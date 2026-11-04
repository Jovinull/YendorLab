"""Bounded, resumable training campaigns with a durable journal.

Write order for each episode: run, journal the result, then checkpoint at the
configured cadence. On resume, journal rows newer than the checkpoint are marked
superseded and those episodes run again from the checkpointed weights, so a crash
between the two writes cannot lose or double-count an episode. Each episode's game
seed and RNG derive from its index, which makes resume exact on the same build and
platform as long as no timing limit fires; that is tested, not promised more widely.
"""

import hashlib
import json
import os
import platform
import resource
import signal
import sqlite3
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from .benchmark import by_seed, evaluate, seed_pairs
from .encoders import Encoder
from .env import OPERATIONAL_REASONS, PROTOCOL, OBSERVATION_SCHEMA, ACTION_SCHEMA
from .experiments import file_sha256, revision, source_fingerprint, worktree_dirty
from .learners import REWARDS, TDTrainer
from .policies import LinearPolicy
from .stats import paired_comparison
from .tasks import m5_seed, run_episode, task_named
from .training import save_checkpoint

ALGORITHM = "linear_td_v2"
CHECKPOINT_VERSION = 2
MAX_RETRIES = 2
SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
  idx INTEGER, attempt INTEGER, seed TEXT, reason TEXT, valid INTEGER, new_cells INTEGER,
  decisions INTEGER, events INTEGER, turns INTEGER, updates INTEGER, mean_abs_error REAL,
  duration_s REAL, weights_sha256 TEXT, superseded INTEGER DEFAULT 0, error TEXT, utc TEXT,
  PRIMARY KEY (idx, attempt));
CREATE TABLE IF NOT EXISTS validations (
  episodes INTEGER, sample INTEGER, invalid INTEGER, mean_new_cells REAL, baseline_difference REAL,
  duration_s REAL, is_best INTEGER, superseded INTEGER DEFAULT 0, utc TEXT);
CREATE TABLE IF NOT EXISTS log (utc TEXT, kind TEXT, detail TEXT);
"""


def utc():
    return datetime.now(timezone.utc).isoformat()


def weights_hash(weights):
    return hashlib.sha256(json.dumps(weights, separators=(",", ":")).encode()).hexdigest()


def load_config(path: Path):
    return tomllib.loads(path.read_text())


def fixed_identity(config, binary):
    """What must not change across a resume."""
    start = config["train"].get("init")
    return {"binary_sha256": file_sha256(binary), "source_tree_sha256": source_fingerprint(),
            "init_sha256": file_sha256(start) if start else None, "protocol": PROTOCOL, "observation_schema": OBSERVATION_SCHEMA,
            "action_schema": ACTION_SCHEMA,
            "config": {section: {k: v for k, v in values.items() if k != "episodes"}
                       for section, values in config.items() if section != "limits"}}


class Campaign:
    def __init__(self, config, directory: Path, binary: Path, fault=None):
        self.config = config
        self.train = config["train"]
        self.limits = config.get("limits", {})
        self.directory = directory
        self.binary = binary
        self.fault = fault or os.environ.get("YENDORLAB_FAULT")
        self.stop_reason = None
        self.task = task_named(self.train["task"], self.train["budget"],
                               episode_seconds=self.limits.get("episode_seconds", 120.0),
                               memory_mb=self.limits.get("memory_mb", 512))
        self.encoder = Encoder(self.train["encoder"], self.train.get("disabled", ()))
        self.latest = directory / "checkpoints/latest.json"
        self.best = directory / "checkpoints/best.json"
        self.journal = None

    # --- persistent state -------------------------------------------------------------
    def _fresh_state(self):
        weights = [0.0] * self.encoder.dimension
        if self.train.get("init"):
            weights = json.loads(Path(self.train["init"]).read_text())["weights"]
        return {"checkpoint_version": CHECKPOINT_VERSION, "kind": "training", "algorithm": ALGORITHM,
                "encoder": self.encoder.spec(), "observation_schema": OBSERVATION_SCHEMA,
                "action_schema": ACTION_SCHEMA, "weights": weights, "episodes_completed": 0,
                "updates": 0, "decisions": 0, "turns": 0, "best_score": None, "best_episode": None,
                "no_improvement": 0, "operational_failures": 0}

    def open(self, resume):
        (self.directory / "checkpoints").mkdir(parents=True, exist_ok=True)
        identity = fixed_identity(self.config, self.binary)
        manifest_path = self.directory / "manifest.json"
        if resume:
            manifest = json.loads(manifest_path.read_text())
            for key, value in identity.items():
                if manifest["identity"].get(key) != value:
                    raise ValueError(f"resume refused: {key} differs from the campaign manifest")
            state = json.loads(self.latest.read_text())
        else:
            if manifest_path.exists():
                raise FileExistsError(f"campaign already exists: {self.directory}")
            state = self._fresh_state()
            save_checkpoint(self.latest, state)
            manifest = {"identity": identity, "algorithm": ALGORITHM, "created_utc": utc(),
                        "yendorlab_revision": revision(), "worktree_dirty": worktree_dirty(),
                        "hardware": platform.platform(), "cpu_count": os.cpu_count(),
                        "limits": self.limits, "artifacts": {}}
            save_checkpoint(manifest_path, manifest)
        self.journal = sqlite3.connect(self.directory / "journal.sqlite3")
        self.journal.executescript("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;" + SCHEMA)
        if resume:
            count = self.journal.execute("UPDATE episodes SET superseded=1 WHERE idx>=? AND superseded=0",
                                         (state["episodes_completed"],)).rowcount
            self.journal.execute("UPDATE validations SET superseded=1 WHERE episodes>? AND superseded=0",
                                 (state["episodes_completed"],))
            self.log("resume", {"episodes_completed": state["episodes_completed"],
                                "superseded_rows": count})
        return state

    def log(self, kind, detail):
        self.journal.execute("INSERT INTO log VALUES (?,?,?)", (utc(), kind, json.dumps(detail)))
        self.journal.commit()

    def journal_episode(self, index, attempt, record, trainer, digest):
        self.journal.execute(
            "INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,0,?,?)",
            (index, attempt, str(record["seed"]), record["reason"], int(record["valid"]),
             record.get("new_cells"), record.get("decisions"), record.get("events"),
             record.get("turn"), trainer.updates,
             trainer.absolute_error / trainer.updates if trainer.updates else None,
             record["duration_s"], digest, record.get("error"), utc()))
        self.journal.commit()

    def save(self, state, index):
        digest = save_checkpoint(self.latest, state)
        every = self.train.get("periodic_checkpoint_every", 0)
        if every and state["episodes_completed"] % every == 0:
            periodic = self.directory / f"checkpoints/ep-{state['episodes_completed']:07d}.json"
            save_checkpoint(periodic, state)
            self._retain()
        return digest

    def _retain(self):
        keep = self.train.get("keep_periodic", 5)
        periodic = sorted((self.directory / "checkpoints").glob("ep-*.json"))
        for path in periodic[:-keep] if keep else []:
            path.unlink()

    # --- limits -----------------------------------------------------------------------
    def apply_process_limits(self):
        cores = self.limits.get("cpu_cores")
        if cores:
            available = sorted(os.sched_getaffinity(0))
            os.sched_setaffinity(0, available[:cores])
        niceness = self.limits.get("nice", 0)
        if niceness:
            os.nice(niceness)

    def request_stop(self, *_):
        self.stop_reason = self.stop_reason or "SIGNAL"

    def budget_stop(self, state, started):
        if self.stop_reason:
            return self.stop_reason
        if (self.directory / "STOP").exists():
            return "STOP_FILE"
        if state["episodes_completed"] >= self.train["episodes"]:
            return "EPISODE_TARGET"
        if time.monotonic() - started >= self.limits.get("max_seconds", 1e12):
            return "TIME_BUDGET"
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        if peak_mb >= self.limits.get("max_rss_mb", 1e12):
            return "MEMORY_LIMIT"
        patience = self.config.get("stop", {}).get("patience_validations")
        if patience and state["no_improvement"] >= patience:
            return "STAGNATION"
        return None

    # --- training ---------------------------------------------------------------------
    def episode_epsilon(self, index):
        start = self.train["epsilon_start"]
        end = self.train.get("epsilon_end", start)
        decay = max(1, self.train.get("epsilon_decay_episodes", self.train["episodes"]))
        return start + (end - start) * min(1.0, index / decay)

    def train_one(self, state, index):
        """Run one episode; only a valid result may touch the committed weights."""
        master = self.train["master_seed"]
        stream = f"run{self.train['run']}"
        seed = m5_seed(master, "train", index, stream)
        policy_seed = m5_seed(master, "train", index, stream + ":policy")
        for attempt in range(MAX_RETRIES + 1):
            policy = LinearPolicy(self.encoder, state["weights"], policy_seed, self.episode_epsilon(index))
            trainer = TDTrainer(policy, self.train["alpha"], self.train["gamma"],
                                REWARDS[self.train["reward"]])
            record = run_episode(self.binary, policy, seed, self.task, policy_seed, trainer=trainer)
            usable = record["reason"] not in OPERATIONAL_REASONS
            digest = weights_hash(policy.weights) if usable else None
            self.journal_episode(index, self.next_attempt(index), record, trainer, digest)
            if usable:
                return policy.weights, trainer, record
            state["operational_failures"] += 1
            self.log("operational_failure", {"episode": index, "attempt": attempt,
                                             "reason": record["reason"], "error": record.get("error")})
            if self.stop_reason:
                return None, None, record
        return None, None, record

    def next_attempt(self, index):
        row = self.journal.execute("SELECT COALESCE(MAX(attempt)+1,0) FROM episodes WHERE idx=?",
                                   (index,)).fetchone()
        return row[0]

    def validate(self, state):
        """Validation seeds only; the selection rule never sees diagnostic or final seeds."""
        settings = self.train
        started = time.monotonic()
        pairs = seed_pairs(settings["master_seed"], "validation", settings["validation_episodes"])
        result = evaluate(self.binary, {"kind": "linear", "encoder": self.encoder.name,
                                        "disabled": list(self.encoder.disabled),
                                        "weights": state["weights"]}, self.task, pairs,
                          self.limits.get("validation_workers", 1))
        summary = result["summary"]
        score = summary["mean_new_cells"]
        baseline_path = settings.get("validation_baseline")
        difference = None
        if baseline_path and Path(baseline_path).is_file():
            baseline = {int(k): v for k, v in json.loads(Path(baseline_path).read_text()).items()}
            difference = paired_comparison(by_seed(result), baseline, resamples=1000).get("mean_difference")
        minimum = self.config.get("stop", {}).get("min_delta", 0.0)
        improved = summary["invalid"] == 0 and score is not None and (
            state["best_score"] is None or score > state["best_score"] + minimum)
        if improved:
            state["best_score"] = score
            state["best_episode"] = state["episodes_completed"]
            state["no_improvement"] = 0
            save_checkpoint(self.best, state)
        else:
            state["no_improvement"] += 1
        self.journal.execute("INSERT INTO validations VALUES (?,?,?,?,?,?,?,0,?)",
                             (state["episodes_completed"], summary["valid"], summary["invalid"], score,
                              difference, time.monotonic() - started, int(improved), utc()))
        self.journal.commit()

    def run(self, resume=False):
        started = time.monotonic()
        state = self.open(resume)
        self.apply_process_limits()
        previous = {number: signal.signal(number, self.request_stop)
                    for number in (signal.SIGINT, signal.SIGTERM)}
        failed = False
        try:
            while True:
                reason = self.budget_stop(state, started)
                if reason:
                    self.stop_reason = reason
                    break
                index = state["episodes_completed"]
                weights, trainer, record = self.train_one(state, index)
                if weights is None:
                    failed = not self.stop_reason
                    self.stop_reason = self.stop_reason or "OPERATIONAL_FAILURE"
                    break
                if self.fault == "after_journal":
                    os._exit(41)
                state.update(weights=weights, episodes_completed=index + 1,
                             updates=state["updates"] + trainer.updates,
                             decisions=state["decisions"] + record["decisions"],
                             turns=state["turns"] + record["turn"])
                if state["episodes_completed"] % self.train.get("validation_every", 10 ** 9) == 0:
                    self.validate(state)
                    self.save(state, index)
                elif state["episodes_completed"] % self.train.get("checkpoint_every", 1) == 0:
                    self.save(state, index)
                if self.fault == "after_checkpoint":
                    os._exit(42)
                if self.fault and self.fault.startswith("stop_at:"):
                    if state["episodes_completed"] == int(self.fault.split(":")[1]):
                        self.stop_reason = "SIGNAL"
            self.save(state, state["episodes_completed"])
            status = "FAILED_OPERATIONAL" if failed else "STOPPED:" + self.stop_reason
            self.log("stop", {"status": status, "episodes_completed": state["episodes_completed"]})
            return self.finish(state, status, time.monotonic() - started)
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
            self.journal.close()

    def finish(self, state, status, duration):
        manifest_path = self.directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        inference = self.directory / "checkpoints/inference-latest.json"
        payload = {"kind": "inference", "algorithm": ALGORITHM, "encoder": state["encoder"],
                   "observation_schema": OBSERVATION_SCHEMA, "action_schema": ACTION_SCHEMA,
                   "weights": state["weights"], "episodes_completed": state["episodes_completed"]}
        save_checkpoint(inference, payload)
        artifacts = {path.name: file_sha256(path)
                     for path in sorted((self.directory / "checkpoints").glob("*.json"))}
        manifest["artifacts"] = artifacts
        manifest["status"] = status
        manifest["updated_utc"] = utc()
        save_checkpoint(manifest_path, manifest)
        return {"status": status, "episodes_completed": state["episodes_completed"],
                "updates": state["updates"], "decisions": state["decisions"], "turns": state["turns"],
                "duration_s": duration, "operational_failures": state["operational_failures"],
                "best_score": state["best_score"], "best_episode": state["best_episode"],
                "weights_sha256": weights_hash(state["weights"]),
                "peak_python_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}


def status(directory: Path):
    """Compact objective report; reads only durable files."""
    manifest = json.loads((directory / "manifest.json").read_text())
    state = json.loads((directory / "checkpoints/latest.json").read_text())
    journal = sqlite3.connect(directory / "journal.sqlite3")
    journal.row_factory = sqlite3.Row
    query = lambda sql, *arguments: journal.execute(sql, arguments).fetchall()
    rows = query("SELECT * FROM episodes WHERE superseded=0 AND valid=1 AND idx<? ORDER BY idx",
                 state["episodes_completed"])
    failures = query("SELECT reason, COUNT(*) AS n FROM episodes WHERE valid=0 GROUP BY reason")
    validations = query("SELECT * FROM validations WHERE superseded=0 ORDER BY episodes")
    recent = rows[-50:]
    spent = sum(row["duration_s"] for row in rows)
    report = {"status": manifest.get("status", "RUNNING_OR_INTERRUPTED"),
              "algorithm": manifest["algorithm"], "created_utc": manifest["created_utc"],
              "episodes_completed": state["episodes_completed"],
              "journal_valid_episodes": len(rows), "updates": state["updates"],
              "decisions": state["decisions"], "turns": state["turns"],
              "episode_seconds_total": spent,
              "decisions_per_s": sum(r["decisions"] for r in rows) / spent if spent else None,
              "recent_50_train_new_cells": sum(r["new_cells"] for r in recent) / len(recent) if recent else None,
              "recent_50_mean_abs_td_error": sum(r["mean_abs_error"] or 0 for r in recent) / len(recent) if recent else None,
              "operational_failures": {row["reason"]: row["n"] for row in failures},
              "best_validation": state["best_score"], "best_episode": state["best_episode"],
              "last_validation": validations[-1]["mean_new_cells"] if validations else None,
              "last_vs_baseline": validations[-1]["baseline_difference"] if validations else None,
              "validations_without_improvement": state["no_improvement"],
              "validation_curve": [[row["episodes"], row["mean_new_cells"]] for row in validations]}
    journal.close()
    return report
