"""Brogue CE protocol v3 environment. One step equals one native input event."""

import hashlib
import json
import os
import select
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import resource
except ImportError:
    resource = None


PROTOCOL = 3
OBSERVATION_SCHEMA = 3
ACTION_SCHEMA = 2
MAX_LINE_BYTES = 8 * 1024 * 1024
RESULTS = {"death", "quit", "victory", "supervictory"}

# Outcome taxonomy. Natural outcomes and configured limits describe an episode of
# the policy; operational failures describe the laboratory and are never competence.
NATURAL_REASONS = {"DEATH", "QUIT", "VICTORY", "SUPERVICTORY"}
LIMIT_REASONS = {"TRUNCATED_ACTION_LIMIT", "TRUNCATED_NO_PROGRESS"}
OPERATIONAL_REASONS = {"TRUNCATED_TIME_LIMIT", "ENGINE_ERROR", "PROTOCOL_ERROR",
                       "RESET_ERROR", "POLICY_ERROR"}


class ResetError(RuntimeError):
    """The native child never reached a valid first observation."""


def runtime_root() -> Path:
    """Private game directories live on a Linux filesystem, preferably tmpfs."""
    configured = os.environ.get("YENDORLAB_RUNTIME_DIR")
    if configured:
        return Path(configured)
    shared_memory = Path("/dev/shm")
    if shared_memory.is_dir() and os.access(shared_memory, os.W_OK):
        return shared_memory
    return Path(tempfile.gettempdir())


@dataclass(frozen=True)
class RewardConfig:
    name: str = "canonical"
    win: float = 1.0
    death: float = -1.0
    new_cell: float = 0.0
    new_depth: float = 0.0

    @classmethod
    def named(cls, name: str) -> "RewardConfig":
        if name == "canonical":
            return cls()
        if name == "exploration_v1":
            return cls(name, new_cell=0.01, new_depth=1.0)
        raise ValueError(f"unknown reward scheme: {name}")


class BrogueEnv:
    def __init__(self, binary: Path, variant: str = "classic", max_events: int = 300,
                 max_seconds: float = 30, max_no_progress: int = 50, memory_mb: int = 512,
                 reward: RewardConfig | None = None, extra_args: tuple = ()):
        if variant not in {"classic", "rapid", "bullet"}:
            raise ValueError("unsupported variant")
        if min(max_events, max_no_progress, memory_mb) <= 0 or max_seconds <= 0:
            raise ValueError("limits must be positive")
        self.binary = binary.resolve()
        self.variant = variant
        self.max_events = max_events
        self.max_seconds = max_seconds
        self.max_no_progress = max_no_progress
        self.memory_mb = memory_mb
        self.reward = reward or RewardConfig()
        self.extra_args = tuple(extra_args)
        self.process = None
        self.observation = None
        self.seed = None
        self.events = 0
        self.deadline = 0.0
        self.no_progress = 0
        self.known = set()
        self.initial_known = 0
        self.known_by_depth = {}
        self.initial_depth = None
        self.max_depth = 0
        self.terminal = None
        self.peak_native_rss_kib = 0
        self.components = {"terminal": 0.0, "exploration": 0.0, "depth": 0.0}
        self.workdir = None
        self.stderr_file = None
        self.stderr_tail = ""
        self._buffer = bytearray()
        self._trace = hashlib.sha256()

    def _read(self):
        """Read one line with a real deadline, independent of chunk boundaries."""
        fd = self.process.stdout.fileno()
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._buffer[:newline])
                del self._buffer[:newline + 1]
                self._trace.update(line)
                try:
                    return json.loads(line.decode("utf-8", "replace"))
                except ValueError as exc:
                    return {"event": "protocol_error", "reason": f"invalid_json: {exc}",
                            "raw": line[:160].decode("utf-8", "replace")}
            if len(self._buffer) > MAX_LINE_BYTES:
                return {"event": "protocol_error", "reason": "line_too_long"}
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                return {"event": "time_limit"}
            ready, _, _ = select.select([fd], [], [], min(0.25, remaining))
            if ready:
                chunk = os.read(fd, 1 << 20)
                if chunk:
                    self._buffer.extend(chunk)
                    continue
                code = self.process.wait()
                suffix = "_partial_line" if self._buffer else ""
                return {"event": "engine_error", "reason": f"exit_{code}{suffix}"}

    def _validate(self, message):
        if (message.get("protocol") != PROTOCOL
                or message.get("observation_schema") != OBSERVATION_SCHEMA):
            raise ValueError(f"protocol or observation schema mismatch: {str(message)[:200]}")
        if message.get("action_schema") != ACTION_SCHEMA or message.get("decision") != self.events + 1:
            raise ValueError("action schema or decision sequence mismatch")
        if message.get("seed") != self.seed:
            raise ValueError("observation seed mismatch")
        if not isinstance(message.get("cells"), list) or not isinstance(message.get("inventory"), list):
            raise ValueError("invalid observation arrays")
        if message.get("context") not in {"gameplay", "inventory", "confirmation",
                                           "acknowledgment", "death_acknowledgment",
                                           "text_input", "target_selection"}:
            raise ValueError(f"unknown input context: {message.get('context')!r}")

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.close()
        if seed is None or seed < 1:
            raise ValueError("explicit positive seed required")
        if resource is None:
            raise RuntimeError("native execution requires Linux/WSL")
        if not self.binary.is_file():
            raise FileNotFoundError(self.binary)
        command = [str(self.binary), "--yendorlab", "-n", "-s", str(seed), *self.extra_args]
        if self.variant != "classic":
            command.extend(["--variant", f"{self.variant}_brogue"])
        # Upstream writes recordings, high scores and run history into the cwd.
        # A private directory makes each episode independent of laboratory history.
        self.workdir = tempfile.mkdtemp(prefix="yendorlab-", dir=runtime_root())
        self.stderr_file = tempfile.TemporaryFile(dir=self.workdir)
        self.process = subprocess.Popen(command, cwd=self.workdir, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.stderr_file,
                                        bufsize=0)
        try:
            memory_bytes = self.memory_mb * 1024 * 1024
            resource.prlimit(self.process.pid, resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        except ProcessLookupError:
            pass  # The child already exited; the read below reports it.
        self.seed = seed
        self.deadline = time.monotonic() + self.max_seconds
        self.events = 0
        self.no_progress = 0
        self.known = set()
        self.known_by_depth = {}
        self.max_depth = 0
        self.terminal = None
        self.peak_native_rss_kib = 0
        self.components = {"terminal": 0.0, "exploration": 0.0, "depth": 0.0}
        self.observation = None
        self.stderr_tail = ""
        self._buffer = bytearray()
        self._trace = hashlib.sha256()
        message = self._read()
        if message.get("event") == "time_limit":
            self.close()
            raise TimeoutError("native reset exceeded episode time limit")
        if message.get("event"):
            self.close()
            raise ResetError(f"native reset failed: {str(message)[:200]} "
                             f"stderr={self.stderr_tail[-200:]!r}")
        self._validate(message)
        self.observation = message
        self._sample_rss()
        self._track(message)
        self.initial_known = len(self.known)
        self.initial_depth = message["depth"]
        # Initial knowledge/depth are a baseline, not a step reward.
        self.components = {"terminal": 0.0, "exploration": 0.0, "depth": 0.0}
        return message, {"seed": seed, "variant": self.variant, "reward_scheme": self.reward.name}

    def _track(self, observation):
        depth = observation["depth"]
        newly_known = {(depth, cell[0], cell[1]) for cell in observation["cells"]} - self.known
        self.known.update(newly_known)
        self.known_by_depth[depth] = self.known_by_depth.get(depth, 0) + len(newly_known)
        depth_gain = max(0, depth - self.max_depth)
        self.max_depth = max(depth, self.max_depth)
        components = {"terminal": 0.0,
                      "exploration": len(newly_known) * self.reward.new_cell,
                      "depth": depth_gain * self.reward.new_depth}
        for key, value in components.items():
            self.components[key] += value
        return components

    def _sample_rss(self):
        if self.process is None:
            return
        try:
            status = Path(f"/proc/{self.process.pid}/status").read_text()
            line = next(line for line in status.splitlines() if line.startswith("VmRSS:"))
            self.peak_native_rss_kib = max(self.peak_native_rss_kib, int(line.split()[1]))
        except (OSError, StopIteration, ValueError):
            pass

    def step(self, action: str):
        if self.process is None or self.observation is None or self.terminal:
            raise RuntimeError("reset required or episode already ended")
        if action not in self.observation["admissible"]:
            raise ValueError(f"action {action!r} is not admissible in {self.observation['context']}")
        previous = self.observation
        self._trace.update(action.encode() + b"\0")
        try:
            os.write(self.process.stdin.fileno(), action.encode() + b"\n")
        except OSError:
            message = {"event": "engine_error", "reason": "broken_pipe"}
        else:
            message = self._read()
        self.events += 1
        if self.events % 16 == 0:
            self._sample_rss()
        reward_parts = {"terminal": 0.0, "exploration": 0.0, "depth": 0.0}
        terminated = truncated = False
        reason = None
        if message.get("event") == "terminal":
            if message.get("result") not in RESULTS:
                reason = "PROTOCOL_ERROR"
                truncated = True
            else:
                reason = message["result"].upper()
                terminated = True
                reward_parts["terminal"] = (self.reward.win if reason in {"VICTORY", "SUPERVICTORY"}
                                             else self.reward.death if reason == "DEATH" else 0.0)
                self.components["terminal"] += reward_parts["terminal"]
            self.terminal = message
        elif message.get("event"):
            reason = "TRUNCATED_TIME_LIMIT" if message["event"] == "time_limit" else (
                "PROTOCOL_ERROR" if message["event"] == "protocol_error" else "ENGINE_ERROR")
            truncated = True
            self.terminal = message
        else:
            try:
                self._validate(message)
            except ValueError as exc:
                reason = "PROTOCOL_ERROR"
                truncated = True
                self.terminal = {"event": "protocol_error", "reason": str(exc)}
            else:
                reward_parts = self._track(message)
                progress = (message["turn"] > previous["turn"] or message["depth"] > previous["depth"]
                            or message["x"] != previous["x"] or message["y"] != previous["y"]
                            or message["context"] != previous["context"]
                            or reward_parts["exploration"] > 0)
                self.no_progress = 0 if progress else self.no_progress + 1
                self.observation = message
                if self.no_progress >= self.max_no_progress:
                    reason = "TRUNCATED_NO_PROGRESS"
                    truncated = True
                elif self.events >= self.max_events:
                    reason = "TRUNCATED_ACTION_LIMIT"
                    truncated = True
                if truncated:
                    self.terminal = {"event": "limit", "result": reason}
        if terminated or truncated:
            self.close()
        info = {"reason": reason, "events": self.events, "turn": self.observation["turn"],
                "depth": self.observation["depth"], "max_depth": self.max_depth,
                "new_cells": len(self.known) - self.initial_known,
                "new_cells_by_depth": self.new_cells_by_depth(),
                "reward_components": reward_parts, "reward_totals": dict(self.components),
                "terminal": self.terminal, "native_rss_peak_sampled_kib": self.peak_native_rss_kib}
        return self.observation, sum(reward_parts.values()), terminated, truncated, info

    def new_cells_by_depth(self):
        """Newly discovered cells per depth; only the start depth has a baseline."""
        counts = dict(self.known_by_depth)
        if self.initial_depth in counts:
            counts[self.initial_depth] -= self.initial_known
        return {str(depth): count for depth, count in sorted(counts.items())}

    def trace_digest(self):
        """SHA-256 over every raw native line and action of this episode."""
        return self._trace.hexdigest()

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            for stream in (self.process.stdin, self.process.stdout):
                try:
                    stream.close()
                except OSError:
                    pass
            self.process = None
        if self.stderr_file is not None:
            self.stderr_file.seek(0, os.SEEK_END)
            self.stderr_file.seek(max(0, self.stderr_file.tell() - 2000))
            self.stderr_tail = self.stderr_file.read().decode("utf-8", "replace")
            self.stderr_file.close()
            self.stderr_file = None
        if self.workdir is not None:
            shutil.rmtree(self.workdir, ignore_errors=True)
            self.workdir = None

    def render(self):
        return self.observation

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
