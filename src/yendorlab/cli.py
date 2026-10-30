"""Bounded, single-worker Brogue CE episode runner."""

import argparse
import hashlib
import json
import platform
import random
import select
import sqlite3
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    import resource
except ImportError:  # Windows can inspect metadata but native execution requires WSL.
    resource = None


ACTIONS = "hjklyubn"
BROGUE_REVISION = "1ba4240b7a928ddf0ffb772717bf1d433cd63804"


def git_revision() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=False)
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def run_episode(binary: Path, seed: int, policy_seed: int, max_actions: int,
                deadline: float, variant: str = "classic", memory_mb: int = 512) -> dict:
    rng = random.Random(policy_seed)
    command = [str(binary), "--yendorlab", "-n", "-s", str(seed)]
    if variant != "classic":
        command.extend(["--variant", f"{variant}_brogue"])
    started = time.monotonic()
    memory_bytes = memory_mb * 1024 * 1024

    if resource is None:
        raise RuntimeError("native execution requires Linux/WSL for resource limits")

    def limit_child_memory() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

    process = subprocess.Popen(command, cwd=binary.parent, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, bufsize=1, preexec_fn=limit_child_memory)
    actions = 0
    observations = 0
    hashes = []
    terminal = None
    last = None
    status = "truncated"
    error = None
    try:
        while time.monotonic() < deadline:
            ready, _, _ = select.select([process.stdout], [], [],
                                        min(0.5, max(0, deadline - time.monotonic())))
            if not ready:
                if process.poll() is not None:
                    status = "failed"
                    break
                continue
            line = process.stdout.readline()
            if not line:
                status = "failed"
                break
            record = json.loads(line)
            if record.get("event") == "terminal":
                terminal = record
                status = "completed"
                break
            if record.get("schema") != 1 or record.get("seed") != seed:
                raise ValueError("invalid native observation")
            last = record
            hashes.append(hashlib.sha256(line.encode()).hexdigest())
            observations += 1
            if actions >= max_actions:
                break
            action = rng.choice(ACTIONS)
            process.stdin.write(action + "\n")
            process.stdin.flush()
            actions += 1
    except KeyboardInterrupt:
        status = "interrupted"
    except (ValueError, OSError, BrokenPipeError) as exc:
        status = "failed"
        error = str(exc)
    finally:
        process.terminate()
        try:
            process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
    elapsed = time.monotonic() - started
    return {"seed": seed, "policy_seed": policy_seed, "status": status,
            "result": terminal["result"] if terminal else None,
            "depth": terminal["depth"] if terminal else (last or {}).get("depth"),
            "turn": terminal["turn"] if terminal else (last or {}).get("turn"),
            "error": error,
            "actions": actions, "observations": observations,
            "duration_s": elapsed, "actions_per_s": actions / elapsed if elapsed else 0,
            "observation_hashes": hashes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path(".local/BrogueCE/bin/brogue"))
    parser.add_argument("--experiment", default="EXP-0004")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-actions", type=int, default=50)
    parser.add_argument("--max-seconds", type=float, default=30)
    parser.add_argument("--memory-mb", type=int, default=512,
                        help="per Brogue subprocess address-space limit")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--variant", choices=["classic", "rapid", "bullet"], default="classic")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.episodes < 1 or args.max_actions < 1 or args.max_seconds <= 0 or args.seed < 1 or args.memory_mb < 64:
        parser.error("episodes, actions, seconds, seed must be positive and memory at least 64 MiB")
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"binary missing: {binary}")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.data_dir / "experiments.sqlite3")
    db.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, experiment_id TEXT, "
               "started_utc TEXT, config_json TEXT, result_json TEXT)")
    run_id = str(uuid.uuid4())
    config = {"binary": str(binary), "episodes": args.episodes, "max_actions": args.max_actions,
              "max_seconds": args.max_seconds, "seed": args.seed, "variant": args.variant,
              "workers": 1, "agent": "uniform_random_v1", "observation_schema": 1,
              "action_schema": 1, "memory_mb_per_child": args.memory_mb}
    started_utc = datetime.now(timezone.utc).isoformat()
    config.update({"brogue_revision": BROGUE_REVISION, "yendorlab_revision": git_revision(),
                   "hardware": platform.platform(), "mode": "normal",
                   "patch_sha256": hashlib.sha256(Path("native/patches/0001-yendorlab-stream.patch").read_bytes()).hexdigest()})
    deadline = time.monotonic() + args.max_seconds
    results = []
    try:
        for index in range(args.episodes):
            if time.monotonic() >= deadline:
                break
            results.append(run_episode(binary, args.seed + index, args.seed + 100000 + index,
                                       args.max_actions, deadline, args.variant, args.memory_mb))
            if results[-1]["status"] == "interrupted":
                break
    except KeyboardInterrupt:
        pass
    finally:
        summary = {"started": len(results), "completed": sum(r["status"] == "completed" for r in results),
                   "truncated": sum(r["status"] == "truncated" for r in results),
                   "interrupted": sum(r["status"] == "interrupted" for r in results),
                   "failed": sum(r["status"] == "failed" for r in results),
                   "duration_s": sum(r["duration_s"] for r in results),
                   "actions": sum(r["actions"] for r in results), "episodes": results}
        db.execute("INSERT INTO runs VALUES (?,?,?,?,?)",
                   (run_id, args.experiment, started_utc, json.dumps(config), json.dumps(summary)))
        db.commit()
        db.close()
        print(json.dumps({"experiment": args.experiment, "run_id": run_id, **summary}))


if __name__ == "__main__":
    main()
