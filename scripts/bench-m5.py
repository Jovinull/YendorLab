"""EXP-0031: where episode time goes, by filesystem, workers and pipeline stage.

Bounded: every section uses fixed diagnostic seeds and a few hundred decisions.
"""

import argparse
import json
import os
import shutil
import statistics
import tempfile
import time
from pathlib import Path

from yendorlab.benchmark import M5_MASTER, evaluate, seed_pairs
from yendorlab.encoders import Encoder
from yendorlab.env import BrogueEnv
from yendorlab.learners import REWARDS, TDTrainer
from yendorlab.policies import FrontierMoves, LinearPolicy, build_policy
from yendorlab.tasks import Perception, task_named
from yendorlab.training import save_checkpoint


def resets(binary, count, runtime=None, cwd_files=0):
    """Reset latency; `cwd_files` reproduces the pre-M5 accumulation of LastGame files."""
    directory = tempfile.mkdtemp(prefix="bench-reset-", dir=runtime)
    for index in range(cwd_files):
        Path(directory, "LastGame.broguesave" if index == 0
             else f"LastGame ({index + 1}).broguesave").write_bytes(b"x" * 38)
    command_cwd = directory
    samples = []
    import subprocess
    for seed in range(1, count + 1):
        started = time.monotonic()
        process = subprocess.Popen([str(binary), "--yendorlab", "-n", "-s", str(seed)], cwd=command_cwd,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        process.stdout.readline()
        samples.append(time.monotonic() - started)
        process.kill()
        process.wait()
        process.stdin.close()
        process.stdout.close()
        leftover = [path for path in Path(directory).iterdir() if path.stat().st_size == 0
                    and path.name.startswith("LastGame")]
        for path in leftover:
            path.unlink()
    shutil.rmtree(directory, ignore_errors=True)
    return {"count": count, "cwd_files": cwd_files, "median_s": statistics.median(samples),
            "max_s": max(samples)}


def stage_profile(binary, decisions=400):
    """Split the wall time of a mature map into native wait, JSON, tracking and policy."""
    encoder = Encoder("rel_v3")
    seed, policy_seed = seed_pairs(M5_MASTER, "diagnostic", 1)[0]
    policy = LinearPolicy(encoder, seed=policy_seed)
    teacher = FrontierMoves(policy_seed)
    perception = Perception()
    env = BrogueEnv(binary, max_events=5000, max_seconds=60, max_no_progress=6000)
    timings = dict.fromkeys(("perceive", "encode", "teacher", "score"), 0.0)
    observation, _ = env.reset(seed=seed)
    raw_read = json_parse = track = 0.0
    original_track = env._track

    def timed_track(message):
        nonlocal track
        started = time.perf_counter()
        result = original_track(message)
        track += time.perf_counter() - started
        return result

    env._track = timed_track
    original_loads = json.loads

    def timed_loads(text, *args, **kwargs):
        nonlocal json_parse
        started = time.perf_counter()
        result = original_loads(text, *args, **kwargs)
        json_parse += time.perf_counter() - started
        return result

    json.loads = timed_loads
    step_total = 0.0
    lengths = []
    count = 0
    started_all = time.perf_counter()
    while count < decisions:
        if observation["context"] == "gameplay":
            before = time.perf_counter()
            view = perception.observe(observation)
            middle = time.perf_counter()
            timings["perceive"] += middle - before
            action = teacher.decide(observation, view)
            third = time.perf_counter()
            timings["teacher"] += third - middle
            encoder(view)
            fourth = time.perf_counter()
            timings["encode"] += fourth - third
            policy.decide(observation, view)
            timings["score"] += time.perf_counter() - fourth
            perception.acted(view, action)
            count += 1
            lengths.append(len(observation["cells"]))
        else:
            from yendorlab.agents import context_action
            action = context_action(observation)
        before = time.perf_counter()
        observation, _, terminated, truncated, _ = env.step(action)
        step_total += time.perf_counter() - before
        if terminated or truncated:
            break
    json.loads = original_loads
    events = env.events
    env.close()
    total = time.perf_counter() - started_all
    per_event_ms = lambda seconds: 1000 * seconds / max(1, events)
    return {"events": events, "decisions": count, "total_s": total,
            "mean_cells_in_observation": statistics.mean(lengths),
            "step_ms_per_event": per_event_ms(step_total),
            "json_ms_per_event": per_event_ms(json_parse),
            "track_ms_per_event": per_event_ms(track),
            "native_and_pipe_ms_per_event": per_event_ms(step_total - json_parse - track),
            **{f"{name}_ms_per_decision": 1000 * value / max(1, count) for name, value in timings.items()}}


def throughput(binary, workers, count=16, budget=150):
    task = task_named("exploration_v2_floor", budget)
    pairs = seed_pairs(M5_MASTER, "diagnostic", count)
    outcome = evaluate(binary, {"kind": "frontier"}, task, pairs, workers)["summary"]
    wall = outcome["wall_s"]
    return {"workers": workers, "episodes": count, "budget": budget, "wall_s": wall,
            "episodes_per_s": count / wall, "decisions_per_s": outcome["mean_decisions"] * count / wall,
            "events_per_s": outcome["mean_events"] * count / wall,
            "turns_per_s": outcome["mean_turns"] * count / wall, "rss_peak_kib": outcome["rss_peak_kib"]}


def learner_costs(updates=20000):
    encoder = Encoder("rel_v3")
    cells = [[x, y, 1, 1, int(x in (0, 40) or y in (0, 25)), 0] for x in range(41) for y in range(26)]
    observation = {"context": "gameplay", "x": 20, "y": 12, "width": 79, "height": 29, "depth": 1,
                   "cells": cells, "creatures": [], "decision": 1, "admissible": []}
    policy = LinearPolicy(encoder, seed=1)
    trainer = TDTrainer(policy, .05, .97, REWARDS["cells_v2"])
    perception = Perception()
    view = perception.observe(observation)
    policy.decide(observation, view)
    started = time.perf_counter()
    for _ in range(updates):
        trainer._update(.5, view)
    update_us = 1e6 * (time.perf_counter() - started) / updates
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory, "checkpoint.json")
        started = time.perf_counter()
        for _ in range(50):
            save_checkpoint(path, {"weights": policy.weights, "episodes_completed": 1})
        checkpoint_ms = 1000 * (time.perf_counter() - started) / 50
    return {"td_update_us": update_us, "checkpoint_ms_tmp": checkpoint_ms}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mounted", type=Path, default=Path(".local/BrogueCE/bin/brogue"))
    parser.add_argument("--linux", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/m5/bench.json"))
    args = parser.parse_args()
    mounted, linux = args.mounted.resolve(), args.linux.resolve()
    report = {"cpu_count": os.cpu_count()}
    report["reset"] = {
        "mounted_binary_mounted_cwd_617_files": resets(mounted, 8, "data", 617),
        "mounted_binary_mounted_cwd_empty": resets(mounted, 8, "data", 0),
        "mounted_binary_tmpfs_cwd": resets(mounted, 16, "/dev/shm", 0),
        "linux_binary_mounted_cwd_617_files": resets(linux, 8, "data", 617),
        "linux_binary_tmpfs_cwd_617_files": resets(linux, 16, "/dev/shm", 617),
        "linux_binary_tmpfs_cwd": resets(linux, 32, "/dev/shm", 0)}
    report["stages_linux_binary"] = stage_profile(linux)
    report["stages_mounted_binary"] = stage_profile(mounted)
    report["throughput_linux"] = [throughput(linux, workers) for workers in (1, 2, 4, 6)]
    report["throughput_mounted"] = [throughput(mounted, 1)]
    report["learner"] = learner_costs()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
