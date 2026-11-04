"""Bounded soak: thousands of resets and long episodes, counting every failure.

M4 recorded one reset failure it could not reproduce. This measures the failure
rate of the current stack and the stability of child memory in long episodes.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

from yendorlab.agents import context_action
from yendorlab.env import BrogueEnv, runtime_root


def worker_pids():
    return [path for path in Path("/proc").iterdir() if path.name.isdigit()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--resets", type=int, default=3000)
    parser.add_argument("--long-episodes", type=int, default=6)
    parser.add_argument("--long-events", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=Path("data/m5/soak.json"))
    args = parser.parse_args()
    binary = args.binary.resolve()
    failures = {}
    started = time.monotonic()
    env = BrogueEnv(binary, max_events=3, max_seconds=10)
    for seed in range(1, args.resets + 1):
        try:
            env.reset(seed=seed * 7919 + 1)
        except (TimeoutError, RuntimeError, ValueError, OSError) as exc:
            failures[type(exc).__name__ + ":" + str(exc)[:160]] = \
                failures.get(type(exc).__name__ + ":" + str(exc)[:160], 0) + 1
    env.close()
    reset_elapsed = time.monotonic() - started
    leftovers = [path.name for path in runtime_root().glob("yendorlab-*")]
    episodes = []
    rng = random.Random(20261103)
    moves = "hjklyubn"
    for index in range(args.long_episodes):
        with BrogueEnv(binary, max_events=args.long_events, max_seconds=180,
                       max_no_progress=args.long_events + 1) as env:
            observation, _ = env.reset(seed=index + 101)
            first_quarter_rss = None
            maximum_cells = 0
            info = None
            while True:
                action = rng.choice(moves) if observation["context"] == "gameplay" else context_action(observation)
                observation, _, terminated, truncated, info = env.step(action)
                maximum_cells = max(maximum_cells, len(observation["cells"]))
                if env.events == args.long_events // 4:
                    first_quarter_rss = env.peak_native_rss_kib
                if terminated or truncated:
                    break
            episodes.append({"seed": index + 101, "reason": info["reason"], "events": info["events"],
                             "turn": info["turn"], "depth": info["max_depth"], "cells": maximum_cells,
                             "rss_first_quarter_kib": first_quarter_rss,
                             "rss_peak_kib": info["native_rss_peak_sampled_kib"]})
    result = {"resets": args.resets, "reset_failures": failures, "reset_seconds": reset_elapsed,
              "resets_per_s": args.resets / reset_elapsed, "leftover_private_directories": len(leftovers),
              "long_episodes": episodes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
