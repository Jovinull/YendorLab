"""Paired bootstrap for frozen EXP-0020 new-cell comparisons."""

import argparse
import json
import random
import sqlite3
import statistics
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("initial_run")
parser.add_argument("candidate_runs", nargs="+")
parser.add_argument("--db", type=Path, default=Path("data/experiments.sqlite3"))
parser.add_argument("--split", choices=["validation", "final"], default="final")
args = parser.parse_args()
database = sqlite3.connect(args.db)


def episodes(run_id):
    row = database.execute("SELECT config_json,result_json FROM runs WHERE run_id=?",
                           (run_id,)).fetchone()
    if row is None:
        raise ValueError(f"unknown run: {run_id}")
    config, result = map(json.loads, row)
    if config.get("split") != args.split or result.get("errors"):
        raise ValueError(f"run is not a complete error-free {args.split} evaluation: {run_id}")
    return {episode["seed"]: episode for episode in result["episodes"]}


baseline = episodes(args.initial_run)
for run_id in args.candidate_runs:
    candidate = episodes(run_id)
    if set(candidate) != set(baseline):
        raise ValueError("paired seed sets differ")
    differences = [candidate[seed]["new_cells"] - baseline[seed]["new_cells"]
                   for seed in sorted(baseline)]
    rng = random.Random(20261030)
    means = sorted(statistics.mean(rng.choice(differences) for _ in differences)
                   for _ in range(10000))
    print(json.dumps({"run_id": run_id, "n": len(differences),
                      "mean_new_cells": statistics.mean(e["new_cells"] for e in candidate.values()),
                      "paired_mean_difference": statistics.mean(differences),
                      "paired_bootstrap_95": [means[249], means[9749]],
                      "positive_pairs": sum(d > 0 for d in differences),
                      "zero_pairs": sum(d == 0 for d in differences),
                      "negative_pairs": sum(d < 0 for d in differences),
                      "candidate_events": sum(e["events"] for e in candidate.values()),
                      "stalled_moves": sum(e.get("stalled_moves", 0) for e in candidate.values()),
                      "action_counts": {action: sum(e.get("action_counts", {}).get(action, 0)
                                                    for e in candidate.values())
                                        for action in "hjklyubn"},
                      "candidate_deaths": sum(e["reason"] == "DEATH" for e in candidate.values())}))
database.close()
