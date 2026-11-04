"""Evaluation suite shared by the diagnostic rehearsal and the frozen final run.

The same code path runs on `diagnostic` and on `final`, so the final evaluation
exercises nothing that has not already been executed on other seeds.
"""

import argparse
import json
import math
from pathlib import Path

from yendorlab.benchmark import M5_MASTER, by_seed, evaluate, seed_pairs
from yendorlab.experiments import record
from yendorlab.stats import bootstrap_mean_interval, holm, interquartile_mean, paired_comparison
from yendorlab.tasks import task_named

OUT = Path("data/m5")
SELECTED = {"bc_planner": "bc_planner_r2", "warm_no_planner": "warm_no_planner_r1"}
MOVES = "hjklyubn"


def learned():
    specs = {}
    for run in range(3):
        specs[f"bc_planner_r{run}"] = {"kind": "linear", "encoder": "rel_v3",
                                       "checkpoint": f"data/m5/bc/rel_full-r{run}.json"}
        specs[f"warm_no_planner_r{run}"] = {
            "kind": "linear", "encoder": "rel_v3", "disabled": ["frontier"],
            "checkpoint": f"data/m5/campaigns/warm-rel_no_planner-r{run}/checkpoints/best.json"}
        specs[f"td_planner_r{run}"] = {
            "kind": "linear", "encoder": "rel_v3",
            "checkpoint": f"data/m5/campaigns/td-rel_v3-r{run}/checkpoints/best.json"}
    return specs


def controls(priors):
    specs = {"random": {"kind": "random"}, "fixed_k": {"kind": "fixed", "action": "k"},
             "frontier_moves": {"kind": "frontier"},
             "m4_a": {"kind": "m4_linear", "checkpoint": "data/checkpoints/m4-linear-a-inference.json"},
             "m4_b": {"kind": "m4_linear", "checkpoint": "data/checkpoints/m4-linear-b-inference.json"}}
    for family, shares in priors.items():
        specs[f"prior_{family}"] = {"kind": "prior", "shares": shares}
    return specs


def entropy(shares):
    """Normalized action entropy: 1 is uniform over the eight moves."""
    values = [value for value in shares.values() if value > 0]
    return -sum(value * math.log(value, 8) for value in values)


def run_task(binary, task, specs, pairs, workers):
    results = {}
    for name, spec in specs.items():
        results[name] = evaluate(binary, spec, task, pairs, workers)
        summary = results[name]["summary"]
        print(task.name, name, round(summary["mean_new_cells"] or 0, 1), "valid", summary["valid"],
              "invalid", summary["invalid"], "unavailable", summary["unavailable"], summary["reasons"],
              "stalled", round(summary["stalled_rate"] or 0, 3), "revisit", round(summary["revisit_rate"] or 0, 3),
              "entropy", round(entropy(summary["action_share"]), 3), flush=True)
    return results


def table(results):
    output = {}
    for name, result in results.items():
        summary = result["summary"]
        values = list(by_seed(result).values())
        output[name] = {"summary": summary, "by_seed": by_seed(result), "action_entropy": entropy(summary["action_share"]),
                        "bootstrap_95": bootstrap_mean_interval(values) if values else None,
                        "interquartile_mean": interquartile_mean(values) if values else None}
    return output


def difficulty(results, reference="frontier_moves"):
    """Means per quartile of map openness, measured by the planner on each seed."""
    planner = by_seed(results[reference])
    ordered = sorted(planner, key=planner.get)
    quarter = max(1, len(ordered) // 4)
    groups = [ordered[start:start + quarter] for start in range(0, quarter * 4, quarter)]
    output = {}
    for name, result in results.items():
        values = by_seed(result)
        output[name] = [sum(values.get(seed, 0) for seed in group) / max(1, len(group)) for group in groups]
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--split", choices=["diagnostic", "final"], required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    pairs = seed_pairs(M5_MASTER, args.split, args.count)
    priors_path = OUT / "priors.json"
    priors = json.loads(priors_path.read_text()) if priors_path.is_file() else {}
    floor = task_named("exploration_v2_floor", 250)
    specs = {**controls(priors), **learned()}
    floor_results = run_task(binary, floor, specs, pairs, args.workers)
    if args.split == "diagnostic":
        priors = {family: floor_results[name]["summary"]["action_share"] for family, name in SELECTED.items()}
        priors_path.write_text(json.dumps(priors, indent=1, sort_keys=True))
        for family, shares in priors.items():
            specs[f"prior_{family}"] = {"kind": "prior", "shares": shares}
            floor_results[f"prior_{family}"] = evaluate(binary, specs[f"prior_{family}"], floor, pairs, args.workers)
    relay_names = ["random", "fixed_k", "frontier_moves", "m4_a", *SELECTED.values()]
    relay_results = run_task(binary, task_named("exploration_v2_relay", 250),
                             {name: specs[name] for name in relay_names}, pairs, args.workers)
    multilevel_specs = {"random": specs["random"], "frontier_stairs": {"kind": "frontier", "stairs_goal": True},
                        **{name: specs[name] for name in SELECTED.values()}}
    multilevel = run_task(binary, task_named("exploration_v2_multilevel", 500), multilevel_specs, pairs, args.workers)
    reference = run_task(binary, task_named("exploration_v2_full_reference", 250),
                         {"heuristic_full": {"kind": "heuristic_full"}}, pairs, args.workers)

    comparisons, p_values = {}, {}

    def compare(label, results, first, second, family=True):
        outcome = paired_comparison(by_seed(results[first]), by_seed(results[second]))
        comparisons[label] = outcome
        if family:
            p_values[label] = outcome["sign_flip_p"]

    for family, name in SELECTED.items():
        compare(f"floor:{name}-random", floor_results, name, "random")
        compare(f"floor:{name}-fixed_k", floor_results, name, "fixed_k")
        compare(f"floor:{name}-prior", floor_results, name, f"prior_{family}")
        compare(f"relay:{name}-random", relay_results, name, "random")
        compare(f"relay:{name}-fixed_k", relay_results, name, "fixed_k")
        compare(f"descriptive:{name}-frontier", floor_results, name, "frontier_moves", False)
    compare("descriptive:frontier-random", floor_results, "frontier_moves", "random", False)
    compare("descriptive:m4_a-random", floor_results, "m4_a", "random", False)
    compare("descriptive:m4_a-fixed_k", floor_results, "m4_a", "fixed_k", False)
    compare("descriptive:relay:m4_a-random", relay_results, "m4_a", "random", False)
    adjusted = holm(p_values)
    for label, value in adjusted.items():
        comparisons[label]["holm_p"] = value
    runs = {family: {name: table({name: floor_results[name]})[name]["summary"]["mean_new_cells"]
                     for name in floor_results if name.startswith(family)}
            for family in ("bc_planner", "warm_no_planner", "td_planner")}
    output = {"split": args.split, "count": args.count, "selected": SELECTED, "priors": priors,
              "floor": table(floor_results), "relay": table(relay_results),
              "multilevel": table(multilevel), "reference": table(reference),
              "comparisons": comparisons, "runs": runs, "difficulty_quartiles": difficulty(floor_results)}
    path = OUT / f"suite-{args.split}.json"
    path.write_text(json.dumps(output))
    print(json.dumps({label: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in value.items()
                              if k in ("pairs", "mean_difference", "bootstrap_95", "positive", "negative",
                                       "sign_flip_p", "holm_p")}
                      for label, value in comparisons.items()}, indent=0))
    print("run", record(Path("data"), args.experiment, {"split": args.split, "count": args.count,
                                                        "budget": 250, "selected": SELECTED}, output, binary))


if __name__ == "__main__":
    main()
