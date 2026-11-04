"""Bounded M5 experiment drivers. Each subcommand writes data/m5/<name>.json and SQLite."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from yendorlab.benchmark import M5_MASTER, by_seed, controls, evaluate, seed_pairs
from yendorlab.encoders import Encoder
from yendorlab.env import BrogueEnv
from yendorlab.evaluation import run_policy_episode, seed_for
from yendorlab.experiments import record
from yendorlab.learners import collect, imitation_metrics, save_demos, train_softmax
from yendorlab.policies import FrontierMoves, LinearPolicy, RawAgent, build_policy
from yendorlab.agents import FixedDirectionAgent, HeuristicAgent, RandomAgent
from yendorlab.skill import LinearFrontierAgent, load_skill_checkpoint
from yendorlab.stats import paired_comparison
from yendorlab.tasks import task_named, m5_seed

DATA = Path("data")
OUT = DATA / "m5"
M4A = "data/checkpoints/m4-linear-a-inference.json"
M4B = "data/checkpoints/m4-linear-b-inference.json"


def save(name, experiment, config, result, binary):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(result))
    trimmed = {k: v for k, v in result.items() if k != "raw"}
    return record(DATA, experiment, config, trimmed, binary)


def show(name, summary):
    keys = ("valid", "invalid", "unavailable", "reasons", "mean_new_cells", "mean_decisions", "mean_turns",
            "stalled_rate", "revisit_rate", "wall_s")
    print(name, {k: round(summary[k], 3) if isinstance(summary[k], float) else summary[k]
                 for k in keys if k in summary}, flush=True)


def m4_specs():
    return {"m4_initial": {"kind": "m4_linear"},
            "m4_a": {"kind": "m4_linear", "checkpoint": M4A},
            "m4_b": {"kind": "m4_linear", "checkpoint": M4B}}


def m4audit(args):
    """EXP-0023: reproduce the M4 final comparison with per-depth and exposure accounting."""
    agents = {
        "m4_initial": lambda s: LinearFrontierAgent(seed=s),
        "m4_a": lambda s: LinearFrontierAgent(load_skill_checkpoint(Path(M4A))["weights"], seed=s),
        "m4_b": lambda s: LinearFrontierAgent(load_skill_checkpoint(Path(M4B))["weights"], seed=s),
        "random": lambda s: RandomAgent(s),
        "heuristic": lambda s: HeuristicAgent(s)}
    output = {}
    for name, factory in agents.items():
        episodes = []
        for index in range(16):
            seed = seed_for(20261031, "final", index)
            agent = factory(seed_for(20261031, "final", index + 100000))
            episodes.append(run_policy_episode(args.binary, agent, seed, "classic", "exploration_v1",
                                               120, 8, 30, 512))
        count = len(episodes)
        output[name] = {
            "mean_new_cells": sum(e["new_cells"] for e in episodes) / count,
            "mean_new_cells_depth1": sum(e["new_cells_by_depth"].get("1", 0) for e in episodes) / count,
            "mean_new_cells_other_depths": sum(e["new_cells"] - e["new_cells_by_depth"].get("1", 0)
                                               for e in episodes) / count,
            "episodes_past_depth1": sum(e["max_depth"] > 1 for e in episodes),
            "mean_events": sum(e["events"] for e in episodes) / count,
            "mean_decisions": sum(e["decisions"] for e in episodes) / count,
            "mean_turns": sum(e["turn"] for e in episodes) / count,
            "stalled": sum(e["stalled_moves"] for e in episodes) / count,
            "reasons": {reason: sum(e["reason"] == reason for e in episodes)
                        for reason in {e["reason"] for e in episodes}},
            "actions": {action: sum(e["action_counts"].get(action, 0) for e in episodes)
                        for action in "hjklyubnz"}}
        print(name, json.dumps(output[name]), flush=True)
    run = save("m4audit", "EXP-0023", {"seeds": "M4 final, master 20261031, 16", "max_events": 120,
                                       "max_no_progress": 30}, output, args.binary)
    print("run", run)


def anisotropy(args):
    """EXP-0024: where does the unknown lie at the start of depth 1 and depth 2?"""
    positions, shares = {}, {"north": 0, "south": 0, "west": 0, "east": 0}
    pairs = seed_pairs(M5_MASTER, "diagnostic", args.count)
    for seed, _ in pairs:
        with BrogueEnv(args.binary, max_events=3) as env:
            observation, _ = env.reset(seed=seed)
        key = f"{observation['x']},{observation['y']}"
        positions[key] = positions.get(key, 0) + 1
        x, y = observation["x"], observation["y"]
        shares["north"] += y
        shares["south"] += observation["height"] - 1 - y
        shares["west"] += x
        shares["east"] += observation["width"] - 1 - x
    relay = evaluate(args.binary, {"kind": "random"}, task_named("exploration_v2_relay", 5), pairs, args.workers)
    second = {}
    rows = []
    for episode in relay["episodes"]:
        if episode["valid"]:
            x, y = episode["start_position"]
            rows.append([x, y])
            second[f"{x},{y}"] = second.get(f"{x},{y}", 0) + 1
    count = len(pairs)
    distinct = len(second)
    result = {"seeds": count, "depth1_distinct_starts": len(positions),
              "depth1_most_common": sorted(positions.items(), key=lambda item: -item[1])[:5],
              "depth1_mean_room_toward": {k: v / count for k, v in shares.items()},
              "depth2_available": len(rows), "depth2_distinct_starts": distinct,
              "depth2_mean_x": sum(r[0] for r in rows) / len(rows) if rows else None,
              "depth2_mean_y": sum(r[1] for r in rows) / len(rows) if rows else None,
              "depth2_y_quartiles": sorted(r[1] for r in rows)[len(rows) // 4:len(rows) // 4 * 3 + 1:len(rows) // 4]
              if len(rows) >= 4 else None,
              "relay_summary": relay["summary"]}
    print(json.dumps({k: v for k, v in result.items() if k != "relay_summary"}))
    save("anisotropy", "EXP-0024", {"count": count}, result, args.binary)


def budget(args):
    """EXP-0025: the budget rule of the protocol."""
    task = task_named("exploration_v2_floor", 1000)
    result = evaluate(args.binary, {"kind": "frontier"}, task, seed_pairs(M5_MASTER, "diagnostic", 32), args.workers)
    episodes = result["episodes"]
    length = max(len(e["curve"]) for e in episodes)
    curve = [sum(e["curve"][i] if i < len(e["curve"]) else e["new_cells"] for e in episodes) / len(episodes)
             for i in range(length)]
    final = result["summary"]["mean_new_cells"]
    chosen = next((i + 1) * 50 for i, value in enumerate(curve) if value >= .8 * final)
    output = {"curve_every_50": [round(v, 1) for v in curve], "mean_at_1000": final, "budget": chosen,
              "summary": result["summary"]}
    print(json.dumps({k: v for k, v in output.items() if k != "summary"}))
    show("frontier", result["summary"])
    save("budget", "EXP-0025", {"budget": 1000, "seeds": 32}, output, args.binary)


def control_suite(args):
    """EXP-0026: every control on validation, plus the reference heuristic."""
    task = task_named("exploration_v2_floor", args.budget)
    pairs = seed_pairs(M5_MASTER, "validation", args.count)
    specs = {**controls(), **m4_specs()}
    results = {}
    for name, spec in specs.items():
        results[name] = evaluate(args.binary, spec, task, pairs, args.workers)
        show(name, results[name]["summary"])
    reference = task_named("exploration_v2_full_reference", args.budget)
    results["heuristic_full_reference"] = evaluate(args.binary, {"kind": "heuristic_full"}, reference,
                                                   pairs, args.workers)
    show("heuristic_full_reference", results["heuristic_full_reference"]["summary"])
    output = {name: {"summary": value["summary"], "by_seed": by_seed(value)} for name, value in results.items()}
    (OUT / "validation-baseline-random.json").write_text(json.dumps(by_seed(results["random"])))
    run = save(f"controls-{args.split_label}", "EXP-0026", {"budget": args.budget, "seeds": args.count}, output, args.binary)
    print("run", run)


def write_config(name, **overrides):
    """Pilot configs are generated here, so this file versions them."""
    train = {"experiment": "EXP-0027", "algorithm": "linear_td_v2", "task": "exploration_v2_floor",
             "budget": 250, "encoder": "rel_v3", "disabled": [], "reward": "cells_v2",
             "master_seed": M5_MASTER, "run": 0, "episodes": 200, "alpha": .05, "gamma": .95,
             "epsilon_start": .3, "epsilon_end": .05, "epsilon_decay_episodes": 150,
             "validation_every": 50, "validation_episodes": 16, "checkpoint_every": 5,
             "periodic_checkpoint_every": 50, "keep_periodic": 8,
             "validation_baseline": "data/m5/validation-baseline-random.json"}
    train.update(overrides)
    limits = {"episode_seconds": 120.0, "memory_mb": 512, "max_seconds": 1800, "max_rss_mb": 1024,
              "validation_workers": 1}

    def value(item):
        if isinstance(item, str):
            return json.dumps(item)
        if isinstance(item, list):
            return json.dumps(item)
        return repr(item)

    lines = ["[train]"] + [f"{key} = {value(item)}" for key, item in train.items()]
    lines += ["", "[limits]"] + [f"{key} = {value(item)}" for key, item in limits.items()]
    path = OUT / "configs" / f"{name}.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def pilot(args):
    """Write the config matrix of one experiment and print the paths."""
    if args.name == "td":
        for encoder in ("rel_v3", "abs_v2", "ego_patch_v1"):
            for run in range(3):
                print(write_config(f"td-{encoder}-r{run}", encoder=encoder, run=run))
    if args.name == "hp":
        for alpha, gamma in ((.02, .95), (.1, .95), (.05, .9), (.05, .98)):
            for run in (0, 1):
                print(write_config(f"hp-a{alpha}-g{gamma}-r{run}", experiment="EXP-0027", alpha=alpha,
                                   gamma=gamma, run=run))
    if args.name == "ablation":
        variants = {"dir": {"encoder": "rel_v3_dir"}, "novisits": {"disabled": ["visits"]},
                    "nofrontier": {"disabled": ["frontier"]}, "nofailed": {"disabled": ["failed"]},
                    "shaped": {"reward": "cells_shaped_v2"}}
        for label, overrides in variants.items():
            for run in range(3):
                print(write_config(f"abl-{label}-r{run}", experiment="EXP-0030", run=run, **overrides))
    if args.name == "warm":
        for label, disabled in (("rel_full", []), ("rel_no_planner", ["frontier"])):
            for run in range(3):
                model = json.loads((OUT / f"bc/{label}-r{run}.json").read_text())
                scaled = OUT / f"bc/scaled-{label}-r{run}.json"
                scaled.write_text(json.dumps({"weights": [w * model["scale"] for w in model["weights"]]}))
                print(write_config(f"warm-{label}-r{run}", experiment="EXP-0029", run=run, disabled=disabled,
                                   init=str(scaled), epsilon_start=.1, epsilon_end=.02, alpha=.02))


def assess(args):
    """Best and latest checkpoints of each campaign on the validation set."""
    task = task_named("exploration_v2_floor", 250)
    pairs = seed_pairs(M5_MASTER, "validation", args.count)
    output = {}
    for directory in sorted(Path(args.root).iterdir()):
        if not (directory / "manifest.json").is_file() or not directory.name.startswith(tuple(args.prefix)):
            continue
        settings = json.loads((directory / "manifest.json").read_text())["identity"]["config"]["train"]
        entry = {}
        for checkpoint in ("best", "latest"):
            path = directory / f"checkpoints/{checkpoint}.json"
            if not path.is_file():
                continue
            spec = {"kind": "linear", "encoder": settings["encoder"], "disabled": settings.get("disabled", []),
                    "checkpoint": str(path)}
            result = evaluate(args.binary, spec, task, pairs, args.workers)
            entry[checkpoint] = {"summary": result["summary"], "by_seed": by_seed(result)}
        output[directory.name] = entry
        print(directory.name, {k: (round(v["summary"]["mean_new_cells"], 1), v["summary"]["invalid"],
                                   round(v["summary"]["stalled_rate"], 3)) for k, v in entry.items()}, flush=True)
    existing = OUT / "assess.json"
    merged = json.loads(existing.read_text()) if existing.is_file() else {}
    merged.update(output)
    existing.write_text(json.dumps(merged))


def launch(args):


def launch(args):
    """Run a list of campaign configs with a bounded number of concurrent processes."""
    configs = [Path(path) for path in args.configs]
    active, started = [], time.monotonic()
    while configs or active:
        while configs and len(active) < args.parallel:
            config = configs.pop(0)
            directory = Path(args.root) / config.stem
            print("start", config.stem, flush=True)
            active.append((config.stem, subprocess.Popen(
                [sys.executable, "-m", "yendorlab.app", "campaign", "start", "--config", str(config),
                 "--directory", str(directory), "--binary", str(args.binary)],
                stdout=subprocess.PIPE, text=True)))
        time.sleep(1)
        for item in list(active):
            name, process = item
            if process.poll() is not None:
                print(name, process.stdout.read().strip()[:400], flush=True)
                active.remove(item)
    print("elapsed", time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path(".local/BrogueCE/bin/brogue"))
    parser.add_argument("--workers", type=int, default=4)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("m4audit")
    item = sub.add_parser("anisotropy")
    item.add_argument("--count", type=int, default=64)
    sub.add_parser("budget")
    item = sub.add_parser("controls")
    item.add_argument("--budget", type=int, required=True)
    item.add_argument("--count", type=int, default=40)
    item.add_argument("--split-label", default="validation")
    item = sub.add_parser("pilot")
    item.add_argument("name")
    item = sub.add_parser("assess")
    item.add_argument("prefix", nargs="+")
    item.add_argument("--count", type=int, default=40)
    item.add_argument("--root", default="data/m5/campaigns")
    item = sub.add_parser("launch")
    item.add_argument("configs", nargs="+")
    item.add_argument("--parallel", type=int, default=4)
    item.add_argument("--root", default="data/m5/campaigns")
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    {"m4audit": m4audit, "anisotropy": anisotropy, "budget": budget, "controls": control_suite,
     "launch": launch, "pilot": pilot, "assess": assess}[args.command](args)


if __name__ == "__main__":
    main()
