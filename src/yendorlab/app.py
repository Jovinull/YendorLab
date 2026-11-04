"""Operational YendorLab CLI for bounded Brogue experiments."""

import argparse
import hashlib
import json
import platform
import subprocess
import tomllib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .agents import FixedDirectionAgent, HeuristicAgent, QAgent, RandomAgent
from .evaluation import run_policy_episode, seed_for, summarize
from .experiments import list_runs, record, show_run
from .training import load_policy_checkpoint, train
from .skill import LinearFrontierAgent, load_skill_checkpoint, train_skill


def agent_named(name, seed, checkpoint=None):
    if name == "random":
        return RandomAgent(seed)
    if name == "heuristic":
        return HeuristicAgent(seed)
    if name == "initial_q":
        return QAgent(seed=seed)
    if name == "q":
        if checkpoint is None:
            raise ValueError("q agent requires --checkpoint")
        return QAgent(load_policy_checkpoint(checkpoint), seed=seed)
    if name == "linear":
        if checkpoint is None:
            raise ValueError("linear agent requires --checkpoint")
        return LinearFrontierAgent(load_skill_checkpoint(checkpoint)["weights"], seed=seed)
    if name == "initial_linear":
        return LinearFrontierAgent(seed=seed)
    if name == "fixed_k":
        return FixedDirectionAgent("k")
    raise ValueError(name)


def common(parser):
    parser.add_argument("--binary", type=Path, default=Path(".local/BrogueCE/bin/brogue"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--variant", choices=["classic", "rapid", "bullet"], default="classic")
    parser.add_argument("--max-events", type=int, default=200)
    parser.add_argument("--max-seconds", type=float, default=20)
    parser.add_argument("--max-no-progress", type=int, default=40)
    parser.add_argument("--memory-mb", type=int, default=512)
    parser.add_argument("--workers", type=int, choices=[1, 2], default=1)


def main():
    parser = argparse.ArgumentParser(prog="yendorlab")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--binary", type=Path, default=Path(".local/BrogueCE/bin/brogue"))
    play = sub.add_parser("play")
    common(play)
    play.add_argument("--agent", choices=["random", "heuristic", "initial_q", "q", "initial_linear", "linear", "fixed_k"], default="heuristic")
    play.add_argument("--checkpoint", type=Path)
    play.add_argument("--seed", type=int, default=123)
    evaluate = sub.add_parser("evaluate")
    common(evaluate)
    evaluate.add_argument("--agent", choices=["random", "heuristic", "initial_q", "q", "initial_linear", "linear", "fixed_k"], required=True)
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--episodes", type=int, default=3)
    evaluate.add_argument("--master-seed", type=int, default=20261030)
    evaluate.add_argument("--split", choices=["validation", "final"], default="validation")
    evaluate.add_argument("--reward", choices=["canonical", "exploration_v1"], default="canonical")
    evaluate.add_argument("--experiment", default="EXP-0012")
    benchmark = sub.add_parser("benchmark")
    common(benchmark)
    benchmark.add_argument("--episodes", type=int, default=3)
    benchmark.add_argument("--master-seed", type=int, default=20261030)
    training = sub.add_parser("train")
    training.add_argument("--config", type=Path, default=Path("configs/train-smoke.toml"))
    training.add_argument("--binary", type=Path, default=Path(".local/BrogueCE/bin/brogue"))
    training.add_argument("--data-dir", type=Path, default=Path("data"))
    training.add_argument("--checkpoint", type=Path, default=Path("data/checkpoints/maia-q.json"))
    training.add_argument("--episodes", type=int)
    training.add_argument("--total-seconds", type=float)
    training.add_argument("--master-seed", type=int)
    training.add_argument("--resume", action="store_true")
    experiments = sub.add_parser("experiments")
    experiments.add_argument("operation", choices=["list", "show"])
    experiments.add_argument("run_id", nargs="?")
    experiments.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    if args.command == "doctor":
        patch = Path("native/patches/0001-yendorlab-stream.patch")
        version = subprocess.run([str(args.binary.resolve()), "--version"],
                                 cwd=args.binary.resolve().parent, capture_output=True,
                                 text=True, timeout=5).stdout.strip() if args.binary.is_file() else None
        print(json.dumps({"binary_exists": args.binary.is_file(), "platform": platform.platform(),
                          "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
                          "python": platform.python_version(), "brogue_version": version}))
        return
    if args.command == "experiments":
        if args.operation == "list":
            print(json.dumps(list_runs(args.data_dir)))
        else:
            if not args.run_id:
                parser.error("show requires run_id")
            print(json.dumps(show_run(args.data_dir, args.run_id)))
        return
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"missing native binary: {binary}; run bash scripts/build-wsl.sh")
    if args.command in {"play", "evaluate", "benchmark"}:
        if args.max_events < 1 or args.max_seconds <= 0 or args.max_no_progress < 1 or args.memory_mb < 64:
            parser.error("event, time, progress and memory limits must be positive")
        if args.command in {"evaluate", "benchmark"} and args.episodes < 1:
            parser.error("episodes must be positive")

    if args.command == "train":
        config = tomllib.loads(args.config.read_text())["train"]
        if args.episodes is not None:
            config["episodes"] = args.episodes
        if args.total_seconds is not None:
            config["total_seconds"] = args.total_seconds
        if args.master_seed is not None:
            config["master_seed"] = args.master_seed
        if config["episodes"] <= 0 or config["total_seconds"] <= 0:
            parser.error("training limits must be positive")
        result = (train_skill(binary, config, args.checkpoint.resolve(), args.resume)
                  if config.get("algorithm") == "linear_frontier_td_v1" else
                  train(binary, config, args.checkpoint.resolve(), args.resume))
        run_id = record(args.data_dir, config["experiment"], config, result, binary)
        print(json.dumps({"run_id": run_id, **{k: v for k, v in result.items() if k != "episodes"}}))
        return

    if args.command == "play":
        agent = agent_named(args.agent, args.seed + 100000, args.checkpoint)
        episode = run_policy_episode(binary, agent, args.seed, args.variant, "canonical",
                                     args.max_events, args.max_seconds, args.max_no_progress, args.memory_mb)
        run_id = record(args.data_dir, "EXP-0008", {"agent": args.agent, "seed": args.seed,
                                                    "variant": args.variant}, episode, binary)
        print(json.dumps({"run_id": run_id, **episode}))
        return

    episodes = []
    name = "random" if args.command == "benchmark" else args.agent
    split = "validation" if args.command == "benchmark" else args.split
    reward = "canonical" if args.command == "benchmark" else args.reward
    def execute(index):
        seed = seed_for(args.master_seed, split, index)
        agent = agent_named(name, seed_for(args.master_seed, split, index + 100000),
                            None if args.command == "benchmark" else args.checkpoint)
        return run_policy_episode(binary, agent, seed, args.variant, reward,
                                  args.max_events, args.max_seconds,
                                  args.max_no_progress, args.memory_mb)
    wall_start = time.monotonic()
    if args.workers == 1:
        episodes = [execute(index) for index in range(args.episodes)]
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            episodes = list(pool.map(execute, range(args.episodes)))
    result = summarize(episodes)
    result["wall_duration_s"] = time.monotonic() - wall_start
    result["wall_events_per_s"] = result["events"] / result["wall_duration_s"]
    experiment = "EXP-0009" if args.command == "benchmark" else args.experiment
    run_id = record(args.data_dir, experiment,
                    {"agent": name, "variant": args.variant, "split": split, "reward": reward,
                     "master_seed": args.master_seed, "max_events": args.max_events,
                     "max_seconds": args.max_seconds, "max_no_progress": args.max_no_progress,
                     "memory_mb": args.memory_mb, "workers": args.workers,
                     "checkpoint": str(args.checkpoint) if args.command != "benchmark" and args.checkpoint else None,
                     "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
                     if args.command != "benchmark" and args.checkpoint else None},
                    result, binary)
    print(json.dumps({"run_id": run_id, **{k: v for k, v in result.items() if k != "episodes"}}))


if __name__ == "__main__":
    main()
