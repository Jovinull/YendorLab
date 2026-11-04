"""EXP-0028: behavior cloning and DAgger from the eight-move planner, one independent run.

Compute is reported in decisions and CPU seconds, separately for demonstrations
and for fitting, so wall time on a loaded laptop does not decide the comparison.
"""

import argparse
import json
import resource
import time
from pathlib import Path

from yendorlab.benchmark import M5_MASTER, by_seed, evaluate, seed_pairs
from yendorlab.encoders import Encoder
from yendorlab.experiments import record
from yendorlab.learners import (collect, imitation_metrics, scale_to_returns, train_softmax)
from yendorlab.policies import FrontierMoves, LinearPolicy
from yendorlab.tasks import m5_seed, task_named

OUT = Path("data/m5/bc")
GAMMA = .95
MODELS = {"rel_full": ("rel_v3", ()), "rel_no_planner": ("rel_v3", ("frontier",)),
          "patch": ("ego_patch_v1", ())}


def cpu_seconds():
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def key_of(encoder):
    return encoder.name + ":" + ",".join(encoder.disabled)


def rollouts(binary, task, make_policy, teacher, encoders, stream, count, run):
    samples, records = [], []
    for index in range(count):
        seed = m5_seed(M5_MASTER, "train", index, f"{stream}{run}")
        policy_seed = m5_seed(M5_MASTER, "train", index, f"{stream}{run}:policy")
        outcome, new = collect(binary, make_policy(policy_seed), teacher, encoders, seed, task, policy_seed)
        if not outcome["valid"]:
            print("invalid rollout discarded and reported", outcome["reason"], flush=True)
        else:
            for sample in new:
                sample["episode"] = f"{stream}{index}"
            samples.extend(new)
        records.append(outcome)
    return samples, records


def returns_of(samples):
    """Discounted future reward of the task, from the known-cell count at each decision."""
    values = [0.0] * len(samples)
    following = 0.0
    for position in range(len(samples) - 1, -1, -1):
        last = position + 1 == len(samples) or samples[position + 1]["episode"] != samples[position]["episode"]
        if last:
            following = 0.0
            values[position] = 0.0
            continue
        gained = max(0, samples[position + 1]["known"] - samples[position]["known"]) * .1
        following = gained + GAMMA * following
        values[position] = following
    return values


def online(binary, encoder, weights, task, count, workers):
    spec = {"kind": "linear", "encoder": encoder.name, "disabled": list(encoder.disabled), "weights": weights}
    return evaluate(binary, spec, task, seed_pairs(M5_MASTER, "validation", count), workers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--demos", type=int, default=60)
    parser.add_argument("--heldout", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--dagger-iterations", type=int, default=3)
    parser.add_argument("--dagger-episodes", type=int, default=20)
    parser.add_argument("--validation", type=int, default=40)
    args = parser.parse_args()
    binary = args.binary.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    task = task_named("exploration_v2_floor", 250)
    encoders = [Encoder(name, disabled) for name, disabled in MODELS.values()]
    teacher = FrontierMoves(0)
    result = {"run": args.run, "models": {}}

    before = cpu_seconds()
    demos, records = rollouts(binary, task, lambda seed: FrontierMoves(seed), teacher, encoders,
                              "demo", args.demos, args.run)
    result["demonstrations"] = {"episodes": len(records), "samples": len(demos),
                                "invalid": sum(not r["valid"] for r in records),
                                "cpu_s": cpu_seconds() - before}
    heldout, _ = rollouts(binary, task, lambda seed: FrontierMoves(seed), teacher, encoders,
                          "heldout", args.heldout, args.run)
    print("demos", len(demos), "heldout", len(heldout), flush=True)
    returns = returns_of(demos)

    for (label, _), encoder in zip(MODELS.items(), encoders):
        key = key_of(encoder)
        before = time.process_time()
        weights, history = train_softmax(demos, key, encoder.dimension, epochs=args.epochs, seed=args.run)
        fit_s = time.process_time() - before
        evaluation = online(binary, encoder, weights, task, args.validation, 2)
        model = {"fit_cpu_s": fit_s, "loss_history": history,
                 "train": imitation_metrics(weights, demos, key),
                 "heldout": imitation_metrics(weights, heldout, key),
                 "online": evaluation["summary"], "online_by_seed": by_seed(evaluation),
                 "scale_to_returns": scale_to_returns(weights, demos, key, returns)}
        Path(OUT / f"{label}-r{args.run}.json").write_text(json.dumps(
            {"weights": weights, "encoder": encoder.spec(), "scale": model["scale_to_returns"]}))

        # DAgger: roll out the learner, label its own states with the planner.
        aggregate = list(demos)
        steps = []
        for iteration in range(args.dagger_iterations):
            before = cpu_seconds()
            stream = f"dagger-{label}-{iteration}-"
            current = list(weights)
            new, _ = rollouts(binary, task, lambda seed: LinearPolicy(encoder, current, seed), teacher,
                              [encoder], stream, args.dagger_episodes, args.run)
            collected = cpu_seconds() - before
            own_states = imitation_metrics(weights, new, key)
            aggregate.extend(new)
            before = time.process_time()
            weights, _ = train_softmax(aggregate, key, encoder.dimension, epochs=max(4, args.epochs // 2),
                                       seed=args.run + iteration + 1)
            fit = time.process_time() - before
            evaluation = online(binary, encoder, weights, task, args.validation, 2)
            steps.append({"iteration": iteration + 1, "samples": len(aggregate), "collect_cpu_s": collected,
                          "fit_cpu_s": fit, "accuracy_on_own_states_before": own_states,
                          "online": evaluation["summary"], "online_by_seed": by_seed(evaluation)})
            print(label, "dagger", iteration + 1, evaluation["summary"]["mean_new_cells"], flush=True)
        Path(OUT / f"{label}-dagger-r{args.run}.json").write_text(json.dumps(
            {"weights": weights, "encoder": encoder.spec()}))
        model["dagger"] = steps
        result["models"][label] = model
        print(label, "bc", model["online"]["mean_new_cells"], model["heldout"], flush=True)
    (OUT / f"result-r{args.run}.json").write_text(json.dumps(result))
    run_id = record(Path("data"), "EXP-0028", {"run": args.run, "demos": args.demos, "epochs": args.epochs,
                                               "budget": 250}, result, binary)
    print("run", run_id)


if __name__ == "__main__":
    main()
