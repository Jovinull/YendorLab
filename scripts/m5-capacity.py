"""EXP-0028b: does capacity help where the linear model underfit?

Linear imitation without the planner signal had equal train and held-out accuracy,
which points at capacity, not data. Same demonstrations, same seeds: linear vs a
small MLP on the no-planner encoders. Budget is fixed in advance; a null result stands.
"""

import argparse
import json
import time
from pathlib import Path

from yendorlab.benchmark import M5_MASTER, by_seed, evaluate, seed_pairs
from yendorlab.encoders import Encoder
from yendorlab.experiments import record
from yendorlab.learners import collect, imitation_metrics, mlp_metrics, train_mlp, train_softmax
from yendorlab.policies import FrontierMoves
from yendorlab.tasks import m5_seed, task_named

OUT = Path("data/m5/mlp")
MODELS = {"rel_no_planner": (("rel_v3", ("frontier",)), 24, 12), "patch": (("ego_patch_v1", ()), 16, 6)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--run", type=int, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    task = task_named("exploration_v2_floor", 250)
    teacher = FrontierMoves(0)
    encoders = {label: Encoder(*definition) for label, (definition, _, _) in MODELS.items()}
    demos, heldout = [], []
    for target, stream, count in ((demos, "demo", 60), (heldout, "heldout", 10)):
        for index in range(count):
            seed = m5_seed(M5_MASTER, "train", index, f"{stream}{args.run}")
            policy_seed = m5_seed(M5_MASTER, "train", index, f"{stream}{args.run}:policy")
            outcome, samples = collect(binary, FrontierMoves(policy_seed), teacher, list(encoders.values()),
                                       seed, task, policy_seed)
            target.extend(samples)
    result = {"run": args.run, "samples": len(demos), "models": {}}
    for label, encoder in encoders.items():
        key = encoder.name + ":" + ",".join(encoder.disabled)
        hidden, epochs = MODELS[label][1:]
        linear, _ = train_softmax(demos, key, encoder.dimension, epochs=12, seed=args.run)
        started = time.process_time()
        model, history = train_mlp(demos, key, encoder.dimension, hidden, epochs, seed=args.run)
        fit = time.process_time() - started
        path = OUT / f"{label}-r{args.run}.json"
        path.write_text(json.dumps({"model": model.to_dict(), "encoder": encoder.spec()}))
        spec = {"kind": "mlp", "encoder": encoder.name, "disabled": list(encoder.disabled), "checkpoint": str(path)}
        online = evaluate(binary, spec, task, seed_pairs(M5_MASTER, "validation", 40), 2)
        linear_online = evaluate(binary, {"kind": "linear", "encoder": encoder.name,
                                          "disabled": list(encoder.disabled), "weights": linear}, task,
                                 seed_pairs(M5_MASTER, "validation", 40), 2)
        result["models"][label] = {
            "hidden": hidden, "epochs": epochs, "mlp_fit_cpu_s": fit, "loss": history,
            "mlp_train": mlp_metrics(model, demos, key)["set_accuracy"],
            "mlp_heldout": mlp_metrics(model, heldout, key)["set_accuracy"],
            "linear_train": imitation_metrics(linear, demos, key)["set_accuracy"],
            "linear_heldout": imitation_metrics(linear, heldout, key)["set_accuracy"],
            "mlp_online": online["summary"], "mlp_by_seed": by_seed(online),
            "linear_online": linear_online["summary"], "linear_by_seed": by_seed(linear_online)}
        print(label, args.run, {k: round(v, 3) for k, v in result["models"][label].items()
                                if isinstance(v, float)}, online["summary"]["mean_new_cells"],
              linear_online["summary"]["mean_new_cells"], flush=True)
    (OUT / f"result-r{args.run}.json").write_text(json.dumps(result))
    print("run", record(Path("data"), "EXP-0028b", {"run": args.run}, result, binary))


if __name__ == "__main__":
    main()
