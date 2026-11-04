"""Aggregate raw M5 outputs under data/ into the committed docs/results/m5-summary.json."""

import glob
import json
import os
import sqlite3
from pathlib import Path

from yendorlab.stats import hierarchical_interval

DATA = Path("data/m5")


def group(assess, prefix, runs=3):
    names = [f"{prefix}-r{run}" for run in range(runs) if f"{prefix}-r{run}" in assess]
    best = [assess[name]["best"]["summary"]["mean_new_cells"] for name in names]
    latest = [assess[name]["latest"]["summary"]["mean_new_cells"] for name in names]
    stalled = [assess[name]["best"]["summary"]["stalled_rate"] for name in names]
    deaths = [assess[name]["best"]["summary"]["deaths"] for name in names]
    return {"runs": len(names), "best_per_run": [round(v, 1) for v in best], "best_mean": round(sum(best) / len(best), 1),
            "latest_per_run": [round(v, 1) for v in latest], "latest_mean": round(sum(latest) / len(latest), 1),
            "stalled_rate_mean": round(sum(stalled) / len(stalled), 3), "deaths_of_40": deaths,
            "hierarchical_95_best": [round(v, 1) for v in hierarchical_interval(
                [list(assess[name]["best"]["by_seed"].values()) for name in names], 2000)]}


def campaigns():
    rows = []
    for path in sorted(glob.glob(str(DATA / "campaigns/*/manifest.json"))):
        directory = Path(path).parent
        manifest = json.loads(Path(path).read_text())
        journal = sqlite3.connect(directory / "journal.sqlite3")
        episodes, failures, seconds, decisions = journal.execute(
            "SELECT COUNT(*), SUM(valid=0), SUM(duration_s), SUM(decisions) FROM episodes WHERE superseded=0").fetchone()
        journal.close()
        rows.append({"campaign": directory.name, "experiment": manifest["identity"]["config"]["train"]["experiment"],
                     "status": manifest.get("status"), "revision": manifest["yendorlab_revision"][:7],
                     "worktree_dirty": manifest["worktree_dirty"], "episodes": episodes,
                     "operational_failures": failures, "episode_seconds": round(seconds, 1), "decisions": decisions})
    return rows


def imitation():
    output = {}
    for run in range(3):
        result = json.loads((DATA / f"bc/result-r{run}.json").read_text())
        for label, model in result["models"].items():
            entry = output.setdefault(label, {"online": [], "heldout_accuracy": [], "train_accuracy": [],
                                              "fit_cpu_s": [], "dagger_online": [],
                                              "accuracy_on_own_states": []})
            entry["online"].append(round(model["online"]["mean_new_cells"], 1))
            entry["heldout_accuracy"].append(round(model["heldout"]["set_accuracy"], 3))
            entry["train_accuracy"].append(round(model["train"]["set_accuracy"], 3))
            entry["fit_cpu_s"].append(round(model["fit_cpu_s"], 1))
            entry["dagger_online"].append([round(s["online"]["mean_new_cells"], 1) for s in model["dagger"]])
            entry["accuracy_on_own_states"].append(round(model["dagger"][0]["accuracy_on_own_states_before"]["set_accuracy"], 3))
        output.setdefault("demonstrations", []).append(result["demonstrations"])
    return output


def capacity():
    output = {}
    for run in range(3):
        result = json.loads((DATA / f"mlp/result-r{run}.json").read_text())
        for label, model in result["models"].items():
            entry = output.setdefault(label, {"mlp_online": [], "linear_online": [], "mlp_heldout": [],
                                              "linear_heldout": [], "mlp_fit_cpu_s": []})
            entry["mlp_online"].append(round(model["mlp_online"]["mean_new_cells"], 1))
            entry["linear_online"].append(round(model["linear_online"]["mean_new_cells"], 1))
            entry["mlp_heldout"].append(round(model["mlp_heldout"], 3))
            entry["linear_heldout"].append(round(model["linear_heldout"], 3))
            entry["mlp_fit_cpu_s"].append(round(model["mlp_fit_cpu_s"], 1))
    return output


def suite(name):
    data = json.loads((DATA / f"suite-{name}.json").read_text())
    tables = {}
    for section in ("floor", "relay", "multilevel", "reference"):
        tables[section] = {}
        for policy, value in data[section].items():
            summary = value["summary"]
            tables[section][policy] = {
                **{key: summary[key] for key in ("valid", "invalid", "unavailable", "deaths", "depth_exits", "reasons",
                                                 "mean_new_cells", "mean_turns", "mean_events", "mean_decisions",
                                                 "stalled_rate", "revisit_rate", "cells_per_decision",
                                                 "cells_per_turn", "cells_per_event", "action_share", "wall_episode_s")},
                "bootstrap_95": value["bootstrap_95"], "interquartile_mean": value["interquartile_mean"],
                "action_entropy": value["action_entropy"]}
    return {"split": data["split"], "count": data["count"], "tables": tables, "comparisons": data["comparisons"],
            "runs": data["runs"], "difficulty_quartiles": data["difficulty_quartiles"]}


def main():
    assess = json.loads((DATA / "assess.json").read_text())
    controls = json.loads((DATA / "controls-validation.json").read_text())
    groups = {label: group(assess, label) for label in (
        "td-rel_v3", "td-abs_v2", "td-ego_patch_v1", "abl-dir", "abl-novisits", "abl-nofrontier", "abl-nofailed",
        "abl-shaped", "warm-rel_full", "warm-rel_no_planner", "stage2-base", "stage2-cold", "stage2-novisits")}
    scan = {label: group(assess, label, 2) for label in ("hp-a0.02-g0.95", "hp-a0.1-g0.95", "hp-a0.05-g0.9", "hp-a0.05-g0.98")}
    database = sqlite3.connect("data/experiments.sqlite3")
    runs = [{"run_id": row[0], "experiment": row[1], "utc": row[2]} for row in database.execute(
        "SELECT run_id, experiment_id, started_utc FROM runs WHERE experiment_id IN "
        "('EXP-0023','EXP-0024','EXP-0025','EXP-0026','EXP-0028','EXP-0028b','EXP-0033','EXP-0036') ORDER BY started_utc")]
    summary = {"validation_controls": {name: {"mean_new_cells": value["summary"]["mean_new_cells"],
                                              "stalled_rate": value["summary"]["stalled_rate"],
                                              "deaths": value["summary"]["deaths"]}
                                       for name, value in controls.items()},
               "m4_audit": json.loads((DATA / "m4audit.json").read_text()),
               "anisotropy": {k: v for k, v in json.loads((DATA / "anisotropy.json").read_text()).items()
                              if k != "relay_summary"},
               "budget": {k: v for k, v in json.loads((DATA / "budget.json").read_text()).items() if k != "summary"},
               "td_groups_validation_40": groups, "hyperparameter_scan": scan,
               "imitation": imitation(), "capacity": capacity(),
               "campaigns": campaigns(), "bench": json.loads((DATA / "bench.json").read_text()),
               "soak": json.loads((DATA / "soak.json").read_text()),
               "diagnostic": suite("diagnostic"), "final": suite("final"), "sqlite_runs": runs}
    target = Path("docs/results/m5-summary.json")
    target.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(os.path.getsize(target))
    print(json.dumps({k: {a: v[a] for a in ("best_per_run", "best_mean", "latest_mean", "stalled_rate_mean",
                                            "hierarchical_95_best", "deaths_of_40")}
                      for k, v in {**groups, **scan}.items()}, indent=0))
    totals = summary["campaigns"]
    print(len(totals), sum(r["episodes"] for r in totals), sum(r["decisions"] for r in totals),
          sum(r["operational_failures"] for r in totals), sum(r["episode_seconds"] for r in totals))
    print({r["campaign"]: (r["decisions"], r["episode_seconds"]) for r in totals if r["campaign"].endswith("r0") and r["campaign"][:4] in ("td-r", "warm")})
    print(summary["imitation"]["demonstrations"], {k: v for k, v in summary["imitation"].items() if k != "demonstrations"})
    print(summary["capacity"])
    print(runs)


if __name__ == "__main__":
    main()
