"""Reusable benchmark suite: equal seeds, budgets and policy RNG for every policy."""

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

from .policies import build_policy, relay_guide
from .tasks import MOVES, m5_seed, metric_summary, run_with_retry

M5_MASTER = 20261103


def controls():
    """Policies with exactly the eight-move contract and shared perception."""
    specs = {"random": {"kind": "random"}, "frontier_moves": {"kind": "frontier"}}
    specs.update({f"fixed_{move}": {"kind": "fixed", "action": move} for move in MOVES})
    return specs


def _job(arguments):
    binary, spec, task, seed, policy_seed = arguments
    guide = relay_guide() if task.preroll_depth else None
    return run_with_retry(binary, lambda: build_policy(spec, policy_seed), seed, task,
                          policy_seed, guide_factory=guide)


def seed_pairs(master, split, count, start=0):
    return [(m5_seed(master, split, index), m5_seed(master, split, index, "policy"))
            for index in range(start, start + count)]


def evaluate(binary, spec, task, pairs, workers=1):
    jobs = [(binary, spec, task, seed, policy_seed) for seed, policy_seed in pairs]
    started = time.monotonic()
    if workers == 1:
        episodes = [_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as pool:
            episodes = list(pool.map(_job, jobs, chunksize=1))
    summary = metric_summary(episodes)
    summary["wall_s"] = time.monotonic() - started
    summary["workers"] = workers
    return {"summary": summary, "episodes": episodes}


def by_seed(result):
    """{seed: metric} for valid episodes, the input of a paired comparison."""
    return {e["seed"]: e["new_cells"] for e in result["episodes"] if e["valid"]}
