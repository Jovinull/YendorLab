"""Small-sample statistics. Game seeds pair comparisons; training runs are the replicates."""

import random
import statistics


def percentile(ordered, q):
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def interquartile_mean(values):
    ordered = sorted(values)
    cut = len(ordered) // 4
    kept = ordered[cut:len(ordered) - cut] or ordered
    return sum(kept) / len(kept)


def bootstrap_mean_interval(values, resamples=10000, seed=20261103, level=0.95):
    rng = random.Random(seed)
    count = len(values)
    means = sorted(sum(rng.choices(values, k=count)) / count for _ in range(resamples))
    tail = (1 - level) / 2
    return [percentile(means, tail), percentile(means, 1 - tail)]


def paired_comparison(candidate, baseline, resamples=10000, seed=20261103):
    """Paired over seeds present in both dicts {seed: metric}; reports what was dropped."""
    shared = sorted(set(candidate) & set(baseline))
    differences = [candidate[key] - baseline[key] for key in shared]
    if not differences:
        return {"pairs": 0}
    rng = random.Random(seed)
    observed = abs(sum(differences))
    extreme = sum(abs(sum(d if rng.random() < .5 else -d for d in differences)) >= observed
                  for _ in range(resamples))
    return {"pairs": len(differences),
            "dropped": len(set(candidate) ^ set(baseline)),
            "mean_difference": statistics.mean(differences),
            "median_difference": statistics.median(differences),
            "bootstrap_95": bootstrap_mean_interval(differences, resamples, seed),
            "positive": sum(d > 0 for d in differences),
            "zero": sum(d == 0 for d in differences),
            "negative": sum(d < 0 for d in differences),
            "sign_flip_p": (extreme + 1) / (resamples + 1)}


def holm(p_values):
    """Holm step-down adjustment for a prespecified family, keyed by comparison name."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (total - rank)))
        adjusted[name] = running
    return adjusted


def hierarchical_interval(runs, resamples=10000, seed=20261103, level=0.95):
    """Bootstrap runs, then seeds inside each run. `runs` is a list of value lists.

    With three runs the outer resampling is coarse: this describes uncertainty, it
    does not support claims about the population of possible training runs.
    """
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        chosen = rng.choices(runs, k=len(runs))
        total = sum(sum(rng.choices(run, k=len(run))) / len(run) for run in chosen)
        means.append(total / len(chosen))
    means.sort()
    tail = (1 - level) / 2
    return [percentile(means, tail), percentile(means, 1 - tail)]
