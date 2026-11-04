"""CPU learners over sparse per-move features: linear TD and softmax imitation."""

import gzip
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from .tasks import MOVES, run_episode


@dataclass(frozen=True)
class RewardScheme:
    """Task reward. The evaluation metric never reads this."""
    name: str
    cell: float = 0.1
    death: float = -1.0
    stall: float = 0.0
    revisit: float = 0.0


REWARDS = {"cells_v2": RewardScheme("cells_v2"),
           "cells_only_v2": RewardScheme("cells_only_v2", death=0.0),
           "cells_shaped_v2": RewardScheme("cells_shaped_v2", stall=-0.1, revisit=-0.02)}


class TDTrainer:
    """TD(0) on the shared linear scorer, with a normalized, clipped step."""

    def __init__(self, policy, alpha, gamma, reward: RewardScheme, error_clip=10.0, weight_clip=50.0):
        self.policy = policy
        self.alpha = alpha
        self.gamma = gamma
        self.reward = reward
        self.error_clip = error_clip
        self.weight_clip = weight_clip
        self.updates = 0
        self.changed = 0
        self.absolute_error = 0.0

    def cell_reward(self, cells):
        return cells * self.reward.cell

    def death_reward(self):
        return self.reward.death

    def step_reward(self, perception):
        return (self.reward.stall if perception.stalled else 0.0) + \
            (self.reward.revisit if perception.revisit else 0.0)

    def _update(self, reward, following):
        features = self.policy.last_features
        if features is None:
            return
        policy = self.policy
        target = reward
        if following is not None:
            encoded = policy.encoder(following)
            target += self.gamma * max(policy.score(encoded[move]) for move in MOVES)
        error = max(-self.error_clip, min(self.error_clip, target - policy.score(features)))
        norm = 1.0 + sum(value * value for _, value in features)
        weights = policy.weights
        step = self.alpha * error / norm
        for index, value in features:
            weights[index] = max(-self.weight_clip, min(self.weight_clip, weights[index] + step * value))
        self.updates += 1
        self.changed += error != 0.0
        self.absolute_error += abs(error)

    def transition(self, reward, view):
        self._update(reward, view)

    def finish(self, reward, view):
        self._update(reward, view)
        self.policy.last_features = None


class DemoRecorder:
    """Collect per-move features and the teacher's optimal move set at each decision."""

    def __init__(self, encoders, teacher):
        self.encoders = encoders
        self.teacher = teacher
        self.samples = []

    def __call__(self, view, action):
        moves = self.teacher.choices(view)
        self.samples.append({"moves": moves, "known": len(view.cells),
                             "features": {encoder.name + ":" + ",".join(encoder.disabled):
                                          [encoder(view)[move] for move in MOVES]
                                          for encoder in self.encoders}})


def collect(binary, policy, teacher, encoders, seed, task, policy_seed):
    """One rollout of `policy`, labeled by `teacher` at every visited state."""
    recorder = DemoRecorder(encoders, teacher)
    record = run_episode(binary, policy, seed, task, policy_seed, observer=recorder)
    return record, recorder.samples


def save_demos(path: Path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as file:
        for sample in samples:
            file.write(json.dumps(sample, separators=(",", ":")) + "\n")


def load_demos(path: Path):
    with gzip.open(path, "rt") as file:
        return [json.loads(line) for line in file]


def softmax_loss_gradient(weights, per_move, optimal):
    """Cross-entropy against a uniform target over the teacher's optimal set."""
    scores = [sum(weights[i] * v for i, v in features) for features in per_move]
    top = max(scores)
    exponentials = [math.exp(max(score - top, -40.0)) for score in scores]
    total = sum(exponentials)
    probabilities = [value / total for value in exponentials]
    share = 1.0 / len(optimal)
    loss = -share * sum(math.log(max(probabilities[move], 1e-12)) for move in optimal)
    return loss, probabilities, share


def train_softmax(samples, key, dimension, epochs=12, rate=0.2, l2=1e-5, seed=0, weights=None):
    """Adagrad on a conditional logit. Plain data, no framework, seeded shuffling."""
    weights = list(weights) if weights is not None else [0.0] * dimension
    accumulator = [1e-8] * dimension
    rng = random.Random(seed)
    order = list(range(len(samples)))
    history = []
    for epoch in range(epochs):
        rng.shuffle(order)
        total = 0.0
        for position in order:
            sample = samples[position]
            per_move = sample["features"][key]
            optimal = sample["moves"]
            loss, probabilities, share = softmax_loss_gradient(weights, per_move, optimal)
            total += loss
            gradient = {}
            for move, features in enumerate(per_move):
                for index, value in features:
                    gradient[index] = gradient.get(index, 0.0) + probabilities[move] * value
            for move in optimal:
                for index, value in per_move[move]:
                    gradient[index] -= share * value
            for index, value in gradient.items():
                value += l2 * weights[index]
                accumulator[index] += value * value
                weights[index] -= rate * value / math.sqrt(accumulator[index])
        history.append(total / max(1, len(samples)))
    return weights, history


def imitation_metrics(weights, samples, key):
    """Greedy agreement with the optimal move set, and mean cross-entropy."""
    agree = loss_total = 0.0
    for sample in samples:
        per_move = sample["features"][key]
        scores = [sum(weights[i] * v for i, v in features) for features in per_move]
        best = max(scores)
        tied = [move for move, score in enumerate(scores) if score == best]
        agree += sum(move in sample["moves"] for move in tied) / len(tied)
        loss_total += softmax_loss_gradient(weights, per_move, sample["moves"])[0]
    count = max(1, len(samples))
    return {"samples": len(samples), "set_accuracy": agree / count, "cross_entropy": loss_total / count}


def scale_to_returns(weights, samples, key, returns):
    """Least-squares scale from logit weights to observed discounted returns."""
    numerator = denominator = 0.0
    for sample, value in zip(samples, returns):
        per_move = sample["features"][key]
        taken = per_move[sample["moves"][0]]
        score = sum(weights[i] * v for i, v in taken)
        numerator += score * value
        denominator += score * score
    return numerator / denominator if denominator else 1.0
