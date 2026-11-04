"""Policies under the M5 decision contract: decide(observation, view) -> key."""

import json
import math
import random
from pathlib import Path

from .agents import DIRECTIONS, FixedDirectionAgent, HeuristicAgent, context_action
from .encoders import Encoder
from .skill import LinearFrontierAgent, load_skill_checkpoint
from .tasks import MOVES, OFFSETS, route, route_to_stairs


class Policy:
    name = "policy"

    def context(self, observation):
        return context_action(observation)


class RandomMoves(Policy):
    name = "random_moves_v1"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def decide(self, observation, view):
        return self.rng.choice(MOVES)


class FixedMove(Policy):
    name = "fixed_move_v1"

    def __init__(self, action):
        if action not in DIRECTIONS:
            raise ValueError("invalid direction")
        self.action = action

    def decide(self, observation, view):
        return self.action


class WeightedMoves(Policy):
    """Memoryless direction prior: a control for what a preference alone can reach."""
    name = "direction_prior_v1"

    def __init__(self, shares, seed=0):
        self.rng = random.Random(seed)
        self.actions = [action for action in MOVES if shares.get(action, 0) > 0]
        self.weights = [shares[action] for action in self.actions]

    def decide(self, observation, view):
        return self.rng.choices(self.actions, self.weights)[0]


class FrontierMoves(Policy):
    """Hand-written planner with exactly the eight moves and the shared perception."""
    name = "frontier_moves_v2"

    def __init__(self, seed=0, stairs_goal=False):
        self.rng = random.Random(seed)
        self.stairs_goal = stairs_goal

    def choices(self, view):
        """The set of moves this planner considers optimal, as action indices."""
        # A chased explorer that never strikes back dies; attacking is a plain move.
        attacks = [index for index, (dx, dy) in enumerate(OFFSETS)
                   if (view.x + dx, view.y + dy) in view.enemies
                   and not view.move_blocked(view.x, view.y, index)
                   and (view.x, view.y, index) not in view.failed]
        if attacks:
            return attacks
        mask = route_to_stairs(view) if self.stairs_goal else 0
        if not mask:
            mask = route(view, avoid_stairs=True)[0]
        if mask:
            return [index for index in range(8) if mask >> index & 1]
        # No reachable frontier: wander to the least visited known neighbor.
        options = []
        for index, (dx, dy) in enumerate(OFFSETS):
            target = (view.x + dx, view.y + dy)
            cell = view.cells.get(target)
            if view.move_blocked(view.x, view.y, index) or (view.x, view.y, index) in view.failed:
                continue
            if cell is not None and cell[5] and not self.stairs_goal:
                continue
            options.append((view.visits.get(target, 0), index))
        if not options:
            return list(range(8))
        least = min(options)[0]
        return [index for count, index in options if count == least]

    def decide(self, observation, view):
        return MOVES[self.rng.choice(self.choices(view))]


class RawAgent(Policy):
    """Adapter for pre-M5 agents, which read the raw observation."""

    def __init__(self, agent):
        self.agent = agent
        self.name = agent.name

    def decide(self, observation, view):
        return self.agent.act(observation)

    def context(self, observation):
        return self.agent.act(observation)


class LinearPolicy(Policy):
    """Linear scorer over per-move features. Ties break by RNG, never by move order."""
    name = "linear_v2"

    def __init__(self, encoder, weights=None, seed=0, epsilon=0.0):
        self.encoder = encoder
        self.weights = list(weights) if weights is not None else [0.0] * encoder.dimension
        if len(self.weights) != encoder.dimension or not all(map(math.isfinite, self.weights)):
            raise ValueError("invalid weights for encoder")
        self.rng = random.Random(seed)
        self.epsilon = epsilon
        self.last_features = None

    def score(self, features):
        weights = self.weights
        return sum(weights[index] * value for index, value in features)

    def decide(self, observation, view):
        encoded = self.encoder(view)
        if self.epsilon and self.rng.random() < self.epsilon:
            action = self.rng.choice(MOVES)
        else:
            scores = [(self.score(encoded[move]), move) for move in MOVES]
            best = max(scores)[0]
            action = self.rng.choice([move for score, move in scores if score == best])
        self.last_features = encoded[action]
        return action


class MLPPolicy(Policy):
    """Greedy scorer over a small imitation network; ties break by RNG."""
    name = "tiny_mlp_v1"

    def __init__(self, encoder, model, seed=0):
        self.encoder = encoder
        self.model = model
        self.rng = random.Random(seed)

    def decide(self, observation, view):
        encoded = self.encoder(view)
        scores = [(self.model.score(encoded[move]), move) for move in MOVES]
        best = max(scores)[0]
        return self.rng.choice([move for score, move in scores if score == best])


def checkpoint_weights(path):
    return json.loads(Path(path).read_text())["weights"]


def build_policy(spec, seed):
    """Policies are described by plain data so worker processes can rebuild them."""
    kind = spec["kind"]
    if kind == "random":
        return RandomMoves(seed)
    if kind == "fixed":
        return FixedMove(spec["action"])
    if kind == "prior":
        return WeightedMoves(spec["shares"], seed)
    if kind == "frontier":
        return FrontierMoves(seed, spec.get("stairs_goal", False))
    if kind == "heuristic_full":
        return RawAgent(HeuristicAgent(seed))
    if kind == "m4_linear":
        weights = load_skill_checkpoint(Path(spec["checkpoint"]))["weights"] if spec.get("checkpoint") else None
        return RawAgent(LinearFrontierAgent(weights, seed=seed))
    if kind == "linear":
        encoder = Encoder(spec["encoder"], spec.get("disabled", ()))
        weights = spec.get("weights")
        if weights is None and spec.get("checkpoint"):
            weights = checkpoint_weights(spec["checkpoint"])
        return LinearPolicy(encoder, weights, seed)
    if kind == "mlp":
        from .learners import TinyMLP
        return MLPPolicy(Encoder(spec["encoder"], spec.get("disabled", ())), TinyMLP.from_dict(
            json.loads(Path(spec["checkpoint"]).read_text())["model"]), seed)
    raise ValueError(f"unknown policy kind: {kind}")


def relay_guide(seed=20261103):
    """The fixed guide of the relay task: same pre-roll for every compared policy."""
    return lambda: FrontierMoves(seed, stairs_goal=True)
