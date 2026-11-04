"""Versioned exploration tasks with equal action, budget and perception contracts.

`exploration_v2_floor` is the comparable skill: eight moves, the start depth only,
and a fixed number of gameplay decisions. The metric is counted from the observation
stream and is independent of any reward. See docs/experiments/m5-protocol.md.
"""

import hashlib
import time
from dataclasses import asdict, dataclass

from .agents import DIRECTIONS, context_action
from .env import BrogueEnv, OPERATIONAL_REASONS

MOVES = tuple(DIRECTIONS)
OFFSETS = tuple(DIRECTIONS.values())
SPLITS = ("train", "validation", "diagnostic", "final")
SEED_PROTOCOL = "yendorlab-m5-v1"
CURVE_STEP = 50


def m5_seed(master: int, split: str, index: int, stream: str = "game") -> int:
    """Domain-separated M5 seeds; M4 seed sets use a different prefix."""
    if split not in SPLITS:
        raise ValueError("invalid seed split")
    text = f"{SEED_PROTOCOL}:{master}:{split}:{stream}:{index}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") or 1


@dataclass(frozen=True)
class TaskSpec:
    name: str
    budget: int
    stop_on_depth_change: bool = True
    preroll_depth: int | None = None
    restrict_moves: bool = True
    preroll_budget: int = 900
    episode_seconds: float = 120.0
    memory_mb: int = 512
    variant: str = "classic"

    @property
    def max_events(self):
        # Guard only: prompts cost events, the budget is counted in decisions.
        return (self.budget + self.preroll_budget) * 4 + 100

    def to_dict(self):
        return asdict(self)


def task_named(name: str, budget: int, **overrides) -> TaskSpec:
    presets = {
        "exploration_v2_floor": {},
        "exploration_v2_multilevel": {"stop_on_depth_change": False},
        "exploration_v2_relay": {"preroll_depth": 2},
        "exploration_v2_full_reference": {"stop_on_depth_change": False, "restrict_moves": False},
    }
    if name not in presets:
        raise ValueError(f"unknown task: {name}")
    return TaskSpec(name=name, budget=budget, **{**presets[name], **overrides})


class MapView:
    """Everything a policy may know at one decision: filtered map plus own memory."""

    __slots__ = ("x", "y", "width", "height", "cells", "enemies", "visits", "failed",
                 "previous", "last_action", "depth", "cache")

    def known_blocked(self, x, y):
        cell = self.cells.get((x, y))
        return cell is not None and bool(cell[4])

    def inside(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def move_blocked(self, x, y, index):
        """Blocked by what the player knows: walls, map edge, diagonal corners."""
        dx, dy = OFFSETS[index]
        nx, ny = x + dx, y + dy
        if not self.inside(nx, ny) or self.known_blocked(nx, ny):
            return True
        return bool(dx and dy) and (self.known_blocked(x + dx, y) or self.known_blocked(x, y + dy))


class Perception:
    """Per-episode memory shared by every comparable policy."""

    def __init__(self):
        self.depth = None
        self.visits = {}
        self.failed = set()
        self.position = None
        self.previous = None
        self.last_action = None
        self.expected_attack = False
        self.stalled = False
        self.revisit = False

    def observe(self, observation):
        depth = observation["depth"]
        position = (observation["x"], observation["y"])
        if depth != self.depth:
            self.depth = depth
            self.visits = {}
            self.failed = set()
            self.position = None
            self.previous = None
            self.last_action = None
            self.expected_attack = False
        self.stalled = self.revisit = False
        if self.last_action is not None:
            if position == self.position:
                if not self.expected_attack:
                    self.failed.add((position[0], position[1], self.last_action))
                    self.stalled = True
            else:
                self.previous = self.position
                self.revisit = position in self.visits
        self.position = position
        self.visits[position] = self.visits.get(position, 0) + 1
        view = MapView()
        view.x, view.y = position
        view.width, view.height = observation["width"], observation["height"]
        view.cells = {(cell[0], cell[1]): cell for cell in observation["cells"]}
        view.enemies = {(creature[0], creature[1]) for creature in observation["creatures"]}
        view.visits = self.visits
        view.failed = self.failed
        view.previous = self.previous
        view.last_action = self.last_action
        view.depth = depth
        view.cache = {}
        return view

    def acted(self, view, action):
        """Record the chosen move; `action` may be outside the move set for references."""
        if action in DIRECTIONS:
            index = MOVES.index(action)
            dx, dy = OFFSETS[index]
            self.last_action = index
            self.expected_attack = (view.x + dx, view.y + dy) in view.enemies
        else:
            self.last_action = None
            self.expected_attack = False


def route(view, avoid_stairs=True):
    """(mask of first moves on every shortest path, distance) to the nearest frontier.

    A frontier is a known passable cell not yet stood on, with an unknown in-bounds
    neighbor. Taking the whole shortest-path set keeps this free of direction order.
    """
    key = ("route", avoid_stairs)
    cached = view.cache.get(key)
    if cached is not None:
        return cached
    cells, visits, failed = view.cells, view.visits, view.failed
    width, height = view.width, view.height
    current = {(view.x, view.y): 0}
    seen = {(view.x, view.y)}
    distance = 0
    result = (0, None)
    while current and not result[0]:
        distance += 1
        following = {}
        for (px, py), mask in current.items():
            for index, (dx, dy) in enumerate(OFFSETS):
                target = (px + dx, py + dy)
                if target in seen:
                    continue
                cell = cells.get(target)
                if cell is None or cell[4] or (avoid_stairs and cell[5]):
                    continue
                if (px, py, index) in failed:
                    continue
                if dx and dy:
                    side = cells.get((px + dx, py))
                    other = cells.get((px, py + dy))
                    if (side is not None and side[4]) or (other is not None and other[4]):
                        continue
                following[target] = following.get(target, 0) | (mask or 1 << index)
        seen.update(following)
        found = 0
        for (tx, ty), mask in following.items():
            if (tx, ty) in visits:
                continue
            for dx, dy in OFFSETS:
                nx, ny = tx + dx, ty + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in cells:
                    found |= mask
                    break
        if found:
            result = (found, distance)
        current = following
    view.cache[key] = result
    return result


def route_to_stairs(view):
    """Mask of first moves toward a known down staircase, or 0."""
    cells, failed = view.cells, view.failed
    current = {(view.x, view.y): 0}
    seen = {(view.x, view.y)}
    while current:
        following = {}
        for (px, py), mask in current.items():
            for index, (dx, dy) in enumerate(OFFSETS):
                target = (px + dx, py + dy)
                if target in seen:
                    continue
                cell = cells.get(target)
                if cell is None or cell[4] or (px, py, index) in failed:
                    continue
                if dx and dy:
                    side = cells.get((px + dx, py))
                    other = cells.get((px, py + dy))
                    if (side is not None and side[4]) or (other is not None and other[4]):
                        continue
                if cell[5] == 1:
                    return mask or 1 << index
                if cell[5]:
                    continue
                following[target] = following.get(target, 0) | (mask or 1 << index)
        seen.update(following)
        current = following
    return 0


def _failure(seed, policy_seed, task, reason, exc, started, **partial):
    return {"seed": seed, "policy_seed": policy_seed, "task": task.name, "reason": reason,
            "valid": False, "error": repr(exc)[:300], "new_cells": None,
            "duration_s": time.monotonic() - started, **partial}


def _preroll(env, observation, task, guide_factory, seed):
    """Reach the task depth with a fixed guide, identical for every compared policy."""
    guide = guide_factory()
    perception = Perception()
    decisions = 0
    while observation["depth"] < task.preroll_depth:
        if observation["context"] == "gameplay":
            if decisions >= task.preroll_budget:
                return observation, "PREROLL_BUDGET"
            view = perception.observe(observation)
            action = guide.decide(observation, view)
            perception.acted(view, action)
            decisions += 1
        else:
            action = context_action(observation)
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return observation, "PREROLL_" + info["reason"]
    return observation, None


def run_episode(binary, policy, seed, task: TaskSpec, policy_seed=0, trainer=None,
                observer=None, guide_factory=None):
    """One bounded episode. Never raises for engine or policy failures."""
    started = time.monotonic()
    env = BrogueEnv(binary, task.variant, task.max_events, task.episode_seconds,
                    task.max_events + 1, task.memory_mb)
    try:
        try:
            observation, _ = env.reset(seed=seed)
        except (TimeoutError, RuntimeError, ValueError, OSError) as exc:
            return _failure(seed, policy_seed, task, "RESET_ERROR", exc, started)
        reset_s = time.monotonic() - started
        if task.preroll_depth:
            observation, problem = _preroll(env, observation, task, guide_factory, seed)
            if problem:
                # The guide never reached the task depth: this seed has no task instance.
                reason = "PREROLL_UNAVAILABLE" if problem.split("_", 1)[1] not in OPERATIONAL_REASONS \
                    else problem.split("_", 1)[1]
                return {"seed": seed, "policy_seed": policy_seed, "task": task.name,
                        "reason": reason, "valid": False, "new_cells": None,
                        "detail": problem, "duration_s": time.monotonic() - started}
            while observation["context"] != "gameplay":
                observation, _, terminated, truncated, info = env.step(context_action(observation))
                if terminated or truncated:
                    return {"seed": seed, "policy_seed": policy_seed, "task": task.name,
                            "reason": "PREROLL_UNAVAILABLE", "valid": False, "new_cells": None,
                            "detail": info["reason"], "duration_s": time.monotonic() - started}
        return _run(env, policy, seed, task, policy_seed, trainer, observer, observation,
                    started, reset_s)
    finally:
        env.close()


def _run(env, policy, seed, task, policy_seed, trainer, observer, observation, started, reset_s):
    perception = Perception()
    start_depth = observation["depth"]
    start_position = [observation["x"], observation["y"]]
    start_turn = observation["turn"]
    start_events = env.events
    baseline = env.known_by_depth.get(start_depth, 0)
    baseline_total = len(env.known)
    decisions = stalled = revisits = 0
    policy_s = step_s = 0.0
    action_counts = {}
    curve = []
    pending = 0.0
    depths = {start_depth}
    last_view = None
    info = None
    reason = None
    while True:
        context = observation["context"]
        if context == "gameplay":
            if decisions >= task.budget:
                reason = "TASK_BUDGET"
                last_view = perception.observe(observation)
                pending += trainer.step_reward(perception) if trainer else 0.0
                stalled += perception.stalled
                revisits += perception.revisit
                break
            before = time.monotonic()
            view = perception.observe(observation)
            stalled += perception.stalled
            revisits += perception.revisit
            try:
                if trainer:
                    pending += trainer.step_reward(perception)
                    trainer.transition(pending, view)
                    pending = 0.0
                action = policy.decide(observation, view)
                if task.restrict_moves and action not in DIRECTIONS:
                    raise ValueError(f"action {action!r} outside the eight-move contract")
                if action not in observation["admissible"]:
                    raise ValueError(f"inadmissible action {action!r}")
                if observer:
                    observer(view, action)
            except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError) as exc:
                return _failure(seed, policy_seed, task, "POLICY_ERROR", exc, started,
                                events=env.events, decisions=decisions)
            perception.acted(view, action)
            policy_s += time.monotonic() - before
            decisions += 1
            action_counts[action] = action_counts.get(action, 0) + 1
            if decisions % CURVE_STEP == 0:
                curve.append(env.known_by_depth.get(start_depth, 0) - baseline)
        else:
            action = policy.context(observation)
        before = time.monotonic()
        before_known = env.known_by_depth.get(start_depth, 0)
        observation, _, terminated, truncated, info = env.step(action)
        step_s += time.monotonic() - before
        if trainer:
            pending += trainer.cell_reward(env.known_by_depth.get(start_depth, 0) - before_known)
        if terminated or truncated:
            reason = info["reason"]
            if trainer and reason == "DEATH":
                pending += trainer.death_reward()
            break
        depths.add(observation["depth"])
        if task.stop_on_depth_change and observation["depth"] != start_depth:
            reason = "TASK_DEPTH_EXIT"
            break
    operational = reason in OPERATIONAL_REASONS
    if trainer and not operational:
        # The budget is a clock outside the state, so it bootstraps; a depth exit or a
        # natural end has no continuation within this task.
        trainer.finish(pending, last_view if reason == "TASK_BUDGET" else None)
    floor_cells = env.known_by_depth.get(start_depth, 0) - baseline
    turns = observation["turn"] - start_turn
    events = env.events - start_events
    total_new = len(env.known) - baseline_total
    record = {"seed": seed, "policy_seed": policy_seed, "task": task.name, "reason": reason,
              "valid": not operational, "budget": task.budget,
              "new_cells": None if operational else floor_cells,
              "new_cells_all_depths": None if operational else total_new,
              "partial_new_cells": floor_cells,
              "new_cells_by_depth": env.new_cells_by_depth(),
              "start_depth": start_depth, "start_position": start_position,
              "depths_visited": sorted(depths), "decisions": decisions, "events": events,
              "turn": turns, "stalled_moves": stalled, "revisit_moves": revisits,
              "unique_positions": len(perception.visits), "action_counts": action_counts,
              "curve": curve, "reset_s": reset_s, "policy_s": policy_s, "step_s": step_s,
              "native_rss_peak_sampled_kib": env.peak_native_rss_kib,
              "trace_sha256": env.trace_digest(), "duration_s": time.monotonic() - started}
    if operational and info:
        record["error"] = str(info["terminal"])[:300]
    return record


def run_with_retry(binary, make_policy, seed, task, policy_seed, retries=2, guide_factory=None):
    """Retry operational failures with the same seed; every attempt stays recorded."""
    attempts = []
    for attempt in range(retries + 1):
        record = run_episode(binary, make_policy(), seed, task, policy_seed,
                             guide_factory=guide_factory)
        attempts.append(record)
        if record["reason"] not in OPERATIONAL_REASONS:
            break
    final = dict(attempts[-1])
    final["attempts"] = len(attempts)
    final["failed_attempts"] = [{"reason": a["reason"], "error": a.get("error")}
                                for a in attempts if a["reason"] in OPERATIONAL_REASONS]
    return final


def metric_summary(episodes):
    """Aggregates over valid episodes; failures stay visible and never enter a mean."""
    valid = [e for e in episodes if e["valid"]]
    unavailable = [e for e in episodes if e["reason"] == "PREROLL_UNAVAILABLE"]
    invalid = [e for e in episodes if not e["valid"] and e["reason"] != "PREROLL_UNAVAILABLE"]

    def mean(key):
        return sum(e[key] for e in valid) / len(valid) if valid else None

    reasons = {}
    for episode in episodes:
        reasons[episode["reason"]] = reasons.get(episode["reason"], 0) + 1
    actions = {}
    for episode in valid:
        for action, count in episode["action_counts"].items():
            actions[action] = actions.get(action, 0) + count
    decisions = sum(e["decisions"] for e in valid)
    turns = sum(e["turn"] for e in valid)
    events = sum(e["events"] for e in valid)
    cells = sum(e["new_cells"] for e in valid)
    wall = sum(e["duration_s"] for e in episodes)
    return {"started": len(episodes), "valid": len(valid), "invalid": len(invalid),
            "unavailable": len(unavailable), "reasons": reasons,
            "retried": sum(1 for e in episodes if e.get("attempts", 1) > 1),
            "mean_new_cells": mean("new_cells"),
            "mean_new_cells_all_depths": mean("new_cells_all_depths"),
            "deaths": reasons.get("DEATH", 0), "depth_exits": reasons.get("TASK_DEPTH_EXIT", 0),
            "mean_decisions": mean("decisions"), "mean_turns": mean("turn"),
            "mean_events": mean("events"),
            "stalled_rate": sum(e["stalled_moves"] for e in valid) / decisions if decisions else None,
            "revisit_rate": sum(e["revisit_moves"] for e in valid) / decisions if decisions else None,
            "cells_per_decision": cells / decisions if decisions else None,
            "cells_per_turn": cells / turns if turns else None,
            "cells_per_event": cells / events if events else None,
            "action_share": {action: count / decisions for action, count in sorted(actions.items())}
            if decisions else {},
            "wall_episode_s": wall,
            "decisions_per_s": decisions / wall if wall else None,
            "policy_s": sum(e["policy_s"] for e in valid), "step_s": sum(e["step_s"] for e in valid),
            "reset_s": sum(e["reset_s"] for e in valid),
            "rss_peak_kib": max((e.get("native_rss_peak_sampled_kib", 0) for e in valid), default=0)}
