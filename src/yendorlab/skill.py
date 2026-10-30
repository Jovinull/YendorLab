"""CPU-only linear action-value learner for player-visible exploration."""

import hashlib
import json
import math
import random
import time
from collections import deque
from pathlib import Path

from .agents import DIRECTIONS, context_action
from .env import BrogueEnv, RewardConfig
from .evaluation import run_policy_episode, seed_for
from .training import save_checkpoint


ALGORITHM = "linear_frontier_td_v1"
ENCODER_VERSION = 2
ACTIONS = tuple(DIRECTIONS)
FEATURE_COUNT = 16


def frontier_action(observation):
    """First known traversable step toward a frontier; unknown cells remain unknown."""
    x, y = observation["x"], observation["y"]
    cells = {(cell[0], cell[1]): cell for cell in observation["cells"]}
    width, height = observation["width"], observation["height"]

    def unknown_neighbor(point):
        for action, (dx, dy) in DIRECTIONS.items():
            nxt = (point[0] + dx, point[1] + dy)
            if 0 <= nxt[0] < width and 0 <= nxt[1] < height and nxt not in cells:
                return action
        return None

    adjacent = unknown_neighbor((x, y))
    if adjacent:
        return adjacent
    queue = deque([((x, y), None)])
    seen = {(x, y)}
    while queue:
        point, first = queue.popleft()
        if first is not None and unknown_neighbor(point):
            return first
        for action, (dx, dy) in DIRECTIONS.items():
            nxt = (point[0] + dx, point[1] + dy)
            cell = cells.get(nxt)
            if nxt not in seen and cell is not None and not cell[4]:
                seen.add(nxt)
                queue.append((nxt, first or action))
    return None


class LinearFrontierAgent:
    name = ALGORITHM

    def __init__(self, weights=None, seed=0, epsilon=0.0):
        self.weights = list(weights) if weights is not None else [0.0] * FEATURE_COUNT
        if len(self.weights) != FEATURE_COUNT or not all(math.isfinite(v) for v in self.weights):
            raise ValueError("invalid linear checkpoint weights")
        self.rng = random.Random(seed)
        self.epsilon = epsilon
        self.visits = {}
        self.last_position = None
        self.last_decision = None
        self.last_features = None

    def features(self, observation):
        if observation["context"] != "gameplay":
            raise ValueError("features require gameplay")
        position = (observation["x"], observation["y"])
        if observation["decision"] != self.last_decision:
            self.visits[position] = self.visits.get(position, 0) + 1
            self.last_decision = observation["decision"]
        cells = {(cell[0], cell[1]): cell for cell in observation["cells"]}
        enemies = {(c[0], c[1]) for c in observation["creatures"]}
        route = frontier_action(observation)
        result = {}
        for index, (action, (dx, dy)) in enumerate(DIRECTIONS.items()):
            nxt = (position[0] + dx, position[1] + dy)
            cell = cells.get(nxt)
            inside = 0 <= nxt[0] < observation["width"] and 0 <= nxt[1] < observation["height"]
            vector = [1.0, float(inside and cell is None),
                      float(not inside or (cell is not None and bool(cell[4]))),
                      float(nxt in enemies), min(self.visits.get(nxt, 0), 5) / 5.0,
                      float(action == route), float(nxt == self.last_position),
                      float(cell is not None and bool(cell[2]))]
            vector.extend(float(i == index) for i in range(8))
            result[action] = vector
        return result

    def score(self, vector):
        return sum(w * x for w, x in zip(self.weights, vector))

    def act(self, observation):
        if observation["context"] != "gameplay":
            return context_action(observation)
        features = self.features(observation)
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(ACTIONS)
        else:
            action = max(ACTIONS, key=lambda candidate: (self.score(features[candidate]),
                                                          -ACTIONS.index(candidate)))
        self.last_features = features[action]
        self.last_position = (observation["x"], observation["y"])
        return action

    def update(self, action_features, reward, next_features, alpha=0.02, gamma=0.95):
        target = reward + (gamma * max(self.score(v) for v in next_features.values())
                           if next_features else 0.0)
        error = max(-10.0, min(10.0, target - self.score(action_features)))
        before = tuple(self.weights)
        self.weights = [max(-10.0, min(10.0, w + alpha * error * x))
                        for w, x in zip(self.weights, action_features)]
        return self.weights != list(before)


def load_skill_checkpoint(path: Path, training=False):
    data = json.loads(path.read_text())
    if (data.get("algorithm") != ALGORITHM or data.get("encoder_version") != ENCODER_VERSION
            or data.get("observation_schema") != 2 or data.get("action_schema") != 2):
        raise ValueError("skill checkpoint is incompatible")
    if training and data.get("kind") != "training":
        raise ValueError("resume requires a training checkpoint")
    LinearFrontierAgent(data["weights"])
    return data


def skill_episode(binary, agent, seed, config):
    with BrogueEnv(binary, config["variant"], config["max_events"],
                   config["episode_seconds"], config["max_no_progress"],
                   config["memory_mb"], RewardConfig.named("exploration_v1")) as env:
        observation, _ = env.reset(seed=seed)
        previous = None
        reward_sum = 0.0
        changed = updates = decisions = 0
        while True:
            if observation["context"] == "gameplay":
                features = agent.features(observation)
                if previous is not None:
                    changed += agent.update(previous, reward_sum, features,
                                            config["alpha"], config["gamma"])
                    updates += 1
                action = agent.act(observation)
                previous = agent.last_features
                reward_sum = 0.0
                decisions += 1
            else:
                action = context_action(observation)
            observation, _, terminated, truncated, info = env.step(action)
            # The task reward is explicit. The primary evaluation metric remains raw new cells.
            reward_sum += info["reward_components"]["exploration"] * 10.0
            reward_sum += info["reward_components"]["terminal"]
            if terminated or truncated:
                if previous is not None:
                    can_bootstrap = (truncated and info["reason"] in
                                     {"TRUNCATED_ACTION_LIMIT", "TRUNCATED_NO_PROGRESS"}
                                     and observation["context"] == "gameplay")
                    if terminated or can_bootstrap:
                        next_features = agent.features(observation) if can_bootstrap else None
                        changed += agent.update(previous, reward_sum, next_features,
                                                config["alpha"], config["gamma"])
                        updates += 1
                break
        return {"seed": seed, "reason": info["reason"], "events": info["events"],
                "decisions": decisions, "turn": info["turn"], "new_cells": info["new_cells"],
                "updates": updates, "changed_updates": changed,
                "native_rss_peak_sampled_kib": info["native_rss_peak_sampled_kib"]}


def train_skill(binary: Path, config: dict, checkpoint_path: Path, resume=False):
    start = time.monotonic()
    if resume:
        checkpoint = load_skill_checkpoint(checkpoint_path, training=True)
        fixed = lambda c: {k: v for k, v in c.items() if k not in {"episodes", "total_seconds"}}
        if fixed(checkpoint["config"]) != fixed(config) or config["episodes"] < checkpoint["episodes_completed"]:
            raise ValueError("resume configuration is incompatible")
        checkpoint["config"] = config
    else:
        checkpoint = {"kind": "training", "algorithm": ALGORITHM,
                      "encoder_version": ENCODER_VERSION, "observation_schema": 2,
                      "action_schema": 2, "config": config, "weights": [0.0] * FEATURE_COUNT,
                      "episodes_completed": 0, "updates": 0, "changed_updates": 0,
                      "validation": []}
        save_checkpoint(checkpoint_path, checkpoint)
    episodes = []
    deadline = start + config["total_seconds"]
    interrupted = False
    try:
        for index in range(checkpoint["episodes_completed"], config["episodes"]):
            if time.monotonic() >= deadline:
                break
            agent = LinearFrontierAgent(checkpoint["weights"],
                                        seed_for(config["master_seed"], "train", index + 1000000),
                                        config["epsilon"])
            seed = seed_for(config["master_seed"], "train", index)
            episode_config = {**config, "episode_seconds": min(config["episode_seconds"],
                              max(0.1, deadline - time.monotonic()))}
            episode = skill_episode(binary, agent, seed, episode_config)
            episodes.append(episode)
            checkpoint["episodes_completed"] = index + 1
            checkpoint["updates"] += episode["updates"]
            checkpoint["changed_updates"] += episode["changed_updates"]
            checkpoint["weights"] = agent.weights
            save_checkpoint(checkpoint_path, checkpoint)
            if (index + 1) % config["validation_every"] == 0 and time.monotonic() < deadline:
                values = []
                for j in range(config["validation_episodes"]):
                    if time.monotonic() >= deadline:
                        break
                    validation_seed = seed_for(config["master_seed"], "validation", j)
                    policy = LinearFrontierAgent(checkpoint["weights"], validation_seed)
                    values.append(run_policy_episode(binary, policy, validation_seed,
                                  config["variant"], "canonical", config["max_events"],
                                  min(config["episode_seconds"], max(0.1, deadline - time.monotonic())),
                                  config["max_no_progress"],
                                  config["memory_mb"])["new_cells"])
                checkpoint["validation"].append({"episodes": index + 1, "sample": len(values),
                                                 "mean_new_cells": sum(values) / len(values) if values else None})
                save_checkpoint(checkpoint_path, checkpoint)
    except KeyboardInterrupt:
        interrupted = True
        checkpoint = load_skill_checkpoint(checkpoint_path, training=True)
    training_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    inference = checkpoint_path.with_name(checkpoint_path.stem + "-inference.json")
    inference_hash = save_checkpoint(inference, {
        "kind": "inference", "algorithm": ALGORITHM, "encoder_version": ENCODER_VERSION,
        "observation_schema": 2, "action_schema": 2, "weights": checkpoint["weights"],
        "source_training_sha256": training_hash})
    return {"status": "INTERRUPTED" if interrupted else
            "COMPLETED" if checkpoint["episodes_completed"] == config["episodes"] else "INCONCLUSIVE",
            "episodes_this_run": len(episodes), "episodes_total": checkpoint["episodes_completed"],
            "updates_total": checkpoint["updates"], "changed_updates_total": checkpoint["changed_updates"],
            "duration_s": time.monotonic() - start, "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": training_hash, "inference_checkpoint": str(inference),
            "inference_checkpoint_sha256": inference_hash,
            "weights": checkpoint["weights"], "validation": checkpoint["validation"],
            "episodes": episodes}
