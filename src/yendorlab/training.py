"""Episode-boundary tabular Q-learning on real Brogue observations."""

import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path

from .agents import GAME_ACTIONS, QAgent, context_action, encode_state
from .env import BrogueEnv, RewardConfig
from .evaluation import run_policy_episode, seed_for


CHECKPOINT_VERSION = 1


def save_checkpoint(path: Path, checkpoint: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(checkpoint, sort_keys=True, separators=(",", ":")).encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as file:
        temp = Path(file.name)
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp, path)
    return hashlib.sha256(payload).hexdigest()


def load_checkpoint(path: Path):
    data = json.loads(path.read_text())
    if data.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("checkpoint version mismatch")
    if data.get("observation_schema") != 2 or data.get("action_schema") != 2:
        raise ValueError("checkpoint schema mismatch")
    if data.get("encoder_version") != 1:
        raise ValueError("encoder version mismatch")
    if any(len(values) != len(GAME_ACTIONS) for values in data["q"].values()):
        raise ValueError("corrupt Q table")
    return data


def load_policy_checkpoint(path: Path):
    data = json.loads(path.read_text())
    if data.get("checkpoint_kind") == "inference":
        if data.get("observation_schema") != 2 or data.get("action_schema") != 2 or data.get("encoder_version") != 1:
            raise ValueError("inference checkpoint schema mismatch")
        return data["q"]
    return load_checkpoint(path)["q"]


def train_episode(binary, agent: QAgent, seed: int, variant: str, max_events: int,
                  max_seconds: float, max_no_progress: int, memory_mb: int,
                  alpha: float, gamma: float):
    with BrogueEnv(binary, variant, max_events, max_seconds, max_no_progress,
                   memory_mb, RewardConfig.named("exploration_v1")) as env:
        observation, _ = env.reset(seed=seed)
        pending = None
        updates = changed = decisions = 0
        total_reward = 0.0
        while True:
            if observation["context"] == "gameplay":
                state = encode_state(observation)
                if pending is not None:
                    old_state, old_action, accumulated = pending
                    changed += agent.update(old_state, old_action, accumulated, state, False,
                                            alpha, gamma)
                    updates += 1
                action = agent.act(observation)
                pending = (state, action, 0.0)
                decisions += 1
            else:
                action = context_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if pending is not None:
                pending = (pending[0], pending[1], pending[2] + reward)
            if terminated or truncated:
                if pending is not None:
                    # An operational cut is not a game terminal. Bootstrap only
                    # when the post-action observation is a valid gameplay state.
                    bootstrappable = (truncated and info["reason"] in
                                      {"TRUNCATED_ACTION_LIMIT", "TRUNCATED_NO_PROGRESS"}
                                      and observation["context"] == "gameplay")
                    if terminated or bootstrappable:
                        changed += agent.update(pending[0], pending[1], pending[2],
                                                encode_state(observation) if bootstrappable else "",
                                                terminated, alpha, gamma)
                        updates += 1
                break
        return {"seed": seed, "reason": info["reason"], "events": info["events"],
                "decisions": decisions, "updates": updates, "changed_updates": changed,
                "turn": info["turn"], "max_depth": info["max_depth"],
                "reward": total_reward, "reward_components": info["reward_totals"]}


def train(binary: Path, config: dict, checkpoint_path: Path, resume: bool = False):
    start = time.monotonic()
    if resume:
        checkpoint = load_checkpoint(checkpoint_path)
        fixed = lambda settings: {k: v for k, v in settings.items() if k not in {"episodes", "total_seconds"}}
        if fixed(checkpoint["config"]) != fixed(config):
            raise ValueError("resume configuration differs from checkpoint")
        if config["episodes"] < checkpoint["episodes_completed"]:
            raise ValueError("episode target precedes saved progress")
        checkpoint["config"] = config
    else:
        checkpoint = {"checkpoint_version": CHECKPOINT_VERSION, "checkpoint_kind": "training",
                      "algorithm": "tabular_q_v1",
                      "observation_schema": 2, "action_schema": 2, "encoder_version": 1,
                      "experiment": config["experiment"], "config": config,
                      "episodes_completed": 0, "updates": 0, "changed_updates": 0,
                      "q": {}, "validation": [], "best_validation_reward": None,
                      "seed_rule": "sha256(yendorlab-v1:master:train:index)"}
        save_checkpoint(checkpoint_path, checkpoint)
    episodes = []
    limit = time.monotonic() + config["total_seconds"]
    interrupted = False
    try:
        for index in range(checkpoint["episodes_completed"], config["episodes"]):
            if time.monotonic() >= limit:
                break
            seed = seed_for(config["master_seed"], "train", index)
            agent = QAgent(checkpoint["q"], seed_for(config["master_seed"], "train", index + 1000000),
                           epsilon=config["epsilon"])
            episode = train_episode(binary, agent, seed, config["variant"], config["max_events"],
                                    min(config["episode_seconds"], max(.1, limit - time.monotonic())),
                                    config["max_no_progress"], config["memory_mb"],
                                    config["alpha"], config["gamma"])
            episodes.append(episode)
            checkpoint["episodes_completed"] = index + 1
            checkpoint["updates"] += episode["updates"]
            checkpoint["changed_updates"] += episode["changed_updates"]
            checkpoint["q"] = agent.q
            save_checkpoint(checkpoint_path, checkpoint)
            if (checkpoint["episodes_completed"] % config.get("validation_every", 2) == 0
                    and time.monotonic() < limit):
                scores = []
                for validation_index in range(config.get("validation_episodes", 1)):
                    if time.monotonic() >= limit:
                        break
                    validation_seed = seed_for(config["master_seed"], "validation", validation_index)
                    validator = QAgent(checkpoint["q"], validation_seed)
                    scores.append(run_policy_episode(binary, validator, validation_seed,
                                                     config["variant"], "exploration_v1",
                                                     config["max_events"],
                                                     min(config["episode_seconds"], max(.1, limit - time.monotonic())),
                                                     config["max_no_progress"], config["memory_mb"])["reward"])
                if not scores:
                    continue
                score = sum(scores) / len(scores)
                checkpoint["validation"].append({"episodes_completed": index + 1,
                                                 "mean_reward": score, "sample": len(scores)})
                if checkpoint["best_validation_reward"] is None or score > checkpoint["best_validation_reward"]:
                    checkpoint["best_validation_reward"] = score
                    save_checkpoint(checkpoint_path.with_name(checkpoint_path.stem + "-best.json"), checkpoint)
                save_checkpoint(checkpoint_path, checkpoint)
    except KeyboardInterrupt:
        interrupted = True
        checkpoint = load_checkpoint(checkpoint_path)
    training_checksum = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    inference_path = checkpoint_path.with_name(checkpoint_path.stem + "-inference.json")
    inference_checksum = save_checkpoint(inference_path, {
        "checkpoint_kind": "inference", "checkpoint_version": CHECKPOINT_VERSION,
        "algorithm": "tabular_q_v1", "observation_schema": 2,
        "action_schema": 2, "encoder_version": 1, "q": checkpoint["q"],
        "source_training_sha256": training_checksum})
    return {"status": "INTERRUPTED" if interrupted else "COMPLETED" if checkpoint["episodes_completed"] >= config["episodes"] else "INCONCLUSIVE",
            "episodes_this_run": len(episodes), "episodes_total": checkpoint["episodes_completed"],
            "updates_total": checkpoint["updates"], "changed_updates_total": checkpoint["changed_updates"],
            "q_states": len(checkpoint["q"]), "duration_s": time.monotonic() - start,
            "checkpoint": str(checkpoint_path),
            "inference_checkpoint": str(inference_path),
            "inference_checkpoint_sha256": inference_checksum,
            "best_checkpoint": str(checkpoint_path.with_name(checkpoint_path.stem + "-best.json")) if checkpoint["best_validation_reward"] is not None else None,
            "best_validation_reward": checkpoint["best_validation_reward"],
            "validation": checkpoint["validation"],
            "checkpoint_sha256": training_checksum,
            "episodes": episodes}
