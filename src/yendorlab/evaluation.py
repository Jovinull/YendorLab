"""Independent, bounded policy evaluation on real Brogue episodes."""

import hashlib
import math
import time

from .env import BrogueEnv, OPERATIONAL_REASONS, RewardConfig


def seed_for(master: int, split: str, index: int) -> int:
    if split not in {"train", "validation", "final"}:
        raise ValueError("invalid seed split")
    digest = hashlib.sha256(f"yendorlab-v1:{master}:{split}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") or 1


def wilson(wins: int, games: int, z: float = 1.96):
    if games == 0:
        return None
    p = wins / games
    denominator = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denominator
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denominator
    return [max(0, center - half), min(1, center + half)]


def failed_episode(seed, reason, exc, started):
    """An operational failure record: never a competence result of zero."""
    return {"seed": seed, "reason": reason, "error": repr(exc), "valid": False,
            "terminated": False, "truncated": True, "events": 0, "decisions": 0, "turn": 0,
            "depth": None, "max_depth": None, "reward": None, "new_cells": None,
            "new_cells_by_depth": {}, "reward_components": {},
            "native_rss_peak_sampled_kib": 0, "duration_s": time.monotonic() - started}


def run_policy_episode(binary, agent, seed, variant="classic", reward="canonical",
                       max_events=300, max_seconds=30, max_no_progress=50,
                       memory_mb=512):
    started = time.monotonic()
    env = BrogueEnv(binary, variant, max_events, max_seconds, max_no_progress,
                    memory_mb, RewardConfig.named(reward))
    try:
        try:
            observation, _ = env.reset(seed=seed)
        except (TimeoutError, RuntimeError, ValueError, OSError) as exc:
            return failed_episode(seed, "RESET_ERROR", exc, started)
        return _run_policy_episode(env, agent, seed, observation, started)
    finally:
        env.close()


def _run_policy_episode(env, agent, seed, observation, started):
    reset_s = time.monotonic() - started
    total_reward = 0.0
    decisions = 0
    policy_s = step_s = 0.0
    action_counts = {}
    stalled_moves = 0
    while True:
        before = time.monotonic()
        try:
            action = agent.act(observation)
            if action not in observation["admissible"]:
                raise ValueError(f"inadmissible action {action!r} in {observation['context']}")
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            failed = failed_episode(seed, "POLICY_ERROR", exc, started)
            failed.update({"events": env.events, "decisions": decisions,
                           "partial_new_cells": len(env.known) - env.initial_known})
            return failed
        policy_s += time.monotonic() - before
        if observation["context"] == "gameplay":
            decisions += 1
            action_counts[action] = action_counts.get(action, 0) + 1
        previous = observation
        before = time.monotonic()
        observation, earned, terminated, truncated, info = env.step(action)
        step_s += time.monotonic() - before
        if (previous["context"] == "gameplay" and observation["context"] == "gameplay"
                and (previous["x"], previous["y"], previous["turn"]) ==
                (observation["x"], observation["y"], observation["turn"])):
            stalled_moves += 1
        total_reward += earned
        if terminated or truncated:
            break
    valid = info["reason"] not in OPERATIONAL_REASONS
    record = {"seed": seed, "reason": info["reason"], "valid": valid, "terminated": terminated,
              "truncated": truncated, "events": info["events"], "decisions": decisions,
              "turn": info["turn"], "depth": info["depth"],
              "max_depth": info["max_depth"], "reward": total_reward,
              "new_cells": info["new_cells"] if valid else None,
              "new_cells_by_depth": info["new_cells_by_depth"],
              "action_counts": action_counts, "stalled_moves": stalled_moves,
              "reset_s": reset_s, "policy_s": policy_s, "step_s": step_s,
              "reward_components": info["reward_totals"],
              "native_rss_peak_sampled_kib": info["native_rss_peak_sampled_kib"],
              "trace_sha256": env.trace_digest(),
              "duration_s": time.monotonic() - started}
    if not valid:
        record["partial_new_cells"] = info["new_cells"]
        record["error"] = str(info["terminal"])[:300]
    return record


def summarize(episodes):
    valid = [e for e in episodes if e.get("valid", True)]
    invalid = [e for e in episodes if not e.get("valid", True)]
    eligible = [e for e in valid if e["reason"] in {"DEATH", "QUIT", "VICTORY", "SUPERVICTORY"}]
    wins = sum(e["reason"] in {"VICTORY", "SUPERVICTORY"} for e in eligible)
    elapsed = sum(e["duration_s"] for e in episodes)

    def mean(values):
        return sum(values) / len(values) if values else None

    return {"started": len(episodes), "completed": len(eligible), "wins": wins,
            "valid": len(valid), "invalid_reasons": sorted(e["reason"] for e in invalid),
            "deaths": sum(e["reason"] == "DEATH" for e in eligible),
            "quits": sum(e["reason"] == "QUIT" for e in eligible),
            "truncated": sum(bool(e["truncated"]) for e in valid),
            "errors": len(invalid),
            "win_rate_eligible": wins / len(eligible) if eligible else None,
            "win_rate_wilson_95": wilson(wins, len(eligible)),
            "mean_max_depth": mean([e["max_depth"] for e in valid]),
            "mean_reward": mean([e["reward"] for e in valid]),
            "mean_new_cells": mean([e.get("new_cells", 0) for e in valid]),
            "events": sum(e["events"] for e in episodes),
            "decisions": sum(e["decisions"] for e in episodes),
            "turns": sum(e["turn"] for e in episodes),
            "stalled_moves": sum(e.get("stalled_moves", 0) for e in episodes),
            "reset_s": sum(e.get("reset_s", 0) for e in episodes),
            "policy_s": sum(e.get("policy_s", 0) for e in episodes),
            "step_s": sum(e.get("step_s", 0) for e in episodes),
            "native_rss_peak_sampled_kib": max((e["native_rss_peak_sampled_kib"] for e in episodes), default=0),
            "duration_s": elapsed,
            "events_per_s": sum(e["events"] for e in episodes) / elapsed if elapsed else 0,
            "decisions_per_s": sum(e["decisions"] for e in episodes) / elapsed if elapsed else 0,
            "turns_per_s": sum(e["turn"] for e in episodes) / elapsed if elapsed else 0,
            "episodes": episodes}
