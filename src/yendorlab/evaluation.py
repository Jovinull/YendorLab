"""Independent, bounded policy evaluation on real Brogue episodes."""

import hashlib
import math
import time

from .env import BrogueEnv, RewardConfig


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


def run_policy_episode(binary, agent, seed, variant="classic", reward="canonical",
                       max_events=300, max_seconds=30, max_no_progress=50,
                       memory_mb=512):
    started = time.monotonic()
    with BrogueEnv(binary, variant, max_events, max_seconds, max_no_progress,
                   memory_mb, RewardConfig.named(reward)) as env:
        observation, _ = env.reset(seed=seed)
        total_reward = 0.0
        decisions = 0
        while True:
            action = agent.act(observation)
            if observation["context"] == "gameplay":
                decisions += 1
            observation, earned, terminated, truncated, info = env.step(action)
            total_reward += earned
            if terminated or truncated:
                break
        return {"seed": seed, "reason": info["reason"], "terminated": terminated,
                "truncated": truncated, "events": info["events"], "decisions": decisions,
                "turn": info["turn"], "depth": info["depth"],
                "max_depth": info["max_depth"], "reward": total_reward,
                "reward_components": info["reward_totals"],
                "native_rss_peak_sampled_kib": info["native_rss_peak_sampled_kib"],
                "duration_s": time.monotonic() - started}


def summarize(episodes):
    eligible = [e for e in episodes if e["reason"] in {"DEATH", "QUIT", "VICTORY", "SUPERVICTORY"}]
    wins = sum(e["reason"] in {"VICTORY", "SUPERVICTORY"} for e in eligible)
    elapsed = sum(e["duration_s"] for e in episodes)
    return {"started": len(episodes), "completed": len(eligible), "wins": wins,
            "deaths": sum(e["reason"] == "DEATH" for e in eligible),
            "quits": sum(e["reason"] == "QUIT" for e in eligible),
            "truncated": sum(e["truncated"] and not e["reason"].endswith("ERROR") for e in episodes),
            "errors": sum(e["reason"].endswith("ERROR") for e in episodes),
            "win_rate_eligible": wins / len(eligible) if eligible else None,
            "win_rate_wilson_95": wilson(wins, len(eligible)),
            "mean_max_depth": sum(e["max_depth"] for e in episodes) / len(episodes) if episodes else None,
            "mean_reward": sum(e["reward"] for e in episodes) / len(episodes) if episodes else None,
            "events": sum(e["events"] for e in episodes),
            "decisions": sum(e["decisions"] for e in episodes),
            "duration_s": elapsed,
            "events_per_s": sum(e["events"] for e in episodes) / elapsed if elapsed else 0,
            "episodes": episodes}
