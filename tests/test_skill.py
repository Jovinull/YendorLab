import tempfile
import hashlib
import unittest
from pathlib import Path

from yendorlab.skill import LinearFrontierAgent, frontier_action, load_skill_checkpoint, train_skill
from yendorlab.evaluation import run_policy_episode, seed_for


BINARY = Path(".local/BrogueCE/bin/brogue")


def observation():
    return {"context": "gameplay", "decision": 1, "x": 2, "y": 2,
            "width": 6, "height": 6, "cells": [[2, 2, 1, 1, 0, 0],
             [3, 2, 1, 1, 0, 0], [1, 2, 1, 1, 1, 0]], "creatures": [],
            "admissible": list("hjklyubn")}


class FeatureTests(unittest.TestCase):
    def test_encoder_distinguishes_known_wall_and_unknown(self):
        obs = observation()
        agent = LinearFrontierAgent()
        features = agent.features(obs)
        self.assertEqual(features["h"][2], 1.0)
        self.assertEqual(features["k"][1], 1.0)
        self.assertEqual(features["l"][1], 0.0)
        self.assertEqual(frontier_action(obs), "j")

    def test_visible_enemy_changes_feature_without_hidden_lookup(self):
        obs = observation()
        agent = LinearFrontierAgent()
        self.assertEqual(agent.features(obs)["l"][3], 0.0)
        obs["creatures"] = [[3, 2, 100]]
        self.assertEqual(agent.features(obs)["l"][3], 1.0)

    def test_reward_changes_weights_and_checkpoint_rejects_wrong_encoder(self):
        agent = LinearFrontierAgent()
        feature = agent.features(observation())["k"]
        self.assertTrue(agent.update(feature, 1.0, None))
        self.assertGreater(agent.score(feature), 0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            path.write_text('{"algorithm":"linear_frontier_td_v1","encoder_version":1}')
            with self.assertRaises(ValueError):
                load_skill_checkpoint(path)


@unittest.skipUnless(BINARY.is_file(), "native build absent")
class SkillNativeTests(unittest.TestCase):
    def test_real_training_and_episode_boundary_resume(self):
        config = {"experiment": "TEST", "algorithm": "linear_frontier_td_v1",
                  "task": "known_map_exploration_v1", "variant": "classic", "episodes": 1,
                  "total_seconds": 20.0, "episode_seconds": 5.0, "max_events": 30,
                  "max_no_progress": 20, "memory_mb": 512, "master_seed": 7788,
                  "epsilon": 0.5, "alpha": 0.02, "gamma": 0.95,
                  "validation_every": 2, "validation_episodes": 1}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            first = train_skill(BINARY.resolve(), config, path)
            self.assertGreater(first["changed_updates_total"], 0)
            self.assertNotEqual(first["weights"], [0.0] * 16)
            self.assertEqual(load_skill_checkpoint(Path(first["inference_checkpoint"]))["weights"],
                             first["weights"])
            inference = Path(first["inference_checkpoint"])
            checksum = hashlib.sha256(inference.read_bytes()).hexdigest()
            game_seed = seed_for(config["master_seed"], "validation", 7)
            first_eval = run_policy_episode(BINARY.resolve(), LinearFrontierAgent(first["weights"]),
                                            game_seed, max_events=20, max_seconds=5)
            second_eval = run_policy_episode(BINARY.resolve(),
                                             LinearFrontierAgent(load_skill_checkpoint(inference)["weights"]),
                                             game_seed, max_events=20, max_seconds=5)
            self.assertEqual(first_eval["new_cells"], second_eval["new_cells"])
            self.assertEqual(checksum, hashlib.sha256(inference.read_bytes()).hexdigest())
            config["episodes"] = 2
            second = train_skill(BINARY.resolve(), config, path, resume=True)
            self.assertEqual(second["episodes_total"], 2)
            uninterrupted = train_skill(BINARY.resolve(), config, Path(directory) / "full.json")
            self.assertEqual(second["weights"], uninterrupted["weights"])


if __name__ == "__main__":
    unittest.main()
