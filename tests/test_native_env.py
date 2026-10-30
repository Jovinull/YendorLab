import unittest
from pathlib import Path

from yendorlab.agents import context_action
from yendorlab.env import BrogueEnv, RewardConfig


BINARY = Path(".local/BrogueCE/bin/brogue")


@unittest.skipUnless(BINARY.is_file(), "native build absent")
class NativeEnvironmentTests(unittest.TestCase):
    def test_turns_and_inventory_context(self):
        with BrogueEnv(BINARY, max_events=20) as env:
            obs, _ = env.reset(seed=123)
            self.assertEqual(obs["turn"], 0)
            self.assertEqual(obs["context"], "gameplay")
            obs, _, ended, truncated, _ = env.step("i")
            self.assertEqual(obs["context"], "inventory")
            self.assertEqual(obs["turn"], 0)
            self.assertFalse(ended or truncated)
            obs, _, _, _, _ = env.step("ESC")
            self.assertEqual(obs["context"], "gameplay")
            obs, _, _, _, _ = env.step("z")
            self.assertEqual(obs["turn"], 1)

    def test_item_selection_and_confirmation_consume_one_turn(self):
        with BrogueEnv(BINARY, max_events=15) as env:
            obs, _ = env.reset(seed=123)
            food = next(item for item in obs["inventory"] if item["name"] == "food")
            obs, _, _, _, _ = env.step("a")
            self.assertEqual(obs["context"], "inventory")
            obs, _, _, _, _ = env.step(food["letter"])
            self.assertEqual(obs["context"], "confirmation")
            self.assertEqual(obs["turn"], 0)
            obs, _, _, _, _ = env.step("y")
            self.assertEqual(obs["context"], "gameplay")
            self.assertEqual(obs["turn"], 1)
            self.assertFalse(any(item["name"] == "food" for item in obs["inventory"]))

    def test_rest_does_not_farm_exploration_reward(self):
        with BrogueEnv(BINARY, max_events=10,
                       reward=RewardConfig.named("exploration_v1")) as env:
            obs, _ = env.reset(seed=123)
            for _ in range(2):
                obs, earned, _, _, info = env.step("z")
                self.assertEqual(earned, 0.0)
                self.assertEqual(sum(info["reward_totals"].values()), 0.0)

    def test_no_progress_limit_is_distinct_from_death(self):
        with BrogueEnv(BINARY, max_events=10, max_no_progress=1) as env:
            obs, _ = env.reset(seed=123)
            self.assertEqual(next(c[4] for c in obs["cells"] if c[0:2] == [37, 27]), 1)
            _, _, terminated, truncated, info = env.step("b")
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info["reason"], "TRUNCATED_NO_PROGRESS")

    def test_death_is_real_terminal_event(self):
        with BrogueEnv(BINARY, max_events=200, max_seconds=15,
                       reward=RewardConfig.named("canonical")) as env:
            obs, _ = env.reset(seed=123)
            for _ in range(200):
                action = "z" if obs["context"] == "gameplay" else context_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["reason"], "DEATH")
            self.assertEqual(info["turn"], 76)
            self.assertEqual(info["events"], 80)
            self.assertEqual(reward, -1.0)

    def test_quit_is_not_death(self):
        with BrogueEnv(BINARY, max_events=20, max_seconds=10) as env:
            obs, _ = env.reset(seed=123)
            for _ in range(20):
                action = {"gameplay": "Q", "confirmation": "y"}.get(obs["context"])
                obs, _, terminated, truncated, info = env.step(action or context_action(obs))
                if terminated or truncated:
                    break
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["reason"], "QUIT")
            self.assertEqual(info["turn"], 0)

    def test_hidden_entities_not_exposed_as_lists(self):
        with BrogueEnv(BINARY) as env:
            obs, _ = env.reset(seed=123)
            visible = {(cell[0], cell[1]) for cell in obs["cells"] if cell[2]}
            for item in obs["floor_items"]:
                self.assertIn((item["x"], item["y"]), visible)
            for creature in obs["creatures"]:
                self.assertIn((creature[0], creature[1]), visible)
            self.assertTrue(all(len(cell) == 6 for cell in obs["cells"]))
            self.assertTrue(all(0 <= cell[0] < obs["width"] and 0 <= cell[1] < obs["height"]
                                for cell in obs["cells"]))

    def test_engine_error_is_not_natural_outcome(self):
        with BrogueEnv(BINARY, max_events=10, max_seconds=5) as env:
            obs, _ = env.reset(seed=123)
            env.process.kill()
            env.process.wait(timeout=2)
            _, _, terminated, truncated, info = env.step("z")
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info["reason"], "ENGINE_ERROR")


if __name__ == "__main__":
    unittest.main()
