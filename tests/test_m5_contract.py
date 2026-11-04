"""M5 task, perception, encoder, learner and statistics contracts."""

import json
import tempfile
import unittest
from pathlib import Path

from yendorlab.benchmark import controls, evaluate, seed_pairs
from yendorlab.encoders import Encoder
from yendorlab.env import BrogueEnv
from yendorlab.evaluation import failed_episode, seed_for, summarize
from yendorlab.learners import REWARDS, TDTrainer, imitation_metrics, train_softmax
from yendorlab.policies import FrontierMoves, LinearPolicy, RandomMoves, build_policy
from yendorlab.stats import holm, hierarchical_interval, paired_comparison
from yendorlab.tasks import (MOVES, OFFSETS, SPLITS, Perception, m5_seed, metric_summary,
                             route, run_episode, run_with_retry, task_named)

BINARY = Path(".local/BrogueCE/bin/brogue")


def observation(cells, x=5, y=5, creatures=(), decision=1, depth=1):
    return {"context": "gameplay", "decision": decision, "x": x, "y": y, "width": 12, "height": 12,
            "depth": depth, "cells": [list(cell) for cell in cells], "creatures": list(creatures),
            "admissible": list(MOVES)}


def room(blocked=(), unknown=()):
    """A known 9x9 block around (5,5); every cell passable unless listed."""
    cells = []
    for x in range(1, 10):
        for y in range(1, 10):
            if (x, y) not in unknown:
                cells.append((x, y, 1, 1, int((x, y) in blocked), 0))
    return cells


def rotate_point(x, y):
    """Quarter turn about the center of a 12x12 map... (5,5) stays fixed."""
    return 5 - (y - 5), 5 + (x - 5)


def rotate_cells(cells):
    return [(*rotate_point(cell[0], cell[1]), *cell[2:]) for cell in cells]


ROTATED_MOVE = {}
for move, (dx, dy) in zip(MOVES, OFFSETS):
    ROTATED_MOVE[move] = MOVES[OFFSETS.index((-dy, dx))]


class SeedDomains(unittest.TestCase):
    def test_m5_splits_are_disjoint_and_separate_from_m4(self):
        groups = {split: {m5_seed(20261103, split, i) for i in range(2000)} for split in SPLITS}
        for first in SPLITS:
            for second in SPLITS:
                if first != second:
                    self.assertTrue(groups[first].isdisjoint(groups[second]))
        historic = {seed_for(master, split, i) for master in (20261030, 20261031, 20261032, 20261033)
                    for split in ("train", "validation", "final") for i in range(2000)}
        for split in SPLITS:
            self.assertTrue(groups[split].isdisjoint(historic))

    def test_game_and_policy_streams_differ(self):
        self.assertNotEqual(m5_seed(1, "final", 0), m5_seed(1, "final", 0, "policy"))


class PerceptionTests(unittest.TestCase):
    def test_failed_move_is_remembered_and_counted(self):
        perception = Perception()
        view = perception.observe(observation(room()))
        perception.acted(view, "h")
        view = perception.observe(observation(room(), decision=2))
        self.assertTrue(perception.stalled)
        self.assertIn((5, 5, MOVES.index("h")), view.failed)

    def test_attack_that_keeps_position_is_not_a_failed_move(self):
        perception = Perception()
        view = perception.observe(observation(room(), creatures=[(4, 5, 100)]))
        perception.acted(view, "h")
        view = perception.observe(observation(room(), creatures=[(4, 5, 100)], decision=2))
        self.assertFalse(perception.stalled)
        self.assertEqual(view.failed, set())

    def test_memory_does_not_cross_depth(self):
        perception = Perception()
        view = perception.observe(observation(room()))
        perception.acted(view, "h")
        view = perception.observe(observation(room(), decision=2, depth=2))
        self.assertFalse(perception.stalled)
        self.assertEqual(view.visits, {(5, 5): 1})

    def test_route_uses_every_shortest_path_and_respects_corners(self):
        cells = room(unknown=[(9, 5)])
        view = Perception().observe(observation(cells))
        mask, distance = route(view)
        self.assertEqual(distance, 3)
        # Cells two steps east reach the unknown cell; all three eastward moves tie.
        self.assertEqual({MOVES[i] for i in range(8) if mask >> i & 1}, {"l", "u", "n"})
        blocked = room(blocked=[(5, 4)], unknown=[(5, 1)])
        view = Perception().observe(observation(blocked))
        self.assertTrue(view.move_blocked(5, 5, MOVES.index("u")), "diagonal past a wall corner")

    def test_stairs_are_not_routed_through_on_a_floor_task(self):
        cells = [(5, 5, 1, 1, 0, 0), (6, 5, 1, 1, 0, 1), (7, 5, 1, 1, 0, 0)]
        view = Perception().observe(observation(cells))
        self.assertEqual(route(view)[0] >> MOVES.index("l") & 1, 0)

    def test_frontier_policy_attacks_an_adjacent_creature(self):
        view = Perception().observe(observation(room(), creatures=[(6, 5, 100)]))
        self.assertEqual(FrontierMoves().choices(view), [MOVES.index("l")])


class EncoderTests(unittest.TestCase):
    def assert_rotation_equivariant(self, encoder):
        cells = room(blocked=[(3, 3), (6, 4), (7, 7)], unknown=[(8, 2), (2, 7), (9, 5)])
        creatures = [(6, 6, 100)]
        first = Perception()
        view = first.observe(observation(cells, creatures=creatures))
        turned = Perception()
        rotated_view = turned.observe(observation(rotate_cells(cells), creatures=[
            (*rotate_point(6, 6), 100)]))
        plain = encoder(view)
        spun = encoder(rotated_view)
        for move in MOVES:
            self.assertEqual(sorted(plain[move]), sorted(spun[ROTATED_MOVE[move]]), move)

    def test_relative_encoder_has_no_absolute_direction(self):
        self.assert_rotation_equivariant(Encoder("rel_v3"))

    def test_patch_encoder_rotates_with_the_map(self):
        self.assert_rotation_equivariant(Encoder("ego_patch_v1"))

    def test_m4_encoder_is_not_equivariant(self):
        with self.assertRaises(AssertionError):
            self.assert_rotation_equivariant(Encoder("abs_v2"))

    def test_ablation_removes_exactly_the_group(self):
        view = Perception().observe(observation(room(unknown=[(9, 5)])))
        full = Encoder("rel_v3")(view)
        without = Encoder("rel_v3", ("frontier",))(view)
        index = Encoder("rel_v3").names.index("on_route")
        self.assertTrue(any(i == index for features in full.values() for i, _ in features))
        self.assertFalse(any(i == index for features in without.values() for i, _ in features))

    def test_unknown_group_is_rejected(self):
        with self.assertRaises(ValueError):
            Encoder("rel_v3", ("nonsense",))


class PolicyTests(unittest.TestCase):
    def test_zero_weights_do_not_prefer_a_direction(self):
        view = Perception().observe(observation(room()))
        policy = LinearPolicy(Encoder("rel_v3"), seed=3)
        chosen = {policy.decide(None, view) for _ in range(300)}
        self.assertEqual(chosen, set(MOVES))

    def test_seeded_policies_repeat(self):
        view = Perception().observe(observation(room()))
        first = [RandomMoves(9).decide(None, view) for _ in range(1)]
        self.assertEqual(first, [RandomMoves(9).decide(None, view)])


class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.encoder = Encoder("rel_v3")
        self.view = Perception().observe(observation(room()))
        self.policy = LinearPolicy(self.encoder, seed=1)
        self.policy.decide(None, self.view)

    def test_terminal_has_no_bootstrap_and_budget_end_does(self):
        trainer = TDTrainer(self.policy, 0.5, 0.9, REWARDS["cells_v2"])
        self.policy.weights = [1.0] * self.encoder.dimension
        scored = self.policy.score(self.policy.last_features)
        trainer._update(0.0, None)
        lowered = self.policy.score(self.policy.last_features)
        self.assertLess(lowered, scored)
        self.policy.weights = [1.0] * self.encoder.dimension
        trainer._update(0.0, self.view)
        self.assertGreater(self.policy.score(self.policy.last_features), lowered)

    def test_weights_stay_bounded_and_finite(self):
        trainer = TDTrainer(self.policy, 1.0, 0.99, REWARDS["cells_v2"])
        for _ in range(3000):
            trainer._update(1000.0, self.view)
        self.assertLessEqual(max(map(abs, self.policy.weights)), 50.0)

    def test_stall_reward_is_separate_from_cells(self):
        perception = Perception()
        view = perception.observe(observation(room()))
        perception.acted(view, "h")
        perception.observe(observation(room(), decision=2))
        plain = TDTrainer(self.policy, .1, .9, REWARDS["cells_v2"])
        shaped = TDTrainer(self.policy, .1, .9, REWARDS["cells_shaped_v2"])
        self.assertEqual(plain.step_reward(perception), 0.0)
        self.assertLess(shaped.step_reward(perception), 0.0)
        self.assertEqual(plain.cell_reward(0), 0.0)


class ImitationTests(unittest.TestCase):
    def test_softmax_learns_a_separable_preference(self):
        samples = [{"moves": [1], "features": {"k": [[(0, 1.0)], [(1, 1.0)]] + [[] for _ in range(6)]}}
                   for _ in range(40)]
        weights, history = train_softmax(samples, "k", 2, epochs=8, rate=.3)
        self.assertLess(history[-1], history[0])
        self.assertEqual(imitation_metrics(weights, samples, "k")["set_accuracy"], 1.0)


class FailureAccounting(unittest.TestCase):
    def test_operational_failure_does_not_enter_the_mean(self):
        good = {"seed": 1, "reason": "TRUNCATED_ACTION_LIMIT", "valid": True, "truncated": True,
                "events": 10, "decisions": 10, "turn": 10, "max_depth": 1, "reward": 1.0,
                "new_cells": 100, "native_rss_peak_sampled_kib": 1, "duration_s": .1}
        import time
        bad = failed_episode(2, "ENGINE_ERROR", RuntimeError("x"), time.monotonic())
        summary = summarize([good, bad])
        self.assertEqual(summary["mean_new_cells"], 100)
        self.assertEqual((summary["started"], summary["valid"], summary["errors"]), (2, 1, 1))
        self.assertEqual(summary["invalid_reasons"], ["ENGINE_ERROR"])

    def test_task_summary_keeps_deaths_and_excludes_failures(self):
        common = {"decisions": 10, "turn": 10, "events": 12, "stalled_moves": 1, "revisit_moves": 1,
                  "action_counts": {"k": 10}, "duration_s": .1, "policy_s": 0, "step_s": 0,
                  "reset_s": 0, "new_cells_all_depths": 5}
        episodes = [{**common, "seed": 1, "reason": "DEATH", "valid": True, "new_cells": 5},
                    {**common, "seed": 2, "reason": "TASK_BUDGET", "valid": True, "new_cells": 15},
                    {"seed": 3, "reason": "ENGINE_ERROR", "valid": False, "new_cells": None, "duration_s": .1}]
        summary = metric_summary(episodes)
        self.assertEqual(summary["mean_new_cells"], 10)
        self.assertEqual((summary["deaths"], summary["invalid"]), (1, 1))


class StatisticsTests(unittest.TestCase):
    def test_paired_comparison_reports_pairs_and_drops(self):
        result = paired_comparison({1: 5, 2: 6, 3: 7}, {1: 1, 2: 2})
        self.assertEqual((result["pairs"], result["dropped"], result["mean_difference"]), (2, 1, 4))

    def test_holm_is_monotone_and_bounded(self):
        adjusted = holm({"a": .01, "b": .04, "c": .03})
        self.assertAlmostEqual(adjusted["a"], .03)
        self.assertAlmostEqual(adjusted["c"], .06)
        self.assertAlmostEqual(adjusted["b"], .06)

    def test_hierarchical_interval_contains_mean(self):
        low, high = hierarchical_interval([[1, 2, 3], [2, 3, 4], [3, 4, 5]], resamples=500)
        self.assertLess(low, 3)
        self.assertGreater(high, 3)


@unittest.skipUnless(BINARY.is_file(), "native build absent")
class NativeTaskTests(unittest.TestCase):
    binary = BINARY.resolve()

    def test_every_control_gets_the_same_budget_and_actions(self):
        task = task_named("exploration_v2_floor", 40)
        pairs = seed_pairs(20261103, "diagnostic", 2)
        for name, spec in controls().items():
            result = evaluate(self.binary, spec, task, pairs)
            for episode in result["episodes"]:
                self.assertTrue(episode["valid"], name)
                self.assertLessEqual(episode["decisions"], 40)
                self.assertTrue(set(episode["action_counts"]) <= set(MOVES))
                if episode["reason"] == "TASK_BUDGET":
                    self.assertEqual(episode["decisions"], 40)

    def test_replay_is_identical_with_policy_rng_fixed(self):
        task = task_named("exploration_v2_floor", 60)
        seed, policy_seed = seed_pairs(20261103, "diagnostic", 1)[0]
        for spec in ({"kind": "random"}, {"kind": "frontier"}, {"kind": "linear", "encoder": "rel_v3"}):
            traces = [run_episode(self.binary, build_policy(spec, policy_seed), seed, task, policy_seed)
                      for _ in range(2)]
            self.assertEqual(traces[0]["trace_sha256"], traces[1]["trace_sha256"])
            self.assertEqual(traces[0]["new_cells"], traces[1]["new_cells"])

    def test_metric_counts_only_the_start_depth_and_never_reward(self):
        task = task_named("exploration_v2_multilevel", 450)
        pairs = seed_pairs(20261103, "diagnostic", 3)
        result = evaluate(self.binary, {"kind": "frontier", "stairs_goal": True}, task, pairs)
        reached = [e for e in result["episodes"] if len(e["depths_visited"]) > 1]
        self.assertTrue(reached, "guide never left depth 1 in this sample")
        for episode in reached:
            self.assertEqual(episode["new_cells"], episode["new_cells_by_depth"]["1"])
            self.assertEqual(episode["new_cells_all_depths"], sum(episode["new_cells_by_depth"].values()))

    def test_floor_task_ends_at_a_depth_change(self):
        task = task_named("exploration_v2_floor", 400)
        pairs = seed_pairs(20261103, "diagnostic", 6)
        result = evaluate(self.binary, {"kind": "frontier", "stairs_goal": True}, task, pairs)
        exits = [e for e in result["episodes"] if e["reason"] == "TASK_DEPTH_EXIT"]
        self.assertTrue(exits)
        self.assertTrue(all(e["depths_visited"] == [1] or len(e["depths_visited"]) == 2 for e in exits))

    def test_relay_hands_over_on_depth_two_with_a_fresh_budget(self):
        task = task_named("exploration_v2_relay", 30)
        pairs = seed_pairs(20261103, "diagnostic", 4)
        result = evaluate(self.binary, {"kind": "random"}, task, pairs)
        positions = set()
        for episode in result["episodes"]:
            if episode["valid"]:
                self.assertEqual(episode["start_depth"], 2)
                self.assertLessEqual(episode["decisions"], 30)
                positions.add(tuple(episode["start_position"]))
        self.assertGreater(len(positions), 1)

    def test_long_stall_does_not_corrupt_the_stream(self):
        # A wall bump costs no turn; upstream flushed its recording only on turns.
        task = task_named("exploration_v2_floor", 1500, preroll_budget=0)
        result = run_episode(self.binary, build_policy({"kind": "fixed", "action": "b"}, 1), 123, task)
        self.assertEqual(result["reason"], "TASK_BUDGET")
        self.assertEqual(result["decisions"], 1500)

    def test_declined_prompt_after_recording_flush_does_not_hang(self):
        # A real training sequence. Flushing the recording while a prompt was open let
        # the decline underflow the buffer position and spin forever writing to disk.
        fixture = json.loads(Path("tests/fixtures/prompt-after-recording-flush.json").read_text())
        with BrogueEnv(self.binary, max_events=len(fixture["keys"]) + 5, max_seconds=15,
                       max_no_progress=10000) as env:
            env.reset(seed=fixture["seed"])
            for key in fixture["keys"]:
                _, _, terminated, truncated, info = env.step(key)
                self.assertFalse(terminated or truncated, info["reason"])

    def test_policy_exception_is_a_policy_error_not_a_result(self):
        class Broken(RandomMoves):
            def decide(self, observation, view):
                raise ValueError("bug in policy")

        task = task_named("exploration_v2_floor", 10)
        record = run_episode(self.binary, Broken(1), 123, task)
        self.assertEqual(record["reason"], "POLICY_ERROR")
        self.assertFalse(record["valid"])
        self.assertIsNone(record["new_cells"])

    def test_action_outside_the_contract_is_refused(self):
        class Rester(RandomMoves):
            def decide(self, observation, view):
                return "z"

        record = run_episode(self.binary, Rester(1), 123, task_named("exploration_v2_floor", 10))
        self.assertEqual(record["reason"], "POLICY_ERROR")

    def test_retry_records_every_failed_attempt(self):
        task = task_named("exploration_v2_floor", 10)
        missing = Path("/nonexistent/brogue")
        record = run_with_retry(missing, lambda: RandomMoves(1), 123, task, 1, retries=2)
        self.assertEqual(record["reason"], "RESET_ERROR")
        self.assertEqual(record["attempts"], 3)
        self.assertEqual(len(record["failed_attempts"]), 3)


if __name__ == "__main__":
    unittest.main()
