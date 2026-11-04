"""Transport stress with a scripted fake engine; no Brogue build is needed."""

import json
import os
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path

from yendorlab.env import BrogueEnv, ResetError

FAKE = r'''#!{python}
import json, os, sys, time
mode = os.environ["FAKE_MODE"]
seed = int(sys.argv[sys.argv.index("-s") + 1])
counter = 0

def observation(cells=1):
    global counter
    counter += 1
    return {{"protocol": 3, "observation_schema": 3, "action_schema": 2, "decision": counter,
             "context": "gameplay", "seed": seed, "depth": 1, "turn": counter, "x": 1, "y": 1,
             "hp": 40, "max_hp": 40, "width": 79, "height": 29, "admissible": ["z"],
             "cells": [[i % 79, i % 29, 1, 0, 0, 0] for i in range(cells)],
             "inventory": [], "floor_items": [], "creatures": []}}

def emit(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()

if mode == "garbage_reset":
    print("Recording buffer length exceeded", flush=True)
elif mode == "silent_exit":
    sys.exit(7)
elif mode == "hang_reset":
    time.sleep(30)
elif mode == "partial_then_exit":
    sys.stdout.write('{{"protocol": 3, "obs')
    sys.stdout.flush()
    sys.exit(9)
elif mode == "wrong_seed":
    seed += 1
    emit(observation())
    sys.stdin.readline()
else:
    emit(observation())
    while True:
        line = sys.stdin.readline()
        if not line:
            sys.exit(64)
        if mode == "burst":  # Two complete lines inside one pipe read.
            text = json.dumps(observation()) + "\n" + json.dumps(
                {{"event": "terminal", "result": "death", "score": 0, "depth": 1, "turn": 2}}) + "\n"
            sys.stdout.write(text)
            sys.stdout.flush()
        elif mode == "split":  # One line delivered across many pipe reads.
            text = json.dumps(observation()) + "\n"
            for start in range(0, len(text), 7):
                sys.stdout.write(text[start:start + 7])
                sys.stdout.flush()
        elif mode == "huge":  # Larger than a Linux pipe buffer.
            emit(observation(cells=12000))
        elif mode == "stderr_flood":
            sys.stderr.write("x" * 200000)
            sys.stderr.flush()
            emit(observation())
        elif mode == "invalid_utf8":
            message = observation()
            message["inventory"] = [{{"name": "PLACEHOLDER"}}]
            data = json.dumps(message).encode().replace(b"PLACEHOLDER", b"bad\xffname")
            sys.stdout.buffer.write(data + b"\n")
            sys.stdout.flush()
        elif mode == "bad_json_step":
            print("not json", flush=True)
        elif mode == "decision_gap":
            counter += 5
            emit(observation())
        elif mode == "hang_step":
            time.sleep(30)
        elif mode == "crash_step":
            sys.exit(11)
        else:
            emit(observation())
'''


@unittest.skipUnless(sys.platform.startswith("linux"), "fake engine needs POSIX pipes")
class TransportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.binary = Path(self.directory.name) / "fake-brogue"
        self.binary.write_text(FAKE.format(python=sys.executable))
        self.binary.chmod(self.binary.stat().st_mode | stat.S_IEXEC)
        self.addCleanup(self.directory.cleanup)

    def env(self, mode, **limits):
        os.environ["FAKE_MODE"] = mode
        self.addCleanup(os.environ.pop, "FAKE_MODE", None)
        settings = {"max_events": 50, "max_seconds": 3, "max_no_progress": 50}
        settings.update(limits)
        return BrogueEnv(self.binary, **settings)

    def test_two_lines_in_one_read_do_not_stall(self):
        with self.env("burst") as env:
            env.reset(seed=5)
            before = time.monotonic()
            _, _, terminated, _, info = env.step("z")
            self.assertFalse(terminated)  # The observation is first.
            self.assertLess(time.monotonic() - before, 1.0)
            _, _, terminated, _, info = env.step("z")
            self.assertTrue(terminated)
            self.assertEqual(info["reason"], "DEATH")

    def test_line_split_across_reads(self):
        with self.env("split") as env:
            env.reset(seed=5)
            for index in range(20):
                obs, _, terminated, truncated, _ = env.step("z")
                self.assertEqual(obs["decision"], index + 2)
                self.assertFalse(terminated or truncated)

    def test_observation_larger_than_pipe_buffer(self):
        with self.env("huge") as env:
            env.reset(seed=5)
            for _ in range(5):
                obs, _, _, truncated, _ = env.step("z")
                self.assertFalse(truncated)
                self.assertEqual(len(obs["cells"]), 12000)

    def test_stderr_flood_cannot_block_the_child(self):
        with self.env("stderr_flood", max_seconds=5) as env:
            env.reset(seed=5)
            for _ in range(5):
                _, _, _, truncated, info = env.step("z")
                self.assertFalse(truncated, info["reason"])

    def test_invalid_utf8_is_replaced_not_fatal(self):
        with self.env("invalid_utf8") as env:
            env.reset(seed=5)
            obs, _, _, truncated, _ = env.step("z")
            self.assertFalse(truncated)
            self.assertIn("�", obs["inventory"][0]["name"])

    def test_garbage_at_reset_names_the_offending_line(self):
        with self.env("garbage_reset") as env:
            with self.assertRaises(ResetError) as caught:
                env.reset(seed=5)
            self.assertIn("Recording buffer", str(caught.exception))

    def test_exit_at_reset_is_reset_error(self):
        with self.env("silent_exit") as env:
            with self.assertRaises(ResetError) as caught:
                env.reset(seed=5)
            self.assertIn("exit_7", str(caught.exception))

    def test_hang_at_reset_honours_deadline(self):
        with self.env("hang_reset", max_seconds=0.6) as env:
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                env.reset(seed=5)
            self.assertLess(time.monotonic() - started, 3.0)

    def test_partial_line_then_exit_is_engine_error(self):
        with self.env("partial_then_exit") as env:
            with self.assertRaises(ResetError) as caught:
                env.reset(seed=5)
            self.assertIn("partial_line", str(caught.exception))

    def test_wrong_seed_is_rejected(self):
        with self.env("wrong_seed") as env:
            with self.assertRaises(ValueError):
                env.reset(seed=5)

    def test_step_failures_are_classified(self):
        expected = {"bad_json_step": "PROTOCOL_ERROR", "decision_gap": "PROTOCOL_ERROR",
                    "crash_step": "ENGINE_ERROR"}
        for mode, reason in expected.items():
            with self.subTest(mode=mode), self.env(mode) as env:
                env.reset(seed=5)
                _, _, terminated, truncated, info = env.step("z")
                self.assertFalse(terminated)
                self.assertTrue(truncated)
                self.assertEqual(info["reason"], reason)

    def test_hang_during_step_honours_deadline(self):
        with self.env("hang_step", max_seconds=0.6) as env:
            env.reset(seed=5)
            _, _, _, truncated, info = env.step("z")
            self.assertTrue(truncated)
            self.assertEqual(info["reason"], "TRUNCATED_TIME_LIMIT")

    def test_private_workdir_is_removed_and_child_reaped(self):
        with self.env("normal") as env:
            env.reset(seed=5)
            workdir = Path(env.workdir)
            process = env.process
            self.assertTrue(workdir.is_dir())
        self.assertFalse(workdir.exists())
        self.assertIsNotNone(process.poll())

    def test_thousand_events_keep_sequence_and_digest(self):
        digests = []
        for _ in range(2):
            with self.env("normal", max_events=1000, max_seconds=20, max_no_progress=2000) as env:
                obs, _ = env.reset(seed=5)
                for index in range(1000):
                    obs, _, _, truncated, info = env.step("z")
                    self.assertEqual(obs["decision"], index + 2)
                self.assertTrue(truncated)
                self.assertEqual(info["reason"], "TRUNCATED_ACTION_LIMIT")
                digests.append(env.trace_digest())
        self.assertEqual(digests[0], digests[1])


if __name__ == "__main__":
    unittest.main()
