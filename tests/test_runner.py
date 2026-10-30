import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path

from yendorlab.cli import ACTIONS, run_episode


class RunnerTests(unittest.TestCase):
    def test_action_contract(self):
        self.assertEqual(len(ACTIONS), 8)
        self.assertEqual(len(set(ACTIONS)), 8)

    def test_sqlite_schema_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = sqlite3.connect(Path(tmp) / "runs.sqlite3")
            db.execute("CREATE TABLE runs (result_json TEXT)")
            db.execute("INSERT INTO runs VALUES (?)", (json.dumps({"completed": 0}),))
            self.assertEqual(json.loads(db.execute("SELECT result_json FROM runs").fetchone()[0])["completed"], 0)


class NativeTests(unittest.TestCase):
    @unittest.skipUnless(Path(".local/BrogueCE/bin/brogue").is_file(), "native build absent")
    def test_observation_excludes_internal_state(self):
        binary = Path(".local/BrogueCE/bin/brogue").resolve()
        process = subprocess.Popen([str(binary), "--yendorlab", "-n", "-s", "123"],
                                   cwd=binary.parent, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, text=True)
        try:
            observation = json.loads(process.stdout.readline())
            self.assertEqual(observation["protocol"], 2)
            self.assertEqual(observation["observation_schema"], 2)
            self.assertEqual(observation["context"], "gameplay")
            self.assertEqual(observation["seed"], 123)
            self.assertIn([observation["x"], observation["y"]],
                          [[cell[0], cell[1]] for cell in observation["cells"]])
            self.assertNotIn("rng", observation)
        finally:
            process.terminate()
            process.communicate(timeout=2)

    @unittest.skipUnless(Path(".local/BrogueCE/bin/brogue").is_file(), "native build absent")
    def test_seed_and_actions_repeat(self):
        import time

        binary = Path(".local/BrogueCE/bin/brogue").resolve()
        a = run_episode(binary, 123, 100123, 15, time.monotonic() + 10)
        b = run_episode(binary, 123, 100123, 15, time.monotonic() + 10)
        self.assertEqual(a["observation_hashes"], b["observation_hashes"])
        self.assertEqual((a["actions"], a["status"]), (15, "truncated"))
        self.assertEqual(b["status"], "truncated")


if __name__ == "__main__":
    unittest.main()
