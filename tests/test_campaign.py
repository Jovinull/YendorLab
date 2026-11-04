"""Campaign recovery: crashes between journal and checkpoint must not lose results."""

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from yendorlab.campaign import Campaign, status, weights_hash

BINARY = Path(".local/BrogueCE/bin/brogue")


def config(episodes=4, **train):
    settings = {"experiment": "TEST", "task": "exploration_v2_floor", "budget": 25,
                "encoder": "rel_v3", "reward": "cells_v2", "master_seed": 777, "run": 0,
                "episodes": episodes, "alpha": .1, "gamma": .95, "epsilon_start": .3,
                "epsilon_end": .1, "epsilon_decay_episodes": 4, "validation_every": 2,
                "validation_episodes": 1, "checkpoint_every": 1,
                "periodic_checkpoint_every": 2, "keep_periodic": 1}
    settings.update(train)
    return {"train": settings, "limits": {"episode_seconds": 30, "memory_mb": 512}}


CHILD = """
import json, sys
from pathlib import Path
from yendorlab.campaign import Campaign
config = json.loads(sys.argv[1])
outcome = Campaign(config, Path(sys.argv[2]), Path(sys.argv[3]).resolve(), fault=sys.argv[4] or None).run(resume=sys.argv[5] == "1")
print(json.dumps(outcome))
"""


def child(settings, directory, fault="", resume=False):
    environment = {**os.environ, "PYTHONPATH": "src"}
    return subprocess.run([sys.executable, "-c", CHILD, json.dumps(settings), str(directory),
                           str(BINARY), fault, "1" if resume else "0"],
                          capture_output=True, text=True, env=environment, timeout=120)


@unittest.skipUnless(BINARY.is_file(), "native build absent")
class CampaignRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="campaign-test-"))
        run = child(config(), cls.root / "clean")
        assert run.returncode == 0, run.stderr
        cls.clean = json.loads(run.stdout.splitlines()[-1])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def rows(self, directory, sql):
        database = sqlite3.connect(directory / "journal.sqlite3")
        rows = database.execute(sql).fetchall()
        database.close()
        return rows

    def test_clean_run_finishes_and_keeps_artifacts(self):
        self.assertEqual(self.clean["status"], "STOPPED:EPISODE_TARGET")
        self.assertEqual(self.clean["episodes_completed"], 4)
        directory = self.root / "clean"
        manifest = json.loads((directory / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "STOPPED:EPISODE_TARGET")
        self.assertIn("best.json", manifest["artifacts"])
        self.assertEqual(len(list((directory / "checkpoints").glob("ep-*.json"))), 1)
        report = status(directory)
        self.assertEqual(report["journal_valid_episodes"], 4)
        self.assertEqual(report["episodes_completed"], 4)
        self.assertEqual(len(report["validation_curve"]), 2)
        self.assertEqual(report["operational_failures"], {})

    def recover(self, fault, code):
        directory = self.root / fault.replace(":", "_")
        crashed = child(config(), directory, fault)
        self.assertEqual(crashed.returncode, code, crashed.stderr)
        resumed = child(config(), directory, resume=True)
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        outcome = json.loads(resumed.stdout.splitlines()[-1])
        self.assertEqual(outcome["weights_sha256"], self.clean["weights_sha256"])
        self.assertEqual(outcome["updates"], self.clean["updates"])
        return directory

    def test_crash_after_journal_before_checkpoint(self):
        directory = self.recover("after_journal", 41)
        live = self.rows(directory, "SELECT idx FROM episodes WHERE superseded=0 ORDER BY idx")
        self.assertEqual([row[0] for row in live], [0, 1, 2, 3])
        superseded = self.rows(directory, "SELECT idx FROM episodes WHERE superseded=1")
        self.assertEqual(superseded, [(0,)], "the orphan row is kept, not deleted")

    def test_crash_after_checkpoint(self):
        directory = self.recover("after_checkpoint", 42)
        live = self.rows(directory, "SELECT idx FROM episodes WHERE superseded=0 ORDER BY idx")
        self.assertEqual([row[0] for row in live], [0, 1, 2, 3])

    def test_controlled_stop_then_resume_matches_uninterrupted(self):
        directory = self.root / "controlled"
        first = child(config(), directory, "stop_at:2")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(json.loads(first.stdout.splitlines()[-1])["episodes_completed"], 2)
        second = child(config(), directory, resume=True)
        outcome = json.loads(second.stdout.splitlines()[-1])
        self.assertEqual(outcome["weights_sha256"], self.clean["weights_sha256"])

    def test_sigterm_stops_at_an_episode_boundary(self):
        directory = self.root / "signal"
        settings = config(episodes=400)
        environment = {**os.environ, "PYTHONPATH": "src"}
        process = subprocess.Popen([sys.executable, "-c", CHILD, json.dumps(settings), str(directory),
                                    str(BINARY), "", "0"], env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not (directory / "journal.sqlite3").exists():
            time.sleep(.05)
        time.sleep(1.0)
        process.terminate()
        stdout, stderr = process.communicate(timeout=30)
        self.assertEqual(process.returncode, 0, stderr)
        outcome = json.loads(stdout.splitlines()[-1])
        self.assertEqual(outcome["status"], "STOPPED:SIGNAL")
        self.assertEqual(json.loads((directory / "checkpoints/latest.json").read_text())["episodes_completed"],
                         outcome["episodes_completed"])

    def test_stop_file_then_larger_episode_target(self):
        directory = self.root / "stopfile"
        directory.mkdir()
        (directory / "STOP").write_text("")
        outcome = Campaign(config(episodes=2), directory, BINARY.resolve()).run()
        self.assertEqual((outcome["status"], outcome["episodes_completed"]), ("STOPPED:STOP_FILE", 0))
        (directory / "STOP").unlink()
        first = Campaign(config(episodes=2), directory, BINARY.resolve()).run(resume=True)
        self.assertEqual(first["episodes_completed"], 2)
        second = Campaign(config(episodes=4), directory, BINARY.resolve()).run(resume=True)
        self.assertEqual(second["weights_sha256"], self.clean["weights_sha256"])

    def test_resume_refuses_a_changed_binary_or_config(self):
        directory = self.root / "clean"
        with self.assertRaises(ValueError):
            Campaign(config(alpha=.2), directory, BINARY.resolve()).run(resume=True)
        fake = self.root / "other-binary"
        fake.write_bytes(b"not brogue")
        fake.chmod(0o755)
        with self.assertRaises(ValueError):
            Campaign(config(), directory, fake).run(resume=True)

    def test_new_campaign_never_overwrites_an_old_one(self):
        with self.assertRaises(FileExistsError):
            Campaign(config(), self.root / "clean", BINARY.resolve()).run()

    def test_operational_failure_commits_nothing_and_is_logged(self):
        directory = self.root / "engine-fails"
        fake = self.root / "broken-brogue"
        fake.write_text("#!/bin/sh\nexit 3\n")
        fake.chmod(0o755)
        outcome = Campaign(config(), directory, fake).run()
        self.assertEqual(outcome["status"], "FAILED_OPERATIONAL")
        self.assertEqual(outcome["episodes_completed"], 0)
        self.assertEqual(outcome["operational_failures"], 3)
        self.assertEqual(json.loads((directory / "checkpoints/latest.json").read_text())["weights"],
                         [0.0] * 15)
        self.assertEqual(status(directory)["operational_failures"], {"RESET_ERROR": 3})


if __name__ == "__main__":
    unittest.main()
