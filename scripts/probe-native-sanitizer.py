"""Bounded ASan/UBSan probes: real death, long stall and planner exploration."""

import json
import select
import subprocess
import tempfile
import time
from pathlib import Path

from yendorlab.agents import context_action
from yendorlab.policies import FrontierMoves
from yendorlab.tasks import Perception

BINARY = Path(".local/BrogueCE/bin/brogue").resolve()


class Child:
    def __init__(self, seed):
        self.directory = tempfile.TemporaryDirectory(prefix="yendorlab-sanitizer-")
        self.process = subprocess.Popen([str(BINARY), "--yendorlab", "-n", "-s", str(seed)],
                                        cwd=self.directory.name, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, bufsize=1)

    def read(self, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
            raise TimeoutError("sanitizer probe timed out")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"native output ended: {self.process.stderr.read()[-3000:]}")
        return json.loads(line)

    def send(self, action):
        self.process.stdin.write(action + "\n")
        self.process.stdin.flush()

    def close(self):
        self.process.terminate()
        try:
            _, stderr = self.process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            _, stderr = self.process.communicate()
        self.directory.cleanup()
        if "ERROR: AddressSanitizer" in stderr or "runtime error:" in stderr:
            raise RuntimeError(stderr[-4000:])


def death():
    child = Child(123)
    try:
        deadline = time.monotonic() + 30
        for index in range(100):
            message = child.read(deadline)
            if message.get("event") == "terminal":
                assert message["result"] == "death", message
                print(f"SANITIZER_DEATH_PASS events={index}")
                return
            child.send("z" if message["context"] == "gameplay" else context_action(message))
        raise RuntimeError("no terminal event within 100 inputs")
    finally:
        child.close()


def stall(events=1600):
    """Turnless inputs past the upstream recording buffer."""
    child = Child(123)
    try:
        deadline = time.monotonic() + 60
        for _ in range(events):
            message = child.read(deadline)
            assert message.get("protocol") == 3, message
            child.send("b" if message["context"] == "gameplay" else context_action(message))
        print(f"SANITIZER_STALL_PASS events={events}")
    finally:
        child.close()


def explore(decisions=500):
    child = Child(7)
    policy, perception = FrontierMoves(1, stairs_goal=True), Perception()
    try:
        deadline = time.monotonic() + 120
        count = depth = 0
        while count < decisions:
            message = child.read(deadline)
            if message.get("event") == "terminal":
                break
            depth = max(depth, message["depth"])
            if message["context"] == "gameplay":
                view = perception.observe(message)
                action = policy.decide(message, view)
                perception.acted(view, action)
                count += 1
            else:
                action = context_action(message)
            child.send(action)
        print(f"SANITIZER_EXPLORE_PASS decisions={count} max_depth={depth}")
    finally:
        child.close()


death()
stall()
explore()
