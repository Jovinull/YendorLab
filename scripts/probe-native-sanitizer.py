"""Bounded real-death probe for an ASan/UBSan Brogue build."""

import json
import select
import subprocess
import time
from pathlib import Path

from yendorlab.agents import context_action


binary = Path(".local/BrogueCE/bin/brogue").resolve()
process = subprocess.Popen([str(binary), "--yendorlab", "-n", "-s", "123"],
                           cwd=binary.parent, stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, bufsize=1)
try:
    deadline = time.monotonic() + 20
    for index in range(100):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([process.stdout], [], [], remaining)[0]:
            raise TimeoutError(f"sanitizer probe timed out at event {index}")
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"native output ended at event {index}")
        message = json.loads(line)
        if message.get("event") == "terminal":
            assert message["result"] == "death", message
            print(f"SANITIZER_DEATH_PASS events={index}")
            break
        action = "z" if message["context"] == "gameplay" else context_action(message)
        process.stdin.write(action + "\n")
        process.stdin.flush()
    else:
        raise RuntimeError("no terminal event within 100 inputs")
finally:
    process.terminate()
    try:
        _, stderr = process.communicate(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        _, stderr = process.communicate()
    if "ERROR: AddressSanitizer" in stderr or "runtime error:" in stderr:
        raise RuntimeError(stderr[-4000:])
