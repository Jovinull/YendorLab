#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
restore() {
  make -C .local/BrogueCE -j2 GRAPHICS=NO TERMINAL=NO RELEASE=YES bin/brogue > .local/native-production-build.log 2>&1
}
trap restore EXIT
make -C .local/BrogueCE -j2 GRAPHICS=NO TERMINAL=NO RELEASE=YES CPPFLAGS=-DYENDORLAB_TESTING bin/brogue > .local/native-fixture-build.log 2>&1
python3 - <<'PY'
import json
import subprocess
from pathlib import Path

binary = Path('.local/BrogueCE/bin/brogue').resolve()
for expected in ('victory', 'supervictory', 'death', 'quit'):
    run = subprocess.run([str(binary), '--yendorlab-test-event', expected],
                         cwd=binary.parent, capture_output=True, text=True, check=True)
    event = json.loads(run.stdout)
    assert event['event'] == 'terminal' and event['result'] == expected
    print(expected, event['score'])
PY
PYTHONPATH=src python3 scripts/test-native-perception.py
restore
trap - EXIT
# The privileged channel must not exist in a production binary.
if grep -q "yendorlab-test" .local/BrogueCE/bin/brogue; then
  echo "production binary contains test-only code" >&2
  exit 1
fi
echo "PRODUCTION_ISOLATION_PASS"
