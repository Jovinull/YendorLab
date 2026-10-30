"""Compare official seed content across checkout line endings and version headers."""

import difflib
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
binary = root / '.local/BrogueCE/bin/brogue'
fixture = root / '.local/BrogueCE/test/seed_catalogs/seed_catalog_brogue.txt'
run = subprocess.run([str(binary), '--print-seed-catalog', '1', '25', '40'],
                     cwd=binary.parent, capture_output=True, text=True,
                     timeout=90, check=True)
expected = fixture.read_text().splitlines()
actual = run.stdout.splitlines()
assert expected[0] == actual[0] == 'Brogue seed catalog, seeds 1 to 25, through depth 40.'
assert 'Dungeons unchanged since CE 1.11.' in expected[1]
assert 'Dungeons unchanged since CE 1.11.' in actual[1]
if expected[4:] != actual[4:]:
    print(''.join(list(difflib.unified_diff(expected[4:], actual[4:], lineterm='\n'))[:80]))
    raise SystemExit(1)
print(f'PASS: {len(actual) - 4} catalog body lines equal; version header and CRLF normalized')
