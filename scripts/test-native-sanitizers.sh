#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
restore() {
  make -C .local/BrogueCE -j2 GRAPHICS=NO TERMINAL=NO RELEASE=YES bin/brogue > .local/native-production-build.log 2>&1
}
trap restore EXIT
make -C .local/BrogueCE -j2 GRAPHICS=NO TERMINAL=NO RELEASE=YES \
  CFLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
  LDLIBS='-fsanitize=address,undefined' bin/brogue > .local/native-sanitizer-build.log 2>&1
PYTHONPATH=src python3 scripts/probe-native-sanitizer.py
