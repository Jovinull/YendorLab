#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .local
if [ ! -d .local/BrogueCE/.git ]; then
  git -c core.autocrlf=true clone --depth 1 --branch v1.15.1 https://github.com/tmewett/BrogueCE.git .local/BrogueCE
fi
git -C .local/BrogueCE config core.autocrlf true
test "$(git -C .local/BrogueCE rev-parse HEAD)" = "1ba4240b7a928ddf0ffb772717bf1d433cd63804"
if ! git -C .local/BrogueCE diff --quiet; then
  git -C .local/BrogueCE apply --reverse --check ../../native/patches/0001-yendorlab-stream.patch
else
  git -C .local/BrogueCE apply ../../native/patches/0001-yendorlab-stream.patch
fi
make -C .local/BrogueCE -j2 GRAPHICS=NO TERMINAL=NO RELEASE=YES bin/brogue
