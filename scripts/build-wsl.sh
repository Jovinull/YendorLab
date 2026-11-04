#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .local
patch=native/patches/0001-yendorlab-stream.patch
if [ ! -d .local/BrogueCE/.git ]; then
  git -c core.autocrlf=true clone --depth 1 --branch v1.15.1 https://github.com/tmewett/BrogueCE.git .local/BrogueCE
fi
git -C .local/BrogueCE config core.autocrlf true
test "$(git -C .local/BrogueCE rev-parse HEAD)" = "1ba4240b7a928ddf0ffb772717bf1d433cd63804"
wanted="$(sha256sum "$patch" | cut -d' ' -f1)"
if [ "$(cat .local/applied-patch.sha256 2>/dev/null || true)" != "$wanted" ]; then
  # The tree must contain exactly the tracked patch; never build a stale one.
  if ! git -C .local/BrogueCE diff --quiet; then
    if ! git -C .local/BrogueCE apply --reverse --check "../../$patch" 2>/dev/null; then
      git -C .local/BrogueCE checkout -- src
    else
      git -C .local/BrogueCE apply --reverse "../../$patch"
    fi
  fi
  git -C .local/BrogueCE apply "../../$patch"
  echo "$wanted" > .local/applied-patch.sha256
fi
make -C .local/BrogueCE -j2 GRAPHICS=NO TERMINAL=NO RELEASE=YES bin/brogue
