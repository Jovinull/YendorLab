#!/usr/bin/env bash
# Copy the built binary to the Linux filesystem under a content-addressed path.
# Git, sources and experiment records stay in the Windows checkout; only this
# immutable, hash-named copy lives elsewhere, so it cannot silently diverge.
set -euo pipefail
cd "$(dirname "$0")/.."
source_binary=.local/BrogueCE/bin/brogue
hash="$(sha256sum "$source_binary" | cut -d' ' -f1)"
target="${YENDORLAB_STAGE_DIR:-$HOME/.local/share/yendorlab}/$hash"
mkdir -p "$target"
if [ ! -f "$target/brogue" ]; then
  cp "$source_binary" "$target/brogue.tmp"
  chmod 755 "$target/brogue.tmp"
  test "$(sha256sum "$target/brogue.tmp" | cut -d' ' -f1)" = "$hash"
  mv "$target/brogue.tmp" "$target/brogue"
fi
test "$(sha256sum "$target/brogue" | cut -d' ' -f1)" = "$hash"
echo "$target/brogue"
