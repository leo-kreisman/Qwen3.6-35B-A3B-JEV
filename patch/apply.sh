#!/usr/bin/env bash
# Clone SemIf at the pinned upstream commit, apply the SSD-scorer patch, and run
# the tests. Idempotent: refuses to clobber an existing checkout.
#
# Usage:
#   ./apply.sh [target-dir]        # default: <repo-root>/semif
#
# SemIf is MIT (Copyright (c) 2026 TheoLeeCJ) and is NOT redistributed in this
# repository -- this script fetches it from upstream. See THIRD_PARTY_NOTICES.md.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-$(cd "$HERE/.." && pwd)/semif}"

UPSTREAM="https://github.com/TheoLeeCJ/SemIf.git"
COMMIT="1f2dea3e25379f9dfc98cb83c324f00ab5deda37"
PATCH="$HERE/semif-ssd-scorer.patch"

[[ -f "$PATCH" ]] || { echo "missing $PATCH" >&2; exit 1; }

if [[ -e "$TARGET" ]]; then
  echo "refusing to touch existing $TARGET -- remove it first, or pass another path" >&2
  exit 1
fi

echo "==> cloning upstream into $TARGET"
git clone --quiet "$UPSTREAM" "$TARGET"
git -C "$TARGET" checkout --quiet "$COMMIT"
echo "==> at $(git -C "$TARGET" rev-parse --short HEAD) (pinned)"

echo "==> applying patch"
if ! git -C "$TARGET" apply --check "$PATCH" 2>/dev/null; then
  echo "patch does not apply cleanly to $COMMIT" >&2
  echo "the patch is pinned to that commit; check none of the three files moved" >&2
  exit 1
fi
git -C "$TARGET" apply "$PATCH"

echo "==> changed files"
git -C "$TARGET" status --porcelain

# The patch is against the pinned commit, so this must hold. It is checked rather
# than assumed because a silently half-applied patch would still import.
echo "==> confirming the patched backend imports"
if [[ -x "$TARGET/.venv/bin/python" ]]; then
  PY="$TARGET/.venv/bin/python"
else
  echo "    no .venv in $TARGET; create one and install first:"
  echo "      uv venv --python 3.11 $TARGET/.venv"
  echo "      uv pip install --python $TARGET/.venv/bin/python -e $TARGET"
  echo "    (see SETUP.md step 2)"
  echo
  echo "==> patch applied. Tests not run -- no interpreter yet."
  exit 0
fi

PYTHONPATH="$TARGET/src" "$PY" -c 'from semif_phase1 import llamacpp_backend as b;
print("   _default_threads() =", b._default_threads());
print("   _ubatch(8) =", b._ubatch(8))'

echo "==> running the test suite"
cd "$TARGET"
PYTHONPATH="$TARGET/src" "$PY" -m pytest -q || {
  echo "tests failed -- the patch is applied but the tree is not healthy" >&2
  exit 1
}
