#!/usr/bin/env bash
# Re-clone the twelve MoE-offload / SSD-streaming engines surveyed in
# docs/OFFLOAD_PROJECTS_ANALYSIS.md, each at the commit that was actually read.
#
# These trees are deliberately NOT in this repository: they are twelve other
# people's projects, they total ~474 MB, and vendoring them would bury this
# repository's own work under someone else's history. This script exists so the
# survey can still be re-checked claim by claim.
#
# Shallow clones at a pinned commit. Read-only: nothing here is built or run.
set -euo pipefail

DEST="${1:-$(cd "$(dirname "$0")/.." && pwd)/offload_projects}"
mkdir -p "$DEST"
cd "$DEST"

# name|url|commit
REPOS="
apus-qwen3.6-35B-A3B|https://github.com/bricesommers/apus-qwen3.6-35B-A3B.git|c4b303b
Edge0|https://github.com/Edge0-AI/Edge0.git|fb4cd2c
flash-moe|https://github.com/danveloper/flash-moe.git|3601d41
Mference|https://github.com/NeelM0906/Mference.git|58a0ec9
q36|https://github.com/Ninnix/q36.git|9fb537a
qwen-fieldfare|https://github.com/tanishqpatil/qwen-fieldfare.git|5287dc4
Qwen-MoE-Router-exp|https://github.com/anwarth/Qwen-MoE-Router-exp.git|dd4bf42
qwisp|https://github.com/penta2himajin/qwisp.git|b7dffc1
samosa-chat|https://github.com/deanwadhwa/samosa-chat.git|b1aa3a7
siphon.cpp|https://github.com/Manohar20006/siphon.cpp.git|5d6c3d6
slipstream-dwijenpatel|https://github.com/dwijenpatel/slipstream.git|3a89246
slipstream-schero94|https://github.com/Schero94/slipstream.git|1518931
"

echo "==> target $DEST"
while IFS='|' read -r name url commit; do
  [[ -z "$name" ]] && continue
  if [[ -d "$name/.git" ]]; then
    echo "==> $name already present, skipping"
    continue
  fi
  echo "==> $name @ $commit"
  git clone --quiet --depth 1 --revision "$commit" "$url" "$name" 2>/dev/null \
    || git clone --quiet --filter=blob:none "$url" "$name" \
       && git -C "$name" checkout --quiet "$commit"
done <<< "$REPOS"

echo
echo "==> done. trees in $DEST are survey material only -- read-only references,"
echo "    not build inputs. See docs/OFFLOAD_PROJECTS_ANALYSIS.md for what each"
echo "    one does and which of its optimisations were measured dead ends here."
