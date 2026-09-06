#!/usr/bin/env bash
# Publish the demo to a Hugging Face Static Space.
#
# Static Spaces are free for everyone; Gradio and Docker Spaces run on compute and require a
# paid plan. Verified against huggingface.co/docs/hub/spaces-overview on 2026-09-06, quoting
# their wording directly:
#
#   "Static Spaces are free for everyone. Gradio and Docker Spaces run on compute and
#    require a paid plan to create: PRO for personal accounts, Team or Enterprise for
#    organizations."
#
# Hugging Face has changed this before, so RE-READ THAT PAGE BEFORE EACH RELEASE rather than
# trusting this comment. That check is the first item of the release checklist.
set -euo pipefail

SPACE=${1:-}
if [ -z "$SPACE" ]; then
    echo "usage: $0 <hf-username>/<space-name>" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "building wasm..."
wasm-pack build crates/overtone-wasm --target web --out-dir pkg --release
mkdir -p web/pkg
cp crates/overtone-wasm/pkg/overtone_wasm.js crates/overtone-wasm/pkg/overtone_wasm_bg.wasm web/pkg/

"$ROOT/scripts/check_js_budget.sh"

WORK="${TMPDIR:-/tmp}/overtone-space.$$"
git clone "https://huggingface.co/spaces/$SPACE" "$WORK"
rm -rf "$WORK"/{index.html,style.css,js,pkg,README.md}
cp -r web/index.html web/style.css web/README.md web/js web/pkg "$WORK/"
cd "$WORK"
git add -A
git commit -m "Deploy Overtone $(git -C "$ROOT" rev-parse --short HEAD)" || { echo "nothing to deploy"; exit 0; }
git push
echo "deployed to https://huggingface.co/spaces/$SPACE"
