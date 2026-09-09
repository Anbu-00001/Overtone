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

# One bundle, one Space card. This used to build the payload itself and copy
# `web/README.md`, while `build_space.sh` copied `deploy/README.md` -- two Space cards that
# drifted apart, and whichever script you happened to run decided which one Hugging Face saw.
# The bundler is now the only path, so the front matter has exactly one source.
"$ROOT/scripts/check_js_budget.sh"

DIST="${TMPDIR:-/tmp}/overtone-dist.$$"
"$ROOT/scripts/build_space.sh" "$DIST"

WORK="${TMPDIR:-/tmp}/overtone-space.$$"
git clone "https://huggingface.co/spaces/$SPACE" "$WORK"
# Same reason as space.yml: without the LFS filters installed, the wasm is committed as
# a plain blob and Hugging Face refuses the push for containing binary files.
git -C "$WORK" lfs install --local
rm -rf "$WORK"/{index.html,style.css,js,pkg,README.md,LICENSE,NOTICE}
cp -r "$DIST"/. "$WORK/"
rm -rf "$DIST"
cd "$WORK"
git add -A
git commit -m "Deploy Overtone $(git -C "$ROOT" rev-parse --short HEAD)" || { echo "nothing to deploy"; exit 0; }
git push
echo "deployed to https://huggingface.co/spaces/$SPACE"
