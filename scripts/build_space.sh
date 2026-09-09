#!/usr/bin/env bash
# Assemble the static bundle that gets pushed to a Hugging Face Space.
#
# Why static, and why there is no backend
# ---------------------------------------
# Hugging Face's own documentation settles it: "Static Spaces are free for everyone. Gradio and
# Docker Spaces run on compute and require a paid plan to create." So on the free tier the only
# SDK available is `static` -- and that suits this project rather than constraining it.
#
# Overtone's engine already compiles to WebAssembly and runs in the browser. Part I 5 requires
# every run to be reproducible from a seed natively and in WASM, and `scripts/check_js_budget.sh`
# enforces that no panel computes a physical quantity. Between them, the frontend *is* the
# engine. There is no backend to deploy, and adding one would move logic out of Rust, which is
# the thing the budget exists to prevent.
#
# What the league needs instead: GitHub Actions runs the matches and pushes results to a
# Hugging Face Dataset. That is Decisions-01 Appendix A's design and it needs no server either.
#
# Two facts from the docs that shape this script:
#   - Persistent storage is no longer available on Spaces. Nothing here may assume writable disk.
#   - A Space rebuilds on every push to its repo, and serves files from the repo root, so the
#     bundle has to be self-contained and `web/pkg/` -- gitignored in this repository, because
#     it is build output -- has to be *in* the bundle.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DIST="${1:-dist}"
rm -rf "$DIST"
mkdir -p "$DIST"

echo "building wasm"
wasm-pack build crates/overtone-wasm --target web --out-dir pkg --release >/dev/null 2>&1
rm -rf web/pkg
cp -r crates/overtone-wasm/pkg web/pkg

echo "assembling $DIST"
cp web/index.html "$DIST/"
cp web/style.css "$DIST/"
cp -r web/js "$DIST/"
cp -r web/pkg "$DIST/"
# The Space's front matter and landing copy. It has to be README.md at the bundle root.
cp deploy/README.md "$DIST/README.md"
cp LICENSE "$DIST/LICENSE"
cp NOTICE "$DIST/NOTICE"

# wasm-pack emits .gitignore inside pkg/, which would stop the Space repo from tracking the
# very files it has to serve. Removing it is not optional.
rm -f "$DIST/pkg/.gitignore"

echo "checking the bundle is self-contained"
fail=0
# Every local href/src in the HTML must resolve inside the bundle.
while read -r ref; do
  case "$ref" in
    http*|//*|data:*|"#"*|"") continue ;;
  esac
  if [ ! -e "$DIST/$ref" ]; then
    echo "  MISSING: index.html references $ref"
    fail=1
  fi
done < <(grep -oE '(href|src)="[^"]+"' "$DIST/index.html" | sed -E 's/.*="([^"]+)"/\1/')

# The module graph: every relative import in the JS must resolve, resolved against the
# importing file's own directory rather than against a guess.
for f in "$DIST"/js/*.js; do
  while read -r spec; do
    case "$spec" in http*|//*) continue ;; esac
    target="$(dirname "$f")/$spec"
    if [ ! -e "$target" ]; then
      echo "  MISSING: $(basename "$f") imports $spec"
      fail=1
    fi
  done < <(grep -oE "from '[^']+'" "$f" | sed -E "s/from '([^']+)'/\1/")
done

[ -f "$DIST/pkg/overtone_wasm_bg.wasm" ] || { echo "  MISSING: the wasm itself"; fail=1; }
[ -f "$DIST/README.md" ] || { echo "  MISSING: the Space front matter"; fail=1; }
grep -q '^sdk: static$' "$DIST/README.md" || { echo "  README.md is not a static Space"; fail=1; }

if [ "$fail" -ne 0 ]; then
  echo "bundle is not self-contained"
  exit 1
fi

bytes=$(du -sb "$DIST" | cut -f1)
wasm=$(stat -c%s "$DIST/pkg/overtone_wasm_bg.wasm")
echo "bundle ok: $(( bytes / 1024 )) KiB total, $(( wasm / 1024 )) KiB wasm"
