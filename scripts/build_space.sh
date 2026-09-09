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

# Track the wasm with Git LFS, and ship the rule so both deploy paths agree.
#
# Measured, not reasoned: pushing the bundle with the wasm as a plain blob is rejected by
# Hugging Face outright --
#
#   remote: Your push was rejected because it contains binary files.
#   remote: Please use https://huggingface.co/docs/hub/xet to store binary files.
#   remote: Offending files:
#   remote:   - pkg/overtone_wasm_bg.wasm
#
# The rule is about binary *content*, not size. A 358 KiB wasm is nowhere near the 10 MiB
# threshold above which LFS is required for large files, and it is still refused -- so the two
# rules are separate and the size one does not grant an exemption from the other. A new Space's
# default .gitattributes already carries `*.wasm filter=lfs`, which is exactly right; the job of
# this file is to make `deploy_space.sh` (which pushes into a clone) and `space.yml` (which does
# a fresh git init and would otherwise have no rules at all) produce identical commits.
cat > "$DIST/.gitattributes" <<'ATTRS'
# Hugging Face refuses a push containing binary files that are not in LFS/Xet, whatever their
# size, so the wasm must be tracked. This mirrors the default .gitattributes a new Space is
# created with, and exists so that a fresh `git init` in the bundle behaves the same way.
*.wasm filter=lfs diff=lfs merge=lfs -text
ATTRS

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

# Every binary in the bundle must be matched by an LFS rule, or the push is refused. Checked
# by extension against the shipped .gitattributes rather than assumed, because the failure
# happens on the remote and costs a full CI run to discover.
while read -r bin; do
  ext="${bin##*.}"
  grep -q "^\*\.${ext} filter=lfs" "$DIST/.gitattributes" || {
    echo "  $bin is binary and no LFS rule in .gitattributes covers *.$ext"
    echo "  Hugging Face rejects a push containing untracked binary files, whatever the size"
    fail=1
  }
done < <(cd "$DIST" && find . -type f ! -path './.git/*' -printf '%P\n' \
         | while read -r f; do case "$(file -b --mime-encoding "$f")" in binary) echo "$f";; esac; done)
[ -f "$DIST/README.md" ] || { echo "  MISSING: the Space front matter"; fail=1; }

# The front matter is the Space's configuration, and Hugging Face validates it after the push
# rather than before. An out-of-set colour or a misspelled sdk is a Space that builds into a
# configuration error, so the allowed values are checked here against the documented set at
# huggingface.co/docs/hub/spaces-config-reference (read 2026-09-09).
python3 - "$DIST" <<'VALIDATE' || fail=1
import re, sys, os
dist = sys.argv[1]
text = open(os.path.join(dist, "README.md"), encoding="utf-8").read()
m = re.match(r"---\n(.*?)\n---\n", text, re.S)
if not m:
    print("  README.md has no YAML front matter, so it is not a Space card"); sys.exit(1)
fm, bad = {}, []
for line in m.group(1).splitlines():
    if line.startswith(("  - ", "- ")) or not line.strip():
        continue
    k, _, v = line.partition(":")
    fm[k.strip()] = v.strip()

COLOURS = {"red", "yellow", "green", "blue", "indigo", "purple", "pink", "gray"}
for key in ("title", "emoji", "colorFrom", "colorTo", "sdk", "app_file", "license",
            "short_description"):
    if key not in fm:
        bad.append(f"front matter is missing `{key}`")
# Only `static` is free: Gradio and Docker Spaces run on compute and require a paid plan.
if fm.get("sdk") != "static":
    bad.append(f"sdk is {fm.get('sdk')!r}, and only 'static' is free")
for key in ("colorFrom", "colorTo"):
    if key in fm and fm[key] not in COLOURS:
        bad.append(f"{key} is {fm[key]!r}, not one of {sorted(COLOURS)}")
if fm.get("header", "default") not in ("mini", "default"):
    bad.append(f"header is {fm['header']!r}, not 'mini' or 'default'")
if fm.get("pinned", "false") not in ("true", "false"):
    bad.append(f"pinned is {fm['pinned']!r}, not a boolean")
# `agpl-3.0` is the Hugging Face identifier; `agpl-3.0-or-later` is the SPDX expression and
# is not in their list, so it belongs in prose and never in the metadata.
if fm.get("license") != "agpl-3.0":
    bad.append(f"license is {fm.get('license')!r}; the HF identifier is 'agpl-3.0'")
app = fm.get("app_file", "")
if app and not os.path.exists(os.path.join(dist, app)):
    bad.append(f"app_file points at {app!r}, which is not in the bundle")
if len(fm.get("short_description", "")) > 60:
    bad.append(f"short_description is {len(fm['short_description'])} chars; the Space form "
               "truncates at 60, so a longer one disagrees with what the card shows")
# The card renderer does KaTeX, not mermaid -- a diagram here would ship as a raw code block.
if "```mermaid" in text:
    bad.append("the Space card renders KaTeX but not mermaid; the diagram would show as code")
for b in bad:
    print("  " + b)
sys.exit(1 if bad else 0)
VALIDATE

if [ "$fail" -ne 0 ]; then
  echo "bundle is not self-contained"
  exit 1
fi

bytes=$(du -sb "$DIST" | cut -f1)
wasm=$(stat -c%s "$DIST/pkg/overtone_wasm_bg.wasm")
echo "bundle ok: $(( bytes / 1024 )) KiB total, $(( wasm / 1024 )) KiB wasm"
