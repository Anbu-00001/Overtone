#!/usr/bin/env bash
# The README's diagrams have to parse, and the README has to stay free of emoji.
#
# Both are failures you cannot see locally. A mermaid block with a syntax error renders on
# GitHub as a grey error box, and the only way to find out is to push -- which is the same
# failure mode `build_space.sh` exists to prevent for the bundle. Mermaid keywords are the
# usual cause: `graph`, `end`, `class`, `style`, `subgraph` and `default` are reserved, so
# `graph["overtone-graph"]` is a parse error while the label alone is fine. That exact
# mistake is what prompted this script.
#
# The emoji rule is CLAUDE.md's, and it was unenforced until a diagram broke it.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

fail=0

# ---------------------------------------------------------------- emoji
# deploy/README.md is exempt on one line only: `emoji:` is a Hugging Face Space config
# field that sets the Space's icon, not prose.
while read -r f; do
  if python3 - "$f" <<'PY'
import sys, re
p = sys.argv[1]
bad = []
for n, line in enumerate(open(p, encoding="utf-8"), 1):
    if p.endswith("deploy/README.md") and line.startswith("emoji:"):
        continue
    for ch in line:
        o = ord(ch)
        if (0x1F000 <= o <= 0x1FAFF) or (0x2600 <= o <= 0x27BF) or o == 0xFE0F:
            bad.append((n, ch))
for n, ch in bad:
    print(f"  {p}:{n}: emoji {ch!r} -- CLAUDE.md 3: no emoji in the README")
sys.exit(1 if bad else 0)
PY
  then :; else fail=1; fi
done < <(git ls-files '*.md' | grep -vE '^(graphify-out|docs/spec)/')

# ---------------------------------------------------------------- mermaid
MMDC="${MMDC:-}"
if [ -z "$MMDC" ]; then
  if command -v mmdc >/dev/null 2>&1; then MMDC="mmdc"
  elif [ -x node_modules/.bin/mmdc ]; then MMDC="node_modules/.bin/mmdc"
  elif command -v npx >/dev/null 2>&1; then MMDC="npx --yes @mermaid-js/mermaid-cli"
  fi
fi

if [ -z "$MMDC" ]; then
  echo "  mermaid-cli unavailable and no npx to fetch it; diagrams NOT checked" >&2
  echo "  install with: npm i -g @mermaid-js/mermaid-cli" >&2
  exit 1
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Which Chrome to drive is a capability question, not a guess. `npm i @mermaid-js/mermaid-cli`
# normally downloads its own matched Chromium, and pointing puppeteer at a system Chrome of a
# different version fails in a way that looks exactly like a broken diagram. But an install
# made with PUPPETEER_SKIP_DOWNLOAD has no browser at all and needs the system one. So probe
# with a trivial diagram and keep whichever config actually renders.
ARGS='"--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"'
printf '{ "args": [%s] }\n' "$ARGS" > "$WORK/pup-own.json"
printf '{ "executablePath": "%s", "args": [%s] }\n' "${CHROME_PATH:-/usr/bin/google-chrome}" "$ARGS" > "$WORK/pup-sys.json"
printf 'flowchart TD\n  A[probe] --> B[ok]\n' > "$WORK/probe.mmd"

PUP=""
for cfg in "$WORK/pup-own.json" "$WORK/pup-sys.json"; do
  if $MMDC -p "$cfg" -i "$WORK/probe.mmd" -o "$WORK/probe.svg" >/dev/null 2>&1; then
    PUP="$cfg"; break
  fi
done
if [ -z "$PUP" ]; then
  echo "  no usable Chrome for mermaid-cli (tried puppeteer's own, then ${CHROME_PATH:-/usr/bin/google-chrome})" >&2
  exit 1
fi

count=0
while read -r doc; do
  n=$(python3 - "$doc" "$WORK" <<'PY'
import re, sys, os, pathlib
doc, work = sys.argv[1], sys.argv[2]
blocks = re.findall(r'```mermaid\n(.*?)```', open(doc, encoding="utf-8").read(), re.S)
stem = doc.replace('/', '_')
for i, b in enumerate(blocks):
    pathlib.Path(f"{work}/{stem}.{i}.mmd").write_text(b, encoding="utf-8")
print(len(blocks))
PY
)
  [ "$n" = "0" ] && continue
  for f in "$WORK"/$(echo "$doc" | tr '/' '_').*.mmd; do
    if $MMDC -p "$PUP" -i "$f" -o "${f%.mmd}.svg" >/dev/null 2>"$WORK/err"; then
      count=$((count + 1))
    else
      echo "  $doc: block $(basename "$f" .mmd | sed 's/.*\.//') failed to parse"
      grep -m1 -A4 'Error' "$WORK/err" | sed 's/^/    /'
      fail=1
    fi
  done
done < <(git ls-files '*.md' | grep -vE '^graphify-out/')

[ "$fail" -eq 0 ] && echo "  $count mermaid diagrams parse, no emoji"
exit "$fail"
