#!/usr/bin/env bash
# Part I 4: the web layer is a renderer and nothing else. Hard budget, 800 lines of JS.
# If logic starts migrating into JS, it belongs in Rust.
set -euo pipefail

BUDGET=800
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB="$ROOT/web"

if [ ! -d "$WEB" ]; then
    echo "no web/ directory yet; budget check vacuously passes"
    exit 0
fi

COUNT=$(find "$WEB" -name '*.js' -not -path '*/node_modules/*' -exec cat {} + 2>/dev/null | wc -l)

echo "JS lines: $COUNT / $BUDGET"
if [ "$COUNT" -gt "$BUDGET" ]; then
    echo "over budget by $((COUNT - BUDGET)) lines"
    exit 1
fi
