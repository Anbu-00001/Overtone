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

# web/pkg/ holds wasm-bindgen's generated glue, which is not hand-written and is not what
# the budget is about. The budget exists so that logic does not migrate out of Rust into
# JS; counting machine-generated bindings against it would measure the wrong thing.
COUNT=$(find "$WEB" -name '*.js' \
    -not -path '*/node_modules/*' \
    -not -path '*/pkg/*' \
    -exec cat {} + 2>/dev/null | wc -l)

echo "JS lines: $COUNT / $BUDGET"
if [ "$COUNT" -gt "$BUDGET" ]; then
    echo "over budget by $((COUNT - BUDGET)) lines"
    exit 1
fi
