#!/usr/bin/env bash
# Part I 4: the web layer is a renderer and nothing else. If logic starts migrating into JS,
# it belongs in Rust.
#
# The number was 800 in Part I 8, written when the demo was one page of panels. It is 1200
# here, raised once and deliberately, because the page now carries three sections rather than
# one -- Lab (Part I), Closure (Part III) and Menagerie (Part IV) -- and 800 lines across
# three would be met by cutting panels rather than by keeping logic in Rust, which is the
# opposite of what the budget is for.
#
# The rule the number enforces is unchanged and is the thing to check in review: no panel
# computes a physical quantity. Every number a panel draws arrives from the WASM boundary
# already fitted, already normalised, already ordered. If that stops being true, the fix is
# to move code into Rust, never to raise this again.
set -euo pipefail

BUDGET=1200
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
