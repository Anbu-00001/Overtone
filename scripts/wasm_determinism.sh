#!/usr/bin/env bash
# Native versus WASM agreement (Part I 5).
#
# Part I asks for "same seed, same trajectory, native and WASM", asserted by a test. Built
# and measured, that claim needs splitting, because one half of it is attainable and the
# other is not:
#
#   Within a target, seeded runs are bit-exact. That is asserted in
#   crates/overtone-sim/tests/determinism.rs, and it is the guarantee the demo actually
#   needs: two people opening the same permalink run the same .wasm binary and get
#   identical numbers.
#
#   Across targets, they are not. IEEE-754 requires correct rounding for +, -, *, / and
#   sqrt, but NOT for transcendental functions. Native links glibc's libm; wasm32 links
#   Rust's own. Measured over 60 arguments, the two disagree by one unit in the last place
#   on roughly 5% of sin and cos evaluations -- with sin_cos and with separate calls alike.
#   A circuit applies thousands of those, so a trajectory drifts in the last couple of
#   digits and no amount of care in our own code prevents it. Shipping our own
#   correctly-rounded transcendentals would fix it and would cost more than it buys.
#
# So this script asserts agreement to a tight relative tolerance and prints the worst
# observed difference, rather than asserting an equality that is not true.
set -euo pipefail

TOL=${TOL:-1e-13}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
WORK="${TMPDIR:-/tmp}/overtone-determinism.$$"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

cargo run -q --release -p overtone-wasm --example trace > "$WORK/native.txt"
wasm-pack build crates/overtone-wasm --target nodejs --out-dir "$WORK/pkg" >/dev/null 2>&1

cat > "$WORK/trace.mjs" <<'JS'
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { Lab } = require(process.argv[2]);
const f = (v) => v.toExponential(17).replace(/e([+-])(\d+)$/, (_, s, d) => 'e' + (s === '-' ? '-' : '') + Number(d));
const lab = new Lab(2, 3, 3, 0, false, true, 1.0, 0.05, 7);
const o = [];
o.push('A ' + f(lab.exact_return()), 'A ' + f(lab.gradient_agreement()));
for (const v of lab.spectrum(64)) o.push('A ' + f(v));
for (const v of lab.gradient_scatter()) o.push('B ' + f(v));
for (let i = 0; i < 20; i++) lab.train_steps(1);
for (const v of lab.return_trace()) o.push('C ' + f(v));
console.log(o.join('\n'));
JS

node "$WORK/trace.mjs" "$WORK/pkg/overtone_wasm.js" > "$WORK/wasm.txt"

paste -d' ' "$WORK/native.txt" "$WORK/wasm.txt" | awk -v tol="$TOL" '
{
    a = $2 + 0; b = $4 + 0;
    scale = (a < 0 ? -a : a); if (scale < 1) scale = 1;
    d = (a - b); if (d < 0) d = -d;
    rel = d / scale;
    if (rel > worst) { worst = rel; wl = NR }
    if ($2 != $4) differing++
    total++
}
END {
    printf "compared %d values across native and wasm32\n", total
    printf "  differing in the last digits : %d\n", differing + 0
    printf "  worst relative difference    : %.3e  (line %d)\n", worst, wl
    printf "  tolerance                    : %s\n", tol
    if (worst > tol + 0) { print "FAILED: divergence exceeds tolerance"; exit 1 }
    print "PASS: native and wasm agree within tolerance"
}'
