#!/usr/bin/env bash
# The gate. Every claim in the README has a line here that fails when it stops being true.
#
# Run from the repository root. Phases 1-7.
set -uo pipefail
cd "$(dirname "$0")/.."

passed=0
failed=0
run() {
  local name="$1"; shift
  if "$@" >/tmp/overtone-gate.log 2>&1; then
    printf '  %-42s PASS\n' "$name"; passed=$((passed + 1))
  else
    printf '  %-42s FAIL\n' "$name"; failed=$((failed + 1))
    sed 's/^/      /' /tmp/overtone-gate.log | tail -20
  fi
}

# A claim about a number in a command's output.
expect() {
  local name="$1" pattern="$2"; shift 2
  local out
  out="$("$@" 2>&1)"
  if grep -qE "$pattern" <<<"$out"; then
    printf '  %-42s PASS\n' "$name"; passed=$((passed + 1))
  else
    printf '  %-42s FAIL\n' "$name"; failed=$((failed + 1))
    echo "      expected /$pattern/ in:"; sed 's/^/      /' <<<"$out" | tail -20
  fi
}

echo "PHASE 1-7 GATE"

run "cargo fmt --check"                cargo fmt --all -- --check
run "clippy -D warnings"               cargo clippy --workspace --all-targets -- -D warnings
run "clippy (parallel)"                cargo clippy -p overtone-sim --features parallel --all-targets -- -D warnings
run "cargo test --workspace"           cargo test --workspace
run "cargo test (parallel)"            cargo test -p overtone-sim --features parallel
for c in sim rl spec lie gsim walk wfc; do
  run "overtone-$c -> wasm32"          cargo build -p "overtone-$c" --target wasm32-unknown-unknown
done
run "wasm-pack build"                  wasm-pack build crates/overtone-wasm --target web --out-dir pkg --release
run "JS budget"                        ./scripts/check_js_budget.sh
run "native vs wasm agreement"         ./scripts/wasm_determinism.sh

# Phase 6 acceptance, in release: these are measurements, not unit tests.
expect "Theorem 1 within 10% at depth 64" \
  '^ 7 +91 +1 +0\.065934 +64 +0\.0(6[5-9]|7[0-2])' \
  cargo run --release -q -p overtone-gsim --example theorem_one

expect "100-qubit policy trains" \
  'return after +\+0\.2[0-9]' \
  cargo run --release -q -p overtone-gsim --example hundred_qubit_policy

expect "dequantization verdict, n=4 L=3" \
  'effective chi = 3' \
  cargo run --release -q -p overtone-cli -- dequantize --qubits 4 --layers 3 --k 3 --tolerance 1e-6 --max-chi 8

expect "predict refuses a false claim" \
  'No trainability claim follows' \
  cargo run --release -q -p overtone-cli -- predict --qubits 4 --layers 2

expect "predict on the TFIM chain" \
  'dim\(g\) +45' \
  cargo run --release -q -p overtone-cli -- predict --family tfim --qubits 5

# Phase 7. The transport exponent is the readout that makes a world an experiment rather
# than a skin, so the gate asserts the numbers the README prints, not merely that it ran.
expect "clean lattice is ballistic" \
  'periodic +1\.000 +1\.000' \
  cargo run --release -q -p overtone-walk --example transport

expect "Rudin-Shapiro saturates, Fibonacci does not" \
  'Rudin-Shapiro +-?0\.0[0-9]+ +0\.1' \
  cargo run --release -q -p overtone-walk --example transport

expect "conditioning pays only on periodic structure" \
  'two-periodic +0\.0197 +0\.0000 +0\.13' \
  cargo run --release -q -p overtone-walk --example transport

# The page must boot and populate its readouts from the engine, Closure chapter included.
if command -v google-chrome >/dev/null 2>&1; then
  mkdir -p web/pkg
  cp crates/overtone-wasm/pkg/overtone_wasm.js \
     crates/overtone-wasm/pkg/overtone_wasm_bg.wasm web/pkg/ 2>/dev/null
  python3 -m http.server 8791 --directory web >/dev/null 2>&1 &
  server=$!
  sleep 2
  dom=$(google-chrome --headless --disable-gpu --no-sandbox --virtual-time-budget=20000 \
        --dump-dom http://localhost:8791/index.html 2>/dev/null)
  kill "$server" >/dev/null 2>&1
  for check in 'id="loading"!' '<b id="r-ceiling">[0-9]+</b>' '<b id="hero-lambda">[0-9]' \
               '<b id="cl-dim">28</b>' '<b id="cl-var">0\.1071</b>' \
               '<b id="mn-beta">0\.827</b>' '<b id="mn-regime">superdiffusive</b>'; do
    if [ "${check: -1}" = "!" ]; then
      if grep -q "${check%!}" <<<"$dom"; then
        printf '  %-42s FAIL\n' "page boots (no loading state)"; failed=$((failed + 1))
      else
        printf '  %-42s PASS\n' "page boots (no loading state)"; passed=$((passed + 1))
      fi
    elif grep -qE "$check" <<<"$dom"; then
      printf '  %-42s PASS\n' "browser: $check"; passed=$((passed + 1))
    else
      printf '  %-42s FAIL\n' "browser: $check"; failed=$((failed + 1))
    fi
  done
else
  printf '  %-42s SKIP (chrome not installed)\n' "browser smoke test"
fi

if command -v julia >/dev/null 2>&1; then
  cargo run -q -p overtone-sim --example emit_oracle_cases > lab/test/cases.txt 2>/dev/null
  run "Yao.jl oracle"                  julia --project=lab lab/test/oracle.jl
else
  printf '  %-42s SKIP (julia not installed)\n' "Yao.jl oracle"
fi

echo
echo "  passed=$passed failed=$failed"
[ "$failed" -eq 0 ]
