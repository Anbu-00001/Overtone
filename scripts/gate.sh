#!/usr/bin/env bash
# The gate. Every claim in the README has a line here that fails when it stops being true.
#
# Run from the repository root. Phases 1-17.
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
  # -e is not optional: a pattern that begins with "-" would otherwise be parsed as options,
  # and the check would fail with "no search PATTERN specified" while looking like a real
  # measurement failure. That happened once and cost a full gate run to diagnose.
  if grep -qE -e "$pattern" <<<"$out"; then
    printf '  %-42s PASS\n' "$name"; passed=$((passed + 1))
  else
    printf '  %-42s FAIL\n' "$name"; failed=$((failed + 1))
    echo "      expected /$pattern/ in:"; sed 's/^/      /' <<<"$out" | tail -20
  fi
}

echo "PHASE 1-17 GATE"

run "cargo fmt --check"                cargo fmt --all -- --check
run "clippy -D warnings"               cargo clippy --workspace --all-targets -- -D warnings
run "clippy (parallel)"                cargo clippy -p overtone-sim --features parallel --all-targets -- -D warnings
run "cargo test --workspace"           cargo test --workspace
run "cargo test (parallel)"            cargo test -p overtone-sim --features parallel
for c in sim rl spec lie gsim walk wfc qd graph opt orbit cgt otn; do
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

expect "policies superpose linearly" \
  '0\.5      0\.5' \
  cargo run --release -q -p overtone-graph --example eigenbasis

expect "the LMDP error falls as 1/rho in log space" \
  '4000        0\.006' \
  cargo run --release -q -p overtone-graph --example eigenbasis

expect "the archive beats a single-peak search" \
  'the archive helped' \
  cargo run --release -q -p overtone-qd --example menagerie -- 1200

expect "conditioning pays only on periodic structure" \
  'two-periodic +0\.0197 +0\.0000 +0\.13' \
  cargo run --release -q -p overtone-walk --example transport

# Phase 8, M24-M27. Each of these is a claim the README makes, and each is a claim the spec
# gets wrong, so the pattern pins the number rather than merely the fact that it ran.
expect "exact arithmetic does not flatline" \
  'Nelder-Mead +0\.0[0-9]+ +0\.00e0 +5/5' \
  cargo run --release -q -p overtone-opt --example flatline

expect "shots needed are exponential, CMA-ES shallowest" \
  'CMA-ES +0\.6[0-9]+ +0\.9[0-9]+' \
  cargo run --release -q -p overtone-opt --example flatline

expect "kappa=1 fits an expectile, kappa=0 a quantile" \
  '100000 +0\.3[0-9]+ .* 0\.01[0-9]+' \
  cargo run --release -q -p overtone-rl --example shot_dial

expect "shots do not narrow the return distribution" \
  '10000 +0\.68' \
  cargo run --release -q -p overtone-rl --example shot_dial

expect "the two Laplacians differ on a maze" \
  'maze \(irregular\): +\|\| L/d - L_norm \|\|_inf = 8\.0' \
  cargo run --release -q -p overtone-graph --example eigenoptions

expect "Go-Explore loses at 2751 vertices" \
  '2751 +160 .* 8/21' \
  cargo run --release -q -p overtone-graph --example eigenoptions

expect "the gate-count penalty has the wrong sign" \
  'gate count +-0\.[45]' \
  cargo run --release -q -p overtone-qd --example architecture

expect "reach alone beats the combined reward" \
  'reach alone, as a selector: top 20 mean J 0\.[56]' \
  cargo run --release -q -p overtone-qd --example architecture

# Phase 9, M34-M37. Orbit is headless, so every claim is asserted from the examples.
expect "the certificate is sound on every pair" \
  'soundness held on every pair' \
  cargo run --release -q -p overtone-orbit --example checkmate

expect "the certificate is incomplete by counting" \
  'local X    4         30         4            8            4         18' \
  cargo run --release -q -p overtone-orbit --example checkmate

expect "checkmate is complete against real positions" \
  'TFIM    4          30             30               30         1.000' \
  cargo run --release -q -p overtone-orbit --example checkmate

expect "the branching factor lands in the band" \
  'pawn 4, rook 4, bishop 6, knight 12' \
  cargo run --release -q -p overtone-orbit --example ladder

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
               '<b id="mn-beta">0\.827</b>' '<b id="mn-regime">superdiffusive</b>' \
               '<b id="lt-eigenvalue">0\.0[0-9]+</b>' '<b id="lt-error">[0-9]' \
               '<b id="lt-classical">[0-9]'; do
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

# Phase 12 acceptance. Part IX.
expect "corridor reproduces Berlekamp & Wolfe" \
  'test result: ok\. 4 passed' \
  cargo test --release -q -p overtone-cgt --test corridor

expect "hottest-first is not optimal" \
  'loss +1\.0000 points' \
  cargo run --release -q -p overtone-cgt --example hottest

expect "hottest-first IS optimal on plain switches" \
  'hottest-first was optimal every time' \
  cargo run --release -q -p overtone-cgt --example hottest

expect "32x32 temperature field fits a frame" \
  'VERDICT frame-budget-ok' \
  cargo run --release -q -p overtone-cgt --example heatmap

expect "seven rings: branching 2, 85 moves" \
  'seven rings: branching 2\.0, and still 85 moves' \
  cargo run --release -q -p overtone-graph --example rings

expect "rings state graph is a path" \
  '^ +7 +128 +127 +2 +85' \
  cargo run --release -q -p overtone-graph --example rings

expect "two temperatures do not track each other" \
  'VERDICT no-correlation' \
  cargo run --release -q -p overtone-orbit --example twotemps

expect "decomposition does not leak" \
  'mean interaction leak .*: 0\.0000' \
  cargo run --release -q -p overtone-orbit --example twotemps

# Phase 13 acceptance. Part VI 1, Part VIII 1, 8, 10.
expect "Pauli exclusion is exact in the mode basis" \
  'fermionic +1\.00 +0\.000000' \
  cargo run --release -q -p overtone-walk --example statistics

expect "two fermions share a site" \
  'fermionic +1\.00 +0\.000000 +0\.093750' \
  cargo run --release -q -p overtone-walk --example statistics

expect "the anyonic dial interpolates" \
  'anyonic +0\.50 +0\.093750' \
  cargo run --release -q -p overtone-walk --example statistics

expect "bosons bunch where fermions do not" \
  'similarity\(bosonic, fermionic\) += 0\.7708' \
  cargo run --release -q -p overtone-walk --example statistics

expect "Overtone-100 is fully verified" \
  'VERDICT all-verified: 100/100' \
  cargo run --release -q -p overtone-otn --example hundred

expect "no benchmark category can be guessed" \
  'reachability balance: 15 reachable, 15 not' \
  cargo run --release -q -p overtone-otn --example hundred

expect "a puzzle is a few hundred bytes" \
  'the whole set serialises to [0-9]+ bytes \(2[0-9][0-9] per puzzle\)' \
  cargo run --release -q -p overtone-otn --example hundred

expect ".otn round-trips byte-exactly" \
  'test result: ok\. 11 passed' \
  cargo test --release -q -p overtone-otn --test roundtrip

# Phase 14 acceptance. Part II 5, 6; Decisions 03 Q7.
expect "the welded tree collapses to 4n+2" \
  '^ 10 +4094 +12280 +42 +welded tree' \
  cargo run --release -q -p overtone-walk --example hitting

expect "Kempe: linear time, exponential classical" \
  '^ 16 +26 +0\.82[0-9]+ +7\.0766e4' \
  cargo run --release -q -p overtone-walk --example hitting

expect "the reduction matches the graph to 1e-15" \
  'welded tree n=3 .*worst \|p_graph - p_reduced\| = [0-9]\.[0-9]+e-1[5-9]' \
  cargo run --release -q -p overtone-walk --example hitting

expect "the welded walk beats its own theorem" \
  '^ 10 +42 +204 +23 +0\.619327 +0\.005000' \
  cargo run --release -q -p overtone-walk --example hitting

expect "the DFT coin cages 3/7 of the amplitude" \
  'hypercube n=4 +Dft +arrives 0\.571429 +never arrives 0\.428571' \
  cargo run --release -q -p overtone-walk --example hitting

expect "Table 2 reproduces, and Long's layer is exact" \
  '^ 150 +323 +2\.153 +-0\.457824 +2 +1\.4818 +1\.00000000000' \
  cargo run --release -q -p overtone-walk --example hitting

# Phase 15 acceptance. Part VIII 2; Decisions-01 Q1, Q4; Decisions-03 Q4, Q5; Decisions-05.
expect "the language rejects with its vocabulary" \
  'unknown feature `reach_margin`; v1 froze dim_g, orbit_size, safe_set_size, half_chain_entropy' \
  cargo run --release -q -p overtone-orbit --example agents

expect "a field that cannot act is rejected" \
  '`rollout` is only meaningful for kind = "mcts", not `greedy`' \
  cargo run --release -q -p overtone-orbit --example agents

expect "progressive widening holds the root to sqrt(N)" \
  '^ +kestrel +512 +513 +23 +1[0-9][0-9] +[0-9]+ +' \
  cargo run --release -q -p overtone-orbit --example agents

expect "the opening is cold, and only the opening" \
  '^ +0 +\[-1\.0, -1\.0, -1\.0\] +0\.00' \
  cargo run --release -q -p overtone-orbit --example agents

expect "budget is a compute axis for all three kinds" \
  '^ +pike +64 +12 +0 +0 +1\.000' \
  cargo run --release -q -p overtone-orbit --example agents

expect "the language digest is pinned" \
  'test result: ok\. 12 passed' \
  cargo test --release -q -p overtone-orbit --test language

expect "presentation constants cannot change a result" \
  'test result: ok\. 3 passed' \
  cargo test --release -q -p overtone-orbit --test presentation

# Phase 16 acceptance. Decisions-06 Q16, Q17.
expect "the game heats up, it is not cold" \
  '^  4   0\.00  0\.04  0\.04  0\.21  0\.21  0\.33  0\.46  0\.62  0\.67  0\.75' \
  cargo run --release -q -p overtone-orbit --example coldness

expect "the hot floor kills the dust at n>=6" \
  '^  6       2        36      0\.0%' \
  cargo run --release -q -p overtone-orbit --example coldness

expect "the standing score dies at n=6, not n=5" \
  '^  5         7\.617e-2         1\.016e-1' \
  cargo run --release -q -p overtone-orbit --example coldness

# Phase 17 acceptance. Decisions-01 Q8; Decisions-06 Q17; Browne CoG 2022.
expect "the memo pays for the sibling searches" \
  '^ +stoat +512 +128 +2 +98\.4%' \
  cargo run --release -q -p overtone-orbit --example memo

expect "and pays nothing for mcts, by construction" \
  '^ +kestrel +512 +1022 +1022 +0\.0%' \
  cargo run --release -q -p overtone-orbit --example memo

expect "Browne's formula, checked against the paper" \
  'test result: ok\. 6 passed' \
  cargo test --release -q -p overtone-orbit --test memo

expect "the Skill Trace at the ruled width" \
  'ST = y \+ \(1 - y\) A = 0\.4305' \
  cargo run --release -q -p overtone-orbit --example trace

# Deployment. The bundle a Hugging Face Static Space serves has to be self-contained, and that
# is only visible after a push otherwise.
run "static bundle is self-contained"  ./scripts/build_space.sh /tmp/overtone-space-check

echo
echo "  passed=$passed failed=$failed"
[ "$failed" -eq 0 ]
