# OVERTONE — phase plan

The specs define milestones M0–M33 across six documents. This file groups them into nine
executable phases with explicit entry and exit criteria. A phase is done when its exit
criteria are green in CI, not when its code is written.

Governing constraint: **Part III §12, "Minimum viable Overtone."** If scope has to be cut,
the version that keeps the thesis intact is Phase 1 + Phase 2 + Phase 3 + the closure
animation from Phase 6. Everything else is elaboration. Re-read that section whenever a new
panel suggests itself.

Second constraint, from **Part VI §0 and §5.5**: the `Braid` arena is the highest-variance
item in the series — "done right it is what puts the project on the front page, done wrong
it is the thing that discredits it." It therefore goes **last** (Phase 9), and `Lab` remains
the default tab regardless of how visually loud `Braid` becomes. Part VI's one governing
rule, **every mechanic must be a theorem**, is a hard filter: a quantity that exists for
balance rather than physics does not exist.

---

## Phase 1 — Foundation: simulator and verified gradients  *(M0, M1)*  — DONE

**Spec:** Part I §5, §6.1, §6.4, §10, §14.

Workspace skeleton, state-vector engine, both gradient paths, differential testing.

- `overtone-sim`: SoA state vector, gate set `RX RY RZ H CZ CNOT` plus arbitrary
  single-qubit unitary, expectation values for Pauli observables.
- Adjoint gradients — constant memory in circuit depth, all parameters in ~two passes.
- Parameter-shift gradients — exact, `O(P)`, hardware-honest.
- Oracles: a dense Kronecker reference sharing no code with the strided kernels, and
  Yao.jl. (A NumPy oracle was scoped and dropped: with Julia installed, Yao.jl is the
  stronger check, and a reference I write myself is not independent in the way that
  matters.)

**Exit criteria**
- Adjoint and parameter-shift agree to `1e-10` on randomised circuits, every gate type.
- Strided kernels match the dense Kronecker reference to `1e-12`.
- Norm preserved to `1e-12` under every unitary.
- Seeded runs reproduce bit-for-bit.
- `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test` all green.

**Gate:** do not start Phase 2 until both gradient paths agree. Part I §14 — "Nothing
renders until the gradients are right."

---

## Phase 2 — RL loop, headless  *(M2)*  — DONE

**Spec:** Part I §6.2, §6.3, §7.1.

- Data re-uploading ansatz, `L` layers, trainable-or-pinned input scaling `λ`.
- RAW-PQC (Born rule, strictly band-limited) and SOFTMAX-PQC (trainable output weights).
- `SpectralControl-k` contextual bandit; REINFORCE with baseline.
- LP ceiling solver — maximise the frequency-`k` Fourier coefficient over degree-`L` trig
  polynomials with `|p| ≤ 1`. Solve numerically; do not guess a closed form.
- Native CLI trains and emits JSONL traces.

**Exit criteria — all met**
- The `L < k` zero-return result reproduces, seeded, in **0.08 s** against a 30 s budget.
  Measured `J = 0.000000` exactly, not merely under the specified `|J| < 0.02`.
- Trained agents land on the LP ceiling staircase `J*(C)`: `0.499656` against a step of
  `0.5`.
- Trainable `λ` dissolves the ceiling: `L = 1`, whose pinned ceiling is `0`, reaches
  `J = 0.500940` with `λ` locked at `3.017`.

**Corrections carried forward from the build**
- The frequency ceiling is `L × (encoding gates per layer)`, not `L`. The zero-return test
  depends on this; see `Ansatz::frequency_ceiling`.
- Returns are asserted by quadrature, not by episode averaging. The sampled mean over 5000
  episodes has a standard error near `0.014`, so the spec's `|J| < 0.02` on sampled return
  would be a 1.4σ test that flakes about one run in six.
- The `λ` return landscape is a resonance curve with a capture range of roughly
  `[1.8, 4.2]` at `k = 3`. Gradient ascent from `λ = 1` locks onto a sidelobe. A coarse
  tune precedes the fine tune.
- The LP staircase bounds RAW-PQC only. SOFTMAX-PQC is not band-limited and measurably
  exceeds it — at `L = 2, k = 3` it scores `0.110` where the ceiling is `0`.

---

## Phase 3 — Instrumentation  *(M3)*  — DONE

**Spec:** Part I §6.5, §6.6, §6.7.

The crown jewel. `overtone-spec` must make each of the five Part I §1 phenomena measurable
by a function that returns numbers, each with a test.

- Fourier extraction: sample the policy on `N = 512` over `[-π, π)`, real FFT.
  **Transform `π(a|s)`, not the logit.** Part I §6.5 says "logit", but for a SOFTMAX-PQC the
  logit is `β·w·⟨Z⟩`, exactly as band-limited as the RAW-PQC's — transforming it would show
  no leakage and quietly falsify a true claim. The leakage lives in the probability.
- Von Neumann entropy across the half-chain bipartition; `--no-entangle` ablation.
- Gradient-variance sweep, `n = 2…12`, global vs local observable, fitted exponent.

**Exit criteria — all met**
- No spectral energy beyond the ceiling for RAW-PQC: measured at the transform's rounding
  floor, `~1e-16`, against the specified `1e-9`. Holds trained and untrained, every seed.
- SOFTMAX-PQC leaks measurably: leakage ratio `0.396` at `L = 3`, against exactly `0` for
  RAW-PQC on the same circuit.
- Global observable collapses exponentially, `Var ~ 2^(-1.03 n)` with `R² = 0.999`; the
  local observable decays at `0.337` with `R² = 0.88` — not exponential, which is the
  finding.

**Corrections carried forward from the build**
- **The `3L` phrasing was too loose and was changed, per Part I §12.** Verified
  numerically: the leakage sits at odd harmonics of the policy's *dominant in-band
  frequency*, not of `L`. Trained at `L = k = 3` the leaked peaks are at 9, 15, 21, 27 with
  even multiples suppressed ~500×; the two coincide only because the trained policy
  concentrates at `k`. At `L = 2` the in-band content is spread and the leakage is
  broadband instead.
- **The plateau probe index is not a free choice.** A mid-circuit probe hits *structurally
  zero* gradients for a local observable (variance `~1e-33`), which inverts the
  global-versus-local result. Part I §6.7's `∂⟨O⟩/∂θ₁` is load-bearing.
- Measured global rate is `1.03`, not the `1.98` Part I §6.7 quotes. That figure is the
  full 2-design result; this ansatz does not reach a 2-design at the depths swept. The
  hypothesis that probing `θ₁` halves the exponent was **tested and rejected** — a
  mid-circuit probe gives 1.08, not 2.
- The local observable's escape is conditional on shallow depth. At linear depth its rate
  rises to `0.914`, essentially the global one. The honest claim is "shallow *and* local".
- `|J| ≤ |c_k|` with equality only at perfect phase alignment, so the spectral instrument
  and the return bound each other rather than being equal. Trained policies saturate it to
  within 0.05 rad.

---

## Phase 4 — The browser  *(M4, M15)*  — DONE

**Spec:** Part I §8; Part IV §1.

WASM bindings, the six panels, the two-zone notebook/instrument design. Deploy to a
Hugging Face **Static Space** (free for everyone; compute Spaces are not) mirrored on
GitHub Pages.

**Exit criteria — met, with one claim corrected**
- The hero loop is the real engine. It runs a `SpectralControl-3` agent with trainable `λ`
  from inside the capture range and locks on at `λ = 3.007`, `J = 0.5005`, live.
- JS is **580 of 800** lines, enforced by `scripts/check_js_budget.sh`. The generated
  wasm-bindgen glue is excluded: the budget exists so logic does not migrate out of Rust,
  and counting machine-generated bindings would measure the wrong thing.
- The page boots and populates every readout from the engine, checked in CI by a headless
  Chrome smoke test that fails if the loading state survives or a readout stays blank.
- No horizontal overflow at a 360px layout viewport, measured in an iframe
  (`scrollWidth == clientWidth`). Headless Chrome clamps its own viewport near 485px, so
  screenshots at 360 are misleading; the iframe measurement is the real one.
- Hugging Face **Static Spaces are still free for everyone**, verified against
  `huggingface.co/docs/hub/spaces-overview` on 2026-09-06 and quoted in
  `scripts/deploy_space.sh`. Re-read that page before each release regardless.

**Corrections carried forward from the build**
- **"WASM and native produce identical seeded trajectories" is false as stated, and the
  test now says what is true.** IEEE-754 requires correct rounding for arithmetic and
  `sqrt` but *not* for transcendentals; native glibc `libm` and wasm32's disagree by one
  ULP on roughly 5% of `sin`/`cos` evaluations. A trajectory therefore drifts in the last
  two digits across targets: measured worst relative difference `5.55e-16`. Within a
  target it *is* bit-exact, and that is the guarantee permalinks actually need — two people
  opening the same link run the same `.wasm`. `scripts/wasm_determinism.sh` asserts a `1e-13`
  relative tolerance and prints the worst difference.
- **`wasm-bindgen` maps Rust `u64` to JavaScript `BigInt`, not `Number`.** Seeds cross the
  boundary as `u32`. Found only by running the page in a real browser; the Rust side
  compiled and the wasm built without complaint.
- **The spectrum's ceiling rule has to track `λ`.** With trainable scaling the reachable set
  is `λ·{-C..C}`, so the ceiling *moves* rather than being exceeded. Drawing it at the
  integer ceiling reported the hero — an agent that had tuned itself into resonance — as
  having leaked, with a ratio of 28. `Lab::effective_ceiling` returns the real-valued
  position, and the leakage readout is suppressed as not meaningful when `λ` is trainable,
  because a non-integer reachable frequency spreads across bins for ordinary sampling
  reasons that have nothing to do with the softmax.

**Not yet done:** the 60fps target at `n = 6, L = 4` is not measured. There is no frame
timing instrumentation and no mid-range laptop in the loop, so the claim is unverified
rather than met.

---

## Phase 5 — The lattice  *(M6–M9, M28)*

**Spec:** Part II.

`overtone-walk` and `overtone-maze`. Sparse infinite lattice with a strict light cone,
procedural mazes from a seeded coordinate hash, DTQW with Hadamard and Grover coins,
trajectory-based decoherence, the designed dark corridor, Anderson localization, glued
trees, Szegedy hitting time, and finally the learned coin.

Plus **M28** from Part VI §1: the two-particle walk, with bosonic, fermionic and anyonic
exchange statistics. Part VI §7 is explicit that this is "a physics milestone, not a game
milestone", so it belongs here with the walk engine rather than in the arena. It is also
what makes the arena's class system a consequence of the spin-statistics theorem rather
than a design choice, so getting it right early sets the tone for Phase 9.

**Exit criteria**
- Fitted spreading exponents `1.00 ± 0.03` (quantum) and `0.50 ± 0.03` (classical).
- Infinite sparse lattice matches a far-boundaried finite lattice to `1e-12`.
- Designed dark corridor below `1e-6`; full dephasing recovers the classical walk to
  TV `< 1e-3`.
- Learned coin benchmarked against Grover, Hadamard, and classical — **publish either way.**
- **M28:** two-particle correlation patterns reproduce Sansoni et al. (PRL 108, 010502) for
  bosonic bunching, fermionic antibunching, and the anyonic `φ = π/2` case. Part VI §8: if
  these do not match the published patterns, the arena's physics is wrong and everything
  built on it is theatre.

May start as soon as Phase 1 lands; it needs the simulator but not the RL loop.

---

## Phase 6 — Closure  *(M10–M14)*

**Spec:** Part III.

`overtone-lie` (Pauli bitsets, Lie closure, `dim(g)`), `overtone-gsim` (Givens-rotation
evolution), `overtone-mps` (the dequantization test).

**Exit criteria**
- `dim(g)` matches published values for hardware-efficient, TFIM, and Heisenberg ansätze.
- Reproduces Figure 2 of Ragone et al. (2024) from our own engine.
- g-sim agrees with the state vector to `1e-12` where both apply.
- The 100-qubit claim ships **with its caveat on the same screen**: polynomial DLA means
  classically simulable.
- Effective `χ` published for every agent in the repo, favourable or not.

---

## Phase 7 — Menagerie  *(M16–M21)*

**Spec:** Part IV.

Substitution-rule worlds with a live transport-exponent readout, sonification, DLA-derived
sigils and stat blocks, the four-agent race, MAP-Elites archive, the WFC entropy panel.

Highest value per line: **sonification** (§5.1) — the project is called Overtone and the
policy is a Fourier series; hearing `λ` beat into resonance lands before the bar chart does.
Off by default, never autoplay.

**Exit criteria**
- Measured `β` matches published regimes; Fibonacci vs Rudin–Shapiro visibly different.
- Every stat-block line is measured, never authored.

---

## Phase 8 — Exotic  *(M22–M27)*

**Spec:** Part V.

Priority order is the spec's own: the shared eigenbasis panel, then solve-by-eigenvector and
policy superposition, then the optimiser flatline demo. The eigenbasis panel is one
eigensolve and one toggle, and it retroactively unifies all four earlier parts — highest
payoff per line in the series.

**Exit criteria**
- Real-vs-imaginary exponent toggle renders diffusion and interference from one eigenvector.
- `α·z_A + β·z_B` produces a correct policy for a third task with no training.
- All four optimisers flatline in a barren plateau, with Arrasmith et al. cited.

---

## Phase 9 — Braid: the adversarial arena  *(M29–M33)*

**Spec:** Part VI. **Gated on Phases 5, 6 and 7**, and on the rest of the project already
reading as serious.

The one rule: **every mechanic must be a theorem.** Nothing is invented for balance; if the
arena is unbalanced, that is a finding. The three ideas it implements each turn out to have
an exact physical counterpart:

- **Fighting → exchange statistics** (Part VI §1). A fermionic agent walls off a corridor
  because Pauli exclusion says so; a bosonic one merges on overlap because that is what
  bunching is. `φ` is a continuous dial between them. Built on M28.
- **Being chased → decoherence** (§2). The pursuer is an advancing dephasing front. Capture
  is the transport exponent falling from `β ≈ 1` to `β ≈ 0.5` — a phase transition in the
  player's own transport, with no separate capture condition in the code. The core dilemma
  is the measurement problem: information costs speed. The Zeno trap (panic-measuring
  freezes you) falls out of `measure` and is never special-cased or explained in advance.
- **Consuming → Lie closure** (§3). Absorption runs a real closure of `g_A ∪ g_B` in
  milliseconds on Pauli bitsets, `dim(g)` jumps, and the sigil redraws because it is a
  function of the algebra. The price is Part III's own results: eat too much and die of a
  barren plateau; stay small and be perfectly predicted by an opponent running `g-sim`.
- **Braiding → topological computation** (§4). Worldlines winding around each other apply a
  unitary that depends only on the topology of the path. In an arena whose antagonist is
  decoherence, the braid is the one thing the pursuer cannot take. The braid word
  accumulates beside the maze in generators `σ₁ σ₂⁻¹ …`.

**Milestones:** M29 pursuer · M30 player control (`evolve`, `measure`, `phase`, `absorb`) ·
M31 absorption and live closure · M32 braiding · M33 the harness.

**Exit criteria**
- Capture is detected as the `β` transition, with no separate capture condition in the code.
- The barren-plateau death is reachable: a player who absorbs everything becomes measurably
  untrainable, verified by the Phase 3 gradient-variance instrument.
- Zeno freezing emerges from repeated `measure` without a special case.
- Braid words are invariant under geometric deformation of the route that preserves its
  topology.
- The run report contains only physical observables, and cites the theorems that decided
  the outcome.

**Structural safeguards (Part VI §5), all load-bearing**
- No invented numbers. State is: wavefunction, coherence, `dim(g)`, `φ`, braid word.
  No HP, damage, XP or cooldowns. If it is not an observable, it is not on screen.
- The Phase 1–3 instruments stay live during play. Playing is how the measurement is driven.
- No win screen — a run report that reads like an experiment log.
- Language: *run*, *opponent*, *report*, *absorb*. Never *level*, *enemy*, *score*, *kill*.
- `Braid` is never the landing page and never precedes `Lab` in the nav.
- The arena **is** the evaluation harness: opponents are Phase 7's MAP-Elites elites, and
  runs push human-in-the-loop evaluation data to the Atlas. Say that in the docs, because
  it is true and it is what keeps a sceptical reader on the page.

**Free theorems that become rules:** no-cloning forbids save-scumming; monogamy of
entanglement (Coffman, Kundu & Wootters 2000) makes alliances mathematically exclusive.
Meyer's penny flip is an optional thirty-second opening duel — ship it with the standing
critique that quantum-game advantages can sometimes be reproduced by classical correlated
equilibria.

**Do not claim quantum advantage in the arena.** The claim is that the mechanics are
theorems. Sometimes the quantum player should lose, and that is the more interesting report.

---

## Cross-cutting, from the start

- **No emoji**, anywhere.
- Every README claim is paired with a CI test that fails when the claim stops being true.
- Negative results are published, not buried.
- Determinism: seed everything; native and WASM agree.
- Scope discipline: Part III §12 governs.
