# OVERTONE — phase plan

The specs define milestones M0–M33 across seven documents (six parts plus the Part VI-A
traps addendum). This file groups them into nine
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

## Phase 5 — The lattice  *(M6–M9, M28)*  — PARTIAL

**The one-dimensional core of M6 and M7 landed early, in Phase 7**, because Part IV's worlds
cannot be measured without a walk. `overtone-walk` has the strict light cone, the sparse
representation, position-dependent coins, trajectory-based decoherence, the fitted spreading
exponent, and the Hadamard/Grover/classical baselines. What is still outstanding is
everything two-dimensional and everything about mazes.

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

## Phase 6 — Closure  *(M10–M14)*  — DONE

**Spec:** Part III.

`overtone-lie` (Pauli bitsets, Lie closure, `dim(g)`), `overtone-gsim` (Givens-rotation
evolution in the DLA basis), `overtone-mps` (the dequantization test), the `Closure` chapter
in the browser, and `overtone predict` / `overtone dequantize` as tools.

Built out of order: Phase 5 is still open. Phase 6 depends on Phase 1 and nothing else, and
Part III §12 puts the closure animation in the minimum-viable core while Part II is
"elaboration on a thesis that this core already establishes."

**Exit criteria — met, with two rewritten**
- `dim(g)` matches the published classification for thirteen generating families, checked
  against closed forms from Theorem IV.1 of Wiersema, Kokcu, Kemper and Bakalov, npj Quantum
  Information 10, 110 (2024): `a_0` `a_1` `a_4` `a_7` `a_8` `a_9` `a_11` `a_14` `a_15`, the
  TFIM open chain and ring, the hardware-efficient ansatz, and the single-qubit case. A
  dense Gram-Schmidt oracle sharing no code with the bitsets agrees on all of them.
- **"Reproduces Figure 2 of Ragone et al." was not achievable as written and was replaced by
  something stronger** — see the corrections below. Theorem 1's closed form is reproduced to
  within 3% at `n = 3..7`.
- g-sim agrees with the state vector to `1e-12` on randomised circuits and on the policy
  itself, value and gradient. Measured worst difference at the assertion: below `1e-12`
  across `n = 2..5`.
- The 100-qubit claim ships with its caveat in the same paragraph, in the README, in the
  example's own output, and in the crate documentation. Measured: `dim(g) = 19900`, closure
  and rotation planes in 1.3 s, 150 exact gradient steps in 3.9 s, return `0` to `0.251`.
- Effective `χ` is published for every agent in the repo. All of them are small tensor
  networks; see the README table.

**Corrections carried forward from the build**

- **Figure 2 of Ragone et al. (2024) is a schematic.** Figures 1 and 2 both illustrate where
  barren plateaus come from; neither is a numerical plot, so M11's acceptance test as
  written has no curve to reproduce. **Theorem 1 is the reproducible object and it is a
  better target**: `Var[loss] = Σ_j P_{g_j}(ρ) P_{g_j}(O) / dim(g_j)` is an exact closed
  form with nothing to eyeball. Measured against 4000 random circuits at depth 64, the ratio
  of measurement to prediction is 0.97, 1.03, 0.99, 0.97, 0.99 for `n = 3..7` on the TFIM
  chain and 0.998, 0.993, 0.979, 0.989 for `n = 3..6` on the Heisenberg chain.
- **The theorem's 2-design assumption is visible in the data**, which is the more
  interesting half. At depth 1 the same ratio is 1.86 at `n = 7`; it falls to 1.64 at depth
  8, 1.08 at 32 and 0.99 at 64. The prediction is not wrong at shallow depth — its
  hypothesis is not yet true.
- **`so` and `sp` are written in different notations inside one theorem.** `a_11(n) ≅
  so(2^n)` is a matrix size; `a_9(n) ≅ sp(2^(n-2))` is not, and its dimension is
  `m(2m+1)` with `m = 2^(n-2)` — 36 at `n = 4`, where reading it the way `so` reads gives
  10. Caught by the test failing, not by reading carefully.
- **The block decomposition this crate computes is the Pauli-visible one, and it can be
  coarser than the abstract one.** The Heisenberg chain at even `n` is `su(2^(n-2))^(+4)`
  in the classification, but its 60 Pauli strings at `n = 4` form a single connected block,
  because a Pauli string there is a sum of pieces from several summands. Theorem 1 evaluated
  on the visible block still lands within 1% for that case, but the two are different
  objects and the difference is not rounding.
- **The DLA of our own Part I ansatz is not the one the informal statement gives it.** "The
  hardware-efficient ansatz has DLA `su(2^n)`" is true of parameterised entanglers and false
  of fixed ones. Part I §6.2 builds fixed CZ layers, so the algebra generated by the
  trainable gates is `su(2)^(+n)` — dimension `3n`, polynomial, nominally "trainable" —
  while Part I §6.7 *measures* its global-observable gradient variance falling like
  `2^(-1.03 n)`. Theorem 1 does not apply: a fixed Clifford is not a one-parameter subgroup,
  so the circuit is not in `exp(g)`. This is now a measured claim, not an argument: the
  Fisher rank of that circuit exceeds `3n`, which an algebra containing it could not allow.
  `overtone predict` refuses to make a trainability claim when fixed entanglers are present
  and says why. See Diaz et al., arXiv:2310.11505, which Part III §14 lists for exactly this.
- **A parity theorem, found because a hundred-qubit agent scored exactly zero.** With
  generators `{X_q} ∪ {Z_qZ_{q+1}}`, observable `Σ_q X_q` and one layer, the policy is an
  odd function of the observation for *every* parameter setting, so a `cos(k s)` reward
  gives `J ≡ 0` and a gradient of zero in every direction. The proof is a conserved count of
  `X`-or-`Y` letters. **I first wrote this as holding at every depth, on the strength of a
  test that had only ever reached one layer**; two layers breaks the conservation and does
  score. Both directions are now asserted.
- **Every agent in this repository is a small tensor network**, and at `χ = 1` — a product
  state — the *return* is already matched. That is the unflattering answer Part III §5 asks
  for and it is in the README.

**Deliberately not built, and not claimed**
- `overtone-serve`: no pyo3 extension, no FastAPI Space, no Atlas dataset. Part III §7's
  architecture is a second surface; Part III §13 says the demo must never depend on it, and
  the demo does not exist on it at all yet.
- Panels C3 (the phase-transition histogram), C4 (the QFIM spectrum and natural gradient)
  and C5 (the dequantization dial). The QFIM itself is built and tested — `rank(QFIM) ≤
  dim(g)` is checked directly — but there is no panel and no natural-gradient optimiser.
- `overtone-mps` compresses exact state vectors, so it is a dequantization instrument for
  circuits the state-vector engine can run, not the `n = 100, χ = 512` Engine B of Part III
  §4. One left-to-right sweep, not a variational optimum, which makes every effective `χ` it
  reports an upper bound.
- The site has no tab bar: it is one scrolling notebook, and `Closure` is its third chapter
  after the Lab instruments. `Lab` still comes first, which is what Part VI §5.5 is
  protecting.

---

## Phase 7 — Menagerie  *(M16, M17, M18, M20, and the 1D core of M6/M7)*  — DONE

**Spec:** Part IV.

Substitution-rule worlds with a live transport-exponent readout, sonification, DLA-derived
sigils, the four-agent race, and the WFC entropy panel.

**Phase 5 was not a prerequisite and it was not skipped so much as partially pulled
forward.** Part IV's M16 has no meaning without a walk to measure — a world with no
measurable transport exponent is a skin — so the one-dimensional core of Part II's M6 and M7
was built here: `overtone-walk`, with the strict light cone, position-dependent coins,
trajectory decoherence and the fitted exponent. What remains Phase 5's is everything
two-dimensional: mazes, the dark corridor, glued trees, Szegedy hitting time, M28's
two-particle statistics, and M9's learned coin.

**Exit criteria — all met**
- Measured exponents match the published regimes. Clean lattice `1.000` with `R^2 = 1.000`
  against Part II M6's `1.00 +- 0.03`; the classical baseline is `0.500` exactly, as a
  closed form rather than a simulation.
- Fibonacci and Rudin-Shapiro are visibly different: `beta = 0.82` against a trace that
  saturates at `sigma = 26` where the clean lattice reaches `325`.
- Every stat-block line is measured. The sigil is a deterministic function of the algebra,
  tested by reaching the same algebra through a reversed generating set and requiring the
  identical mark.
- The three aperiodic words match Lo Gullo et al. Fig. 1 character for character.
- WFC terminates with a seam-consistent grid on every seed, its entropy falls monotonically
  to zero, and the same seed gives the same maze.

**Corrections carried forward from the build**
- **Part IV 3's spectral labels are swapped.** It calls Fibonacci singular-continuous and
  Rudin-Shapiro discrete. Fibonacci has a *pure point* spectrum, Thue-Morse is the singular
  continuous one, and Rudin-Shapiro is *absolutely continuous* — which is exactly why it
  behaves like disorder. The behaviours in that table are roughly right; the labels are not,
  and the panel's caption depends on them.
- **A `beta` without its `R^2` is not a measurement.** Anderson localization does not produce
  a small exponent, it produces a `sigma` that saturates, and a line fitted through a plateau
  reports the plateau's noise as a slope. The first disorder run reported `beta = 0.44` with
  `R^2 = 0.20`; the honest statement is that `sigma` saturates at three sites.
- **A single disorder realisation is not a measurement either.** With a two-letter word the
  fitted exponent swung from `0.15` to `0.34` between seeds. Anderson localization is a
  statement about a continuum of local parameters, so the disorder world now draws a
  continuous coin angle per site.
- **A constant coin cannot see the world, and that is the point.** The substrate selects
  between two coins, so an agent playing the same coin at both letters produces a
  bit-identical trace on the Fibonacci world and on the clean lattice. The first race
  reported exactly that and it read as a bug. Part II 2 says the policy is the coin; an agent
  whose coin ignores the local feature has no policy.
- **Spread is the wrong objective for the race.** `sigma` is maximised by the coin that does
  not mix at all — two ballistic beams sitting on the light cone — and that coin is optimal
  on every substrate, so optimising it teaches an agent nothing about the world. The
  objective is now arrival near a distant target, which is what a hitting time measures.
- **The optimiser needed a multi-start and an explicit search of the blind solution.** With a
  single start it reported that conditioning on the letter helps on the Fibonacci world; with
  the diagonal searched properly the answer reverses. The honest result is a negative one:
  **conditioning the coin on the local letter pays only on the periodic word.** On every
  aperiodic word the best strategy found is to ignore the substrate.
- **The Grover coin is degenerate in one dimension.** `2|s><s| - I` on a two-dimensional coin
  space is exactly the Pauli `X`, so the field oscillates between two sites and `sigma` stays
  at zero. Part II 7 makes it a mandatory baseline; reporting that it is degenerate is the
  honest form of that baseline.
- **Full decoherence is exactly a simple random walk, not approximately.** After a position
  measurement the field is on one site, so each neighbour receives exactly one coin
  component and the next measurement always leaves a basis state, which the Hadamard coin
  splits exactly in half. Measured total variation from the binomial is `0.0025` at 200000
  trajectories and falls as `1/sqrt(N)` — sampling error, not residual coherence.
- **The JS budget was raised from 800 to 1200, once and deliberately**, because the page now
  carries three sections rather than one. The rule it enforces — no panel computes a physical
  quantity — is unchanged, and if it binds again the fix is to move code into Rust.
- The race optimiser costs about a thousand walks, and calling it from the resize handler
  starved the hero on the Lab panel for seconds. It is computed once per world and cached.

**Deferred, with reasons**
- **M19, the MAP-Elites archive.** Part IV 4 ties it to the nightly Actions job that pushes
  to the Atlas, which is Tier 1 infrastructure that does not exist yet. The descriptors it
  needs — `dim(g)`, `beta`, effective `chi` — are all built and measured, so this is a
  scheduling decision, not a missing capability.
- **M21, the hyperbolic tiling and the Hat monotile.** Part IV 6 puts them last itself, as
  the hardest to render and the least load-bearing.
- **Sonification is built but unheard here.** `web/js/sound.js` plays the environment's
  frequency against `lambda` so the beat falls to zero as the policy locks on. It is wired,
  off by default, and never autoplays — but a headless browser cannot verify a beat, so it
  is untested rather than verified.

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
M31 absorption and live closure · M32 braiding and traps · M33 the harness.

### M32's traps (Part VI-A)

Part VI-A folds six sub-milestones into M32. They are the answer to the one thing Part VI
lacks — terrain that matters — and they obey the same rule as everything else in the arena:
every trap is a published localization or topology result, and none of them carries a tuned
constant.

The structural claim is what makes this cheap: **a trap is not an object, it is a per-cell
parameter.** Local flux, local coin, local disorder strength, local observable locality, all
drawn from the same seeded hash the maze already uses. No entity system, no spawn logic, no
collision detection — which is also the guarantee that it stays physics.

| Trap | Catches | Escape | Rests on |
|---|---|---|---|
| T1 Grover well | anyone in range | outlast the over-rotation | `pi/4 sqrt(N)`, Boyer et al. 1998 |
| T2 Aharonov–Bohm cage | the wrong coin | re-parameterise, or inject disorder | Vidal et al. PRL 81, 5888; arXiv:1910.00845 |
| T3 Topological bound state | anyone crossing | change the bulk topology | Kitagawa et al. PRA 82, 033429 |
| T4 Chiral corridor | anyone entering | forward only | same |
| T5 Disorder patch | **coherent** agents | decohere yourself | Anderson localization, Phase 5 |
| T6 Zeno region | anyone evolving | leave spatially | Misra & Sudarshan 1977 |
| T7 Spectral trap | low-`L` agents | raise `L`, or tune `lambda` | Part I §7.1, Phase 2 |
| T8 Plateau region | **learning** agents | freeze the coin | Cerezo et al. Nat. Commun. 12, 1791 |

The composition principle is the reason this is a system rather than a list: **every escape
is a vulnerability to a different trap**, and every one of those trades is a theorem the
earlier phases already implemented and instrumented. Decohere past T5 and the decoherence
front catches you. Widen `dim(g)` past T7 and `Var[∂C] ∝ 1/dim(g)` walks you into T8. Freeze
the coin to survive T8 and you can no longer re-parameterise out of T2.

**Build order** — the first two carry the system:

- **M32a — T5, T6.** Disorder patches and Zeno regions; reuses Phase 5 entirely. Acceptance:
  `σ(t)` saturates inside a patch for a coherent walker and does not for a decohered one.
- **M32b — T7 with band-limited rendering.** Acceptance: an `L < k` agent's return on trap
  structure is zero to `1e-3`, and its rendered view provably lacks frequency-`k` content.
  **This is the best idea in the addendum**: draw the maze as the agent can represent it, so
  two agents with different `L` are visibly walking through different mazes. It turns Part
  I's central abstraction into something you can look at, and it reuses the Phase 3 spectral
  instrument rather than adding an engine.
- **M32c — T2, Aharonov–Bohm cages.** Acceptance: complete confinement, zero amplitude
  outside the cage to `1e-12`, and release on changing the coin. Part VI-A §7 is explicit
  that this test must not be skipped: AB caging is exact, and a cage that "mostly" confines
  is a slow region with a grand name.
- **M32d — T1, Grover wells.** Acceptance: peak capture at `pi/4 sqrt(N)` steps and
  measurable release on over-rotation, both against the closed form.
- **M32e — T3, T4, topological.** Acceptance: the bound state survives perturbation while a
  non-topological trap of similar depth does not.
- **M32f — T8, plateau regions.** Acceptance: the Phase 3 gradient-variance instrument
  registers the collapse; a frozen-coin agent passes through unaffected.

**Dependencies this adds.** M32a needs Phase 5's decoherence; M32b needs Phases 1–3; M32c
and M32e need Phase 5's lattice and coins; M32f needs Phase 3's instrument and Phase 6's
`dim(g)`. Trap *density* is derived from Phase 7's substrate — the Rudin–Shapiro world is
already strongly confining and needs few, the periodic world is nearly frictionless and can
carry more — and is reported as a measured statistic rather than chosen.

**Extra traps carried into `CLAUDE.md`:** no tuned constant anywhere (`pi/4 sqrt(N)` is
derived, the localisation length is measured); no `Trap` object with a lifecycle; no
signposting, because half of them are invisible by nature and the player's own instruments
are the warning; and no ninth trap, because eight already cover localisation by
interference, disorder, topology, measurement, bandwidth and trainability.

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
