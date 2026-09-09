# OVERTONE — phase plan

The specs define milestones M0–M52 across ten documents: Parts I–IX plus the Part VI-A
traps addendum, as amended by the two Decisions documents. This file groups them into
thirteen executable phases with explicit entry and
exit criteria. A phase is done when its exit criteria are green in CI, not when its code is
written.

Governing constraint: **Part III §12, "Minimum viable Overtone."** If scope has to be cut,
the version that keeps the thesis intact is Phase 1 + Phase 2 + Phase 3 + the closure
animation from Phase 6. Everything else is elaboration. Re-read that section whenever a new
panel suggests itself.

Second constraint, from **Part VII §10**: the arena is one tab and it has one design.
**Part VII supersedes Part VI** — a turn-based game replaces the free-form real-time arena,
`Braid` is renamed `Orbit`, and every Part VI mechanic survives inside Part VII's seven rules
as *content* rather than as rules. Do not build both. Part VI's governing rule survives
unchanged and applies to Part VII: **every mechanic must be a theorem**, and a quantity that
exists for balance rather than physics does not exist. `Lab` remains the default tab.

Third constraint, from **Part VII §0**: depth lives on decision complexity, not on state
count. Adding maze adds states and moves depth toward zero. Every instinct to expand is
checked against the measured depth `d` in Phase 9 before it is acted on — and the interface
is not built until `d` has been measured. That gate is structural: Phase 9 is headless and
Phase 10 does not start on a flat ladder.

Fourth constraint, from **Part VIII §0**: the competition layer is a **notation, not a
server**. Chess became a global competitive game three centuries before servers existed, and
it scaled through a file format. Tier 3 — an actual server — is a standing no.

Fifth constraint, from **Part IX §8**: **do not add a die.** Randomness decouples decision
quality from outcome and lowers `d` — Snakes and Ladders is the canonical minimum in Lantz et
al. for exactly that reason. Overtone is already a dice game and has been since M1: you build
the distribution, then collapse it. Two corollaries with the same status. **Do not expand the
generator hand** — Go has one piece type and 10^170 positions, so richness belongs in the
substrate. **Do not hand-tune an evaluation function** — temperature (Phase 12) exists
precisely so that never has to happen, and a tuned eval would be the first invented number.

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

## Phase 5 — The lattice  *(M6–M9, M28)*  — PARTIAL (M8 and M28 done; M9 and the maze open)

**The one-dimensional core of M6 and M7 landed early, in Phase 7**, because Part IV's worlds
cannot be measured without a walk. `overtone-walk` has the strict light cone, the sparse
representation, position-dependent coins, trajectory-based decoherence, the fitted spreading
exponent, and the Hadamard/Grover/classical baselines. **M28 landed in Phase 13** and **M8 in
Phase 14**, the latter in the order Decisions-03 Q7.4 revised — hypercube before welded trees,
with the Szegedy module deleted rather than built. What is still outstanding is the
two-dimensional maze, the dark corridor, and M9's learned coin.

**Spec:** Part II.

`overtone-walk` and `overtone-maze`. Sparse infinite lattice with a strict light cone,
procedural mazes from a seeded coordinate hash, DTQW with Hadamard and Grover coins,
trajectory-based decoherence, the designed dark corridor, Anderson localization, glued
trees, Szegedy hitting time, and finally the learned coin.

Plus **M28** from Part VI §1: the two-particle walk, with bosonic, fermionic and anyonic
exchange statistics. Part VII keeps this load-bearing — rule 6 of the Orbit ruleset is that
overlapping an opponent merges algebras, and exchange statistics are what decide *what
overlap does*. Note also that Part VII §9.3 bounds the arena to a **windowed region of about
32x32** scrolling inside the endless maze, which is a firmer and much more tractable
requirement than "endless": the two-agent joint state only ever has to be representable
inside that window. Part VI §7 is explicit that this is "a physics milestone, not a game
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

**M19's corrections**
- **Part IV §4's justification for MAP-Elites is false.** It says that in a barren plateau
  "gradient descent is a random walk — and quality-diversity search is one of the few methods
  that still functions", and §7 instructs the docs to explain that reasoning. Arrasmith,
  Cerezo, Czarnik, Cincio and Coles, *Effect of barren plateaus on gradient-free
  optimization*, Quantum 5, 558 (2021), prove that **cost-function differences are
  exponentially suppressed in a plateau**, confirmed numerically for Nelder-Mead, Powell and
  COBYLA. MAP-Elites decides whether an offspring replaces an elite by comparing fitness, so
  it decides on cost differences and the theorem covers it. Phase 8's own exit criteria cite
  that paper for "all four optimisers flatline", so following Part IV literally would have
  put two contradictory claims in one repository.
- **The measured justification is better anyway, and it is the ordinary one.** At a matched
  evaluation budget, identical seed and identical mutation operator, MAP-Elites reached
  `J = +0.502` where a hill climb keeping only the best reached `+0.403`. The archive wins by
  holding structurally different agents — qubit counts, depths, entanglers — that a
  single-peak search abandons. That is quality-diversity working in a *multimodal* landscape,
  which needs no appeal to plateaus.
- **The archive does not render the barren plateau, and the first version of the crate docs
  claimed it would.** Best fitness by `dim(g)` band came out flat — `0.492, 0.488, 0.502,
  0.501, 0.471` — because two to five qubits is nowhere near where `Var ~ 2^(-1.03 n)` bites.
  The collapse is a claim about scale and this archive does not reach it.
- **One descriptor is substituted, and it is stated.** Part IV asks for
  `(dim(g), beta, chi)`. A `SpectralControl` policy is a contextual bandit: it does not move,
  so it has no transport exponent. The substitute is **reach**, `C * |lambda|` — how far in
  frequency the policy can see, where `beta` is how far in space an agent can get.

**Also closed, from the standing-claims list**
- **The 60fps target is now measured.** Part I §8's claim was never checked, and the honest
  form of it is not the frame rate a headless browser reports — `requestAnimationFrame` is
  clamped to the display clock whether the page is working or not. What is meaningful is the
  time the step-and-draw block costs against the 16.7 ms budget, and the page now measures
  and displays it.
- **Sonification is testable and tested.** The frequency mapping moved from JavaScript into
  Rust, where the claim it carries can be asserted: the beat rate is the detuning,
  `110 Hz * |k - lambda|`, so it closes monotonically to zero as the policy locks on. That is
  why the ear beats the eye here, and a headless browser could never have verified it.

**Deferred, with reasons**
- **M19, the MAP-Elites archive — no longer deferred. Built, after Part VIII made it
  load-bearing.** `overtone-qd` fills a `dim(g) x reach x bond dimension` grid by exact
  return, and writes `results/menagerie.json`, which is the cold start M43 needs. Three
  corrections came out of building it, below.
- **M21, the hyperbolic tiling and the Hat monotile.** Part IV 6 puts them last itself, as
  the hardest to render and the least load-bearing.
- **Sonification is built but unheard here.** `web/js/sound.js` plays the environment's
  frequency against `lambda` so the beat falls to zero as the policy locks on. It is wired,
  off by default, and never autoplays — but a headless browser cannot verify a beat, so it
  is untested rather than verified.

---

## Phase 8 — Exotic  *(M22–M27)*  — DONE

**Spec:** Part V.

**Ordering changed by Part VII.** Part V §2's linearly-solvable MDP — Bellman as a
largest-eigenvalue problem under `z = exp(-v/lambda)` — was a striking panel. It is now
**infrastructure**: Phase 9's M36 makes it the endgame tablebase, and Phase 11's M40 makes it
the ground truth for the puzzle set. Two later milestones rest on it, so it was built first,
ahead of the eigenbasis panel it previously sat behind.

**The graph came from Phase 7, not from Phase 5.** Both milestones need a maze *graph*, and
the sparse infinite lattice is still outstanding — but `overtone-wfc` already collapses a
real corridor maze, so `overtone-graph` builds the Laplacian from the largest connected
component of one. The eigenbasis is computed on something the reader watched being built.

**M22 — the shared eigenbasis — done.** The eigenvectors of the maze Laplacian are
simultaneously proto-value functions, the eigenmodes of the quantum walk, and the graph
Fourier transform of the maze. One eigendecomposition, three fields.

**M23 — the LMDP — done.** `z = GPz` by Todorov's power iteration, the optimal control
`u*(j|i) ∝ p_ij z(j)`, shortest paths, and policy superposition.

**Exit criteria — met**
- Laplacian spectrum matches the closed form `2 - 2cos(2 pi k / n)` on a cycle to `1e-10`;
  zero eigenvalues count the connected components; the zero mode is constant.
- Diffusion conserves probability and interference conserves norm, both to `1e-9`.
- The eigensolve recovers shortest paths **exactly** — every vertex rounds to its
  breadth-first hop count, on three independent mazes. BFS on an unweighted graph is exact,
  so this is a real oracle rather than a second approximation.
- The optimal policy descends the value function at every vertex: the most likely step under
  `u*` is always onto a neighbour strictly closer to the exit, and it came from one
  eigenvector with no search.
- **Policies superpose, exactly.** `alpha z_A + beta z_B` reproduces the separately solved
  composite task to `4e-22` — a task nobody solved, solved by adding two that were.

**Corrections carried forward from the build**
- **Part V §1.3's toggle uses two different operators.** It writes `e^(-Lt)` against
  `e^(-iAt)` — Laplacian on one side, adjacency on the other — and then rests the panel's
  whole claim on only one character having changed. On a *regular* graph they agree up to an
  unobservable global phase, since `L = dI - A`; on an irregular one, which every maze is,
  the degree term is a position-dependent phase and the probabilities genuinely differ.
  Measured: identical to `1e-9` on a cycle, differing by more than `1e-3` on a maze. The
  panel uses `L` on both sides so the claim is literally true.
- **"The quantum walk spreads faster" is false without a time attached.** It leads early and
  then stops leading: it is unitary on a finite graph, so it never settles, the modes come
  back into phase, and the field returns. On a 101-vertex maze the quantum mean distance
  passes the classical one near `t = 4` and is behind again by `t = 8`. Diffusion, having
  only decaying modes, spreads monotonically to uniform.
- **A tolerance on `z` is a tolerance at the wrong scale.** With `rho = 50` on a five-vertex
  path the desirabilities run from `1` down to `e^(-200)`, so an absolute tolerance of `1e-14`
  is met after two iterations while the far end of the graph is still two hundred orders of
  magnitude from its answer. The first version reported a distance of 2 where the answer was
  4. The convergence test is on `log z`, which is the value function and has a uniform scale.
- **Part V §2.1 omits that the shortest-path reduction is a limit, and the caveat has teeth.**
  Todorov's §3 says it plainly: *"we cannot obtain the exact shortest paths by solving a
  single eigenvalue problem... set rho large enough — but not too large because exp(-rho) may
  become numerically indistinguishable from 0."* Measured, the two demands **collide**. What
  underflows is not `exp(-rho)` but `exp(-rho * diameter)`, so representability needs
  `rho * D < 745` in `f64`, while accuracy needs `rho > ~1.4 D` because the error per hop is
  about `ln(degree)/rho`. Past a diameter near 23 **no `rho` satisfies both**: on a maze of
  diameter 36 the best available `rho` still gave 2.10 hops of error, and four times it
  returned infinities.
- **That ceiling is an artifact of the representation, and removing it is a real fix.**
  Iterating the value function directly with a stable log-sum-exp,
  `v(i) = rho + ln(deg i) - log sum_j exp(-v(j))`, has no floor at all. The error then obeys
  the limit's own rate — `0.634` at `rho = 40`, `0.063` at `400`, `0.0063` at `4000`, a clean
  `1/rho` — and every distance rounds to the exact hop count. The eigenproblem framing, which
  is the interesting part, survives intact.

**M24 — the optimiser flatline — done.** Part V §5.2's demonstration, and the correction it
needs to be true. `overtone-opt` holds all four optimisers — plain gradient, natural gradient
on the QFIM, CMA-ES with Hansen's defaults, Nelder–Mead with Nelder and Mead's — and every
one is checked on the sphere and on Rosenbrock before it is allowed near a plateau, because a
broken optimiser flatlines too.

**M25 — distributional RL and the shot dial — done.** A quantile critic in `overtone-rl`
following Dabney et al., scored against the closed-form two-atom return distribution of
`SpectralControl`, plus the shot-budget dial the same shot machinery provides.

**M26 — eigenoptions and Go-Explore — done.** Machado et al.'s eigenpurposes, eigenbehaviors
and termination sets on the maze Laplacian M22 already computes, and Ecoffet et al.'s
return-then-explore against a loop-erased random walk on the same maze.

**M27 — architecture search into the Menagerie — done**, together with the verification step
Part V §10 refuses to let it ship without.

**The `Lattice` panel — done.** Proto-value function `k` over the maze, the real-versus-
imaginary exponent toggle with the spread readout beside it, and the two-slider composition
panel that prints its own error against solving the composite task from scratch.

**Exit criteria — met**
- All four optimisers reach the minimum of the sphere and of Rosenbrock before any plateau
  result is read. CMA-ES to `1e-6` on Rosenbrock, Nelder–Mead to `1e-4`.
- The finite-shot estimator is the binomial one: unbiased, with the documented variance,
  checked against 4000 draws.
- The metered parameter-shift gradient agrees with the adjoint gradient to `1e-10`.
- The quantile critic reaches the discretisation floor of its own 21-quantile representation,
  `0.0179` against a floor of `0.0174`.
- Every eigenoption terminates somewhere (Machado et al. Theorem 3.1), and running one never
  descends its own eigenvector.
- Go-Explore's archived route is within four hops of the shortest path.
- Selecting architectures on the reach beats random selection on all ten seed/size
  combinations tried.

**Corrections carried forward from M24–M27**
- **The optimiser flatline is a statement about shots, not about arithmetic.** Arrasmith et
  al. say the cost function *differences* are exponentially suppressed, so "the numbers of
  shots required in the optimization grows exponentially with the number of qubits". Written
  against `f64` expectation values every optimiser gets sixteen digits free, which is far more
  than the `2^(-1.03 n)` differences at any simulable width. Measured: with exact arithmetic
  **all four descend at every `n` from 2 to 10, on every seed**, Nelder–Mead included. The
  demonstration as Part V §5.2 describes it shows the opposite of the paper it cites.
- **"All four flatline" is too strong, and the exponents differ by more than a factor of
  two.** Shots per evaluation needed before an optimiser beats its own noise, `n = 2` to `10`:
  gradient `16 → 32768`, natural gradient `32 → 16384`, Nelder–Mead `256 → beyond 65536`,
  **CMA-ES `16 → 512`**. Fitted as `shots ~ 2^(a n)`: `a = 1.45` (`R² = 0.969`), `1.25`
  (`0.936`), `1.25` (`0.954`), and `0.60` (`0.973`). All four are exponential, which is the
  paper's result; they are not the same exponent, and a population method whose decision is a
  rank over `lambda` samples survives roughly four times longer in `n`.
- **A noisy optimiser must not be scored on the best value it saw.** That is a minimum over
  many draws, so it rewards the optimiser that evaluated most and rewards the noise itself.
  Every lane is scored on the *exact* cost at the point it stopped, computed off the meter.
  At `n = 4` with 100 shots the natural-gradient lane believed it had reached `-0.52`; the
  exact cost where it actually stopped was `+0.25`, above where it started.
- **Part V §4 is wrong that more shots sharpen the return distribution.** Two distributions
  are in play and only one depends on the shot count. The return distribution is aleatoric —
  two atoms at `±cos(ks)` — and its spread is flat at `0.68` from 10 shots to 10000. What
  sharpens is the *estimate*. What shots really buy is the **score function**: `grad log pi`
  is nonlinear in the measured `z`, so a cheap measurement biases the gradient, and that bias
  falls as `1/N` — `0.052`, `0.0029`, `0.0004`, `0.00008`.
- **Dabney et al.'s `kappa = 1` fits an expectile, not a quantile, on returns of order one.**
  The published default was chosen against Atari returns in the hundreds. Here every residual
  falls inside the Huber quadratic, so the subgradient is `u` rather than `sign(u)` and the
  fixed point moves. Measured: `kappa = 1` stalls at `W₁ = 0.328` at any sample count, while
  `kappa = 0` reaches `0.0179`. The paper's `rho^0_tau = rho_tau` also has to be special-cased,
  since reading eq. 9 literally at `kappa = 0` gives no gradient at all.
- **Machado et al. use two different Laplacians in two different sections.** §2.3 says
  plainly "the normalized graph Laplacian, which we use in this paper"; §5's sample-based
  algorithm recovers the combinatorial one, by Theorem 5.1's `T^T T = 2(D − W)`. On a regular
  graph those are one diffusion model — `‖L/d − L_norm‖∞ = 2e-16` on a cycle — and on a maze
  they are two, at `0.80`, disagreeing on the chosen action at `0.594` of states.
- **Do not compare eigenvector-derived options on a graph with a degenerate spectrum.** The
  first control here was a 24-cycle, whose Laplacian eigenvalues come in pairs; inside a
  degenerate eigenspace any rotation is a valid basis, so two solvers differ for reasons that
  are not about the graph. The maze's low spectrum is simple to `3.8e-3`, which is why the
  disagreement there is real. The matrix-level statement needs no such caveat and is the one
  the test makes.
- **An option's terminate action has value exactly zero, so the comparison against it needs a
  tolerance at the scale of the eigenvector.** Without one the *constant* mode, whose
  eigenpurpose is identically zero, acquires an initiation set of 94 states on a 101-vertex
  maze built entirely out of `1e-17`.
- **Go-Explore does not beat an undirected walk at every size, and the crossover is the
  budget.** Loop-erasing both trajectories so the comparison is between two post-processed
  answers: at 101 vertices they tie; at 962 the archive finds the goal 21/21 against 12/21;
  at 2751 it finds it **2/21 against 8/21**. At four times the budget on that same maze it is
  21/21 against 20/21. Return-then-explore spends its opening steps building an archive and
  only then aims, so Part V §6's claim holds in a window set by how many steps the agent gets.
- **Part V §3's algebraic reward is beaten by its own reach term.** The verification §10
  demands, over 200 architectures trained to convergence on the exact policy gradient:
  reach `rho = 0.618`, the covers-`k` indicator `0.581`, **the gate-count penalty `-0.486` —
  the wrong sign** — and the `dim(g)` indicator `-0.060`. Gates track layers and layers track
  the frequency ceiling, so charging for gates charges for reach. Selecting the top 20 by the
  three-term reward is indistinguishable from random (five seeds above, five below); selecting
  on reach alone beats random on all ten and beats the combined reward on nine.
- **The `dim(g)` term cannot be validated on a problem small enough to validate it on.**
  It is a *trainability* proxy, and two to five qubits with an exact analytic gradient has no
  barren plateau to be saved from. That is the same reason M19's fitness-by-`dim(g)` profile
  came out flat and the same reason M24's flatline needed shot noise to appear at all. The
  conclusion is not that `dim(g)` fails to predict trainability; it is that this environment
  cannot test the claim, and a term nobody can validate should not carry a weight.

**Outstanding**
- Nothing in Part V. Phase 8 is complete.

---

## Phase 9 — Orbit: the game, headless  *(M34–M37)*  — DONE

**Spec:** Part VII. **Gated on Phases 5, 6 and 8.** **No interface is built in this phase.**

### Part VII supersedes Part VI, and the reason matters

Part VI proposed a free-form real-time arena. Part VII proposes a turn-based one and says
plainly: **do not build both.** They are two designs for one tab, and shipping both would
produce a project that cannot say what it is. `Braid` is renamed `Orbit`; the braid diagram
survives as a panel.

The argument for the replacement is the sharpest correction in the series, and it lands on
work already done. Part VII §0 scores the endless maze honestly as **enormous space
complexity and near-zero decision complexity** — which is Snakes and Ladders, not chess.
Adding maze adds states, and states are the cheap axis. *Endlessness is a liability wearing
the costume of a feature.*

That is not a repudiation of Phase 7. Part IV's worlds move the **transport exponent**,
which changes which strategies work — that is decision complexity, not state count, and the
substrate is one of the knobs §8 says to tune. What the correction kills is the *framing*:
"endless" was never the selling point, and it should stop being described as one.

Nothing else is lost. Every Part VI mechanic survives inside Part VII's rules as **content**
rather than as rules — exchange statistics decide what overlap does (VI §1), the decoherence
front drives the clock (VI §2), braiding stays topologically protected (VI §4), and the
eight traps become terrain (VI-A). The turn structure did not replace them; it made them
legible.

### The rules, entire

```
1.  You are an amplitude field on the maze.
2.  You hold a hand of generators. Together they close into your algebra g.
3.  Each turn: apply one generator to one region, or measure.
4.  Applied generators evolve for k coherent steps, then the turn passes.
5.  Coherence only decreases. dim(g) only increases.
6.  Overlapping an opponent merges algebras: g <- closure(g_you u g_them).
7.  You lose when no unitary in g reaches a safe state.
```

Seven lines. **If a rule needs a paragraph, cut it.**

---

**M34 — checkmate — done.** `overtone-orbit` computes the reachable-orbit certificate and
the check and checkmate predicates. Check is exact: it is an expectation value. Checkmate is
**sound and not complete**, and the shape of that gap turned out to be the whole story.

**M35 — the strategy ladder — done, and it caught a design fault before any interface.**
Generator basis, turn loop, and `d`, all headless.

**M36 — the endgame eigensolve — done.** `Tablebase::solve` reuses Phase 8's `Lmdp`.

**M37 — the complexity dial — done.** Two independent representations, `g`-sim and MPS, with
the verdict written as "hard only when **both** fail" rather than as one threshold.

**Exit criteria — met, with one qualified**
- Soundness is exact: across every algebra, width and position family tested, the certificate
  never contradicted the brute-force search. Zero false checkmates.
- The brute-force oracle finds 8 of 8 paths that exist by construction, so a completeness
  number measured against it is measuring the certificate rather than the search.
- Branching factor **30 at `n = 4`**, inside Part VII §9.2's 25–40 band by construction, and
  reported as a measured statistic. `n = 3` gives 18 and `n = 5` gives 45, both outside it.
- `d = 6` at `n = 4`, against a bar of 4. **Qualified:** the curve is *not* still rising at
  the top of the ladder — see the corrections below for why that is a fact about the strategy
  language rather than about the game.
- The endgame tablebase matches breadth-first optimal play exactly on three independent
  mazes, and solves a 16×12 maze in **under 100 ms**.
- One session moves a position from "`g`-sim and MPS both work" to "neither has an efficient
  description", which is Part VII §6's acceptance.

**Corrections carried forward from the build**
- **"Decidable from `g`, computable in milliseconds" is false as stated, and the certificate
  is provably incomplete by counting.** `exp(g)` is compact, so the ring of invariant
  polynomials separates orbits and is finitely generated (Hilbert–Nagata) — a complete
  certificate exists *in principle*. What is cheap is the degree-≤2 truncation: the commutant
  (exactly conserved, linear) and the per-ideal `g`-purity (quadratic). Truncating leaves it
  sound and not complete, and the gap is a dimension count, not bad luck. Pure states of `n`
  qubits form a manifold of dimension `2·2^n − 2`; fixing `k` invariants cuts it to
  `2·2^n − 2 − k`; the orbit inside has dimension at most `dim(g)`. Measured deficits: local
  X at `n = 4` gives `30 − 8 − 4 = 18`, so the level set holds a **continuum** of distinct
  orbits. TFIM at `n = 3` gives exactly `0` — and there the certificate *is* complete.
- **Measuring completeness on two Haar-random states reports 1.000 for a certificate that is
  provably not complete.** Two random states disagree on essentially every invariant, so the
  certificate fires trivially. On pairs *constructed to share* the invariants it is **0.000**.
  The two numbers differ by the whole of the effect, and only the second is a test.
- **But the game never visits that region, and that is what makes the predicate playable.**
  A safe region in Orbit is a set of maze cells — computational basis states — not an
  adversarially chosen point on an invariant level set, and an opponent cannot choose it
  because it is the maze's geometry. Measured against basis-state safety: **completeness
  1.000** at every algebra and both widths, with zero soundness violations. So Part VII §11's
  "verify the predicate is exactly correct" holds for the position distribution the game
  produces, and fails as a general claim. Both halves are reported.
- **`d = 0` with every win rate at exactly 0.500 is not a flat ladder, it is an absent
  opponent.** The first turn loop gave both players a *fixed* safe subspace, so neither move
  affected the other and the game was two solitaires: seat 0 won 20 of 20 at every budget from
  1 to 128, and colour-swapping averaged that to exactly 0.500 on every rung. Part VII §3
  says check is overlap with **the pursuer's** absorbing subspace, and the pursuer is the
  opponent — the safe set has to be computed from the opponent's amplitude field. With that
  coupling restored, `d = 6` at `n = 4`. The lesson generalises: an exactly-0.500 ladder is a
  bug signature, not a measurement.
- **A pursuer spread uniformly absorbs nothing.** The threshold is `1/dim`, so "the pursuer is
  here more than a uniformly spread field would be" — and a field thinned across the whole
  window threatens no particular cell. A test asserting the opposite was wrong about the
  physics, not about the code.
- **Reusing `recommended_rho` in the endgame reintroduced the bug that log space had
  removed.** That function caps `rho` at 60 to keep `exp(−ρD)` above the `f64` floor, which is
  a constraint of the *`z`-space* solve. `shortest_paths_stable` works in log space where
  there is no floor, so the cap only costs accuracy: at `ρ = 60` on a diameter-36 maze the
  error is still near 0.5, and a true distance of 14 came back as 15. `ρ` is now chosen for
  accuracy alone, `40·D`. A safety cap carried across a representation change is worth
  checking whenever the representation is what made it unnecessary.
- **Cantwell's Quantum Chess: the claim is confirmed, the attribution was not.** The full
  paper does replace checkmate with king capture — rule 5, "There is no concept of check or
  checkmate. Kings are captured like any other piece", with rule 10 awarding the win at zero
  king probability. Two things this plan previously got wrong: the paper **has no abstract**
  (it opens directly into §1), so a quotation attributed to one cannot be right; and the
  sentence in question is in the **conclusion** and says the game "remains **simulable**", not
  "tractable for a classical computer". The mechanism is also not a measurement rule as such
  but rule 2 — no square may ever have non-zero probability of holding two pieces — enforced
  by projective measurements built for it. Part VII inverts *that constraint*, not measurement
  in general.

**Outstanding**
- **The ladder saturates because of the language, and the saturation point scales with it —
  but not the way first predicted.** A depth-1 policy has only `|legal moves| × |angle grid|`
  distinct candidates: 240 at `n = 4`, 360 at `n = 5`. The last rung clearing the step unit
  moves from **64 to 128** as the candidate set grows, which is the evidence that the flatness
  is a property of the language and not of the game — Lantz et al.'s correction 2 exactly.
  The prediction that saturation would arrive *at* the candidate count was wrong: it arrives
  at roughly a quarter of it, because sampling `b` of `N` candidates already lands near the
  top `1/b` quantile, so returns diminish long before enumeration. Establishing whether the
  *game* has more depth needs a richer language — deeper lookahead — not a different game.
  (The `d` values from the two ladders are not comparable: the `d = 6` figure used budgets
  1–128 over 40 games per rung, the scaling check used 16–256 over 12. A `d` is only ever
  quotable against its own settings, which is correction 2 again.)
- Phase 10 remains gated. `d = 6` clears the bar of 4, but the "still rising at the top"
  half of the criterion is not met under this language, and that is the honest reading.

## Phase 10 — Orbit: the board  *(M38, M39)*

**Spec:** Part VII §5, §7, §11. **Gated on Phase 9's measured `d`.** If the ladder is flat,
this phase does not start.

**M38 — the board.** The interface, the windowed arena, the generator hand, and every Part VI
and VI-A mechanic folded in as content.

The move set is a **hand of six to eight generator types**; each turn picks one generator and
one target region. Six to eight, and no more — chess has six piece types and a library of
literature, and rule count and depth are close to unrelated.

The piece mapping is not an analogy. A chess piece *is* a movement geometry, which is the
orbit traced by one generator:

| Piece | What actually defines it | Generator |
|---|---|---|
| Pawn | one step forward, cannot reverse | one coherent step; spends coherence |
| **Bishop** | **cannot change square colour — a conserved `Z2`** | generator confined to a symmetry sector |
| Knight | jumps, ignoring what lies between | long-range hop coupling non-adjacent cells |
| Rook | acts along a whole line | phase gate along a line |
| **Queen** | rook ∪ bishop, and worth more than both | the **commutator closure** of two generators |
| King | the thing that must survive | the protected subspace |

The bishop is the insight: being trapped on one square colour forever is a **superselection
sector**, and chess has had a symmetry-sector piece since the fifteenth century. The queen is
the other one: "a queen is worth more than a rook plus a bishop" is chess folklore for the
**non-additivity of the Lie closure**, and Part III computes `[G_rook, G_bishop]` in
milliseconds. Chess players have had an intuition for non-additive Lie closure for five
hundred years; this can show them the algebra behind it.

**Scale, the honest constraint.** Two agents' joint state on an infinite maze is not
representable. Three layers of mitigation: the light cone bounds support to radius `t`
(Part II §3), turns are short so `t` stays small, and g-sim handles polynomial `dim(g)` at
high qubit count. Concretely, **play inside a windowed region of about 32x32 while the endless
maze scrolls around it** — a local engagement inside an endless world, which is how chess
works too: an 8x8 window on an infinite space of possible games.

**M39 — the ladder published.** **Restated, because the original is not achievable as
written.** Part VII §8 asks for `d` plotted against chess, Go, checkers and tic-tac-toe on
the same axes. Those numbers do not exist — see M35 — and for chess and Go the metric's own
authors say the complete model cannot be applied. What is achievable, and what this milestone
now means:

- **Overtone's own `d`**, with the strategy language, CR ladder and step unit declared in the
  figure caption.
- **Tic-tac-toe as a floor calibration**, computed by us. Perfect play is known, so its ladder
  is a genuine `d` and it should come out at one or two steps. That is the point of including
  it: it calibrates the axis and demonstrates the instrument reports *small* correctly.
- **Chess as an informal comparator**, from published engine Elo-versus-compute data, plotted
  on clearly separate axes and labelled as a different measurement. Not `d`.
- The caveat that `d` is language-relative, in the caption, every time.

A repository that measures its own game's depth on a published metric — and says plainly
which parts of the comparison are not apples to apples — is something that does not exist.
**Report the number even if it is disappointing.** Part VII §12: the entire credibility of
this part rests on that.

---

## Phase 11 — Correspondence: the artifact, the notation, and the ladder  *(M40–M44)*

**Spec:** Part VIII. **Depends on Part VII.**

> **Chess became a global competitive game three centuries before servers existed. It scaled
> through notation.**

Algebraic notation, then PGN and FEN. The entire chess ecosystem runs on a file format.
**Overtone needs a notation, not a server** — and it is unusually well placed for one,
because Part I §5 already mandates that every run be deterministic and reproducible from a
seed, natively and in WASM.

**M40 — the puzzle set.** `Overtone-100`: a hundred curated positions — reachability puzzles,
checkmate-in-`n`, spectral traps, cage escapes, endgame conversions — each with a solution
that is **proven, not agreed**, because the M36 eigensolve gives exact optimal play once the
walker decoheres.

Part VIII §8 recommends building this before the league and the reasoning is sound: it works
with zero participants, it has ground truth where a league never does, benchmarks get cited
where leaderboards go quiet after a month, and it gives every future submission an absolute
score alongside a relative one. Acceptance: every solution verified against brute force on
small instances.

**M41 — the notation.** `.otn`, frozen and versioned, with a reference parser and a
replay-compatibility test in CI.

```
header:  engine version, ruleset version
seed:    maze seed + substrate rule           ~16 bytes
agents:  two declarative specs                ~200 bytes each
moves:   generator index + target region      ~2 bytes per ply
result:  outcome, final report, state hash
```

A few hundred bytes for a whole game. Human-readable, self-verifying via the final-state
hash, and version-pinned in the header because physics changes break replays.
**This format is the actual deliverable of Part VIII**; everything after it is convenience.

**Timing, against a tension in the spec.** Part VIII §9 says freeze the notation early
because "a format that keeps changing is a format nobody builds on". Phase 9's M35
*deliberately retunes* the generator basis, coherence budget and `k`. Those cannot both
happen. The resolution: the notation freezes **after M35's retuning settles**, and the
ruleset version in the header is what absorbs anything later.

**M42 — Tier 0, URL correspondence.** You move, you get a URL, you send it. The fragment
carries the whole match because the match is a few hundred bytes. Play-by-mail chess: works
offline, works forever, costs nothing, and Part IV §5.2's permalinks already implement most
of it. **Zero server.**

**M43 — Tier 1, GitHub is the server.** A submission is a pull request adding
`agents/<name>.toml`; CI validates it against a schema; a nightly job runs the ladder;
`LEADERBOARD.md` is committed back and the match archive goes to a Hugging Face Dataset.

**Verified against GitHub's own documentation on 2026-09-07**, since Part VIII §12 says to
re-check before relying on it: Actions usage is *"free for self-hosted runners and for public
repositories that use standard GitHub-hosted runners"*, and *"Larger runners are always
charged for, even when used by public repositories."* The limits are **6 hours per job**,
**20 concurrent jobs on Free**, **256 jobs per matrix per workflow run**, and 35 days per
workflow run. That is an entire tournament infrastructure, free, for a public repo — and it
arrives with identity, moderation, version history and notifications already built. Re-verify
before launch regardless; this is a release-checklist item, not a fact to trust.

**Agents are declarative, never executable.** A submission is a spec — generator names,
statistics, trained coin weights as *data*, and a policy drawn from a fixed published
vocabulary. This **eliminates arbitrary code execution rather than sandboxing it**, keeps
matches deterministic and verifiable, and makes the competition about physics design rather
than software engineering. A submission is a hypothesis about which algebra wins.

**Cold start is already solved** — and this un-defers a Phase 7 decision. Part VIII §4 seeds
the league with a hundred Menagerie elites spanning `(dim g, beta, chi)`, which is **M19, the
MAP-Elites archive, deferred out of Phase 7**. It is no longer optional: M43 depends on it,
so M19 lands here if it has not landed sooner.

**M44 — the ladder published.**

> **The Elo ladder over submitted agents *is* the strategy ladder. Its length is `d`.**

This is the reason the league belongs in the project at all. A population of
externally-designed agents is a better `d` measurement than self-play: self-play measures how
well the game resists one optimiser, a league measures how well it resists many independent
minds. **Call it the ladder, not the leaderboard**, consistently.

**But the quoted sentence is not Lantz's definition, and M44 must not report the league's Elo
range as `d`.** Lantz et al. draw exactly this distinction and come down the other way. A
*skill chain* is "a sequence of human players of ascending skill"; a *strategy ladder* is "a
sequence of algorithms called strategies" ordered by **CR-levels**, the computational
resources each needs to run — and the paper gives four reasons for studying algorithms rather
than players, including that "the distribution of possible strategies is likely different than
those used by human players influenced by community, opponents, and conventions." It also
requires that "one must first specify a language in which each of these algorithms is
expressed", because "any observations made about a game's depth based on this model must refer
to the language selected." A population of independently-designed submitted agents has no
single language and no resource axis, so its Elo range is the skill-chain shape, not `d`.

There is a second reason to distrust win rate here specifically. The paper warns that "games
with random elements have a looser correlation between strategic decisions and game outcome,
which can obfuscate the signal win rate gives us about strategic strength" — and Overtone has
the Born rule. Phase 9's `d = 6` is already measured this way. The paper's own preferred
alternative is quality-of-move against optimal play, which Overtone can actually compute:
M36's endgame tablebase is exact. **M44 reports two numbers: the league's Elo spread, labelled
as a skill chain, and `d` measured the way Phase 9 measures it, against a declared language.**

**The complexity dial controls verifiability too**, which is the genuinely novel part:

```
low  dim(g), low  chi  ->  every match trustlessly verifiable; CI re-runs everything
high dim(g), high chi  ->  VERIFYING a match is as hard as PLAYING it; sampling and trust
```

The official ladder runs in the verifiable regime; an **open division** runs above it,
labelled as sampled rather than verified. Do not hide the distinction — "here is where
verification becomes intractable" is one of the more interesting sentences the project can
say, and it connects to classical verification of quantum computation (Mahadev, FOCS 2018).

**Design for fifty, not millions.** Round-robin at 50 agents is 1,225 matches, trivial in one
nightly job; at 200 switch to Swiss pairing. Fifty independently-designed agents is a better
`d` measurement than a thousand near-identical ones.

**M45 — Tier 2, WebRTC rooms — deferred, deliberately.** Two browsers can play directly over
a data channel with no server, using serverless matchmaking such as `dmotz/trystero`. Part
VIII §11 says build it only if Tier 0 or Tier 1 produces people asking for it: live rooms are
the most fun to build and the least likely to be needed. **Tier 3, an actual server, is a
no.** It means uptime, cost, auth, abuse, moderation, and a single point of failure for a
project whose most valuable property is that it always works.

---

## Phase 12 — Thermograph: move ordering nobody tuned  *(M46–M52)*  — DONE except the two gated panels

**Spec:** Part IX. **Entry:** Phase 9's `overtone-orbit`, for the positions temperature is
measured on. **Exit:** eight gate lines, below.

Part IX arrived as three separate claims and they did not survive equally. The dice idea was
already refuted in the document itself; the Chinese-rings instinct produced the project's
sharpest counterexample; and the move-generation question turned out to have a real answer in
combinatorial game theory. What follows is what was built and what the measurements changed.

**M48 — temperature.** New crate `overtone-cgt`: short games, thermography from scratch, the
temperature at the base of the mast, and decomposition search over disjunctive sums.

Part IX §7 sets the acceptance test as "reproduce a worked Go endgame temperature from
Berlekamp & Wolfe as a unit test", and §8 makes it a trap: if it cannot, the heat map is
decoration. The family used is the **closed empty corridor**, whose analysis Berlekamp & Wolfe
state as a recursion with no board in it:

```
Corr(0) = 0,   Corr(n+1) = { n | Corr(n) },   f(Corr(n)) = n - 2 + (1/2)^(n-1)
```

`f` is chilling by one, and a chilled corridor is a number, so the published `f` is the mean
value — which the crate computes independently from the walls. It matches for `n = 1..20`
under `assert_eq!` on `f64`, **with no tolerance**: every quantity in temperature theory is a
dyadic rational, and dyadic rationals below `2^52` are exact in binary floating point. The
temperature that falls out alongside it is `1 - (1/2)^(n-1)`, below 1 for every `n`, which is
why chilling by one lands the corridor on a number at all.

**The correction: CGT is defined for games of no chance.** Part IX §5 was written as though
thermography applied to Overtone directly. Berlekamp's own survey opens: *"In its broadest
sense, Combinatorial Game Theory (CGT) is the study of two-person, perfect information games
of no chance."* Overtone has chance — rule 3 lets a player measure, and the Born rule is a
chance node. So temperature is defined on the **coherent segment**, the run of unitary turns
between one measurement and the next, and is undefined across a collapse. That is not a
workaround; it is the honest domain of the tool, and it is the interval a player is actually
reasoning about when deciding where to move.

**The second correction: "optimal play is to move in the hottest region" is false.** Part IX
§5.1 states it as a theorem. An exhaustive search over disjunctive sums finds a two-component
counterexample with **distinct** temperatures — so it is not a gap about ties:

```
{0 | -3}  +  {{1 | -2} | -3}      Left to move
temperatures 1.5 and 1.0, hottest is the first
optimal stop -2, hottest-first -3, loss 1 point
```

The mechanism is visible in the position: the colder component's *left option* is a switch of
temperature 1.5, hotter than the component itself. Greedy ordering assumes the heat it can see
is all the heat there is. The rule is not worthless — on sums of plain switches the search
finds no disagreement at all, which is the case the theory actually covers — so what Overtone
gets from §5.2 is a derived move *ordering*, not a proof of optimality, and the docs say so.

Two filters were needed before that result was real, and both came from wrong answers. The
first search returned `{{-1|-2} | 1}`, which **is a number** by the simplicity rule, because
every left option is strictly below every right option even though no option list is
all-numeric; the crate reports a meaningless temperature for those by design. `Game::is_hot`
is the soundness predicate that fixed it, and `Game::new` now panics on the all-numeric case.
The second filter is distinct temperatures, since the rule says nothing about ties.

**M47 — the Chinese Rings figure.** `overtone-graph::rings`. The state graph is built from the
puzzle's two move rules and everything else is measured off it: it is a path on `2^n` vertices
with exactly two endpoints and no vertex of degree above 2; the Gray code `G(i) = i XOR (i>>1)`
orders it; and the distance from all-on to all-off is `A000975` — 1, 2, 5, 10, 21, 42, 85, …
— recovered three ways, from the closed form, from BFS on the graph, and as the Gray index of
the all-on state.

```
substrate               states   branching   diameter
4x4 maze                    16       3.000          6
7 rings                    128       1.984        127
Pauli hypercube q=3         64       6.000          6
```

That is the figure. **Difficulty does not come from the number of options.** A seven-ring
puzzle branches 1.98 ways and still takes 85 moves; a 4×4 maze branches 3 ways and is six deep.

Two things Part IX §3 gets wrong and the module records instead. The puzzle is **not** two
thousand years old: the Zhuge Liang attribution traces to Stewart Culin relying on an unnamed
informant, and the earliest definitive references are Yang Shen's *Sheng an ji* (early 16th c.)
and Pacioli's *De Viribus Quantitatis* (1509), with Cardano's *De subtilitate* (1550) giving
it the name "Cardan's rings". So it is also not "older than algebra" — it postdates
al-Khwarizmi by seven centuries. The argument does not need the date. Separately, the solution
does **not** traverse the whole path: it is about two-thirds of it, converging to exactly 2/3,
because the solved state is not at the far end.

**M52 — the Pauli-string substrate.** `pauli_hypercube(q)`: cells are Pauli strings on `q`
qubits, legal moves flip one generator, order `4^q`, `2q`-regular, diameter `2q`, and a Gray
code still walks it — now as one Hamiltonian path among many rather than the only route.
Part IX §3.1's observation that Part III has been carrying a hypercube around without walking
on it, made concrete.

**M50 — advantage.** `overtone-orbit::advantage`. Grover iterations as a player-facing dial:
the simulator matches `sin^2((2k+1) theta)` to `1e-9` at every setting from 3 to 9 qubits
without being told the formula, and the simulated argmax lands exactly on the predicted
`round(pi/(4 theta) - 1/2)`.

The comparison with D&D is quantified rather than asserted. One Grover iteration multiplies a
long shot's probability by about **nine**; advantage — roll two dice, take the better —
multiplies it by about **two**, so the equivalent setting of the dial is `k = 1` across the
whole range tested. At its optimum the dial reaches `> 0.999` where advantage from the same
base reaches `0.002`.

And the asymmetry that matters: **a die has a floor and the dial does not.** Disadvantage can
do no worse than squaring the probability. Over-rotation drops below the un-amplified base
rate within one turn past the optimum, and keeping the dial turning falls below what a
disadvantaged die could ever reach. How far you must turn is erratic — between 1 and 62 turns
across 4 to 14 qubits — because it asks how well odd multiples of `theta` approximate multiples
of `pi`, which is an equidistribution question and not a monotone function of register size.
Part VI-A §T1's soufflé, with a number on it.

**M51 — two temperatures.** `overtone-orbit::thermal`. Both quantities are derived, not
invented: a region's CGT temperature is `(a - b)/2` where `a` and `b` are the game's **own**
score — Part VII's `opponent absorbed weight minus own absorbed weight` — after the best move
confined to that region by each side; the physical quantity is the half-chain von Neumann
entropy from `overtone-spec`.

**The first answer was wrong and the correction is the result.** Eight seeds gave a mean rank
correlation of 0.49 with the same sign every time, which looks like a finding. It is not. Both
series climb over the course of a game — temperature against ply at rho 0.76, entropy against
ply at 0.46 — so the correlation between them is mostly a correlation with time. With the ply
partialled out and the seed count doubled:

```
raw rho        mean  0.344   range [-0.089, 0.813]   sign not consistent
partial rho    mean -0.021   range [-0.558,  0.500]  sign mixed 8/15
```

**They share a name. They do not share a behaviour.** Part IX §8's trap said to measure it and
report what is found rather than assume; this is what was found. The panel in §5.4 is worth
building for exactly that reason — a null result about a metaphor is still the project's
characteristic move — but it must be labelled as one.

A second measurement went the other way and supports the spec. Part IX §5.3 assumes the
position splits into weakly-interacting regions; the **interaction leak** — the gap between
the best move over the whole position and the best found one region at a time — is `0.0000`
across every seed tested. Decomposition is exact here, not merely adequate.

**M49 — the heat map.** Gated with Phase 10, but its risky half is measured now. Part IX §7
sets the bar at "temperature updates live at 60fps on a 32×32 window", and a thousand
thermographs turn out to cost **5.6% of a 60fps frame** at the depth that matters (the depth
where hottest-first fails, so the depth the panel has to afford). Thermography is not the
bottleneck and M49 does not depend on making it faster. The budget is spent building each
region's game, which in Overtone means evaluating positions, and one `Position::new` closes an
algebra: 16.3 µs per cell is the number to beat, or the field must be cached and updated
incrementally the way a Zobrist-hashed transposition table updates a chess evaluation.

**M46 — the measurement beat — is gated with Phase 10, and this is not a scope cut.** It is an
interface change to a game that has no board yet: Phase 10 is still blocked on "still rising at
the top of the ladder", and the JS budget stands at 1186 of 1200 lines with CLAUDE.md's
standing instruction to move code into Rust rather than raise it. Building the beat now would
mean building the board now, which is the thing Part VII §12 and this file's third governing
constraint exist to prevent. The physics it renders is already there and has been since M1.

### Exit criteria — met

- `corridor reproduces Berlekamp & Wolfe` — 4 tests, exact `f64` equality, `n = 1..20`
- `hottest-first is not optimal` — a 1-point loss at distinct temperatures
- `hottest-first IS optimal on plain switches` — the boundary of the rule, both sides of it
- `32x32 temperature field fits a frame` — 5.6% of 16.7 ms
- `seven rings: branching 2, 85 moves`
- `rings state graph is a path` — 128 states, 127 edges, 2 endpoints, 85 moves
- `two temperatures do not track each other` — partial rho -0.021, sign mixed
- `decomposition does not leak` — 0.0000

### Deferred from Part IX, deliberately

- **M46** and **M49**'s rendering: gated with Phase 10, above.
- **Zobrist hashing** (§6). The observation is right — a Pauli bitboard and a Zobrist key are
  the same XOR one level apart — but nothing in the workspace is yet slow because of repeated
  position evaluation. It becomes real work the moment M49's per-cell budget binds, and the
  measurement above says exactly when that is.

---

## Phase 13 — Statistics, the notation, and the benchmark  *(M28, M40, M41)*  — DONE

**Spec:** Part VI §1, Part VIII §1, §8, §10. **Entry:** Phase 9's `overtone-orbit` and Phase 7's
`overtone-walk`. **Exit:** the gate lines below.

**M28 — two walkers, and the class system that follows.** `overtone-walk::two`. Part VI §1
claims an agent's class *is* its exchange statistics and that combat is what happens when two
amplitude fields overlap. Part VI §8 sets the stakes: if the published patterns do not
reproduce, "the arena's physics is wrong and everything built on it is theatre."

Sansoni et al.'s Eq. (1) is implemented directly —
`A_KL = U_IK U_JL + e^{i phi} U_IL U_JK` — with `phi = 0` bosonic, `pi` fermionic, and the
generic phase anyonic. Measured, on a 4-step Hadamard walk with both walkers entering one
site in opposite coins:

```
statistics   phi/pi   mode diagonal   position diagonal
bosonic        0.00        0.187500            0.437500
anyonic        0.25        0.160041            0.387159
anyonic        0.50        0.093750            0.265625
anyonic        0.75        0.027459            0.144091
fermionic      1.00        0.000000            0.093750
```

Bunching falls monotonically with `phi`, the endpoints are the two named statistics, and
`similarity(bosonic, fermionic) = 0.771` — three genuinely different distributions, so the
class system is not decorative.

**The correction: Part VI §1's fermionic row is wrong.** It says a fermionic agent "cannot be
entered — occupying a corridor blocks it." Look at the two columns above. The fermionic
**mode** diagonal is exactly zero; the fermionic **position** diagonal is `0.09375`. Sansoni
et al. say why in as many words — "some of the diagonal elements of the fermionic two-particle
walk are nonzero" — because a coined walk carries a site *and* a coin, and their Eq. (4) state
`(|j,U> - |j,D>)/sqrt 2` is antisymmetric while sharing a site. **Pauli exclusion forbids two
fermions in the same mode, not on the same site.** A fermionic agent blocks one coin state in
a corridor and leaves the other open, which is a real defensive property and a smaller one
than the spec claims.

Exclusion is implemented **exactly**: `Statistics::phase_factor` returns a literal `-1` rather
than `expi(PI)`, because `expi(PI)` carries an imaginary part of `1.2e-16` that leaves the
fermionic diagonal at `1e-34` instead of at zero. The test is then an equality, not a
tolerance somebody has to justify.

**A trap worth keeping.** The first version of the experiment put the walkers on *adjacent*
sites and every statistic gave the identical distribution. A coined walk preserves the parity
of `site + step`, so walkers one site apart occupy disjoint sublattices forever, every
exchange term is zero against a non-zero direct term, and a test built that way passes while
measuring nothing. It is pinned as its own test.

**On the citation, against an audit.** An external prior-art audit recorded Part VI §1's
Sansoni citation as CONTRADICTED — "they demonstrated bosonic bunching and fermionic
antibunching only" — and asked for the anyonic claim to be recited to van Exter et al. The
paper's body says otherwise, verbatim: *"we therefore prepared some anyonic states |Psi_phi>,
in particular with phi = pi/4, pi/2, 3pi/4, and measured the output probabilities"*, with
Fig. 4(c) captioned "anyonic (with phi = pi/2)". The audit's error is explicable — the
*abstract* names only bosons and fermions — and the general rule it teaches is worth keeping:
**an abstract is not a source.** The audit's substantive point survives and is kept: Sansoni
et al. *simulate* exchange with photon polarisation rather than producing anyons, so van
Exter, Nienhuis & Woerdman (PRA 85, 033823) is cited alongside, not instead.

**M41 — the notation.** `overtone-otn`. Part VIII §12 says to read PGN and FEN first; four
decisions came straight out of them. A mandatory header roster with free extension around it
(PGN's Seven Tag Roster). A permissive **import** format and a byte-exact **export** format,
because PGN requires that two programs' output be "exactly equivalent, byte for byte". A `%`
escape line for private data, same character and same meaning. And FEN's real lesson —
*record what the engine needs to resume, not what a reader could infer* — which is why an
`.otn` position is a seed plus a move list rather than a dump of `2^n` amplitudes.

Two traps the format had to design around, both load-bearing:

- **`DefaultHasher` cannot back a self-verifying format.** Part VIII §1 wants a hash so that
  re-running is a proof; Rust's default hasher is documented as unstable across releases, so
  a file written by one toolchain would stop verifying under another. FNV-1a is written out.
- **Hashing raw `f64` bits would make the proof platform-dependent.** This repo's own
  `wasm_determinism.sh` measures native-vs-wasm agreement at `5.6e-16` — a tolerance, not
  bit-equality — so a raw-bit hash would disagree between a native and a browser replay of the
  same game, which is the exact case the hash exists to catch. Amplitudes are quantised to
  `1e-9` first: four orders above the asserted tolerance, far below anything playable.

**M40 — `Overtone-100`.** Part VIII §8 argues the benchmark is the better artifact and should
come first, and Part VIII §10 sets the bar: every solution verified against brute force.
100 positions, all 100 answers re-derived by an independent route, `246` bytes each:

```
conversion    40   hop counts 5..9, eigensolve checked against BFS
escape-in-1   30   6..12 escaping moves each, found exhaustively
reachability  30   15 reachable, 15 not
```

Only categories with a *provable* answer are in the set. Part VI-A's spectral traps and cage
escapes are deliberately excluded: their ground truth would come from the same certificate the
puzzle is meant to test, and a benchmark whose answer key is the system under test is
worthless. They wait for an independent oracle.

**Two generator bugs the acceptance criterion caught**, both worth recording because both
would have shipped a benchmark that measured nothing:

- The first mate category asked whether a move made *the mover* checkmate, which is self-mate.
  It produced zero positions and the set-size assertion caught it. Replaced with escape-in-one,
  which is well posed: in check now, fully safe after exactly one move.
- The first reachability category came out **27 reachable to 3** — a category a solver scores
  90% on by answering the same thing every time. Excluding the start cell from the safe set
  puts every position in check so the algebra decides, and the split is now 15/15. `balance()`
  reports it and a test fails if the skew returns.

### Exit criteria — met

- `pauli exclusion is exact in the mode basis` — identically zero, not a tolerance
- `two fermions share a site` — position diagonal `0.09375`, the Part VI §1 correction
- `anyonic phase interpolates` — monotone across `0, pi/4, pi/2, 3pi/4, pi`
- `adjacent inputs cannot interfere` — the parity trap, pinned
- `otn round-trips byte-exactly` — writing what was read is a fixed point
- `every roster tag is required`, `a future version is refused rather than guessed`
- `Overtone-100 all-verified` — 100/100 re-derived independently
- `no category can be guessed` — reachability skew below 0.34

---

## Phase 14 — The walk operator  *(M8, in the order Decisions-03 revised it)*  — DONE

**Spec:** Part II §§5, 6, and Decisions-03 Q7, which supersedes both.

This is Phase 5 material built out of order. The milestone is **M8 — the special graphs**, and
Q7 changed what it contains, how much of it there is, and which of its three acceptance
clauses can still be met.

### One engine, not three

The plan carried `CoinedWalkEngine`, `SzegedyWalkEngine` and a glued-trees walk as separate
objects. Q7.2 deletes the second outright: with a Grover coin and a flip-flop shift, two
coined applications *are* one Szegedy application (Wong 2016; Portugal & Segawa 2017), and the
bipartite double cover is a representation device rather than a second walk. What shipped is
one `Coined` over the arcs of any graph, with the coin taken **per vertex** because the welded
tree is 3-regular everywhere except its two roots, which have degree 2.

```
overtone-walk::coined     U = S C, arc-indexed, variable degree, Grover or DFT coin
overtone-walk::families   hypercube(n), welded_tree(n, seed), classical_hitting(n)
overtone-walk::reduced    Line -- the reduced walk both families collapse to
overtone-walk::hitting    one-shot, concurrent, residual
```

### The reduction is the panel

Part II §5 asked for a panel in which the exponentially large graph collapses to a line in
front of the reader. Q7.5 points out that this stopped being a visualisation choice: both
families confine the walk to a subspace of dimension *linear* in the parameter, and the demo
is that subspace.

```
  n    vertices       arcs    reduced   family
 10        4094      12280         42   welded tree
 20     1048576   20971520         40   hypercube
```

**The two reductions have the same shape, and that was not obvious in advance.** Li, Li and
Luo's `M_U = M_S M_C` (Lemma 3.1) and Krovi and Brun's Hamming-weight reduction of the
hypercube (Eqs. 26–27) are both `diag(1, B₁ … B_m, 1)` against `diag(R, R, …)` — two families
of two-by-two reflections on interleaved pairings of a line, which is the staggered walk model
arrived at from two directions. Only the blocks differ: constant `cos = 1/3` either side of
the weld, versus `cos ω_x = 1 − 2x/n` with Hamming weight. So there is one `Line` type and not
two, and both reductions are real orthogonal matrices — no complex arithmetic survives them.

### M8's acceptance clause, met exactly

> *glued-trees column reduction matches full-graph evolution*

`tests/hitting.rs` runs the coined walk on the actual welded tree — random cycle, random
naming, three seeds — and compares the amplitude on `|t, φ(t)⟩` step by step against the
`4n+2` model. Worst disagreement `7e-16`, which is machine epsilon and not a tolerance.

This is the test Q7.5 says is the one that matters, and it is worth being explicit about why:
an incorrect reduction produces a walk that runs, stays unitary, and yields a plausible
hitting curve. Nothing about it looks wrong. Only the graph it claims to reduce can say.

### Kempe first, deliberately

Q7.4 reorders M8 to put the hypercube ahead of the welded tree: peer-reviewed since 2005
against a result from 2024, and no oracle needed. It validates the engine against something
settled before the engine is used to check something recent.

```
  n      T      p(T)       classical
  8     12    0.9614        3.1208e2
 12     18    0.9586        4.5889e3
 16     26    0.8275        7.0766e4
 20     32    0.8934        1.1114e6
```

`T ≡ n (mod 2)` with `|T − πn/2| ≤ 1`, and the classical column is an exact tridiagonal solve
of the lumped birth–death chain, not a simulation. **The parity condition is load-bearing.**
The hypercube is bipartite by Hamming weight, so a walker is only ever at weight `≡ T (mod 2)`
and the antipode has weight `n`; round `πn/2` to the nearest integer without it and half the
dimensions return exactly zero, which reads as a broken walk rather than a broken test.

The convergence is `1 − O(n^{-1/5})`, and at sizes you can plot it is neither fast nor
monotone: `n = 2` returns an exact zero, because the two-port Grover coin is the Pauli `X` and
the walk is a deterministic cycle. That is what an asymptotic theorem looks like up close, and
the table prints it rather than starting at the size where it flatters.

### Two facts the theorems do not give you

Li, Li and Luo's Theorem 4.1 promises `max{p(t) : t ∈ [2n, 3.6 n log₂(5n)]} > 1/(20n)`.
Measured:

```
  n reduced  horizon      T1      p(T1)    1/(20n)    T1/n
 10      42      204      23   0.619327   0.005000    2.30
 20      82      479      45   0.549208   0.002500    2.25
 30     122      781      67   0.434605   0.001667    2.23
```

**The bound gives away two orders of magnitude** to be a bound. And **`T1` lands at about
`2.2n`** while the horizon it is searched over grows as `n log n`, so the classical
precomputation overshoots by a widening margin — except at `n = 3` and `n = 4`, where the best
time jumps to 33 and 59 for a few percent more probability, which is the only thing that makes
scanning the rest of the window worth doing at all.

*The second of those is the paper's own Conjecture 6.1, not a new observation.* §6 conjectures
`T ∈ [2n, 2.5n]` with `T ≈ n/√(pq) = 2.1213n` from its own numerical simulation, and only odd
`T`, because `p_T = 0` on every even step. Measured here: `T1/n` is `2.30` at `n = 10`, `2.15`
at `n = 100`, `2.137` at `n = 300`, always odd. This is a reproduction of a conjecture, and it
is worth being clear about which of the two facts above is which.

*The base of that logarithm is 2.* The paper writes `log`; Eq. (4.73) pins it down by using
`(1/2)^{log 5n} = 1/(5n)`. Wrong base costs a factor of 1.44 in the horizon and nothing in the
answer, which is exactly the kind of constant a reader reproducing the number would trip over.

*Two versions disagree on the constants.* arXiv:2304.08395v2 states `1/(20n)` at
`T ≈ 3.6 n log₂(5n)`; Decisions-03 quotes `1/(24n)` and `log(24n)` from the Algorithmica
version. The gate asserts against the stronger one.

*And Decisions-03 §12's fourth unknown, answered.* It asks for the smallest `n` at which the
bound actually holds in this implementation, on the grounds that the theorem is only proved
for sufficiently large `n`. **The answer is `n = 1`.** It holds at every size from 1 to 40,
never closer than a factor of 30 to the floor. The asymptotic qualifier is a limitation of the
proof technique, not a property of the walk.

### Algorithm 2, and the two claims the abstract runs together

Q7.1.4 is the correction the amplification layer exists to honour, and it was the largest
single error in the earlier ruling:

```
plain walk alone   ->  p = Omega(1/n)                   <- this is the exponential separation
zero error         =   plain walk + exact amplification  <- this is the paper's title
```

Both are true and they are different claims. The layer is Long's algorithm — `A = M_U^{T1}`,
`G(α, β) = A S₀(β) A† S_t(α)` with `α = −β = 2 arcsin(sin(π/(4T₂+2)) / sin θ)`,
`θ = arcsin|⟨t|A|0⟩|`, `T₂ = ⌈(π/2 − θ)/(2θ)⌉` — and on the reduced walk it is cheap, because a
real orthogonal `M_U` means a complex state is two real vectors stepped by the same routine.

```
   n     T1    T1/n   amplitude   T2     alpha  p after amplifying
  50    109   2.180    0.623306    1    1.8619   1.000000000000060
 100    215   2.150   -0.510514    1    2.7350   1.000000000000118
 150    323   2.153   -0.457824    2    1.4818   1.000000000000295
```

**`T1` at `n = 50, 100, 150` is `109, 215, 323` — the paper's Table 2 exactly**, recomputed
from the reduced matrix with no constant taken from the paper except the horizon. That is the
closest thing to an independent end-to-end replication this result has, and Decisions-03 notes
there was none.

One Grover round suffices up to `n = 100`; `n = 150` is the first size needing two, which is
why it is in the test.

### The cage, and the coin the note has to name

Decisions-03 §11 asks for Krovi and Brun's infinite hitting times as a note under Part VI-A
§T2 — the purest instance of the AB-cage trap, a configuration where the walker provably never
arrives, from interference alone. **The note must name the coin.** Krovi and Brun's infinite
hitting times are a *DFT-coin* phenomenon; the Grover coin on the same graph, from the same
start, arrives with probability one. Written without the coin the note reads as if it
contradicts the Kempe milestone directly above it.

Measured, on the 4-cube, 20 000 measured steps:

```
  hypercube n=4    Grover     arrives 1.000000   never arrives 0.000000
  hypercube n=4    Dft        arrives 0.571429   never arrives 0.428571
```

**Exactly 3/7 of the amplitude never reaches the far corner** — converged to `1e-13`, with the
last quarter of the run detecting `9e-30`. Krovi and Brun give the mechanism for this graph
(the DFT operator on `Q₄` has eigenvalues `1, −1, i, −i`, each eightfold degenerate, leaving a
sixteen-dimensional space of eigenvectors with no amplitude at the target); the 3/7 is measured
here, not quoted from them. And it is dimension-specific: on the 3-cube the same coin arrives
with probability one, which is why they name dimension four.

### M8's third acceptance clause no longer means anything

> *the √HT slope fits*

This came from Part II §P6, which quotes Szegedy's square-root-of-hitting-time result and then
applies it to maze traversal. Decisions-03 §3 separates three things the literature calls
hitting time — one-shot, concurrent, and marked-vertex search — and the `√HT` result is about
the third. **Marked-vertex search is not entrance-to-exit traversal**, so there is no √HT slope
to fit for the thing M8 was measuring, and the Szegedy module that was to produce it is gone.

Replaced by: Kempe's one-shot hitting on the hypercube against the exact classical `2^n`,
which is the same claim M8 wanted — a provable separation in traversal time — with a theorem
that is actually about traversal.

### Exit criteria — met

- Reduced model matches the full graph on the welded tree: `7e-16`, three seeds, `n = 1..3`.
- Reduced model matches the full graph on the hypercube: `n = 1..7`.
- Kempe's hitting probability above `0.45` at the prescribed `T` for every `n` in `8..20`.
- Welded-tree success probability above `1/(20n)` for every `n` tried, `2 ≤ n ≤ 16`.
- `p(t) = 0` **exactly** for `t < 2n`, and on every even `t` — the light cone and Eq. (4.81),
  structurally rather than numerically, so the assertion is equality and not a tolerance.
- Long's amplification reaches `1.0` to `1e-12` at every size tried, including `n = 150` where
  `T₂ = 2`; Table 2 reproduces exactly.
- Both reductions orthogonal to `1e-14`; the arc-indexed walk unitary to `1e-13` on a graph of
  mixed degree, under both coins.

### Q7.4's build order, and what is left of it

```
1. hypercube hitting (Kempe)              done
2. welded trees, reduced (4n+2) model     done
3. welded trees, full coined walk         done -- and it is what validates 2
4. CTQW for Part V 1.3's exponent toggle  already built, in Phase 8
5. exact amplification layer              done, though the ruling called it optional
6. Szegedy                                deleted
```

Item 4 needed nothing: `overtone-graph::evolve` has carried it since M22. `diffuse` is
`e^{-Lt}` and `interfere` is `e^{-iLt}` over the same eigenbasis of the same maze Laplacian,
which is exactly Part V §1.3's real-versus-imaginary toggle. Building a second one would have
been the mistake Q7.4 is about, one crate over.

### What remains of Phase 5

Two-dimensional mazes from a seeded coordinate hash, the designed dark corridor below `1e-6`,
and **M9, the learned coin**. Q7.2 has one thing to say about M9 before it starts: a learned,
position-conditioned coin is precisely the "arbitrary position-dependent coin" that the
coined/Szegedy equivalence excludes, so it inherits **no** quadratic hitting-time guarantee. If
it beats Grover on some family that is an empirical finding needing its own justification, not
a theorem being applied.

---

## Phase 15 — The agent language  *(M41a)*  — DONE

**Spec:** Part VIII §2, and Decisions-01 Q1, which created this milestone and moved it in front
of M39.

### Why a milestone exists at all

> `d` is not a property of the game. It is a property of **(game, strategy language)**.

Lantz et al. measure resistance to partial solutions *by a specific family of agents*. Change
the family and the number changes, so the language is part of the experimental apparatus and
has to be frozen before `d` is measured rather than after. Phase 9's M35 measured a `d` over
`overtone-orbit::ladder`'s language; the league would have measured a different one over a
different language, and neither number would have described the other.

There was a sharper problem underneath, and it is the one that decided the design:

> **A pure declarative feature-weight policy has no compute axis at all.**

It evaluates in `O(1)` per move. A ladder over it has no rungs, so `d` is not merely mismatched
over the league population — it is undefined. The resolution is to **declare the search, not
the policy**: the submitter names a kind and a budget from a published vocabulary and the
engine runs the search. Part VIII §2's rule that no submitted code is ever executed holds
unchanged, and `budget` is the rung index.

```toml
language   = "v1"

[agent]
name       = "kestrel"
generators = ["pawn", "rook", "bishop", "knight"]

[agent.search]
kind       = "mcts"        # "greedy" | "negamax" | "mcts"
budget     = 512           # the compute axis
rollout    = "tablebase"   # "random" | "tablebase"

[agent.eval]
features   = ["dim_g", "orbit_size", "safe_set_size", "half_chain_entropy"]
weights    = [0.31, -0.12, 0.44, 0.08]
```

### Four differences from the ruling's example, each one measured or checked

Decisions-01's TOML is explicitly placeholder — it admits inventing `reach_margin` — so the
differences are deliberate.

| Ruling's example | v1 | Why |
|---|---|---|
| six features | **four** | `freeze.rs` measured all eight candidates. `coherence` and `average_branching` are constant across siblings; `separation_deficit` is dominated by `dim_g` at 14× the cost; `temperature` is the best discriminator measured and is cost-disqualified at 714 µs against the 4096 evaluations a budget-4096 move needs. |
| `reach_margin` | **absent** | It corresponds to nothing implemented. Decisions-03 Q1 withdraws it; `orbit_size` and `safe_set_size` cover its intent. |
| `"bishop-Z2"`, `"knight-hop"` | `"bishop"`, `"knight"` | Those names do not exist. Part VII §2 has four pieces. |
| `statistics = "fermionic"` | **absent** | `Game::merge` does not implement exchange statistics, so the field could not change an outcome — which is precisely what Part VII §0's invented-number test forbids. It goes in when rule 6 consumes it, as a v2 change. |

### The eval is a difference, and that is not a style choice

`Eval::score` is `Σ wᵢ (fᵢ(me) − fᵢ(them))` over normalised features. Negamax is only correct on
a zero-sum evaluation; an eval scoring the side to move alone would make `kind = "negamax"`
quietly unsound, nobody would notice, and the resulting `d` would describe an agent family that
does not do what its spec says.

Each feature is normalised against a **structural** ceiling rather than a tuned scale:
`dim(su(2^n)) = 4^n − 1`, the orbit is a submanifold of a `2^(n+1)`-dimensional real space,
`2^n` cells can be safe, and the half-chain entropy of `n/2` qubits is at most `(n/2) ln 2`.
The two exponential ones are compressed by `ln(1+x)` first, because a linear eval over a
quantity spanning `4^n` is an eval over one feature.

### Q4's canonicalisation, in the order that makes it sound

Decisions-03 Q4 caught a gap: under greedy argmax `w` and `2w` select identically, but UCT
compares the exploitation term against `c√(ln N / n)`, so **scaling the weights changes an MCTS
agent's behaviour**. L2-normalising alone would be a behaviour-changing transform disguised as
canonicalisation — invisible in review, visible only as an unexplained rating shift.

1. `Eval::uct_value` maps the score into `[0, 1]` at point of use, dividing by the weights' L1
   norm, which is the exact bound given features in `[0, 1]`.
2. `Agent::parse` then L2-normalises, rejecting all-zero and non-finite.

And the constant this makes legitimate: UCT's exploration term is `√2`, which is Kocsis and
Szepesvári's value *derived for rewards in `[0, 1]`*. The two halves depend on each other —
normalise the eval differently and `√2` stops being the right number.

### One collapse function, two regimes, differing by one number

MCTS is primary for a physical reason: measurement is Born-random (Part IX §1.2), so Overtone
is a stochastic game, and expectimax's chance nodes would be 1024-way at `n = 10`. But
`Game::apply` deliberately collapses to the **likelier** branch, because the depth ladder
compares strategies and collapse variance is noise there.

Both are now one function. `collapse_at(psi, q, draw)` keeps the one-branch when
`weight_one > draw`, so a uniform `draw` is exactly the Born rule and the deterministic
collapse is the *median* draw, `0.5`. The harness and play differ by one number rather than by
one implementation.

Which one a search uses is then part of what an agent declares, and `LANGUAGE_V1` says so:
`mcts` plays the game as it is, `greedy` and `negamax` search against the deterministic
collapse. That is a modelling error, and it is one the submitter is choosing.

### `presentation.toml`, and the boundary it makes mechanical

Decisions-01 Q4 asks for one file holding every authored presentation constant, plus a CI test
that perturbs each and asserts match results are unchanged. Both shipped. The operational test
for what may live there is the ruling's own: **a number is invented if changing it changes an
outcome**, so a presentation constant is one that can change how something is shown or accepted
and can never change a result.

It is deliberately short — two entries — and the value is not its length. Most numbers in this
repository are measured or derived. The point is that the first mechanic anyone tries to hide
in it has somewhere to be caught.

### Decisions-05, which landed mid-build and changed four frozen things

| Item | Was | Now |
|---|---|---|
| Temperature mechanism | PUCT prior | **Progressive bias**, `W·H/(1+n)` |
| Field name | `temperature_prior` | **`temperature_bias`** |
| Default weight | 0.7 | **0.15** |
| Temperature's second role | none | **expansion ordering under widening** |
| Chance nodes | sampling | **explicit in the tree** |
| `rollout` value | `"random"` | **`"playout"`** |

The mechanism change is the substantive one. Temperature is an unbounded positive scalar, not
a distribution, so a PUCT prior would have to softmax it — imposing a distribution shape on
something that is not one, adding a scaling parameter with no published guidance, and putting a
softmax temperature `τ` three lines from CGT temperature `H` in the same function. Additive
progressive bias needs none of that, and its `1/(1+n)` decay hands control to the empirical
mean as visits accumulate, which is the forgiving behaviour you want from a heuristic nobody
has validated yet.

### The heuristic costs more than the search it guides

A literal reading of §1 wants `H(s,a)` at every node. Measured, at `n = 4` with three regions,
**one temperature field costs 577 µs** — and MCTS creates about one node per playout:

```
budget 512   heuristic per node   0.30 s     entire search   0.09 s
budget 4096  heuristic per node   2.4 s
```

Five to eight times the search. So `H` is a property of the *move*, evaluated once in the root
position, and it is stale deeper in the tree — which is exactly the case progressive bias was
chosen for.

### The heuristic reading that was wrong, and the correction

**This section previously reported the opposite of what is true, and the retraction is more
useful than the claim was.**

What was written: the temperature field is *"uniformly −1 at every width the ladder runs at"*,
therefore temperature is a near-binary indicator rather than a graded field, therefore Phase
12's sibling spread of `1.207` was one outlier over a mean of `−1`.

What is true, from M39a's sweep across seeds and plies:

```
fraction of regions hot, by ply
  n = 4   0.00 0.04 0.04 0.21 0.21 0.33 0.46 0.62 0.67 0.75 0.71 0.71
  n = 5   0.00 0.00 0.00 0.00 0.00 0.04 0.08 0.21 0.29 0.33 0.42 0.54
  n = 6   0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.17 0.21 0.33 0.54 0.67
```

**The field is cold in the opening and hot from the middlegame on**, which is exactly what a
temperature is supposed to do — Part IX §5.3's *"you can watch a region heat up before it
matters"*, measured. And the hotness at depth is real rather than numerical: the median gap
among hot regions at ply 8 to 11 is `0.25` to `0.875`, and the smallest is `0.0625` — ten
orders of magnitude above the floor, so nothing that survives it is numerical.

**One cause, two wrong conclusions: the sample was the opening.** Five plies of one seed is the
cold phase. `freeze.rs` walks four plies in, so its measurement sits in the same phase — which
means the `1.207` was never an outlier artifact and stands as originally reported. The
correction needed correcting.

The general form is worth keeping: **a quantity measured only where a game begins will look
like whatever beginnings look like.** Nothing about the reading was wrong except where it was
taken.

What survives unchanged is the consequence for the implementation. A constant heuristic adds a
constant to every child and reorders nothing, so `temperature_bias` is inert *at the opening*
by construction, and `tests/search.rs` asserts that alongside the middlegame signal rather than
instead of it.

### The tie-break that made a whole move class unreachable

Flat heuristic, stable sort, and the expansion order becomes whatever `legal_moves` emits —
whose first 208 of 240 candidates are all `Apply`. Under widening the root expands about 23
children, so **no measure move was ever tried at all.** The fix is to shuffle before the stable
sort, so ties break on a seeded permutation instead of on the move generator's enumeration
order. Measured after: 141 of 513 nodes at budget 512 are chance nodes.

### Chance nodes: the ruling's premise does not match the rule

§4 requires double progressive widening at chance nodes because *"at `n = 10` a position
measurement has up to 1024 outcomes"*. Part VII §5 rule 3's measure move is
`Move::Measure { qubit }` — it collapses **one** qubit. A chance node here has exactly **two**
outcomes, both enumerated with their exact Born weights. There is nothing to widen and nothing
to sample; a distribution you can write down is not one to draw from.

The ruling's real requirement is implemented, and it was a genuine defect before: chance is
now explicit in the tree rather than folded into whichever outcome happened to be sampled
first. If the rule ever becomes a register-wide measurement, `Kind::Chance` is the node that
needs widening, and the code says so where it is defined.

### Progressive widening, and why it is not optional here

```
children ≤ ceil(k · visits^α),   k = 1, α = 0.5
```

Coulom (2007) and Chaslot et al. (2008), as stated by Couëtoux et al. (2011); Browne et al.
(2012) note widening is *especially* effective when preferred actions are tried first, which is
what §3 puts temperature in the tree to do. At `n = 4` an agent holding all four generators has
**240 candidates**, so full expansion spends the first 240 playouts of a 512 budget giving
every candidate exactly one visit and never reaches a second ply. Measured with widening:

```
   name   budget   nodes  root kids   chance   depth   ms/move
kestrel       16      17          4        5       4      62.3
kestrel      128     129         12       36       6     214.8
kestrel      512     513         23      141       6     866.2
```

### Work units, and where the model stops describing the machine

Decisions-03 Q5: **one work unit is one complex-coefficient update in the active
representation**, computed analytically per gate rather than counted in the inner loop. A
counter is a measurement; a coefficient table is an assumption — and this reads the cost off
the machine instead of asserting it.

```
   name      kind   budget   work units/move   ns per unit
kestrel      mcts      512            999424         781.4
   pike   negamax      512             16384         213.0
  stoat    greedy      512              2048         124.4
```

The six-fold spread in nanoseconds per unit is the honest part. At `n = 4` a state vector is
16 amplitudes, so per-call overhead — building an `Observable`, allocating the new vector —
dominates the updates the unit counts, and MCTS pays it more often. The unit becomes a
faithful currency as `2^n` grows; at demo widths it is a lower bound on a constant-heavy cost,
and printing wall clock beside it is what keeps that visible.

### The compute axis, and a bug it caught

```
   name   budget    W    D    L   win rate     vs stoat@1, 12 games
kestrel        1    7    5    0      0.792
kestrel       64   11    0    1      0.917
   pike        1    0   11    1      0.458
   pike       64   12    0    0      1.000
  stoat        1    0   10    2      0.417
  stoat       64   10    0    2      0.833
```

Wins, draws and losses separately, because the rate alone hides the shape: **low budgets do not
lose, they draw**, and converting draws is what the compute buys.

The first version of this table was flat for `negamax` at every budget below 240, at exactly
`0.479`. The cause was in `negamax_root`: an iteration aborted by the node budget discarded its
best move and fell back to the first candidate, so an agent whose budget was under the
candidate count always played the same move whatever its budget. Every move an aborted
iteration *did* examine was searched to full depth, so keeping the best-so-far is both standard
and correct. The symptom was two identical win rates, and nothing else.

### Decisions-05 §5's open question, first data point

Both research passes came back **NOT FOUND** on whether exact terminal evaluation changes the
shape of the performance-versus-compute curve, and the ruling notes Overtone is unusually well
equipped to answer it: an exact tablebase, a skill-trace protocol, and a compute axis. The
answer is free, because `rollout` is already an agent field and the grid *is* the experiment.

What can be said now: `tablebase` and `playout` differ here **only in the terminal value**, and
both reach it by the same uniform random walk to the coherence horizon. Before the comparison
can mean anything, that walk has to carry signal — which is M39's first job rather than
something to tune quietly now.

### What this unblocks

M39 (the Goodman grid, re-measured over v1 and reported as **Skill Trace**), then M38 the
board, then M43 the league — all of which were waiting on a frozen language and none of which
could have produced a comparable number without one.

---

## Phase 16 — The coldness sweep, and three readings taken in the wrong place  *(M39a)*  — DONE

**Spec:** Decisions-06 Q16 and Q17.

Q17 reverses the question the ladder was about to ask. **Do not run a Skill Trace at a width
until you know the width has anything to trace** — and the diagnostic needs no games at all,
only position evaluation. `overtone-orbit::coldness`, and `examples/coldness.rs`.

### The answer

```
fraction of regions hot, by ply
  n = 4   0.00 0.04 0.04 0.21 0.21 0.33 0.46 0.62 0.67 0.75 0.71 0.71
  n = 5   0.00 0.00 0.00 0.00 0.00 0.04 0.08 0.21 0.29 0.33 0.42 0.54
  n = 6   0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.17 0.21 0.33 0.54 0.67
```

**The game starts cold and heats up**, which is what a temperature is for — Part IX §5.3's
*"you can watch a region heat up before it matters"*, measured. The hotness at depth is real
rather than numerical: median gap `0.25` to `0.875` at ply 8 to 11, and the smallest surviving
gap is `0.0625` — ten orders of magnitude above the `1e-12` floor.

### The mistake that produced three wrong statements

Phase 15 reported the field as *"uniformly −1 at every width the ladder runs at"* and drew two
conclusions from it: that temperature is a near-binary indicator, and that Phase 12's sibling
spread of `1.207` was one outlier over a mean of `−1`. Then this sweep's own first pass read
the standing score as *"exactly zero for both players at every width."*

**All three were the same error: the sample was the opening.** Five plies of one seed is the
cold phase; `freeze.rs` walks four plies in and sits in the same phase; the standing-score
reading came from a single position at six plies. With a real sample:

```
   n  mean absorbed(0)  mean absorbed(1)     1/dim
   4          5.469e-2          6.641e-2   6.250e-2
   5          7.617e-2          1.016e-1   3.125e-2
   6         7.909e-34         2.929e-35   1.562e-2
   7           0.000e0           0.000e0   7.812e-3
```

So the `1.207` was never an artifact and stands as originally reported; the correction needed
correcting. **A quantity measured only where a game begins will look like whatever beginnings
look like**, and this repository made that error three times in one afternoon before the sweep
that was built to catch it caught it.

### The floor has to be used everywhere it is defined

`HOT_FLOOR` was defined, `Reading::is_hot` used it, and the example's by-ply table compared
`left > right` directly — so the same quantity printed two different answers in one run. The
headline read **0.88 hot at ply 6** where the floored value is **0.46**, and the inflated number
had already reached the phase notes, `CLAUDE.md` and a gate line before the two tables were put
side by side. With the floor applied consistently the smallest surviving gap is `0.0625`, ten
orders of magnitude above it, so nothing that passes is numerical.

### `left > right` is not a hotness test

Gaps come in two populations — real ones at `1e-1` to `1e0`, and dust from `1e-32` down to
`1e-96`. Without a floor, `n = 6` at ply 8 reports 50% hot with a *median* gap of `6.7e-32`,
which is 50% of nothing. `HOT_FLOOR = 1e-12` is derived from the engine's own reproducibility:
absorbed weight is a sum of `|amp|²` and native-versus-wasm amplitude agreement is `5.6e-16`,
so weights agree to order `1e-15`. With the floor in place `n = 6, 7, 8` report `0.0%` instead
of a spurious `8.3%`.

### The width ruling: n = 5

Q17 says run M39b at whichever width has hot structure. Measured, that is **`n = 5`** — hot
from ply 4, standing score intact at `1e-1`, and one width wider than the retired `d = 6`.

`n ≥ 6` is not cold. It is **numerically dead at this ply depth**: the players start at
opposite corners of a `2^n` window and a coherent walk needs `O(n)` plies to cross, so at eight
plies the overlap that absorbed weight measures has not happened yet. That is a different
problem with a different fix — more plies, or a score that does not wait for overlap — and it
is a hard ceiling on the current evaluation rather than a property of the game.

### Q16's hypothesis, tested and not supported

Decisions-06 proposed that `d` saturating at `n = 4` and the field being cold at `n = 4` might
be one fact — a game whose components are all numbers has no tactical content, so no
search-language enrichment could extend its ladder. **`n = 4` is 46% hot by ply 6 and 75% hot
by ply 9.** Whatever shortens that ladder, it is not an absence of decisions. A clean negative
on a specific hypothesis, recorded as one.

### The decomposition is not the one Part IX justified

Q16 asks whether the decomposition came apart from its justification in implementation. It did.
Part IX §5.3 argues decomposition is available because *"the maze's corridor structure provides
weakly-interacting regions naturally"*. `thermal::regions` splits **contiguous qubit blocks**.

Those cannot be made into the same object. Orbit's cells are basis indices `0 … 2^n`, and a
generator on qubits `{0, 1}` changes the amplitude on *every* cell — so the game does not
decompose spatially at all, and a qubit-block split is the only decomposition available rather
than the one the spec argued for. It works well enough to produce hot structure, which is the
useful news, but the spec's justification does not cover what was built.

That also re-reads a Phase 12 result. `interaction_leak` measured `0.0000` and was reported as
the decomposition being exact. It is — but the positions it was measured on were cold, and sums
of numbers are exact trivially. **The leak test passed for a degenerate reason.** Re-measuring
it on hot positions is M39b's first line.

**Prompt B gains a sixth item**, per Q16: when a decomposition yields subgames that are all
numbers, is that evidence the decomposition is too fine, along the wrong axis, or that the
position is genuinely cold — and what characterises a good decomposition in the CGT literature?
The measurement above answers the *when* for Overtone (the opening, and only the opening) but
not the *what characterises*.

---

## Phase 17 — The explainers  *(M54, inverted)*  — PLANNED

**Spec:** Part X, as amended by Part XI. **This phase inverts Part X §9's build order**, and the
inversion is the single most consequential thing the sweep found.

### Why the explainers come first

Part X §11 says the risk has moved from the game to the debrief, and that **M54 is the milestone
that decides which project this is**. Part XI §2 then finds that the risk is larger than that,
from the field's own survey and its developers' own traffic data:

> Only around 1% of views for the blog post came from the in-app link. Around 58% came from
> search engines.

and their conclusion: *"the blog posts were a more successful learning resource than the app."*
A second, independent case in the same paper — Battleships with partial NOT gates — *"there is
little evidence of it being played, but the blog post remains the most viewed on the Qiskit blog
by some margin."* Two cases, same direction, large margin.

So a debrief screen inside the Gauntlet inherits the app's reach, which is to say almost none.
Part XI's structural fix is to invert the dependency:

```
was:   level → in-game debrief screen → optional link out
now:   nine standalone explainer pages, publicly indexed, complete without the game
       ↑ the Gauntlet deep-links into the relevant one after each level
```

**Three further reasons to build them first, not merely to restructure them.**

*They are testable immediately.* The acceptance test Part XI §3 specifies — *can a naive reader
state, unprompted, why L3 was impassable?* — can be asked of a page today. It does not need a
level to exist.

*They give M53 a control arm.* With pages first, the same question can be put to a reader who
only read the page and to one who played then read. **If the game arm does not beat the page
arm, this repository has reproduced Part XI §2's finding in-house** and the Gauntlet should stay
small. That comparison is cheap — five readers per arm — and Part XI §10 notes the field has no
evaluation standard at all, so running one on yourself first is the honest order.

*The valuable artifact survives the game not being built.* Wouters et al. find serious games
beat conventional instruction at `d = 0.29` for learning and `d = 0.36` for retention, but **not
in engineering** — one of the two domains where the advantage disappears — and one synthesis puts
it plainly: *"if you compare serious games with active teaching, they do not appear to be more
effective in terms of learning, whereas on average they are more expensive."* Build the cheap
thing that works, and make the expensive thing earn its margin against it.

### What is encouraging, specifically

The one moderator both Clark et al. (2016) and Wouters converge on is **alignment between game
mechanics and instructional content**. Part X §3's rule — *a level is a wall you cannot pass
until you understand the theorem, and the key is always a capability, never a skill* — is that
property taken to its limit. The mechanic **is** the concept. So the evidence is discouraging
about the category and specifically encouraging about this design, which is exactly the position
a two-arm test is for.

### The nine pages

One per level, each complete without the game, each reviewed against the **Minus-Sign Test**.
Every one maps to a panel that already ships:

| Page | The theorem | The instrument it points at |
|---|---|---|
| Spread | ballistic versus diffusive | `overtone-walk --example transport` |
| The dark corridor | destructive interference | Part II P2; the phase-as-hue law |
| The ceiling | `Var[∂C]` and the frequency ceiling — score exactly zero | `overtone-cli -- train`, the LP ceiling |
| Resonance | trainable input scaling `λ` | the `sonify` toggle, already shipped |
| The cage | AB caging, and that the coin is the cage | Phase 14's `3/7` on the 4-cube |
| The feast | `dim(g)` up, gradients down | `overtone-cli -- predict` |
| The front | decoherence and the Zeno effect | the shot dial |
| Checkmate | losing as a controllability statement | `overtone-orbit --example checkmate` |
| The mirror | the trainability–simulability tension | `overtone-cli -- dequantize` |

### One page exists, as a format test

`docs/explainers/the-ceiling.md` is written — L3, the one Part X §9 says has to land. It is the
prototype, not the first of nine, and it exists for the same reason Part X tests M53 on a real
person before building seven more levels: **the format is the risk, and one page is enough to
find out.**

It is written entirely from the instrument's own output rather than from memory. The staircase
it turns on —

```
   C           J*
   2     0.000000
   3     0.500000
```

— is `overtone-cli -- ceiling --k 3` run as printed, and every file and flag it cites was
checked to exist. `docs/explainers/README.md` states the three rules a page follows: complete
without the game, every claim points at a runnable command, and passes the Minus-Sign Test
explicitly rather than incidentally.

### Three corrections to Part X that Phase 17 has to carry

**L4 does not need sonification built.** Part X §9 calls it *"the highest impact-per-line item in
the series and has been waiting since Part IV"*. It shipped in Phase 4 — `web/js/sound.js`,
`sonification_tones(k, lambda)` in the wasm crate, and a `sonify` checkbox that already never
autoplays. M55's largest stated dependency does not exist.

**L9's number is wrong and must be read, not written.** Part X §3 says the ending reports
*"effective χ = 4"*. The measurement at `n = 4, L = 3` is **`χ = 3`**, and `scripts/gate.sh`
asserts it. An ending whose punchline is a written constant is a twist; an ending that runs
`dequantize` on the player's own final circuit and prints what comes back is the thing Part X
§10 asks for when it says do not let L9 be triumphant.

**"Zero new engine work" is not true.** Part X §8 claims zero new physics, zero new mechanics,
zero new engine work. The first two hold. The third does not, and the constraint is already
written down: `scripts/check_js_budget.sh` caps hand-written JS at 1200 lines, the page stands at
**1186**, and the rule the number enforces is that *no panel computes a physical quantity*. So
every wall predicate, every pass condition and every level definition has to live in Rust behind
the wasm boundary, and the sparse renderer needs JS the budget does not have. The honest cost is
a new module plus a deliberate, documented budget raise — the same way it went from 800 to 1200.

---

## Phase 18 — M53, and the two-arm test  *(M53, M54b)*  — PLANNED

**L1 and L3 only**, per Part X §9, with Part XI §3's acceptance and the control arm above.

**M53 acceptance, stated so it can fail:** ten readers with no quantum background, five per arm.
Arm A reads the ceiling explainer. Arm B plays L1 and L3, hits the wall, then reads the same
page. Both are asked, unprompted, why the score was zero and why more training could not fix it.
**The Gauntlet is worth building out if arm B's answers are better; it is not if they are the
same.** Part X §10's own instruction — *test M53 on a real person* — with a comparison attached,
because Part XI §3 says the category's evidence is weak and the one thing that predicts success
is the property this design has.

**M54b — the histogram, which is an instrument before it is a game feature.** Zachtronics'
solution histograms replaced leaderboards for two stated reasons: a leaderboard is *"a fantastic
incentive for cheating"*, and for most players *"the only thing a global leaderboard manages to
tell you is that you suck (and not even by how much)"*. Barth's third point is the one that
matters here — *"because we include three antagonistic metrics, players optimizing for one
criterion often do poorly in the others"*. **A game designer arrived independently at Part VII's
Axiom II and shipped it in 2011.**

Overtone's three are antagonistic **by theorem** rather than by design:

```
dim(g)        expressiveness  ↑  →  Var[∂C] ∝ 1/dim(g) collapses
effective χ   classicality    ↓  →  requires large dim(g)
work units    cost            ↓  →  limits search depth
```

Nobody has shipped a histogram whose axes are antagonistic for a proven reason. And the
distribution of solutions across `(dim g, χ, work)` is exactly the data Part VII §8 wants for the
depth measurement, so **the display and the measurement are one object**. No unlocks, no
achievements — Barth again: *"the players who do optimize are often more intrinsically
motivated."*

---

## Phase 19 — The rest of the Gauntlet  *(M55–M58)*  — PLANNED

L2/L4/L5, then L6/L7/L8, then L9, then nav and first-visit routing — Part X §9's order,
unchanged, and gated on Phase 18's comparison. If arm B does not beat arm A, this phase is one
level and a link rather than seven.

**Two claim-discipline rules carried in from Part XI, both non-negotiable.**

*No "players find what algorithms miss."* The most-cited result in quantum citizen science is
retracted — and both halves of it. Part XI verified the News & Views retraction and flagged the
primary paper as unchecked; it is **also retracted**, in August 2020, withdrawn by its own
authors for an error in their optimisation code that invalidated the quantitative results. The
careful follow-up (Phys. Rev. Research 3, 013057) reports player-assisted results *"roughly on
par with the best of the tested standard optimization methods"* — parity, with the authors'
own caveat attached. **This field has already run that experiment.** Do not claim it unless the
grid says so and it replicates.

*Scale calibration.* HyperRogue has 403 Steam reviews; Quantum Flytrap reports *"around 70 users
on a regular working day."* This category does not produce mass audiences. It produces small,
durable, high-quality ones, and that is the correct expectation.

### One sharpening to Part XI's own analysis

Part XI §8 states *"there is no Sprague–Grundy Theorem for misère play impartial games"*, and
uses it to argue that reductions of this kind are fragile. The claim is too strong and the
accurate version is better for the point being made. The **naive** misère generalisation is
hopelessly complicated and useless beyond simple cases — but Plambeck & Siegel's **misère
quotient** (JCTA 2008) is the working generalisation, and its authors call it the long-sought
natural one.

So the reduction does not break. It **degrades**: from a group to a commutative monoid, from one
integer to a possibly large algebraic object, and from free combination to a multiplication in
that monoid. That is a sharper analogy for `dim(g)` than a missing theorem, and it sits beside
the contrast Part XI draws correctly:

```
nimber(G + H) = nimber(G) ⊕ nimber(H)              combining is free
dim(closure(g₁ ∪ g₂)) ≠ dim(g₁) + dim(g₂)          commutators generate new elements
```

**Sprague–Grundy is exactly the theorem Overtone does not get, and its absence is the mechanic.**

---

## The Decisions documents, and what they change

Six rulings documents supersede parts of the specs where they conflict:
[`decisions-01-phase-10.md`](spec/decisions-01-phase-10.md) rules on the eight open Phase 10
questions; [`decisions-02-synthesis.md`](spec/decisions-02-synthesis.md) synthesises four
external research passes into a claims audit; and
[`decisions-03-revised.md`](spec/decisions-03-revised.md) rules on the walk operator, the v1
feature vocabulary and the licence, and supersedes an earlier Decisions-03 in full.
Decisions-04 amends the MCTS design and **is not in this repository** — see the note below.
[`decisions-04.md`](spec/decisions-04.md) rules on spec immutability, build order, temperature
as move ordering, and the README; [`decisions-05.md`](spec/decisions-05.md) amends Decisions-04
from two MCTS research passes; and [`decisions-06.md`](spec/decisions-06.md) recovers the fifth
feature, sharpens the coldness result and reverses the ladder-width question. All of them are
committed to `docs/spec/` because a ruling that lives outside the repository is a ruling that
gets lost, and [`spec/README.md`](spec/README.md) states the convention they follow.

**The "five features" line, and what actually happened.** Decisions-05 §6 records the frozen
eval vector as five, and this repository freezes four. Decisions-06 Q15 settles it: that is not
a disagreement between documents, it is the Decisions-03 Q1 cut rule running. The five are its
six minus `temperature`, which Decisions-04 Q13 moved to `[agent.search]`:

```
Decisions-03 Q1     six    coherence, dim_g, orbit_size, safe_set_size,
                           temperature, average_branching
Decisions-04 Q13    five   temperature -> [agent.search], as a move-ordering
                           heuristic rather than an eval feature
measurement         four   see below
```

Q1's own rule was *"cut anything with near-zero sibling variance"*, and it pre-authorised the
outcome: *"if it does not, v1 is the other five and you say so in the docs."* This is that
saying, with the numbers:

```
                  sibling spread   leaf spread   us/call   frozen
dim_g                    0.546        32.006       0.28    yes
orbit_size               0.347            --       3.83    yes
safe_set_size            0.000         0.534       0.16    yes -- on the leaf test
half_chain_entropy       0.000         5.053       1.00    yes -- on the leaf test
temperature              1.207            --     714.11    no  -- moved to [agent.search]
coherence                0.000         0.000       0.00    no  -- cut, sibling variance
average_branching        0.000         0.000       0.00    no  -- cut, sibling variance
separation_deficit       0.027            --       4.00    no  -- dominated by dim_g at 14x cost
```

Two of Decisions-04's five are **provably** constant, not merely measured so. `coherence` falls
by `k` on every ply whichever arm of `Game::apply` runs, so it is a pure function of depth;
`average_branching` is `legal_moves(n).len()`, which does not depend on the move. Neither can
ever discriminate between siblings, so including them would be carrying two dimensions that
cannot change a comparison.

`half_chain_entropy` is the one that came *back*. Q1 excluded it on cost without profiling —
*"cost-disqualified unless you have profiled it"* — and profiled it is 1.00 µs and the second
best leaf discriminator in the table.

*(Decisions-06 Q15's worked example guesses that `dim_g` was the feature cut. It was not:
`dim_g` has the largest sibling spread of the survivors. The cuts were `coherence` and
`average_branching`, both at exactly zero.)*

**The one that reorders everything.** Decisions-01 Q1: `d` is a property of *(game, strategy
language)*, so the agent language is part of the experimental apparatus and must be frozen
before `d` is measured. Worse, a pure declarative feature-weight policy has **no compute axis
at all** — it is `O(1)` per move — so if the league's submission format were a policy spec,
`d` would be *undefined* over the league population rather than merely mismatched. The
resolution is to declare the **search**, not the policy: `search.kind ∈ {greedy, negamax,
mcts}` with `budget` as the compute axis, and the feature weights as the `[agent.eval]` block
both searches consume. MCTS is primary for a physical reason — Overtone is stochastic, and
expectimax's chance nodes would be 1024-way at `n = 10`.

```
was:   M38 board → M39 ladder → … → M41 notation → M43 league
is:    M41a agent-language freeze (v1) → M39 ladder → M38 board → M43 league
```

**Status of each ruling.**

| Ruling | Status |
|---|---|
| Q1 — unify the language; declare the search; M41a before M39 | **applied** (Phase 15), with four departures from its illustrative TOML |
| Q2 — work units under a published cost model; publish both curves | **applied** (Phase 15) — `overtone-orbit::work`, analytic per gate |
| Q3 — Goodman grid, not adjacent rungs; pre-register `N`, 0.95, `STEP_UNIT = 2σ` | **pending** (M39) |
| Q4 — invented number = one that changes an outcome; `presentation.toml` + perturbation test | **applied** (Phase 15) |
| Q5 — Born distribution in the measurement basis; no bucketing, no log default | **pending** (M46) |
| Q6 — decouple: field recomputes per ply, progressive fill; 60 fps was never the bar | **pending** (M49) |
| Q7 — fixed scale from the Atlas 99th percentile, visible clipping, numeric max on screen | **pending** (M49) |
| Q8 — memo table keyed on discrete state only; amplitudes are the value | **pending** |
| Q9 — DCO 1.1 plus a non-assignment contributor licence | **half applied** — `CONTRIBUTING.md` ships the DCO; the licence needs counsel and says so |
| Q10 — scoped headline; every README clause maps to a shipped instrument | **applied** — and the bar deleted two of the seven dismissal rows |
| D2 — M28 citation split: Sansoni + van Exter | **applied**, with a correction (Phase 13) |
| D2 — Part VI §1 anyonic citation "CONTRADICTED" | **rejected on the primary source** (Phase 13) |

### The v1 feature freeze, measured

Decisions-03 Q1 ruled six features and said to measure sibling variance before freezing, and
to profile `temperature` before committing to it. Both were done
(`overtone-orbit/examples/freeze.rs`, `freeze2.rs`). **Three of the six do not survive, one
excluded feature comes back, and the ruling's own escape clause fires on `temperature`.**

Two tests are needed, because Q1's sibling test alone is unfair to features that vary with
*depth* rather than across siblings at one node. A feature is useful if it discriminates
*either* the moves at a node *or* the leaves a search evaluates.

```
                      sibling spread   leaf spread   us/call   at 4096/s
dim_g                       0.546         32.006        0.28   fits
orbit_size                  0.347             --        3.83   fits
temperature                 1.207             --      714.11   NO
safe_set_size               0.000          0.534        0.16   fits
half_chain_entropy          0.000          5.053        1.00   fits
separation_deficit          0.027             --        4.00   fits
coherence                   0.000          0.400        0.00   fits
average_branching           0.000          0.000        0.00   fits
```

**Cut — `average_branching`.** Not weak, *structurally constant*. `Game::apply` pushes
`legal_moves(n).len()`, and `legal_moves` is a function of `n` alone, which never changes
during a game. The feature is identical for every position, every line and every agent. It
cannot influence selection under any search.

**Cut — `coherence`.** Both arms of `Game::apply` subtract exactly `self.k`, so coherence is a
pure function of ply depth. Binned by depth, min equals max within every bin. It is a
restatement of something the search already knows.

Q1 asked to "confirm that asymmetry is implemented" between `measure` and generator
application, since `coherence` survives only if it exists. **It is not implemented**, and the
comment above the measure branch claims otherwise — *"Measuring costs the whole remaining
coherence block: it is the move that destroys superposition, and Part VII §4 wants that to be
the expensive one"* — while the code subtracts `k` like the other branch. Part VI §2.2 was
checked before changing anything, and it says measuring costs coherence *and* destroys the
spread; it does **not** say it costs more coherence. So the code is defensible and the comment
is wrong. Making measurement cost extra would change match outcomes, which under Decisions-01
Q4's own test makes it a mechanic — and an invented one. The comment is corrected; the
mechanic is not touched; the feature is cut.

**Cut — `separation_deficit`.** §12's first unknown, answered. It computes
`2·2^n − 2 − (|commutant| + |ideals|) − orbit_dimension`. Every input is a property of the
algebra or the orbit, and a unitary from `exp(g)` changes neither — it moves *within* the
orbit. So it changes only on absorption, exactly like `dim_g`, at 4.00 µs against `dim_g`'s
0.28. Measured sibling spread 0.027 against `dim_g`'s 0.546. Dominated on both axes.

**Restored — `half_chain_entropy`.** Q1 cost-disqualified it, reasoning that at `n = 16` it is
a 256×256 SVD in the inner loop. At the `n = 6` the game actually runs at it costs **1.00 µs**,
a million calls a second, and it has the second-largest leaf spread in the table. Its first
measured sibling spread of zero was a sampling artefact: the sample sat four plies from a
basis-state opening, where single-qubit generators keep the state near-product and every
sibling has entropy zero. At depth it ranges 0 to 1.386. It is in v1, with the caveat that its
cost scales with `n` and must be re-profiled if the window grows.

**Restored — `safe_set_size`.** Its zero sibling spread is structural rather than damning:
`Game::safe_for` reads the **opponent's** field, so a player's own move cannot change it by
construction. Across leaves it ranges 32 to 63. The sibling test was measuring the wrong thing
for this feature.

**`temperature` fires Q1's escape clause.** It is the best discriminator in the table — sibling
spread 1.207, twice `dim_g`'s — and at **714 µs** it sustains 1,400 calls a second against the
4,096 an MCTS budget needs. Decisions-01 Q6's per-position cache does not rescue it: every node
an MCTS expansion touches *is* a different position, so there is nothing to reuse. Q1
pre-authorised this outcome — *"If it does not, v1 is the other five and you say so in the
docs"* — so this is that saying.

**Frozen v1 feature vector, in order (weights are positional):**

```
1. dim_g                material          0.28 us
2. orbit_size           reachability      3.83 us
3. safe_set_size        proximity to loss 0.16 us
4. half_chain_entropy   spread            1.00 us
```

Four features, total 5.27 µs per evaluation, which sustains ~190,000 evaluations a second —
comfortably inside a 4,096-node budget. `temperature` remains the heat map (M49) and the move
*ordering* heuristic, where it is computed once per position rather than once per node.

### Q6, measured: the top of the ladder is the problem

```
budget    s/game    games in a 6 h job    1600 games needs
     8     6.75                  3199              3.0 h
    32    10.05                  2148              4.5 h
   128   134.96                   160             60.0 h
```

Decisions-03 estimated a 13.5 s per-game budget and judged the grid comfortable. At budget 128
a single game takes **135 s**, ten times that, and 1,600 games would need 60 hours against the
6-hour job limit. The cost is super-linear in budget, so the expensive pairings are exactly the
high-budget ones the grid needs most.

The ruling's own lever still applies and is now quantified: pairings are unequal, so a
`(8, 128)` pairing costs roughly half a `(128, 128)` one. But the honest consequence is that
**high-budget pairings get ~160 games, not 1,600** — a 95% Elo error near ±32 rather than ±10.
Q6 said to report heterogeneous precision rather than collapse it, and the heterogeneity is
larger than anticipated. SPRT early-stopping on distant pairs is no longer an optimisation; it
is what makes the top of the grid affordable at all.


**Three rulings are worth restating because they change how things get measured, not just
what gets built.**

*`STEP_UNIT = 0.65` is an invented number and Q3 says so.* The replacement is measured: play
an agent against a bit-identical copy of itself, take `2σ` of the spread. Decisions-02 renames
this a **repeatability calibration** rather than a noise floor, because identical-copy
self-play is not an established chess-testing convention and the weaker word is harder to
attack. And it costs what it costs: the 95% Elo error is about `400/√N`, so ±10 Elo needs
~1,600 games and ±5 needs ~6,400.

*Do not write "the standard Lantz metric."* Lantz et al. explicitly had no system for
evaluating `d` and left the strategy language, resource levels, performance metric and step
definition open. The lineage is Lantz 2017 → Tavener 2020 → Browne 2022 (**Skill Trace**) →
Goodman et al. 2024 (**Skill Depth**). Report Skill Trace, and plot Overtone against Goodman's
sixteen published values — Dots + Boxes `0.353`, Connect 4 `0.282`, Can't Stop `0.028`,
Tic-Tac-Toe `0.000` — which is a far better figure than the eyeballed chess/Go comparison in
Part VII §8 because it is measured on a compatible protocol.

*The headline changes.* Decisions-02 §5: `train a 100-qubit quantum RL policy on a free CPU`
is a capability flex in a field whose mood punishes them, and it invites a dismissal that
would be correct. Lead instead with **"Overtone tells you whether your quantum circuit will
train — before you train it"**, and name the tension in line two rather than hiding it: the
circuits that train are often the ones a classical computer can already simulate, and Overtone
measures both. Six of the seven standard dismissals are things this project *measures* rather
than things it denies, and that table belongs in the README.

**And one claim came back stronger.** Part VII §3's checkmate-as-controllability is novel and
better than the spec claims: Cantwell's Quantum Chess design notes state that "there is no
concept of check or checkmate. Kings are captured like any other piece." The orbit formulation
solves a problem the closest prior work explicitly abandoned. Cite Wu & Tarn (PRA 65, 2002) on
subspace controllability as its basis.

---

## Cross-cutting, from the start

- **No emoji**, anywhere.
- Every README claim is paired with a CI test that fails when the claim stops being true.
- Negative results are published, not buried.
- Determinism: seed everything; native and WASM agree.
- Scope discipline: Part III §12 governs.
