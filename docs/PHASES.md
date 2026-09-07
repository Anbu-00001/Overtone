# OVERTONE — phase plan

The specs define milestones M0–M45 across nine documents: Parts I–VIII plus the Part VI-A
traps addendum. This file groups them into eleven executable phases with explicit entry and
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

## Phase 8 — Exotic  *(M22, M23 done; M24–M27 outstanding)*  — PARTIAL

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

**Outstanding**
- **M24 — the optimiser flatline.** Part V §5.2's demonstration: four optimisers deep in a
  barren plateau, all flat, with Arrasmith et al. cited. The plateau instrument exists
  (Phase 3) and the citation is already load-bearing elsewhere — see Phase 7's M19
  corrections, where the same paper refuted Part IV §4.
- **M25 — distributional RL and the shot-budget dial.**
- **M26 — eigenoptions and Go-Explore.** Eigenoptions are built from the same Laplacian
  spectrum M22 now computes, so the marginal cost is low.
- **M27 — architecture search into the Menagerie.** The archive it feeds exists as of M19.
- **The browser panel for M22 and M23.** The engines and their numbers are done and tested;
  the `Lattice` tab that Part V §1.3 and §2.3 describe is not built.

---

## Phase 9 — Orbit: the game, headless  *(M34–M37)*

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

**M34 — checkmate.** The reachable-orbit computation from `g`, and the checkmate predicate.
Build this first: it is what makes the thing a game rather than a sandbox, and most of it is
already in `overtone-lie`.

Checkmate is not a score threshold. Dynamical Lie algebras come from quantum control theory,
and controllability is exactly what they compute: if `g = su(2^n)` the system is fully
controllable and every state is reachable; if `g` is a proper subalgebra the reachable set is
an **orbit**, and states outside it are unreachable in principle. **Check** is having overlap
with the pursuer's absorbing subspace; **checkmate** is that remaining true at every point of
the reachable orbit.

> **The exchange rate between material and position is a published theorem.** Absorbing an
> agent grows your algebra, which grows your orbit, which makes checkmate harder to deliver
> against you — that is material. The cost is `Var[∂C] ∝ 1/dim(g)`: a larger algebra means
> you can no longer learn — that is position. In chess the value of a bishop is a convention
> refined by tradition; here it is a measured quantity with a citation.

**The technical risk, stated before building.** "Decidable from `g`, computable in
milliseconds" is true at the two ends and not obviously true in between. Deciding whether a
*specific* target state lies in the orbit of a *specific* initial state under a proper
subgroup is a harder question than computing `dim(g)`. What is genuinely cheap:

- **Full controllability.** `dim(g) = 4^n - 1` implies transitivity: never checkmate. Exact,
  and already implemented.
- **Conserved invariants.** The adjoint action of `exp(g)` on the DLA basis is *orthogonal*,
  so the `g`-purity of each simple ideal — `P_j = sum_{alpha in g_j} <psi|b_alpha|psi>^2` —
  is constant along the orbit. `overtone-lie` already computes it and `overtone-gsim`
  already evolves it. Differing invariants **prove** unreachability in `O(dim g)`.

That gives a **sound certificate**, not a decision procedure: it proves "definitely
unreachable" and never "definitely reachable". For a win condition that is the correct
direction — a game must never end in a checkmate that is not one. The predicate may *miss*
checkmates, which makes games longer rather than wrong.

**Acceptance, restated.** Part VII asks to "verify against brute-force reachability that the
predicate is exactly correct." Soundness is the part that must hold exactly: on small
systems, **every position the predicate calls checkmate is verified unreachable by brute
force, with zero false positives.** Completeness is then a *measured* quantity — report the
fraction of genuine checkmates the cheap certificate misses. If that gap is small the game
is playable as specified; if it is large, the honest options are a more expensive predicate
or a stated rule that checkmate means *provable* checkmate. Do not soften it into a score
threshold: Part VII §12 is right that this is the single change that would turn the whole
design back into a toy.

*(Prior art, to be stated as a technical note and not as a dunk: Cantwell's Quantum Chess
(arXiv:1906.05836) introduces a measurement rule that, in its own abstract's words, "helps
limit the size of the superposition, so the game remains tractable for a classical computer."
Part VII deliberately inverts that decision. The claim that it also replaces checkmate with
king-capture is plausible but is **not** confirmed by the abstract — read the full paper
before putting it in the docs.)*

---

**M35 — the strategy ladder.** Generator basis, turn loop, and the depth measurement, all
headless. **No interface.**

**Three corrections to how `d` is defined, from reading the source rather than the summary.**
Lantz, Isaksen, Jaffe, Nealen and Togelius, *Depth in Strategic Games*, AAAI 2017, is the
load-bearing reference, and Part VII §0 paraphrases it in a way that changes what gets built:

1. **`d` is a count of steps, not a length in orders of magnitude.** The paper's procedure is
   explicit: plot the best strategy at each computational-resource level, then walk the curve
   counting how many times strength improves by at least a declared *step unit*. "The number
   of steps you have counted is the `d` for this game for the given settings." Part VII §11's
   acceptance — "a rising region spanning at least three orders of magnitude of compute" — is
   a different quantity. Both are worth reporting; only one of them is `d`.
2. **`d` is relative to a declared strategy language.** "Any observations made about a game's
   depth based on this model must refer to the language selected." Overtone's language must
   be written down before the number is quoted, or the number means nothing.
3. **There are no published `d` values to compare against — for any game.** The paper is a
   proposal. Applying the model to Tic Tac Toe, Blackjack and 3x3 Go is listed as *future
   work*, and it says outright that the complete model "assumes knowledge, not only of a
   strategy for playing perfectly, but also of the minimal computational resources needed for
   such perfect play", which "makes the model impossible to apply completely to complex,
   real-world games." **M39 cannot be a plotting exercise against published numbers.**

**One place Overtone is better placed than the paper's own examples.** It weighs win rate
against quality-of-move as the strength metric, prefers quality-of-move for being "more
simply, clearly, and consistently defined", and rejects it because it needs perfect play.
Overtone *has* perfect play in the decohered endgame, exactly, from M36's eigensolve. So the
strength metric can be quality-of-move where the paper could only wish for it, and win rate
elsewhere. That is a real methodological advantage and it should be used and said.

**Exit criteria — declared before measuring, per the C2 rule.**
- The strategy language, the CR ladder (powers of two in search budget), and the step unit
  (a 65% win rate, mid-range of the paper's 60–75%) are all written down **first**.
- Measured average legal-move count reported per game, targeting **25–40**, as chess reports
  ~35. A measured statistic, never a design intention.
- `d >= 4` steps with the curve still rising at the top of the ladder.
- **If `d` is small, report it and retune.** Substrate, generator-basis size, coherence
  budget, `k` and trap density are all knobs and `d` says which way to turn them. This is the
  stop-the-line gate: **do not proceed to Phase 10 on a flat ladder.** Part VII §12 —
  building the interface first means being attached to it by the time the number arrives.

---

**M36 — the endgame eigensolve.** The LMDP tablebase, triggered by full decoherence.

Axiom II's arrow gives the game chess's phase structure for free: openings are coherent and
classically hard, middlegames are partially decohered and searched, and the endgame is fully
decohered — a classical MDP, and therefore **exactly solvable**. Part V §2's linearly-solvable
MDP turns Bellman into a largest-eigenvalue problem under `z = exp(-v/lambda)`.

> **Overtone's endgame tablebase is an eigenvector.** Chess endgame tablebases cost decades
> of compute and terabytes; this one is a Perron–Frobenius eigenvector computed live, in
> front of the player, the moment coherence runs out.

Tempo is coherence, and it is physical rather than conventional. Zugzwang appears without
being designed, because evolving coherently is usually better than measuring and yet you must
sometimes measure to aim — and the Zeno structure makes the *option* to measure a liability.

**Dependency, load-bearing.** This is Part V §2, which lives in Phase 8, and it needs a maze
with a goal, which is Phase 5. Neither is optional. Within Phase 8, **build §2 first**: it is
promoted from "a striking panel" to infrastructure that two later milestones rest on.

**Exit criteria:** the exact solution appears within 100 ms of the coherence transition, and
matches brute-force optimal play on small boards.

---

**M37 — the complexity dial.** Part VII §6 as an interface control, with a live verdict on
whether the current position is efficiently evaluable, backed by the Phase 6 dequantization
test.

```
low  dim(g), low  chi  ->  g-sim and MPS both work  ->  positions efficiently evaluable
high dim(g), high chi  ->  no efficient classical representation exists  ->  provably hard
```

> **A slider that moves the game between the complexity class of checkers and a class
> strictly beyond chess.** Chess is a fixed point in complexity space; Overtone is a
> trajectory through it, and the player holds the parameter.

This is the strongest claim in the series and the one no board game can make. Part VII §12:
**do not hide it.**

**Exit criteria:** in one session, a position whose optimal move is computed exactly by the
M36 eigensolve, and then — one slider away — a position the dequantization test certifies has
no efficient classical representation.

---

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
minds, which is what Lantz's definition actually asks for. **Call it the ladder, not the
leaderboard**, consistently.

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

## Cross-cutting, from the start

- **No emoji**, anywhere.
- Every README claim is paired with a CI test that fails when the claim stops being true.
- Negative results are published, not buried.
- Determinism: seed everything; native and WASM agree.
- Scope discipline: Part III §12 governs.
