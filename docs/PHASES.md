# OVERTONE — phase plan

The specs define milestones M0–M27 across five documents. This file groups them into eight
executable phases with explicit entry and exit criteria. A phase is done when its exit
criteria are green in CI, not when its code is written.

Governing constraint: **Part III §12, "Minimum viable Overtone."** If scope has to be cut,
the version that keeps the thesis intact is Phase 1 + Phase 2 + Phase 3 + the closure
animation from Phase 6. Everything else is elaboration. Re-read that section whenever a new
panel suggests itself.

---

## Phase 1 — Foundation: simulator and verified gradients  *(M0, M1)*

**Spec:** Part I §5, §6.1, §6.4, §10, §14.

Workspace skeleton, state-vector engine, both gradient paths, differential testing.

- `overtone-sim`: SoA state vector, gate set `RX RY RZ H CZ CNOT` plus arbitrary
  single-qubit unitary, expectation values for Pauli observables.
- Adjoint gradients — constant memory in circuit depth, all parameters in ~two passes.
- Parameter-shift gradients — exact, `O(P)`, hardware-honest.
- Oracles: a dense Kronecker reference sharing no code with the strided kernels; an
  independent NumPy oracle; Yao.jl once Julia is available.

**Exit criteria**
- Adjoint and parameter-shift agree to `1e-10` on randomised circuits, every gate type.
- Strided kernels match the dense Kronecker reference to `1e-12`.
- Norm preserved to `1e-12` under every unitary.
- Seeded runs reproduce bit-for-bit.
- `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test` all green.

**Gate:** do not start Phase 2 until both gradient paths agree. Part I §14 — "Nothing
renders until the gradients are right."

---

## Phase 2 — RL loop, headless  *(M2)*

**Spec:** Part I §6.2, §6.3, §7.1.

- Data re-uploading ansatz, `L` layers, trainable-or-pinned input scaling `λ`.
- RAW-PQC (Born rule, strictly band-limited) and SOFTMAX-PQC (trainable output weights).
- `SpectralControl-k` contextual bandit; REINFORCE with baseline.
- LP ceiling solver — maximise the frequency-`k` Fourier coefficient over degree-`L` trig
  polynomials with `|p| ≤ 1`. Solve numerically; do not guess a closed form.
- Native CLI trains and emits JSONL traces.

**Exit criteria**
- The `L < k` zero-return result reproduces, seeded, in under 30 seconds: train 5k episodes
  at `L = 2, k = 3`, assert `|J| < 0.02`.
- Trained agents land on the LP ceiling staircase `J*(L)`.
- Trainable `λ` dissolves the ceiling: `L = 1` reaches `k = 3`.

---

## Phase 3 — Instrumentation  *(M3)*

**Spec:** Part I §6.5, §6.6, §6.7.

The crown jewel. `overtone-spec` must make each of the five Part I §1 phenomena measurable
by a function that returns numbers, each with a test.

- Fourier extraction: sample the logit function on `N = 512` over `[-π, π)`, real FFT.
- Von Neumann entropy across the half-chain bipartition; `--no-entangle` ablation.
- Gradient-variance sweep, `n = 2…12`, global vs local observable, fitted exponent.

**Exit criteria**
- No spectral energy beyond `L` for RAW-PQC, to `1e-9`.
- SOFTMAX-PQC produces measurable energy at `3L`. **Verify numerically before writing the
  claim down** (Part I §12) — if the numbers disagree with the story, change the story.
- Fitted plateau exponent matches `Var ~ 2^(-cn)` for the global observable; local survives.

---

## Phase 4 — The browser  *(M4, M15)*

**Spec:** Part I §8; Part IV §1.

WASM bindings, the six panels, the two-zone notebook/instrument design. Deploy to a
Hugging Face **Static Space** (free for everyone; compute Spaces are not) mirrored on
GitHub Pages.

**Exit criteria**
- 60fps at `n = 6, L = 4` on a mid-range laptop.
- The hero loop is the real engine, not a recording.
- JS under 800 lines, enforced by `scripts/check_js_budget.sh`.
- WASM and native produce identical seeded trajectories.
- Re-verify Hugging Face's tier documentation at deploy time; it has changed before.

---

## Phase 5 — The lattice  *(M6–M9)*

**Spec:** Part II.

`overtone-walk` and `overtone-maze`. Sparse infinite lattice with a strict light cone,
procedural mazes from a seeded coordinate hash, DTQW with Hadamard and Grover coins,
trajectory-based decoherence, the designed dark corridor, Anderson localization, glued
trees, Szegedy hitting time, and finally the learned coin.

**Exit criteria**
- Fitted spreading exponents `1.00 ± 0.03` (quantum) and `0.50 ± 0.03` (classical).
- Infinite sparse lattice matches a far-boundaried finite lattice to `1e-12`.
- Designed dark corridor below `1e-6`; full dephasing recovers the classical walk to
  TV `< 1e-3`.
- Learned coin benchmarked against Grover, Hadamard, and classical — **publish either way.**

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

## Cross-cutting, from the start

- **No emoji**, anywhere.
- Every README claim is paired with a CI test that fails when the claim stops being true.
- Negative results are published, not buried.
- Determinism: seed everything; native and WASM agree.
- Scope discipline: Part III §12 governs.
