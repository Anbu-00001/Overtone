# OVERTONE — build spec

**Repo:** `overtone` · **Tagline:** *Watch a quantum agent tune itself into resonance with its environment.*

*(Name collides softly with the Clojure audio lib `overtone`. Acceptable — different ecosystem, different topic. Alternates if you want a clean namespace: `resonant`, `bandlimit`, `harmonic-agents`. Pick one at M0 and never rename.)*

This document is the complete brief. Read all of it before writing code. Sections 1–3 are the *why*; deviate from them and the project becomes another quantum-RL notebook with a reward curve, of which GitHub already has forty.

---

## 1. The one idea this repo exists to show

A variational quantum circuit that encodes classical data is, exactly and provably, **a truncated Fourier series in that data**. The frequencies it can reach are fixed by the eigenvalues of the data-encoding gates; the coefficients are set by everything else in the circuit. This is Schuld, Sweke & Meyer (*Phys. Rev. A* 103, 032430, 2021).

Nobody has taken that theorem and pointed it at a **reinforcement learning policy in real time**. That is the entire project.

When you do, five things become visible that no QRL repo currently shows:

1. **A quantum policy has a hard frequency ceiling.** Encode the observation with `L` repeated Pauli rotations and the policy's reachable spectrum is the integers `{-L … L}`. Nothing outside it. Ever.
2. **You can therefore build an environment the agent provably cannot solve** — and then show it failing at exactly the predicted value, not "somewhat worse."
3. **You can watch the agent fix this itself.** Give the encoding a trainable input scaling `λ` and the reachable frequencies become `λ·{-L…L}` — continuous. The agent learns `λ`. On the spectrum panel you literally watch a peak slide along the frequency axis until it locks onto the environment's frequency. That is the money shot of the whole repo.
4. **You can see why `SOFTMAX-PQC` beats `RAW-PQC`.** Jerbi et al. (NeurIPS 2021) found this empirically and explained it as "adjustable greediness." The spectral view gives a sharper reason: the Born-rule policy is *strictly* band-limited, while the softmax is a nonlinearity applied to a band-limited function, and `tanh` of a degree-`L` trig polynomial generates odd harmonics at `3L`, `5L`, … So the softmax policy **leaks past its own circuit's frequency ceiling**. You can watch energy appear at frequencies that are not in the circuit. As far as our search found, this framing is not in the literature. It is not a large claim, but it is a real one, and it is the repo's original contribution.
5. **Barren plateaus become a slider.** Drag qubit count 2→12, watch gradient variance fall off a log-scale cliff, watch it *not* fall when the observable is local (Cerezo et al. 2021). Three seconds to understand what usually takes a paper.

Everything else in this spec — the language choices, the renderer, the panels — is in service of making those five things legible.

---

## 2. What already exists (so we don't rebuild it)

| Thing | State of the art | Gap we fill |
|---|---|---|
| QRL repos on GitHub | Best is ~29★ (`qlan3/QuantumExplorer`, PyTorch+PennyLane). Nearly all are Jupyter notebooks producing reward curves. | No instrumentation of *what the quantum model is doing*. No live demo. |
| Quantum sim in Rust | Several exist (`quantrs2-sim`, `q1tsim`, `qasmsim`, `qukit` — the last two already WASM). All general-purpose. | None have autodiff for RL, none have an RL loop, none have spectral instrumentation. |
| Quantum sim in Julia | `Yao.jl` is genuinely state-of-the-art: QBIR representation, reverse-mode AD that exploits circuit reversibility (constant memory in depth), CUDA backend, batched registers. Outperforms PennyLane and qulacs on small-to-mid circuits. | Almost nobody uses it for RL. Also: it makes an *excellent* correctness oracle. |
| Fourier-spectrum visualisation | PennyLane has a static tutorial demo for supervised learning. | Nothing live, nothing in RL, nothing showing the softmax leakage. |

Conclusion: the simulator is not the contribution. The **instrumentation** is.

---

## 3. Non-negotiables

Rules that make this feel built rather than generated. If a decision is ambiguous, resolve it against these.

- **Every number on screen is computed live in the browser by the real engine.** No pre-recorded traces, no cached JSON masquerading as a simulation, no fake loading state. If a panel can't run at 60fps, cut the panel — don't fake it.
- **No toy visuals.** No cartoon cart-pole, no animated pendulum sprite. The environment is rendered as *the function the agent has to learn*, drawn as a curve or heatmap. The reader should be looking at mathematics, not at a game.
- **The demo runs with zero installs.** GitHub Pages, one URL, works on a phone. This is the single largest determinant of whether the repo gets stars. Everything else is downstream of it.
- **Claims are falsifiable and tested.** Every quantitative statement in the README has a corresponding test in CI that fails if the claim stops being true.
- **Report negative results.** If the quantum policy doesn't beat the classical MLP on parameter count, say so on the front page. Credibility is the scarce resource in this field, and the repos that hype are the repos nobody trusts.
- **No emoji anywhere.** Not in the README, not in commit messages, not in the UI.

---

## 4. Language stack, and the honest reasoning

The brief asked about Julia, Zig, Mojo, C++, Rust. Here is the actual assessment — do not shoehorn a language in for novelty.

**Rust — yes, this is the core.** State-vector simulation is a tight loop over strided complex arrays: the exact workload where Rust matches C++ and beats everything else that also compiles cleanly to WASM. `wasm-bindgen` + `wasm-pack` gives us the zero-install demo from the same source as the native trainer. This is the whole reason the architecture works.

**Julia — yes, as oracle and laboratory.** `Yao.jl` is used two ways, both load-bearing:
- **Differential testing.** Every gate, every gradient, every expectation value in the Rust core is checked against Yao to `1e-12`. A README badge reading *"gradients verified against Yao.jl"* is worth more than a paragraph of prose.
- **Heavy sweeps.** The barren-plateau and spectral-ceiling figures need thousands of circuit evaluations across qubit counts. Yao's batched CUDA registers do this in minutes. Rust does the interactive path; Julia does the science.

**C++ — only if benchmarks demand it.** Optional AVX-512 kernel for the gate inner loop, behind a feature flag, only after profiling shows Rust's autovectorisation leaving >20% on the table. Do not start here.

**Zig — no.** No quantum ecosystem, no autodiff, no BLAS story. Including it would be decoration. Say so in the README's "why these languages" section; the honesty reads well.

**Mojo — no.** Not open enough or stable enough to be a dependency in a repo meant to be cloned and run by strangers. Same treatment: mention, explain, decline.

**JavaScript — kept deliberately dumb.** The web layer is a renderer and nothing else: vanilla JS, Canvas2D and one WebGL2 shader, no framework, no bundler beyond `wasm-pack`. Hard budget: **under 800 lines of JS**, enforced by a CI check. If logic starts migrating into JS, it belongs in Rust.

---

## 5. Architecture

```
overtone/
├── crates/
│   ├── overtone-sim/       state vector, gates, adjoint + parameter-shift gradients
│   ├── overtone-rl/        environments, policies, REINFORCE/PPO, rollout buffers
│   ├── overtone-spec/      the instrumentation: Fourier, entropy, gradient variance, LP ceiling
│   ├── overtone-cli/       native trainer; emits JSONL traces
│   └── overtone-wasm/      wasm-bindgen surface for the browser
├── lab/                    Julia: OvertoneLab.jl — Yao.jl oracle + sweeps + Makie figures
├── web/                    index.html, ~6 canvas modules, one stylesheet. No framework.
├── figures/                generated by lab/, committed, used in README
└── docs/                   the written explanation (this is a teaching repo)
```

**Data flow.** `overtone-sim` knows nothing about RL. `overtone-rl` knows nothing about rendering. `overtone-spec` reads circuits and policies and emits measurements. `overtone-wasm` is a thin FFI shim — if it contains an `if` statement about physics, that logic is in the wrong crate.

**Determinism.** Seed everything (`rand_chacha`). Same seed, same trajectory, native and WASM. A test asserts this. It matters because the demo lets people share a permalink to a run.

---

## 6. The math to implement, precisely

### 6.1 State vector (`overtone-sim`)

Structure-of-arrays: `re: Vec<f64>`, `im: Vec<f64>`, length `2^n`. Not `Vec<Complex64>` — SoA autovectorises, AoS doesn't. Target `n ≤ 16` interactive, `n ≤ 24` native.

Single-qubit gate on qubit `q`: iterate pairs `(i, i | 1<<q)` for `i` with bit `q` clear. Two-qubit controlled gates: same with a control mask test. Use `rayon` on the native build above `n = 12`; single-threaded in WASM (rayon-in-wasm needs cross-origin isolation headers, not worth the deployment friction — revisit only if profiling demands it).

Gate set: `RX RY RZ H CZ CNOT` plus arbitrary single-qubit unitary. That is sufficient; do not gold-plate.

### 6.2 The ansatz

Data re-uploading, `L` layers. Layer `ℓ`:

```
encode:     for each qubit i:  RX(λ[ℓ,i] · s[i])          λ trainable (default) or pinned to 1
variational: for each qubit i:  RZ(θ[ℓ,i,0]) RY(θ[ℓ,i,1])
entangle:   ring of CZ                                     ablatable to identity via --no-entangle
```

`λ` being trainable vs pinned is not a detail — it is the difference between a fixed frequency comb and a tunable one, and toggling it is the demo in §1.3. Expose it as a first-class switch everywhere: CLI flag, WASM parameter, UI toggle.

### 6.3 Policies

**RAW-PQC** — Born rule. Partition computational basis states into `|A|` sets; `π(a|s) = ⟨P_a⟩`. For two actions with a `Z` observable on qubit 0: `π(1|s) = (1 + ⟨Z₀⟩)/2`. **Strictly band-limited.** This is the scientifically clean policy.

**SOFTMAX-PQC** — `π(a|s) = softmax_a(w_a · ⟨O_a⟩_{s,θ})` with trainable output weights `w`. Empirically stronger (Jerbi et al.); spectrally leaky (§1.4). This is the practically useful policy.

Implement both. The comparison *is* the result.

### 6.4 Gradients — two paths that must agree

**Adjoint (simulation, fast).** Forward to `|ψ⟩`; set `|λ⟩ = O|ψ⟩`; walk backwards un-applying gates from both; at each parameterised gate accumulate `∂⟨O⟩/∂θ = 2·Re⟨λ|∂U|ψ⟩`. Constant memory in circuit depth, all parameters in roughly two forward passes. This is what Yao does and why it's fast.

**Parameter-shift (hardware-honest, slow).** For a gate `U(θ) = exp(-iθG/2)` with `G² = I`:

```
∂⟨M⟩/∂θ = ½ [ ⟨M⟩(θ + π/2) − ⟨M⟩(θ − π/2) ]
```

Exact, not a finite difference. Two circuit evaluations per parameter, so `O(P)` cost.

**Test:** both agree to `1e-10` on randomised circuits, for every gate type. Ship this as a UI panel too — a scatter of adjoint-gradient vs shift-gradient landing on `y = x` is a small, satisfying thing to look at, and it silently tells a reader the engine is trustworthy.

### 6.5 The spectral instrument (`overtone-spec`) — the crown jewel

Given a trained policy and a 1-D observation axis:

1. Sample the policy's logit function `h(s)` on a uniform grid over `[-π, π)`, `N = 512`.
2. Real FFT → coefficients `c_ω`.
3. Render `|c_ω|` as a bar spectrum, with a vertical rule at the **theoretical ceiling** `ω = L` (or `λ·L` when scaling is trainable).

For a RAW-PQC with `λ = 1`, no energy may appear beyond `L` — assert this in a test, to `1e-9`. For SOFTMAX-PQC, energy *will* appear at odd harmonics. Render leaked bars in a different treatment so the eye catches them immediately. That contrast, live, is the repo's original observation.

2-D observations (Pendulum): 2-D FFT, render as a lattice of dots on the `(ω₁, ω₂)` integer grid, dot area ∝ `|c|`. The reachable region is a diamond `|ω₁| + |ω₂| ≤ L`; draw its boundary.

### 6.6 Entanglement

Von Neumann entropy `S = −Tr(ρ_A log ρ_A)` across the half-chain bipartition, from the SVD of the reshaped state vector. Track over training.

Then pose the question the repo should honestly ask: **does a good policy actually need entanglement?** Run the `--no-entangle` ablation. There is published evidence that unentangled agents do fine on Gym tasks (arXiv:2203.14348). If our ablation agrees, report it. A repo that deflates its own field's hype is a repo people trust.

### 6.7 Barren plateaus

Sample `M = 200` random parameter vectors, compute `∂⟨O⟩/∂θ₁`, take the variance, sweep `n = 2…12`. Plot on log-y. Two curves: **global** observable (`Z⊗Z⊗…⊗Z`) and **local** (`Z₀`). Global collapses exponentially; local survives at shallow depth. This is McClean et al. 2018 and Cerezo et al. 2021, rendered as a slider.

Fit the exponent live and print it. Seeing `Var ~ 2^(−1.98n)` appear next to the curve is more convincing than the curve.

---

## 7. Environments

Three, in order of importance. Keep the "toy" element minimal — these are rendered as functions, not as games.

### 7.1 `SpectralControl-k` — the original one, and the star

A contextual bandit. Call it that in the docs; don't dress it up as more than it is.

- State `s ~ Uniform[-π, π)`, drawn i.i.d.
- Actions `a ∈ {0, 1}`
- Reward `r(s, a) = (2a − 1) · cos(k·s)`

**Why it matters.** Expected return decomposes as `J = E_s[ cos(ks) · (2π(1|s) − 1) ]`. For a RAW-PQC, `2π(1|s) − 1 = ⟨Z₀⟩_s`, which is a degree-`L` trig polynomial. The expectation picks out exactly its frequency-`k` Fourier coefficient. Therefore:

> **With `λ` pinned to 1 and `L < k`, a RAW-PQC agent scores exactly 0 — chance — forever.** Not "worse." Zero. No amount of training, qubits, depth, or entanglement changes it.

That is an unusually crisp prediction for an ML repo, and it is testable in CI: train for 5k episodes at `L = 2, k = 3`, assert `|J| < 0.02`.

**The ceiling curve for `L ≥ k`.** Maximise `E[cos(ks)·p(s)]` over degree-`L` trig polynomials with `|p| ≤ 1`. Since only `p`'s frequency-`k` component contributes, this reduces to maximising that one coefficient subject to a sup-norm constraint — a small linear program on a fine grid of `s`. Solve it numerically (`good_lp` in Rust, or JuMP in the Julia lab), don't guess a closed form. The result is a **staircase** `J*(L)` rising toward `2/π ≈ 0.6366`. Draw it behind the learning curve. Agents should land on the steps.

**Then turn on trainable `λ`** and the ceiling dissolves: reachable frequencies become `λ·{-L…L}`, `λ` is continuous, so `L = 1` suffices to reach `k`. On the spectrum panel the single peak *slides* until it lands on `k`. Sonically this is a string being tuned. Say so — it is the best available intuition and it is not a metaphor, it is what is happening.

### 7.2 `Pendulum` (2-D observation)

Standard swing-up, discretised torque. Observation `(θ, θ̇)`. The point is the 2-D spectrum lattice and the policy heatmap over the cylinder. Render the policy surface, not a swinging stick.

### 7.3 `CartPole`

Included only so people can compare against every other QRL repo. One paragraph in the README, one figure, no UI panel. It is a reference point, not a feature.

---

## 8. The interface

**Design brief:** this is a physics instrument with a lab notebook attached. The reader is a curious engineer or a physics-adjacent ML person who arrived from Hacker News and will give it about eleven seconds.

### 8.1 Two-zone layout, and why

The page is a **notebook** — light ground, prose, real typography — with **instrument screens** embedded in it: dark, dense, live. This mirrors how the subject is actually practised (paper and oscilloscope) and it sidesteps the two clichés this kind of project falls into: the wall-to-wall dark dashboard with one acid accent, and the cream-and-terracotta essay.

```
┌───────────────────────────────────────────────────────────┐
│  [ HERO: one instrument, running, no controls, no copy ]   │  ← full-bleed dark
│                                                            │
├───────────────────────────────────────────────────────────┤
│   prose column (62ch)          │  margin: small live       │  ← light ground
│   the theorem, in plain words  │  spectra, inline, tiny    │
├────────────────────────────────┴──────────────────────────┤
│  [ INSTRUMENT: circuit │ spectrum │ policy │ return ]      │  ← dark, interactive
│  controls sit inside the dark zone, never floating in prose│
├───────────────────────────────────────────────────────────┤
│   prose: the softmax leakage result                       │
├───────────────────────────────────────────────────────────┤
│  [ INSTRUMENT: barren plateau sweep ]                     │
└───────────────────────────────────────────────────────────┘
```

### 8.2 Colour comes from the physics

Do not choose a palette. Derive one.

- **Phase is cyclic**, so phase is hue — a perceptually-uniform cyclic map (twilight / CET-C2). Never a linear ramp; a linear ramp on a cyclic quantity is a lie and physicists will notice.
- **Amplitude is magnitude**, so amplitude is lightness.
- The legend is therefore literally the complex unit disc. Draw it once, small, near the first state-vector panel.
- Notebook ground: warm neutral paper, around `#F7F5F1` — but do **not** pair it with a terracotta accent; that combination is the current generated-page signature.
- Instrument ground: `#12151A`, a genuinely blue-shifted dark, not tinted near-black standing in for black.
- Exactly **one** non-derived accent, used only for the human's own interactions (slider handles, focus rings). Everything else is coloured by what it means.

### 8.3 Type

Two families, clearly distinct:
- **Prose:** a Dutch old-style or transitional serif with a real italic. Source Serif 4 or Spectral. Generous leading, ≤ 62 characters per line.
- **Numerics, gates, parameters:** one engineering monospace — IBM Plex Mono. Tabular figures on, everywhere a number updates, so digits don't jitter.

No all-caps labels. No eyebrow labels above headings. No `→` glued to link text. Let the type set the tone by its choice, not by decoration.

### 8.4 The hero

Not a headline. The hero is a twelve-second loop of the real engine: a `SpectralControl-3` agent with trainable `λ`, spectrum panel beside it, one peak sliding across the frequency axis and locking on. No controls, no copy, no autoplaying explanation. Then one line of text beneath it, and the reader either scrolls or leaves.

This is the only non-user-triggered motion on the page. Everywhere else, motion answers an action.

### 8.5 Panels

| Panel | Shows | Interaction |
|---|---|---|
| Circuit | Gates as a grid, wire per qubit, each parameterised gate tinted by its current gradient magnitude | Hover a gate → its parameter's history sparkline |
| State | Amplitude bars, hue = phase | Step through gates one at a time |
| **Spectrum** | `\|c_ω\|` bars, ceiling rule, leaked bars distinguished | Drag `L`; toggle `λ` trainable; toggle RAW/SOFTMAX |
| Policy | `π(1\|s)` over `s`, with the optimal policy ghosted behind | — |
| Return | Learning curve with the LP ceiling staircase drawn behind | — |
| Plateau | `Var[∂⟨O⟩]` vs qubits, log-y, global and local | Drag qubits, drag depth |
| Gradient check | Adjoint vs parameter-shift scatter on `y = x` | — |

### 8.6 Quality floor

Responsive to 360px (the spectrum panel is the one that must survive; others may stack or drop). Keyboard focus visible. `prefers-reduced-motion` kills the hero loop and shows its final frame. Contrast passes AA on both grounds. None of this gets announced in the UI.

---

## 9. What the README claims, and how CI enforces it

Write the README last, from measured numbers. Every claim below is paired with a test that fails if it stops being true.

| Claim | Test |
|---|---|
| Gradients match Yao.jl to `1e-12` | `lab/test/oracle.jl`, run in CI on a Julia job |
| Adjoint and parameter-shift agree to `1e-10` | `crates/overtone-sim/tests/gradients.rs` |
| RAW-PQC with `L < k` scores 0 on `SpectralControl-k` | `crates/overtone-rl/tests/spectral_ceiling.rs`, seeded |
| No spectral energy beyond `L` for RAW-PQC | `crates/overtone-spec/tests/bandlimit.rs` |
| SOFTMAX-PQC produces energy at `3L` | same file |
| Gradient variance decays exponentially in `n` for a global observable | `crates/overtone-spec/tests/plateau.rs`, asserts fitted exponent |
| Parameter count at matched return vs a classical MLP | benchmark script; **publish the number whichever way it falls** |
| WASM and native produce identical seeded trajectories | integration test |
| JS under 800 lines | `scripts/check_js_budget.sh` |

---

## 10. Milestones

**M0 — skeleton.** Workspace, CI (fmt, clippy, test, wasm build), licence (Apache-2.0 or MIT), name locked. Empty crates that compile.

**M1 — simulator + oracle.** Gates, state vector, expectation values. Adjoint and parameter-shift gradients. Yao.jl oracle green. *This is the foundation; do not proceed until both gradient paths agree.*

**M2 — RL loop, headless.** `SpectralControl-k`, RAW and SOFTMAX policies, REINFORCE with baseline. LP ceiling solver. Native CLI trains and emits JSONL. Acceptance: the `L < k` zero-return result reproduces, seeded, in under 30 seconds.

**M3 — instrumentation.** Fourier extraction, entropy, gradient-variance sweep. Julia lab generates the static figures for the README with Makie.

**M4 — the browser.** WASM bindings, six panels, the design in §8. Deploy to GitHub Pages. Acceptance: 60fps at `n = 6, L = 4` on a mid-range laptop; the hero loop is the real engine.

**M5 — Pendulum, CartPole, docs, and the writeup.** The `docs/` explanation is a first-class deliverable, not an afterthought — it is why the repo gets shared.

---

## 11. Suggested agent decomposition for Claude Code

These four workstreams have clean interfaces and can run largely in parallel after M1. Give each subagent this document plus its own charter; require each to write its own tests.

- **`sim`** — owns `overtone-sim`. Charter: numerical correctness and speed. Success = Yao oracle green and a published benchmark table against `quantrs2-sim`. Should not know the word "policy."
- **`rl`** — owns `overtone-rl` and the LP ceiling solver. Charter: the `SpectralControl` result reproduces deterministically. Consumes `sim` through a trait; must not reach into its internals.
- **`spec`** — owns `overtone-spec` and the Julia lab. Charter: the five phenomena in §1 are each measurable by a function that returns numbers, and each has a test. This agent is doing science, not plumbing; give it room.
- **`face`** — owns `web/` and the design system. Charter: §8, and the JS budget. Should be handed a *finished, stable* WASM API and told not to negotiate physics.

A fifth pass at the end — **`critic`** — reads the whole repo cold and answers one question: *would a physicist trust this, and would an engineer star it?* Act on what it says.

Sequencing note: `spec` depends on `sim` for gradients but not for the FFT work, so it can start on the spectral extractor against a mock immediately. `face` should not start before M3 is stable, or it will build against a moving API.

---

## 12. Traps

- **Don't let the visualiser drive the physics.** If a panel wants a quantity that isn't physically meaningful, cut the panel.
- **Don't add environments.** Three is already one more than needed. Depth over breadth is the entire positioning.
- **Don't oversell.** No "quantum advantage" language anywhere. The honest claim is *"quantum policies are band-limited function approximators, and here is what that means in practice"* — which is more interesting than a speedup claim nobody believes.
- **Don't put the demo behind a build step.** If a reader has to run `npm install`, the reader is gone.
- **Don't skip the LP ceiling.** It is what converts a nice visualisation into a result. A learning curve that lands exactly on a theoretically predicted staircase is the single most persuasive image the repo can produce.
- **Don't let `λ` be a hidden hyperparameter.** It is a protagonist. Surface it everywhere.
- **Watch the softmax-leakage claim carefully.** Verify numerically before writing it down: confirm the harmonics appear at the predicted orders and that they vanish when the policy is RAW. If the numbers disagree with the story, change the story.

---

## 13. References to keep open

- Schuld, Sweke, Meyer — *Effect of data encoding on the expressive power of variational quantum-machine-learning models*, Phys. Rev. A 103, 032430 (2021), arXiv:2008.08605. **The theorem the repo is built on.**
- Jerbi, Gyurik, Marshall, Briegel, Dunjko — *Parametrized quantum policies for reinforcement learning*, NeurIPS 34 (2021), arXiv:2103.05577. RAW-PQC / SOFTMAX-PQC, trainable scaling.
- Skolik, Jerbi, Dunjko — *Quantum agents in the gym*, Quantum 6, 720 (2022), arXiv:2103.15084. Deep Q-learning with PQCs; reference implementation at `askolik/quantum_agents`.
- McClean, Boixo, Smelyanskiy, Babbush, Neven — *Barren plateaus in quantum neural network training landscapes*, Nat. Commun. 9, 4812 (2018).
- Cerezo et al. — *Cost function dependent barren plateaus in shallow parametrized quantum circuits*, Nat. Commun. 12, 1791 (2021). The global-vs-local distinction the plateau panel shows.
- Luo, Liu, Zhang, Wang — *Yao.jl: Extensible, Efficient Framework for Quantum Algorithm Design*, Quantum 4, 341 (2020).
- arXiv:2203.14348 — *Unentangled quantum reinforcement learning agents in the OpenAI Gym*. Relevant to the entanglement ablation in §6.6.

---

## 14. First command

```
cargo new --lib crates/overtone-sim
```

Then M1. Nothing renders until the gradients are right.
