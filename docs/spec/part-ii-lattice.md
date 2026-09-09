# OVERTONE PART II — The Lattice

**Companion to `overtone-build-spec.md`.** Part I stands as written; nothing here replaces it. This adds a second mode to the same application, sharing the same engine, the same gradient machinery, and the same colour law.

**Tab name in the existing nav:** `Lattice`, sitting beside `Lab`.

---

## Corrections

**The body below is unchanged.** Where measurement or a later ruling has contradicted this
spec, the correction is recorded here rather than edited into the text, so that what was
believed at the time stays readable and the disagreement stays visible. The full reasoning is
in `docs/PHASES.md`; the operational form is in `CLAUDE.md`.

| Section | Correction | Source |
|---|---|---|
| §4 P5 | The `(4n+2)` column reduction is not a rendering choice. It is the literal computation the demo runs: at `n = 10` the welded tree's 4094 vertices and 12 280 arcs become a 42-wide vector. | Decisions-03 Q7.5; Phase 14 |
| §4 P6 | Three different things are called hitting time — one-shot, concurrent, and marked-vertex search. This section quotes Szegedy's `√HT` **search** result and applies it to entrance-to-exit **traversal**. Those are different problems and the square root does not carry across. | Decisions-03 §3; Phase 14 |
| §4 P7 | A learned, position-conditioned coin is exactly the "arbitrary position-dependent coin" that the coined/Szegedy equivalence excludes. It inherits **no** quadratic hitting-time guarantee. Any win over the Grover coin is an empirical finding needing its own justification. | Decisions-03 Q7.2; open (M9) |
| §6 | The Szegedy module is deleted, not deferred. Two coined applications are one Szegedy application under a Grover coin and a flip-flop shift, so there was never a second engine to write. | Decisions-03 Q7.2 |
| §7 M8 | The acceptance clause "the `√HT` slope fits" is unsatisfiable as written, for the P6 reason above. Replaced by Kempe's one-shot hitting on the hypercube against the exact classical `2^n` — the same claim (a provable separation in traversal time) with a theorem that is about traversal. | Decisions-03 §3; Phase 14 |
| §9 | The reference frame the section is missing: the walk's *coin* can cage it. With the DFT coin on the 4-cube exactly `3/7` of the amplitude never reaches the far corner, on a graph with no walls at all. | Krovi & Brun; Phase 14 |

---

## 1. The reframe: this is not a maze game

The request was an endless maze with agents exploring it. The version of that idea worth building is not a sprite in a corridor. It is this:

> A quantum walker does not explore a maze. It floods it.

A classical random walk spreads **diffusively** — standard deviation `σ ∝ √t`. A quantum walk spreads **ballistically** — `σ ∝ t`. That is a quadratic separation, it is provable, and it is *spectacularly* visible. Put the two side by side in the same maze with the same step budget and the classical pane shows a grey blob inching outward while the quantum pane shows a wavefront racing to the causal boundary with interference fringes trailing behind it.

Nothing in this module is rendered as a character. There is no sprite, no goal flag, no score. There is a **probability field flowing through a graph**, and the maze walls are hairline substrate. The reader is looking at physics that happens to be maze-shaped.

Everything else in this document follows from that decision.

---

## 2. How this rhymes with Part I

Part I: the policy is a variational quantum circuit acting on the observation.

Part II: **the policy is the coin.**

In a discrete-time quantum walk, each vertex carries a *coin* operator — a small unitary on the direction space that decides how amplitude distributes among neighbours. That is exactly the object a policy is. So:

```
Part I:   s ──[ encode ]──[ VQC θ ]──[ measure ]──> π(a|s)
Part II:  local maze features ──[ encode ]──[ VQC θ ]──> C(θ; features), the coin
```

Same encoder, same ansatz, same adjoint gradients, same parameter-shift check. One engine, two faces. And the Part I instrumentation carries straight over with new meaning: **the coin's Fourier spectrum tells you which local wall-patterns the walker can distinguish.** A band-limited coin is a walker that is blind to fine maze structure. That is a nice, unforced continuation of the Part I result, and it should be surfaced in the UI as a small inline spectrum next to the Lattice panel.

**The conceptual payload of Part II**, stated plainly for the README:

> In classical RL you tune `ε` to control exploration. Here you tune **coherence length** — how many steps the walker runs before the position is measured. Coherence *is* the exploration mechanism. Turn it down and you recover an ε-greedy random walk. Turn it up and you get ballistic flooding with interference.

That is a real idea, it is cheap to implement, and it is one slider.

---

## 3. "Endless", done exactly

The obvious approach — simulate a big finite maze and pretend — is not necessary. There is an exact method, and it is the nicest engineering idea in this module.

**A discrete-time quantum walk has a strict light cone.** One step moves amplitude to nearest neighbours only. After `t` steps the support of the wavefunction has radius at most `t`. There is no tail beyond it — not small, *zero*.

Therefore:

- Represent the state as a **sparse map** from `(cell, direction) → amplitude`. `HashMap` or, better, a hash grid over `i32` coordinates.
- Grow the support as the wavefront advances. Memory is `O(t²)` in 2D, which is fine for the thousands of steps a browser session will run.
- Generate the maze **procedurally from a seeded hash of the coordinates** — no global maze object, no allocation ahead of time. `wall(x, y, dir) = hash(seed, x, y, dir) < threshold`. Infinite, deterministic, and a URL fragment carrying `(seed, rule)` reproduces someone else's maze exactly.

The result: a genuinely infinite lattice, simulated with **zero truncation error**. Not an approximation dressed up as one. Say this in the README — it is the kind of detail that makes a reader trust everything else.

Draw the light cone. A thin ring at radius `t` around the origin, with the quantum wavefront pressed right against it and the classical blob nowhere near. That single visual explains ballistic-versus-diffusive without a word of text.

**CI test:** results on the infinite sparse lattice must match a large finite lattice with absorbing boundaries placed far outside the light cone, to `1e-12`.

---

## 4. The seven phenomena

Each is a panel. Build them in this order; the first three carry the module.

### P1 — Ballistic vs diffusive *(the hero)*

Two panes, same procedurally generated maze, same step budget, side by side. Quantum left, classical right.

Inset: `σ(t)` on log-log for both, with **live-fitted exponents printed as numbers**. You want the reader to see `1.00` and `0.50` appear on screen. The fitted number is more persuasive than the curve.

Rendering rule that does the explaining for you: the classical pane is **greyscale** (probability has no phase); the quantum pane is **coloured by phase**. The colour difference *is* the physics, and it needs no legend.

### P2 — Dark corridors *(the one people will screenshot)*

Interference is the thing a classical walker cannot do, and it is invisible in a spreading-rate plot. So make it a designed object.

Build a maze cell where two paths of different lengths rejoin at a junction. The amplitudes arrive with a relative phase set by the path-length difference. Tune it and they cancel: **a corridor the quantum walker never enters, that a classical walker enters with high probability.**

Ship a small **corridor designer**: two sliders for the two path lengths, a live readout of the relative phase, and the destructive point going dark as you cross it. This is the module's most memorable ten seconds.

**CI test:** at the designed destructive geometry, probability at the target site `< 1e-6`.

### P3 — The coherence dial

One slider, `p` = dephasing probability per step. `p = 0` is a fully coherent quantum walk; `p = 1` is exactly a classical random walk; in between is the crossover.

**Implement with quantum trajectories, not density matrices.** A density matrix is `N²` and destroys the sparse infinite-lattice representation. Instead: sample many pure trajectories, each with random measurement events at rate `p`, and average. This keeps the sparse structure, parallelises trivially, and is physically the same thing.

The non-obvious payoff, and the reason this panel earns its place: **a small amount of decoherence sometimes helps.** It flattens the distribution toward uniform while retaining much of the ballistic spread, which improves mixing. There is a sweet spot on the dial. Let the reader find it themselves; do not label it.

**CI test:** at `p = 1`, the distribution matches the classical random walk to total-variation distance `< 1e-3`.

### P4 — Anderson localization *(the honest counterweight)*

Add **static** disorder — a fixed random phase per cell, strength `W`. The quantum walker localizes exponentially and **stops**: `σ(t)` saturates to a constant. The classical walker keeps diffusing straight past it.

This is the panel that stops the module being a quantum-is-better advertisement. Static disorder kills the quantum advantage completely, and that is a true and important thing about quantum walks that most popular writing on the subject omits. Note the contrast with P3: *dynamic* disorder (decoherence) produces a quantum-to-classical crossover, while *static* disorder produces localization. Same word, opposite outcomes. Put both sliders on the same panel so the reader can feel the difference.

### P5 — Glued trees *(the provable case)*

Two complete binary trees of depth `d`, joined at the leaves by a random cycle. Enter at one root, find the other. Classically this takes time exponential in `d` — the walker gets lost in the exponentially wide middle. A continuous-time quantum walk crosses in polynomial time. This is Childs, Cleve, Deotto, Farhi, Gutmann & Spielman (STOC 2003), and it is one of very few provable *exponential* quantum speedups.

It is also, structurally, a maze — which is why it belongs here rather than in a theory appendix.

**Rendering insight that makes this panel work:** by symmetry the walk reduces to a one-dimensional walk on the `2d+1` *columns* of the graph. That is the heart of the Childs proof and it renders beautifully — show the full tree faintly, and the column amplitudes as a 1D wavefront sweeping across it. An exponentially large graph collapsing to a line in front of the reader is a genuinely good piece of exposition.

**Required honesty note in the copy:** this is an oracular, artificial problem, and the practical realisability of the speedup under gate imprecision has been challenged in the literature. State it. A repo that volunteers the caveat is a repo people cite.

### P6 — Hitting time and √HT

Szegedy's quantization of a Markov chain detects a marked vertex in about the **square root of the classical hitting time**. Measure both on the same procedurally generated mazes across a range of sizes, plot quantum against classical on log-log, fit the slope.

**CI test:** fitted slope `≈ 0.5` within tolerance, on the graph families where the quadratic speedup is known to hold.

Caveat to carry in the copy: *detecting* a marked vertex and *finding* it are different problems, and the full quadratic speedup for finding took fifteen years and considerable machinery to establish in general. Do not blur them.

### P7 — The learned coin *(the RL part)*

Everything above is fixed-coin physics. This is where the agent enters.

**The model.** A parameterized coin `C(θ; f)` where `f` is a local feature vector — which walls are present, distance-to-goal sensors, local wall density. Encoded and processed by the Part I VQC. So the coin is a function of local maze structure, learned.

**The training loop.** Run the walk for `k` coherent steps, measure the position, receive reward, repeat. `k` is the coherence budget from §2, and it is a first-class hyperparameter:

- `k = 1` → measurement every step → an ordinary classical MDP with a stochastic policy. The quantum machinery reduces to a peculiar way of parameterizing a categorical distribution.
- `k` large → long coherent flooding between decisions → a small number of high-information decisions.

Reward: arrival probability at the marked cell within budget, or negative expected steps. Gradients: adjoint through the `k`-step unitary block (it is just a deep circuit — Part I's `overtone-sim` handles it unchanged), REINFORCE across measurement events.

**Baselines, all three mandatory:** the Grover coin, the Hadamard coin, and a plain classical random walk. Plot all four.

**The open question, stated as a question.** Does a learned, structure-conditioned coin beat the Grover coin on mazes with regularity? Analytically optimized SU(2) coins exist (Chandrashekar, Srikanth & Laflamme 2008) and RL has been used to pick coin *sequences* for entanglement generation (arXiv:2003.07141), but we found no prior work learning a **position-conditioned coin by policy gradient to minimize hitting time on a structured maze**. Verify that gap independently before claiming it in the README, and if the learned coin loses to Grover, **publish that**. A clean negative result here is worth more than a hedged positive one.

---

## 5. Rendering

Continuity with Part I is not optional — the two modes must look like the same instrument.

| Element | Treatment |
|---|---|
| Maze walls | Hairline, low contrast. Substrate, not subject. |
| Quantum field | Per cell: hue = phase (cyclic map), lightness = amplitude. Identical colour law to Part I's state panel. |
| Classical field | Same lightness law, **no hue**. Greyscale. The absence of colour is the information. |
| Light cone | Thin ring at radius `t`. The one non-data annotation allowed on the field. |
| Goal | A single ring on one cell. No flag, no checkerboard, no glow. |
| Camera | Follows the wavefront centroid; the lattice scrolls without bound. |
| Numbers | `σ_q`, `σ_c`, fitted exponents, step count, support size — monospace, tabular figures, always on screen. |

Motion budget: the walk animates because it is running, and that is the only motion. No easing flourishes, no particle effects, no trails beyond the physical amplitude.

**Cut list.** If the frame budget is tight, drop panels before dropping fidelity. Never interpolate frames to fake smoothness — a walk that visibly steps is honest; a walk that glides is a lie about a discrete-time process.

---

## 6. New crates

```
crates/
├── overtone-walk/     sparse infinite lattice; DTQW + CTQW; coins;
│                      trajectory-based decoherence; Szegedy walk
└── overtone-maze/     procedural generation from seeded hash; glued trees;
                       static/dynamic disorder; the corridor designer
```

`overtone-walk` depends on `overtone-sim` for gradients and on nothing else. `overtone-maze` is pure combinatorics and depends on nothing. Neither knows about rendering.

The Julia lab gains an oracle role here too: verify the sparse infinite-lattice evolution against a dense Yao.jl simulation on a small finite graph, and cross-check the glued-trees column reduction against direct evolution on the full graph.

---

## 7. Milestones

**M6 — the walk engine.** Sparse infinite lattice, DTQW with Hadamard and Grover coins, procedural maze. Acceptance: P1 reproduces, fitted exponents `1.00 ± 0.03` and `0.50 ± 0.03`, and the infinite-lattice truncation test passes.

**M7 — interference and decoherence.** P2 and P3. Acceptance: the designed dark corridor goes below `1e-6`; `p = 1` reproduces the classical walk to TV `< 1e-3`.

**M8 — the special graphs.** P4, P5, P6. Acceptance: localization saturates; glued-trees column reduction matches full-graph evolution; the √HT slope fits.

**M9 — the learned coin.** P7 with all three baselines and an honest results table.

Sequencing: M6 can start the moment Part I's M1 lands, since it only needs the simulator. It does **not** need Part I's RL loop. Run the two tracks in parallel — this is the natural place to split agents.

---

## 8. Claims and their tests

| Claim | Test |
|---|---|
| Quantum walk spreads as `σ ∝ t`, classical as `σ ∝ √t` | fitted exponents, `crates/overtone-walk/tests/spreading.rs` |
| Infinite sparse lattice is exact | match a far-boundaried finite lattice to `1e-12` |
| The designed dark corridor is dark | `< 1e-6` at the destructive site |
| Full dephasing recovers the classical walk | TV distance `< 1e-3` |
| Static disorder saturates `σ` | saturation detected within a fixed budget |
| Glued-trees column reduction is exact | match full-graph CTQW on small `d` |
| Quantum hitting time scales as `√HT` | fitted log-log slope, on the graph families where it is known to hold |
| Learned coin vs Grover coin | benchmark table; **publish the result either way** |

---

## 9. Traps

- **Do not sprite it.** The instant a character appears on screen the project becomes a game and loses the audience it was built for.
- **Do not say "the agent walks."** There is no walker. There is an amplitude field, and the agent is the coin. Getting this language right in the docs is most of what separates this from the pop-science version.
- **Do not use a density matrix for decoherence.** It destroys the sparse representation. Trajectories.
- **Do not generalise the speedup.** Quantum walk advantages are graph-dependent. Ballistic spreading on a lattice, exponential on glued trees, quadratic hitting time on some Markov chains — and *nothing at all* under static disorder. The module should teach the reader that the answer is "it depends on the graph," which is the true and interesting answer.
- **Do not interpolate animation frames.** Discrete time is discrete.
- **Do not let this module eat Part I.** The Lattice is visually louder and will attract more attention, but the spectral result in Part I is the more original contribution. Keep `Lab` as the default tab.

---

## 10. Two corrections to the current mockup

Carry these into the build; they are small and will otherwise get copied forward:

- The status panel reads `Simulator: Aer (statevector)`. Aer is Qiskit's simulator. Our engine is our own — it should read something like `overtone-sim (sparse statevector)`. Naming a competitor's backend in our own instrument undermines the "we built this" claim that the Yao-verified badge is meant to establish.
- `four_qubit_cartpole` as the experiment name: per Part I §7.3, CartPole is a reference point, not the flagship. The default experiment should be `spectral_control_k3`.

Otherwise the mockup is the right direction. The two-zone layout works, the phase-as-hue legend is doing real work, and the ceiling-staircase behind the learning curve is exactly the image Part I was asking for.

---

## 11. References

- Childs, Cleve, Deotto, Farhi, Gutmann, Spielman — *Exponential algorithmic speedup by a quantum walk*, STOC 2003. The glued-trees result.
- Kempe — *Quantum random walks: an introductory overview*, Contemp. Phys. 44, 307 (2003). The standard entry point.
- Kendon — *Decoherence in quantum walks: a review*, arXiv:quant-ph/0606016. Ballistic-vs-diffusive, the decoherence crossover, and the case for a small amount of decoherence being useful.
- Szegedy — *Quantum speed-up of Markov chain based algorithms*, FOCS 2004. The `√HT` detection result.
- Ambainis, Gilyén, Jeffery, Kokainis — *Quadratic speedup for finding marked vertices by quantum walks*, STOC 2020. Detection versus finding, resolved after fifteen years.
- Yin, Katsanos, Evangelou — *Quantum walks on a random environment*, arXiv:0708.1137. Anderson localization under static disorder; the `t_c ∝ W⁻²` crossover under dynamic disorder.
- Chandrashekar, Srikanth, Laflamme — *Optimizing the discrete time quantum walk using a SU(2) coin*, arXiv:0711.1882. The analytic coin-optimization baseline P7 must beat.
- arXiv:2003.07141 — RL for quantum walk coin sequences, targeting entanglement rather than traversal. The nearest prior art to P7; read it before claiming novelty.
