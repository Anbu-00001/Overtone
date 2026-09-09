# OVERTONE — DECISIONS 02

**Synthesis of four external research passes.** Supersedes the novelty claims in Parts I–VII where they conflict. Read §2 and §5 first; the rest is a correction list.

Sources: ChatGPT prior-art audit, Gemini technical verification, Grok landscape report, methodology precedent survey. Plus two follow-up searches on the closest competitor and the depth-measurement protocol.

**Summary of what happened.** One claim was contradicted outright. Two were overstated and now carry conditions. Two were anticipated. One came back stronger than written. And the landscape report identified a gap that the project already fills — but not the one the README was going to lead with. **The net effect is a better position, provided the headline changes.**

---

## 1. Claims audit

| Claim | Spec | Verdict | Action |
|---|---|---|---|
| Fourier analysis of QRL policies is unexplored | I §1 | **ANTICIPATED** | Rewrite. See §2. |
| Softmax generates harmonics past the ceiling | I §1.4 | **NOVEL** | Keep; verify numerically first |
| Trainable λ slides the reachable comb | I §1.3 | **STATED, NEVER SHOWN** | Reframe as first demonstration |
| Learned structure-conditioned walk coin | II §P7 | **PARTIALLY ANTICIPATED** | Downgrade to assembly |
| `Var[∂C] ∝ 1/dim(g)` | III §2 | **CONDITIONAL** | Add hypotheses; show a failing case |
| poly `dim(g)` ⟹ classically simulable | III §4 | **CONTRADICTED as written** | Three conditions required |
| g-sim supports gradients | III §4 | **VERIFIED** | Training claim survives |
| `M_c ≤ dim(g)` | III §3 | **UPPER BOUND** | Not an equality |
| Dequantization test | III §5 | **ANTICIPATED** | Established method; reframe |
| Rudin-Shapiro suppression from discrete spectrum | IV §3 | **BACKWARDS** | It's Lebesgue-flat |
| AB caging exact | VI-A §T2 | **VERIFIED** | Add the π-flux condition |
| Sansoni et al. showed anyonic statistics | VI §1 | **CONTRADICTED** | Wrong citation |
| Checkmate as controllability | VII §3 | **NOVEL** | Stronger than written |

---

## 2. The one that changes Part I

Part I §1 opens: *"Nobody has taken that theorem and pointed it at a reinforcement learning policy in real time."*

**That is false.** Wilms, Ohff, Skolik, Eisert, Khatri & Reiss, *Quantum reinforcement learning of classical rare dynamics: Enhancement by intrinsic Fourier features* (arXiv:2504.16258, April 2025) implement softmax-PQC policies and analyse them by computing and fitting their truncated Fourier series. Andrea Skolik is an author of *Quantum agents in the gym*, already cited in Part I; Jens Eisert is a major name in the field. This is a strong group working on the project's exact core idea, sixteen months ago.

### Why this is a gift rather than a threat

Their headline empirical finding: **with increasing data-uploading layers, performance first improves and then deteriorates.** They explain it as expressivity of the softmax policy depending on PQC hyperparameters, and they fit Fourier series *post hoc*.

That non-monotonicity is exactly what Overtone's two instruments explain **at the same time, from opposite directions**:

```
more layers L  →  wider reachable spectrum   (Part I §6.5)   →  performance RISES
more layers L  →  larger dim(g)              (Part III §2)   →  gradients shrink
                                                              →  performance FALLS
```

The rise and the fall have different causes. Wilms et al. observed the curve and attributed it to expressivity. Overtone would show both mechanisms running simultaneously and account for each half.

**So Part I's position becomes: an instrument that explains a published empirical puzzle.** That is stronger than a novel observation — it is falsifiable, it has an existing audience who already care about the result, and it gives you a concrete reproduction target instead of an unfalsifiable claim of priority.

**New acceptance test for M2:** reproduce the Wilms et al. non-monotonic performance-versus-layers curve, and demonstrate the spectrum panel and gradient-variance panel accounting for the two halves separately.

### What remains novel in Part I

Two things, both narrower than written:

1. **The softmax harmonic-generation mechanism.** Nobody has attributed softmax-PQC's advantage to spectral leakage, or studied how output nonlinearities interact with the Schuld et al. frequency ceiling in any setting. The trap in Part I §12 stands: verify numerically before writing it down.

2. **The band-limit ceiling as a designed experiment.** `SpectralControl-k` with the LP-computed staircase and the exact `L < k ⟹ J = 0` prediction is not in the literature. Wilms et al. fit spectra to explain results; they do not build an environment whose optimal policy sits at a chosen frequency.

And one downgrade: Jerbi et al. **already state** that trainable scaling yields "a wider and richer spectrum of frequencies." The λ-resonance demo is therefore stated in the literature and never shown. Reframe it as *the first demonstration*, not the first observation. It is still the best visual in the project.

### A free second reproduction target

Duffy & Jastrzębski (arXiv:2506.22555) find that variational quantum models exhibit **spectral bias — they learn low frequencies first**. That is a prediction the spectrum panel tests directly: watch the low-ω bars fill before the high-ω ones during training. It costs nothing to check, it validates the instrument against published work, and if the RL setting behaves differently from the supervised one, that is a finding.

---

## 3. Corrections, per spec

### Part III §4 — the g-sim claim needs three conditions

**Contradicted as written.** Polynomial `dim(g)` is necessary but **not sufficient**. Goh et al. (PRR 7, 033266, 2025) require all of:

1. The Lie closure yields irreps whose dimensions scale polynomially in `n`
2. An orthonormal basis for those irreps and the non-zero structure constants are efficiently computable classically
3. The initial state and observable are supported on those polynomially-bounded irreps, or expressible as products up to small constant order

If an ansatz has polynomial DLA dimension but the observable projects onto an exponentially large irrep, g-sim is intractable.

**The training claim survives intact:** g-sim supports exact analytical gradients, computed via structure constants in the representation space. And classical simulations above 100 qubits are demonstrated in the paper.

Rewrite the headline with the conditions attached. It is still a remarkable claim; it is just no longer a one-liner.

### Part III §2 — `1/dim(g)` is conditional

Ragone et al. require `ρ ∈ g` **or** `O ∈ g`. Diaz et al. (arXiv:2310.11505) show the formula breaks when both lie outside: parametrized matchgate circuits have `dim(g) = O(n²)` — polynomial — and *still* suffer severe barren plateaus, because the operator space decomposes into Lie-group modules of dimension `C(2n, k)` and higher-degree observables force exponential variance decay.

**This improves the C2 panel.** Display the hypothesis check (`ρ ∈ g`? `O ∈ g`?) beside the prediction, and include a Diaz-style **failing case** as a second preset. A panel that shows its own theory's boundary is far more convincing than one that always works.

**New M11 acceptance test:** reproduce a case where `1/dim(g)` fails.

### Part III §3 — `M_c` is a bound

`M_c ≤ dim(g)` and `rank(QFIM) ≤ dim(g)` are upper bounds, not equalities. Depending on initial-state symmetry, the QFIM rank can saturate strictly below `dim(g)`, and overparameterization can arrive at fewer parameters than the algebra dimension.

C3's phase-transition panel should mark `dim(g)` as a **bound** with the observed `M_c` plotted below it. Watching the transition happen *before* the predicted ceiling is more interesting than watching it land on one.

### Part III §5 — the dequantization test is an established method

**Anticipated.** Rudolph et al., *Synergistic pretraining of parametrized quantum circuits via tensor networks*, Nat. Commun. 14, 8367 (2023) already measures the MPS bond dimension needed to represent trained PQCs, and finds many compress to low bond dimension without losing task performance. Cerezo et al. (arXiv:2312.09121) asks the question directly in its title.

Part III §5 claims *"no other QML repository ships an instrument that tries this hard to disprove its own premise."* The word doing the work is **ships**. The method is standard; running it live on every agent in the repo and publishing the numbers is not. Reframe accordingly, cite both papers, and drop the implication that the idea is new.

### Part IV §3 — Rudin-Shapiro, backwards

The suppression comes from an **absolutely continuous, flat, white-noise-like Fourier power spectrum** — not a discrete one. Fibonacci is singular-continuous with a fractal Cantor spectrum; Rudin-Shapiro is spectrally indistinguishable from spatial white noise, and that pseudo-random character scrambles phases across all spatial frequencies.

The contrast is still real and is arguably a better story: **two perfectly ordered, non-repeating sequences, one with a fractal spectrum and one with a white-noise spectrum, transporting at visibly different rates.** Fix the explanation; keep the panel.

### Part VI §1 — wrong citation for anyonic statistics

**Contradicted.** Sansoni et al. (PRL 108, 010502, 2012) demonstrated bosonic bunching and fermionic antibunching only, via symmetric and antisymmetric polarisation encodings. They did not demonstrate fractional exchange statistics.

Anyonic statistics were realised separately — van Exter, Nienhuis & Woerdman, PRA 85, 033823 (2012), using custom double-retardation phase plates; and in cold-atom optical lattices via Floquet-engineered density-dependent Peierls phases.

The anyonic class in the arena survives. The citation changes, and M28's acceptance test splits into two: bunching/antibunching against Sansoni, anyonic phase against van Exter.

### Part VI-A §T2 — AB caging verified, with the condition made explicit

Confinement is **exact** — amplitude identically zero outside the cage for all time — but requires **π flux (half a flux quantum) per plaquette** *plus* the coin condition. Caging does not hold for arbitrary coin angles without flux tuning. Lattices: diamond chain, dice/T₃, multi-leg ladders.

The `1e-12` acceptance test stands and is the correct test. Add the flux condition to the trap's parameter vector explicitly.

### Part VII §3 — checkmate is stronger than written

**Novel**, and better than the spec claims. Cantwell's Quantum Chess design notes state that *"there is no concept of check or checkmate. Kings are captured like any other piece."*

So the controllability definition does not merely avoid a collision — it solves a problem the closest prior work explicitly abandoned. That is the strongest novelty result of the three, and it should be stated plainly and carefully in the docs: not as a criticism of Cantwell, who had good reasons, but as a note that the orbit formulation is well-defined on superposed states because it is a statement about reachability rather than occupancy.

**Add to the references:** Wu & Tarn, PRA 65 (2002), on subspace controllability. That is the technical machinery the checkmate predicate rests on, and citing it moves the claim from assertion to construction.

### Part II §P7 — downgrade to assembly

**Partially anticipated.** The pieces exist separately:

- Chandrashekar, Srikanth & Laflamme (PRA 77, 032326, 2008) optimise a **global** SU(2) coin analytically, improving mixing and hitting time on a cycle
- Wilms et al. (PRA 101, 062331, 2020) use Q-learning to select coin **sequences**, objective is entanglement generation
- Ahmad, Sajjad & Sajid (2020) analyse position-dependent coins theoretically, no learning
- Patapovich (thesis, 2023) gives circuits for non-homogeneous position-dependent coins, implementation only

The specific combination — learned, position- and structure-conditioned, hitting-time objective — is unoccupied. But it is an assembly of published components, not new territory. Say so. *(Note: the same Wilms appears in both this and §2's Fourier paper. That group is working both of the project's areas and is the closest thing it has to a competitor. Worth watching.)*

---

## 4. Amendments to Decisions 01

### Q1 — validated, with a citation

Goodman, Perez-Liebana & Lucas, *Skill Depth in Tabletop Board Games* (IEEE CoG 2024) find that **expanding the algorithmic space radically changes the estimated depth of some games**, and therefore define best achievable performance at budget `b` as a maximum over an algorithm space rather than for one fixed algorithm.

That is precisely the unified declarative language with `search.kind ∈ {greedy, negamax, mcts}`. The Q1 ruling now has a published justification instead of an argument.

### Q2 — amend: publish two curves

No universal machine-independent compute standard exists. Wall-clock under controlled hardware is dominant in competitions (GGP: 600 s startclock / 30 s playclock; GVGAI: 40 ms per tick). Node counts, search depth, MCTS simulations and forward-model calls are **established local conventions**, but there is no accepted conversion between heterogeneous node types.

**Arimaa 2011 is the direct precedent:** bots were required to expose a hardware-independent play-strength option in nodes or plies — while the championship itself still ran on identical hardware.

**Upgraded recommendation: publish both curves.** Performance versus work units, and performance versus wall-clock. That separates *algorithmic* efficiency from *systems* efficiency, and it is much harder to attack than a calibration footnote. The reviewer question shifts from "why isn't this wall-clock?" to "why is your work unit a valid abstraction?" — which you can answer.

### Q3 — replace the ladder with a grid

Goodman et al. make the point directly: a ladder of adjacent budgets (16 vs 32, 32 vs 64) **may not statistically resolve small win-rate differences**. They extend it to a **grid of all pairwise budgets** (16 vs 32, 16 vs 64, 16 vs 128, …). A gap that is invisible between adjacent rungs is easily resolved between distant ones, and the distant pairs constrain the fit.

**Same number of games, far more statistical power.** Use the grid. Then fit the parameterised skill model to it rather than counting discrete steps.

Two further amendments:

- **Rename the calibration.** Identical-copy self-play is *not* an established chess-testing convention. It is a sensible control, so call it a **repeatability calibration**, not a noise floor. That phrasing is much harder to criticise.
- **Cost it honestly.** For a draw-heavy match the 95% Elo error is roughly `400/√N`: about 1,600 games for ±10 Elo, 6,400 for ±5. Detecting small differences is brutally expensive and the protocol should budget for it.
- **Steal both tools.** SPRT (as in Fishtest, standard `SPRT(0,10)`) for continuous gating; confidence intervals plus LOS for the headline report.

### New — depth terminology

**Do not write "the standard Lantz metric."** Lantz et al. state explicitly that they had no system for evaluating `d` and were not proposing final answers; the framework deliberately leaves the strategy language, resource levels, performance metric, and step definition open.

The lineage: Lantz 2017 → Tavener 2020 (UCT skill ladders) → Browne 2022 (**Skill Trace**) → Goodman et al. 2024 (**Skill Depth**).

Report **Skill Trace**. Correct phrasing: *"We instantiate the strategy-ladder framework of Lantz et al., following Browne's Skill Trace and Goodman et al.'s grid methodology, with our chosen budget, performance measure and step criterion."*

**And this gives you the figure.** Goodman publishes 2-player Skill Trace values for sixteen games:

```
Dots + Boxes 0.353 · Dominion 0.288 · Connect 4 0.282 · Sushi Go 0.189
Can't Stop 0.028 · Love Letter 0.013 · Stratego 0.010 · Diamant 0.002
Tic-Tac-Toe 0.000
```

Plotting Overtone on that axis is a far better figure than the chess/Go/checkers comparison originally planned in Part VII §8, because these are measured on a compatible protocol rather than eyeballed.

*(Also read their companion paper, "Measuring Randomness in Tabletop Games", CoG 2024. Overtone's randomness is the Born rule, and they explicitly note stochasticity changes observed skill rewards — directly relevant to interpreting your trace.)*

### Part VIII §9 — strengthen the overfitting mitigation

**Hidden instances beat hidden scores.** A 2026 Kaggle simulation competition publishes its seeds and hides only scores, and the organisers themselves note this lets participants optimise for the instances. Public seeds plus hidden scores is not enough.

Four layers, per the survey's recommendation:

```
public training seeds
limited public development seeds
hidden test seeds            ← scoring
fresh post-submission seeds  ← audit, never disclosed
```

The fourth layer catches leaderboard overfitting through repeated submission, which the first three do not.

---

## 5. The repositioning

This is the section that matters most, and it follows from combining the corrections with the landscape report.

### What the landscape says

The field is cooling on advantage claims and active on theory. Practitioner mood is tempered to skeptical. Dequantization and classical-surrogate results have eroded earlier advantage claims. **There is no accepted regime that is simultaneously free of barren plateaus, classically hard, and useful.** Cerezo, Larocca et al. asked exactly this in a 2025 *Nature Communications* perspective and concluded that most known trainability-guaranteeing constructions work by restricting to a classically simulable subspace.

And the confirmed gap: **"live visualization of variational circuit training dynamics: almost none with significant visibility."** Most demos stop at static circuits or single forward passes. Quirk is the browser benchmark and it is a static circuit playground.

### Why the headline has to change

The planned headline was *"train a 100-qubit quantum RL policy on a free CPU."* Three problems:

1. It now requires three conditions (§3), so it is no longer a one-liner.
2. It is a capability flex, and the field's mood punishes those.
3. It invites the standard dismissal — *"cute demo, doesn't change the asymptotic picture"* — and that dismissal would be correct.

### The claim to lead with instead

> **Overtone tells you whether your quantum circuit will train — before you train it.**

That is what the tool does. It is a **usefulness** claim, not an advantage claim. It is defensible without any contested novelty. And it sits exactly in the confirmed gap.

Then the second line names the tension rather than hiding it:

> The circuits that train are often the circuits a classical computer can already simulate. Overtone measures both, so you can see where yours sits.

**The trainability–simulability triangle is the project's subject, not its embarrassment.** Cerezo et al. asked the question in a paper. Overtone makes it a slider. The landscape report's own conclusion is that the field needs clarity tools of exactly this kind right now.

### The defensive position is unusually strong

The seven standard dismissals, mapped:

| Dismissal | Overtone's answer |
|---|---|
| "It's classically simulable / dequantized" | The dequantization test measures precisely this and publishes the number |
| "Barren plateaus kill it at scale" | The plateau panel demonstrates it, with the exponent fitted live |
| "No quantum advantage demonstrated" | None is claimed. Say so in line two. |
| "NISQ noise makes it worse than classical" | The shot-budget dial makes measurement cost visible |
| "Only works on 4–8 qubits in simulation" | g-sim, with its three conditions stated honestly |
| "Trainable circuits are the ones classical computers handle" | **That is the thesis, not an objection to it** |
| "Interactive demo doesn't change the asymptotics" | Correct. The README should say so. |

**Six of seven are things Overtone measures rather than things it denies.** Make this an explicit README section — a short table, criticism in the left column, the panel that addresses it in the right. Preempting your own dismissals with instruments rather than arguments is the single most credibility-building thing available here.

---

## 6. What changes in the build

**Corrected or new acceptance tests:**

- **M2** — reproduce Wilms et al.'s non-monotonic performance-vs-layers curve; show both mechanisms accounting for it
- **M2b** — test Duffy & Jastrzębski's spectral bias prediction (low frequencies learned first)
- **M11** — reproduce a Diaz-style case where `1/dim(g)` **fails**
- **M12 (C2)** — display the `ρ ∈ g` / `O ∈ g` hypothesis check beside the prediction
- **M28** — split: bunching/antibunching against Sansoni; anyonic phase against van Exter
- **M32c** — add the explicit π-flux condition to the cage parameters
- **M34** — cite Wu & Tarn subspace controllability as the checkmate predicate's basis
- **M35/M39** — Goodman grid, not adjacent-rung ladder; report Skill Trace against the sixteen published games

**Priority shift.** The 100-qubit g-sim demo drops from headline to documented capability. **M12 (the prediction landing) and M49 (the temperature heat map) become the two panels the project is judged on**, because live training-dynamics visualisation is the confirmed empty space and those two are the clearest instances of it.

Nothing needs cutting. The scope was already governed by Part III §12's minimum viable version, and that cut is unaffected by any of this.

---

## 7. References added

- Wilms, Ohff, Skolik, Eisert, Khatri & Reiss — *Quantum reinforcement learning of classical rare dynamics: enhancement by intrinsic Fourier features*, arXiv:2504.16258 (2025). **The closest prior work; the reproduction target.**
- Duffy & Jastrzębski — spectral bias in variational quantum models, arXiv:2506.22555 (2026).
- Wilms et al. — RL for quantum walk coin sequences, PRA 101, 062331 (2020).
- Ahmad, Sajjad & Sajid — position-dependent coin quantum walks, Commun. Theor. Phys. 72, 065101 (2020).
- Rudolph, Miller, Motlagh, Chen, Acharya & Perdomo-Ortiz — *Synergistic pretraining of parametrized quantum circuits via tensor networks*, Nat. Commun. 14, 8367 (2023). **The dequantization method's origin.**
- Cerezo, Larocca, García-Martín, Diaz et al. — *Does provable absence of barren plateaus imply classical simulability?*, arXiv:2312.09121. **The paper to cite when stating the tension.**
- Diaz, García-Martín, Kazi, Larocca & Cerezo — arXiv:2310.11505. Where `1/dim(g)` fails.
- van Exter, Nienhuis & Woerdman — *Two-photon scattering with customizable exchange statistics*, PRA 85, 033823 (2012). **Correct anyonic citation.**
- Wu & Tarn — subspace controllability, PRA 65 (2002).
- Goodman, Perez-Liebana & Lucas — *Skill Depth in Tabletop Board Games*, IEEE CoG 2024, DOI 10.1109/cog60054.2024.10645624. **The depth protocol and the comparison values.**
- Goodman, Perez-Liebana & Lucas — *Measuring Randomness in Tabletop Games*, IEEE CoG 2024.
- Browne — *Quickly Detecting Skill Trace in Games*, IEEE CoG 2022.
