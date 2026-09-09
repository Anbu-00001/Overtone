# OVERTONE PART VI — Braid

**The adversarial arena.** Sixth companion to Parts I–V. **Tab name:** `Braid`.

You have finished Phase 3, so `overtone-sim`, `overtone-rl`, `overtone-spec` and `overtone-lie` exist. Everything below is built on what you already have.

---

## Corrections

**The body below is unchanged**, per the convention in Part II. Corrections are recorded here
and reasoned out in `docs/PHASES.md`.

| Section | Correction | Source |
|---|---|---|
| §1 | "A fermionic agent walls off a corridor" is wrong. Pauli exclusion forbids two fermions in the same **mode**, not at the same **site**: a coined walk has a site *and* a coin, so two fermions share a site in opposite coin states (Sansoni Eq. 4). Measured in `overtone-walk::two`: fermionic mode diagonal exactly `0`, position diagonal `0.094`. A fermionic class blocks one coin state, not the corridor. | Decisions-03 §11; Phase 13 |
| §1 citation | Decisions-02 §3 records the anyonic citation as contradicted on the grounds that Sansoni et al. name only bosons and fermions. That is what the *abstract* says. The body prepares anyonic states at `φ = π/4, π/2, 3π/4` and Fig. 4(c) plots `φ = π/2`, so the citation stands. The audit's substantive point survives: they simulate exchange with photon polarisation rather than making anyons, so van Exter et al. (PRA 85, 033823) is cited **alongside**, not instead. | Phase 13, on the primary source |

---

## 0. Verdict on the pitch, stated plainly

Your three ideas — agents fighting, being chased, consuming to grow — are **not dumb**. They identify a real gap: five parts in, Overtone has no stakes and no agency. A reader watches. Watching does not spread.

But the obvious implementation would destroy the project. Health bars, damage numbers, cooldowns, powerups: the moment one of those appears, a physicist closes the tab and the previous five parts retroactively become decoration on a game.

One rule prevents that, and it is not a compromise — it makes the design *better* than the version you'd get by inventing mechanics:

> ## Every mechanic must be a theorem.
> If a rule cannot be derived from quantum mechanics, it does not exist in the arena. Nothing is invented for balance. If the game is unbalanced, that is a finding.

Working under that constraint, all three of your ideas turn out to have exact physical counterparts that are stranger and more interesting than the game-design version. Two of them are experimentally verified. One of them makes movement itself into computation.

**This is also the highest-variance thing in the series.** Done right it is what puts the project on the front page. Done wrong it is the thing that discredits it. There is no middle ground and no way to split the difference, so §5 is a set of structural safeguards, not suggestions.

---

## 1. "Agents fighting" → spin-statistics

Two quantum walkers in the same lattice is not a metaphor for combat. It is a well-studied physical system with experimentally verified outcomes, and the outcomes depend entirely on **exchange statistics**.

Sansoni, Sciarrino, Vallone, Mataloni, Crespi, Ramponi & Osellame, *Two-particle bosonic-fermionic quantum walk via integrated photonics*, PRL 108, 010502 (2012), demonstrated all three regimes on a chip:

| Statistics | What happens when two walkers meet | Arena consequence |
|---|---|---|
| **Bosonic** | **bunching** — they exit together (Hong–Ou–Mandel) | overlap merges. This is your "consume." |
| **Fermionic** | **antibunching** — Pauli exclusion forbids the same state | you **cannot be entered**. Occupying a corridor blocks it. |
| **Anyonic** | phase `φ` interpolates continuously; Sansoni et al. ran `φ = π/2` | a dial between the two |

So an agent's class is its exchange statistics, and "combat" is simply what happens when two amplitude fields overlap.

A fermionic agent is defensive by *theorem* — it walls off corridors because Pauli exclusion says so, not because a designer gave it a shield. A bosonic agent is offensive because bunching is what bosons do. And `φ` is a continuous slider between them, which no classical game has ever had because no classical world contains anyons.

**This is a class system derived from the spin-statistics theorem.** Nobody has built it.

---

## 2. "Being chased" → decoherence, and the measurement problem as the core loop

The pursuer is not a monster. It is **the thing that actually kills quantum computers**.

### 2.1 The pursuer is a measurement front

An advancing region of dephasing. Where it reaches you, your off-diagonal density-matrix elements die and your transport exponent falls from `β ≈ 1` to `β ≈ 0.5` (Part IV §3). Ballistic becomes diffusive. Diffusive is slower than the front.

**Capture is a phase transition in your own transport, not a hitpoint reaching zero.** The number on screen when you lose is your measured `β`.

Implement with the trajectory-based decoherence already specified in Part II §P3. No new engine.

### 2.2 The core dilemma is the measurement problem

Your resource is **coherence**, and it buys speed:

- Stay coherent → spread ballistically → outrun the front → **but you do not know where you are.** You are a distribution, not a position.
- Measure → collapse to a definite cell → now you know, and can aim → **but you have paid coherence and destroyed your spread.**

> **Information costs speed.** To learn where you are, you must become slower.

That is the measurement problem, and it is the entire loop. No invented resource, no mana bar. The tension is the one quantum mechanics actually imposes.

### 2.3 The Zeno trap

Frequent measurement suppresses evolution (Misra & Sudarshan, 1977 — the quantum Zeno effect). Measure often enough and the walker **stops moving entirely.**

So the player who panics and measures repeatedly is frozen in place while the decoherence front arrives. This is not a punishment mechanic someone designed. It is a theorem, and it is the most exact model of panic I have ever seen in an interactive system.

Do not explain it in a tooltip. Let the player discover that panicking freezes them, then put the citation in the run report.

---

## 3. "Consume to become powerful" → Lie closure, and the price it charges

This is your strongest idea, and Part III already built the machinery for it.

### 3.1 The mechanic

When agent A absorbs agent B, the merged agent's dynamical Lie algebra is the **closure of `g_A ∪ g_B` under commutation**. Critically, that closure is *not* the union: commutators between the two generator sets produce elements in neither. `dim(g)` can explode.

The computation is Pauli bitsets and XOR (Part III §3) — **milliseconds**. So absorption runs a real Lie closure in real time, the counter jumps, and the agent's **sigil redraws itself** (Part IV §2.1) because the sigil is a deterministic function of the algebra.

Your character visibly mutates into something denser and more chaotic. The "become powerful" feeling is delivered by an actual algebraic computation.

### 3.2 The price, which is where this gets good

Part III's three results tell you precisely what you bought:

```
Var[∂C] ∝ 1/dim(g)     eating exponentially shrinks your gradients
M_c ≤ dim(g)            you need more parameters before the landscape is clean
poly dim(g) ⟺ g-sim     small algebras are efficiently simulable — by your opponent
```

Which gives a three-cornered strategy space, every corner a published theorem:

| | **small `dim(g)`** | **large `dim(g)`** |
|---|---|---|
| **learning** | fast, gradients are healthy | **frozen** — barren plateau |
| **predictability** | fully simulable; the pursuer models you exactly | unpredictable |
| **expressiveness** | limited reach | can represent nearly anything |

> **Eat to become unpredictable. Pay by becoming unable to adapt.**

A greedy player consumes everything, becomes maximally expressive, and **dies of a barren plateau** — powerful, rigid, and unable to learn its way out of anything. A minimal player stays nimble but is perfectly predicted by any opponent running `g-sim`.

That is a genuine strategic tension, and the exchange rate between power and adaptability is a result from *Nature Communications* rather than a balance patch. I do not know of another game where that sentence is true.

**The pursuer's ability to model you is Part III's dequantization test.** Your effective `χ` literally is how well your opponent can predict you. The instrument you built to be honest about your own results becomes the antagonist's sensor.

---

## 4. Braiding — the idea that makes movement into computation

A maze is two-dimensional. Two dimensions is where anyons live. And anyons are how topological quantum computers work.

When the worldlines of two anyonic agents wind around each other, the braid applies a unitary that depends **only on the topology of the path — not its geometry** (Kitaev 2003; Nayak, Simon, Stern, Freedman & Das Sarma, *Rev. Mod. Phys.* 80, 1083 (2008)).

> **Moving through the maze is computing. What matters is how you wound around your opponent, not where you went.**

And braiding is **topologically protected**: local noise cannot corrupt it. So in an arena whose antagonist is decoherence, the braid is *the one thing the pursuer cannot take from you.* Everything else — your coherence, your spread, your phase relationships — decays. The topology of your route survives.

That gives the player a reason to take the longer path. The **shape** of your route is the payload.

**Render it as a braid diagram** accumulating beside the maze, worldlines crossing as you circle an opponent, with the accumulated braid word written out in generators `σ₁ σ₂⁻¹ σ₁ …`. You are writing a program with your movement, and you can read the program.

This is, I think, the best single idea in the six parts. It gives movement meaning, it is real physics with a *Reviews of Modern Physics* article behind it, and no game has ever had it because no classical world has anyons.

---

## 5. Structural safeguards — how this stays a physics instrument

Non-negotiable. Each one is load-bearing.

**5.1 No invented numbers.** Your entire state is: wavefunction, coherence remaining, `dim(g)`, statistics phase `φ`, and braid word. No HP. No damage. No XP. No cooldowns. If a quantity is not a physical observable, it is not on screen.

**5.2 The instruments stay live during play.** You act *inside* the Part I–V panels — your Fourier spectrum, your `dim(g)`, your measured `β`, your effective `χ`, all updating. You are not playing instead of measuring. Playing is how you drive the measurement.

**5.3 No win screen. A run report.** When it ends: measured transport exponent, final `dim(g)` with the closure history, braid word, effective `χ`, coherence integral, and the citations for whichever theorems decided the outcome. It should read like an experiment log, because it is one.

**5.4 Language.** *Run*, not level. *Opponent*, not enemy. *Report*, not score. *Absorb*, not kill. Language does more work here than art direction.

**5.5 Never the landing page.** `Lab` stays the default tab. `Braid` is something a visitor finds after being convinced the project is serious. Reverse that order and you lose the audience that matters.

**5.6 The reframe that makes all of this legitimate — the arena is the evaluation harness.**

The opponents are Part IV's MAP-Elites elites. Playing against them *is* an evaluation run, and the results go into the Atlas. Human play generates human-in-the-loop evaluation data for the archive, which Part IV needed anyway.

So this is not a game bolted onto a research tool. It is the evaluation protocol, with a human in the loop, rendered so a human can participate. Say exactly that in the docs — it is true, and it is the sentence that keeps a skeptical reader on the page.

---

## 6. Two more theorems that become rules for free

**No-cloning.** You cannot copy yourself, fork your state, or save-scum. This is a theorem, not an anti-cheese measure, and it should be stated as one.

**Monogamy of entanglement** (Coffman, Kundu & Wootters, 2000). If you are maximally entangled with one agent you cannot be entangled with another. Alliances are mathematically constrained to be exclusive. A three-way alliance is forbidden by a bound on concurrence, not by a rule.

**Optional tutorial: Meyer's penny flip.** Meyer, *Quantum strategies*, PRL 82, 1052 (1999): in the PQ penny-flip game a quantum player beats a classical player **with probability 1**, using the Hadamard operator. A perfect thirty-second opening duel proving the asymmetry is real rather than flavour.

Carry the caveat honestly, because the literature does: later work showed that under modified rules the classical player can win, and there is a standing critique that quantum-game advantages can sometimes be replicated by classical correlated equilibria. One line in the docs. It costs nothing and buys the reader's trust for everything else.

---

## 7. Build order

**M28 — two walkers.** Two-particle quantum walk with bosonic, fermionic and anyonic statistics. Acceptance: reproduce the bunching/antibunching correlation patterns from Sansoni et al. **This is a physics milestone, not a game milestone — do it first and it sets the tone for everything after.**

**M29 — the pursuer.** Decoherence front using the Part II trajectory machinery. Live `β` readout. Acceptance: capture is detected as the transport-exponent transition, with no separate capture condition in the code.

**M30 — player control.** Four verbs, all physical operations: `evolve`, `measure`, `phase` (apply a phase gate to a region), `absorb`. Zeno freezing falls out of `measure` and is not special-cased.

**M31 — absorption.** Live Lie closure on merge, `dim(g)` jump, sigil redraw. Acceptance: the barren-plateau death is reachable — a player who eats everything must become measurably untrainable, verified by the Part III gradient-variance instrument.

**M32 — braiding.** Anyonic worldlines, braid-word accumulation, the braid diagram. The hardest and the best.

**M33 — the harness.** Wire opponents to the Menagerie archive; push run reports to the Atlas.

---

## 8. Traps

- **The first invented number kills it.** The moment a quantity exists for balance rather than physics, the whole tab becomes a game with quantum flavouring and the previous five parts become its marketing.
- **Do not let a sprite in.** Six parts of holding this line; do not break it in the tab most tempted to.
- **Do not tune the pursuer for difficulty.** Set its dephasing rate from physical parameters and report what happens. If it is unwinnable, that is a measurement.
- **Do not explain the Zeno trap in advance.** Discovery is the entire value; the citation goes in the report.
- **Do not skip M28's acceptance test.** If the two-walker correlations do not match the published patterns, the arena's physics is wrong and everything above it is theatre.
- **Do not put `Braid` in the nav before `Lab`.** Ever.
- **Do not claim quantum advantage in the arena.** The interesting claim is that the mechanics are theorems, not that the quantum player wins. Sometimes it should lose, and when it does, that is the more interesting run report.

---

## 9. References

- Sansoni, Sciarrino, Vallone, Mataloni, Crespi, Ramponi, Osellame — *Two-particle bosonic-fermionic quantum walk via integrated photonics*, PRL 108, 010502 (2012). Bosonic, fermionic and anyonic (`φ = π/2`) statistics, experimentally.
- Peruzzo et al. — *Quantum walks of correlated photons*, Science 329, 1500 (2010).
- Schreiber et al. — *A 2D quantum walk simulation of two-particle dynamics*, Science 336, 55 (2012).
- Nayak, Simon, Stern, Freedman, Das Sarma — *Non-Abelian anyons and topological quantum computation*, Rev. Mod. Phys. 80, 1083 (2008). The braiding module's foundation.
- Kitaev — *Fault-tolerant quantum computation by anyons*, Ann. Phys. 303, 2 (2003).
- Misra & Sudarshan — *The Zeno's paradox in quantum theory*, J. Math. Phys. 18, 756 (1977).
- Coffman, Kundu, Wootters — *Distributed entanglement*, Phys. Rev. A 61, 052306 (2000). Monogamy.
- Meyer — *Quantum strategies*, Phys. Rev. Lett. 82, 1052 (1999); and Eisert, Wilkens, Lewenstein, PRL 83, 3077 (1999). Read the critiques alongside them.
- Ragone et al., Nat. Commun. 15 (2024) and Larocca et al., Nat. Comput. Sci. 3, 542 (2023) — already in Part III; they are what make §3.2 a real tradeoff rather than a designed one.
