# OVERTONE PART VI-A — Traps

**Addendum to Part VI (`Braid`). Folds into M32.** Depends on `overtone-walk` (Part II M6–M8) and `overtone-lie` (Part III M10), both of which you have.

---

## Corrections

**The body below is unchanged**, per the convention in Part II. Corrections are recorded here
and reasoned out in `docs/PHASES.md`.

| Section | Correction | Source |
|---|---|---|
| §T2 | The AB cage has a published instance that needs no flux and no lattice geometry at all. Krovi & Brun exhibit **infinite hitting times on the hypercube** from destructive interference alone — the walker provably never arrives. **It is a DFT-coin phenomenon**: the Grover coin on the same graph, from the same start, arrives with probability one, and on the 3-cube the DFT coin does too. Measured on the 4-cube: `3/7` of the amplitude is trapped forever. Named as a note under T2, per §8's standing rule against a ninth trap. | Decisions-03 §11; Phase 14 |
| §T2 escape | "re-parameterise the coin" is the listed escape from the AB cage, and this instance is the cleanest possible demonstration that it is the *right* escape: the cage is a property of the coin, not of the terrain. | Phase 14 |

---

## 0. Why traps, and the rule they must obey

The arena in Part VI has agents, a pursuer, and movement that computes. What it lacks is **terrain that matters**. Without it the maze is a container and the strategy is a chase.

Traps are the obvious fix and the obvious place the design collapses. Trap damage, trap radius, trap cooldown — three invented numbers and Part VI §5.1 is dead.

It turns out none of that is necessary. Quantum walks have a large, well-studied literature on **localization**, which is the physics of things that cannot leave a region. Every trap below is a published localization or topology result, several experimentally verified, and none of them requires a tuned constant.

One structural consequence sets the tone for the implementation:

> **A trap is not an object. It is a per-cell parameter.**

Local flux, local coin, local disorder strength, local observable locality. The maze generator (Part IV §3) gains a small parameter vector per cell, drawn from the same seeded hash. **No entity system, no spawn logic, no collision detection.** That absence is the structural guarantee that this stays physics — and it means traps cost essentially nothing in the sparse infinite-lattice representation.

---

## 1. The eight traps

### T1 — Grover well

Amplitude amplification, run against the prey. A marked cell plus a local diffusion operator concentrates a walker's amplitude onto the trap site rather than dispersing it.

The timing is not designed, it is derived. Grover reaches maximum amplitude at about `π/4·√N` iterations, and the success amplitude follows `sin²((2j+1)θ)` with `sin θ = 1/√N`. So the trap has:

- a **physically determined arming time** — you cannot trigger it early;
- an **exact failure mode** — run it too long and the amplitude rotates *past* the target and back down. Brassard's soufflé problem.

**A trap left armed too long releases its prey, at a step count you can compute in closed form.** A timing window that is an exact expression rather than a tuned constant.

### T2 — Aharonov–Bohm cage *(the perfect trap)*

Vidal, Mosseri & Douçot, PRL 81, 5888 (1998). On certain lattices — the diamond chain, the dice/`T₃` tiling — threading **half a flux quantum per plaquette** makes every band flat. Fully destructive interference prevents any wavepacket from leaving a finite cluster. **Zero transport**, from geometry and phase alone. No disorder, no measurement, no loss.

Experimentally realised in photonic lattices, ultracold atoms, trapped ions and Rydberg arrays.

Critically for us, there is a paper on exactly the version we need: *Tunable Aharonov–Bohm-like cages for quantum walks* (arXiv:1910.00845) shows AB cages exist for quantum walks on these tilings **for a proper choice of coin.**

> **The cage catches you only if your coin is wrong.** A learned coin can, in principle, learn its way out.

That is a trap whose escape condition is a property of the agent's own policy, and it wires the trap system directly into the Part II §P7 learned coin. Escape by re-parameterisation.

Known leakage mechanism, also published: quenched disorder breaks the cages and produces Anderson-like exponential decay (Vidal, Butaud, Douçot & Mosseri, PRB 64, 155306). So a player can also break a cage by *injecting disorder into it* — which is a phase operation they already have.

### T3 — Topological bound state

Kitagawa, Rudner, Berg & Demler, PRA 82, 033429 (2010): discrete-time quantum walks realise every topological phase classified in 1D and 2D. At a boundary between domains of different topology, a walker becomes **localised in a topologically protected bound state**.

Observed experimentally, including its robustness to perturbation (Kitagawa et al., *Nat. Commun.* 3, 882 (2012)).

That robustness is the mechanic. **You cannot jiggle your way out of a topological trap.** Local noise, phase kicks, small coin adjustments — none of it works, because the binding is protected by a bulk invariant rather than by an energy barrier. The only escape is to change the bulk topology on one side.

This is the hardest trap in the set, and it is hard for a reason a physicist will recognise instantly.

### T4 — Chiral corridor

Same source, 2D case: at a topological boundary in two dimensions the walker **propagates unidirectionally**.

A one-way passage where backscattering is *topologically forbidden* rather than blocked by a wall. Enter it and forward is the only direction that exists. It is the cleanest maze primitive in the whole set, and no rule enforces it — an invariant does.

### T5 — Disorder patch

Static random phases over a region. A coherent walker Anderson-localises: `σ(t)` saturates, exponentially confined (Part IV §3 already has this).

The asymmetry is what makes it interesting. **A classical walker diffuses straight through**, because phase disorder does not touch a diffusive process. So:

> **Disorder traps catch only coherent agents.**

Which produces a genuine strategic loop: the escape is to *deliberately decohere yourself* — and decohering is exactly what the pursuer was trying to do to you anyway. You escape the trap by doing the enemy's work.

### T6 — Zeno region

A region that forces measurement every step. Entering it freezes you (Misra & Sudarshan, 1977). No escape mechanic exists while inside, because frequent observation suppresses evolution — the exit is spatial, not tactical.

The environmental counterpart to Part VI §2.3's panic mechanic. Same citation, no new engine.

### T7 — Spectral trap

The one only this project could have.

Part I §7.1: a RAW-PQC with encoding depth `L < k` scores **exactly zero** on structure at frequency `k`. Not slowed — the solution is not in its function class.

So build maze regions whose local structure sits at frequency `k`. An agent with insufficient depth does not perceive them at all.

> **Two agents with different `L` are walking through different mazes.**

Not fog of war. A bandwidth limit that is a theorem. Escape by raising `L`, or by tuning `λ` until the reachable comb covers `k` — which is Part I's resonance mechanic, now with something at stake.

### T8 — Plateau region

Cerezo et al., *Nat. Commun.* 12, 1791 (2021): **global** cost functions produce barren plateaus even at shallow depth, while local ones remain trainable.

So a region where the reward structure is global rather than local is a place where **learning stops working**. An agent using a fixed coin walks through unaffected. An agent that is still adapting goes flat.

A trap that targets learning itself, with a *Nature Communications* result behind it, measurable live by the Part I §6.7 gradient-variance instrument.

---

## 2. The escape matrix — why this is a system and not a list

| Trap | Catches | Escape |
|---|---|---|
| T1 Grover well | anyone in range | outlast the over-rotation — timing |
| T2 AB cage | wrong coin | re-parameterise the coin, or inject disorder |
| T3 Topological bound state | anyone crossing | change the bulk topology; perturbation fails |
| T4 Chiral corridor | anyone entering | forward only |
| T5 Disorder patch | **coherent** agents | decohere yourself |
| T6 Zeno region | anyone evolving | leave spatially |
| T7 Spectral trap | low `L` agents | raise `L`, or tune `λ` |
| T8 Plateau region | **learning** agents | stop learning; freeze the coin |

Every escape is a physical operation the player already has from Part VI §M30: `measure`, `phase`, `absorb`, plus retuning `λ` and `L`. **Nothing new is added to the verb set.**

---

## 3. The composition principle

This is what turns eight mechanics into a strategy space:

> **Every escape is a vulnerability to a different trap.**

- Decohere to escape T5 → you are now slow → the decoherence front (Part VI §2.1) catches you.
- Raise `dim(g)` to widen your reach past T7 → `Var[∂C] ∝ 1/dim(g)` → you are now vulnerable to T8.
- Freeze your coin to survive T8 → you can no longer re-parameterise out of T2.
- Measure to navigate precisely → Zeno-adjacent, and you have paid coherence.

Every one of those trades is a Part III theorem, already implemented, already instrumented. The trap system does not introduce a single new balance parameter — **it exposes tensions that were already in the physics.**

That is the whole argument for why this belongs in the project. Traps are not content. They are terrain that makes the existing tradeoffs consequential.

---

## 4. Placement in the endless maze

Extend the per-cell procedural generation (Part IV §3) with a parameter vector:

```rust
struct CellParams {
    flux:        f32,   // plaquette flux → T2 at half quantum
    coin_id:     u8,    // local coin → T2 condition, T3/T4 topological domain
    disorder_w:  f32,   // static phase strength → T5
    zeno_p:      f32,   // forced-measurement rate → T6
    structure_k: u8,    // local feature frequency → T7
    obs_global:  bool,  // observable locality → T8
}
```

All six fields come from the same seeded hash of `(x, y)`. Traps are therefore infinite, deterministic, reproducible from a URL, and **free** — they add a handful of bytes per occupied cell to a representation that is already sparse.

T1 (Grover wells) is the one exception: it is player- or agent-placed, so it needs a small list of active wells with their arming step counts. Keep that list, and nothing else, as state.

**Coherence with Part IV's worlds.** Trap density should be a property of the substrate, not a separate difficulty setting. The Rudin–Shapiro world is already strongly subdiffusive; it does not need many traps. The periodic world is ballistic and nearly frictionless; it can carry more. Let the substrate decide, and report the resulting density as a measured statistic rather than choosing it.

---

## 5. Rendering

Traps must read as **terrain**, not as objects. No icons, no glow, no warning markers.

| Trap | How it looks |
|---|---|
| T1 | the amplitude field visibly converging — the trap is the convergence |
| T2 | plaquette flux drawn as faint circulation arrows; the cage boundary emerges from where amplitude stops |
| T3/T4 | domain boundary as a hairline; the bound state is a standing amplitude peak sitting on it |
| T5 | slightly noisier phase texture, nothing more |
| T6 | nothing at all until you enter and stop moving |
| T7 | **invisible to an agent that cannot resolve it** — render the maze *from the agent's spectral perspective*, so a low-`L` agent's view genuinely lacks the structure |
| T8 | nothing visible; the gradient-variance readout falls off a cliff |

**T7's rendering is the best idea here.** Draw the maze as the agent can represent it — band-limit the render to the agent's reachable frequencies. A low-`L` agent's maze is visibly smoother, missing the fine structure that is killing it. Two agents, two views, one maze. It makes Part I's central abstraction into something you can look at.

Half these traps are invisible until they act. That is correct. A trap you can see coming is a game object; a trap you infer from your own instruments is terrain.

---

## 6. Milestones

Folds into M32. Build in this order — the first two carry the system.

**M32a — T5 and T6.** Disorder patches and Zeno regions. Both reuse Part II §P3/§P4 machinery entirely. Acceptance: `σ(t)` saturates inside a disorder patch for a coherent walker and does not for a decohered one.

**M32b — T7 with band-limited rendering.** The spectral trap and the agent's-eye view. Acceptance: a `L < k` agent's measured return on trap structure is zero to `1e-3`, and its rendered view provably lacks frequency-`k` content.

**M32c — T2, Aharonov–Bohm cages.** Diamond-chain and dice-tiling regions at half flux. Acceptance: **reproduce complete confinement** — zero amplitude outside the cage cluster to `1e-12` — and verify that changing the coin releases the walker, per arXiv:1910.00845.

**M32d — T1, Grover wells.** Acceptance: peak capture at `π/4·√N` steps and measurable release on over-rotation, both matching the closed form.

**M32e — T3 and T4, topological.** Domain boundaries, bound states, chiral corridors. Acceptance: the bound state survives perturbation (that is the whole point) while a non-topological trap of similar depth does not.

**M32f — T8, plateau regions.** Acceptance: the gradient-variance instrument registers the collapse, and a frozen-coin agent passes through unaffected.

---

## 7. Design traps

- **The first tuned constant kills it.** There is no trap damage, no trap radius, no cooldown. `π/4·√N` is derived. The localisation length is measured. If you find yourself picking a number to make a trap "feel right," you have left physics.
- **Do not build an entity system.** Traps are cell parameters. The moment there is a `Trap` object with a lifecycle, the codebase has become a game engine and the physics is decoration on top of it.
- **Do not signpost traps.** Half of them are invisible by nature. Let the player's instruments be the warning — that is what the instruments are for.
- **Do not tune trap density for difficulty.** Derive it from the substrate and report it.
- **Do not skip M32c's `1e-12` test.** AB caging is *exact*. If your implementation leaks, the flux or the coin is wrong, and a trap that "mostly" confines is just a slow region.
- **Do not add a ninth trap.** Eight already covers localisation by interference, by disorder, by topology, by measurement, by bandwidth, and by trainability. A ninth would be a variation, and variations are how content creep starts.

---

## 8. References

- Vidal, Mosseri, Douçot — *Aharonov–Bohm cages in two-dimensional structures*, PRL 81, 5888 (1998).
- Vidal, Butaud, Douçot, Mosseri — *Disorder and interactions in Aharonov–Bohm cages*, PRB 64, 155306 (2001). The leakage mechanism.
- *Tunable Aharonov–Bohm-like cages for quantum walks*, arXiv:1910.00845. Cages for quantum walks, conditional on the coin — the version T2 needs.
- Kitagawa, Rudner, Berg, Demler — *Exploring topological phases with quantum walks*, PRA 82, 033429 (2010). T3 and T4.
- Kitagawa, Broome, Fedrizzi, Rudner, Berg, Kassal, Aspuru-Guzik, Demler, White — *Observation of topologically protected bound states in photonic quantum walks*, Nat. Commun. 3, 882 (2012). The robustness that makes T3 hard.
- Asbóth — *Symmetries, topological phases and bound states in the one-dimensional quantum walk*, arXiv:1208.2143. Localisation in 1D, unidirectional propagation in 2D.
- Brassard — the soufflé problem; and Boyer, Brassard, Høyer, Tapp, *Tight bounds on quantum searching*, Fortschr. Phys. 46, 493 (1998), for the `sin²((2j+1)θ)` amplitude and the over-rotation.
- Cerezo, Sone, Volkoff, Cincio, Coles — *Cost function dependent barren plateaus in shallow parametrized quantum circuits*, Nat. Commun. 12, 1791 (2021). T8.
- Misra & Sudarshan — *The Zeno's paradox in quantum theory*, J. Math. Phys. 18, 756 (1977). T6.
