# OVERTONE PART VII — Orbit

**The chess question.** Seventh companion, and the one that changes the arena.

**Tab name:** `Orbit`. **This supersedes Part VI's free-form arena** rather than sitting beside it — see §10.

---

## 0. The correction that has to come first

Your framing was: chess has `n×n` moves and enormous probability, which fascinates humans; Overtone has an endless maze and even more possibility, so it should fascinate agents.

The research says the opposite, and it says it sharply.

**Chess's branching factor is about 35.** Go's is 250. Go is not seven times deeper. Chess has roughly `10^44` legal positions and a game tree around `10^120` — large, but *finite and small* compared with an endless maze, which has infinitely many. Chess is deep **despite** being finite and modest in branching.

Allis's framework, the one Schaeffer used when solving checkers, splits difficulty into two axes:

- **space complexity** — how many states exist
- **decision complexity** — how hard it is to decide correctly

Every game solved so far has low space complexity, low decision complexity, or both. Checkers had *high* decision complexity and only moderate space complexity, and it still took eighteen years.

Now score Overtone's maze honestly:

> **Enormous space complexity. Near-zero decision complexity.**

That is not chess. That is Snakes and Ladders — which also has a large state space and essentially zero depth, because nothing you decide matters.

**Endlessness is currently a liability wearing the costume of a feature.** Adding more maze adds states, and states are the cheap axis. Depth lives on the other one.

### What depth actually is

Lantz, Isaksen, Jaffe, Nealen & Togelius (AAAI 2017) give the definition worth using. Depth `d` is *the capacity of a game to absorb dedicated problem-solving attention and allow sustained long-term learning*, measured by **how resistant the game is to partial solutions as computational resources increase**. They call the resulting structure the **strategy ladder**: a deep game lets you learn something useful immediately, and keeps rewarding better thinking for many orders of magnitude of additional effort.

Operationally:

> **Plot performance against compute budget. The length of the rising portion of that curve is the depth.**

Tic-tac-toe saturates in one rung. Chess is still climbing after seventy years of engines. That is the whole difference, and — critically — **it is measurable.** Overtone is an instrument. It can measure its own depth. §8 makes that the design loop.

---

## 1. Three axioms

From the research and from what chess actually does, three properties generate depth. Everything in this document derives from them.

### Axiom I — Decisions must be few, legible, and heterogeneous

Chess offers ~35 moves from six piece types whose **geometries differ in kind**, not degree. A knight and a bishop are not two settings of one slider; they cover complementary structure, and their interaction is where tactics live.

Overtone currently offers continuous parameters. Continuously many moves is not infinite depth — it is **zero legibility**. You cannot weigh options you cannot enumerate, and weighing options is the experience of depth.

**Fix: discretise the move set into a small heterogeneous generator basis.** §2.

### Axiom II — Resources must be incommensurable and irreversibly spent

Chess trades material against position against tempo with no exchange rate, and pawn moves and captures cannot be undone. That irreversibility is a **thermodynamic arrow**, and it is what produces chess's phase structure: openings, middlegames, endgames are qualitatively different regimes.

Overtone already has this and did not have to design it. Coherence only decreases. `dim(g)` only increases under absorption. §4 turns that into a phase structure.

### Axiom III — Evaluation must be genuinely hard

This is the real reason chess is deep, and the one most designs miss. **If a cheap heuristic reaches 95% of optimal play, the game is shallow at any size.** Chess resists cheap evaluation because material count is sometimes badly wrong in ways that are not locally detectable.

Overtone can satisfy this axiom more strongly than any board game in existence, because when `dim(g)` is exponential and the effective bond dimension is high, **no efficient classical evaluation of the position exists at all.** And it is on a slider. §6.

---

## 2. Pieces are generators

The discretisation writes itself once you notice what a chess piece actually is: a *movement geometry*, which is to say the orbit traced by one generator.

| Chess piece | What actually defines it | Overtone generator |
|---|---|---|
| **Pawn** | one step forward, cannot reverse | one coherent step; spends coherence |
| **Bishop** | **cannot change square colour — conserves a `Z₂` invariant** | generator confined to a symmetry sector |
| **Knight** | jumps, ignores what lies between | long-range hop coupling non-adjacent cells |
| **Rook** | acts along an entire line | phase gate applied along a line |
| **Queen** | rook ∪ bishop | the **commutator closure** of two generators |
| **King** | the thing that must survive | the protected subspace |

### The bishop is the whole insight

A bishop can never change square colour. That is not a rule about diagonals — it is a **conserved quantity**, and being trapped on one colour forever is a superselection sector. Chess has had a symmetry-sector piece since the fifteenth century and calls it "light-squared."

So a piece taxonomy *is* a statement about which subalgebra a generator lives in. The mapping is not an analogy.

### And the queen explains Lie closure

Chess folklore says a queen is worth more than a rook plus a bishop. That is exactly the **non-additivity of the Lie closure**: `[G_rook, G_bishop]` generates directions present in neither. Part III §3 already computes this in milliseconds.

Chess players have had an intuition for the non-additivity of Lie closure for five hundred years. Overtone can show them the algebra behind it.

### The move set

Give the player a **hand of six to eight generator types**. Each turn: pick one generator, pick one target region. Branching factor lands around 25–40 **by construction**, and it must be measured and reported, not assumed (§9).

Finite. Legible. Heterogeneous. Axiom I satisfied.

---

## 3. Checkmate is a controllability condition

This is the centrepiece, and it is the part I would build first.

Checkmate is not "score below a threshold." It is **structural**: the king has no legal move that escapes attack. Every game with a satisfying win condition has one of these — a state from which the reachable set contains nothing acceptable.

Dynamical Lie algebras come from **quantum control theory**, and controllability is precisely what they compute. If `g = su(2ⁿ)` the system is fully controllable and every state is reachable. If `g` is a proper subalgebra, the reachable set is an **orbit**, and states outside it are unreachable *in principle* — not expensive, not unlikely. Impossible.

> ### Quantum checkmate
> **No unitary generated by your algebra maps your state into the safe subspace.**

Decidable from `g`. Computable in milliseconds with the Part III bitset closure. Verifiable, structural, and exactly the shape of the real thing.

The intermediate condition follows too. **Check**: your state has overlap with the pursuer's absorbing subspace. **Checkmate**: that remains true for every point in your reachable orbit.

### Why this makes absorption strategic rather than greedy

Eating an agent grows your algebra, which grows your reachable orbit, which makes checkmate *harder to deliver against you*. That is **material**.

The cost is Part III's: `Var[∂C] ∝ 1/dim(g)`. A larger algebra means you can no longer learn. That is **position**.

> **Material and position, with an exchange rate that is a published theorem.**

I do not know of another game where that sentence is literally true. In chess the value of a bishop is a convention refined by tradition. Here it is a measured quantity with a citation.

*(Note on prior art: Cantwell's Quantum Chess replaced checkmate with king-capture, because checkmate is awkward to define when pieces are in superposition. The controllability definition is a clean answer to that problem — it is well-defined on superposed states because it is a statement about the orbit, not about occupancy. Worth saying carefully in the docs, as a technical note and not as a dunk.)*

---

## 4. Phase structure, and an endgame tablebase that is an eigenvector

Chess has three regimes. Openings are memorised, middlegames are searched, endgames are solved by tablebase. That structure is a large part of what sustains interest across a lifetime.

Overtone gets the same structure free, from Axiom II's arrow:

| Phase | Physical state | Computational character |
|---|---|---|
| **Opening** | high coherence, ballistic, full superposition | classically hard; theory accumulates |
| **Middlegame** | partial decoherence, mixed state | search |
| **Endgame** | fully decohered → classical random walk | **exactly solvable** |

And the endgame really is solvable, by the machinery already specified. Once the walker has decohered, the position is a classical MDP — and Part V §2's linearly-solvable MDP turns Bellman into a **largest-eigenvalue problem** under `z = exp(−v/λ)`.

> **Overtone's endgame tablebase is an eigenvector.**

Chess endgame tablebases cost decades of compute and terabytes of storage. Yours is a Perron–Frobenius eigenvector computed live, in front of the player, the moment coherence runs out. The transition from "no algorithm can evaluate this" to "here is the exact answer, instantly" happens *within a single game*, driven by a physical quantity.

That is a better phase structure than chess has, and it costs one eigensolve.

### Tempo and zugzwang, also free

**Tempo** is coherence. Every action spends it; the clock is physical, not a convention.

**Zugzwang** — the position where you would rather not move — arises naturally, because evolving coherently is often better than measuring, yet you must sometimes measure to aim. And the Zeno structure (Part VI §2.3) means the *option* to measure is itself a liability. Chess's strangest property appears without being designed.

---

## 5. The rules, in full

Axiom I demands legibility, and "simple yet fiery" demands the rules fit on one screen. Here they are, complete:

```
1.  You are an amplitude field on the maze.
2.  You hold a hand of generators. Together they close into your algebra g.
3.  Each turn: apply one generator to one region, or measure.
4.  Applied generators evolve for k coherent steps, then the turn passes.
5.  Coherence only decreases. dim(g) only increases.
6.  Overlapping an opponent merges algebras: g ← closure(g_you ∪ g_them).
7.  You lose when no unitary in g reaches a safe state.
```

Seven lines. That is the entire game.

Every mechanic from Part VI survives inside this structure as *content* rather than as rules: exchange statistics decide what overlap does (VI §1), the decoherence front drives the clock (VI §2), braiding remains available and remains topologically protected (VI §4), and the eight traps become terrain (VI-A). The turn structure did not replace them; it made them legible.

**If a rule needs a paragraph, cut it.** Chess's rules fit on a card and its literature fills a library. Complexity must come from interaction, never from rule count.

---

## 6. The dial chess does not have

Chess's difficulty is fixed forever. Overtone's is a parameter, and this is the strongest claim in the entire seven-part series.

```
low  dim(g), low  χ   →  g-sim and MPS both work
                      →  positions are efficiently evaluable
                      →  the game is SOLVABLE

high dim(g), high χ   →  no efficient classical representation exists
                      →  positions are NOT efficiently evaluable
                      →  the game is PROVABLY HARD
```

> **Overtone contains a slider that moves it between the complexity class of checkers and a class strictly beyond chess.**

In one session you can show a position whose optimal move is computed *exactly* in forty milliseconds by the Part V eigensolve — then move one slider and show a position no classical algorithm can evaluate efficiently, with the Part III dequantization test as the proof.

No board game can do that. Chess is a fixed point in complexity space; Overtone is a trajectory through it, and the player is holding the parameter.

This is the answer to "how do we make it as fiery as chess." Not by adding rules. **By exposing the parameter that governs its complexity class and putting it on screen.**

---

## 7. What "interesting for agents" actually means

Your instinct — that this is a game for agents rather than humans — is right, but it needs sharpening.

Chess is tuned to human cognition: a branching factor a person can survey, a board a person can hold in working memory, piece values a person can learn in an afternoon. It is a *cognitive fit*.

The quantum-native analogues:

| Human-tuned in chess | Agent-tuned in Overtone |
|---|---|
| ~35 moves — surveyable by a person | branching set by the generator basis — tune it to the *agent's* search budget |
| 8×8 board — fits working memory | window size tuned to what the agent can represent |
| material values learnable in a day | generator values that are **not** learnable in a day, because they depend on `dim(g)` non-additively |
| positions evaluable by intuition | positions provably not evaluable by any efficient classical heuristic |

The last row is the point. Chess is interesting to humans partly because human intuition *almost* works. **Overtone is interesting to agents because classical intuition provably does not** — the evaluation function is hard for a reason with a theorem behind it, not because nobody has found the trick yet.

That is a genuinely different kind of game, and it is the honest version of what you were reaching for.

---

## 8. Measure your own depth

Lantz et al. give a measurable `d`, and Overtone is an instrument. So measure it, and use it as the design loop.

**Protocol.** Run agents at increasing compute budgets — search depth, training steps, `dim(g)`, `χ`, coherence budget — and plot win rate against `log(compute)`. Report:

- the **slope** of the rising region
- the **number of orders of magnitude** before saturation
- the **strategy ladder**: how many distinguishable skill tiers exist, measured by round-robin Elo among agents at different budgets

Then **tune the maze parameters to maximise `d`.** Substrate, generator basis size, coherence budget, `k`, trap density — all of them are knobs, and `d` tells you which way to turn them.

**Publish the ladder against chess, Go, checkers, and tic-tac-toe on the same axes.** A repository that measures its own game's depth, on a published metric, and plots it beside chess, is something that does not exist. It is also the most credible possible answer to "is this actually interesting?" — because if `d` is small, the plot says so.

**And if `d` is small, report it and fix the parameters.** The whole point of building this inside an instrument is that you find out in a week rather than a year.

---

## 9. The caveats, and what to do about them

You asked me to mitigate rather than hand-wave. Here they are straight.

**9.1 Turn-based conflicts with continuous evolution.** A quantum walk evolves continuously; chess alternates. *Mitigation:* a turn **is** a coherence block. Choose a generator, it evolves `k` coherent steps, the turn passes. `k` is Part VI's coherence budget, already in the design. This is principled rather than a patch — `k` is the physical parameter that interpolates between "one decision per step" and "few decisions, long evolution."

**9.2 Branching factor must be tuned, not guessed.** Too few generators and you have tic-tac-toe; too many and legibility dies. *Mitigation:* measure the average legal-move count every game and report it, exactly as chess reports ~35. Target 25–40. This is a measured statistic, not a design intention.

**9.3 Scale — the honest one.** Two agents' joint quantum state on an infinite maze is not representable. *Mitigation, three layers:* the light cone bounds support to radius `t` (Part II §3); turns are short, so `t` stays small; and `g-sim` handles polynomial `dim(g)` at high qubit count (Part III §4). Concretely, **play inside a windowed region of about 32×32 while the endless maze scrolls around it.** The arena is a local engagement inside an endless world — which is exactly how chess works, an 8×8 window on an infinite space of possible games.

**9.4 It might not be fun.** Genuinely possible, and no amount of elegant physics guarantees otherwise. *Mitigation:* §8. Measure `d` **before** building the interface. If the ladder is flat, the design is wrong and you will know quickly and cheaply.

**9.5 Scope.** This is a seventh part on a project that already has six. *Mitigation:* §10 — it replaces rather than adds.

---

## 10. This supersedes Part VI

Part VI proposed a free-form real-time arena. Part VII proposes a turn-based one. **Do not build both.** They are two designs for one tab, and shipping both would produce a project that cannot say what it is.

Build Part VII. The reason is Axiom I: Part VI's arena had continuous, illegible decisions, which is precisely the property that produces large state spaces and no depth. The turn structure fixes the one thing that was wrong with it.

Nothing is lost. Every Part VI mechanic survives as content inside the rules in §5 — statistics, absorption, the decoherence pursuer, braiding, and all eight traps. They stop being a physics sandbox and become a game with a win condition, which is what they needed.

Rename `Braid` to `Orbit`. Keep the braid diagram as a panel.

---

## 11. Build order

**M34 — checkmate.** Reachable-orbit computation from `g`; the checkmate predicate. Acceptance: on small systems, verify against brute-force reachability that the predicate is exactly correct. **Build this first — it is the piece that makes it a game rather than a sandbox, and it is mostly already implemented in `overtone-lie`.**

**M35 — the strategy ladder.** Generator basis, turn loop, and `d` measurement, all headless. **No interface.** Acceptance: a published `d` curve with a rising region spanning at least three orders of magnitude of compute. *If it does not, stop and retune before writing a single line of UI.*

**M36 — the endgame eigensolve.** LMDP tablebase triggered by full decoherence. Acceptance: the exact solution appears within 100 ms of the coherence transition, and matches brute-force optimal play on small boards.

**M37 — the complexity dial.** §6 as an interface: one slider, with a live verdict on whether the current position is efficiently evaluable, backed by the dequantization test.

**M38 — the board.** Interface, windowed arena, generator hand, the Part VI mechanics folded in as content.

**M39 — the ladder published.** `d` plotted against chess, Go, checkers and tic-tac-toe. This is the README's headline figure.

---

## 12. Traps

- **Do not add maze to add depth.** More states is the cheap axis and it moves `d` toward zero. Every instinct to expand should be checked against §8 first.
- **Do not build the interface before M35.** If `d` is flat, the interface is wasted work and you will have become attached to it by then.
- **Do not let the generator basis grow.** Six to eight. Chess has six pieces and a library of literature. Rule count and depth are close to unrelated.
- **Do not soften the checkmate predicate.** It is exact, decidable, and structural. A score threshold instead would be the single change that turns this back into a toy.
- **Do not hide the complexity dial.** It is the most remarkable property the game has and the only one no other game can claim.
- **Do not both-build Parts VI and VII.** §10.
- **Do not claim depth before measuring it.** The entire credibility of this part rests on `d` being reported honestly, including if the number is disappointing.

---

## 13. References

- Shannon — *Programming a computer for playing chess*, Phil. Mag. 41 (1950). The `10^120` game-tree bound and the ~35 branching factor.
- Allis, van den Herik, Herschberg — decision complexity versus space complexity; the framework Schaeffer used.
- Schaeffer, Burch, Björnsson, Kishimoto, Müller, Lake, Lu, Sutphen — *Checkers is solved*, Science 317, 1518 (2007). Eighteen years, `10^14` calculations, and a draw.
- Lantz, Isaksen, Jaffe, Nealen, Togelius — *Depth in Strategic Games*, AAAI Workshops 2017. The `d` metric and the strategy ladder. **The most important reference in this document.**
- Cantwell — *Quantum Chess: developing a mathematical framework and design methodology for creating quantum games*, arXiv:1906.05836 (2019). Prior art; note especially the measurement rule introduced to keep the state classically tractable, which is the design decision Part VII deliberately inverts.
- d'Alessandro — *Introduction to Quantum Control and Dynamics*. Controllability and reachable sets from the dynamical Lie algebra; the formal basis for §3.
- Todorov — *Efficient computation of optimal actions*, PNAS 106, 11478 (2009). The endgame eigensolve.
- Ragone et al., Nat. Commun. 15 (2024); Larocca et al., Nat. Comput. Sci. 3, 542 (2023). The material-versus-position exchange rate in §3.
