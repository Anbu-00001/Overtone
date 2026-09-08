# OVERTONE PART IX — Thermograph

**Dice, ancient puzzles, and the math of move generation.** Ninth companion. Extends Part VII.

---

## 0. Verdict, split three ways

**The dice idea is a trap as stated** — and you supplied the disproof yourself when you mentioned Snakes and Ladders. §1 explains why, and what D&D actually gets right, which is not the dice.

**The Chinese-rings instinct is excellent** and produces a fact that demolishes the "endless possibilities" premise more cleanly than anything in Part VII. §3.

**The move-generation question led somewhere better than expected.** Combinatorial game theory has a rigorous notion of *temperature* — the urgency of a position — and it gives Overtone a move-ordering heuristic that doesn't violate the no-invented-numbers rule, plus the best heat map in the project. §5. This is the section to read if you only read one.

---

## 1. The dice trap, and what D&D is actually for

### 1.1 Why "add dice" is backwards

Randomness generally **reduces** depth `d`, because it decouples decision quality from outcome. Snakes and Ladders is pure dice and has `d ≈ 0` — the canonical example in the Lantz et al. framework, and the one you reached for yourself. Bolting a d20 onto Overtone would move it toward Snakes and Ladders, not toward D&D.

### 1.2 But Overtone already has the best dice ever built

The Born rule. Measurement outcomes are irreducibly random. Overtone was never deterministic like chess — it is already a dice game and has been since Part I.

The difference from D&D is the whole point:

| D&D | Overtone |
|---|---|
| roll a **fixed uniform** distribution | **build** the distribution, then collapse it |
| modifiers shift a fixed die | your entire turn *is* shaping the die |
| you know it's 1-in-20 | you can see the shape you made |

> **You don't roll the die. You build the die, then throw it.**

That is strictly richer than D&D's mechanic, it requires no new rules, and it has been sitting in the engine since M1.

### 1.3 What D&D actually gets right

The dice are the least interesting part of D&D. Three things underneath them are load-bearing, and Overtone has analogues for all three:

**The character sheet and its modifier stack.** Already built — Part IV §2.2, where every line is a measured observable rather than an authored number.

**Asymmetric complementary builds.** A party works because a rogue and a cleric cover different failure modes. That is Part VII §2's generator hand: heterogeneous by kind, not by degree.

**Advantage and disadvantage** — roll two dice, take the better or worse. This is the one D&D mechanic that is genuinely about *reshaping a distribution*, and it has an exact quantum counterpart:

> **Grover iterations are advantage.**

Amplitude amplification rotates probability toward a marked outcome. After `k` iterations the success probability is `sin²((2k+1)θ)` — a precise, closed-form advantage dial rather than a binary reroll. And Part VI-A §T1's soufflé problem means **too much advantage becomes disadvantage**, at a computable step count. D&D cannot have that. Overtone gets it free.

### 1.4 The natural 20 you designed yourself

Because you shape the distribution, you can deliberately leave three percent of your amplitude on something spectacular. A calculated long shot, chosen, not granted.

That is a strategic decision no tabletop player has ever made, and it is exactly the kind of choice that makes a moment memorable afterward.

---

## 2. The nerve-wracking answer: hold the frame before the collapse

You asked for nerve-wracking. It is already in the physics and it needs one interface decision to land.

The beat is **the measurement**. You have spent your coherence shaping an amplitude distribution. Now you measure. It resolves to one outcome.

```
1.  the distribution renders — the die you built, spread over outcomes
2.  hold. one beat. nothing moves.
3.  collapse. one outcome. the rest goes dark.
```

That is committed uncertainty with the stakes visible in advance, several times per game, and it beats a d20 because **you can see the die and you made it.** If your plan was good the distribution is sharp and the collapse is a formality. If you gambled, it is not.

Rules for the beat: hold the frame — do not animate through it. Show the actual amplitudes, never a cleaned-up bar chart. No sound cue, no shake, no confetti. The distribution rendering and going dark *is* the drama; anything added subtracts.

---

## 3. The Chinese Rings: hard with a branching factor of one

The medieval-Chinese instinct pays off, and it pays off with a fact that should reframe the whole project.

The Chinese Rings (九连环, baguenaudier) is roughly two thousand years old and still sold as a puzzle. Its mathematics:

- The state graph for `n` rings is **a path of length `2ⁿ − 1`**. Not a tree. A line.
- The solution is the **binary Gray code**, `G(i) = B(i) XOR (B(i) >> 1)`.
- Minimum moves: `(2ⁿ⁺¹ − 2)/3` for even `n`, `(2ⁿ⁺¹ − 1)/3` for odd — OEIS A000975: 1, 2, 5, 10, 21, 42, 85, 170, 341, 682.

Read the first line again. **The entire state space is a single path.** At every state there are exactly two moves: forward and back. Branching factor: one, effectively.

And it is *hard*. People have been failing to solve it for two millennia.

> **Difficulty does not come from the number of options. It comes from the difficulty of knowing which option is forward.**

This is Part VII Axiom III, demonstrated by a puzzle older than algebra, and it is the sharpest available rebuttal to "endless possibilities make it interesting." A seven-ring puzzle needs 85 moves and offers no choices at all. Your endless maze offers infinite choices and, as currently specified, no difficulty.

**Put the Chinese Rings in the docs as the counterexample.** It makes the argument in one figure.

### 3.1 The implementable version

The Chinese Rings state space is the `n`-bit hypercube traversed by a Gray code — each step flips exactly one bit. Part III represents Pauli strings as bitsets under XOR, which is the same object: `F₂²ⁿ` with single-bit-flip moves.

So build a substrate where **cells are Pauli strings and legal moves flip one generator.** A maze whose geometry is the algebra itself. Solutions are Gray-code paths, exponentially long, with a branching factor of two and no greedy shortcut. Part VII's spectral traps and Part VI-A's cages both land naturally on it because it is already the right kind of space.

Overtone has been carrying a hypercube around since Part III without walking on it.

---

## 4. Go: identical stones, and the two arrows

Two lessons, one of them a warning.

**Heterogeneity is optional if the geometry is rich.** Go has ~10¹⁷⁰ legal positions, a branching factor of 250, and **one piece type**. Every stone is identical. All the complexity lives in the substrate.

So Part VII's "give the player six to eight generators" is a design choice, not a requirement, and Go is the proof that the generator hand should stay small. **Do not add generators to add depth.** The substrates from Part IV are where richness should come from — a rule to keep in mind every time a ninth generator suggests itself.

**Additive versus subtractive.** Chess removes pieces and simplifies toward a solvable endgame. Go adds stones and complicates. Overtone has both arrows running at once:

```
coherence ↓   simplification   →  the endgame becomes exactly solvable   (chess-like)
dim(g)    ↑   complication     →  evaluation becomes intractable         (Go-like)
```

They pull in opposite directions, and the interesting play is in the crossing region — where the position is simplifying physically while becoming harder to evaluate algebraically. Neither chess nor Go has that, and it comes free from mechanics already specified.

---

## 5. Temperature — the section that matters

You asked how to script `N×N` moves. The best answer in the literature is not a data structure. It is a theory of *which move matters*.

### 5.1 What temperature is

Conway, Berlekamp and Guy's combinatorial game theory, developed for Go endgames in *Mathematical Go* (Berlekamp & Wolfe, 1994):

- A position decomposes into **independent local regions** whose game values **add**.
- Each region has a **temperature** — the urgency of moving there, computed by *thermography*, where the temperature is the base of the thermograph's mast.
- **Optimal play is to move in the hottest region.**
- Berlekamp estimated the empty 19×19 Go board at temperature ≈ 13.

Müller's *decomposition search* and *temperature discovery search* make it computational rather than theoretical. It is a real, rigorous, working method, and it is strikingly underused outside the Go research community.

### 5.2 Why Overtone needs it specifically

Part VII forbids invented numbers, which leaves a hole: a game needs move ordering, and the usual answer is a hand-tuned evaluation function — exactly the invented number the rules prohibit.

**Temperature fills the hole with a derived quantity.** It is not a heuristic somebody tuned; it is a computed property of the position with a proof behind it. It gives Overtone move ordering without violating its own constitution.

### 5.3 The heat map

And it gives the best visual in the project.

Render **temperature as a field over the maze.** Hot regions glow; cold regions recede. The player sees, at a glance, where the game actually is — which is genuinely hard to perceive otherwise, because an amplitude field spread across a maze does not tell you where the stakes are.

It is information-dense, physically motivated, and it makes the nerve-wracking quality legible: **you can watch a region heat up before it matters.** Tension you can see arriving is much better than tension announced after the fact.

Practical notes: decomposition needs the board to split into weakly-interacting regions, which the maze's corridor structure provides naturally, and the light cone bounds interaction range. Use temperature discovery search rather than exact thermography for live play; approximate is fine for a heat map and exact is affordable in the endgame.

### 5.4 Two temperatures

Combinatorial game theory borrowed "temperature" from thermodynamics as a **metaphor**. Overtone has actual thermodynamics — decoherence, entropy production, an arrow of time.

So put them side by side: **CGT temperature** (urgency of the next move) and **physical entropy** of the quantum state. Ask whether they track each other. Nobody has, because no game has ever had both.

This is now a running motif and it should be deliberate. Part IV pairs WFC's Shannon entropy with the walker's von Neumann entropy. Part IX pairs CGT temperature with physical temperature. **Same word, one metaphorical and one literal, measured together.** It is the project's characteristic move and it teaches more per pixel than any explanation.

---

## 6. The move-generation math you asked for

Three techniques, and you have already built the first one without noticing.

**Bitboards.** Chess engines represent a position as 64-bit words, one bit per square, and generate moves with shifts and masks. Part III §3 represents Pauli strings as bitsets and computes commutators with a popcount and an XOR. **The Pauli representation is a bitboard.** The move generator for Part VII's generator hand should be written in exactly that style, reusing `overtone-lie`'s primitives rather than inventing a parallel representation.

**Zobrist hashing.** Assign a random 64-bit key per (feature, location); the position hash is their XOR; each move updates it in `O(1)` by XOR-ing two keys. This gives transposition tables, repetition detection, and opening books. It composes perfectly with the Pauli XOR structure — the same operation is already doing the same job one level down.

**Decomposition search.** Müller's method: split the position into independent subgames, evaluate each, combine by CGT addition. This is what makes §5 tractable at scale and it is the actual answer to "how do you handle `N×N` possibilities" — you don't, you prove most of them are independent and stop looking at them together.

---

## 7. What to build

**M46 — the measurement beat.** §2. Smallest item here and the largest change to how the game feels.

**M47 — the Chinese Rings figure.** §3 as a documentation panel: state graph as a path, Gray code solution, `A000975` growth curve, plotted beside the maze's branching factor. The argument that reframes the project, in one figure.

**M48 — temperature.** Decomposition search plus temperature discovery search. Acceptance: reproduce a worked Go endgame temperature from Berlekamp & Wolfe as a unit test, then apply the same code to maze regions.

**M49 — the heat map.** §5.3. Acceptance: temperature updates live at 60fps on a 32×32 window.

**M50 — advantage.** §1.3: Grover iterations as an explicit player-facing dial, with the `sin²((2k+1)θ)` curve shown and over-rotation reachable.

**M51 — two temperatures.** §5.4. A panel and an honest answer about whether they correlate.

**M52 — the Pauli-string substrate.** §3.1. Lower priority; build it when the substrate set feels thin.

---

## 8. Traps

- **Do not add a die.** The randomness is already there and it is better than a die. Adding an external random source is the one change that would measurably lower `d`.
- **Do not animate through the collapse.** Hold the frame. The pause is the mechanic.
- **Do not clean up the distribution for display.** Show the amplitudes you actually have, including the ugly ones.
- **Do not expand the generator hand.** Go has one piece type and 10¹⁷⁰ positions. Richness belongs in the substrate.
- **Do not hand-tune an evaluation function.** Temperature exists precisely so you never have to, and a tuned eval would be the first invented number.
- **Do not claim CGT temperature and physical temperature are the same thing.** They share a name and possibly a behaviour. Measure it; report what you find; do not assume.
- **Do not skip M48's unit test.** If the implementation cannot reproduce a published Go endgame temperature, the heat map is decoration.

---

## 9. References

- Berlekamp & Wolfe — *Mathematical Go: Chilling Gets the Last Point*, A K Peters (1994). Temperature and thermography for Go endgames.
- Berlekamp, Conway & Guy — *Winning Ways for Your Mathematical Plays* (1982). The foundation; temperature as the base of the thermograph's mast.
- Berlekamp — *The economist's view of combinatorial games* (1996). Mean value and temperature, defined two equivalent ways.
- Müller — *Computer Go as a sum of local games*; decomposition search and temperature discovery search. The computational path from theory to a live heat map.
- Wolfram MathWorld, *Baguenaudier*; and *An Exploration of Sequence A000975*, arXiv:1608.08245. The `2ⁿ − 1` path state graph and the Gray-code solution.
- OEIS A000975 — 1, 2, 5, 10, 21, 42, 85, 170, 341, 682.
- Zobrist — *A new hashing method with application for game playing*, TR 88, Univ. Wisconsin (1970).
- Lantz, Isaksen, Jaffe, Nealen & Togelius — *Depth in Strategic Games*, AAAI 2017. Why §1.1 is a warning and not a preference.
