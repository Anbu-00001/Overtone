# OVERTONE PART X — Gauntlet

**The on-ramp.** Tenth companion. Adds no mechanics. **Tab name:** `Gauntlet`, and it goes *first* — see §7.

---

## Corrections

**The body below is unchanged**, per the convention in [`docs/spec/README.md`](README.md).
Part XI is a survey written after this document and amends it in five places; three further
corrections come from measuring the repository against its own claims.

| Section | Status | See |
|---|---|---|
| §0 | **amended** — "the on-ramp" becomes "one on-ramp, and probably not the largest". Hello Quantum's explanation reached its audience at 58% from search engines and **1% from the in-app link** | Part XI §2 |
| §3 L4 | **amended** — sonification is not outstanding work. It shipped in Phase 4: `web/js/sound.js`, `sonification_tones(k, lambda)`, and a `sonify` toggle already on the page | PHASES Phase 4 |
| §3 L9 | **superseded** — the ending must read the instrument, not a written number. `effective χ = 4` is asserted here; the measurement at `n = 4, L = 3` is **`χ = 3`**, and the gate asserts it. The ending's punchline is whatever `dequantize` returns on the player's own final circuit | PHASES Phase 17 |
| §6, §9 M54 | **superseded** — the debrief must not live inside the Gauntlet. Nine **standalone, search-indexed explainer pages**, complete without the game, with the Gauntlet deep-linking into them | Part XI §2 |
| §9 M53 | **amended** — acceptance is a knowledge-transfer test, not enjoyment: *can a naive reader state, unprompted, why L3 was impassable?* | Part XI §3 |
| §9, new M54b | **added** — a solution histogram on `(dim g, χ, work units)`: three antagonistic metrics, no leaderboard, no rewards | Part XI §5 |
| §8 | **superseded** — "zero new engine work" is not true. The JS budget is 1200 lines and stands at 1186, and its rule is that **no panel computes a physical quantity**, so every wall predicate and pass condition has to be Rust behind the wasm boundary | `scripts/check_js_budget.sh`; PHASES Phase 17 |

---

---

## 0. You have been right for six rounds and I have been answering the wrong question

Every time you have proposed stakes — the arena, the chase, consumption, chess, now this — I have said some version of *"the naive form would discredit the project"* and redirected to a more rigorous version. That was correct about the naive form and wrong about the underlying need, because the need never went away. Six requests is not restlessness. It is a gap.

Here is the gap, stated properly.

**Lab, Lattice, Closure and Orbit are instruments. Instruments serve people who already care.** A visitor who does not already know what a dynamical Lie algebra is has no reason to stay eleven seconds, and nothing in the project recruits them. Grok's landscape report confirmed the empty space is live training-dynamics visualisation — a *researcher* space — but also confirmed that traction comes from instant browser demos and tutorial quality, and that Quirk is the browser benchmark. Quirk is a toy in the good sense: you drag gates, things happen, and understanding arrives afterwards.

**Overtone has no equivalent of that.** The instruments assume their audience. Nothing builds one.

So: build this. But build it as the thing it actually needs to be, which is not a game with physics in it.

---

## 1. What you are actually asking for is a presentation layer

Go through your proposal element by element and notice that almost none of it is new:

| You proposed | It already exists as |
|---|---|
| items that make you stronger | **absorption** — Part VI §3, acquiring generators grows your algebra |
| one item only, active or passive | **the small generator hand** — Part VII §2, Go's lesson |
| consume the other to become stronger | **Lie closure on merge** — Part VI §3.1 |
| eliminate the incompetent between waves | **king-of-the-hill** — Decisions 02, Part VIII |
| gladiators fighting | **exchange statistics** — Part VI §1 |
| levels that get mathematically harder | *(the one genuinely new thing — see §3)* |

You have independently re-derived most of the arena from Parts VI and VII. What you are asking for is not a fifth system. **It is a fifth view onto systems that already exist, arranged as progression instead of as instruments.**

That reframe is what makes this affordable and what makes it safe.

---

## 2. The rule that still holds, unchanged

Part VI §0: **every mechanic must be a theorem.**

The Gauntlet does not weaken it, because the Gauntlet adds no mechanics. It adds an *order in which you meet them*. Part VI's danger was invented rules; Part VII fixed that with theorem-only rules; Part X arranges those rules into a sequence. Nothing new enters the physics.

If a level ever needs a rule that is not already in Part VII §5's seven lines, that level is wrong and gets cut.

---

## 3. The one new thing: each level is a theorem you cannot climb past

This is the idea that makes the whole thing worth building.

> **A level is a wall you cannot pass until you understand why. The key is never skill. It is always a capability, and every capability is a result from the literature.**

You do not get better at level 3. You get a bigger algebra, and the game shows you exactly which addition was the one that mattered.

That makes the Gauntlet **the best explainer this project can possibly have**, because a reader who hits Part I's frequency ceiling as a wall understands it in a way that no diagram achieves. You do not read the theorem. You run into it.

### The nine levels

Each has a **wall** (the theorem), a **key** (the capability), and a **lesson**.

**L1 — Spread.** *Wall:* reach the exit before the decoherence front arrives. *Key:* none — just evolve coherently instead of measuring. *Lesson:* ballistic beats diffusive, and you feel the difference before anyone names it.

**L2 — The dark corridor.** *Wall:* a passage your amplitude will not enter. *Key:* a phase gate that flips the interference. *Lesson:* amplitudes cancel, and you can engineer where.

**L3 — The ceiling.** *Wall:* structure at frequency 3; your encoding depth is 2. **You score exactly zero, forever, and the game says so in those words.** *Key:* a data-reuploading layer. *Lesson:* your circuit has a hard frequency ceiling and no amount of effort crosses it. *(Part I §7.1 — the crispest result in the project, delivered as a brick wall.)*

**L4 — Resonance.** *Wall:* structure at a non-integer frequency. Adding layers does not help. *Key:* trainable input scaling `λ`. *Lesson:* you can tune the comb. **With sonification on (Part IV §5.1), you hear the beat frequency fall to zero as it locks.** This is the moment the project is named for.

**L5 — The cage.** *Wall:* an Aharonov–Bohm cage with zero escape amplitude. *Key:* change your coin. *Lesson:* confinement from geometry and phase alone — and the escape condition is a property of your own policy, not of the maze.

**L6 — The feast.** *Wall:* an opponent with a larger algebra. *Key:* absorb it. *Consequence:* your gradients collapse and you can no longer adapt. *Lesson:* Part III's tradeoff, felt rather than read. Most players will over-eat here, and that is the level working.

**L7 — The front.** *Wall:* decoherence pursuing you through an open maze. *Key:* coherence budgeting. *Lesson:* information costs speed, and panic-measuring freezes you in place. *(Zeno. Do not warn them. Put the citation in the debrief.)*

**L8 — Checkmate.** *Wall:* your reachable orbit contains no safe state. *Key:* a generator that expands the orbit. *Lesson:* losing is a controllability statement, not a score.

**L9 — The mirror.** *Wall:* none. You win. Then the dequantization test runs on your victorious agent and reports **effective χ = 4.** *Lesson:* the field's central open question, delivered to someone who just spent twenty minutes earning the context to feel it.

L9 is not a level. It is an ending, and it is the best one available. It is also Decisions 02 §5's positioning executed as experience rather than as a README paragraph — and it sends the player straight to `Closure` to find out what just happened, which is the on-ramp completing its job.

---

## 4. Items are generators, and the inventory limit is a theorem

Your instinct to allow only one item is right, and the reason is better than the rule.

**Do not implement a slot limit.** Let the player carry as many generators as they like. The cost is already physical:

```
each generator added  →  dim(g) grows
dim(g) grows          →  Var[∂C] ∝ 1/dim(g) shrinks
                      →  you can no longer learn
```

So a greedy player becomes powerful and rigid, exactly as Part VI §3.2 specified. **The inventory limit enforces itself, and the player discovers it rather than being told.** "One item is usually right" becomes something they work out, which is worth far more than a rule that says so.

That also means item acquisition is a real decision with a measurable cost, rather than a strictly-positive pickup. No game has that. This one gets it free because the physics already charges for it.

**Active versus passive** maps cleanly: a generator you apply on your turn is active; a coin or statistics choice that changes how your field behaves is passive. Both are already in Part VII §5.

---

## 5. Elimination is already the league

Waves that cull the incompetent is king-of-the-hill (Decisions 02 §4), and consumption-to-grow is absorption. So the multi-agent arena you are describing is **Part VIII's league with a different presentation.**

Build it *second*. The Gauntlet needs nobody; the league needs participants. Ship the on-ramp, let it recruit, then the arena has people in it.

When you do build it, the framing from Part VI §5.6 still holds and is what keeps it legitimate: **the arena is the evaluation harness.** Opponents are Menagerie elites, a match is an evaluation run, and the results feed the Atlas. It is not a game bolted onto a research tool. It is the evaluation protocol with a human in the loop.

---

## 6. Visual direction

You named Skul, Tomb of the Mask, and Prince of Persia. **The one to learn from is Prince of Persia (1989)**, and the lesson is *restraint*: a handful of colours, one figure, mostly empty screen, and every ounce of atmosphere coming from motion and negative space rather than from detail. Tomb of the Mask contributes one thing — **instant readability at a glance**, high contrast, no ambiguity about what is where.

Neither requires a sprite, and Part VI §8's rule still stands. **The amplitude field is the character.** It moves, it has a shape, it deforms when it hits a wall, it flinches when the front approaches. That is more animate than a sprite, because a sprite has a fixed silhouette and a field does not.

### The mode shift

Keep the design system. Change the density.

| | Instruments (Lab / Lattice / Closure / Orbit) | Gauntlet |
|---|---|---|
| information density | maximal — every number on screen | minimal — one thing at a time |
| cells | small, many | **large, few** |
| motion | continuous, driven by data | slow, deliberate, one event per beat |
| negative space | little | **most of the screen** |
| readouts | always visible | one, and only the one that matters this level |

Same ground (`#12151A`), same phase-as-hue colour law, same monospace numerals. But the Gauntlet breathes and the instruments do not, and that contrast is what makes it read as a different mode without abandoning the language.

**The debrief is where the density returns.** When a level ends, the screen fills with the instruments — the spectrum panel, the gradient variance, the closure — showing what just happened to you, with the citation. That is the handoff: play sparse, explain dense, and the explanation is the same panel that lives in the other tabs.

---

## 7. Where it sits, and why the earlier rule does not apply

**Nav order: `Gauntlet · Lab · Lattice · Closure · Orbit`. Gauntlet first.**

Part VI §5.5 said never put the game before the instruments, and that rule was written about a free-form arena that could make the project look unserious. **A teaching layer is a different object.** Its entire purpose is to make the instruments legible to someone who arrives without context, and an on-ramp behind a door is not an on-ramp.

**First-time visitors land on Gauntlet. Returning visitors land on Lab.** The recruiter does its job once and then gets out of the way.

If that ever feels wrong, the test is Q10's: does every clause map to a shipped instrument? The Gauntlet's every level maps to a panel in another tab. That is what makes it a front door rather than a facade.

---

## 8. What this costs

Less than it looks, because of one thing: **the Gauntlet is `Overtone-100` with an order and a narrative.**

Part VIII §8 already plans a curated puzzle set with exact ground truth from the M36 eigensolve. That artifact needs to exist anyway, it works with zero participants, and it is the more citable of the two things Part VIII proposes. The Gauntlet is nine of those positions, sequenced, with a debrief attached.

So the marginal cost is:

- **presentation** — the sparse mode in §6
- **sequencing and debrief copy** — real work, mostly writing
- **nine level definitions** — parameter sets, since every mechanic already exists

**Zero new physics. Zero new mechanics. Zero new engine work.**

---

## 9. Build order

**M53 — L1 and L3 only.** Spread and the ceiling. The cheapest possible test of whether the format works, and L3 is the one that has to land. Acceptance: a reader with no quantum background hits the wall at L3, cannot pass it, and correctly explains why afterwards. **Test this on an actual person before building the other seven.**

**M54 — the debrief.** The dense-explanation screen, reusing existing panels. This is what converts play into understanding, and if it does not work the whole layer is a game after all.

**M55 — L2, L4, L5.** Interference, resonance, the cage. L4 needs sonification (Part IV §5.1), which is the highest impact-per-line item in the series and has been waiting since Part IV.

**M56 — L6, L7, L8.** Absorption, the front, checkmate.

**M57 — L9.** The mirror. Write this one last and write it carefully; it is the ending and it carries the project's actual thesis.

**M58 — nav and first-visit routing.**

Arena and league: after Part VIII lands and there are agents to fight.

---

## 10. Traps

- **No new mechanics. Ever.** If a level wants one, cut the level. §2 is the whole safety property.
- **No sprite, no avatar, no character.** Ten parts of holding this line. The field is the character.
- **No score, no stars, no XP, no combo counter.** A level is passed or it is not. The debrief reports observables.
- **Do not warn the player about the wall.** L3's entire value is hitting it. The explanation comes after.
- **Do not let L9 be triumphant.** It is a quiet, deflating, honest ending, and that is what makes it land. If it reads as a twist reveal, rewrite it.
- **Do not add L10.** Nine theorems is a curriculum; twelve is content.
- **Do not let the Gauntlet become the project.** It is the door. `Lab` is the room.
- **Test M53 on a real person.** Every other milestone in this series can be validated by a test. This one cannot, and building nine levels on an unvalidated format is the expensive mistake available here.

---

## 11. The thing I am still uneasy about

I have argued twice that a game layer could retroactively make Parts I–III look like decoration, and I do not think this version has that failure mode — it teaches the instruments rather than competing with them, and its every level points back into them.

But **the risk has not disappeared, it has moved.** It is now in the debrief. If the debrief is thin, the Gauntlet is a nice little maze game with quantum flavouring, and a visitor leaves having enjoyed themselves and learned nothing. If the debrief is good, it is the best explainer of variational quantum machine learning that exists, because it explains a thing the reader has just personally run into.

**M54 is the milestone that decides which project this is.** Weight it accordingly — it is mostly writing, it will feel less important than the levels, and it is not.
