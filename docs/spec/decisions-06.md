# OVERTONE — DECISIONS 06

Three questions. **Q16 is a sharper result than you presented it as**, and it changes Q17 enough that I would reverse your lean there.

---

## Q15 — Decisions 04 re-sent, and the fifth feature is recoverable

The file is attached. But two corrections to the framing.

### The "five features" claim is not in Decisions 04

Decisions 04 has no §6 — its sections are Q11–Q14 plus the commit flag. The row you are thinking of is **Decisions 05 §6**, in the changes table: `Frozen eval vector | five features | unchanged — five`.

The origin is a chain across three documents:

```
Decisions 03 revised, Q1   →  six features listed
Decisions 04, Q13          →  temperature moves to [agent.search]
                           →  "the frozen eval vector stays at five"
Decisions 05 §6            →  records five, unchanged
```

### The five, explicitly

From Decisions 03 revised Q1, minus temperature per Decisions 04 Q13:

```
1. coherence          the clock              Player.coherence
2. dim_g              material               Lie closure, memoised
3. orbit_size         reachability           orbit_dimension
4. safe_set_size      proximity to loss      safe-set spec
5. average_branching  mobility               move generation at node
```

So the fifth is recoverable and there was nothing to guess.

### And the disagreement is probably not a disagreement

Decisions 03 Q1 did not just list six — it gave a **cut rule**:

> A feature is dead weight if it does not vary across the moves available at a node. Measure the sibling variance of each feature across a sample of positions. **Cut anything with near-zero sibling variance.**

It flagged two as at risk: `coherence` (survives only if `measure` and generator-application have different costs) and `dim_g` (discriminates absorb from non-absorb moves and nothing else).

You ran that test — you quote its numbers in Q13. **So if you froze four, you most likely cut one on measured sibling variance, which is the process working exactly as specified.**

**What to record is not a disagreement. It is a measurement.** Replace the `PHASES.md` note with the actual result:

```
Frozen eval vector: 4 features (v1)
  coherence, orbit_size, safe_set_size, average_branching
Cut from the Decisions 03 candidate list:
  temperature  — moved to [agent.search], Decisions 04 Q13
  dim_g        — sibling variance <x>, below threshold, Decisions 03 Q1 cut rule
```

*(Substitute whichever you actually cut and its measured spread.)* A recorded measurement with a citation to the rule that justified it is a better artifact than a noted inconsistency between documents, and it is the honest description of what happened.

---

## Q16 — Yes, and your result is stronger than "almost no signal"

**Confirmed, and sharpen it further.** But first: state precisely what you have found, because it is a cleaner result than you are giving it credit for.

### Temperature ≡ −1 means every region is *cold*

In combinatorial game theory, a game whose value is a **number** has temperature −1 — the minimum. That is not "low signal." It has a specific meaning:

> **A number is a position in which neither player wants to move.** There is no urgency, no contest, nothing to decide. Optimal play in a sum of numbers is arithmetic: add them up, the sign decides.

So a field that is identically −1 is not a weak measurement. It is a strong one: **every subgame in your decomposition is resolved, and the decomposition contains no decisions.**

### Your refinement to Prompt B is right, and it is the more urgent question

*"Whether a decomposition into three contiguous qubit blocks is the wrong decomposition, rather than whether temperature is the wrong idea."* Yes. And it is more urgent than the soundness question, because it is actionable whichever way the soundness question lands.

Note also that Part IX §5.3 assumed something you may not have implemented: it says decomposition is available because *"the maze's corridor structure provides weakly-interacting regions naturally."* **Corridors are spatial. Contiguous qubit blocks are not.** If those two decompositions came apart in implementation, the spec's justification does not cover what was built — and that alone could produce all-numbers, since qubit blocks may have too few contested degrees of freedom to stay hot.

Add to Prompt B, as its own numbered item:

```
6. DEGENERATE DECOMPOSITIONS
When a decomposition yields subgames that are all numbers (temperature = −1
everywhere), what does that indicate? Is it evidence that (a) the decomposition
is too fine — subgames are individually resolved; (b) the decomposition is
along the wrong axis — the chosen components are not where the contest is;
or (c) the position genuinely is cold and the game has no tactical content at
this scale? Is there published guidance on diagnosing which? What decompositions
have been tried and rejected in the CGT literature for producing degenerate
subgames, and what characterises a good decomposition?
```

### The consequence you have not drawn

Here is what makes this a bigger result than a temperature diagnostic.

**Phase 9 found that `d` saturates at n = 4. You have now found that the game is cold at n = 4.** Those may be the same fact.

A game whose components are all numbers has no tactical content — play reduces to counting, and a counting game has no strategy ladder, because more compute buys you nothing once you can add. If the position is genuinely cold at n = 4, then:

- `d` saturating is not (only) a thin strategy language, which was my Q1 diagnosis
- it is the game having no decisions at that width
- and **no amount of search-language enrichment will fix it**, because there is nothing to search

That hypothesis is cheap to test and it changes Q17 completely.

---

## Q17 — I would reverse your lean, and the comparability argument does not hold

### The reason you gave is retired

*"Comparability says n = 4, since that's where Phase 9's `d` was measured."*

**Decisions 03 Q5 retired `d = 6` explicitly**, and you agreed with the phrasing — a quietly-superseded number is worse than a withdrawn one. It was measured in evaluations rather than work units, 40 games per rung, adjacent-rung ladder rather than a Goodman grid. It is not comparable to anything, which is exactly why it was withdrawn.

**So comparability with a retired number is not a reason to hold a width.** With that gone, `n = 4` has no special status, and the right width is whichever produces a measurable ladder.

### The n = 4 measurement is underpowered anyway

12 games per rung puts ±0.14 on every rung. Decisions 03 Q6's cost table: roughly 1,600 games for ±10 Elo, 6,400 for ±5. Twelve games resolves nothing at any width. So `n = 4` is not the cheap safe option — it is a measurement that cannot conclude, at a width that may be cold.

### And Q6's parallelism is not being used

*"Widening costs wall clock against Q6's 6 h cap."* The 6 h cap is **per job**, and the Free plan allows **20 concurrent jobs**. Decisions 03 Q6 worked this: 15 pairings, one job each, 1,600 games per pairing gives a 13.5 s per-game budget. You are budgeting as though the grid must fit in one job.

That is a large factor, and it is available before any width decision is made.

### Ruling: diagnose the width before measuring depth at it

Do not run a Skill Trace at a width until you know the width has anything to trace. The diagnostic is **cheap and needs no games at all**:

**M39a — the coldness sweep.** Compute the temperature field at `n = 4, 6, 8` on a sample of positions. No matches, no Elo, just position evaluation. Report the fraction of regions that are numbers at each width.

```
all numbers at every width   →  the decomposition is wrong (Q16), and
                                 temperature is not your diagnostic. Fix that first.

hot structure appears at n=k →  that is your ladder width. Run M39 there.

never any hot structure      →  Prompt B item 6, and the ladder needs a different
                                 instrument to choose its width.
```

**M39b — the Skill Trace, at whichever width M39a identifies.** Goodman grid, parallelised across pairings per Q6, with `N` chosen from measured single-game cost.

That ordering costs a few hours of compute and it turns Q17 from a judgment call into a measurement — which is the house style, and which is more defensible than either width you were choosing between.

### On your instinct

Running an `n = 6` column was the right idea. You under-scoped it. It is not *"purely to see whether temperature ever earns its place"* — **it is the test of whether `n = 4` was ever a viable ladder width at all**, and if the Q16 hypothesis holds, it is the fix for the saturation rather than a side experiment.

---

## Summary

| | Ruling |
|---|---|
| **Q15** | Decisions 04 attached. The five-feature claim is in Decisions 05 §6, not 04. The fifth is recoverable and listed above. Record the cut as a measurement with its sibling-variance number, not as a disagreement. |
| **Q16** | Confirmed and sharpened. Temperature ≡ −1 means every subgame is *cold*, which is a strong result, not a weak signal. Add item 6 to Prompt B. Check whether you decomposed by qubit blocks where Part IX §5.3 assumed maze corridors. |
| **Q17** | Reversed. Comparability with a retired `d` is not a reason; 12 games resolves nothing at any width; Q6's 15-way parallelism is unused. **Run the coldness sweep first (M39a), then the Skill Trace at whichever width has hot structure (M39b).** |

**And the hypothesis worth holding onto:** `d` saturating at `n = 4` and the field being cold at `n = 4` may be one fact rather than two. If so, the ladder was never going to extend at that width no matter how rich the strategy language got — and M39a settles it for the cost of an afternoon.
