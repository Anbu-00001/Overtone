# OVERTONE — DECISIONS 04

Four questions and a flag. Q13 is right and lands better than you framed it. Q12 I would reverse — the dependency you cited does not exist.

---

## Q11 — Specs stay immutable. Your lean is right, and §11's phrasing was my error.

**Confirmed: `docs/spec/*.md` are immutable historical documents.** Corrections live in `DECISIONS-NN`, `PHASES.md` is authoritative for building.

**§11 was phrased ambiguously and that is on me.** I wrote *"Part II §P6 — Distinguish one-shot hitting, concurrent hitting, and marked-vertex search"* as the natural way to express a correction, but it reads as an edit instruction. It is not. Those are **statements of the corrected position**, to be recorded wherever the convention says. Read every §11 row that way, and treat the same ambiguity as resolved for any future decisions document.

### Why immutable is the right call here specifically

Not just tidiness. Decisions 02 §5 established that this project's asset is credibility in a field with a hype problem, and Decisions 03 §5 endorsed keeping the retired `d = 6` measurement as a visible artifact rather than deleting it. Same principle:

> **A project that shows its corrections rather than absorbing them silently is more trustworthy than one that always appears to have been right.**

The spec files with correction blocks *are* that story — nine documents written on reasoning available at the time, four decision documents recording what external research changed. That is a research record. Editing the specs in place would convert it into a tidy set of documents that appear never to have been wrong, which is both less true and less useful.

### Two sharpenings on your proposal

**Make the correction block section-specific, not a generic pointer.** A block reading "see PHASES.md for corrections" leaves the reader searching. It should be a table:

```markdown
> **Corrections to this document**
> | Section | Status | See |
> |---|---|---|
> | §P5 | amended | Decisions 03 §11 |
> | §P6 | superseded | Decisions 03 §3 |
> | §P7 | amended — no inherited speedup | Decisions 03 §2 |
```

A reader who opens Part II §P7 sees immediately that it is amended and where the amendment is. That is the only failure mode that actually matters: **someone implementing from a stale spec.**

**Add `docs/spec/README.md` stating the convention.** Right now it is implicit, which means a future contributor — or you in four months — will not know whether to edit. Three sentences: these are historical design documents; `PHASES.md` is authoritative for building; corrections are recorded in `DECISIONS-NN` and indexed at the top of each affected spec.

---

## Q12 — M41a first. Your stated dependency does not hold.

**The reason you gave for Kempe-first is wrong, though the instinct behind it is sound in general.**

You wrote that Kempe "de-risks the whole coined core before M41a starts depending on search over it." But M41a is the agent language freeze, and the language covers generators, search kind and budget, and eval features — all Orbit quantities. **None of them depend on the graph-general coined walk that Phase 5 builds.** Orbit runs on the maze lattice, which already exists. Phase 5 generalises the walk to arbitrary graphs for hypercube and welded trees. Those are independent.

The general instinct — validate cheaply before building expensively on top — is right. It just does not apply, because there is nothing to build on top of.

### What actually decides it

**Q13 just settled M41a's last open question.** The frozen vector goes from six features to five, and temperature moves to a different role entirely. The freeze is *settled right now*, and the sibling-variance experiment that was gating it is done — you cite its numbers in Q13.

Freeze while it is settled. Every day M41a sits open is a day something else can surface that reopens it, and Decisions 03 was explicit that freezing it wrong poisons every measurement afterwards. That risk does not decrease with waiting; it increases.

Against that: M41a unblocks ladder → board → league, three phases. Kempe unblocks nothing — it is a leaf.

**Order: M41a, then Kempe as the next self-contained task.** Kempe is still the right first Phase 5 milestone for the reason Decisions 03 gave — a settled 2005 theorem validates the core before you take on a 2024 one — it just does not need to precede the freeze.

---

## Q13 — Yes, and it fits better here than it ever did in the eval vector.

**Confirmed, and the reframing is a genuine improvement on my Decisions 03 ruling.**

Part IX §5.1 says *"optimal play is to move in the hottest region."* That is a statement about **which move to consider**, which is move ordering, literally. Putting temperature in the eval vector was my misplacement, not a failure of the quantity. Your numbers confirm it from the other direction: the best discriminator you measured, at twice `dim_g`'s sibling spread, and too slow for the inner loop. **That is the textbook profile of a move-ordering heuristic** — chess engines use expensive heuristics for ordering precisely because ordering happens once per node while evaluation happens at every leaf.

So: yes, and it makes Phase 12 load-bearing rather than decorative, exactly as you say.

### One correction: in MCTS, "ordering children" does nothing

This matters for the implementation.

Move ordering pays off in alpha-beta because good ordering enables cutoffs — best case reduces the effective branching factor to `√b`. **MCTS has no cutoffs**, so simply sorting the children of a node has no effect on search quality at all.

The correct place for domain knowledge in MCTS is a **prior in the selection term**:

```
PUCT:  argmax_a  Q(s,a) + c · P(s,a) · √(Σ_b N(s,b)) / (1 + N(s,a))
```

`P(s,a)` biases exploration toward moves the heuristic likes, with the bias washing out as visits accumulate. That is where temperature goes. (Progressive bias, in the Chaslot formulation, is the equivalent construction if you prefer an additive term.)

So the ruling splits by search kind:

- **`kind = "mcts"`** → temperature is the PUCT prior `P(s,a)`, normalised over the node's children
- **`kind = "negamax"`** → temperature is classic move ordering, where it earns cutoffs

Both are the same computation, entering at different points.

### Root-only, or shallow plies?

Root-only is affordable and safe: 714 µs per move is nothing.

But it is worth measuring the next step out before committing. At branching ~30, ordering to depth 2 costs roughly `30 × 714 µs ≈ 21 ms` per move — likely still affordable, and priors near the root are where they do the most work. Standard practice is expensive guidance in the first plies, cheap or none in the tail. **Measure it; if 21 ms fits, take it.**

### One design consequence for M41a

Temperature is now an engine-side computation applied uniformly, not an agent-chosen eval weight. But **how much an agent trusts it is a real strategic choice** — and Goodman's finding that algorithmic space determines measured depth argues for keeping genuine choices in the language.

**Ruling: one scalar in `[agent.search]`, not a sixth entry in `[agent.eval]`.**

```toml
[agent.search]
kind              = "mcts"
budget            = 4096
temperature_prior = 0.7     # 0 ignores temperature entirely
```

The engine computes temperature identically for everyone; the agent chooses how strongly it biases selection. **The frozen eval vector stays at five.** That keeps the eval clean, keeps the strategic choice available, and adds one number rather than a dimension.

---

## Q14 — Write it now, at reduced size, with a growth convention.

**Confirmed. A smaller honest table beats a promissory one, and that is not a close call given this project's positioning.**

Q10's scoped headline is safe to write today because `predict` exists and already refuses false claims. That is the part of the README doing the most work, and it does not depend on any Phase 10 panel.

### Two additions

**Make the table's growth a convention, not a chore.** Each new row lands in the same commit as the panel it cites. That makes the README self-maintaining rather than something that silently drifts and needs a big rewrite before launch. Write the rule into `CONTRIBUTING.md` so it survives you forgetting it.

**Say what is not yet covered.** After the four or five rows, one line: *"Other standard criticisms are addressed by instruments not yet shipped; rows appear here when the panels do."* That is the project's own honesty convention applied to its own front page — and it converts an incomplete table from something that looks like an oversight into something that reads as deliberate.

### On timing more broadly

You are at Phase 13 of a plan running to roughly M52, so launch is distant and this README will be rewritten. Write it anyway. Not for the reader — there isn't one yet — but because **the "does every clause map to a shipped instrument" audit is diagnostic.** Any clause you find yourself wanting to write that has no panel behind it is telling you something about the gap between the specs' ambition and what exists. That is worth learning now rather than at launch.

---

## The flag — agreed, and one thing to do about it

Agreed: do not rewrite history over eleven commits. The cost is real and the benefit is cosmetic.

But there is a thirty-second fix that closes it properly, and Q9 is the reason to bother. Add to `AUTHORS` or `NOTICE`:

```
All commits prior to 3ad4e5f were authored by <name>, committing
under the handle <handle>. Sole authorship through Phase 13.
```

Two reasons this is worth the thirty seconds:

**It makes the provenance explicit without touching history.** A future reader — or a lawyer, if the Q9 relicensing question ever becomes live — can establish single authorship from a file in the tree rather than by correlating a handle against commit metadata.

**Sole authorship is the easy case, and it is only true now.** cBioPortal's current problem is 209 contributors and no record. Yours is one contributor and a naming inconsistency. Documenting it while the answer is still one line is the cheapest version of that problem you will ever get to solve.

Do it in the same commit as the DCO adoption, whenever that lands.
