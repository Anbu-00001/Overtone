# The specification documents

**These are historical design documents. Do not edit them.**

Parts I through XI were written on the reasoning available at the time. Several of them contain
statements that later measurement or external research contradicted. Those statements are still
here, unchanged, and that is deliberate.

## The convention, in three sentences

1. **`docs/spec/part-*.md` are immutable.** They record what was believed when they were
   written.
2. **`docs/PHASES.md` is authoritative for building.** If a spec and `PHASES.md` disagree,
   `PHASES.md` is right and the disagreement is recorded there.
3. **Corrections live in `decisions-NN.md`** and are indexed in a table at the top of each
   affected spec, so a reader who opens a section can see immediately whether it still stands.

## Why not just fix the specs

Because the corrections are part of what the project is. Decisions-02 §5 established that this
project's asset is credibility in a field with a hype problem, and Decisions-03 §5 kept the
retired `d = 6` measurement as a visible artifact rather than deleting it, on the same
principle:

> A project that shows its corrections rather than absorbing them silently is more trustworthy
> than one that always appears to have been right.

Editing the specs in place would turn a research record into a tidy set of documents that
appear never to have been wrong. That is both less true and less useful. The failure mode this
convention protects against is exactly one thing — **somebody implementing from a stale
section** — and the correction table at the top of each spec is what prevents it.

## The documents

| File | What it is |
|---|---|
| `part-i-build-spec.md` … `part-xi-the-sweep.md` | The design specifications, in order |
| `decisions-01-phase-10.md` | Rules on the eight open Phase 10 questions |
| `decisions-02-synthesis.md` | Four external research passes as a claims audit |
| `decisions-03-revised.md` | The walk operator, the v1 feature vocabulary, the licence. Supersedes an earlier Decisions-03 in full |
| `decisions-04.md` | Spec immutability, build order, temperature as move ordering, the README |
| `decisions-05.md` | Progressive bias over PUCT, and three amendments from two MCTS research passes |
| `decisions-06.md` | The fifth feature recovered, the coldness result, and the ladder width |

A ruling document supersedes the specs where they conflict, and a later one supersedes an
earlier one. `decisions-03-revised.md` replaced an earlier `decisions-03.md` entirely, which is
why the earlier file is not here.
