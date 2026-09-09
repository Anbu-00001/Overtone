# The explainers

Nine standalone pages, one per Gauntlet level, each **complete without the game**.

## Why they are here and not inside the Gauntlet

IBM's Hello Quantum shipped with companion blog posts explaining how its mechanics map to
quantum computation, and expected players to follow the in-app link. Their own traffic analysis,
published in the field's survey (arXiv:2202.07756):

> Only around 1% of views for the blog post came from the in-app link. Around 58% came from
> search engines.

Their conclusion, in their own words: *"the blog posts were a more successful learning resource
than the app."* A second case in the same paper points the same way — Battleships with partial
NOT gates has *"little evidence of it being played, but the blog post remains the most viewed on
the Qiskit blog by some margin."*

So a debrief screen inside a game inherits the game's reach, which is almost none. These pages
invert that: they stand alone, they are the thing search engines can find, and the Gauntlet
deep-links **into** them rather than the other way round.

## The rules a page here follows

**It is complete without the game.** A reader who never plays should finish it understanding the
theorem. If a sentence only makes sense to somebody who just hit a wall, rewrite it.

**Every claim points at a command.** Not a citation, a command — one the reader can run. The
project's own rule is that every claim is paired with a test that fails when the claim stops
being true; a page that cannot show its working does not belong here.

**It passes the Minus-Sign Test.** Scott Aaronson's standard: a popularisation passes if it
mentions the minus signs — *"interference between positive and negative amplitudes, the defining
feature of quantum mechanics, the thing that makes it different from classical probability
theory."* Most quantum popularisation fails it, because "in two states at once" does not
distinguish superposition from classical uncertainty. Review every page against it explicitly.

**It says what is not known.** Where the theorem has hypotheses, the page states them and says
what happens when they fail.

## Status

| Page | Level | Status |
|---|---|---|
| [the-ceiling.md](the-ceiling.md) | L3 | **prototype** — the format test |
| spread | L1 | not written |
| the dark corridor | L2 | not written |
| resonance | L4 | not written |
| the cage | L5 | not written |
| the feast | L6 | not written |
| the front | L7 | not written |
| checkmate | L8 | not written |
| the mirror | L9 | not written |

One page exists deliberately. Part X §9's own logic — test the format on a real person before
building the other eight — applies to the explainers at least as much as to the levels, and the
acceptance test in `docs/PHASES.md` Phase 18 is a two-arm comparison that needs exactly this one
page to run.
