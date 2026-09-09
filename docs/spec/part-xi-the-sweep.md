# OVERTONE PART XI — The Sweep

**A survey of games related to what Overtone is attempting, ancient to current, organised by what each finding changes.** Six research waves. Eleven findings that alter specs; one that threatens Part X directly.

---

## Corrections

**The body below is unchanged**, per the convention in [`docs/spec/README.md`](README.md).
Every finding here was re-checked at source before being acted on; two came back different.

| Section | Status | See |
|---|---|---|
| §4 | **resolved** — the document flags the primary paper's status as unverified. It is **also retracted**: *Retraction Note: Exploring the quantum speed limit with computer games*, Nature, August 2020, withdrawn by the authors for an error in their optimisation code that invalidates the quantitative results. So both the News & Views and the paper are retracted, which strengthens §4's claim-discipline lesson rather than weakening it | Nature s41586-020-2515-2; PHASES Phase 17 |
| §8 | **amended** — *"there is no Sprague–Grundy Theorem for misère play impartial games"* is too strong. The *naive* misère generalisation is hopelessly complicated and useless beyond simple cases, but Plambeck & Siegel's **misère quotient** (JCTA 2008) is the working one, described by its authors as the long-sought natural generalisation. The reduction survives and **degrades**: from a group element to a commutative monoid, from one integer to a possibly large algebraic object. That is a better analogy for `dim(g)` than a broken theorem | Plambeck & Siegel, arXiv:math/0609825; PHASES Phase 17 |
| §2, §3, §5, §9 | **verified verbatim at source** — the 1%/58% Hello Quantum split, Wouters' *"all domains except biology and engineering"* with language at `d = 0.66`, Aaronson's Minus-Sign Test wording, and *"a proper and comprehensive evaluation and assessment scheme ... is missing in the literature"* | arXiv:2202.07756; Wouters et al. 2013 |

---

**Method note.** I had no subagent capability in this session, so this was done directly across six sequential waves: quantum games; mathematical video games; algorithmic games; educational-game efficacy; combinatorial game theory and its ancient roots; ancient board games. I stopped when further searching was returning more *games* but no more *findings that change the project*. That is the right stopping criterion for a sweep, and it is worth stating plainly rather than claiming exhaustiveness.

---

## 1. Verdict

**Worthy.** Not because the landscape is crowded — it is thinner than expected — but because four findings would have cost real time to learn the hard way, and one of them is a direct challenge to Part X's premise that I could not have anticipated.

The headline results:

| # | Finding | Changes |
|---|---|---|
| 1 | **A quantum game's blog post outperformed the game itself ~100:1** | Part X §11, M54 |
| 2 | **Educational-game evidence is weak, and engineering is one of two domains where it fails** | M53 acceptance |
| 3 | **The one thing that predicts effectiveness is mechanic–content alignment** | Validates Part X's core |
| 4 | **The Zachtronics histogram** — three antagonistic metrics, not a leaderboard | New mechanic, Parts X & VIII |
| 5 | The most cited "players beat algorithms" quantum result carries a retraction | Claim discipline |
| 6 | HyperRogue proves hard math can be the mechanic and still be fun | Validates the bet |
| 7 | Quantum Flytrap independently chose Rust + sparse arrays for browser quantum sim | Validates the stack |
| 8 | Aaronson's **Minus-Sign Test** — a one-line quality standard Overtone passes | Adoptable |
| 9 | **Sprague–Grundy is exactly what Overtone lacks, and the lack is the mechanic** | Sharpens positioning |
| 10 | No evaluation standard exists for quantum games | A gap Overtone can fill |
| 11 | 4,500 years of board games; the deep ones are a tiny minority | Perspective |

---

## 2. The finding that threatens Part X

From the field's own survey (Seskir et al., *Quantum Games and Interactive Tools for Quantum Technologies Outreach and Education*, arXiv:2202.07756), written by the developers themselves:

IBM's **Hello Quantum** app shipped with companion blog posts explaining how the game's mechanics map to quantum computation. The developers expected app players to follow the in-app link to the explanation. Their traffic analysis:

> **Only around 1% of views for the blog post came from the in-app link. Around 58% came from search engines.**

Their conclusion, in their own words: *"the blog posts were a more successful learning resource than the app."*

And a second, independent case in the same paper. The **Battleships with partial NOT gates** game: *"There is therefore little evidence of it being played, but the blog post remains the most viewed on the Qiskit blog by some margin."*

**Two cases, same direction, large margin.** The explanation outperformed the game that was built to deliver it.

Their recommendation is worth quoting because it is exactly the M54 problem: *"It is therefore worth considering making fully blog based companions to any educational quantum game or interactive tool."*

### What this changes

Part X §11 said the risk had moved from the game to the debrief, and that M54 decides which project this is. **This finding says the risk is larger than I estimated**, and it suggests a specific structural fix.

**The debrief must not be inside the Gauntlet.** If it is a screen that appears after a level, it inherits the app's reach — which is to say almost none. It must be a **standalone document, indexed by search engines, that stands on its own as an explanation**, with the Gauntlet linking *into* it rather than the reverse.

Concretely:

```
was:   level → in-game debrief screen → optional link out
now:   nine standalone explainer pages, publicly indexed, complete without the game
       ↑ the Gauntlet deep-links into the relevant one after each level
```

That inverts the dependency. The explainers work for someone who never plays; the game becomes one of several routes into them. Given that 58% of Hello Quantum's explanation traffic came from search and 1% from the app, the explainers are where the audience actually arrives.

**This does not kill the Gauntlet.** It demotes it from "the on-ramp" to "one on-ramp, and probably not the largest." Which is fine, and considerably better than finding out after building nine levels.

---

## 3. The educational-games evidence, honestly

Multiple meta-analyses, and the picture is not flattering.

**Effect sizes are small.** Wouters et al.: serious games more effective than conventional instruction for learning (d = 0.29) and retention (d = 0.36), **but not more motivating** (d = 0.26, p > .05). Sailer & Homner (2020, 38 studies): positive on cognitive outcomes, *unstable* on behavioural and motivational.

**The best studies are the least conclusive.** The strictest meta-analysis, restricted to randomised controlled trials, **finds no evidence of effectiveness.** The pattern — robustness inversely correlated with conclusiveness — is the classic signature of a weak literature.

**Games beat lectures, not other active learning.** The most useful framing found: *"If you compare serious games with active teaching, they do not appear to be more effective in terms of learning, whereas on average they are more expensive."*

**And the domain result is a direct warning.** Wouters found serious games improve learning over conventional instruction *in all domains except biology and engineering*, with language strongest (d = 0.66). **Engineering is one of the two domains where the advantage disappears.** Overtone sits squarely in engineering and physics.

### But one thing does predict success, and Part X has it maximally

Clark, Tanner-Smith & Killingsworth (2016) and Wouters converge on the same moderator: **learning gains depend on the alignment between game mechanics and instructional content.** As one synthesis puts it — *"the effectiveness of serious games does not stem from the technology itself, but from how well the experience integrates learning principles with interactive design."*

Part X §3's design rule is: **a level is a wall you cannot pass until you understand the theorem, and the key is always a capability, never a skill.**

That is the strongest form of mechanic–content alignment available. The mechanic *is* the concept — you do not learn about the frequency ceiling and then apply it, you hit it and cannot proceed. If the literature identifies one design property that predicts effectiveness, the Gauntlet's central idea is that property taken to its limit.

**So the evidence is simultaneously discouraging about the category and specifically encouraging about this design.** That is an unusual and useful position, and it is what M53's test on a real person is for.

**Revised M53 acceptance:** not "did they enjoy it" but **"can they correctly state, unprompted, why they could not pass L3?"** That is a knowledge-transfer test, it is cheap, and it is the only thing the literature says matters.

---

## 4. The retraction, and claim discipline

The most-cited result in quantum citizen science is *Exploring the quantum speed limit with computer games* (Nature 532, 210, 2016) — the Quantum Moves paper reporting that human players found solutions where numerical optimisation failed.

**Nature's accompanying News & Views piece, "Quantum problems solved through games," is marked RETRACTED**, with a correction published 27 April 2016.

*(Precision matters here: I verified the retraction of the News & Views commentary. I did **not** verify the current status of the primary paper. Check both before citing either.)*

The careful follow-up tells the more useful story. *Crowdsourcing human common sense for quantum control* (Phys. Rev. Research 3, 013057, 2021), same group, Quantum Moves 2, three control problems: player-assisted results perform **"roughly on par with the best of the tested standard optimization methods."** Parity, not superiority. And the authors' own warning: results *"should not be taken as a guarantee that player-based seeding is advantageous when comparing to increasingly complex algorithmic strategies."*

**The lesson for Overtone is not about quantum games. It is about claim discipline.** The strongest version of a result got the coverage and the retraction; the honest version got parity and a caveat. Part VIII §8 and Part X will both be tempted to claim that human players discover things algorithms miss. **Do not, unless the grid says so and survives replication.** This field has already run that experiment.

---

## 5. The design pattern to steal

**Zachtronics' solution histograms.** This is the most directly useful mechanic found in the entire sweep, and Zach Barth's own postmortem explains the reasoning better than I could summarise:

> Histograms *"were developed as a replacement for global leaderboards. They solve two common problems: 1. Getting your name at the top of the leaderboards is a fantastic incentive for cheating. 2. For most players, the only thing a global leaderboard manages to tell you is that you suck (and not even by how much)."*

And on what it produces: *"most players discover that their solution is terrible, but quickly formulate a personal challenge after looking at the histogram and replay the puzzle to improve their score."*

Then the part that matters most:

> *"Because we include **three antagonistic metrics** (number of cycles, number of symbols, and number of reactors), players optimizing for one criterion often do poorly in the others."*

**A game designer arrived independently at Part VII's Axiom II** — that resources must be incommensurable — and shipped it as a scoring display in 2011.

Opus Magnum uses cost, cycles, area. TIS-100, Shenzhen I/O, Infinifactory and Exapunks all use variants. A community leaderboard bot spans all of them. Barth's closing note: *"they're not much more difficult than a leaderboard to implement, there's no reason not to include them."*

### Overtone's three antagonistic metrics are already physical

```
dim(g)            expressiveness  ↑  →  gradients collapse   (Part III)
effective χ       classicality    ↓  →  requires large dim(g)
work units        cost            ↓  →  limits search depth
```

Optimising any one degrades another **by theorem**, not by design choice. Nobody has shipped a histogram whose axes are antagonistic for a proven reason.

Two further design points from Zachtronics worth adopting:

- *"We deliberately don't offer any kind of in-game rewards for optimization, so the players who do are often more intrinsically motivated."* No unlocks, no achievements.
- *"As long as the toolset is expressive and the problem is open-ended, we will see creative solutions from players, including optimizations we hadn't really dreamed of before."*

**And the histogram is simultaneously a scientific instrument.** The distribution of player solutions across `(dim g, χ, work)` is exactly the data Part VII §8 wants for the depth measurement. The display and the measurement are the same object.

**Add to Part X as M54b. It is cheap, validated over a decade, and it does two jobs.**

---

## 6. The proof-of-concept

**HyperRogue** — Zeno Rogue, 2011 hobby project, Steam 2015, 97% positive across 403 reviews, still actively developed. A roguelike played on the hyperbolic plane.

The developer's own claim about why it works:

> *"hyperbolic geometry combined with basic roguelike rules makes for exceptionally great gameplay, **even if you do not care about geometry**."*

And on it as a teaching tool:

> *"playing HyperRogue is probably the best way to learn about this, much better and deeper than any mathematical formulas."*

That is the Gauntlet's bet, stated by someone who won it. Three further points:

- **The tutorial doubles as an explorable explanation.** They bridged game and explainer explicitly — the same move §2 recommends.
- **The engine became a research instrument.** RogueViz, built for the game, is now used for research and visualisation in non-Euclidean geometry. Game → tool. Overtone is going tool → game; the dual-use works in both directions.
- **The same author's *Hydra Slayer*** is a resource-management combat game whose system is built on old mathematical puzzles. A second existence proof that combat can be a number-theory problem.

Scale calibration, honestly: 403 Steam reviews is not a large audience. Quantum Flytrap's Virtual Lab reports *"around 70 users on a regular working day, peaks at over 700 during events."* **This category does not produce mass audiences.** It produces small, durable, high-quality ones — which is the correct expectation to hold for Overtone.

---

## 7. The quantum games landscape, complete as of the field's own survey

Everything currently in the field, with what each occupies:

| Tool | Type | Occupies |
|---|---|---|
| **Quantum Odyssey** (Quarks Interactive, 2020) | Paid puzzle game, Windows | Circuit construction. **The Gauntlet's closest competitor.** |
| **Virtual Lab** (Quantum Flytrap, 2019) | Web simulator/puzzle | Photonic optical table, up to 3 entangled photons |
| **Quantum Game with Photons** (2016) | Web puzzle, MIT licence | 34 levels + sandbox. **100k+ gameplays, cited in 7 papers.** |
| **Hello Quantum / Hello Qiskit** (IBM, 2018) | Puzzle, mobile + textbook | Gate effects on states |
| **Quantum Moves 2** (ScienceAtHome, 2018) | Citizen science | Optimal control of ultracold atoms |
| **Alice Challenge** (2017) | Citizen science | Remote control of a real BEC experiment |
| **Particle in a Box / Psi and Delta** (Georgia Tech) | Platformers | Infinite square well, energy levels |
| **VQOL** (UT Austin) | Web simulator | Quantum optics, classical-framework simulation |
| **Quantum Composer** (Aarhus) | Node-based tool | 1D quantum simulation |
| **QPlayLearn** | Platform | Multi-level dictionary + games |

### What this tells you

**Quantum Odyssey is validated and is the competitor to know.** Peer-reviewed evaluation (Nita et al., arXiv:2106.07077) found untrained participants could *"solve problems that are traditionally given today in master's level courses in a mathematical form,"* developing visual rather than mathematical intuition. Also: *"the design of the visuals makes it equally appealing to both female and male participants."* Read that paper before writing M53.

**But nothing in the landscape does what Overtone does.** Every entry above teaches *quantum mechanics* — gates, states, measurement, interference. **None of them touches trainability, barren plateaus, dynamical Lie algebras, or the trainability–simulability tension.** The Gauntlet's curriculum — nine levels of theorems about *variational quantum machine learning* — is unoccupied.

**And the calibration case is Quantum Game with Photons**: 100k+ gameplays, seven citations by unrelated researchers, open source, and *"it was possible to solve a large fraction of levels without prior exposure to quantum physics... At the same time, a few levels provided a challenge to PhD students and established quantum optics professors."*

**That is a strategy ladder in an educational game**, described qualitatively. It is the shape M53 should aim for and the closest thing to a target the field offers.

---

## 8. Where Overtone sits in the mathematical lineage

The sweep from ancient games forward produces a clean taxonomy, and Overtone's position in it is unusually specific.

### Ancient: depth is rare

**Royal Game of Ur** (c. 2600–2400 BC) and **Senet** (c. 3100 BC) are the two oldest playable games, and both are **dice race games**. As one account puts it, they established *"that marriage of luck and position"* which *"structures an enormous share of all gaming since."* Ur's rules survive because Irving Finkel decoded a cuneiform tablet written around 177 BC by a scribe complaining about players who do it wrong.

**They are not deep.** Neither is Liubo, whose rules are lost entirely. The genuinely deep ancient games are a short list:

- **Go** (~2500 years) — ~10¹⁷⁰ positions, branching 250, one piece type
- **Nim** (ancient) — the foundation of impartial combinatorial game theory
- **Chinese Rings** (~2000 years) — state graph is a path of length 2ⁿ−1, solution is the Gray code *(Part IX §3)*
- **Mancala family** — perfect information, no chance; Kalah solved like checkers, Oware and Bao resist

**Finding: 4,500 years of board games, and fewer than half a dozen have real depth.** Depth is not a default property that emerges from rules. It is rare and hard-won, which is why Part VII §8's insistence on *measuring* `d` rather than assuming it is the correct posture.

### Classical: the two branches of CGT

**Impartial games — Sprague–Grundy.** Every finite impartial game position (both players have the same moves) is equivalent to a single Nim heap. Its size is the Grundy value. And critically, positions combine by XOR:

```
nimber(G + H) = nimber(G) ⊕ nimber(H)
```

**This is a classification theorem structurally identical in shape to what Overtone claims with `dim(g)`**: an apparently complex object reduces to a single integer.

But the comparison inverts at the crucial point.

> **Sprague–Grundy's reduction is additive. Overtone's is not.**
>
> `nimber(G + H) = nimber(G) ⊕ nimber(H)` — combining is free.
> `dim(closure(g₁ ∪ g₂)) ≠ dim(g₁) + dim(g₂)` — commutators generate new elements, and the closure can explode.

**Sprague–Grundy is exactly the theorem Overtone does not get, and its absence is the mechanic.** In an impartial game, merging two positions is bookkeeping. In Overtone, merging two agents is a decision with a cost, because the algebra is non-additive under union. That is Part VI §3's absorption tradeoff, expressed as a precise statement about where Overtone sits relative to the deepest result in game mathematics.

Worth stating in the docs. It is the sharpest available characterisation of what makes the absorption mechanic non-trivial.

*(Structural rhyme, not to be overclaimed: Nim-sum is XOR of heap sizes; Pauli commutation is XOR of bitsets. Both are `F₂` arithmetic. Suggestive; not an identity.)*

**Partisan games — Conway, Berlekamp, temperature.** Players have different moves; no reduction to one number; temperature and thermography instead. **This is where Part IX already lives.** Overtone is partisan (different algebras), stochastic (Born rule), and non-additive under combination — the hardest corner of the taxonomy, and the reason Sprague–Grundy offers no shortcut.

Note also: *"there is no Sprague–Grundy Theorem for misère play impartial games."* Even changing the win condition to last-player-loses breaks the theory. Reductions of this kind are fragile, which is worth remembering before claiming `dim(g)` does more than it does.

---

## 9. Two independent validations, and one standard to adopt

**Rust and sparse arrays.** Quantum Flytrap's Virtual Lab needed to simulate three entangled photons on a grid: *"for a typical grid size a single photon requires around 1000-dimensional vector space, so a three-photon simulation requires a billion dimensions. Therefore, Quantum Flytrap needed to develop a custom high-performance sparse array simulation in the low-level programming language Rust — as none of the off-the-shelf solutions were suitable."*

Independent arrival at the same architecture Part I chose, for the same reason, by a company doing browser-based quantum simulation commercially.

**The Minus-Sign Test.** Scott Aaronson's standard, quoted in the field survey:

> *"To pass the Minus-Sign Test, all a popularization needs to do is mention the minus signs: i.e., interference between positive and negative amplitudes, the defining feature of quantum mechanics, the thing that makes it different from classical probability theory."*

The survey notes this is where most quantum popularisation fails — "in two states at once" language fails to distinguish quantum superposition from classical uncertainty.

**Overtone passes trivially.** Phase-as-hue is on every panel; L2 of the Gauntlet is destructive interference; dark corridors are the minus sign made into a mechanic. **Adopt the test explicitly** — cite it in the README and use it as a review criterion for every explainer page. It is a one-line quality bar from a credible source, and passing it is worth saying out loud in a field where most things don't.

---

## 10. The gap Overtone could fill

From the survey, on evaluating quantum games:

> *"a proper and comprehensive evaluation and assessment scheme that can be applied to a variety of quantum games is missing in the literature."*

They propose something like PhysPort for quantum games — a repository with ratings based on whether a tool has been evaluated — and note that no common design methodology exists either.

**Overtone has an unusual asset here: a measurement culture.** The project already has a Skill Trace protocol, a pre-registered step criterion, a repeatability calibration, and a work-unit budget. Applying that apparatus to a *quantum game* — measuring its depth on a published metric — would be the first quantitative evaluation of a quantum game's strategic depth.

That is a small, real, publishable contribution sitting adjacent to work already planned. Lower priority than shipping, but worth knowing it is there.

---

## 11. Changes to specs

| Spec | Change |
|---|---|
| **Part X §6, M54** | Debrief becomes **standalone, search-indexed explainer pages**, complete without the game. The Gauntlet links *into* them. §2. |
| **Part X, new M54b** | **Solution histogram** on `(dim g, χ, work units)` — three antagonistic metrics, no leaderboard, no rewards. §5. |
| **Part X §9, M53** | Acceptance becomes a **knowledge-transfer test**: can a naive player state unprompted why L3 was impassable? §3. |
| **Part X §0** | Soften "the on-ramp" to "one on-ramp." The explainers likely reach more people. §2. |
| **Part VIII** | Histograms over leaderboards, with Barth's two reasons stated. §5. |
| **Part VI §3** | Add the Sprague–Grundy contrast: the reduction is non-additive, and that is the mechanic. §8. |
| **README** | Adopt the **Minus-Sign Test** as an explicit standard. §9. |
| **All claims about players** | No "players find what algorithms miss" unless the grid shows it and it replicates. §4. |

---

## 12. What I did not find, and where I stopped

**Not found, and I looked:**

- No game teaches trainability, barren plateaus, or dynamical Lie algebras. The Gauntlet's curriculum is unoccupied.
- No quantitative depth measurement of any quantum game.
- No published evaluation standard for quantum games.
- No educational game in engineering or physics with a demonstrated large effect size in a well-controlled study.

**Deliberately not pursued, with reasons:**

- **Foldit, EteRNA, EyeWire** — the citizen-science lineage Quantum Moves cites. Real results, but the model is "crowdsource a search problem," which Overtone is not doing.
- **Baba Is You, Zendo, deduction games** — "rules as manipulable objects" and "deduce the hidden constraint" are both structurally relevant, but I found nothing that would change a spec.
- **Amazons, Domineering, Konane, Hackenbush** — the partisan CGT test-beds. Part IX already draws on this branch; Prompt B (outstanding) covers whether temperature extends beyond it.
- **Kerbal Space Program, A Slower Speed of Light, Portal** — physics games with real physics. Different problem: they teach intuition for classical mechanics, which is already intuitive.

**Where I stopped and why.** Wave 6 returned more games but no new findings that changed a spec. That is saturation for a sweep of this kind — not "I have seen every game," but "the marginal game is no longer changing my recommendations." Claiming exhaustiveness over 4,500 years of games would be false; claiming the search converged is true and is the useful statement.

---

## 13. Sources

**Quantum games**
- Seskir, Migdał, Weidner, Anupam, Case, Davis, Decaroli, Ercan, Foti, Gora, Jankiewicz, La Cour, Malo, Maniscalco, Naeemi, Nita, Parvin, Scafirimuto, Sherson, Surer, Wootton, Yeh, Zabello & Chiofalo — *Quantum Games and Interactive Tools for Quantum Technologies Outreach and Education*, arXiv:2202.07756. **The single most useful document in this sweep.**
- Sørensen et al. — *Exploring the quantum speed limit with computer games*, Nature 532, 210 (2016). **Verify status before citing.**
- *Quantum problems solved through games*, Nature 532, 184 (2016) — **RETRACTED**.
- Jensen et al. — *Crowdsourcing human common sense for quantum control*, Phys. Rev. Research 3, 013057 (2021).
- Nita et al. — *Inclusive learning for quantum computing: Quantum Odyssey*, arXiv:2106.07077.
- Cantwell — *Quantum Chess*, arXiv:1906.05836.

**Mathematical and algorithmic games**
- Zeno Rogue — HyperRogue, Hydra Slayer, RogueViz. roguetemple.com/z/hyper.
- Barth — *Postmortem: Zachtronics Industries' SpaceChem*, Game Developer. **The histogram rationale.**
- *Road to the IGF: Zachtronics' Opus Magnum*, Game Developer.
- Migdał — science-based games list, github.com/stared/science-based-games-list.

**Educational efficacy**
- Wouters, van Nimwegen, van Oostendorp & van der Spek — *A meta-analysis of the cognitive and motivational effects of serious games*, J. Educ. Psychol. (2013).
- Clark, Tanner-Smith & Killingsworth (2016) — the alignment moderator.
- Sailer & Homner (2020) — 38-study gamification meta-analysis.
- Sitzmann (2011) — 65-study media comparison.

**Combinatorial game theory**
- Sprague (1935), Grundy (1939) — the theorem.
- Bouton (1901) — Nim solved; XOR characterisation.
- Burke, Ferland & Teng — *Sprague-Grundy-completeness*, FUN 2022.
- Berlekamp & Wolfe — *Mathematical Go* (1994). *(Part IX)*
- Conway, Berlekamp & Guy — *Winning Ways* (1982).

**Ancient games**
- Finkel — the Itti-Marduk-balāṭu tablet, c. 177 BC; British Museum.
- Royal Game of Ur, c. 2600–2400 BC; Senet, c. 3100 BC.
- Mancala family: Oware, Bao, Kalah, Toguz, Tsoro.
