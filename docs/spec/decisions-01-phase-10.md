# OVERTONE — DECISIONS 01 (Phase 10)

**Rulings on Claude Code's eight questions, with consequences for existing specs.** Where a ruling changes something already written, the affected section is named so it can be updated.

Q1 is a real catch and it reorders the phase. Read it first.

---

## Q1 — Unify the languages. And yes, that makes the submission format a blocker.

**Ruling: unify. Freeze the agent language before measuring `d`. The three "unlocks" are not alternatives — two of them become *fields* in the third.**

### Why the catch is correct, and worse than stated

`d` is not a property of the game. It is a property of **(game, strategy language)**. Lantz et al. measure resistance to partial solutions *by a specific family of agents*; change the family and you change the number. The agent language is part of the experimental apparatus, not a downstream consumer of the result.

So a `d` measured on ladder-language agents does not describe league-language agents, exactly as you say. But there is a sharper problem underneath:

> **A pure declarative feature-weight policy has no compute axis at all.**

It evaluates in `O(1)` per move. You cannot give it more compute. If the league's submission format is a policy spec, the league has no strategy ladder in Lantz's sense, and `d` is not merely mismatched — it is undefined over the league population.

### The resolution: declare the *search*, not the *policy*

A declarative spec does not have to describe a function. It can describe **a search agent whose search the engine implements**:

```toml
[agent]
name        = "kestrel"
generators  = ["pawn", "bishop-Z2", "knight-hop", "rook-phase"]
statistics  = "fermionic"

[agent.search]
kind        = "mcts"        # fixed vocabulary: "greedy" | "negamax" | "mcts"
budget      = 4096          # ← THE COMPUTE AXIS (work units, see Q2)
rollout     = "tablebase"   # M36 terminator

[agent.eval]
features    = ["temperature", "orbit_size", "coherence", "reach_margin"]
weights     = [0.31, -0.12, 0.44, 0.08]
```

This satisfies every constraint simultaneously:

- **Still declarative.** The submitter supplies parameters from a published vocabulary; the *engine* implements `mcts` and `negamax`. No submitted code is ever executed. Part VIII §2 holds unchanged.
- **Has a genuine compute axis.** `budget` is the rung index. The same field spans the ladder and the league.
- **One language.** `d` measured on the ladder is `d` over the league population, by construction.
- **Deterministic and verifiable.** Same spec, same seed, same result — Part VIII §6 holds.

The feature-weight language is not a third option; it is the `[agent.eval]` block that `negamax` and `mcts` both consume. Depth-`k` search is not a third option; it is `kind = "negamax"` with `budget` setting the depth.

### Which search is primary: MCTS, and for a physical reason

Not a convenience preference — a correctness one.

**Overtone is a stochastic game.** Measurement outcomes are Born-random (Part IX §1.2). Negamax assumes determinism; the correct deterministic-search generalisation is expectimax, whose chance nodes multiply the branching factor by the number of measurement outcomes. At `n = 10` that is a 1024-way chance node. Expectimax is not viable here.

MCTS samples chance nodes natively. It is the *right* algorithm for this game, not the convenient one.

And Overtone has an unusually strong MCTS setup that most games do not: **M36's tablebase gives exact terminal values.** Rollouts terminate in ground truth rather than in a heuristic playout. That is close to having a perfect rollout policy, and it means MCTS should climb further here than it typically does elsewhere — which is precisely what a long ladder needs.

Keep `negamax` in the vocabulary anyway. It is correct and cheap in the **fully decohered endgame**, where the game becomes deterministic (Part VII §4) and the tablebase makes depth nearly free. Two search kinds, each right in its own phase, is a genuinely nice fit to the game's phase structure rather than an arbitrary menu.

### Consequence: the phase reorders

Yes, the submission format blocks Phase 10, and that is correct sequencing rather than a reversal.

```
was:   M38 board → M39 ladder → ... → M41 notation → M43 league
is:    M41a agent-language freeze (v1) → M39 ladder → M38 board → M43 league
```

Part VIII §11 already says *freeze the notation early*. This extends the same rule to the agent language, for the same reason: **you cannot measure a property of a language you have not fixed.**

**Risk, and its mitigation.** Freezing early means freezing before you know what the language needs. So version it — `language = "v1"` in every spec — and permit a later `v2` with a re-measured ladder. The rule that matters: **never mix language versions inside a single `d` measurement**, and always report `d` as `d(v1)`.

---

## Q2 — Neither. Declared work units under a published cost model.

**Ruling: the compute axis is work units under a published cost model. Calibrate to wall-clock once on a declared reference machine and publish the conversion.**

Both offered options fail, for opposite reasons:

**Wall-clock** is what Lantz specifies, and it is irreproducible in your environment. GitHub Actions runners vary by CPU generation, noisy neighbours, and thermal throttling. A wall-clock `d` cannot be re-derived by a third party, which breaks Part VIII §6's verification story — the property the whole league rests on.

**Evaluations** are reproducible but not algorithm-fair, and the unfairness is large here rather than marginal. A position evaluation at `n = 6` and at `n = 12` differ by a factor of 64 in real work. An MCTS node and a negamax node do different amounts of it. Counting them as equal would let an agent buy ladder rungs by making each evaluation more expensive.

### The cost model

Assign a fixed cost to each engine primitive, published and versioned:

```
gate application         ∝ 2ⁿ      (statevector)  or  ∝ dim(g)  (g-sim)
expectation value        ∝ 2ⁿ
Lie closure step         ∝ dim(g)
tablebase lookup         = 1
temperature, one region  ∝ region size
memo table hit           = 1
```

Budget in work units. This is deterministic across machines, fair across algorithms, and **auditable** — the model is a published file that someone can criticise, which is better than a number nobody can inspect.

### Getting Lantz-compatibility back

Run the reference agent set once on a **declared reference machine** (name the CPU, the clock, the build flags), fit work-units-to-seconds, and publish the conversion factor. Then you can state both: `d` measured in work units, and its wall-clock equivalent under the published calibration.

Say plainly in the writeup that this is **a reproducible variant of Lantz's `d`**, give the conversion, and state why. Reviewers accept a declared, justified variant. They do not accept a number they cannot reproduce.

---

## Q3 — Pre-register three numbers, and derive `STEP_UNIT` from the noise floor.

**Ruling: "still rising" means the lower bound of the 95% confidence interval on the Elo gain exceeds `STEP_UNIT`, for each of the last two rungs. All three numbers are pre-registered before the measurement.**

A bare threshold is both post-hoc-able and noise-vulnerable. The correct form is a hypothesis test with a minimum effect size:

- **Significance** alone fails, because with enough games any real difference becomes significant.
- **Effect size** alone fails, because 0.65 could be noise.
- **Both together** is the standard construction and it is what you want.

### Deriving `STEP_UNIT` instead of choosing it

`0.65` as written is an invented number, and the project's constitution says derive rather than choose. There is a clean way:

> Play an agent against a **bit-identical copy of itself**, `N` games. The true Elo difference is exactly zero. The spread of the measured difference is your noise floor `σ`.
>
> **Set `STEP_UNIT = 2σ`.**

Now it is measured — a property of your match-count and variance structure, not a preference. It also self-adjusts: if you raise `N`, `σ` falls and the bar tightens automatically, which is the correct behaviour.

**Also flag the units.** If `0.65` is Elo, it is far below any plausible noise floor at reasonable `N` and the test would pass on noise alone. If it is a normalised win-rate unit, state that. Whatever it is, the self-play calibration above will tell you whether it was in the right range.

### The three pre-registered numbers

```
N            games per rung pairing        (fixed before measurement)
confidence   0.95
STEP_UNIT    2σ from the self-play calibration
```

Commit them to the repo before running. "Last two rungs" is right — a single rung's gain can be a matchup artefact rather than a genuine ladder step.

---

## Q4 — Your reading is correct. Here is the boundary that closes the loophole.

**Ruling: presentation timing is not covered. The operational test is whether the number changes an outcome.**

> **An invented number is one that changes the outcome.**
>
> Test: *if I changed this number, would any match result change?*
> Yes → mechanic → forbidden.
> No → presentation → allowed.

A 400 ms hold: the collapse has already happened; the hold is when it is *shown*; nothing downstream depends on it. Allowed.

A trap radius: results change. Forbidden.

Your worry — that "mechanics not presentation" is exactly the reasoning that lets invented numbers back in — is right, and the test above is what makes it a boundary rather than a vibe. It is checkable, and it should be checked mechanically:

**Two consequences, both cheap:**

1. **One file.** Every presentation constant lives in `presentation.toml`. The complete list of authored numbers in the project is then visible at a glance, and the claim "no invented numbers" becomes something a reader can audit in thirty seconds instead of taking on trust.

2. **A CI test that proves it.** Perturb every constant in `presentation.toml` and assert that all match results are unchanged. That mechanically enforces the boundary rather than relying on discipline. If someone later sneaks a mechanic into the presentation file, the test fails.

Add this test to the Part VII §9 claims table. It is the sort of thing that makes a reader trust the rest of the table.

---

## Q5 — The measurement basis. Not the state vector, not a summary of it.

**Ruling: show the full Born distribution over the outcomes of the measurement actually being performed.**

The intent of Part IX §2 was *do not show a prettified summary that hides the shape of the risk*. It was not *show everything*.

And the position marginal is **not** a cleaned-up view, if the measurement being performed is a position measurement. It is the literal, complete answer to "what are the odds," because the unmeasured degrees of freedom genuinely do not affect this outcome. Tracing them out is not hiding — it is the correct calculation.

So the rule is basis-matched rather than completeness-based:

- **Show:** the Born distribution over the outcomes of *this* measurement. Complete by construction.
- **Never:** a distribution over a different basis than the one being measured. That would be a misrepresentation, and it is the actual failure mode §2 was written against.

### The part that has teeth

**No bucketing. No default log scale. No dropping small amplitudes.** If 400 outcomes each carry 0.25%, draw 400 bars.

Ugly is the honest rendering of a diffuse state, and **the ugliness is information** — a flat, wide distribution is the interface telling you your plan was bad. Smoothing it into something presentable would delete exactly the signal the panel exists to deliver.

### Optional second panel

The full state vector as a dense phase-coloured strip — 1024 thin columns, hue by phase, lightness by magnitude. Unreadable as numbers, but as a *texture* it truthfully conveys "there is structure here you are not being shown." Available on demand, never the default.

---

## Q6 — The premise is wrong. 60 fps was never the requirement for this layer.

**Ruling: decouple. The amplitude field renders at 60 fps. The temperature field recomputes on position change, with progressive refinement if it overruns a frame.**

Temperature is a property of the position. In a turn-based game (Part VII), the position changes once per ply. Recomputing the temperature field at 60 fps computes the same answer sixty times and then discards fifty-nine of them.

The real requirement is: **recompute within one frame of a move, or fill in visibly.**

### Why this does not collide with the interpolation trap

The standing trap — *do not interpolate animation frames* — is about **the physics**: do not fabricate intermediate states of a discrete-time process, because that is a lie about how a quantum walk moves.

A progressively-refining analysis overlay fabricates nothing. It shows the current state of a computation, coarse-to-fine, region by region. Watching an analysis sharpen is honest; watching a discrete walker glide is not. Different categories, and the distinction is worth writing into Part IX §8 so the trap is not misapplied again.

So: **stepped 15 fps is fine. Progressive fill is better.** Neither is a compromise.

### And take the cache

Most regions do not change between plies. Q8's memo table gives you those for free — key each region's temperature by its local descriptor and only recompute the regions the move touched. That alone will usually get you inside the frame.

---

## Q7 — Fixed scale, derived from the Atlas, with visible clipping.

**Ruling: fixed range, set from measured data, with an explicit over-range treatment and the numeric maximum always on screen.**

Auto-scaling makes every screenshot a lie about magnitude, and two images that cannot be compared are two images that cannot be used as evidence. This project's credibility rests on comparability; that decides it.

Three parts:

**Derive the range, don't choose it.** Set it from the Atlas's measured temperature distribution — zero to the 99th percentile across all recorded positions. The scale then becomes a measured quantity, consistent with the constitution, and it is re-derived when the Atlas grows. Berlekamp put the empty 19×19 Go board at temperature ≈ 13, which suggests practical temperatures sit in a modest range; expect the 99th percentile to cover nearly everything.

**Make clipping visible.** Over-range cells get a distinct treatment — a hatch, or an outline — so "this is off the top of the scale" reads immediately. Clipping is honest when it announces itself and dishonest when it does not.

**Always print the numeric maximum.** Then even a clipped screenshot carries the magnitude, and the comparability survives the clipping.

---

## Q8 — Discrete state only. And this is correct, not a compromise.

**Ruling: key over the discrete state. Amplitudes are the value, never the key.**

Amplitude keying fails twice over. Two different move orders essentially never produce bit-identical amplitude vectors — floating-point accumulation alone guarantees it — so the table would almost never hit. You would pay an invented tolerance for a cache that does not cache.

The key:

```
(DLA fingerprint, safe-set spec, coherence steps remaining,
 substrate window origin, side to move)
```

Coherence is discrete here: Part VII §5 spends it in whole `k`-step blocks, so it is an integer count, not a real number. No bucketing needed.

### Why this is not a compromise

Because **the three expensive computations are exactly the amplitude-independent ones**:

1. **Lie closure** — depends only on the generator set.
2. **The checkmate predicate** — Part VII §3 defines it over the reachable orbit, which depends on `(g, safe set)` and **not on amplitudes at all.** The most important and most expensive computation in the game keys cleanly. This is a load-bearing accident of the design and worth noting in the docs.
3. **Region temperature** — a function of local structure; keyed by the region descriptor.

Amplitudes, meanwhile, are *cheap* — a handful of gate applications. They are the one thing that does not need memoising.

### Name it correctly

Call it a **memo table for the algebraic layer**, not a transposition table. It does not memoise search results across move orders, and calling it a transposition table would set an expectation it never meets.

### The one exception, and it is where you want it

In the **fully decohered endgame** the state is a classical probability distribution over discrete cells with rational transition probabilities. Exact equality is achievable without any tolerance. So the M36 tablebase can use a genuine position key — and that is precisely the phase where transposition matters most.

---

## Appendix A — GitHub tournament infrastructure

Two precedents worth studying, and both validate Part VIII Tier 1.

### A.1 ICRA 2023 Simulated Humanoid Robot Wrestling (Cyberbotics)

An international robot-programming competition run entirely on GitHub. Registration is *creating a repository from a template* — the organisers collect nothing, not even participant details. Games run in Actions on every push to a participant's main branch, triggered by **`repository_dispatch`** events to the CI machine. Results, animations and the leaderboard are published to a **`competition` branch** of the competition repo. Fully open source.

They used a self-hosted runner for 3D acceleration. **Overtone does not need one** — the engine is a CPU workload that fits a standard runner.

This is Part VIII Tier 1 as specified, proven at conference scale.

### A.2 Kata — pull-request-based competition engine

*"An objective, pull-request-based competition engine for autonomous AI agents."* King-of-the-hill in scheduled rounds: a PR adds one agent, which is screened and marked pending; each round every pending agent is scored against the reigning king in an isolated sandbox on **secretly-sampled** benchmark problems; the best agent that objectively beats the king is merged and becomes the new king.

Their framing is the right one to borrow: **agent quality becomes a merge decision, not a review opinion.**

Two things to steal outright:

**King-of-the-hill instead of round-robin.** `O(n)` per round rather than `O(n²)`. At 200 agents that is 200 matches instead of 19,900 — which resolves Part VIII §7's scaling concern entirely. And it composes with the ladder for free: **the king's Elo history over rounds *is* the strategy ladder over time**, which is a better artefact than a snapshot.

**Secretly-sampled problems.** This is the gap in Part VIII. If maze seeds are public, agents overfit to them and the ladder measures memorisation. Fix: **publish a development seed set, hold out a scoring seed set, rotate the held-out set each round.** Add this to Part VIII §9 as a named risk with this mitigation.

**And note what you get to skip.** Kata needs an isolated sandbox because their agents are code. Part VIII §2's declarative decision means **Overtone needs no sandbox at all** — the hardest and most fragile part of their infrastructure simply does not exist for you. That is the declarative choice paying off concretely rather than theoretically.

### A.3 The wrappers

| Action | Role |
|---|---|
| `peter-evans/create-pull-request` | Commits workspace changes to a branch and opens a PR. How tournament results re-enter the repo. Exposes `pull-request-number` and URL as step outputs. |
| `peter-evans/repository-dispatch` | Creates `repository_dispatch` events; dispatch to multiple repos via a matrix. |
| `actions/cache` | Persists the Q8 memo table across runs, keyed by engine version. Large win on repeated tournament rounds. |
| `actions/checkout`, matrix strategy | Parallel match execution. 256 matrix jobs per run; 20 concurrent on Free. |
| `huggingface_hub` | Pushes the match archive and Atlas to a HF Dataset. |

Two operational details that will otherwise cost an afternoon:

- **`repository_dispatch` only triggers workflows committed to the default branch.** A dispatch-receiving workflow on a feature branch silently does nothing.
- **Token scopes:** the default `GITHUB_TOKEN` can dispatch to *its own* repo if you elevate `permissions: actions: write`. Dispatching to a remote repo needs a PAT — `public_repo` scope suffices for public repos. `client-payload` allows at most 10 top-level properties.

### A.4 Recommended shape

```
submission   PR adds agents/<name>.toml         (declarative, language v1)
validate     schema check → run against Overtone-100 → report absolute score
schedule     nightly: king-of-the-hill round on held-out seeds
results      create-pull-request → LEADERBOARD.md + ladder.json → competition branch
archive      match files → HF Dataset
cache        actions/cache holds the algebraic memo table, keyed by engine version
```

---

## Appendix B — Spec changes this document makes

| Spec | Change |
|---|---|
| Part VII §0 | Add the outcome-change test as the operational definition of an invented number. |
| Part VII §9 | Add the `presentation.toml` perturbation test to the claims table. |
| Part VIII §2 | Extend the submission format to declare search kind and budget, not only policy. Freeze as language v1. |
| Part VIII §9 | Add seed overfitting as a named risk; mitigation is held-out rotating seeds. |
| Part VIII §7 | Replace round-robin with king-of-the-hill; note the `O(n)` result. |
| Part VIII §10 | M41 splits: M41a agent-language freeze moves before M39. |
| Part IX §2 | Clarify "actual amplitudes" as the Born distribution in the measurement basis. |
| Part IX §8 | Distinguish physics interpolation (forbidden) from progressive analysis refinement (fine). |
| Phase 9 writeup | Declare `d` as a work-unit variant; publish the cost model and the wall-clock calibration. |
