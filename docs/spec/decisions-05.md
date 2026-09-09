# OVERTONE — DECISIONS 05

**Amendments from the MCTS research.** Two independent passes on Prompt A, broadly agreeing. Four items change frozen content, so this needs to land before M41a.

The item I was most worried about — whether exact terminals compress the skill ladder — came back **NOT FOUND from both passes**. That is worse than a bad answer and better than it looks. §5.

---

## 1. Progressive bias, not a PUCT prior

**Decisions 04 Q13 specified PUCT. Change it.**

Both passes converge on the same distinction, and one states it directly: additive progressive-bias decay is more robust to a poor heuristic, because it gradually hands control to the empirical mean, while multiplicative PUCT priors are the right choice **when the heuristic is already probabilistic**.

Temperature is not probabilistic. It is an unbounded positive scalar. Forcing it into PUCT means softmaxing it, which:

- imposes a distribution shape on something that is not one
- introduces a scaling parameter with no published guidance
- is less forgiving if the heuristic turns out mediocre — and Prompt B has not come back yet, so we do not yet know whether what you are computing is CGT temperature proper or a temperature-shaped heuristic

Progressive bias needs none of that. Add to the UCT value:

```
W · H(s,a) / (1 + n(s,a))
```

No normalisation, explicit decay, and the influence vanishes as visits accumulate.

**Bonus: the name collision resolves itself.** PUCT would have required a softmax temperature parameter `τ` sitting next to CGT temperature `H` in the same function — two unrelated quantities called "temperature" three lines apart. Progressive bias has no softmax, so the collision never arises. Worth noticing, because that class of naming confusion produces bugs that survive review.

**Field rename.** `temperature_prior` is no longer accurate — it is not a prior. Use:

```toml
[agent.search]
kind             = "mcts"
budget           = 4096
temperature_bias = 0.15    # W in the progressive-bias term; 0 disables
```

---

## 2. The default should be 0.1–0.2, not 0.7

My Decisions 04 example used `0.7`. That is far too aggressive given what both passes report.

The one published tuned value that surfaced: Gedda et al. (2018) tuned a progressive-bias weight in Kingdomino and found the optimum around **W ≈ 0.1–0.2**. Both passes independently name low initial weight as *the* mitigation for an uncertain heuristic, and James et al. (2017) show that an incorrect bias focuses search on suboptimal moves and takes many iterations to correct.

**Default `0.15`, and let the league find better.** Since it is an agent field, submitters will explore the range for you, and the value that wins is a measurement rather than a guess — which is the right way for this project to set it.

*(One encouraging data point in the other direction: Gelly & Silver report that in MoGo, handcrafted move evaluation plus RAVE lifted the win rate against GNU Go from 24% to 69%. Good priors are worth a great deal. The uncertainty is about whether yours is good, not about whether the mechanism works.)*

---

## 3. Ordering is not dead after all — and it lands in the same place

I ruled in Decisions 04 that sorting children does nothing in MCTS. Both passes confirm that for vanilla UCT with full expansion, and both confirm the exception I flagged: **progressive widening, progressive unpruning and lazy expansion all consider only a subset of children at low visit counts, so the expansion order determines which moves get tried at all.**

Combine that with §4 below and something useful falls out. If you need double progressive widening for chance nodes — and you do — then you have a widening mechanism in the tree already, and **temperature is the natural expansion ordering for it.**

So temperature does two jobs through one computation:

```
progressive bias      →  biases selection among expanded children
expansion ordering    →  decides which children get expanded first under widening
```

That is a better fit than either role alone, and it costs nothing extra.

---

## 4. Double progressive widening at chance nodes is required, not optional

**This amends Decisions 01 Q1.**

Both passes confirm MCTS handles chance by sampling where expectimax must enumerate, so the original ruling stands. But both also flag documented cases where MCTS *lost* to careful expectimax in stochastic games — Carcassonne appears in both, via Lanctot et al. (2013) and Heyden (2009) — and one quotes Lanctot directly: MCTS performance in large stochastic domains remains unclear.

Decisions 01 said MCTS is right "for a correctness reason." Soften that to: **MCTS is appropriate, and requires widening at chance nodes to be viable.**

The reason is specific to Overtone and sharper than the general case. Your chance nodes are **measurement outcomes**, and at `n = 10` a position measurement has up to 1024 outcomes. Single-sample-per-visit UCT on a 1024-way chance node is unusable variance.

**But the physics fixes it.** Sample outcomes from the Born distribution and DPW caps how many distinct outcomes you track — so the high-probability outcomes get sampled first and often, and the long tail is never expanded. That is not a workaround; it is the correct thing to do, and the distribution doing the work is the one the game already defines.

Note also that chance nodes appear **only at `measure` moves**, not every move (Part VII §5 rule 3). So the widening burden is concentrated rather than pervasive.

---

## 5. The open question is now yours to answer

Both passes confirm that exact terminal evaluation makes MCTS stronger — Isaac & Lorentz (2016) report partial endgame tablebases producing a "much stronger" Breakthrough player that solves otherwise unreachable positions; Winands et al. (2010) report MCTS-Solver beating standard MCTS by about 65% in Lines of Action.

And both say, independently: **no literature addresses how exact terminals change the shape of the performance-versus-compute curve.** Both then construct the same two-sided argument I did, and neither can resolve it.

### Why this is an asset rather than a risk

You are unusually well equipped to answer it. You have an exact tablebase (M36), a skill-trace protocol, and a compute-budget axis. Nobody in the literature has had all three pointed at this question.

And **you get the answer for free**, because `rollout` is already a field in the agent language:

```toml
[agent.search]
rollout = "tablebase"   # or "playout"
```

Keep it. Agents will submit both kinds, Goodman's `J_opt(b,g) = max_a J(g,a,b)` compares them across the algorithm space, and the ladder measurement *is* the experiment. No separate study required.

**Report it as a finding either way.** "Exact terminal evaluation lengthens / compresses measured game depth by X" is a clean, small, genuinely novel result on a question two independent literature searches came back empty on. It costs you nothing beyond writing down what the grid already shows.

One caution: if tablebase rollouts *do* compress the ladder, that is a real problem for the depth measurement and you will want to know early. Watch it in the first grid run rather than at the end.

---

## 6. Changes to frozen content

| Item | Was | Now |
|---|---|---|
| Temperature mechanism | PUCT prior `P(s,a)` | **Progressive bias**, `W·H/(1+n)` |
| Field name | `temperature_prior` | **`temperature_bias`** |
| Default weight | 0.7 | **0.15** |
| Temperature's second role | none | **expansion ordering under widening** |
| Chance nodes | sampling | **sampling + DPW, required** |
| `rollout` field | in the language | **stays, and it answers §5** |
| Frozen eval vector | five features | unchanged — five |

Nothing here touches the eval vector, so Q1's freeze is unaffected. All of it lives in `[agent.search]`.

---

## 7. References

- Gelly & Silver — *Monte-Carlo tree search and rapid action value estimation in computer Go*, Artif. Intell. 175(11) (2011). The 24% → 69% MoGo result.
- Chaslot, Winands, van den Herik, Uiterwijk & Bouzy — progressive strategies for MCTS (2008). Progressive bias and progressive widening.
- James, Konidaris & Rosman (2017) — analysis of how an incorrect bias concentrates search on suboptimal moves.
- Gedda et al. (2018) — progressive-bias weight tuning in Kingdomino; `W ≈ 0.1–0.2`. *Reported by the research pass; verify the exact figure before citing it in the repo.*
- Coulom (2007) — progressive widening and move ranking.
- Winands, Björnsson & Saito (2010) — MCTS-Solver in Lines of Action, ~65% against standard MCTS.
- Isaac & Lorentz (2016) — partial endgame tablebases in Breakthrough MCTS.
- Lanctot, Saffidine, Veness, Archibald & Winands (2013) — Monte Carlo *-Minimax Search; the Carcassonne result and the stochastic-domain caveat.

**Still outstanding:** Prompt B, on whether CGT temperature is sound outside Go-like decomposable structures. Its answer affects how good the heuristic is, which is exactly what §2's weight is hedging against — so a low default is the right posture until it lands.
