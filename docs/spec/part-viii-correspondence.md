# OVERTONE PART VIII — Correspondence

**Global agent competition, without a server.** Eighth companion. Depends on Part VII.

---

## 0. Verdict: right instinct, wrong shape

Not dumb. But "invite agents to compete" and "build multiplayer" are different problems, and conflating them is what kills projects like this.

Here is the reframe the whole document rests on:

> **Chess became a global competitive game three centuries before servers existed. It scaled through notation.**

Algebraic notation, then PGN and FEN. A game is a text file. You could post it in a newspaper, mail it to Argentina, or write it on a napkin. The entire chess ecosystem — opening theory, databases, correspondence leagues, engine tournaments, the whole apparatus — runs on a **file format**. Lichess is a recent convenience layered on something that already worked without it.

**Overtone needs a notation, not a server.** And it is unusually well positioned for one, because Part I §5 already mandates that every run be deterministic and reproducible from a seed, natively and in WASM.

Everything below follows from that. The infrastructure question mostly dissolves.

---

## 1. The notation — `.otn`

A complete match:

```
header:   engine version, ruleset version
seed:     maze seed + substrate rule                  ~16 bytes
agents:   two declarative specs                       ~200 bytes each
moves:    generator index + target region, per ply    ~2 bytes per ply
result:   outcome, final report, state hash
```

**A few hundred bytes for a whole game.** Anyone with the engine reproduces it bit-for-bit.

Three format rules that matter more than they look:

- **Human-readable text.** Chess players read PGN directly, and that is not incidental — a format people can read is a format people build on.
- **Self-verifying.** The file carries a hash of the final state, so re-running is a proof.
- **Version-pinned in the header.** Physics changes break replays. Pin the engine version and keep a replay-compatibility test in CI.

**This format is the actual deliverable of Part VIII.** Everything after it is convenience.

---

## 2. The decision that determines whether this works: agents are declarative, not executable

A submission is a **spec**, never code:

```toml
[agent]
name        = "kestrel"
generators  = ["pawn", "bishop-Z2", "knight-hop", "rook-phase"]
statistics  = "fermionic"          # or bosonic, or anyonic with a phase
coin_weights = [ ... ]             # trained parameters — data, not code
policy      = "greedy-reach"       # from a fixed, published vocabulary
```

Four reasons this is the right call, in order of importance:

1. **It eliminates arbitrary code execution rather than sandboxing it.** You never run a stranger's code in your CI. This is the single largest operational risk in every AI competition, and this design removes the category instead of mitigating it.
2. **Matches stay deterministic and verifiable.** A spec plus a seed is a proof obligation anyone can discharge.
3. **The competition becomes about physics design, not software engineering.** Which is the actual subject of the project. A submission is a *hypothesis about which algebra wins*.
4. **Files stay tiny**, so the whole archive fits in a repo.

The cost is expressiveness: you cannot submit an arbitrary Python bot. The mitigation is that **trained weights are data** — submit what your agent learned, not the code that learned it. That covers the interesting case.

This is exactly the FEN/PGN model: declarative, not executable. It is why chess databases exist and bot-battle platforms mostly do not.

---

## 3. Four tiers, in ascending infrastructure cost

Pick as many as you want. They compose; none requires the next.

### Tier 0 — URL correspondence. Zero server.

You move, you get a URL, you send it. The URL fragment carries the entire match state, because the match state is a few hundred bytes. Your opponent opens it, moves, sends it back.

This is play-by-mail chess, and it is **the honest answer to "a simple room game with no dedicated server."** It works offline, works forever, costs nothing, and is essentially already implemented by Part IV §5.2's permalinks.

### Tier 1 — GitHub is the server ← **recommended**

Verified against GitHub's docs: **Actions on standard runners is free and unmetered for public repositories, with no minute cap.** Limits are 6 hours per job, 20 concurrent jobs on the Free plan, 256 matrix jobs per run — on the order of 120 free compute-hours per workflow run.

That is an entire tournament infrastructure, free, for a public repo.

```
submission   →  a pull request adding agents/<name>.toml
validation   →  CI checks the spec against a schema; rejects anything executable
tournament   →  nightly Actions job runs the ladder
results      →  LEADERBOARD.md committed back; match archive → HF Dataset
```

**GitHub becomes the server**, and it arrives with identity (accounts), moderation (PR review), version history (every agent's evolution is a diff), notifications, and a discussion venue — all of which you would otherwise have to build.

And the star mechanic Part III has been circling since the beginning: **people star repositories they have contributed to.** A merged pull request is an incomparably stronger engagement event than a page view, and it produces an advocate rather than a visitor.

### Tier 2 — WebRTC rooms. Peer-to-peer, no server.

If you want live rooms: the engine is already WASM in the browser, so two browsers can play each other directly over a WebRTC data channel. Match state is small and light-cone bounded, so bandwidth is trivial.

The usual objection is that WebRTC needs a signalling server. It does not have to be yours — libraries like **Trystero** (`dmotz/trystero`) do serverless matchmaking over BitTorrent trackers, Nostr, MQTT or IPFS, with the same API across strategies, and application data never touches the rendezvous medium. Rooms in a few lines, no infrastructure.

Build this only if Tier 0 or Tier 1 produces people asking for it.

### Tier 3 — an actual server

**Don't.** A server means uptime, cost, auth, abuse, moderation, and a single point of failure for a project whose most valuable property is that it always works. Revisit only when Tiers 0–2 are visibly saturated, which is unlikely.

---

## 4. Cold start is already solved

Most competition platforms die of an empty ladder: the first visitor finds nobody to play and never returns.

Overtone does not have this problem. **Part IV's MAP-Elites archive already generates agents.** Seed the league with a hundred Menagerie elites spanning the `(dim g, β, χ)` space, and a human submission on day one joins a populated, genuinely diverse ladder.

That single detail is most of the difference between a league that starts and one that does not.

---

## 5. This is an instrument, not a feature

Part VII §8 said to measure depth `d` by plotting performance against compute budget, with the strategy ladder counting distinguishable skill tiers.

**A population of externally-designed agents is a far better `d` measurement than self-play.** Self-play measures how well the game resists one optimiser. A league measures how well it resists many independent minds — which is precisely what Lantz's definition asks for and what self-play cannot give you.

> **The Elo ladder over submitted agents *is* the strategy ladder. Its length is `d`.**

So the leaderboard is not a scoreboard bolted on for engagement. It is how you measure the thing Part VII said to measure, and it measures it more honestly than anything you can do alone. That is the reason this belongs in the project.

Frame it that way in the docs, consistently. Call it the ladder, not the leaderboard.

---

## 6. Verification, and the complexity dial applied to trust

The novel part, and worth writing about separately.

Matches are deterministic and seeded, so anyone can re-run and verify; the match file is self-verifying and CI re-runs a random sample. Standard.

But Part VII §6's dial does something unexpected here:

```
low  dim(g), low  χ  →  g-sim / MPS evaluate positions efficiently
                     →  every match is trustlessly verifiable
                     →  CI re-runs everything

high dim(g), high χ  →  no efficient classical evaluation exists
                     →  VERIFYING a match is as hard as PLAYING it
                     →  sampling and trust are all that remain
```

> **The complexity dial does not only control game depth. It controls verifiability.**

A game with a tunable trust regime is, as far as I can find, unprecedented. It also connects to a real research area — classical verification of quantum computation (Mahadev, 2018) — and Overtone's version is a much simpler instance of the same shape.

**Practical policy:** the official ladder runs in the verifiable regime, where CI checks everything. An **open division** runs above it, clearly labelled as sampled rather than verified. Do not hide the distinction — make it a visible part of the exhibit, because "here is where verification becomes intractable" is one of the more interesting sentences the project can say.

---

## 7. Design for fifty, not millions

Straight talk about scale, since you raised it.

A repository with good traction gets somewhere between five and fifty agent submissions. Not thousands. **Design for fifty.**

- Round-robin at 50 agents is 1,225 matches — trivial in one nightly job.
- At 200 agents round-robin becomes 19,900 matches; switch to Swiss pairing or Elo-based matchmaking. Well-understood, no new engineering.
- 20 concurrent jobs × 256 matrix entries covers all of this with room to spare.

**You will probably never need a server.** And fifty independently-designed agents is not a disappointing outcome — it is a better `d` measurement than a thousand near-identical ones would be.

---

## 8. The other direction — build the puzzle set first

If the league does not appeal, or as the lower-risk first step, this is arguably the better artifact:

**A benchmark of positions with exact ground truth.**

Part VII §4 gives you something almost no game benchmark has: once the walker decoheres, the LMDP eigensolve produces the **exact optimal play**. So you can publish positions whose correct answers are proven, not merely agreed.

`Overtone-100`: a hundred curated positions — reachability puzzles, checkmate-in-`n`, spectral traps, cage escapes, endgame conversions — each with a verified solution.

Why it may beat the league:

- **It works with zero participants.** Useful the day it ships; a league is useless until people arrive.
- **It has ground truth.** Leagues never do — Elo is relative, and a ladder can be tall and meaningless.
- **Benchmarks get cited.** Leaderboards get played for a month and then go quiet.
- **It makes the league better** if you build both, by giving every submitted agent an absolute score alongside its relative one.

**Recommendation: build the puzzle set first.** It de-risks the league, it is the more durable artifact, and it is the thing a researcher links to.

---

## 9. Risks, honestly

**Nobody submits.** The most likely outcome. Mitigations: the Menagerie bootstrap (§4); one-file submission with a template; and **write the first ten agents yourself, with genuinely different strategies**, so the meta starts diverse instead of converging on whatever the first submitter happened to try.

**Degenerate meta — one strategy dominates.** This is *data*, not failure: it means `d` is small, and Part VII §8 tells you which parameters to retune. A failure mode that produces a measurement is a good failure mode.

**Code execution.** Eliminated by §2 rather than sandboxed.

**Physics drift breaks old replays.** Version-pin in the header; keep a replay-compatibility test in CI; never silently change the ruleset.

**Actions abuse from fork PRs.** Fork pull requests do not receive secrets by default, and first-time contributors require maintainer approval to run workflows. Keep both defaults on.

**The notation drifts.** Freeze it early and version it explicitly. A format that keeps changing is a format nobody builds on, and the format is the product.

---

## 10. Build order

**M40 — the puzzle set.** `Overtone-100` with eigensolve ground truth. Acceptance: every solution verified against brute force on small instances.

**M41 — the notation.** `.otn`, frozen and versioned, with a reference parser and a replay-compatibility test.

**M42 — Tier 0.** URL correspondence. Mostly already built.

**M43 — Tier 1.** GitHub league: PR submission, schema validation, nightly tournament, Menagerie bootstrap.

**M44 — the ladder published.** Elo over all agents, framed and documented as the strategy ladder, feeding Part VII §8's `d` measurement.

**M45 — Tier 2.** WebRTC rooms. Only if someone asks.

---

## 11. Traps

- **Do not build a server.** Every tier above exists to avoid it.
- **Do not accept executable agents.** The moment you do, you own a sandboxing problem forever.
- **Do not launch the league before the puzzle set.** An empty ladder is a bad first impression; a benchmark is useful immediately.
- **Do not call it a leaderboard.** It is the strategy ladder, and the framing is what keeps this a research instrument rather than a growth feature.
- **Do not hide the verification regime.** The point where verification becomes intractable is a feature of the exhibit, not an embarrassment.
- **Do not build Tier 2 speculatively.** Live rooms are the most fun to build and the least likely to be needed.
- **Do not let the notation drift.** Freeze early, version explicitly.

---

## 12. References

- GitHub — Actions usage limits and billing. Standard runners are free and unmetered on public repositories; 6 h per job, 20 concurrent jobs on Free, 256 matrix jobs per run. Re-verify before relying on it.
- `dmotz/trystero` — serverless WebRTC matchmaking over BitTorrent, Nostr, MQTT, Supabase, Firebase or IPFS. Tier 2.
- PGN and FEN specifications — the model for §1. Read them before designing `.otn`; they are unusually well-designed and forty years of use has tested them.
- Lantz, Isaksen, Jaffe, Nealen, Togelius — *Depth in Strategic Games*, AAAI 2017. Why §5 is the real justification for the league.
- Mahadev — *Classical verification of quantum computations*, FOCS 2018. The research context for §6.
- Todorov — *Efficient computation of optimal actions*, PNAS 106, 11478 (2009). The eigensolve that gives §8 its ground truth.
