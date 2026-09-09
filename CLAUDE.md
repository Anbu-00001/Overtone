# OVERTONE — agent instructions

Quantum RL instrumentation. A variational quantum circuit that encodes classical data is
exactly a truncated Fourier series in that data (Schuld, Sweke & Meyer, PRA 103, 032430).
This repo points that theorem at a reinforcement-learning policy in real time.

Build specs live in `docs/spec/` — Parts I through VIII, plus the Part VI-A traps addendum.
**Read the relevant part before touching a crate.** Part I is the spine; Parts II–V are
companions that never replace it. Part III §12 ("Minimum viable Overtone") governs scope
whenever a new panel suggests itself.

**One exception to "companions never replace":** Part VII *does* replace Part VI. The arena
is turn-based, the tab is `Orbit`, and Part VI's mechanics survive inside Part VII's seven
rules as content. Do not build both (Part VII §10).

---

## 1. Code retrieval — route the question, do not grep first

Three retrieval tools are wired in. Each answers a different shape of question. Opening
whole files is the *last* step, for narrowing or for editing specific lines — not the first.

| Question shape | Tool | Why this one |
|---|---|---|
| "where is X used", "what breaks if I change X", blast radius | **CodeGraph** | Returns verbatim line-numbered source of the relevant symbols *plus* call paths and a blast-radius summary in one round-trip |
| precise symbol lookup, symbol-level edit, rename, find refs | **Serena** | Live LSP — never stale, and edits at symbol granularity instead of string matching |
| "how does this fit together", architecture, cross-file/doc structure | **graphify** | Scoped subgraph across code *and* the `docs/spec/` markdown |

**Staleness contract.** Serena is live via LSP and is never stale. CodeGraph and graphify
are derived indexes: CodeGraph has a file-watcher and stays current after `codegraph init`;
graphify goes stale until you run `graphify update .`. So:

- After changing code, run `graphify update .` (AST-only, no API cost) before asking
  graphify anything about what you just changed.
- If a CodeGraph query returns nothing, check for a `.codegraph/` directory before
  concluding the tool is broken.
- Never imply you consulted one of these if it was unavailable in the session. Say so and
  fall back to Read/Grep explicitly.

### 1.1 Per-tool specifics for this repo

**CodeGraph** is global-server but per-project-index. It is initialised here (`.codegraph/`).
Its tools take an explicit `projectPath` — pass the repo root:

```
codegraph_explore(projectPath="/home/anbu/26_class/Playground_3/overtone", query="...")
```

Use it before any change that crosses a crate boundary. The dependency direction is strict
(§2 below), and CodeGraph is how you check you are not about to violate it.

**Serena** — call `initial_instructions` before starting a coding task, then
`activate_project` on the repo root. Prefer `find_symbol` / `find_referencing_symbols` /
`replace_symbol_body` over Read-then-Edit for anything symbol-shaped. `rename_symbol` over
a manual find-and-replace, always: the gate/gradient code has short names (`re`, `im`, `q`,
`theta`) that string-replace will corrupt.

**graphify** — when `graphify-out/graph.json` exists, answer codebase questions with
`graphify query "<question>"` before grepping; `graphify path "<A>" "<B>"` for
relationships; `graphify explain "<concept>"` for a single concept. If
`graphify-out/wiki/index.md` exists, navigate with it instead of browsing source. Read
`GRAPH_REPORT.md` only for broad architecture review. The `docs/spec/` markdown is in the
graph, so `graphify query` is the fastest way to find which spec section governs a piece of
code.

### 1.2 Repo hygiene for these tools

`.codegraph/` and `graphify-out/` are in `.git/info/exclude`, **not** `.gitignore` — they
are local derived indexes and must not appear in a diff sent upstream.

---

## 2. Architecture invariants

```
crates/
├── overtone-sim/    state vector, gates, adjoint + parameter-shift gradients
├── overtone-rl/     environments, policies, REINFORCE, rollout buffers
├── overtone-spec/   instrumentation: Fourier, entropy, gradient variance, QFIM
├── overtone-lie/    Pauli bitsets, Lie closure, dim(g), the prediction report
├── overtone-gsim/   Lie-algebraic simulation: Givens rotations in the DLA basis
├── overtone-mps/    bond spectra, truncation, the dequantization test
├── overtone-walk/   discrete-time quantum walks, substrates, the transport exponent
├── overtone-wfc/    Wave Function Collapse and its Shannon entropy (not physics)
├── overtone-qd/     MAP-Elites over policy agents: the Menagerie archive
├── overtone-graph/  maze Laplacian, shared eigenbasis, LMDPs, eigenoptions, Go-Explore
├── overtone-opt/    shot budgets, four optimisers, the barren-plateau flatline
├── overtone-orbit/  Orbit: reachable sets, checkmate, the ladder, the endgame, the dial
├── overtone-cgt/    short games, thermography, temperature, decomposition search
├── overtone-cli/    native trainer, `predict`, `dequantize`; emits JSONL traces
└── overtone-wasm/   wasm-bindgen surface for the browser
```

Dependency direction is one-way and load-bearing:

- `overtone-sim` **knows nothing about RL.** It must not contain the word "policy".
- `overtone-rl` **knows nothing about rendering.** It consumes `sim` through a trait and
  must not reach into its internals.
- `overtone-spec` reads circuits and policies and emits measurements.
- `overtone-lie` **holds no matrices and no RNG.** It depends on `sim` only for the `Gate`
  and `Pauli` types. The dense oracle that checks it lives in `tests/`, outside the library.
- `overtone-gsim` depends on `lie` and `rl`; `overtone-mps` depends on `sim` and `rl`.
- `overtone-walk` depends on `sim` only. `overtone-wfc` depends on nothing — it is not
  physics and must never import a physics crate, or the panel's whole point is lost.
- `overtone-qd` sits above `rl`, `spec`, `lie` and `mps`, because a behaviour descriptor is
  a measurement and every measurement it needs already exists below it.
- `overtone-graph` depends on `mps` for the eigensolver and on `wfc` for a maze to put a
  Laplacian on. It is the one place a physics crate may import `wfc`, and only for its grid.
- `overtone-opt` sits above `sim`, `spec` and `mps`. **Finite-shot measurement itself lives
  in `overtone-sim`** (`measure.rs`), not here: it is a statement about a state vector and an
  observable and about nothing else, and two unrelated consumers need it — the plateau race
  of Part V §5.2 and the shot dial of Part V §4, which turn out to be one instrument.
- `overtone-wasm` is a thin FFI shim. **If it contains an `if` statement about physics,
  that logic is in the wrong crate.**

Check this with CodeGraph before adding a dependency, not after.

**Determinism.** Seed everything through `rand_chacha`. Same seed ⟹ same trajectory, native
and WASM. There is a test asserting this. It matters because the demo shares permalinks.

---

## 3. Conventions that are easy to get wrong

- **Rotation generator convention:** `RX(θ) = exp(-iθX/2)`, likewise RY, RZ. Every gradient
  formula in the codebase assumes this. If you change it, the parameter-shift rule's factor
  of ½ and the adjoint's `-i/2` both change with it.
- **State layout is structure-of-arrays** — `re: Vec<f64>`, `im: Vec<f64>`, never
  `Vec<Complex64>`. SoA autovectorises; AoS does not. Do not "tidy" this into a complex type.
- **Qubit `q` indexes bit `q`** of the basis-state index, little-endian. Amplitude index `i`
  has qubit `q` set iff `i & (1 << q) != 0`.
- **`λ` (input scaling) is a protagonist, not a hyperparameter.** Trainable-vs-pinned is a
  first-class switch at every layer: CLI flag, WASM parameter, UI toggle. Never bury it.
- **No emoji.** Not in the README, not in commit messages, not in the UI, not in code
  comments.
- **`sigma(t) ~ t^beta`, not the variance.** The walk literature quotes both; this codebase
  fits the standard deviation, so ballistic is `1` and diffusive is `1/2`. A paper quoting
  `sigma^2 ~ t^alpha` has `alpha = 2 beta`.
- **A `beta` without its `R^2` is not a measurement.** Localization saturates rather than
  following a small power law, and a line fitted through a plateau reports the plateau's
  noise as a slope. `PowerLaw::is_power_law` is the gate; `regime_of` refuses to name a
  regime without it.
- **The JS budget is 1200 lines, raised once from Part I's 800** when the page went from one
  section to three. The rule it enforces is unchanged: no panel computes a physical
  quantity. If the budget binds again, move code into Rust rather than raising it.

---

## 4. Verification discipline

Every quantitative claim in the README has a test that fails when the claim stops being
true. Do not write a claim you have not measured.

Gradients have two independent paths — adjoint (fast) and parameter-shift (hardware-honest)
— and they must agree to `1e-10` on randomised circuits, for every gate type. A third
check, a dense Kronecker-product reference that shares no code with the strided kernels,
guards the kernels themselves. **Do not proceed past a red gradient test.** Part I §14:
"Nothing renders until the gradients are right."

Report negative results. The dequantization numbers, the entanglement ablation, and the
parameter-count comparison against a classical MLP go in the README whichever way they fall.
Credibility is the scarce resource in this field.

---

## 5. Traps carried forward from the specs

- Do not let the visualiser drive the physics. If a panel wants a quantity that is not
  physically meaningful, cut the panel.
- Do not add environments. Three is already one more than needed.
- Do not oversell. No "quantum advantage" language anywhere.
- Do not use a density matrix for decoherence (Part II) — it destroys the sparse
  representation. Trajectories.
- Do not compute Lie closures with dense matrices (Part III) — bitsets and XOR. A `2ⁿ`
  matrix anywhere in `overtone-lie` is a bug. Tests are exempt and there is one.
- **`dim(g)` describes an ansatz, not necessarily the circuit you have.** Ragone et al.'s
  Theorem 1 needs the circuit to lie in `exp(g)`. A fixed CZ layer is a Clifford, not a
  one-parameter subgroup, so a circuit with fixed entanglers is *not* in the `exp(g)` of its
  trainable generators and no trainability claim follows from `dim(g)`. `Prediction`
  refuses to make one; do not work around that. See `arXiv:2310.11505`.
- The dequantization test is run and published whatever it says (Part III §13). The
  unflattering answer is the contribution, and today the answer is that every agent here is
  a `chi <= 4` tensor network.
- The 100-qubit g-sim claim never appears without its caveat in the same paragraph:
  polynomial `dim(g)` is why it trains *and* why it is classically simulable.
- Do not interpolate animation frames. Discrete time is discrete.
- **Every mechanic in `Orbit` must be a theorem** (Part VI §0, carried into Part VII).
  Nothing is invented for balance. No HP, damage, XP or cooldowns — if a quantity is not a
  physical observable it is not on screen. If the arena is unbalanced, that is a finding,
  not a bug to tune.
- `Lab` is the default tab. `Orbit` never precedes it in the nav, and is never the landing
  page (Part VI §5.5).
- **Do not add maze to add depth** (Part VII §0, §12). More states is the cheap axis and it
  moves depth toward zero. An endless maze scores enormous space complexity and near-zero
  decision complexity, which is Snakes and Ladders. Every instinct to expand is checked
  against the measured `d` first.
- **Do not build the Orbit interface before `d` is measured** (Part VII §12). Phase 9 is
  headless on purpose. If the ladder is flat the interface is wasted work, and by then you
  will be attached to it.
- **Do not soften the checkmate predicate into a score threshold** (Part VII §12). It is
  structural: no unitary in `g` reaches a safe state. The predicate must be **sound** — never
  a false checkmate — and its incompleteness is a measured, reported number, not a reason to
  weaken it.
- **The generator basis is six to eight types.** Chess has six and a library of literature.
  Rule count and depth are close to unrelated.
- **Do not hide the complexity dial** (Part VII §12). It is the most remarkable property the
  game has and the only one no other game can claim — and per Part VIII §6 it controls
  verifiability as well as depth.
- **Do not claim depth before measuring it** (Part VII §12). `d` is a *count of ladder steps*
  against a declared step unit, relative to a declared strategy language — not a length in
  orders of magnitude, and there are no published values for any game to compare against.
- **Agents are declarative, never executable** (Part VIII §2). A submission is a spec with
  trained weights as data. This eliminates arbitrary code execution rather than sandboxing
  it. The moment executable agents are accepted, the project owns a sandboxing problem
  forever.
- **No server** (Part VIII §3, §11). Tier 0 is URL correspondence, Tier 1 is GitHub Actions,
  Tier 2 is peer-to-peer and speculative. Tier 3 is a standing no.
- **Do not let the `.otn` notation drift** (Part VIII §9). Freeze it after the Phase 9
  retuning settles, version it explicitly in the header, and keep a replay-compatibility
  test in CI. The format is the product.
- **Call it the ladder, not the leaderboard** (Part VIII §11). It is how `d` gets measured,
  and the framing is what keeps it a research instrument rather than a growth feature.
- **A trap is a per-cell parameter, never an object** (Part VI-A §7). The moment there is a
  `Trap` struct with a lifecycle, the codebase has become a game engine and the physics is
  decoration on it. Traps live in the maze's seeded per-cell hash: flux, coin id, disorder
  strength, Zeno rate, structure frequency, observable locality.
- **No tuned constant in a trap.** `pi/4 sqrt(N)` is derived; the localisation length is
  measured. Picking a number to make a trap "feel right" means physics has been left.
  Trap density is derived from the substrate and reported, not chosen for difficulty.
- Do not signpost traps and do not add a ninth (Part VI-A §7). Half of them are invisible by
  nature — that is what the instruments are for — and eight already cover localisation by
  interference, disorder, topology, measurement, bandwidth and trainability.
- **Do not let a stat block acquire an invented number** (Part IV §7). One authored line and
  the whole card is fiction. Same rule as the sigil: if the mark stops being a deterministic
  function of the algebra, delete it.
- **Never autoplay audio** (Part IV §5.1). Off by default, one obvious toggle, instant mute.
- **Do not claim WFC is quantum** (Part IV §7). The entire value of that panel is the
  contrast between a borrowed metaphor and the literal thing.
- **Solve the LMDP in log space, never in `z`.** The desirability at hop distance `d` is
  about `exp(-rho d)`, so `f64` underflows once `rho * diameter` passes ~745 — while accuracy
  needs `rho > ~1.4 * diameter`, since the error per hop is `ln(degree)/rho`. Past a diameter
  near 23 no `rho` satisfies both. Iterating `v` with a stable log-sum-exp has no floor and
  the error falls cleanly as `1/rho`. Todorov's own "not too large" caveat is an artifact of
  the representation.
- **A convergence tolerance must be applied at the scale of the quantity.** An absolute
  tolerance on `z` halts the LMDP iteration after two steps while the far end of the graph is
  200 orders of magnitude from its answer. Converge on `log z`. The same error shape has now
  appeared twice; check the dynamic range before choosing a tolerance.
- **`e^(-Lt)` versus `e^(-iAt)` is two operators, not one character** (Part V §1.3). They
  agree up to a global phase only on a *regular* graph, and a maze is not one. Use `L` on
  both sides or the panel's claim is false.
- **MAP-Elites is not a way out of a barren plateau, and Part IV §4 is wrong to say it is.**
  Arrasmith et al., Quantum 5, 558 (2021) prove that cost-function *differences* are
  exponentially suppressed in a plateau, and MAP-Elites decides by comparing fitness, so the
  theorem covers it. Phase 8 cites the same paper for "all four optimisers flatline", so the
  original justification would have put two contradictory claims in one repository. The
  honest reason for the archive is diversity: at a matched budget it reached `J = 0.502`
  against a single-peak search's `0.403`, by holding structurally different agents. Say that
  instead.

- **A barren-plateau demo with exact expectation values shows the opposite of the paper it
  cites** (Part V §5.2). Arrasmith et al. suppress cost function *differences*, so the
  obstruction is measurement precision. With `f64` values every optimiser gets sixteen digits
  free and all four descend at every simulable `n`. The flatline needs `Budget::Shots`.
- **"All four flatline" is too strong; the exponents differ.** Shots needed grows as
  `2^(a n)` for all four, which is the paper's result, but `a` runs from `0.60` for CMA-ES to
  `1.45` for the gradient. Quote the exponents with their `R²`, not the slogan.
- **Never score a noisy optimiser on the best value it saw.** That is a minimum over draws:
  it rewards evaluating more and rewards noise. Score on the exact cost where it stopped.
- **Shots do not sharpen a return distribution** (Part V §4 says they do). The return
  distribution is aleatoric. What shots buy is the score function, which is nonlinear in the
  measured `z`, so a cheap measurement biases the gradient as `1/N`.
- **Dabney et al.'s `kappa = 1` is not scale-free.** On returns of order one every residual
  is inside the Huber quadratic and the fixed point is an expectile. Use `kappa` well below
  the spread of the returns, and special-case `kappa = 0` to `sign(u)`.
- **Machado et al. use the normalized Laplacian in §2.3 and recover the combinatorial one in
  §5.** They coincide only on a regular graph, and a maze never is one. Name which one every
  call site means; a default silently picks one of the paper's two answers.
- **Do not compare eigenvector-derived quantities on a degenerate spectrum.** Inside a
  degenerate eigenspace any rotation is a valid basis, so two solvers disagree for reasons
  that are not about the graph. Check the gap first, or make the claim at the matrix level.
- **Loop-erase both sides before comparing path lengths.** An archived route against a raw
  wander measures which one was post-processed, not which method is better.
- **Part V §3's algebraic reward is beaten by its own reach term.** The gate-count penalty
  has the wrong sign — gates track layers, layers track the frequency ceiling — and the
  `dim(g)` term carries no signal at widths where there is no plateau to be saved from.
  Verified per term, as Part V §10 requires; do not restore the combined form.

- **Checkmate is sound, not complete, and the gap is a dimension count** (Part VII §3). The
  commutant plus per-ideal `g`-purity is the degree-≤2 truncation of a separating invariant
  ring. When `2·2^n − 2 − k > dim(orbit)` the level set holds a continuum of orbits and no
  such certificate can separate them. Never soften it into a score threshold; never claim it
  is exact either.
- **Do not measure orbit completeness on Haar-random pairs.** They disagree on every
  invariant, so the certificate scores 1.000 on a family that is provably incomplete. Build
  pairs that share the invariants, and report the game-relevant number separately.
- **An exactly-0.500 ladder is a bug signature, not a flat game.** It means the outcome does
  not depend on the strategy — in Orbit's case that the two players never interacted. Check
  that the opponent is actually in the loop before reporting depth.
- **`d` is a count of steps against a declared language, not a span of compute** (Lantz et
  al.). A saturating ladder may be a fact about the language: a depth-1 policy has only
  `|moves| × |angles|` candidates, so budget stops buying at that point.
- **A pursuer spread uniformly absorbs nothing.** The absorbing threshold is `1/dim`, so a
  field thinned across the window threatens no cell.
- **`Lmdp::recommended_rho` is for the `z`-space solve only.** It caps `rho` at 60 to keep
  `exp(-rho D)` representable. In log space there is no floor and the cap only costs accuracy;
  choose `rho` for accuracy alone, around `40 D`.

- **CGT temperature is defined for games of no chance** (Berlekamp, verbatim: "two-person,
  perfect information games of no chance"). Overtone's measurement is a chance node, so every
  temperature is a temperature of the **coherent segment** between collapses. Do not average a
  temperature across a measurement.
- **"Move in the hottest region" is a heuristic, not a theorem.** `{0|-3} + {{1|-2}|-3}` with
  Left to move has distinct temperatures 1.5 and 1.0 and loses a point to hottest-first,
  because the colder component's option is hotter than the component. It *is* exact on sums of
  plain switches, which is the case the theory covers. Use temperature for ordering; never
  claim optimality.
- **A game can be a number without any option list being all-numeric.** `{{-1|-2} | 1}` is a
  number by the simplicity rule, and `overtone-cgt` reports a meaningless temperature for
  those by design. Filter generated games through `Game::is_hot` before searching over them;
  the first hottest-first "counterexample" found was this bug.
- **Both Overtone temperatures climb with the ply.** A raw correlation between CGT temperature
  and von Neumann entropy is mostly a correlation with time: 0.344 raw, -0.021 with the ply
  partialled out, sign mixed. Partial out the ply before reporting any trajectory correlation
  in this repo.
- **The Chinese Rings is not two thousand years old.** The Zhuge Liang attribution is Culin
  via an unnamed informant; the documented references are Yang Shen (early 16th c.) and
  Pacioli (1509). It is therefore not "older than algebra" either. The argument does not need
  the date — use `overtone_graph::rings::PROVENANCE`.
- **Grover over-rotation has no floor; a die does.** Disadvantage bottoms out at `p^2`; keep
  turning the dial and the marked outcome becomes arbitrarily unreachable. How far you must
  turn is an equidistribution question about `theta/pi` and is not monotone in register size.
- **An Elo ladder over submitted agents is a skill chain, not a strategy ladder.** Lantz et
  al. separate the two explicitly and require a *declared language* plus a computational-
  resource axis for `d`. A league of independently-designed agents has neither, so its Elo
  spread is not `d`. Report both, labelled. The same paper warns that random elements loosen
  the link between decisions and outcomes, which applies to every win-rate measurement in this
  repo, Phase 9's ladder included.
- **Mahadev (FOCS 2018) is a classical verifier and a *quantum* prover, under LWE.** The
  asymmetry there is quantum-vs-classical. Overtone's verification cost is classical
  simulation on both sides, so the analogy does not carry and Part VIII 6's framing is wrong
  in kind. Cite it for the idea that verification can be cheaper than execution, not for the
  construction.
- **Sansoni et al. DID measure anyonic statistics; an external audit said they did not.**
  The abstract names only bosons and fermions, but the body prepares anyonic states at
  `phi = pi/4, pi/2, 3pi/4` and Fig. 4(c) plots `phi = pi/2`. Checked against the paper, not
  against the audit. They *simulate* exchange with photon polarisation rather than making
  anyons, so cite van Exter et al. (PRA 85, 033823) alongside -- not instead. General rule
  this is an instance of: **an abstract is not a source.** Read the body before recording a
  contradiction.
- **Pauli exclusion is a statement about modes, not about sites.** Part VI 1's "a fermionic
  agent walls off a corridor" is wrong: a coined walk has a site *and* a coin, so two
  fermions share a site with opposite coins (Sansoni Eq. 4). Measured: fermionic mode
  diagonal exactly 0, position diagonal 0.094. A fermionic class blocks one coin state, not
  the corridor.
- **Two walkers one site apart never interact.** A coined walk preserves the parity of
  `site + step`, so adjacent inputs occupy disjoint sublattices forever and every exchange
  term vanishes. All statistics then give identical distributions and a test built on that
  arrangement passes while measuring nothing.
- **`DefaultHasher` cannot back a self-verifying file format** -- it is not stable across
  Rust releases. And hashing raw `f64` bits would make the hash platform-dependent, since
  native and wasm agree only to ~5.6e-16. `.otn` uses its own FNV-1a over amplitudes
  quantised to `1e-9`.
- **A benchmark category whose answers are lopsided measures nothing.** `Overtone-100`'s
  reachability set first came out 27 reachable to 3 -- 90% for a solver that always answers
  the same way. Check `balance()` on any yes/no category before shipping it.
- **The hypercube's prescribed hitting time has a parity condition.** Kempe's `T ~ pi n / 2`
  is `T = n (mod 2)` with `|T - pi n / 2| <= 1`. The cube is bipartite by Hamming weight and
  the antipode has weight `n`, so rounding to the nearest integer without the parity gives
  *exactly zero* for half the dimensions -- which reads as a broken walk, not a broken test.
- **The Grover coin on two ports is the Pauli `X`.** Any degree-2 vertex, and the whole
  2-cube, therefore evolves deterministically: at `n = 2` the "hitting probability at the
  prescribed time" is an exact `0`. Do not treat an asymptotic hitting theorem as a claim
  about small `n`; report the small cases rather than starting the table where they flatter.
- **The welded tree is 3-regular except at its two roots, which have degree 2.** Li, Li and
  Luo use a genuinely 2-dimensional coin there. Padding the roots to three ports produces a
  walk that runs, stays unitary, and is not the published one. The coin interface takes
  `d_u` per vertex for this reason alone.
- **The welded tree's target is `|t, phi(t)>`, not the vertex `t`.** Theorem 4.1 is about the
  overlap with the uniform superposition of the target's arcs; on a degree-2 root that is not
  the same number as the probability of being at the vertex, and a reduced-versus-full
  comparison against the wrong one disagrees for a reason that has nothing to do with the
  reduction.
- **`3.6 n log(5n)` is base 2.** The paper writes `log`; Eq. (4.73) pins it by using
  `(1/2)^(log 5n) = 1/(5n)`. Also: arXiv v2 says `1/(20n)`, the Algorithmica version says
  `1/(24n)`. Cite the version you measured against.
- **A wrong reduction is invisible.** It runs, it stays unitary, and it produces a plausible
  hitting curve. The only thing that can catch it is the walk on the graph it claims to
  reduce -- with the random parts of the graph actually varied, since the reduction's whole
  claim is that they do not matter.
- **Krovi and Brun's infinite hitting times are a DFT-coin phenomenon.** The Grover coin on
  the same graph, from the same start, arrives with probability one. Measured on the 4-cube:
  DFT traps exactly `3/7` forever, Grover traps nothing, and on the *3*-cube DFT traps
  nothing either. Any note citing this has to name the coin and the dimension or it reads as
  contradicting Kempe.
- **Marked-vertex search is not entrance-to-exit traversal.** Szegedy's `sqrt(HT)` result is
  about finding a marked vertex from a uniform start; the welded-tree and hypercube results
  are about traversal from a fixed source. Part II 6 applied the first to the second. There
  are three distinct notions -- one-shot, concurrent, search -- and only the first two are
  what Phase 5's families measure.
- **A flat heuristic plus a stable sort is the move generator's enumeration order.** The
  temperature field is uniformly `-1` *in the opening*, so every candidate ties; under
  progressive widening the root expands about 23 of 240, and since `legal_moves` emits all
  208 `Apply` candidates before the 32 `Measure` ones, **no measure move was ever tried**.
  Shuffle before a stable sort whenever the key can be constant.
- **Never characterise a game quantity from opening positions alone.** This one cost two
  wrong conclusions in a row. The temperature field is `-1` everywhere for the first few
  plies and then heats up -- 0% of regions hot at ply 0, 46% by ply 6, 75% by ply 9 at
  `n = 4` -- so a sample of five plies from one seed reported "uniformly cold" and produced
  both a false claim about the heuristic and a false correction to a Phase 12 measurement
  that had been right. **A quantity measured only where a game begins will look like
  whatever beginnings look like.**
- **`left > right` is not a hotness test.** Measured gaps come in two populations: real ones
  at `1e-1` to `1e0` and dust from `1e-32` down to `1e-96`. Without a floor, `n = 6` at ply 8
  reports 50% hot with a median gap of `6.7e-32`, which is 50% of nothing. The floor is
  `1e-12`, derived from native-versus-wasm amplitude agreement at `5.6e-16`.
- **A constant heuristic cannot reorder anything.** `W*H/(1+n)` with `H` constant adds the
  same number to every child. A test asserting that the weight changes the search would be
  asserting that a constant offset breaks ties -- so on a flat field, assert the inertness.
- **The move-ordering heuristic can cost more than the search.** One temperature field is
  577 us at `n = 4`; MCTS creates about one node per playout, so per-node evaluation would
  cost 0.3 s at budget 512 against the 0.09 s the whole search takes. Evaluate it once at the
  root and let progressive bias's decay absorb the staleness.
- **An aborted search iteration must keep its best move.** `negamax_root` discarded the best
  move of an iteration cut short by the node budget and fell back to the first candidate, so
  every budget below the candidate count played the same move. The only symptom was two
  identical win rates on a ladder -- which reads as a flat ladder, not as a bug.
- **A rate hides the shape; print W/D/L.** Low-budget agents here do not lose, they *draw*,
  and converting draws is what compute buys. Two rungs can look identical on the rate and be
  completely different underneath.
- **Sample across seeds and plies, or measure nothing.** Three separate wrong statements in
  one afternoon came from the same habit: reading a game quantity off the opening. The
  temperature field is `-1` for the first few plies and then heats up; the standing absorbed
  weight looks like zero at six plies and is `1e-1` at eight. Both readings were taken from
  one seed near the start. **A quantity measured only where a game begins will look like
  whatever beginnings look like**, and the second and third errors were *corrections* to the
  first, which is how a bad sample propagates.
- **A threshold defined in one place and bypassed in another prints two different answers
  for the same quantity.** `coldness.rs` defines `HOT_FLOOR` and `Reading::is_hot` uses it --
  but the by-ply table in the example compared `l > rr` directly, so the headline read `0.88`
  hot at ply 6 where the floored value is `0.46`, and the inflated number reached the phase
  notes, the traps file and a gate line before the two tables were compared. If a predicate
  is worth a named constant, nothing may re-implement it inline.
- **The enumeration order of `legal_moves` leaks into behaviour wherever a search stops
  early.** Three incidents, one shape. Progressive widening expanded a prefix and no measure
  move was ever tried; `negamax_root` scanned candidates in generation order and aborted on
  the node budget, so "budget b" meant "best of the first b candidates" -- budget 6 beat
  budget 48 sixty games to nil and budget 96 lost to budget 6 by the same margin. **Any
  place a budget truncates a candidate scan must sample or shuffle first**, or the compute
  axis is measuring the move generator.
