# OVERTONE — DECISIONS 03 (revised)

**Supersedes the earlier Decisions 03 in full.** Both external research passes are in. Q7 is materially refined, Q9 reverses, and Q7's findings have knock-on corrections for Parts II and VI-A that are collected in §11.

Two things to read before you build anything: **§1.4** (the zero-error claim is not what the abstract implies) and **§2** (Szegedy needs no engine at all).

---

## Q7 — the walk operator

### 1. Welded trees: build the coined version. Confirmed, with four corrections.

The paper is no longer just a preprint. It appeared at **SODA 2024** (pp. 2454–2480) and as **Algorithmica 86(12), 3719–3758** (October 2024), by Li, Li & Luo, journal title *Recovering the Original Simplicity: Succinct and Exact Quantum Algorithm for the Welded Tree Problem*. Peer-reviewed, cited, no independent end-to-end replication found.

Your option (b) — implement coined and report the speedup absent — remains **disqualified**. The exponential separation genuinely survives in the plain coined model. But four details change what you build.

#### 1.1 The operator is exactly as elementary as hoped

```
U_walk = S · C

C_u = 2|φ(u)⟩⟨φ(u)| − I          Grover reflection at each vertex
|φ(u)⟩ = (1/√3) Σᵢ |Γ(u,i)⟩       uniform neighbour superposition
S|u,v⟩ = |v,u⟩                    flip-flop shift, a SWAP of registers
```

Standard Grover coined walk. Nothing exotic.

#### 1.2 Variable degree at the roots — this will bite you

The welded graph is 3-regular **except** at the entrance and exit, which have degree 2. The paper handles this explicitly with a 2-dimensional superposition there:

```
|φ(r)⟩ = (1/√2)(|Γ(r,i₂)⟩ + |Γ(r,i₃)⟩),   r ∈ {s, t}
```

So the implementation is **variable-degree Grover reflection**, not a uniform 3-port coin. Your coin interface must take `d_u` per vertex:

```
|φ_u⟩ = (1/√d_u) Σ_{v~u} |v⟩
```

Build that generality in from the start. Padding the roots to three ports is not what the paper does, and it is the kind of shortcut that produces a subtly wrong result you cannot debug.

#### 1.3 The state space maps cleanly — my "verify this" is resolved

`|u,i⟩ ↔ |u, Γ(u,i)⟩`, so edge states and position⊗coin are the same object on a regular graph, and the flip-flop shift becomes `S|u,i⟩ = |v,j⟩` where `v = Γ(u,i)` and `u = Γ(v,j)`.

One implementation difference worth knowing: they do **not** store an abstract 3-dimensional coin register. They store the neighbour's `2n`-bit label and manufacture the Grover superposition from the adjacency oracle via `C = U_φ Ref_⊥ U_φ†`. Support both views behind an encoding layer rather than committing to one.

#### 1.4 The zero-error claim is not a bare-walk result

**This is the most important correction, and my earlier ruling overstated it.**

```
plain walk alone      →  p = Ω(1/n)          ← this is what gives the exponential separation
zero error            =  plain walk  +  exact amplitude amplification
```

The amplification layer is `G(α,β) = A S₀(β) A† S_t(α)` with `A = U_walk^{T₁} U_p`, phases derived from the *known* overlap: `θ = arcsin(p_{T₁})`, `T₂ = ⌈(π/2 − θ)/(2θ)⌉`.

So Decisions 03's "there is a zero-error target to hit" was wrong as written. **The milestone's acceptance test is `p = Ω(1/n)`, not zero error**, unless you also build the amplification layer — which is optional, because the exponential quantum/classical query separation is already established by the plain walk against `2^Ω(n)` classical queries.

Two further qualifications:

- **`T₁` is classically precomputed**, not discovered by the circuit: `2n < T₁ < 3.6n·log(24n)`, chosen by iterating the reduced matrix.
- **The theorem is asymptotic.** It states that for *sufficiently large* `n`, `max_{t∈[2n,T]} p(t) > 1/(24n)`. It is not proven for every finite `n`. Do not write an acceptance test that assumes it holds at `n = 4`.

#### 1.5 The gift: a (4n+2)-dimensional reduction

By the welded-tree symmetry, evolution stays inside a **(4n+2)-dimensional invariant subspace**. Directed edges group by layer and direction; the walk becomes a `(4n+2)×(4n+2)` matrix `M_U = M_S M_C`, with the initial state `|0,R⟩ = |s,φ(s)⟩` as the first basis vector and the target `|2n+1,L⟩ = |t,φ(t)⟩` as the last.

At `n = 10` that is a **42×42 matrix**. Trivially cheap.

This substantially changes the build calculus. The glued-trees demo is no longer an expensive Phase 5 item — it is a small dense matrix iterated in a loop. And it renders exactly as Part II §P5 specified: the exponentially large graph collapsing to a line in front of the reader, which is now the literal computation rather than a visualisation choice.

**Acceptance test, two levels:**
1. Implement the reduced `M_U`, iterate to find `T₁`, verify `max_{t∈[2n,T]} p(t) > 1/(24n)` across a range of `n`.
2. For small `n`, verify the reduced model against the full coined walk on the actual graph.

The second test is what catches an incorrect reduction, which is otherwise invisible.

#### 1.6 The oracle is the real cost

```
O: |u⟩ ⊗ᵢ|vᵢ⟩  →  |u⟩ ⊗ᵢ|vᵢ ⊕ Γ(u,i)⟩
```

One query returns all three neighbours coherently. The coin costs **2 oracle calls** under the explicit `U_φ` construction. *(Note: the paper's introduction says 4; Lemma 2.1 and the explicit construction give 2. The construction is the better evidence — expect this discrepancy when reading.)* A weaker Childs-style one-neighbour oracle costs only a constant factor more, which the authors state.

#### 1.7 Prior art, correctly classified

- **Carneiro, Loo, Xu, Girerd, Kendon & Knight (2005)** — coined walks on glued trees, but an *entanglement and spreading study*, not a traversal algorithm. Useful precedent, not a competing result.
- **Ide, Konno, Segawa & Xu (2014)** — DTQW localisation on glued trees via reduction to path graphs and Jacobi-matrix spectral analysis. Cited by Li–Li–Luo as inspiration for their reduction. Localisation ≠ traversal.
- **Jeffery & Zur (STOC 2023)** — multidimensional DTQW, `O(n)` queries, genuine discrete-time exponential speedup, but not a plain coined walk.

The progression is: Childs et al. CTQW → Jeffery–Zur multidimensional DTQW → **Li–Li–Luo plain coined**. Only the last is relevant to your engine, and citing the lineage correctly is worth doing.

### 2. Szegedy: build nothing

**Resolved, and it deletes the largest open item in Phase 5.**

Three results establish the equivalence:

- **Portugal (2015, arXiv:1509.08852)** — conditions under which coined and Szegedy walks correspond, via the staggered model after bipartite conversion.
- **Wong (2016)** — with a Grover coin and flip-flop shift, **two coined applications correspond to one Szegedy application**.
- **Portugal & Segawa (2017)** — flip-flop coined models with generalised Grover coins convert directly to Szegedy walks on the **subdivision graph**.

The mechanism is transparent once seen. Szegedy reflects about `|ψ_u⟩ = (1/√d) Σ_{v~u}|u,v⟩` — the same object as a Grover coin reflection around the uniform neighbour state. `W = R₂R₁` needs two reflections; `S` flips orientation, so `U² = (SC)²` reproduces the two-reflection structure.

**The bipartite double cover is a representation device, not a second engine.** Delete the Szegedy module from scope.

#### The caveat that matters most for Overtone

The equivalence is **class-specific**: Grover coin plus flip-flop shift, on a regular graph, with the appropriate search construction. A DFT coin, an arbitrary position-dependent coin, a non-reversible walk, or an arbitrary scattering walk does **not** inherit Szegedy's theorem.

> **Part II §P7's learned, position-conditioned coin is precisely "arbitrary position-dependent coin" — outside the equivalence class.**

So a learned coin carries **no automatic quadratic hitting-time guarantee**. If the learned coin beats the Grover coin on some family, that is an empirical finding requiring its own justification, not an inherited theorem. Add this to Part II §P7 explicitly, because it is exactly the kind of thing that would otherwise get claimed by accident.

### 3. Hitting-time speedups: three notions, do not conflate

The literature uses one phrase for three different things, and Part II §P6 currently blurs two of them.

| Notion | Definition | Status |
|---|---|---|
| **One-shot hitting** | `\|⟨v\|U^T\|u⟩\|² ≥ p`, no measurement during the walk | The welded-tree and hypercube results |
| **Concurrent hitting** | measure every step, first detection time | Krovi & Brun's setting |
| **Quantum-walk search** | uniform state, marked subset, phase/coin modification | Grids, Johnson graphs, complete graphs |

**Marked-vertex search is not entrance-to-exit traversal.** Part II §P6 describes Szegedy's `√HT` result and then applies it to maze traversal; those are different problems, and the section needs the distinction stated.

**By family:**

- **Hypercube — the clean example.** Kempe (2005, *Probab. Theory Relat. Fields*) proves a discrete-time walk reaches the opposite corner in polynomial time where the classical process is exponential, explicitly the first exponential gap in *discrete* quantum-walk hitting time. Peer-reviewed, twenty years old, uncontroversial.
- **Krovi & Brun** — repeated-measurement hitting on the hypercube. Coin choice matters dramatically, and they exhibit **infinite hitting times caused by destructive interference**. See §11 for why that matters to you.
- **Cubelike graphs** — a September 2026 preprint (arXiv:2609.04503) proves `p_T = 1 − O(Δ^{-1/5})` at `T ≈ πΔ/2` for degree `Δ → ∞`. Very recent, not independently confirmed. Useful test-suite material; do not build a claim on it yet.
- **Grids, complete graphs, Johnson graphs** — quadratic *search* speedups, well established, different problem.
- **NAND/AND-OR trees** — Childs, Reichardt, Špalek & Zhang evaluate any NAND formula in `N^{1/2+o(1)}` using a coined walk. Demonstrates the model's reach; not a hitting-time result.

### 4. Architecture and build order

**One generalised coined core. Not three engines.**

```
                    U = SC
          ┌───────────┴───────────┐
   encoding layer            coin interface
   |u,i⟩ ⟷ |u,v⟩         C_u = 2|φ_u⟩⟨φ_u| − I
                          variable degree d_u
          └───────────┬───────────┘
                 flip-flop shift
                      │
        ┌─────────────┼─────────────┐
   hypercube       grids       welded trees
                      │
          search / exact amplification layer
                      │
             Szegedy-equivalent results
```

Do **not** build `CoinedWalkEngine` + `SzegedyWalkEngine` + `WeldedTreeEngine` on the assumption that welded trees need a qualitatively different walk. They do not.

**Revised build order for Phase 5:**

1. **Hypercube hitting (Kempe).** Clean, peer-reviewed since 2005, no oracle needed, needs only the Grover coin and correct initial/target states. Do this first — it validates the core against a settled result before you take on a 2024 one.
2. **Welded trees, reduced model.** The (4n+2) matrix. Cheap, and it is the Part II §P5 panel.
3. **Welded trees, full coined walk on the graph.** Validates the reduction. Needs the coherent adjacency oracle.
4. **CTQW.** Still worth building, but for Part V §1.3's real-versus-imaginary exponent toggle — not for glued trees.
5. **Exact amplification layer.** Optional. Buys the exactness claim; the separation does not need it.
6. **Szegedy.** Deleted.

---

## Group A — the freeze

### Q1 — the v1 feature vocabulary

**Admission first: I invented `reach_margin`.** It was placeholder TOML in Decisions 01 and corresponds to nothing implemented. Your guess (`1 − absorbed_weight`) is a sensible quantity but measures the wrong kind of thing — absorbed weight is a *state* measure, whereas the gap I was gesturing at is a *reachability* measure. Part VII §3's checkmate is binary and a linear eval needs a gradient toward it, and two quantities you already have serve that better.

**Ruling — six features, one per resource axis:**

```
1. coherence          the clock              Player.coherence
2. dim_g              material               Lie closure, memoised
3. orbit_size         reachability           orbit_dimension
4. safe_set_size      proximity to loss      safe-set spec
5. temperature        positional urgency     M48
6. average_branching  mobility               move generation at node
```

The organising principle: **each feature tracks a different one of the game's resources.** A linear eval over correlated features wastes dimensions, and these six map onto the six things Part VII says a position consists of. `reach_margin`'s intent is covered by 3 and 4 together.

**Excluded:**

- **half-chain entropy** — an SVD of the reshaped state vector, called thousands of times per move. At `n = 16` that is a 256×256 SVD in the inner loop. Cost-disqualified unless you have profiled it.
- **separation_deficit** — **I do not know what this computes.** Apply the two tests below and decide. Do not include it merely because it exists.

#### The test that matters more than the list

A feature is dead weight if it **does not vary across the moves available at a node**. A constant contributes equally to every sibling and cannot influence selection.

Two of the six need checking:

- **`coherence`** — if every move costs the same coherence, all leaves at fixed depth are identical on this axis and the feature contributes nothing. It survives *only because* `measure` and generator-application have different costs (Part VI §2.2). Confirm that asymmetry is implemented.
- **`dim_g`** — changes only on absorption, so it discriminates absorb from non-absorb moves and nothing else. Still valuable, since absorption is the loaded decision, but know that is its entire contribution.

**Before freezing, measure the sibling variance of each feature across a sample of positions.** Cut anything near zero. Twenty minutes, and it is the difference between a considered freeze and a guessed one.

#### Freeze the computation, not the name

If `temperature`'s implementation changes in Phase 13, the trace is invalidated exactly as surely as adding a seventh feature. Pin:

- the feature **set**, **order** and **vector length** — weights are positional
- the feature **computation**, via a golden test: each feature's output pinned on a fixed position set, hashed in the spec header
- `language = "v1"` in every file, checked at parse

**Profile `temperature` before committing.** Decisions 01 Q6 established it recomputes per position, not per frame — but MCTS at budget 4096 expands thousands of nodes per move. If it must be computed per node expansion rather than looked up from a cached per-position field, it may not fit. If it does not, v1 is the other five and you say so in the docs.

### Q2 — illustrative. Your lean is correct.

`bishop-Z2`, `knight-hop` and `rook-phase` were illustrative names implying variants that do not exist. **v1 uses the four implemented pieces verbatim: Pawn `X_q`, Rook `Z_q`, Bishop `X_qX_p`, Knight `X_qZ_p`.** Part IX §8's trap stands.

**Keep the field, because subset choice is strategically real.** Four pieces give fifteen non-empty subsets, and the subset interacts with the Lie closure — Part VII §2 notes that `[G_rook, G_bishop]` generates directions in neither, so a rook-plus-bishop agent has a strictly larger algebra than the union suggests. That is what makes `generators` a decision rather than a constant.

### Q3 — out of v1. Your lean is correct, for the reason you give.

Shipping a field the engine ignores is disqualified by the format's own premise: a notation whose entire value is honesty cannot contain a decorative entry, because the first one teaches readers that fields may be fiction.

You have also identified a genuine spec problem worth recording. **Orbit's two-independent-`StateVec`s-with-a-merge-rule is not a symmetrised two-particle state**, and Part VI §1 assumes it is. Wiring statistics in properly changes rule 6 of Part VII §5 — architecture, not a field.

Your M28 finding narrows it further: fermionic blocks one coin state, not a corridor. **Add that to Part VI §1 as a correction**, because the spec currently describes the class system as stronger than the implemented physics delivers.

*(Minor note from Q7: the edge-state encoding `|u,v⟩` that the coined core now needs makes a genuinely symmetrised two-particle state more architecturally reachable in v2 than it was. Not a reason to rush it, but the path is shorter than it looked.)*

### Q4 — L2-normalise, but your analysis missed a second invariance

Your lean is right and your reasoning is right, with one gap underneath it.

**Scale is not behaviourally neutral under MCTS.** Under greedy argmax, `w` and `2w` select identically. But UCT compares an exploitation term against `c·√(ln N / n)`, so scaling the eval output shifts the exploration–exploitation balance. Under `kind = "mcts"`, doubling the weights produces a **different agent**.

L2-normalising alone would therefore be a behaviour-changing transform disguised as canonicalisation — the worst kind, because it is invisible in review and only shows up as an unexplained rating shift.

**Ruling, both halves, in order:**

1. **Normalise the eval output at point of use** into UCT's value range. Standard MCTS practice, and it makes weight scale genuinely irrelevant.
2. **Then L2-normalise weights at parse**, as pure canonicalisation. Reject all-zero and non-finite.

Without (1), (2) is unsound.

For league dedup, round the normalised vector to fixed precision for the **submission-uniqueness check only**, never for execution. That rounding is a presentation constant under Decisions 01 Q4's test — it changes whether a PR is accepted, never a match result — so it lives in `presentation.toml` and is covered by the perturbation test.

---

## Group B — the `d` re-measurement

### Q5 — adopt your work unit. It improves on mine.

**1 work unit = one complex-coefficient update in the active representation.**

Better than the proportional cost model I wrote, for a reason that is the project's own constitution: **a counter is a measurement; a coefficient table is an assumption.** Mine required me to assert relative costs; yours reads them off the machine.

```
statevector   single-qubit gate on n qubits  →  2ⁿ updates
g-sim         Givens rotation                →  count anticommuting coefficients
MPS           contraction cost
```

The obvious objection — that this makes units engine-dependent, so a g-sim agent gets more gates per unit — is not an objection. g-sim genuinely *is* cheaper, so that agent genuinely does get more search for the same real cost, exactly as a chess engine with a faster evaluation gets more nodes per second. The methodology survey was explicit that no universal cross-agent currency exists and you must defend your abstraction. This one is the actual work, which is the strongest defence available.

**Implementation:** do not increment inside the innermost loop. Compute the update count analytically per gate from gate type and `n`. Exact, and free.

**Retire `d = 6` explicitly.** Your phrasing — *"a quietly-superseded number is worse than a withdrawn one"* — is right. State the reason in the docs and **keep the measurement as a documented historical artifact**. A visible retirement with an explanation demonstrates the methodology tightening; a deletion looks like hiding. Re-measure under frozen v1, work units, and the Goodman grid, and report it as **Skill Trace**, not `d`.

### Q6 — your ordering is right; the constraint is looser than you assumed

Measure single-game cost, pick `N` to fit, report what that buys — correct ordering, and far better than choosing `N` to hit a pretty number.

But **the grid is embarrassingly parallel across pairings, and Actions gives 20 concurrent jobs.**

```
6 budget levels  →  C(6,2) = 15 pairings
one job per pairing  →  15 concurrent, inside the 20 limit
6 h per job, 1,600 games  →  13.5 s per game budget
```

Far more achievable than fitting 24,000 games into one 6-hour job.

Two further levers before accepting wider bars:

- **Pairings are not equal cost.** A `(1, 128)` pairing has one cheap side; the total is well under 15× the most expensive.
- **SPRT per pairing with a cap.** Distant pairs have large effects and stop early — which is where the grid's statistical advantage converts into saved compute.

**Expect heterogeneous precision and report it.** Distant pairings tight, adjacent ones wide. That is correct, not a flaw. Publish as an error-bar plot rather than collapsing to one number, so a reader can see which comparisons are resolved.

---

## Q8 — the salt. Confirmed, with two additions.

Your reasoning holds, and retroactive auditability is the deciding argument: publishing the salt after a round closes makes the whole competition history verifiable by anyone.

**Derive per-round salts from a master secret.** `salt_round = hash(master_secret, round)`, publishing `salt_round` after round `N` closes. Then disclosing one round tells an attacker nothing about future rounds, whereas a single fixed salt would compromise everything at once. The never-disclosed audit layer needs a second master secret, never published.

**The thing that will actually bite you: fork PRs do not receive secrets by default.** That is correct security behaviour — a submitter must not be able to extract the salt — but it means scoring cannot run in the fork's PR context. Use `pull_request_target` or a manual approval gate, with the scoring job executing against the base repo. Get this right before the first external submission; it is the standard failure point.

**And note what the declarative decision buys here.** In a code-submission competition, exfiltration through submitted code is live and Actions log-masking is defeatable. Part VIII §2 means no submitted code executes, so the risk is structurally absent rather than mitigated.

---

## Q9 — reversed. Take Option B to counsel, not pure DCO.

**I am not a lawyer.** What follows is the factual landscape and a judgment call, and my earlier ruling was wrong on two points of fact and overstated on a third.

### Three corrections to what I told you

**1. "DCO establishes provenance only" is too narrow.** DCO 1.1 is a contributor *warranty* about authority to submit under the project's stated license, plus a public record. It gives an explicit representation of authority, an evidentiary record tying contribution to person, and a statement that the contribution enters under the project's existing licensing framework. My operational conclusion — it does not enable relicensing — was right; the characterisation was not.

**2. CLA does not imply assignment.** There are three models, not two:

| Model | Contributor keeps copyright | Future relicensing | Friction |
|---|---|---|---|
| DCO / inbound=outbound | Yes | Usually no | Very low |
| **CLA, broad non-exclusive licence** | **Yes** | **Potentially yes** | **Low–medium** |
| Copyright assignment | No | Yes | Highest |

The Apache ICLA is a **broad non-exclusive licence, not an assignment** — contributors retain ownership. So does Google's. So does Project Harmony's licence form. I framed it as a binary and it is a spectrum.

**3. "A CLA will cost you exactly the contributors you want" was overstated.** The evidence is qualitative and mixed. Node.js removed its CLA in 2014 explicitly to lower the barrier to entry; Cesium reported that large organisations took months to sign but individual contributions were unaffected. A 2017 study of 200 widely used OSS projects found 67% used no additional condition, 19% individual CLAs, 15% corporate — prevalence, not causation. **No controlled evidence quantifies a CLA's effect on scientific-software contribution.** I stated a causal claim the literature does not support.

### The example that should decide this for you

**cBioPortal is living your risk right now.** AGPL-3.0 cancer-genomics software, no prior CLA, currently running a contributor-consent process across **209 contributors** to move to Apache-2.0, with a consent deadline in September 2026. Their own project notes say: *no prior CLA exists — must get consent from all copyright holders*, and unreachable code may have to be rewritten or removed.

That is scientific software, AGPL, distributed authorship, and the exact cost of Option A, happening this month.

**And mpv is the worked outcome.** Their GPL→LGPL effort establishes that it is not all-or-nothing: silence did not count as consent, and where permission could not be obtained the project removed, replaced, or segregated the affected code. Some GPL-only components remain to this day. So "we can never relicense" is too strong — the accurate framing is *"we must obtain permission from the copyright holders of the material we want to relicense, or remove and rewrite it."* The problem is not impossibility. It is the cost of legal archaeology across a large contributor graph.

**One more comparator worth knowing:** Qhronology, a quantum-computing package, uses AGPL **plus a CLA**, openly, because its strategy requires distributing contributions under non-AGPL terms. The closest analogue to your project in the wild chose flexibility over minimum friction.

### Revised recommendation

**Option B: DCO 1.1 for provenance, plus a short non-exclusive contributor licence in which the contributor retains copyright.**

The asymmetry decides it. Option B costs one signing step now. Option A costs what cBioPortal is paying — and that cost grows with every merged PR and is unrecoverable once a contributor becomes unreachable. Adopting a CLA before the first external contribution is cheap; adopting it after twenty contributors is impossible.

I previously argued that pure DCO reads as a trust signal. I still think that is true, but it is a small effect and it is bought with a large, irreversible option. A non-assignment CLA does not take ownership from anyone, and saying so plainly in `CONTRIBUTING.md` largely neutralises the reputational concern.

**The drafting point that matters most:** "broad licence" is not sufficient. A clause saying *"you grant us a licence to distribute your contribution"* may not do what you need. The grant must expressly cover **sublicensing, derivative works, and use under licences other than the current outbound licence.** Have counsel check that specifically.

**The efficient question for a lawyer:**

> *"Can we use a short Apache/Harmony-style non-exclusive contributor licence, retaining contributor ownership, whose grant expressly covers future relicensing and dual-licensing, while using DCO 1.1 as the provenance and authority attestation?"*

That frames the decision around the capability you need rather than the DCO-versus-CLA label, and it is a fifteen-minute conversation rather than an open-ended one. Project Harmony's licence form is the cleanest template to bring.

---

## Q10 — yours to execute, and here is the bar

**Yes. And your hesitation is correct — the headline as I drafted it overclaims.**

"Whether your quantum circuit will train" is broader than what `predict` covers, because Decisions 02 §3 established that `Var[∂C] ∝ 1/dim(g)` holds only when `ρ ∈ g` or `O ∈ g`, and Diaz et al. exhibit poly-`dim(g)` circuits that plateau anyway. An unconditioned headline is exactly the overreach the rest of Decisions 02 exists to avoid.

**Scoped version:**

> **Overtone reads your circuit's algebra and predicts whether it will train — before you train it.**
>
> The prediction holds when the theory's hypotheses hold. Overtone checks them and tells you when they don't. It also measures whether your trainable circuit is one a classical computer could already simulate — because for most known constructions, it is.

The second sentence is load-bearing. **A tool that says "I cannot predict this one, and here is why" is more trustworthy than one that always answers**, and `predict` already refuses false claims, so the behaviour exists. Put it on the front page; it reads as a feature because it is one.

**The bar, and it is checkable:**

> **Every clause in the README must map to a shipped instrument.**

If a phrase does not correspond to a panel or a CLI output, cut the phrase. Apply the same rule to the dismissals table — right column is a panel name, not an argument. If a row has no panel, drop the row. Six of seven have one, and six is plenty.

---

## §11 — Spec corrections from Q7

Collected so they do not get lost.

| Spec | Correction |
|---|---|
| **Part II §P6** | Distinguish one-shot hitting, concurrent hitting, and marked-vertex search. The section currently applies Szegedy's `√HT` search result to entrance-to-exit traversal; those are different problems. |
| **Part II §P7** | The learned position-conditioned coin is **outside** the Grover-plus-flip-flop equivalence class, so it inherits **no** automatic quadratic speedup. Any advantage over the Grover coin is an empirical finding needing its own justification. |
| **Part II §P5** | The (4n+2) column reduction is not a rendering choice — it is the literal computation. Update the section to say so; it makes the panel cheaper and more honest at once. |
| **Part VI §1** | Fermionic statistics block one coin state, not a corridor. The spec overstates what the implemented physics delivers. |
| **Part VI-A §T2** | **Krovi & Brun exhibit infinite hitting times from destructive interference on the hypercube.** That is the purest possible instance of the AB-cage trap — a configuration where the walker provably never arrives, from interference alone. Add as a note under T2 rather than a ninth trap, per §8's standing rule. |
| **Phase 5 scope** | Szegedy module deleted. Hypercube (Kempe) added as the first milestone. |

---

## §12 — What I still do not know

- **What `separation_deficit` computes.** Yours. Apply Q1's two tests.
- **Whether `temperature` profiles cheaply enough for MCTS inner-loop use.** Must be measured.
- **Your single-game wall-clock cost.** Determines everything in Q6.
- **Whether the Li–Li–Luo result holds at the small `n` you will actually demo.** The theorem is asymptotic. Test empirically across `n` and report the smallest `n` at which the bound holds in your implementation — that is a useful contribution in itself.
- **Whether the cubelike-graph result (arXiv:2609.04503) survives review.** September 2026 preprint, no independent confirmation. Fine as test-suite material; do not build a claim on it.
- **Legal specifics of the Q9 grant language.** Genuinely a lawyer question, and now a fifteen-minute one.

---

## §13 — References added

- Li, Li & Luo — *Recovering the Original Simplicity: Succinct and Exact Quantum Algorithm for the Welded Tree Problem*, SODA 2024 pp. 2454–2480; Algorithmica 86(12), 3719–3758 (2024); arXiv:2304.08395.
- Jeffery & Zur — multidimensional quantum walks, STOC 2023. The predecessor.
- Ide, Konno, Segawa & Xu — *Localization of discrete time quantum walks on the glued trees*, arXiv:1312.1149.
- Kempe — *Discrete quantum walks hit exponentially faster*, Probab. Theory Relat. Fields (2005).
- Krovi & Brun — *Hitting time for quantum walks on the hypercube*, arXiv:quant-ph/0510136. Infinite hitting times from interference.
- Portugal — *Establishing the equivalence between Szegedy's and coined quantum walks using the staggered model*, arXiv:1509.08852.
- Wong — *Direct equivalence of coined and Szegedy's quantum walks* (2016).
- Portugal & Segawa — *Connecting coined quantum walks with Szegedy's model*, Interdiscip. Inf. Sci. 23(1) (2017).
- Mulherkar — *One-shot and concurrent hitting times for Grover-coined quantum walks on cubelike graphs*, arXiv:2609.04503 (2026). Unconfirmed.
- Developer Certificate of Origin 1.1 — developercertificate.org.
- Apache ICLA and ASF contributor agreements — apache.org/licenses/contributor-agreements.html.
- Project Harmony contributor agreement templates — licence form, not assignment form.
- cBioPortal relicensing project — github.com/cBioPortal/relicensing. The live worked example.
- mpv LGPL relicensing — github.com/mpv-player/mpv/issues/2033.
