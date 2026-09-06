# OVERTONE PART III — Closure

**Companion to `overtone-build-spec.md` (Part I) and `overtone-part-ii-lattice.md` (Part II).** Neither is replaced. This is the layer that makes the other two into one project, and it is the reason the repo will be taken seriously.

**Tab name:** `Closure`, third beside `Lab` and `Lattice`.

---

## 1. The new thesis

Parts I and II *show* what a quantum policy does. That is a visualiser. Good visualisers get looked at once.

Part III makes Overtone **predict what a circuit can learn, and whether it will train, before a single gradient step** — from a purely algebraic computation on the circuit's generators — and then trains the thing to show the prediction landing.

> **A quantum policy's capabilities are fixed before training by the algebra of its generators. Overtone computes that algebra, tells you what it implies, and then verifies it empirically.**

That is a thesis, not a demo. It is also a *tool*: `overtone predict circuit.qasm` returns a report on a circuit you are about to spend a week training. Tools get used; demos get bookmarked.

---

## 2. The prediction chain

Take the circuit's gate generators `{iG₁, …, i G_K}`. Close them under nested commutators until the set stops growing. That is the **dynamical Lie algebra** `g`. Its dimension is one integer, and that integer predicts three separate things that are otherwise only discoverable by training:

```
generators ──[ Lie closure ]──> dim(g)
                                   │
     ┌─────────────────────────────┼──────────────────────────────┐
     ▼                             ▼                              ▼
Var[∂C] ∝ 1/dim(g)          M_c ≤ dim(g)              rank(QFIM) ≤ dim(g)
"will it train?"        "how many parameters       "how many directions
                         before the landscape       in state space can
                         stops having spurious      it actually move?"
                         local minima?"
```

- **Barren plateaus.** Ragone, Bakalov, Sauvage, Kemper, Ortiz Marrero, Larocca & Cerezo, *A Lie algebraic theory of barren plateaus for deep parameterized quantum circuits*, Nat. Commun. 15 (2024). The variance of the loss gradient falls inversely with `dim(g)`. Exponentially large algebra ⟹ barren plateau; polynomial algebra ⟹ trainable. Companion paper: Fontana et al., Nat. Commun. 15 (2024).
- **Overparameterization.** Larocca, Ju, García-Martín, Coles & Cerezo, *Theory of overparametrization in quantum neural networks*, Nat. Comput. Sci. 3, 542 (2023). There is a critical parameter count `M_c ≤ dim(g)` marking a genuine **computational phase transition**: below it the landscape carries spurious local minima, above it they disappear.
- **Effective dimension.** Same paper: `dim(g)` upper-bounds the rank the quantum Fisher information matrix and the Hessian can reach. Most VQCs are overparameterized in a precisely measurable way — the QFIM has a large, visible null space.

**Part I is the abelian shadow of this.** The reachable Fourier frequencies are set by the eigenvalues of the *encoding* generators; the DLA is the closure of *all* generators under commutation. Part I asks which functions the circuit can represent. Part III asks which unitaries it can reach. Same question, two levels of generality, and now the project has a single spine instead of three modules.

---

## 3. Computing the closure is cheap, and that is the whole opportunity

The Lie closure looks expensive. For Pauli generators it is not, and this is the key implementation insight.

Represent a Pauli string on `n` qubits as two bitsets `(x, z) ∈ F₂ⁿ × F₂ⁿ` plus a phase. Then:

- Two Paulis **anticommute** iff the symplectic form `⟨x₁,z₂⟩ + ⟨x₂,z₁⟩ = 1 (mod 2)` — a couple of AND-and-popcount word operations.
- If they commute, the commutator is zero.
- If they anticommute, `[P₁,P₂] ∝ P₁P₂`, and the product is another Pauli string, obtained by XOR-ing the bitsets.

So the closure is: repeatedly take pairs, test one bit, XOR. **No `2ⁿ` matrices anywhere.** Memory is `dim(g) × 2n` bits. You can compute the DLA of a 200-qubit circuit on a laptop, provided `dim(g)` stays polynomial — which is exactly the regime you care about.

And it gets better. The structure constants are maximally sparse: `ad_G(P)` is zero if they commute, and otherwise a single Pauli `Q`. Since `ad_G(Q) ∝ P` again, `ad_G` restricted to `span{P, Q}` is a 2×2 rotation generator. Therefore:

> **`exp(θ·ad_G)` acting on the coefficient vector is a set of independent 2D Givens rotations.**

Each gate costs `O(dim g)`, not `O(dim(g)²)`. That makes the Lie-algebraic simulator in §4 almost embarrassingly simple to implement correctly, and very fast.

---

## 4. Three engines, and the 12 GB budget

The Hugging Face free CPU Space changes the architecture in a way that is genuinely good for the project. The browser keeps the interactive path; the Space runs the things the browser cannot.

### Engine A — state vector (Part I, unchanged)

`complex128` needs `16 · 2ⁿ` bytes. Adjoint gradients need two vectors live at once.

| qubits | one vector | with gradients (×2) |
|---|---|---|
| 24 | 268 MB | 537 MB |
| 26 | 1.07 GB | 2.15 GB |
| 28 | 4.29 GB | 8.59 GB |
| 29 | 8.59 GB | — over budget |

**Ceiling on a 12 GB Space: 28 qubits with gradients, 29 without.** In `complex64`, one more each. State the ceiling in the README with the arithmetic shown; people check.

### Engine B — matrix product state

Memory `≈ n · χ² · 16` bytes for complex128. At `n = 100`, `χ = 512`: about 840 MB. `χ = 1024` fits comfortably. Good to hundreds of qubits **when entanglement is low**, and the bond dimension `χ` is a physically meaningful dial rather than a hyperparameter (see §5).

### Engine C — g-sim (the new one)

Goh, Larocca, Cincio, Cerezo & Sauvage, *Lie-algebraic classical simulations for quantum computing*, Phys. Rev. Research 7, 033266 (2025). Instead of evolving a `2ⁿ` state, evolve a `dim(g)`-dimensional vector of expectation values in the DLA basis. When `dim(g) ∈ poly(n)`, this is a **scalable exact simulator**, and the authors report simulating 200 qubits in minutes on a single CPU core.

Memory: `dim(g)` doubles for the coefficient vector, plus sparse structure constants. `dim(g)` up to ~10⁴ is comfortable inside 12 GB. Gradients are also efficient in this representation, so you can *train* here — which means:

> **A 100-qubit quantum RL policy, trained exactly, on a free Hugging Face CPU Space.**

That is the number that goes in the README's first line. It is true, it is checkable, and it will make people click.

### The caveat that must be on the same page as the claim

`g-sim` is efficient exactly when `dim(g)` is polynomial — and a polynomial DLA is exactly the condition for *no barren plateau*. So the circuits that train well are, by the same algebraic fact, the circuits that admit efficient classical simulation.

This tension is the live question in the field, and the repo should name it rather than route around it. Put it in the README, in plain words, near the 100-qubit claim. A project that volunteers the strongest objection to its own headline is a project people trust — and trust is the thing that converts a visit into a star and a star into a citation.

---

## 5. The Dequantization Test

This falls out of having three engines, and it is the most useful thing in Part III.

Take a trained quantum policy. Run it through:

- **Engine B at increasing `χ`.** At what bond dimension does the MPS reproduce the exact policy to within tolerance, and at what `χ` does the achieved return saturate?
- **Engine C**, if the DLA is polynomial. Does it reproduce the policy exactly?

If a `χ = 4` tensor network reproduces your quantum agent, **your quantum agent is a small tensor network.** No advantage, no ambiguity, one number.

Ship this as `overtone dequantize run.json`, and as a UI dial. The output is a single integer — the effective bond dimension of the learned policy — and a verdict sentence. No other QML repository ships an instrument that tries this hard to disprove its own premise, and that is precisely why this one will be shared.

Run it on the Part I `SpectralControl` agents and the Part II learned coins, and publish the numbers whatever they are.

---

## 6. What the Closure tab shows

Five panels. Two of them are the most visually striking things in the entire project.

### C1 — The closure, animated *(build this first)*

Start with the generator set: a handful of Pauli strings, each drawn as a strip of `n` glyphs coloured by `I / X / Y / Z`. Compute commutators. New strings appear and connect by edges. The set grows — fast, alarmingly fast — and then **stops**.

A counter ticks `dim(g)` upward and freezes. That frozen number is the prediction.

A combinatorial explosion resolving into a finite structure is inherently dramatic, and as far as we can find **nobody has ever animated a Lie closure**. Layout: force-directed for small `dim(g)`, switching to a sorted lattice above a few hundred elements so it stays legible. Show the growth curve `|S|` vs iteration in an inset; the plateau *is* the closure.

Two presets, side by side, that make the point in five seconds: a hardware-efficient ansatz whose algebra explodes toward `4ⁿ − 1`, and a symmetry-restricted ansatz whose algebra closes at `O(n²)`. Same qubit count. One will train; one will not. You can see which before either has been trained.

### C2 — The prediction landing

Draw the predicted `1/dim(g)` line **before** training. Then run the empirical gradient-variance measurement from Part I §6.7 and let the measured points appear on it.

The whole rhetorical weight of the module is in the ordering: prediction first, measurement second. Never render them simultaneously.

### C3 — The phase transition *(the second striking one)*

Sweep the parameter count `M`. At each `M`, launch many randomly initialised training runs and plot the **distribution of final losses** as a histogram.

Below `M_c`: broad, multimodal, agents stuck in spurious minima.
Above `M_c`: the histogram **collapses to a spike** at the optimum.

A computational phase transition rendered as a histogram collapsing under a slider. Mark `dim(g)` on the axis as the predicted upper bound for `M_c` and let the reader watch the collapse happen at or before it.

### C4 — QFIM spectrum and the natural gradient

Sorted eigenvalues of the quantum Fisher information matrix, log scale. The rank saturates; the null space appears as a cliff. Add layers and watch the rank climb toward `dim(g)` and stop.

Then the payoff: two optimisation trajectories on the same landscape — vanilla gradient descent and **quantum natural gradient** (Stokes, Izaac, Killoran & Carleo, *Quantum Natural Gradient*, Quantum 4, 269 (2020)), which preconditions by the QFIM. The natural-gradient path takes the geometrically correct route and gets there in a fraction of the steps. Same loss surface, two metrics, visibly different journeys.

This also earns its place practically: natural gradient is a genuine improvement for VQC training, and offering it as a working optimiser makes the repo useful rather than merely instructive.

### C5 — The dequantization dial

§5 as an interface: `χ` slider, fidelity-to-exact curve, return curve, and the verdict integer.

---

## 7. Hugging Face architecture

Keep the browser demo primary. The Space is a second, optional surface — the demo must never require it, because a sleeping Space would take the whole project down with it.

```
web (GitHub Pages, WASM)          ← interactive, ≤16 qubits, always works
        │  optional fetch
        ▼
HF Space (FastAPI, 2 vCPU, 12 GB) ← the heavy work
        ├── /predict     POST circuit → { dim_g, predicted_var, M_c,
        │                                 reachable_frequencies, verdict }
        ├── /dequantize  POST policy  → { effective_chi, verdict }
        ├── /simulate    g-sim / MPS runs the browser can't host
        └── nightly job  → sweeps → the Atlas dataset
```

Practical notes for the build:

- **Assume the Space is asleep.** Free Spaces idle out. The UI degrades gracefully: show the cached Atlas result, note it is cached, and offer to wake the Space. Never a spinner that hangs.
- **Rust core, thin Python shell.** Compile `overtone-*` as a `pyo3` extension and let FastAPI be forty lines. Do not reimplement any physics in Python.
- **Cache aggressively.** `dim(g)` for a given generator set is deterministic — hash the generators, cache forever. Most `/predict` calls should never compute anything.
- **Hard caps on every endpoint,** returning a clear message rather than an OOM: qubits, `dim(g)`, `χ`, wall-clock. A Space that dies under a Hacker News front page is worse than no Space.
- **Pin the memory ceiling below the real one.** Budget to 9 GB and leave headroom; the container has other tenants.

---

## 8. The Atlas

A nightly sweep over circuit families, published as a **Hugging Face Dataset**:

```
ansatz · n · layers · encoding · generators · seed
    →  dim(g) · predicted Var · measured Var · M_c · rank(QFIM)
    ·  reachable frequency set · effective χ · final return · parameter count
```

This is the piece that changes the repo's category. A visualiser gets stars from people who looked at it. **A dataset gets cited**, and citations bring a durable, compounding kind of attention that a demo never does. It also gives the Space something useful to do while nobody is watching.

Ship it with a datasheet: how it was generated, what the seeds were, what is missing, and what it should not be used for.

---

## 9. New crates and milestones

```
crates/
├── overtone-lie/      Pauli bitset arithmetic; Lie closure; dim(g);
│                      structure constants; the prediction report
├── overtone-gsim/     Lie-algebraic simulation via Givens rotations;
│                      gradients in the DLA basis
├── overtone-mps/      matrix product state engine; the dequantization test
└── overtone-serve/    pyo3 bindings + FastAPI shell for the Space
```

**M10 — the closure engine.** Pauli bitsets, Lie closure, `dim(g)`. Acceptance: reproduces published DLA dimensions for standard ansätze (hardware-efficient, transverse-field Ising, Heisenberg) — these are tabulated in the literature and make excellent unit tests.

**M11 — prediction.** Wire `dim(g)` to predicted gradient variance and `M_c`. Acceptance: **reproduce Figure 2 of Ragone et al. (2024)** from our own engine. Put the badge in the README.

**M12 — the Closure tab.** C1 and C2. Acceptance: the closure animation is legible up to `dim(g) = 500`, and the prediction is rendered before the measurement, always.

**M13 — g-sim.** Givens-rotation evolution, gradients, and a trained 100-qubit policy. Acceptance: matches Engine A exactly on small circuits where both apply; the 100-qubit training run completes inside the Space's limits.

**M14 — MPS and dequantization.** C3, C4, C5. The Atlas job. Acceptance: effective-`χ` numbers published for every agent in the repo, favourable or not.

---

## 10. Claims and tests

| Claim | Test |
|---|---|
| `dim(g)` matches published values for standard ansätze | unit tests against literature tables |
| `Var[∂C] ∝ 1/dim(g)` | fitted exponent; reproduces Ragone et al. Fig. 2 |
| Spurious minima vanish at `M ≈ M_c ≤ dim(g)` | the C3 histogram collapse, measured not asserted |
| `rank(QFIM) ≤ dim(g)` | direct rank computation, several ansätze |
| g-sim agrees with the state vector where both apply | `1e-12`, randomised circuits |
| Natural gradient converges in fewer steps than vanilla | benchmark; report the ratio |
| Effective `χ` of every published agent | table in the README, whatever the numbers say |
| Space endpoints stay inside 9 GB under load | soak test in CI |

Reproducing figures from *Nature Communications* and *Nature Computational Science* papers, with the badge to prove it, is the cheapest credibility available to this project. Do it early.

---

## 11. On getting recognised, honestly

You asked about GitHub stars. The mechanics that actually work, in order:

1. **A live demo, zero install.** Already the plan. Everything else is downstream.
2. **A true claim in the first line that sounds impossible.** *"Train a 100-qubit quantum RL policy on a free CPU."* True, checkable, and the check is one click.
3. **Reproduce published figures with a badge.** Converts "someone's side project" into "someone who knows the literature."
4. **One screenshot worth sharing.** C1 or C3. Make one of them genuinely beautiful and put it at the top of the README as a still, linked to the live version.
5. **A tool, not just a demo.** `overtone predict circuit.qasm` is something a person with a real VQC uses on a Tuesday. Usage is stickier than admiration.
6. **A citable dataset.** The Atlas. Different audience, longer half-life.
7. **Publish the negative results.** The dequantization numbers, the entanglement ablation, the parameter-count comparison. Counterintuitively this is the strongest signal of all — a repo that tries to falsify itself reads as a research artifact rather than a pitch.
8. **Post once, factually.** Title the submission with the most surprising *true* sentence, not with adjectives. Let the demo do the arguing.

What does not work: feature lists, roadmap emoji, "awesome" framing, or claiming quantum advantage. This field has a credibility deficit and the fastest way to inherit it is to sound like everyone who caused it.

---

## 12. Minimum viable Overtone

This is now three parts and eleven crates, which is a lot. If the scope has to be cut — and it probably should be — this is the version that keeps the thesis intact:

**Ship:** Part I §6.5 (the spectral instrument), §7.1 (`SpectralControl-k` with the LP ceiling), Part III C1 and C2 (closure animation and the prediction landing), and the browser demo. Nothing else.

That is one environment, two instruments, one animation, one live page. It still says the whole thing: *the algebra of the generators determines what the policy can learn and whether it will train, and here is the proof, live.* Everything in Part II and the rest of Part III is elaboration on a thesis that this core already establishes.

Build that. Ship it. Then decide what the response tells you to build next.

---

## 13. Traps

- **Do not reverse the ordering in C2.** Prediction before measurement, every time. It is the entire rhetorical structure of the module.
- **Do not claim the 100-qubit result without the caveat beside it.** Polynomial DLA means classically simulable. Say it on the same screen.
- **Do not let the Space become a dependency.** Free Spaces sleep. If the demo needs the backend, the demo is broken half the time.
- **Do not implement physics in Python.** The shell is a shell.
- **Do not compute closures with dense matrices.** Bitsets and XOR. A `2ⁿ` matrix anywhere in `overtone-lie` is a bug.
- **Do not skip the dequantization test because the answer might be unflattering.** The unflattering answer is the contribution.
- **Watch scope.** Three parts is already ambitious for one repository. §12 exists for a reason; re-read it whenever a new panel suggests itself.

---

## 14. References

- Ragone, Bakalov, Sauvage, Kemper, Ortiz Marrero, Larocca, Cerezo — *A Lie algebraic theory of barren plateaus for deep parameterized quantum circuits*, Nat. Commun. 15 (2024), arXiv:2309.09342.
- Fontana et al. — *Characterizing barren plateaus in quantum ansätze with the adjoint representation*, Nat. Commun. 15 (2024). Companion result.
- Larocca, Ju, García-Martín, Coles, Cerezo — *Theory of overparametrization in quantum neural networks*, Nat. Comput. Sci. 3, 542 (2023).
- Goh, Larocca, Cincio, Cerezo, Sauvage — *Lie-algebraic classical simulations for quantum computing*, Phys. Rev. Research 7, 033266 (2025), arXiv:2308.01432. The g-sim framework.
- Stokes, Izaac, Killoran, Carleo — *Quantum Natural Gradient*, Quantum 4, 269 (2020).
- Wiersema et al. — *Classification of dynamical Lie algebras for translation-invariant 2-local spin systems in one dimension*, arXiv:2309.05690. Tabulated DLA dimensions — use these as unit tests.
- Diaz, García-Martín, Kazi, Larocca, Cerezo — *Showcasing a barren plateau theory beyond the dynamical Lie algebra*, arXiv:2310.11505. Where the `1/dim(g)` picture stops holding; read before overstating it.
