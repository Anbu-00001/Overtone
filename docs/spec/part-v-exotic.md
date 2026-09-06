# OVERTONE PART V — Exotic

**Reinforcement learning algorithms beyond REINFORCE and PPO, selected for one reason only.** Fifth companion to Parts I–IV.

---

## 0. The filter

There are hundreds of exotic RL algorithms and most of them would be decoration here. One test decides:

> **Does the algorithm rhyme with quantum mechanics at the level of the operator, or only at the level of the vocabulary?**

Wave Function Collapse rhymes at the level of vocabulary — which is why Part IV uses it as a *joke with a lesson* rather than as physics. The algorithms below rhyme at the level of the operator: the same matrix, the same eigenproblem, the same exponential. Those are worth building. Everything else is a feature list.

Section 1 is the important one. It contains a unification I did not expect to find, and it makes Parts I through IV into a single object.

---

## 1. Proto-value functions — the thing that unifies the whole project

### 1.1 The observation

Mahadevan's proto-value functions (ICML 2005; Mahadevan & Maggioni, JMLR 2007) are basis functions for RL learned **not from rewards but from the topology of the state space**. Formally, in his own words, they are *Fourier eigenfunctions of the Laplace–Beltrami diffusion operator on the state-space manifold* — in the discrete case, the eigenvectors of the graph Laplacian `L = D − A`.

Now put that next to Part II. A classical random walk on a maze evolves under `e^{−Lt}`. A continuous-time quantum walk evolves under `e^{−iAt}`. Same graph. Same matrix, up to the degree term. **The only difference between the classical walker and the quantum walker is whether the exponent is real or imaginary.**

And both are diagonal in the same basis. So:

> The eigenvectors of the maze's Laplacian are **simultaneously** the RL basis functions of proto-value function theory, the eigenmodes of the quantum walk, and the graph Fourier transform of the maze.

Three fields, one eigendecomposition.

### 1.2 Why this matters for the project specifically

Overtone's thesis has been spectral from the beginning, and this closes it:

| Part | What is being Fourier-decomposed |
|---|---|
| I | the **observation** — reachable frequencies of the encoding |
| II | the **lattice** — transport exponents, interference |
| III | the **algebra** — the DLA as the general case |
| V | the **state space itself** — proto-value functions |

The project is not "quantum RL with nice graphics." It is one claim — *spectral structure determines what an agent can learn* — applied at four levels. That is a thesis a researcher will repeat to a colleague, and repeated theses are how repositories spread.

### 1.3 What to build

**Panel: the shared eigenbasis.** *(goes in `Lattice`)*

Compute the low-order eigenvectors of the maze Laplacian. Draw mode `k` over the maze. Then one toggle:

```
   exponent:  [ real  e^{−Lt} ]  ←→  [ imaginary  e^{−iAt} ]
```

Real: the mode decays. Diffusion. Imaginary: the mode *oscillates* and the modes interfere. Same eigenvector, same maze, one character changed in the exponent, and diffusion becomes interference in front of the reader.

This is the single clearest explanation of quantum-versus-classical transport that the project can produce, and it costs one eigensolve.

**Eigenoptions.** *(Machado, Bellemare, Bowling et al., ICLR 2018)* Options — temporally extended actions — constructed from the principal eigenvectors of the Laplacian or the successor representation. On a maze they are "traverse the slow modes of diffusion," which are exactly the modes a quantum walk excites most strongly.

So: build eigenoptions, hand them to the agent as a macro-action set, and compare against the flat learned coin from Part II §P7. Prediction worth testing publicly: eigenoptions should help most on precisely the graphs where the quantum walk's advantage is largest, because they are built from the same spectrum. If they do not, that is also worth publishing.

**Successor representation.** Modern formulations take the top singular vectors of `(I − γPπ)⁻¹`. Same spectral object, learnable from experience rather than requiring the graph. Include it so the module works on the endless maze where you never have the full adjacency matrix.

---

## 2. Linearly-solvable MDPs — Bellman as an eigenproblem

### 2.1 The result

Todorov (NeurIPS 2006; PNAS 2009). Restrict the controller to reshaping the passive dynamics, and charge a KL-divergence cost for doing so. Then substitute the **desirability function** `z = exp(−v/λ)`.

The Bellman equation becomes **linear**.

Consequences, all from the original paper:

- **Shortest-path problems become largest-eigenvalue problems**, solvable in `O(n)`.
- **Z-learning**, the off-policy algorithm that follows, does not need state-action values and outperforms Q-learning on the problems tested.
- **Compositionality**: because the equation is linear, the optimal policy for a composite task is a **linear combination of the desirability functions of previously solved tasks**.

That last one deserves a second reading. In a project about superposition, here is an RL formalism in which **optimal policies genuinely superpose** — linearly, exactly, non-metaphorically.

### 2.2 The quantum bridge is published, not invented

The connection is not something we would be asserting. arXiv:2012.07063 (*Ground States of Quantum Many Body Lattice Models via Reinforcement Learning*) reinterprets the stochastic formulation of the Schrödinger equation directly as a Todorov linearly-solvable MDP. The bridge exists in the literature; Overtone would be the first place you can *watch* it.

On the endless maze this means: the optimal controller and the quantum walker are governed by the same operator. The walker asks about the whole spectrum; the controller asks only for the extremal eigenvector. **One matrix, two questions.**

### 2.3 What to build

**Panel: solve by eigenvector.** *(goes in `Lattice`)*

Give the maze a goal. Instead of training, solve the largest-eigenvalue problem and render the desirability function `z` as a field over the maze. It appears **instantly** — no learning curve, no episodes. Then run the Part II learned-coin agent beside it and watch the agent's policy converge toward a field that was already sitting there.

The contrast between "trained for ten thousand episodes" and "solved in one eigensolve" is a genuinely provocative thing to put on a screen, and it is honest: LMDPs buy that speed with a restricted control formulation, and the panel should say so.

**Panel: policy superposition.** Solve task A (reach the north exit) and task B (reach the east exit). Then take `α·z_A + β·z_B` and watch a correct policy for a *third* task appear without any training at all. Add a two-slider control for `α, β`.

Nothing else in the project produces a comparable "wait, what?" and it is a thirty-year-old result that almost no one has animated.

---

## 3. RL for architecture search, with an algebraic reward

### 3.1 Closing the loop

Parts I–III use quantum circuits to do RL. Invert it: use RL to design the quantum circuit. Quantum architecture search is an established line of work — an agent adds, deletes, and rearranges gates to maximise some objective.

The reason it is usually impractical is cost: every reward requires training the candidate circuit, so a search over thousands of architectures means thousands of training runs.

**Part III removes that cost.** `dim(g)`, the predicted gradient variance, the reachable frequency set, and the QFIM rank bound are all computable from the generators in **milliseconds**, with no training at all (Part III §3 — Pauli bitsets and XOR).

So the search agent is rewarded by algebra:

```
reward = w₁·(is dim(g) polynomial?)          will it train
       + w₂·(does the reach cover the task?)  can it represent the answer
       − w₃·(gate count)                      is it implementable
```

An architecture search with an **algebraic** reward rather than an empirical one. Millions of candidate evaluations become affordable, and every candidate is scored by a quantity backed by a *Nature Communications* result rather than by a noisy training run.

Then verify: take the top candidates, train them properly, and check whether the algebraic reward predicted the empirical outcome. That verification step is the contribution, and it is publishable whichever way it lands.

### 3.2 Pairing

Run this **inside** Part IV's MAP-Elites archive rather than as a separate search. The descriptors are already algebraic quantities; the search agent becomes an emitter proposing new architectures into the Menagerie. The two modules were built for each other.

---

## 4. Distributional RL — because the Born rule already gave you a distribution

Standard RL learns the *expected* return. Distributional RL (C51, QR-DQN) learns the whole return distribution.

The rhyme is exact and it is not a stretch: **a quantum measurement never returns an expectation value.** It returns samples. What Part I calls `⟨O⟩` is an average over shots, and on real hardware you get the distribution, not the mean. Distributional RL is simply the value-side formalism that matches what the policy side was already doing.

Two concrete payoffs:

- **Shot noise becomes first-class.** The variance from finite measurement is exactly the aleatoric uncertainty distributional RL is built to model, rather than something to be averaged away and apologised for.
- **A shot-budget dial.** Sweep shots per estimate from 10 to 10,000 and watch the return distribution sharpen. This is the most honest NISQ-realism knob the project can offer, and it makes every other result more credible by showing you know what it costs.

Cheap to add (quantile regression is a small change to the critic) and it slots into `Lab` without a new panel.

---

## 5. Evolution strategies, and a myth worth killing

### 5.1 The rhyme

Natural Evolution Strategies preconditions its update by the Fisher information of the *search distribution*. Quantum Natural Gradient (Part III §C4) preconditions by the Fisher information of the *quantum state*. Two Fisher metrics, one algorithmic idea, and the project already has the machinery for both. Worth a short docs section drawing the parallel.

### 5.2 The myth

There is a widespread belief in quantum machine learning that barren plateaus can be sidestepped by switching to a gradient-free optimiser — Nelder–Mead, Powell, CMA-ES. It is intuitive and it is wrong.

Arrasmith et al., *Effect of barren plateaus on gradient-free optimization* (Quantum, 2021): in a barren plateau the **cost function differences themselves are exponentially suppressed**, not merely the gradients. A gradient-free optimiser has nothing to compare either, so without exponential measurement precision it makes no progress. They confirm this numerically for Nelder–Mead, Powell, and others.

**Build the demonstration.** Same circuit, deep in a plateau, four optimisers racing: vanilla gradient, natural gradient, CMA-ES, Nelder–Mead. All four flatline. It takes an afternoon, it corrects a real and common misconception, and *a repository that debunks something* gets shared in a way that a repository that demonstrates something does not.

This also sharpens Part IV's MAP-Elites framing: quality-diversity helps by *restructuring the search problem* — by keeping stepping stones in behaviourally distinct cells — not by being gradient-free. Say that precisely, or a reader will assume we fell for the same myth.

---

## 6. Go-Explore — the right exploration algorithm for an endless maze

Return-then-explore: keep an archive of visited states, deliberately return to a promising one, then explore from there. It was designed for exactly the failure mode an endless sparse-reward maze produces, where undirected exploration detaches from the frontier and wanders.

No deep quantum rhyme — it earns its place on fit alone. It shares the archive machinery with the Menagerie, so the marginal cost is low, and it makes the Part II learned-coin results meaningfully stronger by removing "the agent never found the goal" as a confound.

---

## 7. Speculative — put these in `experiments/`, not in the claims

Flagged as unproven. Report outcomes either way.

**Von Neumann entropy as the max-entropy regulariser.** Soft actor-critic regularises by the Shannon entropy of the policy. A quantum policy has a native alternative: the von Neumann entropy of its own state. Substituting one for the other is a small code change with genuinely unclear consequences — they measure different things, and which one is the right regulariser is an open question rather than a known answer.

**Empowerment via quantum channel capacity.** Empowerment is the channel capacity from an agent's actions to its future states. The quantum analogue — Holevo information, quantum channel capacity — is a real, well-defined object. An intrinsic reward with no classical counterpart. Related to the QFI-curiosity idea in Part IV §5.3; build at most one of the two.

**Amplitude-amplification exploration.** Grover-style action selection, amplifying the amplitudes of good actions instead of ε-greedy sampling (Dong et al., 2008 — the older QRL lineage, distinct from the variational branch). Quadratic speedup, but only in an oracularised environment, which is a strong assumption. Include it as a clearly-labelled idealised comparison, never as a headline.

---

## 8. What I considered and rejected

Included because the rejections are as informative as the picks.

| Algorithm | Why not |
|---|---|
| MuZero / model-based | Large engineering cost, no operator-level rhyme. Would consume the project. |
| Decision Transformer | No rhyme, and the compute is far outside a free CPU budget. |
| Offline RL / CQL | Requires a dataset we do not have and would have to fabricate. |
| Active inference | Genuinely contested, and hard to falsify — which is exactly the wrong property for a repo whose credibility rests on falsifiable claims. |
| Curriculum learning | Useful, unremarkable. Include it silently in training if it helps; do not present it as a feature. |
| Multi-agent RL | Part IV's race is presentation, not MARL. Actual MARL adds a research problem without adding a quantum one. |

---

## 9. Priority

If you build three things from this document, build these:

**1 — the shared eigenbasis panel (§1.3).** One eigensolve, one toggle, and it explains quantum-versus-classical transport better than anything else in the project while retroactively unifying all four parts. Highest payoff per line of code in the entire series.

**2 — solve-by-eigenvector and policy superposition (§2.3).** A thirty-year-old result nobody has animated, and "policies superpose linearly" is the single most surprising true sentence available to this repository.

**3 — the optimiser flatline demo (§5.2).** An afternoon's work that corrects a widespread misconception, with a citation behind it.

Then the shot-budget dial (§4) whenever you want every other result to become more credible for very little effort.

**Build order:** M22 eigenbasis · M23 LMDP · M24 optimiser flatline · M25 distributional and the shot dial · M26 eigenoptions and Go-Explore · M27 architecture search into the Menagerie.

---

## 10. Traps

- **Do not present LMDPs as a free lunch.** The linearity is bought with a restricted control formulation and a KL cost. State the restriction on the same panel as the instant solution, or the demo is a magic trick.
- **Do not claim MAP-Elites escapes barren plateaus because it is gradient-free.** §5.2 shows that reasoning is wrong. It helps by restructuring the search, and the docs must say so precisely.
- **Do not let the eigenbasis panel become a spectrogram.** One mode at a time, one toggle, nothing else on screen. Its power is that it changes exactly one character in an equation.
- **Do not skip the verification step in §3.** An architecture search rewarded by a proxy is only interesting if someone checks whether the proxy was right.
- **Do not build more than one speculative item from §7.** Two unproven ideas in one repository read as unfocused; one clearly-labelled experiment reads as curiosity.
- **Do not let this part expand the scope again.** Part III §12's minimum viable cut still governs. §9's item 1 is small enough to add to that cut; the rest is not.

---

## 11. References

- Mahadevan — *Proto-value functions: developmental reinforcement learning*, ICML 2005; and Mahadevan & Maggioni, *Proto-value functions: a Laplacian framework for learning representation and control in MDPs*, JMLR 8 (2007).
- Machado, Bellemare, Bowling — *A Laplacian framework for option discovery in reinforcement learning*, ICML 2017, arXiv:1703.00956; and *Eigenoption discovery through the deep successor representation*, ICLR 2018, arXiv:1710.11089.
- Wu, Tucker, Nachum — *The Laplacian in RL: learning representations with efficient approximations*, ICLR 2019.
- Todorov — *Linearly-solvable Markov decision problems*, NeurIPS 2006; and *Efficient computation of optimal actions*, PNAS 106, 11478 (2009).
- Kappen — *Path integrals and symmetry breaking for optimal control theory*, J. Stat. Mech. (2005). The continuous-time relative.
- Barry, Barry, Aaronson (and, separately, arXiv:2012.07063) — *Ground states of quantum many-body lattice models via reinforcement learning*. The published Schrödinger-to-LMDP bridge.
- Arrasmith, Cerezo, Czarnik, Cincio, Coles — *Effect of barren plateaus on gradient-free optimization*, Quantum 5, 558 (2021). The myth, killed.
- Bellemare, Dabney, Munos — *A distributional perspective on reinforcement learning*, ICML 2017; Dabney et al., *Distributional RL with quantile regression*, AAAI 2018.
- Wierstra et al. — *Natural Evolution Strategies*, JMLR 15 (2014).
- Ecoffet, Huizinga, Lehman, Stanley, Clune — *Go-Explore: a new approach for hard-exploration problems*, Nature 590 (2021).
- Dong, Chen, Li, Tarn — *Quantum reinforcement learning*, IEEE Trans. SMC-B 38 (2008). Amplitude amplification for action selection.
