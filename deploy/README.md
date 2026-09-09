---
title: Overtone
emoji: 🎵
colorFrom: indigo
colorTo: gray
sdk: static
app_file: index.html
header: mini
pinned: false
license: agpl-3.0
short_description: Reads your circuit's algebra and predicts if it will train.
tags:
  - quantum-machine-learning
  - reinforcement-learning
  - webassembly
  - rust
  - visualization
---

# Overtone

**Overtone reads your circuit's algebra and predicts whether it will train — before you train
it.**

The prediction holds when the theory's hypotheses hold. Overtone checks them and tells you when
they do not. It also measures whether your trainable circuit is one a classical computer could
already simulate — because for most known constructions, it is.

## The theorem this is pointed at

A variational quantum circuit that encodes classical data is, exactly and provably, a truncated
Fourier series in that data:

$$ f_{\theta}(x) \;=\; \sum_{\omega \in \Omega} c_{\omega}(\theta)\, e^{i \omega x} $$

The reachable frequencies $\Omega$ are fixed by the eigenvalues of the data-encoding gates; the
coefficients $c_\omega$ are set by everything else in the circuit. That is Schuld, Sweke and
Meyer, *Phys. Rev. A* **103**, 032430 (2021). Overtone points it at a reinforcement-learning
policy in real time, and you watch a single spectral peak slide up the frequency axis and lock
onto the environment.

## What the page is doing

Everything here runs in your browser. There is no server, no API and no inference endpoint: the
whole engine is Rust compiled to WebAssembly, and the JavaScript is a renderer that computes no
physical quantity of its own — 1186 lines against a budget of 1200, enforced in CI.

That is not a concession to the free tier. Two checks in the repository make the frontend *be*
the engine: `wasm_determinism.sh` requires native and WASM to agree to `1e-13`, and
`check_js_budget.sh` forbids a panel from computing a physical quantity. A backend would have
nothing left to do.

| | |
|---|---|
| Bundle | 539 KiB total, 358 KiB of it WebAssembly |
| Sections | `Lab` · `Closure` · `Menagerie` · `Lattice` |
| Reproducibility | every run is seeded; no pre-recorded traces, no cached results |
| Source | [github.com/Anbu-00001/Overtone](https://github.com/Anbu-00001/Overtone) |

The architecture diagrams, the full build specification (Parts I–XI) and the measurements —
including the negative results — are in the repository README, which renders Mermaid; this card
does not, so the diagrams are not duplicated here.

## Licence

AGPL-3.0-or-later. Section 13 is the clause that does the work: running a modified Overtone as a
network service counts as conveying it, so a hosted fork owes its source to the people using it.
This is an instrument — its value is that its measurements can be checked.
