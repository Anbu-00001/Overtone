---
title: Overtone
emoji: 🎵
colorFrom: indigo
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: agpl-3.0
short_description: Reads your circuit's algebra and predicts whether it will train.
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

A variational quantum circuit that encodes classical data is, exactly and provably, a truncated
Fourier series in that data (Schuld, Sweke & Meyer, *Phys. Rev. A* **103**, 032430). Overtone
points that theorem at a reinforcement-learning policy in real time.

Everything on this page runs in your browser. There is no server: the whole engine is Rust
compiled to WebAssembly, and the JavaScript is a renderer that computes no physical quantity of
its own. Source and the full build specification are at
[github.com/Anbu-00001/Overtone](https://github.com/Anbu-00001/Overtone).

Licensed AGPL-3.0-or-later.
