# Overtone

Watch a quantum agent tune itself into resonance with its environment.

A variational quantum circuit that encodes classical data is, exactly and provably, a
truncated Fourier series in that data. The frequencies it can reach are fixed by the
eigenvalues of the data-encoding gates; the coefficients are set by everything else in the
circuit. That is Schuld, Sweke and Meyer, *Phys. Rev. A* **103**, 032430 (2021).

Overtone points that theorem at a reinforcement-learning policy in real time.

**Status: Phase 1 of 8.** The simulator and both gradient paths are built and verified.
Nothing renders yet. See [docs/PHASES.md](docs/PHASES.md) for the plan and
[docs/spec/](docs/spec/) for the full build specification.

---

## What is verified today

| Claim | Measured | Tolerance | Test |
|---|---|---|---|
| Adjoint and parameter-shift gradients agree | `1.4e-15` | `1e-10` | `crates/overtone-sim/tests/gradients.rs` |
| Gradients and expectation values match Yao.jl | `2.2e-15` | `1e-12` | `lab/test/oracle.jl` |
| Strided kernels match a dense Kronecker reference | `< 1e-12` | `1e-12` | `crates/overtone-sim/tests/kernels.rs` |
| Parallel kernels match serial ones | bit-for-bit | exact | `crates/overtone-sim/tests/parallel.rs` |
| Seeded runs reproduce | bit-for-bit | exact | `crates/overtone-sim/tests/determinism.rs` |
| Norm preserved under every unitary | `< 1e-12` | `1e-12` | `crates/overtone-sim/tests/kernels.rs` |

Every number in that table is produced by the test suite, not by hand. The measured column
is the worst case observed across the full sweep, not a typical value.

## Why three gradient checks and not one

The adjoint path is what training will use: constant memory in circuit depth, every
parameter in roughly two forward passes. It is also the one where a sign error hides
comfortably, because it is only ever compared against itself.

So it is checked three ways. The **parameter-shift rule** is exact rather than a finite
difference and is what real hardware would evaluate, so it shares no implementation with
the adjoint sweep. A **central finite difference** is a third opinion: if the two analytic
paths ever agree with each other but disagree with it, the fault is in an assumption they
share, such as the generator convention, rather than in either implementation. And
**Yao.jl** checks the whole engine from outside — different language, different author,
different algorithm.

The dense Kronecker reference in `overtone_sim::reference` plays the same role one level
down. It builds the full `2^n x 2^n` matrix of every gate, which is exactly what the fast
path exists to avoid; that is what makes it useful. An indexing mistake in the pair loop or
a transposed matrix shows up as a disagreement rather than as two copies of one bug.

## Conventions

- `RX(t) = exp(-i t X / 2)`, and likewise RY and RZ.
- Qubit `q` is bit `q` of the basis-state index, little-endian.
- Amplitudes are stored structure-of-arrays, two `f64` planes. SoA autovectorises;
  array-of-structs does not.
- Every random draw is seeded. `overtone-sim` has no entropy dependency at all — dropping
  `rand`'s default features removes `getrandom` from the tree, which enforces the
  determinism requirement at the dependency level and lets the crate build for
  `wasm32-unknown-unknown` with no shim.

## Building

```
cargo test --workspace                          # 24 tests
cargo test -p overtone-sim --features parallel  # 26, adding the threaded kernels
cargo clippy --workspace --all-targets -- -D warnings
cargo build -p overtone-sim --target wasm32-unknown-unknown
```

The Yao.jl oracle needs Julia:

```
cargo run -q -p overtone-sim --example emit_oracle_cases > lab/test/cases.txt
julia --project=lab -e 'using Pkg; Pkg.instantiate()'
julia --project=lab lab/test/oracle.jl
```

## Layout

```
crates/overtone-sim/    state vector, gates, adjoint and parameter-shift gradients
crates/overtone-rl/     environments, policies, REINFORCE          (Phase 2)
crates/overtone-spec/   Fourier, entropy, gradient variance        (Phase 3)
crates/overtone-cli/    native trainer, JSONL traces               (Phase 2)
crates/overtone-wasm/   wasm-bindgen surface                       (Phase 4)
lab/                    Yao.jl oracle and heavy sweeps
docs/spec/              the build specification, Parts I to V
```

The dependency direction is one-way and load-bearing. `overtone-sim` knows nothing about
reinforcement learning; `overtone-rl` knows nothing about rendering; `overtone-wasm` is a
shim, and if it ever contains an `if` statement about physics, that logic is in the wrong
crate.

## Licence

Apache-2.0.
