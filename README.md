# Overtone

Watch a quantum agent tune itself into resonance with its environment.

A variational quantum circuit that encodes classical data is, exactly and provably, a
truncated Fourier series in that data. The frequencies it can reach are fixed by the
eigenvalues of the data-encoding gates; the coefficients are set by everything else in the
circuit. That is Schuld, Sweke and Meyer, *Phys. Rev. A* **103**, 032430 (2021).

Overtone points that theorem at a reinforcement-learning policy in real time.

**Status: Phase 2 of 8.** The simulator, both gradient paths, the RL loop and the LP ceiling
are built and verified. Nothing renders yet. See [docs/PHASES.md](docs/PHASES.md) for the
plan and [docs/spec/](docs/spec/) for the full build specification.

---

## The result, in one table

`SpectralControl-k` is a contextual bandit with reward `r(s, a) = (2a - 1) cos(k s)`. Its
expected return is `J = E_s[cos(ks) (2 pi(1|s) - 1)]`, which extracts precisely the
frequency-`k` Fourier coefficient of the policy and ignores everything else. So a policy
that cannot reach frequency `k` cannot score. At all.

Measured, at `k = 3`, 5000 episodes, seeded:

| Policy | Encoding ceiling | LP ceiling | Achieved `J` |
|---|---|---|---|
| RAW-PQC, `lambda` pinned | 2 | 0 | **0.000000** |
| RAW-PQC, `lambda` pinned | 3 | 0.5 | 0.499656 |
| RAW-PQC, `lambda` **trainable** | 1 | 0 | **0.500940** |
| SOFTMAX-PQC, `lambda` pinned | 2 | 0 | **0.110075** |

Row one is the prediction that makes this repo worth reading: not "worse", not "near
chance" — exactly zero, for every parameter setting, forever. Rows three and four are the
two different ways out of that box, and they are not the same escape.

Reproduce all four:

```
cargo run --release -p overtone-cli -- train --k 3 --layers 2 --policy raw
cargo run --release -p overtone-cli -- train --k 3 --layers 1 --scaling trainable --coarse-tune
cargo run --release -p overtone-cli -- ceiling --k 3 --max-c 12
```

Each takes under a quarter of a second.

## What is verified today

| Claim | Measured | Tolerance | Test |
|---|---|---|---|
| Adjoint and parameter-shift gradients agree | `1.4e-15` | `1e-10` | `overtone-sim/tests/gradients.rs` |
| Gradients and expectation values match Yao.jl | `2.2e-15` | `1e-12` | `lab/test/oracle.jl` |
| Strided kernels match a dense Kronecker reference | `< 1e-12` | `1e-12` | `overtone-sim/tests/kernels.rs` |
| Parallel kernels match serial ones | bit-for-bit | exact | `overtone-sim/tests/parallel.rs` |
| Seeded runs reproduce | bit-for-bit | exact | `overtone-sim/tests/determinism.rs` |
| RAW-PQC below the ceiling scores zero | `0.0` exactly | `1e-12` | `overtone-rl/tests/spectral_ceiling.rs` |
| Trained agents land on the LP staircase | within `0.0004` | `0.05` | `overtone-rl/tests/spectral_ceiling.rs` |
| Policy gradients match a finite difference | `< 1e-5` | `1e-5` | `overtone-rl/tests/policy_gradient.rs` |
| REINFORCE is unbiased for the policy gradient | within 8% | 8% | `overtone-rl/tests/policy_gradient.rs` |
| Reduced LP ceiling matches the unreduced one | `< 5e-5` | `5e-5` | `overtone-rl/src/ceiling.rs` |

48 tests. Every number in the measured column is produced by the suite, and is the worst
case across the full sweep rather than a typical value.

## Three things the specification did not say, that turned out to matter

**The frequency ceiling counts encoding gates, not layers.** Part I states the reachable
spectrum as `{-L..L}` for `L` layers. That holds only when each layer applies one encoding
rotation per observation component. Encode the same scalar on all `n` qubits every layer and
the ceiling is `n*L`. This is not pedantry: the headline test is RAW-PQC at `L = 2` scoring
zero against `k = 3`, and on a two-qubit register encoding everywhere the ceiling would be
`4 > 3`, the agent would score, and the repo's central claim would appear to fail. The
ceiling is computed from the encoding map, and there is a test pinning both cases.

**The `lambda` landscape is a resonance curve with a capture range, not a hill.** Part I
says trainable input scaling makes the reachable frequencies continuous, so the peak slides
until it locks onto the environment. True — but only from within the capture range.
Measured at `L = 1` against `k = 3`:

```
main peak       lambda = 3.05   J = 0.502
sidelobes       lambda = 0.66   J = 0.021 · 5.47 -> 0.083 · 7.48 -> 0.051
capture range   lambda_0 in roughly [1.8, 4.2] reaches the main peak
```

Gradient ascent started at `lambda = 1` — the obvious default, since it matches the pinned
circuit — falls outside that range and converges to a sidelobe at `J = 0.021`. So the
demonstration needs a coarse tune before the fine tune, exactly as a radio does. That is a
property of resonance rather than a defect, and `--coarse-tune` is the coarse tune.
`examples/lambda_scan.rs` reproduces the numbers.

**The softmax leakage shows up as return, not only as spectrum.** Part I predicts that a
SOFTMAX-PQC leaks past its own circuit's frequency ceiling, because a sigmoid of a
band-limited function is not band-limited. On this environment that leakage is worth reward:
at `L = 2` against `k = 3`, where a RAW-PQC scores exactly zero, the softmax policy scores
`0.110` by generating a frequency-3 harmonic out of its frequency-1 and 2 content. It stays
below the unconstrained optimum `2/pi`, so the leakage is bounded, not magical.

The consequence is that **the LP staircase bounds RAW-PQC only.** It is not a bound on
softmax policies, and drawing it behind a softmax learning curve would misrepresent both.

A fourth, recorded for Phase 3: the leakage lives in `pi(a|s)`, not in the logit. Part I 6.5
says to transform "the policy's logit function", but for a SOFTMAX-PQC the logit is
`beta * w * <Z>`, which is exactly as band-limited as the RAW-PQC's. The spectral instrument
has to transform the probability, or it will show no leakage and quietly falsify a true
claim.

## Why three gradient checks and not one

The adjoint path is what training uses: constant memory in circuit depth, every parameter in
roughly two forward passes. It is also where a sign error hides comfortably, because it is
only ever compared against itself.

So it is checked three ways. The **parameter-shift rule** is exact rather than a finite
difference and is what hardware would evaluate, sharing no implementation with the adjoint
sweep. A **central finite difference** is a third opinion: if the two analytic paths agree
with each other but disagree with it, the fault is in an assumption they share, such as the
generator convention. And **Yao.jl** checks the whole engine from outside — different
language, different author, different algorithm.

The same discipline runs one level up. `REINFORCE` is checked against the analytic policy
gradient it is supposed to estimate, and the reduced LP ceiling against the unreduced
formulation it was derived from.

## The LP ceiling

Given a frequency ceiling `C`, the best expected return any *strictly band-limited* policy
can reach on `SpectralControl-k` is the answer to: maximise the frequency-`k` Fourier
coefficient of a degree-`C` trig polynomial whose sup norm is at most 1. That is a linear
program.

Averaging over the `k` shifts `s -> s + 2 pi j / k` preserves the sup-norm bound and the
frequency-`k` component while annihilating every frequency that is not a multiple of `k`;
taking the even part then kills the sines. So the problem reduces from `2C+1` variables to
`floor(C/k)+1`, and `J*` depends on `C` only through `floor(C/k)` — which is *why* the
ceiling is a staircase, with steps at multiples of `k`.

There is no usable closed form. The one-sided Carathéodory–Fejér bound `2 cos(pi/(C+2))`
does not apply, since `|p| <= 1` is two-sided and strictly stronger; it returns 0.707 where
the true ceiling is 0.5, and passes straight through the `2/pi` supremum. Part I's
instruction to solve it numerically is well taken.

```
   C           J*        (k = 3)
   0-2   0.000000
   3-8   0.500000
   9-11  0.577350
   inf   0.636620   (2/pi, unconstrained optimum)
```

## Conventions

- `RX(t) = exp(-i t X / 2)`, and likewise RY and RZ.
- Qubit `q` is bit `q` of the basis-state index, little-endian.
- Amplitudes are stored structure-of-arrays, two `f64` planes. SoA autovectorises;
  array-of-structs does not.
- Every random draw is seeded. `overtone-sim` has no entropy dependency at all — dropping
  `rand`'s default features removes `getrandom` from the tree, which enforces the
  determinism requirement at the dependency level and lets the crate build for
  `wasm32-unknown-unknown` with no shim.
- The LP solver is `good_lp` on the pure-Rust `microlp` backend, so `overtone-rl` also
  builds for `wasm32`. The ceiling is drawn behind the learning curve in the browser, so it
  cannot be a native-only convenience.

## Building

```
cargo test --workspace                          # 48 tests
cargo test -p overtone-sim --features parallel  # plus the threaded kernels
cargo clippy --workspace --all-targets -- -D warnings
cargo build -p overtone-sim --target wasm32-unknown-unknown
cargo build -p overtone-rl  --target wasm32-unknown-unknown
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
crates/overtone-rl/     ansatz, policies, SpectralControl-k, REINFORCE, LP ceiling
crates/overtone-spec/   Fourier, entropy, gradient variance        (Phase 3)
crates/overtone-cli/    native trainer, JSONL traces
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
