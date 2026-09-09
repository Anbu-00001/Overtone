# Your circuit has a highest note, and no amount of training reaches above it

There is a quantum machine-learning circuit that scores **exactly zero** on a task. Not nearly
zero. Not zero on average. Zero, for every setting of every parameter, forever.

That is not a bug, and it is not a hard optimisation problem. It is a theorem, and once you have
seen it you can predict it in advance for any circuit you like, in about a second, without
training anything.

## The task

`SpectralControl-3` is about as small as a reinforcement-learning problem gets. A number `s`
arrives. You choose one of two actions. Your reward is

```
r(s, a) = (2a − 1) · cos(3s)
```

Take action 1 where `cos(3s)` is positive and action 0 where it is negative, and you collect
reward. Get it backwards and you lose the same amount. That is the whole game.

The expected return works out to

```
J = E_s[ cos(3s) · (2·π(1|s) − 1) ]
```

which is worth staring at for a moment, because it is doing something specific. It is picking
out **precisely the frequency-3 Fourier coefficient** of your policy and ignoring everything
else. Whatever else your policy does — however cleverly it varies, however well it fits some
other part of the problem — only its frequency-3 component is paid.

## The circuit

A variational quantum circuit that encodes a classical number `s` is, exactly and provably, a
**truncated Fourier series in `s`**. This is Schuld, Sweke and Meyer, *Phys. Rev. A* **103**,
032430 (2021), and "exactly" is meant literally: not an approximation, not a useful analogy. The
circuit's output as a function of `s` is

```
f(s) = Σ_{ω ∈ Ω}  c_ω  e^{iωs}
```

and the two halves of that expression come from different parts of the machine:

- **The frequencies `Ω` are fixed by the eigenvalues of the data-encoding gates.** How many times
  you feed `s` into the circuit, and with what generators, decides which `ω` can appear at all.
- **The coefficients `c_ω` are set by everything else** — every trainable parameter, every
  entangling gate, the observable you measure.

Training moves the coefficients. **Training cannot move the frequency set.** The frequencies were
decided when you wrote the circuit down.

For the standard encoding, with `L` repetitions of the data-encoding layer, the reachable set is
every integer frequency from `−L` to `L`. So `L` is your circuit's highest note.

## The wall

Put those together. The task pays only frequency 3. A circuit with `L = 2` can produce
frequencies `−2, −1, 0, 1, 2`. Its frequency-3 coefficient is **structurally absent** — there is
no parameter setting in which it is small, because there is no parameter that could make it
anything at all.

So `J = 0`. Every time. And the number that says so is computable before you train:

```
$ cargo run --release -p overtone-cli -- ceiling --k 3 --max-c 6

# LP ceiling J*(C) for SpectralControl-3
# the best any strictly band-limited policy with frequency ceiling C can score
   C           J*
   0     0.000000
   1     0.000000
   2     0.000000
   3     0.500000
   4     0.500000
   5     0.500000
   6     0.500000
 inf     0.636620  (2/pi, unconstrained optimum)
```

`J*(C)` is the best score achievable by **any** policy whose frequency ceiling is `C` — the
solution of a linear program over all band-limited policies, not a training result. It is zero
at `C = 2` and it is `0.5` at `C = 3`. That is a wall, not a slope.

Train the circuit and you get what the ceiling promised:

| Policy | Encoding ceiling | LP ceiling | Achieved `J` |
|---|---|---|---|
| RAW-PQC, `λ` pinned | 2 | 0 | **0.000000** |
| RAW-PQC, `λ` pinned | 3 | 0.5 | 0.499656 |

The zero is exact to `1e-12`, and there is a test in this repository that fails if it ever stops
being exact.

## Where the minus signs are

It matters that this is interference and not ignorance, because the two look the same from
outside and behave completely differently.

A classical model that cannot represent frequency 3 gets it *wrong* — it produces some
approximation, badly. The quantum circuit does not approximate frequency 3 poorly. It produces
**nothing at all** at frequency 3, because the amplitudes that would carry it cancel. Positive
and negative amplitudes meet and sum to zero, everywhere, for every parameter setting.

That cancellation is the same phenomenon as a dark fringe in a two-slit experiment, and it is the
thing that separates quantum amplitudes from classical probabilities. Probabilities only ever add
up. Amplitudes can subtract, and a frequency that is absent from `Ω` is absent because they did.

## Two ways past it, and they are not the same

**Add layers.** More repetitions of the data-encoding gate, more reachable frequencies. `L = 3`
puts frequency 3 in range and the score jumps to `0.5`. This is the obvious fix and it costs
circuit depth.

**Make the input scaling trainable.** Let the circuit learn a scale factor `λ` on the input, and
the frequency comb stops being locked to the integers — it stretches. A circuit with `L = 1` and
trainable `λ` scores `0.500940` on a task it has no integer frequency for, because it *tunes* the
one frequency it has onto the one the task pays.

That second one is a genuinely different escape, and this repository has a page about it. It is
also the reason the project is called Overtone.

## What this does not say

The theorem is about which frequencies are *reachable*. It says nothing about whether your
optimiser will find good coefficients among the reachable ones — that is a separate problem with
separate failure modes, and the biggest of them (barren plateaus) has its own page.

And the frequency ceiling of a real circuit is not always the textbook `L`. Repeated encoding
gates, non-integer eigenvalues, and clever generators all move it. The point is not that the
answer is always `L`; it is that **there is always an answer, it is a property of the circuit
rather than of the training run, and you can compute it first.**

```
$ cargo run --release -p overtone-cli -- predict --qubits 4 --layers 2
```

## Reproduce everything above

```bash
git clone https://github.com/Anbu-00001/Overtone && cd Overtone
cargo run --release -p overtone-cli -- ceiling --k 3 --max-c 12
cargo run --release -p overtone-cli -- train --k 3 --layers 2 --policy raw
cargo run --release -p overtone-cli -- train --k 3 --layers 1 --scaling trainable --coarse-tune
```

## Sources

- Schuld, Sweke & Meyer — *Effect of data encoding on the expressive power of variational
  quantum machine learning models*, Phys. Rev. A **103**, 032430 (2021).
- The LP ceiling construction and its reduction: `crates/overtone-rl/src/ceiling.rs`.
- The exactness of the zero: `crates/overtone-rl/tests/spectral_ceiling.rs`.
