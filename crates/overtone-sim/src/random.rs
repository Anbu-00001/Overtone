//! Seeded random circuits, for differential testing and for the plateau sweeps of
//! Part I 6.7.
//!
//! Everything here takes an explicit seed and uses ChaCha8. Part I 5: same seed, same
//! trajectory, native and WASM. There is no thread-local or entropy-seeded RNG anywhere
//! in this crate.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::circuit::Circuit;
use crate::gate::{Angle, Gate};

/// A deterministic RNG from a `u64` seed.
pub fn rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

/// A uniform parameter vector over `[-pi, pi)`.
pub fn random_params(seed: u64, count: usize) -> Vec<f64> {
    let mut r = rng(seed);
    (0..count)
        .map(|_| r.gen_range(-std::f64::consts::PI..std::f64::consts::PI))
        .collect()
}

/// Options for [`random_circuit`].
#[derive(Clone, Copy, Debug)]
pub struct RandomCircuitOpts {
    pub num_qubits: usize,
    pub depth: usize,
    /// Include CZ and CNOT.
    pub entangle: bool,
    /// Give some rotation gates a non-unit chain-rule factor, as an encoding gate has.
    /// This exercises the chain rule in both gradient paths rather than leaving it at 1.
    pub scaled_angles: bool,
    /// Let two gates share one parameter index, which both gradient paths must accumulate
    /// across rather than overwrite.
    pub share_params: bool,
}

impl Default for RandomCircuitOpts {
    fn default() -> Self {
        RandomCircuitOpts {
            num_qubits: 4,
            depth: 3,
            entangle: true,
            scaled_angles: true,
            share_params: true,
        }
    }
}

/// A random circuit exercising every gate type.
///
/// Returns the circuit; its parameter count is `circuit.num_params()`.
pub fn random_circuit(seed: u64, opts: RandomCircuitOpts) -> Circuit {
    let n = opts.num_qubits;
    let mut r = rng(seed);
    let mut c = Circuit::new(n);
    let mut next_param = 0usize;

    for layer in 0..opts.depth {
        for q in 0..n {
            // A rotation whose axis varies, so all three derivative matrices get used.
            let scale = if opts.scaled_angles && r.gen_bool(0.5) {
                r.gen_range(-2.0..2.0)
            } else {
                1.0
            };

            // Occasionally point a later gate at an earlier parameter.
            let index = if opts.share_params && next_param > 0 && r.gen_bool(0.25) {
                r.gen_range(0..next_param)
            } else {
                let i = next_param;
                next_param += 1;
                i
            };

            let angle = Angle::scaled(index, scale);
            match r.gen_range(0..3) {
                0 => c.rx(q, angle),
                1 => c.ry(q, angle),
                _ => c.rz(q, angle),
            };

            if r.gen_bool(0.3) {
                c.h(q);
            }
        }

        if opts.entangle && n > 1 {
            for q in 0..n {
                let a = q;
                let b = (q + 1) % n;
                if a == b {
                    continue;
                }
                if (layer + q) % 2 == 0 {
                    c.cz(a, b);
                } else {
                    c.cnot(a, b);
                }
            }
        }
    }

    // A fixed angle, so the "no gradient from a fixed gate" path is covered too.
    c.push(Gate::Rz {
        q: 0,
        angle: Angle::Fixed(0.371),
    });

    c
}
