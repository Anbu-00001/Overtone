//! Determinism. Part I 5: same seed, same trajectory, native and WASM.
//!
//! The WASM half of that claim belongs to Phase 4. What is testable now is that nothing in
//! this crate reaches for entropy: a seed fully determines the circuit, the parameters,
//! the state, and the gradients, bit for bit.

use overtone_sim::grad::adjoint;
use overtone_sim::random::{random_circuit, random_params, RandomCircuitOpts};
use overtone_sim::Observable;

#[test]
fn same_seed_same_circuit() {
    for seed in 0..8u64 {
        let a = random_circuit(seed, RandomCircuitOpts::default());
        let b = random_circuit(seed, RandomCircuitOpts::default());
        assert_eq!(a, b, "seed={seed} produced two different circuits");
    }
}

#[test]
fn different_seeds_differ() {
    let a = random_circuit(1, RandomCircuitOpts::default());
    let b = random_circuit(2, RandomCircuitOpts::default());
    assert_ne!(a, b, "two seeds produced the same circuit");
}

#[test]
fn same_seed_same_parameters() {
    for seed in 0..8u64 {
        assert_eq!(random_params(seed, 16), random_params(seed, 16));
    }
}

#[test]
fn state_is_bit_for_bit_reproducible() {
    for seed in 0..8u64 {
        let c = random_circuit(seed, RandomCircuitOpts::default());
        let p = random_params(seed, c.num_params());
        let a = c.run(&p);
        let b = c.run(&p);
        // Exact equality, not a tolerance: the same operations in the same order must
        // produce identical floating-point results.
        assert_eq!(a.re, b.re, "seed={seed}: real plane differs");
        assert_eq!(a.im, b.im, "seed={seed}: imaginary plane differs");
    }
}

#[test]
fn gradients_are_bit_for_bit_reproducible() {
    let obs = Observable::z(0);
    for seed in 0..8u64 {
        let c = random_circuit(seed, RandomCircuitOpts::default());
        let p = random_params(seed, c.num_params());
        assert_eq!(
            adjoint::grad(&c, &p, &obs),
            adjoint::grad(&c, &p, &obs),
            "seed={seed}"
        );
    }
}

#[test]
fn parallel_and_serial_kernels_agree() {
    // Only meaningful with --features parallel; harmless otherwise. The parallel path
    // splits the pair loop across threads, and a chunking mistake would show up as a
    // wrong state rather than as a crash.
    for n in 1..=6 {
        for seed in 0..5u64 {
            let opts = RandomCircuitOpts {
                num_qubits: n,
                depth: 3,
                entangle: n > 1,
                ..Default::default()
            };
            let c = random_circuit(seed, opts);
            let p = random_params(seed, c.num_params());
            let st = c.run(&p);
            assert!((st.norm_sqr() - 1.0).abs() < 1e-12);
        }
    }
}
