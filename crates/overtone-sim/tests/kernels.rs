//! Differential tests of the strided gate kernels against the dense Kronecker reference.
//!
//! These guard the kernels themselves: qubit ordering, the pair-loop indexing, the control
//! mask test, and matrix orientation. The reference in `overtone_sim::reference` shares no
//! code with the kernels, so a disagreement is a real bug rather than two copies of one.

use overtone_sim::complex::is_unitary;
use overtone_sim::gate::{h_matrix, rx_matrix, ry_matrix, rz_matrix};
use overtone_sim::random::{random_circuit, random_params, RandomCircuitOpts};
use overtone_sim::reference;
use overtone_sim::{Angle, Circuit, Observable, Pauli, PauliTerm, StateVec};

const TOL: f64 = 1e-12;

#[test]
fn rotation_matrices_are_unitary() {
    for t in [0.0, 0.3, 1.0, -2.2, std::f64::consts::PI, 7.5] {
        assert!(is_unitary(&rx_matrix(t), TOL), "RX({t}) not unitary");
        assert!(is_unitary(&ry_matrix(t), TOL), "RY({t}) not unitary");
        assert!(is_unitary(&rz_matrix(t), TOL), "RZ({t}) not unitary");
    }
    assert!(is_unitary(&h_matrix(), TOL));
}

#[test]
fn single_qubit_analytics() {
    // <Z_0> after RY(t)|0> is cos(t); after RX(t)|0> it is also cos(t).
    for t in [0.0, 0.4, 1.3, -0.9, 2.8] {
        let mut c = Circuit::new(1);
        c.ry(0, Angle::param(0));
        let v = Observable::z(0).expectation(&c.run(&[t]));
        assert!((v - t.cos()).abs() < TOL, "RY: {v} vs {}", t.cos());

        let mut c = Circuit::new(1);
        c.rx(0, Angle::param(0));
        let v = Observable::z(0).expectation(&c.run(&[t]));
        assert!((v - t.cos()).abs() < TOL, "RX: {v} vs {}", t.cos());

        // RZ leaves |0> in the Z eigenbasis, so <Z> stays 1 regardless of the angle.
        let mut c = Circuit::new(1);
        c.rz(0, Angle::param(0));
        let v = Observable::z(0).expectation(&c.run(&[t]));
        assert!((v - 1.0).abs() < TOL, "RZ: {v}");
    }
}

#[test]
fn hadamard_puts_x_at_one() {
    let mut c = Circuit::new(1);
    c.h(0);
    let st = c.run(&[]);
    let x = Observable::new(vec![PauliTerm::new(1.0, vec![(0, Pauli::X)])]);
    assert!((x.expectation(&st) - 1.0).abs() < TOL);
    assert!(Observable::z(0).expectation(&st).abs() < TOL);
}

#[test]
fn bell_state_correlations() {
    // H(0) then CNOT(0 -> 1) gives (|00> + |11>)/sqrt(2).
    let mut c = Circuit::new(2);
    c.h(0);
    c.cnot(0, 1);
    let st = c.run(&[]);

    let amp = |i: usize| st.amp(i);
    let s = std::f64::consts::FRAC_1_SQRT_2;
    assert!((amp(0).re - s).abs() < TOL, "|00> amplitude");
    assert!(amp(1).norm_sqr() < TOL, "|01> should be empty");
    assert!(amp(2).norm_sqr() < TOL, "|10> should be empty");
    assert!((amp(3).re - s).abs() < TOL, "|11> amplitude");

    // Perfectly correlated in Z, and each marginal is unbiased.
    let zz = Observable::new(vec![PauliTerm::new(
        1.0,
        vec![(0, Pauli::Z), (1, Pauli::Z)],
    )]);
    assert!((zz.expectation(&st) - 1.0).abs() < TOL);
    assert!(Observable::z(0).expectation(&st).abs() < TOL);
    assert!(Observable::z(1).expectation(&st).abs() < TOL);
}

#[test]
fn cz_is_symmetric_and_diagonal() {
    // CZ(a, b) and CZ(b, a) must produce the same state from a uniform superposition.
    let mut c1 = Circuit::new(3);
    c1.h(0);
    c1.h(1);
    c1.h(2);
    c1.cz(0, 2);

    let mut c2 = Circuit::new(3);
    c2.h(0);
    c2.h(1);
    c2.h(2);
    c2.cz(2, 0);

    assert!(c1.run(&[]).max_diff(&c2.run(&[])) < TOL);
}

#[test]
fn kernels_match_dense_reference() {
    // Every gate type, several qubit counts and depths, twenty seeds each.
    for n in 1..=5 {
        for depth in 1..=3 {
            for seed in 0..20u64 {
                let opts = RandomCircuitOpts {
                    num_qubits: n,
                    depth,
                    entangle: n > 1,
                    scaled_angles: true,
                    share_params: true,
                };
                let c = random_circuit(seed, opts);
                let p = random_params(seed ^ 0xabcd, c.num_params());

                let fast = c.run(&p);
                let dense = reference::run_as_state(&c, &p);

                let diff = fast.max_diff(&dense);
                assert!(
                    diff < TOL,
                    "n={n} depth={depth} seed={seed}: state differs by {diff:e}"
                );
            }
        }
    }
}

#[test]
fn expectations_match_dense_reference() {
    for n in 1..=5 {
        for seed in 0..15u64 {
            let opts = RandomCircuitOpts {
                num_qubits: n,
                depth: 3,
                entangle: n > 1,
                scaled_angles: true,
                share_params: true,
            };
            let c = random_circuit(seed, opts);
            let p = random_params(seed ^ 0x1234, c.num_params());

            let observables = [
                Observable::z(0),
                Observable::z_global(n),
                Observable::new(vec![
                    PauliTerm::new(0.5, vec![(0, Pauli::X)]),
                    PauliTerm::new(-1.5, vec![(n - 1, Pauli::Y)]),
                ]),
            ];

            let fast_state = c.run(&p);
            let dense_state = reference::run(&c, &p);

            for obs in &observables {
                let a = obs.expectation(&fast_state);
                let b = reference::expectation(n, obs, &dense_state);
                assert!(
                    (a - b).abs() < TOL,
                    "n={n} seed={seed}: expectation {a} vs reference {b}"
                );
            }
        }
    }
}

#[test]
fn norm_is_preserved() {
    for n in 1..=6 {
        for seed in 0..10u64 {
            let opts = RandomCircuitOpts {
                num_qubits: n,
                depth: 4,
                entangle: n > 1,
                scaled_angles: true,
                share_params: false,
            };
            let c = random_circuit(seed, opts);
            let p = random_params(seed, c.num_params());
            let norm = c.run(&p).norm_sqr();
            assert!(
                (norm - 1.0).abs() < TOL,
                "n={n} seed={seed}: norm^2 drifted to {norm}"
            );
        }
    }
}

#[test]
fn adjoint_gate_application_inverts() {
    // Applying a gate then its adjoint must return the original state exactly.
    for seed in 0..10u64 {
        let opts = RandomCircuitOpts {
            num_qubits: 4,
            depth: 3,
            ..Default::default()
        };
        let c = random_circuit(seed, opts);
        let p = random_params(seed, c.num_params());

        let start = StateVec::zero(4);
        let mut st = start.clone();
        for gate in c.gates() {
            gate.apply(&mut st, &p);
        }
        for gate in c.gates().iter().rev() {
            gate.apply_adjoint(&mut st, &p);
        }
        assert!(
            st.max_diff(&start) < TOL,
            "seed={seed}: round trip drifted by {:e}",
            st.max_diff(&start)
        );
    }
}
