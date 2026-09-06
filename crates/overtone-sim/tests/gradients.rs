//! The Phase 1 gate: adjoint and parameter-shift gradients must agree to `1e-10`.
//!
//! Part I 6.4 and 9. Part I 14: "Nothing renders until the gradients are right."
//!
//! Three independent opinions are compared, not two. If adjoint and parameter-shift ever
//! agree with each other but disagree with the finite difference, the fault is in an
//! assumption they share -- the generator convention, or the chain rule -- rather than in
//! either implementation.

use overtone_sim::grad::{adjoint, shift};
use overtone_sim::random::{random_circuit, random_params, RandomCircuitOpts};
use overtone_sim::{Angle, Circuit, Observable, Pauli, PauliTerm};

/// The tolerance the README claims.
const GRAD_TOL: f64 = 1e-10;
/// A central difference at this step size is good to about 1e-8, no better.
const FD_TOL: f64 = 1e-7;
const FD_EPS: f64 = 1e-5;

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn analytic_single_qubit_gradient() {
    // <Z_0> after RY(t)|0> is cos(t), so the gradient is -sin(t).
    let mut c = Circuit::new(1);
    c.ry(0, Angle::param(0));
    let obs = Observable::z(0);

    for t in [0.0, 0.3, 1.1, -2.0, 3.0] {
        let p = [t];
        let vg = adjoint::value_and_grad(&c, &p, &obs);
        assert!((vg.value - t.cos()).abs() < 1e-12);
        assert!(
            (vg.grad[0] + t.sin()).abs() < 1e-12,
            "adjoint at t={t}: {} vs {}",
            vg.grad[0],
            -t.sin()
        );

        let s = shift::grad(&c, &p, &obs);
        assert!(
            (s[0] + t.sin()).abs() < 1e-12,
            "shift at t={t}: {} vs {}",
            s[0],
            -t.sin()
        );
    }
}

#[test]
fn chain_rule_on_a_scaled_encoding_angle() {
    // RX(lambda * s) is the data-re-uploading encoding gate. <Z_0> is cos(lambda * s), so
    // d/dlambda is -s sin(lambda * s). Getting the chain factor wrong is invisible when
    // s = 1, which is exactly why this test uses s != 1.
    let obs = Observable::z(0);
    for s in [0.0, 0.5, 1.0, -2.3, 3.7] {
        let mut c = Circuit::new(1);
        c.rx(0, Angle::scaled(0, s));
        for lambda in [0.4, -1.2, 2.5] {
            let p = [lambda];
            let expected = -s * (lambda * s).sin();

            let a = adjoint::grad(&c, &p, &obs);
            let sh = shift::grad(&c, &p, &obs);
            assert!(
                (a[0] - expected).abs() < 1e-12,
                "adjoint s={s} lambda={lambda}: {} vs {expected}",
                a[0]
            );
            assert!(
                (sh[0] - expected).abs() < 1e-12,
                "shift s={s} lambda={lambda}: {} vs {expected}",
                sh[0]
            );
        }
    }
}

#[test]
fn shared_parameter_gradients_accumulate() {
    // One parameter driving two gates: the total derivative is the sum of both
    // contributions. RY(t) on qubit 0 twice in a row is RY(2t), so <Z_0> = cos(2t) and
    // the gradient is -2 sin(2t).
    let mut c = Circuit::new(1);
    c.ry(0, Angle::param(0));
    c.ry(0, Angle::param(0));
    let obs = Observable::z(0);

    for t in [0.2_f64, 0.9, -1.4] {
        let p = [t];
        let expected = -2.0 * (2.0 * t).sin();
        let a = adjoint::grad(&c, &p, &obs);
        let sh = shift::grad(&c, &p, &obs);
        assert!(
            (a[0] - expected).abs() < 1e-12,
            "adjoint {} vs {expected}",
            a[0]
        );
        assert!(
            (sh[0] - expected).abs() < 1e-12,
            "shift {} vs {expected}",
            sh[0]
        );
    }
}

#[test]
fn fixed_angles_contribute_no_gradient() {
    let mut c = Circuit::new(1);
    c.ry(0, Angle::param(0));
    c.rz(0, Angle::Fixed(0.9));
    let obs = Observable::z(0);
    let p = [0.6];

    // RZ after RY does not change <Z>, so the value and gradient match the bare RY.
    let vg = adjoint::value_and_grad(&c, &p, &obs);
    assert!((vg.value - 0.6_f64.cos()).abs() < 1e-12);
    assert!((vg.grad[0] + 0.6_f64.sin()).abs() < 1e-12);
    assert_eq!(vg.grad.len(), 1);
}

#[test]
fn adjoint_matches_parameter_shift() {
    // The headline claim. Every gate type, one to five qubits, depths one to four.
    let mut worst = 0.0_f64;

    for n in 1..=5 {
        for depth in 1..=4 {
            for seed in 0..25u64 {
                let opts = RandomCircuitOpts {
                    num_qubits: n,
                    depth,
                    entangle: n > 1,
                    scaled_angles: true,
                    share_params: true,
                };
                let c = random_circuit(seed, opts);
                let p = random_params(seed ^ 0x5eed, c.num_params());

                for obs in [Observable::z(0), Observable::z_global(n)] {
                    let a = adjoint::grad(&c, &p, &obs);
                    let s = shift::grad(&c, &p, &obs);
                    let d = max_abs_diff(&a, &s);
                    worst = worst.max(d);
                    assert!(
                        d < GRAD_TOL,
                        "n={n} depth={depth} seed={seed}: adjoint and shift differ by {d:e}"
                    );
                }
            }
        }
    }

    println!("worst adjoint-vs-shift disagreement: {worst:e}");
}

#[test]
fn both_paths_match_a_finite_difference() {
    // The third opinion. Loose tolerance because the finite difference is the inaccurate
    // one here, not the analytic paths.
    for n in 1..=4 {
        for seed in 0..10u64 {
            let opts = RandomCircuitOpts {
                num_qubits: n,
                depth: 3,
                entangle: n > 1,
                scaled_angles: true,
                share_params: true,
            };
            let c = random_circuit(seed, opts);
            let p = random_params(seed ^ 0xfd, c.num_params());
            let obs = Observable::new(vec![
                PauliTerm::new(1.0, vec![(0, Pauli::Z)]),
                PauliTerm::new(0.3, vec![(n - 1, Pauli::X)]),
            ]);

            let a = adjoint::grad(&c, &p, &obs);
            let s = shift::grad(&c, &p, &obs);
            let f = shift::finite_difference_grad(&c, &p, &obs, FD_EPS);

            let da = max_abs_diff(&a, &f);
            let ds = max_abs_diff(&s, &f);
            assert!(da < FD_TOL, "n={n} seed={seed}: adjoint vs FD {da:e}");
            assert!(ds < FD_TOL, "n={n} seed={seed}: shift vs FD {ds:e}");
        }
    }
}

#[test]
fn gradients_hold_for_multi_term_observables() {
    // A weighted Pauli sum, which the adjoint path handles through obs.apply_into rather
    // than through a single string.
    let obs = Observable::new(vec![
        PauliTerm::new(0.7, vec![(0, Pauli::Z), (1, Pauli::Z)]),
        PauliTerm::new(-1.3, vec![(0, Pauli::X)]),
        PauliTerm::new(2.0, vec![(1, Pauli::Y)]),
    ]);

    for seed in 0..20u64 {
        let opts = RandomCircuitOpts {
            num_qubits: 3,
            depth: 3,
            ..Default::default()
        };
        let c = random_circuit(seed, opts);
        let p = random_params(seed ^ 0xbeef, c.num_params());

        let a = adjoint::grad(&c, &p, &obs);
        let s = shift::grad(&c, &p, &obs);
        let d = max_abs_diff(&a, &s);
        assert!(d < GRAD_TOL, "seed={seed}: differ by {d:e}");
    }
}

#[test]
fn value_from_gradient_path_matches_plain_evaluation() {
    for seed in 0..10u64 {
        let c = random_circuit(seed, RandomCircuitOpts::default());
        let p = random_params(seed, c.num_params());
        let obs = Observable::z(0);
        let vg = adjoint::value_and_grad(&c, &p, &obs);
        let plain = obs.expectation(&c.run(&p));
        assert!((vg.value - plain).abs() < 1e-14);
    }
}
