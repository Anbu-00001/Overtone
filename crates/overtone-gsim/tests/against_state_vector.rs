//! M13's acceptance test: g-sim must match Engine A exactly where both apply.
//!
//! The two engines share no arithmetic. Engine A carries `2^n` complex amplitudes and
//! applies strided single- and two-qubit kernels; g-sim carries `dim(g)` real expectation
//! values and applies Givens rotations. They agree because the algebra says they must, and
//! that agreement is the only evidence that the rotation signs in `evolve.rs` are right --
//! a sign error there is invisible in `dim(g)` and invisible in the norm.
//!
//! The bridge is the standard identity `exp(-i t Z_a Z_b / 2) = CNOT RZ_b(t) CNOT`, which
//! is how a two-site generator is expressed in the gate set of Part I 6.1.

use overtone_gsim::{GsimCircuit, GsimPolicy, PolicyLayout, PolicyObservable};
use overtone_lie::{closure_unbounded, family, PauliString};
use overtone_sim::{grad, Angle, Circuit, Observable, Pauli, PauliTerm};

/// Deterministic angles; this crate seeds everything explicitly.
fn angles(count: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..count)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 6.0 - 3.0
        })
        .collect()
}

fn x0(n: usize) -> Observable {
    let _ = n;
    Observable::new(vec![PauliTerm::new(1.0, vec![(0, Pauli::X)])])
}

/// Generators in the order `overtone-gsim` uses: X_q first, then Z_qZ_{q+1}.
fn tfim_generators(n: usize) -> Vec<PauliString> {
    let mut g: Vec<PauliString> = (0..n).map(|q| PauliString::single(q, Pauli::X)).collect();
    g.extend((0..n - 1).map(|q| PauliString::from_factors(&[(q, Pauli::Z), (q + 1, Pauli::Z)])));
    g
}

/// Append the state-vector realisation of `exp(-i theta G / 2)` for generator index `gi`.
fn push_equivalent(c: &mut Circuit, n: usize, gi: usize, angle: Angle) {
    if gi < n {
        c.rx(gi, angle);
    } else {
        let q = gi - n;
        c.cnot(q, q + 1);
        c.rz(q + 1, angle);
        c.cnot(q, q + 1);
    }
}

#[test]
fn random_dla_circuits_match_the_state_vector() {
    for n in 2..=5 {
        let generators = tfim_generators(n);
        let algebra = closure_unbounded(&family::tfim(n), n);
        assert_eq!(algebra.dim(), n * (2 * n - 1));
        let mut g = GsimCircuit::new(algebra, generators.clone());
        let mut c = Circuit::new(n);

        // Three sweeps of every generator, each with its own parameter.
        let mut p = 0usize;
        for _ in 0..3 {
            for gi in 0..generators.len() {
                g.push(gi, Angle::param(p));
                push_equivalent(&mut c, n, gi, Angle::param(p));
                p += 1;
            }
        }
        let params = angles(p, 900 + n as u64);

        let index = g
            .observable_index(&PauliString::single(0, Pauli::X))
            .expect("X_0 is in the TFIM algebra");
        let mut weights = vec![0.0; g.dim()];
        weights[index] = 1.0;

        let (gv, gg) = g.value_and_grad(&params, &weights);
        let exact = grad::value_and_grad(&c, &params, &x0(n));

        assert!(
            (gv - exact.value).abs() < 1e-12,
            "n = {n}: g-sim {gv} vs state vector {}",
            exact.value
        );
        let worst = gg
            .iter()
            .zip(&exact.grad)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-12, "n = {n}: worst gradient difference {worst}");
    }
}

#[test]
fn the_policy_matches_a_state_vector_policy() {
    for (n, observable) in [
        (2, PolicyObservable::FirstX),
        (3, PolicyObservable::FirstX),
        (4, PolicyObservable::FirstX),
        (3, PolicyObservable::MeanX),
        (4, PolicyObservable::MeanX),
        (3, PolicyObservable::MeanZz),
        (5, PolicyObservable::MeanZz),
    ] {
        let layers = 2;
        let policy = GsimPolicy::new(PolicyLayout {
            num_qubits: n,
            layers,
            lambda_trainable: true,
            observable,
        });
        let params = {
            let mut p = angles(policy.num_params(), 4242 + n as u64);
            p[0] = 1.7; // lambda
            p
        };

        for &s in &[-2.4_f64, -0.3, 0.0, 1.1, 2.9] {
            // The same circuit, gate for gate, in the state-vector engine.
            let mut c = Circuit::new(n);
            let mut next = 1usize;
            for _ in 0..layers {
                for q in 0..n {
                    c.rx(q, Angle::scaled(0, s));
                }
                for q in 0..n - 1 {
                    push_equivalent(&mut c, n, n + q, Angle::param(next));
                    next += 1;
                }
                for q in 0..n {
                    c.rx(q, Angle::param(next));
                    next += 1;
                }
            }
            let obs = match observable {
                PolicyObservable::FirstX => x0(n),
                PolicyObservable::MeanX => Observable::new(
                    (0..n)
                        .map(|q| PauliTerm::new(1.0 / n as f64, vec![(q, Pauli::X)]))
                        .collect(),
                ),
                PolicyObservable::MeanZz => Observable::new(
                    (0..n - 1)
                        .map(|q| {
                            PauliTerm::new(
                                1.0 / (n - 1) as f64,
                                vec![(q, Pauli::Z), (q + 1, Pauli::Z)],
                            )
                        })
                        .collect(),
                ),
            };
            let exact = grad::value_and_grad(&c, &params, &obs);
            let (p1, dp) = policy.prob_and_grad(&params, s);

            let expected_p1 = 0.5 * (1.0 + exact.value);
            assert!(
                (p1 - expected_p1).abs() < 1e-12,
                "n = {n}, s = {s}: pi {p1} vs {expected_p1}"
            );
            let worst = dp
                .iter()
                .zip(&exact.grad)
                .map(|(a, b)| (a - 0.5 * b).abs())
                .fold(0.0, f64::max);
            assert!(
                worst < 1e-12,
                "n = {n}, s = {s}: worst gradient diff {worst}"
            );
        }
    }
}

#[test]
fn evolution_preserves_the_length_of_the_expectation_vector() {
    // Every gate is an orthogonal rotation in the DLA basis, so the norm is conserved
    // exactly. This is the g-sim analogue of the state vector's unitarity check, and it
    // fails immediately if the pairing signs do not oppose.
    let n = 4;
    let generators = tfim_generators(n);
    let algebra = closure_unbounded(&family::tfim(n), n);
    let mut g = GsimCircuit::new(algebra, generators.clone());
    for (p, gi) in (0..generators.len()).cycle().take(30).enumerate() {
        g.push(gi, Angle::param(p));
    }
    let params = angles(30, 77);
    let e0 = g.initial_vector();
    let e = g.evolve(&params);
    let n0: f64 = e0.iter().map(|x| x * x).sum();
    let n1: f64 = e.iter().map(|x| x * x).sum();
    assert!((n0 - n1).abs() < 1e-12, "norm {n0} -> {n1}");
}

#[test]
fn at_one_layer_the_mean_x_observable_is_odd_and_scores_exactly_zero() {
    // The conserved X-or-Y count proved in `policy.rs`. This is why the first hundred-qubit
    // run reported J = 0.000000 with a gradient of zero everywhere: not a plateau, not a
    // bug, a symmetry -- and one that holds at a single layer only, because a second layer
    // gives the variational RX gates something to act on and they break the count.
    use overtone_rl::SpectralControl;

    let env = SpectralControl::new(3);
    for n in [3usize, 5, 8, 12] {
        let odd = GsimPolicy::new(PolicyLayout {
            num_qubits: n,
            layers: 1,
            lambda_trainable: true,
            observable: PolicyObservable::MeanX,
        });
        for seed in 0..4u64 {
            let mut params = angles(odd.num_params(), 31 + n as u64 * 7 + seed);
            params[0] = 1.3;
            for &s in &[0.4_f64, 1.9, -2.7] {
                let a = odd.prob_action1(&params, s) - 0.5;
                let b = odd.prob_action1(&params, -s) - 0.5;
                assert!((a + b).abs() < 1e-13, "n={n} seed={seed} s={s}: {a} vs {b}");
            }
            let j = env.expected_return(256, |s| odd.prob_action1(&params, s));
            assert!(
                j.abs() < 1e-13,
                "n={n} seed={seed}: return {j} was not zero"
            );
        }

        // Two layers, same observable: the symmetry is gone and the policy scores.
        let two = GsimPolicy::new(PolicyLayout {
            num_qubits: n,
            layers: 2,
            lambda_trainable: true,
            observable: PolicyObservable::MeanX,
        });
        let mut params = angles(two.num_params(), 99 + n as u64);
        params[0] = 1.3;
        let j = env.expected_return(256, |s| two.prob_action1(&params, s));
        assert!(j.abs() > 1e-6, "n={n}: two layers still scored zero ({j})");

        // The Z-type observable carries no such law at any depth.
        for layers in [1usize, 2, 4] {
            let even = GsimPolicy::new(PolicyLayout {
                num_qubits: n,
                layers,
                lambda_trainable: true,
                observable: PolicyObservable::MeanZz,
            });
            let mut params = angles(even.num_params(), 7 + n as u64 + layers as u64);
            params[0] = 1.3;
            let j = env.expected_return(256, |s| even.prob_action1(&params, s));
            assert!(j.abs() > 1e-6, "n={n} L={layers}: Z-type scored zero ({j})");
        }
    }
}
