//! M34's claims. Soundness is the one that must hold exactly.

use overtone_lie::{closure, PauliString};
use overtone_orbit::checkmate::{apply_exponential, best_reachable_fidelity, haar_state, Position};
use overtone_orbit::invariant::{commutant_basis, expectation};
use overtone_orbit::OrbitCertificate;
use overtone_sim::StateVec;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// The transverse-field Ising algebra on `n` qubits: a proper subalgebra, so it has an orbit
/// structure worth testing.
fn tfim(n: usize) -> Vec<PauliString> {
    let mut g: Vec<PauliString> = (0..n)
        .map(|q| PauliString::single(q, overtone_sim::Pauli::X))
        .collect();
    for q in 0..n - 1 {
        g.push(PauliString::from_factors(&[
            (q, overtone_sim::Pauli::Z),
            (q + 1, overtone_sim::Pauli::Z),
        ]));
    }
    g
}

/// A deliberately small algebra: single-qubit X rotations only. Every qubit's Z-parity in
/// the other qubits is conserved, so the commutant is large and the orbit tiny.
fn local_x(n: usize) -> Vec<PauliString> {
    (0..n)
        .map(|q| PauliString::single(q, overtone_sim::Pauli::X))
        .collect()
}

fn random_orbit_point(
    algebra: &overtone_lie::Algebra,
    start: &StateVec,
    steps: usize,
    rng: &mut ChaCha8Rng,
) -> StateVec {
    let basis = algebra.basis();
    let mut psi = start.clone();
    for _ in 0..steps {
        let g = basis[rng.gen_range(0..basis.len())];
        let theta: f64 = rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI);
        apply_exponential(&mut psi, &g, theta);
    }
    psi
}

/// Every element of the commutant commutes with every element of the algebra. If this fails
/// the GF(2) null space is wrong and every invariant downstream is meaningless.
#[test]
fn the_commutant_commutes() {
    for n in 2..=5 {
        let a = closure(&tfim(n), n, 4096);
        for q in commutant_basis(&a) {
            for p in a.basis() {
                assert!(
                    q.commutes(p),
                    "n={n}: {} does not commute with {}",
                    q.render(n),
                    p.render(n)
                );
            }
        }
    }
}

/// A fully controllable algebra has no commutant beyond the identity, and the certificate
/// says so rather than computing an empty invariant and calling everything unreachable.
#[test]
fn full_controllability_admits_no_certificate() {
    let n = 3;
    let mut g = local_x(n);
    for q in 0..n {
        g.push(PauliString::single(q, overtone_sim::Pauli::Z));
    }
    for q in 0..n - 1 {
        g.push(PauliString::from_factors(&[
            (q, overtone_sim::Pauli::Y),
            (q + 1, overtone_sim::Pauli::Z),
        ]));
    }
    let a = closure(&g, n, 4096);
    assert_eq!(a.dim(), 4usize.pow(n as u32) - 1, "expected su(2^n)");
    let cert = OrbitCertificate::new(&a);
    assert!(cert.fully_controllable());
    assert!(cert.commutant().is_empty());

    let mut rng = ChaCha8Rng::seed_from_u64(1);
    for _ in 0..20 {
        let s = haar_state(n, &mut rng);
        let t = haar_state(n, &mut rng);
        assert!(
            !cert.certainly_unreachable(&s, &t, 1e-9),
            "nothing is unreachable under su(2^n)"
        );
    }
}

/// **Soundness, the claim that must be exact.** The invariants are conserved along the
/// orbit, so the certificate can never fire on a pair that is genuinely connected.
#[test]
fn the_certificate_never_fires_on_a_reachable_pair() {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    for n in 2..=4 {
        for generators in [tfim(n), local_x(n)] {
            let a = closure(&generators, n, 4096);
            let cert = OrbitCertificate::new(&a);
            for _ in 0..40 {
                let source = haar_state(n, &mut rng);
                let target = random_orbit_point(&a, &source, 12, &mut rng);
                let d = cert.invariants(&source).distance(&cert.invariants(&target));
                assert!(
                    d < 1e-9,
                    "n={n} dim={}: invariants moved by {d} along the orbit",
                    a.dim()
                );
                assert!(!cert.certainly_unreachable(&source, &target, 1e-6));
            }
        }
    }
}

/// The commutant invariants are conserved *exactly*, not merely to the purity's tolerance:
/// they are linear, so they are the sharper half of the certificate.
#[test]
fn commutant_expectations_are_conserved_exactly() {
    let n = 4;
    let a = closure(&local_x(n), n, 4096);
    let commutant = commutant_basis(&a);
    assert!(!commutant.is_empty(), "local X should conserve Z-parities");
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let start = haar_state(n, &mut rng);
    let before: Vec<f64> = commutant.iter().map(|q| expectation(q, &start)).collect();
    let moved = random_orbit_point(&a, &start, 30, &mut rng);
    let after: Vec<f64> = commutant.iter().map(|q| expectation(q, &moved)).collect();
    let worst = before
        .iter()
        .zip(&after)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    assert!(worst < 1e-12, "commutant drifted by {worst}");
}

/// The brute-force oracle must actually work, or a soundness test that relies on it proves
/// nothing. Given a pair connected by construction, it has to find the connection.
#[test]
fn the_brute_force_oracle_finds_paths_that_exist() {
    let n = 3;
    let a = closure(&tfim(n), n, 4096);
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let mut found = 0;
    for _ in 0..8 {
        let source = haar_state(n, &mut rng);
        let target = random_orbit_point(&a, &source, 3, &mut rng);
        let f = best_reachable_fidelity(&a, &source, &target, 2, 8, &mut rng);
        if f > 0.99 {
            found += 1;
        }
    }
    assert_eq!(
        found, 8,
        "the oracle found only {found}/8 paths that exist by construction"
    );
}

/// **Soundness against the oracle.** Every pair the certificate calls unreachable must
/// defeat the search. A single counterexample would mean the certificate is unsound.
#[test]
fn nothing_the_certificate_rejects_can_be_reached() {
    let n = 3;
    let a = closure(&local_x(n), n, 4096);
    let cert = OrbitCertificate::new(&a);
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let mut checked = 0;
    for _ in 0..30 {
        let source = haar_state(n, &mut rng);
        let target = haar_state(n, &mut rng);
        if !cert.certainly_unreachable(&source, &target, 1e-6) {
            continue;
        }
        checked += 1;
        let f = best_reachable_fidelity(&a, &source, &target, 2, 8, &mut rng);
        assert!(
            f < 0.999,
            "certificate called a pair unreachable that the oracle connected at {f}"
        );
    }
    assert!(
        checked >= 10,
        "only {checked} pairs were rejected; test is weak"
    );
}

/// Check is exact: it is an expectation value and nothing more.
#[test]
fn check_is_an_expectation_value() {
    let n = 2;
    let a = closure(&local_x(n), n, 4096);
    let dim = 1usize << n;
    // Amplitude entirely on basis state 0, which is safe: no check.
    let mut re = vec![0.0; dim];
    re[0] = 1.0;
    let p = Position::new(
        StateVec::from_amplitudes(re, vec![0.0; dim]),
        a.clone(),
        vec![0],
    );
    assert!(!p.is_check(1e-12));
    assert!((p.absorbed_weight() - 0.0).abs() < 1e-15);

    // Half the amplitude on an absorbing state: check, with weight one half.
    let mut re = vec![0.0; dim];
    re[0] = std::f64::consts::FRAC_1_SQRT_2;
    re[1] = std::f64::consts::FRAC_1_SQRT_2;
    let p = Position::new(StateVec::from_amplitudes(re, vec![0.0; dim]), a, vec![0]);
    assert!(p.is_check(1e-12));
    assert!((p.absorbed_weight() - 0.5).abs() < 1e-12);
}

/// The certificate is incomplete **by counting**, not by bad luck. When the invariant level
/// set has more dimensions than the orbit, it contains a continuum of distinct orbits and no
/// certificate built from those invariants can separate them.
#[test]
fn a_low_degree_certificate_is_provably_incomplete() {
    use overtone_orbit::separation_deficit;
    let mut rng = ChaCha8Rng::seed_from_u64(21);
    for n in [3usize, 4] {
        let a = closure(&local_x(n), n, 4096);
        let cert = OrbitCertificate::new(&a);
        let psi = haar_state(n, &mut rng);
        let deficit = separation_deficit(&a, &cert, &psi);
        assert!(
            deficit > 0,
            "n={n}: deficit {deficit}, so the counting argument would not apply"
        );
    }
    // And where the deficit vanishes the certificate can be complete: the TFIM orbit at
    // three qubits fills its level set.
    let a = closure(&tfim(3), 3, 4096);
    let cert = OrbitCertificate::new(&a);
    let psi = haar_state(3, &mut rng);
    assert_eq!(separation_deficit(&a, &cert, &psi), 0);
}

/// Two Haar-random states disagree on essentially every invariant, so measuring completeness
/// on them reports 1.0 for a certificate that is provably not complete. The hard pairs have
/// to be constructed, and this checks the constructor actually constructs them.
#[test]
fn invariant_matched_pairs_defeat_the_certificate() {
    use overtone_orbit::checkmate::matched_invariant_state;
    let n = 4;
    let a = closure(&local_x(n), n, 4096);
    let cert = OrbitCertificate::new(&a);
    let mut rng = ChaCha8Rng::seed_from_u64(23);
    let mut matched = 0;
    for _ in 0..6 {
        let source = haar_state(n, &mut rng);
        let (target, residual) = matched_invariant_state(&cert, &source, &mut rng, 60);
        if residual > 1e-6 {
            continue;
        }
        matched += 1;
        assert!(
            !cert.certainly_unreachable(&source, &target, 1e-6),
            "a matched pair should be invisible to the certificate"
        );
    }
    assert!(matched >= 4, "only built {matched} matched pairs of 6");
}

/// The number the game depends on. Against a safe subspace made of computational basis
/// states -- which is what a maze region is -- the certificate catches the checkmates.
#[test]
fn checkmate_is_complete_against_basis_state_safety() {
    use overtone_orbit::checkmate::best_reachable_fidelity;
    let n = 3;
    let a = closure(&local_x(n), n, 4096);
    let mut rng = ChaCha8Rng::seed_from_u64(29);
    let (mut lost, mut called) = (0, 0);
    for _ in 0..12 {
        let state = haar_state(n, &mut rng);
        let safe: Vec<usize> = (0..2).collect();
        let position = Position::new(state.clone(), a.clone(), safe);
        let says = position.is_checkmate(1e-6);
        let escapable = position
            .safe_samples()
            .iter()
            .any(|s| best_reachable_fidelity(&a, &state, s, 2, 4, &mut rng) > 0.999);
        assert!(!(says && escapable), "unsound checkmate call");
        if !escapable {
            lost += 1;
            if says {
                called += 1;
            }
        }
    }
    assert!(lost > 0, "no lost positions were generated");
    assert_eq!(
        called,
        lost,
        "missed {} of {lost} checkmates",
        lost - called
    );
}
