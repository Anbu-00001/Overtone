//! M28, against Sansoni et al. (PRL 108, 010502).
//!
//! Part VI 8: "if these do not match the published patterns, the arena's physics is wrong and
//! everything built on it is theatre." So these tests check the published *structure* --
//! bunching, exclusion, and the phase that interpolates -- rather than eyeballing a figure.

use overtone_walk::two::{correlate, evolve, similarity, Statistics};
use overtone_walk::{Coin, Substrate, Word};
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

fn clean() -> Substrate {
    Substrate::new(Word::Periodic, 64, 0)
}

fn run(steps: usize, s: Statistics) -> overtone_walk::Correlation {
    let (a, b) = (Coin::hadamard(), Coin::hadamard());
    correlate(steps, (0, 0), (0, 1), s, &clean(), &a, &b)
}

#[test]
fn pauli_exclusion_is_exact_in_the_mode_basis() {
    // Two fermions never occupy the same mode. Not "nearly never" -- the amplitude is
    // identically zero, so this is an exact equality and not a tolerance.
    for steps in 1..=6 {
        let c = run(steps, Statistics::Fermionic);
        for k in 0..c.modes.len() {
            assert_eq!(c.modes_matrix[k][k], 0.0, "{steps} steps, mode {k}");
        }
        assert_eq!(c.mode_bunching(), 0.0, "{steps} steps");
    }
}

#[test]
fn two_fermions_can_share_a_site() {
    // The correction to Part VI 1. Sansoni et al.: "some of the diagonal elements of the
    // fermionic two-particle walk are nonzero", because a coined walk has two degrees of
    // freedom and their Eq. (4) state |j,U> - |j,D> is antisymmetric while sharing a site.
    // A fermionic agent therefore does not wall off a corridor; it blocks one coin state.
    for steps in 2..=6 {
        let c = run(steps, Statistics::Fermionic);
        assert!(
            c.position_bunching() > 1e-3,
            "{steps} steps: fermionic position diagonal is {}, but exclusion applies to \
             modes, not sites",
            c.position_bunching()
        );
    }
    let c = run(4, Statistics::Fermionic);
    assert!((c.position_bunching() - 0.09375).abs() < 1e-12);
}

#[test]
fn bosons_bunch_and_fermions_do_not() {
    for steps in 2..=6 {
        let b = run(steps, Statistics::Bosonic);
        let f = run(steps, Statistics::Fermionic);
        assert!(
            b.mode_bunching() > 0.0,
            "{steps} steps: bosons should bunch"
        );
        assert_eq!(f.mode_bunching(), 0.0);
        assert!(
            b.position_bunching() > f.position_bunching(),
            "{steps} steps: {} vs {}",
            b.position_bunching(),
            f.position_bunching()
        );
    }
}

#[test]
fn the_anyonic_phase_interpolates_monotonically() {
    // "phi interpolates continuously; a dial between the two" -- Part VI 1, checked.
    let steps = 4;
    let ladder = [
        Statistics::Bosonic,
        Statistics::Anyonic(FRAC_PI_4),
        Statistics::Anyonic(FRAC_PI_2),
        Statistics::Anyonic(3.0 * FRAC_PI_4),
        Statistics::Fermionic,
    ];
    let bunching: Vec<f64> = ladder
        .iter()
        .map(|&s| run(steps, s).mode_bunching())
        .collect();
    for w in bunching.windows(2) {
        assert!(w[0] > w[1], "bunching should fall with phi: {bunching:?}");
    }
    assert_eq!(bunching[4], 0.0, "the fermionic endpoint is exactly zero");
    // Anyonic(PI) is the same physics but not the same arithmetic: expi(pi) has an imaginary
    // part of 1.2e-16, so the cancellation lands near 1e-34 instead of on zero.
    let via_phase = run(steps, Statistics::Anyonic(PI)).mode_bunching();
    assert!(
        via_phase < 1e-30 && via_phase > 0.0,
        "expected a tiny non-zero, got {via_phase}"
    );
    // And Sansoni et al.'s reported phi = pi/2 shows both behaviours at once.
    let mid = run(steps, Statistics::Anyonic(FRAC_PI_2));
    assert!(mid.mode_bunching() > 0.0, "anyons bunch");
    assert!(mid.mode_bunching() < run(steps, Statistics::Bosonic).mode_bunching());
    assert!(mid.position_bunching() < 1.0, "and antibunch");
}

#[test]
fn the_three_statistics_give_different_distributions() {
    // If they did not, the class system would be decorative.
    let b = run(4, Statistics::Bosonic);
    let f = run(4, Statistics::Fermionic);
    let a = run(4, Statistics::Anyonic(FRAC_PI_2));
    assert!(similarity(&b.positions, &f.positions) < 0.9);
    assert!(similarity(&b.positions, &b.positions) > 0.999_999);
    // The anyon sits between them and resembles neither exactly.
    let (sb, sf) = (
        similarity(&a.positions, &b.positions),
        similarity(&a.positions, &f.positions),
    );
    assert!(sb < 0.99 && sf < 0.99, "anyon similarity {sb} / {sf}");
}

#[test]
fn adjacent_inputs_cannot_interfere_at_all() {
    // The parity trap, pinned so nobody restores it. A coined walk preserves the parity of
    // site + step, so walkers starting one site apart occupy disjoint sublattices forever.
    // Every exchange term is then zero against a non-zero direct term, all statistics agree,
    // and a test built on that arrangement would pass while measuring nothing.
    let (a, b) = (Coin::hadamard(), Coin::hadamard());
    let bos = correlate(4, (0, 0), (1, 0), Statistics::Bosonic, &clean(), &a, &b);
    let fer = correlate(4, (0, 0), (1, 0), Statistics::Fermionic, &clean(), &a, &b);
    assert_eq!(bos.mode_bunching(), 0.0);
    assert_eq!(fer.mode_bunching(), 0.0);
    assert!(
        similarity(&bos.positions, &fer.positions) > 0.999_999,
        "adjacent inputs must be indistinguishable across statistics"
    );
}

#[test]
fn the_distribution_is_normalised() {
    for steps in 1..=6 {
        for s in [
            Statistics::Bosonic,
            Statistics::Fermionic,
            Statistics::Anyonic(FRAC_PI_4),
        ] {
            let c = run(steps, s);
            assert!(
                (c.total() - 1.0).abs() < 1e-12,
                "{} at {steps} steps sums to {}",
                s.name(),
                c.total()
            );
        }
    }
}

#[test]
fn the_single_particle_walk_stays_unitary() {
    let (a, b) = (Coin::hadamard(), Coin::hadamard());
    for steps in 1..=8 {
        for pos in [-2i64, 0, 3] {
            for coin in 0..2 {
                let radius = pos.unsigned_abs() as usize + steps;
                let col = evolve(steps, pos, coin, radius, &clean(), &a, &b);
                assert!((col.norm() - 1.0).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn a_disordered_substrate_still_obeys_exclusion() {
    // Exclusion is a statement about the state, not about the terrain, so it must survive
    // Anderson-style disorder untouched.
    let sub = Substrate::new(Word::Disorder, 64, 7);
    let (a, b) = (Coin::hadamard(), Coin::generic(0.3, 1.1, 0.7));
    let f = correlate(5, (0, 0), (0, 1), Statistics::Fermionic, &sub, &a, &b);
    assert_eq!(f.mode_bunching(), 0.0);
    assert!((f.total() - 1.0).abs() < 1e-12);
    assert!(f.position_bunching() > 0.0);
}
