//! M37: the dial, and the two representations it reads.

use overtone_lie::{closure, PauliString};
use overtone_orbit::checkmate::{apply_exponential, haar_state};
use overtone_orbit::dial::measure;
use overtone_sim::{Pauli, StateVec};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn product_state(n: usize) -> StateVec {
    let dim = 1usize << n;
    let mut re = vec![0.0; dim];
    re[0] = 1.0;
    StateVec::from_amplitudes(re, vec![0.0; dim])
}

/// Part VII 6's acceptance, in one test: a position at one end of the dial is efficiently
/// evaluable, and a position at the other end is not.
#[test]
fn the_dial_moves_between_evaluable_and_hard() {
    let n = 6;
    // Low end: a small algebra and a product state.
    let small = closure(
        &(0..n)
            .map(|q| PauliString::single(q, Pauli::X))
            .collect::<Vec<_>>(),
        n,
        4096,
    );
    let easy = measure(&small, &product_state(n), 1e-8);
    assert!(easy.efficiently_evaluable(), "{}", easy.verdict());
    assert!(easy.gsim_works() && easy.mps_works());

    // High end: the full algebra and a Haar-random state.
    let mut g: Vec<PauliString> = (0..n).map(|q| PauliString::single(q, Pauli::X)).collect();
    for q in 0..n {
        g.push(PauliString::single(q, Pauli::Z));
    }
    for q in 0..n - 1 {
        g.push(PauliString::from_factors(&[
            (q, Pauli::Y),
            (q + 1, Pauli::Z),
        ]));
    }
    let big = closure(&g, n, 8192);
    let mut rng = ChaCha8Rng::seed_from_u64(4);
    let hard = measure(&big, &haar_state(n, &mut rng), 1e-8);
    assert!(hard.provably_hard(), "{}", hard.verdict());
    assert!(!hard.gsim_works() && !hard.mps_works());
}

/// The two axes are independent: a large algebra with a product state is still evaluable,
/// because `g`-sim and MPS fail for different reasons. A verdict built on one number would
/// get this case wrong.
#[test]
fn the_two_axes_are_independent() {
    let n = 6;
    let mut g: Vec<PauliString> = (0..n).map(|q| PauliString::single(q, Pauli::X)).collect();
    for q in 0..n {
        g.push(PauliString::single(q, Pauli::Z));
    }
    for q in 0..n - 1 {
        g.push(PauliString::from_factors(&[
            (q, Pauli::Y),
            (q + 1, Pauli::Z),
        ]));
    }
    let big = closure(&g, n, 8192);
    let c = measure(&big, &product_state(n), 1e-8);
    assert!(!c.gsim_works(), "the algebra is exponential");
    assert!(c.mps_works(), "but a product state has bond dimension one");
    assert!(c.efficiently_evaluable(), "{}", c.verdict());
}

/// Entangling a product state raises the bond dimension, so the dial actually moves when the
/// position does rather than being a property of the algebra alone.
#[test]
fn entangling_moves_the_position_along_the_dial() {
    let n = 6;
    let small = closure(
        &(0..n)
            .map(|q| PauliString::single(q, Pauli::X))
            .collect::<Vec<_>>(),
        n,
        4096,
    );
    let before = measure(&small, &product_state(n), 1e-8).chi;
    let mut psi = product_state(n);
    // Order matters and it is not a detail: `exp(i t X_q Z_p)` applied to |0...0> leaves a
    // product state, because `Z` acts trivially on |0>. The single-qubit rotations have to
    // come first so the coupling has something to correlate.
    for q in 0..n {
        apply_exponential(&mut psi, &PauliString::single(q, Pauli::Y), 0.6);
    }
    for q in 0..n - 1 {
        apply_exponential(
            &mut psi,
            &PauliString::from_factors(&[(q, Pauli::Z), (q + 1, Pauli::Z)]),
            0.7,
        );
    }
    let after = measure(&small, &psi, 1e-8).chi;
    assert!(
        after > before,
        "bond dimension did not rise: {before} then {after}"
    );
}
