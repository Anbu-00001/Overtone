//! M10's acceptance test: `dim(g)` against the published classification.
//!
//! Every expected value below is a closed form from Theorem IV.1 of Wiersema, Kokcu,
//! Kemper and Bakalov, *Classification of dynamical Lie algebras of 2-local spin systems on
//! linear, circular and fully connected topologies*, npj Quantum Information 10, 110 (2024)
//! (arXiv:2309.05690), combined with the dimensions of the classical algebras:
//!
//! ```text
//! dim su(N) = N^2 - 1     dim so(N) = N(N-1)/2     dim sp(2m) = m(2m+1)
//! ```
//!
//! The theorem is stated for chains of length `n >= 3`, and a few entries for `n >= 4`; the
//! ranges below respect that.

use overtone_lie::{closure_unbounded, family};

fn dim(generators: Vec<overtone_lie::PauliString>, n: usize) -> usize {
    closure_unbounded(&generators, n).dim()
}

fn so(n: usize) -> usize {
    n * (n - 1) / 2
}

fn su(dim: usize) -> usize {
    dim * dim - 1
}

/// `dim sp(2m) = m(2m + 1)`. The paper writes the a_9 case as `sp(2^(n-2))`, and the
/// dimension it means is the one with `m = 2^(n-2)`: at n = 4 that is 36, not 10. The
/// `so(2^n)` entries in the same theorem are matrix sizes, so the two notations differ
/// inside one list -- worth stating, since reading `sp` the way `so` reads there gives an
/// answer wrong by a factor of three and a half.
fn sp(m: usize) -> usize {
    m * (2 * m + 1)
}

#[test]
fn a0_ising_coupling_is_abelian() {
    // a_0(n) = u(1)^(+(n-1)).
    for n in 3..=8 {
        assert_eq!(dim(family::ising(n), n), n - 1, "n = {n}");
    }
}

#[test]
fn a1_kitaev_chain_is_so_n() {
    // a_1(n) = so(n). Wiersema et al. Example II.4 gives dim 3 at n = 3 explicitly.
    for n in 3..=8 {
        assert_eq!(dim(family::kitaev(n), n), so(n), "n = {n}");
    }
}

#[test]
fn a4_xy_model_is_so_n_plus_so_n() {
    // a_2(n) = a_4(n) = so(n) (+) so(n).
    for n in 3..=8 {
        assert_eq!(dim(family::xy(n), n), 2 * so(n), "n = {n}");
    }
}

#[test]
fn a8_is_so_2n_minus_1() {
    // a_8(n) = so(2n - 1).
    for n in 3..=7 {
        assert_eq!(dim(family::a8(n), n), so(2 * n - 1), "n = {n}");
    }
}

#[test]
fn a9_is_symplectic() {
    // a_9(n) = sp(2^(n-2)).
    for n in 4..=6 {
        assert_eq!(dim(family::a9(n), n), sp(1 << (n - 2)), "n = {n}");
        assert_eq!(
            sp(1 << (n - 2)),
            (1 << (n - 2)) * ((1 << (n - 1)) + 1),
            "n = {n}"
        );
    }
}

#[test]
fn a11_is_so_two_to_the_n() {
    // a_11(n) = so(2^n), n >= 4.
    for n in 4..=6 {
        assert_eq!(dim(family::a11(n), n), so(1 << n), "n = {n}");
    }
}

#[test]
fn a14_is_so_2n() {
    // a_14(n) = so(2n).
    for n in 3..=8 {
        assert_eq!(dim(family::a14(n), n), so(2 * n), "n = {n}");
    }
}

#[test]
fn a15_is_two_copies_of_su() {
    // a_13(n) = a_15(n) = a_20(n) = su(2^(n-1))^(+2).
    for n in 3..=5 {
        assert_eq!(dim(family::a15(n), n), 2 * su(1 << (n - 1)), "n = {n}");
    }
}

#[test]
fn a7_heisenberg_chain_is_exponential() {
    // a_6(n) = a_7(n) = a_10(n) = su(2^(n-1)) for odd n, su(2^(n-2))^(+4) for even n >= 4.
    for n in 3..=6 {
        let expected = if n % 2 == 1 {
            su(1 << (n - 1))
        } else {
            4 * su(1 << (n - 2))
        };
        assert_eq!(dim(family::heisenberg(n), n), expected, "n = {n}");
    }
}

#[test]
fn transverse_field_ising_is_so_2n() {
    // The TFIM generating set {ZZ, XI} generates a_14 -- at n = 2 both close on the same
    // six strings -- so the open chain is so(2n), dimension n(2n - 1). This is the value
    // PennyLane's g-sim demo quotes as 2n(2n-1)/2, and it is the algebra the 100-qubit run
    // in overtone-gsim uses.
    for n in 2..=9 {
        assert_eq!(dim(family::tfim(n), n), n * (2 * n - 1), "n = {n}");
    }
}

#[test]
fn transverse_field_ising_on_a_ring_doubles() {
    // Wiersema et al. Fig. 7(a): the TFIM on a ring has DLA a_8^o(n) = so(2n) (+) so(2n).
    for n in 5..=8 {
        assert_eq!(dim(family::tfim_ring(n), n), 2 * so(2 * n), "n = {n}");
    }
}

#[test]
fn hardware_efficient_with_trainable_entanglers_is_all_of_su() {
    for n in 2..=5 {
        assert_eq!(dim(family::hardware_efficient(n), n), su(1 << n), "n = {n}");
    }
}

#[test]
fn fixed_entanglers_leave_only_single_qubit_algebra() {
    // The distinction Part III elides and this crate does not: with *fixed* CZ layers the
    // parameterised generators are single-qubit only, and the DLA is su(2)^(+n) -- 3n, not
    // 4^n - 1. See `family::EntanglerPolicy`.
    for n in 2..=6 {
        assert_eq!(dim(family::single_qubit_only(n), n), 3 * n, "n = {n}");
    }
}

#[test]
fn commuting_blocks_are_found_where_they_are_visible_in_the_pauli_basis() {
    use overtone_lie::predict::components;

    // su(2)^(+n): n blocks of three, each one qubit's worth.
    for n in 2..=5 {
        let g = closure_unbounded(&family::single_qubit_only(n), n);
        let blocks = components(&g);
        assert_eq!(blocks.len(), n, "single-qubit n = {n}");
        assert!(blocks.iter().all(|b| b.len() == 3));
    }

    // so(n) (+) so(n): two blocks of equal size.
    for n in 4..=7 {
        let g = closure_unbounded(&family::xy(n), n);
        let blocks = components(&g);
        assert_eq!(blocks.len(), 2, "xy n = {n}");
        assert_eq!(blocks[0].len(), so(n));
        assert_eq!(blocks[1].len(), so(n));
    }

    // so(2n) is simple: one block, no centre.
    for n in 3..=6 {
        let g = closure_unbounded(&family::tfim(n), n);
        assert_eq!(components(&g).len(), 1, "tfim n = {n}");
    }

    // The abelian case is all centre: n - 1 blocks of one.
    let g = closure_unbounded(&family::ising(6), 6);
    assert_eq!(components(&g).len(), 5);
}
