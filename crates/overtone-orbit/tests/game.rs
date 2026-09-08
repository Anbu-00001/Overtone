//! M35: the rules, as tests. The ladder itself is measured in `examples/ladder.rs`.

use overtone_lie::PauliString;
use overtone_orbit::game::{legal_moves, Game, Move, Piece, Player};
use overtone_sim::{Pauli, StateVec};

fn basis(n: usize, i: usize) -> StateVec {
    let dim = 1usize << n;
    let mut re = vec![0.0; dim];
    re[i] = 1.0;
    StateVec::from_amplitudes(re, vec![0.0; dim])
}

fn game(n: usize) -> Game {
    let dim = 1usize << n;
    let a = Player::new(
        basis(n, 0),
        (0..n).map(|q| PauliString::single(q, Pauli::X)).collect(),
        16,
    );
    let b = Player::new(
        basis(n, dim - 1),
        (0..n).map(|q| PauliString::single(q, Pauli::Z)).collect(),
        16,
    );
    Game::new([a, b], (0..dim).collect(), 2)
}

/// **The bishop is the whole insight.** Square colour is the parity of the position index,
/// read by `Z Z ... Z`. A bishop commutes with it and a pawn does not, so a bishop is
/// confined to one colour for exactly the reason a superselection sector is.
#[test]
fn a_bishop_conserves_square_colour_and_a_pawn_does_not() {
    let n = 4;
    let colour = PauliString::from_factors(&(0..n).map(|q| (q, Pauli::Z)).collect::<Vec<_>>());
    let bishop = Piece::Bishop.generator(&[0, 1]);
    let pawn = Piece::Pawn.generator(&[0]);
    assert!(
        bishop.commutes(&colour),
        "a bishop must preserve square colour"
    );
    assert!(!pawn.commutes(&colour), "a pawn must change square colour");
    // And a knight, which carries one X, changes it too.
    assert!(!Piece::Knight.generator(&[0, 1]).commutes(&colour));
}

/// **The queen is not additive.** `[rook, bishop]` generates a direction in neither, which is
/// the five-hundred-year-old chess intuition that a queen beats a rook plus a bishop.
#[test]
fn the_queen_is_the_commutator_closure_and_beats_its_parts() {
    let rook = Piece::Rook.generator(&[0]);
    let bishop = Piece::Bishop.generator(&[0, 1]);
    let separate =
        overtone_lie::closure(&[rook], 3, 64).dim() + overtone_lie::closure(&[bishop], 3, 64).dim();
    let together = overtone_lie::closure(&[rook, bishop], 3, 64).dim();
    assert!(
        together > separate - 1,
        "closure {together} against parts {separate}"
    );
    let (c, _) = rook
        .commutator(&bishop)
        .expect("rook and bishop anticommute");
    assert_ne!(c, rook);
    assert_ne!(c, bishop);
}

/// Part VII 9.2: the branching factor is measured, not intended. Chess reports about 35 and
/// the target band is 25 to 40.
#[test]
fn the_branching_factor_lands_in_the_declared_band() {
    assert_eq!(legal_moves(4).len(), 30);
    assert_eq!(legal_moves(5).len(), 45);
    let n4 = legal_moves(4).len();
    assert!(
        (25..=40).contains(&n4),
        "n=4 branching {n4} is outside the 25 to 40 band"
    );
}

/// Rule 5: coherence only decreases, and `dim(g)` only increases.
#[test]
fn coherence_falls_and_the_algebra_grows() {
    let mut g = game(4);
    let before_coherence = g.players[0].coherence;
    let before_dim = g.players[0].dim_g();
    g.apply(
        &Move::Apply {
            piece: Piece::Knight,
            targets: vec![0, 2],
        },
        0.5,
    );
    assert!(g.players[0].coherence < before_coherence, "rule 5");
    assert!(g.players[0].dim_g() >= before_dim, "rule 5");
}

/// Rule 6: overlapping merges algebras, and the merged algebra contains both.
#[test]
fn overlap_merges_algebras() {
    let mut g = game(3);
    let before = g.players[0].dim_g();
    g.merge(0, 1);
    let after = g.players[0].dim_g();
    assert!(
        after > before,
        "merging did not grow the algebra: {before} to {after}"
    );
    for h in g.players[1].hand.clone() {
        assert!(g.players[0].hand.contains(&h));
    }
}

/// The pursuer defines the absorbing subspace, so a player's safe set depends on where the
/// opponent actually is. Without this coupling the two players never interact and the ladder
/// is flat for a reason that has nothing to do with the game.
///
/// Note which way round it goes. The threshold is `1 / dim`, so a pursuer spread *uniformly*
/// sits exactly at it and absorbs nothing: a field thinned across the whole window threatens
/// no particular cell. A concentrated pursuer absorbs the cells it occupies. The first
/// version of this test asserted the opposite and was wrong about the physics, not about the
/// code.
#[test]
fn the_safe_set_follows_where_the_pursuer_is() {
    let n = 3;
    let dim = 1usize << n;
    let hunted = Player::new(basis(n, 0), vec![PauliString::single(0, Pauli::X)], 8);

    let spread = StateVec::from_amplitudes(vec![1.0 / (dim as f64).sqrt(); dim], vec![0.0; dim]);
    let thin = Game::new(
        [
            hunted.clone(),
            Player::new(spread, vec![PauliString::single(0, Pauli::Z)], 8),
        ],
        (0..dim).collect(),
        2,
    );
    assert_eq!(
        thin.safe_for(0).len(),
        dim,
        "a uniformly spread pursuer sits at the threshold and absorbs nothing"
    );

    // Concentrate it onto two cells and those two become absorbing.
    let mut re = vec![0.0; dim];
    re[3] = std::f64::consts::FRAC_1_SQRT_2;
    re[5] = std::f64::consts::FRAC_1_SQRT_2;
    let sharp = Game::new(
        [
            hunted,
            Player::new(
                StateVec::from_amplitudes(re, vec![0.0; dim]),
                vec![PauliString::single(0, Pauli::Z)],
                8,
            ),
        ],
        (0..dim).collect(),
        2,
    );
    let safe = sharp.safe_for(0);
    assert_eq!(safe.len(), dim - 2, "two occupied cells should absorb");
    assert!(!safe.contains(&3) && !safe.contains(&5));
}

/// Measuring collapses the field and renormalises it.
#[test]
fn measuring_collapses_and_renormalises() {
    let n = 3;
    let dim = 1usize << n;
    let mut psi = StateVec::from_amplitudes(vec![1.0 / (dim as f64).sqrt(); dim], vec![0.0; dim]);
    overtone_orbit::game::collapse(&mut psi, 0);
    assert!((psi.norm_sqr() - 1.0).abs() < 1e-12, "norm not restored");
    let live = (0..dim)
        .filter(|i| {
            let a = psi.amp(*i);
            a.re * a.re + a.im * a.im > 1e-12
        })
        .count();
    assert_eq!(live, dim / 2, "collapse did not halve the support");
}
