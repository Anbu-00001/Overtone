//! Q8: the memo table for the algebraic layer.

use overtone_lie::PauliString;
use overtone_orbit::language::Agent;
use overtone_orbit::memo::{fingerprint, Memo};
use overtone_orbit::search::choose_with_stats;
use overtone_sim::Pauli;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const STOAT: &str = include_str!("../../../agents/stoat.toml");
const KESTREL: &str = include_str!("../../../agents/kestrel.toml");

#[test]
fn a_hand_is_a_set_and_the_fingerprint_knows_it() {
    let x0 = PauliString::single(0, Pauli::X);
    let z1 = PauliString::single(1, Pauli::Z);
    // Order-independence is the whole reason the table hits: two players who acquired the same
    // generators in a different order have the same algebra.
    assert_eq!(fingerprint(&[x0, z1]), fingerprint(&[z1, x0]));
    assert_eq!(fingerprint(&[x0, z1]), fingerprint(&[x0, z1, x0]));
    assert_ne!(fingerprint(&[x0, z1]), fingerprint(&[x0]));
    assert_ne!(fingerprint(&[x0]), fingerprint(&[z1]));
    assert_eq!(fingerprint(&[]), fingerprint(&[]));
}

#[test]
fn the_same_hand_returns_the_same_object() {
    let mut memo = Memo::new();
    let hand = vec![
        PauliString::single(0, Pauli::X),
        PauliString::single(1, Pauli::X),
    ];
    let a = memo.get(&hand, 3);
    let b = memo.get(&[hand[1], hand[0]], 3);
    assert!(std::sync::Arc::ptr_eq(&a, &b), "reordering caused a miss");
    assert_eq!(memo.hits(), 1);
    assert_eq!(memo.misses(), 1);
    assert_eq!(memo.distinct(), 1);
}

#[test]
fn the_memo_pays_for_the_sibling_searches_and_not_for_mcts() {
    // The finding, pinned. greedy and negamax evaluate siblings of one position and the
    // rotation angle does not change the hand, so eight candidates share an algebra. MCTS
    // evaluates only at rollout leaves, and a rollout reaches a hand no other leaf has -- which
    // is what a rollout is for, so a cache cannot help it.
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let game = overtone_orbit::ladder::opening(4, 24, 2, &mut rng);

    let greedy = Agent::parse(STOAT).unwrap();
    let (_, gs) = choose_with_stats(&greedy, &game, &mut ChaCha8Rng::seed_from_u64(1));
    let lookups = gs.memo_hits + gs.memo_misses;
    assert!(lookups > 0, "greedy made no lookups");
    assert!(
        gs.memo_hits as f64 / lookups as f64 > 0.9,
        "greedy hit rate {} of {lookups}",
        gs.memo_hits
    );
    assert!(gs.memo_misses <= 4, "{} distinct hands", gs.memo_misses);

    let mcts = Agent::parse(KESTREL).unwrap();
    let (_, ms) = choose_with_stats(&mcts, &game, &mut ChaCha8Rng::seed_from_u64(1));
    assert!(ms.memo_hits + ms.memo_misses > 0, "mcts made no lookups");
    assert_eq!(
        ms.memo_hits, 0,
        "mcts hit the memo -- if this ever becomes non-zero the rollout has stopped exploring"
    );
}

#[test]
fn memoising_does_not_change_what_the_eval_says() {
    // The correctness half. A cache that changes an answer is not a cache.
    let a = Agent::parse(KESTREL).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let game = overtone_orbit::ladder::opening(4, 24, 2, &mut rng);
    let mut memo = Memo::new();
    for player in 0..2 {
        let fresh = a.eval.raw(&game, player);
        let cached = a.eval.raw_with(&game, player, &mut memo);
        assert_eq!(fresh, cached);
        let again = a.eval.raw_with(&game, player, &mut memo);
        assert_eq!(fresh, again);
    }
    assert!(memo.hits() > 0, "the second pass should have hit");
}

#[test]
fn brownes_formula_is_implemented_as_published() {
    use overtone_orbit::trace::Trace;
    // Worked by hand from Browne (CoG 2022) Algorithm 1, so the code is checked against the
    // paper rather than against itself.
    //
    // scores = [0.9, 0.2, 0.4, 0.6]. The regression runs from the second match on, over the
    // points (2, 0.2), (3, 0.4), (4, 0.6): slope 0.2, intercept -0.2. y = f(5) = 0.8.
    // A = 0.81 + 0.04 + 0.16 + 0.36 = 1.37 over all four matches.
    // ST = 0.8 + 0.2 * 1.37 = 1.074.
    let t = Trace::of(&[0.9, 0.2, 0.4, 0.6]);
    assert!((t.slope - 0.2).abs() < 1e-12, "slope {}", t.slope);
    assert!((t.y - 0.8).abs() < 1e-12, "y {}", t.y);
    assert!((t.area - 1.37).abs() < 1e-12, "A {}", t.area);
    assert!((t.value - 1.074).abs() < 1e-12, "ST {}", t.value);
    // And the property the comparison chart has to carry: ST is not bounded by 1.
    assert!(t.value > 1.0);

    // Negative scores contribute nothing to the area -- max(0, s).
    let u = Trace::of(&[0.0, -0.5, -0.5, -0.5]);
    assert_eq!(u.area, 0.0);
    assert_eq!(u.value, u.y);

    // ST grows with ladder length at the same per-match score, which is why two ST values
    // measured over ladders of different length are not the same quantity.
    let short = Trace::of(&[0.3, 0.3, 0.3]);
    let long = Trace::of(&[0.3, 0.3, 0.3, 0.3, 0.3, 0.3]);
    assert!(long.area > short.area);
    assert!(long.value > short.value);
}

#[test]
fn the_grid_measures_every_pair_and_is_order_independent() {
    use overtone_orbit::trace::{Grid, Setting};
    let a = Agent::parse(STOAT).unwrap();
    let set = Setting {
        n: 4,
        coherence: 8,
        k: 2,
    };
    let one = Grid::measure(&a, set, &[4, 8, 16], 6, 99, 1);
    let many = Grid::measure(&a, set, &[4, 8, 16], 6, 99, 4);
    assert_eq!(one.pairings.len(), 3, "three pairs from three budgets");
    // Part VIII 6: same spec, same seed, same result -- including across thread counts.
    for (p, q) in one.pairings.iter().zip(many.pairings.iter()) {
        assert_eq!((p.weak, p.strong), (q.weak, q.strong));
        assert_eq!((p.wins, p.draws, p.losses), (q.wins, q.draws, q.losses));
    }
    assert_eq!(Grid::games_for(10.0), 1600);
    assert_eq!(Grid::games_for(5.0), 6400);
}
