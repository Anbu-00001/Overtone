//! What decomposition search buys, and where its ordering rule actually breaks.

use overtone_cgt::{find_disagreement, hottest_first, hottest_index, stop, temperatures, Game};

/// The counterexample the search finds, written out so the test does not depend on the
/// search finding it again. Part IX 5.1 says "Optimal play is to move in the hottest
/// region"; here Left's correct move is in the *colder* one.
fn counterexample() -> Vec<Game> {
    vec![
        Game::switch(0.0, -3.0),
        Game::new(vec![Game::switch(1.0, -2.0)], vec![Game::number(-3.0)]),
    ]
}

#[test]
fn hottest_first_is_not_optimal() {
    let c = counterexample();
    let t = temperatures(&c);
    assert_eq!(
        t,
        vec![1.5, 1.0],
        "temperatures are distinct, so this is not a tie-break gap"
    );
    assert_eq!(hottest_index(&c), Some(0));
    // Left to move. Optimal is -2; playing the hottest region first gives -3.
    assert_eq!(stop(&c, true), -2.0);
    assert_eq!(hottest_first(&c, true, true), -3.0);
}

#[test]
fn the_colder_region_is_urgent_because_its_option_is_hot() {
    // The mechanism, not just the number. Component 1 is colder (1.0 < 1.5), but its left
    // option is a switch of temperature 1.5 -- the heat is one move down, where the
    // temperature of the region as a whole does not show it.
    let c = counterexample();
    assert_eq!(c[1].temperature(), 1.0);
    assert_eq!(c[1].left_options()[0].temperature(), 1.5);
    assert!(c[1].left_options()[0].temperature() > c[1].temperature());
    assert!(
        c.iter().all(Game::is_hot),
        "both components are hot all the way down"
    );
}

#[test]
fn the_search_finds_it_from_a_generated_catalogue() {
    let n = Game::number;
    let mut cat = Vec::new();
    for a in -1..=2 {
        for b in -3..=0 {
            if a >= b {
                cat.push(Game::switch(a as f64, b as f64));
            }
        }
    }
    for a in 0..=1 {
        for b in -2..=-1 {
            for c in -3..=-2 {
                if a >= b {
                    cat.push(Game::new(
                        vec![Game::switch(a as f64, b as f64)],
                        vec![n(c as f64)],
                    ));
                }
            }
        }
    }
    cat.retain(|g| g.is_hot());
    let found = find_disagreement(&cat, 2).expect("a disagreement exists in this catalogue");
    assert!(found.loss() >= 1.0, "loss {}", found.loss());
    assert_eq!(found.components.len(), 2);
}

#[test]
fn hottest_first_is_optimal_on_sums_of_plain_switches() {
    // The rule is not worthless -- this is the case the theory actually covers. Every sum of
    // simple switches with distinct temperatures is played correctly by the greedy rule.
    let mut cat = Vec::new();
    for a in 0..=3 {
        for b in -3..=0 {
            cat.push(Game::switch(a as f64, b as f64));
        }
    }
    cat.retain(|g| g.is_hot());
    assert!(cat.len() > 8);
    assert!(
        find_disagreement(&cat, 3).is_none(),
        "hottest-first should be exact on sums of plain switches"
    );
}

#[test]
fn a_settled_position_scores_the_sum_and_has_no_hottest_region() {
    let c = vec![Game::number(1.5), Game::number(-0.5)];
    assert_eq!(stop(&c, true), 1.0);
    assert_eq!(stop(&c, false), 1.0);
    assert_eq!(hottest_index(&c), None);
    assert_eq!(temperatures(&c), vec![-1.0, -1.0]);
}

#[test]
fn decomposition_beats_the_undecomposed_move_count() {
    // The point of 6: you do not look at N x N moves together. A sum of k components each
    // with m options has k*m moves at the root, not m^k.
    let c: Vec<Game> = (0..4)
        .map(|i| Game::switch(i as f64, -(i as f64) - 1.0))
        .collect();
    let root_moves: usize = c.iter().map(|g| g.left_options().len()).sum();
    assert_eq!(root_moves, 4);
    assert!(stop(&c, true).is_finite());
}
