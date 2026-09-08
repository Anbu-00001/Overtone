//! Decomposition search, and the check on the rule that motivates it.
//!
//! Muller's decomposition search is the computational half of Part IX 5: split a position
//! into independent regions, evaluate each, combine by CGT addition. The combining step is
//! what makes an `N x N` move set tractable -- you do not look at `N^2` moves together, you
//! prove most of them are independent and stop looking at them together.
//!
//! The rule that makes the split *useful* is "move in the hottest region", and Part IX 5.1
//! states it as though it were a theorem. It is not, and [`find_disagreement`] goes looking
//! for the counterexample rather than taking the sentence on trust. What the theory actually
//! gives is a bound, not an identity: hottest-first is optimal for sums of simple switches,
//! and elsewhere it can lose, by at most about the ambient temperature. The interesting case
//! is a *tepid* component whose temperature goes **up** when it is played -- Blom's
//! `{0 | {0 | -2}}`, where Right's only move raises the temperature from 0 to 1. Greedy
//! ordering assumes the heat it can see is all the heat there is.
//!
//! # Scores
//!
//! Play stops when every component is a number, and the score is their sum -- the stopping
//! formulation, which is the right one for a scored game like Go or Overtone and unlike the
//! last-player-wins convention of impartial theory. [`stop`] is the exact value under optimal
//! play; [`hottest_first`] is the value the greedy rule actually achieves.

use crate::Game;

/// The index of the hottest component, or `None` if every component is settled.
pub fn hottest_index(components: &[Game]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, g) in components.iter().enumerate() {
        if g.is_number() {
            continue;
        }
        let t = g.temperature();
        if best.map_or(true, |(_, bt)| t > bt) {
            best = Some((i, t));
        }
    }
    best.map(|(i, _)| i)
}

/// Every component's temperature, with `-1` for the settled ones. This is the field Part
/// IX 5.3 wants to draw: one number per region, derived, not tuned.
pub fn temperatures(components: &[Game]) -> Vec<f64> {
    components.iter().map(Game::temperature).collect()
}

fn settled(components: &[Game]) -> Option<f64> {
    components.iter().map(Game::value).sum()
}

/// The score under optimal play by both sides, by exhaustive minimax over the sum.
pub fn stop(components: &[Game], left_to_move: bool) -> f64 {
    if let Some(total) = settled(components) {
        return total;
    }
    let mut best = if left_to_move {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    for i in 0..components.len() {
        let options = if left_to_move {
            components[i].left_options()
        } else {
            components[i].right_options()
        };
        for opt in options {
            let mut next = components.to_vec();
            next[i] = opt.clone();
            let v = stop(&next, !left_to_move);
            best = if left_to_move {
                best.max(v)
            } else {
                best.min(v)
            };
        }
    }
    best
}

/// The score when the side named plays hottest-region-first and the *opponent* plays
/// optimally. Both play locally-best options within the chosen region.
///
/// Scoring the greedy player against an optimal opponent is the honest comparison: a greedy
/// rule that only loses to another greedy rule has not been tested.
pub fn hottest_first(components: &[Game], left_to_move: bool, greedy_is_left: bool) -> f64 {
    if let Some(total) = settled(components) {
        return total;
    }
    if left_to_move != greedy_is_left {
        let mut best = if left_to_move {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        for i in 0..components.len() {
            let options = if left_to_move {
                components[i].left_options()
            } else {
                components[i].right_options()
            };
            for opt in options {
                let mut next = components.to_vec();
                next[i] = opt.clone();
                let v = hottest_first(&next, !left_to_move, greedy_is_left);
                best = if left_to_move {
                    best.max(v)
                } else {
                    best.min(v)
                };
            }
        }
        return best;
    }
    let i = hottest_index(components).expect("some component is unsettled");
    let options = if left_to_move {
        components[i].left_options()
    } else {
        components[i].right_options()
    };
    // Within the hottest region, take the locally best option by mean value -- the region is
    // all the greedy player is looking at, so the mean is all it has to go on.
    let mut chosen = &options[0];
    for opt in options {
        let better = if left_to_move {
            opt.mean() > chosen.mean()
        } else {
            opt.mean() < chosen.mean()
        };
        if better {
            chosen = opt;
        }
    }
    let mut next = components.to_vec();
    next[i] = chosen.clone();
    hottest_first(&next, !left_to_move, greedy_is_left)
}

/// A position where hottest-first play does not achieve the optimal stop.
#[derive(Clone, Debug)]
pub struct Disagreement {
    /// The independent components of the position.
    pub components: Vec<Game>,
    /// Whether Left moves first.
    pub left_to_move: bool,
    /// The score under optimal play.
    pub optimal: f64,
    /// The score the greedy rule achieves against an optimal opponent.
    pub greedy: f64,
    /// The temperatures the greedy rule was ordering by.
    pub temperatures: Vec<f64>,
}

impl Disagreement {
    /// How much the greedy rule gave away, in points, always non-negative.
    pub fn loss(&self) -> f64 {
        if self.left_to_move {
            self.optimal - self.greedy
        } else {
            self.greedy - self.optimal
        }
    }
}

/// Exhaustively search sums of up to `max_components` games drawn from `catalogue` for a
/// position where hottest-first loses points against optimal play, returning the smallest
/// such position found, measured in total tree nodes and breaking ties by largest loss.
///
/// Two filters keep the answer honest, and both were added because the first version of this
/// search returned nonsense:
///
/// * every catalogue entry must be [`Game::is_hot`]. Without it the search happily returns
///   positions built from games that are *equal to numbers* by the simplicity rule -- the
///   first "counterexample" it found was `{{-1|-2} | 1}`, which is a number, and this crate
///   reports a meaningless temperature for those by design.
/// * component temperatures must be pairwise distinct. "Move in the hottest region" says
///   nothing about ties, so a loss that only appears when two regions are equally hot is a
///   gap in the statement, not a refutation of it.
///
/// # Panics
///
/// If any catalogue entry is not hot all the way down.
pub fn find_disagreement(catalogue: &[Game], max_components: usize) -> Option<Disagreement> {
    for g in catalogue {
        assert!(
            g.is_hot(),
            "catalogue entry is not hot all the way down: see Game::is_hot"
        );
    }
    let mut best: Option<(usize, f64, Disagreement)> = None;
    for k in 2..=max_components {
        let mut idx = vec![0usize; k];
        'sums: loop {
            // Only non-decreasing index tuples: the sum is commutative.
            if idx.windows(2).all(|w| w[0] <= w[1]) {
                let components: Vec<Game> = idx.iter().map(|&i| catalogue[i].clone()).collect();
                let temps = temperatures(&components);
                let distinct = (0..temps.len()).all(|i| {
                    (0..temps.len()).all(|j| i == j || (temps[i] - temps[j]).abs() > 1e-9)
                });
                if distinct {
                    for &left_to_move in &[true, false] {
                        let optimal = stop(&components, left_to_move);
                        let greedy = hottest_first(&components, left_to_move, left_to_move);
                        let loss = if left_to_move {
                            optimal - greedy
                        } else {
                            greedy - optimal
                        };
                        if loss > 1e-9 {
                            let size: usize = components.iter().map(Game::nodes).sum();
                            let better = best
                                .as_ref()
                                .map_or(true, |b| size < b.0 || (size == b.0 && loss > b.1));
                            if better {
                                best = Some((
                                    size,
                                    loss,
                                    Disagreement {
                                        temperatures: temps.clone(),
                                        components: components.clone(),
                                        left_to_move,
                                        optimal,
                                        greedy,
                                    },
                                ));
                            }
                        }
                    }
                }
            }
            let mut w = k;
            loop {
                if w == 0 {
                    break 'sums;
                }
                w -= 1;
                idx[w] += 1;
                if idx[w] < catalogue.len() {
                    break;
                }
                idx[w] = 0;
            }
        }
    }
    best.map(|(_, _, d)| d)
}
