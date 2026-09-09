//! M39a: is there a game here at all, at this width?
//!
//! Decisions-06 Q17 reverses the question the ladder was about to ask. Before measuring depth
//! at a width, find out whether the width has anything to measure — and the diagnostic needs
//! no games at all, only position evaluation.
//!
//! # The answer, and a correction to what this repository said an hour earlier
//!
//! **The field is not cold. It starts cold and heats up, which is what a temperature is for.**
//!
//! ```text
//! fraction of regions hot, by ply
//!   n = 4   0.00 0.04 0.04 0.21 0.21 0.33 0.46 0.62 0.67 0.75 0.71 0.71
//!   n = 5   0.00 0.00 0.00 0.00 0.00 0.04 0.08 0.21 0.29 0.33 0.42 0.54
//!   n = 6   0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.17 0.21 0.33 0.54 0.67
//! ```
//!
//! and the hotness at depth is real rather than numerical: median gap `0.25` to `0.875` at ply
//! 8 to 11, against a floor of `1e-12`.
//!
//! Phase 15 reported the field as *"uniformly -1 at every width the ladder runs at"* and drew
//! two conclusions from it -- that temperature is a near-binary indicator, and that Phase 12's
//! sibling spread of `1.207` was one outlier over a mean of `-1`. **Both were wrong, and for
//! one reason: the sample was the opening.** Five plies of one seed is exactly the cold phase.
//! `freeze.rs` walks four plies in, so its measurement sits in the same phase, and the 1.207
//! stands better than the correction did.
//!
//! The lesson is not about temperature. A quantity measured only where a game begins will look
//! like whatever beginnings look like.
//!
//! # What temperature = -1 actually means
//!
//! In combinatorial game theory a game whose value is a **number** has temperature `-1`, the
//! minimum. That is not a weak reading. A number is a position in which *neither player wants
//! to move*: no urgency, no contest, nothing to decide, and optimal play in a sum of numbers
//! is arithmetic rather than strategy.
//!
//! So a field that is identically `-1` is a strong statement — every subgame in the
//! decomposition is resolved and the decomposition contains no decisions. And it carries a
//! hypothesis worth testing directly:
//!
//! Decisions-06 Q16 raised the hypothesis that this was the same fact as Phase 9's `d`
//! saturating at `n = 4` -- a game whose components are all numbers has no tactical content, so
//! no search-language enrichment could extend its ladder. **The measurement does not support
//! it.** `n = 4` is 46% hot by ply 6 and 75% hot by ply 9, so whatever saturates `d` there, it
//! is not an absence of decisions. That is a clean negative on a specific hypothesis, and it is
//! worth recording as one.
//!
//! What the sweep did turn up instead is in [`standing_value`]: **`absorbed_weight` reads
//! exactly zero for both players** in every sampled position past the opening. The standing
//! score is degenerate, and every bit of the contest lives in what a trial move would change.
//! An evaluation with nothing to grade a position by until it plays a move is a better
//! candidate explanation for a short ladder than coldness was.
//!
//! # The gap this measures, and the one it exposes
//!
//! [`Sweep`] reports more than the fraction of regions that are cold, because "all cold" has at
//! least three causes and they need different fixes:
//!
//! ```text
//! left - right   far below zero   the region is decisively resolved
//!                just below zero  the region is one hair from contested
//!                above zero       hot: a switch, and whoever moves gains
//! ```
//!
//! A field that is cold *by a hair* everywhere says the decomposition is nearly right and the
//! move set is too weak. A field that is cold by a mile says the decomposition is looking in
//! the wrong place.
//!
//! # The decomposition is not the one Part IX justified
//!
//! Worth stating plainly, because it is the most likely explanation and it was invisible until
//! Decisions-06 Q16 asked. Part IX 5.3 justifies decomposition by saying the board splits into
//! weakly-interacting regions *"which the maze's corridor structure provides naturally"*.
//! **Corridors are spatial. [`crate::thermal::regions`] splits contiguous qubit blocks.**
//!
//! Those are not the same object and cannot be made into it: Orbit's cells are basis indices
//! `0 .. 2^n`, and a generator on qubits `{0, 1}` changes the amplitude on *every* cell. So the
//! game does not decompose spatially at all, and a qubit-block decomposition is the only one
//! available rather than the one the spec argued for.
//!
//! That also re-reads a Phase 12 result. `interaction_leak` measured `0.0000` and was reported
//! as the decomposition being exact. It is — but sums of numbers are exact trivially, so the
//! leak test passed for a degenerate reason. A decomposition into all-numbers cannot leak.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::game::Game;
use crate::ladder::{opening, Strategy, ANGLES};
use crate::thermal::{region_options, regions};

/// Below this, a gap is not a difference the engine can resolve.
///
/// Absorbed weight is a sum of `|amp|^2`, and `scripts/wasm_determinism.sh` measures
/// native-versus-wasm amplitude agreement at about `5.6e-16`, so weights agree to order
/// `1e-15`. Measured gaps come in two populations: a real one at `1e-1` to `1e0`, and dust
/// running from `1e-32` down to `1e-96`. A bare `left > right` test counts the dust as contest
/// -- at `n = 6` ply 8 it reports 50% hot with a median gap of `6.7e-32`, which is 50% of
/// nothing. This floor is what separates them, and it is derived from the engine's own
/// reproducibility rather than chosen to make a number look better.
pub const HOT_FLOOR: f64 = 1e-12;

/// One region's reading in one position.
#[derive(Clone, Copy, Debug)]
pub struct Reading {
    /// Best value the player to move can reach inside the region.
    pub left: f64,
    /// Worst the opponent can inflict, moving inside the region.
    pub right: f64,
}

impl Reading {
    /// `left - right`. Positive is a switch, and the region is hot.
    pub fn gap(&self) -> f64 {
        self.left - self.right
    }

    /// Hot means `left > right` by more than the engine can resolve: whoever moves gains, so
    /// there is something to decide. See [`HOT_FLOOR`] for why the margin is not optional.
    pub fn is_hot(&self) -> bool {
        self.left - self.right > HOT_FLOOR
    }

    /// The temperature the region presents, on the same convention as [`crate::thermal`].
    pub fn temperature(&self) -> f64 {
        if self.is_hot() {
            (self.left - self.right) / 2.0
        } else {
            -1.0
        }
    }
}

/// Every region reading over a sample of positions at one width and granularity.
#[derive(Clone, Debug)]
pub struct Sweep {
    pub n: usize,
    pub count: usize,
    pub readings: Vec<Reading>,
    /// Regions that had no legal move inside them, which are not readings at all.
    pub empty: usize,
}

impl Sweep {
    /// Evaluate every region of every sampled position.
    pub fn measure(n: usize, count: usize, positions: &[Game], angles: &[f64]) -> Sweep {
        let rs = regions(n, count);
        let mut readings = Vec::new();
        let mut empty = 0;
        for game in positions {
            for r in &rs {
                match region_options(game, r, angles) {
                    Some((left, right)) => readings.push(Reading { left, right }),
                    None => empty += 1,
                }
            }
        }
        Sweep {
            n,
            count,
            readings,
            empty,
        }
    }

    /// Fraction of readings that are hot. Zero means the decomposition contains no decisions.
    pub fn hot_fraction(&self) -> f64 {
        if self.readings.is_empty() {
            return 0.0;
        }
        self.readings.iter().filter(|r| r.is_hot()).count() as f64 / self.readings.len() as f64
    }

    /// Mean of `left - right`. The sign says hot or cold; the magnitude says by how much.
    pub fn mean_gap(&self) -> f64 {
        if self.readings.is_empty() {
            return 0.0;
        }
        self.readings.iter().map(|r| r.gap()).sum::<f64>() / self.readings.len() as f64
    }

    /// The largest gap seen — how close the coldest configuration ever came to contested.
    pub fn best_gap(&self) -> f64 {
        self.readings
            .iter()
            .map(|r| r.gap())
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Spread of `left` across readings. A decomposition whose options never move is a
    /// different failure from one whose options move together.
    pub fn left_spread(&self) -> f64 {
        let (lo, hi) = self
            .readings
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), r| {
                (lo.min(r.left), hi.max(r.left))
            });
        if hi < lo {
            0.0
        } else {
            hi - lo
        }
    }
}

/// Positions sampled along real games, not from the opening alone.
///
/// The opening is one position and a sweep over it would measure the opening. These are taken
/// every ply of a short self-play game at a small budget, which is the same generator
/// `freeze.rs` used so the two measurements are on comparable positions.
pub fn sample_positions(
    n: usize,
    coherence: usize,
    k: usize,
    plies: usize,
    games: usize,
    seed: u64,
) -> Vec<Game> {
    let mut out = Vec::new();
    for g in 0..games {
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ g as u64);
        let mut game = opening(n, coherence, k, &mut rng);
        let strategy = Strategy { budget: 6 };
        let mut choose = ChaCha8Rng::seed_from_u64(seed ^ (g as u64).wrapping_mul(0x9e37_79b9));
        for _ in 0..plies {
            out.push(game.clone());
            let (mv, a) = strategy.choose(&game, &mut choose);
            game.apply(&mv, a);
        }
    }
    out
}

/// Mean absorbed weight over a sample, for each player.
///
/// The sweep's other finding, and the more useful one. Past the opening this is **exactly
/// zero** for both players at every width tried, so the game's own standing score cannot rank
/// two positions at all -- the whole contest is in what a trial move would change. An
/// evaluation with nothing to grade a position by until it plays is a better candidate for a
/// short ladder than an absence of decisions was.
pub fn standing_value(positions: &[Game]) -> (f64, f64) {
    if positions.is_empty() {
        return (0.0, 0.0);
    }
    let n = positions.len() as f64;
    let a: f64 = positions
        .iter()
        .map(|g| g.position(0).absorbed_weight())
        .sum();
    let b: f64 = positions
        .iter()
        .map(|g| g.position(1).absorbed_weight())
        .sum();
    (a / n, b / n)
}

/// The default angle grid, for callers that do not want to reach into the ladder.
pub fn angles() -> Vec<f64> {
    ANGLES.to_vec()
}
