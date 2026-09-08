//! M51: two temperatures, measured side by side.
//!
//! Part IX 5.4 asks the project's characteristic question one more time. Combinatorial game
//! theory borrowed "temperature" from thermodynamics as a *metaphor*: it is the urgency of
//! the next move, the base of a thermograph's mast. Overtone has actual thermodynamics --
//! decoherence, entropy production, an arrow of time. Same word, one metaphorical and one
//! literal. Do they track each other?
//!
//! Part IX 8's trap is explicit that this must be measured and not assumed: "Do not claim CGT
//! temperature and physical temperature are the same thing. They share a name and possibly a
//! behaviour. Measure it; report what you find; do not assume." So this module measures both
//! along a real trajectory and hands back the rank correlation, whatever it is.
//!
//! # Where the numbers come from
//!
//! Neither temperature is invented, which is the whole point of Part VII's rule against
//! tuned evaluation.
//!
//! * The **CGT temperature** of a region is `(a - b)/2` where `a` and `b` are the position's
//!   *own* score -- Part VII's `opponent absorbed weight minus own absorbed weight` -- after
//!   the best move confined to that region by each side in turn. The score function was
//!   already there; thermography turns it into an ordering.
//! * The **physical entropy** is the half-chain von Neumann entropy of the mover's state,
//!   from `overtone-spec`.
//!
//! # The boundary, again
//!
//! Temperature is defined for games of no chance (see the `overtone-cgt` crate docs), so
//! every temperature here is a temperature of the coherent segment. [`Trace`] records where
//! the measurements were taken so a collapse cannot be silently averaged across.

use crate::game::{legal_moves, Game, Move};
use overtone_cgt::Game as Cgt;
use overtone_spec::entropy::half_chain_entropy;
use overtone_spec::spearman;

/// A region of the window: the qubits a move may target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    /// The qubits in this region.
    pub qubits: Vec<usize>,
}

impl Region {
    /// Does every target of this move lie inside the region?
    pub fn contains_move(&self, mv: &Move) -> bool {
        match mv {
            Move::Apply { targets, .. } => targets.iter().all(|t| self.qubits.contains(t)),
            _ => false,
        }
    }
}

/// Split `n` qubits into `count` contiguous regions, as evenly as they divide.
///
/// Part IX 5.3 notes that decomposition needs the board to split into weakly-interacting
/// pieces, which a maze's corridor structure provides. Contiguity is the cheap stand-in for
/// that here, and [`interaction_leak`] measures how badly it holds rather than assuming it.
pub fn regions(n: usize, count: usize) -> Vec<Region> {
    assert!(
        count > 0 && count <= n,
        "cannot split {n} qubits into {count} regions"
    );
    let mut out = Vec::with_capacity(count);
    let mut start = 0;
    for r in 0..count {
        let len = (n - start) / (count - r);
        out.push(Region {
            qubits: (start..start + len).collect(),
        });
        start += len;
    }
    out
}

/// The game's own score, from the point of view of `me`, after a trial move.
fn value_to(game: &Game, me: usize) -> f64 {
    game.position(me ^ 1).absorbed_weight() - game.position(me).absorbed_weight()
}

/// The best value either side can reach by a move confined to `region`, in the units of the
/// player to move: `(left, right)`.
pub fn region_options(game: &Game, region: &Region, angles: &[f64]) -> Option<(f64, f64)> {
    let n = game.players[game.to_move].num_qubits;
    let me = game.to_move;
    let moves: Vec<Move> = legal_moves(n)
        .into_iter()
        .filter(|m| region.contains_move(m))
        .collect();
    if moves.is_empty() {
        return None;
    }
    let mut left = f64::NEG_INFINITY;
    let mut right = f64::INFINITY;
    for mv in &moves {
        for &angle in angles {
            let mut trial = game.clone();
            trial.to_move = me;
            trial.apply(mv, angle);
            left = left.max(value_to(&trial, me));

            let mut trial = game.clone();
            trial.to_move = me ^ 1;
            trial.apply(mv, angle);
            right = right.min(value_to(&trial, me));
        }
    }
    Some((left, right))
}

/// The combinatorial game a region presents to the player to move.
///
/// A region where the mover cannot reach as much as the opponent would concede is *settled*
/// -- nobody gains by moving first -- and is returned as a number, which is what the
/// simplicity rule says it is.
pub fn region_game(game: &Game, region: &Region, angles: &[f64]) -> Cgt {
    match region_options(game, region, angles) {
        None => Cgt::number(0.0),
        Some((a, b)) if a > b => Cgt::switch(a, b),
        Some((a, b)) => Cgt::number((a + b) / 2.0),
    }
}

/// One temperature per region: the field Part IX 5.3 wants to draw.
pub fn temperature_field(game: &Game, regions: &[Region], angles: &[f64]) -> Vec<f64> {
    regions
        .iter()
        .map(|r| region_game(game, r, angles).temperature())
        .collect()
}

/// The hottest region's temperature -- the ambient temperature of the position.
pub fn ambient_temperature(game: &Game, regions: &[Region], angles: &[f64]) -> f64 {
    temperature_field(game, regions, angles)
        .into_iter()
        .fold(f64::NEG_INFINITY, f64::max)
}

/// One turn's worth of both temperatures.
#[derive(Clone, Copy, Debug)]
pub struct Reading {
    /// Ply index within the coherent segment.
    pub ply: usize,
    /// Hottest region temperature, in the game's own score units.
    pub cgt: f64,
    /// Half-chain von Neumann entropy of the mover's state.
    pub entropy: f64,
}

/// A run of readings taken inside a single coherent segment.
#[derive(Clone, Debug, Default)]
pub struct Trace {
    /// The readings, in order.
    pub readings: Vec<Reading>,
}

impl Trace {
    /// Rank correlation between the two temperatures. `NaN` if either is constant, which is
    /// itself an answer worth printing rather than hiding.
    pub fn correlation(&self) -> f64 {
        let cgt: Vec<f64> = self.readings.iter().map(|r| r.cgt).collect();
        let ent: Vec<f64> = self.readings.iter().map(|r| r.entropy).collect();
        spearman(&cgt, &ent)
    }

    /// How much each series actually moved. A correlation computed over a series that never
    /// varies says nothing, and this is how a reader checks.
    pub fn spread(&self) -> (f64, f64) {
        let range = |v: Vec<f64>| {
            let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            hi - lo
        };
        (
            range(self.readings.iter().map(|r| r.cgt).collect()),
            range(self.readings.iter().map(|r| r.entropy).collect()),
        )
    }
}

/// Play `plies` turns from `game` with the given strategy, recording both temperatures each
/// turn. Stops early if the game ends.
pub fn trace<F>(
    game: &mut Game,
    regions: &[Region],
    angles: &[f64],
    plies: usize,
    mut choose: F,
) -> Trace
where
    F: FnMut(&Game) -> (Move, f64),
{
    let mut out = Trace::default();
    for ply in 0..plies {
        if game.outcome(1e-6).is_some() {
            break;
        }
        let cgt = ambient_temperature(game, regions, angles);
        let entropy = half_chain_entropy(&game.players[game.to_move].state);
        out.readings.push(Reading { ply, cgt, entropy });
        let (mv, angle) = choose(game);
        game.apply(&mv, angle);
    }
    out
}

/// How much the decomposition lies: the gap between the whole position's best move value and
/// the best value found by looking at regions one at a time.
///
/// Zero means the regions really are independent for this position. Part IX 5.3 assumes the
/// corridor structure makes the gap small; this returns the number instead.
pub fn interaction_leak(game: &Game, regions: &[Region], angles: &[f64]) -> f64 {
    let n = game.players[game.to_move].num_qubits;
    let me = game.to_move;
    let mut whole = f64::NEG_INFINITY;
    for mv in legal_moves(n) {
        for &angle in angles {
            let mut trial = game.clone();
            trial.apply(&mv, angle);
            whole = whole.max(value_to(&trial, me));
        }
    }
    let mut split = f64::NEG_INFINITY;
    for r in regions {
        if let Some((a, _)) = region_options(game, r, angles) {
            split = split.max(a);
        }
    }
    if !whole.is_finite() || !split.is_finite() {
        return 0.0;
    }
    whole - split
}
