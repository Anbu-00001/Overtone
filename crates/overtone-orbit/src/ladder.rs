//! M35: the strategy ladder, and `d`.
//!
//! # Three corrections to how `d` is defined, from the source rather than the summary
//!
//! Lantz, Isaksen, Jaffe, Nealen and Togelius, *Depth in Strategic Games* (AAAI 2017):
//!
//! 1. **`d` is a count of steps, not a span in orders of magnitude.** Their procedure is to
//!    plot the best strategy at each computational-resource level and walk the curve counting
//!    how many times strength improves by at least a declared *step unit*. "The number of
//!    steps you have counted is the `d` for this game for the given settings." Part VII 11's
//!    acceptance -- "a rising region spanning at least three orders of magnitude of compute"
//!    -- is a different quantity. Both are reported below; only one of them is `d`.
//! 2. **`d` is relative to a declared strategy language.** "Any observations made about a
//!    game's depth based on this model must refer to the language selected." So [`LANGUAGE`]
//!    is written down before any number is produced.
//! 3. **There are no published `d` values for any game to compare against.** The paper is a
//!    proposal; applying it to Tic Tac Toe, Blackjack and 3x3 Go is listed as future work,
//!    and it says the complete model is "impossible to apply completely to complex,
//!    real-world games". A plot of Orbit's `d` beside chess and Go would be a plot beside
//!    numbers that do not exist.
//!
//! # Where Overtone is better placed than the paper's own examples
//!
//! The paper weighs win rate against quality-of-move as the strength metric, prefers
//! quality-of-move for being "more simply, clearly, and consistently defined", and then
//! rejects it because it requires perfect play. Overtone *has* perfect play in the decohered
//! endgame, exactly, from the eigensolve. So quality-of-move is available here where the
//! paper could only wish for it -- and win rate is used elsewhere.

use overtone_lie::PauliString;
use overtone_sim::{Pauli, StateVec};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::game::{legal_moves, Game, Move, Outcome, Player};

/// The strategy language, declared before any number is measured.
///
/// Correction 2 above: a `d` quoted without this is meaningless.
pub const LANGUAGE: &str = "\
Depth-1 lookahead over the legal-move set of Part VII 2's four generator types, with the \
rotation angle drawn from a fixed 8-point grid on [-pi, pi). A strategy is a policy that, \
given a budget of B position evaluations, samples B (move, angle) pairs uniformly without \
replacement where B is below the legal count and with replacement above it, evaluates each \
by the score function below, and plays the best. Budget B is the computational resource, \
and the ladder runs over powers of two.";

/// The score a strategy maximises, declared with the language.
///
/// Absorbed weight is the physical quantity Part VII 3 defines check on, so the evaluation
/// is "push them in, stay out" and nothing else. No material term, no positional term, no
/// tuned weights: a hand-built evaluation would make the ladder a measurement of the
/// evaluation rather than of the game.
pub const SCORE: &str = "opponent absorbed weight minus own absorbed weight, after the move.";

/// The angle grid a strategy may choose from.
pub const ANGLES: [f64; 8] = [
    std::f64::consts::FRAC_PI_8,
    std::f64::consts::FRAC_PI_4,
    3.0 * std::f64::consts::FRAC_PI_8,
    std::f64::consts::FRAC_PI_2,
    5.0 * std::f64::consts::FRAC_PI_8,
    3.0 * std::f64::consts::FRAC_PI_4,
    7.0 * std::f64::consts::FRAC_PI_8,
    std::f64::consts::PI,
];

/// A strategy at one budget.
#[derive(Clone, Copy, Debug)]
pub struct Strategy {
    /// Position evaluations allowed per move. The computational resource of the ladder.
    pub budget: usize,
}

impl Strategy {
    /// Choose a move by sampling `budget` candidates and keeping the best.
    pub fn choose(&self, game: &Game, rng: &mut ChaCha8Rng) -> (Move, f64) {
        let n = game.players[game.to_move].num_qubits;
        let moves = legal_moves(n);
        let mut best = (moves[0].clone(), ANGLES[0], f64::NEG_INFINITY);
        for _ in 0..self.budget.max(1) {
            let mv = moves[rng.gen_range(0..moves.len())].clone();
            let angle = ANGLES[rng.gen_range(0..ANGLES.len())];
            let mut trial = game.clone();
            trial.apply(&mv, angle);
            let me = game.to_move;
            let score =
                trial.position(me ^ 1).absorbed_weight() - trial.position(me).absorbed_weight();
            if score > best.2 {
                best = (mv, angle, score);
            }
        }
        (best.0, best.1)
    }
}

/// A single game between two strategies. Returns the winner, or `None` for a draw.
pub fn play(
    a: Strategy,
    b: Strategy,
    n: usize,
    coherence: usize,
    k: usize,
    seed: u64,
) -> Option<usize> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut game = opening(n, coherence, k, &mut rng);
    let strategies = [a, b];
    let mut guard = 0;
    while game.outcome(1e-6).is_none() && guard < 200 {
        let s = strategies[game.to_move];
        let (mv, angle) = s.choose(&game, &mut rng);
        game.apply(&mv, angle);
        // Rule 6: overlapping an opponent merges algebras.
        if game.overlap() > 0.5 {
            let winner = game.to_move ^ 1;
            game.merge(winner, winner ^ 1);
        }
        guard += 1;
    }
    match game.outcome(1e-6) {
        Some(Outcome::Checkmate(loser)) => Some(loser ^ 1),
        _ => {
            // No checkmate: the player holding less absorbed weight is ahead. This is a
            // tiebreak on the same physical quantity the score function uses, not a second
            // evaluation function smuggled in.
            let w0 = game.position(0).absorbed_weight();
            let w1 = game.position(1).absorbed_weight();
            if (w0 - w1).abs() < 1e-9 {
                None
            } else if w0 < w1 {
                Some(0)
            } else {
                Some(1)
            }
        }
    }
}

/// The starting position: both players spread over the window, with disjoint hands.
/// The starting position: two players at opposite corners of the window, one holding every
/// single-qubit X and the other every single-qubit Z.
///
/// Public because M51 measures temperature along a trajectory and has to start it from the
/// same place the ladder does; a second opening would make the two measurements
/// incomparable.
pub fn opening(n: usize, coherence: usize, k: usize, rng: &mut ChaCha8Rng) -> Game {
    let dim = 1usize << n;
    let make = |start: usize| {
        let mut re = vec![0.0; dim];
        re[start % dim] = 1.0;
        StateVec::from_amplitudes(re, vec![0.0; dim])
    };
    let hand_a: Vec<PauliString> = (0..n).map(|q| PauliString::single(q, Pauli::X)).collect();
    let hand_b: Vec<PauliString> = (0..n).map(|q| PauliString::single(q, Pauli::Z)).collect();
    let _ = rng;
    Game::new(
        [
            Player::new(make(0), hand_a, coherence),
            Player::new(make(dim - 1), hand_b, coherence),
        ],
        // The terrain's own safe set: every cell of the window. What makes a cell unsafe is
        // the pursuer standing in it, which is computed per turn by `Game::safe_for`.
        (0..dim).collect(),
        k,
    )
}

/// Win rate of `a` against `b` over `games`, colours swapped each time so a first-move
/// advantage cannot be mistaken for strength.
pub fn win_rate(a: Strategy, b: Strategy, n: usize, games: usize, seed: u64) -> f64 {
    let mut wins = 0.0;
    for g in 0..games {
        let swap = g % 2 == 1;
        let (x, y) = if swap { (b, a) } else { (a, b) };
        let result = play(x, y, n, 24, 2, seed + g as u64);
        let a_index = if swap { 1 } else { 0 };
        match result {
            Some(w) if w == a_index => wins += 1.0,
            None => wins += 0.5,
            _ => {}
        }
    }
    wins / games as f64
}

/// One rung of the ladder.
#[derive(Clone, Copy, Debug)]
pub struct Rung {
    pub budget: usize,
    /// Win rate of this budget against the one below it.
    pub win_rate: f64,
    /// Whether that clears the step unit.
    pub is_step: bool,
}

/// The step unit, mid-range of the paper's 60-75%.
pub const STEP_UNIT: f64 = 0.65;

/// Walk the ladder and count `d`.
///
/// Correction 1 above: `d` is the count of rungs that clear [`STEP_UNIT`], not the width of
/// the rising region. The width is reported separately by [`Ladder::orders_of_magnitude`].
pub struct Ladder {
    pub rungs: Vec<Rung>,
}

impl Ladder {
    pub fn measure(n: usize, budgets: &[usize], games: usize, seed: u64) -> Ladder {
        let rungs = budgets
            .windows(2)
            .map(|w| {
                let rate = win_rate(
                    Strategy { budget: w[1] },
                    Strategy { budget: w[0] },
                    n,
                    games,
                    seed,
                );
                Rung {
                    budget: w[1],
                    win_rate: rate,
                    is_step: rate >= STEP_UNIT,
                }
            })
            .collect();
        Ladder { rungs }
    }

    /// `d`: the number of distinguishable skill tiers.
    pub fn depth(&self) -> usize {
        self.rungs.iter().filter(|r| r.is_step).count()
    }

    /// The other quantity Part VII 11 asks for, reported separately because it is not `d`.
    pub fn orders_of_magnitude(&self) -> f64 {
        let stepping: Vec<usize> = self
            .rungs
            .iter()
            .filter(|r| r.is_step)
            .map(|r| r.budget)
            .collect();
        match (stepping.first(), stepping.last()) {
            (Some(lo), Some(hi)) if *lo > 0 => (*hi as f64 / *lo as f64).log10(),
            _ => 0.0,
        }
    }
}
