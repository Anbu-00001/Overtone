//! The seven rules, as code. No interface: Part VII 12 forbids one before `d` is measured.
//!
//! # Pieces are generators
//!
//! Part VII 2's claim is that a chess piece *is* a movement geometry, which is to say the
//! orbit traced by one generator. Position is encoded in binary across `n` qubits, so a
//! `2^n`-cell window is `n` qubits, and the piece taxonomy falls out of the Pauli algebra
//! rather than being imposed on it:
//!
//! | Chess piece | What defines it | Generator here |
//! |---|---|---|
//! | Pawn | one step, cannot reverse | `X_q` -- flips one position bit |
//! | Bishop | cannot change square colour | `X_q X_p` -- flips two, so parity survives |
//! | Knight | jumps, ignoring what lies between | `X_q Z_p` -- a hop carrying a phase |
//! | Rook | acts along an entire line | `Z_q` -- phases the half-space `bit q = 1` |
//! | Queen | rook and bishop together | the **commutator closure** of the two |
//! | King | the thing that must survive | the safe subspace |
//!
//! **The bishop is not an analogy.** Square colour is the parity of the position index, and
//! the operator that reads it is `Z Z ... Z`. A generator commutes with it exactly when it
//! carries an even number of `X` factors, so `X_q X_p` preserves colour and `X_q` does not.
//! A bishop is confined to one colour for the same reason a superselection sector is
//! confined: an operator in the commutant forbids the transition. That commutant is what
//! [`crate::invariant::commutant_basis`] computes, and it is what the checkmate predicate
//! reads.
//!
//! **And the queen is not additive.** `[Z_q, X_q X_p] = 2i Y_q X_p`, a direction present in
//! neither. Chess folklore has said a queen is worth more than a rook plus a bishop for five
//! hundred years; that is the non-additivity of the Lie closure, and it is measured here
//! rather than asserted -- see `examples/ladder.rs`.

use overtone_lie::{closure, Algebra, PauliString};
use overtone_sim::{Pauli, StateVec};

use crate::checkmate::{apply_exponential, Position};

/// The generator types a hand is drawn from. Six, and Part VII 12 says it stays six to
/// eight: chess has six pieces and a library of literature, and rule count is close to
/// unrelated to depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Piece {
    /// `X_q`. One step; changes square colour.
    Pawn,
    /// `Z_q`. Phases an entire half-space.
    Rook,
    /// `X_q X_p`. Two steps at once, so colour is conserved.
    Bishop,
    /// `X_q Z_p`. A hop that carries a phase, and ignores what lies between.
    Knight,
}

impl Piece {
    pub const ALL: [Piece; 4] = [Piece::Pawn, Piece::Rook, Piece::Bishop, Piece::Knight];

    pub fn name(&self) -> &'static str {
        match self {
            Piece::Pawn => "pawn",
            Piece::Rook => "rook",
            Piece::Bishop => "bishop",
            Piece::Knight => "knight",
        }
    }

    /// Whether this piece needs one target or two.
    pub fn arity(&self) -> usize {
        match self {
            Piece::Pawn | Piece::Rook => 1,
            Piece::Bishop | Piece::Knight => 2,
        }
    }

    /// The Pauli string this piece generates on the given targets.
    pub fn generator(&self, targets: &[usize]) -> PauliString {
        match self {
            Piece::Pawn => PauliString::single(targets[0], Pauli::X),
            Piece::Rook => PauliString::single(targets[0], Pauli::Z),
            Piece::Bishop => {
                PauliString::from_factors(&[(targets[0], Pauli::X), (targets[1], Pauli::X)])
            }
            Piece::Knight => {
                PauliString::from_factors(&[(targets[0], Pauli::X), (targets[1], Pauli::Z)])
            }
        }
    }
}

/// One legal move: a generator applied to a region, or a measurement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Move {
    Apply {
        piece: Piece,
        targets: Vec<usize>,
    },
    /// Rule 3's other option. Measuring aims but costs the superposition, which is what
    /// makes zugzwang appear without being designed.
    Measure {
        qubit: usize,
    },
}

impl Move {
    pub fn generator(&self) -> Option<PauliString> {
        match self {
            Move::Apply { piece, targets } => Some(piece.generator(targets)),
            Move::Measure { .. } => None,
        }
    }
}

/// Every legal move at a given window width.
///
/// The branching factor is a *measured* statistic, never a design intention (Part VII 9.2).
/// At `n = 4` this is 26 and at `n = 5` it is 40, which brackets chess's ~35 by construction
/// rather than by tuning -- but the number that gets quoted is the one
/// [`Game::average_branching`] measures over a real game.
pub fn legal_moves(n: usize) -> Vec<Move> {
    let mut out = Vec::new();
    for piece in Piece::ALL {
        match piece.arity() {
            1 => {
                for q in 0..n {
                    out.push(Move::Apply {
                        piece,
                        targets: vec![q],
                    });
                }
            }
            _ => {
                for q in 0..n {
                    for p in 0..n {
                        if p == q {
                            continue;
                        }
                        // A bishop is symmetric in its two targets; a knight is not, because
                        // its X and Z factors sit on different qubits.
                        if piece == Piece::Bishop && p < q {
                            continue;
                        }
                        out.push(Move::Apply {
                            piece,
                            targets: vec![q, p],
                        });
                    }
                }
            }
        }
    }
    for q in 0..n {
        out.push(Move::Measure { qubit: q });
    }
    out
}

/// One player.
#[derive(Clone)]
pub struct Player {
    pub state: StateVec,
    /// The generators held. Rule 2: together they close into `g`.
    pub hand: Vec<PauliString>,
    /// Rule 5: coherence only decreases.
    pub coherence: usize,
    pub num_qubits: usize,
}

impl Player {
    pub fn new(state: StateVec, hand: Vec<PauliString>, coherence: usize) -> Player {
        let num_qubits = state.num_qubits();
        Player {
            state,
            hand,
            coherence,
            num_qubits,
        }
    }

    /// Rule 2: the algebra the hand closes into.
    pub fn algebra(&self) -> Algebra {
        closure(&self.hand, self.num_qubits, 4096)
    }

    /// Rule 5's other half: `dim(g)` only increases, because a hand only ever grows.
    pub fn dim_g(&self) -> usize {
        self.algebra().dim()
    }
}

/// The board: two players on one window, with a shared safe subspace.
#[derive(Clone)]
pub struct Game {
    pub players: [Player; 2],
    pub to_move: usize,
    /// Basis indices that are safe -- the complement of the pursuer's absorbing subspace.
    pub safe: Vec<usize>,
    /// Rule 4: an applied generator evolves for `k` coherent steps.
    pub k: usize,
    pub turns: usize,
    /// Probability above which a pursuer's presence in a cell makes it absorbing.
    ///
    /// Not tuned: it is one over the window size, so "the pursuer is here more than a
    /// uniformly spread field would be" is the condition, and it scales with the board
    /// rather than being a number chosen at one width.
    pub absorb_threshold: f64,
    branching: Vec<usize>,
}

/// What ended a game, if anything.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Rule 7: no unitary in `g` reaches a safe state. The index is the *loser*.
    Checkmate(usize),
    /// Both ran out of coherence with neither proved lost.
    Exhausted,
}

impl Game {
    pub fn new(players: [Player; 2], safe: Vec<usize>, k: usize) -> Game {
        let dim = players[0].state.dim() as f64;
        Game {
            players,
            to_move: 0,
            safe,
            k,
            turns: 0,
            absorb_threshold: 1.0 / dim,
            branching: Vec::new(),
        }
    }

    /// Rule 3: apply one generator to one region, or measure.
    ///
    /// Rule 4: an applied generator evolves for `k` coherent steps, which is one exponential
    /// of angle `k * theta` rather than `k` separate ones -- the flow of a single generator
    /// is a one-parameter group, so those are the same operator and the loop would only cost
    /// time. `k` is Part VI's coherence budget and is the physical parameter that
    /// interpolates between one decision per step and few decisions over long evolutions.
    pub fn apply(&mut self, mv: &Move, angle: f64) {
        let player = &mut self.players[self.to_move];
        match mv {
            Move::Apply { piece, targets } => {
                let g = piece.generator(targets);
                apply_exponential(&mut player.state, &g, angle * self.k as f64);
                if !player.hand.contains(&g) {
                    player.hand.push(g);
                }
                player.coherence = player.coherence.saturating_sub(self.k);
            }
            Move::Measure { qubit } => {
                collapse(&mut player.state, *qubit);
                // Measuring costs the same `k` as applying a generator. An earlier comment
                // here claimed it "costs the whole remaining coherence block" and cited Part
                // VII 4 for it; the code never did that, and Part VI 2.2 does not ask for it
                // either -- it says measuring costs coherence *and* destroys the spread, and
                // the collapse above is that second cost. Charging measurement extra would
                // change match outcomes, which makes it a mechanic under Decisions 01 Q4's
                // test, and an invented one. Left symmetric deliberately.
                //
                // The consequence: coherence falls by `k` every ply whatever is played, so it
                // is a pure function of depth and was cut from the v1 feature vocabulary.
                player.coherence = player.coherence.saturating_sub(self.k);
            }
        }
        self.branching
            .push(legal_moves(self.players[self.to_move].num_qubits).len());
        self.to_move ^= 1;
        self.turns += 1;
    }

    /// Rule 6: overlapping an opponent merges algebras.
    ///
    /// The merge is on the *hands*, not on the closures: closing the union of two hands is
    /// the same algebra as closing the union of two closures, and keeping hands means
    /// `dim(g)` is recomputed from generators rather than accumulated. Rule 5 is then
    /// automatic -- a hand only grows, so its closure only grows.
    pub fn merge(&mut self, a: usize, b: usize) {
        let taken: Vec<PauliString> = self.players[b].hand.clone();
        for g in taken {
            if !self.players[a].hand.contains(&g) {
                self.players[a].hand.push(g);
            }
        }
    }

    /// Overlap between the two amplitude fields: the trigger for rule 6.
    pub fn overlap(&self) -> f64 {
        let (a, b) = (&self.players[0].state, &self.players[1].state);
        let c = a.inner(b);
        c.re * c.re + c.im * c.im
    }

    /// The safe cells for one player: the terrain's own safe set, minus wherever the
    /// **pursuer** has support.
    ///
    /// Part VII 3 says check is "overlap with the pursuer's absorbing subspace", and the
    /// pursuer is the opponent. A first version used a fixed half-window instead, and the
    /// result was not a game: the two players never touched, each minimised its own absorbed
    /// weight against a constant target, and whichever seat started inside the safe half won
    /// 20 of 20 at every budget from 1 to 128. Colour-swapping then averaged that to exactly
    /// 0.500 on every rung of the ladder, which reads as a flat ladder and is really an
    /// absent opponent. The coupling between the players *is* this function.
    pub fn safe_for(&self, player: usize) -> Vec<usize> {
        let pursuer = &self.players[player ^ 1].state;
        self.safe
            .iter()
            .copied()
            .filter(|&i| {
                let a = pursuer.amp(i);
                a.re * a.re + a.im * a.im < self.absorb_threshold
            })
            .collect()
    }

    /// Rule 7, for one player.
    pub fn position(&self, player: usize) -> Position {
        let p = &self.players[player];
        Position::new(p.state.clone(), p.algebra(), self.safe_for(player))
    }

    /// Whether the game is over, and for whom.
    pub fn outcome(&self, tolerance: f64) -> Option<Outcome> {
        for i in 0..2 {
            if self.position(i).is_checkmate(tolerance) {
                return Some(Outcome::Checkmate(i));
            }
        }
        if self.players.iter().all(|p| p.coherence == 0) {
            return Some(Outcome::Exhausted);
        }
        None
    }

    /// Mean legal-move count over the game so far. Part VII 9.2 asks for this to be reported
    /// exactly as chess reports ~35, and never assumed.
    pub fn average_branching(&self) -> f64 {
        if self.branching.is_empty() {
            return 0.0;
        }
        self.branching.iter().sum::<usize>() as f64 / self.branching.len() as f64
    }
}

/// Projective measurement of one position bit, renormalised.
///
/// The outcome is chosen deterministically as the more likely branch. That is not a
/// simplification of the physics but a decision about what is being measured: the depth
/// ladder compares strategies, and a stochastic collapse would add variance that has nothing
/// to do with the strategies being compared. A stochastic version belongs in play, not in the
/// measurement harness.
pub fn collapse(psi: &mut StateVec, qubit: usize) {
    let dim = psi.dim();
    let bit = 1usize << qubit;
    let mut weight_one = 0.0;
    for i in 0..dim {
        if i & bit != 0 {
            let a = psi.amp(i);
            weight_one += a.re * a.re + a.im * a.im;
        }
    }
    let keep_one = weight_one > 0.5;
    let norm = if keep_one {
        weight_one
    } else {
        1.0 - weight_one
    };
    if norm < 1e-15 {
        return;
    }
    let s = 1.0 / norm.sqrt();
    let mut re = vec![0.0; dim];
    let mut im = vec![0.0; dim];
    for i in 0..dim {
        if (i & bit != 0) == keep_one {
            let a = psi.amp(i);
            re[i] = a.re * s;
            im[i] = a.im * s;
        }
    }
    *psi = StateVec::from_amplitudes(re, im);
}
