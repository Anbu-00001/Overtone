//! Orbit: the game of Part VII, headless.
//!
//! Phase 9 builds no interface at all. Part VII 12 is explicit about why: if the strategy
//! ladder is flat the design is wrong, and building a board first means being attached to it
//! by the time the number arrives. So this crate is the rules, the win condition, the
//! endgame solver and the depth measurement, and nothing that draws.
//!
//! The seven rules, entire:
//!
//! ```text
//! 1.  You are an amplitude field on the maze.
//! 2.  You hold a hand of generators. Together they close into your algebra g.
//! 3.  Each turn: apply one generator to one region, or measure.
//! 4.  Applied generators evolve for k coherent steps, then the turn passes.
//! 5.  Coherence only decreases. dim(g) only increases.
//! 6.  Overlapping an opponent merges algebras: g <- closure(g_you u g_them).
//! 7.  You lose when no unitary in g reaches a safe state.
//! ```
//!
//! Rule 7 is [`invariant`] and [`checkmate`]; rules 2 to 6 are [`game`].

#![forbid(unsafe_code)]

pub mod checkmate;
pub mod dial;
pub mod endgame;
pub mod game;
pub mod invariant;
pub mod ladder;

pub use dial::Complexity;
pub use endgame::{is_endgame, Tablebase};
pub use game::{legal_moves, Game, Move, Outcome, Piece, Player};
pub use invariant::{
    commutant_basis, orbit_dimension, separation_deficit, Invariants, OrbitCertificate,
};
pub use ladder::{Ladder, Strategy, LANGUAGE, STEP_UNIT};
