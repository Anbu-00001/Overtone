// Overtone. Copyright (C) 2026 Anbuchelvan Ganesan.
// Licensed under the GNU Affero General Public License, version 3 or (at your option) any
// later version. See LICENSE, or <https://www.gnu.org/licenses/>. This program is
// distributed WITHOUT ANY WARRANTY; see the licence for details.

//! Discrete-time quantum walks, the substrates they run on, and the transport exponent.
//!
//! Part II 2's reframe: **the policy is the coin.** In a discrete-time quantum walk each
//! vertex carries a small unitary on the direction space that decides how amplitude
//! distributes among neighbours — exactly the object a policy is. Part IV 3 then makes the
//! substrate a dial: change the word written on the lattice and `sigma(t) ~ t^beta` moves
//! across its whole range, from ballistic to localized, with `beta` measured live.
//!
//! This crate is the one-dimensional core of Part II's M6 built early, because Part IV's
//! M16 cannot exist without it: a world with no measurable transport exponent is a skin.
//! The two-dimensional lattice, mazes, glued trees and Szegedy walks remain Phase 5's.
//!
//! # Example
//!
//! ```
//! use overtone_walk::{fit_exponent, sigma_trace, Coin, Run, Substrate, Word};
//!
//! // A clean lattice is ballistic: the standard deviation grows linearly in time.
//! let s = Substrate::new(Word::Periodic, 200, 0);
//! let h = Coin::hadamard();
//! let trace = sigma_trace(&s, &h, &h, Run::coherent(200));
//! let fit = fit_exponent(&trace, 0.5);
//! assert!((fit.beta - 1.0).abs() < 0.03, "beta was {}", fit.beta);
//! ```

#![forbid(unsafe_code)]

pub mod coin;
pub mod coined;
pub mod families;
pub mod hitting;
pub mod race;
pub mod reduced;
pub mod stats;
pub mod substrate;
pub mod two;
pub mod walk;

pub use coin::Coin;
pub use coined::{Coined, VertexCoin};
pub use families::{classical_hitting, hypercube, welded_tree, Family};
pub use hitting::{concurrent, one_shot, one_shot_uniform, residual, Concurrent};
pub use race::{optimise_coins, race, Entrant};
pub use reduced::{amplify, hypercube_time, welded_best, welded_horizon, Amplification, Line};
pub use stats::{classical_sigma, fit_exponent, regime, regime_of, PowerLaw};
pub use substrate::{Substrate, Word, ALL_WORDS};
pub use two::{correlate, evolve, similarity, Column, Correlation, Modes, Statistics};
pub use walk::{sigma_trace, Run, Walk};
