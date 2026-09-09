// Overtone. Copyright (C) 2026 Anbuchelvan Ganesan.
// Licensed under the GNU Affero General Public License, version 3 or (at your option) any
// later version. See LICENSE, or <https://www.gnu.org/licenses/>. This program is
// distributed WITHOUT ANY WARRANTY; see the licence for details.

//! Temperature: move ordering that nobody tuned.
//!
//! Part IX 5.2 states the problem this crate exists to solve. Part VII forbids invented
//! numbers, and a game still needs to know which move matters; the usual answer is a
//! hand-tuned evaluation function, which is exactly the invented number the rules prohibit.
//! Combinatorial game theory has a *derived* answer. A position decomposes into independent
//! regions, each region has a **temperature** -- the urgency of moving there -- and the
//! temperature is computed from the position by thermography, not chosen.
//!
//! # The correction this crate carries
//!
//! Part IX 5 was written as though thermography applied to Overtone directly. It does not,
//! and the boundary is sharp. Berlekamp's own survey opens:
//!
//! > In its broadest sense, Combinatorial Game Theory (CGT) is the study of two-person,
//! > **perfect information games of no chance.**
//!
//! Overtone has chance: rule 3 lets a player *measure*, and the Born rule is a chance node.
//! So temperature is defined on the **coherent segment** -- the run of unitary turns between
//! one measurement and the next -- and is undefined across a collapse. That is not fatal and
//! it is not a workaround: it is the honest domain of the tool, and it happens to be the
//! interval a player is actually reasoning about when they decide where to move. Every
//! temperature this crate reports is a temperature of a coherent segment. See
//! [`thermo::Thermograph`] and Part IX's trap list.
//!
//! # Exactness
//!
//! Every quantity in temperature theory is a dyadic rational (Blom, Thm 5.8 with Prop 5.11:
//! a short game's temperature is a dyadic rational). Dyadic rationals with denominator below
//! `2^52` are represented *exactly* by `f64`, so this crate computes in `f64` and is exact,
//! not approximate, over the range it is used in. [`tests/thermograph.rs`] tests that claim
//! rather than assuming it.
//!
//! # Example
//!
//! ```
//! use overtone_cgt::Game;
//!
//! // A switch: Left moves to 3, Right moves to 1.
//! let g = Game::switch(3.0, 1.0);
//! let t = g.thermograph();
//! assert_eq!(t.mean, 2.0);          // (3 + 1) / 2
//! assert_eq!(t.temperature, 1.0);   // (3 - 1) / 2
//! ```

#![forbid(unsafe_code)]

pub mod decompose;
pub mod game;
pub mod thermo;

pub use decompose::{
    find_disagreement, hottest_first, hottest_index, stop, temperatures, Disagreement,
};
pub use game::Game;
pub use thermo::{Thermograph, Wall};

/// Comparison tolerance. Every value in this crate is a dyadic rational, so exact equality
/// would in principle do; the tolerance exists only to keep breakpoint merging robust when a
/// crossing lands a few ULPs off a breakpoint that is already in the list.
pub const EPS: f64 = 1e-12;
