// Overtone. Copyright (C) 2026 Anbuchelvan Ganesan.
// Licensed under the GNU Affero General Public License, version 3 or (at your option) any
// later version. See LICENSE, or <https://www.gnu.org/licenses/>. This program is
// distributed WITHOUT ANY WARRANTY; see the licence for details.

//! The closure engine: Pauli bitsets, the dynamical Lie algebra, and what it predicts.
//!
//! Part III's thesis is that a quantum policy's capabilities are fixed before training by
//! the algebra of its generators. This crate computes that algebra. It holds no matrices:
//! Part III 13 is explicit that a `2^n` matrix anywhere in here is a bug, and the dense
//! oracle that checks the closure lives in `tests/`, outside the library, for that reason.
//!
//! # Example
//!
//! ```
//! use overtone_lie::{closure, family, PauliString, Prediction};
//!
//! // The transverse-field Ising chain closes at so(2n), dimension n(2n - 1).
//! let n = 5;
//! let g = closure::closure_unbounded(&family::tfim(n), n);
//! assert_eq!(g.dim(), n * (2 * n - 1));
//!
//! // The Heisenberg chain on the same five qubits does not.
//! let h = closure::closure_unbounded(&family::heisenberg(n), n);
//! assert_eq!(h.dim(), 4usize.pow(4) - 1);
//! ```

#![forbid(unsafe_code)]

pub mod closure;
pub mod family;
pub mod pauli;
pub mod predict;
pub mod sigil;

pub use closure::{closure, closure_unbounded, Algebra};
pub use pauli::{PauliString, MAX_QUBITS};
pub use predict::{Prediction, Scaling};
pub use sigil::{Sigil, Spoke};
