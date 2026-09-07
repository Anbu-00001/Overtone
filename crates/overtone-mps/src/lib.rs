//! Engine B, and the instrument that tries to disprove the project's own premise.
//!
//! Part III 5: if a `chi = 4` tensor network reproduces a quantum agent, the quantum agent
//! *is* a small tensor network. The output is one integer and a sentence, and it is
//! published whatever it says.
//!
//! Scope, stated rather than implied: this crate compresses an exact state vector, so it is
//! a dequantization instrument for circuits the state-vector engine can run, not a
//! large-`n` MPS time evolution. Part III 4's `n = 100, chi = 512` Engine B is not built
//! here; Part III 12's minimum viable scope does not need it, and claiming it without
//! building it would be exactly the kind of thing this crate exists to catch.

#![forbid(unsafe_code)]

pub mod dequantize;
pub mod linalg;
pub mod truncate;

pub use dequantize::{dequantize, ChiRow, DequantizeReport};
pub use truncate::{bond_spectrum, compress, exact_bond_dimension};
