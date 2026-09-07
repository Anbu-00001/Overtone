//! Engine C: Lie-algebraic simulation.
//!
//! Part III 4. Instead of a `2^n` state vector, evolve a `dim(g)`-dimensional vector of
//! expectation values in the dynamical Lie algebra basis. When `dim(g)` is polynomial in
//! the qubit count this is exact and it scales — which is how a hundred-qubit policy is
//! trained on one CPU core in `examples/hundred_qubit_policy.rs`.
//!
//! The caveat travels with the claim, here as well as in the README: g-sim is efficient
//! exactly when `dim(g)` is polynomial, and a polynomial DLA is exactly the condition for
//! *not* having a barren plateau. The circuits that train are the circuits that are
//! classically simulable. Part III 4 says to name that tension rather than route around it.

#![forbid(unsafe_code)]

pub mod evolve;
pub mod policy;

pub use evolve::{DlaGate, GsimCircuit};
pub use policy::{GsimPolicy, PolicyLayout, PolicyObservable};
