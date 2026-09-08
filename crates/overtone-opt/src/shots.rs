//! Why the shot budget is the whole demonstration.
//!
//! The mechanism lives in [`overtone_sim::measure`] -- it is a property of a state vector
//! and an observable, not of an optimiser -- and is re-exported here so that this crate's
//! reader meets it in context.
//!
//! # The correction this crate exists to record
//!
//! Arrasmith et al. (*Quantum* **5**, 558) do not say gradient-free optimisers stall in a
//! barren plateau as a matter of arithmetic. They say the *cost function differences* are
//! exponentially suppressed, so that "the numbers of shots required in the optimization
//! grows exponentially with the number of qubits". The obstruction is **measurement
//! precision**, not the landscape's shape in exact arithmetic.
//!
//! That distinction has teeth. A demonstration written against `f64` expectation values
//! computed from a state vector gives every optimiser roughly sixteen digits for free,
//! which is far more than the `2^(-1.03 n)` differences at any qubit count a browser can
//! simulate. Measured in `examples/flatline.rs`: with exact arithmetic all four optimisers
//! descend at every qubit count from 2 to 10, on every seed, Nelder-Mead included. Part V
//! 5.2's demonstration written the obvious way shows the opposite of the paper it cites.

pub use overtone_sim::measure::{Budget, DiagonalTerm, EXACT_SAMPLING_LIMIT};
