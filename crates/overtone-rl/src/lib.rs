//! Environments, policies and the policy-gradient loop.
//!
//! This crate consumes `overtone-sim` and knows nothing about rendering. The one idea it
//! exists to make measurable is Part I 7.1: on `SpectralControl-k`, a band-limited policy
//! whose frequency ceiling falls below `k` scores exactly zero, and above `k` it is capped
//! by a staircase that a linear program can compute in advance.
//!
//! See `docs/PHASES.md` for where this sits, and `docs/spec/part-i-build-spec.md` for the
//! specification it implements.

#![forbid(unsafe_code)]

pub mod ansatz;
pub mod ceiling;
pub mod env;
pub mod policy;
pub mod reinforce;

pub use ansatz::{Ansatz, AnsatzConfig, EncodingSite, Scaling, SpectralControlAnsatz};
pub use env::SpectralControl;
pub use policy::{Policy, PolicyKind};
pub use reinforce::{train, TrainConfig, TrainOutcome};
