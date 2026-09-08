//! Optimisers, shot budgets, and the barren-plateau flatline (Part V 4, 5.2).
//!
//! Part V asks for two things that turn out to be one thing. Section 5.2 wants four
//! optimisers racing in a barren plateau and all four flat, citing Arrasmith et al.
//! Section 4 wants a shot-budget dial, "the most honest NISQ-realism knob the project can
//! offer". They are the same instrument: the flatline **is** a statement about shots, and
//! without the dial the demonstration shows the opposite of what it cites.
//!
//! See [`shots`] for why, [`optim`] for the four optimisers and the check that they work at
//! all, and `examples/flatline.rs` for the measurement.

#![forbid(unsafe_code)]

pub mod optim;
pub mod plateau;
pub mod race;
pub mod shots;

pub use optim::{cmaes, descend, nelder_mead, Differentiable, Objective, Optimiser, Run};
pub use plateau::PlateauCost;
pub use race::{fit_shot_scaling, lane, race, Lane, RaceConfig, ShotScaling};
pub use shots::{Budget, DiagonalTerm};
