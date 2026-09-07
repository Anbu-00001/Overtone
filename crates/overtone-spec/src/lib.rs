//! The instrumentation. This is the crown jewel (Part I 6.5).
//!
//! Part I 1 lists five things a spectral view of a quantum policy makes visible. This crate
//! turns each into a function that returns numbers, and each of those into a test.
//!
//! It reads circuits and policies and emits measurements. It does not train and it does not
//! render.

#![forbid(unsafe_code)]

pub mod entropy;
pub mod fft;
pub mod plateau;
pub mod qfim;
pub mod spectrum;

pub use entropy::{bipartition, half_chain_entropy, Bipartition};
pub use plateau::{
    fit_exponential, sweep, CostLocality, DepthPolicy, ExponentialFit, PlateauPoint,
};
pub use qfim::{qfim, qfim_spectrum, rank};
pub use spectrum::{spectrum_of, Spectrum};
