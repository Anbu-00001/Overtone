//! Overtone's state-vector simulator.
//!
//! A variational quantum circuit that encodes classical data is exactly a truncated
//! Fourier series in that data (Schuld, Sweke and Meyer, Phys. Rev. A 103, 032430). This
//! crate is the engine that claim is measured with. It knows nothing about reinforcement
//! learning and nothing about rendering; see `docs/PHASES.md` for what depends on it.
//!
//! # Conventions
//!
//! - `RX(t) = exp(-i t X / 2)`, and likewise RY and RZ.
//! - Qubit `q` is bit `q` of the basis-state index, little-endian.
//! - Amplitudes are stored structure-of-arrays, two `f64` planes.
//! - Every random draw is seeded explicitly. No entropy-seeded RNG appears here.
//!
//! # Example
//!
//! ```
//! use overtone_sim::{Angle, Circuit, Observable, grad};
//!
//! // <Z_0> after RY(t)|0> is cos(t).
//! let mut c = Circuit::new(1);
//! c.ry(0, Angle::param(0));
//! let obs = Observable::z(0);
//! let params = [0.7_f64];
//!
//! let vg = grad::value_and_grad(&c, &params, &obs);
//! assert!((vg.value - 0.7_f64.cos()).abs() < 1e-12);
//! assert!((vg.grad[0] + 0.7_f64.sin()).abs() < 1e-12);
//! ```

#![forbid(unsafe_code)]

pub mod circuit;
pub mod complex;
pub mod gate;
pub mod grad;
pub mod observable;
pub mod random;
pub mod reference;
pub mod state;

pub use circuit::Circuit;
pub use complex::C64;
pub use gate::{Angle, Gate};
pub use observable::{Observable, Pauli, PauliTerm};
pub use state::StateVec;
