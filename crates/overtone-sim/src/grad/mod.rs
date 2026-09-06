//! Two independent gradient paths that must agree.
//!
//! [`adjoint`] is fast and is what training uses. [`shift`] is the parameter-shift rule:
//! exact, hardware-evaluable, and `O(P)`. Part I 6.4 requires them to agree to `1e-10` on
//! randomised circuits for every gate type, and the repo does not proceed past a red
//! gradient test.

pub mod adjoint;
pub mod shift;

pub use adjoint::{value_and_grad, ValueGrad};
