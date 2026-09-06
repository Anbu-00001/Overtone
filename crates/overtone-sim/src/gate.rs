//! Gates, their matrices, and their derivatives.
//!
//! **Generator convention:** `RX(t) = exp(-i t X / 2)`, and likewise for RY and RZ. Every
//! gradient formula in this crate assumes it. Changing it changes the parameter-shift
//! rule's factor of one half and the adjoint's `-i/2` together; do not change one alone.

use crate::complex::{dagger, Mat2, C64};
use crate::state::StateVec;

/// Where a gate's rotation angle comes from.
///
/// The `Param` variant carries a `scale` so that a gate can rotate by `scale * theta`
/// rather than by `theta` itself. This is what makes the data-re-uploading encoding of
/// Part I 6.2 expressible: the encoding gate is `RX(lambda * s)`, where `lambda` is the
/// trainable parameter and `s` is the observation component, so `scale = s`. Gradients
/// pick up that factor by the chain rule, in both gradient paths.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Angle {
    /// A constant angle. Contributes no gradient.
    Fixed(f64),
    /// `scale * params[index]`.
    Param { index: usize, scale: f64 },
}

impl Angle {
    /// A trainable angle used directly, with unit scale.
    pub fn param(index: usize) -> Self {
        Angle::Param { index, scale: 1.0 }
    }

    /// A trainable angle multiplied by a datum, as in `RX(lambda * s)`.
    pub fn scaled(index: usize, scale: f64) -> Self {
        Angle::Param { index, scale }
    }

    /// Resolve to a concrete rotation angle.
    #[inline]
    pub fn value(&self, params: &[f64]) -> f64 {
        match *self {
            Angle::Fixed(t) => t,
            Angle::Param { index, scale } => scale * params[index],
        }
    }

    /// `(parameter index, d angle / d parameter)`, or `None` for a fixed angle.
    #[inline]
    pub fn param_ref(&self) -> Option<(usize, f64)> {
        match *self {
            Angle::Fixed(_) => None,
            Angle::Param { index, scale } => Some((index, scale)),
        }
    }
}

/// The gate set. Part I 6.1: "That is sufficient; do not gold-plate."
#[derive(Clone, Debug, PartialEq)]
pub enum Gate {
    Rx {
        q: usize,
        angle: Angle,
    },
    Ry {
        q: usize,
        angle: Angle,
    },
    Rz {
        q: usize,
        angle: Angle,
    },
    H {
        q: usize,
    },
    Cz {
        a: usize,
        b: usize,
    },
    Cnot {
        control: usize,
        target: usize,
    },
    /// An arbitrary single-qubit unitary, row-major.
    U {
        q: usize,
        m: Mat2,
    },
}

const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// `RX(t) = exp(-i t X / 2)`.
pub fn rx_matrix(t: f64) -> Mat2 {
    let (s, c) = (t * 0.5).sin_cos();
    [
        C64::new(c, 0.0),
        C64::new(0.0, -s),
        C64::new(0.0, -s),
        C64::new(c, 0.0),
    ]
}

/// `RY(t) = exp(-i t Y / 2)`.
pub fn ry_matrix(t: f64) -> Mat2 {
    let (s, c) = (t * 0.5).sin_cos();
    [
        C64::new(c, 0.0),
        C64::new(-s, 0.0),
        C64::new(s, 0.0),
        C64::new(c, 0.0),
    ]
}

/// `RZ(t) = exp(-i t Z / 2)`.
pub fn rz_matrix(t: f64) -> Mat2 {
    [
        C64::expi(-t * 0.5),
        C64::ZERO,
        C64::ZERO,
        C64::expi(t * 0.5),
    ]
}

/// `d RX / dt = -i (X/2) RX(t)`.
pub fn drx_matrix(t: f64) -> Mat2 {
    let (s, c) = (t * 0.5).sin_cos();
    [
        C64::new(-0.5 * s, 0.0),
        C64::new(0.0, -0.5 * c),
        C64::new(0.0, -0.5 * c),
        C64::new(-0.5 * s, 0.0),
    ]
}

/// `d RY / dt = -i (Y/2) RY(t)`.
pub fn dry_matrix(t: f64) -> Mat2 {
    let (s, c) = (t * 0.5).sin_cos();
    [
        C64::new(-0.5 * s, 0.0),
        C64::new(-0.5 * c, 0.0),
        C64::new(0.5 * c, 0.0),
        C64::new(-0.5 * s, 0.0),
    ]
}

/// `d RZ / dt = -i (Z/2) RZ(t)`.
pub fn drz_matrix(t: f64) -> Mat2 {
    let e = C64::expi(-t * 0.5);
    let f = C64::expi(t * 0.5);
    [
        C64::new(0.5 * e.im, -0.5 * e.re),
        C64::ZERO,
        C64::ZERO,
        C64::new(-0.5 * f.im, 0.5 * f.re),
    ]
}

/// The Hadamard matrix.
pub fn h_matrix() -> Mat2 {
    [
        C64::new(INV_SQRT2, 0.0),
        C64::new(INV_SQRT2, 0.0),
        C64::new(INV_SQRT2, 0.0),
        C64::new(-INV_SQRT2, 0.0),
    ]
}

/// The Pauli X matrix.
pub fn x_matrix() -> Mat2 {
    [C64::ZERO, C64::ONE, C64::ONE, C64::ZERO]
}

impl Gate {
    /// The parameter this gate differentiates against, with its chain-rule factor.
    pub fn param_ref(&self) -> Option<(usize, f64)> {
        match self {
            Gate::Rx { angle, .. } | Gate::Ry { angle, .. } | Gate::Rz { angle, .. } => {
                angle.param_ref()
            }
            _ => None,
        }
    }

    /// The gate's rotation angle under `params`, or `None` if it does not rotate.
    pub fn angle_value(&self, params: &[f64]) -> Option<f64> {
        match self {
            Gate::Rx { angle, .. } | Gate::Ry { angle, .. } | Gate::Rz { angle, .. } => {
                Some(angle.value(params))
            }
            _ => None,
        }
    }

    /// Apply the gate. `angle_offset` is added to the rotation angle, which is how the
    /// parameter-shift rule displaces one gate without disturbing any other gate that
    /// happens to share the same parameter.
    pub fn apply_with_offset(&self, st: &mut StateVec, params: &[f64], angle_offset: f64) {
        match self {
            Gate::Rx { q, angle } => {
                st.apply_1q(*q, &rx_matrix(angle.value(params) + angle_offset))
            }
            Gate::Ry { q, angle } => {
                st.apply_1q(*q, &ry_matrix(angle.value(params) + angle_offset))
            }
            Gate::Rz { q, angle } => {
                st.apply_1q(*q, &rz_matrix(angle.value(params) + angle_offset))
            }
            Gate::H { q } => st.apply_1q(*q, &h_matrix()),
            Gate::Cz { a, b } => st.apply_phase_on_mask((1 << a) | (1 << b), C64::new(-1.0, 0.0)),
            Gate::Cnot { control, target } => {
                st.apply_controlled_1q(*control, *target, &x_matrix())
            }
            Gate::U { q, m } => st.apply_1q(*q, m),
        }
    }

    /// Apply the gate.
    #[inline]
    pub fn apply(&self, st: &mut StateVec, params: &[f64]) {
        self.apply_with_offset(st, params, 0.0);
    }

    /// Apply the gate's inverse. Used to walk the adjoint pass backwards.
    pub fn apply_adjoint(&self, st: &mut StateVec, params: &[f64]) {
        match self {
            // Pauli rotations invert by negating the angle.
            Gate::Rx { q, angle } => st.apply_1q(*q, &rx_matrix(-angle.value(params))),
            Gate::Ry { q, angle } => st.apply_1q(*q, &ry_matrix(-angle.value(params))),
            Gate::Rz { q, angle } => st.apply_1q(*q, &rz_matrix(-angle.value(params))),
            // H, CZ and CNOT are their own inverses.
            Gate::H { .. } | Gate::Cz { .. } | Gate::Cnot { .. } => self.apply(st, params),
            Gate::U { q, m } => st.apply_1q(*q, &dagger(m)),
        }
    }

    /// Apply `dU/d(angle)`. Not a unitary; [`StateVec::apply_1q`] does not require one.
    ///
    /// Panics on a non-rotation gate, which has no angle to differentiate against.
    pub fn apply_derivative(&self, st: &mut StateVec, params: &[f64]) {
        match self {
            Gate::Rx { q, angle } => st.apply_1q(*q, &drx_matrix(angle.value(params))),
            Gate::Ry { q, angle } => st.apply_1q(*q, &dry_matrix(angle.value(params))),
            Gate::Rz { q, angle } => st.apply_1q(*q, &drz_matrix(angle.value(params))),
            _ => panic!("apply_derivative called on a non-rotation gate: {self:?}"),
        }
    }

    /// The qubits this gate touches. Used by the dense reference and by circuit validation.
    pub fn qubits(&self) -> Vec<usize> {
        match self {
            Gate::Rx { q, .. } | Gate::Ry { q, .. } | Gate::Rz { q, .. } => vec![*q],
            Gate::H { q } | Gate::U { q, .. } => vec![*q],
            Gate::Cz { a, b } => vec![*a, *b],
            Gate::Cnot { control, target } => vec![*control, *target],
        }
    }
}
