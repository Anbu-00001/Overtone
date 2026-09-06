//! A minimal complex scalar.
//!
//! Deliberately hand-rolled rather than pulled from `num-complex`: the hot path stores
//! amplitudes as two `f64` planes (see [`crate::state`]), so this type is only used for
//! 2x2 gate matrices and the dense reference in [`crate::reference`]. Keeping it local
//! keeps the crate dependency-free apart from the RNG.

use std::ops::{Add, Mul, Sub};

/// A double-precision complex number.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

impl C64 {
    pub const ZERO: C64 = C64 { re: 0.0, im: 0.0 };
    pub const ONE: C64 = C64 { re: 1.0, im: 0.0 };

    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        C64 { re, im }
    }

    /// `exp(i*phi)`.
    #[inline]
    pub fn expi(phi: f64) -> Self {
        C64::new(phi.cos(), phi.sin())
    }

    #[inline]
    pub fn conj(self) -> Self {
        C64::new(self.re, -self.im)
    }

    #[inline]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Largest absolute difference across the real and imaginary parts.
    #[inline]
    pub fn max_diff(self, other: Self) -> f64 {
        (self.re - other.re).abs().max((self.im - other.im).abs())
    }
}

impl Add for C64 {
    type Output = C64;
    #[inline]
    fn add(self, rhs: C64) -> C64 {
        C64::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Sub for C64 {
    type Output = C64;
    #[inline]
    fn sub(self, rhs: C64) -> C64 {
        C64::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for C64 {
    type Output = C64;
    #[inline]
    fn mul(self, rhs: C64) -> C64 {
        C64::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl Mul<f64> for C64 {
    type Output = C64;
    #[inline]
    fn mul(self, rhs: f64) -> C64 {
        C64::new(self.re * rhs, self.im * rhs)
    }
}

/// A 2x2 complex matrix in row-major order: `[m00, m01, m10, m11]`.
pub type Mat2 = [C64; 4];

/// Conjugate transpose of a 2x2 matrix.
#[inline]
pub fn dagger(m: &Mat2) -> Mat2 {
    [m[0].conj(), m[2].conj(), m[1].conj(), m[3].conj()]
}

/// Whether a 2x2 matrix is unitary to within `tol`.
pub fn is_unitary(m: &Mat2, tol: f64) -> bool {
    let d = dagger(m);
    // d * m, row-major.
    let p = [
        d[0] * m[0] + d[1] * m[2],
        d[0] * m[1] + d[1] * m[3],
        d[2] * m[0] + d[3] * m[2],
        d[2] * m[1] + d[3] * m[3],
    ];
    p[0].max_diff(C64::ONE) < tol
        && p[1].max_diff(C64::ZERO) < tol
        && p[2].max_diff(C64::ZERO) < tol
        && p[3].max_diff(C64::ONE) < tol
}
