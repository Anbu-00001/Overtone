//! The state vector, stored structure-of-arrays.
//!
//! Amplitudes live in two parallel `Vec<f64>` planes rather than a `Vec<C64>`. This is not
//! a style preference: the gate inner loop is a strided read-modify-write over pairs of
//! amplitudes, and the split layout autovectorises where an array-of-structs does not.
//! Do not "tidy" this into a complex element type.
//!
//! Qubit `q` indexes bit `q` of the basis-state index, little-endian: amplitude `i` has
//! qubit `q` set iff `i & (1 << q) != 0`.

use crate::complex::{Mat2, C64};

/// A pure state of `n` qubits.
#[derive(Clone, Debug, PartialEq)]
pub struct StateVec {
    /// Real parts, length `2^n`.
    pub re: Vec<f64>,
    /// Imaginary parts, length `2^n`.
    pub im: Vec<f64>,
    n: usize,
}

/// Above this qubit count the parallel feature splits the gate loop across threads.
/// Below it the rayon overhead dominates the work.
#[cfg(feature = "parallel")]
const PARALLEL_THRESHOLD: usize = 12;

impl StateVec {
    /// `|0...0>` on `n` qubits.
    pub fn zero(n: usize) -> Self {
        assert!(n > 0, "a state needs at least one qubit");
        assert!(n <= 30, "2^{n} amplitudes will not fit in memory");
        let dim = 1usize << n;
        let mut re = vec![0.0; dim];
        re[0] = 1.0;
        StateVec {
            re,
            im: vec![0.0; dim],
            n,
        }
    }

    /// Build from explicit amplitudes. Panics unless the length is a power of two.
    pub fn from_amplitudes(re: Vec<f64>, im: Vec<f64>) -> Self {
        assert_eq!(re.len(), im.len(), "amplitude planes must match in length");
        assert!(
            re.len().is_power_of_two(),
            "dimension must be a power of two"
        );
        let n = re.len().trailing_zeros() as usize;
        StateVec { re, im, n }
    }

    /// Number of qubits.
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// Number of amplitudes, `2^n`.
    #[inline]
    pub fn dim(&self) -> usize {
        self.re.len()
    }

    /// Amplitude `i`.
    #[inline]
    pub fn amp(&self, i: usize) -> C64 {
        C64::new(self.re[i], self.im[i])
    }

    /// Overwrite this state with `other`, reusing the allocation.
    ///
    /// Used by the adjoint pass, which needs a scratch buffer per parameterised gate but
    /// must not allocate one per gate.
    pub fn copy_from(&mut self, other: &StateVec) {
        debug_assert_eq!(self.n, other.n);
        self.re.copy_from_slice(&other.re);
        self.im.copy_from_slice(&other.im);
    }

    /// Squared norm. Should stay at 1 to rounding under any unitary.
    pub fn norm_sqr(&self) -> f64 {
        self.re
            .iter()
            .zip(&self.im)
            .map(|(r, i)| r * r + i * i)
            .sum()
    }

    /// `Re<self|other>`.
    ///
    /// The adjoint gradient needs only the real part, which is why this returns `f64`.
    pub fn inner_re(&self, other: &StateVec) -> f64 {
        debug_assert_eq!(self.n, other.n);
        self.re
            .iter()
            .zip(&self.im)
            .zip(other.re.iter().zip(&other.im))
            .map(|((ar, ai), (br, bi))| ar * br + ai * bi)
            .sum()
    }

    /// Full complex inner product `<self|other>`.
    pub fn inner(&self, other: &StateVec) -> C64 {
        debug_assert_eq!(self.n, other.n);
        let mut re = 0.0;
        let mut im = 0.0;
        for i in 0..self.dim() {
            // conj(self) * other
            re += self.re[i] * other.re[i] + self.im[i] * other.im[i];
            im += self.re[i] * other.im[i] - self.im[i] * other.re[i];
        }
        C64::new(re, im)
    }

    /// Largest absolute amplitude difference against `other`.
    pub fn max_diff(&self, other: &StateVec) -> f64 {
        self.re
            .iter()
            .zip(&other.re)
            .map(|(a, b)| (a - b).abs())
            .chain(self.im.iter().zip(&other.im).map(|(a, b)| (a - b).abs()))
            .fold(0.0, f64::max)
    }

    /// Apply an arbitrary 2x2 matrix to qubit `q`.
    ///
    /// The matrix need not be unitary: the adjoint pass applies gate derivatives, which
    /// are not. Nothing in this routine assumes unitarity.
    pub fn apply_1q(&mut self, q: usize, m: &Mat2) {
        #[cfg(feature = "parallel")]
        if self.n > PARALLEL_THRESHOLD {
            self.apply_1q_parallel(q, m);
            return;
        }
        self.apply_1q_serial(q, m);
    }

    /// The serial pair loop.
    ///
    /// Exposed rather than private so that the parallel path can be tested against it
    /// directly. Dispatching on qubit count alone would leave the parallel kernel
    /// unexercised, since the test suite runs well below the threshold.
    pub fn apply_1q_serial(&mut self, q: usize, m: &Mat2) {
        assert!(q < self.n, "qubit {q} out of range for {} qubits", self.n);
        let stride = 1usize << q;
        let block = stride << 1;
        self.re
            .chunks_mut(block)
            .zip(self.im.chunks_mut(block))
            .for_each(|(rc, ic)| apply_1q_block(rc, ic, stride, m));
    }

    /// The same pair loop, split across threads. Must agree with the serial path exactly.
    #[cfg(feature = "parallel")]
    pub fn apply_1q_parallel(&mut self, q: usize, m: &Mat2) {
        use rayon::prelude::*;
        assert!(q < self.n, "qubit {q} out of range for {} qubits", self.n);
        let stride = 1usize << q;
        let block = stride << 1;
        self.re
            .par_chunks_mut(block)
            .zip(self.im.par_chunks_mut(block))
            .for_each(|(rc, ic)| apply_1q_block(rc, ic, stride, m));
    }

    /// Apply a 2x2 matrix to target qubit `t`, conditioned on control qubit `c` being set.
    pub fn apply_controlled_1q(&mut self, c: usize, t: usize, m: &Mat2) {
        #[cfg(feature = "parallel")]
        if self.n > PARALLEL_THRESHOLD {
            self.apply_controlled_1q_parallel(c, t, m);
            return;
        }
        self.apply_controlled_1q_serial(c, t, m);
    }

    /// The serial controlled pair loop.
    pub fn apply_controlled_1q_serial(&mut self, c: usize, t: usize, m: &Mat2) {
        assert!(c < self.n, "control {c} out of range for {} qubits", self.n);
        assert!(t < self.n, "target {t} out of range for {} qubits", self.n);
        assert_ne!(c, t, "control and target must differ");
        let stride = 1usize << t;
        let block = stride << 1;
        let cmask = 1usize << c;
        self.re
            .chunks_mut(block)
            .zip(self.im.chunks_mut(block))
            .enumerate()
            .for_each(|(k, (rc, ic))| apply_controlled_block(k * block, rc, ic, stride, cmask, m));
    }

    /// The same loop across threads. The control test needs the *global* index, which the
    /// chunk index reconstructs; getting that wrong is the likely bug here, so the test
    /// suite compares this against the serial path directly.
    #[cfg(feature = "parallel")]
    pub fn apply_controlled_1q_parallel(&mut self, c: usize, t: usize, m: &Mat2) {
        use rayon::prelude::*;
        assert!(c < self.n, "control {c} out of range for {} qubits", self.n);
        assert!(t < self.n, "target {t} out of range for {} qubits", self.n);
        assert_ne!(c, t, "control and target must differ");
        let stride = 1usize << t;
        let block = stride << 1;
        let cmask = 1usize << c;
        self.re
            .par_chunks_mut(block)
            .zip(self.im.par_chunks_mut(block))
            .enumerate()
            .for_each(|(k, (rc, ic))| apply_controlled_block(k * block, rc, ic, stride, cmask, m));
    }

    /// Multiply amplitudes by `phase` wherever every qubit in `mask` is set.
    ///
    /// CZ is the only user, and it is diagonal, so it never touches the pair loop.
    pub fn apply_phase_on_mask(&mut self, mask: usize, phase: C64) {
        for i in 0..self.dim() {
            if i & mask == mask {
                let a = C64::new(self.re[i], self.im[i]) * phase;
                self.re[i] = a.re;
                self.im[i] = a.im;
            }
        }
    }

    /// Scale every amplitude by a real factor.
    pub fn scale(&mut self, f: f64) {
        for r in &mut self.re {
            *r *= f;
        }
        for i in &mut self.im {
            *i *= f;
        }
    }

    /// Add `f * other` into this state.
    pub fn add_scaled(&mut self, other: &StateVec, f: f64) {
        debug_assert_eq!(self.n, other.n);
        for i in 0..self.dim() {
            self.re[i] += f * other.re[i];
            self.im[i] += f * other.im[i];
        }
    }
}

/// One block of the single-qubit pair loop.
///
/// A block spans `2 * stride` amplitudes; within it, `off` and `off + stride` are the two
/// partners that differ only in the target bit.
#[inline]
fn apply_1q_block(re: &mut [f64], im: &mut [f64], stride: usize, m: &Mat2) {
    for off in 0..stride.min(re.len()) {
        let j = off + stride;
        if j >= re.len() {
            break;
        }
        let (a0r, a0i) = (re[off], im[off]);
        let (a1r, a1i) = (re[j], im[j]);
        re[off] = m[0].re * a0r - m[0].im * a0i + m[1].re * a1r - m[1].im * a1i;
        im[off] = m[0].re * a0i + m[0].im * a0r + m[1].re * a1i + m[1].im * a1r;
        re[j] = m[2].re * a0r - m[2].im * a0i + m[3].re * a1r - m[3].im * a1i;
        im[j] = m[2].re * a0i + m[2].im * a0r + m[3].re * a1i + m[3].im * a1r;
    }
}

/// One block of the controlled pair loop. `base` is the global index of `re[0]`.
#[inline]
fn apply_controlled_block(
    base: usize,
    re: &mut [f64],
    im: &mut [f64],
    stride: usize,
    cmask: usize,
    m: &Mat2,
) {
    for off in 0..stride.min(re.len()) {
        let j = off + stride;
        if j >= re.len() {
            break;
        }
        // The control bit is never the target bit, so it takes the same value at `off`
        // and `off + stride`; testing one of them is enough.
        if (base + off) & cmask == 0 {
            continue;
        }
        let (a0r, a0i) = (re[off], im[off]);
        let (a1r, a1i) = (re[j], im[j]);
        re[off] = m[0].re * a0r - m[0].im * a0i + m[1].re * a1r - m[1].im * a1i;
        im[off] = m[0].re * a0i + m[0].im * a0r + m[1].re * a1i + m[1].im * a1r;
        re[j] = m[2].re * a0r - m[2].im * a0i + m[3].re * a1r - m[3].im * a1i;
        im[j] = m[2].re * a0i + m[2].im * a0r + m[3].re * a1i + m[3].im * a1r;
    }
}
