//! Coins.
//!
//! Part II 2: **the policy is the coin.** In a discrete-time walk each vertex carries a
//! small unitary on the direction space that decides how amplitude distributes among
//! neighbours, and that is exactly the object a policy is. Everything here is a `2 x 2`
//! unitary; nothing here knows about learning.

use overtone_sim::C64;

/// A `2 x 2` unitary on the coin space, row-major.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Coin(pub [C64; 4]);

impl Coin {
    /// The one-parameter real family `[[cos t, sin t], [sin t, -cos t]]`.
    ///
    /// This is the coin the aperiodic-walk literature varies: Lo Gullo et al. fix
    /// `theta_1 = pi/4` for one letter and sweep `theta_2` for the other.
    pub fn theta(t: f64) -> Coin {
        let (s, c) = t.sin_cos();
        Coin([
            C64::new(c, 0.0),
            C64::new(s, 0.0),
            C64::new(s, 0.0),
            C64::new(-c, 0.0),
        ])
    }

    /// `theta = pi/4`.
    pub fn hadamard() -> Coin {
        Coin::theta(std::f64::consts::FRAC_PI_4)
    }

    /// The Grover diffusion operator `2|s><s| - I` on a two-dimensional coin.
    ///
    /// On two directions this is exactly the Pauli `X`, so the walk is deterministic: the
    /// coin swaps left for right every step and the amplitude oscillates between two sites
    /// forever. The Grover coin only becomes a real coin at three directions or more. Part
    /// II 7 asks for it as a mandatory baseline, so it is here — and reporting that it is
    /// degenerate in one dimension is the honest form of that baseline.
    pub fn grover() -> Coin {
        Coin([C64::ZERO, C64::ONE, C64::ONE, C64::ZERO])
    }

    /// The generic coin of arXiv:2307.06332 §2:
    /// `[[sqrt(rho), sqrt(1-rho) e^(i theta)], [sqrt(1-rho) e^(i phi), -sqrt(rho) e^(i(theta+phi))]]`.
    ///
    /// `rho` sets how asymmetrically the coin superposes; `theta` and `phi` are the relative
    /// phases. `generic(0.5, 0, 0)` is the Hadamard coin.
    pub fn generic(rho: f64, theta: f64, phi: f64) -> Coin {
        let a = rho.clamp(0.0, 1.0).sqrt();
        let b = (1.0 - rho.clamp(0.0, 1.0)).sqrt();
        let e = |x: f64| C64::new(x.cos(), x.sin());
        Coin([
            C64::new(a, 0.0),
            e(theta) * b,
            e(phi) * b,
            e(theta + phi) * (-a),
        ])
    }

    /// Apply to `(a0, a1)`.
    #[inline]
    pub fn apply(&self, a0: C64, a1: C64) -> (C64, C64) {
        (
            self.0[0] * a0 + self.0[1] * a1,
            self.0[2] * a0 + self.0[3] * a1,
        )
    }

    /// Worst deviation of `C^dagger C` from the identity. Zero for a unitary.
    pub fn unitarity_error(&self) -> f64 {
        let m = &self.0;
        let col = |i: usize, j: usize| m[i].conj() * m[j] + m[i + 2].conj() * m[j + 2];
        let d00 = col(0, 0) - C64::ONE;
        let d11 = col(1, 1) - C64::ONE;
        let d01 = col(0, 1);
        [d00, d11, d01]
            .iter()
            .map(|c| c.re.hypot(c.im))
            .fold(0.0, f64::max)
    }
}
