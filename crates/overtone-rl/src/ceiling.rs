//! The LP ceiling `J*(C)` (Part I 7.1).
//!
//! What is the best expected return any policy with frequency ceiling `C` can achieve on
//! `SpectralControl-k`? Since `J = E_s[cos(ks) p(s)]` with `p(s) = 2 pi(1|s) - 1`, and
//! `|p| <= 1` because `p` is a difference of probabilities, the question is:
//!
//! > maximise the frequency-`k` Fourier coefficient of a degree-`C` trig polynomial whose
//! > sup norm is at most 1.
//!
//! That is a linear objective over a convex set: a linear program on a grid of `s`. Part I
//! 12 is explicit that it must be solved numerically rather than guessed, and having looked
//! for a closed form, that instruction is well taken. The one-sided Caratheodory-Fejer
//! bound `2 cos(pi/(C+2))` does not apply -- it assumes `p >= 0`, whereas `|p| <= 1` is
//! two-sided and strictly stronger, and the Fejer value exceeds the true answer (it gives
//! 0.707 at a point where the true ceiling is 0.5, and the supremum over all degrees is
//! `2/pi = 0.6366`, which Fejer's bound passes straight through).
//!
//! # The symmetry reduction
//!
//! The naive LP has `2C+1` variables. It can be cut to `floor(C/k)+1`.
//!
//! Average `p` over the `k` shifts `s -> s + 2 pi j / k`. The average is still degree `C`;
//! it still satisfies `|p| <= 1`, being a mean of such functions; and in frequency space
//! the shift multiplies component `n` by `exp(2 pi i n j / k)`, so averaging over `j`
//! annihilates every `n` that is not a multiple of `k` while leaving `n = k` untouched.
//! Then replace `p` by its even part, which preserves the cosine-`k` coefficient and kills
//! the sines. So without loss of generality
//!
//! ```text
//! p(s) = q(ks),   q(u) = sum_{m=0..M} c_m cos(m u),   M = floor(C / k),
//! ```
//!
//! and `J = c_1 / 2`.
//!
//! Two consequences worth stating. `J*` depends on `C` only through `floor(C/k)`, so the
//! ceiling really is a staircase and its steps fall at multiples of `k`. And `J*(C) = 0`
//! whenever `C < k`, which is the zero-return prediction, recovered here from the
//! optimisation side rather than from the theorem.
//!
//! [`ceiling_unreduced`] keeps the naive formulation so the reduction can be tested rather
//! than trusted.

use good_lp::{
    constraint, default_solver, variable, Expression, ProblemVariables, Solution, SolverModel,
};

/// Grid resolution used by [`ceiling`]. Convergence is fast; see the tests.
pub const DEFAULT_GRID: usize = 2001;

/// Coefficient bound imposed on the LP variables.
///
/// `|p| <= 1` implies `|a_j| = |(1/pi) integral p cos(js) ds| <= 2`, so this cannot exclude
/// an optimum. It is here because a solver given genuinely free variables reports the
/// problem unbounded.
const COEFF_BOUND: f64 = 4.0;

/// The best expected return attainable on `SpectralControl-k` by any policy whose
/// reachable frequency ceiling is `c`.
///
/// Uses the reduced formulation. Returns exactly `0.0` when `c < k`.
pub fn ceiling(c: usize, k: usize, grid: usize) -> f64 {
    assert!(k > 0, "k must be positive");
    let m = c / k;
    if m == 0 {
        // The frequency-k component is outside the reachable spectrum, so J is identically
        // zero for every parameter setting. No optimisation needed, and no floating-point
        // noise introduced by pretending otherwise.
        return 0.0;
    }

    let mut vars = ProblemVariables::new();
    let coeffs: Vec<_> = (0..=m)
        .map(|_| vars.add(variable().min(-COEFF_BOUND).max(COEFF_BOUND)))
        .collect();

    let mut problem = vars
        .maximise(Expression::from(coeffs[1]))
        .using(default_solver);

    // q is even and 2pi-periodic in u, so [0, pi] covers every value it takes.
    for i in 0..grid {
        let u = std::f64::consts::PI * (i as f64) / (grid as f64 - 1.0);
        let q: Expression = coeffs
            .iter()
            .enumerate()
            .map(|(j, v)| *v * ((j as f64) * u).cos())
            .sum();
        problem = problem
            .with(constraint!(q.clone() <= 1.0))
            .with(constraint!(q >= -1.0));
    }

    problem
        .solve()
        .expect("ceiling LP is feasible and bounded")
        .value(coeffs[1])
        / 2.0
}

/// The same quantity from the unreduced formulation: all `2c+1` Fourier coefficients, no
/// symmetry argument used.
///
/// Slower, and slightly less accurate on a fixed grid -- more variables means more freedom
/// to violate the sup-norm constraint between grid points, so it reads slightly high. It
/// exists to test [`ceiling`], not to be called.
pub fn ceiling_unreduced(c: usize, k: usize, grid: usize) -> f64 {
    assert!(k > 0, "k must be positive");
    if c < k {
        return 0.0;
    }

    let pi = std::f64::consts::PI;
    let mut vars = ProblemVariables::new();
    let a: Vec<_> = (0..=c)
        .map(|_| vars.add(variable().min(-COEFF_BOUND).max(COEFF_BOUND)))
        .collect();
    let b: Vec<_> = (0..=c)
        .map(|_| vars.add(variable().min(-COEFF_BOUND).max(COEFF_BOUND)))
        .collect();

    let mut problem = vars.maximise(Expression::from(a[k])).using(default_solver);

    for i in 0..grid {
        let s = -pi + 2.0 * pi * (i as f64) / grid as f64;
        let p: Expression = (0..=c)
            .map(|j| a[j] * ((j as f64) * s).cos() + b[j] * ((j as f64) * s).sin())
            .sum();
        problem = problem
            .with(constraint!(p.clone() <= 1.0))
            .with(constraint!(p >= -1.0));
    }

    problem.solve().expect("unreduced ceiling LP").value(a[k]) / 2.0
}

/// The staircase `J*(0..=max_c)`, for drawing behind a learning curve.
pub fn staircase(max_c: usize, k: usize) -> Vec<f64> {
    (0..=max_c).map(|c| ceiling(c, k, DEFAULT_GRID)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    #[test]
    fn below_the_environment_frequency_the_ceiling_is_exactly_zero() {
        for k in 1..=5usize {
            for c in 0..k {
                assert_eq!(ceiling(c, k, 401), 0.0, "c={c} k={k}");
            }
        }
    }

    #[test]
    fn at_the_environment_frequency_the_ceiling_is_one_half() {
        // M = 1 admits q(u) = c_0 + c_1 cos u with |q| <= 1, whose best c_1 is 1 at
        // c_0 = 0, giving J = 1/2. A rare point where the LP has a closed form to check.
        for k in 1..=4usize {
            let j = ceiling(k, k, DEFAULT_GRID);
            assert!((j - 0.5).abs() < TOL, "k={k}: {j}");
        }
    }

    #[test]
    fn the_ceiling_is_a_staircase_stepping_at_multiples_of_k() {
        // J* depends on c only through floor(c/k). This is the symmetry reduction's
        // structural prediction, and it is what makes the curve a staircase.
        let k = 3;
        let s = staircase(11, k);
        for (c, got) in s.iter().enumerate() {
            let expected = ceiling(k * (c / k), k, DEFAULT_GRID);
            assert!(
                (got - expected).abs() < TOL,
                "c={c}: {got} vs floor-matched {expected}"
            );
        }
    }

    #[test]
    fn the_ceiling_is_monotone_and_bounded_by_two_over_pi() {
        let two_over_pi = 2.0 / std::f64::consts::PI;
        for k in [1usize, 2, 3] {
            let s = staircase(4 * k + 2, k);
            for w in s.windows(2) {
                assert!(w[1] >= w[0] - TOL, "not monotone: {:?}", w);
            }
            for j in &s {
                assert!(*j <= two_over_pi + TOL, "{j} exceeds 2/pi");
            }
        }
    }

    #[test]
    fn reduced_agrees_with_unreduced() {
        // The reduction is a proof, but a proof I wrote. Tolerance is loose because the
        // unreduced LP reads slightly high on a finite grid, always in the same direction.
        for k in 1..=3usize {
            for c in 0..=7usize {
                let r = ceiling(c, k, 1201);
                let u = ceiling_unreduced(c, k, 1201);
                assert!(
                    (r - u).abs() < 5e-5,
                    "c={c} k={k}: reduced {r} vs unreduced {u}"
                );
                assert!(u >= r - TOL, "c={c} k={k}: unreduced should not read low");
            }
        }
    }

    #[test]
    fn the_ceiling_is_grid_converged() {
        for c in [1usize, 3, 5, 7] {
            let coarse = ceiling(c, 1, 401);
            let fine = ceiling(c, 1, 6401);
            assert!(
                (coarse - fine).abs() < 1e-4,
                "c={c}: {coarse} at 401 nodes vs {fine} at 6401"
            );
        }
    }

    #[test]
    fn the_staircase_approaches_two_over_pi_from_below() {
        // Not a closed form, just the limit the sequence is climbing toward.
        let deep = ceiling(41, 1, 4001);
        let two_over_pi = 2.0 / std::f64::consts::PI;
        assert!(deep < two_over_pi, "{deep} should stay under 2/pi");
        assert!(deep > 0.63, "{deep} should be close to 2/pi by C = 41");
    }
}
