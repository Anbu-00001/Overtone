//! Barren plateaus as a measurement (Part I 6.7).
//!
//! Sample `M` random parameter vectors, take `d<O>/dtheta_1` at each, and report the
//! variance. Sweep the qubit count and fit the exponent.
//!
//! Two observables, and the contrast between them is the entire point:
//!
//! - **Global**, `Z (x) Z (x) ... (x) Z`. McClean et al. (2018): the variance vanishes
//!   exponentially in `n`. Training is hopeless at scale.
//! - **Local**, `Z_0`. Cerezo et al. (2021): for a *shallow* circuit the variance is
//!   bounded below by an inverse polynomial. No barren plateau.
//!
//! # What this ansatz actually measures
//!
//! With the hardware-efficient ansatz below (`RY`, `RZ`, ring of `CZ`) and the `theta_1`
//! probe, the measured global rate is `Var ~ 2^(-1.03 n)` with `R^2 = 0.999`, at both
//! logarithmic and linear depth. Part I 6.7 quotes `2^(-1.98 n)` as the illustrative
//! figure; that is the full 2-design result, and this circuit family does not reach a
//! 2-design at the depths swept here. The number reported is the one measured.
//!
//! A plausible explanation -- that probing `theta_1` leaves the circuit to its left trivial,
//! so only one side Haar-averages and the exponent halves -- was tested and **rejected**:
//! moving the probe to mid-circuit changes the global rate from 1.035 to 1.080, not to
//! 2. See `examples/plateau_probe.rs`. The gap from the textbook exponent is a property of
//! the ansatz, not of where the derivative is taken.
//!
//! # Depth is not a free parameter here
//!
//! The Cerezo result is conditional on shallow depth -- `O(log n)` -- and the local
//! observable only survives in that regime. Run the same sweep at linear depth and the
//! local curve collapses too, because the circuit approaches a 2-design and forgets that
//! the observable was local. So [`DepthPolicy`] makes the choice explicit rather than
//! leaving a magic number in a loop: a sweep that quietly used constant depth would show
//! the right picture for the wrong reason.

use overtone_sim::grad::adjoint;
use overtone_sim::random::{random_params, rng};
use overtone_sim::{Angle, Circuit, Observable};
use rand::Rng;

/// How circuit depth scales with the qubit count across a sweep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthPolicy {
    /// Fixed depth, independent of `n`.
    Constant(usize),
    /// `ceil(c * log2(n))`, the regime in which Cerezo et al. guarantee the local
    /// observable escapes the plateau.
    Logarithmic(usize),
    /// `c * n`. Deep enough to approach a 2-design, where even a local observable
    /// develops a plateau.
    Linear(usize),
}

impl DepthPolicy {
    pub fn depth_for(&self, n: usize) -> usize {
        match *self {
            DepthPolicy::Constant(d) => d.max(1),
            DepthPolicy::Logarithmic(c) => (c as f64 * (n as f64).log2()).ceil().max(1.0) as usize,
            DepthPolicy::Linear(c) => (c * n).max(1),
        }
    }
}

/// Which observable the cost is built from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CostLocality {
    /// `Z_0`, acting on one qubit.
    Local,
    /// `Z` on every qubit.
    Global,
}

impl CostLocality {
    pub fn observable(&self, n: usize) -> Observable {
        match self {
            CostLocality::Local => Observable::z(0),
            CostLocality::Global => Observable::z_global(n),
        }
    }
}

/// A hardware-efficient ansatz: `RY`, `RZ` per qubit per layer, then a ring of `CZ`.
///
/// No data encoding. The plateau is a property of the variational circuit and the
/// observable, not of any environment, so introducing an observation here would only add a
/// variable that has to be held fixed.
pub fn hardware_efficient(n: usize, depth: usize) -> Circuit {
    let mut c = Circuit::new(n);
    let mut idx = 0;
    for _ in 0..depth {
        for q in 0..n {
            c.ry(q, Angle::param(idx));
            idx += 1;
            c.rz(q, Angle::param(idx));
            idx += 1;
        }
        if n > 1 {
            if n == 2 {
                c.cz(0, 1);
            } else {
                for q in 0..n {
                    c.cz(q, (q + 1) % n);
                }
            }
        }
    }
    c
}

/// One point of a plateau sweep.
#[derive(Clone, Debug, PartialEq)]
pub struct PlateauPoint {
    pub num_qubits: usize,
    pub depth: usize,
    /// Variance of `d<O>/dtheta_1` over the sampled parameter vectors.
    pub variance: f64,
    /// Mean gradient. Should sit at zero; a drift away from it means the sample is too
    /// small to trust the variance either.
    pub mean: f64,
    pub samples: usize,
}

/// Gradient variance at one qubit count, probing the **first** parameter.
///
/// Part I 6.7 specifies `d<O>/dtheta_1`, and that specification turns out to be load
/// bearing rather than arbitrary. Probing a parameter in the middle of the circuit -- which
/// seems more representative, since it sees an entangled state -- silently breaks the
/// measurement: for a local observable many mid-circuit parameters have gradients that are
/// *identically* zero, not merely small. Sampling one of those gives a variance around
/// `1e-33`, which then enters the exponential fit as a catastrophic collapse and makes the
/// local observable look worse than the global one, exactly inverting the result.
///
/// The zeros are structural, not numerical: Phase 1 verified the gradient engine against
/// Yao.jl to `1e-12`, and they persist across seeds and parameter draws. A simple light-cone
/// argument predicts many of them but not all, so the honest summary is that a fixed
/// mid-circuit probe is not a reliable estimator of "the" gradient variance for a local
/// cost. The first parameter acts on the same qubit the local observable reads, so it is
/// never structurally decoupled from it.
///
/// [`gradient_variance_at`] exposes the probe index so this choice is visible rather than
/// buried, and `structurally_zero_probes` in the tests documents the phenomenon.
pub fn gradient_variance(
    n: usize,
    depth: usize,
    locality: CostLocality,
    samples: usize,
    seed: u64,
) -> PlateauPoint {
    gradient_variance_at(n, depth, locality, samples, seed, 0)
}

/// As [`gradient_variance`], with an explicit probe parameter index.
pub fn gradient_variance_at(
    n: usize,
    depth: usize,
    locality: CostLocality,
    samples: usize,
    seed: u64,
    probe: usize,
) -> PlateauPoint {
    let circuit = hardware_efficient(n, depth);
    let observable = locality.observable(n);
    let num_params = circuit.num_params();
    assert!(
        probe < num_params,
        "probe {probe} outside {num_params} parameters"
    );

    let mut values = Vec::with_capacity(samples);
    for i in 0..samples {
        let params = random_params(
            seed.wrapping_add(i as u64).wrapping_mul(0x9E37_79B9),
            num_params,
        );
        let grad = adjoint::grad(&circuit, &params, &observable);
        values.push(grad[probe]);
    }

    let mean = values.iter().sum::<f64>() / samples as f64;
    // Population variance: this is the variance of the sampling distribution itself, not an
    // estimate of a parent population's, and the theory predicts E[grad] = 0 exactly.
    let variance = values.iter().map(|g| (g - mean) * (g - mean)).sum::<f64>() / samples as f64;

    PlateauPoint {
        num_qubits: n,
        depth,
        variance,
        mean,
        samples,
    }
}

/// A full sweep over qubit counts.
pub fn sweep(
    qubit_range: std::ops::RangeInclusive<usize>,
    depth: DepthPolicy,
    locality: CostLocality,
    samples: usize,
    seed: u64,
) -> Vec<PlateauPoint> {
    qubit_range
        .map(|n| gradient_variance(n, depth.depth_for(n), locality, samples, seed))
        .collect()
}

/// The fitted decay of an exponential model `Var ~ 2^(-rate * n)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExponentialFit {
    /// `rate` in `Var ~ 2^(-rate * n)`. Larger means a faster collapse.
    pub rate: f64,
    pub intercept: f64,
    /// Coefficient of determination on `log2(Var)`. Near 1 means the exponential model
    /// describes the data; well below it means the decay is not exponential, which is
    /// itself the finding for a local observable.
    pub r_squared: f64,
}

/// Least squares on `log2(variance)` against `n`.
///
/// Reported as a *rate*, so a barren plateau is a positive number and the flat case is near
/// zero. Part I 6.7 asks for the fitted exponent to be printed next to the curve, because
/// seeing `2^(-1.98 n)` appear is more convincing than the curve alone.
pub fn fit_exponential(points: &[PlateauPoint]) -> ExponentialFit {
    let usable: Vec<(f64, f64)> = points
        .iter()
        .filter(|p| p.variance > 0.0)
        .map(|p| (p.num_qubits as f64, p.variance.log2()))
        .collect();
    assert!(
        usable.len() >= 2,
        "need at least two positive variances to fit"
    );

    let n = usable.len() as f64;
    let mean_x = usable.iter().map(|(x, _)| x).sum::<f64>() / n;
    let mean_y = usable.iter().map(|(_, y)| y).sum::<f64>() / n;

    let sxy: f64 = usable
        .iter()
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum();
    let sxx: f64 = usable
        .iter()
        .map(|(x, _)| (x - mean_x) * (x - mean_x))
        .sum();
    let slope = sxy / sxx;
    let intercept = mean_y - slope * mean_x;

    let ss_tot: f64 = usable
        .iter()
        .map(|(_, y)| (y - mean_y) * (y - mean_y))
        .sum();
    let ss_res: f64 = usable
        .iter()
        .map(|(x, y)| {
            let predicted = slope * x + intercept;
            (y - predicted) * (y - predicted)
        })
        .sum();
    let r_squared = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        1.0
    };

    ExponentialFit {
        rate: -slope,
        intercept,
        r_squared,
    }
}

/// Random parameters drawn uniformly, as [`gradient_variance`] does, exposed so a caller
/// can reproduce a specific sample.
pub fn sample_params(seed: u64, count: usize) -> Vec<f64> {
    let mut r = rng(seed);
    (0..count)
        .map(|_| r.gen_range(-std::f64::consts::PI..std::f64::consts::PI))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_policies_scale_as_advertised() {
        assert_eq!(DepthPolicy::Constant(3).depth_for(10), 3);
        assert_eq!(DepthPolicy::Linear(2).depth_for(5), 10);
        assert_eq!(DepthPolicy::Logarithmic(1).depth_for(8), 3);
        assert_eq!(DepthPolicy::Logarithmic(2).depth_for(16), 8);
        // Never zero, whatever the arithmetic says.
        assert!(DepthPolicy::Logarithmic(1).depth_for(1) >= 1);
    }

    #[test]
    fn the_mean_gradient_is_zero() {
        // The theory says E[grad] = 0 by symmetry of the parameter distribution. If this
        // drifted, the variance would be measuring the wrong thing.
        for n in [2usize, 4, 6] {
            let p = gradient_variance(n, 3, CostLocality::Global, 400, 1);
            assert!(
                p.mean.abs() < 5.0 * (p.variance / p.samples as f64).sqrt() + 1e-12,
                "n={n}: mean {} against standard error of the mean",
                p.mean
            );
        }
    }

    #[test]
    fn the_global_observable_collapses_exponentially() {
        let points = sweep(
            2..=10,
            DepthPolicy::Logarithmic(2),
            CostLocality::Global,
            300,
            7,
        );
        let fit = fit_exponential(&points);
        assert!(
            fit.rate > 0.8,
            "global rate {} is not an exponential collapse: {points:?}",
            fit.rate
        );
        assert!(
            fit.r_squared > 0.9,
            "global fit is not exponential, R^2 = {}",
            fit.r_squared
        );
    }

    #[test]
    fn the_local_observable_survives_at_shallow_depth() {
        // Cerezo et al.: shallow depth plus a local cost gives at worst polynomial decay.
        // The decisive comparison is against the global sweep on the same circuits.
        let depth = DepthPolicy::Logarithmic(2);
        let local = fit_exponential(&sweep(2..=10, depth, CostLocality::Local, 300, 7));
        let global = fit_exponential(&sweep(2..=10, depth, CostLocality::Global, 300, 7));
        assert!(
            global.rate > 2.0 * local.rate,
            "global rate {} should far exceed local rate {}",
            global.rate,
            local.rate
        );
        // The local variance must stay usable at the top of the sweep. This is the claim
        // that matters practically: a gradient you can still measure.
        let deepest = sweep(10..=10, depth, CostLocality::Local, 300, 7);
        assert!(
            deepest[0].variance > 1e-3,
            "local variance at n=10 collapsed to {}",
            deepest[0].variance
        );
    }

    #[test]
    fn a_mid_circuit_probe_hits_structurally_zero_gradients() {
        // Documents why the probe index is not a free choice. These zeros are structural --
        // the gradient engine is verified against Yao.jl to 1e-12 -- and they are what makes
        // a mid-circuit probe invert the global-versus-local result.
        let (n, depth) = (10usize, 7usize);
        let num_params = hardware_efficient(n, depth).num_params();

        let middle = gradient_variance_at(n, depth, CostLocality::Local, 64, 7, num_params / 2);
        assert!(
            middle.variance < 1e-30,
            "expected an identically-zero gradient at the middle probe, got {}",
            middle.variance
        );

        // The first parameter, which the protocol specifies, is well away from zero.
        let first = gradient_variance_at(n, depth, CostLocality::Local, 64, 7, 0);
        assert!(
            first.variance > 1e-3,
            "first-parameter variance {} should be usable",
            first.variance
        );
    }

    #[test]
    fn deep_circuits_bury_the_local_observable_too() {
        // The Cerezo escape is conditional on shallow depth. At linear depth the circuit
        // approaches a 2-design and the locality of the observable stops helping, so the
        // local curve collapses as well. Reporting this keeps the panel honest: the answer
        // is "shallow and local", not "local".
        let shallow = fit_exponential(&sweep(
            2..=8,
            DepthPolicy::Logarithmic(2),
            CostLocality::Local,
            200,
            7,
        ));
        let deep = fit_exponential(&sweep(
            2..=8,
            DepthPolicy::Linear(2),
            CostLocality::Local,
            200,
            7,
        ));
        assert!(
            deep.rate > shallow.rate,
            "deep local rate {} should exceed shallow local rate {}",
            deep.rate,
            shallow.rate
        );
    }

    #[test]
    fn variance_is_reproducible_from_a_seed() {
        let a = gradient_variance(4, 3, CostLocality::Global, 128, 99);
        let b = gradient_variance(4, 3, CostLocality::Global, 128, 99);
        assert_eq!(a, b);
    }

    #[test]
    fn the_fit_recovers_a_planted_exponent() {
        // Feed the fitter data it should get exactly right, so a failure elsewhere is not
        // blamed on the regression.
        let points: Vec<PlateauPoint> = (2..=12)
            .map(|n| PlateauPoint {
                num_qubits: n,
                depth: 1,
                variance: 2.0_f64.powf(-1.75 * n as f64),
                mean: 0.0,
                samples: 1,
            })
            .collect();
        let fit = fit_exponential(&points);
        assert!((fit.rate - 1.75).abs() < 1e-9, "rate {}", fit.rate);
        assert!((fit.r_squared - 1.0).abs() < 1e-9, "R^2 {}", fit.r_squared);
    }
}
