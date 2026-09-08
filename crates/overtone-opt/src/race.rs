//! The race itself: same circuit, same start, same shot allowance, four optimisers.
//!
//! # How a lane is scored
//!
//! Not by the best value the optimiser *saw*. Under shot noise the minimum of many noisy
//! draws is mostly a statement about how many draws were taken, so scoring on it rewards
//! the optimiser that evaluated most and rewards noise itself. Every lane is scored by the
//! **exact** cost at the point the optimiser stopped -- a number the optimiser never had
//! access to, computed off the meter afterwards.
//!
//! That choice is the difference between a demonstration and an artefact, and it is
//! measurable: at four qubits with a hundred shots per evaluation the two scorings disagree
//! by more than the entire effect being reported.

use overtone_spec::plateau::{CostLocality, DepthPolicy};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::optim::{cmaes, descend, nelder_mead, Optimiser, Run};
use crate::plateau::PlateauCost;
use overtone_sim::measure::Budget;

/// One race.
#[derive(Clone, Copy, Debug)]
pub struct RaceConfig {
    pub num_qubits: usize,
    pub depth: DepthPolicy,
    pub locality: CostLocality,
    /// Shots per expectation value, or `Exact` for the unfair control.
    pub budget: Budget,
    /// Total shots each optimiser may spend.
    pub allowance: u64,
    pub seed: u64,
    /// Step for the gradient methods and the initial simplex edge / CMA-ES sigma.
    pub step: f64,
}

impl RaceConfig {
    /// The configuration Part V 5.2 describes: global observable, linear depth -- deep
    /// enough for the ansatz to approach a 2-design, which is what makes the plateau.
    pub fn plateau(num_qubits: usize, budget: Budget, allowance: u64, seed: u64) -> RaceConfig {
        RaceConfig {
            num_qubits,
            depth: DepthPolicy::Linear(1),
            locality: CostLocality::Global,
            budget,
            allowance,
            seed,
            step: 0.1,
        }
    }
}

/// What one optimiser did.
#[derive(Clone, Debug)]
pub struct Lane {
    pub optimiser: Optimiser,
    /// Exact cost at the start, before any measurement.
    pub start_exact: f64,
    /// Exact cost where the optimiser stopped. The score.
    pub final_exact: f64,
    /// Exact cost at its own best-measured point, to show how much of `best_value` was luck.
    pub best_exact: f64,
    /// The value it believed it had reached.
    pub reported_best: f64,
    pub evaluations: usize,
    pub shots: u64,
    /// Standard error of one measurement at the start point.
    pub noise_floor: f64,
}

impl Lane {
    /// Descent in the exact cost. Positive is real progress.
    pub fn progress(&self) -> f64 {
        self.start_exact - self.final_exact
    }

    /// True when the lane moved further than a single measurement could resolve.
    ///
    /// Not a tuned threshold: it is the standard error of the estimator the optimiser was
    /// reading, which is the smallest difference it could have detected.
    pub fn beat_the_noise(&self) -> bool {
        self.progress() > self.noise_floor
    }
}

/// Run all four optimisers on one configuration, from the same start point.
pub fn race(cfg: RaceConfig) -> Vec<Lane> {
    ALL.iter().map(|&which| lane(cfg, which)).collect()
}

/// Part V 5.2's four, in the order the spec lists them.
pub const ALL: [Optimiser; 4] = [
    Optimiser::Gradient,
    Optimiser::NaturalGradient,
    Optimiser::Cmaes,
    Optimiser::NelderMead,
];

fn new_cost(cfg: &RaceConfig) -> PlateauCost {
    PlateauCost::new(
        cfg.num_qubits,
        cfg.depth,
        cfg.locality,
        cfg.budget,
        cfg.allowance,
        ChaCha8Rng::seed_from_u64(cfg.seed),
    )
}

fn drive(which: Optimiser, cost: &mut PlateauCost, x0: &[f64], cfg: &RaceConfig) -> Run {
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed ^ 0xc3a);
    match which {
        Optimiser::Gradient => descend(cost, x0, cfg.step, false),
        Optimiser::NaturalGradient => descend(cost, x0, cfg.step, true),
        Optimiser::Cmaes => cmaes(cost, x0, cfg.step, &mut rng),
        Optimiser::NelderMead => nelder_mead(cost, x0, cfg.step),
    }
}

/// The smallest shots-per-evaluation at which a given optimiser makes real progress.
///
/// A doubling search, which is what turns Arrasmith et al.'s claim -- "the numbers of shots
/// required in the optimization grows exponentially with the number of qubits" -- into a
/// curve this repository can plot. `None` means it never did, by `max_shots`.
///
/// Reported **per optimiser**, not for the field, because the four do not fail together and
/// pooling them would hide the one result here that the spec did not anticipate.
///
/// Progress is required on a majority of `seeds`. One seed decides this by coin flip: the
/// quantity being tested is whether a descent exceeded a standard error, and a single draw
/// crosses that line about as often as not.
pub fn shots_to_progress(
    which: Optimiser,
    num_qubits: usize,
    seeds: &[u64],
    min_shots: usize,
    max_shots: usize,
    evaluations_allowed: u64,
) -> Option<usize> {
    let mut s = min_shots;
    while s <= max_shots {
        let wins = seeds
            .iter()
            .filter(|&&seed| {
                let cfg = RaceConfig::plateau(
                    num_qubits,
                    Budget::Shots(s),
                    evaluations_allowed * s as u64,
                    seed,
                );
                lane(cfg, which).beat_the_noise()
            })
            .count();
        if wins * 2 > seeds.len() {
            return Some(s);
        }
        s *= 2;
    }
    None
}

/// One optimiser on one configuration.
pub fn lane(cfg: RaceConfig, which: Optimiser) -> Lane {
    let probe = new_cost(&cfg);
    let p = probe.num_params();
    let x0 = overtone_sim::random::random_params(cfg.seed ^ 0x5eed, p);
    let start_exact = probe.exact(&x0);
    let noise_floor = probe.noise_floor(&x0);
    let mut cost = new_cost(&cfg);
    let run = drive(which, &mut cost, &x0, &cfg);
    Lane {
        optimiser: which,
        start_exact,
        final_exact: cost.exact(&run.final_x),
        best_exact: cost.exact(&run.best_x),
        reported_best: run.best_value,
        evaluations: run.evaluations,
        shots: run.shots,
        noise_floor,
    }
}

/// A fit of `shots ~ 2^(a n)` to a shots-to-progress sweep.
///
/// Arrasmith et al.'s claim is *exponential* growth in the shot requirement, so a straight
/// line through `log2(shots)` against `n` is the right fit and its slope is the number to
/// quote. The `R^2` travels with it: this repository does not report an exponent without
/// one, because a curve that has saturated will still yield a slope.
#[derive(Clone, Copy, Debug)]
pub struct ShotScaling {
    /// `a` in `shots ~ 2^(a n)`.
    pub exponent: f64,
    pub intercept: f64,
    pub r_squared: f64,
    pub points: usize,
}

/// Least squares on `(n, log2 shots)`.
pub fn fit_shot_scaling(points: &[(usize, usize)]) -> ShotScaling {
    let n = points.len();
    if n < 2 {
        return ShotScaling {
            exponent: f64::NAN,
            intercept: f64::NAN,
            r_squared: f64::NAN,
            points: n,
        };
    }
    let xs: Vec<f64> = points.iter().map(|(q, _)| *q as f64).collect();
    let ys: Vec<f64> = points.iter().map(|(_, s)| (*s as f64).log2()).collect();
    let mx = xs.iter().sum::<f64>() / n as f64;
    let my = ys.iter().sum::<f64>() / n as f64;
    let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let sxx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    let slope = if sxx == 0.0 { f64::NAN } else { sxy / sxx };
    let intercept = my - slope * mx;
    let ss_res: f64 = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| (y - (slope * x + intercept)).powi(2))
        .sum();
    let ss_tot: f64 = ys.iter().map(|y| (y - my).powi(2)).sum();
    ShotScaling {
        exponent: slope,
        intercept,
        r_squared: if ss_tot == 0.0 {
            f64::NAN
        } else {
            1.0 - ss_res / ss_tot
        },
        points: n,
    }
}
