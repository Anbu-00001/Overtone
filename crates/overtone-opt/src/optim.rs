//! Four optimisers, and the one thing they must all be able to do first.
//!
//! Part V 5.2 asks for a race: vanilla gradient, natural gradient, CMA-ES and Nelder-Mead,
//! deep in a barren plateau, all four flat. The obvious failure mode of that demonstration
//! is that a *broken* optimiser also produces a flat line, and produces it for a reason
//! that has nothing to do with barren plateaus. So `tests/optimisers.rs` runs all four on
//! the sphere and on Rosenbrock, where the answer is known, and requires each to find it.
//! Nothing here may be trusted on the plateau until it is trusted off the plateau.
//!
//! Everything is budgeted in **shots**, not in iterations. That is the axis Arrasmith et
//! al. measure -- their result is that the shot count needed for progress grows
//! exponentially with `n` -- and it is the only axis on which a gradient method and a
//! simplex method can be compared at all, since one function evaluation and one gradient
//! evaluation are not the same purchase.

use overtone_mps::linalg::jacobi_eigh;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

/// A cost function that knows what it has spent.
///
/// Implementors count shots inside [`Self::value`]; the optimisers only ever ask whether
/// the budget is gone. Keeping the meter inside the objective rather than inside each
/// optimiser is what stops a subtly different accounting rule from deciding the race.
pub trait Objective {
    /// The cost at `x`, charged to the budget.
    fn value(&mut self, x: &[f64]) -> f64;
    /// Shots spent so far.
    fn spent(&self) -> u64;
    /// Total allowance.
    fn budget(&self) -> u64;
    fn exhausted(&self) -> bool {
        self.spent() >= self.budget()
    }
    fn dim(&self) -> usize;
}

/// What an optimiser leaves behind.
#[derive(Clone, Debug)]
pub struct Run {
    /// The point with the lowest *measured* value. Under shot noise this selects on the
    /// noise as much as on the landscape, so a race scored on it flatters every optimiser
    /// equally and flatters the noisiest most. Score on [`Self::final_x`] instead, or on
    /// the exact cost at either -- never on `best_value`.
    pub best_x: Vec<f64>,
    /// Where the optimiser actually stopped: the simplex's best vertex, CMA-ES's
    /// distribution mean, the gradient method's current iterate.
    pub final_x: Vec<f64>,
    /// Best value ever *evaluated*, in the objective's own units.
    pub best_value: f64,
    /// Value at the starting point, for the same objective.
    pub start_value: f64,
    pub evaluations: usize,
    pub shots: u64,
}

impl Run {
    /// How far the optimiser descended. Positive is progress.
    pub fn improvement(&self) -> f64 {
        self.start_value - self.best_value
    }
}

/// Which of Part V 5.2's four is being run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Optimiser {
    /// Gradient descent on a parameter-shift gradient, every shifted term paid for in
    /// shots. The hardware-honest gradient.
    Gradient,
    /// The same gradient, preconditioned by the quantum Fisher information matrix.
    NaturalGradient,
    /// Hansen's covariance-matrix adaptation evolution strategy.
    Cmaes,
    /// The Nelder-Mead simplex.
    NelderMead,
}

impl Optimiser {
    pub fn name(&self) -> &'static str {
        match self {
            Optimiser::Gradient => "gradient",
            Optimiser::NaturalGradient => "natural gradient",
            Optimiser::Cmaes => "CMA-ES",
            Optimiser::NelderMead => "Nelder-Mead",
        }
    }

    /// True when the optimiser needs derivative information the objective must supply.
    pub fn is_gradient_based(&self) -> bool {
        matches!(self, Optimiser::Gradient | Optimiser::NaturalGradient)
    }
}

fn track(best: &mut (Vec<f64>, f64), x: &[f64], v: f64) {
    if v < best.1 {
        best.0.copy_from_slice(x);
        best.1 = v;
    }
}

/// The Nelder-Mead simplex, with Nelder and Mead's own coefficients.
///
/// `alpha = 1` reflection, `gamma = 2` expansion, `rho = 0.5` contraction, `sigma = 0.5`
/// shrink. No restarts and no adaptive coefficients: the point of the demonstration is the
/// textbook algorithm, and a tuned variant would invite the objection that the flatline was
/// a tuning artefact.
pub fn nelder_mead<O: Objective>(obj: &mut O, x0: &[f64], step: f64) -> Run {
    let n = x0.len();
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());
    for i in 0..n {
        let mut x = x0.to_vec();
        x[i] += step;
        simplex.push(x);
    }
    let mut f: Vec<f64> = simplex.iter().map(|x| obj.value(x)).collect();
    let start_value = f[0];
    let mut best = (x0.to_vec(), f[0]);
    for (x, &v) in simplex.iter().zip(f.iter()) {
        track(&mut best, x, v);
    }
    let mut evaluations = n + 1;

    while !obj.exhausted() {
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| f[a].partial_cmp(&f[b]).unwrap());
        let (lo, hi, second) = (order[0], order[n], order[n - 1]);

        // Centroid of everything but the worst.
        let mut centroid = vec![0.0; n];
        for &i in &order[..n] {
            for (c, v) in centroid.iter_mut().zip(&simplex[i]) {
                *c += v / n as f64;
            }
        }
        let combine = |a: &[f64], b: &[f64], t: f64| -> Vec<f64> {
            a.iter().zip(b).map(|(p, q)| p + t * (q - p)).collect()
        };

        let reflected = combine(&centroid, &simplex[hi], -1.0);
        let fr = obj.value(&reflected);
        evaluations += 1;
        track(&mut best, &reflected, fr);

        if fr < f[lo] {
            let expanded = combine(&centroid, &simplex[hi], -2.0);
            let fe = obj.value(&expanded);
            evaluations += 1;
            track(&mut best, &expanded, fe);
            if fe < fr {
                simplex[hi] = expanded;
                f[hi] = fe;
            } else {
                simplex[hi] = reflected;
                f[hi] = fr;
            }
        } else if fr < f[second] {
            simplex[hi] = reflected;
            f[hi] = fr;
        } else {
            let contracted = if fr < f[hi] {
                combine(&centroid, &simplex[hi], -0.5)
            } else {
                combine(&centroid, &simplex[hi], 0.5)
            };
            let fc = obj.value(&contracted);
            evaluations += 1;
            track(&mut best, &contracted, fc);
            let accept = if fr < f[hi] { fc <= fr } else { fc < f[hi] };
            if accept {
                simplex[hi] = contracted;
                f[hi] = fc;
            } else {
                for i in 0..=n {
                    if i == lo {
                        continue;
                    }
                    simplex[i] = combine(&simplex[lo], &simplex[i], 0.5);
                    f[i] = obj.value(&simplex[i]);
                    evaluations += 1;
                    track(&mut best, &simplex[i], f[i]);
                }
            }
        }
    }

    let lo = (0..=n)
        .min_by(|&a, &b| f[a].partial_cmp(&f[b]).unwrap())
        .unwrap();
    Run {
        best_x: best.0,
        final_x: simplex[lo].clone(),
        best_value: best.1,
        start_value,
        evaluations,
        shots: obj.spent(),
    }
}

/// (mu/mu_w, lambda)-CMA-ES with Hansen's default parameters.
///
/// `lambda = 4 + floor(3 ln n)`, `mu = lambda/2`, log-weights, and the standard rank-one
/// plus rank-mu covariance update. `C^(-1/2)` comes from a Jacobi eigendecomposition --
/// the same routine the eigenbasis panel uses -- refreshed every `n/10` generations, which
/// is Hansen's own lazy-update rule.
pub fn cmaes<O: Objective>(obj: &mut O, x0: &[f64], sigma0: f64, rng: &mut ChaCha8Rng) -> Run {
    let n = x0.len();
    let nf = n as f64;
    let lambda = 4 + (3.0 * nf.ln()).floor() as usize;
    let mu = lambda / 2;

    let weights_raw: Vec<f64> = (0..mu)
        .map(|i| ((mu as f64 + 0.5).ln()) - ((i + 1) as f64).ln())
        .collect();
    let wsum: f64 = weights_raw.iter().sum();
    let weights: Vec<f64> = weights_raw.iter().map(|w| w / wsum).collect();
    let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

    let cc = (4.0 + mu_eff / nf) / (nf + 4.0 + 2.0 * mu_eff / nf);
    let cs = (mu_eff + 2.0) / (nf + mu_eff + 5.0);
    let c1 = 2.0 / ((nf + 1.3).powi(2) + mu_eff);
    let cmu = (1.0 - c1).min(2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((nf + 2.0).powi(2) + mu_eff));
    let damps = 1.0 + 2.0 * (0.0f64).max(((mu_eff - 1.0) / (nf + 1.0)).sqrt() - 1.0) + cs;
    let chi_n = nf.sqrt() * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));

    let mut mean = x0.to_vec();
    let mut sigma = sigma0;
    let mut p_c = vec![0.0; n];
    let mut p_s = vec![0.0; n];
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        c[i * n + i] = 1.0;
    }
    let (mut b, mut d) = (identity(n), vec![1.0; n]);
    let mut eigen_age = 0usize;

    let start_value = obj.value(x0);
    let mut best = (x0.to_vec(), start_value);
    let mut evaluations = 1usize;
    let mut generation = 0usize;

    while !obj.exhausted() {
        generation += 1;
        let mut zs = Vec::with_capacity(lambda);
        let mut ys = Vec::with_capacity(lambda);
        let mut fs = Vec::with_capacity(lambda);
        for _ in 0..lambda {
            let z: Vec<f64> = (0..n).map(|_| standard_normal(rng)).collect();
            // y = B D z
            let mut y = vec![0.0; n];
            for i in 0..n {
                for k in 0..n {
                    y[i] += b[i * n + k] * d[k] * z[k];
                }
            }
            let x: Vec<f64> = (0..n).map(|i| mean[i] + sigma * y[i]).collect();
            let v = obj.value(&x);
            evaluations += 1;
            track(&mut best, &x, v);
            zs.push(z);
            ys.push(y);
            fs.push(v);
        }

        let mut order: Vec<usize> = (0..lambda).collect();
        order.sort_by(|&a, &b2| fs[a].partial_cmp(&fs[b2]).unwrap());

        let mut y_w = vec![0.0; n];
        let mut z_w = vec![0.0; n];
        for (rank, &i) in order.iter().take(mu).enumerate() {
            for k in 0..n {
                y_w[k] += weights[rank] * ys[i][k];
                z_w[k] += weights[rank] * zs[i][k];
            }
        }
        for k in 0..n {
            mean[k] += sigma * y_w[k];
        }

        // p_s uses C^(-1/2) y_w, which in the eigenbasis is just B z_w.
        let mut bz = vec![0.0; n];
        for i in 0..n {
            for k in 0..n {
                bz[i] += b[i * n + k] * z_w[k];
            }
        }
        for k in 0..n {
            p_s[k] = (1.0 - cs) * p_s[k] + (cs * (2.0 - cs) * mu_eff).sqrt() * bz[k];
        }
        let ps_norm = p_s.iter().map(|v| v * v).sum::<f64>().sqrt();
        let h_sigma = ps_norm / (1.0 - (1.0 - cs).powi(2 * generation as i32)).sqrt() / chi_n
            < 1.4 + 2.0 / (nf + 1.0);
        let hs = if h_sigma { 1.0 } else { 0.0 };
        for k in 0..n {
            p_c[k] = (1.0 - cc) * p_c[k] + hs * (cc * (2.0 - cc) * mu_eff).sqrt() * y_w[k];
        }

        let delta = (1.0 - hs) * cc * (2.0 - cc);
        for i in 0..n {
            for j in 0..n {
                let mut rank_mu = 0.0;
                for (rank, &idx) in order.iter().take(mu).enumerate() {
                    rank_mu += weights[rank] * ys[idx][i] * ys[idx][j];
                }
                c[i * n + j] = (1.0 - c1 - cmu) * c[i * n + j]
                    + c1 * (p_c[i] * p_c[j] + delta * c[i * n + j])
                    + cmu * rank_mu;
            }
        }
        sigma *= ((cs / damps) * (ps_norm / chi_n - 1.0)).exp();
        if !sigma.is_finite() || sigma <= 0.0 {
            break;
        }

        eigen_age += 1;
        if eigen_age as f64 > nf / 10.0 {
            eigen_age = 0;
            let mut sym = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    sym[i * n + j] = 0.5 * (c[i * n + j] + c[j * n + i]);
                }
            }
            let mut vecs = vec![0.0; n * n];
            let vals = jacobi_eigh(&mut sym, n, &mut vecs);
            if vals.iter().all(|v| v.is_finite() && *v > 0.0) {
                d = vals.iter().map(|v| v.sqrt()).collect();
                b = vecs;
            }
        }
    }

    Run {
        best_x: best.0,
        final_x: mean,
        best_value: best.1,
        start_value,
        evaluations,
        shots: obj.spent(),
    }
}

fn identity(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// Box-Muller, from the seeded ChaCha stream. No entropy source anywhere in this workspace.
fn standard_normal(rng: &mut ChaCha8Rng) -> f64 {
    let u1: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
    let u2: f64 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// An objective that can also be differentiated, at a price.
pub trait Differentiable: Objective {
    /// The gradient at `x`, charged to the same budget as [`Objective::value`].
    fn gradient(&mut self, x: &[f64]) -> Vec<f64>;
    /// A metric to precondition with, row-major and `dim x dim`, or `None` for the identity.
    ///
    /// Returning `Some` costs nothing here, and that is deliberate: the quantum Fisher
    /// information would cost shots on hardware, so giving it away free can only *help*
    /// the natural gradient. If it flatlines anyway, the result is stronger than it would
    /// have been with honest accounting.
    fn metric(&mut self, x: &[f64]) -> Option<Vec<f64>>;
}

/// Plain gradient descent with a fixed step, optionally preconditioned by a metric.
pub fn descend<O: Differentiable>(obj: &mut O, x0: &[f64], rate: f64, natural: bool) -> Run {
    let n = x0.len();
    let mut x = x0.to_vec();
    let start_value = obj.value(&x);
    let mut best = (x.clone(), start_value);
    let mut evaluations = 1usize;

    while !obj.exhausted() {
        let mut g = obj.gradient(&x);
        if natural {
            if let Some(metric) = obj.metric(&x) {
                g = solve_regularised(&metric, &g, n, 1e-3);
            }
        }
        for i in 0..n {
            x[i] -= rate * g[i];
        }
        let v = obj.value(&x);
        evaluations += 1;
        track(&mut best, &x, v);
        if !v.is_finite() {
            break;
        }
    }

    Run {
        best_x: best.0,
        final_x: x,
        best_value: best.1,
        start_value,
        evaluations,
        shots: obj.spent(),
    }
}

/// `(A + eps I) y = b` by Gaussian elimination with partial pivoting.
///
/// The QFIM is singular whenever the circuit has redundant parameters, which a
/// hardware-efficient ansatz always does, so the shift is not optional.
pub fn solve_regularised(a: &[f64], b: &[f64], n: usize, eps: f64) -> Vec<f64> {
    let mut m = vec![0.0; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            m[i * (n + 1) + j] = a[i * n + j];
        }
        m[i * (n + 1) + i] += eps;
        m[i * (n + 1) + n] = b[i];
    }
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&r1, &r2| {
                m[r1 * (n + 1) + col]
                    .abs()
                    .partial_cmp(&m[r2 * (n + 1) + col].abs())
                    .unwrap()
            })
            .unwrap();
        if m[pivot * (n + 1) + col].abs() < 1e-14 {
            continue;
        }
        if pivot != col {
            for k in col..=n {
                m.swap(col * (n + 1) + k, pivot * (n + 1) + k);
            }
        }
        let p = m[col * (n + 1) + col];
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = m[row * (n + 1) + col] / p;
            if factor == 0.0 {
                continue;
            }
            for k in col..=n {
                m[row * (n + 1) + k] -= factor * m[col * (n + 1) + k];
            }
        }
    }
    (0..n)
        .map(|i| {
            let p = m[i * (n + 1) + i];
            if p.abs() < 1e-14 {
                0.0
            } else {
                m[i * (n + 1) + n] / p
            }
        })
        .collect()
}
