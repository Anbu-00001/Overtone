//! The four optimisers must work before a flat line from any of them means anything.
//!
//! Part V 5.2's demonstration has an obvious way to be accidentally right: a broken
//! optimiser flatlines everywhere, including in a barren plateau, and the demo would look
//! exactly the same. These tests are the difference between "all four flatline" and "all
//! four are broken", and they run on functions whose minima are known in closed form.

use overtone_opt::optim::{cmaes, descend, nelder_mead, Differentiable, Objective};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// A noiseless analytic objective with a counter, so the same budget machinery is exercised.
struct Analytic {
    f: fn(&[f64]) -> f64,
    grad: fn(&[f64]) -> Vec<f64>,
    n: usize,
    spent: u64,
    allowance: u64,
}

impl Analytic {
    fn new(f: fn(&[f64]) -> f64, grad: fn(&[f64]) -> Vec<f64>, n: usize, allowance: u64) -> Self {
        Analytic {
            f,
            grad,
            n,
            spent: 0,
            allowance,
        }
    }
}

impl Objective for Analytic {
    fn value(&mut self, x: &[f64]) -> f64 {
        self.spent += 1;
        (self.f)(x)
    }
    fn spent(&self) -> u64 {
        self.spent
    }
    fn budget(&self) -> u64 {
        self.allowance
    }
    fn dim(&self) -> usize {
        self.n
    }
}

impl Differentiable for Analytic {
    fn gradient(&mut self, x: &[f64]) -> Vec<f64> {
        self.spent += self.n as u64;
        (self.grad)(x)
    }
    fn metric(&mut self, _x: &[f64]) -> Option<Vec<f64>> {
        None
    }
}

fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}
fn sphere_grad(x: &[f64]) -> Vec<f64> {
    x.iter().map(|v| 2.0 * v).collect()
}

/// Rosenbrock, the standard hard case for a simplex: a curved valley with a flat floor.
fn rosenbrock(x: &[f64]) -> f64 {
    (0..x.len() - 1)
        .map(|i| 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2))
        .sum()
}
fn rosenbrock_grad(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut g = vec![0.0; n];
    for i in 0..n - 1 {
        let t = x[i + 1] - x[i] * x[i];
        g[i] += -400.0 * x[i] * t - 2.0 * (1.0 - x[i]);
        g[i + 1] += 200.0 * t;
    }
    g
}

#[test]
fn nelder_mead_finds_the_sphere_minimum() {
    let mut obj = Analytic::new(sphere, sphere_grad, 5, 4_000);
    let run = nelder_mead(&mut obj, &[1.5; 5], 0.5);
    assert!(
        run.best_value < 1e-8,
        "Nelder-Mead left the sphere at {}",
        run.best_value
    );
}

#[test]
fn nelder_mead_finds_the_rosenbrock_valley() {
    let mut obj = Analytic::new(rosenbrock, rosenbrock_grad, 4, 60_000);
    let run = nelder_mead(&mut obj, &[-1.2, 1.0, -1.2, 1.0], 0.5);
    assert!(
        run.best_value < 1e-4,
        "Nelder-Mead stalled on Rosenbrock at {}",
        run.best_value
    );
}

#[test]
fn cmaes_finds_the_sphere_minimum() {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let mut obj = Analytic::new(sphere, sphere_grad, 6, 6_000);
    let run = cmaes(&mut obj, &[2.0; 6], 0.6, &mut rng);
    assert!(
        run.best_value < 1e-10,
        "CMA-ES left the sphere at {}",
        run.best_value
    );
}

#[test]
fn cmaes_finds_the_rosenbrock_valley() {
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let mut obj = Analytic::new(rosenbrock, rosenbrock_grad, 4, 40_000);
    let run = cmaes(&mut obj, &[-1.2, 1.0, -1.2, 1.0], 0.5, &mut rng);
    assert!(
        run.best_value < 1e-6,
        "CMA-ES stalled on Rosenbrock at {}",
        run.best_value
    );
}

#[test]
fn gradient_descent_finds_the_sphere_minimum() {
    let mut obj = Analytic::new(sphere, sphere_grad, 5, 2_000);
    let run = descend(&mut obj, &[1.5; 5], 0.1, false);
    assert!(
        run.best_value < 1e-12,
        "gradient descent left the sphere at {}",
        run.best_value
    );
}

/// The preconditioned path must reduce to the plain one when the metric is the identity,
/// or "natural gradient" in the race would be a second, differently-tuned gradient lane.
#[test]
fn natural_gradient_matches_plain_under_an_identity_metric() {
    let mut a = Analytic::new(sphere, sphere_grad, 4, 800);
    let plain = descend(&mut a, &[1.0, -1.0, 0.5, 0.25], 0.05, false);
    let mut b = Analytic::new(sphere, sphere_grad, 4, 800);
    let natural = descend(&mut b, &[1.0, -1.0, 0.5, 0.25], 0.05, true);
    assert_eq!(plain.best_x, natural.best_x);
}

/// Regularised solve: `(A + eps I) y = b` on a singular `A` must not blow up, because the
/// QFIM of any hardware-efficient ansatz is singular.
#[test]
fn regularised_solve_survives_a_singular_metric() {
    use overtone_opt::optim::solve_regularised;
    let a = vec![1.0, 1.0, 1.0, 1.0];
    let y = solve_regularised(&a, &[1.0, 1.0], 2, 1e-3);
    assert!(y.iter().all(|v| v.is_finite()));
    // (A + eps I) y should reproduce b to the size of eps.
    let r0 = (1.0 + 1e-3) * y[0] + y[1];
    let r1 = y[0] + (1.0 + 1e-3) * y[1];
    assert!(
        (r0 - 1.0).abs() < 1e-6 && (r1 - 1.0).abs() < 1e-6,
        "{r0} {r1}"
    );
}
