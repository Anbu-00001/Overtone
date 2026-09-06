//! REINFORCE with a baseline, plus Adam.
//!
//! The estimator is the standard one: for a batch of `(s, a, r)` drawn on-policy,
//!
//! ```text
//! grad J ~= (1/N) sum_i grad log pi(a_i | s_i) * (r_i - b)
//! ```
//!
//! with `b` the batch mean return. Subtracting a baseline leaves the estimator unbiased --
//! `E[grad log pi] = 0` at fixed `s` -- while cutting its variance, which on this bandit is
//! the difference between converging in a few hundred steps and wandering.
//!
//! Because `SpectralControl` is a one-step bandit there is no discounting, no bootstrapping
//! and no credit assignment across time. That is deliberate. The environment exists to make
//! a spectral statement legible, and every mechanism not needed for that is a mechanism
//! that could be blamed when a number comes out wrong.

use rand::Rng;

use crate::env::SpectralControl;
use crate::policy::Policy;

/// Adam, as the optimiser for the policy parameters.
#[derive(Clone, Debug)]
pub struct Adam {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    m: Vec<f64>,
    v: Vec<f64>,
    t: u64,
}

impl Adam {
    pub fn new(num_params: usize, learning_rate: f64) -> Self {
        Adam {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m: vec![0.0; num_params],
            v: vec![0.0; num_params],
            t: 0,
        }
    }

    /// One ascent step: `params += lr * adam(gradient)`.
    pub fn ascend(&mut self, params: &mut [f64], gradient: &[f64]) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..params.len() {
            let g = gradient[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] += self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrainConfig {
    /// Episodes per gradient step.
    pub batch_size: usize,
    /// Total episodes. The number of gradient steps is `episodes / batch_size`.
    pub episodes: usize,
    pub learning_rate: f64,
    /// Quadrature nodes for the exact return recorded in the trace.
    pub eval_nodes: usize,
    /// Record a trace row every this many gradient steps.
    pub record_every: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            batch_size: 50,
            episodes: 5000,
            learning_rate: 0.05,
            eval_nodes: 512,
            record_every: 10,
        }
    }
}

/// One row of the training trace.
#[derive(Clone, Debug, PartialEq)]
pub struct TraceRow {
    pub step: usize,
    pub episodes: usize,
    /// Mean reward actually collected in this batch. Noisy.
    pub sampled_return: f64,
    /// Exact expected return by quadrature. This is the one to plot and to assert on.
    pub exact_return: f64,
}

#[derive(Clone, Debug)]
pub struct TrainOutcome {
    pub params: Vec<f64>,
    pub trace: Vec<TraceRow>,
    pub final_exact_return: f64,
}

/// One REINFORCE gradient estimate from a fresh on-policy batch.
pub fn reinforce_gradient<R: Rng + ?Sized>(
    policy: &Policy,
    params: &[f64],
    env: &SpectralControl,
    batch_size: usize,
    rng: &mut R,
) -> (Vec<f64>, f64) {
    let mut states = Vec::with_capacity(batch_size);
    let mut actions = Vec::with_capacity(batch_size);
    let mut rewards = Vec::with_capacity(batch_size);

    for _ in 0..batch_size {
        let s = env.sample_state(rng);
        let a = policy.sample(params, &[s], rng);
        rewards.push(env.reward(s, a));
        states.push(s);
        actions.push(a);
    }

    let baseline = rewards.iter().sum::<f64>() / batch_size as f64;

    let mut grad = vec![0.0; policy.num_params()];
    for i in 0..batch_size {
        let advantage = rewards[i] - baseline;
        if advantage == 0.0 {
            continue;
        }
        let score = policy.grad_log_prob(params, &[states[i]], actions[i]);
        for (g, sc) in grad.iter_mut().zip(&score) {
            *g += advantage * sc;
        }
    }
    for g in &mut grad {
        *g /= batch_size as f64;
    }

    (grad, baseline)
}

/// Train a policy on `SpectralControl-k` with REINFORCE.
pub fn train<R: Rng + ?Sized>(
    policy: &Policy,
    initial_params: &[f64],
    env: &SpectralControl,
    cfg: &TrainConfig,
    rng: &mut R,
) -> TrainOutcome {
    let mut params = initial_params.to_vec();
    let mut adam = Adam::new(params.len(), cfg.learning_rate);
    let mut trace = Vec::new();
    let steps = cfg.episodes / cfg.batch_size;

    for step in 0..steps {
        let (grad, sampled) = reinforce_gradient(policy, &params, env, cfg.batch_size, rng);
        adam.ascend(&mut params, &grad);

        if step % cfg.record_every == 0 || step + 1 == steps {
            trace.push(TraceRow {
                step,
                episodes: (step + 1) * cfg.batch_size,
                sampled_return: sampled,
                exact_return: policy.expected_return(&params, env.k, cfg.eval_nodes),
            });
        }
    }

    let final_exact_return = policy.expected_return(&params, env.k, cfg.eval_nodes);
    TrainOutcome {
        params,
        trace,
        final_exact_return,
    }
}

/// Initial parameters: small random variational angles, `lambda` at `lambda_init`, softmax
/// weights at `-1` and `+1`.
///
/// # `lambda_init` is not a free choice, and this is the interesting part
///
/// With `L = 1` on `SpectralControl-3` the reachable frequencies are `lambda * {-1, 0, 1}`,
/// so the agent can only score where `lambda` is near `k`. The return as a function of
/// `lambda` is therefore not a hill but a **resonance curve**: a main peak at `lambda ~ k`
/// flanked by decaying sidelobes, exactly the shape a detuned oscillator traces out.
///
/// Measured on a two-qubit `L = 1` RAW-PQC against `SpectralControl-3`
/// (`examples/lambda_scan.rs` reproduces it):
///
/// ```text
/// main peak       lambda = 3.05   J = 0.502
/// sidelobes       lambda = 0.66   J = 0.021
///                 lambda = 5.47   J = 0.083
///                 lambda = 7.48   J = 0.051
/// capture range   lambda_0 in roughly [1.8, 4.2] reaches the main peak
/// ```
///
/// Gradient ascent started outside that capture range climbs a sidelobe and stays there. So
/// "the agent learns lambda and the peak slides until it locks on" is true, but only from
/// within the capture range -- which is a property of resonance, not a defect. It is the
/// same reason a radio needs a coarse tune before the fine tune, and
/// [`coarse_tune_lambda`] is that coarse tune.
///
/// Starting at `lambda = 1` -- the pinned-equivalent value, which looks like the natural
/// default -- lands outside the capture range for `k = 3` and converges to `J = 0.021`.
/// Report this rather than quietly initialising at the answer.
pub fn initial_params<R: Rng + ?Sized>(
    policy: &Policy,
    spread: f64,
    lambda_init: f64,
    rng: &mut R,
) -> Vec<f64> {
    let mut params = vec![0.0; policy.num_params()];

    for p in params.iter_mut().take(policy.ansatz.num_variational()) {
        *p = rng.gen_range(-spread..spread);
    }
    for i in policy.ansatz.lambda_range() {
        params[i] = lambda_init;
    }
    let w = policy.weight_range();
    if !w.is_empty() {
        params[w.start] = -1.0;
        params[w.start + 1] = 1.0;
    }

    params
}

/// Coarse tune: sweep `lambda` over `[0, max_lambda]` and return the value with the best
/// exact return, leaving every other parameter alone.
///
/// This is a scan, not an optimisation. Its job is to drop the fine tune inside the capture
/// range described on [`initial_params`], after which gradient ascent locks on by itself.
pub fn coarse_tune_lambda(
    policy: &Policy,
    params: &[f64],
    k: usize,
    max_lambda: f64,
    steps: usize,
) -> f64 {
    let range = policy.ansatz.lambda_range();
    if range.is_empty() {
        return 1.0;
    }

    let mut probe = params.to_vec();
    let mut best = (f64::NEG_INFINITY, 1.0);

    for i in 0..=steps {
        let lambda = max_lambda * (i as f64) / (steps as f64);
        for idx in range.clone() {
            probe[idx] = lambda;
        }
        let j = policy.expected_return(&probe, k, 256);
        if j > best.0 {
            best = (j, lambda);
        }
    }

    best.1
}
