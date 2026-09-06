//! Gradient checks for the policy layer.
//!
//! Phase 1 established that `overtone-sim` differentiates `<Z_0>` correctly. That does not
//! establish that the policy differentiates *itself* correctly: the score function, the
//! softmax weight gradients and the chain rule through the Born rule are all new code, and
//! all of them are places a sign survives undetected because REINFORCE still trains,
//! merely slower.
//!
//! So every analytic gradient here is checked against a central finite difference of the
//! quantity it claims to differentiate.

use overtone_rl::env::SpectralControl;
use overtone_rl::policy::{Policy, PolicyKind};
use overtone_rl::reinforce::{initial_params, reinforce_gradient};
use overtone_rl::{Scaling, SpectralControlAnsatz};
use overtone_sim::random::rng;

const FD_EPS: f64 = 1e-6;
const FD_TOL: f64 = 1e-6;

fn policies() -> Vec<(&'static str, Policy)> {
    vec![
        (
            "raw-pinned",
            Policy::raw(SpectralControlAnsatz::build(2, 3, Scaling::Pinned, true)),
        ),
        (
            "raw-trainable",
            Policy::raw(SpectralControlAnsatz::build(2, 2, Scaling::Trainable, true)),
        ),
        (
            "softmax-pinned",
            Policy::softmax(
                SpectralControlAnsatz::build(2, 3, Scaling::Pinned, true),
                1.0,
            ),
        ),
        (
            "softmax-trainable",
            Policy::softmax(
                SpectralControlAnsatz::build(3, 2, Scaling::Trainable, true),
                1.7,
            ),
        ),
    ]
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn grad_prob_matches_finite_difference() {
    for (name, policy) in policies() {
        for seed in 0..5u64 {
            let mut r = rng(seed);
            let params = initial_params(&policy, 1.0, 1.3, &mut r);
            for s in [-2.0, -0.4, 0.0, 0.9, 2.7] {
                let analytic = policy.grad_prob_action1(&params, &[s]);

                let mut fd = vec![0.0; params.len()];
                let mut probe = params.clone();
                for i in 0..params.len() {
                    let original = probe[i];
                    probe[i] = original + FD_EPS;
                    let plus = policy.prob_action1(&probe, &[s]);
                    probe[i] = original - FD_EPS;
                    let minus = policy.prob_action1(&probe, &[s]);
                    probe[i] = original;
                    fd[i] = (plus - minus) / (2.0 * FD_EPS);
                }

                let d = max_abs_diff(&analytic, &fd);
                assert!(
                    d < FD_TOL,
                    "{name} seed={seed} s={s}: grad pi differs by {d:e}"
                );
            }
        }
    }
}

#[test]
fn score_function_matches_finite_difference_of_log_prob() {
    for (name, policy) in policies() {
        for seed in 0..5u64 {
            let mut r = rng(seed);
            let params = initial_params(&policy, 1.0, 1.3, &mut r);
            for s in [-1.8, -0.3, 0.6, 2.2] {
                for action in 0..2 {
                    // Skip states where the Born rule pins the probability to a boundary;
                    // the score function is genuinely singular there and the clamp in the
                    // policy is a deliberate deviation, not something to test against.
                    let p = policy.prob(&params, &[s], action);
                    if !(1e-4..=1.0 - 1e-4).contains(&p) {
                        continue;
                    }

                    let analytic = policy.grad_log_prob(&params, &[s], action);

                    let mut fd = vec![0.0; params.len()];
                    let mut probe = params.clone();
                    for i in 0..params.len() {
                        let original = probe[i];
                        probe[i] = original + FD_EPS;
                        let plus = policy.prob(&probe, &[s], action).ln();
                        probe[i] = original - FD_EPS;
                        let minus = policy.prob(&probe, &[s], action).ln();
                        probe[i] = original;
                        fd[i] = (plus - minus) / (2.0 * FD_EPS);
                    }

                    let d = max_abs_diff(&analytic, &fd);
                    assert!(
                        d < 1e-5,
                        "{name} seed={seed} s={s} a={action}: score differs by {d:e}"
                    );
                }
            }
        }
    }
}

#[test]
fn analytic_return_gradient_matches_finite_difference() {
    for (name, policy) in policies() {
        let mut r = rng(2);
        let params = initial_params(&policy, 1.0, 1.3, &mut r);
        let k = 3;
        let nodes = 512;

        let analytic = policy.analytic_return_grad(&params, k, nodes);

        let mut fd = vec![0.0; params.len()];
        let mut probe = params.clone();
        for i in 0..params.len() {
            let original = probe[i];
            probe[i] = original + FD_EPS;
            let plus = policy.expected_return(&probe, k, nodes);
            probe[i] = original - FD_EPS;
            let minus = policy.expected_return(&probe, k, nodes);
            probe[i] = original;
            fd[i] = (plus - minus) / (2.0 * FD_EPS);
        }

        let d = max_abs_diff(&analytic, &fd);
        assert!(d < 1e-5, "{name}: analytic grad J differs by {d:e}");
    }
}

#[test]
fn reinforce_is_an_unbiased_estimator_of_the_policy_gradient() {
    // The RL analogue of Phase 1's adjoint-versus-parameter-shift check: the sampled
    // estimator is compared against the quantity it is supposed to estimate. Averaging many
    // independent batches should converge on the analytic gradient, and the residual should
    // shrink like 1/sqrt(N).
    let env = SpectralControl::new(3);
    let policy = Policy::raw(SpectralControlAnsatz::build(2, 3, Scaling::Pinned, true));
    let mut r = rng(19);
    let params = initial_params(&policy, 0.8, 1.0, &mut r);

    let analytic = policy.analytic_return_grad(&params, env.k, 2048);

    let mut mean = vec![0.0; policy.num_params()];
    let batches = 4000;
    for _ in 0..batches {
        let (g, _) = reinforce_gradient(&policy, &params, &env, 32, &mut r);
        for (m, gi) in mean.iter_mut().zip(&g) {
            *m += gi;
        }
    }
    for m in &mut mean {
        *m /= batches as f64;
    }

    let scale = analytic
        .iter()
        .fold(0.0_f64, |a, b| a.max(b.abs()))
        .max(1e-3);
    let d = max_abs_diff(&mean, &analytic);
    assert!(
        d < 0.08 * scale.max(1.0),
        "REINFORCE mean over {batches} batches differs from the analytic gradient by {d:e}\n\
         analytic: {analytic:?}\n     mean: {mean:?}"
    );
}

#[test]
fn softmax_beta_scales_the_score_function() {
    // Lemma 1 of Jerbi et al. carries an explicit factor of beta. Doubling beta at fixed
    // observable output must double the weight-gradient entries.
    let ansatz = SpectralControlAnsatz::build(2, 2, Scaling::Pinned, true);
    let one = Policy::softmax(ansatz.clone(), 1.0);
    let two = Policy::softmax(ansatz, 2.0);

    let mut r = rng(4);
    let mut params = initial_params(&one, 0.7, 1.0, &mut r);
    // Equal weights make the policy uniform, so pi(b|s) = 1/2 independently of beta and the
    // weight gradient is beta * z * (delta - 1/2): exactly proportional to beta.
    let w = one.weight_range();
    params[w.start] = 0.0;
    params[w.start + 1] = 0.0;

    let g1 = one.grad_log_prob(&params, &[0.8], 1);
    let g2 = two.grad_log_prob(&params, &[0.8], 1);
    for i in w {
        assert!(
            (g2[i] - 2.0 * g1[i]).abs() < 1e-12,
            "weight gradient {i} did not scale with beta: {} vs {}",
            g2[i],
            g1[i]
        );
    }
}

#[test]
fn raw_policy_probability_is_the_born_rule() {
    let policy = Policy::raw(SpectralControlAnsatz::build(2, 3, Scaling::Pinned, true));
    assert_eq!(policy.kind, PolicyKind::Raw);
    let mut r = rng(8);
    let params = initial_params(&policy, 1.5, 1.0, &mut r);

    for s in [-2.5, -0.7, 0.2, 1.9] {
        let z = policy.observable_value(&params, &[s]);
        let p1 = policy.prob_action1(&params, &[s]);
        assert!((p1 - (1.0 + z) / 2.0).abs() < 1e-14, "s={s}");
        assert!((0.0..=1.0).contains(&p1), "probability {p1} out of range");
        assert!(
            (policy.prob(&params, &[s], 0) + p1 - 1.0).abs() < 1e-14,
            "probabilities must sum to one"
        );
    }
}
