//! Part V 4's claims, each with a test that fails when it stops being true.

use overtone_rl::ansatz::{Scaling, SpectralControlAnsatz};
use overtone_rl::distributional::{measured_observable, score_bias, QuantileCritic, TwoAtoms};
use overtone_rl::{Policy, SpectralControl};
use overtone_sim::measure::Budget;
use overtone_sim::random::{random_params, rng};

fn setup() -> (SpectralControl, Policy, Vec<f64>) {
    let env = SpectralControl::new(2);
    let ansatz = SpectralControlAnsatz::build(2, 2, Scaling::Pinned, true);
    let policy = Policy::raw(ansatz);
    let params = random_params(31, policy.num_params());
    (env, policy, params)
}

/// The quantile midpoints are Dabney et al.'s `(2i - 1) / 2N`, not `i / N`. Getting this
/// wrong biases every learned quantile toward one tail and is invisible in a plot.
#[test]
fn quantile_midpoints_follow_the_paper() {
    let c = QuantileCritic::new(4, 0.0);
    let taus: Vec<f64> = (0..4).map(|i| c.tau(i)).collect();
    assert_eq!(taus, vec![0.125, 0.375, 0.625, 0.875]);
}

/// The critic reaches the discretisation floor of its own representation -- the best any
/// `N`-quantile approximation of this distribution can do -- rather than merely getting
/// close to the mean.
#[test]
fn the_critic_reaches_its_representation_floor() {
    let (env, policy, params) = setup();
    let s = 0.4;
    let truth = TwoAtoms::for_state(&env, policy.prob_action1(&params, &[s]), s);

    let mut ideal = QuantileCritic::new(21, 0.0);
    for i in 0..ideal.len() {
        ideal.theta[i] = truth.inverse_cdf(ideal.tau(i));
    }
    let floor = ideal.wasserstein_to(|p| truth.inverse_cdf(p), 2001);

    let mut r = rng(7);
    let mut critic = QuantileCritic::new(21, 0.0);
    for t in 0..100_000 {
        critic.observe(truth.sample(&mut r), 0.05 / (1.0 + t as f64 / 2_000.0), 0.0);
    }
    let learned = critic.wasserstein_to(|p| truth.inverse_cdf(p), 2001);
    assert!(
        learned < floor * 1.15,
        "learned {learned} against a floor of {floor}"
    );
}

/// The published `kappa = 1` does not converge on returns of order one: the Huber
/// quadratic swallows every residual and the fixed point becomes an expectile. This is the
/// correction the module documents, so it is tested rather than asserted.
#[test]
fn kappa_one_fits_an_expectile_not_a_quantile() {
    let (env, policy, params) = setup();
    let s = 0.4;
    let truth = TwoAtoms::for_state(&env, policy.prob_action1(&params, &[s]), s);

    let run = |kappa: f64| {
        let mut r = rng(7);
        let mut critic = QuantileCritic::new(21, 0.0);
        for t in 0..100_000 {
            critic.observe(
                truth.sample(&mut r),
                0.05 / (1.0 + t as f64 / 2_000.0),
                kappa,
            );
        }
        critic.wasserstein_to(|p| truth.inverse_cdf(p), 2001)
    };
    let huber = run(1.0);
    let plain = run(0.0);
    assert!(
        huber > 10.0 * plain,
        "kappa=1 gave {huber} and kappa=0 gave {plain}; the correction no longer holds"
    );
}

/// The closed form is the closed form: two atoms with the policy's own probability, mean
/// equal to the environment's expected reward.
#[test]
fn the_two_atom_form_matches_the_environment() {
    let (env, policy, params) = setup();
    for &s in &[-2.0, -0.4, 0.0, 0.9, 2.7] {
        let p1 = policy.prob_action1(&params, &[s]);
        let atoms = TwoAtoms::for_state(&env, p1, s);
        let direct = p1 * env.reward(s, 1) + (1.0 - p1) * env.reward(s, 0);
        assert!((atoms.mean() - direct).abs() < 1e-12, "{s}");
    }
}

/// Shots do not narrow the return distribution. Part V 4 says they do; they do not, and a
/// panel that showed them doing so would be showing an artefact.
#[test]
fn shots_do_not_narrow_the_return_distribution() {
    let (env, policy, params) = setup();
    let s = 0.4;
    // 2000 episodes puts the standard error of the spread near 0.01, inside the tolerance the
    // assertion needs, and the top of the range is 1000 shots rather than 10000 because the
    // claim is that this column does *not* move: one more decade of shots on a flat quantity
    // buys nothing and costs a factor of ten every time the tests run unoptimised.
    let spread = |shots: usize| {
        let mut r = rng(shots as u64 + 1);
        let xs: Vec<f64> = (0..2_000)
            .map(|_| {
                let z = measured_observable(&policy, &params, &[s], Budget::Shots(shots), &mut r);
                let p = ((1.0 + z) * 0.5).clamp(0.0, 1.0);
                let a = usize::from(rand::Rng::gen::<f64>(&mut r) < p);
                env.reward(s, a)
            })
            .collect();
        let m = xs.iter().sum::<f64>() / xs.len() as f64;
        (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64).sqrt()
    };
    let (few, many) = (spread(10), spread(1_000));
    let exact = TwoAtoms::for_state(&env, policy.prob_action1(&params, &[s]), s)
        .variance()
        .sqrt();
    assert!(
        (few - many).abs() < 0.03,
        "spread moved with the shot budget: {few} against {many}"
    );
    assert!(
        (few - exact).abs() < 0.02,
        "{few} against the exact {exact}"
    );
}

/// The score function is nonlinear in the measured observable, so a small shot budget
/// biases it. The bias must fall as the budget grows, or the dial measures nothing.
#[test]
fn the_score_function_bias_falls_with_the_shot_budget() {
    let (_, policy, params) = setup();
    let mut r = rng(19);
    let cheap = score_bias(
        &policy,
        &params,
        &[0.4],
        1,
        Budget::Shots(10),
        1e-3,
        4_000,
        &mut r,
    );
    let dear = score_bias(
        &policy,
        &params,
        &[0.4],
        1,
        Budget::Shots(1_000),
        1e-3,
        4_000,
        &mut r,
    );
    assert!(
        cheap > 20.0 * dear,
        "bias did not fall with the budget: {cheap} at 10 shots, {dear} at 1000"
    );
}

/// `Budget::Exact` must reproduce the policy's own observable exactly, or every comparison
/// in this module is between two different quantities.
#[test]
fn the_exact_budget_reproduces_the_policy_observable() {
    let (_, policy, params) = setup();
    let mut r = rng(3);
    for &s in &[-1.0, 0.25, 2.0] {
        let measured = measured_observable(&policy, &params, &[s], Budget::Exact, &mut r);
        let direct = policy.observable_value(&params, &[s]);
        assert!((measured - direct).abs() < 1e-15, "{s}");
    }
}
