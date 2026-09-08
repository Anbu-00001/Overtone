//! Part V 4: the shot-budget dial, and the two distributions it is easy to confuse.
//!
//! `cargo run --release -p overtone-rl --example shot_dial`

use overtone_rl::ansatz::{Scaling, SpectralControlAnsatz};
use overtone_rl::distributional::{measured_observable, score_bias, QuantileCritic, TwoAtoms};
use overtone_rl::{Policy, SpectralControl};
use overtone_sim::measure::Budget;
use overtone_sim::random::{random_params, rng};

fn main() {
    let env = SpectralControl::new(2);
    let ansatz = SpectralControlAnsatz::build(2, 2, Scaling::Pinned, true);
    let policy = Policy::raw(ansatz);
    let params = random_params(31, policy.num_params());
    let s = 0.4;

    let p1 = policy.prob_action1(&params, &[s]);
    let truth = TwoAtoms::for_state(&env, p1, s);
    println!("state s = {s}, pi(1|s) = {p1:.6}");
    println!(
        "the return distribution is two atoms: {:+.6} w.p. {:.4}, {:+.6} otherwise",
        truth.high, truth.p_high, truth.low
    );
    println!(
        "  mean {:+.6}   standard deviation {:.6}",
        truth.mean(),
        truth.variance().sqrt()
    );

    println!("\na quantile critic learns it from samples alone -- if kappa is set for the scale");
    println!(
        "{:>10}  {:>12}  {:>12}  {:>12}  {:>12}",
        "samples", "kappa 1.0", "kappa 0.1", "kappa 0.01", "kappa 0"
    );
    for &total in &[1_000usize, 10_000, 100_000] {
        let mut cells = Vec::new();
        for &kappa in &[1.0f64, 0.1, 0.01, 0.0] {
            let mut r = rng(7);
            let mut critic = QuantileCritic::new(21, 0.0);
            for t in 0..total {
                let rate = 0.05 / (1.0 + t as f64 / 2_000.0);
                critic.observe(truth.sample(&mut r), rate, kappa);
            }
            cells.push(critic.wasserstein_to(|p| truth.inverse_cdf(p), 2001));
        }
        println!(
            "{total:>10}  {:>12.6}  {:>12.6}  {:>12.6}  {:>12.6}",
            cells[0], cells[1], cells[2], cells[3]
        );
    }
    let mut ideal = QuantileCritic::new(21, 0.0);
    for i in 0..ideal.len() {
        ideal.theta[i] = truth.inverse_cdf(ideal.tau(i));
    }
    println!(
        "  the floor for 21 quantiles on this distribution is {:.6}",
        ideal.wasserstein_to(|p| truth.inverse_cdf(p), 2001)
    );
    println!("  kappa trades the fixed point against the step size. At kappa = 0 the subgradient");
    println!("  is sign(u) and the critic reaches that floor. At kappa = 1 -- the published");
    println!("  default, chosen for Atari returns in the hundreds -- every residual on a return");
    println!("  in [-1, 1] falls inside the Huber quadratic, so the fixed point is an expectile");
    println!("  and no number of samples fixes it. In between the fixed point is right and the");
    println!("  gradient is scaled by kappa, so convergence is just slower.");

    println!("\nthe shot dial, on both distributions at once");
    println!(
        "{:>8}  {:>18}  {:>18}  {:>16}",
        "shots", "return sd (aleatoric)", "sd of J-hat", "score bias"
    );
    for &shots in &[10usize, 100, 1_000, 10_000] {
        let budget = Budget::Shots(shots);
        let mut r = rng(shots as u64);

        // The aleatoric spread: sample actual returns through a measured policy.
        let returns: Vec<f64> = (0..4000)
            .map(|_| {
                let z = measured_observable(&policy, &params, &[s], budget, &mut r);
                let p = ((1.0 + z) * 0.5).clamp(0.0, 1.0);
                let a = usize::from(rand::Rng::gen::<f64>(&mut r) < p);
                env.reward(s, a)
            })
            .collect();
        let m = returns.iter().sum::<f64>() / returns.len() as f64;
        let sd =
            (returns.iter().map(|x| (x - m).powi(2)).sum::<f64>() / returns.len() as f64).sqrt();

        // The epistemic spread: repeated 200-episode estimates of the same J.
        let estimates: Vec<f64> = (0..200)
            .map(|_| {
                (0..200)
                    .map(|_| {
                        let z = measured_observable(&policy, &params, &[s], budget, &mut r);
                        let p = ((1.0 + z) * 0.5).clamp(0.0, 1.0);
                        let a = usize::from(rand::Rng::gen::<f64>(&mut r) < p);
                        env.reward(s, a)
                    })
                    .sum::<f64>()
                    / 200.0
            })
            .collect();
        let me = estimates.iter().sum::<f64>() / estimates.len() as f64;
        let sde = (estimates.iter().map(|x| (x - me).powi(2)).sum::<f64>()
            / estimates.len() as f64)
            .sqrt();

        let bias = score_bias(&policy, &params, &[s], 1, budget, 1e-3, 4000, &mut r);
        println!("{shots:>8}  {sd:>18.6}  {sde:>18.6}  {bias:>16.6}");
    }
    println!("  the first column is flat: shots do not narrow the return distribution, and a");
    println!("  critic that claimed they did would be wrong. The second falls as 1/sqrt of the");
    println!("  episode count, not of the shot count -- with two atoms the aleatoric term");
    println!("  dominates. The third is what a shot budget really costs: the score function is");
    println!("  nonlinear in the measured z, so a cheap measurement biases the gradient.");
}
