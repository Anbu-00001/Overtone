// How does the exact return depend on the input scaling lambda, at L = 1 on
// SpectralControl-3? The reachable frequencies are lambda * {-1, 0, 1}, so the agent can
// only score where lambda ~ 3. This scan asks what the optimiser actually sees.
use overtone_rl::policy::Policy;
use overtone_rl::reinforce::initial_params;
use overtone_rl::{Scaling, SpectralControlAnsatz};
use overtone_sim::random::rng;

fn main() {
    let ansatz = SpectralControlAnsatz::build(2, 1, Scaling::Trainable, true);
    let policy = Policy::raw(ansatz);
    let mut r = rng(7);
    // Fix the variational angles at a decent setting, then sweep lambda alone.
    let mut best = (f64::MIN, vec![]);
    for seed in 0..40u64 {
        let mut rr = rng(seed);
        let mut p = initial_params(&policy, 3.0, 1.0, &mut rr);
        for i in policy.ansatz.lambda_range() {
            p[i] = 3.0;
        }
        let j = policy.expected_return(&p, 3, 1024);
        if j > best.0 {
            best = (j, p);
        }
    }
    let mut p = best.1;
    println!(
        "best-of-40 variational init at lambda=3 gives J = {:.5}\n",
        best.0
    );

    println!("{:>7} {:>12}", "lambda", "J");
    let mut prev = f64::NAN;
    let mut peaks = vec![];
    let mut vals = vec![];
    for i in 0..=400 {
        let lam = 0.02 * i as f64;
        for idx in policy.ansatz.lambda_range() {
            p[idx] = lam;
        }
        let j = policy.expected_return(&p, 3, 1024);
        vals.push((lam, j));
        if i % 20 == 0 {
            println!("{lam:>7.2} {j:>12.6}");
        }
        prev = j;
    }
    let _ = prev;
    // local maxima
    for w in vals.windows(3) {
        if w[1].1 > w[0].1 && w[1].1 > w[2].1 && w[1].1 > 0.01 {
            peaks.push(w[1]);
        }
    }
    println!("\nlocal maxima with J > 0.01:");
    for (lam, j) in &peaks {
        println!("  lambda = {lam:.2}   J = {j:.5}");
    }

    // Capture range: from which starting lambda does plain gradient ascent on lambda reach 3?
    println!("\ncapture range (gradient ascent on lambda alone, 3000 steps, lr 0.01):");
    for start in [0.5, 1.0, 1.5, 2.0, 2.2, 2.5, 2.8, 3.5, 4.0, 4.5] {
        for idx in policy.ansatz.lambda_range() {
            p[idx] = start;
        }
        let mut q = p.clone();
        for _ in 0..3000 {
            let g = policy.analytic_return_grad(&q, 3, 512);
            for idx in policy.ansatz.lambda_range() {
                q[idx] += 0.01 * g[idx];
            }
        }
        let end: f64 = policy.ansatz.lambda_range().map(|i| q[i]).next().unwrap();
        println!(
            "  start {start:>4.1} -> lambda {end:>6.3}, J = {:.5}",
            policy.expected_return(&q, 3, 1024)
        );
    }
    let _ = &mut r;
}
