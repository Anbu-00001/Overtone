use overtone_rl::ceiling;
use overtone_rl::policy::{Policy, PolicyKind};
use overtone_rl::reinforce::{initial_params, train, TrainConfig};
use overtone_rl::{Scaling, SpectralControl, SpectralControlAnsatz};
use overtone_sim::random::rng;

fn main() {
    let k = 3;
    let env = SpectralControl::new(k);
    println!(
        "{:>4} {:>8} {:>10} {:>10} {:>10}",
        "L", "policy", "ceiling", "trained", "gap"
    );

    for l in [1usize, 2, 3, 4, 6] {
        for kind in [PolicyKind::Raw, PolicyKind::Softmax] {
            let ansatz = SpectralControlAnsatz::build(2, l, Scaling::Pinned, true);
            let c = ansatz.frequency_ceiling(0);
            let policy = match kind {
                PolicyKind::Raw => Policy::raw(ansatz),
                PolicyKind::Softmax => Policy::softmax(ansatz, 1.0),
            };
            let mut r = rng(7);
            let p0 = initial_params(&policy, 0.3, 1.0, &mut r);
            let cfg = TrainConfig {
                episodes: 5000,
                batch_size: 50,
                learning_rate: 0.05,
                ..Default::default()
            };
            let out = train(&policy, &p0, &env, &cfg, &mut r);
            let cap = ceiling::ceiling(c, k, ceiling::DEFAULT_GRID);
            println!(
                "{l:>4} {:>8} {cap:>10.5} {:>10.5} {:>10.5}",
                if kind == PolicyKind::Raw {
                    "raw"
                } else {
                    "softmax"
                },
                out.final_exact_return,
                cap - out.final_exact_return
            );
        }
    }

    println!("\ntrainable lambda, L = 1 (ceiling 1 < k = 3):");
    for kind in [PolicyKind::Raw, PolicyKind::Softmax] {
        let ansatz = SpectralControlAnsatz::build(2, 1, Scaling::Trainable, true);
        let policy = match kind {
            PolicyKind::Raw => Policy::raw(ansatz),
            PolicyKind::Softmax => Policy::softmax(ansatz, 1.0),
        };
        let mut r = rng(7);
        let p0 = initial_params(&policy, 0.3, 1.0, &mut r);
        let cfg = TrainConfig {
            episodes: 20000,
            batch_size: 50,
            learning_rate: 0.05,
            ..Default::default()
        };
        let out = train(&policy, &p0, &env, &cfg, &mut r);
        let lam: Vec<f64> = policy
            .ansatz
            .lambda_range()
            .map(|i| out.params[i])
            .collect();
        println!(
            "  {:>8}: J = {:>9.5}   lambda = {:?}",
            if kind == PolicyKind::Raw {
                "raw"
            } else {
                "softmax"
            },
            out.final_exact_return,
            lam.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>()
        );
    }
}
