// Part I 12: "Watch the softmax-leakage claim carefully. Verify numerically before writing
// it down: confirm the harmonics appear at the predicted orders and that they vanish when
// the policy is RAW. If the numbers disagree with the story, change the story."
//
// This is that verification. It is run before the claim is written, not after.
use overtone_rl::policy::Policy;
use overtone_rl::reinforce::{initial_params, train, TrainConfig};
use overtone_rl::{Scaling, SpectralControl, SpectralControlAnsatz};
use overtone_sim::random::rng;
use overtone_spec::spectrum_of;

const N: usize = 512;

fn bars(label: &str, mag: &[f64], ceiling: usize, upto: usize) {
    print!("{label:>18} |");
    for (k, m) in mag.iter().enumerate().take(upto + 1) {
        let ch = if *m < 1e-9 {
            '.'
        } else if *m < 0.01 {
            ':'
        } else if *m < 0.05 {
            'o'
        } else {
            '#'
        };
        if k == ceiling {
            print!("{ch}|");
        } else {
            print!("{ch}");
        }
    }
    println!();
}

fn main() {
    let k = 3;
    let env = SpectralControl::new(k);
    println!("SpectralControl-{k}.  '|' marks the encoding ceiling.  . <1e-9  : <0.01  o <0.05  # bigger\n");
    println!(
        "{:>18}  {:>7} {:>10} {:>10} {:>9} {:>7}",
        "", "ceiling", "max above", "sum above", "leak/band", "J"
    );

    for layers in [2usize, 3, 4] {
        for softmax in [false, true] {
            let ansatz = SpectralControlAnsatz::build(2, layers, Scaling::Pinned, true);
            let c = ansatz.frequency_ceiling(0);
            let policy = if softmax {
                Policy::softmax(ansatz, 1.0)
            } else {
                Policy::raw(ansatz)
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

            let s = spectrum_of(N, c, |x| policy.prob_action1(&out.params, &[x]));
            let label = format!("L={layers} {}", if softmax { "softmax" } else { "raw    " });
            println!(
                "{label:>18}  {c:>7} {:>10.2e} {:>10.2e} {:>9.4} {:>7.4}",
                s.energy_above_ceiling(),
                s.total_above_ceiling(),
                s.leakage_ratio(),
                out.final_exact_return
            );
            bars(&label, &s.magnitude, c, 16);
        }
    }

    // The cleanest possible case: is the leaked energy concentrated at odd harmonics?
    println!("\nleaked bars above the ceiling (magnitude > 1e-6), L=3 softmax, ceiling 3:");
    let ansatz = SpectralControlAnsatz::build(2, 3, Scaling::Pinned, true);
    let policy = Policy::softmax(ansatz, 1.0);
    let mut r = rng(7);
    let p0 = initial_params(&policy, 0.3, 1.0, &mut r);
    let out = train(
        &policy,
        &p0,
        &env,
        &TrainConfig {
            episodes: 5000,
            batch_size: 50,
            learning_rate: 0.05,
            ..Default::default()
        },
        &mut r,
    );
    let s = spectrum_of(N, 3, |x| policy.prob_action1(&out.params, &[x]));
    for (freq, m) in s.leaked_bars(1e-6).into_iter().take(24) {
        let marker = if freq % 3 == 0 {
            "  <- multiple of the in-band peak (3)"
        } else {
            ""
        };
        println!("   omega = {freq:>3}   |c| = {m:.6}{marker}");
    }
    println!(
        "\n   highest occupied above 1e-6: {}",
        s.highest_occupied(1e-6)
    );
    println!("   ceiling = 3, so 3*ceiling = 9");
}
