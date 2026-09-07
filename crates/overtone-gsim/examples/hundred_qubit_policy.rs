//! M13: train a hundred-qubit quantum RL policy, exactly, on one CPU core.
//!
//! The claim Part III 4 wants in the README's first line. It is true because the
//! transverse-field Ising algebra is `so(2n)`, of dimension `n(2n - 1)` -- 19900 numbers at
//! `n = 100`, against `2^100` amplitudes for a state vector. Nothing here is approximate:
//! g-sim is exact whenever the circuit stays inside `exp(g)`, and this one does by
//! construction.
//!
//! **The caveat belongs on the same screen as the claim.** g-sim is efficient exactly when
//! `dim(g)` is polynomial, and a polynomial `dim(g)` is exactly the condition for having no
//! barren plateau. So this policy is trainable *because* it is classically simulable. That
//! is the live tension in the field and Part III 4 says to name it rather than route round
//! it.
//!
//! `cargo run --release -p overtone-gsim --example hundred_qubit_policy`

use std::time::Instant;

use overtone_gsim::{GsimPolicy, PolicyLayout, PolicyObservable};
use overtone_rl::reinforce::Adam;
use overtone_rl::SpectralControl;

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(100);
    let k = 3;
    let nodes = 32;
    let steps = 150;

    let built = Instant::now();
    let policy = GsimPolicy::new(PolicyLayout {
        num_qubits: n,
        layers: 1,
        lambda_trainable: true,
        observable: PolicyObservable::MeanZz,
    });
    let build_secs = built.elapsed().as_secs_f64();

    let env = SpectralControl::new(k);
    println!("qubits            {n}");
    println!("dim(g)            {}  (so(2n), n(2n-1))", policy.dim_g());
    println!("state vector      2^{n} amplitudes -- not representable");
    println!("gates             {}", policy.num_gates());
    println!("parameters        {}", policy.num_params());
    println!("frequency ceiling {} * lambda", policy.frequency_ceiling());
    println!("closure + planes  {build_secs:.2} s");
    println!(
        "memory (vectors)  {:.1} kB",
        (policy.dim_g() * 8 * 3) as f64 / 1024.0
    );

    // Deterministic start: small variational angles, lambda at one.
    let mut params = vec![0.0; policy.num_params()];
    params[0] = 1.0;
    let mut seed = 20260907u64;
    for p in params.iter_mut().skip(1) {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *p = ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.2;
    }

    let quadrature: Vec<f64> = (0..nodes)
        .map(|i| {
            -std::f64::consts::PI + 2.0 * std::f64::consts::PI * (i as f64 + 0.5) / nodes as f64
        })
        .collect();

    let exact_return = |p: &[f64]| env.expected_return(256, |s| policy.prob_action1(p, s));

    let start_return = exact_return(&params);
    let mut adam = Adam::new(policy.num_params(), 0.05);
    let trained = Instant::now();
    for step in 0..steps {
        // The return is a quadrature integral of a smooth periodic integrand, so the
        // gradient is exact rather than sampled -- Part I 7.1's lesson, carried over.
        let mut grad = vec![0.0; policy.num_params()];
        for &s in &quadrature {
            let (_, dp) = policy.prob_and_grad(&params, s);
            let w = 2.0 * ((k as f64) * s).cos() / nodes as f64;
            for (g, d) in grad.iter_mut().zip(&dp) {
                *g += w * d;
            }
        }
        adam.ascend(&mut params, &grad);
        if step % 30 == 0 || step == steps - 1 {
            println!(
                "  step {step:>4}   J = {:+.6}   lambda = {:.4}",
                exact_return(&params),
                params[0]
            );
        }
    }
    let train_secs = trained.elapsed().as_secs_f64();
    let end_return = exact_return(&params);

    println!("\nreturn before     {start_return:+.6}");
    println!("return after      {end_return:+.6}");
    println!("optimal (2/pi)    {:+.6}", env.optimal_return());
    println!("training          {train_secs:.1} s for {steps} exact gradient steps");
    println!(
        "\ncaveat: dim(g) is polynomial, which is why this trains and also why it is\n\
         classically simulable. The two are the same fact."
    );
}
