// Why is the fitted global rate ~1.0 and not the ~2.0 that a full 2-design argument gives?
// Hypothesis: probing theta_1 leaves the circuit to its LEFT trivial (the state is still
// |0..0>), so only one side Haar-averages. Probing mid-circuit puts a 2-design on both
// sides. If that is the explanation, a mid-circuit probe on the GLOBAL observable should
// roughly double the rate. (Global is used because the local observable's mid-circuit
// gradients hit structural zeros - see plateau.rs.)
use overtone_spec::plateau::*;
fn main() {
    for depth in [DepthPolicy::Logarithmic(2), DepthPolicy::Linear(2)] {
        for &mid in &[false, true] {
            let pts: Vec<_> = (2..=11usize)
                .map(|n| {
                    let d = depth.depth_for(n);
                    let np = hardware_efficient(n, d).num_params();
                    let probe = if mid { np / 2 } else { 0 };
                    gradient_variance_at(n, d, CostLocality::Global, 400, 7, probe)
                })
                .collect();
            let f = fit_exponential(&pts);
            println!(
                "{:>18?}  probe={:>6}  global Var ~ 2^(-{:.3} n)   R^2 = {:.4}",
                depth,
                if mid { "middle" } else { "first" },
                f.rate,
                f.r_squared
            );
        }
    }
}
