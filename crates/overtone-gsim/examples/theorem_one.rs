//! M11: reproduce the barren-plateau theory's variance formula from our own engine.
//!
//! Ragone, Bakalov, Sauvage, Kemper, Ortiz Marrero, Larocca and Cerezo, *A Lie algebraic
//! theory of barren plateaus for deep parameterized quantum circuits*, Nat. Commun. 15,
//! 7172 (2024), Theorem 1:
//!
//! ```text
//! Var_theta[ loss ] = sum_j P_{g_j}(rho) P_{g_j}(O) / dim(g_j)
//! ```
//!
//! over the simple components of the algebra, assuming the circuit is deep enough to form
//! a 2-design over `exp(g)`. Part III 9 asks for "Figure 2 of Ragone et al." -- but Figures
//! 1 and 2 of that paper are both schematics of where barren plateaus come from, so there
//! is no curve there to reproduce. The theorem is the reproducible object, and it is the
//! better target: an exact closed form with nothing to eyeball.
//!
//! Run with `cargo run --release -p overtone-gsim --example theorem_one`.

use overtone_gsim::GsimCircuit;
use overtone_lie::{closure_unbounded, family, predict::components, PauliString};
use overtone_sim::Pauli;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn sweep(
    name: &str,
    ns: &[usize],
    build: impl Fn(usize) -> (Vec<PauliString>, Vec<PauliString>, PauliString),
) {
    println!("\n{name}");
    println!(
        "{:>2}  {:>6}  {:>7}  {:>12}  {:>6}  {:>12}  {:>9}  {:>10}",
        "n", "dim(g)", "ideals", "predicted", "depth", "measured", "ratio", "mean"
    );

    for &n in ns {
        let (family_gens, gens, obs) = build(n);
        let algebra = closure_unbounded(&family_gens, n);
        let oi = algebra.index_of(&obs).unwrap();
        let comps = components(&algebra);
        let comp = comps.iter().find(|c| c.contains(&oi)).unwrap();
        let z_type = comp.iter().filter(|&&i| algebra.basis()[i].x == 0).count();
        let predicted = z_type as f64 / comp.len() as f64;
        let dim = algebra.dim();

        let mut weights = vec![0.0; dim];
        weights[oi] = 1.0;

        for &depth in &[1usize, 8, 32, 64] {
            let mut circuit = GsimCircuit::new(algebra.clone(), gens.clone());
            for _ in 0..depth {
                for gi in 0..gens.len() {
                    circuit.push_raw(gi);
                }
            }
            let ngates = circuit.gates().len();
            let mut rng = ChaCha8Rng::seed_from_u64(20260907 + n as u64);
            let samples = 4000;
            let (mut sum, mut sum_sq) = (0.0, 0.0);
            for _ in 0..samples {
                let angles: Vec<f64> = (0..ngates)
                    .map(|_| rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI))
                    .collect();
                let e = circuit.evolve_angles(&angles);
                let v: f64 = weights.iter().zip(&e).map(|(w, x)| w * x).sum();
                sum += v;
                sum_sq += v * v;
            }
            let mean = sum / samples as f64;
            let measured = sum_sq / samples as f64 - mean * mean;
            println!(
                "{:>2}  {:>6}  {:>7}  {:>12.6}  {:>6}  {:>12.6}  {:>9.4}  {:>10.2e}",
                n,
                dim,
                comps.iter().filter(|c| c.len() > 1).count(),
                predicted,
                depth,
                measured,
                measured / predicted,
                mean
            );
        }
        println!();
    }
}

fn main() {
    println!("Theorem 1: Var[loss] = sum_j P_gj(rho) P_gj(O) / dim(gj),  rho = |0..0>");
    println!("Prediction is algebraic and exact; measurement is 4000 random circuits.");
    println!("The ratio approaches one as the circuit deepens into a 2-design over exp(g).");

    sweep(
        "transverse-field Ising, algebra so(2n), observable X_0 -- polynomial",
        &[3, 4, 5, 6, 7],
        |n| {
            let mut g: Vec<PauliString> =
                (0..n).map(|q| PauliString::single(q, Pauli::X)).collect();
            g.extend(
                (0..n - 1).map(|q| PauliString::from_factors(&[(q, Pauli::Z), (q + 1, Pauli::Z)])),
            );
            (family::tfim(n), g, PauliString::single(0, Pauli::X))
        },
    );

    sweep(
        "Heisenberg chain, algebra su(2^(n-1)) or su(2^(n-2))^4, observable X_0X_1 -- exponential",
        &[3, 4, 5, 6],
        |n| {
            let g = family::heisenberg(n);
            (
                g.clone(),
                g,
                PauliString::from_factors(&[(0, Pauli::X), (1, Pauli::X)]),
            )
        },
    );
}
