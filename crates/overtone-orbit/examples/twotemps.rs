//! M51: do the two temperatures track each other?
//!
//! `cargo run --release -p overtone-orbit --example twotemps`
//!
//! Part IX 5.4 pairs CGT temperature (urgency of the next move) with the physical entropy of
//! the quantum state and asks whether they correlate. Nobody has, because no game has ever
//! had both. This prints the answer for as many seeds as it is given, and prints the spread
//! of each series alongside it so a correlation over a flat line cannot pass as a result.

use overtone_orbit::ladder::{opening, Strategy, ANGLES};
use overtone_orbit::thermal::{interaction_leak, regions, trace, Trace};
use overtone_spec::spearman;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn main() {
    let n = 6;
    let regions = regions(n, 3);
    let angles = [ANGLES[1], ANGLES[3], ANGLES[5]];
    let strategy = Strategy { budget: 12 };

    println!(
        "n = {n} qubits, {} regions, {} angles sampled",
        regions.len(),
        angles.len()
    );
    println!(
        "{:>5}  {:>6}  {:>9}  {:>9}  {:>9}  {:>8}  {:>8}  {:>8}",
        "seed", "plies", "rho", "cgt range", "S range", "T~ply", "S~ply", "partial"
    );

    let mut rhos = Vec::new();
    let mut leaks = Vec::new();
    let mut cgt_vs_ply = Vec::new();
    let mut ent_vs_ply = Vec::new();
    let mut partials = Vec::new();
    for seed in 0..16u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut game = opening(n, 24, 2, &mut rng);
        leaks.push(interaction_leak(&game, &regions, &angles));
        let mut choose_rng = ChaCha8Rng::seed_from_u64(seed ^ 0x5eed);
        let t: Trace = trace(&mut game, &regions, &angles, 14, |g| {
            strategy.choose(g, &mut choose_rng)
        });
        let (cs, es) = t.spread();
        let rho = t.correlation();
        // The confound: if both series simply drift with the ply index, a correlation
        // between them is a correlation with time and nothing more. So measure each against
        // the ply, and partial the ply out of the pair.
        let ply: Vec<f64> = t.readings.iter().map(|r| r.ply as f64).collect();
        let cgt: Vec<f64> = t.readings.iter().map(|r| r.cgt).collect();
        let ent: Vec<f64> = t.readings.iter().map(|r| r.entropy).collect();
        let rcp = spearman(&cgt, &ply);
        let rep = spearman(&ent, &ply);
        let partial = (rho - rcp * rep) / ((1.0 - rcp * rcp).sqrt() * (1.0 - rep * rep).sqrt());
        println!(
            "{seed:>5}  {:>6}  {rho:>9.3}  {cs:>9.4}  {es:>9.4}  {rcp:>8.3}  {rep:>8.3}  {partial:>8.3}",
            t.readings.len()
        );
        if rho.is_finite() {
            rhos.push(rho);
            cgt_vs_ply.push(rcp);
            ent_vs_ply.push(rep);
            if partial.is_finite() {
                partials.push(partial);
            }
        }
    }

    println!();
    if rhos.is_empty() {
        println!("no finite correlation on any seed: one of the two series never varied");
    } else {
        let mean = rhos.iter().sum::<f64>() / rhos.len() as f64;
        let lo = rhos.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = rhos.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let same_sign = rhos.iter().all(|r| *r > 0.0) || rhos.iter().all(|r| *r < 0.0);
        println!(
            "rho over {} seeds: mean {mean:.3}, range [{lo:.3}, {hi:.3}]",
            rhos.len()
        );
        println!("consistent sign across seeds: {same_sign}");
    }
    let avg = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len().max(1) as f64;
    println!(
        "drift with ply: temperature {:.3}, entropy {:.3}",
        avg(&cgt_vs_ply),
        avg(&ent_vs_ply)
    );
    if partials.is_empty() {
        println!("partial correlation undefined on every seed");
    } else {
        let lo = partials.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = partials.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "partial rho with ply removed: mean {:.3}, range [{lo:.3}, {hi:.3}] over {} seeds",
            avg(&partials),
            partials.len()
        );
    }
    let leak = leaks.iter().sum::<f64>() / leaks.len() as f64;
    println!("mean interaction leak (whole-position best minus best-by-region): {leak:.4}");

    // The verdict, in a form the gate can check. Part IX 8: measure it, report what you find,
    // do not assume. What we find is that the raw correlation is an artefact of both series
    // climbing over the course of a game, and that nothing survives removing the ply.
    let raw_mean = avg(&rhos);
    let par_mean = avg(&partials);
    let positive = partials.iter().filter(|x| **x > 0.0).count();
    let mixed = positive > 2 && positive < partials.len() - 2;
    if par_mean.abs() < 0.25 && mixed {
        println!(
            "VERDICT no-correlation: raw rho {raw_mean:.3} is drift; partial rho {par_mean:.3}, \
             sign mixed {positive}/{}",
            partials.len()
        );
    } else {
        println!(
            "VERDICT correlated: partial rho {par_mean:.3} over {} seeds",
            partials.len()
        );
    }
}
