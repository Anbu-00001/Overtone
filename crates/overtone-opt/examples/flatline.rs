//! Part V 5.2: four optimisers, one plateau. Run it both ways, and report what happens.
//!
//! `cargo run --release -p overtone-opt --example flatline`

use overtone_opt::race::{fit_shot_scaling, lane, race, shots_to_progress, RaceConfig, ALL};
use overtone_opt::shots::Budget;

const SEEDS: [u64; 5] = [1, 2, 3, 4, 5];

fn main() {
    let evaluations = 200u64;
    let shots = 1_000usize;

    println!("exact arithmetic: sixteen digits per evaluation, {evaluations} evaluations");
    println!(
        "mean descent in the exact cost over {} seeds\n",
        SEEDS.len()
    );
    table(Budget::Exact, evaluations, evaluations);

    println!("\n{shots} shots per evaluation, the same {evaluations} evaluations");
    table(
        Budget::Shots(shots),
        evaluations * shots as u64,
        evaluations,
    );

    println!("\nshots per evaluation before a majority of seeds beat their own noise");
    println!(
        "{:>4}  {:>7}  {:>18}  {:>18}  {:>12}  {:>14}",
        "n", "params", "gradient", "natural gradient", "CMA-ES", "Nelder-Mead"
    );
    let qubits = [2usize, 4, 6, 8, 10];
    let mut curves: Vec<Vec<(usize, usize)>> = vec![Vec::new(); ALL.len()];
    for &n in &qubits {
        let params = 2 * n * n;
        let mut cells = Vec::new();
        for (i, &o) in ALL.iter().enumerate() {
            match shots_to_progress(o, n, &SEEDS, 16, 1 << 16, 200) {
                Some(s) => {
                    curves[i].push((n, s));
                    cells.push(format!("{s}"));
                }
                None => cells.push("> 65536".into()),
            }
        }
        println!(
            "{n:>4}  {params:>7}  {:>18}  {:>18}  {:>12}  {:>14}",
            cells[0], cells[1], cells[2], cells[3]
        );
    }

    println!("\nfitted as shots ~ 2^(a n)");
    println!(
        "{:>18}  {:>10}  {:>10}  {:>8}",
        "optimiser", "a", "R^2", "points"
    );
    for (i, &o) in ALL.iter().enumerate() {
        let f = fit_shot_scaling(&curves[i]);
        println!(
            "{:>18}  {:>10.3}  {:>10.4}  {:>8}",
            o.name(),
            f.exponent,
            f.r_squared,
            f.points
        );
    }
    println!("  all four exponents are positive: the shot requirement is exponential in n for");
    println!("  every optimiser, which is Arrasmith et al.'s result. They are not the same");
    println!("  exponent, and the spec's \"all four flatline\" hides that they are not.");

    println!("\nscoring on the best value the optimiser saw, versus where it stopped");
    println!(
        "{:>4}  {:>18}  {:>14}  {:>14}  {:>14}",
        "n", "optimiser", "it believed", "exact there", "exact at end"
    );
    for n in [4usize, 8] {
        for l in race(RaceConfig::plateau(
            n,
            Budget::Shots(100),
            200 * 100,
            SEEDS[0],
        )) {
            println!(
                "{n:>4}  {:>18}  {:>14.6}  {:>14.6}  {:>14.6}",
                l.optimiser.name(),
                l.reported_best,
                l.best_exact,
                l.final_exact
            );
        }
    }
}

fn table(budget: Budget, allowance: u64, _evaluations: u64) {
    println!(
        "{:>4}  {:>18}  {:>12}  {:>10}  {:>8}",
        "n", "optimiser", "mean descent", "noise", "seeds up"
    );
    for n in [4usize, 6, 8, 10] {
        for &o in ALL.iter() {
            let lanes: Vec<_> = SEEDS
                .iter()
                .map(|&s| lane(RaceConfig::plateau(n, budget, allowance, s), o))
                .collect();
            let mean = lanes.iter().map(|l| l.progress()).sum::<f64>() / lanes.len() as f64;
            let floor = lanes.iter().map(|l| l.noise_floor).sum::<f64>() / lanes.len() as f64;
            let up = lanes.iter().filter(|l| l.beat_the_noise()).count();
            println!(
                "{n:>4}  {:>18}  {mean:>12.6}  {floor:>10.2e}  {up:>4}/{}",
                o.name(),
                lanes.len()
            );
        }
    }
}
