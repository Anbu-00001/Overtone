//! M19: fill the Menagerie archive, and test the claim that motivates it.
//!
//! `cargo run --release -p overtone-qd --example menagerie`

use overtone_qd::behaviour::dequantize_elite;
use overtone_qd::search::hill_climb;
use overtone_qd::{map_elites, Config};

fn main() {
    let evaluations: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(3000);
    let cfg = Config {
        evaluations,
        ..Config::default()
    };

    let archive = map_elites(&cfg);
    println!("MAP-Elites over SpectralControl-{} policies", cfg.k);
    println!("axes: dim(g) x reach x bond dimension   fitness: exact return by quadrature\n");
    println!("evaluations   {}", archive.evaluations());
    println!(
        "cells filled  {} of {}  ({:.0}% of the grid)",
        archive.len(),
        archive.bins.capacity(),
        100.0 * archive.coverage()
    );

    if let Some(best) = archive.best() {
        println!(
            "\nbest elite    J = {:+.6}   dim(g) = {}   reach = {:.2}   chi = {}   S = {:.3}",
            best.fitness,
            best.behaviour.dim_g,
            best.behaviour.reach,
            best.behaviour.bond_dimension,
            best.behaviour.entropy
        );
        println!(
            "              {} qubits, {} layers, {}, lambda {}",
            best.genome.qubits,
            best.genome.layers,
            if best.genome.softmax {
                "softmax"
            } else {
                "raw"
            },
            if best.genome.trainable_lambda {
                "free"
            } else {
                "pinned"
            }
        );
    }

    // The barren plateau, rendered as a map rather than as a curve. Part III: expressiveness
    // costs trainability, so the best elite in each dim(g) band should fall as the band
    // rises -- and this is a *measurement* of that, not an assertion of it.
    println!("\nbest fitness by dim(g) band");
    println!("{:>5}  {:>12}  {:>10}", "band", "best J", "cells");
    for band in 0..archive.bins.dim_g {
        let cells = archive.cells().filter(|(k, _)| k.0 == band).count();
        match archive.best_in_dim_band(band) {
            Some(f) => println!("{band:>5}  {f:>12.6}  {cells:>10}"),
            None => println!("{band:>5}  {:>12}  {cells:>10}", "-"),
        }
    }

    // Part IV 4 claims quality-diversity keeps working where gradient methods stop.
    // Arrasmith et al. (Quantum 5, 558) prove no optimiser that decides on cost differences
    // does. Matched budget, same seed, same mutation operator: the only difference is
    // whether an archive is kept.
    let (_, hill) = hill_climb(&cfg);
    let peak = archive.best().map(|c| c.fitness).unwrap_or(0.0);
    println!("\nsame budget, no archive");
    println!("  MAP-Elites peak   {peak:+.6}");
    println!("  hill climb peak   {hill:+.6}");
    println!(
        "  difference        {:+.6}   ({})",
        peak - hill,
        if (peak - hill).abs() < 0.02 {
            "no material difference"
        } else if peak > hill {
            "the archive helped"
        } else {
            "the archive cost peak fitness, which is the expected trade"
        }
    );

    // The archive is the artifact Part VIII 4 needs, so it has to leave the process.
    if let Some(path) = std::env::args().nth(2) {
        std::fs::write(&path, archive.to_json()).expect("could not write the archive");
        println!("\narchive written to {path}");
    }

    println!("\ndequantization verdict for the three best elites");
    for cell in archive.ranked().into_iter().take(3) {
        let report = dequantize_elite(&cell.genome, cfg.k, 8, 1e-6);
        println!(
            "  J = {:+.6}  n = {}  {}",
            cell.fitness,
            cell.genome.qubits,
            report.verdict()
        );
    }
}
