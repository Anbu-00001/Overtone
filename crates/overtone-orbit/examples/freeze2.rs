//! Follow-up to `freeze`: Q1's sibling test is unfair to features that vary with *depth*
//! rather than across siblings at one node.
//!
//! `cargo run --release -p overtone-orbit --example freeze2`
//!
//! Three of the four features the sibling test killed have identifiable causes:
//!
//! * `average_branching` is `legal_moves(n).len()`, a function of `n` alone — constant
//!   everywhere, for every agent, forever. Genuinely dead.
//! * `coherence` falls by exactly `k` every ply in both arms of `Game::apply`, so it is a
//!   pure function of depth. Constant among siblings *and* redundant with depth, which the
//!   search already knows. Dead as an independent feature.
//! * `safe_set_size` reads the **opponent's** field (`Game::safe_for`), so a player's own
//!   move cannot change it by construction. Zero among siblings, but it should move when the
//!   opponent moves.
//! * `half_chain_entropy` was zero on the sampled positions, but those sit four plies from a
//!   basis-state opening where single-qubit generators keep the state near-product. That
//!   smells like a sampling artefact rather than a property of the feature.
//!
//! The last two need measuring across mixed-depth leaves before they are cut.

use overtone_orbit::game::Game;
use overtone_orbit::ladder::{opening, Strategy, ANGLES};
use overtone_spec::entropy::half_chain_entropy;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const N: usize = 6;

fn read(game: &Game) -> [f64; 4] {
    let me = game.to_move;
    let p = &game.players[me];
    [
        p.coherence as f64,
        game.safe_for(me).len() as f64,
        half_chain_entropy(&p.state),
        p.algebra().dim() as f64,
    ]
}

const NAMES: [&str; 4] = ["coherence", "safe_set_size", "half_chain_entropy", "dim_g"];

fn main() {
    // Collect leaves at mixed depths along many different lines -- the population an MCTS
    // evaluation actually sees.
    let mut leaves: Vec<[f64; 4]> = Vec::new();
    let mut by_depth: Vec<Vec<[f64; 4]>> = vec![Vec::new(); 5];
    for seed in 0..24u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut game = opening(N, 40, 2, &mut rng);
        let strategy = Strategy { budget: 4 };
        let mut choose = ChaCha8Rng::seed_from_u64(seed ^ 0x1234);
        for ply in 0..16 {
            let (mv, a) = strategy.choose(&game, &mut choose);
            game.apply(&mv, a);
            let _ = ANGLES;
            if ply >= 4 {
                leaves.push(read(&game));
                if ply % 3 == 0 {
                    by_depth[(ply / 3).min(4)].push(read(&game));
                }
            }
        }
    }

    println!(
        "Q1 follow-up: spread across {} mixed-depth leaves\n",
        leaves.len()
    );
    println!(
        "{:>20}  {:>10}  {:>10}  {:>12}  {:>10}",
        "feature", "min", "max", "rel spread", "verdict"
    );
    for (i, name) in NAMES.iter().enumerate() {
        let v: Vec<f64> = leaves.iter().map(|l| l[i]).collect();
        let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let rel = if mean.abs() > 1e-12 {
            (hi - lo) / mean.abs()
        } else {
            hi - lo
        };
        let verdict = if rel < 1e-9 {
            "DEAD"
        } else if rel < 0.05 {
            "weak"
        } else {
            "lives"
        };
        println!("{name:>20}  {lo:>10.4}  {hi:>10.4}  {rel:>12.6}  {verdict:>10}");
    }

    // The distinguishing question for `coherence`: does it carry anything beyond depth?
    println!();
    println!("Is coherence anything but a restatement of depth?\n");
    println!(
        "{:>8}  {:>12}  {:>12}",
        "depth bin", "coherence min", "coherence max"
    );
    for (d, rows) in by_depth.iter().enumerate() {
        if rows.is_empty() {
            continue;
        }
        let v: Vec<f64> = rows.iter().map(|r| r[0]).collect();
        let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("{d:>8}  {lo:>12.1}  {hi:>12.1}");
    }
    println!();
    println!("If min == max in every bin, coherence is a function of depth alone and adds");
    println!("nothing a search does not already know.");
}
