//! M39a: the coldness sweep. Decisions-06 Q17 -- diagnose the width before measuring depth at it.
//!
//! `cargo run --release -p overtone-orbit --example coldness`

use overtone_orbit::coldness::{angles, sample_positions, standing_value, Sweep, HOT_FLOOR};
use overtone_orbit::ladder::{opening, Strategy};
use overtone_orbit::thermal::{region_options, regions};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn main() {
    let a = angles();

    println!("M39a -- is there a game at this width?\n");
    println!(
        "{:>3} {:>7} {:>9} {:>9} {:>11} {:>11} {:>11}",
        "n", "regions", "readings", "hot", "mean gap", "best gap", "left spread"
    );
    for n in [4usize, 5, 6, 7, 8] {
        let positions = sample_positions(n, 24, 2, 6, 3, 17);
        for count in 2..=n.min(4) {
            let s = Sweep::measure(n, count, &positions, &a);
            println!(
                "{n:>3} {count:>7} {:>9} {:>8.1}% {:>11.3e} {:>11.3e} {:>11.3e}",
                s.readings.len(),
                100.0 * s.hot_fraction(),
                s.mean_gap(),
                s.best_gap(),
                s.left_spread()
            );
        }
    }

    println!("\n\nHot by ply -- where in a game the contest is\n");
    println!("{:>3}  fraction of regions hot, ply 0 onward", "n");
    for n in [4usize, 5, 6] {
        let mut row = Vec::new();
        for ply in 0..12 {
            let mut hot = 0;
            let mut total = 0;
            for g in 0..8u64 {
                let mut rng = ChaCha8Rng::seed_from_u64(100 + g);
                let mut game = opening(n, 24, 2, &mut rng);
                let strategy = Strategy { budget: 6 };
                let mut ch = ChaCha8Rng::seed_from_u64(900 + g);
                for _ in 0..ply {
                    let (mv, ang) = strategy.choose(&game, &mut ch);
                    game.apply(&mv, ang);
                }
                for r in &regions(n, 3.min(n)) {
                    if let Some((l, rr)) = region_options(&game, r, &a) {
                        total += 1;
                        if l - rr > HOT_FLOOR {
                            hot += 1;
                        }
                    }
                }
            }
            row.push(format!("{:>5.2}", hot as f64 / total.max(1) as f64));
        }
        println!("{n:>3}  {}", row.join(" "));
    }

    println!("\n\nIs the deep hotness real, or floating-point dust?\n");
    println!(
        "{:>3} {:>5} {:>7} {:>12} {:>12} {:>12}",
        "n", "ply", "hot", "median gap", "max gap", "min hot gap"
    );
    for n in [4usize, 5, 6] {
        for ply in [4usize, 8, 11] {
            let mut gaps: Vec<f64> = Vec::new();
            let mut total = 0;
            for g in 0..8u64 {
                let mut rng = ChaCha8Rng::seed_from_u64(100 + g);
                let mut game = opening(n, 24, 2, &mut rng);
                let strategy = Strategy { budget: 6 };
                let mut ch = ChaCha8Rng::seed_from_u64(900 + g);
                for _ in 0..ply {
                    let (mv, ang) = strategy.choose(&game, &mut ch);
                    game.apply(&mv, ang);
                }
                for r in &regions(n, 3.min(n)) {
                    if let Some((l, rr)) = region_options(&game, r, &a) {
                        total += 1;
                        if l - rr > HOT_FLOOR {
                            gaps.push(l - rr);
                        }
                    }
                }
            }
            gaps.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let med = if gaps.is_empty() {
                0.0
            } else {
                gaps[gaps.len() / 2]
            };
            let max = gaps.last().copied().unwrap_or(0.0);
            let min = gaps.first().copied().unwrap_or(0.0);
            println!(
                "{n:>3} {ply:>5} {:>6.0}% {med:>12.3e} {max:>12.3e} {min:>12.3e}",
                100.0 * gaps.len() as f64 / total.max(1) as f64
            );
        }
    }

    println!("\n\nThe other finding: the standing score has no range at all\n");
    println!(
        "{:>3} {:>16} {:>16} {:>12}",
        "n", "mean absorbed(0)", "mean absorbed(1)", "1/dim"
    );
    for n in [4usize, 5, 6, 7, 8] {
        let positions = sample_positions(n, 24, 2, 8, 4, 17);
        let (a0, a1) = standing_value(&positions);
        println!(
            "{n:>3} {a0:>16.3e} {a1:>16.3e} {:>12.3e}",
            1.0 / (1usize << n) as f64
        );
    }
    println!(
        "\nThe score has real range at n = 4 and 5 -- the mean sits at about 1/dim, which is the\n\
         threshold that decides whether a cell is absorbing -- and it **collapses at n = 6** to\n\
         1e-34, then to exactly zero. That is a hard width ceiling on the current evaluation,\n\
         and it is not a game-theoretic fact: the players start at opposite corners of a 2^n\n\
         window and a coherent walk needs O(n) plies to reach the other, so at eight plies the\n\
         overlap that absorbed weight measures has not happened yet.\n\
         \n\
         So Decisions-06 Q17's ruling resolves to **n = 5**: hot from ply 4, standing score\n\
         intact, and one width wider than the retired measurement. n = 6 and above are not\n\
         cold, they are numerically dead at this ply depth, which is a different problem with a\n\
         different fix -- more plies, or a score that does not wait for overlap.\n\
         \n\
         And Decisions-06 Q16's hypothesis does not hold. It proposed that `d` saturating at\n\
         n = 4 and the field being cold at n = 4 might be one fact; the ply table says n = 4 is\n\
         46% hot by ply 6. Whatever shortens that ladder, it is not an absence of decisions."
    );
}
