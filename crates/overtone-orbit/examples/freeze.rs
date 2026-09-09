//! The measurements Decisions 03 §12 says it cannot make, and Q1 says to make before the
//! v1 feature vocabulary is frozen.
//!
//! `cargo run --release -p overtone-orbit --example freeze`
//!
//! Q1: "A feature is dead weight if it does not vary across the moves available at a node. A
//! constant contributes equally to every sibling and cannot influence selection. Before
//! freezing, measure the sibling variance of each feature across a sample of positions. Cut
//! anything near zero."
//!
//! Three things get measured here:
//!
//! 1. **Sibling variance** — the spread of each candidate feature over the positions reachable
//!    by one move from a node. This is Q1's test.
//! 2. **Cost per call** — because Q1 also asks whether `temperature` is affordable inside an
//!    MCTS expansion at budget 4096, and whether `half_chain_entropy` is cost-disqualified.
//! 3. **Single-game wall clock** — which Q6 says determines the whole `d` re-measurement
//!    budget.

use overtone_orbit::checkmate::Position;
use overtone_orbit::game::{legal_moves, Game};
use overtone_orbit::invariant::{orbit_dimension, separation_deficit};
use overtone_orbit::ladder::{opening, play, Strategy, ANGLES};
use overtone_orbit::thermal::{ambient_temperature, regions, Region};
use overtone_spec::entropy::half_chain_entropy;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

const N: usize = 6;

/// Every candidate feature, evaluated on a game state for the side to move.
fn features(game: &Game, rs: &[Region], angles: &[f64]) -> Vec<f64> {
    let me = game.to_move;
    let p = &game.players[me];
    let algebra = p.algebra();
    let position = Position::new(p.state.clone(), algebra.clone(), game.safe_for(me));
    vec![
        p.coherence as f64,
        algebra.dim() as f64,
        orbit_dimension(&algebra, &p.state, 1e-9) as f64,
        game.safe_for(me).len() as f64,
        ambient_temperature(game, rs, angles),
        game.average_branching(),
        separation_deficit(&algebra, position.certificate(), &p.state) as f64,
        half_chain_entropy(&p.state),
    ]
}

const NAMES: [&str; 8] = [
    "coherence",
    "dim_g",
    "orbit_size",
    "safe_set_size",
    "temperature",
    "average_branching",
    "separation_deficit",
    "half_chain_entropy",
];

fn main() {
    let rs = regions(N, 3);
    let angles = [ANGLES[1], ANGLES[3], ANGLES[5]];

    // ---- 1. sibling variance -------------------------------------------------------------
    let mut spread = vec![Vec::new(); NAMES.len()];
    for seed in 0..6u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut game = opening(N, 24, 2, &mut rng);
        // Walk a few plies in so the position is not the pristine opening.
        let strategy = Strategy { budget: 6 };
        let mut choose = ChaCha8Rng::seed_from_u64(seed ^ 9);
        for _ in 0..4 {
            let (mv, a) = strategy.choose(&game, &mut choose);
            game.apply(&mv, a);
        }
        // Now evaluate every sibling.
        let moves = legal_moves(N);
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for mv in moves.iter().take(24) {
            let mut trial = game.clone();
            trial.apply(mv, ANGLES[3]);
            // The feature is read for the player who just moved, not the next mover.
            trial.to_move ^= 1;
            rows.push(features(&trial, &rs, &angles));
        }
        for (f, s) in spread.iter_mut().enumerate() {
            let vals: Vec<f64> = rows.iter().map(|r| r[f]).collect();
            let lo = vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let rel = if mean.abs() > 1e-12 {
                (hi - lo) / mean.abs()
            } else {
                hi - lo
            };
            s.push(rel);
        }
    }

    println!("Q1 sibling variance -- relative spread (max-min)/|mean| over 24 sibling moves\n");
    println!(
        "{:>20}  {:>12}  {:>10}",
        "feature", "mean spread", "verdict"
    );
    let mut dead = Vec::new();
    for (i, name) in NAMES.iter().enumerate() {
        let m = spread[i].iter().sum::<f64>() / spread[i].len() as f64;
        let verdict = if m < 1e-9 {
            "DEAD"
        } else if m < 0.01 {
            "weak"
        } else {
            "lives"
        };
        if verdict == "DEAD" {
            dead.push(*name);
        }
        println!("{name:>20}  {m:>12.6}  {verdict:>10}");
    }
    println!();
    if dead.is_empty() {
        println!("no feature is constant across siblings");
    } else {
        println!("CONSTANT ACROSS SIBLINGS, cut from v1: {}", dead.join(", "));
    }

    // ---- 2. cost per call ----------------------------------------------------------------
    println!();
    println!("Q1 cost per call -- can it live inside an MCTS expansion?\n");
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let game = opening(N, 24, 2, &mut rng);
    println!(
        "{:>20}  {:>14}  {:>26}",
        "feature", "us per call", "calls in a 4096 budget (1s)"
    );

    // Time each feature separately rather than the whole vector.
    let time_one = |f: &dyn Fn() -> f64| -> f64 {
        let reps = 50;
        let t0 = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(f());
        }
        t0.elapsed().as_secs_f64() * 1e6 / reps as f64
    };
    let me = game.to_move;
    let p = &game.players[me];
    let algebra = p.algebra();
    let position = Position::new(p.state.clone(), algebra.clone(), game.safe_for(me));
    let costs = [
        ("coherence", time_one(&|| p.coherence as f64)),
        ("dim_g", time_one(&|| p.algebra().dim() as f64)),
        (
            "orbit_size",
            time_one(&|| orbit_dimension(&algebra, &p.state, 1e-9) as f64),
        ),
        (
            "safe_set_size",
            time_one(&|| game.safe_for(me).len() as f64),
        ),
        (
            "temperature",
            time_one(&|| ambient_temperature(&game, &rs, &angles)),
        ),
        ("average_branching", time_one(&|| game.average_branching())),
        (
            "separation_deficit",
            time_one(&|| separation_deficit(&algebra, position.certificate(), &p.state) as f64),
        ),
        (
            "half_chain_entropy",
            time_one(&|| half_chain_entropy(&p.state)),
        ),
    ];
    for (name, us) in costs {
        let per_second = 1e6 / us;
        let fits = if per_second > 4096.0 {
            "fits 4096/s"
        } else {
            "TOO SLOW"
        };
        println!("{name:>20}  {us:>14.2}  {per_second:>14.0}/s  {fits}");
    }

    // ---- 3. single-game wall clock -------------------------------------------------------
    println!();
    println!("Q6 single-game wall clock\n");
    for budget in [8usize, 32, 128] {
        let games = 6;
        let t0 = Instant::now();
        for g in 0..games {
            play(Strategy { budget }, Strategy { budget }, N, 24, 2, g as u64);
        }
        let per = t0.elapsed().as_secs_f64() / games as f64;
        let in_6h = (6.0 * 3600.0 / per) as u64;
        println!(
            "budget {budget:>4}: {per:>7.3} s/game   {in_6h:>9} games in a 6 h job   \
             1600 games in {:.2} h",
            1600.0 * per / 3600.0
        );
    }
}
