//! M39b: the Skill Trace, on Browne's protocol over Goodman's grid.
//!
//! `cargo run --release -p overtone-orbit --example trace`

use std::time::Instant;

use overtone_orbit::game::legal_moves;
use overtone_orbit::language::Agent;
use overtone_orbit::trace::{budget_ladder_from, saturation, Grid, Setting, Trace, PUBLISHED};

const PIKE: &str = include_str!("../../../agents/pike.toml");
const STOAT: &str = include_str!("../../../agents/stoat.toml");

/// Decisions-06 Q17's ruling: measured by M39a as the width with hot structure and an intact
/// standing score.
const N: usize = 5;
const COHERENCE: usize = 24;
const K: usize = 2;

fn main() {
    let bf = legal_moves(N).len() * 8;
    println!("M39b -- the Skill Trace at n = {N}\n");
    println!(
        "Browne bases the ladder on the branching factor. Overtone's action is a (move, angle)\n\
         pair, so BF = {} legal moves x 8 angles = {bf}, not {}.\n",
        legal_moves(N).len(),
        legal_moves(N).len()
    );

    println!("What the protocol costs, measured, and why it decides the shape of this run\n");
    println!(
        "{:>9} {:>8} {:>14}  1600 games",
        "kind", "budget", "s per game"
    );
    for (name, s) in [("mcts", 271.812f64), ("negamax", 42.734), ("greedy", 0.032)] {
        println!(
            "{name:>9} {:>8} {s:>14.3}  {:>10.1} h",
            320,
            s * 1600.0 / 3600.0
        );
    }
    println!(
        "\nBrowne's ladder starts at BF and doubles, so the *first* match for mcts is 360 vs 720\n\
         -- above the 320 measured here. At 272 s per game a single 1600-game pairing is 120\n\
         hours, and the grid has fifteen of them. **The published protocol run with mcts at\n\
         BF-scaled budgets is not affordable**, and that is a measurement rather than a\n\
         complaint: it is the number that decides what this run can be.\n\
         \n\
         So the trace is reported per search kind, over the range in which each kind has a\n\
         compute axis. That is not a retreat from the protocol -- it is Goodman's own\n\
         contribution made concrete. Their finding is that expanding the algorithmic space\n\
         changes the estimated depth of some games, and that best achievable performance at a\n\
         budget must therefore be a maximum over an algorithm space rather than one algorithm's\n\
         curve. A frozen language with three declarable kinds is that space."
    );

    println!("\n\nWhere each kind's compute axis actually lives\n");
    println!(
        "  greedy   saturates AT {bf}: at or above the candidate count it searches everything,\n\
             \x20          so every budget from {bf} up is the same agent and a ladder based\n\
             \x20          there would be flat by construction. Its range is below BF.\n\
      negamax   the node allowance; depth 2 becomes reachable near BF, which is where the\n\
             \x20          cost jumps from 0.016 to 42.7 s per game.\n\
         mcts   playouts; no saturation, and no affordable ladder."
    );

    // The grids that fit. greedy is traced at the ruled width; negamax is traced at n = 4
    // because at n = 5 it is not affordable; mcts is not traced at all. Those are measured
    // exclusions, and the fact that only one of three declared kinds can be traced at the ruled
    // width is a result rather than a limitation of this run.
    let plans: [(&str, &str, usize, usize, usize, usize); 2] = [
        ("greedy, n=5", STOAT, N, 6, 6, 240),
        ("negamax, n=4", PIKE, 4, 6, 4, 60),
    ];
    let grids: Vec<(&str, Grid, f64)> = plans
        .iter()
        .map(|&(name, src, width, base, matches, games)| {
            let agent = Agent::parse(src).unwrap();
            let mut budgets = budget_ladder_from(base, matches);
            // Cut the ladder where the kind stops changing: a saturated pair is 100% draws and
            // drags the regression rather than measuring anything.
            if let Some(cap) = saturation(&agent, width) {
                budgets.retain(|b| *b <= cap);
            }
            let t = Instant::now();
            let setting = Setting {
                n: width,
                coherence: COHERENCE,
                k: K,
            };
            let grid = Grid::measure(&agent, setting, &budgets, games, 4242, 8);
            (name, grid, t.elapsed().as_secs_f64())
        })
        .collect();

    for (name, grid, secs) in &grids {
        println!("\n\nThe grid, {name}: every pairwise budget\n");
        println!(
            "{:>7} {:>7} {:>6} {:>5} {:>5} {:>5} {:>9} {:>9}  resolves",
            "weak", "strong", "games", "W", "D", "L", "score", "+/- 95%"
        );
        for p in &grid.pairings {
            println!(
                "{:>7} {:>7} {:>6} {:>5} {:>5} {:>5} {:>9.3} {:>9.3}  {}",
                p.weak,
                p.strong,
                p.games,
                p.wins,
                p.draws,
                p.losses,
                p.score(),
                p.half_width(),
                if p.resolves() { "yes" } else { "" }
            );
        }
        let t = grid.skill_trace();
        println!(
            "\n  {} of {} pairings resolve at 95%. Measured in {:.1} s.",
            grid.resolved(),
            grid.pairings.len(),
            secs
        );
        println!(
            "  Browne's ST from the adjacent diagonal: slope {:+.4}, y = {:.4}, A = {:.4}\n\
             \x20 ST = y + (1 - y) A = {:.4}   (A excluding the discarded first match: {:.4})",
            t.slope, t.y, t.area, t.value, t.area_excluding_first
        );
    }

    println!("\n\nAgainst Goodman's published two-player values\n");
    let mut rows: Vec<(String, f64)> = PUBLISHED.iter().map(|(g, v)| (g.to_string(), *v)).collect();
    for (name, grid, _) in &grids {
        rows.push((format!("Orbit {name}"), grid.skill_trace().value));
    }
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (g, v) in &rows {
        let bar = "#".repeat((v * 60.0).round() as usize);
        println!("{:>22}  {v:>6.3}  {bar}", g);
    }
    println!(
        "\nThe comparison is worth making only because it is on a compatible protocol -- which\n\
         is the whole reason Part VII 8's original chess-and-Go figure was replaced. It is still\n\
         not a like-for-like number, in three ways that all point the same direction.\n\
         \n\
         Goodman's values are measured with MCTS agents; the two rows above are greedy and\n\
         negamax, because mcts is not affordable here. They are not the same width as each\n\
         other either. And the games per pairing are 240 and 60 against the 1600 that buys\n\
         +/- 10 Elo.\n\
         \n\
         **ST is also not scale-free in ladder length**, which is a property of the formula\n\
         rather than of this run. A = sum over matches of max(0, score)^2 is a *sum*, so a\n\
         longer ladder has more terms and a larger A, and ST = y + (1 - y) A is increasing in A.\n\
         Browne runs until a time limit, so ladder length varies between games -- and two ST\n\
         values measured over ladders of different length are not measuring the same quantity.\n\
         These ladders have three and four adjacent matches, which is short. Nothing about that\n\
         is hidden by the number, and it is why the rows above are a position on a chart rather\n\
         than a claim about Orbit being deeper than Dots + Boxes.\n\
         \n\
         (A further consequence: ST is not bounded above by 1. With y clamped low and A above\n\
         one, y + (1 - y) A exceeds it. An earlier run of this grid produced exactly that.)"
    );

    println!("\n\nWhat it would take to say this properly\n");
    println!(
        "  {:>6} games for +/- 10 Elo, {:>6} for +/- 5, on the draw-heavy 400/sqrt(N)\n\
         \x20 approximation. This run used {} and {} games per pairing.",
        Grid::games_for(10.0),
        Grid::games_for(5.0),
        240,
        60
    );
    println!(
        "\n  And only one of the three declared kinds could be traced at the ruled width at\n\
         \x20 all. That is the sharper finding: Decisions-06 Q17 chose n = 5 because it has hot\n\
         \x20 structure and an intact standing score, and at that width two of the three search\n\
         \x20 kinds in the frozen language cost more per game than the protocol can spend."
    );
    let _ = Trace::of(&[0.0]);
}
