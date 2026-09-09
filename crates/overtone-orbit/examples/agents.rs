//! M41a: the agent language, validated and run.
//!
//! `cargo run --release -p overtone-orbit --example agents`

use std::time::Instant;

use overtone_orbit::ladder::ANGLES;
use overtone_orbit::language::{Agent, Feature, LANGUAGE_V1, VERSION};
use overtone_orbit::search::{candidates, choose, choose_with_stats, play};
use overtone_orbit::thermal::{ambient_temperature, regions, temperature_field};
use overtone_orbit::work;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const KESTREL: &str = include_str!("../../../agents/kestrel.toml");
const PIKE: &str = include_str!("../../../agents/pike.toml");
const STOAT: &str = include_str!("../../../agents/stoat.toml");

const N: usize = 4;
// The ladder's parameters exactly (`ladder::win_rate` plays at 24 and 2), because M39 will
// compare a number measured here against one measured there and a different horizon would
// make them incomparable.
const COHERENCE: usize = 24;
const K: usize = 2;

fn main() {
    println!("Agent language {VERSION}, frozen\n");
    println!("{LANGUAGE_V1}\n");

    println!("The vocabulary, and what each feature is normalised against at n = {N}\n");
    println!("{:>20}  {:>12}", "feature", "ceiling");
    for f in Feature::ALL {
        println!("{:>20}  {:>12.4}", f.name(), f.ceiling(N));
    }

    println!("\n\nThe three shipped specs\n");
    let agents: Vec<Agent> = [KESTREL, PIKE, STOAT]
        .iter()
        .map(|s| Agent::parse(s).expect("shipped specs must parse"))
        .collect();
    println!(
        "{:>10} {:>9} {:>8} {:>11} {:>10}  digest",
        "name", "kind", "budget", "generators", "features"
    );
    for a in &agents {
        println!(
            "{:>10} {:>9} {:>8} {:>11} {:>10}  {:016x}",
            a.name,
            a.search.kind.name(),
            a.search.budget,
            a.generators.len(),
            a.eval.features.len(),
            a.digest()
        );
    }

    println!("\nCanonical form of the first, which is what the digest is taken over:\n");
    for line in agents[0].to_toml().lines() {
        println!("    {line}");
    }

    println!("\n\nWhat a rejected submission is told\n");
    for (what, src) in [
        (
            "an invented feature",
            STOAT.replace("\"safe_set_size\"", "\"reach_margin\""),
        ),
        (
            "a search we do not implement",
            STOAT.replace("\"greedy\"", "\"expectimax\""),
        ),
        (
            "a field that could not do anything",
            STOAT.replace(
                "budget     = 32",
                "budget     = 32\nrollout    = \"random\"",
            ),
        ),
        (
            "no evaluation at all",
            STOAT.replace("weights    = [1.0]", "weights    = [0.0]"),
        ),
    ] {
        match Agent::parse(&src) {
            Ok(_) => println!("  {what:>34}: ACCEPTED -- this is a bug"),
            Err(e) => println!("  {what:>34}: {e}"),
        }
    }

    println!("\n\nWhat the MCTS tree looks like, and what it costs\n");
    println!(
        "{:>10} {:>8} {:>7} {:>9} {:>8} {:>7} {:>12}",
        "name", "budget", "nodes", "root kids", "chance", "depth", "ms per move"
    );
    for a in &agents {
        for budget in [16usize, 128, 512] {
            let mut spec = a.clone();
            spec.search.budget = budget;
            let mut rng = ChaCha8Rng::seed_from_u64(3);
            let game = overtone_orbit::ladder::opening(N, COHERENCE, K, &mut rng);
            let t = Instant::now();
            let (_, st) = choose_with_stats(&spec, &game, &mut rng);
            println!(
                "{:>10} {budget:>8} {:>7} {:>9} {:>8} {:>7} {:>12.2}",
                spec.name,
                st.nodes,
                st.root_children,
                st.chance_nodes,
                st.depth,
                t.elapsed().as_secs_f64() * 1000.0
            );
        }
    }
    println!(
        "\nZeros are the two tree-less kinds. For mcts, `root kids` is progressive widening at\n\
         work: ceil(sqrt(visits)) of {} candidates, which is what stops the first 240 playouts\n\
         of a 512 budget from going into giving every candidate exactly one visit and never\n\
         reaching a second ply. `chance` counts measure moves reached in the tree -- each is a\n\
         two-outcome Born node, not a sampled successor folded into a decision node.",
        candidates(&agents[0], N).len()
    );

    println!("\n\nThe heuristic the bias is built on, measured\n");
    let rs = regions(N, 3);
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    let mut game = overtone_orbit::ladder::opening(N, COHERENCE, K, &mut rng);
    let strategy = overtone_orbit::ladder::Strategy { budget: 6 };
    let mut ch = ChaCha8Rng::seed_from_u64(9);
    for ply in 0..5 {
        let field = temperature_field(&game, &rs, &ANGLES);
        println!("  ply {ply}: temperature field = {field:?}");
        let (mv, ang) = strategy.choose(&game, &mut ch);
        game.apply(&mv, ang);
    }
    let warm = candidates(&agents[0], N)
        .iter()
        .filter(|(mv, ang)| {
            let mut t = game.clone();
            t.apply(mv, *ang);
            ambient_temperature(&t, &rs, &ANGLES) > -1.0
        })
        .count();
    println!(
        "  {warm} of {} candidates raise a region above -1 from here",
        candidates(&agents[0], N).len()
    );
    println!(
        "\nThe field is uniformly -1 -- every region is a *number*, which is the CGT convention\n\
         for cold -- and about a tenth of the moves lift one region to 0. So temperature here is a\n\
         near-binary `this move heats a region` indicator rather than a graded urgency field,\n\
         and the sibling spread of 1.207 that freeze.rs measured is one outlier in twenty-four\n\
         divided by a mean sitting at -1.\n\
         \n\
         That is a finding, not a defect, and Decisions-05 §2 already hedges it: the default\n\
         weight is 0.15 precisely because nobody yet knows whether this is CGT temperature\n\
         proper or something temperature-shaped. A constant heuristic adds a constant to every\n\
         child and reorders nothing, so on a flat field the bias is inert by construction --\n\
         which is the correct behaviour and is asserted as such in tests/search.rs."
    );

    println!(
        "\n\nWork units -- Decisions-03 Q5: one complex-coefficient update, counted analytically\n"
    );
    println!(
        "{:>10} {:>9} {:>8} {:>16} {:>14}",
        "name", "kind", "budget", "work units/move", "per unit (ns)"
    );
    for a in &agents {
        for budget in [16usize, 128, 512] {
            let mut spec = a.clone();
            spec.search.budget = budget;
            let mut rng = ChaCha8Rng::seed_from_u64(3);
            let game = overtone_orbit::ladder::opening(N, COHERENCE, K, &mut rng);
            let dim_g = [
                game.players[0].algebra().dim(),
                game.players[1].algebra().dim(),
            ];
            let units = work::per_move(&spec, N, dim_g, 2 * COHERENCE.div_ceil(K) + 2);
            let t = Instant::now();
            let _ = choose(&spec, &game, &mut rng);
            let ns = t.elapsed().as_secs_f64() * 1e9 / units as f64;
            println!(
                "{:>10} {:>9} {budget:>8} {units:>16} {ns:>14.1}",
                spec.name,
                spec.search.kind.name(),
            );
        }
    }
    println!(
        "\nThe bound is analytic and free -- no counter runs in the inner loop -- and it is an\n\
         upper bound for the two searches that stop early. Nanoseconds per unit is printed\n\
         beside it so the gap between the model and the machine is visible rather than\n\
         asserted: greedy and negamax pay for evaluation work the unit does not count, because\n\
         the Lie closure is bitset combinatorics and updates no complex coefficient."
    );

    println!("\n\nThe compute axis, which is the whole reason the language declares a search\n");
    println!(
        "{:>10} {:>8} {:>5} {:>5} {:>5} {:>10}  vs stoat@1, 12 games",
        "name", "budget", "W", "D", "L", "win rate"
    );
    let floor = Agent::parse(&STOAT.replace("budget     = 32", "budget     = 1")).unwrap();
    let games = 12;
    for a in &agents {
        for budget in [1usize, 4, 16, 64] {
            let mut spec = a.clone();
            spec.search.budget = budget;
            let (mut w, mut d, mut l) = (0, 0, 0);
            for g in 0..games {
                let swap = g % 2 == 1;
                let (x, y) = if swap {
                    (&floor, &spec)
                } else {
                    (&spec, &floor)
                };
                let seat = usize::from(swap);
                let seed = 2024u64 ^ (g as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                match play(x, y, N, COHERENCE, K, seed) {
                    Some(win) if win == seat => w += 1,
                    Some(_) => l += 1,
                    None => d += 1,
                }
            }
            let rate = (w as f64 + 0.5 * d as f64) / games as f64;
            println!(
                "{:>10} {budget:>8} {w:>5} {d:>5} {l:>5} {rate:>10.3}",
                spec.name
            );
        }
    }
    println!(
        "\nWins, draws and losses separately, because the rate alone hides the shape. The low\n\
         budgets do not *lose* -- they draw, which here means neither side reached checkmate and\n\
         the absorbed weights tied. Converting those draws is what the extra compute buys, and\n\
         it is why a rate averaged over a small sample can look flat across two rungs and then\n\
         move sharply at the third."
    );

    println!(
        "\nThis is not yet `d`. It is the axis `d` is measured along: one rung is one budget,\n\
         and M39 walks it counting how many times strength improves by a declared step unit.\n\
         What matters here is only that the axis exists and that the language is frozen while\n\
         it is walked -- Decisions-01 Q1's point is that a `d` measured before the freeze\n\
         describes an agent family that no longer exists."
    );
}
