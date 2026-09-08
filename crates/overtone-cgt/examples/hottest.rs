//! Does "move in the hottest region" actually play optimally?
//!
//! Part IX 5.1 states it flat: "Optimal play is to move in the hottest region." This example
//! searches small disjunctive sums for a position where that rule gives away points against
//! an optimal opponent, and prints the smallest one it finds.
//!
//! `cargo run --release -p overtone-cgt --example hottest`

use overtone_cgt::{find_disagreement, hottest_first, stop, temperatures, Game};

fn catalogue() -> Vec<Game> {
    let n = Game::number;
    let mut cat = Vec::new();
    for a in -1..=2 {
        for b in -3..=0 {
            if a >= b {
                cat.push(Game::switch(a as f64, b as f64));
            }
        }
    }
    // Components with a hot follow-up: the heat is one move down, where a region's own
    // temperature does not show it.
    for a in 0..=1 {
        for b in -2..=-1 {
            for c in -3..=-2 {
                if a >= b {
                    cat.push(Game::new(
                        vec![Game::switch(a as f64, b as f64)],
                        vec![n(c as f64)],
                    ));
                }
            }
        }
    }
    cat.retain(|g| g.is_hot());
    cat
}

fn show(g: &Game) -> String {
    match g {
        Game::Number(x) => format!("{x}"),
        Game::Options { left, right } => {
            let l: Vec<String> = left.iter().map(show).collect();
            let r: Vec<String> = right.iter().map(show).collect();
            format!("{{{} | {}}}", l.join(","), r.join(","))
        }
    }
}

fn main() {
    let cat = catalogue();
    println!("catalogue:");
    for g in &cat {
        println!(
            "  {:>28}   temperature {:>6.3}  mean {:>6.3}",
            show(g),
            g.temperature(),
            g.mean()
        );
    }
    println!();

    match find_disagreement(&cat, 3) {
        None => println!("no disagreement found: hottest-first was optimal on every sum searched"),
        Some(d) => {
            let parts: Vec<String> = d.components.iter().map(show).collect();
            println!("hottest-first is NOT optimal. Smallest counterexample found:");
            println!("  position     {}", parts.join("  +  "));
            println!("  temperatures {:?}", d.temperatures);
            println!(
                "  {} moves first",
                if d.left_to_move { "Left" } else { "Right" }
            );
            println!("  optimal stop {:.4}", d.optimal);
            println!("  hottest-first {:.4}", d.greedy);
            println!("  loss          {:.4} points", d.loss());
            println!();
            println!(
                "  the hottest component is index {}, but optimal play starts elsewhere",
                overtone_cgt::hottest_index(&d.components).unwrap()
            );
        }
    }

    // Where the rule *is* exact: sums of plain switches, the case the theory covers.
    let switches: Vec<Game> = catalogue().into_iter().filter(|g| g.nodes() == 3).collect();
    match find_disagreement(&switches, 3) {
        None => println!(
            "on {} plain switches, sums of up to 3: hottest-first was optimal every time",
            switches.len()
        ),
        Some(d) => println!("unexpected: plain switches disagree by {}", d.loss()),
    }

    // How often does it happen on the full catalogue?
    let cat = catalogue();
    let (mut pairs, mut bad, mut worst) = (0, 0, 0.0f64);
    for a in 0..cat.len() {
        for b in a..cat.len() {
            for &ltm in &[true, false] {
                let comps = vec![cat[a].clone(), cat[b].clone()];
                let t = temperatures(&comps);
                if (t[0] - t[1]).abs() < 1e-9 {
                    continue;
                }
                let opt = stop(&comps, ltm);
                let gre = hottest_first(&comps, ltm, ltm);
                let loss = if ltm { opt - gre } else { gre - opt };
                pairs += 1;
                if loss > 1e-9 {
                    bad += 1;
                    worst = worst.max(loss);
                }
            }
        }
    }
    println!();
    println!("distinct-temperature pairs searched {pairs}, hottest-first suboptimal on {bad}, worst loss {worst}");
}
