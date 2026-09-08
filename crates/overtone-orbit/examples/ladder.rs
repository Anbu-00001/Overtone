//! M35: the strategy ladder, measured. No interface, by Part VII 12.
//!
//! `cargo run --release -p overtone-orbit --example ladder`

use overtone_lie::{closure, PauliString};
use overtone_orbit::game::{legal_moves, Piece};
use overtone_orbit::ladder::{Ladder, SCORE};
use overtone_orbit::{LANGUAGE, STEP_UNIT};
use overtone_sim::Pauli;

fn main() {
    println!("the strategy language, declared before any number is measured");
    for line in textwrap(LANGUAGE, 78) {
        println!("  {line}");
    }
    println!("  score: {SCORE}");
    println!("  step unit: {STEP_UNIT} win rate, mid-range of the paper's 0.60 to 0.75.");

    println!("\nbranching factor, by construction and then measured");
    println!("{:>4}  {:>10}  {:>28}", "n", "legal moves", "breakdown");
    for n in [3usize, 4, 5] {
        let moves = legal_moves(n);
        let counts: Vec<String> = Piece::ALL
            .iter()
            .map(|p| {
                let c = moves
                    .iter()
                    .filter(
                        |m| matches!(m, overtone_orbit::Move::Apply { piece, .. } if piece == p),
                    )
                    .count();
                format!("{} {}", p.name(), c)
            })
            .collect();
        println!("{n:>4}  {:>10}  {:>28}", moves.len(), counts.join(", "));
    }
    println!("  chess reports about 35. Part VII 9.2 asks for 25 to 40 and asks for it to be");
    println!("  measured rather than intended.");

    println!("\nthe queen is not additive: [rook, bishop] leaves the span of either");
    let rook = PauliString::single(0, Pauli::Z);
    let bishop = PauliString::from_factors(&[(0, Pauli::X), (1, Pauli::X)]);
    let separate = closure(&[rook], 3, 64).dim() + closure(&[bishop], 3, 64).dim();
    let together = closure(&[rook, bishop], 3, 64).dim();
    println!("  dim(closure(rook)) + dim(closure(bishop)) = {separate}");
    println!("  dim(closure(rook, bishop))                 = {together}");
    if let Some((c, _)) = rook.commutator(&bishop) {
        println!("  [Z_0, X_0 X_1] = {}, present in neither", c.render(3));
    }
    println!("  chess folklore has said a queen beats a rook plus a bishop for five hundred");
    println!("  years. That is the non-additivity of the Lie closure, and it is one line.");

    println!("\nthe ladder");
    let budgets = [1usize, 2, 4, 8, 16, 32, 64, 128];
    for n in [3usize, 4, 5] {
        let ladder = Ladder::measure(n, &budgets, 40, 17);
        println!("  n = {n}");
        println!("  {:>10}  {:>12}  {:>8}", "budget", "win vs half", "step?");
        for r in &ladder.rungs {
            println!(
                "  {:>10}  {:>12.3}  {:>8}",
                r.budget,
                r.win_rate,
                if r.is_step { "yes" } else { "" }
            );
        }
        println!(
            "  d = {}   rising region = {:.2} orders of magnitude   candidates = {}",
            ladder.depth(),
            ladder.orders_of_magnitude(),
            legal_moves(n).len() * overtone_orbit::ladder::ANGLES.len()
        );
    }
    println!("  d is the count of rungs clearing the step unit -- Lantz et al.'s own");
    println!("  procedure. The orders of magnitude is a different quantity and is reported");
    println!("  separately because Part VII 11 asks for it under the name of d.");
}

fn textwrap(s: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut line = String::new();
    for word in s.split_whitespace() {
        if line.len() + word.len() + 1 > width {
            out.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        out.push(line);
    }
    out
}
