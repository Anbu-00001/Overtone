//! Does the ladder's saturation move with the size of the candidate set?
//!
//! Lantz et al.'s correction 2: `d` is relative to a declared strategy language. If the
//! ladder flattens because the *language* runs out of distinct policies rather than because
//! the game runs out of depth, that is a fact about the language. A depth-1 policy has only
//! `|legal moves| x |angle grid|` candidates, so the prediction is that saturation tracks
//! that number -- 240 at `n = 4` and 360 at `n = 5`.
use overtone_orbit::game::legal_moves;
use overtone_orbit::ladder::{Ladder, ANGLES};

fn main() {
    let budgets = [16usize, 32, 64, 128, 256];
    println!(
        "{:>3}  {:>10}  {:>44}  {:>3}",
        "n", "candidates", "win rate vs half budget", "d"
    );
    for n in [4usize, 5] {
        let candidates = legal_moves(n).len() * ANGLES.len();
        let l = Ladder::measure(n, &budgets, 12, 17);
        let cells: Vec<String> = l
            .rungs
            .iter()
            .map(|r| {
                format!(
                    "{}:{:.2}{}",
                    r.budget,
                    r.win_rate,
                    if r.is_step { "*" } else { " " }
                )
            })
            .collect();
        println!(
            "{n:>3}  {candidates:>10}  {:>44}  {:>3}",
            cells.join(" "),
            l.depth()
        );
    }
    println!("  a star marks a rung clearing the 0.65 step unit.");
}
