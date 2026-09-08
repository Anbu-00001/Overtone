//! M47: the Chinese Rings figure, as data.
//!
//! `cargo run --release -p overtone-graph --example rings`
//!
//! Part IX 3 wants one figure: the state graph as a path, the Gray-code solution, the
//! A000975 growth curve, plotted beside the maze's branching factor. This prints the numbers
//! that figure is drawn from, each one measured off a state graph built from the puzzle's two
//! move rules rather than asserted.

use overtone_graph::rings::{
    a000975, branching_factor, gray, moves, pauli_hypercube, rings_graph, solution_length,
    PROVENANCE,
};
use overtone_graph::Graph;

fn main() {
    println!("provenance: {PROVENANCE}");
    println!();

    println!(
        "{:>3}  {:>8}  {:>8}  {:>7}  {:>8}  {:>6}",
        "n", "states", "edges", "ends", "moves", "b"
    );
    for n in 1..=14 {
        let g = rings_graph(n);
        let ends = (0..g.order()).filter(|&i| g.degree(i) == 1).count();
        println!(
            "{n:>3}  {:>8}  {:>8}  {:>7}  {:>8}  {:>6.3}",
            g.order(),
            g.edge_count(),
            ends,
            solution_length(n),
            branching_factor(&g)
        );
    }
    println!();
    println!("A000975: {:?}", a000975(10));
    println!();

    // The solution path itself, for the small case the figure can actually draw.
    let n = 4;
    let all_on = (1usize << n) - 1;
    let steps: Vec<usize> = (0..=solution_length(n) as usize)
        .map(|i| gray(solution_length(n) as usize - i))
        .collect();
    println!(
        "the {n}-ring solution, {} moves, as ring states:",
        steps.len() - 1
    );
    for w in steps.windows(2) {
        let flipped = (w[0] ^ w[1]).trailing_zeros();
        println!(
            "  {:0width$b} -> {:0width$b}   flip ring {flipped}",
            w[0],
            w[1],
            width = n
        );
    }
    assert_eq!(*steps.first().unwrap(), all_on);
    assert_eq!(*steps.last().unwrap(), 0);
    assert!(steps.windows(2).all(|w| moves(w[0], n).contains(&w[1])));
    println!();

    // The other axis of the figure: choice against difficulty.
    let maze = Graph::from_edges(
        16,
        &(0..4)
            .flat_map(|r| {
                (0..4).flat_map(move |c| {
                    let i = r * 4 + c;
                    let mut e = Vec::new();
                    if c + 1 < 4 {
                        e.push((i, i + 1));
                    }
                    if r + 1 < 4 {
                        e.push((i, i + 4));
                    }
                    e
                })
            })
            .collect::<Vec<_>>(),
    );
    let rings7 = rings_graph(7);
    let cube = pauli_hypercube(3);
    println!(
        "{:>22}  {:>8}  {:>10}  {:>9}",
        "substrate", "states", "branching", "diameter"
    );
    for (name, g) in [
        ("4x4 maze", &maze),
        ("7 rings", &rings7),
        ("Pauli hypercube q=3", &cube),
    ] {
        println!(
            "{name:>22}  {:>8}  {:>10.3}  {:>9}",
            g.order(),
            branching_factor(g),
            g.diameter()
        );
    }
    println!();
    println!(
        "seven rings: branching {:.1}, and still {} moves. Difficulty is not option count.",
        branching_factor(&rings7),
        solution_length(7)
    );
}
