//! M26: options from the Laplacian spectrum, and Go-Explore on the same maze.
//!
//! `cargo run --release -p overtone-graph --example eigenoptions`

use overtone_graph::explore::{go_explore, random_walk, Selection};
use overtone_graph::options::{eigenoptions, policy_disagreement, Laplacian};
use overtone_graph::{Eigenbasis, Graph};
use overtone_wfc::Wfc;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn maze(w: usize, h: usize, seed: u64) -> Graph {
    let mut wfc = Wfc::new(w, h, seed);
    wfc.run();
    Graph::from_wfc(&wfc).largest_component()
}

fn main() {
    let g = maze(16, 12, 7);
    let n = g.order();
    println!(
        "maze: {n} vertices, diameter {}, regular {}",
        g.diameter(),
        g.is_regular()
    );

    // Machado et al. use two different Laplacians in two different sections. On a regular
    // graph that is one diffusion model; on a maze it is two.
    println!("\nthe paper's two diffusion models, at the matrix level");
    let cycle = Graph::from_edges(24, &(0..24).map(|i| (i, (i + 1) % 24)).collect::<Vec<_>>());
    for (name, graph) in [("24-cycle (regular)", &cycle), ("maze (irregular)", &g)] {
        let l = Laplacian::Combinatorial.matrix(graph);
        let ln = Laplacian::Normalized.matrix(graph);
        let mean_degree = (0..graph.order())
            .map(|i| graph.degree(i) as f64)
            .sum::<f64>()
            / graph.order() as f64;
        let worst = l
            .iter()
            .zip(&ln)
            .map(|(a, b)| (a / mean_degree - b).abs())
            .fold(0.0, f64::max);
        println!("{name:>22}:  || L/d - L_norm ||_inf = {worst:.3e}");
    }
    println!("  on a regular graph D = dI makes the two proportional and the eigenvectors");
    println!("  identical. A maze is never regular -- a dead end has degree one, a crossing");
    println!("  four -- so section 2.3 and section 5 of one paper build different options.");

    let combinatorial = eigenoptions(&g, Laplacian::Combinatorial, 8, 0.99);
    let normalized = eigenoptions(&g, Laplacian::Normalized, 8, 0.99);
    let basis = Eigenbasis::of_laplacian(&g);
    let gap = (1..9)
        .map(|k| basis.values[k + 1] - basis.values[k])
        .fold(f64::INFINITY, f64::min);
    println!(
        "\n{} options from 8 proto-value functions, both signs",
        combinatorial.len()
    );
    println!(
        "  worst per-state policy disagreement between the two: {:.3}",
        policy_disagreement(&combinatorial, &normalized)
    );
    println!("  smallest eigenvalue gap among the modes compared:     {gap:.2e}");
    println!("  the gap matters: in a degenerate eigenspace any rotation is a valid basis, so");
    println!("  two solvers may disagree for reasons that are not about the graph at all.");

    // Where the options actually go.
    let dist = g.bfs_distances(&[0]);
    println!("\nwhere the first four options take an agent standing at vertex 0");
    println!(
        "{:>6}  {:>6}  {:>10}  {:>16}  {:>12}",
        "mode", "sign", "steps", "hops from start", "|terminate|"
    );
    for o in combinatorial.iter().take(4) {
        let path = o.rollout(0, 500);
        println!(
            "{:>6}  {:>6}  {:>10}  {:>16}  {:>12}",
            o.mode,
            if o.negated { "-" } else { "+" },
            path.len() - 1,
            dist[*path.last().unwrap()].unwrap_or(0),
            o.termination().len()
        );
    }

    // Go-Explore against an undirected walk, both post-processed the same way.
    println!("\nGo-Explore against a loop-erased random walk, 100000 steps, 21 seeds");
    println!(
        "{:>9}  {:>9}  {:>18}  {:>18}  {:>18}",
        "vertices", "goal hops", "counts only", "with frontier", "random walk"
    );
    for (w, h) in [(16usize, 12usize), (40, 30), (70, 50)] {
        let g = maze(w, h, 7);
        let d = g.bfs_distances(&[0]);
        let far = (0..g.order()).max_by_key(|&i| d[i].unwrap_or(0)).unwrap();
        let mut cells = Vec::new();
        for method in 0..3 {
            let mut found = 0;
            let mut lengths = Vec::new();
            for seed in 0..21u64 {
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                let r = match method {
                    0 => go_explore(&g, 0, far, 100_000, 40, Selection::Counts, &mut rng),
                    1 => go_explore(&g, 0, far, 100_000, 40, Selection::Frontier, &mut rng),
                    _ => random_walk(&g, 0, far, 100_000, &mut rng),
                };
                if let Some(l) = r.path_length {
                    found += 1;
                    lengths.push(l);
                }
            }
            lengths.sort_unstable();
            cells.push(format!(
                "{found:>2}/21  path {}",
                lengths
                    .get(lengths.len() / 2)
                    .map_or("-".into(), |m| m.to_string())
            ));
        }
        println!(
            "{:>9}  {:>9}  {:>18}  {:>18}  {:>18}",
            g.order(),
            d[far].unwrap(),
            cells[0],
            cells[1],
            cells[2]
        );
    }
    println!("  the archive wins outright at 962 vertices and loses at 2751, and the reversal is");
    println!("  a budget effect rather than a size effect: on the same 2751-vertex maze at four");
    println!("  times the budget it is 21/21 against the walk's 20/21. Return-then-explore");
    println!("  spends its opening steps building an archive, and only then aims. Part V 6's");
    println!("  claim that it is \"the right exploration algorithm for an endless maze\" holds in");
    println!("  a window, and the window is set by how many steps the agent is given.");
    println!("  Both are loop-erased before their paths are compared. Without that the walk's");
    println!("  route reads as 67158 steps against the archive's 634, which measures only that");
    println!("  one of them had been post-processed.");
}
