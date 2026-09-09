//! Phase 5, milestones 1-3: the collapse, the gap, and the cage.
//!
//! `cargo run --release -p overtone-walk --example hitting`

use overtone_walk::coined::{Coined, VertexCoin};
use overtone_walk::families::{classical_hitting, hypercube, welded_tree};
use overtone_walk::hitting::concurrent;
use overtone_walk::reduced::{amplify, hypercube_time, welded_best, welded_horizon, Line};

fn main() {
    // Part II 5's panel, which Decisions 03 Q7.5 turns from a picture into the computation.
    // The arc counts are formulas rather than built graphs so the table can reach n = 20; the
    // formulas are checked against real graphs in tests/hitting.rs.
    println!("The collapse -- what the walk actually runs on\n");
    println!(
        "{:>3}  {:>10} {:>10} {:>10}   family",
        "n", "vertices", "arcs", "reduced"
    );
    for n in [2usize, 4, 6, 8, 10] {
        let order = (1usize << (n + 2)) - 2;
        // Three arcs per vertex, less one at each degree-two root.
        println!(
            "{n:>3}  {order:>10} {:>10} {:>10}   welded tree",
            3 * order - 2,
            4 * n + 2
        );
    }
    for n in [4usize, 8, 12, 16, 20] {
        println!(
            "{n:>3}  {:>10} {:>10} {:>10}   hypercube",
            1usize << n,
            n << n,
            2 * n
        );
    }

    println!("\n\nKempe -- the corner-to-corner gap on the hypercube\n");
    println!(
        "{:>3} {:>6} {:>9}  {:>14}  T = n mod 2, |T - pi n / 2| <= 1",
        "n", "T", "p(T)", "classical"
    );
    for n in 2..=20 {
        let t = hypercube_time(n);
        let p = Line::hypercube(n).arrival(t)[t - 1];
        println!("{n:>3} {t:>6} {p:>9.4}  {:>14.4e}", classical_hitting(n));
    }
    println!(
        "\nThe convergence is 1 - O(n^-1/5) and it is not monotone: n=2 lands on an exact zero\n\
         because the two-port Grover coin is the Pauli X and the walk is a deterministic cycle,\n\
         and n=3 and n=9 sit at the bad end of the parity window. This is what an asymptotic\n\
         theorem looks like when you plot it at sizes you can afford."
    );

    println!("\n\nLi, Li and Luo -- the welded tree, and how loose the bound is\n");
    println!(
        "{:>3} {:>7} {:>8} {:>7} {:>10} {:>10} {:>7}",
        "n", "reduced", "horizon", "T1", "p(T1)", "1/(20n)", "T1/n"
    );
    for n in [2usize, 3, 4, 6, 8, 10, 12, 16, 20, 30] {
        let (t1, p) = welded_best(n);
        println!(
            "{n:>3} {:>7} {:>8} {t1:>7} {p:>10.6} {:>10.6} {:>7.2}",
            4 * n + 2,
            welded_horizon(n),
            1.0 / (20.0 * n as f64),
            t1 as f64 / n as f64
        );
    }
    println!(
        "\nTwo measured facts the theorem does not give you. The success probability sits\n\
         between 0.43 and 0.68 -- two orders of magnitude above the 1/(20n) floor at every size\n\
         tried, which says how much the proof gives away to be a proof. And T1 lands at about\n\
         2.2n for n >= 5, while the horizon it is searched over grows as n log n: the classical\n\
         precomputation overshoots by a widening margin. The exceptions are n = 3 and n = 4,\n\
         where the best time jumps to 33 and 59 for a few percent more probability, and those\n\
         two are the whole reason the rest of the window is worth scanning at all."
    );

    println!("\n\nAlgorithm 2 -- the plain walk, then Long's algorithm, then certainty\n");
    println!(
        "{:>4} {:>6} {:>7} {:>11} {:>4} {:>9}  {:>18}",
        "n", "T1", "T1/n", "amplitude", "T2", "alpha", "p after amplifying"
    );
    for n in [4usize, 10, 30, 50, 100, 150, 200] {
        let a = amplify(n);
        println!(
            "{n:>4} {:>6} {:>7.3} {:>11.6} {:>4} {:>9.4}  {:>18.15}",
            a.t1,
            a.t1 as f64 / n as f64,
            a.amplitude,
            a.t2,
            a.alpha,
            a.probability
        );
    }
    println!(
        "\nT1 at n = 50, 100 and 150 is 109, 215 and 323, which is the paper's Table 2 exactly,\n\
         recomputed here from the reduced matrix with no constant taken from the paper except\n\
         the horizon. T1/n approaches 2.137 at n = 300, against the 2.1213 of Conjecture 6.1.\n\
         And the separation and the exactness are two claims, not one: the exponential speedup\n\
         is the plain walk's, at p = Omega(1/n) against 2^Omega(n) classical queries, and the\n\
         zero error is this layer on top of it -- one Grover round up to n = 100, two beyond."
    );

    println!("\n\nThe reduction is not assumed -- it is checked against the graph\n");
    for n in 1..=3 {
        let g = welded_tree(n, 4242);
        let mut w = Coined::new(&g.adjacency, VertexCoin::Grover);
        w.set_uniform(g.source);
        let steps = 6 * n;
        let mut worst: f64 = 0.0;
        for &r in Line::welded(n).arrival(steps).iter() {
            w.step();
            worst = worst.max((w.uniform_overlap(g.target).norm_sqr() - r).abs());
        }
        println!(
            "  {}  {} vertices, {} arcs vs {} dimensions: worst |p_graph - p_reduced| = {worst:.2e}",
            g.name,
            g.order(),
            2 * g.edges(),
            4 * n + 2
        );
    }

    println!("\n\nPart VI-A T2 -- the cage, on a graph with no walls\n");
    for n in [3usize, 4] {
        let g = hypercube(n);
        for coin in [VertexCoin::Grover, VertexCoin::Dft] {
            let mut w = Coined::new(&g.adjacency, coin);
            w.set_uniform(g.source);
            let c = concurrent(&mut w, g.target, 20_000);
            println!(
                "  {:<16} {:<8}   arrives {:.6}   never arrives {:.6}",
                g.name,
                format!("{coin:?}"),
                c.total,
                c.residual
            );
        }
    }
    println!(
        "\nOn the 4-cube the DFT coin leaves exactly 3/7 of the amplitude permanently unable to\n\
         reach the far corner -- on a 4-regular graph with no dead ends, whose far corner is\n\
         four steps away and reachable by 24 distinct shortest paths. No\n\
         classical walk on a connected graph can do that. The Grover coin on the same graph,\n\
         from the same start, arrives with probability one -- so the cage is the walk's, not\n\
         the maze's, which is the whole content of the trap."
    );
}
