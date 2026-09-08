//! Part IX 3's three claims about the Chinese Rings, checked against a state graph built
//! from the puzzle's move rules and nothing else.

use overtone_graph::rings::{
    a000975, branching_factor, gray, gray_path, is_legal, moves, pauli_hypercube, rings_graph,
    solution_length, ungray,
};
use overtone_graph::Graph;

#[test]
fn the_state_graph_is_a_path() {
    // "The state graph for n rings is a path of length 2^n - 1. Not a tree. A line."
    for n in 1..=12 {
        let g = rings_graph(n);
        assert_eq!(g.order(), 1 << n, "{n} rings");
        assert_eq!(
            g.edge_count(),
            (1 << n) - 1,
            "{n} rings: a path has V - 1 edges"
        );
        assert_eq!(g.components().len(), 1, "{n} rings: connected");
        let ends = (0..g.order()).filter(|&i| g.degree(i) == 1).count();
        assert_eq!(ends, 2, "{n} rings: a path has exactly two endpoints");
        assert!(
            (0..g.order()).all(|i| g.degree(i) <= 2),
            "{n} rings: no vertex branches"
        );
        assert_eq!(g.diameter(), (1 << n) - 1, "{n} rings: the path's length");
    }
}

#[test]
fn the_gray_code_orders_the_path() {
    // "The solution is the binary Gray code, G(i) = B(i) XOR (B(i) >> 1)."
    for n in 1..=12 {
        for i in 0..(1usize << n) - 1 {
            let (a, b) = (gray(i), gray(i + 1));
            assert_eq!(
                (a ^ b).count_ones(),
                1,
                "consecutive Gray codes differ in one bit"
            );
            assert!(
                moves(a, n).contains(&b),
                "{n} rings: {a} -> {b} is a Gray step but not a legal move"
            );
        }
        // ...and it is a bijection, so the Gray order is the path order.
        assert!((0..(1usize << n)).all(|i| ungray(gray(i)) == i));
    }
}

#[test]
fn the_distance_to_solved_is_a000975() {
    // Minimum moves: (2^(n+1) - 2)/3 for even n, (2^(n+1) - 1)/3 for odd -- OEIS A000975.
    assert_eq!(a000975(10), vec![1, 2, 5, 10, 21, 42, 85, 170, 341, 682]);
    // Seven rings need 85 moves, as Part IX 3 says.
    assert_eq!(solution_length(7), 85);

    // And the graph agrees: all-on to all-off, measured by BFS on the state graph.
    for n in 1..=12 {
        let g = rings_graph(n);
        let all_on = (1usize << n) - 1;
        let d = g.bfs_distances(&[all_on])[0].expect("all-off is reachable");
        assert_eq!(d as u64, solution_length(n), "{n} rings");
        // The same number is the Gray index of the all-on state, which is why it is A000975.
        assert_eq!(
            ungray(all_on) as u64,
            solution_length(n),
            "{n} rings, via ungray"
        );
    }
}

#[test]
fn the_solution_is_shorter_than_the_path() {
    // The puzzle does not traverse the whole state space: the solved state sits about
    // two-thirds of the way along. A fact worth having straight before drawing the figure.
    for n in 2..=20 {
        let path = (1u64 << n) - 1;
        assert!(solution_length(n) < path, "{n} rings");
        let ratio = solution_length(n) as f64 / path as f64;
        // The ratio alternates above and below 2/3 with parity -- 5/7 at n = 3, 2/3 at every
        // even n -- and converges to 2/3 from both sides.
        assert!((0.66..=0.72).contains(&ratio), "{n} rings: ratio {ratio}");
    }
    let far = solution_length(20) as f64 / ((1u64 << 20) - 1) as f64;
    assert!(
        (far - 2.0 / 3.0).abs() < 1e-5,
        "converges to two thirds, got {far}"
    );
}

#[test]
fn the_branching_factor_is_two_and_it_is_still_hard() {
    // Part IX 3's whole point: hardness is not option count. A 7-ring puzzle offers two
    // moves everywhere and takes 85 of them, and 84 of the 85 are "not back the way I came".
    let g = rings_graph(7);
    let b = branching_factor(&g);
    assert!((b - 2.0).abs() < 0.02, "branching factor {b}");
    assert_eq!(solution_length(7), 85);

    // A maze with far more choice per step and far less depth, for the figure's other axis.
    let maze = Graph::from_edges(
        9,
        &[
            (0, 1),
            (1, 2),
            (3, 4),
            (4, 5),
            (6, 7),
            (7, 8),
            (0, 3),
            (3, 6),
            (1, 4),
            (4, 7),
            (2, 5),
            (5, 8),
        ],
    );
    assert!(branching_factor(&maze) > b, "the grid branches more");
    assert!(
        (maze.diameter() as u64) < solution_length(7),
        "and is shallower"
    );
}

#[test]
fn only_one_ring_moves_at_a_time_and_ring_zero_is_always_free() {
    for n in 1..=8 {
        for s in 0..(1usize << n) {
            assert!(is_legal(s, 0));
            let m = moves(s, n);
            assert!(
                !m.is_empty() && m.len() <= 2,
                "state {s} has {} moves",
                m.len()
            );
            for &t in &m {
                assert_eq!((s ^ t).count_ones(), 1);
            }
        }
    }
}

#[test]
fn the_pauli_hypercube_is_the_same_object_with_the_path_lifted() {
    // Part IX 3.1: cells are Pauli strings, legal moves flip one generator.
    for q in 1..=4 {
        let g = pauli_hypercube(q);
        assert_eq!(g.order(), 1 << (2 * q));
        assert!(
            g.is_regular(),
            "every Pauli string has the same number of neighbours"
        );
        assert_eq!(g.degree(0), 2 * q, "one move per generator");
        assert_eq!(g.components().len(), 1);
        assert_eq!(
            g.diameter(),
            2 * q,
            "the hypercube diameter is its dimension"
        );
        // A Gray code still walks it, but now it is one path among many.
        let path = gray_path(2 * q);
        assert_eq!(path.len(), g.order());
        for w in path.windows(2) {
            assert!(g.neighbours(w[0]).contains(&w[1]));
        }
    }
}

#[test]
fn the_hypercube_branches_and_the_rings_do_not() {
    // Same underlying space, opposite difficulty story: the hypercube gives 2q choices at
    // every cell, the rings give two. Part IX 3.1 wants the substrate, not the puzzle.
    let rings = rings_graph(8);
    let cube = pauli_hypercube(4);
    assert_eq!(rings.order(), cube.order(), "both are 2^8 states");
    assert!(branching_factor(&cube) > 3.9 * branching_factor(&rings) / 2.0);
    assert!(
        rings.diameter() > cube.diameter() * 20,
        "and the line is far longer than the cube"
    );
}
