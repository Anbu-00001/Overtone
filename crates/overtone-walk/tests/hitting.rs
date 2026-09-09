//! Phase 5, milestones 1-3 of Decisions 03 Q7.4: the hypercube, the reduction, and the walk
//! on the actual graph that proves the reduction was not wishful.

use overtone_walk::coined::{Coined, VertexCoin};
use overtone_walk::families::{classical_hitting, hypercube, welded_tree};
use overtone_walk::hitting::{concurrent, one_shot_uniform};
use overtone_walk::reduced::{amplify, hypercube_time, welded_best, welded_horizon, Line};

#[test]
fn the_welded_tree_is_three_regular_except_at_its_roots() {
    for n in 1..=5 {
        let g = welded_tree(n, 7);
        assert!(g.is_consistent(), "{} is not a graph", g.name);
        assert_eq!(g.order(), (1usize << (n + 2)) - 2);
        // Two roots of degree 2, everything else degree 3. Decisions 03 Q7.2: padding the
        // roots to three ports would give a walk that runs and is not the published one.
        assert_eq!(
            g.degrees(),
            vec![(2, 2), (3, g.order() - 2)],
            "n={n} degree distribution"
        );
        assert_eq!(g.degree_of_source_and_target(), (2, 2));
    }
}

#[test]
fn the_hypercube_is_n_regular_and_classically_exponential() {
    for n in 1..=10 {
        let g = hypercube(n);
        assert!(g.is_consistent());
        assert_eq!(g.order(), 1 << n);
        assert_eq!(g.degrees(), vec![(n, 1 << n)]);
    }
    // The classical corner-to-corner walk is the 2^n the quantum walk is measured against.
    for n in [4, 8, 12, 16] {
        let h = classical_hitting(n);
        let two_n = (1u64 << n) as f64;
        assert!(
            h > two_n && h < 1.4 * two_n,
            "n={n}: classical hitting {h} is not near 2^n = {two_n}"
        );
    }
}

#[test]
fn the_panel_counts_arcs_the_way_the_graphs_do() {
    // examples/hitting.rs prints the collapse table from closed forms so it can reach n = 20.
    // A wrong formula there would be a wrong claim in the panel, so the formulas are pinned
    // against graphs that are actually built.
    for n in 1..=6 {
        let g = welded_tree(n, 3);
        let order = (1usize << (n + 2)) - 2;
        assert_eq!(g.order(), order);
        assert_eq!(2 * g.edges(), 3 * order - 2, "welded n={n} arcs");
        assert_eq!(Line::welded(n).dimension(), 4 * n + 2);
    }
    for n in 1..=10 {
        let g = hypercube(n);
        assert_eq!(2 * g.edges(), n << n, "hypercube n={n} arcs");
        assert_eq!(Line::hypercube(n).dimension(), 2 * n);
    }
}

#[test]
fn both_reductions_are_orthogonal() {
    for n in 1..=8 {
        assert!(
            Line::hypercube(n).orthogonality_error() < 1e-14,
            "hypercube n={n}"
        );
        assert!(
            Line::welded(n).orthogonality_error() < 1e-14,
            "welded n={n}"
        );
    }
}

#[test]
fn the_welded_walk_has_a_light_cone() {
    // Li, Li and Luo, after Eq. (4.75): M_U|0> = |1>, and one step moves amplitude at most two
    // indices, so p(t) = 0 for t < 2n. Structurally zero, not numerically small -- Part II 3's
    // light cone showing up in the reduced model.
    for n in 1..=6 {
        let ps = Line::welded(n).arrival(4 * n);
        for (t, &p) in ps.iter().enumerate() {
            let t = t + 1;
            if t < 2 * n {
                assert_eq!(p, 0.0, "n={n} t={t} should be exactly zero, got {p}");
            }
        }
    }
}

#[test]
fn the_welded_success_probability_beats_the_theorem() {
    // Theorem 4.1 promises max{p(t) : t in [2n, 3.6 n log2(5n)]} > 1/(20n) for sufficiently
    // large n. Decisions 03 12 asks for the smallest n at which it actually holds here.
    //
    // The answer is n = 1: it holds at every size from 1 to 40, with two orders of magnitude
    // of headroom. The theorem being asymptotic is a limitation of the proof, not of the walk.
    for n in 1..=40 {
        let (t1, p) = welded_best(n);
        assert!(
            p > 1.0 / (20.0 * n as f64),
            "n={n}: best p={p} at T1={t1} fails 1/(20n)"
        );
        assert!(t1 >= 2 * n && t1 <= welded_horizon(n), "n={n}: T1={t1}");
        assert!(
            p > 0.35,
            "n={n}: measured p={p} lower than every earlier run"
        );
    }
}

#[test]
fn the_reduced_welded_walk_is_the_walk_on_the_graph() {
    // The test Decisions 03 Q7.5 says is the one that matters: an incorrect reduction is
    // invisible until it is checked against the graph it claims to reduce. Seeds vary because
    // Lemma 3.1 holds regardless of the vertex naming or of which random cycle came out.
    for n in 1..=3 {
        for seed in [1u64, 2, 99] {
            let g = welded_tree(n, seed);
            let mut walk = Coined::new(&g.adjacency, VertexCoin::Grover);
            walk.set_uniform(g.source);
            let steps = 6 * n + 6;
            let full = one_shot_uniform(&mut walk, g.target, steps);
            let reduced = Line::welded(n).arrival(steps);
            for t in 0..steps {
                assert!(
                    (full[t] - reduced[t]).abs() < 1e-12,
                    "n={n} seed={seed} t={}: graph {} vs reduced {}",
                    t + 1,
                    full[t],
                    reduced[t]
                );
            }
        }
    }
}

#[test]
fn the_reduced_hypercube_walk_is_the_walk_on_the_graph() {
    for n in 1..=7 {
        let g = hypercube(n);
        let mut walk = Coined::new(&g.adjacency, VertexCoin::Grover);
        walk.set_uniform(g.source);
        let steps = 2 * n + 8;
        let full = one_shot_uniform(&mut walk, g.target, steps);
        let reduced = Line::hypercube(n).arrival(steps);
        for t in 0..steps {
            assert!(
                (full[t] - reduced[t]).abs() < 1e-12,
                "n={n} t={}: graph {} vs reduced {}",
                t + 1,
                full[t],
                reduced[t]
            );
        }
    }
}

#[test]
fn kempe_the_walk_reaches_the_far_corner_in_linear_time() {
    // The settled result, and the reason it goes first: for the n-cube the Grover-coined walk
    // is at the antipode with probability tending to one at T ~ pi*n/2, where the classical
    // walk needs 2^n steps.
    //
    // The convergence is slow and not monotone -- 1 - O(n^{-1/5}) -- so the assertion is a
    // floor that rises with n rather than a limit, and small n is *reported*, not asserted.
    for n in 8..=20 {
        let t = hypercube_time(n);
        assert_eq!(t % 2, n % 2, "n={n}: parity");
        assert!(
            (t as f64 - std::f64::consts::PI * n as f64 / 2.0).abs() <= 1.0,
            "n={n}: T={t} is not within 1 of pi*n/2"
        );
        let p = Line::hypercube(n).arrival(t)[t - 1];
        assert!(p > 0.45, "n={n}: p({t}) = {p}");
    }
}

#[test]
fn the_parity_of_the_prescribed_time_is_load_bearing() {
    // Ignore T = n (mod 2) and the antipode's probability is exactly zero, because the
    // hypercube is bipartite by Hamming weight. This is the trap the acceptance test would
    // otherwise fail on, so it is pinned as its own claim.
    for n in [6, 9, 12] {
        let t = hypercube_time(n);
        let ps = Line::hypercube(n).arrival(t + 2);
        assert!(ps[t - 1] > 0.4, "n={n}: p({t}) = {}", ps[t - 1]);
        assert_eq!(ps[t - 2], 0.0, "n={n}: p({}) must vanish by parity", t - 1);
        assert_eq!(ps[t], 0.0, "n={n}: p({}) must vanish by parity", t + 1);
    }
}

#[test]
fn krovi_brun_the_dft_coin_builds_a_cage_the_grover_coin_does_not() {
    // Part VI-A T2, made concrete. Same graph, same start, same measurement schedule: on the
    // 4-cube with the DFT coin a fixed slice of the amplitude is trapped by interference and
    // never arrives -- which no classical walk on a connected graph can do. With the Grover
    // coin it all arrives. The coin is the cage, not the maze.
    //
    // The trapped fraction measures as exactly 3/7. Krovi and Brun give the mechanism for this
    // graph -- the DFT walk operator on Q_4 has eigenvalues 1, -1, i, -i each eightfold
    // degenerate, leaving a sixteen-dimensional space of eigenvectors with no amplitude at the
    // target -- but the 3/7 is measured here, not quoted from them.
    let g = hypercube(4);
    let steps = 2_000;

    let mut dft = Coined::new(&g.adjacency, VertexCoin::Dft);
    dft.set_uniform(g.source);
    let caged = concurrent(&mut dft, g.target, steps);

    let mut grover = Coined::new(&g.adjacency, VertexCoin::Grover);
    grover.set_uniform(g.source);
    let free = concurrent(&mut grover, g.target, steps);

    assert!(
        (caged.residual - 3.0 / 7.0).abs() < 1e-9,
        "DFT residual {} is not 3/7",
        caged.residual
    );
    assert!(
        free.residual < 1e-12,
        "Grover residual {} -- this walk should always arrive",
        free.residual
    );
    // A plateau, not a slow leak: what still arrives in the last quarter of the run is twelve
    // orders of magnitude below what never arrives, and it keeps falling -- at 20000 steps the
    // same tail is 9e-30.
    let tail: f64 = caged.detected[steps - steps / 4..].iter().sum();
    assert!(tail < 1e-12, "DFT detected {tail} in the tail; not trapped");
}

#[test]
fn the_cage_is_not_a_property_of_the_dft_coin_alone() {
    // The obvious over-reading of the previous test is "the DFT coin traps". It does not: on
    // the 3-cube the same coin arrives with probability one. Dimension four is where the
    // degeneracy appears, which is why Krovi and Brun name that dimension specifically.
    let g = hypercube(3);
    let mut dft = Coined::new(&g.adjacency, VertexCoin::Dft);
    dft.set_uniform(g.source);
    assert!(
        concurrent(&mut dft, g.target, 2_000).residual < 1e-12,
        "the 3-cube should have no cage"
    );
}

#[test]
fn the_walk_stays_unitary_on_a_graph_of_mixed_degree() {
    for coin in [VertexCoin::Grover, VertexCoin::Dft] {
        let g = welded_tree(2, 5);
        let walk = Coined::new(&g.adjacency, coin);
        assert!(
            walk.unitarity_error() < 1e-13,
            "{coin:?} on {}: {}",
            g.name,
            walk.unitarity_error()
        );
    }
}

#[test]
fn the_walk_amplitude_vanishes_on_every_even_step() {
    // Li, Li and Luo Eq. (4.81) and Remark 6.1: p_T = 0 for even T, which is why their
    // conjecture only considers odd T. Exactly zero, like the light cone -- so the assertion
    // is equality, not a tolerance.
    for n in 1..=6 {
        for (t, a) in Line::welded(n).amplitudes(6 * n).iter().enumerate() {
            if (t + 1) % 2 == 0 {
                assert_eq!(*a, 0.0, "n={n} t={} amplitude {a}", t + 1);
            }
        }
    }
}

#[test]
fn conjecture_six_one_the_best_time_is_about_2_12_n() {
    // The paper's §6 conjectures T in [2n, 2.5n] with T ~ n / sqrt(pq) = 2.1213n, from
    // numerical simulation. Independently reproduced here, and it is the reason T1 is not
    // deep inside the 3.6 n log2(5n) horizon the algorithm scans.
    for n in [10usize, 20, 30, 50, 100] {
        let a = amplify(n);
        assert_eq!(a.t1 % 2, 1, "n={n}: T1={} should be odd", a.t1);
        let ratio = a.t1 as f64 / n as f64;
        assert!(
            (2.0..=2.5).contains(&ratio),
            "n={n}: T1/n = {ratio} outside the conjectured window"
        );
        assert!(
            a.amplitude.abs() > (n as f64).powf(-1.0 / 3.0),
            "n={n}: amplitude {} below n^-1/3",
            a.amplitude.abs()
        );
    }
}

#[test]
fn table_two_reproduces() {
    // The paper's Table 2 gives the exact T for n = 50, 100 and 150. Recomputed from the
    // reduced matrix here, with no constant taken from the paper except the horizon.
    for (n, t) in [(50usize, 109usize), (100, 215), (150, 323)] {
        assert_eq!(amplify(n).t1, t, "Table 2 disagrees at n={n}");
    }
}

#[test]
fn long_amplification_reaches_certainty() {
    // Decisions 03 Q7.1.4: the exponential separation is the plain walk's; the zero-error
    // claim is the walk plus this layer. Both are true, they are different claims, and this is
    // the second one. n = 150 is in the test because it is the first size where T2 > 1.
    for n in [2usize, 4, 8, 16, 32, 150] {
        let a = amplify(n);
        assert!(
            (a.probability - 1.0).abs() < 1e-12,
            "n={n}: amplified to {} with T1={} T2={} alpha={}",
            a.probability,
            a.t1,
            a.t2,
            a.alpha
        );
    }
    assert_eq!(amplify(150).t2, 2, "n=150 should need two Grover rounds");
    assert_eq!(amplify(32).t2, 1, "n=32 should need one");
}
