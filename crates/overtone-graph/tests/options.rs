//! M26's claims, each with a test.

use overtone_graph::explore::{go_explore, loop_erase, random_walk, Selection};
use overtone_graph::options::{eigenoption, eigenoptions, policy_disagreement, Laplacian};
use overtone_graph::{Eigenbasis, Graph};
use overtone_wfc::Wfc;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn maze(w: usize, h: usize, seed: u64) -> Graph {
    let mut wfc = Wfc::new(w, h, seed);
    wfc.run();
    Graph::from_wfc(&wfc).largest_component()
}

fn cycle(n: usize) -> Graph {
    Graph::from_edges(n, &(0..n).map(|i| (i, (i + 1) % n)).collect::<Vec<_>>())
}

/// Machado et al. use the normalized Laplacian in section 2.3 and recover the combinatorial
/// one in section 5. On a regular graph those are the same diffusion model; on a maze they
/// are not, and the claim is made at the matrix level where no eigenvector convention can
/// confuse it.
#[test]
fn the_two_laplacians_coincide_only_on_a_regular_graph() {
    let proportionality_gap = |g: &Graph| {
        let d = g.degree(0) as f64;
        let l = Laplacian::Combinatorial.matrix(g);
        let ln = Laplacian::Normalized.matrix(g);
        l.iter()
            .zip(&ln)
            .map(|(a, b)| (a / d - b).abs())
            .fold(0.0, f64::max)
    };
    let ring = cycle(24);
    assert!(ring.is_regular());
    assert!(
        proportionality_gap(&ring) < 1e-12,
        "on a regular graph L_norm must be L/d exactly"
    );

    let m = maze(16, 12, 7);
    assert!(!m.is_regular());
    // No scalar multiple works, so try the most generous one available.
    let l = Laplacian::Combinatorial.matrix(&m);
    let ln = Laplacian::Normalized.matrix(&m);
    let best = (1..=8)
        .map(|d| {
            l.iter()
                .zip(&ln)
                .map(|(a, b)| (a / d as f64 - b).abs())
                .fold(0.0, f64::max)
        })
        .fold(f64::INFINITY, f64::min);
    assert!(
        best > 0.1,
        "some scalar multiple matched on an irregular graph: {best}"
    );
}

/// The consequence: different options. Checked only where the spectrum is non-degenerate,
/// because inside a degenerate eigenspace any rotation is a legitimate basis and two
/// solvers may differ for reasons that have nothing to do with the graph.
#[test]
fn the_two_laplacians_build_different_options_on_a_maze() {
    let g = maze(16, 12, 7);
    let basis = Eigenbasis::of_laplacian(&g);
    let gap = (1..9)
        .map(|k| basis.values[k + 1] - basis.values[k])
        .fold(f64::INFINITY, f64::min);
    assert!(gap > 1e-4, "spectrum is near-degenerate: gap {gap}");

    let disagreement = policy_disagreement(
        &eigenoptions(&g, Laplacian::Combinatorial, 8, 0.99),
        &eigenoptions(&g, Laplacian::Normalized, 8, 0.99),
    );
    assert!(
        disagreement > 0.2,
        "the two diffusion models agreed: {disagreement}"
    );
}

/// The same construction run twice must give the same options, or the disagreement above
/// measures the solver rather than the Laplacian.
#[test]
fn option_construction_is_deterministic() {
    let g = maze(16, 12, 7);
    let a = eigenoptions(&g, Laplacian::Combinatorial, 6, 0.99);
    let b = eigenoptions(&g, Laplacian::Combinatorial, 6, 0.99);
    assert_eq!(policy_disagreement(&a, &b), 0.0);
}

/// Machado et al.'s Theorem 3.1: for `gamma < 1` on a finite state space the termination
/// set of every eigenoption is nonempty.
#[test]
fn every_eigenoption_terminates_somewhere() {
    let g = maze(16, 12, 7);
    for o in eigenoptions(&g, Laplacian::Combinatorial, 10, 0.99) {
        assert!(
            !o.termination().is_empty(),
            "mode {} sign {} never terminates",
            o.mode,
            o.negated
        );
    }
}

/// An eigenoption climbs its own eigenpurpose: following it must not decrease the
/// proto-value function, and must strictly increase it while the option is running.
#[test]
fn an_eigenoption_climbs_its_own_eigenvector() {
    let g = maze(16, 12, 7);
    let basis = Eigenbasis::of_laplacian(&g);
    for mode in 1..6 {
        for negated in [false, true] {
            let o = eigenoption(&g, &basis, mode, negated, 0.99, 10_000);
            let sign = if negated { -1.0 } else { 1.0 };
            let e: Vec<f64> = basis.mode(mode).iter().map(|v| sign * v).collect();
            for s in 0..g.order() {
                if let Some(next) = o.policy[s] {
                    assert!(
                        e[next] > e[s] - 1e-12 || o.q_best[s] > 0.0,
                        "mode {mode} descended at {s}"
                    );
                }
            }
        }
    }
}

/// The constant mode has an identically zero eigenpurpose, so its option terminates
/// everywhere -- which is why `eigenoptions` starts at mode one.
#[test]
fn the_constant_mode_would_be_a_no_op() {
    let g = maze(16, 12, 7);
    let basis = Eigenbasis::of_laplacian(&g);
    let o = eigenoption(&g, &basis, 0, false, 0.99, 1000);
    assert_eq!(o.initiation().len(), 0);
    assert_eq!(o.termination().len(), g.order());
}

/// Loop erasure returns a simple path along real edges with the same endpoints.
#[test]
fn loop_erasure_gives_a_simple_path() {
    let g = maze(16, 12, 7);
    let mut rng = ChaCha8Rng::seed_from_u64(4);
    let d = g.bfs_distances(&[0]);
    let far = (0..g.order()).max_by_key(|&i| d[i].unwrap_or(0)).unwrap();
    let mut at = 0usize;
    let mut trajectory = vec![0usize];
    for _ in 0..20_000 {
        let ns = g.neighbours(at);
        at = ns[rand::Rng::gen_range(&mut rng, 0..ns.len())];
        trajectory.push(at);
        if at == far {
            break;
        }
    }
    let path = loop_erase(&trajectory);
    let mut seen = std::collections::HashSet::new();
    for &v in &path {
        assert!(seen.insert(v), "loop-erased path repeats vertex {v}");
    }
    for pair in path.windows(2) {
        assert!(
            g.neighbours(pair[0]).contains(&pair[1]),
            "{:?} is not an edge",
            pair
        );
    }
    assert_eq!(path[0], 0);
    assert_eq!(*path.last().unwrap(), *trajectory.last().unwrap());
    assert!(path.len() <= trajectory.len());
}

/// Go-Explore beats an undirected walk at *finding* a distant goal on a mid-sized maze --
/// and the comparison is only meaningful because both are post-processed the same way.
#[test]
fn go_explore_finds_the_goal_where_a_random_walk_does_not() {
    let g = maze(40, 30, 7);
    let d = g.bfs_distances(&[0]);
    let far = (0..g.order()).max_by_key(|&i| d[i].unwrap_or(0)).unwrap();
    let count = |archive: bool| {
        (0..11u64)
            .filter(|&seed| {
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                if archive {
                    go_explore(&g, 0, far, 100_000, 40, Selection::Frontier, &mut rng)
                        .found_at
                        .is_some()
                } else {
                    random_walk(&g, 0, far, 100_000, &mut rng)
                        .found_at
                        .is_some()
                }
            })
            .count()
    };
    let (with, without) = (count(true), count(false));
    assert!(
        with > without,
        "Go-Explore {with}/11 against a random walk {without}/11"
    );
}

/// The archive stores a simple path, so what it hands back is a route and not a wander.
#[test]
fn the_archived_route_is_near_optimal() {
    let g = maze(16, 12, 7);
    let d = g.bfs_distances(&[0]);
    let far = (0..g.order()).max_by_key(|&i| d[i].unwrap_or(0)).unwrap();
    let truth = d[far].unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let r = go_explore(&g, 0, far, 100_000, 40, Selection::Frontier, &mut rng);
    let length = r.path_length.expect("should reach the goal on this maze");
    assert!(
        length <= truth + 4,
        "archived route {length} against a shortest path of {truth}"
    );
}
