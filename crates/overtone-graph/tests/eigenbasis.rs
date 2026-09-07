//! M22 and M23's acceptance tests.
//!
//! The eigenbasis claims are checked against closed forms where one exists — a cycle graph's
//! Laplacian spectrum is `2 - 2cos(2 pi k / n)` and needs no oracle — and the LMDP against
//! breadth-first search, which on an unweighted graph is exact rather than approximate.

use overtone_graph::{compose, mean_distance, shortest_path_window, Eigenbasis, Graph, Lmdp};
use overtone_wfc::Wfc;

fn cycle(n: usize) -> Graph {
    let edges: Vec<(usize, usize)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
    Graph::from_edges(n, &edges)
}

fn maze_graph(seed: u64) -> Graph {
    let mut w = Wfc::new(14, 10, seed);
    w.run();
    Graph::from_wfc(&w).largest_component()
}

#[test]
fn the_laplacian_spectrum_of_a_cycle_matches_the_closed_form() {
    let n = 12;
    let basis = Eigenbasis::of_laplacian(&cycle(n));
    let mut expected: Vec<f64> = (0..n)
        .map(|k| 2.0 - 2.0 * (2.0 * std::f64::consts::PI * k as f64 / n as f64).cos())
        .collect();
    expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for (got, want) in basis.values.iter().zip(&expected) {
        assert!((got - want).abs() < 1e-10, "{got} vs {want}");
    }
}

#[test]
fn the_zero_mode_is_constant_and_counts_the_components() {
    // A graph Laplacian has one zero eigenvalue per connected component, and the zero mode
    // of a connected graph is the constant vector. Both are standard, and both are the
    // reason the diffusion below conserves probability.
    let basis = Eigenbasis::of_laplacian(&cycle(9));
    assert!(basis.values[0].abs() < 1e-10);
    assert!(basis.values[1] > 1e-6, "a cycle is connected");
    let mode = basis.mode(0);
    let first = mode[0];
    for m in &mode {
        assert!((m - first).abs() < 1e-10, "zero mode was not constant");
    }

    // Two disjoint triangles: two zero eigenvalues.
    let two = Graph::from_edges(6, &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]);
    let b2 = Eigenbasis::of_laplacian(&two);
    assert_eq!(two.components().len(), 2);
    assert!(b2.values[0].abs() < 1e-10 && b2.values[1].abs() < 1e-10);
    assert!(b2.values[2] > 1e-6);
}

#[test]
fn diffusion_conserves_probability_and_interference_conserves_norm() {
    let g = maze_graph(3);
    let n = g.order();
    let basis = Eigenbasis::of_laplacian(&g);
    let mut p0 = vec![0.0; n];
    p0[0] = 1.0;

    for t in [0.0, 0.5, 2.0, 10.0, 100.0] {
        let p = basis.diffuse(&p0, t);
        let total: f64 = p.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "diffusion lost probability at t = {t}: {total}"
        );
        let q = basis.interference_probability(&p0, t);
        let norm: f64 = q.iter().sum();
        assert!(
            (norm - 1.0).abs() < 1e-9,
            "interference lost norm at t = {t}: {norm}"
        );
    }
}

#[test]
fn diffusion_settles_and_interference_does_not() {
    // The panel's whole content: one character in the exponent. Real, and the modes decay to
    // the uniform distribution. Imaginary, and they rotate forever, so the field keeps
    // moving and never settles.
    let g = maze_graph(11);
    let n = g.order();
    let basis = Eigenbasis::of_laplacian(&g);
    let mut p0 = vec![0.0; n];
    p0[0] = 1.0;

    // How long relaxation takes is set by the spectral gap, not by a round number: the
    // slowest mode decays as exp(-lambda_1 t), and a long thin maze has a very small gap.
    let gap = basis.values[1];
    let t = 25.0 / gap;
    let late = basis.diffuse(&p0, t);
    let uniform = 1.0 / n as f64;
    let worst = late.iter().map(|p| (p - uniform).abs()).fold(0.0, f64::max);
    assert!(
        worst < 1e-6,
        "diffusion did not reach uniform by t = {t} (gap {gap}): {worst}"
    );

    // The quantum walk is unitary, so it cannot converge to anything: the distance between
    // two well-separated times stays finite forever.
    let a = basis.interference_probability(&p0, t);
    let b = basis.interference_probability(&p0, t + 1.5);
    let moved: f64 = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum();
    assert!(moved > 1e-3, "interference settled down: {moved}");
}

#[test]
fn interference_outruns_diffusion_at_short_times() {
    let g = maze_graph(5);
    let n = g.order();
    let basis = Eigenbasis::of_laplacian(&g);
    let dist = g.bfs_distances(&[0]);
    let mut p0 = vec![0.0; n];
    p0[0] = 1.0;

    let t = 3.0;
    let classical = mean_distance(&dist, &basis.diffuse(&p0, t));
    let quantum = mean_distance(&dist, &basis.interference_probability(&p0, t));
    assert!(
        quantum > classical,
        "the quantum walk did not spread further: {quantum} vs {classical}"
    );
}

#[test]
fn the_laplacian_and_the_adjacency_convention_differ_on_an_irregular_graph() {
    // Part V 1.3 writes the toggle as e^(-Lt) against e^(-iAt). On a *regular* graph those
    // agree up to a global phase, since L = dI - A; on an irregular one they do not, and the
    // difference is in the probabilities rather than in an unobservable phase. A maze is
    // irregular, so the panel has to use one operator on both sides for its claim to hold.
    let regular = cycle(10);
    assert!(regular.is_regular());
    let n = regular.order();
    let mut psi = vec![0.0; n];
    psi[0] = 1.0;
    let by_l = Eigenbasis::of_laplacian(&regular).interference_probability(&psi, 2.0);
    let by_a = Eigenbasis::of_adjacency(&regular).interference_probability(&psi, 2.0);
    let worst = by_l
        .iter()
        .zip(&by_a)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    assert!(worst < 1e-9, "a regular graph should agree: {worst}");

    let maze = maze_graph(2);
    assert!(!maze.is_regular(), "a maze should not be regular");
    let m = maze.order();
    let mut psi = vec![0.0; m];
    psi[0] = 1.0;
    let by_l = Eigenbasis::of_laplacian(&maze).interference_probability(&psi, 2.0);
    let by_a = Eigenbasis::of_adjacency(&maze).interference_probability(&psi, 2.0);
    let worst = by_l
        .iter()
        .zip(&by_a)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    assert!(
        worst > 1e-3,
        "an irregular graph should disagree, but the worst difference was {worst}"
    );
}

#[test]
fn the_eigensolve_recovers_shortest_paths_exactly() {
    // M23's acceptance. BFS on an unweighted graph is exact, so this is a real oracle.
    for seed in [1u64, 4, 9] {
        let g = maze_graph(seed);
        let exit = vec![0usize];
        let truth = g.bfs_distances(&exit);
        // rho is free once the solve is in log space: the error per hop is about
        // ln(degree)/rho, so a large rho is what makes the estimate round to the exact
        // answer, and nothing underflows to stop it.
        let lmdp = Lmdp::new(&g, exit.clone(), 400.0);
        let est = lmdp.shortest_paths_stable(20_000, 1e-12);
        for (i, d) in truth.iter().enumerate() {
            let Some(d) = d else { continue };
            assert_eq!(
                est[i].round() as usize,
                *d,
                "seed {seed}, vertex {i}: eigensolve gave {} for a true distance of {d}",
                est[i]
            );
        }
    }
}

#[test]
fn the_z_space_solve_has_no_usable_rho_on_a_large_maze_and_log_space_does() {
    // Todorov is explicit that (30) is a limit, so no finite rho is exact, and that rho
    // cannot be pushed arbitrarily high because exp(-rho) underflows. Part V 2.1 quotes the
    // headline without either half. Measured, the two demands turn out to *collide*:
    //
    //   accuracy       error per hop is about ln(degree) / rho, so rho must be large
    //   representable  exp(-rho * diameter) must clear the f64 floor near exp(-745)
    //
    // Past a diameter near 23 there is no rho that satisfies both, and this maze is past it.
    let g = maze_graph(7);
    let good = Lmdp::recommended_rho(&g, &[0]);
    let rhos = [0.5, good, 4.0 * good];
    let window = shortest_path_window(&g, &[0], &rhos);
    let row = |r: f64| *window.iter().find(|(x, _, _)| *x == r).unwrap();

    // Too small: the control cost is not dominated by the state cost.
    assert!(row(0.5).1 > 0.5, "rho = 0.5 was unexpectedly accurate");
    assert!(!row(0.5).2, "rho = 0.5 should not underflow");

    // The largest rho that still fits in f64 is *still* not sharp enough to round correctly.
    assert!(
        row(good).1 > 0.1,
        "the collision did not appear: rho = {good} gave error {}",
        row(good).1
    );

    // And past it, exp(-rho * diameter) passes the floor. Partial underflow is the dangerous
    // case: the near half of the graph looks healthy while the far half is silently infinite.
    assert!(
        row(4.0 * good).2,
        "rho = {} on a graph of diameter {} should have underflowed",
        4.0 * good,
        g.diameter()
    );

    // Iterating the value function in log space has no floor, so rho is free -- and the
    // residual error then obeys the limit's own rate. The estimate is d + O(ln(deg) * d / rho),
    // so ten times the rho is a tenth of the error, forever, with nothing to stop it.
    let truth = g.bfs_distances(&[0]);
    let worst_at = |rho: f64| {
        let est = Lmdp::new(&g, vec![0], rho).shortest_paths_stable(20_000, 1e-12);
        truth
            .iter()
            .enumerate()
            .filter_map(|(i, d)| d.map(|d| (est[i] - d as f64).abs()))
            .fold(0.0, f64::max)
    };
    let (near, far) = (worst_at(400.0), worst_at(4000.0));
    assert!(
        near < 0.05,
        "log-space solve was not sharp at rho = 400: {near}"
    );
    assert!(
        far < 0.005,
        "log-space solve was not sharp at rho = 4000: {far}"
    );
    let ratio = near / far;
    assert!(
        (8.0..12.0).contains(&ratio),
        "the error did not fall as 1/rho: {near} -> {far} is a ratio of {ratio}"
    );
    // Every distance rounds to the exact hop count, which is what M23 asked for.
    let est = Lmdp::new(&g, vec![0], 4000.0).shortest_paths_stable(20_000, 1e-12);
    for (i, d) in truth.iter().enumerate() {
        if let Some(d) = d {
            assert_eq!(est[i].round() as usize, *d, "vertex {i}: {}", est[i]);
        }
    }
}

#[test]
fn the_optimal_policy_descends_the_value_function() {
    let g = maze_graph(6);
    let exit = vec![0usize];
    let lmdp = Lmdp::new(&g, exit.clone(), 30.0);
    let sol = lmdp.desirability(20_000, 1e-13);
    let dist = g.bfs_distances(&exit);

    for i in 0..g.order() {
        if i == 0 || g.degree(i) == 0 {
            continue;
        }
        let step = lmdp.optimal_step(&sol.z, i);
        // The most likely step under the optimal control must be onto a neighbour strictly
        // closer to the exit. That is the whole policy, and it came from one eigenvector.
        let best = step
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;
        let (Some(here), Some(there)) = (dist[i], dist[best]) else {
            continue;
        };
        assert!(
            there < here,
            "from {i} at distance {here}, the policy preferred {best} at {there}"
        );
    }
}

#[test]
fn optimal_policies_superpose_linearly() {
    // Todorov's compositionality, and it is exact rather than approximate: the equation in z
    // is linear, so a task weighting the same absorbing set by alpha and beta is solved by
    // exactly alpha z_A + beta z_B. Composing the *value* functions instead -- averaging
    // them -- is the obvious mistake and solves nothing, which is also checked.
    let g = maze_graph(8);
    let n = g.order();
    let far = (0..n)
        .max_by_key(|&i| g.bfs_distances(&[0])[i].unwrap_or(0))
        .unwrap();
    let exits = vec![0usize, far];
    let rho = 12.0;

    let a = Lmdp::with_terminal_values(&g, exits.clone(), vec![1.0, 0.0], rho)
        .desirability(20_000, 1e-14)
        .z;
    let b = Lmdp::with_terminal_values(&g, exits.clone(), vec![0.0, 1.0], rho)
        .desirability(20_000, 1e-14)
        .z;

    let (alpha, beta) = (0.3, 0.7);
    let composed = compose(&a, &b, alpha, beta);
    let solved = Lmdp::with_terminal_values(&g, exits.clone(), vec![alpha, beta], rho)
        .desirability(20_000, 1e-14)
        .z;

    let worst = composed
        .iter()
        .zip(&solved)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    assert!(
        worst < 1e-12,
        "policies did not superpose: worst difference {worst}"
    );

    // And the same combination taken in the value function does not solve it.
    let mixed: Vec<f64> = a
        .iter()
        .zip(&b)
        .map(|(x, y)| {
            let (va, vb) = (-x.max(1e-300).ln(), -y.max(1e-300).ln());
            (-(alpha * va + beta * vb)).exp()
        })
        .collect();
    let wrong = mixed
        .iter()
        .zip(&solved)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    assert!(
        wrong > 1e-6,
        "averaging the value functions accidentally worked, which would mean the test is \
         not distinguishing the two compositions"
    );
}
