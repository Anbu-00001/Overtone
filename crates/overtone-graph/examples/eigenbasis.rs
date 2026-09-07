//! M22 and M23: the shared eigenbasis, and Bellman as an eigenproblem.
//!
//! `cargo run --release -p overtone-graph --example eigenbasis`

use overtone_graph::{compose, mean_distance, shortest_path_window, Eigenbasis, Graph, Lmdp};
use overtone_wfc::Wfc;

fn main() {
    let mut w = Wfc::new(16, 12, 7);
    w.run();
    let g = Graph::from_wfc(&w).largest_component();
    let n = g.order();
    let basis = Eigenbasis::of_laplacian(&g);

    println!("a Wave Function Collapse maze, as a graph");
    println!(
        "  vertices {n}  edges {}  diameter {}",
        g.edge_count(),
        g.diameter()
    );
    println!("  regular  {}", g.is_regular());
    println!(
        "  Laplacian spectrum: {:.6} {:.6} {:.6} ... {:.6}",
        basis.values[0],
        basis.values[1],
        basis.values[2],
        basis.values[n - 1]
    );
    println!(
        "  zero modes = connected components = {}",
        g.components().len().min(1)
    );

    // Part V 1.3: one character in the exponent.
    println!("\nthe same eigenbasis, two exponents");
    println!(
        "{:>8}  {:>16}  {:>16}",
        "t", "real: <d> diffuse", "imaginary: <d>"
    );
    let dist = g.bfs_distances(&[0]);
    let mut p0 = vec![0.0; n];
    p0[0] = 1.0;
    for t in [0.5, 1.0, 2.0, 4.0, 8.0] {
        println!(
            "{t:>8.1}  {:>16.4}  {:>16.4}",
            mean_distance(&dist, &basis.diffuse(&p0, t)),
            mean_distance(&dist, &basis.interference_probability(&p0, t))
        );
    }
    println!("  the real exponent decays the modes; the imaginary one rotates them, so the");
    println!("  amplitudes interfere. The quantum field outruns diffusion early and then");
    println!("  stops doing so -- it is unitary on a finite graph, so it never settles and");
    println!("  the modes come back into phase. Diffusion only ever spreads.");

    // Part V 2: Bellman as an eigenproblem, and the caveat Part V 2.1 omits.
    println!("\nthe LMDP shortest-path reduction, in z-space (Todorov eq. 19-20)");
    println!(
        "{:>10}  {:>14}  {:>12}",
        "rho", "worst error", "underflowed"
    );
    let good = Lmdp::recommended_rho(&g, &[0]);
    for (rho, err, under) in shortest_path_window(&g, &[0], &[0.5, 2.0, good, 4.0 * good]) {
        let shown = if err.is_finite() {
            format!("{err:.4}")
        } else {
            "inf".into()
        };
        println!("{rho:>10.2}  {shown:>14}  {under:>12}");
    }
    println!("  accuracy needs a large rho; representability needs exp(-rho * diameter) to");
    println!("  clear the f64 floor near exp(-745). Past a diameter of about 23 no rho does");
    println!("  both, and this maze is past it.");

    println!("\nthe same solve in log space, where there is no floor");
    println!("{:>10}  {:>14}", "rho", "worst error");
    let truth = g.bfs_distances(&[0]);
    for rho in [40.0, 400.0, 4000.0] {
        let est = Lmdp::new(&g, vec![0], rho).shortest_paths_stable(20_000, 1e-12);
        let worst = truth
            .iter()
            .enumerate()
            .filter_map(|(i, d)| d.map(|d| (est[i] - d as f64).abs()))
            .fold(0.0, f64::max);
        println!("{rho:>10.0}  {worst:>14.6}");
    }
    println!("  the error falls as 1/rho, exactly as the limit says, with nothing to stop it.");

    // Part V 2.3: policies superpose.
    println!("\npolicies superpose, linearly and exactly");
    let far = (0..n).max_by_key(|&i| truth[i].unwrap_or(0)).unwrap();
    let exits = vec![0usize, far];
    let rho = 12.0;
    let a = Lmdp::with_terminal_values(&g, exits.clone(), vec![1.0, 0.0], rho)
        .desirability(20_000, 1e-14)
        .z;
    let b = Lmdp::with_terminal_values(&g, exits.clone(), vec![0.0, 1.0], rho)
        .desirability(20_000, 1e-14)
        .z;
    println!(
        "{:>8} {:>8}  {:>18}",
        "alpha", "beta", "worst |composed-solved|"
    );
    for (alpha, beta) in [(1.0, 0.0), (0.7, 0.3), (0.5, 0.5), (0.1, 0.9)] {
        let composed = compose(&a, &b, alpha, beta);
        let solved = Lmdp::with_terminal_values(&g, exits.clone(), vec![alpha, beta], rho)
            .desirability(20_000, 1e-14)
            .z;
        let worst = composed
            .iter()
            .zip(&solved)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        println!("{alpha:>8.1} {beta:>8.1}  {worst:>18.3e}");
    }
    println!("  a task nobody solved, solved by adding two that were -- because the equation");
    println!("  in z is linear. The linearity is bought with a restricted control formulation");
    println!("  and a KL cost, and that restriction belongs on the panel beside the result.");
}
