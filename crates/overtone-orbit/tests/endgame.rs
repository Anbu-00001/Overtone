//! M36: the endgame tablebase, against a brute-force oracle.

use overtone_graph::Graph;
use overtone_orbit::endgame::{brute_force, Tablebase};
use overtone_wfc::Wfc;

fn maze(w: usize, h: usize, seed: u64) -> Graph {
    let mut wfc = Wfc::new(w, h, seed);
    wfc.run();
    Graph::from_wfc(&wfc).largest_component()
}

/// The eigensolve recovers exact optimal play: every cell's value rounds to its true hop
/// count, on three independent mazes.
#[test]
fn the_tablebase_matches_brute_force() {
    for seed in [7u64, 11, 23] {
        let g = maze(12, 9, seed);
        let goal = vec![0usize];
        let table = Tablebase::solve(&g, &goal);
        let truth = brute_force(&g, &goal);
        for (i, t) in truth.iter().enumerate() {
            if let Some(d) = *t {
                let rounded = table.value[i].round() as usize;
                assert_eq!(
                    rounded, d,
                    "seed {seed} cell {i}: tablebase {} against true {d}",
                    table.value[i]
                );
            }
        }
    }
}

/// The optimal step strictly descends the value function, and the line it traces reaches the
/// goal in exactly the shortest number of hops.
#[test]
fn the_tablebase_line_is_a_shortest_path() {
    let g = maze(12, 9, 7);
    let goal = vec![0usize];
    let table = Tablebase::solve(&g, &goal);
    let truth = brute_force(&g, &goal);
    let far = (0..g.order())
        .max_by_key(|&i| truth[i].unwrap_or(0))
        .unwrap();
    let line = table.line(far, 500);
    assert_eq!(*line.last().unwrap(), 0, "the line did not reach the goal");
    assert_eq!(
        line.len() - 1,
        truth[far].unwrap(),
        "the line is not a shortest path"
    );
    for pair in line.windows(2) {
        assert!(table.value[pair[1]] < table.value[pair[0]]);
    }
}

/// Part VII 4's acceptance: the exact solution appears within 100 ms of the coherence
/// transition. Measured on a window larger than the arena Part VII 9.3 specifies.
#[test]
fn the_eigensolve_lands_inside_the_budget() {
    let g = maze(16, 12, 7);
    let start = std::time::Instant::now();
    let table = Tablebase::solve(&g, &[0]);
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_millis() < 100,
        "the endgame took {elapsed:?}, past the 100 ms budget"
    );
    assert!(table.value.iter().all(|v| v.is_finite()));
}
