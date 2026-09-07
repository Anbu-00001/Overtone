//! M20: the solver has to actually solve, and its entropy has to actually fall.

use overtone_wfc::{Wfc, TILES};

#[test]
fn every_run_terminates_with_a_consistent_grid() {
    for seed in 0..20u64 {
        let mut w = Wfc::new(14, 10, seed);
        w.run();
        assert!(w.is_finished(), "seed {seed} did not finish");
        let g = w.grid();
        // Every adjacent pair must agree on the socket they share. This is the property the
        // whole algorithm exists to maintain, so it is the property to assert.
        for y in 0..w.height {
            for x in 0..w.width {
                let Some(t) = g[y * w.width + x] else {
                    continue;
                };
                if x + 1 < w.width {
                    if let Some(r) = g[y * w.width + x + 1] {
                        assert_eq!(
                            TILES[t].right, TILES[r].left,
                            "seed {seed}: horizontal seam at ({x}, {y})"
                        );
                    }
                }
                if y + 1 < w.height {
                    if let Some(d) = g[(y + 1) * w.width + x] {
                        assert_eq!(
                            TILES[t].down, TILES[d].up,
                            "seed {seed}: vertical seam at ({x}, {y})"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn entropy_falls_monotonically_to_zero() {
    let mut w = Wfc::new(16, 12, 7);
    let start = w.total_entropy();
    // Uniform weights, so the initial entropy is exactly cells * ln(tiles).
    let expected = (16.0 * 12.0) * (TILES.len() as f64).ln();
    assert!((start - expected).abs() < 1e-12, "{start} vs {expected}");
    w.run();
    let trace = w.entropy_trace();
    for pair in trace.windows(2) {
        assert!(
            pair[1] <= pair[0] + 1e-12,
            "entropy rose from {} to {}",
            pair[0],
            pair[1]
        );
    }
    assert!(
        trace.last().copied().unwrap_or(1.0) < 1e-12,
        "entropy did not reach zero: {:?}",
        trace.last()
    );
}

#[test]
fn the_same_seed_gives_the_same_maze() {
    let mut a = Wfc::new(12, 9, 20260907);
    let mut b = Wfc::new(12, 9, 20260907);
    a.run();
    b.run();
    assert_eq!(a.grid(), b.grid());
    assert_eq!(a.entropy_trace(), b.entropy_trace());
    let mut c = Wfc::new(12, 9, 20260908);
    c.run();
    assert_ne!(a.grid(), c.grid(), "different seeds gave the same maze");
}
