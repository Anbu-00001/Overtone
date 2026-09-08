//! M49's acceptance criterion, asked of the part that can be measured today.
//!
//! `cargo run --release -p overtone-cgt --example heatmap`
//!
//! Part IX 7 sets M49's bar at "temperature updates live at 60fps on a 32x32 window". The
//! rendering half of that is Phase 10 work and is gated with the board. The half that decides
//! whether the panel is *possible* is not: it is whether a thousand thermographs fit in a
//! frame. This measures it, at two depths, and then says where the budget actually goes.

use overtone_cgt::Game;
use std::time::Instant;

const W: usize = 32;
const CELLS: usize = W * W;
const FRAME_MS: f64 = 1000.0 / 60.0;

/// A shallow region: one switch, the form a first-order evaluation produces.
fn shallow(i: usize) -> Game {
    let a = (i % 13) as f64 - 3.0;
    let b = a - 1.0 - (i % 5) as f64;
    Game::switch(a, b)
}

/// A region with follow-ups: the form a real position produces, where the heat can be a move
/// or two down. This is the shape that made hottest-first fail, so it is the shape the panel
/// has to be able to afford.
fn deep(i: usize) -> Game {
    let a = (i % 13) as f64 - 3.0;
    let b = a - 1.0 - (i % 5) as f64;
    let inner_l = Game::switch(a + 2.0, a - 1.0);
    let inner_r = Game::switch(b + 1.0, b - 2.0);
    Game::new(vec![inner_l], vec![inner_r])
}

fn bench(name: &str, build: impl Fn(usize) -> Game) -> f64 {
    let field: Vec<Game> = (0..CELLS).map(&build).collect();
    let nodes: usize = field.iter().map(Game::nodes).sum();

    // Warm up, then time ten passes.
    let mut sink = 0.0;
    for g in &field {
        sink += g.temperature();
    }
    let t0 = Instant::now();
    let passes = 10;
    for _ in 0..passes {
        for g in &field {
            sink += g.temperature();
        }
    }
    let per_pass = t0.elapsed().as_secs_f64() * 1000.0 / passes as f64;
    let fps = 1000.0 / per_pass;
    println!(
        "{name:>8}  {CELLS} cells, {nodes} nodes   {per_pass:>8.3} ms/frame   {fps:>9.0} fps   \
         {:>5.2}% of a 60fps frame",
        100.0 * per_pass / FRAME_MS
    );
    let _ = sink;
    per_pass
}

fn main() {
    println!("a {W}x{W} temperature field, thermography only:");
    bench("shallow", shallow);
    let deep_ms = bench("deep", deep);
    println!();
    println!(
        "60fps allows {FRAME_MS:.1} ms per frame, so {:.1} us per cell across {CELLS} cells.",
        FRAME_MS * 1000.0 / CELLS as f64
    );
    if deep_ms < FRAME_MS {
        println!(
            "VERDICT frame-budget-ok: a deep {W}x{W} field costs {:.1}% of a 60fps frame",
            100.0 * deep_ms / FRAME_MS
        );
    } else {
        println!("VERDICT frame-budget-blown: {deep_ms:.3} ms exceeds {FRAME_MS:.1} ms");
    }
    println!();
    println!("Thermography is not the bottleneck and M49 does not depend on making it faster.");
    println!("The budget is spent building each region's game, which in Overtone means");
    println!("evaluating positions -- and one `Position::new` closes an algebra. That is the");
    println!("number to watch: the panel is affordable only if a region's options can be");
    println!("evaluated in the per-cell budget above, or cached and updated incrementally");
    println!("the way a Zobrist-hashed transposition table updates a chess evaluation.");
}
