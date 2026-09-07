//! The race: four agents, one world, four amplitude fields.
//!
//! Part IV 2.3. No sprite, and yet you will find yourself wanting one of them to win — which
//! is what a character is. The four are the three baselines Part II 7 makes mandatory plus
//! an optimised coin.
//!
//! # A constant coin cannot see the world
//!
//! The substrate assigns one of two coins to each site, so an agent that plays the *same*
//! coin at both letters never touches the substrate at all: its spread on the Fibonacci
//! world, the Rudin-Shapiro world and the clean lattice is not merely similar, it is
//! bit-identical. The first version of this race reported exactly that and it looked like a
//! bug. It is the point. Part II 2 says the policy is the coin, and an agent whose coin does
//! not depend on the local feature is an agent with no policy — the world is information it
//! declines to use.
//!
//! Only static disorder separates them, and only because there the local parameter is drawn
//! from a continuum rather than from two letters, so no single coin can be blind to it.
//!
//! **The optimised coin is not yet the learned coin.** Part II's M9 wants a coin conditioned
//! on a whole local feature vector and trained by policy gradient. What is here is a direct
//! search over the two coin angles the substrate selects between, maximising the spread
//! reached on this world. That is a real optimisation against a real objective and a fair
//! baseline to beat, but calling it "learned" would be claiming M9, so it is called what it
//! is.

use crate::coin::Coin;
use crate::stats::{classical_sigma, fit_exponent};
use crate::substrate::Substrate;
use crate::walk::{sigma_trace, Run};

/// One agent's result on one world.
#[derive(Clone, Debug)]
pub struct Entrant {
    pub name: &'static str,
    pub sigma: Vec<f64>,
    pub beta: f64,
    pub r_squared: f64,
    /// The two coin angles the agent plays, where it has any.
    pub angles: Option<(f64, f64)>,
    /// Whether the agent's coin depends on the local feature at all.
    pub sees_the_world: bool,
}

/// Probability of arriving within `window` sites of `target` after `steps`.
///
/// Spread is the wrong objective and the first version of this used it. `sigma` is maximised
/// by the coin that does not mix at all: the field splits into two ballistic beams that sit
/// exactly on the light cone, `sigma` comes out at very nearly `t`, and since that coin is
/// optimal on every substrate the optimised agent again learns nothing about the world. The
/// objective has to be one the world can help or hinder — which is Part II P7's own choice,
/// arrival, the thing a hitting time measures.
fn arrival(substrate: &Substrate, ta: f64, tb: f64, steps: usize, target: i64, window: i64) -> f64 {
    let mut w = crate::walk::Walk::new(steps);
    let (ca, cb) = (Coin::theta(ta), Coin::theta(tb));
    for _ in 0..steps {
        w.step(substrate, &ca, &cb);
    }
    w.distribution()
        .iter()
        .filter(|(x, _)| (x - target).abs() <= window)
        .map(|(_, p)| p)
        .sum()
}

/// Search the two coin angles for the widest spread on this substrate.
///
/// Coarse grid, then local refinement. Deterministic — no RNG — so a world always produces
/// the same optimised agent and the race is reproducible from its seed alone.
pub fn optimise_coins(substrate: &Substrate, steps: usize) -> ((f64, f64), f64) {
    let target = steps as i64 / 2;
    let window = 2;
    let spread =
        |sub: &Substrate, ta: f64, tb: f64, st: usize| arrival(sub, ta, tb, st, target, window);
    let pi = std::f64::consts::PI;
    let grid = 16;
    // Multi-start: refine from the best few grid points, not just the single best. A single
    // start gets stuck, and it gets stuck in a way that matters here -- on the Fibonacci
    // world it settled on a world-conditioned pair scoring 0.074 while a blind pair scores
    // 0.118, which would have been reported as "conditioning helps on Fibonacci" when the
    // truth is the opposite.
    let mut points: Vec<(f64, f64, f64)> = Vec::new();
    for i in 1..grid {
        for j in 1..grid {
            let ta = pi * i as f64 / grid as f64;
            let tb = pi * j as f64 / grid as f64;
            points.push((spread(substrate, ta, tb, steps), ta, tb));
        }
    }
    // The blind solution -- one coin at both letters -- has to be searched properly rather
    // than left to a 16-point grid, because arrival probability at a single site oscillates
    // sharply in the coin angle and a coarse grid ranks the diagonal badly. Without this
    // the optimiser reports "conditioning helps" on worlds where it does not.
    let fine = 400;
    for i in 1..fine {
        let t = pi * i as f64 / fine as f64;
        points.push((spread(substrate, t, t, steps), t, t));
    }
    points.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let mut best = (points[0].1, points[0].2);
    let mut best_score = points[0].0;
    for &(_, sa, sb) in points.iter().take(6) {
        let mut local = (sa, sb);
        let mut local_score = spread(substrate, sa, sb, steps);
        let mut step = pi / (2.0 * grid as f64);
        for _ in 0..6 {
            let centre = local;
            for di in -1..=1i32 {
                for dj in -1..=1i32 {
                    let ta = centre.0 + di as f64 * step;
                    let tb = centre.1 + dj as f64 * step;
                    let sc = spread(substrate, ta, tb, steps);
                    if sc > local_score {
                        local_score = sc;
                        local = (ta, tb);
                    }
                }
            }
            step *= 0.5;
        }
        if local_score > best_score {
            best_score = local_score;
            best = local;
        }
    }
    (best, best_score)
}

/// Run all four entrants on one world.
pub fn race(substrate: &Substrate, steps: usize) -> Vec<Entrant> {
    let make = |sigma: Vec<f64>, name: &'static str, angles, sees| {
        let f = fit_exponent(&sigma, 0.5);
        Entrant {
            name,
            sigma,
            beta: f.beta,
            r_squared: f.r_squared,
            angles,
            sees_the_world: sees,
        }
    };
    let pi4 = std::f64::consts::FRAC_PI_4;
    let h = Coin::hadamard();
    let g = Coin::grover();

    let (best, _) = optimise_coins(substrate, steps);
    vec![
        make(
            sigma_trace(substrate, &h, &h, Run::coherent(steps)),
            "Hadamard",
            Some((pi4, pi4)),
            false,
        ),
        make(
            sigma_trace(substrate, &g, &g, Run::coherent(steps)),
            "Grover",
            None,
            false,
        ),
        make(
            sigma_trace(
                substrate,
                &Coin::theta(best.0),
                &Coin::theta(best.1),
                Run::coherent(steps),
            ),
            "optimised",
            Some(best),
            (best.0 - best.1).abs() > 1e-6,
        ),
        make(classical_sigma(steps), "classical", None, false),
    ]
}
