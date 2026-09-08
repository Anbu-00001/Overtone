//! Go-Explore on a maze (Part V 6).
//!
//! Return-then-explore. Keep an archive of visited cells; each round, *deliberately return*
//! to a chosen cell rather than hoping to wander back, then explore randomly from there and
//! file whatever is new. Ecoffet et al. designed it for exactly the failure mode a sparse
//! endless maze produces, where undirected exploration detaches from the frontier.
//!
//! Part V is explicit that this one has "no deep quantum rhyme -- it earns its place on fit
//! alone", and it is built that way: no physics, no algebra, one archive and a return rule.
//! Its value here is that it removes "the agent never found the goal" as a confound from
//! every sparse-maze result downstream.
//!
//! # The deterministic return is the whole algorithm
//!
//! Ecoffet et al. note that returning is trivial when the environment is deterministic and
//! resettable, because the stored trajectory can simply be replayed. A maze graph is both,
//! so the return here is a replay and not a re-search. Saying so matters: on a stochastic
//! environment this step is the hard part of the paper and the version implemented here
//! would not apply.
//!
//! # Selection is not a detail, and the first version of it here was wrong
//!
//! Appendix A.5 of Ecoffet et al. gives the cell score as a sum of subscores:
//!
//! ```text
//! CntScore(c, a)   = w_a (1 / (v(c, a) + eps1))^p_a + eps2
//! NeighScore(c, n) = w_n (1 - HasNeighbor(c, n))
//! CellScore(c)     = [ sum_n NeighScore + sum_a CntScore ] + 1
//! ```
//!
//! over three counts -- times chosen, times chosen since the cell last produced something
//! new, times seen -- and over *missing neighbours*, which is available whenever the cell
//! representation carries position. A maze graph carries position by construction, so the
//! neighbour term is free here.
//!
//! A first implementation used only the times-chosen term. Measured against a plain random
//! walk it was **worse**: on a 962-vertex maze it reached a 153-hop goal on 2 seeds of 21
//! where the random walk reached it on 12. The reason is visible in the formula: with only
//! a count term, the archive's mass sits wherever the archive is largest, which is behind
//! the agent, so returns go backwards. The neighbour term is what points selection at the
//! frontier, and it is the one the paper's own domain-knowledge configuration leans on --
//! its Montezuma weights set every count weight to zero.
//!
//! The weights are grid-searched per game in the paper, so they are not imported. What is
//! imported is the *shape*, and [`Selection`] names the two configurations rather than
//! hiding a tuned number: `Counts` is the paper's no-domain-knowledge setting, `Frontier`
//! adds the missing-neighbour term. `examples/eigenoptions.rs` measures both, which is the
//! only honest way to present a hyperparameter this repository cannot re-derive.

use crate::graph::Graph;
use rand::Rng;

/// One archived cell: the best trajectory found to it.
#[derive(Clone, Debug)]
pub struct Cell {
    pub vertex: usize,
    /// Steps taken to reach it on the stored trajectory.
    pub cost: usize,
    /// How many times this cell has been chosen as a return point.
    pub chosen: usize,
    /// How many times it has been stepped onto during exploration, chosen or not.
    pub seen: usize,
    /// The trajectory that reaches it, from the start vertex.
    pub trajectory: Vec<usize>,
}

/// What one exploration run cost.
#[derive(Clone, Debug)]
pub struct Exploration {
    /// Environment steps taken. The only currency; wall-clock is not comparable.
    pub steps: usize,
    /// Steps at which the goal was first reached, or `None`.
    pub found_at: Option<usize>,
    /// Cells in the archive when the run ended.
    pub cells: usize,
    /// Length of the trajectory to the goal, against the true shortest path.
    pub path_length: Option<usize>,
}

/// Which cell-selection subscores are switched on.
///
/// The paper's `p_a = 0.5` and `eps1 = 0.001` are used throughout: those two the grid
/// search found identical in every game and every treatment, so they are the algorithm
/// rather than a tuning. The weights are not, and are named here instead of chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selection {
    /// Count subscores only -- times chosen and times seen. Ecoffet et al.'s
    /// no-domain-knowledge configuration.
    Counts,
    /// Count subscores plus the missing-neighbour term, which a maze graph supplies for
    /// free. The configuration the paper's domain-knowledge runs rely on.
    Frontier,
}

const EPS1: f64 = 0.001;

fn cell_score(
    which: Selection,
    graph: &Graph,
    vertex: usize,
    chosen: usize,
    seen: usize,
    in_archive: &[bool],
) -> f64 {
    let cnt = |v: usize, w: f64| w * (1.0 / (v as f64 + EPS1)).sqrt();
    let mut score = cnt(chosen, 0.1) + cnt(seen, 0.3);
    if which == Selection::Frontier {
        let missing = graph
            .neighbours(vertex)
            .iter()
            .filter(|&&u| !in_archive[u])
            .count();
        score += missing as f64;
    }
    score + 1.0
}

/// Go-Explore: return to an archived cell, then take `explore_steps` random steps from it.
pub fn go_explore<R: Rng + ?Sized>(
    graph: &Graph,
    start: usize,
    goal: usize,
    budget: usize,
    explore_steps: usize,
    which: Selection,
    rng: &mut R,
) -> Exploration {
    let n = graph.order();
    let mut archive: Vec<Option<Cell>> = vec![None; n];
    archive[start] = Some(Cell {
        vertex: start,
        cost: 0,
        chosen: 0,
        seen: 1,
        trajectory: vec![start],
    });
    let mut steps = 0usize;
    let mut found_at = None;
    let mut path_length = None;

    while steps < budget {
        // Choose a cell to return to.
        let in_archive: Vec<bool> = archive.iter().map(|c| c.is_some()).collect();
        let candidates: Vec<usize> = (0..n).filter(|&i| in_archive[i]).collect();
        let weights: Vec<f64> = candidates
            .iter()
            .map(|&i| {
                let c = archive[i].as_ref().unwrap();
                cell_score(which, graph, i, c.chosen, c.seen, &in_archive)
            })
            .collect();
        let total: f64 = weights.iter().sum();
        let mut pick = rng.gen::<f64>() * total;
        let mut chosen_index = candidates[0];
        for (&c, &w) in candidates.iter().zip(&weights) {
            pick -= w;
            if pick <= 0.0 {
                chosen_index = c;
                break;
            }
        }

        let (mut at, mut path) = {
            let cell = archive[chosen_index].as_mut().unwrap();
            cell.chosen += 1;
            (cell.vertex, cell.trajectory.clone())
        };
        // The archive keeps a *simple* path to each cell -- an excursion that returns to a
        // vertex already on the route contributes nothing to reaching the cell, and storing
        // it would make every archived route longer than the walk needed it to be. Loop
        // erasure is maintained here **incrementally**, one vertex at a time, rather than by
        // re-erasing the whole trajectory each step: the walk only ever appends, so the
        // erased prefix of a prefix is the prefix of the erased path. Re-erasing per step is
        // `O(length)` and turns a long run quadratic; this is amortised `O(1)`.
        let mut position: std::collections::HashMap<usize, usize> =
            path.iter().enumerate().map(|(i, &v)| (v, i)).collect();

        // The return is a replay, not a search: it costs no environment steps because the
        // environment is deterministic and resettable. Counting it would be counting the
        // same steps twice, and not counting it is exactly what Ecoffet et al. assume.
        for _ in 0..explore_steps {
            if steps >= budget {
                break;
            }
            let ns = graph.neighbours(at);
            if ns.is_empty() {
                break;
            }
            at = ns[rng.gen_range(0..ns.len())];
            steps += 1;
            match position.get(&at).copied() {
                Some(i) => {
                    for dropped in path.drain(i + 1..) {
                        position.remove(&dropped);
                    }
                }
                None => {
                    position.insert(at, path.len());
                    path.push(at);
                }
            }

            // Ecoffet et al.'s "or to a cell being improved", applied to the route the
            // archive already holds.
            let (better, seen, chosen) = match &archive[at] {
                None => (true, 1, 0),
                Some(c) => (path.len() - 1 < c.cost, c.seen + 1, c.chosen),
            };
            if better {
                archive[at] = Some(Cell {
                    vertex: at,
                    cost: path.len() - 1,
                    chosen,
                    seen,
                    trajectory: path.clone(),
                });
            } else if let Some(c) = archive[at].as_mut() {
                c.seen = seen;
            }
            if at == goal && found_at.is_none() {
                found_at = Some(steps);
                path_length = archive[at].as_ref().map(|c| c.cost);
            }
        }
    }

    Exploration {
        steps,
        found_at,
        cells: archive.iter().filter(|c| c.is_some()).count(),
        path_length,
    }
}

/// An undirected random walk, for the comparison that makes the above mean anything.
///
/// The trajectory it returns is **loop-erased** before its length is reported. Comparing an
/// archive's stored path against a raw wander would be comparing a post-processed answer
/// against an unprocessed one, and the walk's raw length -- tens of thousands of steps on a
/// maze -- would flatter Go-Explore by a factor that is entirely an artefact of not having
/// done the same cheap post-processing to both.
pub fn random_walk<R: Rng + ?Sized>(
    graph: &Graph,
    start: usize,
    goal: usize,
    budget: usize,
    rng: &mut R,
) -> Exploration {
    let mut at = start;
    let mut seen = vec![false; graph.order()];
    seen[start] = true;
    let mut trajectory = vec![start];
    let mut found_at = None;
    let mut path_length = None;
    for step in 1..=budget {
        let ns = graph.neighbours(at);
        if ns.is_empty() {
            break;
        }
        at = ns[rng.gen_range(0..ns.len())];
        seen[at] = true;
        trajectory.push(at);
        if at == goal && found_at.is_none() {
            found_at = Some(step);
            path_length = Some(loop_erase(&trajectory).len() - 1);
            break;
        }
    }
    Exploration {
        steps: budget,
        found_at,
        cells: seen.iter().filter(|s| **s).count(),
        path_length,
    }
}

/// Loop erasure: drop every excursion that returns to a vertex already on the path.
///
/// The result is a simple path along graph edges with the same endpoints, which is what
/// makes it a fair comparator for an archived trajectory.
pub fn loop_erase(trajectory: &[usize]) -> Vec<usize> {
    let mut path: Vec<usize> = Vec::with_capacity(trajectory.len());
    let mut position: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for &v in trajectory {
        if let Some(&i) = position.get(&v) {
            for dropped in path.drain(i + 1..) {
                position.remove(&dropped);
            }
        } else {
            position.insert(v, path.len());
            path.push(v);
        }
    }
    path
}
