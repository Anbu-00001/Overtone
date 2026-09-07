//! Linearly-solvable MDPs: Bellman as an eigenproblem.
//!
//! Todorov, *Linearly-solvable Markov decision problems*, NeurIPS 2006, and *Efficient
//! computation of optimal actions*, PNAS 106, 11478 (2009). Restrict the controller to
//! reshaping the passive dynamics, charge a KL-divergence cost for doing so, and the Bellman
//! equation becomes linear under the **desirability function** `z = exp(-v)`:
//!
//! ```text
//! z(i) = exp(-q(i)) sum_j p_ij z(j)          (Todorov eq. 18)
//! z    = G P z,  G = diag(exp(-q))           (eq. 19)
//! ```
//!
//! and `z` is the largest eigenvector of `GP`, reachable by the power iteration
//! `z_{k+1} = G P z_k` from `z_0 = 1` (eq. 20) — with no rescaling, because a stochastic `P`
//! scaled down by `G` has spectral radius exactly one when an absorbing set exists.
//!
//! # Two things that must be said on the same screen as the result
//!
//! **The linearity is bought, not free** (Part V 10). The controller may only reshape the
//! passive dynamics, and it pays a KL cost for doing so. That is a real restriction on the
//! class of problems, and a panel showing an instant solution without it is a magic trick.
//!
//! **The shortest-path reduction is a limit, so at any finite cost it is an approximation.**
//! Todorov's §3 sets `q(i) = rho` off the destination set and recovers path length as
//! `s(i) = lim_{rho -> inf} v(i)/rho`, then says so plainly: *"(30) involves a limit and so we
//! cannot obtain the exact shortest paths by solving a single eigenvalue problem. However we
//! can obtain a good approximation by setting rho large enough — but not too large because
//! exp(-rho) may become numerically indistinguishable from 0."* Part V 2.1 quotes the
//! headline without the caveat. There is therefore a usable window in `rho`, and
//! [`shortest_path_window`] measures where it is rather than assuming it.
//!
//! **And the ceiling has a quantitative form Todorov leaves as "not too large".** The
//! desirability at hop distance `d` from the exit is about `exp(-rho * d)`, so what
//! underflows is not `exp(-rho)` but `exp(-rho * diameter)`. In `f64` the smallest subnormal
//! is near `exp(-745)`, which gives
//!
//! ```text
//! rho * diameter < 745
//! ```
//!
//! A `rho` of 40 is comfortable on a ten-vertex path and silently returns infinities on a
//! maze of diameter 31 — which is exactly what the first run of the acceptance test did.
//! [`Lmdp::recommended_rho`] derives a safe value from the graph instead of guessing one.

use crate::graph::Graph;

/// A first-exit problem on a graph: reach the destination set, paying `rho` per step spent
/// outside it.
#[derive(Clone, Debug)]
pub struct Lmdp<'g> {
    pub graph: &'g Graph,
    pub destinations: Vec<usize>,
    /// Desirability held at each destination — `exp(-terminal cost)`. All ones for a plain
    /// "reach the exit" problem. Distinct values are what make Todorov's compositionality
    /// exactly testable: two tasks that share an absorbing set but weight it differently
    /// compose linearly, and a task that weights it `alpha` and `beta` is solved by exactly
    /// `alpha z_A + beta z_B`.
    pub terminal_z: Vec<f64>,
    /// State cost off the destination set. Larger is a sharper approximation to true path
    /// length, until `exp(-rho)` underflows.
    pub rho: f64,
}

/// The desirability field and what it cost to find.
#[derive(Clone, Debug)]
pub struct Solution {
    pub z: Vec<f64>,
    pub iterations: usize,
    /// Largest change in `z` on the final iteration.
    pub residual: f64,
    /// True when *any* reachable vertex underflowed to zero, which is what happens when
    /// `rho * diameter` passes the floating-point floor. Partial underflow is the dangerous
    /// case: the near half of the graph looks perfectly healthy while the far half has
    /// silently become infinite.
    pub underflowed: bool,
}

impl<'g> Lmdp<'g> {
    pub fn new(graph: &'g Graph, destinations: Vec<usize>, rho: f64) -> Lmdp<'g> {
        let terminal_z = vec![1.0; destinations.len()];
        Lmdp::with_terminal_values(graph, destinations, terminal_z, rho)
    }

    /// A first-exit problem with a chosen desirability at each destination.
    pub fn with_terminal_values(
        graph: &'g Graph,
        destinations: Vec<usize>,
        terminal_z: Vec<f64>,
        rho: f64,
    ) -> Lmdp<'g> {
        assert!(
            !destinations.is_empty(),
            "a first-exit problem needs an exit"
        );
        assert_eq!(destinations.len(), terminal_z.len());
        Lmdp {
            graph,
            destinations,
            terminal_z,
            rho,
        }
    }

    /// The largest `rho` that keeps the whole graph above the `f64` floor, with headroom.
    ///
    /// `exp(-rho * diameter)` must stay representable, and the smallest subnormal is near
    /// `exp(-745)`. Half of that bound leaves room for the `1/degree` factors the iteration
    /// also accumulates, and it is still far into the regime where the shortest-path estimate
    /// rounds to the exact answer.
    pub fn recommended_rho(graph: &Graph, destinations: &[usize]) -> f64 {
        let reach = graph
            .bfs_distances(destinations)
            .into_iter()
            .flatten()
            .max()
            .unwrap_or(1)
            .max(1);
        (350.0 / reach as f64).clamp(1.0, 60.0)
    }

    fn terminal_value(&self, i: usize) -> Option<f64> {
        self.destinations
            .iter()
            .position(|&d| d == i)
            .map(|k| self.terminal_z[k])
    }

    fn is_destination(&self, i: usize) -> bool {
        self.destinations.contains(&i)
    }

    /// Solve `z = GPz` by the power iteration of Todorov eq. 20.
    ///
    /// **The convergence test is on `log z`, not on `z`, and that is not a detail.** With
    /// `rho = 50` on a five-vertex path the desirabilities run from `1` down to `e^(-200)`,
    /// so an absolute tolerance of `1e-14` on `z` is met after two iterations — while the
    /// far end of the graph is still two hundred orders of magnitude from its answer. The
    /// first version of this did exactly that and reported a distance of 2 where the answer
    /// was 4. `log z` is the value function, it is the quantity that is actually being
    /// solved for, and its scale is uniform across the graph.
    pub fn desirability(&self, max_iterations: usize, tolerance: f64) -> Solution {
        let n = self.graph.order();
        let mut z = vec![1.0f64; n];
        for (k, &d) in self.destinations.iter().enumerate() {
            z[d] = self.terminal_z[k];
        }
        let decay = (-self.rho).exp();
        let mut residual = f64::INFINITY;
        let mut iterations = 0;

        for step in 0..max_iterations {
            let mut next = vec![0.0; n];
            for (i, slot) in next.iter_mut().enumerate() {
                if let Some(v) = self.terminal_value(i) {
                    // Row i of GP is the identity row: a destination is absorbing and its
                    // desirability is held at its boundary value for every iterate.
                    *slot = v;
                    continue;
                }
                let d = self.graph.degree(i);
                if d == 0 {
                    // An isolated vertex can never reach the exit. Its desirability is zero,
                    // its value infinite, and saying so is more useful than dividing by zero.
                    *slot = 0.0;
                    continue;
                }
                let mean: f64 =
                    self.graph.neighbours(i).iter().map(|&j| z[j]).sum::<f64>() / d as f64;
                *slot = decay * mean;
            }
            residual = next
                .iter()
                .zip(&z)
                .filter(|(a, b)| **a > 0.0 && **b > 0.0)
                .map(|(a, b)| (a.ln() - b.ln()).abs())
                .fold(0.0, f64::max);
            z = next;
            iterations = step + 1;
            // Information travels one hop per iteration, so a converged residual before the
            // graph's diameter has been crossed is a coincidence, not a solution.
            if residual < tolerance && step >= 2 {
                break;
            }
        }

        let reachable = self.graph.bfs_distances(&self.destinations);
        let underflowed = (0..n).any(|i| reachable[i].is_some() && z[i] == 0.0);
        Solution {
            z,
            iterations,
            residual,
            underflowed,
        }
    }

    /// The same fixed point, iterated in `log z` instead of `z`.
    ///
    /// **This removes the ceiling on `rho` that Todorov's §3 states as a limitation.** In
    /// `z`-space the far end of the graph holds `exp(-rho * d)`, so `f64` underflows once
    /// `rho * diameter` passes about 745 — and the accuracy of the estimate needs `rho`
    /// *large*, because the error per hop is about `ln(degree) / rho`. Those two demands
    /// collide: accuracy wants `rho > 1.4 * diameter` and representability wants
    /// `rho < 745 / diameter`, and past a diameter near 23 no `rho` satisfies both. On a
    /// maze of diameter 31 the best available `rho` returned 33.4 hops for a true 31.
    ///
    /// Iterating the value function directly, with a stable log-sum-exp, has no such floor:
    ///
    /// ```text
    /// v(i) = rho + ln(deg i) - log sum_j exp(-v(j))
    /// ```
    ///
    /// `rho` can then be pushed as far as the accuracy needs. The limitation was an artifact
    /// of the representation, not of the method — which is worth knowing, because the
    /// eigenproblem framing is the interesting part and it survives intact.
    pub fn value_log_space(&self, max_iterations: usize, tolerance: f64) -> Vec<f64> {
        let n = self.graph.order();
        let mut v = vec![0.0f64; n];
        for (k, &d) in self.destinations.iter().enumerate() {
            v[d] = if self.terminal_z[k] <= 0.0 {
                f64::INFINITY
            } else {
                -self.terminal_z[k].ln()
            };
        }

        for step in 0..max_iterations {
            let mut next = v.clone();
            let mut residual: f64 = 0.0;
            for (i, slot) in next.iter_mut().enumerate() {
                if self.is_destination(i) {
                    continue;
                }
                let d = self.graph.degree(i);
                if d == 0 {
                    *slot = f64::INFINITY;
                    continue;
                }
                // Stable log-sum-exp: factor out the smallest value so the largest exponent
                // is exactly zero and nothing underflows that would have mattered.
                let lo = self
                    .graph
                    .neighbours(i)
                    .iter()
                    .map(|&j| v[j])
                    .fold(f64::INFINITY, f64::min);
                if !lo.is_finite() {
                    *slot = f64::INFINITY;
                    continue;
                }
                let sum: f64 = self
                    .graph
                    .neighbours(i)
                    .iter()
                    .map(|&j| (-(v[j] - lo)).exp())
                    .sum();
                *slot = self.rho + (d as f64).ln() + lo - sum.ln();
                if slot.is_finite() && v[i].is_finite() {
                    residual = residual.max((*slot - v[i]).abs());
                }
            }
            v = next;
            if residual < tolerance && step >= 2 {
                break;
            }
        }
        v
    }

    /// Todorov's shortest-path estimate from the log-space solve, which is the one to use.
    pub fn shortest_paths_stable(&self, max_iterations: usize, tolerance: f64) -> Vec<f64> {
        self.value_log_space(max_iterations, tolerance)
            .iter()
            .map(|v| v / self.rho)
            .collect()
    }

    /// `v = -log z`. Infinite where the exit is unreachable.
    pub fn value(&self, z: &[f64]) -> Vec<f64> {
        z.iter()
            .map(|&x| if x <= 0.0 { f64::INFINITY } else { -x.ln() })
            .collect()
    }

    /// Todorov's shortest-path estimate, `v / rho`.
    pub fn shortest_paths(&self, z: &[f64]) -> Vec<f64> {
        self.value(z).iter().map(|v| v / self.rho).collect()
    }

    /// The optimal controlled transition probabilities out of vertex `i`:
    /// `u*(j|i) = p_ij z(j) / sum_k p_ik z(k)`.
    ///
    /// The passive dynamics `p` are uniform over neighbours, so this reweights a random walk
    /// by the desirability of where each step lands — which is the whole content of the
    /// optimal policy, and it needs no search.
    pub fn optimal_step(&self, z: &[f64], i: usize) -> Vec<(usize, f64)> {
        let neighbours = self.graph.neighbours(i);
        let total: f64 = neighbours.iter().map(|&j| z[j]).sum();
        if total <= 0.0 {
            let p = 1.0 / neighbours.len().max(1) as f64;
            return neighbours.iter().map(|&j| (j, p)).collect();
        }
        neighbours.iter().map(|&j| (j, z[j] / total)).collect()
    }
}

/// Compose two solved tasks.
///
/// Todorov's compositionality: because the equation in `z` is linear, a task whose terminal
/// desirability is `alpha z_A + beta z_B` at every destination is solved by exactly that
/// combination of the two solutions. **Optimal policies superpose — linearly, exactly, and
/// not as a metaphor**, which is a thirty-year-old result that reads very differently in a
/// project about superposition.
///
/// The combination is in *desirability*, not in value: `z = exp(-v)`, so adding desirabilities
/// is a log-sum-exp of the value functions and not an average of them. Averaging the values
/// instead is the obvious mistake and it does not solve anything.
pub fn compose(a: &[f64], b: &[f64], alpha: f64, beta: f64) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| alpha * x + beta * y).collect()
}

/// Sweep `rho` and report where the shortest-path approximation is usable.
///
/// Returns `(rho, max absolute error against BFS, underflowed)` per sample. The window has
/// both ends: too small and the control cost is not dominated by the state cost, too large
/// and `exp(-rho)` is zero in floating point.
pub fn shortest_path_window(
    graph: &Graph,
    destinations: &[usize],
    rhos: &[f64],
) -> Vec<(f64, f64, bool)> {
    let truth = graph.bfs_distances(destinations);
    rhos.iter()
        .map(|&rho| {
            let lmdp = Lmdp::new(graph, destinations.to_vec(), rho);
            let sol = lmdp.desirability(20_000, 1e-14);
            let est = lmdp.shortest_paths(&sol.z);
            let mut worst: f64 = 0.0;
            for (i, d) in truth.iter().enumerate() {
                if let Some(d) = d {
                    if est[i].is_finite() {
                        worst = worst.max((est[i] - *d as f64).abs());
                    } else {
                        worst = f64::INFINITY;
                    }
                }
            }
            (rho, worst, sol.underflowed)
        })
        .collect()
}
