//! Work units: what a search actually costs, counted rather than assumed.
//!
//! Decisions-03 Q5 adopts one definition:
//!
//! > **1 work unit = one complex-coefficient update in the active representation.**
//!
//! and one implementation rule, which is what makes it free:
//!
//! > Do not increment inside the innermost loop. Compute the update count analytically per
//! > gate from gate type and `n`.
//!
//! The obvious objection is that this makes the unit engine-dependent, so a g-sim agent gets
//! more gates per unit than a statevector agent. That is not an objection: g-sim genuinely
//! *is* cheaper, so that agent genuinely does get more search for the same real cost, exactly
//! as a chess engine with a faster evaluation gets more nodes per second. No universal
//! cross-agent currency exists; this one is the actual work, which is the strongest defence
//! available for an abstraction.
//!
//! # Every formula here is read off the code it models
//!
//! | operation | updates | where |
//! |---|---|---|
//! | `apply_exponential` | `2 * 2^n` | one pass for `P psi`, one for `cos psi + i sin (P psi)` |
//! | `collapse_at` | `2^n` | one write pass; the weight scan reads and does not update |
//! | `orbit_dimension` | `dim(g) * 2^n + (dim(g)+1)^2 * 2^n` | one `P psi` per generator, then Gram-Schmidt over `dim(g)+1` vectors of `2 * 2^n` reals |
//! | `half_chain_entropy` | `2^n + 2^(3n/2)` | forming `rho_A`, then diagonalising a `2^(n/2)` matrix |
//! | `dim_g`, `safe_set_size` | `0` | Pauli-string bitset algebra and an amplitude scan: no complex coefficient is *updated* |
//!
//! The last row is the one worth arguing about, and it is deliberate. The unit is defined on
//! updates in the active representation; the Lie closure is combinatorics over bitsets and the
//! safe-set filter reads amplitudes without writing any. Counting them here would be counting
//! a different thing, and `examples/agents.rs` reports wall-clock beside work units so the gap
//! between the two is visible rather than hidden.

use crate::game::Move;
use crate::language::{Agent, Feature, SearchKind};

/// Updates for one generator exponential on `n` qubits.
pub fn exponential(n: usize) -> u64 {
    2 * (1u64 << n)
}

/// Updates for one measurement collapse on `n` qubits.
pub fn collapse(n: usize) -> u64 {
    1u64 << n
}

/// Updates for applying one move.
pub fn of_move(mv: &Move, n: usize) -> u64 {
    match mv {
        Move::Apply { .. } => exponential(n),
        Move::Measure { .. } => collapse(n),
    }
}

/// Updates for one feature on one player, given that player's `dim(g)`.
pub fn of_feature(f: Feature, n: usize, dim_g: usize) -> u64 {
    let d = 1u64 << n;
    match f {
        Feature::DimG | Feature::SafeSetSize => 0,
        Feature::OrbitSize => {
            let g = dim_g as u64;
            g * d + (g + 1) * (g + 1) * d
        }
        Feature::HalfChainEntropy => d + (1u64 << (3 * n / 2)),
    }
}

/// Updates for one evaluation, which is both players' features.
pub fn of_eval(agent: &Agent, n: usize, dim_g: [usize; 2]) -> u64 {
    (0..2)
        .map(|p| {
            agent
                .eval
                .features
                .iter()
                .map(|f| of_feature(*f, n, dim_g[p]))
                .sum::<u64>()
        })
        .sum()
}

/// Analytic work for one move of the declared search.
///
/// `greedy` is exact. `negamax` and `mcts` are upper bounds at the declared budget: alpha-beta
/// cuts and terminal rollouts both stop early, so the realised figure is lower. The bound is
/// what the ladder plots, because it is the quantity a submitter can reason about before
/// spending it — and `examples/agents.rs` prints the realised wall clock beside it.
pub fn per_move(agent: &Agent, n: usize, dim_g: [usize; 2], horizon: usize) -> u64 {
    let candidates = agent.moves(n).len() as u64 * 8;
    let budget = agent.search.budget as u64;
    let one = exponential(n) + of_eval(agent, n, dim_g);
    match agent.search.kind {
        SearchKind::Greedy => budget.min(candidates) * one,
        SearchKind::Negamax => budget * one,
        // Each playout expands one node and rolls out to the horizon; the rollout applies
        // moves without evaluating, and the leaf is evaluated once.
        SearchKind::Mcts => budget * (one + horizon as u64 * exponential(n)),
    }
}
