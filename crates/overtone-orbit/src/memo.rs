//! Q8's memo table for the algebraic layer — and the one field of its key that survives.
//!
//! Decisions-01 Q8 rules that the table keys on the **discrete state only**, never on
//! amplitudes, and the reasoning is right: two move orders essentially never produce
//! bit-identical amplitude vectors, so an amplitude-keyed table would pay an invented tolerance
//! for a cache that never hits. It proposes
//!
//! ```text
//! (DLA fingerprint, safe-set spec, coherence steps remaining,
//!  substrate window origin, side to move)
//! ```
//!
//! on the grounds that the three expensive computations are the amplitude-independent ones:
//! the Lie closure, the checkmate predicate, and region temperature. **Two of those three are
//! not amplitude-independent in this implementation, and the key is wider than what is left.**
//!
//! # What actually keys, checked against the code
//!
//! ```text
//! closure(hand)                    f(hand)                       keys, and is the expensive one
//! OrbitCertificate::new(algebra)   f(algebra) = f(hand)          keys, and is expensive
//! Position::is_checkmate           f(certificate, STATE, safe)   does not key
//! Position::absorbed_weight        f(STATE, safe)                does not key
//! thermal::region_options          applies moves, reads weights  does not key
//! ```
//!
//! `is_checkmate` calls `is_check`, which is `absorbed_weight > tolerance`, and then
//! `certainly_unreachable(&self.state, …)`. Both take the amplitudes. Q8 calls the checkmate
//! predicate keying cleanly *"a load-bearing accident of the design and worth noting in the
//! docs"* — it is not one, and a table that memoised it on the discrete key would return stale
//! answers rather than slow ones, which is the worse failure.
//!
//! What is left is the algebraic layer, and Q8 named that correctly: this is a **memo table for
//! the algebraic layer**, not a transposition table. But it keys on **the DLA fingerprint
//! alone**. The other four fields would only fragment it: the same hand under a different safe
//! set has the identical algebra and the identical certificate, and including the safe set in
//! the key would turn a hit into a miss for no reason.
//!
//! The ruling's *rule* is what holds — key on the discrete state, amplitudes are the value —
//! and applying it honestly leaves a narrower key than the ruling drew.
//!
//! # Why it is worth having anyway
//!
//! `Player::algebra()` calls `closure(&hand, n, 4096)` on every call and `Game::position()`
//! calls it and then builds a certificate. Under MCTS that runs once per node and once per
//! rollout leaf, on a hand that changes at most once per ply — so the miss rate is bounded by
//! the number of *distinct hands* in a search rather than by the number of nodes.

use std::collections::HashMap;
use std::sync::Arc;

use overtone_lie::{closure, Algebra, PauliString};
use overtone_spec::digest::fnv1a;

use crate::invariant::OrbitCertificate;

/// The commutator-closure iteration cap, matching `Player::algebra`.
const CLOSURE_CAP: usize = 4096;

/// A stable fingerprint of a generator set, independent of the order it was collected in.
///
/// Order-independence is the whole point: a hand is a *set*, and two players who acquired the
/// same generators in different orders have the same algebra. Sorting before hashing is what
/// makes the table hit at all.
pub fn fingerprint(hand: &[PauliString]) -> u64 {
    let mut keys: Vec<(u128, u128)> = hand.iter().map(|p| (p.x, p.z)).collect();
    keys.sort_unstable();
    keys.dedup();
    let mut bytes = Vec::with_capacity(keys.len() * 32);
    for (x, z) in keys {
        bytes.extend_from_slice(&x.to_le_bytes());
        bytes.extend_from_slice(&z.to_le_bytes());
    }
    fnv1a(&bytes)
}

/// What a hand determines, once.
#[derive(Clone, Debug)]
pub struct Algebraic {
    pub algebra: Algebra,
    pub certificate: OrbitCertificate,
}

/// The memo table.
///
/// Deliberately not global and not behind a lock: it is threaded through the caller that owns
/// it, so a parallel Skill Trace gets one per worker and there is no shared mutable state to
/// make a measurement non-reproducible.
#[derive(Debug, Default)]
pub struct Memo {
    entries: HashMap<u64, Arc<Algebraic>>,
    hits: usize,
    misses: usize,
}

impl Memo {
    pub fn new() -> Memo {
        Memo::default()
    }

    /// The algebra and certificate for this hand, computing them only on a miss.
    pub fn get(&mut self, hand: &[PauliString], num_qubits: usize) -> Arc<Algebraic> {
        let key = fingerprint(hand);
        if let Some(hit) = self.entries.get(&key) {
            self.hits += 1;
            return Arc::clone(hit);
        }
        self.misses += 1;
        let algebra = closure(hand, num_qubits, CLOSURE_CAP);
        let certificate = OrbitCertificate::new(&algebra);
        let entry = Arc::new(Algebraic {
            algebra,
            certificate,
        });
        self.entries.insert(key, Arc::clone(&entry));
        entry
    }

    pub fn hits(&self) -> usize {
        self.hits
    }

    pub fn misses(&self) -> usize {
        self.misses
    }

    /// Distinct hands seen. The table's size is bounded by this, not by node count.
    pub fn distinct(&self) -> usize {
        self.entries.len()
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}
