//! The self-verifying half of the format.
//!
//! Part VIII 1 asks for a hash of the final state so that "re-running is a proof". Two
//! decisions make that actually true rather than nearly true; both are argued in the crate
//! docs and both are load-bearing.

use overtone_sim::StateVec;

/// Amplitudes are rounded to this grid before hashing.
///
/// The number is not a taste. `scripts/wasm_determinism.sh` measures native-versus-wasm
/// agreement at a worst relative difference of about `5.6e-16`, and asserts a tolerance of
/// `1e-13`. A hash over raw bits would therefore differ between a native replay and a browser
/// replay of the same game. `1e-9` sits four orders above the asserted tolerance, so no
/// platform difference can cross a quantum, and far below any amplitude a player could act
/// on. Changing it changes every hash ever written, so it is pinned with the format version.
pub const HASH_QUANTUM: f64 = 1e-9;

/// FNV-1a, 64-bit, written out because the format has to outlive a toolchain.
///
/// `std::collections::hash_map::DefaultHasher` is documented as not guaranteed stable across
/// Rust releases. A format whose verification depends on it would silently stop verifying
/// after an upgrade, which is the opposite of what Part VIII 1 asks for.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Quantise one amplitude component onto the hash grid.
fn quantise(x: f64) -> i64 {
    // Round half away from zero so that -0.0 and +0.0 agree, and so the grid is symmetric.
    (x / HASH_QUANTUM).round() as i64
}

/// A stable hash of a state vector, invariant to differences below [`HASH_QUANTUM`].
pub fn state_hash(state: &StateVec) -> u64 {
    let mut bytes = Vec::with_capacity(state.dim() * 16);
    for i in 0..state.dim() {
        let a = state.amp(i);
        bytes.extend_from_slice(&quantise(a.re).to_le_bytes());
        bytes.extend_from_slice(&quantise(a.im).to_le_bytes());
    }
    fnv1a(&bytes)
}

/// The hash of a whole position: both players' states, in seat order.
pub fn position_hash(states: &[&StateVec]) -> u64 {
    let mut bytes = Vec::new();
    for s in states {
        bytes.extend_from_slice(&state_hash(s).to_le_bytes());
    }
    fnv1a(&bytes)
}
