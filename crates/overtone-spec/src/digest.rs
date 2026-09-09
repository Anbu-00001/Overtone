//! FNV-1a, in one place, because two crates need the same bytes to hash the same.
//!
//! `.otn` uses it to make a replay a proof (Part VIII 1) and the agent language uses it to
//! give a submitted spec a stable name (Part VIII 2). Those are different problems with the
//! same requirement: **the hash has to outlive a toolchain.**
//! `std::collections::hash_map::DefaultHasher` is documented as *not* guaranteed stable across
//! Rust releases, so a format or a league that verified through it would silently stop
//! verifying after an upgrade.
//!
//! It lives here rather than in `overtone-otn` because `overtone-otn` depends on
//! `overtone-orbit`, so the agent language cannot reach it without a dependency cycle. Copying
//! eight lines would have worked and would have been two implementations of one guarantee.

/// FNV-1a, 64-bit.
///
/// The constants are the published ones: offset basis `0xcbf29ce484222325`, prime
/// `0x100000001b3`. `fnv1a(b"overtone")` is `0x8e1b756d79746b0b`, which is checked against an
/// independent implementation in `overtone-otn/tests/roundtrip.rs` rather than against this
/// one — a self-consistent hash proves nothing.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}
