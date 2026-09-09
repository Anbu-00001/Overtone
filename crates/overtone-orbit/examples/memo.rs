//! Q8: what the algebraic memo table is worth, measured.
//!
//! `cargo run --release -p overtone-orbit --example memo`

use std::time::Instant;

use overtone_orbit::language::Agent;
use overtone_orbit::memo::{fingerprint, Memo};
use overtone_orbit::search::choose_with_stats;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const KESTREL: &str = include_str!("../../../agents/kestrel.toml");
const PIKE: &str = include_str!("../../../agents/pike.toml");
const STOAT: &str = include_str!("../../../agents/stoat.toml");

fn main() {
    println!("Q8 -- the memo table for the algebraic layer\n");
    println!("A hand is a set, so the fingerprint has to be order-independent:");
    let a = overtone_lie::PauliString::single(0, overtone_sim::Pauli::X);
    let b = overtone_lie::PauliString::single(1, overtone_sim::Pauli::Z);
    println!(
        "  fingerprint([X0, Z1]) == fingerprint([Z1, X0])   {}",
        fingerprint(&[a, b]) == fingerprint(&[b, a])
    );
    println!(
        "  fingerprint([X0, Z1]) != fingerprint([X0])       {}",
        fingerprint(&[a, b]) != fingerprint(&[a])
    );
    println!(
        "  a repeated generator does not change the hand    {}",
        fingerprint(&[a, b]) == fingerprint(&[a, b, a])
    );

    println!("\n\nHits, misses, and what the table costs to keep\n");
    println!(
        "{:>10} {:>8} {:>9} {:>8} {:>8} {:>10} {:>12}",
        "agent", "budget", "lookups", "distinct", "hit rate", "ms/move", "n"
    );
    for (name, src) in [("kestrel", KESTREL), ("pike", PIKE), ("stoat", STOAT)] {
        for n in [4usize, 5] {
            for budget in [128usize, 512] {
                let spec = Agent::parse(src).unwrap();
                let mut spec = spec;
                spec.search.budget = budget;
                let mut rng = ChaCha8Rng::seed_from_u64(3);
                let game = overtone_orbit::ladder::opening(n, 24, 2, &mut rng);
                let t = Instant::now();
                let (_, st) = choose_with_stats(&spec, &game, &mut rng);
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                let lookups = st.memo_hits + st.memo_misses;
                println!(
                    "{name:>10} {budget:>8} {lookups:>9} {:>8} {:>7.1}% {ms:>10.1} {n:>12}",
                    st.memo_misses,
                    100.0 * st.memo_hits as f64 / lookups.max(1) as f64
                );
            }
        }
    }

    println!(
        "\nThe miss count is the number of *distinct hands* seen, not the number of lookups, so\n\
         the table is small and nothing is ever evicted. The split between the kinds is the\n\
         result, and it is structural rather than a tuning outcome.\n\
         \n\
         greedy and negamax evaluate the *siblings* of one position, and **the rotation angle\n\
         does not change the hand** -- eight candidates per move share one algebra. Two distinct\n\
         hands cover 160 lookups for a pawns-only agent. That is a 95 to 99 percent hit rate\n\
         bought by the shape of the move set.\n\
         \n\
         mcts gets **nothing**, and not because the table is wrong. MCTS evaluates only at\n\
         rollout leaves, and a rollout applies random generators down a path, so every leaf has\n\
         a hand no other leaf has: 1022 lookups, 1022 distinct. A rollout's whole purpose is to\n\
         reach positions nobody has seen, and a cache cannot help with that by construction.\n\
         \n\
         So Q8's table pays for the two searches Decisions-01 calls secondary and not for the\n\
         one it calls primary. Worth having -- pike at budget 512 makes 1486 lookups against 46\n\
         closures -- and worth saying plainly, because the obvious expectation is the opposite."
    );

    println!("\n\nWhat the ruling got right, and the one thing it did not\n");
    let mut memo = Memo::new();
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let game = overtone_orbit::ladder::opening(4, 24, 2, &mut rng);
    let hand = &game.players[0].hand;
    let cold = Instant::now();
    let first = memo.get(hand, 4);
    let cold = cold.elapsed().as_secs_f64() * 1e6;
    let warm = Instant::now();
    let second = memo.get(hand, 4);
    let warm = warm.elapsed().as_secs_f64() * 1e6;
    println!(
        "  closure + certificate, cold  {cold:>9.2} us   dim(g) = {}",
        first.algebra.dim()
    );
    println!(
        "  the same hand again, warm    {warm:>9.2} us   same object = {}",
        std::ptr::eq(
            std::sync::Arc::as_ptr(&first),
            std::sync::Arc::as_ptr(&second)
        )
    );
    println!(
        "\n  Q8 names three expensive amplitude-independent computations: the Lie closure, the\n\
           checkmate predicate, and region temperature. Checked against the code, only the\n\
           first is amplitude-independent. `is_checkmate` calls `is_check`, which is\n\
           `absorbed_weight > tolerance`, and then `certainly_unreachable(&self.state, ..)`;\n\
           `region_options` applies moves and reads weights. Memoising either on a discrete key\n\
           would return stale answers rather than slow ones, which is the worse failure.\n\
         \n\
           So the key is one field, not five. The same hand under a different safe set has the\n\
           identical algebra and the identical certificate, and putting the safe set in the key\n\
           would turn a hit into a miss for nothing. The ruling's rule holds -- key on the\n\
           discrete state, amplitudes are the value -- and applying it honestly leaves a\n\
           narrower key than the ruling drew."
    );
}
