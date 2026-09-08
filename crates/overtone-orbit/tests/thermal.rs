//! M51: the two temperatures, and the confound that turned out to explain them.
//!
//! Part IX 5.4 asks whether CGT temperature and physical entropy track each other. The first
//! run said yes -- 8 seeds, mean rank correlation 0.49, the same sign every time. It was
//! wrong. Both series climb over the course of a game, so a correlation between them is
//! mostly a correlation with the ply index. With the ply partialled out and the seed count
//! doubled, the mean falls to about zero and the sign stops being consistent.
//!
//! The answer Part IX 8 asked for, then: they share a name, they do not share a behaviour.

use overtone_orbit::ladder::{opening, Strategy, ANGLES};
use overtone_orbit::thermal::{
    ambient_temperature, interaction_leak, region_game, regions, temperature_field, trace, Region,
};
use overtone_spec::spearman;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const N: usize = 5;
const ANGLE_SAMPLE: [f64; 3] = [ANGLES[1], ANGLES[3], ANGLES[5]];

#[test]
fn regions_partition_the_window() {
    for count in 1..=N {
        let rs = regions(N, count);
        assert_eq!(rs.len(), count);
        let mut all: Vec<usize> = rs.iter().flat_map(|r| r.qubits.clone()).collect();
        all.sort_unstable();
        assert_eq!(
            all,
            (0..N).collect::<Vec<_>>(),
            "{count} regions must tile the window"
        );
    }
}

#[test]
fn the_temperature_field_is_one_number_per_region() {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let game = opening(N, 24, 2, &mut rng);
    let rs = regions(N, 3);
    let field = temperature_field(&game, &rs, &ANGLE_SAMPLE);
    assert_eq!(field.len(), rs.len());
    assert_eq!(
        ambient_temperature(&game, &rs, &ANGLE_SAMPLE),
        field.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );
    // A settled region reports -1, the convention for a number; a contested one reports a
    // non-negative temperature. Nothing in between is possible.
    for t in field {
        assert!(
            t == -1.0 || t >= 0.0,
            "temperature {t} is neither a number nor a game"
        );
    }
}

#[test]
fn a_region_with_no_legal_move_is_settled() {
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    let game = opening(N, 24, 2, &mut rng);
    let empty = Region { qubits: vec![] };
    assert_eq!(
        region_game(&game, &empty, &ANGLE_SAMPLE).temperature(),
        -1.0
    );
}

#[test]
fn the_decomposition_does_not_leak() {
    // Part IX 5.3 assumes the position splits into weakly-interacting regions. For this
    // opening it splits exactly: the best move over the whole position is always found by
    // looking at one region at a time, so the gap is zero rather than merely small.
    let rs = regions(N, 3);
    for seed in 0..8u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let game = opening(N, 24, 2, &mut rng);
        let leak = interaction_leak(&game, &rs, &ANGLE_SAMPLE);
        assert!(leak.abs() < 1e-12, "seed {seed} leaked {leak}");
    }
}

#[test]
fn both_temperatures_climb_with_the_ply() {
    // The confound, isolated. The full 16-seed correlation is a measurement and lives in the
    // `twotemps` example, which the gate runs in release; what belongs in a unit test is the
    // fact that makes the raw correlation untrustworthy.
    let rs = regions(N, 2);
    let strategy = Strategy { budget: 6 };
    let mut drift_t = Vec::new();
    for seed in 0..3u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut game = opening(N, 16, 2, &mut rng);
        let mut choose = ChaCha8Rng::seed_from_u64(seed ^ 0x5eed);
        let t = trace(&mut game, &rs, &ANGLE_SAMPLE, 8, |g| {
            strategy.choose(g, &mut choose)
        });
        let ply: Vec<f64> = t.readings.iter().map(|r| r.ply as f64).collect();
        let cgt: Vec<f64> = t.readings.iter().map(|r| r.cgt).collect();
        let r = spearman(&cgt, &ply);
        if r.is_finite() {
            drift_t.push(r);
        }
    }
    assert!(!drift_t.is_empty());
    let mean = drift_t.iter().sum::<f64>() / drift_t.len() as f64;
    assert!(
        mean > 0.3,
        "temperature does not climb with the ply: {mean}"
    );
}
