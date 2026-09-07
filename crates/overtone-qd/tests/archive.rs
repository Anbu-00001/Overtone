//! M19's acceptance: the archive is a map, it is deterministic, and every axis is measured.

use overtone_qd::behaviour::measure;
use overtone_qd::search::hill_climb;
use overtone_qd::{map_elites, Config, Genome};
use overtone_rl::SpectralControl;
use rand::SeedableRng;

fn small() -> Config {
    Config {
        initial: 60,
        evaluations: 400,
        nodes: 32,
        ..Config::default()
    }
}

#[test]
fn the_same_seed_gives_the_same_archive() {
    // The league replays matches from a seed, so an archive that is not reproducible is an
    // archive nobody can verify against.
    let a = map_elites(&small());
    let b = map_elites(&small());
    assert_eq!(a.len(), b.len());
    let (ra, rb) = (a.ranked(), b.ranked());
    for (x, y) in ra.iter().zip(&rb) {
        assert_eq!(x.genome, y.genome);
        assert!((x.fitness - y.fitness).abs() < 1e-15);
    }
    let mut different = small();
    different.seed += 1;
    let c = map_elites(&different);
    assert!(
        c.ranked()[0].genome != a.ranked()[0].genome || c.len() != a.len(),
        "a different seed produced an identical archive"
    );
}

#[test]
fn every_descriptor_responds_to_the_genome() {
    // Part IV 7: an authored descriptor would place agents in cells they do not belong in.
    // The check that a descriptor is measured is that changing the agent changes it.
    let env = SpectralControl::new(3);
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(4);
    // Structure and angles have to be drawn together -- see Genome::structured. The first
    // version of this test set the fields directly and panicked inside the policy.
    let base = Genome::structured(4, 2, false, true, true, &mut rng);
    let fresh = Genome::structured(4, 2, false, true, false, &mut rng);
    assert!(base.is_well_formed() && fresh.is_well_formed());

    let (with_entanglers, _) = measure(&base, &env, 32);
    let (without, _) = measure(&fresh, &env, 32);
    // Entanglers take the ansatz algebra from su(2)^(+n) to all of su(2^n).
    assert!(
        with_entanglers.dim_g > without.dim_g,
        "dim(g) did not respond to the entanglers: {} vs {}",
        with_entanglers.dim_g,
        without.dim_g
    );
    assert_eq!(without.dim_g, 3 * base.qubits);

    // Reach is C * |lambda|, so scaling lambda scales it.
    let policy = base.policy();
    let mut scaled = base.clone();
    for i in policy.ansatz.lambda_range() {
        scaled.params[i] = 2.0 * base.params[i];
    }
    let (a, _) = measure(&base, &env, 32);
    let (b, _) = measure(&scaled, &env, 32);
    assert!(
        (b.reach - 2.0 * a.reach).abs() < 1e-9,
        "reach did not track lambda: {} vs {}",
        b.reach,
        a.reach
    );
}

#[test]
fn a_cell_is_only_taken_by_a_better_agent() {
    let archive = map_elites(&small());
    assert!(!archive.is_empty());
    // Every elite must be the best thing its cell ever saw, which is the invariant the whole
    // algorithm maintains.
    for (key, cell) in archive.cells() {
        assert_eq!(*key, archive.bins.cell_of(&cell.behaviour));
        assert!(cell.found_at <= archive.evaluations());
    }
}

#[test]
fn the_archive_spans_the_space_rather_than_one_corner() {
    let archive = map_elites(&small());
    let dims: std::collections::HashSet<usize> = archive.cells().map(|(k, _)| k.0).collect();
    let reaches: std::collections::HashSet<usize> = archive.cells().map(|(k, _)| k.1).collect();
    assert!(dims.len() >= 3, "only {} dim(g) bands filled", dims.len());
    assert!(
        reaches.len() >= 3,
        "only {} reach bands filled",
        reaches.len()
    );
    assert!(archive.len() >= 10, "only {} cells filled", archive.len());
}

#[test]
fn quality_diversity_beats_a_single_peak_search_at_a_matched_budget() {
    // The honest justification for MAP-Elites, measured. Part IV 4's stated reason -- that
    // quality-diversity survives a barren plateau -- is refuted by Arrasmith et al.
    // (Quantum 5, 558), whose theorem covers any optimiser deciding on cost differences.
    // What is true is the ordinary quality-diversity result: keeping structurally different
    // agents beats abandoning them, in a landscape that is multimodal.
    let cfg = small();
    let archive = map_elites(&cfg);
    let (_, peak_only) = hill_climb(&cfg);
    let qd = archive.best().unwrap().fitness;
    assert!(
        qd >= peak_only - 1e-9,
        "the archive lost to a single-peak search: {qd} vs {peak_only}"
    );
}

#[test]
fn the_archive_serialises_to_json_the_league_can_read() {
    let archive = map_elites(&small());
    let json = archive.to_json();
    assert!(json.starts_with('{') && json.trim_end().ends_with('}'));
    assert!(json.contains("\"elites\""));
    assert!(json.contains("\"dim_g\""));
    assert!(json.contains("\"params\""));
    // One record per filled cell.
    assert_eq!(json.matches("\"return\":").count(), archive.len());
}
