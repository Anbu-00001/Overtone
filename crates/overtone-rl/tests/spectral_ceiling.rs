//! The Phase 2 gate (Part I 7.1, 9).
//!
//! The claim the repo is built on, stated so it can fail:
//!
//! > With `lambda` pinned and a frequency ceiling below `k`, a RAW-PQC agent on
//! > `SpectralControl-k` scores exactly zero. Not "worse". Zero, for any parameters,
//! > forever.
//!
//! Returns here are measured by quadrature, not by averaging episodes. Part I 9 asks for
//! `|J| < 0.02` after 5000 episodes; the *sampled* mean over 5000 episodes has a standard
//! error near 0.014, so asserting 0.02 on it would be a 1.4-sigma test that flakes roughly
//! one run in six. The exact return is the quantity the claim is actually about, and it
//! comes out at rounding.

use overtone_rl::ceiling::{self, DEFAULT_GRID};
use overtone_rl::policy::{Policy, PolicyKind};
use overtone_rl::reinforce::{coarse_tune_lambda, initial_params, train, TrainConfig};
use overtone_rl::{Scaling, SpectralControl, SpectralControlAnsatz};
use overtone_sim::random::rng;

fn raw_policy(qubits: usize, layers: usize, scaling: Scaling) -> Policy {
    Policy::raw(SpectralControlAnsatz::build(qubits, layers, scaling, true))
}

#[test]
fn the_frequency_ceiling_counts_encoding_gates_not_layers() {
    // Part I 1.1 states the ceiling as L. That holds only when each layer encodes once.
    let one_site = SpectralControlAnsatz::build(4, 3, Scaling::Pinned, true);
    assert_eq!(one_site.frequency_ceiling(0), 3);

    // Encode the same scalar on every qubit and the ceiling is n*L, which would silently
    // break the zero-return test below.
    use overtone_rl::{Ansatz, AnsatzConfig, EncodingSite};
    let all_sites = Ansatz::new(AnsatzConfig {
        num_qubits: 4,
        layers: 3,
        encoding: (0..4)
            .map(|q| EncodingSite {
                qubit: q,
                component: 0,
            })
            .collect(),
        scaling: Scaling::Pinned,
        entangle: true,
    });
    assert_eq!(all_sites.frequency_ceiling(0), 12);
}

#[test]
fn untrained_raw_pqc_below_the_ceiling_scores_exactly_zero() {
    // No training involved. The claim is about the function class, so it must hold for
    // every parameter setting, not merely at a converged one.
    let k = 3;
    for layers in 1..k {
        let policy = raw_policy(2, layers, Scaling::Pinned);
        for seed in 0..25u64 {
            let mut r = rng(seed);
            let params = initial_params(&policy, 3.0, 1.0, &mut r);
            let j = policy.expected_return(&params, k, 1024);
            assert!(
                j.abs() < 1e-12,
                "L={layers} seed={seed}: J = {j:e}, expected exactly zero"
            );
        }
    }
}

#[test]
fn trained_raw_pqc_below_the_ceiling_still_scores_zero() {
    // Part I 9's headline CI test: L = 2, k = 3, 5000 episodes, seeded.
    let k = 3;
    let env = SpectralControl::new(k);
    let policy = raw_policy(2, 2, Scaling::Pinned);
    assert_eq!(
        policy.ansatz.frequency_ceiling(0),
        2,
        "ceiling must sit below k"
    );

    let mut r = rng(11);
    let params = initial_params(&policy, 0.3, 1.0, &mut r);
    let cfg = TrainConfig {
        episodes: 5000,
        batch_size: 50,
        learning_rate: 0.05,
        ..Default::default()
    };
    let out = train(&policy, &params, &env, &cfg, &mut r);

    assert!(
        out.final_exact_return.abs() < 0.02,
        "5000 episodes at L=2, k=3 gave J = {}",
        out.final_exact_return
    );
    // The theorem says zero, so hold it to zero rather than to the paper's threshold.
    assert!(
        out.final_exact_return.abs() < 1e-12,
        "J should be exactly zero, got {:e}",
        out.final_exact_return
    );
}

#[test]
fn training_never_escapes_the_ceiling_at_any_point_in_the_run() {
    // A converged zero could hide a run that scored and then lost it. Check the trace.
    let env = SpectralControl::new(3);
    let policy = raw_policy(3, 2, Scaling::Pinned);
    let mut r = rng(5);
    let params = initial_params(&policy, 0.5, 1.0, &mut r);
    let out = train(&policy, &params, &env, &TrainConfig::default(), &mut r);

    for row in &out.trace {
        assert!(
            row.exact_return.abs() < 1e-12,
            "step {}: J = {:e}",
            row.step,
            row.exact_return
        );
    }
}

#[test]
fn trained_raw_pqc_lands_on_the_lp_ceiling_staircase() {
    // Above the environment frequency the agent should reach the step, not merely improve.
    let k = 3;
    let env = SpectralControl::new(k);

    for layers in [3usize, 4, 6] {
        let policy = raw_policy(2, layers, Scaling::Pinned);
        let c = policy.ansatz.frequency_ceiling(0);
        let cap = ceiling::ceiling(c, k, DEFAULT_GRID);

        let mut r = rng(7);
        let params = initial_params(&policy, 0.3, 1.0, &mut r);
        let cfg = TrainConfig {
            episodes: 5000,
            batch_size: 50,
            learning_rate: 0.05,
            ..Default::default()
        };
        let out = train(&policy, &params, &env, &cfg, &mut r);

        // Never above the ceiling: that would falsify the band-limit argument.
        assert!(
            out.final_exact_return <= cap + 1e-9,
            "L={layers}: J = {} exceeds the ceiling {cap}",
            out.final_exact_return
        );
        // And close enough to it that "agents land on the steps" is a fair description.
        assert!(
            out.final_exact_return > cap - 0.05,
            "L={layers}: J = {} fell short of the ceiling {cap}",
            out.final_exact_return
        );
    }
}

#[test]
fn trainable_lambda_dissolves_the_ceiling() {
    // L = 1, so the pinned ceiling is 1 < k = 3 and the pinned agent scores exactly zero.
    // Trainable lambda makes the reachable set lambda * {-1, 0, 1}, which is continuous, so
    // the same circuit can reach k. Coarse tune first: the return is a resonance curve in
    // lambda, and gradient ascent from outside the capture range locks onto a sidelobe.
    let k = 3;
    let env = SpectralControl::new(k);

    let pinned = raw_policy(2, 1, Scaling::Pinned);
    let mut r = rng(3);
    let pinned_params = initial_params(&pinned, 0.5, 1.0, &mut r);
    let pinned_out = train(
        &pinned,
        &pinned_params,
        &env,
        &TrainConfig::default(),
        &mut r,
    );
    assert!(
        pinned_out.final_exact_return.abs() < 1e-12,
        "pinned L=1 should score zero, got {}",
        pinned_out.final_exact_return
    );

    let trainable = raw_policy(2, 1, Scaling::Trainable);
    let mut r = rng(3);
    let mut params = initial_params(&trainable, 0.5, 1.0, &mut r);
    let tuned = coarse_tune_lambda(&trainable, &params, k, 8.0, 200);
    for idx in trainable.ansatz.lambda_range() {
        params[idx] = tuned;
    }
    let cfg = TrainConfig {
        episodes: 20000,
        batch_size: 50,
        learning_rate: 0.05,
        ..Default::default()
    };
    let out = train(&trainable, &params, &env, &cfg, &mut r);

    assert!(
        out.final_exact_return > 0.4,
        "trainable lambda at L=1 reached only J = {} (lambda started at {tuned})",
        out.final_exact_return
    );
    let lambda: f64 = trainable
        .ansatz
        .lambda_range()
        .map(|i| out.params[i])
        .next()
        .unwrap();
    assert!(
        (lambda.abs() - k as f64).abs() < 0.5,
        "lambda locked at {lambda}, expected near {k}"
    );
}

#[test]
fn softmax_pqc_leaks_past_the_band_limited_ceiling() {
    // Part I 1.4, measured as return rather than as spectrum. The softmax policy is a
    // sigmoid of a band-limited function, so it is not itself band-limited and the LP
    // staircase -- which bounds strictly band-limited policies -- does not bound it.
    //
    // The sharpest case is L = 2 against k = 3: a RAW-PQC scores exactly zero there,
    // because frequency 3 is outside its reach, while the softmax generates a frequency-3
    // harmonic from its frequency-1 and 2 content and scores on it.
    let k = 3;
    let env = SpectralControl::new(k);
    let ansatz = SpectralControlAnsatz::build(2, 2, Scaling::Pinned, true);
    assert_eq!(ansatz.frequency_ceiling(0), 2);
    let policy = Policy::softmax(ansatz, 1.0);
    assert_eq!(policy.kind, PolicyKind::Softmax);

    let mut r = rng(7);
    let params = initial_params(&policy, 0.3, 1.0, &mut r);
    let cfg = TrainConfig {
        episodes: 5000,
        batch_size: 50,
        learning_rate: 0.05,
        ..Default::default()
    };
    let out = train(&policy, &params, &env, &cfg, &mut r);

    let raw_cap = ceiling::ceiling(2, k, DEFAULT_GRID);
    assert_eq!(raw_cap, 0.0, "the band-limited ceiling at C=2, k=3 is zero");
    assert!(
        out.final_exact_return > 0.05,
        "softmax should leak past a zero ceiling, got J = {}",
        out.final_exact_return
    );
    // Still bounded by the unconstrained optimum: leakage is not magic.
    assert!(
        out.final_exact_return < env.optimal_return() + 1e-9,
        "J = {} exceeds 2/pi",
        out.final_exact_return
    );
}
