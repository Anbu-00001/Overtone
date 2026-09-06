//! The Phase 3 gate (Part I 1.4, 6.5, 9).
//!
//! Two claims, one of which had to be sharpened before it could be tested.
//!
//! **RAW-PQC is strictly band-limited.** No spectral energy above the encoding ceiling.
//! Part I 9 asks for `1e-9`; measured, it is at the rounding floor of the transform.
//!
//! **SOFTMAX-PQC leaks past its own circuit's ceiling.** Part I 1.4 states this as "odd
//! harmonics at `3L`, `5L`". Verified numerically (`examples/verify_leakage.rs`), the
//! precise statement is: **odd harmonics of the policy's dominant in-band frequency**, which
//! equals `L` only when the trained policy happens to concentrate there. On
//! `SpectralControl-k` a trained policy concentrates at `k`, so the leakage appears at
//! `3k, 5k, 7k, ...` — and at `L = k` that coincides with `3L, 5L`, which is why the
//! original phrasing looked right.
//!
//! The distinction matters. At `L = 2` against `k = 3` the policy cannot concentrate at 3,
//! its in-band content is spread over frequencies 1 and 2, and the leakage is correspondingly
//! broadband rather than a clean harmonic ladder.

use overtone_rl::policy::Policy;
use overtone_rl::reinforce::{initial_params, train, TrainConfig};
use overtone_rl::{Scaling, SpectralControl, SpectralControlAnsatz};
use overtone_sim::random::rng;
use overtone_spec::spectrum::Spectrum;
use overtone_spec::{half_chain_entropy, spectrum_of};

const N: usize = 512;

fn trained(layers: usize, softmax: bool, entangle: bool, seed: u64) -> (Policy, Vec<f64>, usize) {
    let env = SpectralControl::new(3);
    let ansatz = SpectralControlAnsatz::build(2, layers, Scaling::Pinned, entangle);
    let ceiling = ansatz.frequency_ceiling(0);
    let policy = if softmax {
        Policy::softmax(ansatz, 1.0)
    } else {
        Policy::raw(ansatz)
    };
    let mut r = rng(seed);
    let p0 = initial_params(&policy, 0.3, 1.0, &mut r);
    let cfg = TrainConfig {
        episodes: 5000,
        batch_size: 50,
        learning_rate: 0.05,
        ..Default::default()
    };
    let out = train(&policy, &p0, &env, &cfg, &mut r);
    (policy, out.params, ceiling)
}

fn policy_spectrum(policy: &Policy, params: &[f64], ceiling: usize) -> Spectrum {
    spectrum_of(N, ceiling, |s| policy.prob_action1(params, &[s]))
}

#[test]
fn raw_pqc_has_no_energy_beyond_its_ceiling() {
    // Part I 9, at 1e-9. Holds for every layer count and every seed, trained or not,
    // because it is a statement about the function class rather than about a fixed point.
    for layers in 1..=5usize {
        for seed in 0..5u64 {
            let (policy, params, ceiling) = trained(layers, false, true, seed);
            let s = policy_spectrum(&policy, &params, ceiling);
            assert!(
                s.energy_above_ceiling() < 1e-9,
                "L={layers} seed={seed}: leaked {:e} above ceiling {ceiling}",
                s.energy_above_ceiling()
            );
            // And the in-band content is real, so this is not vacuously true.
            assert!(
                s.total_in_band() > 1e-3,
                "L={layers} seed={seed}: spectrum is empty in band"
            );
        }
    }
}

#[test]
fn raw_pqc_is_band_limited_before_training_too() {
    for layers in 1..=4usize {
        let ansatz = SpectralControlAnsatz::build(3, layers, Scaling::Pinned, true);
        let ceiling = ansatz.frequency_ceiling(0);
        let policy = Policy::raw(ansatz);
        for seed in 0..10u64 {
            let mut r = rng(seed);
            let params = initial_params(&policy, 3.0, 1.0, &mut r);
            let s = policy_spectrum(&policy, &params, ceiling);
            assert!(
                s.energy_above_ceiling() < 1e-9,
                "L={layers} seed={seed}: leaked {:e}",
                s.energy_above_ceiling()
            );
        }
    }
}

#[test]
fn softmax_pqc_leaks_above_its_ceiling() {
    for layers in [2usize, 3, 4] {
        let (policy, params, ceiling) = trained(layers, true, true, 7);
        let s = policy_spectrum(&policy, &params, ceiling);
        assert!(
            s.energy_above_ceiling() > 1e-3,
            "L={layers}: softmax should leak, found only {:e}",
            s.energy_above_ceiling()
        );
    }
}

#[test]
fn softmax_leakage_sits_on_odd_harmonics_of_the_dominant_tone() {
    // The sharpened claim. At L = k = 3 the trained policy concentrates in-band at
    // frequency 3, and the leakage appears at 9, 15, 21, ... with even multiples of 3
    // suppressed by orders of magnitude. That asymmetry is the signature of an odd
    // nonlinearity, and it is what distinguishes "the softmax leaked" from "the numerics
    // are noisy".
    let (policy, params, ceiling) = trained(3, true, true, 7);
    assert_eq!(ceiling, 3);
    let s = policy_spectrum(&policy, &params, ceiling);

    // The dominant in-band frequency should be the environment's.
    let dominant = (1..=ceiling)
        .max_by(|a, b| s.at(*a).partial_cmp(&s.at(*b)).unwrap())
        .unwrap();
    assert_eq!(dominant, 3, "expected the policy to concentrate at k = 3");

    let odd: Vec<f64> = [3usize, 5, 7].iter().map(|m| s.at(m * dominant)).collect();
    let even: Vec<f64> = [2usize, 4, 6].iter().map(|m| s.at(m * dominant)).collect();

    assert!(odd[0] > 1e-2, "third harmonic too weak: {}", odd[0]);
    assert!(odd[1] > 1e-3, "fifth harmonic too weak: {}", odd[1]);
    assert!(odd[2] > 1e-4, "seventh harmonic too weak: {}", odd[2]);

    // Odd harmonics dominate even ones by orders of magnitude.
    for (i, e) in even.iter().enumerate() {
        assert!(
            odd[i] > 20.0 * e,
            "harmonic {} ({}) should dominate even neighbour ({e})",
            2 * i + 3,
            odd[i]
        );
    }

    // And the ladder decays, as a Taylor series in the nonlinearity should.
    assert!(
        odd[0] > odd[1] && odd[1] > odd[2],
        "ladder not decaying: {odd:?}"
    );
}

#[test]
fn softmax_leakage_grows_with_the_inverse_temperature() {
    // beta controls how hard the softmax bends the band-limited input. More bend, more
    // harmonic content. This ties the leakage to its mechanism rather than to a
    // coincidence of one trained parameter set.
    let env = SpectralControl::new(3);
    let ansatz = SpectralControlAnsatz::build(2, 3, Scaling::Pinned, true);
    let ceiling = ansatz.frequency_ceiling(0);

    let mut previous = 0.0;
    for beta in [0.25_f64, 1.0, 4.0] {
        let policy = Policy::softmax(ansatz.clone(), beta);
        let mut r = rng(7);
        let p0 = initial_params(&policy, 0.3, 1.0, &mut r);
        let out = train(
            &policy,
            &p0,
            &env,
            &TrainConfig {
                episodes: 4000,
                batch_size: 50,
                learning_rate: 0.05,
                ..Default::default()
            },
            &mut r,
        );
        let s = policy_spectrum(&policy, &out.params, ceiling);
        let leak = s.leakage_ratio();
        assert!(
            leak > previous,
            "beta={beta}: leakage ratio {leak} did not exceed {previous}"
        );
        previous = leak;
    }
}

#[test]
fn the_two_policies_are_separated_by_one_number() {
    // The panel needs a single readout that distinguishes them. Leakage ratio is it.
    let (raw, raw_p, raw_c) = trained(3, false, true, 7);
    let (sm, sm_p, sm_c) = trained(3, true, true, 7);

    let raw_leak = policy_spectrum(&raw, &raw_p, raw_c).leakage_ratio();
    let sm_leak = policy_spectrum(&sm, &sm_p, sm_c).leakage_ratio();

    assert!(
        raw_leak < 1e-12,
        "raw leakage {raw_leak:e} should be at rounding"
    );
    assert!(
        sm_leak > 0.05,
        "softmax leakage {sm_leak} should be visible"
    );
}

#[test]
fn the_entanglement_ablation_removes_entanglement() {
    // Part I 6.6: the --no-entangle ablation must actually ablate. Without the CZ ring the
    // circuit is a product of single-qubit rotations, so the half-chain entropy is zero to
    // rounding; with it, the state is genuinely entangled.
    let env = SpectralControl::new(3);
    for entangle in [false, true] {
        let ansatz = SpectralControlAnsatz::build(4, 3, Scaling::Pinned, entangle);
        let policy = Policy::raw(ansatz);
        let mut r = rng(5);
        let p0 = initial_params(&policy, 0.7, 1.0, &mut r);
        let out = train(&policy, &p0, &env, &TrainConfig::default(), &mut r);

        let circuit = policy.ansatz.build(&[0.9]);
        let state = circuit.run(&out.params[..policy.ansatz.num_params()]);
        let entropy = half_chain_entropy(&state);

        if entangle {
            assert!(entropy > 1e-3, "entangled circuit gave entropy {entropy}");
        } else {
            assert!(entropy < 1e-12, "ablated circuit gave entropy {entropy}");
        }
    }
}

#[test]
fn the_frequency_k_magnitude_equals_the_expected_return() {
    // A cross-check between two independent instruments. The return is computed by
    // quadrature in overtone-rl; the spectrum by FFT in overtone-spec. They share no code.
    //
    // J = E_s[cos(ks) (2 pi(1|s) - 1)] extracts the frequency-k component of pi(1|s),
    // doubled, and the expectation halves it again. Writing that component as
    // a_k cos(ks + phi) gives
    //
    //     J = |c_k| cos(phi),      hence      |J| <= |c_k|,
    //
    // with equality exactly when the policy is in phase with the reward. So the identity is
    // an inequality, and it is saturated only by a policy that has aligned its phase --
    // which is what maximising the return does. Both halves are worth asserting: the bound
    // is a hard consequence of the two instruments agreeing, and the near-saturation says
    // the agent found the phase.
    //
    // A factor-of-two or normalisation error in either instrument would break the bound.
    for (layers, softmax) in [(3usize, false), (4, false), (3, true), (4, true)] {
        let env = SpectralControl::new(3);
        let (policy, params, ceiling) = trained(layers, softmax, true, 7);
        let j = env.expected_return(4096, |s| policy.prob_action1(&params, &[s]));
        let s = policy_spectrum(&policy, &params, ceiling);

        assert!(
            j.abs() <= s.at(3) + 1e-9,
            "L={layers} softmax={softmax}: |J| = {} exceeds |c_3| = {}",
            j.abs(),
            s.at(3)
        );
        // Near-saturation, stated as the phase error it actually measures rather than as
        // an absolute difference. Trained policies here land within about 0.05 radians of
        // perfect alignment; L = 4 is the loosest, at 0.044.
        let cos_phi = j.abs() / s.at(3);
        let phase_error = cos_phi.clamp(-1.0, 1.0).acos();
        assert!(
            phase_error < 0.08,
            "L={layers} softmax={softmax}: phase error {phase_error:.4} rad \
             (|c_3| = {}, |J| = {})",
            s.at(3),
            j.abs()
        );
    }
}

#[test]
fn a_policy_below_the_ceiling_has_no_frequency_k_component_at_all() {
    // The Phase 2 zero-return result, seen from the spectral side. The return is zero
    // because the coefficient it extracts does not exist, and here that absence is visible
    // directly rather than inferred from a number that happens to be zero.
    let (policy, params, ceiling) = trained(2, false, true, 7);
    assert_eq!(ceiling, 2);
    let s = policy_spectrum(&policy, &params, ceiling);
    assert!(
        s.at(3) < 1e-12,
        "frequency 3 should be unreachable, found {:e}",
        s.at(3)
    );
}
