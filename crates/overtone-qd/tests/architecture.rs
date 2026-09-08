//! M27: the verification Part V 10 insists on, as tests.

use overtone_qd::architecture::{
    algebraic, algebraic_reward, is_polynomial, selection_gain, spearman, verify, Candidate,
};
use overtone_qd::Genome;
use overtone_rl::SpectralControl;
use overtone_sim::random::rng;

/// The verification sweep, computed once for the whole test binary.
///
/// Five tests ask questions of the same 60 architectures, and training them is the expensive
/// part -- recomputing it per test would multiply the cost of `cargo test` by five to answer
/// the same question five times. The larger sweep lives in `examples/architecture.rs`, which
/// the gate runs in release and asserts numbers from.
fn sample() -> &'static [Candidate] {
    static SAMPLE: std::sync::OnceLock<Vec<Candidate>> = std::sync::OnceLock::new();
    SAMPLE.get_or_init(|| {
        let env = SpectralControl::new(3);
        let mut r = rng(11);
        verify(&env, 60, 120, &mut r)
    })
}

/// Spearman on known input, so the verification statistic is not itself the thing being
/// trusted.
#[test]
fn spearman_is_spearman() {
    assert!((spearman(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]) - 1.0).abs() < 1e-12);
    assert!((spearman(&[1.0, 2.0, 3.0], &[30.0, 20.0, 10.0]) + 1.0).abs() < 1e-12);
    // A monotone but nonlinear map still ranks perfectly, which is why rank was chosen.
    assert!((spearman(&[1.0, 2.0, 3.0], &[1.0, 4.0, 9.0]) - 1.0).abs() < 1e-12);
    // Ties take the average rank.
    let r = spearman(&[1.0, 1.0, 2.0], &[5.0, 5.0, 9.0]);
    assert!((r - 1.0).abs() < 1e-12, "{r}");
}

/// The algebra is read without training anything: the reported reach and gate count must
/// match what the policy actually is.
#[test]
fn the_algebraic_reading_matches_the_circuit() {
    let env = SpectralControl::new(3);
    let mut r = rng(5);
    for _ in 0..20 {
        let g = Genome::random(&mut r);
        let a = algebraic(&g, &env);
        let policy = g.policy();
        assert_eq!(a.params, policy.num_params());
        assert_eq!(
            a.gates,
            policy.ansatz.build(&[1.0]).gates().len(),
            "gate count disagrees with the built circuit"
        );
        assert_eq!(a.covers, a.reach >= env.k as f64 - 1e-9);
        assert_eq!(a.polynomial, is_polynomial(a.dim_g, g.qubits));
    }
}

/// The reach term predicts the trained return. This is the half of Part V 3's reward that
/// survives verification.
#[test]
fn the_reach_term_predicts_the_trained_return() {
    let candidates = sample();
    let trained: Vec<f64> = candidates.iter().map(|c| c.trained_return).collect();
    let reach: Vec<f64> = candidates.iter().map(|c| c.algebra.reach).collect();
    let rho = spearman(&reach, &trained);
    assert!(rho > 0.4, "reach predicted nothing: rho {rho}");

    let (top, all) = selection_gain(candidates, 20, |c| c.algebra.reach);
    assert!(
        top > all + 0.1,
        "selecting on reach did not beat the field: {top} against {all}"
    );
}

/// And the other half does not. Part V 3 subtracts the gate count; measured, more gates go
/// with a *better* return, because gates track layers and layers track the frequency
/// ceiling. Reported because the spec asked for the check, not because the answer was
/// wanted.
#[test]
fn the_gate_count_penalty_has_the_wrong_sign() {
    let candidates = sample();
    let trained: Vec<f64> = candidates.iter().map(|c| c.trained_return).collect();
    let penalty: Vec<f64> = candidates
        .iter()
        .map(|c| -(c.algebra.gates as f64))
        .collect();
    let rho = spearman(&penalty, &trained);
    assert!(
        rho < -0.2,
        "the gate penalty is not anti-correlated after all: rho {rho}"
    );
}

/// The `dim(g)` indicator carries no signal at these widths, which is a statement about
/// the environment rather than about Ragone et al.
#[test]
fn the_dim_g_indicator_carries_no_signal_here() {
    let candidates = sample();
    let trained: Vec<f64> = candidates.iter().map(|c| c.trained_return).collect();
    let poly: Vec<f64> = candidates
        .iter()
        .map(|c| f64::from(c.algebra.polynomial))
        .collect();
    let rho = spearman(&poly, &trained);
    assert!(
        rho.abs() < 0.2,
        "dim(g) suddenly predicts something: rho {rho}"
    );
}

/// The whole reward, as written, is beaten by its own reach term. This is the finding, and
/// it is stated at the strength the data supports and no higher.
///
/// A single sample once showed the combined reward selecting *worse* than random, and that
/// claim did not survive: across five seeds at two sample sizes it lands above random five
/// times and below it five times. What is stable is the comparison against reach alone --
/// reach wins on 9 of those 10 runs and beats random selection on all 10 -- so that is what
/// is asserted here.
#[test]
fn the_combined_reward_is_beaten_by_its_own_reach_term() {
    let candidates = sample();
    let gate_scale = candidates
        .iter()
        .map(|c| c.algebra.gates as f64)
        .fold(0.0, f64::max);
    let (combined, all) = selection_gain(candidates, 20, |c| {
        algebraic_reward(&c.algebra, 1.0, 1.0, 1.0, gate_scale)
    });
    let (reach_only, _) = selection_gain(candidates, 20, |c| c.algebra.reach);
    assert!(
        reach_only > combined,
        "reach alone did not beat the combined reward: {reach_only} against {combined}"
    );
    assert!(
        reach_only > all + 0.05,
        "reach alone did not beat random selection: {reach_only} against {all}"
    );
}

/// Training does the work: the untrained return must not predict the trained one, or the
/// whole sweep is measuring initial luck.
#[test]
fn training_is_what_produces_the_return() {
    let candidates = sample();
    let trained: Vec<f64> = candidates.iter().map(|c| c.trained_return).collect();
    let initial: Vec<f64> = candidates.iter().map(|c| c.initial_return).collect();
    let rho = spearman(&initial, &trained);
    assert!(rho.abs() < 0.3, "trained return tracks the draw: rho {rho}");
    let improved = candidates
        .iter()
        .filter(|c| c.trained_return > c.initial_return + 1e-6)
        .count();
    assert!(
        improved * 2 > candidates.len(),
        "only {improved} of {} candidates improved",
        candidates.len()
    );
}
