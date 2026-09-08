//! Every claim this crate makes about the plateau, as a test that fails when it stops
//! being true.

use overtone_opt::optim::{Differentiable, Objective, Optimiser};
use overtone_opt::race::{fit_shot_scaling, lane, shots_to_progress, RaceConfig};
use overtone_opt::shots::{Budget, DiagonalTerm, EXACT_SAMPLING_LIMIT};
use overtone_opt::PlateauCost;
use overtone_sim::{Observable, Pauli, PauliTerm};
use overtone_spec::plateau::{hardware_efficient, CostLocality, DepthPolicy};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// The shot estimator is the binomial one, not an approximation of it: unbiased, with the
/// variance the module documents.
#[test]
fn shot_estimator_is_unbiased_with_the_stated_variance() {
    let circuit = hardware_efficient(4, 4);
    let x = overtone_sim::random::random_params(9, circuit.num_params());
    let psi = circuit.run(&x);
    let term = DiagonalTerm::from_observable(&Observable::z_global(4)).unwrap();
    let truth = term.exact(&psi);
    let shots = 400;

    let mut rng = ChaCha8Rng::seed_from_u64(4);
    let draws: Vec<f64> = (0..4000)
        .map(|_| term.sample(&psi, shots, &mut rng))
        .collect();
    let mean = draws.iter().sum::<f64>() / draws.len() as f64;
    let var = draws.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / draws.len() as f64;

    assert!(
        (mean - truth).abs() < 4.0 * term.standard_error(&psi, shots) / (draws.len() as f64).sqrt(),
        "biased: sample mean {mean}, truth {truth}"
    );
    let predicted = term.standard_error(&psi, shots).powi(2);
    assert!(
        (var / predicted - 1.0).abs() < 0.1,
        "variance {var} against predicted {predicted}"
    );
}

/// The Gaussian branch above `EXACT_SAMPLING_LIMIT` must match the Bernoulli branch in mean
/// and variance, or the large-shot end of every sweep is measuring the approximation.
#[test]
fn the_two_sampling_branches_agree_across_the_boundary() {
    let circuit = hardware_efficient(3, 3);
    let x = overtone_sim::random::random_params(2, circuit.num_params());
    let psi = circuit.run(&x);
    let term = DiagonalTerm::from_observable(&Observable::z_global(3)).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(5);

    for shots in [EXACT_SAMPLING_LIMIT, EXACT_SAMPLING_LIMIT + 1] {
        let draws: Vec<f64> = (0..400)
            .map(|_| term.sample(&psi, shots, &mut rng))
            .collect();
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        let var = draws.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / draws.len() as f64;
        let predicted = term.standard_error(&psi, shots).powi(2);
        assert!(
            (mean - term.exact(&psi)).abs() < 1e-3,
            "{shots} shots: mean {mean}"
        );
        assert!(
            (var / predicted - 1.0).abs() < 0.25,
            "{shots} shots: variance {var} against {predicted}"
        );
    }
}

/// A non-diagonal observable cannot be sampled from the computational basis, and is
/// rejected rather than silently mis-estimated.
#[test]
fn a_non_diagonal_observable_is_refused() {
    let x_only = Observable::new(vec![PauliTerm::new(1.0, vec![(0, Pauli::X)])]);
    assert!(DiagonalTerm::from_observable(&x_only).is_none());
    let two_terms = Observable::new(vec![
        PauliTerm::new(1.0, vec![(0, Pauli::Z)]),
        PauliTerm::new(1.0, vec![(1, Pauli::Z)]),
    ]);
    assert!(DiagonalTerm::from_observable(&two_terms).is_none());
}

/// The metered parameter-shift gradient is the same gradient the adjoint path computes.
/// If it were not, the gradient lanes would be racing on a different function.
#[test]
fn the_metered_gradient_is_the_adjoint_gradient() {
    let mut cost = PlateauCost::new(
        4,
        DepthPolicy::Constant(3),
        CostLocality::Global,
        Budget::Exact,
        1_000_000,
        ChaCha8Rng::seed_from_u64(1),
    );
    let x = overtone_sim::random::random_params(6, cost.num_params());
    let shifted = cost.gradient(&x);
    let exact = cost.exact_gradient(&x);
    let worst = shifted
        .iter()
        .zip(&exact)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(worst < 1e-10, "parameter-shift against adjoint: {worst}");
}

/// The headline. With exact arithmetic Nelder-Mead descends in the plateau; with a
/// thousand shots per evaluation, at the same qubit count, it does not. Part V 5.2's
/// demonstration is a statement about measurement, and written without shots it shows the
/// opposite of the paper it cites.
#[test]
fn the_flatline_is_a_property_of_shots_not_of_the_landscape() {
    let n = 8;
    let evaluations = 200;
    let exact = lane(
        RaceConfig::plateau(n, Budget::Exact, evaluations, 3),
        Optimiser::NelderMead,
    );
    assert!(
        exact.progress() > 0.01,
        "exact arithmetic should descend, got {}",
        exact.progress()
    );

    let noisy = lane(
        RaceConfig::plateau(n, Budget::Shots(1000), evaluations * 1000, 3),
        Optimiser::NelderMead,
    );
    assert!(
        !noisy.beat_the_noise(),
        "a thousand shots should not resolve the plateau at n={n}, got {} against a floor of {}",
        noisy.progress(),
        noisy.noise_floor
    );
}

/// Arrasmith et al.: the shots required grow exponentially with the qubit count. Measured
/// here on the cheapest lane, with the `R^2` that makes the exponent a measurement.
#[test]
fn shots_required_grow_exponentially() {
    let seeds = [1u64, 2, 3];
    let points: Vec<(usize, usize)> = [2usize, 4, 6, 8]
        .into_iter()
        .filter_map(|n| {
            shots_to_progress(Optimiser::Cmaes, n, &seeds, 16, 1 << 14, 100).map(|s| (n, s))
        })
        .collect();
    assert!(points.len() >= 3, "sweep did not resolve: {points:?}");
    let fit = fit_shot_scaling(&points);
    assert!(
        fit.exponent > 0.2,
        "exponent {} is not exponential growth",
        fit.exponent
    );
    assert!(
        fit.r_squared > 0.8,
        "R^2 {} too low to quote",
        fit.r_squared
    );
}

/// Scoring a noisy optimiser on the best value it saw credits it with the noise. At least
/// one lane must end up worse than its own best-seen point claims, or the warning in
/// `race.rs` is decoration.
#[test]
fn best_seen_value_overstates_the_descent() {
    let lanes: Vec<_> =
        overtone_opt::race::race(RaceConfig::plateau(4, Budget::Shots(100), 200 * 100, 1));
    let overstated = lanes
        .iter()
        .any(|l| l.reported_best < l.final_exact - 0.1 && l.reported_best < l.best_exact - 0.05);
    assert!(
        overstated,
        "no lane overstated its descent; the scoring warning would be untested"
    );
}

/// `Budget::Exact` must still advance the meter, or every optimiser loops forever.
#[test]
fn the_exact_budget_still_terminates() {
    let mut cost = PlateauCost::new(
        3,
        DepthPolicy::Constant(2),
        CostLocality::Local,
        Budget::Exact,
        5,
        ChaCha8Rng::seed_from_u64(0),
    );
    let x = vec![0.1; cost.num_params()];
    for _ in 0..5 {
        cost.value(&x);
    }
    assert!(cost.exhausted());
}
