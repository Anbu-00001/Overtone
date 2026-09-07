//! M16's acceptance test, plus the pieces of Part II's M6 and M7 this engine carries.
//!
//! Every expected value here is either a closed form or a published regime. The three
//! aperiodic words are pinned character-for-character against Fig. 1 of Lo Gullo,
//! Di Molfetta, Sanchez-Palencia and Wilkowski, Phys. Rev. E 96, 012111 (2017),
//! arXiv:1611.04427, which is the strongest available check that the substrate is the one
//! the transport results were measured on.

use overtone_walk::{
    classical_sigma, fit_exponent, sigma_trace, Coin, Run, Substrate, Walk, Word, ALL_WORDS,
};

const HALF: f64 = 0.5;

fn beta_of(word: Word, theta_b: f64, steps: usize) -> f64 {
    let s = Substrate::new(word, steps, 20260907);
    let trace = sigma_trace(
        &s,
        &Coin::hadamard(),
        &Coin::theta(theta_b),
        Run::coherent(steps),
    );
    fit_exponent(&trace, HALF).beta
}

#[test]
fn words_match_the_published_sequences() {
    // Lo Gullo et al. Fig. 1 prints the first fifteen letters of each word.
    assert_eq!(
        Substrate::new(Word::Fibonacci, 64, 0).render(15),
        "ABAABABAABAABAB"
    );
    assert_eq!(
        Substrate::new(Word::ThueMorse, 64, 0).render(15),
        "ABBABAABBAABABB"
    );
    assert_eq!(
        Substrate::new(Word::RudinShapiro, 64, 0).render(15),
        "AAABAABAAAABBBA"
    );
}

#[test]
fn the_clean_lattice_is_ballistic() {
    // Part II 7, M6: fitted exponent 1.00 +- 0.03.
    let beta = beta_of(Word::Periodic, std::f64::consts::FRAC_PI_3, 400);
    assert!((beta - 1.0).abs() < 0.03, "periodic beta was {beta}");
    // The two-periodic word is equivalent to a two-period walk in time and stays ballistic.
    let beta2 = beta_of(Word::TwoPeriodic, std::f64::consts::FRAC_PI_3, 400);
    assert!((beta2 - 1.0).abs() < 0.03, "two-periodic beta was {beta2}");
}

#[test]
fn the_classical_baseline_is_exactly_one_half() {
    // Part II 7, M6: 0.50 +- 0.03. It is a closed form, so it is exact.
    let fit = fit_exponent(&classical_sigma(400), HALF);
    assert!(
        (fit.beta - 0.5).abs() < 1e-12,
        "classical beta was {}",
        fit.beta
    );
}

#[test]
fn the_light_cone_is_strict() {
    // Part II 3: outside radius t the amplitude is not small, it is zero. That is what makes
    // the sparse representation exact rather than a truncation, so it is worth asserting
    // rather than assuming.
    let steps = 60;
    let sub = Substrate::new(Word::Fibonacci, steps, 5);
    let (a, b) = (Coin::hadamard(), Coin::theta(1.1));
    let mut w = Walk::new(steps);
    for t in 1..=steps {
        w.step(&sub, &a, &b);
        let (lo, hi) = w.support();
        assert_eq!((lo, hi), (-(t as i64), t as i64), "light cone at t = {t}");
        assert!(
            (w.total_probability() - 1.0).abs() < 1e-12,
            "norm drifted to {} at t = {t}",
            w.total_probability()
        );
    }
    // Nothing outside the cone, by construction: the distribution has no other sites.
    assert_eq!(w.distribution().len(), 2 * steps + 1);
}

#[test]
fn rudin_shapiro_localizes_where_fibonacci_and_thue_morse_spread() {
    // The ordering the literature reports: Fibonacci (pure point spectrum) spreads,
    // Thue-Morse (singular continuous) spreads with a localized component alongside, and
    // Rudin-Shapiro (absolutely continuous, the closest deterministic word to noise) is the
    // one that localizes. Measured as medians over the coin, because the exponent is
    // strongly coin-dependent and a single coin is not a claim about a substrate.
    let medians: Vec<(Word, f64)> = [Word::Fibonacci, Word::ThueMorse, Word::RudinShapiro]
        .iter()
        .map(|&w| {
            let mut betas: Vec<f64> = (1..=18)
                .map(|j| std::f64::consts::PI * j as f64 / 19.0)
                .filter(|t| {
                    (t - std::f64::consts::FRAC_PI_2).abs() > 0.12
                        && *t > 0.12
                        && *t < std::f64::consts::PI - 0.12
                })
                .map(|t| beta_of(w, t, 250))
                .collect();
            betas.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (w, betas[betas.len() / 2])
        })
        .collect();
    let get = |w: Word| medians.iter().find(|(x, _)| *x == w).unwrap().1;
    assert!(
        get(Word::RudinShapiro) < get(Word::Fibonacci) - 0.2,
        "Rudin-Shapiro {} was not clearly below Fibonacci {}",
        get(Word::RudinShapiro),
        get(Word::Fibonacci)
    );
    assert!(
        get(Word::RudinShapiro) < get(Word::ThueMorse) - 0.2,
        "Rudin-Shapiro {} was not clearly below Thue-Morse {}",
        get(Word::RudinShapiro),
        get(Word::ThueMorse)
    );
    assert!(
        get(Word::RudinShapiro) < 0.5,
        "Rudin-Shapiro median {} was not subdiffusive",
        get(Word::RudinShapiro)
    );
    for w in [Word::Fibonacci, Word::ThueMorse] {
        assert!(
            get(w) > 0.6,
            "{} median {} did not spread",
            w.name(),
            get(w)
        );
    }
}

#[test]
fn static_disorder_localizes() {
    // Anderson localization: sigma saturates, so the power law fit is both flat and bad.
    let steps = 400;
    let s = Substrate::new(Word::Disorder, steps, 20260907);
    let trace = sigma_trace(
        &s,
        &Coin::hadamard(),
        &Coin::theta(std::f64::consts::FRAC_PI_3),
        Run::coherent(steps),
    );
    let fit = fit_exponent(&trace, HALF);
    // The signature is saturation, not a small exponent: a line fitted through a plateau
    // reports the plateau's noise as a slope, which is why the fit must be rejected on its
    // R^2 before its beta is read at all.
    assert!(
        !fit.is_power_law(),
        "disorder produced a clean power law, beta = {} R^2 = {}",
        fit.beta,
        fit.r_squared
    );
    // Saturation, stated directly: doubling the time barely moves the spread, and the
    // spread itself is a few sites against a ballistic 300.
    let ratio = trace[steps - 1] / trace[steps / 2 - 1];
    assert!(ratio < 1.35, "sigma still growing: ratio {ratio}");
    assert!(
        trace[steps - 1] < 0.05 * steps as f64,
        "sigma {} was not localized against a ballistic {steps}",
        trace[steps - 1]
    );
}

#[test]
fn full_decoherence_is_exactly_a_simple_random_walk() {
    // Part II 7, M7 asks that p = 1 reproduce the classical walk. It does, and exactly
    // rather than approximately, for a reason worth recording: after a position measurement
    // the field sits on one site, so the next step sends component 0 to the left neighbour
    // and component 1 to the right, and each neighbour receives exactly one component. The
    // following measurement therefore always leaves a coin *basis* state, and the Hadamard
    // coin splits a basis state exactly in half. There is no residual coherence to decay --
    // the process is a simple symmetric random walk from the second step onward.
    use rand::SeedableRng;
    let steps = 24;
    let sub = Substrate::new(Word::Periodic, steps, 0);
    let h = Coin::hadamard();

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(11);
    let mut w = Walk::new(steps);
    for t in 1..=steps {
        w.step(&sub, &h, &h);
        w.measure_position(&mut rng);
        let (lo, hi) = w.support();
        assert_eq!(lo, hi, "measurement left more than one site at t = {t}");
    }

    // And the ensemble is the binomial, to sampling error.
    let trajectories = 20_000;
    let mut hist = vec![0.0f64; 2 * steps + 1];
    for k in 0..trajectories {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(555 ^ k as u64);
        let mut w = Walk::new(steps);
        for _ in 0..steps {
            w.step(&sub, &h, &h);
            w.measure_position(&mut rng);
        }
        hist[(w.support().0 + steps as i64) as usize] += 1.0 / trajectories as f64;
    }
    let mut binom = vec![0.0f64; 2 * steps + 1];
    let mut c = 1.0f64;
    for j in 0..=steps {
        binom[2 * j] = c * 0.5f64.powi(steps as i32);
        c = c * (steps - j) as f64 / (j + 1) as f64;
    }
    let tv: f64 = 0.5
        * hist
            .iter()
            .zip(&binom)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>();
    // Sampling error at 20000 trajectories, not a physical discrepancy: it falls as
    // 1/sqrt(N), and at 200000 it measures 0.0025.
    assert!(tv < 0.02, "total variation from the binomial was {tv}");
}

#[test]
fn the_grover_coin_is_degenerate_in_one_dimension() {
    // Part II 7 makes the Grover coin a mandatory baseline. On a two-dimensional coin space
    // 2|s><s| - I is exactly the Pauli X, so the field oscillates between two sites and
    // never spreads. Reporting that is the honest form of the baseline.
    let steps = 60;
    let trace = sigma_trace(
        &Substrate::new(Word::Periodic, steps, 0),
        &Coin::grover(),
        &Coin::grover(),
        Run::coherent(steps),
    );
    assert!(
        trace[steps - 1] < 1e-9,
        "Grover coin spread to {}",
        trace[steps - 1]
    );
}

#[test]
fn every_coin_is_unitary() {
    let mut coins = vec![Coin::hadamard(), Coin::grover()];
    for j in 0..8 {
        coins.push(Coin::theta(j as f64 * 0.7));
        coins.push(Coin::generic(
            j as f64 / 8.0,
            0.3 * j as f64,
            1.1 * j as f64,
        ));
    }
    for c in coins {
        assert!(c.unitarity_error() < 1e-14, "{:?}", c);
    }
    // generic(1/2, 0, 0) is the Hadamard coin.
    let g = Coin::generic(0.5, 0.0, 0.0);
    let h = Coin::hadamard();
    for i in 0..4 {
        assert!((g.0[i].re - h.0[i].re).abs() < 1e-15);
        assert!((g.0[i].im - h.0[i].im).abs() < 1e-15);
    }
}

#[test]
fn every_world_conserves_probability() {
    for w in ALL_WORDS {
        let steps = 80;
        let sub = Substrate::new(w, steps, 3);
        let (a, b) = (Coin::hadamard(), Coin::theta(0.9));
        let mut walk = Walk::new(steps);
        for _ in 0..steps {
            walk.step(&sub, &a, &b);
        }
        assert!(
            (walk.total_probability() - 1.0).abs() < 1e-12,
            "{} lost probability: {}",
            w.name(),
            walk.total_probability()
        );
    }
}

#[test]
fn a_constant_coin_cannot_see_the_world() {
    // Part II 2: the policy is the coin. An agent that plays the same coin at both letters
    // never reads the substrate, so its trace on every binary word is bit-identical -- not
    // approximately, exactly. This looks like a bug in the race and is the whole point of it.
    let steps = 120;
    let h = Coin::hadamard();
    let reference = sigma_trace(
        &Substrate::new(Word::Periodic, steps, 0),
        &h,
        &h,
        Run::coherent(steps),
    );
    for w in [
        Word::TwoPeriodic,
        Word::Fibonacci,
        Word::ThueMorse,
        Word::RudinShapiro,
    ] {
        let trace = sigma_trace(
            &Substrate::new(w, steps, 20260907),
            &h,
            &h,
            Run::coherent(steps),
        );
        assert_eq!(trace, reference, "{} moved a world-blind agent", w.name());
    }
    // Static disorder is the exception, and for a stated reason: its local parameter is drawn
    // from a continuum, so no single coin can be blind to it.
    let d = sigma_trace(
        &Substrate::new(Word::Disorder, steps, 20260907),
        &h,
        &h,
        Run::coherent(steps),
    );
    assert!(d[steps - 1] < 0.2 * reference[steps - 1]);
}

#[test]
fn conditioning_the_coin_on_the_world_pays_only_where_there_is_structure_to_exploit() {
    // Part II 7 asks for the learned-coin benchmark to be published either way. This is the
    // weaker, direct-search version of it, and the answer is a negative result: for arrival
    // at a distant target, the optimum on every aperiodic word is a coin that ignores the
    // substrate entirely. Only the periodic word rewards reading it.
    let steps = 160;
    for w in [Word::Fibonacci, Word::ThueMorse, Word::RudinShapiro] {
        let ((a, b), _) = overtone_walk::optimise_coins(&Substrate::new(w, steps, 20260907), steps);
        assert!(
            (a - b).abs() < 0.05,
            "{}: optimiser conditioned on the world, {a} vs {b}",
            w.name()
        );
    }
    let ((a, b), score) =
        overtone_walk::optimise_coins(&Substrate::new(Word::TwoPeriodic, steps, 0), steps);
    assert!(
        (a - b).abs() > 0.02,
        "two-periodic optimum did not condition: {a} vs {b}"
    );
    let (_, blind) = {
        let sub = Substrate::new(Word::Periodic, steps, 0);
        overtone_walk::optimise_coins(&sub, steps)
    };
    assert!(
        score > blind,
        "conditioning did not beat the blind optimum: {score} vs {blind}"
    );
}
