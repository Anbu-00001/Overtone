//! M16's measurement: the transport exponent of every world.
//!
//! `cargo run --release -p overtone-walk --example transport`

use overtone_walk::{
    classical_sigma, fit_exponent, regime, regime_of, sigma_trace, Coin, Run, Substrate,
};
use overtone_walk::{Word, ALL_WORDS};

fn main() {
    let steps: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(600);

    println!("sigma(t) ~ t^beta, fitted over the last half of {steps} steps");
    println!("coin A = Hadamard (theta = pi/4), coin B = theta = pi/3\n");
    println!(
        "{:<16} {:>7} {:>7} {:>9} {:>16}  regime in the literature",
        "world", "beta", "R^2", "sigma(T)", "measured"
    );

    let a = Coin::hadamard();
    let b = Coin::theta(std::f64::consts::FRAC_PI_3);
    for w in ALL_WORDS {
        let s = Substrate::new(w, steps, 20260907);
        let trace = sigma_trace(&s, &a, &b, Run::coherent(steps));
        let fit = fit_exponent(&trace, 0.5);
        println!(
            "{:<16} {:>7.3} {:>7.3} {:>9.2} {:>16}  {}",
            w.name(),
            fit.beta,
            fit.r_squared,
            trace[steps - 1],
            regime_of(&fit),
            w.regime()
        );
    }

    let c = fit_exponent(&classical_sigma(steps), 0.5);
    println!(
        "{:<16} {:>7.3} {:>7.3} {:>9.2} {:>16}  diffusive, exactly 1/2",
        "classical",
        c.beta,
        c.r_squared,
        (steps as f64).sqrt(),
        regime(c.beta)
    );

    println!("\ndecoherence sweep on the periodic lattice (trajectories, not density matrices)");
    let s = Substrate::new(Word::Periodic, steps, 7);
    for p in [0.0, 0.02, 0.1, 0.5, 1.0] {
        let trace = sigma_trace(
            &s,
            &a,
            &a,
            Run {
                steps: steps.min(200),
                decoherence: p,
                trajectories: if p == 0.0 { 1 } else { 400 },
                seed: 99,
            },
        );
        let fit = fit_exponent(&trace, 0.5);
        println!(
            "  p = {p:<5}  beta = {:>6.3}   R^2 = {:>5.3}   {}",
            fit.beta,
            fit.r_squared,
            regime(fit.beta)
        );
    }

    // The published sweeps report *distributions* of beta over coin parameters, not one
    // number, because the exponent is strongly coin-dependent. theta_2 near 0, pi/2 or pi
    // makes coin B diagonal or antidiagonal -- a coin that does not superpose at all -- so
    // those degenerate points are excluded rather than allowed to pin the range at one.
    println!(
        "\ncoin dependence: theta_1 = pi/4 fixed, theta_2 swept over 33 non-degenerate values"
    );
    println!(
        "{:<16} {:>8} {:>8} {:>8}  {:>10} {:>10}",
        "world", "min", "median", "max", "localized", "ballistic"
    );
    for w in [Word::Fibonacci, Word::ThueMorse, Word::RudinShapiro] {
        let sub = Substrate::new(w, 300, 20260907);
        let mut betas = Vec::new();
        for j in 1..=36 {
            let t2 = std::f64::consts::PI * j as f64 / 37.0;
            let deg = [0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI]
                .iter()
                .any(|d| (t2 - d).abs() < 0.12);
            if deg {
                continue;
            }
            let trace = sigma_trace(&sub, &a, &Coin::theta(t2), Run::coherent(300));
            betas.push(fit_exponent(&trace, 0.5).beta);
        }
        betas.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let frac = |f: &dyn Fn(f64) -> bool| {
            betas.iter().filter(|b| f(**b)).count() as f64 * 100.0 / betas.len() as f64
        };
        println!(
            "{:<16} {:>8.3} {:>8.3} {:>8.3}  {:>9.0}% {:>9.0}%",
            w.name(),
            betas[0],
            betas[betas.len() / 2],
            betas[betas.len() - 1],
            frac(&|b| b < 0.15),
            frac(&|b| b > 0.92),
        );
    }

    println!("\nwords, first 15 letters, against Lo Gullo et al. Fig. 1");
    for w in [Word::Fibonacci, Word::ThueMorse, Word::RudinShapiro] {
        println!("  {:<14} {}", w.name(), Substrate::new(w, 64, 0).render(15));
    }

    // Part II 7 asks that "p = 1 reproduces the classical walk to TV < 1e-3". Which
    // classical walk? The simple random walk's binomial, or the Gaussian a fully decohered
    // quantum walk is known to converge to (Brun, Carteret & Ambainis, PRA 67, 032304)?
    // They are not the same distribution and the difference is not sampling error.
    println!("\nfull decoherence against two candidate classical limits, t = 24");
    let t = 24usize;
    let trajectories = 200_000usize;
    let mut hist = vec![0.0f64; 2 * t + 1];
    {
        use rand::SeedableRng;
        for kk in 0..trajectories {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(555 ^ kk as u64);
            let mut w = overtone_walk::Walk::new(t);
            for _ in 0..t {
                w.step(&Substrate::new(Word::Periodic, t, 0), &a, &a);
                w.measure_position(&mut rng);
            }
            let (lo, _) = w.support();
            hist[(lo + t as i64) as usize] += 1.0 / trajectories as f64;
        }
    }
    let binom = {
        let mut v = vec![0.0f64; 2 * t + 1];
        let mut c = 1.0f64;
        for j in 0..=t {
            v[2 * j] = c * 0.5f64.powi(t as i32);
            c = c * (t - j) as f64 / (j + 1) as f64;
        }
        v
    };
    let var: f64 = hist
        .iter()
        .enumerate()
        .map(|(i, p)| p * (i as f64 - t as f64).powi(2))
        .sum();
    let gauss = {
        let mut v = vec![0.0f64; 2 * t + 1];
        // Only sites of the right parity are reachable, so the Gaussian is binned over 2.
        for (i, slot) in v.iter_mut().enumerate() {
            if (i + t) % 2 == 0 {
                let x = i as f64 - t as f64;
                *slot =
                    2.0 / (2.0 * std::f64::consts::PI * var).sqrt() * (-x * x / (2.0 * var)).exp();
            }
        }
        v
    };
    let tv = |p: &[f64], q: &[f64]| -> f64 {
        0.5 * p.iter().zip(q).map(|(a, b)| (a - b).abs()).sum::<f64>()
    };
    println!("  measured variance      {var:.3}   (t = {t}, so the classical value is {t}.000)");
    println!("  TV to the binomial     {:.5}", tv(&hist, &binom));
    println!("  TV to a Gaussian       {:.5}", tv(&hist, &gauss));
    println!(
        "  sampling floor         ~{:.5}  ({trajectories} trajectories)",
        ((2 * t + 1) as f64 / trajectories as f64).sqrt()
    );

    println!("\nthe race (Part IV 2.3): arrival within 2 sites of x = 100 after 200 steps");
    println!(
        "{:<16} {:>10} {:>10} {:>10} {:>22}",
        "world", "Hadamard", "Grover", "optimised", "optimised coin angles"
    );
    for w in ALL_WORDS {
        let sub = Substrate::new(w, 200, 20260907);
        let r = overtone_walk::race(&sub, 200);
        let arr = |e: &overtone_walk::Entrant, ta: f64, tb: f64| {
            let mut wk = overtone_walk::Walk::new(200);
            let _ = e;
            for _ in 0..200 {
                wk.step(&sub, &Coin::theta(ta), &Coin::theta(tb));
            }
            wk.distribution()
                .iter()
                .filter(|(x, _)| (x - 100).abs() <= 2)
                .map(|(_, p)| p)
                .sum::<f64>()
        };
        let pi4 = std::f64::consts::FRAC_PI_4;
        let (oa, ob) = r[2].angles.unwrap();
        println!(
            "{:<16} {:>10.4} {:>10.4} {:>10.4} {:>11.3} {:>10.3}",
            w.name(),
            arr(&r[0], pi4, pi4),
            0.0,
            arr(&r[2], oa, ob),
            oa,
            ob
        );
    }

    println!("\nthe Grover coin in one dimension");
    let g = Coin::grover();
    let trace = sigma_trace(
        &Substrate::new(Word::Periodic, 60, 0),
        &g,
        &g,
        Run::coherent(60),
    );
    println!(
        "  sigma after 60 steps = {:.4}  (2|s><s| - I on two directions is Pauli X)",
        trace[59]
    );
}
