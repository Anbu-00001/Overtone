//! The walk itself: a discrete-time quantum walk on the integer line.
//!
//! Part II 3: **a discrete-time quantum walk has a strict light cone.** One step moves
//! amplitude to nearest neighbours only, so after `t` steps the support has radius at most
//! `t` — and outside it the amplitude is not small, it is *zero*. The representation is
//! that fact: the occupied window grows by one site per step and there is no lattice
//! outside it to truncate.
//!
//! Part II 9 is also emphatic about the language, and it is worth keeping in the code as
//! well as the prose: there is no walker. There is an amplitude field, and the agent is the
//! coin.

use overtone_sim::C64;
use rand::Rng;

use crate::coin::Coin;
use crate::substrate::Substrate;

/// A coined walk on the line, stored as the light cone and nothing else.
#[derive(Clone, Debug)]
pub struct Walk {
    /// `amps[2 * i + c]`, where site `i` holds lattice position `i - centre`.
    amps: Vec<C64>,
    scratch: Vec<C64>,
    centre: i64,
    steps: usize,
    /// Occupied window, inclusive, as site indices.
    lo: usize,
    hi: usize,
}

impl Walk {
    /// A field localised at the origin, in the symmetric coin state `(|0> + i|1>)/sqrt(2)`.
    ///
    /// That initial coin is the standard choice for a symmetric Hadamard walk; the real part
    /// alone would produce the familiar lopsided distribution and every substrate comparison
    /// would then be reading an artefact of the start rather than of the world.
    pub fn new(max_steps: usize) -> Self {
        let sites = 2 * max_steps + 1;
        let centre = max_steps as i64;
        let mut amps = vec![C64::ZERO; 2 * sites];
        let r = std::f64::consts::FRAC_1_SQRT_2;
        amps[2 * centre as usize] = C64::new(r, 0.0);
        amps[2 * centre as usize + 1] = C64::new(0.0, r);
        Walk {
            scratch: vec![C64::ZERO; 2 * sites],
            amps,
            centre,
            steps: 0,
            lo: centre as usize,
            hi: centre as usize,
        }
    }

    /// A field localised at lattice position `x` in an arbitrary coin state.
    ///
    /// [`Walk::new`] is the symmetric start every substrate comparison uses; this is the one
    /// M28 needs, because building a single-particle unitary means evolving each basis mode
    /// `(site, coin)` on its own and reading off where it went.
    pub fn localised(max_steps: usize, x: i64, coin: [C64; 2]) -> Self {
        let sites = 2 * max_steps + 1;
        let centre = max_steps as i64;
        let site = centre + x;
        assert!(
            site >= 0 && (site as usize) < sites,
            "position {x} is outside a {max_steps}-step window"
        );
        let mut amps = vec![C64::ZERO; 2 * sites];
        amps[2 * site as usize] = coin[0];
        amps[2 * site as usize + 1] = coin[1];
        Walk {
            scratch: vec![C64::ZERO; 2 * sites],
            amps,
            centre,
            steps: 0,
            lo: site as usize,
            hi: site as usize,
        }
    }

    /// The raw mode amplitudes, `amps[2 * site + coin]`, with `site` measured from the left
    /// edge of the window rather than from the origin. Paired with [`Walk::centre_site`].
    pub fn mode_amplitudes(&self) -> &[C64] {
        &self.amps
    }

    /// The site index holding lattice position zero.
    pub fn centre_site(&self) -> usize {
        self.centre as usize
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Lattice positions currently inside the light cone, inclusive.
    pub fn support(&self) -> (i64, i64) {
        (self.lo as i64 - self.centre, self.hi as i64 - self.centre)
    }

    /// Probability at each occupied site, paired with its lattice position.
    pub fn distribution(&self) -> Vec<(i64, f64)> {
        (self.lo..=self.hi)
            .map(|i| {
                let a = self.amps[2 * i];
                let b = self.amps[2 * i + 1];
                (
                    i as i64 - self.centre,
                    a.re * a.re + a.im * a.im + b.re * b.re + b.im * b.im,
                )
            })
            .collect()
    }

    pub fn total_probability(&self) -> f64 {
        self.amps.iter().map(|a| a.re * a.re + a.im * a.im).sum()
    }

    /// Standard deviation of position. Part IV 3's transport dial is `sigma(t) ~ t^beta`.
    pub fn sigma(&self) -> f64 {
        let (mut m1, mut m2, mut p) = (0.0, 0.0, 0.0);
        for (x, w) in self.distribution() {
            let x = x as f64;
            m1 += w * x;
            m2 += w * x * x;
            p += w;
        }
        if p <= 0.0 {
            return 0.0;
        }
        (m2 / p - (m1 / p).powi(2)).max(0.0).sqrt()
    }

    /// One step: coin, then shift. The coin at each site comes from the substrate.
    pub fn step(&mut self, substrate: &Substrate, coin_a: &Coin, coin_b: &Coin) {
        let (lo, hi) = (self.lo, self.hi);
        for i in lo..=hi {
            self.scratch[2 * i] = C64::ZERO;
            self.scratch[2 * i + 1] = C64::ZERO;
        }
        if lo > 0 {
            self.scratch[2 * (lo - 1)] = C64::ZERO;
            self.scratch[2 * (lo - 1) + 1] = C64::ZERO;
        }
        if hi + 1 < self.amps.len() / 2 {
            self.scratch[2 * (hi + 1)] = C64::ZERO;
            self.scratch[2 * (hi + 1) + 1] = C64::ZERO;
        }

        for i in lo..=hi {
            let x = i as i64 - self.centre;
            let coin = substrate.coin(x, coin_a, coin_b);
            let (a, b) = coin.apply(self.amps[2 * i], self.amps[2 * i + 1]);
            // Direction 0 moves left, direction 1 moves right.
            self.scratch[2 * (i - 1)] = self.scratch[2 * (i - 1)] + a;
            self.scratch[2 * (i + 1) + 1] = self.scratch[2 * (i + 1) + 1] + b;
        }

        std::mem::swap(&mut self.amps, &mut self.scratch);
        self.lo = lo - 1;
        self.hi = hi + 1;
        self.steps += 1;
    }

    /// Collapse the field onto one site, as a measurement of position does.
    ///
    /// Part II 5 is explicit that decoherence must be done with trajectories rather than a
    /// density matrix: a density matrix is `N^2` and destroys the sparse representation,
    /// while sampling pure trajectories with random measurement events keeps it and is
    /// physically the same thing.
    pub fn measure_position<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        let dist = self.distribution();
        let total: f64 = dist.iter().map(|(_, p)| p).sum();
        let mut u = rng.gen_range(0.0f64..1.0) * total;
        let mut chosen = dist[0].0;
        for (x, p) in &dist {
            u -= p;
            if u <= 0.0 {
                chosen = *x;
                break;
            }
        }
        let i = (chosen + self.centre) as usize;
        let a = self.amps[2 * i];
        let b = self.amps[2 * i + 1];
        let norm = (a.re * a.re + a.im * a.im + b.re * b.re + b.im * b.im).sqrt();
        for j in self.lo..=self.hi {
            self.amps[2 * j] = C64::ZERO;
            self.amps[2 * j + 1] = C64::ZERO;
        }
        self.amps[2 * i] = a * (1.0 / norm);
        self.amps[2 * i + 1] = b * (1.0 / norm);
        self.lo = i;
        self.hi = i;
    }
}

/// How the walk is run: coherently, or with measurement at some rate.
#[derive(Clone, Copy, Debug)]
pub struct Run {
    pub steps: usize,
    /// Probability of a position measurement after each step. `0` is fully coherent, `1`
    /// measures every step and must reproduce the classical random walk.
    pub decoherence: f64,
    /// Trajectories to average when `decoherence > 0`.
    pub trajectories: usize,
    pub seed: u64,
}

impl Run {
    pub fn coherent(steps: usize) -> Run {
        Run {
            steps,
            decoherence: 0.0,
            trajectories: 1,
            seed: 0,
        }
    }
}

/// `sigma(t)` for `t = 1..steps`, averaged over trajectories where that applies.
pub fn sigma_trace(substrate: &Substrate, coin_a: &Coin, coin_b: &Coin, run: Run) -> Vec<f64> {
    use rand::SeedableRng;

    if run.decoherence <= 0.0 {
        let mut w = Walk::new(run.steps);
        return (0..run.steps)
            .map(|_| {
                w.step(substrate, coin_a, coin_b);
                w.sigma()
            })
            .collect();
    }

    // Averaging sigma over trajectories would measure the wrong thing: the ensemble's
    // spread is the second moment of the *mixture*, so the moments are what must be
    // averaged, not the standard deviations.
    let mut m1 = vec![0.0; run.steps];
    let mut m2 = vec![0.0; run.steps];
    for k in 0..run.trajectories {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(
            run.seed ^ (k as u64).wrapping_mul(0x9e3779b97f4a7c15),
        );
        let mut w = Walk::new(run.steps);
        for t in 0..run.steps {
            w.step(substrate, coin_a, coin_b);
            if rng.gen_range(0.0f64..1.0) < run.decoherence {
                w.measure_position(&mut rng);
            }
            for (x, p) in w.distribution() {
                m1[t] += p * x as f64;
                m2[t] += p * (x as f64) * (x as f64);
            }
        }
    }
    let n = run.trajectories as f64;
    (0..run.steps)
        .map(|t| (m2[t] / n - (m1[t] / n).powi(2)).max(0.0).sqrt())
        .collect()
}
