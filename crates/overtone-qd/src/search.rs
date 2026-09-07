//! The MAP-Elites loop.
//!
//! Mouret & Clune's algorithm, unmodified: draw an initial population at random, then
//! repeatedly pick a random elite, mutate it, measure it, and offer it to the archive. No
//! inner gradient step — the genome carries its own angles, so this is a search in parameter
//! space and the comparison against REINFORCE in `examples/menagerie.rs` is a fair one.

use overtone_rl::SpectralControl;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::archive::{Archive, Bins};
use crate::behaviour::measure;
use crate::genome::Genome;

#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Random genomes drawn before the loop starts.
    pub initial: usize,
    /// Total evaluations, including the initial population.
    pub evaluations: usize,
    /// Angle mutation width.
    pub sigma: f64,
    /// Quadrature nodes for the exact return. The integrand is smooth and periodic, so a few
    /// dozen give more precision than thousands of sampled episodes would.
    pub nodes: usize,
    pub k: usize,
    pub bins: Bins,
    pub seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            initial: 200,
            evaluations: 3000,
            sigma: 0.35,
            nodes: 48,
            k: 3,
            bins: Bins::default(),
            seed: 20260907,
        }
    }
}

/// Fill an archive.
pub fn map_elites(cfg: &Config) -> Archive {
    let env = SpectralControl::new(cfg.k);
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let mut archive = Archive::new(cfg.bins);

    for _ in 0..cfg.initial.min(cfg.evaluations) {
        let g = Genome::random(&mut rng);
        let (b, f) = measure(&g, &env, cfg.nodes);
        archive.offer(g, b, f);
    }

    while archive.evaluations() < cfg.evaluations {
        let parent = {
            let elites = archive.ranked();
            if elites.is_empty() {
                Genome::random(&mut rng)
            } else {
                // Uniformly over *cells*, not weighted by fitness: weighting would pull the
                // search back toward the peak and undo the reason for keeping a map.
                elites
                    .choose(&mut rng)
                    .map(|c| c.genome.clone())
                    .unwrap_or_else(|| Genome::random(&mut rng))
            }
        };
        let child = parent.mutate(&mut rng, cfg.sigma);
        let (b, f) = measure(&child, &env, cfg.nodes);
        archive.offer(child, b, f);
    }
    archive
}

/// The control: the same evaluation budget spent maximising return alone, with no archive.
///
/// This is what MAP-Elites has to be compared against. Part IV 4 claims quality-diversity
/// keeps working where gradient methods stop; Arrasmith et al. say no optimiser that decides
/// on cost differences does. Running both at a matched budget is how the claim gets settled
/// rather than repeated.
pub fn hill_climb(cfg: &Config) -> (Genome, f64) {
    let env = SpectralControl::new(cfg.k);
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let mut best = Genome::random(&mut rng);
    let (_, mut best_f) = measure(&best, &env, cfg.nodes);
    let mut used = 1;
    while used < cfg.evaluations {
        let child = if rng.gen_bool(0.05) {
            Genome::random(&mut rng)
        } else {
            best.mutate(&mut rng, cfg.sigma)
        };
        let (_, f) = measure(&child, &env, cfg.nodes);
        used += 1;
        if f > best_f {
            best_f = f;
            best = child;
        }
    }
    (best, best_f)
}
