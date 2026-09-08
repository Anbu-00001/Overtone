//! Architecture search with an algebraic reward, and the check that it was worth anything
//! (Part V 3, M27).
//!
//! # The idea
//!
//! Parts I to III use quantum circuits to do RL. Invert it: use a search agent to design
//! the circuit. Quantum architecture search is usually impractical because every reward
//! needs the candidate trained, so thousands of architectures means thousands of training
//! runs. Part III removes that cost: `dim(g)`, the reachable frequency set and the gate
//! count all come from the generators in milliseconds, with no training at all.
//!
//! ```text
//! reward = w1 (is dim(g) polynomial?)      will it train
//!        + w2 (does the reach cover k?)    can it represent the answer
//!        - w3 (gate count)                 is it implementable
//! ```
//!
//! # The verification is the contribution, not the search
//!
//! Part V 10 is explicit: "an architecture search rewarded by a proxy is only interesting if
//! someone checks whether the proxy was right." So [`verify`] trains every candidate to
//! convergence on the exact policy gradient and reports the rank correlation between the
//! algebraic score and the return actually achieved -- and reports it whichever way it comes
//! out.
//!
//! Each term is scored **separately** as well as in combination. Part V gives `w1`, `w2` and
//! `w3` as symbols with no values, and this repository does not invent constants: a combined
//! score needs weights, but the question "does the reach term predict anything" does not,
//! and it is the more informative question anyway.

use overtone_lie::{closure, family::EntanglerPolicy};
use overtone_rl::SpectralControl;
/// Re-exported: the shared implementation now lives in `overtone-spec`.
pub use overtone_spec::spearman;
use rand::Rng;

use crate::genome::Genome;

/// What the algebra says about a candidate, before any training.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Algebraic {
    /// `dim(g)` of the ansatz family.
    pub dim_g: usize,
    /// `dim(g)` against the exponential ceiling `4^n - 1`, as a fraction. Ragone et al.'s
    /// variance bound falls with `dim(g)`, so *small* is the trainable end.
    pub dim_fraction: f64,
    /// True when `dim(g)` stays below a polynomial cut in the qubit count.
    pub polynomial: bool,
    /// `C * |lambda|`: how far in frequency the policy can see.
    pub reach: f64,
    /// True when the reach covers the task frequency.
    pub covers: bool,
    /// Gates in the built circuit.
    pub gates: usize,
    /// Trainable parameters.
    pub params: usize,
}

/// Whether `dim(g)` is small enough to call polynomial.
///
/// `n^4` is the cut, and it is not a tuning knob: the classification of Wiersema et al.
/// puts the polynomial dynamical Lie algebras of spin chains at `O(n)` to `O(n^2)`, while
/// the exponential ones sit at `4^n - 1` or a constant fraction of it. Any cut between the
/// two families separates them, and `n^4` is comfortably inside the gap at every width this
/// archive searches.
pub fn is_polynomial(dim_g: usize, qubits: usize) -> bool {
    dim_g <= qubits.pow(4)
}

/// Read a candidate's algebra. No training, no optimiser, no rollouts.
pub fn algebraic(genome: &Genome, env: &SpectralControl) -> Algebraic {
    let policy = genome.policy();
    let circuit = policy
        .ansatz
        .build(&[crate::behaviour::REFERENCE_OBSERVATION]);
    let generators = overtone_lie::family::from_circuit(&circuit, EntanglerPolicy::Parameterised);
    let algebra = closure(&generators.generators, genome.qubits, 4096);
    let dim_g = algebra.dim();

    let ceiling = policy.ansatz.frequency_ceiling(0) as f64;
    let lambda = policy
        .ansatz
        .lambda_range()
        .next()
        .map(|i| genome.params[i].abs())
        .unwrap_or(1.0);
    let reach = ceiling * lambda;

    let full = 4usize
        .saturating_pow(genome.qubits as u32)
        .saturating_sub(1);
    Algebraic {
        dim_g,
        dim_fraction: dim_g as f64 / full.max(1) as f64,
        polynomial: is_polynomial(dim_g, genome.qubits),
        reach,
        covers: reach >= env.k as f64 - 1e-9,
        gates: circuit.gates().len(),
        params: policy.num_params(),
    }
}

/// Part V 3's reward, with its weights made arguments rather than constants.
///
/// The gate-count term is divided by `gate_scale` so that all three terms are `O(1)` and
/// the weights mean what they look like they mean. Leaving it in raw gates would make `w3`
/// a number whose magnitude depends on how large the circuits happen to be, which is how a
/// weight becomes a tuned constant without anyone deciding to tune it.
pub fn algebraic_reward(a: &Algebraic, w1: f64, w2: f64, w3: f64, gate_scale: f64) -> f64 {
    w1 * f64::from(a.polynomial) + w2 * f64::from(a.covers) - w3 * (a.gates as f64 / gate_scale)
}

/// A candidate, its cheap score, and what it was actually worth.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub genome: Genome,
    pub algebra: Algebraic,
    /// Exact return after training, or `f64::NAN` before [`verify`] has run.
    pub trained_return: f64,
    /// Exact return before training, at the drawn angles.
    pub initial_return: f64,
}

/// Quadrature nodes used to evaluate the exact return and its gradient.
///
/// `J = E_s[cos(ks)(2 pi(1|s) - 1)]`, and `pi(1|s)` is a trig polynomial of degree at most
/// the ansatz's frequency ceiling `C`, so the integrand has degree at most `C + k`. The
/// midpoint rule on `N` equispaced nodes integrates `e^(ims)` exactly for every `m` that is
/// not a multiple of `N`, so it is exact whenever `C + k < N` -- there is no accuracy to buy
/// above that, only cost.
///
/// The widest architecture `Genome::random` can draw is five qubits by four layers with an
/// encoding rotation per qubit per layer, giving `C = 20`, against `k = 3`. So 64 nodes is
/// exact with a factor of two to spare, and doubling it would double the cost of the sweep
/// to change nothing.
pub const NODES: usize = 64;

/// How often the return is evaluated during training. Adam's step does not read `J`, so
/// evaluating it every iteration doubles the cost of the sweep to refine a number that is
/// only used for the running maximum.
const EVALUATE_EVERY: usize = 10;

/// Train one candidate to convergence on the exact policy gradient.
///
/// The gradient is analytic here -- `SpectralControl` has a closed-form return -- so this is
/// as close to "trained properly" as the environment allows, and any failure to reach a
/// good return is the architecture's rather than the optimiser's.
pub fn train_to_convergence(genome: &Genome, env: &SpectralControl, steps: usize) -> f64 {
    let policy = genome.policy();
    let mut params = genome.params.clone();
    let mut adam = overtone_rl::reinforce::Adam::new(params.len(), 0.05);
    let mut best = policy.expected_return(&params, env.k, NODES);
    for step in 0..steps {
        let g = policy.analytic_return_grad(&params, env.k, NODES);
        adam.ascend(&mut params, &g);
        if step % EVALUATE_EVERY == 0 || step + 1 == steps {
            let j = policy.expected_return(&params, env.k, NODES);
            if j > best {
                best = j;
            }
        }
    }
    best
}

/// Draw `count` architectures, score them algebraically, then train every one and report
/// how well the cheap score predicted the expensive answer.
pub fn verify<R: Rng + ?Sized>(
    env: &SpectralControl,
    count: usize,
    training_steps: usize,
    rng: &mut R,
) -> Vec<Candidate> {
    (0..count)
        .map(|_| {
            let genome = Genome::random(rng);
            let algebra = algebraic(&genome, env);
            let policy = genome.policy();
            Candidate {
                initial_return: policy.expected_return(&genome.params, env.k, NODES),
                trained_return: train_to_convergence(&genome, env, training_steps),
                algebra,
                genome,
            }
        })
        .collect()
}

/// Mean trained return of the top `k` by a scoring function, against the mean over all.
///
/// The practical question an architecture search has to answer: if I spend my training
/// budget on what the proxy liked, do I do better than spending it at random?
pub fn selection_gain<F: Fn(&Candidate) -> f64>(
    candidates: &[Candidate],
    k: usize,
    score: F,
) -> (f64, f64) {
    let mut sorted: Vec<&Candidate> = candidates.iter().collect();
    sorted.sort_by(|a, b| score(b).partial_cmp(&score(a)).unwrap());
    let top =
        sorted.iter().take(k).map(|c| c.trained_return).sum::<f64>() / k.min(sorted.len()) as f64;
    let all = candidates.iter().map(|c| c.trained_return).sum::<f64>() / candidates.len() as f64;
    (top, all)
}
