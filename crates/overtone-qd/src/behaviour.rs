//! The descriptors, all measured and none authored.
//!
//! Part IV 7: the moment one line of a stat block is authored rather than measured, the
//! whole card becomes fiction. That applies to archive axes with more force, because an
//! authored descriptor would place agents in cells they do not belong in and every
//! conclusion drawn from the map would inherit the error.

use overtone_lie::{closure, family::EntanglerPolicy};
use overtone_rl::SpectralControl;

use crate::genome::Genome;

/// Where an agent sits in behaviour space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Behaviour {
    /// Dimension of the ansatz's dynamical Lie algebra, entanglers counted as trainable.
    ///
    /// Part III's caveat applies and is why this is labelled as the *ansatz's* algebra: a
    /// fixed CZ layer is a Clifford, not a one-parameter subgroup, so the strict DLA of the
    /// built circuit is smaller and no trainability claim follows from either number. What
    /// this axis measures is expressiveness of the ansatz family, which is what Part IV 4
    /// asks the archive to span.
    pub dim_g: usize,
    /// `C * |lambda|`: how far in frequency the policy can see. The substitute for Part IV's
    /// transport exponent, which a contextual bandit does not have — see the crate docs.
    pub reach: f64,
    /// Bond dimension of the policy's state across the half-chain cut, at a reference
    /// observation. The cheap cousin of the full dequantization verdict, which costs a
    /// sweep over every bond dimension and is run on the finished elites instead.
    pub bond_dimension: usize,
    /// Half-chain von Neumann entropy at the same reference observation. Reported rather
    /// than binned; it is highly correlated with the bond dimension, so making it an axis
    /// would spend a dimension of the archive on the same information twice.
    pub entropy: f64,
}

/// The observation the entanglement descriptors are read at.
///
/// Fixed, and away from zero: at `s = 0` every encoding gate is the identity regardless of
/// `lambda`, so every agent would look identical on both entanglement axes and the archive
/// would collapse onto one column.
pub const REFERENCE_OBSERVATION: f64 = 1.0;

/// Measure an agent. Returns its behaviour and its exact return.
pub fn measure(genome: &Genome, env: &SpectralControl, nodes: usize) -> (Behaviour, f64) {
    assert!(
        genome.is_well_formed(),
        "genome has {} parameters but its structure needs {} -- a structural field was \
         changed without redrawing the angles; use Genome::structured",
        genome.params.len(),
        genome.policy().num_params()
    );
    let policy = genome.policy();
    let params = &genome.params;

    let generators = overtone_lie::family::from_circuit(
        &policy.ansatz.build(&[REFERENCE_OBSERVATION]),
        EntanglerPolicy::Parameterised,
    );
    // Capped: a five-qubit hardware-efficient ansatz closes at 1023, and the cap only ever
    // bites on an algebra that is already exponential, where the exact value changes nothing
    // about which cell the agent lands in.
    let algebra = closure(&generators.generators, genome.qubits, 4096);

    let ceiling = policy.ansatz.frequency_ceiling(0) as f64;
    let lambda = policy
        .ansatz
        .lambda_range()
        .next()
        .map(|i| params[i].abs())
        .unwrap_or(1.0);

    let state = policy.ansatz.build(&[REFERENCE_OBSERVATION]).run(params);
    let cut = genome.qubits / 2;
    let bond_dimension = overtone_mps::exact_bond_dimension(&state, 1e-8).max(1);
    let entropy = if cut == 0 {
        0.0
    } else {
        overtone_spec::entropy::half_chain_entropy(&state)
    };

    let fitness = env.expected_return(nodes, |s| policy.prob_action1(params, &[s]));

    (
        Behaviour {
            dim_g: algebra.dim(),
            reach: ceiling * lambda,
            bond_dimension,
            entropy,
        },
        fitness,
    )
}

/// The full dequantization verdict for one agent. Expensive — a sweep over every bond
/// dimension at every quadrature node — so it is run on the finished elites, never inside
/// the search loop.
pub fn dequantize_elite(
    genome: &Genome,
    k: usize,
    max_chi: usize,
    tolerance: f64,
) -> overtone_mps::DequantizeReport {
    let policy = genome.policy();
    let params = genome.params.clone();
    overtone_mps::dequantize(
        genome.qubits,
        k,
        64,
        max_chi,
        tolerance,
        |s| policy.ansatz.build(&[s]).run(&params),
        |st| policy.prob_from_observable(&params, policy.observable.expectation(st)),
    )
}

/// True when this agent's fitness is meaningless to compare — used to keep an unreachable
/// cell from being reported as a discovery.
pub fn is_degenerate(b: &Behaviour) -> bool {
    b.reach < 1e-9
}
