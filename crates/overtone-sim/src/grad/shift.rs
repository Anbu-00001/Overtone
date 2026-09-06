//! The parameter-shift rule.
//!
//! Exact, not a finite difference, and evaluable on hardware. Slow: two circuit
//! evaluations per parameterised gate, so `O(P)` where the adjoint path is `O(1)` passes.
//! It exists to keep the adjoint path honest.
//!
//! For a gate `U(t) = exp(-i t G / 2)` whose generator squares to the identity,
//!
//! ```text
//!   d<M>/dt = 1/2 [ <M>(t + pi/2) - <M>(t - pi/2) ].
//! ```
//!
//! Two details are easy to get wrong and are handled explicitly here.
//!
//! 1. **The shift is per gate, not per parameter.** When one parameter drives several
//!    gates, each gate is shifted alone and the results are summed. Shifting every gate
//!    that shares the parameter simultaneously computes a different quantity.
//! 2. **The chain rule is separate from the shift.** An encoding gate rotates by
//!    `lambda * s`, so the shift displaces the *angle* by `pi/2` and the resulting
//!    derivative is multiplied by `s`. When `s` is zero the gradient is zero, which the
//!    formula gives for free.

use std::f64::consts::FRAC_PI_2;

use crate::circuit::Circuit;
use crate::observable::Observable;

/// Compute `d<M>/dtheta` by the parameter-shift rule.
pub fn grad(circuit: &Circuit, params: &[f64], obs: &Observable) -> Vec<f64> {
    let mut grad = vec![0.0; params.len()];

    for gate_index in circuit.parameterised_gates() {
        let (index, chain) = circuit.gates()[gate_index]
            .param_ref()
            .expect("parameterised_gates only yields gates with a parameter");

        let plus = obs.expectation(&circuit.run_with_gate_offset(params, gate_index, FRAC_PI_2));
        let minus = obs.expectation(&circuit.run_with_gate_offset(params, gate_index, -FRAC_PI_2));

        grad[index] += chain * 0.5 * (plus - minus);
    }

    grad
}

/// A central finite difference on the parameter vector.
///
/// Not used for training. It is a third opinion: if the adjoint and the parameter-shift
/// path ever agree with each other but disagree with this, the shared assumption between
/// them is what is wrong.
pub fn finite_difference_grad(
    circuit: &Circuit,
    params: &[f64],
    obs: &Observable,
    eps: f64,
) -> Vec<f64> {
    let mut grad = vec![0.0; params.len()];
    let mut probe = params.to_vec();

    for i in 0..params.len() {
        let original = probe[i];

        probe[i] = original + eps;
        let plus = obs.expectation(&circuit.run(&probe));

        probe[i] = original - eps;
        let minus = obs.expectation(&circuit.run(&probe));

        probe[i] = original;
        grad[i] = (plus - minus) / (2.0 * eps);
    }

    grad
}
