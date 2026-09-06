//! Adjoint differentiation.
//!
//! Constant memory in circuit depth, every parameter in roughly two forward passes. This
//! is the fast path; [`crate::grad::shift`] is the hardware-honest one, and the two must
//! agree to `1e-10` (Part I 6.4).
//!
//! # Derivation
//!
//! Write the circuit as `|psi> = U_m ... U_1 |0>` and the cost as `<M> = <psi|M|psi>`.
//! For parameter `theta_i` driving gate `i`,
//!
//! ```text
//!   d<M>/dtheta_i = <dpsi|M|psi> + <psi|M|dpsi> = 2 Re <psi|M|dpsi>,
//! ```
//!
//! because `M` is Hermitian and the two terms are complex conjugates. Splitting the
//! circuit at gate `i` and defining
//!
//! ```text
//!   |k_i> = U_{i-1} ... U_1 |0>            (the state entering gate i)
//!   |b_i> = U_{i+1}^dag ... U_m^dag M|psi> (the observable pulled back to gate i)
//! ```
//!
//! gives `d<M>/dtheta_i = 2 Re <b_i| dU_i/dtheta_i |k_i>`.
//!
//! Both `|k>` and `|b>` are obtained by walking backwards from the end, un-applying one
//! gate at a time, so a single reverse sweep produces every gradient. The ordering inside
//! the loop matters: `|k>` is un-applied *before* the gradient is read (it must be the
//! state entering the gate) and `|b>` *after* (it must still be the pullback past the
//! gate). Swapping either is a silent off-by-one-gate error.

use crate::circuit::Circuit;
use crate::observable::Observable;
use crate::state::StateVec;

/// The expectation value and its gradient with respect to every circuit parameter.
#[derive(Clone, Debug, PartialEq)]
pub struct ValueGrad {
    pub value: f64,
    pub grad: Vec<f64>,
}

/// Compute `<M>` and `d<M>/dtheta` by adjoint differentiation.
pub fn value_and_grad(circuit: &Circuit, params: &[f64], obs: &Observable) -> ValueGrad {
    let ket_final = circuit.run(params);
    let value = obs.expectation(&ket_final);

    // Three vectors live at once: the ket, the pulled-back bra, and one scratch buffer for
    // the gate derivative. All allocated here, none inside the loop.
    let mut ket = ket_final.clone();
    let mut bra = ket_final.clone();
    let mut scratch = ket_final.clone();
    obs.apply_into(&ket_final, &mut bra, &mut scratch);

    let mut grad = vec![0.0; params.len()];

    for gate in circuit.gates().iter().rev() {
        // |k> now enters this gate.
        gate.apply_adjoint(&mut ket, params);

        if let Some((index, chain)) = gate.param_ref() {
            scratch.copy_from(&ket);
            gate.apply_derivative(&mut scratch, params);
            // chain = d(angle)/d(parameter), which is the observation value for an
            // encoding gate RX(lambda * s) and 1 for a plain variational rotation.
            grad[index] += 2.0 * chain * bra.inner_re(&scratch);
        }

        // |b> is pulled back past this gate only after it has been used.
        gate.apply_adjoint(&mut bra, params);
    }

    ValueGrad { value, grad }
}

/// Gradient only.
pub fn grad(circuit: &Circuit, params: &[f64], obs: &Observable) -> Vec<f64> {
    value_and_grad(circuit, params, obs).grad
}

/// The final state and the expectation value, without differentiating.
pub fn value(circuit: &Circuit, params: &[f64], obs: &Observable) -> (StateVec, f64) {
    let st = circuit.run(params);
    let v = obs.expectation(&st);
    (st, v)
}
