//! Circuits: an ordered gate list over a fixed qubit register.

use crate::gate::{Angle, Gate};
use crate::state::StateVec;

/// An ordered list of gates on `n` qubits, parameterised by a flat vector.
///
/// Several gates may reference the same parameter index. That is not an edge case: it is
/// how a pinned or shared input scaling is expressed. Both gradient paths accumulate
/// across every gate that references a parameter.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Circuit {
    n: usize,
    gates: Vec<Gate>,
    num_params: usize,
}

impl Circuit {
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "a circuit needs at least one qubit");
        Circuit {
            n,
            gates: Vec::new(),
            num_params: 0,
        }
    }

    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// One past the highest parameter index referenced by any gate.
    #[inline]
    pub fn num_params(&self) -> usize {
        self.num_params
    }

    #[inline]
    pub fn gates(&self) -> &[Gate] {
        &self.gates
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.gates.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.gates.is_empty()
    }

    /// Append a gate, validating its qubits and widening the parameter count.
    pub fn push(&mut self, gate: Gate) -> &mut Self {
        for q in gate.qubits() {
            assert!(
                q < self.n,
                "gate {gate:?} touches qubit {q}, outside a {}-qubit register",
                self.n
            );
        }
        if let Some((index, _)) = gate.param_ref() {
            self.num_params = self.num_params.max(index + 1);
        }
        self.gates.push(gate);
        self
    }

    pub fn rx(&mut self, q: usize, angle: Angle) -> &mut Self {
        self.push(Gate::Rx { q, angle })
    }

    pub fn ry(&mut self, q: usize, angle: Angle) -> &mut Self {
        self.push(Gate::Ry { q, angle })
    }

    pub fn rz(&mut self, q: usize, angle: Angle) -> &mut Self {
        self.push(Gate::Rz { q, angle })
    }

    pub fn h(&mut self, q: usize) -> &mut Self {
        self.push(Gate::H { q })
    }

    pub fn cz(&mut self, a: usize, b: usize) -> &mut Self {
        self.push(Gate::Cz { a, b })
    }

    pub fn cnot(&mut self, control: usize, target: usize) -> &mut Self {
        self.push(Gate::Cnot { control, target })
    }

    /// Indices, into `gates()`, of the gates that carry a trainable angle.
    pub fn parameterised_gates(&self) -> Vec<usize> {
        (0..self.gates.len())
            .filter(|&i| self.gates[i].param_ref().is_some())
            .collect()
    }

    /// Run the circuit on `|0...0>`.
    pub fn run(&self, params: &[f64]) -> StateVec {
        let mut st = StateVec::zero(self.n);
        self.run_on(&mut st, params);
        st
    }

    /// Run the circuit on an existing state, in place.
    pub fn run_on(&self, st: &mut StateVec, params: &[f64]) {
        assert_eq!(
            st.num_qubits(),
            self.n,
            "state has {} qubits, circuit has {}",
            st.num_qubits(),
            self.n
        );
        assert!(
            params.len() >= self.num_params,
            "circuit needs {} parameters, got {}",
            self.num_params,
            params.len()
        );
        for gate in &self.gates {
            gate.apply(st, params);
        }
    }

    /// Run the circuit with one gate's angle displaced by `offset`.
    ///
    /// This is the primitive the parameter-shift rule is built on. The displacement is
    /// per *gate*, not per parameter: when a parameter drives several gates, each gate is
    /// shifted on its own and the contributions are summed. Shifting them together would
    /// compute something else entirely.
    pub fn run_with_gate_offset(&self, params: &[f64], gate_index: usize, offset: f64) -> StateVec {
        let mut st = StateVec::zero(self.n);
        for (i, gate) in self.gates.iter().enumerate() {
            let o = if i == gate_index { offset } else { 0.0 };
            gate.apply_with_offset(&mut st, params, o);
        }
        st
    }
}
