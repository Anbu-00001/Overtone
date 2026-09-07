//! Evolution in the DLA basis: Givens rotations on a `dim(g)`-vector of expectation values.
//!
//! Part III 3, and Goh, Larocca, Cincio, Cerezo and Sauvage, *Lie-algebraic classical
//! simulations for quantum computing*, Phys. Rev. Research 7, 033266 (2025). Instead of a
//! `2^n` state vector, carry `e_alpha = <psi| b_alpha |psi>` for each basis element of the
//! algebra. When `dim(g)` is polynomial this is an exact simulator that scales.
//!
//! # The rotation, derived
//!
//! For a gate `R = exp(-i theta G / 2)` and a basis element `P` that anticommutes with `G`,
//!
//! ```text
//! R^dagger P R = (c^2 - s^2) P - 2 i c s P G     with c = cos(theta/2), s = sin(theta/2)
//!              = cos(theta) P + sigma sin(theta) Q
//! ```
//!
//! where `[P, G] = 2i sigma Q`. The half-angle disappears: the rotation in the
//! `span{P, Q}` plane is by `theta`, not `theta/2`. Elements that commute with `G` are
//! untouched, which is why a gate costs `O(dim g)` and not `O(dim(g)^2)`.
//!
//! Sanity check against the one everybody knows: `G = Z`, `P = X`. `[X, Z] = -2iY`, so
//! `sigma = -1` and `Q = Y`, giving `RZ^dagger X RZ = cos(theta) X - sin(theta) Y`.

use overtone_lie::{Algebra, PauliString};
use overtone_sim::Angle;

/// One plane of the rotation a generator induces: indices `(alpha, gamma)` with
/// `alpha < gamma`, and the sign from `[b_alpha, G] = 2i sigma b_gamma`.
#[derive(Clone, Copy, Debug)]
struct Plane {
    alpha: u32,
    gamma: u32,
    sigma: f64,
}

/// A gate: a generator drawn from the algebra, and where its angle comes from.
#[derive(Clone, Debug)]
pub struct DlaGate {
    /// Index into [`GsimCircuit::generators`].
    pub generator: usize,
    pub angle: Angle,
}

/// A circuit whose gates are all one-parameter subgroups of `exp(g)`.
#[derive(Clone, Debug)]
pub struct GsimCircuit {
    algebra: Algebra,
    generators: Vec<PauliString>,
    planes: Vec<Vec<Plane>>,
    gates: Vec<DlaGate>,
    num_params: usize,
}

impl GsimCircuit {
    /// Precompute the rotation planes of each generator. `O(dim(g))` per generator, once.
    ///
    /// Panics if a generator is not in the algebra: that would mean the circuit leaves
    /// `exp(g)` and the simulation would be silently wrong rather than approximate.
    pub fn new(algebra: Algebra, generators: Vec<PauliString>) -> Self {
        let mut planes = Vec::with_capacity(generators.len());
        for g in &generators {
            assert!(
                algebra.contains(g),
                "generator {} is not in the algebra",
                g.render(algebra.num_qubits())
            );
            let pairing = algebra.pairing(g);
            let mut ps = Vec::new();
            for (alpha, entry) in pairing.iter().enumerate() {
                if let Some((gamma, sigma)) = *entry {
                    if alpha < gamma {
                        // The pairing must be an involution with opposite signs, or the
                        // 2x2 block would not be a rotation and the evolution would not
                        // preserve the state's purity.
                        let back = pairing[gamma].expect("pairing is not involutive");
                        debug_assert_eq!(back.0, alpha, "pairing is not involutive");
                        debug_assert_eq!(back.1, -sigma, "pairing signs did not oppose");
                        ps.push(Plane {
                            alpha: alpha as u32,
                            gamma: gamma as u32,
                            sigma: sigma as f64,
                        });
                    }
                }
            }
            planes.push(ps);
        }
        GsimCircuit {
            algebra,
            generators,
            planes,
            gates: Vec::new(),
            num_params: 0,
        }
    }

    pub fn algebra(&self) -> &Algebra {
        &self.algebra
    }

    pub fn generators(&self) -> &[PauliString] {
        &self.generators
    }

    pub fn gates(&self) -> &[DlaGate] {
        &self.gates
    }

    pub fn dim(&self) -> usize {
        self.algebra.dim()
    }

    pub fn num_params(&self) -> usize {
        self.num_params
    }

    pub fn push(&mut self, generator: usize, angle: Angle) -> &mut Self {
        assert!(generator < self.generators.len());
        if let Angle::Param { index, .. } = angle {
            self.num_params = self.num_params.max(index + 1);
        }
        self.gates.push(DlaGate { generator, angle });
        self
    }

    /// Push a gate whose angle is supplied per call rather than by a parameter index.
    ///
    /// This is what an encoding gate needs: its angle is `lambda * s`, and `s` changes
    /// every observation. Rebuilding the circuit each time would mean recomputing the
    /// rotation planes, which is the expensive part; supplying the angle instead costs
    /// nothing. Circuits built this way must be driven through [`Self::evolve_angles`] and
    /// [`Self::value_and_grad_per_gate`] -- the parameter-indexed entry points see the
    /// placeholder angle, not the real one.
    pub fn push_raw(&mut self, generator: usize) -> &mut Self {
        assert!(generator < self.generators.len());
        self.gates.push(DlaGate {
            generator,
            angle: Angle::Fixed(f64::NAN),
        });
        self
    }

    /// `e_alpha = <0...0| b_alpha |0...0>`: one on the Z-type strings, zero elsewhere.
    pub fn initial_vector(&self) -> Vec<f64> {
        self.algebra
            .basis()
            .iter()
            .map(|p| p.expectation_in_zero_state())
            .collect()
    }

    /// Index of a Pauli observable in the algebra, or `None` if it is outside — in which
    /// case g-sim cannot see it at all, which is a fact about the method, not a bug.
    pub fn observable_index(&self, o: &PauliString) -> Option<usize> {
        self.algebra.index_of(o)
    }

    fn apply(&self, v: &mut [f64], generator: usize, cos: f64, sin: f64) {
        for p in &self.planes[generator] {
            let (a, c) = (p.alpha as usize, p.gamma as usize);
            let (ea, eg) = (v[a], v[c]);
            v[a] = cos * ea + sin * p.sigma * eg;
            v[c] = cos * eg - sin * p.sigma * ea;
        }
    }

    /// The angle of each gate, resolved against `params`.
    pub fn gate_angles(&self, params: &[f64]) -> Vec<f64> {
        self.gates.iter().map(|g| g.angle.value(params)).collect()
    }

    /// Evolve the expectation vector, one angle per gate.
    pub fn evolve_angles(&self, gate_angles: &[f64]) -> Vec<f64> {
        let mut e = self.initial_vector();
        for (gate, t) in self.gates.iter().zip(gate_angles) {
            self.apply(&mut e, gate.generator, t.cos(), t.sin());
        }
        e
    }

    /// Evolve the expectation vector through the whole circuit.
    pub fn evolve(&self, params: &[f64]) -> Vec<f64> {
        self.evolve_angles(&self.gate_angles(params))
    }

    /// `<O>` for an observable given as coefficients over the algebra basis.
    pub fn expectation(&self, params: &[f64], weights: &[f64]) -> f64 {
        let e = self.evolve(params);
        weights.iter().zip(&e).map(|(w, x)| w * x).sum()
    }

    /// `<O>` and its derivative with respect to each *gate angle*, by an adjoint pass.
    ///
    /// One forward pass, then one backward pass that un-rotates the state and rotates the
    /// observable covector, exactly mirroring the state-vector adjoint in `overtone-sim`.
    /// Cost is `O(gates * dim(g))` for the whole gradient, not per parameter. Mapping gate
    /// angles back to parameters is the caller's chain rule, which is what lets an encoding
    /// gate carry a per-observation scale without rebuilding the circuit.
    pub fn value_and_grad_per_gate(&self, gate_angles: &[f64], weights: &[f64]) -> (f64, Vec<f64>) {
        let mut e = self.evolve_angles(gate_angles);
        let value: f64 = weights.iter().zip(&e).map(|(w, x)| w * x).sum();
        let mut b = weights.to_vec();
        let mut grad = vec![0.0; self.gates.len()];

        for (k, gate) in self.gates.iter().enumerate().rev() {
            let (c, s) = (gate_angles[k].cos(), gate_angles[k].sin());
            // Undo this gate, leaving the state as it was before it.
            self.apply(&mut e, gate.generator, c, -s);
            // b . (dR/dtheta) e. The derivative of the rotation is zero outside the planes,
            // not the identity, which is what reusing `apply` here would wrongly give.
            let mut d = 0.0;
            for p in &self.planes[gate.generator] {
                let (a, g) = (p.alpha as usize, p.gamma as usize);
                d += b[a] * (-s * e[a] + c * p.sigma * e[g])
                    + b[g] * (-s * e[g] - c * p.sigma * e[a]);
            }
            grad[k] = d;
            // Pull the observable back through the gate: R^T = R(-theta).
            self.apply(&mut b, gate.generator, c, -s);
        }
        (value, grad)
    }

    /// `<O>` and its gradient with respect to the circuit parameters.
    pub fn value_and_grad(&self, params: &[f64], weights: &[f64]) -> (f64, Vec<f64>) {
        let angles = self.gate_angles(params);
        let (value, per_gate) = self.value_and_grad_per_gate(&angles, weights);
        let mut grad = vec![0.0; params.len()];
        for (k, gate) in self.gates.iter().enumerate() {
            if let Some((index, scale)) = gate.angle.param_ref() {
                grad[index] += scale * per_gate[k];
            }
        }
        (value, grad)
    }
}
