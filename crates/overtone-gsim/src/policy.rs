//! A RAW-PQC policy that never builds a state vector.
//!
//! The same construction as Part I 6.2 and 6.3 -- data re-uploading, encoding angle
//! `lambda * s`, Born rule on a single Pauli -- but evolved in the DLA basis, so the qubit
//! count is limited by `dim(g)` rather than by `2^n`.
//!
//! The generators are the transverse-field Ising set `{X_i} u {Z_i Z_{i+1}}`, whose algebra
//! is `so(2n)` of dimension `n(2n - 1)`. At `n = 100` that is 19900 numbers instead of
//! `2^100` amplitudes, and the simulation is exact rather than approximate.

use overtone_lie::{closure_unbounded, family, PauliString};
use overtone_sim::Pauli;

use crate::evolve::GsimCircuit;

/// Where a gate's angle comes from, once an observation is known.
#[derive(Clone, Copy, Debug)]
enum Source {
    /// `lambda * s`, the data re-uploading encoding. `lambda` is parameter zero.
    Encoding,
    /// A free variational angle.
    Variational(usize),
}

/// Which observable the Born rule reads. Both lie in the transverse-field Ising algebra,
/// so both are visible to g-sim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyObservable {
    /// `X_0`. Local, and after one entangling layer it depends on two qubits only -- the
    /// light cone, not the qubit count, sets the reachable frequency.
    FirstX,
    /// `(1/n) sum_q X_q`. Every encoding gate reaches it immediately -- but its expectation
    /// is an **odd** function of the observation, for any depth and any parameters, so a
    /// `SpectralControl` return built on `cos(k s)` is identically zero. See
    /// [`GsimPolicy`] for why; it is a theorem about this ansatz, not a training failure.
    MeanX,
    /// `(1/(n-1)) sum_q Z_q Z_{q+1}`. Carries no parity law, so it has a nonzero even
    /// component and can score. This is the observable the hundred-qubit run uses.
    MeanZz,
}

/// Shape of the policy circuit.
#[derive(Clone, Copy, Debug)]
pub struct PolicyLayout {
    pub num_qubits: usize,
    pub layers: usize,
    /// Hold `lambda` at its initial value instead of training it (Part I 6.2's ablation).
    pub lambda_trainable: bool,
    pub observable: PolicyObservable,
}

/// A band-limited policy simulated in the algebra.
///
/// # A parity theorem, found by the return being exactly zero
///
/// With generators `{X_q} u {Z_qZ_{q+1}}`, observable `sum_q X_q` and **one** layer, the
/// expectation is an odd function of the observation `s` for every parameter setting, so
/// the `SpectralControl` return -- a correlation against `cos(k s)` -- vanishes
/// identically. The first hundred-qubit run scored exactly zero with a zero gradient in
/// every direction for this reason.
///
/// The proof is a conserved quantity. Multiplying a Pauli string by `Z_aZ_b` toggles
/// `X <-> Y` and `I <-> Z` at both sites, and a string only fails to commute with `Z_aZ_b`
/// when exactly one of the two sites carries an `X` or a `Y`; so every `Z_aZ_b` gate leaves
/// the number of `X`-or-`Y` letters unchanged. Starting from `X_q` that count is one
/// forever. An encoding gate `RX(lambda s)` leaves `X` alone and mixes `Y` with `Z` at the
/// cost of one factor of `sin(lambda s)`, and only strings of `I` and `Z` survive in
/// `|0...0>`; so every surviving term carries exactly one `Y`, hence exactly one sine, hence
/// is odd.
///
/// The argument needs the single layer. A second layer gives the variational `RX` gates a
/// string with a `Z` to act on, the count can then visit two, and the parity is no longer
/// pinned -- measured, not assumed: two layers score.
///
/// [`PolicyObservable::MeanZz`] starts at count zero instead of one and shows no such
/// symmetry, which is why the hundred-qubit run uses it.
pub struct GsimPolicy {
    circuit: GsimCircuit,
    sources: Vec<Source>,
    weights: Vec<f64>,
    layout: PolicyLayout,
    num_params: usize,
}

impl GsimPolicy {
    /// Build the ansatz and close its algebra. The closure is the expensive part and it
    /// happens once; the planes it produces are what every later evaluation reuses.
    pub fn new(layout: PolicyLayout) -> Self {
        let n = layout.num_qubits;
        assert!(n >= 2, "a chain needs two sites");
        let algebra = closure_unbounded(&family::tfim(n), n);
        // Generators, in a fixed order: X_0..X_{n-1}, then Z_0Z_1..Z_{n-2}Z_{n-1}.
        let mut generators: Vec<PauliString> =
            (0..n).map(|q| PauliString::single(q, Pauli::X)).collect();
        generators.extend(
            (0..n - 1).map(|q| PauliString::from_factors(&[(q, Pauli::Z), (q + 1, Pauli::Z)])),
        );
        let mut circuit = GsimCircuit::new(algebra, generators);

        // Parameter zero is lambda; the variational angles follow.
        let mut sources = Vec::new();
        let mut next_param = 1usize;
        for _ in 0..layout.layers {
            for q in 0..n {
                circuit.push_raw(q);
                sources.push(Source::Encoding);
            }
            for q in 0..n - 1 {
                circuit.push_raw(n + q);
                sources.push(Source::Variational(next_param));
                next_param += 1;
            }
            for q in 0..n {
                circuit.push_raw(q);
                sources.push(Source::Variational(next_param));
                next_param += 1;
            }
        }

        // The Born rule reads a Pauli, or an average of them; both are in g by
        // construction, since every X_q is a generator.
        let mut weights = vec![0.0; circuit.dim()];
        match layout.observable {
            PolicyObservable::FirstX => {
                let i = circuit
                    .observable_index(&PauliString::single(0, Pauli::X))
                    .expect("X_0 generates the algebra and must be in it");
                weights[i] = 1.0;
            }
            PolicyObservable::MeanX => {
                for q in 0..n {
                    let i = circuit
                        .observable_index(&PauliString::single(q, Pauli::X))
                        .expect("X_q generates the algebra and must be in it");
                    weights[i] = 1.0 / n as f64;
                }
            }
            PolicyObservable::MeanZz => {
                for q in 0..n - 1 {
                    let i = circuit
                        .observable_index(&PauliString::from_factors(&[
                            (q, Pauli::Z),
                            (q + 1, Pauli::Z),
                        ]))
                        .expect("Z_qZ_{q+1} generates the algebra and must be in it");
                    weights[i] = 1.0 / (n - 1) as f64;
                }
            }
        }

        GsimPolicy {
            circuit,
            sources,
            weights,
            layout,
            num_params: next_param,
        }
    }

    pub fn num_params(&self) -> usize {
        self.num_params
    }

    pub fn layout(&self) -> PolicyLayout {
        self.layout
    }

    pub fn dim_g(&self) -> usize {
        self.circuit.dim()
    }

    pub fn num_gates(&self) -> usize {
        self.circuit.gates().len()
    }

    /// The reachable frequency ceiling in units of `lambda`: one per encoding gate, since
    /// each contributes eigenvalue gaps of one under the `exp(-i t X / 2)` convention.
    pub fn frequency_ceiling(&self) -> usize {
        self.layout.num_qubits * self.layout.layers
    }

    fn angles(&self, params: &[f64], s: f64) -> Vec<f64> {
        self.sources
            .iter()
            .map(|src| match src {
                Source::Encoding => params[0] * s,
                Source::Variational(i) => params[*i],
            })
            .collect()
    }

    /// `<O>` for observation `s`.
    pub fn observable_value(&self, params: &[f64], s: f64) -> f64 {
        let angles = self.angles(params, s);
        let e = self.circuit.evolve_angles(&angles);
        self.weights.iter().zip(&e).map(|(w, x)| w * x).sum()
    }

    /// The Born rule: `pi(1|s) = (1 + <O>) / 2`. Strictly band-limited by construction.
    pub fn prob_action1(&self, params: &[f64], s: f64) -> f64 {
        0.5 * (1.0 + self.observable_value(params, s))
    }

    /// `pi(1|s)` and its gradient with respect to every parameter.
    pub fn prob_and_grad(&self, params: &[f64], s: f64) -> (f64, Vec<f64>) {
        let angles = self.angles(params, s);
        let (value, per_gate) = self.circuit.value_and_grad_per_gate(&angles, &self.weights);
        let mut grad = vec![0.0; self.num_params];
        for (k, src) in self.sources.iter().enumerate() {
            match src {
                // d(lambda * s)/d lambda = s.
                Source::Encoding => {
                    if self.layout.lambda_trainable {
                        grad[0] += s * per_gate[k];
                    }
                }
                Source::Variational(i) => grad[*i] += per_gate[k],
            }
        }
        for g in &mut grad {
            *g *= 0.5;
        }
        (0.5 * (1.0 + value), grad)
    }

    /// `grad log pi(a|s)`, the REINFORCE score function.
    pub fn grad_log_prob(&self, params: &[f64], s: f64, action: usize) -> Vec<f64> {
        let (p1, grad) = self.prob_and_grad(params, s);
        let (p, sign) = if action == 1 {
            (p1, 1.0)
        } else {
            (1.0 - p1, -1.0)
        };
        let denom = p.max(1e-12);
        grad.into_iter().map(|g| sign * g / denom).collect()
    }
}
