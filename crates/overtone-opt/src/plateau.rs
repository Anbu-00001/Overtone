//! The objective the four optimisers race on: a global observable, deep in a plateau,
//! measured with a finite number of shots.

use overtone_sim::grad::adjoint;
use overtone_sim::{Circuit, Observable, StateVec};
use overtone_spec::plateau::{hardware_efficient, CostLocality, DepthPolicy};
use overtone_spec::qfim::qfim;
use rand_chacha::ChaCha8Rng;

use crate::optim::{Differentiable, Objective};
use overtone_sim::measure::{Budget, DiagonalTerm};

/// A variational cost `<O>(theta)`, metered in shots.
pub struct PlateauCost {
    circuit: Circuit,
    observable: Observable,
    term: DiagonalTerm,
    budget: Budget,
    allowance: u64,
    spent: u64,
    rng: ChaCha8Rng,
}

impl PlateauCost {
    /// `n` qubits under `depth`, with the global or local observable.
    ///
    /// `allowance` is the total shot budget for one optimiser run; `budget` is what a single
    /// expectation value costs.
    pub fn new(
        n: usize,
        depth: DepthPolicy,
        locality: CostLocality,
        budget: Budget,
        allowance: u64,
        rng: ChaCha8Rng,
    ) -> PlateauCost {
        let circuit = hardware_efficient(n, depth.depth_for(n));
        let observable = locality.observable(n);
        let term = DiagonalTerm::from_observable(&observable)
            .expect("both plateau observables are products of Z");
        PlateauCost {
            circuit,
            observable,
            term,
            budget,
            allowance,
            spent: 0,
            rng,
        }
    }

    pub fn num_params(&self) -> usize {
        self.circuit.num_params()
    }

    fn state(&self, x: &[f64]) -> StateVec {
        self.circuit.run(x)
    }

    /// `<O>(x)` to machine precision, off the meter.
    ///
    /// The scoring function, never the optimising one. An optimiser scored on its own noisy
    /// evaluations is scored partly on luck; scored on this, it is scored on where it
    /// actually ended up.
    pub fn exact(&self, x: &[f64]) -> f64 {
        self.term.exact(&self.state(x))
    }

    /// The standard error of one measurement at `x`, the floor any real progress has to
    /// clear.
    pub fn noise_floor(&self, x: &[f64]) -> f64 {
        match self.budget {
            Budget::Exact => 0.0,
            Budget::Shots(s) => self.term.standard_error(&self.state(x), s),
        }
    }

    pub fn remaining(&self) -> u64 {
        self.allowance.saturating_sub(self.spent)
    }
}

impl Objective for PlateauCost {
    fn value(&mut self, x: &[f64]) -> f64 {
        let psi = self.state(x);
        self.spent += self.budget.cost();
        self.term.estimate(&psi, self.budget, &mut self.rng)
    }

    fn spent(&self) -> u64 {
        self.spent
    }

    fn budget(&self) -> u64 {
        self.allowance
    }

    fn dim(&self) -> usize {
        self.circuit.num_params()
    }
}

impl Differentiable for PlateauCost {
    /// Parameter-shift, with every shifted expectation paid for in shots.
    ///
    /// This is the hardware-honest path: `2P` measurements for `P` parameters. The adjoint
    /// gradient exists in `overtone-sim` and is exact, but charging nothing for it would
    /// make the gradient methods win a race about measurement cost by not measuring.
    fn gradient(&mut self, x: &[f64]) -> Vec<f64> {
        let p = self.circuit.num_params();
        let mut g = vec![0.0; p];
        let mut shifted = x.to_vec();
        for i in 0..p {
            let original = shifted[i];
            shifted[i] = original + std::f64::consts::FRAC_PI_2;
            let plus = self.value(&shifted);
            shifted[i] = original - std::f64::consts::FRAC_PI_2;
            let minus = self.value(&shifted);
            shifted[i] = original;
            g[i] = 0.5 * (plus - minus);
        }
        g
    }

    fn metric(&mut self, x: &[f64]) -> Option<Vec<f64>> {
        Some(qfim(&self.circuit, x))
    }
}

impl PlateauCost {
    pub fn circuit(&self) -> &Circuit {
        &self.circuit
    }

    pub fn observable(&self) -> &Observable {
        &self.observable
    }

    /// The exact adjoint gradient at `x`, off the meter.
    ///
    /// Used by `tests/` to confirm that the metered parameter-shift path is the *same*
    /// gradient measured expensively, not a different one measured cheaply.
    pub fn exact_gradient(&self, x: &[f64]) -> Vec<f64> {
        adjoint::grad(&self.circuit, x, &self.observable)
    }
}
