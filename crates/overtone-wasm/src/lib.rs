//! The browser surface.
//!
//! This crate is a shim. Part I 5: "if it contains an `if` statement about physics, that
//! logic is in the wrong crate." Everything here either forwards a call or reshapes a
//! result into a flat `Vec<f64>`, which `wasm-bindgen` hands to JavaScript as a
//! `Float64Array` without a copy on the JS side.
//!
//! The shape of this API is set by the JS line budget (Part I 4: under 800 lines, enforced
//! in CI). Every quantity a panel needs is returned render-ready — already normalised,
//! already ordered, already in the units the axis uses — so the renderer never computes,
//! it only draws. Where a panel needs a derived index, such as which spectral bars sit
//! above the ceiling, that index is returned too rather than recomputed in JS.

use wasm_bindgen::prelude::*;

use overtone_rl::ceiling;
use overtone_rl::policy::Policy;
use overtone_rl::reinforce::{coarse_tune_lambda, initial_params, reinforce_gradient, Adam};
use overtone_rl::{Scaling, SpectralControl, SpectralControlAnsatz};
use overtone_sim::grad::{adjoint, shift};
use overtone_sim::random::rng;
use overtone_sim::{Gate, Observable};
use overtone_spec::plateau::{fit_exponential, sweep, CostLocality, DepthPolicy};
use overtone_spec::spectrum_of;
use rand_chacha::ChaCha8Rng;

/// Gate kinds, as a numeric code for the circuit panel. JavaScript switches on the number
/// rather than parsing a string.
const GATE_RX: f64 = 0.0;
const GATE_RY: f64 = 1.0;
const GATE_RZ: f64 = 2.0;
const GATE_H: f64 = 3.0;
const GATE_CZ: f64 = 4.0;
const GATE_CNOT: f64 = 5.0;

/// The `Lab` tab: one policy, trained live, with every Part I instrument reading off it.
#[wasm_bindgen]
pub struct Lab {
    policy: Policy,
    params: Vec<f64>,
    env: SpectralControl,
    optimiser: Adam,
    rng: ChaCha8Rng,
    trace: Vec<f64>,
    episodes: usize,
    batch_size: usize,
    seed: u64,
}

#[wasm_bindgen]
impl Lab {
    /// Build a lab.
    ///
    /// `policy_kind` is 0 for RAW-PQC and 1 for SOFTMAX-PQC.
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        qubits: usize,
        layers: usize,
        k: usize,
        policy_kind: u8,
        trainable_lambda: bool,
        entangle: bool,
        beta: f64,
        learning_rate: f64,
        // u32, not u64: wasm-bindgen maps u64 to BigInt, and forcing the renderer to
        // construct BigInts to seed a run would be ergonomics leaking out of the shim.
        seed: u32,
    ) -> Lab {
        let scaling = if trainable_lambda {
            Scaling::Trainable
        } else {
            Scaling::Pinned
        };
        let ansatz = SpectralControlAnsatz::build(qubits, layers, scaling, entangle);
        let policy = if policy_kind == 1 {
            Policy::softmax(ansatz, beta)
        } else {
            Policy::raw(ansatz)
        };

        let mut r = rng(seed as u64);
        let params = initial_params(&policy, 0.3, 1.0, &mut r);
        let optimiser = Adam::new(policy.num_params(), learning_rate);

        Lab {
            policy,
            params,
            env: SpectralControl::new(k),
            optimiser,
            rng: r,
            trace: Vec::new(),
            episodes: 0,
            batch_size: 50,
            seed: seed as u64,
        }
    }

    /// Start `lambda` inside the capture range of the environment frequency.
    ///
    /// The return is a resonance curve in `lambda`, so gradient ascent from the pinned-
    /// equivalent value of 1 locks onto a sidelobe rather than the main peak. This is the
    /// coarse tune; the training loop is the fine tune. Returns the chosen `lambda`.
    pub fn coarse_tune(&mut self) -> f64 {
        let range = self.policy.ansatz.lambda_range();
        if range.is_empty() {
            return 1.0;
        }
        let tuned = coarse_tune_lambda(&self.policy, &self.params, self.env.k, 8.0, 200);
        for i in range {
            self.params[i] = tuned;
        }
        tuned
    }

    /// Run `steps` REINFORCE gradient steps. Returns the exact return afterwards.
    pub fn train_steps(&mut self, steps: usize) -> f64 {
        for _ in 0..steps {
            let (grad, _) = reinforce_gradient(
                &self.policy,
                &self.params,
                &self.env,
                self.batch_size,
                &mut self.rng,
            );
            self.optimiser.ascend(&mut self.params, &grad);
            self.episodes += self.batch_size;
            let j = self.policy.expected_return(&self.params, self.env.k, 256);
            self.trace.push(j);
        }
        self.exact_return()
    }

    /// Restore the initial parameters for this seed and clear the trace.
    pub fn reset(&mut self) {
        let mut r = rng(self.seed);
        self.params = initial_params(&self.policy, 0.3, 1.0, &mut r);
        self.rng = r;
        self.optimiser = Adam::new(self.policy.num_params(), self.optimiser.learning_rate);
        self.trace.clear();
        self.episodes = 0;
    }

    pub fn episodes(&self) -> usize {
        self.episodes
    }

    /// Exact expected return by quadrature, not a sampled average.
    pub fn exact_return(&self) -> f64 {
        self.policy.expected_return(&self.params, self.env.k, 512)
    }

    /// The reachable frequency ceiling of the encoding.
    pub fn frequency_ceiling(&self) -> usize {
        self.policy.ansatz.frequency_ceiling(0)
    }

    /// The best any strictly band-limited policy could score here.
    pub fn lp_ceiling(&self) -> f64 {
        ceiling::ceiling(self.frequency_ceiling(), self.env.k, 801)
    }

    /// The unconstrained optimum, `2/pi`.
    pub fn optimal_return(&self) -> f64 {
        self.env.optimal_return()
    }

    /// Current input scaling, or 1 when pinned.
    pub fn lambda(&self) -> f64 {
        self.policy
            .ansatz
            .lambda_range()
            .map(|i| self.params[i])
            .next()
            .unwrap_or(1.0)
    }

    /// Whether `lambda` is trainable. Part I 6.2: this is a protagonist, not a detail.
    pub fn lambda_trainable(&self) -> bool {
        !self.policy.ansatz.lambda_range().is_empty()
    }

    /// The learning curve so far.
    pub fn return_trace(&self) -> Vec<f64> {
        self.trace.clone()
    }

    /// The LP ceiling staircase `J*(0..=max_c)`, drawn behind the learning curve.
    pub fn ceiling_staircase(&self, max_c: usize) -> Vec<f64> {
        (0..=max_c)
            .map(|c| ceiling::ceiling(c, self.env.k, 401))
            .collect()
    }

    /// `|c_omega|` for `omega = 0 ..= samples/2`, of `pi(1|s)`.
    ///
    /// The probability, not the logit. A SOFTMAX-PQC's logit is as band-limited as a
    /// RAW-PQC's, so transforming it would show no leakage at all.
    pub fn spectrum(&self, samples: usize) -> Vec<f64> {
        spectrum_of(samples, self.frequency_ceiling(), |s| {
            self.policy.prob_action1(&self.params, &[s])
        })
        .magnitude
    }

    /// Where the ceiling actually sits on the frequency axis, as a real number.
    ///
    /// With `lambda` pinned this is the integer `frequency_ceiling()`. With `lambda`
    /// trainable the reachable set is `lambda * {-C..C}`, so the ceiling **moves** with
    /// `lambda` rather than being exceeded — the whole point of Part I 1.3. Drawing the rule
    /// at the integer ceiling in that mode would report a policy that had tuned itself into
    /// resonance as though it had leaked, which is the opposite of what happened.
    pub fn effective_ceiling(&self) -> f64 {
        self.frequency_ceiling() as f64 * self.lambda().abs()
    }

    /// Whether a leakage measurement is meaningful for this policy.
    ///
    /// It is not when `lambda` is trainable: the reachable frequencies are then generally
    /// non-integer, so a discrete transform spreads them across neighbouring bins for
    /// ordinary sampling reasons that have nothing to do with the softmax. Reporting a
    /// number there would be measuring the instrument, not the policy.
    pub fn leakage_is_meaningful(&self) -> bool {
        !self.lambda_trainable()
    }

    /// Total spectral magnitude above the ceiling, over that in band. Exactly zero for a
    /// RAW-PQC; positive for a SOFTMAX-PQC. Only meaningful when
    /// [`Lab::leakage_is_meaningful`] holds.
    pub fn leakage_ratio(&self, samples: usize) -> f64 {
        spectrum_of(samples, self.frequency_ceiling(), |s| {
            self.policy.prob_action1(&self.params, &[s])
        })
        .leakage_ratio()
    }

    /// `pi(1|s)` sampled uniformly on `[-pi, pi)`.
    pub fn policy_curve(&self, samples: usize) -> Vec<f64> {
        let pi = std::f64::consts::PI;
        (0..samples)
            .map(|i| {
                let s = -pi + 2.0 * pi * (i as f64) / samples as f64;
                self.policy.prob_action1(&self.params, &[s])
            })
            .collect()
    }

    /// The optimal policy on the same grid, for the ghost behind the policy curve.
    pub fn optimal_curve(&self, samples: usize) -> Vec<f64> {
        let pi = std::f64::consts::PI;
        (0..samples)
            .map(|i| {
                let s = -pi + 2.0 * pi * (i as f64) / samples as f64;
                self.env.optimal_action(s) as f64
            })
            .collect()
    }

    /// Amplitudes of the circuit state at observation `s`, interleaved `re, im, re, im...`.
    ///
    /// The state panel draws bar height from magnitude and hue from phase, so both parts
    /// have to cross the boundary.
    pub fn state_amplitudes(&self, s: f64) -> Vec<f64> {
        let circuit = self.policy.ansatz.build(&[s]);
        let state = circuit.run(&self.params[..self.policy.ansatz.num_params()]);
        let mut out = Vec::with_capacity(state.dim() * 2);
        for i in 0..state.dim() {
            out.push(state.re[i]);
            out.push(state.im[i]);
        }
        out
    }

    /// Half-chain entanglement entropy at observation `s`, in nats.
    pub fn entropy(&self, s: f64) -> f64 {
        let circuit = self.policy.ansatz.build(&[s]);
        let state = circuit.run(&self.params[..self.policy.ansatz.num_params()]);
        if state.num_qubits() < 2 {
            return 0.0;
        }
        overtone_spec::half_chain_entropy(&state)
    }

    /// Gate kinds, one per gate, using the `GATE_*` codes.
    pub fn gate_kinds(&self) -> Vec<f64> {
        self.circuit_at_zero()
            .gates()
            .iter()
            .map(|g| match g {
                Gate::Rx { .. } => GATE_RX,
                Gate::Ry { .. } => GATE_RY,
                Gate::Rz { .. } => GATE_RZ,
                Gate::H { .. } => GATE_H,
                Gate::Cz { .. } => GATE_CZ,
                Gate::Cnot { .. } => GATE_CNOT,
                Gate::U { .. } => GATE_H,
            })
            .collect()
    }

    /// Primary qubit of each gate.
    pub fn gate_qubits(&self) -> Vec<f64> {
        self.circuit_at_zero()
            .gates()
            .iter()
            .map(|g| g.qubits()[0] as f64)
            .collect()
    }

    /// Second qubit of each gate, or `-1` for single-qubit gates.
    pub fn gate_targets(&self) -> Vec<f64> {
        self.circuit_at_zero()
            .gates()
            .iter()
            .map(|g| {
                let q = g.qubits();
                if q.len() > 1 {
                    q[1] as f64
                } else {
                    -1.0
                }
            })
            .collect()
    }

    /// Magnitude of the current gradient at each gate, for the circuit panel's tint.
    ///
    /// Non-parameterised gates report zero.
    pub fn gate_gradient_magnitudes(&self) -> Vec<f64> {
        let circuit = self.circuit_at_zero();
        let grad = adjoint::grad(
            &circuit,
            &self.params[..self.policy.ansatz.num_params()],
            &Observable::z(0),
        );
        circuit
            .gates()
            .iter()
            .map(|g| match g.param_ref() {
                Some((index, _)) => grad[index].abs(),
                None => 0.0,
            })
            .collect()
    }

    /// Adjoint and parameter-shift gradients interleaved as `x, y, x, y, ...`.
    ///
    /// The gradient-check panel plots these as a scatter that must land on `y = x`. It is a
    /// small, satisfying thing to look at, and it silently tells a reader the engine is
    /// trustworthy.
    pub fn gradient_scatter(&self) -> Vec<f64> {
        let circuit = self.circuit_at_zero();
        let p = &self.params[..self.policy.ansatz.num_params()];
        let observable = Observable::z(0);
        let a = adjoint::grad(&circuit, p, &observable);
        let s = shift::grad(&circuit, p, &observable);
        let mut out = Vec::with_capacity(a.len() * 2);
        for (x, y) in a.iter().zip(&s) {
            out.push(*x);
            out.push(*y);
        }
        out
    }

    /// Worst absolute disagreement between the two gradient paths.
    pub fn gradient_agreement(&self) -> f64 {
        let scatter = self.gradient_scatter();
        scatter
            .chunks(2)
            .map(|p| (p[0] - p[1]).abs())
            .fold(0.0, f64::max)
    }

    /// The circuit at `s = 0`. Structure does not depend on the observation, only the
    /// encoding angles do, so this is the right thing to draw.
    fn circuit_at_zero(&self) -> overtone_sim::Circuit {
        self.policy.ansatz.build(&[0.0])
    }
}

/// The barren-plateau panel. Independent of any `Lab`, since the plateau is a property of a
/// circuit family and an observable rather than of a trained agent.
#[wasm_bindgen]
pub struct Plateau {
    local: Vec<f64>,
    global: Vec<f64>,
    qubits: Vec<f64>,
    local_rate: f64,
    global_rate: f64,
    local_r2: f64,
    global_r2: f64,
}

#[wasm_bindgen]
impl Plateau {
    /// Sweep the qubit count. `depth_kind` is 0 for constant, 1 for logarithmic, 2 for
    /// linear; Cerezo et al.'s local-observable escape holds only in the shallow regime, so
    /// the choice is exposed rather than fixed.
    #[wasm_bindgen(constructor)]
    pub fn new(
        min_qubits: usize,
        max_qubits: usize,
        depth_kind: u8,
        depth_value: usize,
        samples: usize,
        seed: u32,
    ) -> Plateau {
        let depth = match depth_kind {
            0 => DepthPolicy::Constant(depth_value),
            2 => DepthPolicy::Linear(depth_value),
            _ => DepthPolicy::Logarithmic(depth_value),
        };
        let l = sweep(
            min_qubits..=max_qubits,
            depth,
            CostLocality::Local,
            samples,
            seed as u64,
        );
        let g = sweep(
            min_qubits..=max_qubits,
            depth,
            CostLocality::Global,
            samples,
            seed as u64,
        );
        let fl = fit_exponential(&l);
        let fg = fit_exponential(&g);

        Plateau {
            qubits: l.iter().map(|p| p.num_qubits as f64).collect(),
            local: l.iter().map(|p| p.variance).collect(),
            global: g.iter().map(|p| p.variance).collect(),
            local_rate: fl.rate,
            global_rate: fg.rate,
            local_r2: fl.r_squared,
            global_r2: fg.r_squared,
        }
    }

    pub fn qubits(&self) -> Vec<f64> {
        self.qubits.clone()
    }

    pub fn local_variance(&self) -> Vec<f64> {
        self.local.clone()
    }

    pub fn global_variance(&self) -> Vec<f64> {
        self.global.clone()
    }

    /// `rate` in `Var ~ 2^(-rate * n)` for the local observable.
    pub fn local_rate(&self) -> f64 {
        self.local_rate
    }

    /// The same for the global observable.
    pub fn global_rate(&self) -> f64 {
        self.global_rate
    }

    /// Coefficient of determination on `log2(Var)`. Well below 1 means the decay is not
    /// exponential, which for the local observable is the finding rather than a bad fit.
    pub fn local_r_squared(&self) -> f64 {
        self.local_r2
    }

    pub fn global_r_squared(&self) -> f64 {
        self.global_r2
    }
}

/// Library version, so the page can show what it is running.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
