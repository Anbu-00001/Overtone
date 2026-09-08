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

/// A2, so that a frequency of three lands near an audible A4. Any base works; this one keeps
/// the whole reachable comb inside the range where human pitch discrimination is sharpest,
/// which is the entire reason Part IV 5.1 expects the ear to beat the eye here.
const SONIFICATION_BASE: f64 = 110.0;

/// `[environment, policy, beat]` in hertz. A free function so the claim it carries can be
/// tested directly over the whole range of detunings, rather than only at the few values a
/// constructor happens to reach.
fn sonification_tones(k: usize, lambda: f64) -> [f64; 3] {
    let env = SONIFICATION_BASE * k as f64;
    // Never zero hertz: a silent tone would misreport a policy that has merely scaled down.
    let policy = SONIFICATION_BASE * lambda.abs().max(0.05);
    [env, policy, (env - policy).abs()]
}

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
    /// The two tones the sonification plays: the environment's frequency and the policy's
    /// input scaling, in hertz.
    ///
    /// Part IV 5.1. The mapping lives here rather than in JavaScript for the same reason
    /// every other number does — it carries a claim, and a claim in the renderer cannot be
    /// tested. The claim is that the **beat rate is the detuning**: two tones an interval
    /// apart beat at the difference of their frequencies, so what a listener hears slow to a
    /// stop as `lambda` approaches `k` is `SONIFICATION_BASE * |k - lambda|` going to zero.
    /// That is why the agent can be heard locking on before the bar chart shows it.
    ///
    /// Returns `[environment, policy, beat]`.
    pub fn sonification(&self) -> Vec<f64> {
        sonification_tones(self.env.k, self.lambda()).to_vec()
    }

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

// ---------------------------------------------------------------------------------------
// Part III: the Closure tab.
// ---------------------------------------------------------------------------------------

/// C1: the closure, animated.
///
/// Everything the panel needs is precomputed here in one pass -- the strings as glyph
/// codes, the edge each new string arrived on, and the growth curve whose plateau *is* the
/// closure. JavaScript replays that history frame by frame; it never takes a commutator.
#[wasm_bindgen]
pub struct Closure {
    observable: String,
    algebra: overtone_lie::Algebra,
    prediction: overtone_lie::Prediction,
    num_qubits: usize,
    generators: usize,
}

#[wasm_bindgen]
impl Closure {
    /// `family`: 0 Part I ansatz with fixed entanglers, 1 the same with entanglers made
    /// trainable, 2 transverse-field Ising, 3 Heisenberg, 4 XY, 5 hardware-efficient.
    #[wasm_bindgen(constructor)]
    pub fn new(family: u32, num_qubits: usize, layers: usize, limit: usize) -> Closure {
        use overtone_lie::family::{self, EntanglerPolicy};

        let mut fixed = 0usize;
        let generators = match family {
            0 | 1 => {
                let policy = if family == 0 {
                    EntanglerPolicy::Strict
                } else {
                    EntanglerPolicy::Parameterised
                };
                let ansatz =
                    SpectralControlAnsatz::build(num_qubits, layers.max(1), Scaling::Pinned, true);
                let extracted = family::from_circuit(&ansatz.build(&[0.3]), policy);
                fixed = extracted.fixed_entanglers;
                extracted.generators
            }
            2 => family::tfim(num_qubits),
            3 => family::heisenberg(num_qubits),
            4 => family::xy(num_qubits),
            _ => family::hardware_efficient(num_qubits),
        };
        let algebra = overtone_lie::closure(&generators, num_qubits, limit.max(4));
        let observable = overtone_lie::predict::pick_observable(
            &algebra,
            &overtone_lie::PauliString::single(0, overtone_sim::Pauli::Z),
        );
        let mut prediction = overtone_lie::Prediction::new(&algebra, &observable);
        prediction.note_fixed_entanglers(fixed);
        Closure {
            observable: observable.render(num_qubits),
            algebra,
            prediction,
            num_qubits,
            generators: generators.len(),
        }
    }

    pub fn dim(&self) -> usize {
        self.algebra.dim()
    }

    pub fn num_generators(&self) -> usize {
        self.generators
    }

    pub fn truncated(&self) -> bool {
        self.algebra.truncated()
    }

    pub fn dim_su(&self) -> f64 {
        self.prediction.dim_su
    }

    /// `|S|` after each round: the growth curve of the C1 inset.
    pub fn growth(&self) -> Vec<f64> {
        self.algebra.growth().iter().map(|&v| v as f64).collect()
    }

    /// Glyph codes, `num_qubits` per string, `0..=3` for `I X Y Z`.
    pub fn glyphs(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.algebra.dim() * self.num_qubits);
        for p in self.algebra.basis() {
            for q in 0..self.num_qubits {
                out.push(match p.at(q) {
                    overtone_sim::Pauli::I => 0,
                    overtone_sim::Pauli::X => 1,
                    overtone_sim::Pauli::Y => 2,
                    overtone_sim::Pauli::Z => 3,
                });
            }
        }
        out
    }

    /// Two parent indices per string; a generator points at itself.
    pub fn parents(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.algebra.dim() * 2);
        for &(a, b) in self.algebra.parents() {
            out.push(a as f64);
            out.push(b as f64);
        }
        out
    }

    /// Theorem 1's variance of the loss, or `NaN` where the theorem does not apply.
    pub fn predicted_variance(&self) -> f64 {
        self.prediction.loss_variance.unwrap_or(f64::NAN)
    }

    pub fn leaves_the_group(&self) -> bool {
        self.prediction.leaves_the_group
    }

    pub fn verdict(&self) -> String {
        self.prediction.verdict()
    }

    /// Which observable the variance refers to. Not always `Z_0`; see `pick_observable`.
    pub fn observable(&self) -> String {
        self.observable.clone()
    }

    pub fn caveats(&self) -> String {
        self.prediction.caveats.join(" ")
    }
}

/// C2: the prediction landing.
///
/// The algebraic prediction is computed for every width first, then the measurement is run.
/// Part III 13: the ordering is the argument, so the two arrays are returned separately and
/// the panel is expected to draw the line before the points.
#[wasm_bindgen]
pub struct Landing {
    widths: Vec<f64>,
    predicted: Vec<f64>,
    measured: Vec<f64>,
    dims: Vec<f64>,
}

#[wasm_bindgen]
impl Landing {
    /// Sweep `n` from `min_qubits` to `max_qubits` for the transverse-field Ising family
    /// (`family = 0`, polynomial) or the Heisenberg chain (`family = 1`, exponential).
    #[wasm_bindgen(constructor)]
    pub fn new(
        family: u32,
        min_qubits: usize,
        max_qubits: usize,
        depth: usize,
        samples: usize,
        seed: u32,
    ) -> Landing {
        use overtone_gsim::GsimCircuit;
        use overtone_lie::{closure_unbounded, family as fam, predict::components, PauliString};
        use rand::{Rng, SeedableRng};

        let mut widths = Vec::new();
        let mut predicted = Vec::new();
        let mut measured = Vec::new();
        let mut dims = Vec::new();

        for n in min_qubits..=max_qubits {
            let (generating, gates, obs) = if family == 0 {
                let mut g: Vec<PauliString> = (0..n)
                    .map(|q| PauliString::single(q, overtone_sim::Pauli::X))
                    .collect();
                g.extend((0..n - 1).map(|q| {
                    PauliString::from_factors(&[
                        (q, overtone_sim::Pauli::Z),
                        (q + 1, overtone_sim::Pauli::Z),
                    ])
                }));
                (
                    fam::tfim(n),
                    g,
                    PauliString::single(0, overtone_sim::Pauli::X),
                )
            } else {
                let g = fam::heisenberg(n);
                (
                    g.clone(),
                    g,
                    PauliString::from_factors(&[
                        (0, overtone_sim::Pauli::X),
                        (1, overtone_sim::Pauli::X),
                    ]),
                )
            };
            let algebra = closure_unbounded(&generating, n);
            let oi = match algebra.index_of(&obs) {
                Some(i) => i,
                None => continue,
            };
            let comps = components(&algebra);
            let comp = match comps.iter().find(|c| c.contains(&oi)) {
                Some(c) => c,
                None => continue,
            };
            let z_type = comp.iter().filter(|&&i| algebra.basis()[i].x == 0).count();
            let dim = algebra.dim();

            let mut weights = vec![0.0; dim];
            weights[oi] = 1.0;
            let mut circuit = GsimCircuit::new(algebra, gates.clone());
            for _ in 0..depth {
                for gi in 0..gates.len() {
                    circuit.push_raw(gi);
                }
            }
            let ngates = circuit.gates().len();
            let mut r = ChaCha8Rng::seed_from_u64(seed as u64 + n as u64);
            let (mut sum, mut sum_sq) = (0.0, 0.0);
            for _ in 0..samples {
                let angles: Vec<f64> = (0..ngates)
                    .map(|_| r.gen_range(-std::f64::consts::PI..std::f64::consts::PI))
                    .collect();
                let e = circuit.evolve_angles(&angles);
                let v: f64 = weights.iter().zip(&e).map(|(w, x)| w * x).sum();
                sum += v;
                sum_sq += v * v;
            }
            let mean = sum / samples as f64;
            widths.push(n as f64);
            dims.push(dim as f64);
            predicted.push(z_type as f64 / comp.len() as f64);
            measured.push(sum_sq / samples as f64 - mean * mean);
        }
        Landing {
            widths,
            predicted,
            measured,
            dims,
        }
    }

    pub fn widths(&self) -> Vec<f64> {
        self.widths.clone()
    }

    /// Computed from the algebra alone, before anything is run.
    pub fn predicted(&self) -> Vec<f64> {
        self.predicted.clone()
    }

    /// Measured afterwards, from random circuits.
    pub fn measured(&self) -> Vec<f64> {
        self.measured.clone()
    }

    pub fn dims(&self) -> Vec<f64> {
        self.dims.clone()
    }

    /// Worst relative disagreement over the sweep, for the readout.
    pub fn worst_ratio(&self) -> f64 {
        self.predicted
            .iter()
            .zip(&self.measured)
            .map(|(p, m)| (m / p - 1.0).abs())
            .fold(0.0, f64::max)
    }
}

#[wasm_bindgen]
impl Closure {
    /// The agent's sigil: `[angle, radius, x, y, z]` per basis element, sorted canonically.
    ///
    /// Part IV 2.1. A function of the algebra and nothing else -- not of the generating set,
    /// not of the order the closure discovered elements in, and not of a name.
    pub fn sigil(&self) -> Vec<f64> {
        overtone_lie::Sigil::of(&self.algebra).to_flat()
    }

    /// Rings to draw: the largest Pauli weight in the algebra.
    pub fn sigil_rings(&self) -> usize {
        overtone_lie::Sigil::of(&self.algebra).rings
    }

    /// Blocks that commute elementwise -- the mark's rotational symmetry order.
    pub fn sigil_blocks(&self) -> usize {
        overtone_lie::Sigil::of(&self.algebra).blocks
    }
}

/// The `Menagerie` tab: one world, four agents, and the transport exponent.
///
/// Part IV 3 makes the substrate a dial and `beta` the readout, so world selection is an
/// experiment rather than a skin. Everything below is already fitted, already normalised,
/// already ordered; the renderer draws it.
#[wasm_bindgen]
pub struct Menagerie {
    substrate: overtone_walk::Substrate,
    word: overtone_walk::Word,
    steps: usize,
    sigma: Vec<f64>,
    fit: overtone_walk::PowerLaw,
    theta_b: f64,
}

#[wasm_bindgen]
impl Menagerie {
    /// `world` is the code from `overtone_walk::Word`: 0 periodic, 1 two-periodic,
    /// 2 Fibonacci, 3 Thue-Morse, 4 Rudin-Shapiro, 5 static disorder.
    #[wasm_bindgen(constructor)]
    pub fn new(world: u32, steps: usize, theta_b: f64, seed: u32) -> Menagerie {
        let word = overtone_walk::Word::from_code(world);
        let steps = steps.clamp(20, 600);
        let substrate = overtone_walk::Substrate::new(word, steps, seed as u64);
        let mut m = Menagerie {
            substrate,
            word,
            steps,
            sigma: Vec::new(),
            fit: overtone_walk::fit_exponent(&[1.0, 1.0, 1.0], 1.0),
            theta_b,
        };
        m.recompute();
        m
    }

    fn recompute(&mut self) {
        self.sigma = overtone_walk::sigma_trace(
            &self.substrate,
            &overtone_walk::Coin::hadamard(),
            &overtone_walk::Coin::theta(self.theta_b),
            overtone_walk::Run::coherent(self.steps),
        );
        self.fit = overtone_walk::fit_exponent(&self.sigma, 0.5);
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    pub fn beta(&self) -> f64 {
        self.fit.beta
    }

    pub fn r_squared(&self) -> f64 {
        self.fit.r_squared
    }

    /// False when the trace saturates instead of following a power law, in which case
    /// `beta` is the slope of noise on a plateau and must not be shown as an exponent.
    pub fn is_power_law(&self) -> bool {
        self.fit.is_power_law()
    }

    pub fn regime(&self) -> String {
        overtone_walk::regime_of(&self.fit).to_string()
    }

    pub fn world_name(&self) -> String {
        self.word.name().to_string()
    }

    pub fn world_regime(&self) -> String {
        self.word.regime().to_string()
    }

    pub fn sigma_trace(&self) -> Vec<f64> {
        self.sigma.clone()
    }

    /// `sqrt(t)`, the classical baseline, as a closed form rather than a simulation.
    pub fn classical_trace(&self) -> Vec<f64> {
        overtone_walk::classical_sigma(self.steps)
    }

    /// The probability at every site inside the light cone, left to right.
    pub fn distribution(&self) -> Vec<f64> {
        let mut w = overtone_walk::Walk::new(self.steps);
        let (a, b) = (
            overtone_walk::Coin::hadamard(),
            overtone_walk::Coin::theta(self.theta_b),
        );
        for _ in 0..self.steps {
            w.step(&self.substrate, &a, &b);
        }
        w.distribution().into_iter().map(|(_, p)| p).collect()
    }

    /// The substrate's letters over the light cone, as zeros and ones.
    pub fn letters(&self) -> Vec<f64> {
        (-(self.steps as i64)..=self.steps as i64)
            .map(|x| self.substrate.letter(x) as f64)
            .collect()
    }

    /// The race, flattened as `[beta, sigma_final, sees_world, angle_a, angle_b]` per
    /// entrant, in the order Hadamard, Grover, optimised, classical.
    pub fn race(&self) -> Vec<f64> {
        overtone_walk::race(&self.substrate, self.steps)
            .iter()
            .flat_map(|e| {
                let (a, b) = e.angles.unwrap_or((f64::NAN, f64::NAN));
                [
                    e.beta,
                    *e.sigma.last().unwrap_or(&0.0),
                    if e.sees_the_world { 1.0 } else { 0.0 },
                    a,
                    b,
                ]
            })
            .collect()
    }
}

/// The `Wfc` panel: Part IV 3.1's metaphor, kept beside the physics.
#[wasm_bindgen]
pub struct Wfc {
    inner: overtone_wfc::Wfc,
}

#[wasm_bindgen]
impl Wfc {
    #[wasm_bindgen(constructor)]
    pub fn new(width: usize, height: usize, seed: u32) -> Wfc {
        Wfc {
            inner: overtone_wfc::Wfc::new(width.clamp(2, 64), height.clamp(2, 64), seed as u64),
        }
    }

    pub fn width(&self) -> usize {
        self.inner.width
    }

    pub fn height(&self) -> usize {
        self.inner.height
    }

    /// One observation plus propagation. Returns false when the grid is fully collapsed.
    pub fn step(&mut self) -> bool {
        self.inner.step().is_some()
    }

    pub fn total_entropy(&self) -> f64 {
        self.inner.total_entropy()
    }

    pub fn entropy_trace(&self) -> Vec<f64> {
        self.inner.entropy_trace().to_vec()
    }

    pub fn contradictions(&self) -> usize {
        self.inner.contradictions()
    }

    /// Per cell: the tile index if collapsed, otherwise `-1 - options_remaining`, so the
    /// renderer can shade a cell by how undecided it still is without a second call.
    pub fn cells(&self) -> Vec<f64> {
        (0..self.inner.height)
            .flat_map(|y| (0..self.inner.width).map(move |x| (x, y)))
            .map(|(x, y)| match self.inner.tile_at(x, y) {
                Some(t) => t as f64,
                None => -1.0 - self.inner.options_at(x, y).count_ones() as f64,
            })
            .collect()
    }

    /// The eight tiles as `[left, right, up, down]` sockets, so the renderer draws pipes
    /// from the tileset rather than from a hard-coded copy of it.
    pub fn tileset() -> Vec<f64> {
        overtone_wfc::TILES
            .iter()
            .flat_map(|t| [t.left as f64, t.right as f64, t.up as f64, t.down as f64])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Part IV 5.1's audible claim, made testable by living in Rust: the beat rate is the
    /// detuning, so it falls to zero exactly when the policy's scaling reaches the
    /// environment's frequency. A headless browser cannot verify a beat; this can.
    #[test]
    fn the_beat_rate_is_the_detuning_and_vanishes_on_resonance() {
        let k = 3;
        let mut previous = f64::INFINITY;
        for step in 0..=20 {
            let lambda = 1.0 + (k as f64 - 1.0) * step as f64 / 20.0;
            let s = sonification_tones(k, lambda);
            assert!((s[0] - SONIFICATION_BASE * k as f64).abs() < 1e-12);
            assert!((s[1] - SONIFICATION_BASE * lambda).abs() < 1e-9);
            // The beat is the difference of the two tones, and it closes monotonically.
            assert!((s[2] - (s[0] - s[1]).abs()).abs() < 1e-12);
            assert!(s[2] < previous + 1e-9, "beat rose at lambda = {lambda}");
            previous = s[2];
        }
        // On resonance the two tones coincide and the beat stops.
        assert!(sonification_tones(k, k as f64)[2] < 1e-9);
        // And the method a browser calls is the same function, so the test covers it.
        let lab = Lab::new(2, 1, k, 0, true, true, 1.0, 0.05, 7);
        assert_eq!(
            lab.sonification(),
            sonification_tones(k, lab.lambda()).to_vec()
        );
    }

    /// A silent policy would be a lie about a trained agent, so the tone never collapses to
    /// zero hertz even when lambda is driven to zero.
    #[test]
    fn the_policy_tone_stays_audible_at_zero_scaling() {
        assert!(sonification_tones(3, 0.0)[1] > 0.0);
        assert!(
            sonification_tones(3, -2.0)[1] > 0.0,
            "a negative scaling still sounds"
        );
    }
}

// ---------------------------------------------------------------------------------------
// Part V: the Lattice tab.
// ---------------------------------------------------------------------------------------

/// M22 and M23 for the browser: one maze, one eigendecomposition, four fields.
///
/// Everything the panel draws is computed here. The page picks a mode, a time and two
/// mixing weights; it never exponentiates a matrix, never solves an eigenproblem, and never
/// decides what "reached the exit" means.
///
/// The eigensolve is the expensive step and it happens once, in [`Lattice::new`]. A maze of
/// a hundred vertices is a hundred-by-hundred Jacobi sweep, which is milliseconds; the
/// per-frame calls below are all `O(n^2)` recombinations of a basis that is already there.
#[wasm_bindgen]
pub struct Lattice {
    graph: overtone_graph::Graph,
    basis: overtone_graph::Eigenbasis,
    coords: Vec<(usize, usize)>,
    width: usize,
    height: usize,
    source: usize,
    exits: [usize; 2],
    z_a: Vec<f64>,
    z_b: Vec<f64>,
    rho: f64,
}

#[wasm_bindgen]
impl Lattice {
    /// Collapse a maze, take its largest component, and diagonalise its Laplacian.
    ///
    /// Both walks run under `L`. Part V 1.3 writes the toggle as `e^(-Lt)` against
    /// `e^(-iAt)`, but those are two different operators and agree only on a regular graph;
    /// a maze never is one. See `overtone_graph::evolve`.
    #[wasm_bindgen(constructor)]
    pub fn new(width: usize, height: usize, seed: u32) -> Lattice {
        // `u32`, not `u64`, and the reason is not style. wasm-bindgen maps `u64` to a JS
        // `BigInt`, so a page passing an ordinary Number throws a TypeError at the boundary
        // -- and because this constructor runs during start-up, that exception took down the
        // whole init chain, including the hero loop three panels away. Every other
        // constructor in this file already takes `u32` and widens here.
        let mut wfc = overtone_wfc::Wfc::new(width, height, seed as u64);
        wfc.run();
        let graph = overtone_graph::Graph::from_wfc(&wfc).largest_component();
        let basis = overtone_graph::Eigenbasis::of_laplacian(&graph);
        let coords = graph.coords().to_vec();
        let n = graph.order();

        // The source is a corner-most vertex and the two exits are the two vertices furthest
        // from it and from each other, so that task A and task B are genuinely different
        // tasks rather than two names for the same corner.
        let source = (0..n)
            .min_by_key(|&i| coords[i].0 + coords[i].1)
            .unwrap_or(0);
        let from_source = graph.bfs_distances(&[source]);
        let first = (0..n)
            .max_by_key(|&i| from_source[i].unwrap_or(0))
            .unwrap_or(0);
        let from_first = graph.bfs_distances(&[first]);
        let second = (0..n)
            .max_by_key(|&i| from_first[i].unwrap_or(0).min(from_source[i].unwrap_or(0)))
            .unwrap_or(0);

        let rho = overtone_graph::Lmdp::recommended_rho(&graph, &[first, second]);
        let exits = [first, second];
        let z_a =
            overtone_graph::Lmdp::with_terminal_values(&graph, exits.to_vec(), vec![1.0, 0.0], rho)
                .desirability(20_000, 1e-14)
                .z;
        let z_b =
            overtone_graph::Lmdp::with_terminal_values(&graph, exits.to_vec(), vec![0.0, 1.0], rho)
                .desirability(20_000, 1e-14)
                .z;

        Lattice {
            graph,
            basis,
            coords,
            width,
            height,
            source,
            exits,
            z_a,
            z_b,
            rho,
        }
    }

    pub fn order(&self) -> usize {
        self.graph.order()
    }

    pub fn diameter(&self) -> usize {
        self.graph.diameter()
    }

    /// True when the two exponents of Part V 1.3 would be indistinguishable. A maze is
    /// never regular, so this is how the panel says why it uses `L` on both sides.
    pub fn is_regular(&self) -> bool {
        self.graph.is_regular()
    }

    /// Grid coordinates of every vertex, flattened as `x0, y0, x1, y1, ...`.
    pub fn coords(&self) -> Vec<f64> {
        self.coords
            .iter()
            .flat_map(|&(x, y)| [x as f64, y as f64])
            .collect()
    }

    /// Edges, flattened as `a0, b0, a1, b1, ...`, each listed once.
    pub fn edges(&self) -> Vec<f64> {
        let mut out = Vec::new();
        for i in 0..self.graph.order() {
            for &j in self.graph.neighbours(i) {
                if i < j {
                    out.push(i as f64);
                    out.push(j as f64);
                }
            }
        }
        out
    }

    pub fn grid_width(&self) -> usize {
        self.width
    }

    pub fn grid_height(&self) -> usize {
        self.height
    }

    pub fn source(&self) -> usize {
        self.source
    }

    pub fn exit_a(&self) -> usize {
        self.exits[0]
    }

    pub fn exit_b(&self) -> usize {
        self.exits[1]
    }

    pub fn eigenvalues(&self) -> Vec<f64> {
        self.basis.values.clone()
    }

    /// Proto-value function `k`, normalised to `[-1, 1]` so the page maps it to colour
    /// without deciding a scale of its own.
    pub fn mode(&self, k: usize) -> Vec<f64> {
        let v = self.basis.mode(k.min(self.graph.order() - 1));
        let peak = v.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-12);
        v.iter().map(|x| x / peak).collect()
    }

    /// `e^(-Lt)` from the source: diffusion. Normalised to its own peak.
    pub fn diffuse(&self, t: f64) -> Vec<f64> {
        Lattice::normalise(self.basis.diffuse(&self.impulse(), t))
    }

    /// `|e^(-iLt) psi|^2` from the source: interference. Same basis, same normalisation.
    pub fn interfere(&self, t: f64) -> Vec<f64> {
        Lattice::normalise(self.basis.interference_probability(&self.impulse(), t))
    }

    /// Mean hop distance from the source under each evolution, as `[classical, quantum]`.
    ///
    /// The honest caption for the toggle. The quantum field leads early and then stops
    /// leading -- it is unitary on a finite graph, so the modes come back into phase and it
    /// returns. The panel reads these numbers rather than asserting a winner.
    pub fn spread(&self, t: f64) -> Vec<f64> {
        let d = self.graph.bfs_distances(&[self.source]);
        let p0 = self.impulse();
        vec![
            overtone_graph::mean_distance(&d, &self.basis.diffuse(&p0, t)),
            overtone_graph::mean_distance(&d, &self.basis.interference_probability(&p0, t)),
        ]
    }

    /// The time at which diffusion has effectively relaxed, `25 / lambda_1`.
    ///
    /// Set by the spectral gap rather than by a round number, so the slider's range is a
    /// property of this maze and not of anyone's taste.
    pub fn relaxation_time(&self) -> f64 {
        let gap = self.basis.values.get(1).copied().unwrap_or(1.0).max(1e-9);
        25.0 / gap
    }

    /// Log-spaced times from `0.1` to the relaxation time, for the toggle's slider.
    ///
    /// Linear would be useless: on this maze the spectral gap puts relaxation past `t = 3000`
    /// while everything the panel is about -- the crossover where the quantum field passes
    /// diffusion and falls behind again -- happens between `t = 2` and `t = 8`. A linear
    /// slider would spend 99.8% of its travel on a settled field. The grid is computed here
    /// so the page does not have to know that.
    pub fn time_grid(&self, steps: usize) -> Vec<f64> {
        let hi = self.relaxation_time().max(1.0);
        let (lo, n) = (0.1f64, steps.max(2));
        (0..n)
            .map(|i| lo * (hi / lo).powf(i as f64 / (n - 1) as f64))
            .collect()
    }

    /// The desirability field of the composed task `alpha z_A + beta z_B`, normalised.
    ///
    /// Not solved: *added*. The equation in `z` is linear, so a task nobody solved is the
    /// sum of two that were. The panel's slider moves `alpha` and `beta` and nothing is
    /// recomputed but a weighted sum.
    pub fn composed(&self, alpha: f64, beta: f64) -> Vec<f64> {
        Lattice::normalise(overtone_graph::compose(&self.z_a, &self.z_b, alpha, beta))
    }

    /// How far the composed field is from solving that task from scratch.
    ///
    /// The claim under the slider, computed rather than asserted. Solving it costs one power
    /// iteration, which is why the panel can afford to check itself every time the slider
    /// moves.
    pub fn composition_error(&self, alpha: f64, beta: f64) -> f64 {
        let solved = overtone_graph::Lmdp::with_terminal_values(
            &self.graph,
            self.exits.to_vec(),
            vec![alpha, beta],
            self.rho,
        )
        .desirability(20_000, 1e-14)
        .z;
        overtone_graph::compose(&self.z_a, &self.z_b, alpha, beta)
            .iter()
            .zip(&solved)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    /// The optimal next vertex from each state under the composed policy, or the state
    /// itself at an exit.
    pub fn policy(&self, alpha: f64, beta: f64) -> Vec<f64> {
        let z = overtone_graph::compose(&self.z_a, &self.z_b, alpha, beta);
        (0..self.graph.order())
            .map(|i| {
                self.graph
                    .neighbours(i)
                    .iter()
                    .copied()
                    .max_by(|&a, &b| z[a].partial_cmp(&z[b]).unwrap())
                    .unwrap_or(i) as f64
            })
            .collect()
    }

    /// Hop distance from the source, for the caption that says how far the exits are.
    pub fn distance_to(&self, vertex: usize) -> usize {
        self.graph.bfs_distances(&[self.source])[vertex].unwrap_or(0)
    }

    fn impulse(&self) -> Vec<f64> {
        let mut p = vec![0.0; self.graph.order()];
        p[self.source] = 1.0;
        p
    }

    fn normalise(v: Vec<f64>) -> Vec<f64> {
        let peak = v.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-300);
        v.iter().map(|x| x / peak).collect()
    }
}
