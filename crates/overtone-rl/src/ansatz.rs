//! The data re-uploading ansatz (Part I 6.2).
//!
//! Layer `l` is:
//!
//! ```text
//! encode:      for each encoding site:  RX(lambda[l, site] * s[component])
//! variational: for each qubit i:        RZ(theta[l, i, 0]) RY(theta[l, i, 1])
//! entangle:    ring of CZ                                  (ablatable to identity)
//! ```
//!
//! `lambda` trainable versus pinned is not a detail. Pinned, the reachable frequencies are
//! the integers up to a hard ceiling; trainable, they become `lambda * {-C..C}`, which is
//! continuous. It is a first-class switch, not a hyperparameter.
//!
//! # The frequency ceiling counts encoding gates, not layers
//!
//! Part I 1.1 states the ceiling as `L`, the layer count. That is only true when each
//! layer applies **one** encoding rotation per observation component. Each `RX(lambda*s)`
//! has generator eigenvalues `+-1/2`, so it contributes frequency differences in
//! `{-1, 0, 1}`; `E` such gates acting on the same component give a reachable set inside
//! `{-E, ..., E}`. Encode the same scalar on all `n` qubits every layer and the ceiling is
//! `n*L`, not `L`.
//!
//! This matters because Part I 7.1's headline test -- RAW-PQC at `L = 2` scoring exactly
//! zero on `SpectralControl-3` -- needs the ceiling to be `2 < 3`. On a two-qubit register
//! encoding `s` everywhere the ceiling would be `4`, the frequency-3 component would be
//! reachable, and the agent would *not* score zero. So the ceiling is computed here from
//! the encoding map rather than assumed equal to the layer count, and
//! [`SpectralControlAnsatz`] defaults to a single encoding site to keep the crisp claim.

use overtone_sim::{Angle, Circuit};

/// One encoding site: qubit `qubit` receives `RX(lambda * s[component])` every layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodingSite {
    pub qubit: usize,
    pub component: usize,
}

/// How the input scaling is treated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scaling {
    /// `lambda` fixed at 1: a rigid, integer frequency comb.
    Pinned,
    /// `lambda` trained: the comb slides continuously.
    Trainable,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AnsatzConfig {
    pub num_qubits: usize,
    /// `L`, the number of re-uploading layers.
    pub layers: usize,
    pub encoding: Vec<EncodingSite>,
    pub scaling: Scaling,
    /// Ring of CZ. Ablatable to identity for the entanglement question of Part I 6.6.
    pub entangle: bool,
}

/// A parameterised data re-uploading circuit family.
///
/// Parameter layout, flat:
///
/// ```text
/// [0, 2*L*n)                        variational angles, RZ then RY per qubit per layer
/// [2*L*n, 2*L*n + L*|encoding|)     lambda, present only when scaling is Trainable
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Ansatz {
    cfg: AnsatzConfig,
    num_variational: usize,
    num_lambda: usize,
}

impl Ansatz {
    pub fn new(cfg: AnsatzConfig) -> Self {
        assert!(cfg.num_qubits > 0, "an ansatz needs at least one qubit");
        assert!(cfg.layers > 0, "an ansatz needs at least one layer");
        for site in &cfg.encoding {
            assert!(
                site.qubit < cfg.num_qubits,
                "encoding site on qubit {} outside a {}-qubit register",
                site.qubit,
                cfg.num_qubits
            );
        }
        let num_variational = 2 * cfg.layers * cfg.num_qubits;
        let num_lambda = match cfg.scaling {
            Scaling::Pinned => 0,
            Scaling::Trainable => cfg.layers * cfg.encoding.len(),
        };
        Ansatz {
            cfg,
            num_variational,
            num_lambda,
        }
    }

    pub fn config(&self) -> &AnsatzConfig {
        &self.cfg
    }

    pub fn num_qubits(&self) -> usize {
        self.cfg.num_qubits
    }

    pub fn layers(&self) -> usize {
        self.cfg.layers
    }

    pub fn num_params(&self) -> usize {
        self.num_variational + self.num_lambda
    }

    pub fn num_variational(&self) -> usize {
        self.num_variational
    }

    /// Index range of the trainable input scalings, empty when pinned.
    pub fn lambda_range(&self) -> std::ops::Range<usize> {
        self.num_variational..self.num_variational + self.num_lambda
    }

    /// Upper bound on the reachable integer frequencies in observation component `c`.
    ///
    /// With pinned scaling the policy's Fourier support is contained in
    /// `{-C, ..., C}` for this `C`; nothing outside it, ever. With trainable scaling the
    /// support becomes `lambda * {-C, ..., C}`, which is continuous in `lambda`.
    pub fn frequency_ceiling(&self, component: usize) -> usize {
        let sites = self
            .cfg
            .encoding
            .iter()
            .filter(|s| s.component == component)
            .count();
        self.cfg.layers * sites
    }

    fn variational_index(&self, layer: usize, qubit: usize, which: usize) -> usize {
        2 * (layer * self.cfg.num_qubits + qubit) + which
    }

    fn lambda_index(&self, layer: usize, site: usize) -> usize {
        self.num_variational + layer * self.cfg.encoding.len() + site
    }

    /// Build the circuit for one observation.
    ///
    /// The observation is baked into the encoding gates as the chain-rule factor on
    /// `lambda`, so the circuit is rebuilt per observation. That is cheap -- it allocates a
    /// gate list, it does not simulate -- and it keeps `overtone-sim` free of any notion of
    /// an observation.
    pub fn build(&self, s: &[f64]) -> Circuit {
        let n = self.cfg.num_qubits;
        let mut c = Circuit::new(n);

        for layer in 0..self.cfg.layers {
            for (site_index, site) in self.cfg.encoding.iter().enumerate() {
                let datum = s[site.component];
                let angle = match self.cfg.scaling {
                    // Pinned: the rotation is exactly the datum, contributing no gradient.
                    Scaling::Pinned => Angle::Fixed(datum),
                    // Trainable: rotate by lambda * s, so d(angle)/d(lambda) = s.
                    Scaling::Trainable => {
                        Angle::scaled(self.lambda_index(layer, site_index), datum)
                    }
                };
                c.rx(site.qubit, angle);
            }

            for q in 0..n {
                c.rz(q, Angle::param(self.variational_index(layer, q, 0)));
                c.ry(q, Angle::param(self.variational_index(layer, q, 1)));
            }

            if self.cfg.entangle && n > 1 {
                if n == 2 {
                    c.cz(0, 1);
                } else {
                    for q in 0..n {
                        c.cz(q, (q + 1) % n);
                    }
                }
            }
        }

        c
    }
}

/// The Part I 7.1 configuration: a scalar observation encoded once per layer, so the
/// frequency ceiling is exactly `L` and the zero-return prediction is sharp.
pub struct SpectralControlAnsatz;

impl SpectralControlAnsatz {
    pub fn config(
        num_qubits: usize,
        layers: usize,
        scaling: Scaling,
        entangle: bool,
    ) -> AnsatzConfig {
        AnsatzConfig {
            num_qubits,
            layers,
            encoding: vec![EncodingSite {
                qubit: 0,
                component: 0,
            }],
            scaling,
            entangle,
        }
    }

    pub fn build(num_qubits: usize, layers: usize, scaling: Scaling, entangle: bool) -> Ansatz {
        Ansatz::new(Self::config(num_qubits, layers, scaling, entangle))
    }
}
