//! The dequantization test.
//!
//! Part III 5: take a trained quantum policy, run it through an MPS at increasing bond
//! dimension, and report the smallest `chi` that reproduces it. If `chi = 4` suffices, the
//! agent is a small tensor network and there is no advantage to argue about.
//!
//! Part III 13 is explicit that this must be run and published whatever it says: "Do not
//! skip the dequantization test because the answer might be unflattering. The unflattering
//! answer is the contribution."

use overtone_sim::StateVec;

use crate::truncate::compress;

/// One row of the dial: what the policy looks like at a given bond dimension.
#[derive(Clone, Copy, Debug)]
pub struct ChiRow {
    pub chi: usize,
    /// Largest `|pi(1|s) - pi_chi(1|s)|` over the observation grid.
    pub max_policy_error: f64,
    /// Mean `|<psi|psi_chi>|^2` over the grid.
    pub mean_fidelity: f64,
    /// The return this truncated policy achieves on `SpectralControl-k`.
    pub achieved_return: f64,
}

#[derive(Clone, Debug)]
pub struct DequantizeReport {
    pub num_qubits: usize,
    pub tolerance: f64,
    pub exact_return: f64,
    pub rows: Vec<ChiRow>,
    /// Smallest bond dimension reproducing the policy within `tolerance`, if any was tried.
    pub effective_chi: Option<usize>,
}

impl DequantizeReport {
    /// The single sentence Part III 5 asks for.
    pub fn verdict(&self) -> String {
        let full = 1usize << (self.num_qubits / 2);
        match self.effective_chi {
            None => format!(
                "no bond dimension up to {} reproduced the policy to {:.0e}",
                self.rows.last().map(|r| r.chi).unwrap_or(0),
                self.tolerance
            ),
            Some(chi) if chi <= 4 => format!(
                "effective chi = {chi}: this agent is a small tensor network, and a chi = \
                 {chi} MPS reproduces it to {:.0e}. No quantum advantage is being claimed \
                 and none is present.",
                self.tolerance
            ),
            Some(chi) => format!(
                "effective chi = {chi} of a maximum {full} for {} qubits: the policy needs \
                 {:.0}% of the available bond dimension.",
                self.num_qubits,
                100.0 * chi as f64 / full as f64
            ),
        }
    }
}

/// Run the dial.
///
/// `state_of` returns the exact circuit state at an observation; `prob` reads a policy
/// probability off a state, which keeps the test independent of the Born rule in use.
pub fn dequantize<S, P>(
    num_qubits: usize,
    k: usize,
    nodes: usize,
    max_chi: usize,
    tolerance: f64,
    state_of: S,
    prob: P,
) -> DequantizeReport
where
    S: Fn(f64) -> StateVec,
    P: Fn(&StateVec) -> f64,
{
    let pi = std::f64::consts::PI;
    let grid: Vec<f64> = (0..nodes)
        .map(|i| -pi + 2.0 * pi * (i as f64 + 0.5) / nodes as f64)
        .collect();
    let states: Vec<StateVec> = grid.iter().map(|&s| state_of(s)).collect();
    let exact: Vec<f64> = states.iter().map(&prob).collect();

    // Midpoint rule, the same quadrature `SpectralControl::expected_return` uses.
    let ret = |ps: &[f64]| -> f64 {
        grid.iter()
            .zip(ps)
            .map(|(s, p)| (2.0 * p - 1.0) * ((k as f64) * s).cos())
            .sum::<f64>()
            / nodes as f64
    };
    let exact_return = ret(&exact);

    let mut rows = Vec::new();
    let mut effective_chi = None;
    for chi in 1..=max_chi {
        let mut max_err: f64 = 0.0;
        let mut fid = 0.0;
        let mut ps = Vec::with_capacity(nodes);
        for (st, e) in states.iter().zip(&exact) {
            let (trunc, f) = compress(st, chi);
            let p = prob(&trunc);
            max_err = max_err.max((p - e).abs());
            fid += f;
            ps.push(p);
        }
        rows.push(ChiRow {
            chi,
            max_policy_error: max_err,
            mean_fidelity: fid / nodes as f64,
            achieved_return: ret(&ps),
        });
        if effective_chi.is_none() && max_err <= tolerance {
            effective_chi = Some(chi);
            break;
        }
    }

    DequantizeReport {
        num_qubits,
        tolerance,
        exact_return,
        rows,
        effective_chi,
    }
}
