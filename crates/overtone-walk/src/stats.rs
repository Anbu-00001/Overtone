//! The transport exponent.
//!
//! Part IV 3: quantum transport on a lattice obeys `sigma(t) ~ t^beta`, and changing the
//! substrate moves `beta` continuously across its whole range — `1` ballistic, `1/2`
//! diffusive, `0` localized. The number is measured live, which is what turns choosing a
//! world into an experiment rather than a skin.
//!
//! The fit is over the tail of the trace. Early steps are dominated by the initial
//! condition rather than by the substrate — the light cone has not yet met enough of the
//! word for the word to mean anything — so including them biases every exponent towards
//! one. arXiv:2307.06332 fits the last 4500 of 5000 steps for the same reason.

/// A least-squares fit of `log sigma = log a + beta log t`.
#[derive(Clone, Copy, Debug)]
pub struct PowerLaw {
    pub beta: f64,
    pub log_amplitude: f64,
    /// Coefficient of determination on the log-log fit. A low value means the trace is not
    /// a power law at all, which is itself a finding and must not be reported as a `beta`.
    pub r_squared: f64,
    pub points: usize,
}

/// Fit `sigma(t) ~ t^beta` over the last `tail` fraction of the trace.
pub fn fit_exponent(sigma: &[f64], tail: f64) -> PowerLaw {
    let start = ((sigma.len() as f64) * (1.0 - tail.clamp(0.0, 1.0))).floor() as usize;
    let pts: Vec<(f64, f64)> = sigma
        .iter()
        .enumerate()
        .skip(start.max(1))
        .filter(|(_, s)| **s > 1e-12)
        .map(|(i, s)| (((i + 1) as f64).ln(), s.ln()))
        .collect();
    if pts.len() < 3 {
        return PowerLaw {
            beta: 0.0,
            log_amplitude: 0.0,
            r_squared: 0.0,
            points: pts.len(),
        };
    }
    let n = pts.len() as f64;
    let mx = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let my = pts.iter().map(|p| p.1).sum::<f64>() / n;
    let sxx: f64 = pts.iter().map(|p| (p.0 - mx).powi(2)).sum();
    let sxy: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let beta = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    let intercept = my - beta * mx;
    let ss_tot: f64 = pts.iter().map(|p| (p.1 - my).powi(2)).sum();
    let ss_res: f64 = pts
        .iter()
        .map(|p| (p.1 - (intercept + beta * p.0)).powi(2))
        .sum();
    PowerLaw {
        beta,
        log_amplitude: intercept,
        r_squared: if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            1.0
        },
        points: pts.len(),
    }
}

impl PowerLaw {
    /// Whether the trace is a power law at all.
    ///
    /// Anderson localization does not produce a small exponent; it produces a `sigma` that
    /// *saturates*, and a straight line fitted through a plateau reports whatever the noise
    /// on the plateau happens to slope. The `R^2` is what separates the two, and a `beta`
    /// without it is not a measurement.
    pub fn is_power_law(&self) -> bool {
        self.r_squared > 0.9
    }
}

/// The regime a measured exponent falls in, named as the literature names it.
pub fn regime(beta: f64) -> &'static str {
    if beta < 0.15 {
        "localized"
    } else if beta < 0.42 {
        "subdiffusive"
    } else if beta < 0.58 {
        "diffusive"
    } else if beta < 0.92 {
        "superdiffusive"
    } else {
        "ballistic"
    }
}

/// The regime of a whole fit, refusing to name one when the trace is not a power law.
pub fn regime_of(fit: &PowerLaw) -> &'static str {
    if fit.is_power_law() {
        regime(fit.beta)
    } else {
        "saturated (not a power law)"
    }
}

/// The classical random walk on the line: `sigma(t) = sqrt(t)` exactly.
///
/// Kept as a closed form rather than as a simulation, because it is the baseline every
/// quantum trace is read against and a baseline with its own sampling error is no baseline.
pub fn classical_sigma(steps: usize) -> Vec<f64> {
    (1..=steps).map(|t| (t as f64).sqrt()).collect()
}
