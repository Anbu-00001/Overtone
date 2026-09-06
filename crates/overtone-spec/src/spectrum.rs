//! The spectral instrument (Part I 6.5).
//!
//! Sample a policy over the observation axis, transform, and compare the result against the
//! frequency ceiling the encoding allows. For a strictly band-limited policy nothing may
//! appear beyond the ceiling; for one that is not, the energy above it is the measurement.
//!
//! # Transform the probability, not the logit
//!
//! Part I 6.5 says to sample "the policy's logit function". Taken literally that would
//! defeat the instrument. A SOFTMAX-PQC's logit is `beta * w * <Z_0>`, a rescaling of the
//! same band-limited expectation value the RAW-PQC uses -- so its spectrum is *also*
//! strictly band-limited, and the panel would show no leakage at all.
//!
//! The leakage is created by the softmax itself. It lives in `pi(a|s)`, which is a sigmoid
//! of a band-limited function and therefore not band-limited. So the instrument transforms
//! `pi(1|s)`. This also makes the two policies directly comparable: for a RAW-PQC,
//! `pi(1|s) = (1 + <Z_0>)/2` is band-limited, and the same measurement on the same quantity
//! separates them.

use overtone_sim::C64;

use crate::fft::rfft;

/// A sampled spectrum of a policy over `[-pi, pi)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Spectrum {
    /// `|c_omega|` for `omega = 0, 1, ..., n/2`, normalised so a unit-amplitude cosine at
    /// frequency `omega > 0` reads exactly 1.
    pub magnitude: Vec<f64>,
    /// The reachable frequency ceiling this spectrum should be judged against.
    pub ceiling: usize,
    /// Number of sample points.
    pub samples: usize,
}

/// Sample `f` on a uniform grid over `[-pi, pi)` and transform.
///
/// The grid excludes the right endpoint, so the samples are exactly one period of a
/// `2*pi`-periodic function and the transform is free of leakage from windowing. Any energy
/// found above the ceiling is therefore the policy's, not the instrument's.
pub fn spectrum_of<F>(samples: usize, ceiling: usize, f: F) -> Spectrum
where
    F: Fn(f64) -> f64,
{
    assert!(
        samples.is_power_of_two(),
        "sample count {samples} must be a power of two"
    );
    assert!(
        samples / 2 > ceiling,
        "need more than {} samples to resolve a ceiling of {ceiling} without aliasing",
        2 * ceiling
    );

    let pi = std::f64::consts::PI;
    let values: Vec<f64> = (0..samples)
        .map(|j| f(-pi + 2.0 * pi * (j as f64) / samples as f64))
        .collect();

    let transformed = rfft(&values);
    let magnitude = normalised_magnitudes(&transformed, samples);

    Spectrum {
        magnitude,
        ceiling,
        samples,
    }
}

/// `|c_omega|`, scaled so that `cos(omega s)` reads 1 at `omega > 0` and a constant `c`
/// reads `c` at `omega = 0`.
fn normalised_magnitudes(transformed: &[C64], samples: usize) -> Vec<f64> {
    let half = samples / 2;
    (0..=half)
        .map(|k| {
            let raw = transformed[k].norm_sqr().sqrt();
            // Bins 0 and N/2 have no negative-frequency partner to pair with; every other
            // bin represents half of a real cosine's energy.
            let scale = if k == 0 || k == half {
                samples as f64
            } else {
                samples as f64 / 2.0
            };
            raw / scale
        })
        .collect()
}

impl Spectrum {
    /// Largest magnitude strictly above the ceiling. Zero for a band-limited policy.
    pub fn energy_above_ceiling(&self) -> f64 {
        self.magnitude
            .iter()
            .enumerate()
            .filter(|(k, _)| *k > self.ceiling)
            .map(|(_, m)| *m)
            .fold(0.0, f64::max)
    }

    /// Total magnitude strictly above the ceiling.
    pub fn total_above_ceiling(&self) -> f64 {
        self.magnitude
            .iter()
            .enumerate()
            .filter(|(k, _)| *k > self.ceiling)
            .map(|(_, m)| *m)
            .sum()
    }

    /// Total magnitude at or below the ceiling, excluding DC.
    pub fn total_in_band(&self) -> f64 {
        self.magnitude
            .iter()
            .enumerate()
            .filter(|(k, _)| *k > 0 && *k <= self.ceiling)
            .map(|(_, m)| *m)
            .sum()
    }

    /// Leaked magnitude as a fraction of in-band magnitude.
    ///
    /// A single number for "how much has this policy escaped its own circuit". Exactly zero
    /// for RAW-PQC; positive for SOFTMAX-PQC.
    pub fn leakage_ratio(&self) -> f64 {
        let in_band = self.total_in_band();
        if in_band <= 0.0 {
            return 0.0;
        }
        self.total_above_ceiling() / in_band
    }

    /// Magnitude at one frequency, or zero if it is outside the resolved range.
    pub fn at(&self, omega: usize) -> f64 {
        self.magnitude.get(omega).copied().unwrap_or(0.0)
    }

    /// The highest frequency carrying at least `threshold` magnitude.
    pub fn highest_occupied(&self, threshold: f64) -> usize {
        self.magnitude
            .iter()
            .enumerate()
            .rev()
            .find(|(_, m)| **m >= threshold)
            .map(|(k, _)| k)
            .unwrap_or(0)
    }

    /// Frequencies above the ceiling carrying at least `threshold`, with their magnitudes.
    pub fn leaked_bars(&self, threshold: f64) -> Vec<(usize, f64)> {
        self.magnitude
            .iter()
            .enumerate()
            .filter(|(k, m)| *k > self.ceiling && **m >= threshold)
            .map(|(k, m)| (k, *m))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 512;

    #[test]
    fn a_band_limited_signal_has_no_energy_above_its_ceiling() {
        // Exactly the RAW-PQC situation: a trig polynomial of known degree.
        let s = spectrum_of(N, 3, |x| {
            0.2 + 0.5 * x.cos() + 0.3 * (2.0 * x).sin() - 0.4 * (3.0 * x).cos()
        });
        // Rounding, not zero: the transform accumulates error at the 1e-15 level. Part I 9
        // asks for 1e-9, so state the bound that is actually true rather than an exact zero
        // that would be a lie about floating point.
        assert!(
            s.energy_above_ceiling() < 1e-14,
            "leaked {:e}",
            s.energy_above_ceiling()
        );
        assert!(s.leakage_ratio() < 1e-14, "ratio {:e}", s.leakage_ratio());
    }

    #[test]
    fn magnitudes_are_normalised_so_a_unit_cosine_reads_one() {
        let s = spectrum_of(N, 8, |x| (3.0 * x).cos());
        assert!((s.at(3) - 1.0).abs() < 1e-10, "{}", s.at(3));
        assert!(s.at(2) < 1e-10);
        assert!(s.at(4) < 1e-10);

        let dc = spectrum_of(N, 8, |_| 0.75);
        assert!((dc.at(0) - 0.75).abs() < 1e-10, "{}", dc.at(0));
    }

    #[test]
    fn sine_and_cosine_have_the_same_magnitude() {
        let c = spectrum_of(N, 8, |x| (5.0 * x).cos());
        let s = spectrum_of(N, 8, |x| (5.0 * x).sin());
        assert!((c.at(5) - s.at(5)).abs() < 1e-10);
    }

    #[test]
    fn a_cubed_tone_lands_on_the_third_harmonic() {
        // cos^3(u) = (3 cos u + cos 3u)/4. This is the mechanism behind softmax leakage in
        // its cleanest form: an odd nonlinearity applied to a single tone.
        let s = spectrum_of(N, 1, |x| (2.0 * x).cos().powi(3));
        assert!((s.at(2) - 0.75).abs() < 1e-10, "fundamental {}", s.at(2));
        assert!((s.at(6) - 0.25).abs() < 1e-10, "third harmonic {}", s.at(6));
        assert!(s.at(4) < 1e-10, "no even harmonic expected");
    }

    #[test]
    fn tanh_of_a_single_tone_produces_only_odd_harmonics() {
        // The textbook case the leakage claim is usually stated from.
        let s = spectrum_of(N, 2, |x| (3.0 * (2.0 * x).cos()).tanh());
        assert!(s.at(6) > 0.05, "third harmonic missing: {}", s.at(6));
        assert!(s.at(10) > 0.005, "fifth harmonic missing: {}", s.at(10));
        for even in [4usize, 8, 12] {
            assert!(
                s.at(even) < 1e-9,
                "even harmonic {even} present: {}",
                s.at(even)
            );
        }
    }

    #[test]
    fn a_dc_offset_breaks_the_odd_harmonic_pattern() {
        // The reason the "harmonics at 3L, 5L" story needs care in practice: a constant
        // term destroys the odd symmetry and even harmonics appear.
        let s = spectrum_of(N, 2, |x| (0.7 + 3.0 * (2.0 * x).cos()).tanh());
        assert!(s.at(4) > 1e-3, "even harmonic should appear: {}", s.at(4));
    }

    #[test]
    fn aliasing_is_refused_rather_than_silently_wrong() {
        let result = std::panic::catch_unwind(|| spectrum_of(8, 4, |x| x.cos()));
        assert!(result.is_err(), "a ceiling at Nyquist should be rejected");
    }
}
