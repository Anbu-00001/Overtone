//! A radix-2 FFT, and the naive DFT it is tested against.
//!
//! Hand-rolled rather than pulled from a crate for two reasons. The transform length is
//! always a power of two by construction (Part I 6.5 samples on `N = 512`), so radix-2 is
//! the whole algorithm rather than a special case. And a second, independent
//! implementation is what the rest of this repository does everywhere else: the naive DFT
//! in [`dft_naive`] shares no code with [`fft_in_place`], so a butterfly indexing error
//! shows up as a disagreement rather than as a plausible-looking spectrum.

use overtone_sim::C64;

/// In-place radix-2 decimation-in-time FFT.
///
/// Computes `X[k] = sum_j x[j] exp(-2 pi i j k / N)`. Panics unless `N` is a power of two.
pub fn fft_in_place(data: &mut [C64]) {
    let n = data.len();
    assert!(n.is_power_of_two(), "FFT length {n} is not a power of two");
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation.
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = (i as u32).reverse_bits() >> (32 - bits);
        let j = j as usize;
        if j > i {
            data.swap(i, j);
        }
    }

    // Butterflies, stage by stage.
    let mut len = 2;
    while len <= n {
        let theta = -2.0 * std::f64::consts::PI / len as f64;
        let half = len / 2;
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let w = C64::expi(theta * k as f64);
                let a = data[start + k];
                let b = data[start + k + half] * w;
                data[start + k] = a + b;
                data[start + k + half] = a - b;
            }
        }
        len <<= 1;
    }
}

/// Allocating wrapper around [`fft_in_place`].
pub fn fft(input: &[C64]) -> Vec<C64> {
    let mut out = input.to_vec();
    fft_in_place(&mut out);
    out
}

/// FFT of a real signal.
pub fn rfft(input: &[f64]) -> Vec<C64> {
    let buffer: Vec<C64> = input.iter().map(|x| C64::new(*x, 0.0)).collect();
    fft(&buffer)
}

/// The direct `O(N^2)` transform. Reference only; never called on a hot path.
pub fn dft_naive(input: &[C64]) -> Vec<C64> {
    let n = input.len();
    (0..n)
        .map(|k| {
            let mut acc = C64::ZERO;
            for (j, x) in input.iter().enumerate() {
                let angle = -2.0 * std::f64::consts::PI * (j * k) as f64 / n as f64;
                acc = acc + *x * C64::expi(angle);
            }
            acc
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use overtone_sim::random::rng;
    use rand::Rng;

    fn max_diff(a: &[C64], b: &[C64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| x.max_diff(*y))
            .fold(0.0, f64::max)
    }

    #[test]
    fn radix2_matches_the_naive_transform() {
        for bits in 1..=9u32 {
            let n = 1usize << bits;
            let mut r = rng(bits as u64);
            let input: Vec<C64> = (0..n)
                .map(|_| C64::new(r.gen_range(-1.0..1.0), r.gen_range(-1.0..1.0)))
                .collect();
            let d = max_diff(&fft(&input), &dft_naive(&input));
            // Tolerance scales with N: both transforms accumulate rounding, and the naive
            // one accumulates more of it.
            assert!(d < 1e-12 * n as f64, "N={n}: differ by {d:e}");
        }
    }

    #[test]
    fn a_pure_tone_lands_in_exactly_one_bin() {
        let n = 512;
        for freq in [1usize, 3, 7, 64] {
            let signal: Vec<f64> = (0..n)
                .map(|j| (2.0 * std::f64::consts::PI * (freq * j) as f64 / n as f64).cos())
                .collect();
            let spectrum = rfft(&signal);
            for (k, c) in spectrum.iter().enumerate().take(n / 2) {
                let magnitude = c.norm_sqr().sqrt() / (n as f64 / 2.0);
                let expected = if k == freq { 1.0 } else { 0.0 };
                assert!(
                    (magnitude - expected).abs() < 1e-10,
                    "freq={freq} bin={k}: {magnitude} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn a_constant_signal_is_pure_dc() {
        let n = 64;
        let spectrum = rfft(&vec![2.5; n]);
        assert!((spectrum[0].re - 2.5 * n as f64).abs() < 1e-10);
        for c in spectrum.iter().skip(1) {
            assert!(c.norm_sqr().sqrt() < 1e-10);
        }
    }
}
