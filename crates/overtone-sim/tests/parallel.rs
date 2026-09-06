//! The parallel kernels must produce exactly the same amplitudes as the serial ones.
//!
//! Dispatch in `apply_1q` only reaches the parallel path above twelve qubits, where a
//! dense cross-check is unaffordable. So these tests call the parallel kernels directly at
//! small sizes and compare against the serial path amplitude for amplitude. The likely bug
//! is the control-mask test, which needs the global amplitude index reconstructed from the
//! chunk index.
#![cfg(feature = "parallel")]

use overtone_sim::gate::{h_matrix, rx_matrix, ry_matrix, rz_matrix, x_matrix};
use overtone_sim::random::{random_params, rng};
use overtone_sim::StateVec;
use rand::Rng;

/// A random normalised state, so the comparison is not run on a mostly-zero vector.
fn random_state(seed: u64, n: usize) -> StateVec {
    let mut r = rng(seed);
    let dim = 1usize << n;
    let re: Vec<f64> = (0..dim).map(|_| r.gen_range(-1.0..1.0)).collect();
    let im: Vec<f64> = (0..dim).map(|_| r.gen_range(-1.0..1.0)).collect();
    let mut st = StateVec::from_amplitudes(re, im);
    let norm = st.norm_sqr().sqrt();
    st.scale(1.0 / norm);
    st
}

#[test]
fn parallel_single_qubit_matches_serial() {
    for n in 1..=9 {
        for (i, angle) in random_params(n as u64, 6).iter().enumerate() {
            let m = match i % 4 {
                0 => rx_matrix(*angle),
                1 => ry_matrix(*angle),
                2 => rz_matrix(*angle),
                _ => h_matrix(),
            };
            for q in 0..n {
                let base = random_state(n as u64 * 100 + q as u64, n);
                let mut a = base.clone();
                let mut b = base.clone();
                a.apply_1q_serial(q, &m);
                b.apply_1q_parallel(q, &m);
                // Exact equality: same arithmetic, same order, just a different schedule.
                assert_eq!(a.re, b.re, "n={n} q={q} kind={i}: real plane differs");
                assert_eq!(a.im, b.im, "n={n} q={q} kind={i}: imaginary plane differs");
            }
        }
    }
}

#[test]
fn parallel_controlled_matches_serial() {
    for n in 2..=9 {
        for c in 0..n {
            for t in 0..n {
                if c == t {
                    continue;
                }
                let base = random_state((n * 31 + c * 7 + t) as u64, n);
                let m = x_matrix();
                let mut a = base.clone();
                let mut b = base.clone();
                a.apply_controlled_1q_serial(c, t, &m);
                b.apply_controlled_1q_parallel(c, t, &m);
                assert_eq!(a.re, b.re, "n={n} c={c} t={t}: real plane differs");
                assert_eq!(a.im, b.im, "n={n} c={c} t={t}: imaginary plane differs");
            }
        }
    }
}
