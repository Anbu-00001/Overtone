//! Truncation has to be exactly right before any dequantization verdict means anything.

use overtone_mps::{bond_spectrum, compress, exact_bond_dimension};
use overtone_sim::{Angle, Circuit, StateVec};

fn product_state(n: usize) -> StateVec {
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.ry(q, Angle::Fixed(0.3 + 0.2 * q as f64));
    }
    c.run(&[])
}

fn ghz(n: usize) -> StateVec {
    let mut c = Circuit::new(n);
    c.h(0);
    for q in 1..n {
        c.cnot(0, q);
    }
    c.run(&[])
}

fn entangled(n: usize, seed: u64) -> StateVec {
    let mut c = Circuit::new(n);
    let mut s = seed;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 6.0 - 3.0
    };
    for _ in 0..4 {
        for q in 0..n {
            c.ry(q, Angle::Fixed(next()));
            c.rz(q, Angle::Fixed(next()));
        }
        for q in 0..n - 1 {
            c.cnot(q, q + 1);
        }
    }
    c.run(&[])
}

#[test]
fn a_product_state_has_bond_dimension_one() {
    for n in 2..=6 {
        let st = product_state(n);
        assert_eq!(exact_bond_dimension(&st, 1e-10), 1, "n = {n}");
        let (out, fid) = compress(&st, 1);
        assert!((fid - 1.0).abs() < 1e-12, "n = {n}: fidelity {fid}");
        assert!(st.max_diff(&out) < 1e-10, "n = {n}");
    }
}

#[test]
fn ghz_has_bond_dimension_two_and_chi_one_destroys_it() {
    for n in 3..=6 {
        let st = ghz(n);
        assert_eq!(exact_bond_dimension(&st, 1e-10), 2, "n = {n}");
        let spectrum = bond_spectrum(&st, n / 2);
        assert!((spectrum[0] - 0.5).abs() < 1e-10, "n = {n}: {spectrum:?}");
        assert!((spectrum[1] - 0.5).abs() < 1e-10, "n = {n}");
        let (_, fid) = compress(&st, 1);
        assert!((fid - 0.5).abs() < 1e-9, "n = {n}: chi=1 fidelity {fid}");
        let (_, fid2) = compress(&st, 2);
        assert!((fid2 - 1.0).abs() < 1e-12, "n = {n}: chi=2 fidelity {fid2}");
    }
}

#[test]
fn full_bond_dimension_is_the_identity_and_fidelity_rises_with_chi() {
    for n in 3..=6 {
        let st = entangled(n, 1000 + n as u64);
        let full = 1usize << (n / 2);
        let (out, fid) = compress(&st, full);
        assert!(
            (fid - 1.0).abs() < 1e-10,
            "n = {n}: full-chi fidelity {fid}"
        );
        assert!(st.max_diff(&out) < 1e-9, "n = {n}");

        let mut previous = -1.0;
        for chi in 1..=full {
            let (_, f) = compress(&st, chi);
            assert!(
                f >= previous - 1e-9,
                "n = {n}: fidelity fell from {previous} to {f} at chi = {chi}"
            );
            previous = f;
        }
    }
}

#[test]
fn the_schmidt_spectrum_sums_to_one() {
    for n in 2..=6 {
        let st = entangled(n, 55 + n as u64);
        for cut in 1..n {
            let total: f64 = bond_spectrum(&st, cut).iter().sum();
            assert!((total - 1.0).abs() < 1e-10, "n = {n}, cut = {cut}: {total}");
        }
    }
}
