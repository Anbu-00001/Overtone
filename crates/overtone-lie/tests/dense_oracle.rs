//! An independent check on the bitset algebra, using matrices.
//!
//! Part III 13: "Do not compute closures with dense matrices. A `2^n` matrix anywhere in
//! `overtone-lie` is a bug." That is a rule about the library, not about its tests. This
//! file is the same pattern as `overtone-sim`'s dense Kronecker oracle: a slow, obviously
//! correct implementation that shares no code with the fast one, kept outside the shipped
//! crate so the rule stays true where it matters.
//!
//! Two things are checked. First the commutator's *phase*, since `[P, Q] = 2i s R` with a
//! single sign bit is the claim that g-sim's rotations are built on and a sign error there
//! would be invisible in `dim(g)`. Then the closure dimension itself, by Gram-Schmidt over
//! the Hilbert-Schmidt inner product.

use overtone_lie::{closure_unbounded, family, PauliString};
use overtone_sim::reference::{observable_matrix, DenseMat};
use overtone_sim::{Observable, PauliTerm, C64};

fn dense(p: &PauliString, n: usize) -> DenseMat {
    observable_matrix(n, &Observable::new(vec![PauliTerm::new(1.0, p.factors(n))]))
}

fn commutator(a: &DenseMat, b: &DenseMat) -> DenseMat {
    let ab = a.matmul(b);
    let ba = b.matmul(a);
    let mut out = ab;
    for (o, s) in out.data.iter_mut().zip(&ba.data) {
        *o = *o - *s;
    }
    out
}

fn max_abs_diff(a: &DenseMat, b: &DenseMat) -> f64 {
    a.data
        .iter()
        .zip(&b.data)
        .map(|(x, y)| ((x.re - y.re).hypot(x.im - y.im)).abs())
        .fold(0.0, f64::max)
}

/// Deterministic pseudo-random strings; the crate carries no RNG on purpose.
fn strings(n: usize, count: usize, seed: u64) -> Vec<PauliString> {
    let mut s = seed;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 33) as u128
    };
    let mask = (1u128 << n) - 1;
    (0..count)
        .map(|_| PauliString::new(next() & mask, next() & mask))
        .filter(|p| !p.is_identity())
        .collect()
}

#[test]
fn commutator_sign_and_target_match_dense_matrices() {
    let n = 4;
    let ps = strings(n, 60, 20260906);
    let mut anticommuting = 0;
    for a in &ps {
        for b in &ps {
            let dense_c = commutator(&dense(a, n), &dense(b, n));
            match a.commutator(b) {
                None => {
                    let zero = DenseMat::zeros(1 << n);
                    assert!(
                        max_abs_diff(&dense_c, &zero) < 1e-12,
                        "{} and {} were called commuting",
                        a.render(n),
                        b.render(n)
                    );
                }
                Some((prod, sign)) => {
                    anticommuting += 1;
                    // [a, b] = 2i * sign * prod
                    let mut expect = dense(&prod, n);
                    for e in &mut expect.data {
                        *e = *e * C64::new(0.0, 2.0 * sign as f64);
                    }
                    assert!(
                        max_abs_diff(&dense_c, &expect) < 1e-12,
                        "[{}, {}] disagreed with 2i*{}*{}",
                        a.render(n),
                        b.render(n),
                        sign,
                        prod.render(n)
                    );
                }
            }
        }
    }
    assert!(anticommuting > 100, "test never exercised the sign branch");
}

/// Close a generating set with matrices and Gram-Schmidt, sharing nothing with the bitsets.
///
/// The algebra is a *real* vector space of anti-Hermitian matrices, so the inner product is
/// `Re tr[A^dagger B]` and the coefficients are real.
fn dense_closure_dim(generators: &[PauliString], n: usize) -> usize {
    let dim = 1usize << n;
    let ip = |a: &DenseMat, b: &DenseMat| -> f64 {
        a.data
            .iter()
            .zip(&b.data)
            .map(|(x, y)| x.re * y.re + x.im * y.im)
            .sum()
    };
    let scale = |a: &DenseMat, s: f64| -> DenseMat {
        let mut o = a.clone();
        for e in &mut o.data {
            *e = *e * s;
        }
        o
    };
    let sub = |a: &DenseMat, b: &DenseMat| -> DenseMat {
        let mut o = a.clone();
        for (e, s) in o.data.iter_mut().zip(&b.data) {
            *e = *e - *s;
        }
        o
    };

    // Orthonormal basis, and the raw elements alongside it for taking commutators.
    let mut ortho: Vec<DenseMat> = Vec::new();
    let mut raw: Vec<DenseMat> = Vec::new();
    let push = |m: DenseMat, ortho: &mut Vec<DenseMat>, raw: &mut Vec<DenseMat>| -> bool {
        let mut r = m.clone();
        for o in ortho.iter() {
            let c = ip(o, &r);
            r = sub(&r, &scale(o, c));
        }
        let norm = ip(&r, &r).sqrt();
        if norm < 1e-8 {
            return false;
        }
        ortho.push(scale(&r, 1.0 / norm));
        raw.push(m);
        true
    };

    for g in generators {
        // i * P: anti-Hermitian, so the span stays a real Lie algebra.
        let mut m = dense(g, n);
        for e in &mut m.data {
            *e = *e * C64::new(0.0, 1.0);
        }
        push(m, &mut ortho, &mut raw);
    }

    let mut done = 1usize;
    while done < raw.len() {
        let len = raw.len();
        for j in done..len {
            for i in 0..j {
                let c = commutator(&raw[i], &raw[j]);
                if ip(&c, &c).sqrt() > 1e-8 {
                    push(c, &mut ortho, &mut raw);
                }
            }
        }
        done = len;
    }
    assert_eq!(ortho.len(), raw.len());
    assert!(ortho.len() <= dim * dim);
    ortho.len()
}

#[test]
fn closure_dimension_matches_a_dense_gram_schmidt() {
    let cases: Vec<(&str, Vec<PauliString>, usize)> = vec![
        ("kitaev n=4", family::kitaev(4), 4),
        ("tfim n=3", family::tfim(3), 3),
        ("tfim n=4", family::tfim(4), 4),
        ("xy n=4", family::xy(4), 4),
        ("heisenberg n=3", family::heisenberg(3), 3),
        ("heisenberg n=4", family::heisenberg(4), 4),
        ("a8 n=4", family::a8(4), 4),
        ("hardware-efficient n=3", family::hardware_efficient(3), 3),
        ("single-qubit n=4", family::single_qubit_only(4), 4),
    ];
    for (name, gens, n) in cases {
        let bitset = closure_unbounded(&gens, n).dim();
        let matrices = dense_closure_dim(&gens, n);
        assert_eq!(bitset, matrices, "{name}");
    }
}
