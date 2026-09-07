//! The one linear-algebra routine the dequantization test needs.
//!
//! A Schmidt decomposition at a cut is the eigendecomposition of the reduced density
//! matrix, and truncating to bond dimension `chi` is projecting onto its top `chi`
//! eigenvectors. So the only primitive required is a Hermitian eigensolver that returns
//! vectors, which the cyclic Jacobi method provides with no dependencies and no pivoting
//! decisions to get wrong.
//!
//! Complex Hermitian matrices go through the standard real embedding
//! `H = A + iB  ->  [[A, -B], [B, A]]`, which is real symmetric of twice the size and has
//! every eigenvalue twice. Its eigenvector `(u; v)` is the complex vector `u + iv`.

/// Cyclic Jacobi for a real symmetric matrix, row-major, in place.
///
/// Returns eigenvalues, and fills `vectors` with the eigenvectors as columns.
pub fn jacobi_eigh(a: &mut [f64], n: usize, vectors: &mut [f64]) -> Vec<f64> {
    assert_eq!(a.len(), n * n);
    assert_eq!(vectors.len(), n * n);
    vectors.fill(0.0);
    for i in 0..n {
        vectors[i * n + i] = 1.0;
    }
    for _ in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .map(|(i, j)| a[i * n + j] * a[i * n + j])
            .sum();
        if off.sqrt() < 1e-14 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = vectors[k * n + p];
                    let vkq = vectors[k * n + q];
                    vectors[k * n + p] = c * vkp - s * vkq;
                    vectors[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    (0..n).map(|i| a[i * n + i]).collect()
}

/// Eigenvalues and eigenvectors of a `d x d` complex Hermitian matrix.
///
/// Input is row-major real and imaginary planes. Output is `(eigenvalues, re, im)` with
/// eigenvectors as columns, sorted by descending eigenvalue.
pub fn hermitian_eigh(re: &[f64], im: &[f64], d: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let m = 2 * d;
    let mut embedded = vec![0.0; m * m];
    for r in 0..d {
        for c in 0..d {
            embedded[r * m + c] = re[r * d + c];
            embedded[(r + d) * m + (c + d)] = re[r * d + c];
            embedded[r * m + (c + d)] = -im[r * d + c];
            embedded[(r + d) * m + c] = im[r * d + c];
        }
    }
    let mut vecs = vec![0.0; m * m];
    let vals = jacobi_eigh(&mut embedded, m, &mut vecs);

    // Every eigenvalue appears twice; take one of each pair, largest first.
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&i, &j| vals[j].partial_cmp(&vals[i]).unwrap());

    let mut out_vals = Vec::with_capacity(d);
    let mut out_re = vec![0.0; d * d];
    let mut out_im = vec![0.0; d * d];
    let mut taken = 0usize;
    let mut used: Vec<(f64, Vec<f64>, Vec<f64>)> = Vec::new();
    for &col in &order {
        if taken == d {
            break;
        }
        let u: Vec<f64> = (0..d).map(|r| vecs[r * m + col]).collect();
        let v: Vec<f64> = (0..d).map(|r| vecs[(r + d) * m + col]).collect();
        // Reject the duplicate partner: it is the same complex vector up to a phase, so it
        // has a nonzero overlap with one already taken.
        let duplicate = used.iter().any(|(lam, ur, ui)| {
            (lam - vals[col]).abs() < 1e-9 * (1.0 + lam.abs()) && {
                let (mut dr, mut di) = (0.0, 0.0);
                for k in 0..d {
                    dr += ur[k] * u[k] + ui[k] * v[k];
                    di += ur[k] * v[k] - ui[k] * u[k];
                }
                dr.hypot(di) > 1e-6
            }
        });
        if duplicate {
            continue;
        }
        let norm: f64 = (0..d)
            .map(|k| u[k] * u[k] + v[k] * v[k])
            .sum::<f64>()
            .sqrt();
        if norm < 1e-12 {
            continue;
        }
        for r in 0..d {
            out_re[r * d + taken] = u[r] / norm;
            out_im[r * d + taken] = v[r] / norm;
        }
        out_vals.push(vals[col]);
        used.push((
            vals[col],
            u.iter().map(|x| x / norm).collect(),
            v.iter().map(|x| x / norm).collect(),
        ));
        taken += 1;
    }
    (out_vals, out_re, out_im)
}
