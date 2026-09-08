//! Small statistics shared across the workspace.
//!
//! This module exists because two different questions in this project turned out to want the
//! same answer -- whether one measured quantity tracks another when neither is expected to be
//! a *linear* function of the other. M27 asks it of an algebraic proxy against a return; M51
//! asks it of a combinatorial-game temperature against a von Neumann entropy. Rank
//! correlation is the right tool for both, and having one tested copy is better than two.

/// Spearman's rank correlation, with ties given their average rank.
///
/// Rank rather than Pearson because the algebraic score is built from two indicator
/// variables: it is an ordering, and asking whether it is linearly related to a return would
/// be asking the wrong question of it.
pub fn spearman(xs: &[f64], ys: &[f64]) -> f64 {
    assert_eq!(xs.len(), ys.len());
    let rx = ranks(xs);
    let ry = ranks(ys);
    let n = xs.len() as f64;
    let mx = rx.iter().sum::<f64>() / n;
    let my = ry.iter().sum::<f64>() / n;
    let cov: f64 = rx.iter().zip(&ry).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sx: f64 = rx.iter().map(|a| (a - mx).powi(2)).sum::<f64>().sqrt();
    let sy: f64 = ry.iter().map(|b| (b - my).powi(2)).sum::<f64>().sqrt();
    if sx == 0.0 || sy == 0.0 {
        f64::NAN
    } else {
        cov / (sx * sy)
    }
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..v.len()).collect();
    order.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
    let mut out = vec![0.0; v.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i;
        while j + 1 < order.len() && v[order[j + 1]] == v[order[i]] {
            j += 1;
        }
        let average = (i + j) as f64 / 2.0 + 1.0;
        for &k in &order[i..=j] {
            out[k] = average;
        }
        i = j + 1;
    }
    out
}
