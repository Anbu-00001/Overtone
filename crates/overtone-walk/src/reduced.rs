//! The line every one of these walks turns out to be.
//!
//! Part II 5 asked for a panel in which the exponentially large graph collapses to a line in
//! front of the reader. Decisions 03 Q7.5 points out that this stopped being a visualisation
//! choice and became the literal computation: by symmetry both families Phase 5 cares about
//! confine the walk to a subspace whose dimension is *linear* in the parameter, and inside it
//! the walk operator is a tridiagonal-shaped product of two-by-two reflections.
//!
//! ```text
//! welded tree G_n     2^(n+2) - 2 vertices     ->   4n + 2 dimensions
//! hypercube  Q_n      2^n vertices             ->   2n     dimensions
//! ```
//!
//! At `n = 10` the welded tree has 4094 vertices and 12278 arcs; the reduced walk is a 42-wide
//! vector. That is the whole demo, and it costs microseconds.
//!
//! # The shape both reductions share
//!
//! This was not obvious in advance and it is the reason there is one type here and not two.
//! Li, Li and Luo's `M_U = M_S M_C` (Lemma 3.1) and Krovi and Brun's Hamming-weight reduction
//! of the hypercube (Eqs. 26-27) are the *same matrix shape*:
//!
//! ```text
//! M_C = diag(1, B_1, B_2, ..., B_m, 1)     two-by-two blocks on pairs (2k-1, 2k)
//! M_S = diag(R, R, ..., R),  R = [[0,1],[1,0]]   swaps on pairs (2k, 2k+1)
//! ```
//!
//! Two families of reflections on interleaved pairings of a line — which is the staggered
//! walk model, arrived at from two different directions. Only the blocks differ: the welded
//! tree's are constant with `cos = 1/3` on either side of the weld, the hypercube's vary with
//! Hamming weight as `cos w_x = 1 - 2x/n`.
//!
//! Everything here is **real**. Both reductions are real orthogonal matrices, so no complex
//! arithmetic survives them, and `unitarity_error` is an orthogonality check.

/// A reduced coined walk on a line: `M_U = M_S M_C`.
///
/// Dimension is `2 * blocks + 2`. Index `0` is the start, the last index is the target.
#[derive(Clone, Debug)]
pub struct Line {
    /// Row-major two-by-two coin blocks, acting on basis pairs `(2k+1, 2k+2)`.
    blocks: Vec<[f64; 4]>,
    name: String,
}

impl Line {
    /// The reduced hypercube walk: `2n` dimensions, one per (Hamming weight, direction).
    ///
    /// Basis `|R,0>, |L,1>, |R,1>, ..., |L,n-1>, |R,n-1>, |L,n>`, where `R` steps away from
    /// the source corner and `L` steps back toward it. `|L,0>` and `|R,n>` do not exist —
    /// there is nothing below weight 0 or above weight `n` — which is what makes the ends of
    /// the line scalar blocks rather than two-by-two ones.
    ///
    /// The blocks are the Grover coin restricted to the symmetric subspace. At weight `x`
    /// there are `n - x` ports up and `x` ports down, so the uniform coin state is
    /// `sqrt((n-x)/n)|R> + sqrt(x/n)|L>`, and reflecting about it gives
    /// `cos w_x = 1 - 2x/n`, `sin w_x = 2 sqrt(x(n-x))/n` — Krovi and Brun Eq. (27).
    pub fn hypercube(n: usize) -> Line {
        assert!(n >= 1, "hypercube dimension must be at least 1");
        let nf = n as f64;
        let blocks = (1..n)
            .map(|x| {
                let c = 1.0 - 2.0 * x as f64 / nf;
                let s = 2.0 * (x as f64 * (nf - x as f64)).sqrt() / nf;
                // On (|L,x>, |R,x>): C|L> = -c|L> + s|R>,  C|R> = s|L> + c|R>.
                [-c, s, s, c]
            })
            .collect();
        Line {
            blocks,
            name: format!("hypercube n={n}"),
        }
    }

    /// The reduced welded-tree walk: `4n + 2` dimensions — Li, Li and Luo, Lemma 3.1.
    ///
    /// Basis `|0,R>, |1,L>, |1,R>, ..., |2n,L>, |2n,R>, |2n+1,L>`, where `|k,L>` is the
    /// uniform superposition of the arcs of layer `k` pointing toward the source root and
    /// `|k,R>` of those pointing toward the weld.
    ///
    /// Inside the left tree a vertex has one arc toward the root and two away, so the uniform
    /// coin state is `sqrt(1/3)|L> + sqrt(2/3)|R>`; in the right tree the ratio inverts. Both
    /// roots have degree 2 and their state is already the uniform one, so the coin fixes them
    /// — those are the two scalar `1`s at the ends of `M_C`.
    pub fn welded(n: usize) -> Line {
        assert!(n >= 1, "welded tree depth must be at least 1");
        let r = 2.0 * 2.0f64.sqrt() / 3.0;
        let left = [-1.0 / 3.0, r, r, 1.0 / 3.0];
        let right = [1.0 / 3.0, r, r, -1.0 / 3.0];
        let mut blocks = vec![left; n];
        blocks.extend(vec![right; n]);
        Line {
            blocks,
            name: format!("welded tree n={n}"),
        }
    }

    /// What the walk is, for panels and failure messages.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Dimension of the reduced space.
    pub fn dimension(&self) -> usize {
        2 * self.blocks.len() + 2
    }

    /// The initial vector `|0>`.
    pub fn start(&self) -> Vec<f64> {
        let mut v = vec![0.0; self.dimension()];
        v[0] = 1.0;
        v
    }

    /// One step of `M_U = M_S M_C`, in place.
    pub fn step(&self, v: &mut [f64]) {
        debug_assert_eq!(v.len(), self.dimension());
        // M_C: the ends are fixed, the interior rotates in pairs.
        for (k, b) in self.blocks.iter().enumerate() {
            let (i, j) = (2 * k + 1, 2 * k + 2);
            let (a0, a1) = (v[i], v[j]);
            v[i] = b[0] * a0 + b[1] * a1;
            v[j] = b[2] * a0 + b[3] * a1;
        }
        // M_S: swap each adjacent pair, offset by one from the coin's pairing.
        for k in (0..v.len()).step_by(2) {
            v.swap(k, k + 1);
        }
    }

    /// One step of the adjoint, `M_U^T = M_C M_S` — shift first, then coin.
    ///
    /// Both `M_C` and `M_S` are symmetric, so transposing the product only reverses the order.
    /// Long's amplification needs `A^dagger`, and this is what it is made of.
    pub fn step_adjoint(&self, v: &mut [f64]) {
        debug_assert_eq!(v.len(), self.dimension());
        for k in (0..v.len()).step_by(2) {
            v.swap(k, k + 1);
        }
        for (k, b) in self.blocks.iter().enumerate() {
            let (i, j) = (2 * k + 1, 2 * k + 2);
            let (a0, a1) = (v[i], v[j]);
            v[i] = b[0] * a0 + b[1] * a1;
            v[j] = b[2] * a0 + b[3] * a1;
        }
    }

    /// `<target| M_U^t |0>` for `t = 1 ..= steps` — the amplitude, sign included.
    ///
    /// Theorem 4.1 is stated on the probability and §6's conjecture on the amplitude; the sign
    /// matters to neither but it is what the amplification consumes, so it is not discarded.
    pub fn amplitudes(&self, steps: usize) -> Vec<f64> {
        let last = self.dimension() - 1;
        let mut v = self.start();
        (0..steps)
            .map(|_| {
                self.step(&mut v);
                v[last]
            })
            .collect()
    }

    /// `p(t) = |<target| M_U^t |0>|^2` for `t = 1 ..= steps`.
    pub fn arrival(&self, steps: usize) -> Vec<f64> {
        let last = self.dimension() - 1;
        let mut v = self.start();
        (0..steps)
            .map(|_| {
                self.step(&mut v);
                v[last] * v[last]
            })
            .collect()
    }

    /// Worst deviation of `M_U^T M_U` from the identity. Zero for an orthogonal matrix.
    pub fn orthogonality_error(&self) -> f64 {
        let d = self.dimension();
        let cols: Vec<Vec<f64>> = (0..d)
            .map(|a| {
                let mut v = vec![0.0; d];
                v[a] = 1.0;
                self.step(&mut v);
                v
            })
            .collect();
        let mut worst: f64 = 0.0;
        for a in 0..d {
            for b in a..d {
                let dot: f64 = (0..d).map(|k| cols[a][k] * cols[b][k]).sum();
                let want = if a == b { 1.0 } else { 0.0 };
                worst = worst.max((dot - want).abs());
            }
        }
        worst
    }
}

/// The step count Kempe's hitting result prescribes for the `n`-cube.
///
/// `T = n (mod 2)` and `|T - pi*n/2| <= 1` — the cubelike-graph statement (arXiv:2609.04503,
/// Theorem 7) of what Kempe proved for the hypercube in 2005.
///
/// **The parity condition is not decoration.** The hypercube is bipartite by Hamming weight,
/// so after `T` steps the walker is only ever at weight `= T (mod 2)`; the antipode has
/// weight `n`. Round `pi*n/2` to the nearest integer without the parity constraint and half
/// the dimensions come back with a probability of exactly zero, which reads as a broken walk
/// rather than as a broken test.
pub fn hypercube_time(n: usize) -> usize {
    let centre = std::f64::consts::PI * n as f64 / 2.0;
    let mut best = None;
    for t in (centre.floor() as usize).saturating_sub(2)..=(centre.ceil() as usize + 2) {
        if t % 2 != n % 2 {
            continue;
        }
        let d = (t as f64 - centre).abs();
        let better = match best {
            None => true,
            Some((_, bd)) => d < bd,
        };
        if d <= 1.0 && better {
            best = Some((t, d));
        }
    }
    best.expect("a parity-matching integer always lies within 1 of pi*n/2")
        .0
}

/// How far Li, Li and Luo's classical precomputation searches: `3.6 n log2(5n)`.
///
/// The base is `2`. The paper writes `log`, and the proof pins it down at Eq. (4.73): it uses
/// `(1/2)^(log 5n) = 1/(5n)`, which is only true of the binary logarithm. Getting this wrong
/// costs a factor of 1.44 in the horizon and nothing in the answer — but it is the kind of
/// silent constant that a reader reproducing the number would trip over.
pub fn welded_horizon(n: usize) -> usize {
    (3.6 * n as f64 * (5.0 * n as f64).log2()).ceil() as usize
}

/// The classical precomputation of Algorithm 1, step 1: the best `T1` in `[2n, horizon]`.
///
/// Returns `(T1, p(T1))`. Theorem 4.1 promises `p(T1) > 1/(20n)` for sufficiently large `n`;
/// what is actually measured is far better than that, and `examples/hitting.rs` prints both.
pub fn welded_best(n: usize) -> (usize, f64) {
    let line = Line::welded(n);
    let horizon = welded_horizon(n);
    let ps = line.arrival(horizon);
    let mut best = (2 * n, 0.0f64);
    for (t, &p) in ps.iter().enumerate() {
        let t = t + 1;
        if t >= 2 * n && p > best.1 {
            best = (t, p);
        }
    }
    best
}

/// What Long's exact amplification does to a walk whose overlap is already known.
#[derive(Clone, Debug)]
pub struct Amplification {
    /// The classically precomputed walk length.
    pub t1: usize,
    /// `<t| M_U^T1 |0>`, sign included.
    pub amplitude: f64,
    /// `theta = arcsin(|amplitude|)`.
    pub theta: f64,
    /// Number of generalised Grover iterations, `ceil((pi/2 - theta) / (2 theta))`.
    pub t2: usize,
    /// The phase, with `beta = -alpha`.
    pub alpha: f64,
    /// Probability on the target after amplification. One, to floating-point.
    pub probability: f64,
}

/// Li, Li and Luo's Algorithm 2: the plain walk, then Long's algorithm, then certainty.
///
/// Decisions 03 Q7.1.4 is the correction this exists to honour. The exponential separation is
/// a property of the **plain walk** — `p = Omega(1/n)` against `2^Omega(n)` classical queries —
/// and the paper's zero-error headline is that walk *plus* an exact amplification layer. Two
/// separate claims that the abstract runs together, and only the first one is the speedup.
///
/// ```text
/// A       = M_U^T1                                the walk, as a preparation unitary
/// S_t(a)  = exp(i a |t><t|)                       phase on the target
/// S_0(b)  = exp(-i b |0><0|)                      phase on the start
/// G(a, b) = A S_0(b) A^dagger S_t(a)              one generalised Grover iteration
/// a = -b  = 2 arcsin( sin(pi / (4 T2 + 2)) / sin theta ),   T2 = ceil((pi/2 - theta) / 2 theta)
/// ```
///
/// The reduced walk is real, so a complex state is two real vectors and every step is the same
/// real `step` applied to each — no complex arithmetic is needed anywhere in the layer.
pub fn amplify(n: usize) -> Amplification {
    let line = Line::welded(n);
    let d = line.dimension();
    let horizon = welded_horizon(n);
    let amps = line.amplitudes(horizon);
    let mut t1 = 2 * n;
    let mut amplitude = 0.0f64;
    for (t, &a) in amps.iter().enumerate() {
        let t = t + 1;
        if t >= 2 * n && a.abs() > amplitude.abs() {
            t1 = t;
            amplitude = a;
        }
    }

    let theta = amplitude.abs().asin();
    let t2 = ((std::f64::consts::FRAC_PI_2 - theta) / (2.0 * theta)).ceil() as usize;
    let alpha = 2.0 * ((std::f64::consts::PI / (4.0 * t2 as f64 + 2.0)).sin() / theta.sin()).asin();

    // The state after A, as (real, imaginary).
    let mut re = line.start();
    let mut im = vec![0.0; d];
    for _ in 0..t1 {
        line.step(&mut re);
    }

    let rotate = |re: &mut f64, im: &mut f64, phi: f64| {
        let (s, c) = phi.sin_cos();
        let (r0, i0) = (*re, *im);
        *re = c * r0 - s * i0;
        *im = s * r0 + c * i0;
    };

    for _ in 0..t2 {
        rotate(&mut re[d - 1], &mut im[d - 1], alpha);
        for _ in 0..t1 {
            line.step_adjoint(&mut re);
            line.step_adjoint(&mut im);
        }
        // beta = -alpha, and S_0 applies exp(-i beta), so this is another +alpha.
        rotate(&mut re[0], &mut im[0], alpha);
        for _ in 0..t1 {
            line.step(&mut re);
            line.step(&mut im);
        }
    }

    Amplification {
        t1,
        amplitude,
        theta,
        t2,
        alpha,
        probability: re[d - 1] * re[d - 1] + im[d - 1] * im[d - 1],
    }
}
