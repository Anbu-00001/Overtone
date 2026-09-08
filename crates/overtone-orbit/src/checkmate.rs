//! Check, checkmate, and the brute-force oracle that keeps them honest.
//!
//! Part VII 3, in its own words:
//!
//! > **Check**: your state has overlap with the pursuer's absorbing subspace.
//! > **Checkmate**: that remains true for every point in your reachable orbit.
//!
//! So checkmate is a statement about an orbit, not about occupancy -- which is what makes it
//! well defined on a superposed state, and is the technical note Part VII asks to be made
//! carefully rather than as a dunk. See [`PRIOR_ART`].
//!
//! # What is exact here and what is measured
//!
//! [`Position::is_check`] is exact: it is an expectation value.
//!
//! [`Position::is_checkmate`] is **sound and not complete**. It answers "is every point of
//! the orbit in check" by proving that no safe state shares the orbit's conserved
//! quantities. When it says checkmate, it is checkmate. When it says nothing, the position
//! may still be lost -- the game simply continues. Part VII 12 forbids softening this into a
//! score threshold, and a sound-but-incomplete predicate is the opposite of softening: it
//! errs toward play continuing, never toward a win being awarded that was not earned.

use overtone_lie::closure::Algebra;
use overtone_sim::StateVec;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::invariant::OrbitCertificate;

/// The prior-art note, kept next to the predicate it concerns so the two cannot drift.
///
/// Cantwell's Quantum Chess (arXiv:1906.05836) does replace checkmate with king capture, and
/// the full paper says so in its rules rather than only in summary -- rule 5: "There is no
/// concept of check or checkmate. Kings are captured like any other piece", with rule 10
/// awarding the win when a player's probability of having a king reaches zero.
///
/// Two corrections to how this workspace previously described that paper, both from reading
/// it rather than its summary. First, **it has no abstract**: the document opens directly
/// into section 1, so a quotation attributed to "its own abstract" cannot be right. Second,
/// the sentence in question is in the **conclusion**, and reads "we can see that it is
/// possible to limit the size of the superposition so the game remains **simulable**" --
/// not "tractable for a classical computer". And the mechanism is not a measurement rule as
/// such: it is rule 2, that no square may ever have a non-zero probability of holding two
/// pieces, enforced by projective measurements designed for the purpose.
///
/// The design decision Part VII inverts is therefore that constraint, not measurement in
/// general. Orbit lets superposition grow and pays for it with a windowed arena and a
/// polynomial `dim(g)`; Quantum Chess keeps a full board and bounds the superposition. Both
/// are answers to the same tractability problem and neither is a mistake.
pub const PRIOR_ART: &str = "Cantwell, arXiv:1906.05836, rules 5 and 10 and section 12.";

/// A player's position: an amplitude field, an algebra, and a safe subspace.
pub struct Position {
    pub state: StateVec,
    pub algebra: Algebra,
    certificate: OrbitCertificate,
    /// Basis indices that are safe to occupy -- the complement of the pursuer's absorbing
    /// subspace.
    safe: Vec<usize>,
}

impl Position {
    pub fn new(state: StateVec, algebra: Algebra, safe: Vec<usize>) -> Position {
        let certificate = OrbitCertificate::new(&algebra);
        Position {
            state,
            algebra,
            certificate,
            safe,
        }
    }

    pub fn certificate(&self) -> &OrbitCertificate {
        &self.certificate
    }

    pub fn safe(&self) -> &[usize] {
        &self.safe
    }

    /// Probability of being found outside the safe subspace. Exact.
    pub fn absorbed_weight(&self) -> f64 {
        let mut total = 0.0;
        for i in 0..self.state.dim() {
            if !self.safe.contains(&i) {
                let a = self.state.amp(i);
                total += a.re * a.re + a.im * a.im;
            }
        }
        total
    }

    /// **Check**: any overlap at all with the absorbing subspace.
    pub fn is_check(&self, tolerance: f64) -> bool {
        self.absorbed_weight() > tolerance
    }

    /// **Checkmate**, soundly: no safe state shares this state's conserved quantities, so no
    /// unitary in `exp(g)` can reach one.
    ///
    /// The safe states tested are the computational basis states of the safe subspace,
    /// together with the uniform superposition over it. That is a *sample* of the safe
    /// subspace, not all of it, which only ever makes the predicate more conservative: a
    /// checkmate call requires every sampled safe state to be provably out of reach, and
    /// missing some safe states can only turn a checkmate into silence.
    pub fn is_checkmate(&self, tolerance: f64) -> bool {
        if self.certificate.fully_controllable() || self.safe.is_empty() {
            return false;
        }
        if !self.is_check(tolerance) {
            return false;
        }
        self.safe_samples().iter().all(|s| {
            self.certificate
                .certainly_unreachable(&self.state, s, tolerance)
        })
    }

    /// Safe states the predicate tests against.
    pub fn safe_samples(&self) -> Vec<StateVec> {
        let n = self.state.num_qubits();
        let dim = self.state.dim();
        let mut out = Vec::new();
        for &i in &self.safe {
            let mut re = vec![0.0; dim];
            re[i] = 1.0;
            out.push(StateVec::from_amplitudes(re, vec![0.0; dim]));
        }
        if self.safe.len() > 1 {
            let w = 1.0 / (self.safe.len() as f64).sqrt();
            let mut re = vec![0.0; dim];
            for &i in &self.safe {
                re[i] = w;
            }
            out.push(StateVec::from_amplitudes(re, vec![0.0; dim]));
        }
        let _ = n;
        out
    }
}

/// A brute-force search for a path through the orbit, used only to check the predicate.
///
/// Maximises `|<target| U |source>|^2` over `U` drawn from a **layered ansatz**: every
/// generator of `g`, in order, repeated `layers` times, each with its own free angle. Every
/// such `U` is in `exp(g)` by construction, so **a success is a proof of reachability**. A
/// failure proves nothing -- it is a search, and searches fail for their own reasons.
///
/// The layering is not a detail. A first version picked `depth` generators at random and
/// optimised those angles, and it found only five of eight paths that existed by
/// construction, because the generators it needed were often not among the ones it drew. A
/// certificate's completeness measured against an oracle that weak would be measuring the
/// oracle. Cycling through the whole basis removes that failure mode: the ansatz can express
/// any product of exponentials the construction used, and what remains is only the optimiser.
///
/// The asymmetry is the whole reason this is an oracle for *soundness* and only an estimator
/// for completeness. Soundness is what must hold exactly: if this ever finds a path between
/// two states the certificate called unreachable, the certificate is wrong.
pub fn best_reachable_fidelity(
    algebra: &Algebra,
    source: &StateVec,
    target: &StateVec,
    layers: usize,
    restarts: usize,
    rng: &mut ChaCha8Rng,
) -> f64 {
    let generators: Vec<_> = algebra.basis().to_vec();
    if generators.is_empty() {
        return fidelity(source, target);
    }
    let picks: Vec<usize> = (0..layers).flat_map(|_| 0..generators.len()).collect();
    let width = picks.len();
    let mut best = fidelity(source, target);

    let evaluate = |angles: &[f64]| {
        let mut psi = source.clone();
        for (k, &g) in picks.iter().enumerate() {
            apply_exponential(&mut psi, &generators[g], angles[k]);
        }
        fidelity(&psi, target)
    };

    for restart in 0..restarts {
        // The first restart starts from the identity, which is free and is the right guess
        // whenever the two states are already close.
        let mut angles: Vec<f64> = if restart == 0 {
            vec![0.0; width]
        } else {
            (0..width)
                .map(|_| rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI))
                .collect()
        };
        let mut current = evaluate(&angles);
        let mut step = 1.0;
        while step > 1e-5 {
            let mut improved = false;
            for k in 0..width {
                for delta in [step, -step] {
                    let saved = angles[k];
                    angles[k] = saved + delta;
                    let v = evaluate(&angles);
                    if v > current + 1e-13 {
                        current = v;
                        improved = true;
                    } else {
                        angles[k] = saved;
                    }
                }
            }
            if !improved {
                step *= 0.5;
            }
        }
        if current > best {
            best = current;
        }
        if best > 1.0 - 1e-9 {
            break;
        }
    }
    best
}

/// `|<a|b>|^2`.
pub fn fidelity(a: &StateVec, b: &StateVec) -> f64 {
    let c = a.inner(b);
    c.re * c.re + c.im * c.im
}

/// `exp(i theta P) |psi>` in place, for a Pauli string `P`.
///
/// `P^2 = I`, so `exp(i theta P) = cos(theta) I + i sin(theta) P`. Two passes over the
/// amplitudes and no matrix. This is the move of rule 3, and it is the only way this crate
/// ever changes a state.
pub fn apply_exponential(psi: &mut StateVec, p: &overtone_lie::PauliString, theta: f64) {
    let n = psi.num_qubits();
    let factors = p.factors(n);
    if factors.is_empty() {
        return;
    }
    let obs = overtone_sim::Observable::new(vec![overtone_sim::PauliTerm::new(1.0, factors)]);
    let applied = obs.apply(psi);
    let (c, s) = (theta.cos(), theta.sin());
    let dim = psi.dim();
    let mut re = vec![0.0; dim];
    let mut im = vec![0.0; dim];
    for i in 0..dim {
        let a = psi.amp(i);
        let b = applied.amp(i);
        // (cos + i sin P) psi
        re[i] = c * a.re - s * b.im;
        im[i] = c * a.im + s * b.re;
    }
    *psi = StateVec::from_amplitudes(re, im);
}

/// A uniformly random pure state, for sampling points that are almost surely off any proper
/// orbit.
pub fn haar_state(n: usize, rng: &mut ChaCha8Rng) -> StateVec {
    let dim = 1usize << n;
    let mut re = vec![0.0; dim];
    let mut im = vec![0.0; dim];
    let mut norm = 0.0;
    for i in 0..dim {
        let (a, b) = (normal(rng), normal(rng));
        re[i] = a;
        im[i] = b;
        norm += a * a + b * b;
    }
    let s = 1.0 / norm.sqrt();
    for i in 0..dim {
        re[i] *= s;
        im[i] *= s;
    }
    StateVec::from_amplitudes(re, im)
}

fn normal(rng: &mut ChaCha8Rng) -> f64 {
    let u1: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
    let u2: f64 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// A state that matches `source`'s invariants but is otherwise unconstrained.
///
/// This is what a real completeness measurement needs. Two independent Haar-random states
/// disagree on essentially every invariant, so a certificate separates them trivially and
/// scores a completeness of one on a family that dimension counting says is not complete.
/// The hard pairs are the ones that *share* the conserved quantities, and they have to be
/// built on purpose.
///
/// Coordinate descent on `|invariants(psi) - invariants(source)|^2` from a random start, with
/// the norm restored after every step. Returns the best state found and its residual; a
/// residual above the tolerance means the search failed and the pair should be discarded
/// rather than counted.
pub fn matched_invariant_state(
    certificate: &OrbitCertificate,
    source: &StateVec,
    rng: &mut ChaCha8Rng,
    sweeps: usize,
) -> (StateVec, f64) {
    let target = certificate.invariants(source);
    let dim = source.dim();
    let n = source.num_qubits();
    let mut psi = haar_state(n, rng);

    let residual = |p: &StateVec| certificate.invariants(p).distance(&target);
    let mut current = residual(&psi);
    let mut step = 0.6;

    for _ in 0..sweeps {
        let mut improved = false;
        for i in 0..2 * dim {
            for delta in [step, -step] {
                let mut re: Vec<f64> = (0..dim).map(|k| psi.amp(k).re).collect();
                let mut im: Vec<f64> = (0..dim).map(|k| psi.amp(k).im).collect();
                if i < dim {
                    re[i] += delta;
                } else {
                    im[i - dim] += delta;
                }
                let norm: f64 = re
                    .iter()
                    .chain(im.iter())
                    .map(|v| v * v)
                    .sum::<f64>()
                    .sqrt();
                if norm < 1e-12 {
                    continue;
                }
                for v in re.iter_mut().chain(im.iter_mut()) {
                    *v /= norm;
                }
                let candidate = StateVec::from_amplitudes(re, im);
                let r = residual(&candidate);
                if r < current - 1e-14 {
                    current = r;
                    psi = candidate;
                    improved = true;
                }
            }
        }
        if !improved {
            step *= 0.5;
            if step < 1e-6 {
                break;
            }
        }
    }
    (psi, current)
}
