//! M28: two walkers in one lattice, and the class system that follows from it.
//!
//! Part VI 1 is the claim this module has to earn: an agent's class is its **exchange
//! statistics**, and "combat" is what happens when two amplitude fields overlap. That is not
//! a metaphor. Sansoni et al. (PRL 108, 010502) ran all three regimes on a chip, and Part
//! VI 8 is blunt about the stakes: if these do not match the published patterns, the arena's
//! physics is wrong and everything built on it is theatre.
//!
//! # The construction
//!
//! Two indistinguishable particles enter a network in modes `I` and `J`. The network's
//! single-particle unitary is `U`. Sansoni et al.'s Eq. (1) gives the output amplitude:
//!
//! ```text
//! |Psi_IJ> -> (1/sqrt 2) sum_{K,L} (U_IK U_JL  +/-  U_JK U_IL) a*_K b*_L |0>
//! ```
//!
//! with `+` for bosons and `-` for fermions. The anyonic case replaces the sign with a
//! phase: the paper prepares `|Psi_phi> = (|H>|V> + e^{i phi} |V>|H>)/sqrt 2`, simulating
//! particles whose creation operators obey `c_i c_j = e^{i phi} c_j c_i`, and reports
//! `phi = pi/4, pi/2, 3pi/4`. So one function covers all three, with `phi = 0` bosonic and
//! `phi = pi` fermionic.
//!
//! # On the citation, against an audit that got it wrong
//!
//! An external prior-art audit recorded that Sansoni et al. "demonstrated bosonic bunching
//! and fermionic antibunching only" and "did not demonstrate fractional exchange
//! statistics", and asked for the anyonic claim to be recited to van Exter et al. **The
//! paper's body says otherwise, and it was checked here rather than taken on either
//! authority.** Section text, verbatim:
//!
//! > "As a further measurement, we therefore prepared some anyonic states |Psi_phi>, in
//! > particular with phi = pi/4, pi/2, 3pi/4, and measured the output probabilities."
//!
//! and its Fig. 4 caption reads "(a) bosonic, (b) fermionic and (c) anyonic (with
//! phi = pi/2)". The audit's error is explicable: the *abstract* mentions only "the
//! bunching-antibunching feature of non interacting bosons and fermions", so a reader who
//! stopped there would conclude exactly what the audit concluded.
//!
//! One qualification the audit was right to press on, and it is kept: Sansoni et al.
//! **simulate** anyonic exchange with photon polarisation -- their own verb -- rather than
//! producing physical anyons. van Exter, Nienhuis & Woerdman (PRA 85, 033823, 2012) is a
//! separate realisation with customizable exchange statistics and is cited alongside, not
//! instead.
//!
//! # The trap, which is the whole reason this module has two correlation functions
//!
//! Part VI 1's table says a fermionic agent "cannot be entered -- occupying a corridor blocks
//! it". **That is wrong as stated, and the paper says so directly.** Sansoni et al. observe
//! that "some of the diagonal elements of the fermionic two-particle walk are nonzero both in
//! the theoretical and experimental distribution", because Fig. 4 plots the distribution over
//! *walk positions*, not over physical modes. A coined walk has two degrees of freedom, the
//! lattice site and the internal coin, so two fermions can sit on the same site provided
//! their coins differ -- their Eq. (4):
//!
//! ```text
//! |Psi_kk> = (1/sqrt 2) [ |j, U> - |j, D> ]
//! ```
//!
//! which is antisymmetric overall and therefore perfectly legal. Pauli exclusion forbids two
//! fermions in the same **mode**, not on the same **site**.
//!
//! The arena consequence is real and it is smaller than Part VI 1 claims: a fermionic agent
//! does not wall off a corridor, it blocks one coin state in that corridor and leaves the
//! other open. [`Correlation::modes`] is where exclusion is exact; [`Correlation::positions`]
//! is what a player would see on a board, and its diagonal is not zero.

use crate::coin::Coin;
use crate::substrate::Substrate;
use crate::walk::Walk;
use overtone_sim::complex::C64;
use std::f64::consts::PI;

/// Exchange statistics, as a phase on particle exchange.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Statistics {
    /// `phi = 0`. Bunching: they exit together.
    Bosonic,
    /// `phi = pi`. Antibunching: no two in the same mode.
    Fermionic,
    /// The dial between them. Sansoni et al. ran `pi/4`, `pi/2` and `3 pi/4`.
    Anyonic(f64),
}

impl Statistics {
    /// The exchange phase `phi`.
    pub fn phase(&self) -> f64 {
        match self {
            Statistics::Bosonic => 0.0,
            Statistics::Fermionic => PI,
            Statistics::Anyonic(p) => *p,
        }
    }

    /// The exchange factor `e^{i phi}` used in the amplitude.
    ///
    /// The two named cases return exact `+1` and `-1` rather than `expi(0)` and `expi(pi)`.
    /// That is not a micro-optimisation: `expi(pi)` carries an imaginary part of `1.2e-16`,
    /// which leaves the fermionic diagonal at about `1e-34` instead of at zero. Pauli
    /// exclusion is exact, so the code that expresses it should be exact too, and the test
    /// for it is then an equality rather than a tolerance somebody has to justify.
    /// `Anyonic(PI)` is *not* bit-identical to `Fermionic` for exactly this reason.
    pub fn phase_factor(&self) -> C64 {
        match self {
            Statistics::Bosonic => C64::ONE,
            Statistics::Fermionic => C64::new(-1.0, 0.0),
            Statistics::Anyonic(p) => C64::expi(*p),
        }
    }

    /// A short label for tables and panels.
    pub fn name(&self) -> &'static str {
        match self {
            Statistics::Bosonic => "bosonic",
            Statistics::Fermionic => "fermionic",
            Statistics::Anyonic(_) => "anyonic",
        }
    }
}

/// One particle's amplitude over every mode after `steps` steps, starting from a single mode.
///
/// This is one column of the single-particle unitary `U`. Modes are indexed
/// `2 * site + coin`, with `site` counted from the left edge of the window; use
/// [`Modes::position_of`] to recover the lattice position.
#[derive(Clone, Debug)]
pub struct Column {
    amps: Vec<C64>,
}

impl Column {
    /// Amplitude in mode `k`.
    pub fn amp(&self, k: usize) -> C64 {
        self.amps[k]
    }

    /// How many modes the window holds.
    pub fn len(&self) -> usize {
        self.amps.len()
    }

    /// True if the column is empty.
    pub fn is_empty(&self) -> bool {
        self.amps.is_empty()
    }

    /// Total probability, which a unitary walk preserves.
    pub fn norm(&self) -> f64 {
        self.amps.iter().map(|a| a.norm_sqr()).sum()
    }
}

/// The mode layout of a window: `2 * (2 * radius + 1)` modes, site-major, coin-minor.
///
/// `radius` is chosen so the window holds the whole light cone of both inputs. Sizing it from
/// the step count alone silently truncates any input that does not start at the origin, and
/// truncation shows up as lost probability rather than as an error, so [`Column::norm`] is
/// checked rather than trusted.
#[derive(Clone, Copy, Debug)]
pub struct Modes {
    /// Half-width of the window, in sites.
    pub radius: usize,
}

impl Modes {
    /// Number of sites in the window.
    pub fn sites(&self) -> usize {
        2 * self.radius + 1
    }

    /// Number of modes: one per (site, coin).
    pub fn len(&self) -> usize {
        2 * self.sites()
    }

    /// True if the window holds no modes, which it never does.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The lattice position a mode sits at.
    pub fn position_of(&self, mode: usize) -> i64 {
        (mode / 2) as i64 - self.radius as i64
    }

    /// The coin index a mode carries.
    pub fn coin_of(&self, mode: usize) -> usize {
        mode % 2
    }

    /// The mode index for a lattice position and coin.
    pub fn mode(&self, position: i64, coin: usize) -> usize {
        let site = (position + self.radius as i64) as usize;
        2 * site + coin
    }
}

/// Evolve one particle from `(position, coin)` for `steps` steps and return its amplitude in
/// every mode of a window of half-width `radius`.
///
/// This is one column of the single-particle unitary `U`.
///
/// # Panics
///
/// If the window is too small to hold the light cone, which would silently lose probability.
pub fn evolve(
    steps: usize,
    position: i64,
    coin: usize,
    radius: usize,
    substrate: &Substrate,
    coin_a: &Coin,
    coin_b: &Coin,
) -> Column {
    let modes = Modes { radius };
    let mut state = [C64::ZERO; 2];
    state[coin] = C64::ONE;
    let span = radius + steps;
    let mut walk = Walk::localised(span, position, state);
    for _ in 0..steps {
        walk.step(substrate, coin_a, coin_b);
    }
    let amps = walk.mode_amplitudes();
    let centre = walk.centre_site();
    let mut out = vec![C64::ZERO; modes.len()];
    for m in 0..modes.len() {
        let pos = modes.position_of(m);
        let site = (centre as i64 + pos) as usize;
        out[m] = amps[2 * site + modes.coin_of(m)];
    }
    let col = Column { amps: out };
    assert!(
        (col.norm() - 1.0).abs() < 1e-12,
        "the window of radius {radius} truncated a {steps}-step walk from position \
         {position}: {} of the probability is outside it",
        1.0 - col.norm()
    );
    col
}

/// The two-particle correlation for one pair of input modes and one statistics.
#[derive(Clone, Debug)]
pub struct Correlation {
    /// The mode layout both matrices are indexed by.
    pub modes: Modes,
    /// `modes[k][l]`: probability of finding one particle in mode `k` and one in mode `l`.
    /// The fermionic diagonal of this matrix is exactly zero -- this is where Pauli
    /// exclusion lives.
    pub modes_matrix: Vec<Vec<f64>>,
    /// `positions[i][j]`: the same distribution marginalised over the coin, which is what a
    /// board would show. The fermionic diagonal of *this* matrix is not zero.
    pub positions: Vec<Vec<f64>>,
}

/// Build the correlation for two particles entering at `(position, coin)` each.
///
/// The amplitude is `A_KL = U_IK U_JL + e^{i phi} U_IL U_JK`. For distinct output modes the
/// detection probability is `|A_KL|^2`, since `A` already sums both orderings; for a repeated
/// mode it is `|A_KK|^2 / 2`, which removes the double count and reduces to the textbook
/// `2 |U_IK U_JK|^2` for bosons and to zero for fermions.
///
/// Inputs are given as positions and coins rather than mode indices because the window is
/// sized from them, and a caller holding a mode index would have to know the window first.
#[allow(clippy::too_many_arguments)]
pub fn correlate(
    steps: usize,
    input_i: (i64, usize),
    input_j: (i64, usize),
    statistics: Statistics,
    substrate: &Substrate,
    coin_a: &Coin,
    coin_b: &Coin,
) -> Correlation {
    let radius = input_i.0.unsigned_abs().max(input_j.0.unsigned_abs()) as usize + steps;
    let modes = Modes { radius };
    let ui = evolve(
        steps, input_i.0, input_i.1, radius, substrate, coin_a, coin_b,
    );
    let uj = evolve(
        steps, input_j.0, input_j.1, radius, substrate, coin_a, coin_b,
    );
    let phase = statistics.phase_factor();
    let n = modes.len();

    let mut m = vec![vec![0.0; n]; n];
    for (k, row) in m.iter_mut().enumerate() {
        for (l, cell) in row.iter_mut().enumerate() {
            let a = ui.amp(k) * uj.amp(l) + phase * (ui.amp(l) * uj.amp(k));
            let p = a.norm_sqr();
            *cell = if k == l { p / 2.0 } else { p };
        }
    }
    // Ordered pairs double-count the off-diagonal, so the normaliser counts each unordered
    // pair once.
    let total: f64 = (0..n).map(|k| (k..n).map(|l| m[k][l]).sum::<f64>()).sum();
    if total > 0.0 {
        for row in m.iter_mut() {
            for v in row.iter_mut() {
                *v /= total;
            }
        }
    }

    let sites = modes.sites();
    let mut pos = vec![vec![0.0; sites]; sites];
    for k in 0..n {
        for l in 0..n {
            pos[k / 2][l / 2] += m[k][l];
        }
    }

    Correlation {
        modes,
        modes_matrix: m,
        positions: pos,
    }
}

impl Correlation {
    /// Total weight on the mode diagonal: the probability that both particles end in the
    /// same mode. Zero for fermions, by exclusion.
    pub fn mode_bunching(&self) -> f64 {
        (0..self.modes.len()).map(|k| self.modes_matrix[k][k]).sum()
    }

    /// Total weight on the position diagonal: the probability that both particles end on the
    /// same *site*, whatever their coins. Not zero for fermions.
    pub fn position_bunching(&self) -> f64 {
        (0..self.modes.sites()).map(|i| self.positions[i][i]).sum()
    }

    /// Sum of the whole distribution over unordered pairs, which should be 1.
    pub fn total(&self) -> f64 {
        let n = self.modes.len();
        (0..n)
            .map(|k| (k..n).map(|l| self.modes_matrix[k][l]).sum::<f64>())
            .sum()
    }
}

/// The similarity Sansoni et al. use to compare two distributions,
/// `S = (sum sqrt(p q))^2 / (sum p * sum q)`.
///
/// Their measured values against theory were `0.982` bosonic and `0.973` fermionic; here it
/// is used to show that the three statistics produce *different* distributions rather than
/// to compare against a laboratory.
pub fn similarity(p: &[Vec<f64>], q: &[Vec<f64>]) -> f64 {
    let mut cross = 0.0;
    let mut sp = 0.0;
    let mut sq = 0.0;
    for (rp, rq) in p.iter().zip(q.iter()) {
        for (a, b) in rp.iter().zip(rq.iter()) {
            cross += (a * b).sqrt();
            sp += a;
            sq += b;
        }
    }
    if sp <= 0.0 || sq <= 0.0 {
        return 0.0;
    }
    cross * cross / (sp * sq)
}
