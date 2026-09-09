//! M39b: the Skill Trace, on Browne's protocol and Goodman's grid.
//!
//! # Getting the name right, because three papers are involved
//!
//! Lantz, Isaksen, Jaffe, Nealen & Togelius (AAAI 2017) proposed the strategy-ladder framework
//! and were explicit that they had no system for evaluating `d` — the strategy language,
//! resource levels, performance metric and step definition are all left open. So *"the standard
//! Lantz metric"* does not exist and this module does not report one.
//!
//! The lineage is Lantz 2017 → Tavener 2020 → **Browne 2022 (Skill Trace)** → Goodman,
//! Perez-Liebana & Lucas 2024 (Skill Depth). What is computed here is **Browne's Skill Trace**,
//! measured over **Goodman's grid**, and the correct phrasing for it is: *we instantiate the
//! strategy-ladder framework of Lantz et al., following Browne's Skill Trace and Goodman et
//! al.'s grid methodology, with our chosen budget, performance measure and step criterion.*
//!
//! # Browne's formula, implemented as published
//!
//! ```text
//! matches      UCT_{2^(m-1) BF}  vs  UCT_{2^m BF},  m = 1, 2, 3, ...
//! score        the strong agent's mean result, 1 = win, 0 = draw, -1 = loss
//! f            least-squares line through the scores, FROM THE SECOND MATCH ON
//! y            f(M + 1), clamped to [0, 1]
//! A            sum over all M matches of max(0, score)^2
//! ST           y + (1 - y) A
//! ```
//!
//! Two details that are easy to lose and change the number.
//!
//! **The score is a mean in `[-1, 1]`, not a win fraction in `[0, 1]`.** A draw is `0`, not a
//! half. `crate::search::win_rate` returns the other convention and is not what this uses.
//!
//! **The first match is discarded from the regression.** Browne's reason is specific to UCT:
//! the first `BF` iterations are random move choices as each root child is tried in turn, so
//! `UCT_2BF` unduly outperforms `UCT_BF` and the first point defies the trend. The pseudocode
//! nonetheless sums `A` over *all* `M` matches, including the discarded one — the text and the
//! algorithm differ here, and this follows the algorithm, with [`Trace::area_excluding_first`]
//! available for the other reading so the difference is visible rather than chosen silently.
//!
//! # Why the grid, and not the ladder
//!
//! Goodman et al. find that a ladder of adjacent budgets may not statistically resolve small
//! win-rate differences, and extend it to all pairwise budgets. A gap invisible between 16 and
//! 32 is easily resolved between 16 and 128, and the distant pairs constrain the fit — the same
//! games buy far more statistical power. So [`Grid`] measures every pair and Browne's ST is
//! computed from the adjacent diagonal of it, which keeps the headline number on the published
//! protocol while the off-diagonal pairs carry the error bars.
//!
//! # And what a result here can and cannot say
//!
//! For a draw-heavy match the 95% Elo error is about `400/sqrt(N)`: roughly 1600 games for
//! ±10 Elo and 6400 for ±5. [`Grid::games_for`] computes it, and any run below that budget is
//! reported with its interval rather than as a point estimate. Detecting small differences here
//! is brutally expensive and saying so is part of the measurement.

use std::thread;

use crate::language::Agent;
use crate::search::play;

/// One cell of the grid: a weak budget against a strong one.
#[derive(Clone, Copy, Debug)]
pub struct Pairing {
    pub weak: usize,
    pub strong: usize,
    pub games: usize,
    pub wins: usize,
    pub draws: usize,
    pub losses: usize,
}

impl Pairing {
    /// The strong agent's mean result on Browne's scale: `1` win, `0` draw, `-1` loss.
    pub fn score(&self) -> f64 {
        if self.games == 0 {
            return 0.0;
        }
        (self.wins as f64 - self.losses as f64) / self.games as f64
    }

    /// The same result as a win fraction in `[0, 1]`, for Elo-style reporting.
    pub fn win_fraction(&self) -> f64 {
        if self.games == 0 {
            return 0.5;
        }
        (self.wins as f64 + 0.5 * self.draws as f64) / self.games as f64
    }

    /// 95% interval half-width on the win fraction, by the normal approximation.
    pub fn half_width(&self) -> f64 {
        if self.games == 0 {
            return 1.0;
        }
        let p = self.win_fraction();
        1.96 * (p * (1.0 - p) / self.games as f64).sqrt()
    }

    /// Whether the pairing resolves at all: does the interval exclude a coin flip?
    pub fn resolves(&self) -> bool {
        (self.win_fraction() - 0.5).abs() > self.half_width()
    }
}

/// The match conditions a grid is measured under.
///
/// These three travel together everywhere and mean nothing apart -- a trace at one width is not
/// comparable with a trace at another, and neither is one at a different coherence horizon. So
/// they are one value, and a grid that carries its own settings cannot be misread as a grid
/// measured under different ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Setting {
    pub n: usize,
    pub coherence: usize,
    pub k: usize,
}

/// Every pairwise budget comparison at one width.
#[derive(Clone, Debug)]
pub struct Grid {
    pub setting: Setting,
    pub budgets: Vec<usize>,
    pub pairings: Vec<Pairing>,
}

/// Browne's budget ladder: `BF, 2BF, 4BF, ...`, where `BF` is the action count at the root.
///
/// Browne bases the iteration counts on the branching factor rather than on absolute numbers,
/// so that the same protocol means the same thing in games of different width. Overtone's
/// action is a `(move, angle)` pair, so `BF` is the candidate count and not the legal-move
/// count — eight times larger, and using the smaller one would be quietly running a different
/// protocol.
pub fn budget_ladder(branching: usize, matches: usize) -> Vec<usize> {
    (0..=matches).map(|m| branching << m).collect()
}

/// The budget above which a kind stops changing, or `None` if it never does.
///
/// greedy searches every candidate once its budget reaches the candidate count, so every budget
/// from there up is the same agent — a ladder running past it measures nothing and reports
/// 100% draws, which then drags the regression that Browne's `y` is read from. Measured before
/// this was applied: three of the greedy ladder's six matches were saturated pairs at exactly
/// `0.000`, pinning `y` at zero.
pub fn saturation(agent: &Agent, n: usize) -> Option<usize> {
    match agent.search.kind {
        crate::language::SearchKind::Greedy => Some(agent.moves(n).len() * 8),
        _ => None,
    }
}

/// A doubling ladder from an arbitrary base, for the kinds that saturate at or below `BF`.
///
/// Browne bases the ladder at `BF` because that is where UCT stops making random choices. The
/// other two kinds saturate rather than start there: **greedy searches every candidate once its
/// budget reaches the candidate count**, so every budget at or above `BF` is the same agent and
/// a ladder starting at `BF` would be flat by construction. Its meaningful range is below `BF`,
/// and running it there is not a weaker protocol — it is the same protocol over the range in
/// which the kind has a compute axis at all.
pub fn budget_ladder_from(base: usize, matches: usize) -> Vec<usize> {
    (0..=matches).map(|m| base << m).collect()
}

impl Grid {
    /// Games needed for a given 95% Elo half-width, on the draw-heavy approximation `400/sqrt(N)`.
    pub fn games_for(elo: f64) -> usize {
        ((400.0 / elo).powi(2)).ceil() as usize
    }

    /// Measure every pair of budgets, `games` games each, colours alternating.
    ///
    /// Pairings are independent and each is seeded from its own index, so running them across
    /// threads cannot change a result — Part VIII 6 requires same spec, same seed, same answer,
    /// and a parallel measurement that failed that would be worthless. `workers` is the only
    /// thing threading changes.
    pub fn measure(
        agent: &Agent,
        setting: Setting,
        budgets: &[usize],
        games: usize,
        seed: u64,
        workers: usize,
    ) -> Grid {
        let pairs: Vec<(usize, usize)> = (0..budgets.len())
            .flat_map(|i| ((i + 1)..budgets.len()).map(move |j| (i, j)))
            .collect();

        let run = |&(i, j): &(usize, usize)| -> Pairing {
            let (weak, strong) = (budgets[i], budgets[j]);
            let mut lo = agent.clone();
            lo.search.budget = weak;
            let mut hi = agent.clone();
            hi.search.budget = strong;
            let (mut wins, mut draws, mut losses) = (0, 0, 0);
            for g in 0..games {
                // Alternate play order so a first-move advantage cannot be read as skill.
                let swap = g % 2 == 1;
                let (a, b) = if swap { (&lo, &hi) } else { (&hi, &lo) };
                let strong_seat = usize::from(swap);
                let s = seed
                    ^ ((i * 97 + j) as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    ^ (g as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                match play(a, b, setting.n, setting.coherence, setting.k, s) {
                    Some(w) if w == strong_seat => wins += 1,
                    Some(_) => losses += 1,
                    None => draws += 1,
                }
            }
            Pairing {
                weak,
                strong,
                games,
                wins,
                draws,
                losses,
            }
        };

        let workers = workers.max(1).min(pairs.len().max(1));
        let mut pairings: Vec<Pairing> = if workers <= 1 {
            pairs.iter().map(run).collect()
        } else {
            let chunk = pairs.len().div_ceil(workers);
            let slices: Vec<&[(usize, usize)]> = pairs.chunks(chunk).collect();
            thread::scope(|scope| {
                let handles: Vec<_> = slices
                    .into_iter()
                    .map(|s| scope.spawn(move || s.iter().map(&run).collect::<Vec<_>>()))
                    .collect();
                handles
                    .into_iter()
                    .flat_map(|h| h.join().expect("pairing worker panicked"))
                    .collect()
            })
        };
        pairings.sort_by_key(|p| (p.weak, p.strong));

        Grid {
            setting,
            budgets: budgets.to_vec(),
            pairings,
        }
    }

    /// The adjacent-budget diagonal, in order. Browne's protocol runs on exactly these.
    pub fn adjacent(&self) -> Vec<Pairing> {
        self.budgets
            .windows(2)
            .filter_map(|w| {
                self.pairings
                    .iter()
                    .find(|p| p.weak == w[0] && p.strong == w[1])
                    .copied()
            })
            .collect()
    }

    /// How many pairings resolve — the reason Goodman's grid exists.
    pub fn resolved(&self) -> usize {
        self.pairings.iter().filter(|p| p.resolves()).count()
    }

    /// Browne's Skill Trace, from the adjacent diagonal.
    pub fn skill_trace(&self) -> Trace {
        Trace::of(
            &self
                .adjacent()
                .iter()
                .map(|p| p.score())
                .collect::<Vec<_>>(),
        )
    }
}

/// The Skill Trace and the pieces it is built from, so a reader can check the arithmetic.
#[derive(Clone, Copy, Debug)]
pub struct Trace {
    /// Least-squares slope through the scores, first match excluded.
    pub slope: f64,
    pub intercept: f64,
    /// `f(M + 1)`, clamped to `[0, 1]`.
    pub y: f64,
    /// `sum max(0, score)^2` over all matches, per Browne's Algorithm 1.
    pub area: f64,
    /// The same sum with the discarded first match left out, per Browne's prose.
    pub area_excluding_first: f64,
    /// `ST = y + (1 - y) A`.
    pub value: f64,
    pub matches: usize,
}

impl Trace {
    /// Compute the trace from per-match scores in `[-1, 1]`, in ladder order.
    pub fn of(scores: &[f64]) -> Trace {
        let m = scores.len();
        // The regression runs from the second match on: Browne discards the first because the
        // first BF UCT iterations are random move choices, so UCT_2BF unduly outperforms
        // UCT_BF and the point defies the trend.
        let fitted: Vec<(f64, f64)> = scores
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, &s)| ((i + 1) as f64, s))
            .collect();
        let (slope, intercept) = least_squares(&fitted);
        let y = (slope * (m as f64 + 1.0) + intercept).clamp(0.0, 1.0);
        let area: f64 = scores.iter().map(|s| s.max(0.0).powi(2)).sum();
        let area_excluding_first: f64 = scores.iter().skip(1).map(|s| s.max(0.0).powi(2)).sum();
        Trace {
            slope,
            intercept,
            y,
            area,
            area_excluding_first,
            value: y + (1.0 - y) * area,
            matches: m,
        }
    }
}

fn least_squares(points: &[(f64, f64)]) -> (f64, f64) {
    let n = points.len() as f64;
    if points.len() < 2 {
        return (0.0, points.first().map(|p| p.1).unwrap_or(0.0));
    }
    let sx: f64 = points.iter().map(|p| p.0).sum();
    let sy: f64 = points.iter().map(|p| p.1).sum();
    let sxx: f64 = points.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = points.iter().map(|p| p.0 * p.1).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-15 {
        return (0.0, sy / n);
    }
    let slope = (n * sxy - sx * sy) / denom;
    (slope, (sy - slope * sx) / n)
}

/// Goodman's published two-player Skill Trace values, for the comparison figure.
///
/// These are what Overtone's number is plotted against. They are measured on a compatible
/// protocol, which is why this comparison is worth making where the chess-and-Go comparison
/// originally planned in Part VII 8 was not — those were eyeballed.
pub const PUBLISHED: [(&str, f64); 9] = [
    ("Dots + Boxes", 0.353),
    ("Dominion", 0.288),
    ("Connect 4", 0.282),
    ("Sushi Go", 0.189),
    ("Can't Stop", 0.028),
    ("Love Letter", 0.013),
    ("Stratego", 0.010),
    ("Diamant", 0.002),
    ("Tic-Tac-Toe", 0.000),
];
