//! Short games, in the stopping formulation a scored game needs.
//!
//! A position is either a [`Game::Number`] -- play there is over and the score is settled --
//! or a set of Left and Right options. This is deliberately *not* a canonical-form
//! implementation: it does not detect that some `{L | R}` is secretly a number, because it
//! does not have to. Every game this crate is asked about is either given with its numeric
//! positions already declared (the corridor recursion below is stated that way in the
//! literature) or built by [`crate::decompose`] from measured Overtone quantities, which
//! bottom out in numbers by construction.
//!
//! # The corridor
//!
//! [`Game::corridor`] is the family that makes M48's acceptance test possible. A closed,
//! empty corridor of length `n` on a Go board is a real endgame position, and Berlekamp &
//! Wolfe's analysis of it reduces to a recursion with no board in it at all:
//!
//! ```text
//! Corr(0)     = 0
//! Corr(n + 1) = { n | Corr(n) }
//! ```
//!
//! One player can take `n - 1` points in gote; the other can shrink the corridor by one
//! intersection; those options dominate every other move. The published result is that
//! chilling by one turns it into a number:
//!
//! ```text
//! f(Corr(n)) = n - 2 + (1/2)^(n-1)
//! ```
//!
//! which is the mean value, and the temperature that falls out alongside it is
//! `1 - (1/2)^(n-1)` -- below 1 for every `n`, which is *why* chilling by one lands on a
//! number. `tests/corridor.rs` checks both against the closed forms, so the implementation
//! is pinned to a published Go endgame and not to itself.

use crate::thermo::Thermograph;

/// A short combinatorial game.
#[derive(Clone, Debug, PartialEq)]
pub enum Game {
    /// A settled position worth `x` to Left.
    Number(f64),
    /// `{ left | right }`. Both lists must be non-empty.
    Options {
        /// Left's options.
        left: Vec<Game>,
        /// Right's options.
        right: Vec<Game>,
    },
}

impl Game {
    /// A settled position.
    pub fn number(x: f64) -> Game {
        Game::Number(x)
    }

    /// `{ left | right }`.
    ///
    /// # Panics
    ///
    /// If an option list is empty, or if every option is a number and the left options do not
    /// reach the right ones. That second case is the crate's stated limitation made into a
    /// guard: `{a | b}` with `a < b` is *equal to a number* by the simplicity rule, and this
    /// crate does not detect that, so it would report a nonsense temperature for it. Catching
    /// it at construction is cheap and it caught a bad search result during M48's development.
    pub fn new(left: Vec<Game>, right: Vec<Game>) -> Game {
        assert!(
            !left.is_empty() && !right.is_empty(),
            "an option list is empty"
        );
        let ln: Option<Vec<f64>> = left.iter().map(Game::value).collect();
        let rn: Option<Vec<f64>> = right.iter().map(Game::value).collect();
        if let (Some(ln), Some(rn)) = (ln, rn) {
            let lo = ln.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let hi = rn.iter().cloned().fold(f64::INFINITY, f64::min);
            assert!(
                lo >= hi,
                "{{{lo} | {hi}}} is a number by the simplicity rule; this crate takes games \
                 with their numeric positions already declared (see the module docs)"
            );
        }
        Game::Options { left, right }
    }

    /// The switch `{ a | b }`: mean `(a + b) / 2`, temperature `(a - b) / 2`.
    pub fn switch(a: f64, b: f64) -> Game {
        Game::new(vec![Game::number(a)], vec![Game::number(b)])
    }

    /// `* = { 0 | 0 }`: temperature 0, mean 0.
    pub fn star() -> Game {
        Game::switch(0.0, 0.0)
    }

    /// The Go corridor `Corr(n)`, from the recursion in the module docs.
    pub fn corridor(n: usize) -> Game {
        let mut g = Game::number(0.0);
        for k in 0..n {
            g = Game::new(vec![Game::number(k as f64)], vec![g]);
        }
        g
    }

    /// Left's options, or an empty slice for a number.
    pub fn left_options(&self) -> &[Game] {
        match self {
            Game::Number(_) => &[],
            Game::Options { left, .. } => left,
        }
    }

    /// Right's options, or an empty slice for a number.
    pub fn right_options(&self) -> &[Game] {
        match self {
            Game::Number(_) => &[],
            Game::Options { right, .. } => right,
        }
    }

    /// True if play here is over.
    pub fn is_number(&self) -> bool {
        matches!(self, Game::Number(_))
    }

    /// The settled score, for a number.
    pub fn value(&self) -> Option<f64> {
        match self {
            Game::Number(x) => Some(*x),
            Game::Options { .. } => None,
        }
    }

    /// The full thermograph, computed bottom-up.
    pub fn thermograph(&self) -> Thermograph {
        match self {
            Game::Number(x) => Thermograph::number(*x),
            Game::Options { left, right } => {
                let l: Vec<Thermograph> = left.iter().map(|g| g.thermograph()).collect();
                let r: Vec<Thermograph> = right.iter().map(|g| g.thermograph()).collect();
                Thermograph::from_options(&l, &r)
            }
        }
    }

    /// The base of the mast. `-1` for a number, by convention.
    pub fn temperature(&self) -> f64 {
        self.thermograph().temperature
    }

    /// The height of the mast.
    pub fn mean(&self) -> f64 {
        self.thermograph().mean
    }

    /// True if this position and every unsettled position under it is strictly **hot** --
    /// left stop above right stop, equivalently temperature above zero.
    ///
    /// This is the crate's soundness predicate. Thermography here assumes the games it is
    /// given are not secretly numbers, and the simplicity rule makes "secretly a number"
    /// easy to stumble into: `{{-1|-2} | 1}` is a number, because every left option is
    /// strictly below every right option, even though no option list is all-numeric. A game
    /// that is hot all the way down cannot be a number, since numbers have equal stops. Any
    /// experiment that generates games rather than declaring them should filter on this.
    pub fn is_hot(&self) -> bool {
        match self {
            Game::Number(_) => true,
            Game::Options { left, right } => {
                self.temperature() > 0.0
                    && left.iter().all(Game::is_hot)
                    && right.iter().all(Game::is_hot)
            }
        }
    }

    /// The left stop: the score if Left moves first and both sides stop at a number.
    pub fn left_stop(&self) -> f64 {
        match self {
            Game::Number(x) => *x,
            Game::Options { left, .. } => left
                .iter()
                .map(Game::right_stop)
                .fold(f64::NEG_INFINITY, f64::max),
        }
    }

    /// The right stop: the score if Right moves first.
    pub fn right_stop(&self) -> f64 {
        match self {
            Game::Number(x) => *x,
            Game::Options { right, .. } => right
                .iter()
                .map(Game::left_stop)
                .fold(f64::INFINITY, f64::min),
        }
    }

    /// How many nodes the tree holds -- a cost proxy for [`crate::decompose`].
    pub fn nodes(&self) -> usize {
        1 + self.left_options().iter().map(Game::nodes).sum::<usize>()
            + self.right_options().iter().map(Game::nodes).sum::<usize>()
    }
}
