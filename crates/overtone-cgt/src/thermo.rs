//! Thermographs: two walls, a mast, and the temperature at its base.
//!
//! The construction is the standard one (Winning Ways; Berlekamp 1996; Blom ch. 5). For a
//! position `G` that is not a number, the walls before clamping are
//!
//! ```text
//! LW(t) = max over G^L of  RW_{G^L}(t) - t
//! RW(t) = min over G^R of  LW_{G^R}(t) + t
//! ```
//!
//! `LW` is non-increasing, `RW` is non-decreasing, so they cross exactly once; the crossing
//! is the **temperature**, the common value there is the **mean**, and above the crossing
//! both walls are the vertical mast at the mean. A number's thermograph is a mast all the
//! way down; by the usual convention its temperature is `-1`.
//!
//! Because the left wall is a max of right walls shifted by `-t`, and the right wall a min of
//! left walls shifted by `+t`, left walls have slopes in `{-1, 0}` and right walls slopes in
//! `{0, +1}`. That is what makes [`Wall`] representable exactly as a breakpoint list plus a
//! trailing slope, and what guarantees the crossing exists: past every option's own
//! temperature the two walls have slopes `-1` and `+1`, so their difference falls at rate 2.

use crate::EPS;

/// A piecewise-linear wall of a thermograph, as a function of temperature `t >= 0`.
///
/// `pts` is a list of breakpoints in strictly increasing `t`, starting at `t = 0`; the wall
/// is linear between consecutive breakpoints and continues past the last one with slope
/// [`Wall::tail`].
#[derive(Clone, Debug, PartialEq)]
pub struct Wall {
    pts: Vec<(f64, f64)>,
    tail: f64,
}

impl Wall {
    /// The constant wall at `v` -- the thermograph of a number, and the mast of any game.
    pub fn constant(v: f64) -> Wall {
        Wall {
            pts: vec![(0.0, v)],
            tail: 0.0,
        }
    }

    /// The wall's breakpoints, in increasing `t`.
    pub fn breakpoints(&self) -> &[(f64, f64)] {
        &self.pts
    }

    /// The slope beyond the last breakpoint.
    pub fn tail(&self) -> f64 {
        self.tail
    }

    /// The wall's height at temperature `t`.
    pub fn eval(&self, t: f64) -> f64 {
        let last = self.pts.len() - 1;
        if t >= self.pts[last].0 {
            return self.pts[last].1 + self.tail * (t - self.pts[last].0);
        }
        for w in 0..last {
            let (t0, v0) = self.pts[w];
            let (t1, v1) = self.pts[w + 1];
            if t <= t1 {
                return v0 + (v1 - v0) * (t - t0) / (t1 - t0);
            }
        }
        self.pts[0].1
    }

    /// The slope of the segment starting at `t`.
    fn slope_at(&self, t: f64) -> f64 {
        let last = self.pts.len() - 1;
        if t >= self.pts[last].0 - EPS {
            return self.tail;
        }
        for w in 0..last {
            let (t0, v0) = self.pts[w];
            let (t1, v1) = self.pts[w + 1];
            if t < t1 - EPS {
                let _ = t0;
                return (v1 - v0) / (t1 - t0);
            }
        }
        self.tail
    }

    /// Add `s * t` to the wall: the `-t` of the left wall and the `+t` of the right wall.
    pub fn add_slope(&self, s: f64) -> Wall {
        Wall {
            pts: self.pts.iter().map(|&(t, v)| (t, v + s * t)).collect(),
            tail: self.tail + s,
        }
    }

    /// Pointwise maximum -- the combining rule for left walls.
    pub fn max(&self, other: &Wall) -> Wall {
        self.combine(other, true)
    }

    /// Pointwise minimum -- the combining rule for right walls.
    pub fn min(&self, other: &Wall) -> Wall {
        self.combine(other, false)
    }

    fn combine(&self, other: &Wall, take_max: bool) -> Wall {
        let mut ts: Vec<f64> = self
            .pts
            .iter()
            .map(|p| p.0)
            .chain(other.pts.iter().map(|p| p.0))
            .collect();
        sort_dedup(&mut ts);
        // A crossing inside a segment becomes a breakpoint of the result. Both walls are
        // linear on each interval between consecutive `ts`, so at most one crossing lands in
        // each, and it is dyadic whenever the endpoints are.
        let mut extra = Vec::new();
        for w in 0..ts.len() {
            let t0 = ts[w];
            let t1 = ts.get(w + 1).copied();
            let (sa, sb) = (self.slope_at(t0), other.slope_at(t0));
            if (sa - sb).abs() <= EPS {
                continue;
            }
            let tc = t0 + (other.eval(t0) - self.eval(t0)) / (sa - sb);
            let inside = tc > t0 + EPS && t1.map_or(true, |t1| tc < t1 - EPS);
            if inside {
                extra.push(tc);
            }
        }
        ts.extend(extra);
        sort_dedup(&mut ts);
        let pick = |a: f64, b: f64| if take_max { a.max(b) } else { a.min(b) };
        let pts = ts
            .iter()
            .map(|&t| (t, pick(self.eval(t), other.eval(t))))
            .collect();
        let tail = pick(self.tail, other.tail);
        let mut w = Wall { pts, tail };
        w.simplify();
        w
    }

    /// Replace everything above `t` with the mast at `v`.
    fn clamp(&self, t: f64, v: f64) -> Wall {
        let mut pts: Vec<(f64, f64)> = self.pts.iter().copied().filter(|p| p.0 < t - EPS).collect();
        pts.push((t, v));
        let mut w = Wall { pts, tail: 0.0 };
        w.simplify();
        w
    }

    /// Drop breakpoints that sit on the straight line between their neighbours.
    fn simplify(&mut self) {
        let mut out: Vec<(f64, f64)> = Vec::with_capacity(self.pts.len());
        for w in 0..self.pts.len() {
            if w == 0 || w + 1 == self.pts.len() {
                out.push(self.pts[w]);
                continue;
            }
            let (t0, v0) = *out.last().unwrap();
            let (t1, v1) = self.pts[w];
            let (t2, v2) = self.pts[w + 1];
            let s1 = (v1 - v0) / (t1 - t0);
            let s2 = (v2 - v1) / (t2 - t1);
            if (s1 - s2).abs() > EPS {
                out.push(self.pts[w]);
            }
        }
        // The last breakpoint is redundant if the segment before it already has the tail slope.
        while out.len() >= 2 {
            let n = out.len();
            let (t0, v0) = out[n - 2];
            let (t1, v1) = out[n - 1];
            if ((v1 - v0) / (t1 - t0) - self.tail).abs() > EPS {
                break;
            }
            out.remove(n - 1);
        }
        self.pts = out;
    }
}

fn sort_dedup(ts: &mut Vec<f64>) {
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ts.dedup_by(|a, b| (*a - *b).abs() < EPS);
}

/// A game's thermograph: both walls, the temperature, and the mean value.
#[derive(Clone, Debug, PartialEq)]
pub struct Thermograph {
    /// The left wall, clamped at the mast.
    pub left: Wall,
    /// The right wall, clamped at the mast.
    pub right: Wall,
    /// The base of the mast. `-1` for a number, by the usual convention.
    pub temperature: f64,
    /// The mean value: the height of the mast.
    pub mean: f64,
}

impl Thermograph {
    /// The thermograph of a number: a mast at `v` with no walls above it.
    pub fn number(v: f64) -> Thermograph {
        Thermograph {
            left: Wall::constant(v),
            right: Wall::constant(v),
            temperature: -1.0,
            mean: v,
        }
    }

    /// Build a thermograph from the thermographs of the options, per the recursion in the
    /// module docs. Panics if either option list is empty: this crate represents a position
    /// with no options on one side as a [`crate::Game::Number`], which is what the stopping
    /// formulation of a scored game requires.
    pub fn from_options(left: &[Thermograph], right: &[Thermograph]) -> Thermograph {
        assert!(
            !left.is_empty() && !right.is_empty(),
            "an option list is empty"
        );
        let raw_left = left
            .iter()
            .map(|t| t.right.add_slope(-1.0))
            .reduce(|a, b| a.max(&b))
            .expect("non-empty");
        let raw_right = right
            .iter()
            .map(|t| t.left.add_slope(1.0))
            .reduce(|a, b| a.min(&b))
            .expect("non-empty");
        let (temperature, mean) = crossing(&raw_left, &raw_right);
        Thermograph {
            left: raw_left.clamp(temperature, mean),
            right: raw_right.clamp(temperature, mean),
            temperature,
            mean,
        }
    }
}

/// The unique `t >= 0` where a non-increasing left wall meets a non-decreasing right wall,
/// and the common value there.
fn crossing(left: &Wall, right: &Wall) -> (f64, f64) {
    let d = |t: f64| left.eval(t) - right.eval(t);
    if d(0.0) <= EPS {
        return (0.0, left.eval(0.0).max(right.eval(0.0)));
    }
    let mut ts: Vec<f64> = left
        .pts
        .iter()
        .map(|p| p.0)
        .chain(right.pts.iter().map(|p| p.0))
        .collect();
    sort_dedup(&mut ts);
    for w in 0..ts.len() {
        let t0 = ts[w];
        let t1 = ts.get(w + 1).copied();
        let slope = left.slope_at(t0) - right.slope_at(t0);
        if slope >= -EPS {
            continue;
        }
        let tc = t0 - d(t0) / slope;
        if tc >= t0 - EPS && t1.map_or(true, |t1| tc <= t1 + EPS) {
            return (tc, left.eval(tc));
        }
    }
    unreachable!(
        "a short game's walls always cross: past every option's temperature the \
                  left wall falls at rate 1 and the right wall rises at rate 1"
    );
}
