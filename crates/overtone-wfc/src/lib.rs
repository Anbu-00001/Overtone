// Overtone. Copyright (C) 2026 Anbuchelvan Ganesan.
// Licensed under the GNU Affero General Public License, version 3 or (at your option) any
// later version. See LICENSE, or <https://www.gnu.org/licenses/>. This program is
// distributed WITHOUT ANY WARRANTY; see the licence for details.

//! Wave Function Collapse — the metaphor, kept honestly beside the physics.
//!
//! Part IV 3.1. WFC borrows quantum vocabulary: tiles in "superposition", an "observation"
//! that "collapses" the lowest-entropy cell, constraints propagating outward. Overtone
//! contains the literal version of every one of those words, so the two are worth showing
//! at once — WFC laying down a maze with its Shannon entropy over remaining tile options on
//! one side, the amplitude field on the finished maze with its von Neumann entropy on the
//! other.
//!
//! **WFC is not quantum** (Part IV 7). Nothing here is a physical claim, and the entire
//! value of the panel is the contrast. Two differences are worth naming, because they are
//! what a reader takes away:
//!
//! - WFC's superposition is a *set of possibilities with no phase*. There is no
//!   interference, so no possibility can ever cancel another. The whole of Part I is about
//!   what phase does.
//! - WFC's collapse is a choice made by a random number generator, and its "entropy" is a
//!   property of the solver's ignorance. Von Neumann entropy is a property of the state.
//!
//! Tile weights are uniform here, deliberately. A weighted tileset would need numbers
//! chosen to make the output look right, and with uniform weights a cell's Shannon entropy
//! is exactly `ln(options)` — which states the contrast more sharply than any tuning could.

#![forbid(unsafe_code)]

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Sockets on the four edges of a tile, in the order left, right, up, down.
/// `1` is an opening, `0` a wall. Two tiles may sit side by side when the sockets they
/// present to each other agree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    pub name: &'static str,
    pub left: u8,
    pub right: u8,
    pub up: u8,
    pub down: u8,
}

/// A pipe tileset: the eight ways a corridor can pass through a cell.
pub const TILES: [Tile; 8] = [
    Tile {
        name: "empty",
        left: 0,
        right: 0,
        up: 0,
        down: 0,
    },
    Tile {
        name: "horizontal",
        left: 1,
        right: 1,
        up: 0,
        down: 0,
    },
    Tile {
        name: "vertical",
        left: 0,
        right: 0,
        up: 1,
        down: 1,
    },
    Tile {
        name: "elbow-ur",
        left: 0,
        right: 1,
        up: 1,
        down: 0,
    },
    Tile {
        name: "elbow-ul",
        left: 1,
        right: 0,
        up: 1,
        down: 0,
    },
    Tile {
        name: "elbow-dr",
        left: 0,
        right: 1,
        up: 0,
        down: 1,
    },
    Tile {
        name: "elbow-dl",
        left: 1,
        right: 0,
        up: 0,
        down: 1,
    },
    Tile {
        name: "cross",
        left: 1,
        right: 1,
        up: 1,
        down: 1,
    },
];

/// The state of one WFC run.
pub struct Wfc {
    pub width: usize,
    pub height: usize,
    /// Bitmask of still-possible tiles per cell.
    options: Vec<u16>,
    rng: ChaCha8Rng,
    /// Total Shannon entropy of the grid after each observation.
    entropy_trace: Vec<f64>,
    contradictions: usize,
}

const ALL: u16 = (1u16 << 8) - 1;

/// A neighbour direction: the offset, the socket this tile presents that way, and the socket
/// the neighbour presents back.
type Direction = (i64, i64, fn(&Tile) -> u8, fn(&Tile) -> u8);

impl Wfc {
    pub fn new(width: usize, height: usize, seed: u64) -> Self {
        Wfc {
            width,
            height,
            options: vec![ALL; width * height],
            rng: ChaCha8Rng::seed_from_u64(seed),
            entropy_trace: Vec::new(),
            contradictions: 0,
        }
    }

    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    /// Tiles still possible at a cell.
    pub fn options_at(&self, x: usize, y: usize) -> u16 {
        self.options[self.idx(x, y)]
    }

    /// The collapsed tile index, or `None` while the cell is still in superposition.
    pub fn tile_at(&self, x: usize, y: usize) -> Option<usize> {
        let o = self.options_at(x, y);
        (o.count_ones() == 1).then(|| o.trailing_zeros() as usize)
    }

    /// Shannon entropy of one cell. With uniform weights this is exactly `ln(options)`.
    pub fn cell_entropy(&self, x: usize, y: usize) -> f64 {
        let n = self.options_at(x, y).count_ones();
        if n <= 1 {
            0.0
        } else {
            (n as f64).ln()
        }
    }

    /// Summed Shannon entropy of the whole grid — the quantity that falls to zero as the
    /// solver learns, and the one to put beside the von Neumann entropy.
    pub fn total_entropy(&self) -> f64 {
        (0..self.height)
            .flat_map(|y| (0..self.width).map(move |x| (x, y)))
            .map(|(x, y)| self.cell_entropy(x, y))
            .sum()
    }

    pub fn entropy_trace(&self) -> &[f64] {
        &self.entropy_trace
    }

    pub fn contradictions(&self) -> usize {
        self.contradictions
    }

    pub fn is_finished(&self) -> bool {
        self.options.iter().all(|o| o.count_ones() <= 1)
    }

    /// The lowest-entropy cell still in superposition, ties broken by a seeded draw.
    fn lowest_entropy_cell(&mut self) -> Option<(usize, usize)> {
        let mut best = u32::MAX;
        let mut count = 0;
        let mut chosen = None;
        for y in 0..self.height {
            for x in 0..self.width {
                let n = self.options[self.idx(x, y)].count_ones();
                if n <= 1 {
                    continue;
                }
                if n < best {
                    best = n;
                    count = 1;
                    chosen = Some((x, y));
                } else if n == best {
                    count += 1;
                    if self.rng.gen_range(0..count) == 0 {
                        chosen = Some((x, y));
                    }
                }
            }
        }
        chosen
    }

    /// One observation, then constraint propagation. Returns the collapsed cell.
    pub fn step(&mut self) -> Option<(usize, usize, usize)> {
        let (x, y) = self.lowest_entropy_cell()?;
        let i = self.idx(x, y);
        let opts = self.options[i];
        let n = opts.count_ones();
        let pick = self.rng.gen_range(0..n);
        let mut seen = 0;
        let mut chosen = 0;
        for t in 0..TILES.len() {
            if opts & (1 << t) != 0 {
                if seen == pick {
                    chosen = t;
                    break;
                }
                seen += 1;
            }
        }
        self.options[i] = 1 << chosen;
        self.propagate(x, y);
        self.entropy_trace.push(self.total_entropy());
        Some((x, y, chosen))
    }

    /// Constraints spread outward until nothing changes: the part of WFC that is genuinely
    /// clever and has no quantum analogue at all.
    fn propagate(&mut self, sx: usize, sy: usize) {
        let mut stack = vec![(sx, sy)];
        while let Some((x, y)) = stack.pop() {
            let here = self.options[self.idx(x, y)];
            // (dx, dy, socket on this tile, socket on the neighbour)
            let dirs: [Direction; 4] = [
                (-1, 0, |t| t.left, |t| t.right),
                (1, 0, |t| t.right, |t| t.left),
                (0, -1, |t| t.up, |t| t.down),
                (0, 1, |t| t.down, |t| t.up),
            ];
            for (dx, dy, mine, theirs) in dirs {
                let nx = x as i64 + dx;
                let ny = y as i64 + dy;
                if nx < 0 || ny < 0 || nx >= self.width as i64 || ny >= self.height as i64 {
                    continue;
                }
                let (nx, ny) = (nx as usize, ny as usize);
                // Which sockets can this cell present in that direction?
                let mut allowed = 0u8;
                for (t, tile) in TILES.iter().enumerate() {
                    if here & (1 << t) != 0 {
                        allowed |= 1 << mine(tile);
                    }
                }
                let j = self.idx(nx, ny);
                let before = self.options[j];
                let mut after = 0u16;
                for (t, tile) in TILES.iter().enumerate() {
                    if before & (1 << t) != 0 && allowed & (1 << theirs(tile)) != 0 {
                        after |= 1 << t;
                    }
                }
                if after == 0 {
                    // A contradiction. Recorded rather than retried: the panel is about
                    // watching the entropy fall, and a silent restart would hide the one
                    // genuinely interesting failure mode the algorithm has.
                    self.contradictions += 1;
                    continue;
                }
                if after != before {
                    self.options[j] = after;
                    stack.push((nx, ny));
                }
            }
        }
    }

    /// Run to completion. Returns the number of observations it took.
    pub fn run(&mut self) -> usize {
        let mut steps = 0;
        while self.step().is_some() {
            steps += 1;
        }
        steps
    }

    /// The finished grid as tile indices, or `None` for any cell left in contradiction.
    pub fn grid(&self) -> Vec<Option<usize>> {
        (0..self.height)
            .flat_map(|y| (0..self.width).map(move |x| (x, y)))
            .map(|(x, y)| self.tile_at(x, y))
            .collect()
    }
}
