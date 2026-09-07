//! The grid: one elite per cell of behaviour space.

use std::collections::HashMap;

use crate::behaviour::Behaviour;
use crate::genome::Genome;

/// How each descriptor axis is discretised.
///
/// `dim(g)` is binned on a log scale because it spans `3n` to `4^n - 1` — six to a thousand
/// across the genome range — and a linear axis would put every polynomial algebra in one
/// cell and spend the rest of the grid on the exponential ones.
#[derive(Clone, Copy, Debug)]
pub struct Bins {
    pub dim_g: usize,
    pub reach: usize,
    pub bond: usize,
    pub reach_max: f64,
}

impl Default for Bins {
    fn default() -> Self {
        Bins {
            dim_g: 6,
            reach: 8,
            bond: 4,
            reach_max: 32.0,
        }
    }
}

impl Bins {
    /// The cell a behaviour falls in.
    pub fn cell_of(&self, b: &Behaviour) -> (usize, usize, usize) {
        let d = ((b.dim_g.max(1) as f64).log2() / 11.0 * self.dim_g as f64) as usize;
        let r = (b.reach / self.reach_max * self.reach as f64) as usize;
        let c = b.bond_dimension.saturating_sub(1);
        (
            d.min(self.dim_g - 1),
            r.min(self.reach - 1),
            c.min(self.bond - 1),
        )
    }

    pub fn capacity(&self) -> usize {
        self.dim_g * self.reach * self.bond
    }
}

/// One occupied cell.
#[derive(Clone, Debug)]
pub struct Cell {
    pub genome: Genome,
    pub behaviour: Behaviour,
    pub fitness: f64,
    /// Which evaluation first filled this cell. The fill order is what the coverage curve
    /// is drawn from.
    pub found_at: usize,
}

/// The Menagerie.
#[derive(Clone, Debug)]
pub struct Archive {
    pub bins: Bins,
    cells: HashMap<(usize, usize, usize), Cell>,
    coverage: Vec<usize>,
    evaluations: usize,
}

impl Archive {
    pub fn new(bins: Bins) -> Archive {
        Archive {
            bins,
            cells: HashMap::new(),
            coverage: Vec::new(),
            evaluations: 0,
        }
    }

    /// Offer an agent to the archive. Returns true if it took a cell — either an empty one,
    /// or one whose elite it beat.
    pub fn offer(&mut self, genome: Genome, behaviour: Behaviour, fitness: f64) -> bool {
        self.evaluations += 1;
        let key = self.bins.cell_of(&behaviour);
        let better = match self.cells.get(&key) {
            None => true,
            Some(existing) => fitness > existing.fitness,
        };
        if better {
            self.cells.insert(
                key,
                Cell {
                    genome,
                    behaviour,
                    fitness,
                    found_at: self.evaluations,
                },
            );
        }
        self.coverage.push(self.cells.len());
        better
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    pub fn evaluations(&self) -> usize {
        self.evaluations
    }

    /// Fraction of reachable cells filled. Not every cell is reachable — `dim(g)` and reach
    /// are not independent, since both grow with the qubit count — so this is a coverage of
    /// the grid, not of the space, and it is reported as such.
    pub fn coverage(&self) -> f64 {
        self.cells.len() as f64 / self.bins.capacity() as f64
    }

    /// Cells filled after each evaluation.
    pub fn coverage_trace(&self) -> &[usize] {
        &self.coverage
    }

    pub fn cells(&self) -> impl Iterator<Item = (&(usize, usize, usize), &Cell)> {
        self.cells.iter()
    }

    pub fn get(&self, key: &(usize, usize, usize)) -> Option<&Cell> {
        self.cells.get(key)
    }

    /// The single best agent, which is what a return-maximising search would have aimed at.
    pub fn best(&self) -> Option<&Cell> {
        self.cells
            .values()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
    }

    /// Best fitness among cells in a given `dim(g)` band. This is the quantity that renders
    /// the barren plateau as a map: it should fall as the band index rises.
    pub fn best_in_dim_band(&self, band: usize) -> Option<f64> {
        self.cells
            .iter()
            .filter(|(k, _)| k.0 == band)
            .map(|(_, c)| c.fitness)
            .fold(None, |acc: Option<f64>, f| {
                Some(acc.map_or(f, |a: f64| a.max(f)))
            })
    }

    /// Elites sorted by fitness, best first.
    pub fn ranked(&self) -> Vec<&Cell> {
        let mut v: Vec<&Cell> = self.cells.values().collect();
        v.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        v
    }
}

impl Archive {
    /// The archive as JSON, hand-written to keep this crate free of a serialisation
    /// dependency.
    ///
    /// This is the artifact Part VIII 4 wants: the league's cold start is a hundred elites
    /// spanning the descriptor space, so that the first person to submit an agent joins a
    /// populated and genuinely diverse ladder rather than an empty one. Every field here is
    /// measured; nothing in it is authored.
    pub fn to_json(&self) -> String {
        let mut out = String::from("{\n");
        out.push_str(&format!(
            "  \"evaluations\": {},\n  \"cells_filled\": {},\n  \"grid_capacity\": {},\n",
            self.evaluations,
            self.cells.len(),
            self.bins.capacity()
        ));
        out.push_str("  \"elites\": [\n");
        let ranked = self.ranked();
        for (i, c) in ranked.iter().enumerate() {
            out.push_str(&format!(
                "    {{\"return\": {:.9}, \"dim_g\": {}, \"reach\": {:.6}, \
                 \"bond_dimension\": {}, \"entropy\": {:.6}, \"qubits\": {}, \"layers\": {}, \
                 \"policy\": \"{}\", \"lambda\": \"{}\", \"entangle\": {}, \"found_at\": {}, \
                 \"params\": [{}]}}{}\n",
                c.fitness,
                c.behaviour.dim_g,
                c.behaviour.reach,
                c.behaviour.bond_dimension,
                c.behaviour.entropy,
                c.genome.qubits,
                c.genome.layers,
                if c.genome.softmax { "softmax" } else { "raw" },
                if c.genome.trainable_lambda {
                    "trainable"
                } else {
                    "pinned"
                },
                c.genome.entangle,
                c.found_at,
                c.genome
                    .params
                    .iter()
                    .map(|p| format!("{p:.9}"))
                    .collect::<Vec<_>>()
                    .join(", "),
                if i + 1 == ranked.len() { "" } else { "," }
            ));
        }
        out.push_str("  ]\n}\n");
        out
    }
}
