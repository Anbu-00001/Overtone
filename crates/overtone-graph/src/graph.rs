//! The maze as a graph.
//!
//! Part V 1.1's observation needs one: a classical random walk evolves under `e^(-Lt)`, a
//! continuous-time quantum walk under `e^(-iLt)`, and both are diagonal in the eigenbasis of
//! the same Laplacian. Everything in this crate hangs off that one matrix.

use overtone_wfc::{Wfc, TILES};

/// An undirected, unweighted graph.
#[derive(Clone, Debug)]
pub struct Graph {
    adjacency: Vec<Vec<usize>>,
    /// Grid coordinates of each vertex, when the graph came from a maze. Rendering needs
    /// them; nothing in the mathematics does.
    coords: Vec<(usize, usize)>,
}

impl Graph {
    pub fn from_edges(n: usize, edges: &[(usize, usize)]) -> Graph {
        let mut adjacency = vec![Vec::new(); n];
        for &(a, b) in edges {
            if a != b && !adjacency[a].contains(&b) {
                adjacency[a].push(b);
                adjacency[b].push(a);
            }
        }
        Graph {
            adjacency,
            coords: (0..n).map(|i| (i, 0)).collect(),
        }
    }

    /// The corridor graph of a finished Wave Function Collapse maze.
    ///
    /// Two neighbouring cells are joined when the sockets they present to each other are both
    /// open. WFC's own constraint guarantees those agree, so checking one side would do — both
    /// are checked anyway, because a graph built on an unverified assumption would put the
    /// error in every eigenvector downstream.
    pub fn from_wfc(wfc: &Wfc) -> Graph {
        let (w, h) = (wfc.width, wfc.height);
        let mut edges = Vec::new();
        let idx = |x: usize, y: usize| y * w + x;
        for y in 0..h {
            for x in 0..w {
                let Some(t) = wfc.tile_at(x, y) else { continue };
                if x + 1 < w {
                    if let Some(r) = wfc.tile_at(x + 1, y) {
                        if TILES[t].right == 1 && TILES[r].left == 1 {
                            edges.push((idx(x, y), idx(x + 1, y)));
                        }
                    }
                }
                if y + 1 < h {
                    if let Some(d) = wfc.tile_at(x, y + 1) {
                        if TILES[t].down == 1 && TILES[d].up == 1 {
                            edges.push((idx(x, y), idx(x, y + 1)));
                        }
                    }
                }
            }
        }
        let mut g = Graph::from_edges(w * h, &edges);
        g.coords = (0..w * h).map(|i| (i % w, i / w)).collect();
        g
    }

    pub fn order(&self) -> usize {
        self.adjacency.len()
    }

    pub fn neighbours(&self, i: usize) -> &[usize] {
        &self.adjacency[i]
    }

    pub fn degree(&self, i: usize) -> usize {
        self.adjacency[i].len()
    }

    pub fn coords(&self) -> &[(usize, usize)] {
        &self.coords
    }

    pub fn edge_count(&self) -> usize {
        self.adjacency.iter().map(|a| a.len()).sum::<usize>() / 2
    }

    /// True when every vertex has the same degree, which is the condition under which the
    /// two exponents of Part V 1.3 differ by nothing observable — see [`crate::evolve`].
    pub fn is_regular(&self) -> bool {
        let d = self.degree(0);
        (0..self.order()).all(|i| self.degree(i) == d)
    }

    /// `L = D - A`, dense and row-major.
    pub fn laplacian(&self) -> Vec<f64> {
        let n = self.order();
        let mut l = vec![0.0; n * n];
        for i in 0..n {
            l[i * n + i] = self.degree(i) as f64;
            for &j in &self.adjacency[i] {
                l[i * n + j] = -1.0;
            }
        }
        l
    }

    /// `A`, dense and row-major.
    pub fn adjacency_matrix(&self) -> Vec<f64> {
        let n = self.order();
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for &j in &self.adjacency[i] {
                a[i * n + j] = 1.0;
            }
        }
        a
    }

    /// Connected components, as vertex lists.
    pub fn components(&self) -> Vec<Vec<usize>> {
        let n = self.order();
        let mut seen = vec![false; n];
        let mut out = Vec::new();
        for start in 0..n {
            if seen[start] {
                continue;
            }
            let mut stack = vec![start];
            let mut comp = Vec::new();
            seen[start] = true;
            while let Some(v) = stack.pop() {
                comp.push(v);
                for &u in &self.adjacency[v] {
                    if !seen[u] {
                        seen[u] = true;
                        stack.push(u);
                    }
                }
            }
            comp.sort_unstable();
            out.push(comp);
        }
        out
    }

    /// The largest connected component, renumbered from zero.
    ///
    /// A WFC maze leaves isolated cells wherever the empty tile landed, and an isolated
    /// vertex contributes a zero eigenvalue of its own. Left in, the low-order eigenvectors
    /// would be a basis for the *disconnected* part of the maze — technically correct and
    /// visually useless.
    pub fn largest_component(&self) -> Graph {
        let comps = self.components();
        let biggest = comps
            .iter()
            .max_by_key(|c| c.len())
            .cloned()
            .unwrap_or_default();
        let mut remap = vec![usize::MAX; self.order()];
        for (new, &old) in biggest.iter().enumerate() {
            remap[old] = new;
        }
        let mut edges = Vec::new();
        for &old in &biggest {
            for &j in &self.adjacency[old] {
                if remap[j] != usize::MAX && remap[old] < remap[j] {
                    edges.push((remap[old], remap[j]));
                }
            }
        }
        let mut g = Graph::from_edges(biggest.len(), &edges);
        g.coords = biggest.iter().map(|&o| self.coords[o]).collect();
        g
    }

    /// Longest shortest-path between any two vertices, within each component.
    ///
    /// The LMDP needs it: the desirability at distance `d` is about `exp(-rho * d)`, so the
    /// diameter is what decides how large `rho` can be before the far end of the graph
    /// underflows. See [`crate::lmdp::Lmdp::recommended_rho`].
    pub fn diameter(&self) -> usize {
        (0..self.order())
            .filter_map(|i| self.bfs_distances(&[i]).into_iter().flatten().max())
            .max()
            .unwrap_or(0)
    }

    /// Hop distance from the nearest source, by breadth-first search.
    ///
    /// The exact answer the LMDP eigensolve is checked against. Edges are unweighted, so BFS
    /// *is* Dijkstra here, and it is exact rather than approximate — which is the whole point
    /// of using it as the oracle.
    pub fn bfs_distances(&self, sources: &[usize]) -> Vec<Option<usize>> {
        let mut dist = vec![None; self.order()];
        let mut queue = std::collections::VecDeque::new();
        for &s in sources {
            dist[s] = Some(0);
            queue.push_back(s);
        }
        while let Some(v) = queue.pop_front() {
            let d = dist[v].unwrap();
            for &u in &self.adjacency[v] {
                if dist[u].is_none() {
                    dist[u] = Some(d + 1);
                    queue.push_back(u);
                }
            }
        }
        dist
    }
}
