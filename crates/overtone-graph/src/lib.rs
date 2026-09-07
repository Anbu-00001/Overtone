//! The maze's Laplacian, and the two things that fall out of diagonalising it.
//!
//! Part V 1.1 is the unification the series was missing. Proto-value functions (Mahadevan,
//! ICML 2005) are RL basis functions learned from the *topology* of the state space rather
//! than from rewards — in the discrete case, the eigenvectors of the graph Laplacian. A
//! classical random walk evolves under `e^(-Lt)`; a continuous-time quantum walk evolves
//! under an imaginary exponent of the same operator. Same graph, same matrix, same basis:
//!
//! > The eigenvectors of the maze's Laplacian are simultaneously the RL basis functions of
//! > proto-value function theory, the eigenmodes of the quantum walk, and the graph Fourier
//! > transform of the maze.
//!
//! And Part V 2 puts a second thing in the same eigenbasis: Todorov's linearly-solvable MDP,
//! where the exponential transform `z = exp(-v)` turns Bellman's equation into `z = GPz` and
//! optimal control into a largest-eigenvector problem. One decomposition, three fields.
//!
//! The graph itself comes from `overtone-wfc` — the maze the Part IV panel collapses is a
//! real corridor graph, so the eigenbasis is computed on something the reader watched being
//! built rather than on a lattice conjured for the occasion.
//!
//! # Example
//!
//! ```
//! use overtone_graph::{Graph, Lmdp};
//!
//! // A path of five vertices; the exit is at one end.
//! let g = Graph::from_edges(5, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
//! let lmdp = Lmdp::new(&g, vec![0], 50.0);
//! let sol = lmdp.desirability(20_000, 1e-14);
//! let hops = lmdp.shortest_paths(&sol.z);
//! // Vertex 4 is four hops from the exit, recovered from an eigenproblem.
//! assert!((hops[4] - 4.0).abs() < 0.05, "got {}", hops[4]);
//! ```

#![forbid(unsafe_code)]

pub mod evolve;
pub mod graph;
pub mod lmdp;

pub use evolve::{mean_distance, Eigenbasis};
pub use graph::Graph;
pub use lmdp::{compose, shortest_path_window, Lmdp, Solution};
