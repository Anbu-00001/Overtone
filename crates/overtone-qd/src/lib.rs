//! MAP-Elites over policy agents — the Menagerie archive.
//!
//! Mouret & Clune, *Illuminating search spaces by mapping elites*, arXiv:1504.04909. Instead
//! of maximising return, keep a grid over *behaviour descriptors* and hold the best agent in
//! each cell. One trained agent is a result; an archive of them is a map.
//!
//! # The reason Part IV gives for this is wrong, and the real reason is better
//!
//! Part IV 4 justifies MAP-Elites like this: "Part III establishes that large `dim(g)`
//! implies vanishing gradients. **In a barren plateau, gradient descent is a random walk —
//! and quality-diversity search is one of the few methods that still functions.**" Part IV 7
//! then instructs that the docs explain that reasoning rather than let a reader assume the
//! grid was added because it looked nice.
//!
//! The reasoning is false, and the paper that refutes it is one this repository already
//! cites. Arrasmith, Cerezo, Czarnik, Cincio and Coles, *Effect of barren plateaus on
//! gradient-free optimization*, Quantum 5, 558 (2021), prove that **cost function
//! differences — the basis on which gradient-free optimizers make decisions — are
//! exponentially suppressed in a barren plateau**, and confirm it numerically for
//! Nelder-Mead, Powell and COBYLA. MAP-Elites decides whether an offspring replaces an elite
//! by comparing fitness, so it decides on cost differences and sits squarely inside that
//! theorem. It is not a way out. Phase 8's own exit criteria cite the same paper for "all
//! four optimisers flatline", so following Part IV's instruction literally would have put
//! two contradictory claims in one repository.
//!
//! What survives, and is the honest justification:
//!
//! - **The archive is the deliverable, not the peak.** Quality-diversity illuminates a
//!   space; the Menagerie is a map of what is reachable, not an attempt to climb higher than
//!   gradient ascent can.
//! - **The descriptors keep working where the fitness does not.** `dim(g)`, reach and bond
//!   dimension are measurable whatever the return signal does, so the map exists even where
//!   the ranking within it stops meaning anything.
//! - **A diverse population is a better depth measurement than self-play** (Part VIII 5),
//!   and Part VIII 4 makes this archive the league's cold start. That is what it is for.
//!
//! `examples/menagerie.rs` measures those rather than asserting them, and the first
//! measurement already corrected a claim that stood in this comment: the archive does **not**
//! render the barren plateau at the scale it runs at. Best fitness by `dim(g)` band comes out
//! flat — `0.492, 0.488, 0.502, 0.501, 0.471` across five bands — because two to five qubits
//! is nowhere near the regime where `Var ~ 2^(-1.03 n)` bites. The collapse is a claim about
//! scale, and this archive does not reach it. Reported, not hidden.
//!
//! What the measurement does support is the plainer thing. At a matched evaluation budget,
//! the same mutation operator and the same seed, MAP-Elites reaches `J = +0.502` where a hill
//! climb that keeps only the best reaches `+0.403`. The archive wins by holding structurally
//! different agents — qubit counts, depths, entanglers — that a single-peak search abandons
//! early. That is quality-diversity doing what quality-diversity is for, in a landscape that
//! is multimodal rather than flat, and it needs no appeal to barren plateaus at all.
//!
//! # One descriptor is substituted, deliberately
//!
//! Part IV 4 asks for axes `(dim(g), transport exponent beta, effective chi)`. A
//! `SpectralControl` policy is a contextual bandit: **it does not move**, so it has no
//! transport exponent, and `beta` belongs to the walk agents of Part IV 2.3's race. The
//! substitute is **reach**, `C * |lambda|` — the width of the frequency comb the policy can
//! represent. The analogy is exact in the way that matters: `beta` is how far in space an
//! agent can get, and reach is how far in frequency it can see. Part IV's own stat block
//! lists both lines for the same reason.

#![forbid(unsafe_code)]

pub mod archive;
pub mod behaviour;
pub mod genome;
pub mod search;

pub use archive::{Archive, Bins, Cell};
pub use behaviour::{measure, Behaviour};
pub use genome::Genome;
pub use search::{map_elites, Config};
