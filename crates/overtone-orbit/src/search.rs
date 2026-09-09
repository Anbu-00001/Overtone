//! The three searches the engine implements on a submitted agent's behalf.
//!
//! Part VIII 2's rule is that **no submitted code is ever executed**: a submission names a
//! search from a fixed vocabulary and the engine runs it. That is what makes the league
//! verifiable, and it is also what gives the language a compute axis, which Decisions-01 Q1
//! shows a declarative policy language cannot have.
//!
//! # Why MCTS is the primary kind, and what the other two are declaring
//!
//! Overtone is a **stochastic** game: rule 3's measure move collapses the field, and the
//! outcome is Born-random. Negamax assumes determinism; the correct deterministic-search
//! generalisation is expectimax, whose chance nodes have one branch per measurement outcome —
//! 1024-way at `n = 10`. That is not viable, and MCTS samples chance nodes natively.
//!
//! So the three kinds are not three implementations of the same idea:
//!
//! - `mcts` plays the game as it is, sampling collapses from the Born rule.
//! - `greedy` and `negamax` search against [`crate::game::collapse`], which takes the
//!   *likelier* branch deterministically. That is a modelling error, and it is one the
//!   submitter is choosing when they declare those kinds. It is stated in
//!   [`crate::language::LANGUAGE_V1`] because a `d` measured over this population has to say
//!   what the population believes.
//!
//! # The one constant here, and where it comes from
//!
//! [`EXPLORATION`] is `sqrt(2)`, which is Kocsis and Szepesvári's UCT constant and is derived
//! for rewards in `[0, 1]` — which is exactly the range Decisions-03 Q4 requires
//! [`crate::language::Eval::uct_value`] to map into. It is not a tuned number, and the two
//! halves of that sentence depend on each other: normalise the eval differently and `sqrt(2)`
//! stops being the right constant.
//!
//! Everything else is derived from the position. The rollout cap is the coherence horizon —
//! `2 * ceil(coherence / k) + 2` plies, after which no player can move at all — rather than a
//! chosen depth.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::game::{collapse_at, one_weight, Game, Move, Outcome};
use crate::ladder::ANGLES;
use crate::language::{Agent, Rollout, SearchKind};
use crate::thermal::{regions, temperature_field};

/// UCT's exploration constant, valid because the eval is mapped into `[0, 1]` first.
pub const EXPLORATION: f64 = std::f64::consts::SQRT_2;

/// Checkmate tolerance, the same one [`crate::game::Game::outcome`] is called with elsewhere.
const TOLERANCE: f64 = 1e-6;

/// One (move, angle) pair the agent is allowed to play.
type Candidate = (Move, f64);

/// Every candidate this agent may play at this width, in a fixed order.
///
/// Order matters: it is what makes a seeded search reproducible, and Part VIII 6 requires
/// same spec, same seed, same result.
pub fn candidates(agent: &Agent, n: usize) -> Vec<Candidate> {
    let mut out = Vec::new();
    for mv in agent.moves(n) {
        for &a in ANGLES.iter() {
            out.push((mv.clone(), a));
        }
    }
    out
}

/// How many plies remain before neither player can move.
fn horizon(game: &Game) -> usize {
    let k = game.k.max(1);
    let worst = game
        .players
        .iter()
        .map(|p| p.coherence.div_ceil(k))
        .max()
        .unwrap_or(0);
    2 * worst + 2
}

/// What the tree looked like, for the claims that are otherwise invisible from outside.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SearchStats {
    /// Nodes created, decision and chance together.
    pub nodes: usize,
    /// Children of the root. Under widening this is far below the candidate count.
    pub root_children: usize,
    /// Chance nodes -- one per measure move reached in the tree.
    pub chance_nodes: usize,
    /// Deepest path from the root, in nodes.
    pub depth: usize,
}

/// Run the agent's declared search and return the move it plays.
pub fn choose(agent: &Agent, game: &Game, rng: &mut ChaCha8Rng) -> Candidate {
    choose_with_stats(agent, game, rng).0
}

/// As [`choose`], with what the search actually built. Empty stats for the two tree-less kinds.
pub fn choose_with_stats(
    agent: &Agent,
    game: &Game,
    rng: &mut ChaCha8Rng,
) -> (Candidate, SearchStats) {
    match agent.search.kind {
        SearchKind::Greedy => (greedy(agent, game, rng), SearchStats::default()),
        SearchKind::Negamax => (negamax_root(agent, game), SearchStats::default()),
        SearchKind::Mcts => mcts(agent, game, rng),
    }
}

/// Depth-one search over as many candidates as the budget allows.
fn greedy(agent: &Agent, game: &Game, rng: &mut ChaCha8Rng) -> Candidate {
    let n = game.players[game.to_move].num_qubits;
    let cands = candidates(agent, n);
    let me = game.to_move;
    let mut best = (cands[0].clone(), f64::NEG_INFINITY);
    let budget = agent.search.budget;
    // Below the candidate count the budget samples without replacement; at or above it the
    // whole set is searched, because sampling with replacement would only re-evaluate.
    let order: Vec<usize> = if budget >= cands.len() {
        (0..cands.len()).collect()
    } else {
        let mut idx: Vec<usize> = (0..cands.len()).collect();
        let len = idx.len();
        for i in 0..budget {
            idx.swap(i, rng.gen_range(i..len));
        }
        idx.truncate(budget);
        idx
    };
    for i in order {
        let (mv, angle) = &cands[i];
        let mut trial = game.clone();
        trial.apply(mv, *angle);
        let s = agent.eval.score(&trial, me);
        if s > best.1 {
            best = ((mv.clone(), *angle), s);
        }
    }
    best.0
}

/// Iteratively deepened alpha-beta, stopped by the node budget rather than by a fixed depth.
///
/// Budget is the node allowance, not the depth: the two are related by the branching factor,
/// which Part VII 9.2 insists is measured rather than assumed, so a spec that named a depth
/// would be naming different amounts of compute at different widths. The last *completed*
/// iteration is the one whose move is played, which is the standard way to make a
/// budget-limited search safe to interrupt.
fn negamax_root(agent: &Agent, game: &Game) -> Candidate {
    let n = game.players[game.to_move].num_qubits;
    let cands = candidates(agent, n);
    let mut best = cands[0].clone();
    let budget = agent.search.budget;
    let mut depth = 1;
    loop {
        let mut nodes = 0usize;
        let mut alpha = f64::NEG_INFINITY;
        let mut this = None;
        let mut ran_out = false;
        for (mv, angle) in &cands {
            if nodes >= budget {
                ran_out = true;
                break;
            }
            nodes += 1;
            let mut child = game.clone();
            child.apply(mv, *angle);
            let v = -negamax(
                agent,
                &child,
                depth - 1,
                f64::NEG_INFINITY,
                -alpha,
                &mut nodes,
                budget,
            );
            if v > alpha {
                alpha = v;
                this = Some((mv.clone(), *angle));
            }
        }
        // Keep the best move an aborted iteration found. Every move it *did* examine was
        // searched to full depth, so the choice is grounded; discarding it would fall back to
        // the first candidate, and a negamax agent whose budget is below the candidate count
        // would then always play the same move whatever its budget was. That is what the first
        // version of this function did, and it made the compute axis flat for `kind =
        // "negamax"` at every budget under 240 -- visible only as two identical win rates.
        match this {
            Some(m) => best = m,
            None => break,
        }
        if ran_out || nodes >= budget || depth >= horizon(game) {
            break;
        }
        depth += 1;
    }
    best
}

fn negamax(
    agent: &Agent,
    game: &Game,
    depth: usize,
    mut alpha: f64,
    beta: f64,
    nodes: &mut usize,
    budget: usize,
) -> f64 {
    if depth == 0 || *nodes >= budget {
        // The eval is antisymmetric, so the value from the side to move is exactly what
        // negamax's sign convention wants. An eval scoring one side only would be wrong here
        // and would look right.
        return agent.eval.score(game, game.to_move);
    }
    let n = game.players[game.to_move].num_qubits;
    let mut best = f64::NEG_INFINITY;
    for (mv, angle) in candidates(agent, n) {
        if *nodes >= budget {
            break;
        }
        *nodes += 1;
        let mut child = game.clone();
        child.apply(&mv, angle);
        let v = -negamax(agent, &child, depth - 1, -beta, -alpha, nodes, budget);
        if v > best {
            best = v;
        }
        if v > alpha {
            alpha = v;
        }
        if alpha >= beta {
            break;
        }
    }
    if best == f64::NEG_INFINITY {
        agent.eval.score(game, game.to_move)
    } else {
        best
    }
}

/// How many regions the temperature field is taken over.
///
/// Three, matching `examples/twotemps.rs` and `examples/freeze.rs`, so a temperature that
/// orders a move here is the same number M51 measured and the same number the panel draws.
/// Capped by the width, since a region cannot be empty.
const TEMPERATURE_REGIONS: usize = 3;

/// Progressive widening: a node holds at most `ceil(WIDEN_K * visits^WIDEN_ALPHA)` children.
///
/// The rule is Coulom (2007) and Chaslot et al. (2008) as stated by Couetoux et al. (2011):
/// *the number of children of a node is artificially limited to `k N^alpha`*. `alpha = 0.5`
/// and `k = 1` are the standard values, and Browne et al. (2012) note that widening is
/// *especially* effective when a set of preferred actions is tried first — which is precisely
/// what Decisions-05 3 puts temperature in the tree to do.
///
/// Why it is needed here and not optional: at `n = 4` an agent holding all four generators has
/// 208 (move, angle) candidates. Full expansion spends the first 208 playouts of a 512 budget
/// giving every candidate exactly one visit, and the tree never gets a second ply.
const WIDEN_K: f64 = 1.0;
const WIDEN_ALPHA: f64 = 0.5;

/// One node of the tree. Decision nodes are where a player chooses; chance nodes are where the
/// Born rule does.
struct Node {
    game: Game,
    parent: Option<usize>,
    /// The player who moved into this node, if any.
    mover: Option<usize>,
    visits: f64,
    value: f64,
    kind: Kind,
}

enum Kind {
    /// `children` pairs a candidate index with its node. The expansion *order* is shared by
    /// every decision node in the tree, because the heuristic is a property of the move
    /// evaluated once at the root -- see [`heuristic`].
    Decision { children: Vec<(usize, usize)> },
    /// A measurement in progress: which qubit, the Born weight of the one-branch, and the two
    /// outcome nodes.
    ///
    /// **This is where Decisions-05 4 and the implemented rule disagree, and the code follows
    /// the rule.** The ruling requires double progressive widening at chance nodes on the
    /// grounds that "at `n = 10` a position measurement has up to 1024 outcomes". Part VII 5
    /// rule 3's measure move is `Move::Measure { qubit }` — it collapses **one** qubit, so a
    /// chance node here has exactly **two** outcomes and both are enumerated with their exact
    /// Born weights. There is nothing to widen and nothing to sample: a distribution you can
    /// write down is not one you should be drawing from.
    ///
    /// The ruling's underlying point survives and is implemented: chance is explicit in the
    /// tree rather than folded into whichever outcome happened to be sampled first. If the
    /// rule ever becomes a register-wide measurement, this is the node that needs widening.
    Chance {
        qubit: usize,
        one_weight: f64,
        outcomes: [Option<usize>; 2],
    },
}

/// The progressive-bias heuristic for each candidate: the temperature of the hottest region
/// the move touches, in the position the search starts from.
///
/// # Once per move, not once per node, and that is a measurement not a shortcut
///
/// A literal reading of Decisions-05 1 wants `H(s,a)` at every node. Measured, at `n = 4` with
/// three regions, one temperature field costs **577 microseconds** — and MCTS creates about one
/// node per playout, so recomputing it per node would cost 0.3 s at budget 512 and 2.4 s at
/// 4096, against the 90 ms the entire rest of the search takes at 512. **The heuristic would
/// cost five to eight times the search it is guiding.**
///
/// So `H` is a property of the *move*, evaluated once in the root position, and it is stale
/// deeper in the tree. That is the situation progressive bias is built for: the `1/(1+n)` decay
/// hands control back to the empirical mean exactly where the heuristic is least trustworthy,
/// which is the property Decisions-05 1 chose it over PUCT for.
fn heuristic(agent: &Agent, game: &Game, cands: &[Candidate]) -> Vec<f64> {
    let n = game.players[game.to_move].num_qubits;
    if agent.search.temperature_bias <= 0.0 {
        return vec![0.0; cands.len()];
    }
    let rs = regions(n, TEMPERATURE_REGIONS.min(n));
    let field = temperature_field(game, &rs, &ANGLES);
    cands
        .iter()
        .map(|(mv, _)| {
            rs.iter()
                .zip(field.iter())
                .filter(|(r, _)| r.touches_move(mv))
                .map(|(_, t)| *t)
                .fold(0.0f64, f64::max)
        })
        .collect()
}

/// UCT with progressive widening, explicit chance nodes, and Decisions-05's progressive bias.
fn mcts(agent: &Agent, game: &Game, rng: &mut ChaCha8Rng) -> (Candidate, SearchStats) {
    let n = game.players[game.to_move].num_qubits;
    let cands = candidates(agent, n);
    let reference = game.to_move;
    let cap = horizon(game);
    let bias = agent.search.temperature_bias;

    // Decisions-05 3: temperature does two jobs through one computation -- it biases selection
    // among expanded children, and it decides which children get expanded first under widening.
    let h = heuristic(agent, game, &cands);
    let mut order: Vec<usize> = (0..cands.len()).collect();
    // Shuffle before the stable sort so that ties in the heuristic are broken by a seeded
    // permutation and not by the order the move generator happens to emit.
    //
    // This is not a nicety. Measured, the temperature field is uniformly -1 at the opening at
    // every width tried, so every candidate ties -- and a stable sort on a flat heuristic
    // leaves generation order, whose first 208 of 240 entries are all `Apply`. Under widening
    // the root expands about 23 children, so **no measure move would ever be tried at all**:
    // a whole move class made unreachable by the enumeration order of `legal_moves`.
    let len = order.len();
    for i in (1..len).rev() {
        order.swap(i, rng.gen_range(0..=i));
    }
    order.sort_by(|&a, &b| h[b].partial_cmp(&h[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut nodes = vec![Node {
        game: game.clone(),
        parent: None,
        mover: None,
        visits: 0.0,
        value: 0.0,
        kind: Kind::Decision {
            children: Vec::new(),
        },
    }];

    for _ in 0..agent.search.budget {
        let mut at = 0usize;
        loop {
            let visits = nodes[at].visits.max(1.0);
            match &nodes[at].kind {
                Kind::Decision { children } => {
                    let allowed = (WIDEN_K * visits.powf(WIDEN_ALPHA)).ceil() as usize;
                    if children.len() < allowed.min(order.len()) {
                        let ci = order[children.len()];
                        let mover = nodes[at].game.to_move;
                        let child = expand(&mut nodes, at, ci, &cands, mover);
                        at = child;
                        break;
                    }
                    if children.is_empty() {
                        break;
                    }
                    let mut best = (children[0].1, f64::NEG_INFINITY);
                    for &(ci, c) in children {
                        let v = nodes[c].visits.max(1e-12);
                        let score = nodes[c].value / v
                            + EXPLORATION * (visits.ln() / v).sqrt()
                            + bias * h[ci] / (1.0 + nodes[c].visits);
                        if score > best.1 {
                            best = (c, score);
                        }
                    }
                    at = best.0;
                }
                Kind::Chance {
                    qubit,
                    one_weight,
                    outcomes,
                } => {
                    let (qubit, one_weight) = (*qubit, *one_weight);
                    let one = rng.gen_range(0.0f64..1.0) < one_weight;
                    let slot = usize::from(one);
                    match outcomes[slot] {
                        Some(c) => at = c,
                        None => {
                            let mover = nodes[at].mover;
                            let mut child = nodes[at].game.clone();
                            // The draw that selects this branch: below the weight keeps the
                            // one-branch, above it keeps the zero-branch.
                            let draw = if one { 0.0 } else { 1.0 };
                            collapse_at(&mut child.players[child.to_move ^ 1].state, qubit, draw);
                            let idx = nodes.len();
                            nodes.push(Node {
                                game: child,
                                parent: Some(at),
                                mover,
                                visits: 0.0,
                                value: 0.0,
                                kind: Kind::Decision {
                                    children: Vec::new(),
                                },
                            });
                            if let Kind::Chance { outcomes, .. } = &mut nodes[at].kind {
                                outcomes[slot] = Some(idx);
                            }
                            at = idx;
                            break;
                        }
                    }
                }
            }
        }

        let v = rollout(agent, &nodes[at].game, reference, cap, rng);
        let mut cursor = Some(at);
        while let Some(i) = cursor {
            nodes[i].visits += 1.0;
            nodes[i].value += match nodes[i].mover {
                Some(m) if m != reference => 1.0 - v,
                _ => v,
            };
            cursor = nodes[i].parent;
        }
    }

    let stats = SearchStats {
        nodes: nodes.len(),
        root_children: match &nodes[0].kind {
            Kind::Decision { children, .. } => children.len(),
            Kind::Chance { .. } => 0,
        },
        chance_nodes: nodes
            .iter()
            .filter(|n| matches!(n.kind, Kind::Chance { .. }))
            .count(),
        depth: depth_of(&nodes),
    };

    // The most visited child, which is the standard robust choice: a high-value child visited
    // twice is a claim with no evidence behind it.
    let played = match &nodes[0].kind {
        Kind::Decision { children, .. } if !children.is_empty() => {
            let mut best = (children[0].0, -1.0);
            for &(ci, c) in children {
                if nodes[c].visits > best.1 {
                    best = (ci, nodes[c].visits);
                }
            }
            cands[best.0].clone()
        }
        _ => cands[order[0]].clone(),
    };
    (played, stats)
}

/// Longest root-to-leaf path, in nodes. Parents always precede children in the arena, so one
/// forward pass is enough.
fn depth_of(nodes: &[Node]) -> usize {
    let mut d = vec![1usize; nodes.len()];
    let mut worst = 1;
    for i in 1..nodes.len() {
        if let Some(p) = nodes[i].parent {
            d[i] = d[p] + 1;
            worst = worst.max(d[i]);
        }
    }
    worst
}

/// Apply one candidate at `at` and return the new node.
///
/// A generator application is deterministic and gives a decision node. A measurement is not:
/// it gives a **chance node** holding the pre-collapse position and the exact Born weight, and
/// its two outcomes become decision nodes as the search reaches them.
fn expand(nodes: &mut Vec<Node>, at: usize, ci: usize, cands: &[Candidate], mover: usize) -> usize {
    let (mv, angle) = &cands[ci];
    let idx = nodes.len();
    let mut child = nodes[at].game.clone();
    let kind = match mv {
        Move::Measure { qubit } => {
            // A measure move's bookkeeping -- the coherence cost, the turn passing -- is
            // `apply`'s, and only the collapse belongs to the outcome nodes. So the move is
            // applied and then the mover's field is put back uncollapsed; the two outcomes
            // collapse it themselves, each with its own branch.
            let uncollapsed = child.players[child.to_move].state.clone();
            child.apply(mv, *angle);
            child.players[mover].state = uncollapsed;
            Kind::Chance {
                qubit: *qubit,
                one_weight: one_weight(&child.players[mover].state, *qubit),
                outcomes: [None, None],
            }
        }
        Move::Apply { .. } => {
            child.apply(mv, *angle);
            Kind::Decision {
                children: Vec::new(),
            }
        }
    };
    nodes.push(Node {
        game: child,
        parent: Some(at),
        mover: Some(mover),
        visits: 0.0,
        value: 0.0,
        kind,
    });
    if let Kind::Decision { children, .. } = &mut nodes[at].kind {
        children.push((ci, idx));
    }
    idx
}

/// Play out to the coherence horizon, then take the value the declared rollout asks for.
fn rollout(agent: &Agent, from: &Game, reference: usize, cap: usize, rng: &mut ChaCha8Rng) -> f64 {
    let n = from.players[from.to_move].num_qubits;
    let cands = candidates(agent, n);
    let mut g = from.clone();
    let mut steps = 0;
    while steps < cap && !g.players.iter().all(|p| p.coherence == 0) {
        let (mv, angle) = &cands[rng.gen_range(0..cands.len())];
        g.apply_stochastic(mv, *angle, rng);
        steps += 1;
    }
    match agent.search.rollout {
        // Ground truth at the horizon rather than a heuristic playout value. At zero coherence
        // the position is a classical distribution -- rule 5's arrow has run out -- so
        // `Game::outcome` is exact there, which is the property Decisions-01 Q1 named the
        // tablebase for. It is not yet `endgame::Tablebase` itself: that is defined on the
        // maze graph and needs the arena's cell mapping, which is a Phase 10 integration.
        Rollout::Tablebase => match g.outcome(TOLERANCE) {
            Some(Outcome::Checkmate(loser)) => {
                if loser == reference {
                    0.0
                } else {
                    1.0
                }
            }
            _ => agent.eval.uct_value(&g, reference),
        },
        Rollout::Playout => agent.eval.uct_value(&g, reference),
    }
}

/// A whole game between two agents. Returns the winner, or `None` for a draw.
///
/// The tiebreak is the same absorbed-weight comparison [`crate::ladder::play`] uses, on the
/// same physical quantity, so a result here is comparable with a result there.
pub fn play(
    a: &Agent,
    b: &Agent,
    n: usize,
    coherence: usize,
    k: usize,
    seed: u64,
) -> Option<usize> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut game = crate::ladder::opening(n, coherence, k, &mut rng);
    let agents = [a, b];
    let cap = horizon(&game);
    let mut guard = 0;
    while game.outcome(TOLERANCE).is_none() && guard < cap {
        let who = game.to_move;
        let (mv, angle) = choose(agents[who], &game, &mut rng);
        game.apply_stochastic(&mv, angle, &mut rng);
        if game.overlap() > 0.5 {
            let winner = who ^ 1;
            game.merge(winner, winner ^ 1);
        }
        guard += 1;
    }
    match game.outcome(TOLERANCE) {
        Some(Outcome::Checkmate(loser)) => Some(loser ^ 1),
        _ => {
            let w0 = game.position(0).absorbed_weight();
            let w1 = game.position(1).absorbed_weight();
            if (w0 - w1).abs() < 1e-9 {
                None
            } else if w0 < w1 {
                Some(0)
            } else {
                Some(1)
            }
        }
    }
}

/// Win rate of `a` against `b`, colours swapped every other game.
pub fn win_rate(
    a: &Agent,
    b: &Agent,
    n: usize,
    coherence: usize,
    k: usize,
    games: usize,
    seed: u64,
) -> f64 {
    let mut wins = 0.0;
    for g in 0..games {
        let swap = g % 2 == 1;
        let (x, y) = if swap { (b, a) } else { (a, b) };
        match play(
            x,
            y,
            n,
            coherence,
            k,
            seed ^ (g as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        ) {
            Some(w) => {
                let a_seat = if swap { 1 } else { 0 };
                if w == a_seat {
                    wins += 1.0;
                }
            }
            None => wins += 0.5,
        }
    }
    wins / games as f64
}
