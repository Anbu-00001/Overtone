//! M41a: the agent language, frozen at v1.
//!
//! Decisions-01 Q1 is the ruling that reordered Phase 10, and it is worth restating because
//! everything here follows from it:
//!
//! > `d` is not a property of the game. It is a property of **(game, strategy language)**.
//!
//! So the language is part of the experimental apparatus and has to be frozen *before* `d` is
//! measured, not after. And a declarative feature-weight policy cannot be that language,
//! because it evaluates in `O(1)` per move and therefore **has no compute axis at all** — a
//! ladder over it would have no rungs. The resolution is to declare the *search*, not the
//! policy: the submitter names a search kind and a budget from a published vocabulary, and the
//! engine implements the search. No submitted code is ever executed.
//!
//! ```toml
//! language   = "v1"
//!
//! [agent]
//! name       = "kestrel"
//! generators = ["pawn", "bishop", "knight", "rook"]
//!
//! [agent.search]
//! kind       = "mcts"      # "greedy" | "negamax" | "mcts"
//! budget     = 4096        # the compute axis: the rung index of the ladder
//! rollout    = "tablebase" # "random" | "tablebase"
//!
//! [agent.eval]
//! features   = ["dim_g", "orbit_size", "safe_set_size", "half_chain_entropy"]
//! weights    = [0.31, -0.12, 0.44, 0.08]
//! ```
//!
//! # Three ways this differs from the ruling's illustrative TOML, and why
//!
//! Decisions-01's example is explicitly placeholder — it admits inventing `reach_margin` — so
//! the differences are deliberate and each one is a thing that was measured or checked.
//!
//! **The feature vocabulary is four, not six.** `overtone-orbit/examples/freeze.rs` measured
//! all eight candidates for sibling spread, leaf spread and cost. `coherence` and
//! `average_branching` are *constant across siblings* — `coherence` is a pure function of
//! depth because both arms of `Game::apply` subtract `k`, and `average_branching` is
//! structurally `legal_moves(n).len()`. `separation_deficit` is dominated by `dim_g` at 14x
//! the cost. `temperature` is the best discriminator measured and is **cost-disqualified**: at
//! 714 us per call it manages 1400 evaluations a second against the 4096 a budget-4096 move
//! needs. What survives is `dim_g`, `orbit_size`, `safe_set_size`, `half_chain_entropy`.
//!
//! **`reach_margin` is not here**, because it corresponds to nothing implemented. Decisions-03
//! Q1 withdraws it; its intent is covered by `orbit_size` and `safe_set_size` together.
//!
//! **The generator names are the four the engine has.** The ruling's `"bishop-Z2"`,
//! `"knight-hop"` and `"rook-phase"` do not exist; Part VII's pieces are `pawn`, `rook`,
//! `bishop`, `knight`.
//!
//! **`statistics` is not a field.** The ruling's example declares one, but `Game::merge` does
//! not yet implement exchange statistics, and a declared parameter the engine ignores is
//! exactly what Part VII 0's invented-number test forbids: it cannot change an outcome. It
//! goes in when rule 6 consumes it, which is a v2 change.
//!
//! # The eval is a difference, not a score for one side
//!
//! `Eval::score` is `sum_i w_i (f_i(me) - f_i(them))` over normalised features. Antisymmetry is
//! not a stylistic choice: negamax is only correct on a zero-sum evaluation, and an eval that
//! scored the side to move alone would make `kind = "negamax"` quietly unsound. Nobody would
//! notice, and the resulting `d` would be measured on an agent family that does not do what
//! its spec says.
//!
//! # Q4's two-step canonicalisation, in order
//!
//! Decisions-03 Q4: under greedy argmax `w` and `2w` are the same agent, but UCT compares the
//! exploitation term against `c sqrt(ln N / n)`, so **scaling the weights changes an MCTS
//! agent's behaviour**. L2-normalising alone would therefore be a behaviour-changing transform
//! disguised as canonicalisation. So, in this order:
//!
//! 1. [`Eval::uct_value`] maps the score into `[0, 1]` at point of use, by dividing by the
//!    weights' L1 norm — which is the exact bound, since normalised features lie in `[0, 1]`.
//! 2. [`Agent::parse`] then L2-normalises the weights as pure canonicalisation, rejecting
//!    all-zero and non-finite vectors.
//!
//! Without the first, the second is unsound.

use overtone_lie::Algebra;
use overtone_spec::digest::fnv1a;
use overtone_spec::entropy::half_chain_entropy;

use crate::game::{Game, Piece};
use crate::invariant::orbit_dimension;

/// The frozen version of this language. A spec declaring anything else is rejected.
pub const VERSION: &str = "v1";

/// The language, declared in prose, as Lantz et al. require of any `d` that gets quoted.
///
/// Correction 2 in [`crate::ladder`]: "any observations made about a game's depth based on
/// this model must refer to the language selected". This is that reference for v1, and it
/// supersedes [`crate::ladder::LANGUAGE`] for everything measured after M41a.
pub const LANGUAGE_V1: &str = "\
An agent is a search kind from {greedy, negamax, mcts}, a budget in position evaluations, and \
a weighted linear evaluation over the four frozen features {dim_g, orbit_size, safe_set_size, \
half_chain_entropy}, each normalised to [0, 1] against its structural maximum at the window \
width, scored as the difference between the side to move and its opponent. The move set is \
Part VII 2's legal moves restricted to the declared generator types, with the rotation angle \
drawn from a fixed 8-point grid on [-pi, pi). Budget is the computational resource and the \
ladder runs over powers of two. Measurement collapse is Born-random under mcts and \
deterministic-to-the-likelier-branch under greedy and negamax, which is the modelling error \
those two kinds are declaring.";

/// The tolerance the closure and orbit computations use. One number, used everywhere.
const TOLERANCE: f64 = 1e-9;

/// The four features that survived measurement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Feature {
    /// `dim(g)`, the Lie closure of the hand. Material.
    DimG,
    /// The dimension of the orbit through the current state. Reachability.
    OrbitSize,
    /// How many basis cells are still safe for this player. Proximity to loss.
    SafeSetSize,
    /// Von Neumann entropy across the half-chain cut, in nats. Structure.
    HalfChainEntropy,
}

impl Feature {
    /// Every feature in the frozen vocabulary, in canonical order.
    pub const ALL: [Feature; 4] = [
        Feature::DimG,
        Feature::OrbitSize,
        Feature::SafeSetSize,
        Feature::HalfChainEntropy,
    ];

    /// The name a spec writes.
    pub fn name(&self) -> &'static str {
        match self {
            Feature::DimG => "dim_g",
            Feature::OrbitSize => "orbit_size",
            Feature::SafeSetSize => "safe_set_size",
            Feature::HalfChainEntropy => "half_chain_entropy",
        }
    }

    /// Parse a name from a spec.
    pub fn parse(s: &str) -> Option<Feature> {
        Feature::ALL.into_iter().find(|f| f.name() == s)
    }

    /// The structural maximum at `n` qubits — what the raw value is divided by.
    ///
    /// Every one of these is a property of the state space rather than a tuned scale:
    /// `dim(su(2^n)) = 4^n - 1`, the orbit is a real submanifold of a `2^(n+1)`-dimensional
    /// real space, there are `n` cells that can be safe, and the half-chain entropy of `n/2`
    /// qubits is at most `(n/2) ln 2`. The two exponential ones are compressed by `ln(1 + x)`
    /// first, because a linear eval over a quantity spanning `4^n` is an eval over one feature.
    pub fn ceiling(&self, n: usize) -> f64 {
        let nf = n as f64;
        match self {
            Feature::DimG => (4.0f64.powf(nf) - 1.0).ln_1p(),
            Feature::OrbitSize => (2.0f64.powf(nf + 1.0)).ln_1p(),
            Feature::SafeSetSize => (1usize << n) as f64,
            Feature::HalfChainEntropy => (nf / 2.0) * std::f64::consts::LN_2,
        }
    }

    /// Map a raw value into `[0, 1]`.
    pub fn normalise(&self, raw: f64, n: usize) -> f64 {
        let c = self.ceiling(n);
        if c <= 0.0 {
            return 0.0;
        }
        let x = match self {
            Feature::DimG | Feature::OrbitSize => raw.max(0.0).ln_1p(),
            _ => raw,
        };
        (x / c).clamp(0.0, 1.0)
    }
}

/// Raw values of every feature for one player, computed with the algebra shared between them.
///
/// `dim_g` and `orbit_size` both need the Lie closure, which is the expensive part; computing
/// them from separate calls would pay for it twice and is the obvious way to make a
/// 5.27-microsecond evaluation cost 9.
pub fn raw_features(game: &Game, player: usize) -> [f64; 4] {
    let p = &game.players[player];
    let algebra: Algebra = p.algebra();
    [
        algebra.dim() as f64,
        orbit_dimension(&algebra, &p.state, TOLERANCE) as f64,
        game.safe_for(player).len() as f64,
        half_chain_entropy(&p.state),
    ]
}

/// Feature values are quantised onto this grid before being hashed.
///
/// The same `1e-9` `.otn` uses, for the same reason: `scripts/wasm_determinism.sh` measures
/// native-versus-wasm agreement at about `5.6e-16`, so a digest over raw bits would differ
/// between a native run and a browser run of the same position.
pub const FEATURE_QUANTUM: f64 = 1e-9;

/// The position every frozen-language claim is made on.
///
/// Fixed opening, fixed moves, deterministic collapse — so the digest below is a statement
/// about the *features*, not about a sampler.
pub fn reference_position() -> Game {
    use crate::game::{Move, Piece};
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(41);
    let mut game = crate::ladder::opening(4, 8, 2, &mut rng);
    let script = [
        (
            Move::Apply {
                piece: Piece::Pawn,
                targets: vec![0],
            },
            std::f64::consts::FRAC_PI_4,
        ),
        (
            Move::Apply {
                piece: Piece::Bishop,
                targets: vec![1, 2],
            },
            std::f64::consts::FRAC_PI_8,
        ),
        (Move::Measure { qubit: 3 }, 0.0),
    ];
    for (mv, angle) in script {
        game.apply(&mv, angle);
    }
    game
}

/// A digest of what the frozen features *compute*, not of what they are called.
///
/// Freezing the vocabulary is not the same as freezing the language. Changing
/// `orbit_dimension`'s tolerance, or the cut `half_chain_entropy` takes, would leave every
/// spec parsing, every name intact, and every test green — while silently making v1 a
/// different language and every `d` measured under it incomparable with every other. This is
/// the thing that notices.
pub fn language_digest() -> u64 {
    let game = reference_position();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(VERSION.as_bytes());
    for player in 0..2 {
        for v in raw_features(&game, player) {
            bytes.extend_from_slice(&((v / FEATURE_QUANTUM).round() as i64).to_le_bytes());
        }
    }
    for f in Feature::ALL {
        bytes.extend_from_slice(f.name().as_bytes());
        bytes.extend_from_slice(&((f.ceiling(4) / FEATURE_QUANTUM).round() as i64).to_le_bytes());
    }
    fnv1a(&bytes)
}

/// Which search the engine runs on the agent's behalf.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchKind {
    /// Evaluate every candidate one ply deep and play the best. Budget caps the candidates.
    Greedy,
    /// Alpha-beta over the same evaluation. Budget is the node allowance, not the depth.
    Negamax,
    /// UCT with the tablebase or a random playout at the leaves. Budget is the playout count.
    Mcts,
}

impl SearchKind {
    pub fn name(&self) -> &'static str {
        match self {
            SearchKind::Greedy => "greedy",
            SearchKind::Negamax => "negamax",
            SearchKind::Mcts => "mcts",
        }
    }

    pub fn parse(s: &str) -> Option<SearchKind> {
        match s {
            "greedy" => Some(SearchKind::Greedy),
            "negamax" => Some(SearchKind::Negamax),
            "mcts" => Some(SearchKind::Mcts),
            _ => None,
        }
    }
}

/// What terminates an MCTS playout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rollout {
    /// Play uniformly at random to the coherence horizon, then evaluate.
    Playout,
    /// Stop at the first decohered position and take M36's exact value.
    Tablebase,
}

impl Rollout {
    pub fn name(&self) -> &'static str {
        match self {
            Rollout::Playout => "playout",
            Rollout::Tablebase => "tablebase",
        }
    }

    pub fn parse(s: &str) -> Option<Rollout> {
        match s {
            "playout" => Some(Rollout::Playout),
            "tablebase" => Some(Rollout::Tablebase),
            _ => None,
        }
    }
}

/// Decisions-05 §2's default progressive-bias weight.
///
/// `0.15`, the midpoint of the one published tuned value that surfaced: Gedda et al. (2018)
/// tuned a progressive-bias weight in Kingdomino to `W ~ 0.1-0.2`. Decisions-05 flags that
/// figure as reported by a research pass and not yet verified at source, which is one more
/// reason it is a *default* on an agent field rather than an engine constant: the league
/// explores the range and the value that wins is a measurement.
///
/// Decisions-04's example used `0.7`. Decisions-05 revises it down on the grounds that an
/// incorrect bias concentrates search on suboptimal moves and takes many iterations to
/// correct (James, Konidaris & Rosman 2017), and that nobody yet knows whether Overtone's
/// temperature is CGT temperature proper or a temperature-shaped heuristic.
pub const DEFAULT_TEMPERATURE_BIAS: f64 = 0.15;

/// The search block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Search {
    pub kind: SearchKind,
    /// The compute axis. One rung of the ladder is one value of this.
    pub budget: usize,
    pub rollout: Rollout,
    /// `W` in the progressive-bias term `W * H(s,a) / (1 + n(s,a))`. Zero disables it.
    ///
    /// **Not a PUCT prior.** Decisions-05 §1 changes the mechanism, and the reason is that
    /// temperature is an unbounded positive scalar rather than a distribution. Forcing it into
    /// PUCT means softmaxing it, which imposes a distribution shape on something that is not
    /// one, introduces a scaling parameter with no published guidance, and is less forgiving
    /// of a mediocre heuristic. Additive progressive bias needs none of that and hands control
    /// to the empirical mean as visits accumulate.
    ///
    /// It also removes a name collision that would have been a real hazard: PUCT would have
    /// put a softmax temperature `tau` three lines away from CGT temperature `H` in the same
    /// function.
    pub temperature_bias: f64,
}

/// The evaluation block: features and their weights, canonicalised.
#[derive(Clone, Debug, PartialEq)]
pub struct Eval {
    pub features: Vec<Feature>,
    /// L2-normalised at parse. Q4 step 2.
    pub weights: Vec<f64>,
}

impl Eval {
    /// Raw feature values, in the order the spec declared them.
    pub fn raw(&self, game: &Game, player: usize) -> Vec<f64> {
        let all = raw_features(game, player);
        self.features
            .iter()
            .map(|f| match f {
                Feature::DimG => all[0],
                Feature::OrbitSize => all[1],
                Feature::SafeSetSize => all[2],
                Feature::HalfChainEntropy => all[3],
            })
            .collect()
    }

    /// The antisymmetric score for `player`: their features minus their opponent's.
    pub fn score(&self, game: &Game, player: usize) -> f64 {
        let n = game.players[player].num_qubits;
        let mine = self.raw(game, player);
        let theirs = self.raw(game, player ^ 1);
        self.features
            .iter()
            .enumerate()
            .map(|(i, f)| self.weights[i] * (f.normalise(mine[i], n) - f.normalise(theirs[i], n)))
            .sum()
    }

    /// Sum of absolute weights — the exact bound on `score`, since features are in `[0, 1]`.
    pub fn l1(&self) -> f64 {
        self.weights.iter().map(|w| w.abs()).sum()
    }

    /// Q4 step 1: the score mapped into UCT's `[0, 1]` at point of use.
    ///
    /// This is what makes weight scale behaviourally irrelevant, and it has to happen before
    /// L2-normalisation is honest.
    pub fn uct_value(&self, game: &Game, player: usize) -> f64 {
        let l1 = self.l1();
        if l1 <= 0.0 {
            return 0.5;
        }
        (0.5 + 0.5 * self.score(game, player) / l1).clamp(0.0, 1.0)
    }
}

/// A submitted agent.
#[derive(Clone, Debug, PartialEq)]
pub struct Agent {
    pub name: String,
    /// Which piece types the agent may play. A subset of `Piece::ALL`, in canonical order.
    pub generators: Vec<Piece>,
    pub search: Search,
    pub eval: Eval,
}

/// What a spec can be rejected for, with the line it was rejected on.
#[derive(Clone, Debug, PartialEq)]
pub struct SpecError {
    pub line: usize,
    pub message: String,
}

impl std::fmt::Display for SpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.line == 0 {
            write!(f, "{}", self.message)
        } else {
            write!(f, "line {}: {}", self.line, self.message)
        }
    }
}

impl std::error::Error for SpecError {}

fn err<T>(line: usize, message: impl Into<String>) -> Result<T, SpecError> {
    Err(SpecError {
        line,
        message: message.into(),
    })
}

/// One parsed `key = value` line, before it is given a meaning.
enum Value {
    Str(String),
    Int(i64),
    Float(f64),
    Strings(Vec<String>),
    Floats(Vec<f64>),
}

impl Agent {
    /// Parse a spec. The accepted grammar is a deliberately small subset of TOML.
    ///
    /// Tables, `key = value`, `#` comments, strings, integers, floats, and arrays of one type.
    /// No inline tables, no dotted keys, no multi-line strings. A submission format that
    /// accepts less is a submission format with fewer ways to be surprising, and the parser is
    /// hand-written for the same reason `.otn`'s is: the format is part of the artefact, and a
    /// dependency's idea of TOML is not something this repository should be able to change.
    pub fn parse(src: &str) -> Result<Agent, SpecError> {
        let mut table = String::new();
        let mut language: Option<String> = None;
        let mut name: Option<String> = None;
        let mut generators: Option<Vec<String>> = None;
        let mut kind: Option<String> = None;
        let mut budget: Option<i64> = None;
        let mut rollout: Option<String> = None;
        let mut bias: Option<f64> = None;
        let mut bias_line = 0usize;
        let mut features: Option<Vec<String>> = None;
        let mut weights: Option<Vec<f64>> = None;
        let mut kind_line = 0usize;
        let mut rollout_line = 0usize;

        for (i, raw) in src.lines().enumerate() {
            let line = i + 1;
            let text = match raw.find('#') {
                Some(p) => &raw[..p],
                None => raw,
            }
            .trim();
            if text.is_empty() {
                continue;
            }
            if let Some(rest) = text.strip_prefix('[') {
                let Some(head) = rest.strip_suffix(']') else {
                    return err(line, "unterminated table header");
                };
                table = head.trim().to_string();
                match table.as_str() {
                    "agent" | "agent.search" | "agent.eval" => {}
                    other => return err(line, format!("unknown table [{other}]")),
                }
                continue;
            }
            let Some((key, rhs)) = text.split_once('=') else {
                return err(line, "expected `key = value`");
            };
            let key = key.trim();
            let value = parse_value(rhs.trim(), line)?;
            match (table.as_str(), key) {
                ("", "language") => language = Some(as_str(value, line, key)?),
                ("agent", "name") => name = Some(as_str(value, line, key)?),
                ("agent", "generators") => generators = Some(as_strings(value, line, key)?),
                ("agent.search", "kind") => {
                    kind_line = line;
                    kind = Some(as_str(value, line, key)?);
                }
                ("agent.search", "budget") => budget = Some(as_int(value, line, key)?),
                ("agent.search", "rollout") => {
                    rollout_line = line;
                    rollout = Some(as_str(value, line, key)?);
                }
                ("agent.search", "temperature_bias") => {
                    bias_line = line;
                    bias = Some(as_one_float(value, line, key)?);
                }
                ("agent.eval", "features") => features = Some(as_strings(value, line, key)?),
                ("agent.eval", "weights") => weights = Some(as_floats(value, line, key)?),
                ("", k) => return err(line, format!("`{k}` outside any table")),
                (t, k) => return err(line, format!("unknown key `{k}` in [{t}]")),
            }
        }

        // Version first: every message after this one assumes v1's vocabulary.
        let language = language.ok_or(SpecError {
            line: 0,
            message: "no `language` declared; this engine speaks v1".into(),
        })?;
        if language != VERSION {
            return err(
                0,
                format!("language `{language}` is not `{VERSION}`, the only version this engine implements"),
            );
        }

        let name = name.unwrap_or_default();
        if name.is_empty() || name.len() > 32 || !name.bytes().all(|b| b.is_ascii_graphic()) {
            return err(0, "`agent.name` must be 1 to 32 printable ASCII characters");
        }

        let generators = {
            let names = generators.unwrap_or_default();
            if names.is_empty() {
                return err(0, "`agent.generators` must name at least one piece");
            }
            let mut out = Vec::new();
            for g in &names {
                let Some(p) = Piece::ALL.into_iter().find(|p| p.name() == g) else {
                    return err(
                        0,
                        format!(
                            "unknown generator `{g}`; v1 has {}",
                            Piece::ALL.map(|p| p.name()).join(", ")
                        ),
                    );
                };
                if out.contains(&p) {
                    return err(0, format!("generator `{g}` declared twice"));
                }
                out.push(p);
            }
            out.sort_by_key(|p| Piece::ALL.iter().position(|q| q == p).unwrap_or(0));
            out
        };

        let kind = match kind {
            None => return err(0, "`agent.search.kind` is required"),
            Some(k) => match SearchKind::parse(&k) {
                Some(v) => v,
                None => {
                    return err(
                        kind_line,
                        format!("unknown search kind `{k}`; v1 has greedy, negamax, mcts"),
                    )
                }
            },
        };
        let budget = match budget {
            None => {
                return err(
                    0,
                    "`agent.search.budget` is required; it is the compute axis",
                )
            }
            Some(b) if b < 1 => return err(0, "`agent.search.budget` must be at least 1"),
            Some(b) => b as usize,
        };
        let rollout = match rollout {
            None => Rollout::Tablebase,
            Some(r) => {
                if kind != SearchKind::Mcts {
                    return err(
                        rollout_line,
                        format!(
                            "`rollout` is only meaningful for kind = \"mcts\", not `{}`",
                            kind.name()
                        ),
                    );
                }
                match Rollout::parse(&r) {
                    Some(v) => v,
                    None => {
                        return err(
                            rollout_line,
                            format!("unknown rollout `{r}`; v1 has playout, tablebase"),
                        )
                    }
                }
            }
        };

        let temperature_bias = match bias {
            None => {
                if kind == SearchKind::Mcts {
                    DEFAULT_TEMPERATURE_BIAS
                } else {
                    0.0
                }
            }
            Some(w) => {
                if kind != SearchKind::Mcts {
                    return err(
                        bias_line,
                        format!(
                            "`temperature_bias` is a UCT term and only means something for kind = \"mcts\", not `{}`",
                            kind.name()
                        ),
                    );
                }
                if !w.is_finite() || w < 0.0 {
                    return err(
                        bias_line,
                        "`temperature_bias` must be finite and not negative",
                    );
                }
                w
            }
        };

        let feature_names = features.unwrap_or_default();
        if feature_names.is_empty() {
            return err(0, "`agent.eval.features` must name at least one feature");
        }
        let mut feats = Vec::new();
        for f in &feature_names {
            let Some(p) = Feature::parse(f) else {
                return err(
                    0,
                    format!(
                        "unknown feature `{f}`; v1 froze {}",
                        Feature::ALL.map(|x| x.name()).join(", ")
                    ),
                );
            };
            if feats.contains(&p) {
                return err(0, format!("feature `{f}` declared twice"));
            }
            feats.push(p);
        }
        let mut ws = weights.unwrap_or_default();
        if ws.len() != feats.len() {
            return err(
                0,
                format!(
                    "{} weights for {} features; they are positional",
                    ws.len(),
                    feats.len()
                ),
            );
        }
        if ws.iter().any(|w| !w.is_finite()) {
            return err(0, "weights must all be finite");
        }
        let norm = ws.iter().map(|w| w * w).sum::<f64>().sqrt();
        if norm <= 0.0 {
            return err(
                0,
                "weights must not be all zero: that agent has no evaluation",
            );
        }
        // Canonicalisation has to be idempotent or the canonical form is not canonical: an
        // already-unit vector divided by a norm that is 1.0 to within an ulp comes back with
        // different last bits, and `parse(to_toml(a)) == a` then fails on a spec that was
        // already canonical. Which would make `digest` a name for a file, not for an agent.
        if (norm - 1.0).abs() > 8.0 * f64::EPSILON {
            for w in &mut ws {
                *w /= norm;
            }
        }

        Ok(Agent {
            name,
            generators,
            search: Search {
                kind,
                budget,
                rollout,
                temperature_bias,
            },
            eval: Eval {
                features: feats,
                weights: ws,
            },
        })
    }

    /// The canonical spec: what this agent is, written back out.
    ///
    /// Round-tripping through this is the identity on the parsed form, weights included, which
    /// is what makes [`Agent::digest`] a stable name for an agent rather than for a file.
    pub fn to_toml(&self) -> String {
        let mut s = format!(
            "language   = \"{VERSION}\"\n\n[agent]\nname       = \"{}\"\n",
            self.name
        );
        s.push_str(&format!(
            "generators = [{}]\n\n[agent.search]\nkind       = \"{}\"\nbudget     = {}\n",
            self.generators
                .iter()
                .map(|p| format!("\"{}\"", p.name()))
                .collect::<Vec<_>>()
                .join(", "),
            self.search.kind.name(),
            self.search.budget
        ));
        if self.search.kind == SearchKind::Mcts {
            s.push_str(&format!(
                "rollout    = \"{}\"\ntemperature_bias = {}\n",
                self.search.rollout.name(),
                self.search.temperature_bias
            ));
        }
        s.push_str(&format!(
            "\n[agent.eval]\nfeatures   = [{}]\nweights    = [{}]\n",
            self.eval
                .features
                .iter()
                .map(|f| format!("\"{}\"", f.name()))
                .collect::<Vec<_>>()
                .join(", "),
            self.eval
                .weights
                .iter()
                .map(|w| format!("{w}"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        s
    }

    /// A stable 64-bit name for this agent, over the canonical form.
    ///
    /// The league needs a uniqueness check on submissions, and Decisions-03 Q4 rules that the
    /// rounding it uses is a *presentation* constant: it decides whether a pull request is
    /// accepted and can never change a match result. It lives in `presentation.toml`.
    pub fn digest(&self) -> u64 {
        fnv1a(self.to_toml().as_bytes())
    }

    /// The legal moves this agent is allowed to play, in a fixed order.
    pub fn moves(&self, n: usize) -> Vec<crate::game::Move> {
        crate::game::legal_moves(n)
            .into_iter()
            .filter(|m| match m {
                crate::game::Move::Apply { piece, .. } => self.generators.contains(piece),
                crate::game::Move::Measure { .. } => true,
            })
            .collect()
    }
}

fn parse_value(s: &str, line: usize) -> Result<Value, SpecError> {
    if let Some(rest) = s.strip_prefix('[') {
        let Some(inner) = rest.strip_suffix(']') else {
            return err(line, "unterminated array");
        };
        let items: Vec<&str> = if inner.trim().is_empty() {
            Vec::new()
        } else {
            inner.split(',').map(|x| x.trim()).collect()
        };
        if items.iter().all(|x| x.starts_with('"')) {
            let mut out = Vec::new();
            for it in items {
                out.push(unquote(it, line)?);
            }
            return Ok(Value::Strings(out));
        }
        let mut out = Vec::new();
        for it in items {
            match it.parse::<f64>() {
                Ok(v) => out.push(v),
                Err(_) => return err(line, format!("`{it}` is not a number")),
            }
        }
        return Ok(Value::Floats(out));
    }
    if s.starts_with('"') {
        return Ok(Value::Str(unquote(s, line)?));
    }
    if let Ok(v) = s.parse::<i64>() {
        return Ok(Value::Int(v));
    }
    match s.parse::<f64>() {
        Ok(v) => Ok(Value::Float(v)),
        Err(_) => err(line, format!("`{s}` is not a string, number or array")),
    }
}

fn unquote(s: &str, line: usize) -> Result<String, SpecError> {
    let t = s.trim();
    if t.len() >= 2 && t.starts_with('"') && t.ends_with('"') {
        Ok(t[1..t.len() - 1].to_string())
    } else {
        err(line, format!("{t} is not a quoted string"))
    }
}

fn as_str(v: Value, line: usize, key: &str) -> Result<String, SpecError> {
    match v {
        Value::Str(s) => Ok(s),
        _ => err(line, format!("`{key}` must be a string")),
    }
}

fn as_int(v: Value, line: usize, key: &str) -> Result<i64, SpecError> {
    match v {
        Value::Int(i) => Ok(i),
        _ => err(line, format!("`{key}` must be an integer")),
    }
}

fn as_strings(v: Value, line: usize, key: &str) -> Result<Vec<String>, SpecError> {
    match v {
        Value::Strings(s) => Ok(s),
        Value::Floats(f) if f.is_empty() => Ok(Vec::new()),
        _ => err(line, format!("`{key}` must be an array of strings")),
    }
}

fn as_floats(v: Value, line: usize, key: &str) -> Result<Vec<f64>, SpecError> {
    match v {
        Value::Floats(f) => Ok(f),
        Value::Int(i) => Ok(vec![i as f64]),
        Value::Float(f) => Ok(vec![f]),
        Value::Strings(s) if s.is_empty() => Ok(Vec::new()),
        _ => err(line, format!("`{key}` must be an array of numbers")),
    }
}

fn as_one_float(v: Value, line: usize, key: &str) -> Result<f64, SpecError> {
    match v {
        Value::Float(f) => Ok(f),
        Value::Int(i) => Ok(i as f64),
        _ => err(line, format!("`{key}` must be a number")),
    }
}
