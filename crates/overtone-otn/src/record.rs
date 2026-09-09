//! The record itself: a header, a move list, and a result.

use overtone_orbit::game::{Move, Piece};
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// The format version, written into every file as the first tag and checked on parse.
///
/// Part VIII 11: freeze early, version explicitly. Bumping this is a deliberate act that
/// requires a compatibility entry in `tests/roundtrip.rs`, not a side effect of a refactor.
pub const FORMAT_VERSION: u32 = 1;

/// The ruleset the moves are interpreted under: Part VII's seven rules.
pub const RULESET: &str = "vii.7";

/// The mandatory header roster -- `.otn`'s answer to PGN's Seven Tag Roster.
///
/// Every conforming file carries these, in this order, before any other tag. Anything else a
/// producer wants to record is welcome and is preserved on round-trip, but these are the
/// common ground.
pub const REQUIRED: [&str; 7] = [
    "Otn",
    "Engine",
    "Ruleset",
    "Qubits",
    "Coherence",
    "K",
    "Seed",
];

/// How a game ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Result_ {
    /// Rule 7: the named seat has no unitary in `g` reaching a safe state.
    Checkmate(usize),
    /// Both ran out of coherence with neither proved lost.
    Exhausted,
    /// The record is a position, not a finished game.
    Unfinished,
}

impl Result_ {
    /// The token written into the `Result` tag.
    pub fn token(&self) -> String {
        match self {
            Result_::Checkmate(loser) => format!("checkmate/{loser}"),
            Result_::Exhausted => "exhausted".to_string(),
            Result_::Unfinished => "*".to_string(),
        }
    }

    /// Parse a `Result` tag value.
    pub fn from_token(s: &str) -> Option<Result_> {
        match s {
            "*" => Some(Result_::Unfinished),
            "exhausted" => Some(Result_::Exhausted),
            _ => s
                .strip_prefix("checkmate/")
                .and_then(|n| n.parse().ok())
                .map(Result_::Checkmate),
        }
    }
}

/// One ply: a move and the angle it was applied at.
///
/// Angles are recorded as an index into Part VII's grid of eighths of `pi`, never as a
/// float. A float in the notation would make two records of the same game differ in their
/// last digit and would put the format at the mercy of `f64` formatting; an index cannot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ply {
    /// The move.
    pub mv: Move,
    /// `1..=8`, meaning `index * pi / 8`. Ignored for a measurement.
    pub angle: usize,
}

fn piece_letter(p: Piece) -> char {
    match p {
        Piece::Pawn => 'P',
        Piece::Rook => 'R',
        Piece::Bishop => 'B',
        // N, as in chess: K would collide with nothing here but the convention is free.
        Piece::Knight => 'N',
    }
}

fn letter_piece(c: char) -> Option<Piece> {
    match c {
        'P' => Some(Piece::Pawn),
        'R' => Some(Piece::Rook),
        'B' => Some(Piece::Bishop),
        'N' => Some(Piece::Knight),
        _ => None,
    }
}

impl Ply {
    /// The token for this ply: `P0/4`, `B0-1/2`, `M3`.
    pub fn token(&self) -> String {
        match &self.mv {
            Move::Measure { qubit } => format!("M{qubit}"),
            Move::Apply { piece, targets } => {
                let t: Vec<String> = targets.iter().map(|q| q.to_string()).collect();
                format!("{}{}/{}", piece_letter(*piece), t.join("-"), self.angle)
            }
        }
    }

    /// Parse one ply token.
    pub fn from_token(s: &str) -> Option<Ply> {
        if let Some(q) = s.strip_prefix('M') {
            return q.parse().ok().map(|qubit| Ply {
                mv: Move::Measure { qubit },
                angle: 0,
            });
        }
        let mut chars = s.chars();
        let piece = letter_piece(chars.next()?)?;
        let rest: String = chars.collect();
        let (targets, angle) = rest.split_once('/')?;
        let angle: usize = angle.parse().ok()?;
        if !(1..=8).contains(&angle) {
            return None;
        }
        let targets: Option<Vec<usize>> = targets.split('-').map(|t| t.parse().ok()).collect();
        let targets = targets?;
        if targets.len() != piece.arity() {
            return None;
        }
        Some(Ply {
            mv: Move::Apply { piece, targets },
            angle,
        })
    }

    /// The angle in radians.
    pub fn radians(&self) -> f64 {
        self.angle as f64 * std::f64::consts::FRAC_PI_8
    }
}

/// A complete `.otn` record.
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    /// Number of qubits in the window.
    pub qubits: usize,
    /// Starting coherence per seat.
    pub coherence: usize,
    /// Rule 4's coherent-step count.
    pub k: usize,
    /// The opening seed. With the ruleset and the move list, this *is* the position.
    pub seed: u64,
    /// The engine that produced the record.
    pub engine: String,
    /// The plies, in order.
    pub plies: Vec<Ply>,
    /// How it ended.
    pub result: Result_,
    /// The state hash, if the producer computed one.
    pub hash: Option<u64>,
    /// Any non-roster tags, preserved in sorted order the way PGN's export format does it.
    pub extra: BTreeMap<String, String>,
}

impl Record {
    /// A new unfinished record for a given opening.
    pub fn new(qubits: usize, coherence: usize, k: usize, seed: u64) -> Record {
        Record {
            qubits,
            coherence,
            k,
            seed,
            engine: format!("overtone {}", env!("CARGO_PKG_VERSION")),
            plies: Vec::new(),
            result: Result_::Unfinished,
            hash: None,
            extra: BTreeMap::new(),
        }
    }

    /// Serialise in **export format**: canonical, byte-exact, and stable across producers.
    ///
    /// Roster tags first in roster order, then `Result`, then `Hash` when present, then any
    /// extra tags in ASCII order -- the same layering PGN's export format uses. Moves are
    /// numbered in pairs, six per line, so a long game stays readable in a terminal.
    pub fn to_otn(&self) -> String {
        let mut s = String::new();
        let values = [
            FORMAT_VERSION.to_string(),
            self.engine.clone(),
            RULESET.to_string(),
            self.qubits.to_string(),
            self.coherence.to_string(),
            self.k.to_string(),
            self.seed.to_string(),
        ];
        for (tag, value) in REQUIRED.iter().zip(values.iter()) {
            let _ = writeln!(s, "[{tag} \"{value}\"]");
        }
        let _ = writeln!(s, "[Result \"{}\"]", self.result.token());
        if let Some(h) = self.hash {
            let _ = writeln!(s, "[Hash \"{h:016x}\"]");
        }
        for (k, v) in &self.extra {
            let _ = writeln!(s, "[{k} \"{v}\"]");
        }
        s.push('\n');

        let mut line = String::new();
        for (i, pair) in self.plies.chunks(2).enumerate() {
            let mut item = format!("{}. {}", i + 1, pair[0].token());
            if let Some(second) = pair.get(1) {
                let _ = write!(item, " {}", second.token());
            }
            if !line.is_empty() && line.len() + item.len() + 1 > 78 {
                s.push_str(line.trim_end());
                s.push('\n');
                line.clear();
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(&item);
        }
        if !line.is_empty() {
            s.push_str(line.trim_end());
            s.push('\n');
        }
        s
    }
}
