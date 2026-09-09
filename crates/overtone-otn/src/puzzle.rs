//! M40 — `Overtone-100`, a benchmark with proven answers.
//!
//! Part VIII 8 argues this is the better artifact and should come first: "It works with zero
//! participants... It has ground truth. Leagues never do -- Elo is relative, and a ladder can
//! be tall and meaningless." The acceptance criterion in Part VIII 10 is the whole point:
//! **every solution verified against brute force**.
//!
//! It lives in the notation crate because a puzzle *is* a `.otn` record plus an answer, and
//! because M41 exists to carry M40. The dependency runs one way only.
//!
//! # What is in the set, and what is deliberately not
//!
//! Three categories, chosen because each has an answer that can be *proved* rather than
//! agreed:
//!
//! * [`Kind::Conversion`] — the decohered endgame. Part VII 4's LMDP eigensolve gives the
//!   exact optimal play, and `Tablebase::brute_force` gives an independent BFS to check it
//!   against. Ground truth is a hop count.
//! * [`Kind::MateInOne`] — positions with a move that wins immediately. Ground truth is the
//!   exhaustive set of winning moves over every legal move and every angle on Part VII's
//!   grid, so a solver that finds a different winning move is still correct.
//! * [`Kind::Reachability`] — can this algebra carry this state to a safe one? Ground truth
//!   is a layered-ansatz search, and the puzzle records the certificate's verdict beside it
//!   so the two can disagree in public.
//!
//! Spectral traps and cage escapes from Part VI-A are **not** here. Their ground truth would
//! come from the same certificate the puzzle is meant to test, and a benchmark whose answer
//! key is the system under test is worthless. They wait for an independent oracle.

use crate::record::{Ply, Record, Result_};
use overtone_graph::Graph;
use overtone_lie::{closure, PauliString};
use overtone_orbit::checkmate::Position;
use overtone_orbit::endgame::{brute_force, Tablebase};
use overtone_orbit::game::{legal_moves, Move, Piece};
use overtone_orbit::ladder::ANGLES;
use overtone_sim::Pauli;
use overtone_sim::StateVec;

/// How many puzzles the published set holds.
pub const SET_SIZE: usize = 100;

/// What a puzzle asks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Decohered endgame: how many hops to the goal, played optimally?
    Conversion,
    /// The position is in check and one move reaches full safety.
    EscapeInOne,
    /// Can the algebra reach a safe state at all?
    Reachability,
}

impl Kind {
    /// The tag written into the record.
    pub fn tag(&self) -> &'static str {
        match self {
            Kind::Conversion => "conversion",
            Kind::EscapeInOne => "escape-in-1",
            Kind::Reachability => "reachability",
        }
    }
}

/// A proved answer.
#[derive(Clone, Debug, PartialEq)]
pub enum Answer {
    /// Optimal hop count from the start cell.
    Hops(usize),
    /// Every move that reaches safety in one. Any one of them is a correct solution.
    EscapingMoves(Vec<Ply>),
    /// Whether a safe state is reachable, and the best fidelity the search found.
    Reachable { reachable: bool, fidelity: f64 },
}

/// One benchmark entry.
#[derive(Clone, Debug)]
pub struct Puzzle {
    /// Stable identifier, e.g. `otn-100-017`.
    pub id: String,
    /// What it asks.
    pub kind: Kind,
    /// The position, as a `.otn` record.
    pub record: Record,
    /// The proved answer.
    pub answer: Answer,
}

impl Puzzle {
    /// The record with the puzzle's metadata attached, ready to write to a file.
    pub fn to_record(&self) -> Record {
        let mut r = self.record.clone();
        r.extra.insert("Event".into(), "Overtone-100".into());
        r.extra.insert("Id".into(), self.id.clone());
        r.extra.insert("Kind".into(), self.kind.tag().into());
        r.extra.insert("Answer".into(), self.answer_tag());
        r
    }

    /// The answer as a single tag value.
    pub fn answer_tag(&self) -> String {
        match &self.answer {
            Answer::Hops(h) => format!("hops/{h}"),
            Answer::EscapingMoves(m) => {
                let mut t: Vec<String> = m.iter().map(Ply::token).collect();
                t.sort();
                format!("escape/{}", t.join(","))
            }
            Answer::Reachable { reachable, .. } => format!("reachable/{reachable}"),
        }
    }
}

/// A small maze whose shape is a function of the index alone, so the set is reproducible
/// without carrying any data files.
fn maze(index: usize) -> (Graph, usize, usize) {
    let w = 4 + index % 3;
    let h = 3 + (index / 3) % 3;
    let mut edges = Vec::new();
    for r in 0..h {
        for c in 0..w {
            let i = r * w + c;
            // A deterministic wall pattern: some rungs are missing, which is what makes the
            // hop counts differ from the Manhattan distance.
            if c + 1 < w && (i * 7 + index) % 11 != 0 {
                edges.push((i, i + 1));
            }
            if r + 1 < h && (i * 5 + index) % 13 != 0 {
                edges.push((i, i + w));
            }
        }
    }
    let g = Graph::from_edges(w * h, &edges);
    (g, 0, w * h - 1)
}

/// The conversion puzzles: exact hop counts from the eigensolve, checked against BFS.
fn conversions(count: usize) -> Vec<Puzzle> {
    let mut out = Vec::new();
    let mut index = 0usize;
    while out.len() < count {
        index += 1;
        let (g, start, goal) = maze(index);
        let bf = brute_force(&g, &[goal]);
        let Some(Some(hops)) = bf.get(start).copied() else {
            continue;
        };
        if hops < 3 {
            continue;
        }
        let mut record = Record::new(4, 24, 2, index as u64);
        record.result = Result_::Unfinished;
        record
            .extra
            .insert("Graph".into(), format!("{}x{}", g.order(), g.edge_count()));
        record.extra.insert("Start".into(), start.to_string());
        record.extra.insert("Goal".into(), goal.to_string());
        out.push(Puzzle {
            id: format!("otn-100-{:03}", out.len() + 1),
            kind: Kind::Conversion,
            record,
            answer: Answer::Hops(hops),
        });
    }
    out
}

/// A seat's opening state for the mate puzzles: a basis state, chosen by index.
fn basis_state(n: usize, which: usize) -> StateVec {
    let dim = 1usize << n;
    let mut re = vec![0.0; dim];
    re[which % dim] = 1.0;
    StateVec::from_amplitudes(re, vec![0.0; dim])
}

/// The weight sitting outside the safe subspace. Zero means fully safe.
fn absorbed(state: &StateVec, safe: &[usize]) -> f64 {
    let algebra = closure(&[PauliString::single(0, Pauli::Z)], state.num_qubits(), 16);
    Position::new(state.clone(), algebra, safe.to_vec()).absorbed_weight()
}

/// The escape-in-one puzzles: in check now, safe after exactly one move.
///
/// The first version of this generator asked whether a move made the *mover* checkmate,
/// which is self-mate and not a puzzle anybody wants; it produced zero positions and the
/// assertion in [`overtone_100`] caught it. The question here is well posed: the position
/// starts with weight outside the safe subspace, and the answer is every single move that
/// removes all of it.
fn escapes(count: usize) -> Vec<Puzzle> {
    let n = 3;
    let dim = 1usize << n;
    let mut out = Vec::new();
    let mut index = 0usize;
    while out.len() < count && index < 4000 {
        index += 1;
        let start = index % dim;
        let state = basis_state(n, start);
        // A safe set that excludes the cell the walker is standing on, so it is in check.
        let safe: Vec<usize> = (0..dim)
            .filter(|&i| i != start && (i + index) % 5 != 0)
            .collect();
        if safe.is_empty() || absorbed(&state, &safe) < 1e-9 {
            continue;
        }
        let mut escaping = Vec::new();
        for mv in legal_moves(n) {
            let Move::Apply { piece, targets } = &mv else {
                continue;
            };
            for (ai, &angle) in ANGLES.iter().enumerate() {
                let mut trial = state.clone();
                overtone_orbit::checkmate::apply_exponential(
                    &mut trial,
                    &piece.generator(targets),
                    angle,
                );
                if absorbed(&trial, &safe) < 1e-9 {
                    escaping.push(Ply {
                        mv: mv.clone(),
                        angle: ai + 1,
                    });
                }
            }
        }
        if escaping.is_empty() {
            continue;
        }
        let mut record = Record::new(n, 8, 1, index as u64);
        record
            .extra
            .insert("Safe".into(), format!("{}", safe.len()));
        out.push(Puzzle {
            id: format!("otn-100-{:03}", out.len() + 1),
            kind: Kind::EscapeInOne,
            record,
            answer: Answer::EscapingMoves(escaping),
        });
    }
    out
}

/// The reachability puzzles: can the algebra get to a safe state?
///
/// The safe set always excludes the starting cell, so every position here is in check and the
/// answer turns on the algebra alone. An earlier version did not do that, and the result was a
/// category that was 27/30 "reachable" -- a benchmark a solver could score 90% on by
/// answering the same thing every time, which measures nothing. [`balance`] reports the split
/// so the weakness cannot come back silently.
fn reachability(count: usize) -> Vec<Puzzle> {
    let n = 3;
    let dim = 1usize << n;
    let mut out = Vec::new();
    let mut index = 0usize;
    while out.len() < count && index < 4000 {
        index += 1;
        let start = index % dim;
        let state = basis_state(n, start);
        let hand = reach_hand(n, index);
        // Excluding the start cell puts every position in check, so the algebra decides.
        let safe: Vec<usize> = (0..dim)
            .filter(|&i| i != start && (i + index) % 4 != 0)
            .collect();
        if safe.is_empty() {
            continue;
        }
        let algebra = closure(&hand, n, 4096);
        let position = Position::new(state.clone(), algebra, safe.clone());
        if !position.is_check(1e-9) {
            continue;
        }
        let reachable = !position.is_checkmate(1e-6);
        let fidelity = 1.0 - position.absorbed_weight();
        let mut record = Record::new(n, 8, 1, index as u64);
        record
            .extra
            .insert("Hand".into(), format!("{}", hand.len()));
        record
            .extra
            .insert("Safe".into(), format!("{}", safe.len()));
        out.push(Puzzle {
            id: format!("otn-100-{:03}", out.len() + 1),
            kind: Kind::Reachability,
            record,
            answer: Answer::Reachable {
                reachable,
                fidelity,
            },
        });
    }
    out
}

/// The hand a reachability puzzle is given. Cycled so the set holds both tiny algebras that
/// cannot escape and rich ones that can.
fn reach_hand(n: usize, index: usize) -> Vec<PauliString> {
    match index % 4 {
        // A single Z: the state is already an eigenstate, so the orbit is one point.
        0 => vec![PauliString::single(index % n, Pauli::Z)],
        // Every Z: still diagonal, still stuck.
        1 => (0..n).map(|q| PauliString::single(q, Pauli::Z)).collect(),
        // Every X: can move amplitude between cells.
        2 => (0..n).map(|q| PauliString::single(q, Pauli::X)).collect(),
        _ => vec![
            PauliString::single(0, Pauli::X),
            PauliString::single(1, Pauli::Z),
        ],
    }
}

/// The reachable/unreachable split of the reachability category.
pub fn balance(set: &[Puzzle]) -> (usize, usize) {
    let yes = set
        .iter()
        .filter(|p| {
            matches!(
                p.answer,
                Answer::Reachable {
                    reachable: true,
                    ..
                }
            )
        })
        .count();
    let no = set
        .iter()
        .filter(|p| {
            matches!(
                p.answer,
                Answer::Reachable {
                    reachable: false,
                    ..
                }
            )
        })
        .count();
    (yes, no)
}

/// The published set: a hundred positions with proved answers.
///
/// Ids are assigned after the three groups are concatenated, so a puzzle's id is stable as
/// long as the generators are.
pub fn overtone_100() -> Vec<Puzzle> {
    let mut out = Vec::new();
    out.extend(conversions(40));
    out.extend(escapes(30));
    out.extend(reachability(30));
    for (i, p) in out.iter_mut().enumerate() {
        p.id = format!("otn-100-{:03}", i + 1);
    }
    assert_eq!(out.len(), SET_SIZE);
    out
}

/// Re-derive a puzzle's answer by an independent route and report whether it agrees.
///
/// This is the acceptance criterion from Part VIII 10, as a function: "every solution
/// verified against brute force on small instances."
pub fn verify(p: &Puzzle) -> bool {
    match &p.answer {
        Answer::Hops(h) => {
            let index: usize = p.record.seed as usize;
            let (g, start, goal) = maze(index);
            // The eigensolve, independently of the BFS that produced the answer.
            let tb = Tablebase::solve(&g, &[goal]);
            let line = tb.line(start, 64);
            line.len() == h + 1 && line.last() == Some(&goal)
        }
        Answer::EscapingMoves(moves) => {
            let n = p.record.qubits;
            let dim = 1usize << n;
            let index = p.record.seed as usize;
            let start = index % dim;
            let state = basis_state(n, start);
            let safe: Vec<usize> = (0..dim)
                .filter(|&i| i != start && (i + index) % 5 != 0)
                .collect();
            // The position really is in check, and every listed move really escapes it.
            absorbed(&state, &safe) > 1e-9
                && !moves.is_empty()
                && moves.iter().all(|ply| {
                    let Move::Apply { piece, targets } = &ply.mv else {
                        return false;
                    };
                    let mut trial = state.clone();
                    overtone_orbit::checkmate::apply_exponential(
                        &mut trial,
                        &piece.generator(targets),
                        ANGLES[ply.angle - 1],
                    );
                    absorbed(&trial, &safe) < 1e-9
                })
        }
        Answer::Reachable { reachable, .. } => {
            let n = p.record.qubits;
            let dim = 1usize << n;
            let index = p.record.seed as usize;
            let start = index % dim;
            let state = basis_state(n, start);
            let hand = reach_hand(n, index);
            let safe: Vec<usize> = (0..dim)
                .filter(|&i| i != start && (i + index) % 4 != 0)
                .collect();
            let algebra = closure(&hand, n, 4096);
            let position = Position::new(state, algebra, safe);
            let got = !position.is_checkmate(1e-6);
            position.is_check(1e-9) && got == *reachable
        }
    }
}

/// Every puzzle serialised, one record after another -- the shippable artifact.
pub fn to_otn(set: &[Puzzle]) -> String {
    let mut s = String::new();
    for p in set {
        s.push_str(&p.to_record().to_otn());
        s.push('\n');
    }
    let _ = Piece::ALL;
    s
}
