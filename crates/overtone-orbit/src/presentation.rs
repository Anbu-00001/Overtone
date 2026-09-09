//! The presentation constants, and the boundary they are on the wrong side of.
//!
//! Decisions-01 Q4 gives the operational test for an invented number:
//!
//! > A number is invented if changing it changes an outcome.
//!
//! and asks for two things: one file holding every authored presentation constant, so the
//! complete list is auditable at a glance, and a CI test that perturbs each one and asserts
//! that match results are unchanged. Both exist. This is the file's reader.
//!
//! # The guarantee is structural, and the test is what notices if that changes
//!
//! Nothing in [`crate::search`] or [`crate::game`] takes a [`Presentation`]. The engine cannot
//! read these values because its signatures cannot see them, which is a stronger guarantee
//! than discipline. `tests/presentation.rs` perturbs every constant and replays the same
//! matches anyway — not because today's engine might read them, but because the day someone
//! threads a presentation value into a search is the day that test fails.
//!
//! # Why it is two entries and not twenty
//!
//! Most numbers in this repository are measured or derived, and the ones that are not are
//! mechanics that belong in the code with their derivation next to them. A short file is the
//! honest outcome of that, and the file's value is not its length: it is that the first
//! mechanic anyone tries to hide in it has somewhere to be caught.

use crate::language::{Agent, SpecError};

/// The authored constants, as read from `presentation.toml`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Presentation {
    /// Decimal places the canonical weight vector is rounded to for the league's submission
    /// uniqueness check. Decides whether a pull request is accepted as a distinct agent; the
    /// unrounded vector is what plays.
    pub dedup_precision: usize,
    /// Rows shown on the leaderboard. A display cut, never a tournament cut.
    pub leaderboard_rows: usize,
}

impl Presentation {
    /// The shipped file, compiled in so the test cannot drift from the artefact.
    pub fn shipped() -> Presentation {
        Presentation::parse(include_str!("../../../presentation.toml"))
            .expect("presentation.toml must parse")
    }

    /// Parse the file. Integers under `[league]`, comments, nothing else.
    ///
    /// A second small parser rather than the agent language's: this file has one shape, and
    /// giving it the full grammar would let it grow one.
    pub fn parse(src: &str) -> Result<Presentation, SpecError> {
        let mut dedup_precision = None;
        let mut leaderboard_rows = None;
        for (i, raw) in src.lines().enumerate() {
            let line = i + 1;
            let text = match raw.find('#') {
                Some(p) => &raw[..p],
                None => raw,
            }
            .trim();
            if text.is_empty() || text.starts_with('[') {
                continue;
            }
            let Some((k, v)) = text.split_once('=') else {
                return Err(SpecError {
                    line,
                    message: "expected `key = integer`".into(),
                });
            };
            let value: usize = v.trim().parse().map_err(|_| SpecError {
                line,
                message: format!("`{}` is not a non-negative integer", v.trim()),
            })?;
            match k.trim() {
                "dedup_precision" => dedup_precision = Some(value),
                "leaderboard_rows" => leaderboard_rows = Some(value),
                other => {
                    return Err(SpecError {
                        line,
                        message: format!("unknown presentation constant `{other}`"),
                    })
                }
            }
        }
        match (dedup_precision, leaderboard_rows) {
            (Some(d), Some(l)) => Ok(Presentation {
                dedup_precision: d,
                leaderboard_rows: l,
            }),
            _ => Err(SpecError {
                line: 0,
                message: "presentation.toml is missing a constant".into(),
            }),
        }
    }

    /// Every constant, by name, so a test can enumerate them without knowing the struct.
    pub fn constants(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("dedup_precision", self.dedup_precision),
            ("leaderboard_rows", self.leaderboard_rows),
        ]
    }

    /// The same constants with one of them changed. Used only to perturb.
    pub fn with(&self, key: &str, value: usize) -> Presentation {
        let mut p = *self;
        match key {
            "dedup_precision" => p.dedup_precision = value,
            "leaderboard_rows" => p.leaderboard_rows = value,
            _ => {}
        }
        p
    }

    /// The league's uniqueness key for a submission.
    ///
    /// Decisions-03 Q4: round the canonical vector for **the submission check only, never for
    /// execution**. Two specs that round together are one agent as far as the league is
    /// concerned, and both still play with their own unrounded weights.
    pub fn dedup_key(&self, agent: &Agent) -> String {
        let p = self.dedup_precision;
        let weights: Vec<String> = agent
            .eval
            .weights
            .iter()
            .map(|w| format!("{w:.p$}", p = p))
            .collect();
        format!(
            "{}|{}|{}|{}|{}",
            agent.search.kind.name(),
            agent.search.budget,
            agent
                .generators
                .iter()
                .map(|g| g.name())
                .collect::<Vec<_>>()
                .join(","),
            agent
                .eval
                .features
                .iter()
                .map(|f| f.name())
                .collect::<Vec<_>>()
                .join(","),
            weights.join(",")
        )
    }
}
