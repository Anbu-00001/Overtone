//! The **import format**: permissive, the way PGN's is.
//!
//! A reader accepts more than a writer emits. Blank lines anywhere, any amount of
//! whitespace, tags in any order, unknown tags kept, `;` comments to end of line, `{...}`
//! comments inline, and `%` escape lines ignored entirely. What it will not accept is a
//! missing roster tag or a format version it does not know, because those are the two things
//! that make a record ambiguous rather than merely untidy.

use crate::record::{Ply, Record, Result_, FORMAT_VERSION, REQUIRED, RULESET};
use std::collections::BTreeMap;

/// Why a record would not parse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseError {
    /// A roster tag is missing.
    MissingTag(String),
    /// A tag's value did not parse as the type the roster requires.
    BadValue(String, String),
    /// The file declares a format version this build does not implement.
    UnknownVersion(String),
    /// The file declares a different ruleset.
    WrongRuleset(String),
    /// A move token did not parse.
    BadPly(String),
    /// A line was neither a tag, a comment, nor movetext.
    Malformed(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::MissingTag(t) => write!(f, "missing required tag [{t}]"),
            ParseError::BadValue(t, v) => write!(f, "tag [{t}] has an unusable value {v:?}"),
            ParseError::UnknownVersion(v) => {
                write!(
                    f,
                    "format version {v:?}; this build implements {FORMAT_VERSION}"
                )
            }
            ParseError::WrongRuleset(r) => write!(f, "ruleset {r:?}; this build plays {RULESET}"),
            ParseError::BadPly(p) => write!(f, "unparseable move {p:?}"),
            ParseError::Malformed(l) => write!(f, "malformed line {l:?}"),
        }
    }
}

impl std::error::Error for ParseError {}

/// Strip `{...}` comments, which may not span a token but may span a line.
fn strip_braces(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut depth = 0usize;
    for c in line.chars() {
        match c {
            '{' => depth += 1,
            '}' => depth = depth.saturating_sub(1),
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out
}

/// Parse a `.otn` document.
pub fn parse(text: &str) -> Result<Record, ParseError> {
    let mut tags: BTreeMap<String, String> = BTreeMap::new();
    let mut movetext = String::new();

    for raw in text.lines() {
        let line = raw.trim();
        // PGN's escape convention, kept verbatim: a leading percent is private data.
        if line.is_empty() || line.starts_with('%') || line.starts_with(';') {
            continue;
        }
        let line = match line.split_once(';') {
            Some((before, _)) => before.trim(),
            None => line,
        };
        let line = strip_braces(line);
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix('[') {
            let rest = rest
                .strip_suffix(']')
                .ok_or_else(|| ParseError::Malformed(line.into()))?;
            let (name, value) = rest
                .split_once(' ')
                .ok_or_else(|| ParseError::Malformed(line.into()))?;
            let value = value
                .trim()
                .strip_prefix('"')
                .and_then(|v| v.strip_suffix('"'))
                .ok_or_else(|| ParseError::Malformed(line.into()))?;
            tags.insert(name.trim().to_string(), value.to_string());
        } else {
            movetext.push(' ');
            movetext.push_str(line);
        }
    }

    for t in REQUIRED {
        if !tags.contains_key(t) {
            return Err(ParseError::MissingTag(t.to_string()));
        }
    }
    let version = &tags["Otn"];
    if version.parse::<u32>() != Ok(FORMAT_VERSION) {
        return Err(ParseError::UnknownVersion(version.clone()));
    }
    if tags["Ruleset"] != RULESET {
        return Err(ParseError::WrongRuleset(tags["Ruleset"].clone()));
    }

    let num = |t: &str| -> Result<u64, ParseError> {
        tags[t]
            .parse::<u64>()
            .map_err(|_| ParseError::BadValue(t.into(), tags[t].clone()))
    };
    let qubits = num("Qubits")? as usize;
    let coherence = num("Coherence")? as usize;
    let k = num("K")? as usize;
    let seed = num("Seed")?;
    let engine = tags["Engine"].clone();

    let result = tags
        .get("Result")
        .map(|r| {
            Result_::from_token(r).ok_or_else(|| ParseError::BadValue("Result".into(), r.clone()))
        })
        .transpose()?
        .unwrap_or(Result_::Unfinished);

    let hash = tags
        .get("Hash")
        .map(|h| {
            u64::from_str_radix(h, 16).map_err(|_| ParseError::BadValue("Hash".into(), h.clone()))
        })
        .transpose()?;

    let mut plies = Vec::new();
    for token in movetext.split_whitespace() {
        // Move numbers are decoration: they are written for a reader and re-derived on
        // output, so a parser that trusted them could be desynchronised by a typo.
        if token.ends_with('.') && token[..token.len() - 1].chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        if token == "*" {
            continue;
        }
        plies.push(Ply::from_token(token).ok_or_else(|| ParseError::BadPly(token.to_string()))?);
    }

    let mut extra = tags;
    for t in REQUIRED {
        extra.remove(t);
    }
    extra.remove("Result");
    extra.remove("Hash");

    Ok(Record {
        qubits,
        coherence,
        k,
        seed,
        engine,
        plies,
        result,
        hash,
        extra,
    })
}
