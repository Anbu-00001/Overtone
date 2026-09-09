//! M40's acceptance criterion, from Part VIII 10: "every solution verified against brute
//! force on small instances."

use overtone_otn::puzzle::{balance, overtone_100, to_otn, verify, Answer, Kind};
use overtone_otn::{parse, SET_SIZE};

#[test]
fn every_answer_is_independently_re_derived() {
    let set = overtone_100();
    assert_eq!(set.len(), SET_SIZE);
    for p in &set {
        assert!(verify(p), "{} ({}) did not verify", p.id, p.kind.tag());
    }
}

#[test]
fn the_set_is_reproducible_from_nothing() {
    // No data files: the set is a function of its generators, so two runs agree exactly.
    let a = overtone_100();
    let b = overtone_100();
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.id, y.id);
        assert_eq!(x.answer_tag(), y.answer_tag());
    }
}

#[test]
fn ids_are_unique_and_ordered() {
    let set = overtone_100();
    for (i, p) in set.iter().enumerate() {
        assert_eq!(p.id, format!("otn-100-{:03}", i + 1));
    }
}

#[test]
fn no_category_can_be_guessed() {
    // A benchmark whose answers are lopsided measures nothing. The first version of the
    // reachability generator came out 27 reachable to 3, which a solver could score 90% on
    // by answering the same thing every time; excluding the start cell from the safe set put
    // every position in check and let the algebra decide instead.
    let set = overtone_100();
    let (yes, no) = balance(&set);
    assert!(yes > 0 && no > 0);
    let skew = (yes as f64 - no as f64).abs() / (yes + no) as f64;
    assert!(skew < 0.34, "reachability is {yes}/{no}, skew {skew:.2}");

    // The other two categories are not yes/no questions, but they must still vary.
    let hops: Vec<usize> = set
        .iter()
        .filter_map(|p| match p.answer {
            Answer::Hops(h) => Some(h),
            _ => None,
        })
        .collect();
    assert!(
        hops.iter().max() > hops.iter().min(),
        "every conversion has the same answer"
    );
}

#[test]
fn every_puzzle_serialises_and_parses_back() {
    for p in overtone_100() {
        let text = p.to_record().to_otn();
        let back = parse(&text).unwrap_or_else(|e| panic!("{}: {e}", p.id));
        assert_eq!(back, p.to_record(), "{}", p.id);
        assert_eq!(back.extra.get("Id"), Some(&p.id));
        assert_eq!(
            back.extra.get("Kind").map(String::as_str),
            Some(p.kind.tag())
        );
    }
}

#[test]
fn the_whole_set_is_a_small_text_file() {
    // Part VIII 1's size claim, applied to the benchmark rather than to one game.
    let set = overtone_100();
    let bytes = to_otn(&set).len();
    assert!(bytes < 64_000, "the set serialised to {bytes} bytes");
    assert!(
        bytes / set.len() < 400,
        "{} bytes per puzzle",
        bytes / set.len()
    );
}

#[test]
fn escape_puzzles_really_start_in_check() {
    // If a puzzle were already safe, "escape in one" would be vacuous. `verify` checks this,
    // so the test is that the category is non-empty and every member passes it.
    let set = overtone_100();
    let escapes: Vec<_> = set.iter().filter(|p| p.kind == Kind::EscapeInOne).collect();
    assert!(escapes.len() >= 20);
    for p in escapes {
        assert!(verify(p), "{}", p.id);
        match &p.answer {
            Answer::EscapingMoves(m) => assert!(!m.is_empty()),
            other => panic!("wrong answer kind: {other:?}"),
        }
    }
}
