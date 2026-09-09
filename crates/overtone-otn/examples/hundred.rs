//! M40: generate `Overtone-100` and verify every answer against an independent route.
//!
//! `cargo run --release -p overtone-otn --example hundred`

use overtone_otn::puzzle::{balance, overtone_100, to_otn, verify, Answer, Kind};

fn main() {
    let set = overtone_100();
    println!("Overtone-100: {} positions\n", set.len());

    let mut counts = [0usize; 3];
    let mut verified = 0usize;
    let mut failures = Vec::new();
    for p in &set {
        counts[match p.kind {
            Kind::Conversion => 0,
            Kind::EscapeInOne => 1,
            Kind::Reachability => 2,
        }] += 1;
        if verify(p) {
            verified += 1;
        } else {
            failures.push(p.id.clone());
        }
    }

    println!("{:>14}  {:>6}", "kind", "count");
    for (k, n) in [Kind::Conversion, Kind::EscapeInOne, Kind::Reachability]
        .iter()
        .zip(counts)
    {
        println!("{:>14}  {n:>6}", k.tag());
    }
    println!();

    let hops: Vec<usize> = set
        .iter()
        .filter_map(|p| match p.answer {
            Answer::Hops(h) => Some(h),
            _ => None,
        })
        .collect();
    let mates: Vec<usize> = set
        .iter()
        .filter_map(|p| match &p.answer {
            Answer::EscapingMoves(m) => Some(m.len()),
            _ => None,
        })
        .collect();
    let (reach, unreach) = balance(&set);
    if !hops.is_empty() {
        println!(
            "conversion hop counts: min {} max {}",
            hops.iter().min().unwrap(),
            hops.iter().max().unwrap()
        );
    }
    if !mates.is_empty() {
        println!(
            "escape-in-1 escaping moves per puzzle: min {} max {}",
            mates.iter().min().unwrap(),
            mates.iter().max().unwrap()
        );
    }
    println!("reachability balance: {reach} reachable, {unreach} not");
    println!();

    let bytes = to_otn(&set).len();
    println!(
        "the whole set serialises to {bytes} bytes ({} per puzzle)",
        bytes / set.len()
    );
    println!();

    if failures.is_empty() {
        println!(
            "VERDICT all-verified: {verified}/{} answers re-derived independently",
            set.len()
        );
    } else {
        println!("VERDICT unverified: {failures:?}");
    }
}
