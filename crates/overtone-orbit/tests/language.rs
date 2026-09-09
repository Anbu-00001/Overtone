//! M41a: the language is frozen, so every one of these is a claim about what v1 *is*.

use overtone_orbit::language::{Agent, Feature, Rollout, SearchKind, VERSION};

const KESTREL: &str = include_str!("../../../agents/kestrel.toml");
const PIKE: &str = include_str!("../../../agents/pike.toml");
const STOAT: &str = include_str!("../../../agents/stoat.toml");

#[test]
fn the_three_shipped_specs_parse() {
    let k = Agent::parse(KESTREL).expect("kestrel");
    assert_eq!(k.name, "kestrel");
    assert_eq!(k.search.kind, SearchKind::Mcts);
    assert_eq!(k.search.budget, 512);
    assert_eq!(k.search.rollout, Rollout::Tablebase);
    assert_eq!(k.eval.features.len(), 4);
    assert_eq!(k.generators.len(), 4);

    let p = Agent::parse(PIKE).expect("pike");
    assert_eq!(p.search.kind, SearchKind::Negamax);
    assert_eq!(p.eval.features, vec![Feature::SafeSetSize, Feature::DimG]);

    let s = Agent::parse(STOAT).expect("stoat");
    assert_eq!(s.search.kind, SearchKind::Greedy);
    assert_eq!(s.generators.len(), 1);
}

#[test]
fn weights_are_l2_normalised_at_parse() {
    for src in [KESTREL, PIKE, STOAT] {
        let a = Agent::parse(src).unwrap();
        let n: f64 = a.eval.weights.iter().map(|w| w * w).sum::<f64>().sqrt();
        assert!((n - 1.0).abs() < 1e-12, "{} has |w| = {n}", a.name);
    }
}

#[test]
fn scale_is_canonicalised_away_but_only_after_the_eval_is_bounded() {
    // Decisions-03 Q4, both halves. Under greedy argmax `w` and `7w` are already the same
    // agent; under UCT they are not, because scaling the eval shifts the exploration balance.
    // So the eval is mapped into [0, 1] at point of use *first*, and only then is the
    // canonicalisation sound. What this test can check is the second half: the two specs are
    // the same agent, with the same name, byte for byte.
    let scaled = STOAT.replace("weights    = [1.0]", "weights    = [7.0]");
    assert_eq!(Agent::parse(STOAT).unwrap(), Agent::parse(&scaled).unwrap());
    assert_eq!(
        Agent::parse(STOAT).unwrap().digest(),
        Agent::parse(&scaled).unwrap().digest()
    );

    // And a sign flip is *not* canonicalised away: it is a different agent.
    let flipped = STOAT.replace("weights    = [1.0]", "weights    = [-1.0]");
    assert_ne!(
        Agent::parse(STOAT).unwrap().digest(),
        Agent::parse(&flipped).unwrap().digest()
    );
}

#[test]
fn the_round_trip_is_the_identity() {
    for src in [KESTREL, PIKE, STOAT] {
        let a = Agent::parse(src).unwrap();
        let b = Agent::parse(&a.to_toml()).expect("canonical form must parse");
        assert_eq!(a, b, "{} did not survive a round trip", a.name);
        assert_eq!(a.digest(), b.digest());
    }
}

#[test]
fn the_version_is_checked_before_anything_else() {
    // A v2 spec may legitimately use a vocabulary this engine has never heard of, so reporting
    // "unknown feature" first would be a misleading error. The version is the first thing.
    let src = "language = \"v2\"\n[agent]\nname = \"x\"\ngenerators = [\"pawn\"]\n\
               [agent.search]\nkind = \"swarm\"\nbudget = 1\n\
               [agent.eval]\nfeatures = [\"vibes\"]\nweights = [1.0]\n";
    let e = Agent::parse(src).unwrap_err();
    assert!(e.message.contains("v2"), "{}", e.message);
    assert!(e.message.contains(VERSION), "{}", e.message);
}

#[test]
fn the_rejections_name_the_vocabulary() {
    let bad_feature = STOAT.replace("\"safe_set_size\"", "\"reach_margin\"");
    let e = Agent::parse(&bad_feature).unwrap_err();
    // reach_margin is the one Decisions-01 invented and Decisions-03 withdrew. A spec asking
    // for it should be told what does exist.
    assert!(e.message.contains("reach_margin"), "{}", e.message);
    for f in Feature::ALL {
        assert!(e.message.contains(f.name()), "{}", e.message);
    }

    let bad_kind = STOAT.replace("\"greedy\"", "\"expectimax\"");
    let e = Agent::parse(&bad_kind).unwrap_err();
    assert!(e.message.contains("greedy, negamax, mcts"), "{}", e.message);
}

#[test]
fn a_field_that_could_not_do_anything_is_rejected() {
    // `rollout` only means something under mcts. Accepting it silently on a greedy agent would
    // put a declared parameter in the language that cannot change an outcome, which is exactly
    // what Part VII 0's invented-number test forbids.
    let src = STOAT.replace(
        "budget     = 32",
        "budget     = 32\nrollout    = \"tablebase\"",
    );
    let e = Agent::parse(&src).unwrap_err();
    assert!(e.message.contains("mcts"), "{}", e.message);
    assert!(e.line > 0, "the error should name the line");
}

#[test]
fn an_agent_with_no_evaluation_is_rejected() {
    let zeros = PIKE.replace("weights    = [1.0, 0.25]", "weights    = [0.0, 0.0]");
    assert!(Agent::parse(&zeros).is_err());
    let short = PIKE.replace("weights    = [1.0, 0.25]", "weights    = [1.0]");
    let e = Agent::parse(&short).unwrap_err();
    assert!(e.message.contains("positional"), "{}", e.message);
    let dup = STOAT
        .replace(
            "features   = [\"safe_set_size\"]",
            "features   = [\"safe_set_size\", \"safe_set_size\"]",
        )
        .replace("weights    = [1.0]", "weights    = [1.0, 1.0]");
    assert!(Agent::parse(&dup).is_err());
}

#[test]
fn the_budget_is_required_because_it_is_the_compute_axis() {
    let src = STOAT.replace("budget     = 32\n", "");
    let e = Agent::parse(&src).unwrap_err();
    assert!(e.message.contains("compute axis"), "{}", e.message);
    let zero = STOAT.replace("budget     = 32", "budget     = 0");
    assert!(Agent::parse(&zero).is_err());
}

#[test]
fn every_feature_normalises_into_the_unit_interval() {
    for f in Feature::ALL {
        for n in 1..=8 {
            assert!(f.ceiling(n) > 0.0, "{} has no ceiling at n={n}", f.name());
            for raw in [0.0, 1.0, 3.0, 1e6] {
                let v = f.normalise(raw, n);
                assert!(
                    (0.0..=1.0).contains(&v),
                    "{} {raw} at n={n} -> {v}",
                    f.name()
                );
            }
        }
    }
}

#[test]
fn the_language_digest_pins_what_the_features_compute() {
    // Freezing the vocabulary is not freezing the language. Every name here could stay the
    // same, every spec could keep parsing, and v1 could still become a different language --
    // change `orbit_dimension`'s tolerance, or the cut the entropy is taken across, and every
    // `d` measured before the change stops being comparable with every `d` measured after.
    //
    // The literal is computed, not chosen. If this test fails, either the freeze was broken or
    // it was broken deliberately, and the second case is a v2.
    assert_eq!(
        overtone_orbit::language::language_digest(),
        0x883f_6657_a605_7861,
        "the frozen feature computations changed"
    );
}

#[test]
fn the_reference_position_is_the_same_every_time() {
    let a =
        overtone_orbit::language::raw_features(&overtone_orbit::language::reference_position(), 0);
    let b =
        overtone_orbit::language::raw_features(&overtone_orbit::language::reference_position(), 0);
    assert_eq!(a, b);
}
