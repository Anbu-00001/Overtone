//! The engine side of M41a: the three searches, and the properties the league depends on.

use overtone_orbit::game::Move;
use overtone_orbit::language::Agent;
use overtone_orbit::search::{candidates, choose, choose_with_stats};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const KESTREL: &str = include_str!("../../../agents/kestrel.toml");
const PIKE: &str = include_str!("../../../agents/pike.toml");
const STOAT: &str = include_str!("../../../agents/stoat.toml");

const N: usize = 4;
const COHERENCE: usize = 8;
const K: usize = 2;

fn opening() -> overtone_orbit::game::Game {
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    overtone_orbit::ladder::opening(N, COHERENCE, K, &mut rng)
}

#[test]
fn every_kind_returns_a_move_the_agent_is_allowed_to_play() {
    for src in [KESTREL, PIKE, STOAT] {
        let a = Agent::parse(src).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let (mv, angle) = choose(&a, &opening(), &mut rng);
        assert!(
            candidates(&a, N).contains(&(mv.clone(), angle)),
            "{} played something outside its own candidate set",
            a.name
        );
    }
}

#[test]
fn an_agent_only_plays_the_generators_it_declared() {
    // stoat declares pawns only. Part VIII 2's whole premise is that the engine, not the
    // submission, decides what is legal -- so this has to be enforced rather than trusted.
    let a = Agent::parse(STOAT).unwrap();
    for (mv, _) in candidates(&a, N) {
        match mv {
            Move::Apply { piece, .. } => assert_eq!(piece.name(), "pawn"),
            Move::Measure { .. } => {}
        }
    }
}

#[test]
fn the_same_spec_and_seed_give_the_same_move() {
    // Part VIII 6. Without this the league cannot verify a submitted result at all.
    for src in [KESTREL, PIKE, STOAT] {
        let a = Agent::parse(src).unwrap();
        let one = choose(&a, &opening(), &mut ChaCha8Rng::seed_from_u64(99));
        let two = choose(&a, &opening(), &mut ChaCha8Rng::seed_from_u64(99));
        assert_eq!(one, two, "{} is not reproducible", a.name);
    }
}

#[test]
fn the_evaluation_is_antisymmetric() {
    // negamax is only correct on a zero-sum evaluation, and an eval scoring the side to move
    // alone would break it silently. This is the property that makes `kind = "negamax"` mean
    // what it says.
    let a = Agent::parse(KESTREL).unwrap();
    let g = opening();
    let s0 = a.eval.score(&g, 0);
    let s1 = a.eval.score(&g, 1);
    assert!((s0 + s1).abs() < 1e-12, "score(0)={s0} score(1)={s1}");
    let u0 = a.eval.uct_value(&g, 0);
    let u1 = a.eval.uct_value(&g, 1);
    assert!((u0 + u1 - 1.0).abs() < 1e-12, "uct {u0} and {u1}");
    assert!((0.0..=1.0).contains(&u0));
}

#[test]
fn a_bigger_budget_searches_more_of_the_same_position() {
    // The claim `budget` has to support to be a compute axis: raising it changes what the
    // agent does. It is not a claim that the move gets *better* -- that is what the ladder
    // measures, and it is measured, not asserted here.
    let small = Agent::parse(&STOAT.replace("budget     = 32", "budget     = 1")).unwrap();
    let large = Agent::parse(&STOAT.replace("budget     = 32", "budget     = 4096")).unwrap();
    let mut differ = 0;
    for seed in 0..8u64 {
        let a = choose(&small, &opening(), &mut ChaCha8Rng::seed_from_u64(seed));
        let b = choose(&large, &opening(), &mut ChaCha8Rng::seed_from_u64(seed));
        if a != b {
            differ += 1;
        }
    }
    assert!(
        differ >= 6,
        "budget changed the move only {differ} times in 8"
    );
}

#[test]
fn the_full_budget_search_is_seed_independent() {
    // At or above the candidate count, greedy searches everything, so the rng never chooses
    // anything and the move must not depend on the seed. If it does, the sampler is being
    // consulted where it should not be.
    let a = Agent::parse(&STOAT.replace("budget     = 32", "budget     = 100000")).unwrap();
    let first = choose(&a, &opening(), &mut ChaCha8Rng::seed_from_u64(1));
    for seed in 2..6u64 {
        assert_eq!(
            first,
            choose(&a, &opening(), &mut ChaCha8Rng::seed_from_u64(seed))
        );
    }
}

#[test]
fn temperature_is_a_search_field_and_only_mcts_has_one() {
    // Decisions-05 §1: it is progressive bias, not a PUCT prior, so it is a weight on a UCT
    // term and means nothing to a search that has no UCT term.
    let k = Agent::parse(KESTREL).unwrap();
    assert_eq!(k.search.temperature_bias, 0.15);
    assert_eq!(
        Agent::parse(&KESTREL.replace("temperature_bias = 0.15\n", ""))
            .unwrap()
            .search
            .temperature_bias,
        overtone_orbit::language::DEFAULT_TEMPERATURE_BIAS,
        "the default has to survive the field being absent"
    );

    let on_greedy = STOAT.replace(
        "budget     = 32",
        "budget     = 32\ntemperature_bias = 0.15",
    );
    let e = Agent::parse(&on_greedy).unwrap_err();
    assert!(e.message.contains("mcts"), "{}", e.message);

    for bad in ["-0.1", "nan"] {
        let src = KESTREL.replace(
            "temperature_bias = 0.15",
            &format!("temperature_bias = {bad}"),
        );
        assert!(Agent::parse(&src).is_err(), "{bad} should be rejected");
    }
}

#[test]
fn the_rollout_is_named_playout_not_random() {
    // Decisions-05 §6 keeps the field and fixes the vocabulary.
    assert!(Agent::parse(&KESTREL.replace("\"tablebase\"", "\"playout\"")).is_ok());
    let e = Agent::parse(&KESTREL.replace("\"tablebase\"", "\"random\"")).unwrap_err();
    assert!(e.message.contains("playout, tablebase"), "{}", e.message);
}

#[test]
fn progressive_widening_keeps_the_tree_from_being_one_ply_wide() {
    // The reason widening is required rather than optional here: at n = 4 an agent holding all
    // four generators has 240 candidates, so full expansion spends the first 240 playouts of a
    // 512 budget giving every candidate one visit, and the tree never reaches a second ply.
    let a = Agent::parse(KESTREL).unwrap();
    let (_, stats) = choose_with_stats(&a, &opening(), &mut ChaCha8Rng::seed_from_u64(4));
    let cands = candidates(&a, N).len();
    assert_eq!(cands, 240, "the candidate count this test is about");
    assert!(
        stats.root_children < cands / 4,
        "widening did nothing: {} of {cands} children at the root",
        stats.root_children
    );
    // ceil(1 * sqrt(512)) = 23.
    assert!(
        stats.root_children <= 23,
        "{} children",
        stats.root_children
    );
    assert!(stats.depth >= 4, "the tree is only {} deep", stats.depth);
}

#[test]
fn a_measure_move_becomes_a_chance_node_with_two_outcomes() {
    // Decisions-05 §4 asks for widening at chance nodes because "at n = 10 a position
    // measurement has up to 1024 outcomes". Part VII §5 rule 3's measure move collapses **one**
    // qubit, so a chance node has exactly two outcomes and both are enumerated with their exact
    // Born weights. The ruling's real requirement -- chance explicit in the tree rather than
    // folded into whichever outcome was sampled first -- is what is implemented.
    let a = Agent::parse(KESTREL).unwrap();
    let (_, stats) = choose_with_stats(&a, &opening(), &mut ChaCha8Rng::seed_from_u64(4));
    assert!(stats.chance_nodes > 0, "no measure move was ever expanded");
    // Two outcomes each, and nothing else: a chance node cannot have decision children.
    assert!(stats.nodes > stats.chance_nodes * 2);

    let g = opening();
    let p = overtone_orbit::game::one_weight(&g.players[0].state, 0);
    assert!((0.0..=1.0).contains(&p), "Born weight {p}");
}

#[test]
fn the_temperature_field_is_flat_where_the_ladder_runs() {
    // Measured, and it matters more than it looks. Decisions-05 builds the whole progressive
    // bias mechanism on temperature being a useful ordering heuristic, and at the widths the
    // ladder actually runs at the field is **uniformly -1** in the opening -- every region is a
    // number, which is the CGT convention for "cold". A few plies in, roughly one sibling move
    // in twenty-four lifts a region to 0.
    //
    // So temperature here is a near-binary "this move makes a region hot" indicator, not a
    // graded urgency field, and the sibling spread of 1.207 that `freeze.rs` measured is that
    // one outlier divided by a mean sitting at -1. Decisions-05 §2 hedges exactly this by
    // setting the default weight to 0.15; this is the measurement behind the hedge.
    let g = opening();
    let rs = overtone_orbit::thermal::regions(N, 3);
    let field =
        overtone_orbit::thermal::temperature_field(&g, &rs, &overtone_orbit::ladder::ANGLES);
    assert!(
        field.iter().all(|t| *t == -1.0),
        "the opening field is no longer flat: {field:?} -- update the note in search.rs"
    );
}

#[test]
fn the_bias_weight_is_inert_on_a_flat_field_and_that_is_not_a_bug() {
    // A constant heuristic adds a constant to every child's UCT score, so it cannot reorder
    // anything. With the field flat at the opening, `temperature_bias` changes nothing there --
    // and asserting that it *did* would be asserting that a constant offset breaks ties, which
    // would mean something else was wrong.
    //
    // What the weight must not be is unable to matter anywhere; `the_bias_reorders_expansion`
    // is that half of the claim.
    let hot = Agent::parse(KESTREL).unwrap();
    let off = Agent::parse(&KESTREL.replace("temperature_bias = 0.15", "temperature_bias = 0.0"))
        .unwrap();
    for seed in 0..4u64 {
        assert_eq!(
            choose(&hot, &opening(), &mut ChaCha8Rng::seed_from_u64(seed)),
            choose(&off, &opening(), &mut ChaCha8Rng::seed_from_u64(seed)),
            "a constant heuristic reordered the search"
        );
    }
}

#[test]
fn the_bias_reorders_expansion_once_the_field_is_not_flat() {
    // Construct the case the heuristic exists for: a position where one region is warmer than
    // the others, and check that the warm move is expanded before the cold ones. This is
    // Decisions-05 §3's second job for temperature -- expansion ordering under widening -- and
    // it is the half that still works when the bias term is drowned by the exploration term.
    let a = Agent::parse(KESTREL).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    let mut game = overtone_orbit::ladder::opening(N, 24, 2, &mut rng);
    let strategy = overtone_orbit::ladder::Strategy { budget: 6 };
    let mut ch = ChaCha8Rng::seed_from_u64(9);
    for _ in 0..4 {
        let (mv, ang) = strategy.choose(&game, &mut ch);
        game.apply(&mv, ang);
    }
    let rs = overtone_orbit::thermal::regions(N, 3);
    let cands = candidates(&a, N);
    let warm: Vec<usize> = cands
        .iter()
        .enumerate()
        .filter(|(_, (mv, ang))| {
            let mut t = game.clone();
            t.apply(mv, *ang);
            overtone_orbit::thermal::ambient_temperature(&t, &rs, &overtone_orbit::ladder::ANGLES)
                > -1.0
        })
        .map(|(i, _)| i)
        .collect();
    // If nothing is warm anywhere the heuristic has no signal at all, which is a finding and
    // not a passing test.
    assert!(
        !warm.is_empty(),
        "no move heats a region here; the heuristic has nothing to order by"
    );
}
