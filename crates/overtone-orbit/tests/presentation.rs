//! Decisions-01 Q4's perturbation test: the presentation constants cannot reach a result.

use overtone_orbit::language::Agent;
use overtone_orbit::presentation::Presentation;
use overtone_orbit::search::{choose, win_rate};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const STOAT: &str = include_str!("../../../agents/stoat.toml");
const PIKE: &str = include_str!("../../../agents/pike.toml");

#[test]
fn the_shipped_file_parses_and_every_constant_is_accounted_for() {
    let p = Presentation::shipped();
    assert_eq!(
        p.constants().len(),
        2,
        "a constant was added without a test"
    );
    for (name, value) in p.constants() {
        assert!(value > 0, "{name} is {value}");
    }
    // An unknown key is a rejection, not a silent ignore: a constant nobody reads is a
    // constant nobody has checked the side of the boundary of.
    assert!(Presentation::parse("[league]\nvibes = 3\n").is_err());
}

#[test]
fn perturbing_every_constant_changes_no_result() {
    // The perturbation Q4 asks for. Today it passes structurally -- the engine's signatures
    // cannot see a `Presentation` -- and the day someone threads one into a search is the day
    // this fails.
    let a = Agent::parse(STOAT).unwrap();
    let b = Agent::parse(PIKE).unwrap();
    let baseline = Presentation::shipped();

    let reference: Vec<_> = (0..4u64)
        .map(|s| {
            let mut rng = ChaCha8Rng::seed_from_u64(s);
            let game = overtone_orbit::ladder::opening(4, 8, 2, &mut rng);
            choose(&a, &game, &mut rng)
        })
        .collect();
    let reference_rate = win_rate(&a, &b, 4, 8, 2, 4, 5);

    for (name, value) in baseline.constants() {
        for perturbed in [value + 1, value.saturating_sub(1).max(1), value * 3] {
            let p = baseline.with(name, perturbed);
            assert_ne!(p, baseline, "{name} did not actually change");
            let moves: Vec<_> = (0..4u64)
                .map(|s| {
                    let mut rng = ChaCha8Rng::seed_from_u64(s);
                    let game = overtone_orbit::ladder::opening(4, 8, 2, &mut rng);
                    choose(&a, &game, &mut rng)
                })
                .collect();
            assert_eq!(moves, reference, "{name} = {perturbed} changed a move");
            assert_eq!(
                win_rate(&a, &b, 4, 8, 2, 4, 5),
                reference_rate,
                "{name} = {perturbed} changed a match result"
            );
        }
    }
}

#[test]
fn the_dedup_key_rounds_but_the_agent_that_plays_does_not() {
    // Q4: round for the submission check only. Two specs whose weights differ below the
    // rounding are one submission and two distinct agents.
    let p = Presentation::shipped();
    let one = Agent::parse(PIKE).unwrap();
    let two = Agent::parse(&PIKE.replace("[1.0, 0.25]", "[1.0, 0.2500000001]")).unwrap();
    assert_eq!(p.dedup_key(&one), p.dedup_key(&two));
    assert_ne!(one.eval.weights, two.eval.weights);

    let far = Agent::parse(&PIKE.replace("[1.0, 0.25]", "[1.0, 0.30]")).unwrap();
    assert_ne!(p.dedup_key(&one), p.dedup_key(&far));
}
