//! M50: the advantage dial, and the point where it turns into disadvantage.

use overtone_orbit::advantage::{dnd_advantage, dnd_disadvantage, Advantage};

#[test]
fn simulation_matches_the_closed_form() {
    // Part IX 1.3 quotes sin^2((2k+1) theta). The simulator is not told the formula.
    for qubits in 3..=9 {
        let a = Advantage::new(qubits, vec![5]);
        for k in 0..=a.optimal_iterations() + 3 {
            let got = a.measured(k);
            let want = a.closed_form(k);
            assert!(
                (got - want).abs() < 1e-9,
                "{qubits} qubits, k = {k}: simulated {got}, closed form {want}"
            );
        }
    }
}

#[test]
fn the_dial_reaches_near_certainty_and_dnd_advantage_cannot() {
    // Grover at its optimum: essentially 1. D&D advantage from the same base: 1 - (1-p)^2,
    // which for a long shot is barely better than the long shot.
    let a = Advantage::new(10, vec![512]);
    let p0 = a.base_probability();
    assert!((p0 - 1.0 / 1024.0).abs() < 1e-12);
    let best = a.closed_form(a.optimal_iterations());
    assert!(best > 0.999, "Grover reached {best}");
    let dnd = dnd_advantage(p0);
    assert!(dnd < 0.002, "D&D advantage only reached {dnd}");
    assert!(best > 500.0 * dnd);
}

#[test]
fn one_iteration_already_beats_a_dnd_advantage_roll() {
    // For a small base probability, one rotation multiplies it by about nine; advantage
    // multiplies it by about two. So the equivalent setting of the dial is k = 1.
    for qubits in 6..=12 {
        let a = Advantage::new(qubits, vec![1]);
        assert_eq!(
            a.dnd_equivalent_iterations(),
            Some(1),
            "{qubits} qubits: one iteration should already beat advantage"
        );
        let p0 = a.base_probability();
        let ratio = a.closed_form(1) / p0;
        assert!(
            (ratio - 9.0).abs() < 0.6,
            "{qubits} qubits: one iteration gave {ratio}x"
        );
        assert!((dnd_advantage(p0) / p0 - 2.0).abs() < 0.05);
    }
}

#[test]
fn too_much_advantage_becomes_disadvantage() {
    // Part VI-A T1's souffle, at a computable step count. Past the optimum every further
    // turn of the dial costs probability, and it eventually costs almost all of it.
    let a = Advantage::new(10, vec![7]);
    let k = a.optimal_iterations();
    assert!(k > 10, "expected a real rotation budget, got {k}");
    assert!(a.closed_form(k) > 0.999);
    assert!(!a.over_rotated(k));
    for j in 1..=3 {
        assert!(a.over_rotated(k + j), "k + {j} should be over-rotated");
        assert!(a.closed_form(k + j) < a.closed_form(k));
    }
    // Turn it far enough and the marked outcome becomes nearly unreachable: worse than
    // never having amplified at all.
    let ruin = (0..=3 * k).map(|j| a.closed_form(j)).fold(1.0f64, f64::min);
    assert!(ruin < a.base_probability(), "over-rotation fell to {ruin}");
}

#[test]
fn the_souffle_point_is_where_the_simulator_says_it_is() {
    // The predicted optimum is not a fitted number: simulate the whole curve and check the
    // argmax lands on it.
    for qubits in 4..=10 {
        let a = Advantage::new(qubits, vec![3]);
        let k = a.optimal_iterations();
        let mut best_k = 0;
        let mut best_p = 0.0;
        for j in 0..=2 * k + 2 {
            let p = a.measured(j);
            if p > best_p {
                best_p = p;
                best_k = j;
            }
        }
        assert_eq!(
            best_k, k,
            "{qubits} qubits: simulated argmax {best_k}, predicted {k}"
        );
    }
}

#[test]
fn more_marked_states_means_a_shorter_dial() {
    // The dial's length is set by the position, not chosen: sqrt(N/M) iterations.
    let one = Advantage::new(12, vec![0]);
    let many = Advantage::new(12, (0..16).collect());
    assert!(one.optimal_iterations() > 3 * many.optimal_iterations());
    assert!(many.closed_form(many.optimal_iterations()) > 0.99);
}

#[test]
fn a_die_has_a_floor_and_the_dial_does_not() {
    // Rolling with disadvantage squares the probability, and that is as bad as a die can be.
    // The dial has no such floor: turn it far enough past the optimum and the marked outcome
    // becomes arbitrarily unreachable. This is the asymmetry Part IX 1.3 is reaching for, and
    // it cuts both ways -- the richer mechanic is also the one that can be ruined.
    for qubits in 4..=14 {
        let a = Advantage::new(qubits, vec![1]);
        let p0 = a.base_probability();
        let k = a.optimal_iterations();

        // One turn past the optimum already costs more than never amplifying.
        let one_turn = (0..=2 * k + 1)
            .map(|j| a.closed_form(j))
            .fold(1.0f64, f64::min);
        assert!(
            one_turn < p0,
            "{qubits} qubits: one turn bottomed out at {one_turn} >= {p0}"
        );

        // Keep turning and it falls below what a disadvantaged die could ever reach. How far
        // you have to turn is erratic -- between 1 and 62 turns across this range -- because
        // it asks how well odd multiples of theta approximate multiples of pi, which is an
        // equidistribution question and not a monotone function of the register size. The
        // budget below is generous for that reason; the claim being tested is that a floor
        // does not exist, not that it is reached on any particular schedule.
        let mut fell = false;
        for turns in 1..=100u64 {
            let m = (0..=(2 * k as u64 + 1) * turns)
                .map(|j| a.closed_form(j as usize))
                .fold(1.0f64, f64::min);
            if m < dnd_disadvantage(p0) {
                fell = true;
                break;
            }
        }
        assert!(
            fell,
            "{qubits} qubits: never fell below disadvantage {}",
            dnd_disadvantage(p0)
        );
    }
}

#[test]
fn the_curve_is_the_panel_data() {
    let a = Advantage::new(6, vec![9]);
    let c = a.curve(2 * a.optimal_iterations());
    assert_eq!(c.len(), 2 * a.optimal_iterations() + 1);
    assert!(c.iter().all(|p| (0.0..=1.0).contains(p)));
    assert!((c[0] - a.base_probability()).abs() < 1e-12);
}
