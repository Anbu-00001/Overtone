//! The thermograph construction, on games whose values are known by hand.

use overtone_cgt::{Game, Thermograph, Wall};

#[test]
fn a_switch_has_mean_and_temperature_from_its_two_numbers() {
    for &(a, b) in &[(3.0, 1.0), (2.0, -2.0), (0.5, 0.25), (7.0, -1.0)] {
        let t = Game::switch(a, b).thermograph();
        assert_eq!(t.mean, (a + b) / 2.0, "mean of {{{a} | {b}}}");
        assert_eq!(t.temperature, (a - b) / 2.0, "temperature of {{{a} | {b}}}");
    }
}

#[test]
fn a_number_has_a_mast_all_the_way_down() {
    let t = Game::number(2.5).thermograph();
    assert_eq!(t.mean, 2.5);
    assert_eq!(t.temperature, -1.0, "the convention for numbers");
    assert_eq!(t.left.eval(0.0), 2.5);
    assert_eq!(t.right.eval(99.0), 2.5);
}

#[test]
fn star_is_the_coldest_non_number() {
    let t = Game::star().thermograph();
    assert_eq!(t.temperature, 0.0);
    assert_eq!(t.mean, 0.0);
}

#[test]
fn walls_have_the_slopes_the_theory_says() {
    // Left walls fall at 0 or 1; right walls rise at 0 or 1. This is what makes the crossing
    // exist and what makes the breakpoint representation exact.
    for n in 1..=8 {
        let t = Game::corridor(n).thermograph();
        for w in t.left.breakpoints().windows(2) {
            let s = (w[1].1 - w[0].1) / (w[1].0 - w[0].0);
            assert!(
                (-1.0 - 1e-12..=1e-12).contains(&s),
                "left wall slope {s} out of [-1, 0]"
            );
        }
        for w in t.right.breakpoints().windows(2) {
            let s = (w[1].1 - w[0].1) / (w[1].0 - w[0].0);
            assert!(
                (-1e-12..=1.0 + 1e-12).contains(&s),
                "right wall slope {s} out of [0, 1]"
            );
        }
    }
}

#[test]
fn both_walls_are_the_mast_above_the_temperature() {
    for n in 1..=10 {
        let t = Game::corridor(n).thermograph();
        for step in 0..5 {
            let above = t.temperature + 0.5 * step as f64 + 1e-9;
            assert!((t.left.eval(above) - t.mean).abs() < 1e-9);
            assert!((t.right.eval(above) - t.mean).abs() < 1e-9);
        }
    }
}

#[test]
fn a_tepid_game_can_get_hotter_when_it_is_played() {
    // Blom's example, and the reason hottest-first is a heuristic and not a theorem:
    // Right's only move here raises the temperature from 0 to 1.
    let inner = Game::switch(0.0, -2.0);
    assert_eq!(inner.temperature(), 1.0);
    let g = Game::new(vec![Game::number(0.0)], vec![inner]);
    assert_eq!(g.temperature(), 0.0);
    assert!(
        g.right_options()[0].temperature() > g.temperature(),
        "the option is hotter than the position it came from"
    );
}

#[test]
fn wall_arithmetic_is_pointwise() {
    let a = Wall::constant(1.0).add_slope(-1.0); // 1 - t
    let b = Wall::constant(-1.0).add_slope(1.0); // t - 1
    let m = a.max(&b);
    assert_eq!(m.eval(0.0), 1.0);
    assert_eq!(m.eval(1.0), 0.0);
    assert_eq!(m.eval(3.0), 2.0);
    let n = a.min(&b);
    assert_eq!(n.eval(0.0), -1.0);
    assert_eq!(n.eval(3.0), -2.0);
}

#[test]
fn from_options_agrees_with_the_game_recursion() {
    let l = [Thermograph::number(4.0)];
    let r = [Thermograph::number(0.0)];
    let t = Thermograph::from_options(&l, &r);
    assert_eq!(t, Game::switch(4.0, 0.0).thermograph());
}
