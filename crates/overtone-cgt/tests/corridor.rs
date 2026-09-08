//! M48's acceptance test: reproduce a published Go endgame temperature.
//!
//! Part IX 8 makes this the trap that decides whether the heat map is real: "If the
//! implementation cannot reproduce a published Go endgame temperature, the heat map is
//! decoration." The corridor is the right family to be pinned to, because Berlekamp & Wolfe
//! state its analysis as a recursion with no board in it, so the test checks thermography
//! rather than a Go rules engine:
//!
//! ```text
//! Corr(0) = 0,   Corr(n+1) = { n | Corr(n) },   f(Corr(n)) = n - 2 + (1/2)^(n-1)
//! ```
//!
//! `f` is chilling by one, and a chilled corridor is a *number*, so `f(Corr(n))` is the mean
//! value of `Corr(n)` -- which this crate computes independently, from the walls.

use overtone_cgt::Game;

/// The published closed form for the chilled corridor.
fn published_mean(n: usize) -> f64 {
    n as f64 - 2.0 + 0.5f64.powi(n as i32 - 1)
}

#[test]
fn corridor_means_match_berlekamp_and_wolfe() {
    for n in 1..=20 {
        let g = Game::corridor(n);
        let got = g.mean();
        let want = published_mean(n);
        assert_eq!(
            got, want,
            "Corr({n}) mean: thermography gave {got}, Berlekamp & Wolfe give {want}"
        );
    }
}

#[test]
fn corridor_temperatures_are_one_minus_a_dyadic() {
    // The temperature that falls out alongside the published mean. It is strictly below 1
    // for every n, which is exactly why chilling by one lands the corridor on a number.
    for n in 1..=20 {
        let t = Game::corridor(n).temperature();
        let want = 1.0 - 0.5f64.powi(n as i32 - 1);
        assert_eq!(t, want, "Corr({n}) temperature");
        assert!(t < 1.0, "Corr({n}) temperature {t} is not below 1");
    }
}

#[test]
fn the_short_corridors_are_the_hand_computed_ones() {
    // Corr(1) = {0 | 0} = *, the canonical temperature-zero game.
    assert_eq!(Game::corridor(1), Game::star());
    assert_eq!(Game::corridor(1).temperature(), 0.0);
    assert_eq!(Game::corridor(1).mean(), 0.0);
    // Corr(2) = {1 | *}: the left wall is 1 - t, the right wall is t, they meet at 1/2.
    let t2 = Game::corridor(2).thermograph();
    assert_eq!(t2.temperature, 0.5);
    assert_eq!(t2.mean, 0.5);
    assert_eq!(t2.left.eval(0.0), 1.0);
    assert_eq!(t2.right.eval(0.0), 0.0);
    // Corr(3) = {2 | Corr(2)}: 2 - t against 1/2 + t, meeting at 3/4.
    let t3 = Game::corridor(3).thermograph();
    assert_eq!(t3.temperature, 0.75);
    assert_eq!(t3.mean, 1.25);
}

#[test]
fn corridor_thermography_is_exact_in_f64() {
    // Every quantity in temperature theory is a dyadic rational, and dyadic rationals below
    // 2^52 are exactly representable in f64. So the equalities above are not "within
    // tolerance" -- they are equalities, and this test says so out loud by checking that the
    // computed temperature is a dyadic rational with the denominator the theory predicts.
    for n in 1..=20 {
        let t = Game::corridor(n).temperature();
        let scaled = t * 2f64.powi(n as i32 - 1);
        assert_eq!(
            scaled,
            scaled.round(),
            "Corr({n}) temperature is not dyadic at 2^{}",
            n - 1
        );
    }
}
