//! M28: the three exchange statistics, side by side.
//!
//! `cargo run --release -p overtone-walk --example statistics`

use overtone_walk::two::{correlate, similarity, Statistics};
use overtone_walk::{Coin, Substrate, Word};

fn main() {
    let steps = 4;
    let sub = Substrate::new(Word::Periodic, 64, 0);
    let (a, b) = (Coin::hadamard(), Coin::hadamard());
    // Both walkers enter the same site in opposite coin states -- the coined-walk analogue
    // of two photons meeting at one beam splitter, and the only arrangement where exchange
    // can do anything.
    //
    // Adjacent sites do NOT work, and the reason is worth stating: a coined walk preserves
    // the parity of `site + step`, so a walker starting at site 0 and one starting at site 1
    // occupy disjoint sublattices forever. Every exchange term U_IL U_JK is then zero against
    // a non-zero U_IK U_JL, all five statistics give the identical distribution, and the test
    // passes while measuring nothing.
    let (i, j) = ((0i64, 0usize), (0i64, 1usize));

    let cases = [
        Statistics::Bosonic,
        Statistics::Anyonic(std::f64::consts::FRAC_PI_4),
        Statistics::Anyonic(std::f64::consts::FRAC_PI_2),
        Statistics::Anyonic(3.0 * std::f64::consts::FRAC_PI_4),
        Statistics::Fermionic,
    ];

    println!("{steps}-step Hadamard walk, both inputs at site 0 with opposite coins\n");
    println!(
        "{:>10}  {:>6}  {:>14}  {:>18}  {:>10}",
        "statistics", "phi/pi", "mode diagonal", "position diagonal", "total"
    );
    let mut all = Vec::new();
    for s in cases {
        let c = correlate(steps, i, j, s, &sub, &a, &b);
        println!(
            "{:>10}  {:>6.2}  {:>14.6}  {:>18.6}  {:>10.6}",
            s.name(),
            s.phase() / std::f64::consts::PI,
            c.mode_bunching(),
            c.position_bunching(),
            c.total()
        );
        all.push((s, c));
    }

    println!();
    println!("Pauli exclusion is exact in the MODE basis and absent in the POSITION basis.");
    println!("Two fermions share a site by taking opposite coins -- Sansoni et al. Eq. (4).\n");

    let bos = &all[0].1;
    let fer = &all[4].1;
    println!(
        "similarity(bosonic, fermionic)  = {:.4}",
        similarity(&bos.positions, &fer.positions)
    );
    for (s, c) in &all[1..4] {
        println!(
            "similarity(bosonic, phi={:.2}pi) = {:.4}   similarity(fermionic, same) = {:.4}",
            s.phase() / std::f64::consts::PI,
            similarity(&bos.positions, &c.positions),
            similarity(&fer.positions, &c.positions)
        );
    }

    println!();
    println!("fermionic position-space correlation, rows/cols are lattice sites:");
    let modes = fer.modes;
    let sites = modes.sites();
    print!("       ");
    for x in 0..sites {
        print!("{:>7}", modes.position_of(2 * x));
    }
    println!();
    for x in 0..sites {
        print!("{:>5}  ", modes.position_of(2 * x));
        for y in 0..sites {
            print!("{:>7.3}", fer.positions[x][y]);
        }
        println!();
    }
}
