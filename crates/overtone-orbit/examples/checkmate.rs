//! M34: what the cheap checkmate certificate proves, and what it misses.
//!
//! `cargo run --release -p overtone-orbit --example checkmate`

use overtone_lie::{closure, Algebra, PauliString};
use overtone_orbit::checkmate::{best_reachable_fidelity, haar_state, matched_invariant_state};
use overtone_orbit::separation_deficit;
use overtone_orbit::OrbitCertificate;
use overtone_sim::Pauli;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn tfim(n: usize) -> Vec<PauliString> {
    let mut g: Vec<PauliString> = (0..n).map(|q| PauliString::single(q, Pauli::X)).collect();
    for q in 0..n - 1 {
        g.push(PauliString::from_factors(&[
            (q, Pauli::Z),
            (q + 1, Pauli::Z),
        ]));
    }
    g
}

fn local_x(n: usize) -> Vec<PauliString> {
    (0..n).map(|q| PauliString::single(q, Pauli::X)).collect()
}

/// One `Z2` sector generator per neighbouring pair: the bishop of Part VII 2.
fn bishop(n: usize) -> Vec<PauliString> {
    (0..n - 1)
        .map(|q| PauliString::from_factors(&[(q, Pauli::X), (q + 1, Pauli::X)]))
        .collect()
}

fn full(n: usize) -> Vec<PauliString> {
    let mut g = local_x(n);
    for q in 0..n {
        g.push(PauliString::single(q, Pauli::Z));
    }
    for q in 0..n - 1 {
        g.push(PauliString::from_factors(&[
            (q, Pauli::Y),
            (q + 1, Pauli::Z),
        ]));
    }
    g
}

fn main() {
    println!("the certificate, on four algebras at three qubits and four");
    println!(
        "{:>12}  {:>3}  {:>7}  {:>10}  {:>8}  {:>7}",
        "algebra", "n", "dim(g)", "commutant", "ideals", "full?"
    );
    type Family = (&'static str, fn(usize) -> Vec<PauliString>);
    let families: Vec<Family> = vec![
        ("local X", local_x),
        ("bishop XX", bishop),
        ("TFIM", tfim),
        ("su(2^n)", full),
    ];
    for n in [3usize, 4] {
        for (name, f) in &families {
            let a = closure(&f(n), n, 4096);
            let cert = OrbitCertificate::new(&a);
            println!(
                "{name:>12}  {n:>3}  {:>7}  {:>10}  {:>8}  {:>7}",
                a.dim(),
                cert.commutant().len(),
                cert.ideals().len(),
                cert.fully_controllable()
            );
        }
    }

    println!("\nsoundness and completeness against a layered-ansatz search");
    println!(
        "{:>12}  {:>3}  {:>9}  {:>12}  {:>13}  {:>12}",
        "algebra", "n", "pairs", "unreachable", "proved by cert", "completeness"
    );
    for n in [3usize, 4] {
        for (name, f) in &families {
            let a = closure(&f(n), n, 4096);
            if OrbitCertificate::new(&a).fully_controllable() {
                // Not measured, and deliberately. Under su(2^n) the group is transitive on
                // the sphere, so every pair is connected as a theorem -- running a search to
                // rediscover that would be spending minutes to confirm an identity, and the
                // 510-parameter ansatz it would need is the slowest lane in the sweep.
                println!(
                    "{name:>12}  {n:>3}  {:>9}  {:>12}  {:>13}  {:>12}",
                    "-", "0 (theory)", "-", "-"
                );
                continue;
            }
            let (unreachable, proved, unsound) = measure(&a, n, 60, 4242 + n as u64);
            let completeness = if unreachable == 0 {
                f64::NAN
            } else {
                proved as f64 / unreachable as f64
            };
            let shown = if completeness.is_nan() {
                "-".to_string()
            } else {
                format!("{:.3}", completeness)
            };
            println!(
                "{name:>12}  {n:>3}  {:>9}  {unreachable:>12}  {proved:>13}  {shown:>12}",
                60
            );
            assert_eq!(unsound, 0, "SOUNDNESS VIOLATED on {name} at n={n}");
        }
    }
    println!("  soundness held on every pair: the certificate never contradicted the search.");
    println!("  completeness is the fraction of pairs the search could not connect that the");
    println!("  certificate also proved disconnected. A search failure is not a proof, so this");
    println!("  is an estimate of the gap and not a bound on it.");

    println!("\nwhy random pairs cannot measure completeness: a counting argument");
    println!(
        "{:>12}  {:>3}  {:>9}  {:>8}  {:>11}  {:>11}  {:>9}",
        "algebra", "n", "ambient", "dim(g)", "invariants", "orbit dim", "deficit"
    );
    let mut rng = ChaCha8Rng::seed_from_u64(99);
    for n in [3usize, 4] {
        for (name, f) in &families {
            let a = closure(&f(n), n, 4096);
            let cert = OrbitCertificate::new(&a);
            let psi = haar_state(n, &mut rng);
            let orbit = overtone_orbit::orbit_dimension(&a, &psi, 1e-9);
            println!(
                "{name:>12}  {n:>3}  {:>9}  {:>8}  {:>11}  {orbit:>11}  {:>9}",
                2 * (1 << n) - 2,
                a.dim(),
                cert.commutant().len() + cert.ideals().len(),
                separation_deficit(&a, &cert, &psi)
            );
        }
    }
    println!("  a positive deficit means the invariant level set contains a continuum of");
    println!("  distinct orbits, so no certificate built from those invariants can separate");
    println!("  them. Every proper algebra here has one. The certificate is incomplete as a");
    println!("  matter of counting, before any experiment is run.");

    println!("\ncompleteness measured on pairs built to share the invariants");
    println!(
        "{:>12}  {:>3}  {:>10}  {:>12}  {:>13}  {:>12}",
        "algebra", "n", "hard pairs", "unreachable", "proved by cert", "completeness"
    );
    for n in [3usize, 4] {
        for (name, f) in &families {
            let a = closure(&f(n), n, 4096);
            let cert = OrbitCertificate::new(&a);
            if cert.fully_controllable() {
                continue;
            }
            let mut rng = ChaCha8Rng::seed_from_u64(31 + n as u64);
            let (mut built, mut unreachable, mut proved) = (0, 0, 0);
            for _ in 0..24 {
                let source = haar_state(n, &mut rng);
                let (target, residual) = matched_invariant_state(&cert, &source, &mut rng, 60);
                if residual > 1e-6 {
                    continue;
                }
                built += 1;
                let says = cert.certainly_unreachable(&source, &target, 1e-6);
                let connected =
                    best_reachable_fidelity(&a, &source, &target, 2, 6, &mut rng) > 0.999;
                assert!(
                    !(says && connected),
                    "SOUNDNESS VIOLATED on {name} at n={n}"
                );
                if !connected {
                    unreachable += 1;
                    if says {
                        proved += 1;
                    }
                }
            }
            let shown = if unreachable == 0 {
                "-".to_string()
            } else {
                format!("{:.3}", proved as f64 / unreachable as f64)
            };
            println!(
                "{name:>12}  {n:>3}  {built:>10}  {unreachable:>12}  {proved:>13}  {shown:>12}"
            );
        }
    }
    println!("  this is the number Part VII 11 is really asking for, and it is the one that");
    println!("  says whether the cheap predicate is playable.");

    println!("\nthe number the game actually depends on: checkmate against basis-state safety");
    println!(
        "{:>12}  {:>3}  {:>10}  {:>13}  {:>15}  {:>12}",
        "algebra", "n", "positions", "truly lost", "called checkmate", "completeness"
    );
    for n in [3usize, 4] {
        for (name, f) in &families {
            let a = closure(&f(n), n, 4096);
            let cert = OrbitCertificate::new(&a);
            if cert.fully_controllable() {
                continue;
            }
            let mut rng = ChaCha8Rng::seed_from_u64(77 + n as u64);
            let (mut lost, mut called) = (0, 0);
            let positions = 30;
            for _ in 0..positions {
                let state = haar_state(n, &mut rng);
                // The safe subspace: the first quarter of the computational basis. A real
                // position's safe region is a set of maze cells, which is a set of basis
                // states -- not an adversarially chosen point on an invariant level set.
                let safe: Vec<usize> = (0..(1usize << n) / 4).collect();
                let position = overtone_orbit::checkmate::Position::new(
                    state.clone(),
                    a.clone(),
                    safe.clone(),
                );
                let says = position.is_checkmate(1e-6);
                let escapable = position
                    .safe_samples()
                    .iter()
                    .any(|s| best_reachable_fidelity(&a, &state, s, 2, 4, &mut rng) > 0.999);
                assert!(
                    !(says && escapable),
                    "SOUNDNESS VIOLATED on {name} at n={n}"
                );
                if !escapable {
                    lost += 1;
                    if says {
                        called += 1;
                    }
                }
            }
            let shown = if lost == 0 {
                "-".to_string()
            } else {
                format!("{:.3}", called as f64 / lost as f64)
            };
            println!("{name:>12}  {n:>3}  {positions:>10}  {lost:>13}  {called:>15}  {shown:>12}");
        }
    }

    println!("\nwhat the certificate is made of");
    let a = closure(&local_x(4), 4, 4096);
    let cert = OrbitCertificate::new(&a);
    println!("  local X at n=4, dim(g) = {}", a.dim());
    print!("  commutant:");
    for q in cert.commutant().iter().take(8) {
        print!(" {}", q.render(4));
    }
    println!();
    println!("  those are the superselection sectors -- the bishop's square colour, computed.");
    println!("  Every one of them is a Pauli string commuting with all of g, so its");
    println!("  expectation is conserved exactly and linearly along the whole orbit.");
}

/// Returns `(unreachable, proved, unsound)`.
fn measure(a: &Algebra, n: usize, pairs: usize, seed: u64) -> (usize, usize, usize) {
    let cert = OrbitCertificate::new(a);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let (mut unreachable, mut proved, mut unsound) = (0, 0, 0);
    for _ in 0..pairs {
        let source = haar_state(n, &mut rng);
        let target = haar_state(n, &mut rng);
        let says = cert.certainly_unreachable(&source, &target, 1e-6);
        let f = best_reachable_fidelity(a, &source, &target, 2, 6, &mut rng);
        let connected = f > 0.999;
        if says && connected {
            unsound += 1;
        }
        if !connected {
            unreachable += 1;
            if says {
                proved += 1;
            }
        }
    }
    (unreachable, proved, unsound)
}
