//! Staged trace, used by `scripts/wasm_determinism.sh` to compare native and wasm.
fn main() {
    let mut lab = overtone_wasm::Lab::new(2, 3, 3, 0, false, true, 1.0, 0.05, 7);
    // Stage A: no training at all. Isolates the simulator and quadrature.
    println!("A {:.17e}", lab.exact_return());
    println!("A {:.17e}", lab.gradient_agreement());
    for v in lab.spectrum(64) {
        println!("A {v:.17e}");
    }
    // Stage B: gradients only, still no RNG or optimiser state.
    for v in lab.gradient_scatter() {
        println!("B {v:.17e}");
    }
    // Stage C: the full training loop, which adds the RNG and Adam.
    for _ in 0..20 {
        lab.train_steps(1);
    }
    for v in lab.return_trace() {
        println!("C {v:.17e}");
    }
}
