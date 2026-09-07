//! The native trainer. Emits JSONL traces (Part I 5).
//!
//! Argument parsing and JSON writing are hand-rolled. The trace has four fields and the
//! argument list is flat, so a dependency here would buy nothing and would have to be
//! justified to the WASM build later.

use std::fmt::Write as _;

use overtone_rl::ceiling::{self, DEFAULT_GRID};
use overtone_rl::policy::Policy;
use overtone_rl::reinforce::{coarse_tune_lambda, initial_params, train, TrainConfig};
use overtone_rl::{Scaling, SpectralControl, SpectralControlAnsatz};
use overtone_sim::random::rng;

const USAGE: &str = "\
overtone - quantum RL spectral instrumentation

USAGE:
    overtone train [OPTIONS]
    overtone ceiling [OPTIONS]
    overtone spectrum [OPTIONS]
    overtone plateau [OPTIONS]
    overtone predict [OPTIONS]
    overtone dequantize [OPTIONS]

TRAIN OPTIONS:
    --k <N>            environment frequency for SpectralControl-k   [default: 3]
    --qubits <N>       register size                                 [default: 2]
    --layers <N>       re-uploading layers, L                        [default: 2]
    --policy <KIND>    raw | softmax                                 [default: raw]
    --scaling <KIND>   pinned | trainable                            [default: pinned]
    --beta <F>         softmax inverse temperature                   [default: 1.0]
    --episodes <N>     total episodes                                [default: 5000]
    --batch <N>        episodes per gradient step                    [default: 50]
    --lr <F>           learning rate                                 [default: 0.05]
    --seed <N>         RNG seed; same seed, same trajectory          [default: 7]
    --no-entangle      ablate the ring of CZ to identity
    --coarse-tune      sweep lambda before training (trainable only)
    --out <PATH>       write JSONL trace here instead of stdout

CEILING OPTIONS:
    --k <N>            environment frequency                         [default: 3]
    --max-c <N>        highest frequency ceiling to tabulate         [default: 12]

SPECTRUM OPTIONS:
    accepts the TRAIN options; trains, then transforms pi(1|s) and reports the
    magnitude at each frequency together with what leaked above the ceiling

PREDICT OPTIONS:
    --qubits <N>       register size                                 [default: 4]
    --layers <N>       re-uploading layers, L                        [default: 2]
    --no-entangle      ablate the ring of CZ to identity
    --entanglers <K>   fixed | parameterised: how to treat CZ layers [default: fixed]
    --family <NAME>    ansatz | tfim | heisenberg | xy | kitaev | hea
    --max-dim <N>      abandon the closure past this many elements   [default: 200000]

DEQUANTIZE OPTIONS:
    accepts the TRAIN options; trains, then compresses the policy state to each bond
    dimension in turn and reports the smallest one that reproduces it
    --tolerance <F>    policy agreement required                     [default: 1e-6]
    --max-chi <N>      largest bond dimension to try                 [default: 16]

PLATEAU OPTIONS:
    --min-qubits <N>   sweep start                                   [default: 2]
    --max-qubits <N>   sweep end                                     [default: 10]
    --depth <KIND>     const:<d> | log:<c> | linear:<c>              [default: log:2]
    --samples <N>      random parameter vectors per point            [default: 300]
    --seed <N>         RNG seed                                      [default: 7]
";

struct Args {
    map: std::collections::HashMap<String, String>,
    flags: std::collections::HashSet<String>,
}

impl Args {
    fn parse(raw: &[String]) -> Args {
        let mut map = std::collections::HashMap::new();
        let mut flags = std::collections::HashSet::new();
        let mut i = 0;
        while i < raw.len() {
            let a = &raw[i];
            if let Some(name) = a.strip_prefix("--") {
                if i + 1 < raw.len() && !raw[i + 1].starts_with("--") {
                    map.insert(name.to_string(), raw[i + 1].clone());
                    i += 2;
                } else {
                    flags.insert(name.to_string());
                    i += 1;
                }
            } else {
                i += 1;
            }
        }
        Args { map, flags }
    }

    fn get<T: std::str::FromStr>(&self, name: &str, default: T) -> T {
        match self.map.get(name) {
            Some(v) => v
                .parse()
                .unwrap_or_else(|_| fail(&format!("could not parse --{name} value {v:?}"))),
            None => default,
        }
    }

    fn text(&self, name: &str, default: &str) -> String {
        self.map
            .get(name)
            .cloned()
            .unwrap_or_else(|| default.to_string())
    }

    fn flag(&self, name: &str) -> bool {
        self.flags.contains(name)
    }
}

fn fail(msg: &str) -> ! {
    eprintln!("overtone: {msg}");
    std::process::exit(2);
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match argv.first().map(String::as_str) {
        Some("train") => cmd_train(&Args::parse(&argv[1..])),
        Some("ceiling") => cmd_ceiling(&Args::parse(&argv[1..])),
        Some("spectrum") => cmd_spectrum(&Args::parse(&argv[1..])),
        Some("plateau") => cmd_plateau(&Args::parse(&argv[1..])),
        Some("predict") => cmd_predict(&Args::parse(&argv[1..])),
        Some("dequantize") => cmd_dequantize(&Args::parse(&argv[1..])),
        Some("--help") | Some("-h") | None => print!("{USAGE}"),
        Some(other) => fail(&format!("unknown command {other:?}\n\n{USAGE}")),
    }
}

fn cmd_ceiling(args: &Args) {
    let k: usize = args.get("k", 3);
    let max_c: usize = args.get("max-c", 12);

    println!("# LP ceiling J*(C) for SpectralControl-{k}");
    println!("# the best any strictly band-limited policy with frequency ceiling C can score");
    println!("{:>4} {:>12}", "C", "J*");
    for c in 0..=max_c {
        println!("{c:>4} {:>12.6}", ceiling::ceiling(c, k, DEFAULT_GRID));
    }
    println!(
        "{:>4} {:>12.6}  (2/pi, unconstrained optimum)",
        "inf",
        2.0 / std::f64::consts::PI
    );
}

fn cmd_train(args: &Args) {
    let k: usize = args.get("k", 3);
    let qubits: usize = args.get("qubits", 2);
    let layers: usize = args.get("layers", 2);
    let beta: f64 = args.get("beta", 1.0);
    let seed: u64 = args.get("seed", 7);
    let entangle = !args.flag("no-entangle");

    let scaling = match args.text("scaling", "pinned").as_str() {
        "pinned" => Scaling::Pinned,
        "trainable" => Scaling::Trainable,
        other => fail(&format!(
            "--scaling must be pinned or trainable, got {other:?}"
        )),
    };

    let ansatz = SpectralControlAnsatz::build(qubits, layers, scaling, entangle);
    let ceiling_c = ansatz.frequency_ceiling(0);

    let policy = match args.text("policy", "raw").as_str() {
        "raw" => Policy::raw(ansatz),
        "softmax" => Policy::softmax(ansatz, beta),
        other => fail(&format!("--policy must be raw or softmax, got {other:?}")),
    };

    let cfg = TrainConfig {
        episodes: args.get("episodes", 5000),
        batch_size: args.get("batch", 50),
        learning_rate: args.get("lr", 0.05),
        eval_nodes: 512,
        record_every: 1,
    };

    let env = SpectralControl::new(k);
    let mut r = rng(seed);
    let mut params = initial_params(&policy, 0.3, 1.0, &mut r);

    let mut tuned_lambda = None;
    if args.flag("coarse-tune") && !policy.ansatz.lambda_range().is_empty() {
        // The return is a resonance curve in lambda; gradient ascent started outside the
        // capture range locks onto a sidelobe. Scan first, then descend.
        let lambda = coarse_tune_lambda(&policy, &params, k, 8.0, 200);
        for idx in policy.ansatz.lambda_range() {
            params[idx] = lambda;
        }
        tuned_lambda = Some(lambda);
    }

    let started = std::time::Instant::now();
    let out = train(&policy, &params, &env, &cfg, &mut r);
    let elapsed = started.elapsed();

    let cap = ceiling::ceiling(ceiling_c, k, DEFAULT_GRID);
    let mut jsonl = String::new();

    let _ = writeln!(
        jsonl,
        "{{\"type\":\"meta\",\"k\":{k},\"qubits\":{qubits},\"layers\":{layers},\
         \"frequency_ceiling\":{ceiling_c},\"lp_ceiling\":{},\"policy\":\"{}\",\
         \"scaling\":\"{}\",\"beta\":{beta},\"entangle\":{entangle},\"seed\":{seed},\
         \"episodes\":{},\"batch_size\":{},\"learning_rate\":{}{}}}",
        json_f64(cap),
        if matches!(policy.kind, overtone_rl::PolicyKind::Raw) {
            "raw"
        } else {
            "softmax"
        },
        if scaling == Scaling::Pinned {
            "pinned"
        } else {
            "trainable"
        },
        cfg.episodes,
        cfg.batch_size,
        json_f64(cfg.learning_rate),
        match tuned_lambda {
            Some(l) => format!(",\"coarse_tuned_lambda\":{}", json_f64(l)),
            None => String::new(),
        }
    );

    for row in &out.trace {
        let _ = writeln!(
            jsonl,
            "{{\"type\":\"step\",\"step\":{},\"episodes\":{},\"sampled_return\":{},\"exact_return\":{}}}",
            row.step,
            row.episodes,
            json_f64(row.sampled_return),
            json_f64(row.exact_return)
        );
    }

    let lambdas: Vec<String> = policy
        .ansatz
        .lambda_range()
        .map(|i| json_f64(out.params[i]))
        .collect();
    let _ = writeln!(
        jsonl,
        "{{\"type\":\"final\",\"exact_return\":{},\"lp_ceiling\":{},\"lambda\":[{}],\
         \"wall_clock_seconds\":{}}}",
        json_f64(out.final_exact_return),
        json_f64(cap),
        lambdas.join(","),
        json_f64(elapsed.as_secs_f64())
    );

    match args.map.get("out") {
        Some(path) => {
            std::fs::write(path, &jsonl).unwrap_or_else(|e| fail(&format!("writing {path}: {e}")));
            eprintln!(
                "wrote {} trace rows to {path}\nJ = {:.6}   LP ceiling = {:.6}   {:.2}s",
                out.trace.len() + 2,
                out.final_exact_return,
                cap,
                elapsed.as_secs_f64()
            );
        }
        None => print!("{jsonl}"),
    }
}

/// Serialise a float as valid JSON. Non-finite values have no JSON representation, so they
/// become null rather than the bare `NaN` that most writers emit and no parser accepts.
fn json_f64(x: f64) -> String {
    if x.is_finite() {
        format!("{x:.10}")
    } else {
        "null".to_string()
    }
}

fn cmd_spectrum(args: &Args) {
    use overtone_spec::spectrum_of;

    let k: usize = args.get("k", 3);
    let qubits: usize = args.get("qubits", 2);
    let layers: usize = args.get("layers", 2);
    let beta: f64 = args.get("beta", 1.0);
    let seed: u64 = args.get("seed", 7);
    let softmax = args.text("policy", "raw") == "softmax";

    let ansatz =
        SpectralControlAnsatz::build(qubits, layers, Scaling::Pinned, !args.flag("no-entangle"));
    let ceiling = ansatz.frequency_ceiling(0);
    let policy = if softmax {
        Policy::softmax(ansatz, beta)
    } else {
        Policy::raw(ansatz)
    };

    let env = SpectralControl::new(k);
    let mut r = rng(seed);
    let p0 = initial_params(&policy, 0.3, 1.0, &mut r);
    let cfg = TrainConfig {
        episodes: args.get("episodes", 5000),
        batch_size: args.get("batch", 50),
        learning_rate: args.get("lr", 0.05),
        eval_nodes: 512,
        record_every: 1,
    };
    let out = train(&policy, &p0, &env, &cfg, &mut r);

    let s = spectrum_of(512, ceiling, |x| policy.prob_action1(&out.params, &[x]));

    println!(
        "# spectrum of pi(1|s) for a {} policy, L={layers}, ceiling={ceiling}, k={k}",
        if softmax { "SOFTMAX-PQC" } else { "RAW-PQC" }
    );
    println!("# J = {:.6}", out.final_exact_return);
    println!("{:>6} {:>14}  in band?", "omega", "|c|");
    for (omega, magnitude) in s.magnitude.iter().enumerate().take(4 * ceiling.max(1) + 4) {
        let tag = if omega <= ceiling { "in" } else { "LEAKED" };
        println!("{omega:>6} {magnitude:>14.9}  {tag}");
    }
    println!("#");
    println!("# max above ceiling  {:.3e}", s.energy_above_ceiling());
    println!("# sum above ceiling  {:.3e}", s.total_above_ceiling());
    println!("# leakage ratio      {:.6}", s.leakage_ratio());
}

fn cmd_plateau(args: &Args) {
    use overtone_spec::plateau::{fit_exponential, sweep, CostLocality, DepthPolicy};

    let lo: usize = args.get("min-qubits", 2);
    let hi: usize = args.get("max-qubits", 10);
    let samples: usize = args.get("samples", 300);
    let seed: u64 = args.get("seed", 7);

    let spec = args.text("depth", "log:2");
    let (kind, value) = spec.split_once(':').unwrap_or(("log", "2"));
    let value: usize = value
        .parse()
        .unwrap_or_else(|_| fail(&format!("bad --depth value in {spec:?}")));
    let depth = match kind {
        "const" => DepthPolicy::Constant(value),
        "log" => DepthPolicy::Logarithmic(value),
        "linear" => DepthPolicy::Linear(value),
        other => fail(&format!(
            "--depth kind must be const, log or linear, got {other:?}"
        )),
    };

    let local = sweep(lo..=hi, depth, CostLocality::Local, samples, seed);
    let global = sweep(lo..=hi, depth, CostLocality::Global, samples, seed);

    println!("# gradient variance vs qubit count, depth policy {depth:?}, {samples} samples");
    println!(
        "{:>3} {:>6} {:>16} {:>16}",
        "n", "depth", "Var local Z_0", "Var global ZZ..Z"
    );
    for (l, g) in local.iter().zip(&global) {
        println!(
            "{:>3} {:>6} {:>16.6e} {:>16.6e}",
            l.num_qubits, l.depth, l.variance, g.variance
        );
    }

    let fl = fit_exponential(&local);
    let fg = fit_exponential(&global);
    println!("#");
    println!(
        "# local   Var ~ 2^(-{:.3} n)   R^2 = {:.4}",
        fl.rate, fl.r_squared
    );
    println!(
        "# global  Var ~ 2^(-{:.3} n)   R^2 = {:.4}",
        fg.rate, fg.r_squared
    );
    println!("#");
    println!("# Cerezo et al. 2021: a local observable escapes the plateau only while the");
    println!("# circuit stays shallow. Re-run with --depth linear:2 to watch it stop working.");
}

/// `overtone predict`: what the algebra says before a gradient step is taken.
///
/// Part III 1 wants this to be a tool, not a demo -- something a person with a real VQC
/// runs on a Tuesday. The report is the prediction chain of Part III 2: `dim(g)`, the loss
/// variance it implies, the rank it caps the Fisher matrix at, and the parameter count past
/// which overparameterisation sets in.
fn cmd_predict(args: &Args) {
    use overtone_lie::family::{self, EntanglerPolicy};
    use overtone_lie::{closure, PauliString, Prediction};

    let qubits: usize = args.get("qubits", 4);
    let layers: usize = args.get("layers", 2);
    let max_dim: usize = args.get("max-dim", 200_000);
    let entangle = !args.flag("no-entangle");
    let policy = match args.text("entanglers", "fixed").as_str() {
        "fixed" => EntanglerPolicy::Strict,
        "parameterised" | "parameterized" => EntanglerPolicy::Parameterised,
        other => fail(&format!(
            "--entanglers must be fixed or parameterised, got {other:?}"
        )),
    };
    let family_name = args.text("family", "ansatz");

    let mut fixed_entanglers = 0usize;
    let (generators, note) = match family_name.as_str() {
        "ansatz" => {
            let ansatz = SpectralControlAnsatz::build(qubits, layers, Scaling::Pinned, entangle);
            let circuit = ansatz.build(&[0.3]);
            let extracted = family::from_circuit(&circuit, policy);
            fixed_entanglers = extracted.fixed_entanglers;
            let note = format!(
                "Part I ansatz, {} parameterised generators, {} fixed entanglers, \
                 {} other fixed gates",
                extracted.generators.len(),
                extracted.fixed_entanglers,
                extracted.fixed_local
            );
            (extracted.generators, note)
        }
        "tfim" => (family::tfim(qubits), "transverse-field Ising chain".into()),
        "heisenberg" => (family::heisenberg(qubits), "Heisenberg chain".into()),
        "xy" => (family::xy(qubits), "XY model".into()),
        "kitaev" => (family::kitaev(qubits), "Kitaev chain".into()),
        "hea" => (
            family::hardware_efficient(qubits),
            "hardware-efficient, entanglers parameterised".into(),
        ),
        other => fail(&format!("unknown --family {other:?}")),
    };

    let started = std::time::Instant::now();
    let algebra = closure(&generators, qubits, max_dim);
    let elapsed = started.elapsed();

    // Z on qubit zero is what the Born rule reads in Part I 6.3 -- but it is not always in
    // the algebra, and Theorem 1 needs it to be.
    let preferred = PauliString::single(0, overtone_sim::Pauli::Z);
    let observable = overtone_lie::predict::pick_observable(&algebra, &preferred);
    let mut report = Prediction::new(&algebra, &observable);
    report.note_fixed_entanglers(fixed_entanglers);

    println!("# overtone predict");
    println!("family              {family_name}  ({note})");
    println!("qubits              {qubits}");
    println!("generators          {}", generators.len());
    println!(
        "closure             {} rounds, {:.3} s",
        algebra.growth().len(),
        elapsed.as_secs_f64()
    );
    println!("growth |S|          {:?}", algebra.growth());
    println!();
    println!("dim(g)              {}", report.dim_g);
    println!("dim su(2^n)         {:.0}", report.dim_su);
    println!(
        "blocks              {:?} (centre {})",
        report.ideals, report.centre
    );
    println!("scaling             {:?}", report.scaling);
    println!("observable          {}", observable.render(qubits));
    match report.loss_variance {
        Some(v) => println!("Var[loss]           {v:.6}   (Ragone et al. 2024, Theorem 1)"),
        None => println!("Var[loss]           not applicable"),
    }
    println!("rank(QFIM) <=       {}", report.qfim_rank_bound);
    println!("overparam. at M ~   {}", report.overparameterisation_bound);
    println!();
    for c in &report.caveats {
        println!("caveat: {c}");
    }
    println!("verdict: {}", report.verdict());
}

/// `overtone dequantize`: the smallest tensor network that is this agent.
fn cmd_dequantize(args: &Args) {
    let k: usize = args.get("k", 3);
    let qubits: usize = args.get("qubits", 4);
    let layers: usize = args.get("layers", 3);
    let seed: u64 = args.get("seed", 7);
    let tolerance: f64 = args.get("tolerance", 1e-6);
    let max_chi: usize = args.get("max-chi", 16);
    let entangle = !args.flag("no-entangle");
    let scaling = match args.text("scaling", "pinned").as_str() {
        "pinned" => Scaling::Pinned,
        "trainable" => Scaling::Trainable,
        other => fail(&format!(
            "--scaling must be pinned or trainable, got {other:?}"
        )),
    };

    let ansatz = SpectralControlAnsatz::build(qubits, layers, scaling, entangle);
    let policy = match args.text("policy", "raw").as_str() {
        "raw" => Policy::raw(ansatz),
        "softmax" => Policy::softmax(ansatz, args.get("beta", 1.0)),
        other => fail(&format!("--policy must be raw or softmax, got {other:?}")),
    };
    let cfg = TrainConfig {
        episodes: args.get("episodes", 4000),
        batch_size: args.get("batch", 50),
        learning_rate: args.get("lr", 0.05),
        eval_nodes: 256,
        record_every: 1000,
    };
    let env = SpectralControl::new(k);
    let mut r = rng(seed);
    let mut params = initial_params(&policy, 0.3, 1.0, &mut r);
    if !policy.ansatz.lambda_range().is_empty() {
        let lambda = coarse_tune_lambda(&policy, &params, k, 8.0, 200);
        for idx in policy.ansatz.lambda_range() {
            params[idx] = lambda;
        }
    }
    let out = train(&policy, &params, &env, &cfg, &mut r);
    let trained = out.params.clone();

    let report = overtone_mps::dequantize(
        qubits,
        k,
        64,
        max_chi,
        tolerance,
        |s| policy.ansatz.build(&[s]).run(&trained),
        |st| policy.prob_from_observable(&trained, policy.observable.expectation(st)),
    );

    println!("# overtone dequantize");
    println!("qubits {qubits}  layers {layers}  k {k}  seed {seed}");
    println!("exact return        {:+.6}", report.exact_return);
    println!("tolerance           {:.0e}", report.tolerance);
    println!();
    println!(
        "{:>5} {:>18} {:>16} {:>14}",
        "chi", "max policy error", "mean fidelity", "return"
    );
    for row in &report.rows {
        println!(
            "{:>5} {:>18.3e} {:>16.9} {:>+14.6}",
            row.chi, row.max_policy_error, row.mean_fidelity, row.achieved_return
        );
    }
    println!();
    println!("verdict: {}", report.verdict());
}
