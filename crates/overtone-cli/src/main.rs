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
