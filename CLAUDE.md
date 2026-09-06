# OVERTONE — agent instructions

Quantum RL instrumentation. A variational quantum circuit that encodes classical data is
exactly a truncated Fourier series in that data (Schuld, Sweke & Meyer, PRA 103, 032430).
This repo points that theorem at a reinforcement-learning policy in real time.

Build specs live in `docs/spec/` — Parts I through V. **Read the relevant part before
touching a crate.** Part I is the spine; Parts II–V are companions that never replace it.
Part III §12 ("Minimum viable Overtone") governs scope whenever a new panel suggests itself.

---

## 1. Code retrieval — route the question, do not grep first

Three retrieval tools are wired in. Each answers a different shape of question. Opening
whole files is the *last* step, for narrowing or for editing specific lines — not the first.

| Question shape | Tool | Why this one |
|---|---|---|
| "where is X used", "what breaks if I change X", blast radius | **CodeGraph** | Returns verbatim line-numbered source of the relevant symbols *plus* call paths and a blast-radius summary in one round-trip |
| precise symbol lookup, symbol-level edit, rename, find refs | **Serena** | Live LSP — never stale, and edits at symbol granularity instead of string matching |
| "how does this fit together", architecture, cross-file/doc structure | **graphify** | Scoped subgraph across code *and* the `docs/spec/` markdown |

**Staleness contract.** Serena is live via LSP and is never stale. CodeGraph and graphify
are derived indexes: CodeGraph has a file-watcher and stays current after `codegraph init`;
graphify goes stale until you run `graphify update .`. So:

- After changing code, run `graphify update .` (AST-only, no API cost) before asking
  graphify anything about what you just changed.
- If a CodeGraph query returns nothing, check for a `.codegraph/` directory before
  concluding the tool is broken.
- Never imply you consulted one of these if it was unavailable in the session. Say so and
  fall back to Read/Grep explicitly.

### 1.1 Per-tool specifics for this repo

**CodeGraph** is global-server but per-project-index. It is initialised here (`.codegraph/`).
Its tools take an explicit `projectPath` — pass the repo root:

```
codegraph_explore(projectPath="/home/anbu/26_class/Playground_3/overtone", query="...")
```

Use it before any change that crosses a crate boundary. The dependency direction is strict
(§2 below), and CodeGraph is how you check you are not about to violate it.

**Serena** — call `initial_instructions` before starting a coding task, then
`activate_project` on the repo root. Prefer `find_symbol` / `find_referencing_symbols` /
`replace_symbol_body` over Read-then-Edit for anything symbol-shaped. `rename_symbol` over
a manual find-and-replace, always: the gate/gradient code has short names (`re`, `im`, `q`,
`theta`) that string-replace will corrupt.

**graphify** — when `graphify-out/graph.json` exists, answer codebase questions with
`graphify query "<question>"` before grepping; `graphify path "<A>" "<B>"` for
relationships; `graphify explain "<concept>"` for a single concept. If
`graphify-out/wiki/index.md` exists, navigate with it instead of browsing source. Read
`GRAPH_REPORT.md` only for broad architecture review. The `docs/spec/` markdown is in the
graph, so `graphify query` is the fastest way to find which spec section governs a piece of
code.

### 1.2 Repo hygiene for these tools

`.codegraph/` and `graphify-out/` are in `.git/info/exclude`, **not** `.gitignore` — they
are local derived indexes and must not appear in a diff sent upstream.

---

## 2. Architecture invariants

```
crates/
├── overtone-sim/    state vector, gates, adjoint + parameter-shift gradients
├── overtone-rl/     environments, policies, REINFORCE, rollout buffers
├── overtone-spec/   instrumentation: Fourier, entropy, gradient variance, LP ceiling
├── overtone-cli/    native trainer; emits JSONL traces
└── overtone-wasm/   wasm-bindgen surface for the browser
```

Dependency direction is one-way and load-bearing:

- `overtone-sim` **knows nothing about RL.** It must not contain the word "policy".
- `overtone-rl` **knows nothing about rendering.** It consumes `sim` through a trait and
  must not reach into its internals.
- `overtone-spec` reads circuits and policies and emits measurements.
- `overtone-wasm` is a thin FFI shim. **If it contains an `if` statement about physics,
  that logic is in the wrong crate.**

Check this with CodeGraph before adding a dependency, not after.

**Determinism.** Seed everything through `rand_chacha`. Same seed ⟹ same trajectory, native
and WASM. There is a test asserting this. It matters because the demo shares permalinks.

---

## 3. Conventions that are easy to get wrong

- **Rotation generator convention:** `RX(θ) = exp(-iθX/2)`, likewise RY, RZ. Every gradient
  formula in the codebase assumes this. If you change it, the parameter-shift rule's factor
  of ½ and the adjoint's `-i/2` both change with it.
- **State layout is structure-of-arrays** — `re: Vec<f64>`, `im: Vec<f64>`, never
  `Vec<Complex64>`. SoA autovectorises; AoS does not. Do not "tidy" this into a complex type.
- **Qubit `q` indexes bit `q`** of the basis-state index, little-endian. Amplitude index `i`
  has qubit `q` set iff `i & (1 << q) != 0`.
- **`λ` (input scaling) is a protagonist, not a hyperparameter.** Trainable-vs-pinned is a
  first-class switch at every layer: CLI flag, WASM parameter, UI toggle. Never bury it.
- **No emoji.** Not in the README, not in commit messages, not in the UI, not in code
  comments.

---

## 4. Verification discipline

Every quantitative claim in the README has a test that fails when the claim stops being
true. Do not write a claim you have not measured.

Gradients have two independent paths — adjoint (fast) and parameter-shift (hardware-honest)
— and they must agree to `1e-10` on randomised circuits, for every gate type. A third
check, a dense Kronecker-product reference that shares no code with the strided kernels,
guards the kernels themselves. **Do not proceed past a red gradient test.** Part I §14:
"Nothing renders until the gradients are right."

Report negative results. The dequantization numbers, the entanglement ablation, and the
parameter-count comparison against a classical MLP go in the README whichever way they fall.
Credibility is the scarce resource in this field.

---

## 5. Traps carried forward from the specs

- Do not let the visualiser drive the physics. If a panel wants a quantity that is not
  physically meaningful, cut the panel.
- Do not add environments. Three is already one more than needed.
- Do not oversell. No "quantum advantage" language anywhere.
- Do not use a density matrix for decoherence (Part II) — it destroys the sparse
  representation. Trajectories.
- Do not compute Lie closures with dense matrices (Part III) — bitsets and XOR. A `2ⁿ`
  matrix anywhere in `overtone-lie` is a bug.
- Do not interpolate animation frames. Discrete time is discrete.
