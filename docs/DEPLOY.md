# Deploying Overtone

**Target: a Hugging Face Static Space.** The choice is forced and it happens to be the right one.

**The Space exists:** [huggingface.co/spaces/Anbu-00001/Overtone](https://huggingface.co/spaces/Anbu-00001/Overtone)
— `sdk: static`, public, and currently serving the Blank template. So `HF_SPACE` is
`Anbu-00001/Overtone`.

Its card was filled in from the creation form and differs from `deploy/README.md` in three
fields — the emoji is a trophy rather than a musical note, the gradient is `gray` to `red`
rather than `indigo` to `gray`, and `short_description` was cut at 60 characters. The first
deploy force-pushes our card over it, so all three correct themselves; there is nothing to fix
by hand. **The 60-character cut is the useful part:** it is the Space form's real limit, the
card's description was 64, and `build_space.sh` now rejects anything longer so the card and the
form can never disagree again.

## Why static, and why there is no backend

Hugging Face's documentation settles the first half:

> Static Spaces are free for everyone. Gradio and Docker Spaces run on compute and require a
> paid plan to create: PRO for personal accounts, Team or Enterprise for organizations.

So on the free tier the only SDK available is `static`. That would be a constraint for most
projects. Here it is not, because **the frontend already is the engine**:

- Part I §5 requires every run to be reproducible from a seed, natively *and* in WASM, and
  `scripts/wasm_determinism.sh` checks that they agree to `1e-13`.
- `scripts/check_js_budget.sh` enforces that **no panel computes a physical quantity** — every
  number a panel draws arrives from the WASM boundary already fitted and already normalised.

Between them there is nothing left for a server to do. Adding one would move logic out of Rust,
which is exactly what the budget exists to prevent.

**The league does not change this.** Decisions-01 Appendix A already designs it as GitHub
Actions running the matches and pushing results to a Hugging Face Dataset — a scheduled job and
a data repository, not a service. `actions/cache` holds the memo table across runs, keyed by
engine version.

## The wasm must be in Git LFS, whatever its size

Measured, on the first real deploy, not reasoned:

```
remote: Your push was rejected because it contains binary files.
remote: Please use https://huggingface.co/docs/hub/xet to store binary files.
remote: Offending files:
remote:   - pkg/overtone_wasm_bg.wasm
```

**The rule is about binary content, not size.** Hugging Face separately requires LFS for files
over 10 MiB, and it is tempting to read that as the only constraint and conclude that a 358 KiB
wasm can be a plain blob. It cannot. The two rules are independent, and being far below the size
threshold grants no exemption from the binary one.

So the bundle ships a `.gitattributes` carrying `*.wasm filter=lfs diff=lfs merge=lfs -text` —
the same rule a new Space is created with. Shipping it rather than relying on the Space's own
copy is what makes the two deploy paths identical: `deploy_space.sh` pushes into a clone that
already has the file, while `space.yml` does a fresh `git init` that would otherwise have no
rules at all.

A rule is not enough on its own. The LFS filters have to be installed in the repository doing
the commit (`git lfs install --local`), or `git add` stores the binary directly and the remote
refuses the push regardless of what `.gitattributes` says. Both deploy paths do this, and
`space.yml` then asserts that the staged wasm really is a pointer before pushing — the failure
otherwise arrives minutes later, from the remote, at the end of a full build.

## Two facts from the docs that constrain the bundle

**Persistent storage is no longer available on Spaces.** Nothing may assume writable disk. The
bundle is read-only and the engine keeps its state in the browser.

**A Space serves files from its repo and rebuilds on every push.** So the bundle must be
self-contained, and `web/pkg/` — gitignored here, because it is build output — has to be *in*
it. `scripts/build_space.sh` assembles it and refuses to finish if anything is missing:

```
$ ./scripts/build_space.sh dist
building wasm
assembling dist
checking the bundle is self-contained
bundle ok: 539 KiB total, 358 KiB wasm
```

The check is not decorative — it resolves every `href` and `src` in the HTML and every relative
import in every JS module against the bundle, and fails on the first one that does not exist.
A Space that builds and then 404s on a module is the failure this prevents, and it is only
visible after a push otherwise.

## Deploying

### Automatically, on every push to `main`

[`.github/workflows/space.yml`](../.github/workflows/space.yml) builds the bundle and pushes it.
It is a **no-op until you configure it**, which is deliberate — a fork or a fresh clone should
not go red on every push for a deployment it was never going to do:

| Where | Name | Value |
|---|---|---|
| Repository **variable** | `HF_SPACE` | `Anbu-00001/Overtone` |
| Repository **secret** | `HF_TOKEN` | a write token for that Space |

`pages.yml` already describes itself as a mirror of this Space, so once both are configured
neither host is a single point of failure and both serve the same bytes.

### Creating the Space

The form at [huggingface.co/new-space](https://huggingface.co/new-space), field by field. Every
value here also appears in `deploy/README.md`, and the first deploy force-pushes that card — so
anything you type that disagrees with it is overwritten rather than merged. Match them.

| Field | Value | Why |
|---|---|---|
| Owner / Space name | `<you>` / `Overtone` | Together they are `HF_SPACE` |
| Short description | `Reads your circuit's algebra and predicts whether it will train.` | Must equal `short_description:` in the card |
| License | `agpl-3.0` — *GNU Affero General Public License v3.0* | The exact Hugging Face identifier. `agpl-3.0-or-later` is the SPDX expression and is **not** in their list |
| Space SDK | **Static** | Gradio and Docker run on compute and need a paid plan |
| Template | **Blank** | The first deploy replaces the tree wholesale |
| Space hardware | **CPU Basic** (free) | A Static Space serves files; there is no process to attach a GPU to, so ZeroGPU buys nothing |
| Storage Bucket | **off** | Persistent storage is no longer available, and `build_space.sh` assumes no writable disk |
| Space Dev Mode | **off** | PRO-only, and CI regenerates the Space on every push, so an SSH edit would be overwritten |
| Visibility | **Public** | Protected and Private need PRO — and AGPL §13 means a hosted instance owes its source to its users anyway |

The emoji and gradient (a musical note, `indigo` to `gray`) are in the card too. `scripts/build_space.sh`
validates all of this against Hugging Face's documented allowed values before the bundle is
allowed to exist, so a misspelled colour or SDK fails locally instead of becoming a
configuration error after the push.

### Manually


The Space repository is a build artefact, not a second source of truth. It holds the bundle and
nothing else; the history that matters is in this repository.

```bash
# once: create the Space (sdk: static) at https://huggingface.co/new-space
#       and authenticate
pip install -U "huggingface_hub[cli]"
hf auth login

# each deploy
./scripts/build_space.sh dist
cd dist
git init -b main
git remote add origin https://huggingface.co/spaces/Anbu-00001/Overtone
git add -A
git commit -m "Overtone $(git -C .. rev-parse --short HEAD)"
git push --force origin main
```

`--force` is correct here and only here: the Space repo is regenerated from the bundle each
time, so its history carries no information that is not already in this repository's.

## What to check after the first deploy

**That the `.wasm` is served as `application/wasm`.** `wasm-bindgen`'s glue calls
`WebAssembly.instantiateStreaming` and falls back to `arrayBuffer` on a MIME mismatch, so the
page works either way — but the fallback logs a console warning and costs a second pass over
358 KiB. If the warning appears, the fix is a `custom_headers` block in the Space's front
matter.

**That the free Space's sleep behaviour is acceptable.** Free hardware Spaces go to sleep after
a period of inactivity. For a static bundle this costs a cold start on the first request after
sleeping and nothing else, since there is no process holding state.

## The explainers

Phase 18's explainer pages are the part of the plan that most needs this deployment, and for a
specific reason: Hello Quantum's companion blog post drew **58% of its traffic from search
engines and 1% from the in-app link**, and its developers concluded the post was the more
successful learning resource. The pages therefore have to be *crawlable*, which means real URLs
served as HTML — not a debrief screen inside a demo, and not Markdown in a repository.

They are written as Markdown in [`docs/explainers/`](explainers/) today, which is where a
contributor reads them. Shipping them means serving them as HTML from the same bundle, and the
JS budget rules out doing the conversion in the browser. That conversion is Phase 18's work and
it is not done yet.
