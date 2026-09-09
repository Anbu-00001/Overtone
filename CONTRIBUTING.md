# Contributing to Overtone

Overtone is AGPL-3.0-or-later. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

Two things are asked of every contribution, and one is still being drafted. Both are explained
rather than asserted, because a contributor who understands why a rule exists can tell when it
is being applied badly.

---

## 1. Sign off your commits (DCO 1.1)

Every commit must carry a `Signed-off-by` line matching the author:

```
Signed-off-by: Jane Doe <jane@example.com>
```

`git commit -s` adds it. By adding it you certify the
[Developer Certificate of Origin 1.1](https://developercertificate.org/), reproduced verbatim
below because it is short and because a rule nobody has read is not a rule.

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

The DCO is a warranty about your authority to submit under the project's licence, plus a
public record tying the contribution to a person. **It does not transfer anything and it does
not permit relicensing.**

---

## 2. A contributor licence — not yet in force, and here is exactly why

**Status: drafted intent only. Nothing is being asked of you today beyond the DCO above.**

This section is published early and unfinished on purpose, because the alternative is
discovering the terms after you have contributed.

The intent is a **short, non-exclusive contributor licence in which you keep your copyright**
— the Apache ICLA / Project Harmony shape, which is a licence and *not* an assignment. What it
would add over the DCO is the ability to sublicense, to make derivative works, and to
distribute contributions **under licences other than the current outbound one**.

### Why this is being considered at all

Because the cost of not having it is known and is being paid by other projects right now.
cBioPortal — AGPL-3.0 scientific software with no prior CLA — is running a contributor-consent
process across **209 contributors** to move to Apache-2.0, and unreachable contributors'
code may have to be rewritten or removed. mpv's GPL-to-LGPL effort is the worked outcome:
silence did not count as consent, and code whose authors could not be reached was removed,
replaced, or segregated. Some GPL-only components remain there to this day.

So "we could never relicense" would be too strong. The accurate statement is: *we would have
to obtain permission from the copyright holders of the material, or remove and rewrite it.*
The problem is not impossibility. It is legal archaeology across a contributor graph, and its
cost grows with every merged pull request and becomes unrecoverable the moment a contributor
becomes unreachable.

The asymmetry is what decides it. Adopting a contributor licence before the first external
contribution costs one signing step. Adopting it after twenty contributors is not a decision
anybody gets to make.

### What is honestly uncertain

- **Whether a CLA costs contributors.** Node.js removed its CLA in 2014 specifically to lower
  the barrier to entry; Cesium reported that large organisations took months to sign while
  individual contributions were unaffected. A 2017 survey of 200 widely used projects found
  67% used no additional condition, 19% individual CLAs, 15% corporate. That is prevalence,
  not causation, and **no controlled evidence quantifies the effect on scientific-software
  contribution.** Anyone who tells you otherwise, including a previous version of this
  document, is stating a causal claim the literature does not support.
- **The drafting.** "You grant us a licence to distribute your contribution" is *not*
  sufficient for the purpose above. The grant has to expressly cover sublicensing, derivative
  works, and use under licences other than the current outbound licence. That is a question for
  a lawyer and not for a repository maintainer, and it is why this section is unfinished:

  > Can we use a short Apache/Harmony-style non-exclusive contributor licence, retaining
  > contributor ownership, whose grant expressly covers future relicensing and dual-licensing,
  > while using DCO 1.1 as the provenance and authority attestation?

**You retain your copyright under every option on the table.** No version of this asks you to
assign anything, and if that ever changes it will change in a commit you can read.

---

## 3. How work is accepted here

These are not style preferences. Each one exists because something went wrong without it, and
`docs/PHASES.md` records which.

**Every claim is paired with a test that fails when the claim stops being true.**
`scripts/gate.sh` is that pairing, and a pull request that adds a claim to a README, a module
doc, or a panel adds a line there too. A claim with no line in the gate is a claim nobody is
checking.

**Negative results are published, not buried.** If a thing you built does not work, the
finding is the contribution. Several of this repository's better sections are failures written
up carefully — a sentence in Part IX that turned out to be false, a correlation that turned
out to be a confound, a feature that measured constant across every position it was supposed to
discriminate.

**Numbers are measured or derived, never chosen.** The operational test is: *a number is
invented if changing it changes an outcome.* Anything that passes that test and is still needed
belongs in the code with its derivation beside it; anything that fails it belongs in
[`presentation.toml`](presentation.toml), which has a test that perturbs every constant in it
and asserts no result moves.

**The dismissals table grows one row per panel, in the panel's own commit.** The README's
table of standard criticisms has a rule: a row appears when the instrument that answers it
ships, in the same commit, and never before. That makes the front page self-maintaining rather
than something that drifts and needs a rewrite before launch. Rows whose panels do not exist
yet are not written as promises; the table says so in a line of its own instead.

**Cite the source, not the summary.** More than one correction in this repository came from
reading a paper's body after its abstract had been trusted. If a citation is load-bearing, the
docstring says which section of it.

**No emoji, anywhere.**

**Determinism.** Seed everything. Native and WASM must agree; `scripts/wasm_determinism.sh`
checks it.

### Before you open a pull request

```
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
./scripts/gate.sh
```

The gate is the whole acceptance criterion. It takes a while, and it is the thing that makes
the rest of this document enforceable rather than aspirational.

### Agent submissions

An agent is a `.toml` file in [`agents/`](agents/) in language v1, and **no submitted code is
ever executed** — the engine implements the search you name. The vocabulary, the four frozen
evaluation features, and the reasons each of them survived measurement are in
`crates/overtone-orbit/src/language.rs`. A submission that names something outside the
vocabulary is rejected with the vocabulary in the error.
