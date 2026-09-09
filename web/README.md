---
title: Overtone
colorFrom: indigo
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: agpl-3.0
---

# Overtone

A quantum policy is, exactly and provably, a truncated Fourier series in its input. This
page measures what that implies, live, in your browser.

Everything on the page is computed here by the same Rust engine the test suite runs
against, compiled to WebAssembly. There are no pre-recorded traces and no cached results.

Source and the written explanation: https://github.com/Anbu-00001/Overtone

Note on the Space card: Hugging Face's metadata block accepts an `emoji:` field, which is
deliberately omitted. Part I of the build specification rules out emoji everywhere in this
project, and the Space card is not an exception worth making.
