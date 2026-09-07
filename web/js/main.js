// Bootstrap and the render loop.
//
// Physics steps on its own clock; requestAnimationFrame draws whatever is current
// (Part IV 5.4). Frames are never interpolated -- a discrete-time process that glides is a
// lie about the process.

import init, { Closure, Lab, Landing, Plateau, version } from '../pkg/overtone_wasm.js';
import * as panel from './panels.js';
import { landing, lattice } from './closure.js';

const $ = (id) => document.getElementById(id);
const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

let lab = null;
let hero = null;
let sweepData = null;
let running = false;
let lastStep = 0;
let clo = null;
let land = null;

function config() {
  return {
    qubits: +$('qubits').value,
    layers: +$('layers').value,
    k: +$('k').value,
    softmax: $('policy').value === 'softmax',
    trainable: $('lambda').checked,
    entangle: $('entangle').checked,
    seed: +$('seed').value,
  };
}

function build() {
  const c = config();
  if (lab) lab.free();
  lab = new Lab(c.qubits, c.layers, c.k, c.softmax ? 1 : 0, c.trainable, c.entangle, 1.0, 0.05, c.seed);
  if (c.trainable) lab.coarse_tune();
  syncReadouts();
  draw();
}

function syncReadouts() {
  $('layers-value').textContent = $('layers').value;
  $('qubits-value').textContent = $('qubits').value;
  $('k-value').textContent = $('k').value;
  $('r-ceiling').textContent = lab.frequency_ceiling();
  $('r-lp').textContent = lab.lp_ceiling().toFixed(4);
  $('r-return').textContent = lab.exact_return().toFixed(4);
  $('r-episodes').textContent = lab.episodes();
  $('r-lambda').textContent = lab.lambda_trainable() ? lab.lambda().toFixed(3) : 'pinned';
  $('r-leak').textContent = lab.leakage_is_meaningful() ? lab.leakage_ratio(512).toFixed(4) : 'n/a';
}

function draw() {
  if (!lab) return;
  const c = config();
  panel.circuit($('c-circuit'), lab, c.qubits);
  panel.state($('c-state'), lab, 0.9);
  panel.spectrum($('c-spectrum'), lab, 512);
  panel.policy($('c-policy'), lab, 256);
  panel.ret($('c-return'), lab);
  panel.gradcheck($('c-gradcheck'), lab);
}

// The Closure chapter. The closure is computed in one call and replayed frame by frame;
// JavaScript never takes a commutator.
function buildClosure() {
  const n = +$('cl-qubits').value;
  $('cl-qubits-value').textContent = n;
  if (clo) clo.g.free();
  const g = new Closure(+$('cl-family').value, n, 2, 4096);
  clo = {
    g, n, dim: g.dim(), shown: 0, last: 0,
    glyphs: g.glyphs(), parents: Array.from(g.parents()),
    growth: Array.from(g.growth()), generators: g.num_generators(),
  };
  $('cl-dim').textContent = g.dim() + (g.truncated() ? '+ (abandoned)' : '');
  $('cl-su').textContent = g.dim_su().toFixed(0);
  const v = g.predicted_variance();
  $('cl-var').textContent = Number.isNaN(v) ? 'not applicable' : v.toFixed(4);
  $('cl-obs').textContent = g.observable();
  $('cl-verdict').textContent = g.verdict();
}

function closureFrame(t) {
  if (!clo || t - clo.last < 24) return;
  clo.last = t;
  if (clo.shown >= clo.dim && land && land.shown >= land.measured.length) return;
  clo.shown = Math.min(clo.dim, clo.shown + Math.max(1, Math.ceil(clo.dim / 90)));
  lattice($('c-closure'), {
    ...clo,
    growth: clo.growth.filter((v, i) => i === 0 || v <= clo.shown),
  });
  if (land) {
    if (clo.shown >= clo.dim) land.shown = Math.min(land.measured.length, land.shown + 1);
    landing($('c-landing'), land, land.shown);
  }
}

function buildLanding() {
  const l = new Landing(0, 3, 7, 32, 400, 7);
  land = {
    widths: Array.from(l.widths()), predicted: Array.from(l.predicted()),
    measured: Array.from(l.measured()), worst: l.worst_ratio(), shown: 0,
  };
  l.free();
  landing($('c-landing'), land, 0);
}

function loop(t) {
  if (running && t - lastStep > 60) {
    lastStep = t;
    lab.train_steps(2);
    syncReadouts();
    draw();
  }
  if (hero) heroFrame(t);
  closureFrame(t);
  requestAnimationFrame(loop);
}

// The hero (Part I 8.4): a SpectralControl-3 agent with trainable lambda, its spectrum
// beside it, one peak sliding across the frequency axis and locking on. The real engine,
// not a recording. This is the only motion on the page that a human did not ask for.
function heroFrame(t) {
  if (t - hero.last < 70) return;
  hero.last = t;
  if (hero.steps < 260) {
    hero.lab.train_steps(1);
    hero.steps += 1;
  } else if (!reduced) {
    hero.lab.free();
    startHero();
    return;
  }
  panel.spectrum($('c-hero-spectrum'), hero.lab, 512);
  panel.ret($('c-hero-return'), hero.lab);
  $('hero-lambda').textContent = hero.lab.lambda().toFixed(3);
  $('hero-return').textContent = hero.lab.exact_return().toFixed(4);
}

function startHero() {
  // lambda starts below the environment frequency but inside the capture range, so the peak
  // visibly slides up and locks on rather than starting at the answer.
  const h = new Lab(2, 1, 3, 0, true, true, 1.0, 0.06, 7);
  h.coarse_tune();
  hero = { lab: h, steps: 0, last: 0 };
}

async function runSweep() {
  $('plateau-status').textContent = 'sweeping';
  await new Promise((r) => setTimeout(r, 16));
  const p = new Plateau(2, +$('p-max').value, +$('p-depth').value, 2, 200, 7);
  sweepData = {
    qubits: Array.from(p.qubits()),
    local: Array.from(p.local_variance()),
    global: Array.from(p.global_variance()),
    localRate: p.local_rate(),
    globalRate: p.global_rate(),
    localR2: p.local_r_squared(),
    globalR2: p.global_r_squared(),
  };
  p.free();
  $('plateau-status').textContent =
    `local R2 ${sweepData.localR2.toFixed(3)} · global R2 ${sweepData.globalR2.toFixed(3)}`;
  panel.plateau($('c-plateau'), sweepData);
}

function wire() {
  ['qubits', 'layers', 'k', 'policy', 'lambda', 'entangle', 'seed'].forEach((id) =>
    $(id).addEventListener('input', build));
  $('run').addEventListener('click', () => {
    running = !running;
    $('run').textContent = running ? 'pause' : 'run';
  });
  $('reset').addEventListener('click', () => {
    lab.reset();
    if (config().trainable) lab.coarse_tune();
    syncReadouts();
    draw();
  });
  $('p-run').addEventListener('click', runSweep);
  ['cl-family', 'cl-qubits'].forEach((id) => $(id).addEventListener('input', () => {
    buildClosure();
    if (land) land.shown = 0;
  }));
  $('cl-replay').addEventListener('click', () => {
    clo.shown = 0;
    if (land) land.shown = 0;
  });
  $('p-max').addEventListener('input', () => { $('p-max-value').textContent = $('p-max').value; });
  window.addEventListener('resize', () => {
    draw();
    if (sweepData) panel.plateau($('c-plateau'), sweepData);
    if (land) landing($('c-landing'), land, land.shown);
  });
}

init().then(() => {
  $('version').textContent = `overtone-sim ${version()}`;
  $('loading').remove();
  wire();
  build();
  startHero();
  if (reduced) {
    // Show the loop's final frame instead of running it.
    for (let i = 0; i < 260; i++) hero.lab.train_steps(1);
    hero.steps = 260;
  }
  panel.plateau($('c-plateau'), null);
  buildClosure();
  requestAnimationFrame(loop);
  // The measurement is a few hundred random circuits per width; let the page paint first.
  setTimeout(buildLanding, 32);
});
