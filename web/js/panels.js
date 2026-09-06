// The seven panels of Part I 8.5. Each takes a canvas and a Lab, reads numbers, and draws.
// No panel computes a physical quantity; if one needed to, that quantity would belong in
// Rust.

import { ACCENT, GRID, INK, INK_DIM, drawPhaseDisc, phaseColor, phaseTick } from './color.js';
import { axes, bars, fit, fmt, frame, label, polyline, readout, scatter, sci, stepline, vrule } from './draw.js';

const GATE_NAMES = ['RX', 'RY', 'RZ', 'H', 'CZ', 'CX'];

// Circuit: gates as a grid, one wire per qubit, parameterised gates tinted by the magnitude
// of their current gradient.
export function circuit(canvas, lab, qubits) {
  const { ctx, w, h } = fit(canvas);
  const r = frame(ctx, w, h, 'circuit');
  const kinds = lab.gate_kinds();
  const qs = lab.gate_qubits();
  const tg = lab.gate_targets();
  const gr = lab.gate_gradient_magnitudes();
  const gmax = Math.max(1e-12, ...gr);
  const cols = Math.max(1, kinds.length);
  const cw = Math.min(24, r.w / cols);
  const rh = r.h / Math.max(1, qubits);

  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  for (let q = 0; q < qubits; q++) {
    const y = r.y0 + (q + 0.5) * rh;
    ctx.beginPath();
    ctx.moveTo(r.x0, y);
    ctx.lineTo(r.x0 + cols * cw, y);
    ctx.stroke();
    label(ctx, `q${q}`, 4, y + 3);
  }

  for (let i = 0; i < kinds.length; i++) {
    const x = r.x0 + i * cw;
    const y = r.y0 + (qs[i] + 0.5) * rh;
    if (tg[i] >= 0) {
      const y2 = r.y0 + (tg[i] + 0.5) * rh;
      ctx.strokeStyle = INK_DIM;
      ctx.beginPath();
      ctx.moveTo(x + cw / 2, y);
      ctx.lineTo(x + cw / 2, y2);
      ctx.stroke();
      ctx.fillStyle = INK_DIM;
      ctx.beginPath();
      ctx.arc(x + cw / 2, y2, 2.5, 0, 2 * Math.PI);
      ctx.fill();
    }
    // Tint by gradient magnitude: a gate the optimiser can feel is brighter.
    const t = gr[i] / gmax;
    ctx.fillStyle = gr[i] > 0 ? `rgba(62,111,217,${0.25 + 0.75 * t})` : '#2a2f38';
    ctx.fillRect(x + 1, y - rh * 0.3, Math.max(3, cw - 3), rh * 0.6);
    if (cw > 13) label(ctx, GATE_NAMES[kinds[i]], x + cw / 2, y + 3, 'center', INK);
  }
}

// State: amplitude bars, hue from phase. The legend is the complex unit disc.
export function state(canvas, lab, s) {
  const { ctx, w, h } = fit(canvas);
  const r = frame(ctx, w, h, 'state  |psi>');
  const a = lab.state_amplitudes(s);
  const n = a.length / 2;
  const mags = new Array(n);
  const phases = new Array(n);
  let max = 0;
  for (let i = 0; i < n; i++) {
    const re = a[2 * i];
    const im = a[2 * i + 1];
    mags[i] = Math.hypot(re, im);
    phases[i] = Math.atan2(im, re);
    if (mags[i] > max) max = mags[i];
  }
  axes(ctx, r);
  bars(ctx, r, mags, max, (i, v) => phaseColor(phases[i], v / (max || 1)));

  // Redundant channel for phase, so hue is not the only carrier.
  if (r.w / n > 6) {
    ctx.strokeStyle = 'rgba(232,230,227,0.55)';
    ctx.lineWidth = 1;
    const bw = r.w / n;
    for (let i = 0; i < n; i++) {
      if (mags[i] < 0.04 * max) continue;
      const x = r.x0 + (i + 0.5) * bw;
      const y = r.y1 - (mags[i] / max) * r.h - 5;
      phaseTick(ctx, x, y, phases[i], 4);
    }
  }
  if (w > 190) drawPhaseDisc(ctx, w - 22, 26, 13);
  readout(ctx, `S = ${fmt(lab.entropy(s), 3)}`, r.x1, r.y1 + 5);
}

// Spectrum: the crown jewel. Bars of |c_omega| with a rule at the reachable ceiling; bars
// beyond it are drawn in a different treatment so the eye catches them immediately.
export function spectrum(canvas, lab, samples) {
  const { ctx, w, h } = fit(canvas);
  const r = frame(ctx, w, h, 'spectrum  |c|');
  const mag = lab.spectrum(samples);
  // With lambda trainable the reachable set is lambda * {-C..C}, so the rule moves with
  // lambda instead of being exceeded. Drawing it at the integer ceiling would show a policy
  // that had tuned into resonance as though it had leaked.
  const ceiling = lab.effective_ceiling();
  const shown = Math.min(mag.length, Math.max(Math.ceil(4 * ceiling) + 6, 26));
  const view = Array.from(mag.slice(0, shown));
  const max = Math.max(...view, 1e-9);

  axes(ctx, r);
  bars(ctx, r, view, max, (i) => (i <= ceiling + 0.5 ? '#7fb2d9' : '#d98a6a'));
  vrule(ctx, r, (ceiling + 1) / shown, ACCENT);

  label(ctx, '0', r.x0, r.y1 + 11);
  label(ctx, ceiling.toFixed(ceiling % 1 ? 2 : 0),
    r.x0 + ((ceiling + 0.5) / shown) * r.w, r.y1 + 11, 'center', ACCENT);
  label(ctx, `${shown - 1}`, r.x1, r.y1 + 11, 'right');
  if (lab.leakage_is_meaningful()) {
    const leak = lab.leakage_ratio(samples);
    readout(ctx, `leak ${leak < 1e-9 ? '0' : fmt(leak, 3)}`, r.x1, 3, leak < 1e-9 ? INK_DIM : '#d98a6a');
  } else {
    readout(ctx, 'reach lambda-tuned', r.x1, 3, INK_DIM);
  }
}

// Policy: pi(1|s) over the observation axis, with the optimal policy ghosted behind.
export function policy(canvas, lab, samples) {
  const { ctx, w, h } = fit(canvas);
  const r = frame(ctx, w, h, 'policy  pi(1|s)');
  axes(ctx, r);
  polyline(ctx, r, Array.from(lab.optimal_curve(samples)), 0, 1, '#333a45', 1);
  polyline(ctx, r, Array.from(lab.policy_curve(samples)), 0, 1, '#7fb2d9', 1.5);
  label(ctx, '-pi', r.x0, r.y1 + 11);
  label(ctx, 'pi', r.x1, r.y1 + 11, 'right');
  label(ctx, '1', r.x0 - 4, r.y0 + 4, 'right');
  label(ctx, '0', r.x0 - 4, r.y1, 'right');
}

// Return: the learning curve, with the LP ceiling staircase drawn behind it.
export function ret(canvas, lab) {
  const { ctx, w, h } = fit(canvas);
  const r = frame(ctx, w, h, 'return  J');
  const trace = Array.from(lab.return_trace());
  const cap = lab.lp_ceiling();
  const opt = lab.optimal_return();
  const hi = Math.max(opt * 1.08, 0.05);
  const lo = Math.min(0, ...trace) - 0.02;

  axes(ctx, r);
  // The unconstrained optimum, and the ceiling this circuit is actually held to.
  stepline(ctx, r, [opt, opt], lo, hi, '#333a45', [2, 3]);
  stepline(ctx, r, [cap, cap], lo, hi, ACCENT, [4, 3]);
  polyline(ctx, r, trace, lo, hi, '#7fb2d9', 1.5);

  label(ctx, `2/pi`, r.x1 - 2, r.y1 - ((opt - lo) / (hi - lo)) * r.h - 3, 'right', '#5a616c');
  label(ctx, `J*`, r.x1 - 2, r.y1 - ((cap - lo) / (hi - lo)) * r.h - 3, 'right', ACCENT);
  readout(ctx, `${fmt(lab.exact_return())}`, r.x1, 3);
}

// Plateau: gradient variance against qubit count, log-y, global and local.
export function plateau(canvas, sweepData) {
  const { ctx, w, h } = fit(canvas);
  const r = frame(ctx, w, h, 'gradient variance  log2 Var');
  if (!sweepData) {
    label(ctx, 'sweeping...', r.x0 + 6, r.y0 + 14, 'left', INK_DIM);
    return;
  }
  const { qubits, local, global } = sweepData;
  const all = [...local, ...global].filter((v) => v > 0).map((v) => Math.log2(v));
  const hi = Math.max(...all) + 0.5;
  const lo = Math.min(...all) - 0.5;
  axes(ctx, r);

  const line = (vals, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    vals.forEach((v, i) => {
      const x = r.x0 + (i / (vals.length - 1)) * r.w;
      const y = r.y1 - ((Math.log2(Math.max(v, 1e-300)) - lo) / (hi - lo)) * r.h;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  };
  line(local, '#7fb2d9');
  line(global, '#d98a6a');

  label(ctx, `n=${qubits[0]}`, r.x0, r.y1 + 11);
  label(ctx, `n=${qubits[qubits.length - 1]}`, r.x1, r.y1 + 11, 'right');
  label(ctx, `local  2^(-${fmt(sweepData.localRate, 2)}n)`, r.x0 + 4, r.y0 + 11, 'left', '#7fb2d9');
  label(ctx, `global 2^(-${fmt(sweepData.globalRate, 2)}n)`, r.x0 + 4, r.y0 + 23, 'left', '#d98a6a');
}

// Gradient check: adjoint against parameter-shift, which must land on y = x.
export function gradcheck(canvas, lab) {
  const { ctx, w, h } = fit(canvas);
  const r = frame(ctx, w, h, 'adjoint vs parameter-shift');
  const flat = lab.gradient_scatter();
  const xs = [];
  const ys = [];
  for (let i = 0; i < flat.length; i += 2) {
    xs.push(flat[i]);
    ys.push(flat[i + 1]);
  }
  const m = Math.max(1e-6, ...xs.map(Math.abs), ...ys.map(Math.abs)) * 1.15;
  axes(ctx, r);
  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(r.x0, r.y1);
  ctx.lineTo(r.x1, r.y0);
  ctx.stroke();
  scatter(ctx, r, xs, ys, -m, m, '#7fb2d9');
  readout(ctx, `max diff ${sci(lab.gradient_agreement())}`, r.x1, 3);
}
