// The Menagerie: worlds, the transport dial, the race, the sigil, and WFC beside it.
//
// Part IV 3. Nothing here computes a physical quantity -- beta, R^2, the fitted regime, the
// distribution and the race all arrive from Rust already reduced.

import { ACCENT, GRID, INK, INK_DIM } from './color.js';
import { axes, bars, fit, fmt, frame, label, polyline, readout } from './draw.js';

const AGENTS = ['Hadamard', 'Grover', 'optimised', 'classical'];
const AGENT_COLOUR = ['#7fb2d9', '#8a8f98', ACCENT, '#d98a6a'];

// Transport: sigma(t) against t on log-log, with the classical sqrt(t) behind it. A power
// law is a straight line here, which is why the axes are the ones the fit was made on.
export function transport(canvas, men) {
  const { ctx, w, h } = fit(canvas);
  const r = frame(ctx, w, h, 'transport  log sigma vs log t');
  const s = Array.from(men.sigma_trace());
  const c = Array.from(men.classical_trace());
  const lg = (v) => Math.log(Math.max(v, 1e-6));
  const all = [...s, ...c].map(lg);
  const hi = Math.max(...all) + 0.3;
  const lo = Math.min(...all) - 0.3;
  axes(ctx, r);
  polyline(ctx, r, c.map(lg), lo, hi, '#d98a6a', 1);
  polyline(ctx, r, s.map(lg), lo, hi, '#7fb2d9', 1.6);
  label(ctx, 'sqrt(t)', r.x0 + 4, r.y0 + 23, 'left', '#d98a6a');
  label(ctx, `t = ${men.steps()}`, r.x1, r.y1 + 11, 'right');
  const pl = men.is_power_law();
  readout(
    ctx,
    pl ? `beta ${fmt(men.beta(), 3)}  R2 ${fmt(men.r_squared(), 3)}` : `saturated  R2 ${fmt(men.r_squared(), 2)}`,
    r.x1, 3, pl ? INK : '#d98a6a',
  );
}

// The field itself, with the substrate's own letters drawn as a strip beneath it. The strip
// is the world; the field is what the world did.
export function field(canvas, men) {
  const { ctx, w, h } = fit(canvas);
  const r = frame(ctx, w, h, 'amplitude field  P(x)');
  const p = Array.from(men.distribution());
  const letters = Array.from(men.letters());
  const max = Math.max(...p, 1e-12);
  const strip = 7;
  const inner = { ...r, h: r.h - strip - 3, y1: r.y1 - strip - 3 };
  axes(ctx, inner);
  bars(ctx, inner, p, max, () => '#7fb2d9');

  const bw = r.w / letters.length;
  for (let i = 0; i < letters.length; i++) {
    ctx.fillStyle = letters[i] === 0 ? '#232830' : '#495569';
    ctx.fillRect(r.x0 + i * bw, r.y1 - strip, Math.max(1, bw), strip);
  }
  label(ctx, `-${men.steps()}`, r.x0, r.y1 + 11);
  label(ctx, '0', r.x0 + r.w / 2, r.y1 + 11, 'center');
  label(ctx, `${men.steps()}`, r.x1, r.y1 + 11, 'right');
  readout(ctx, men.world_name(), r.x1, 3, INK);
}

// The race. Four agents, one world; the bars are how far each got.
//
// `flat` is passed in rather than fetched, because the optimised agent is found by a search
// over coin angles that costs a thousand walks. Running that inside a resize handler starved
// the hero on the Lab panel for seconds. It is computed once per world.
export function race(canvas, flat) {
  const { ctx, w, h } = fit(canvas);
  const r = frame(ctx, w, h, 'race  sigma at t');
  const vals = [];
  for (let i = 0; i < 4; i++) vals.push(flat[i * 5 + 1]);
  const max = Math.max(...vals, 1e-9);
  axes(ctx, r);
  bars(ctx, r, vals, max, (i) => AGENT_COLOUR[i]);
  const bw = r.w / 4;
  for (let i = 0; i < 4; i++) {
    label(ctx, AGENTS[i], r.x0 + (i + 0.5) * bw, r.y1 + 11, 'center', AGENT_COLOUR[i]);
    if (flat[i * 5 + 2] === 1) {
      label(ctx, 'reads the world', r.x0 + (i + 0.5) * bw, r.y0 + 11, 'center', INK_DIM);
    }
  }
}

// The sigil: a picture of the algebra. Spokes at the support's centre of mass, radius from
// the Pauli weight, colour from the X/Y/Z content.
export function sigil(canvas, closure) {
  const { ctx, w, h } = fit(canvas);
  const r = frame(ctx, w, h, 'sigil');
  const flat = Array.from(closure.sigil());
  const cx = r.x0 + r.w / 2;
  const cy = r.y0 + r.h / 2;
  const rad = Math.min(r.w, r.h) / 2 - 6;
  const rings = closure.sigil_rings();

  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  for (let k = 1; k <= rings; k++) {
    ctx.beginPath();
    ctx.arc(cx, cy, (rad * k) / rings, 0, Math.PI * 2);
    ctx.stroke();
  }
  for (let i = 0; i < flat.length; i += 5) {
    const [a, rr, x, y, z] = flat.slice(i, i + 5);
    const r0 = rad * Math.max(rr - 1 / rings, 0);
    const r1 = rad * rr;
    ctx.strokeStyle = `rgb(${Math.round(90 + 140 * x)},${Math.round(90 + 140 * y)},${Math.round(120 + 120 * z)})`;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(cx + r0 * Math.cos(a), cy + r0 * Math.sin(a));
    ctx.lineTo(cx + r1 * Math.cos(a), cy + r1 * Math.sin(a));
    ctx.stroke();
  }
  readout(ctx, `dim ${flat.length / 5} · ${closure.sigil_blocks()} blocks`, r.x1, 3, INK_DIM);
}

// Wave Function Collapse, and its Shannon entropy. Part IV 7: do not claim this is quantum.
// The whole value of the panel is that it is not.
export function wfc(canvas, w0, tiles) {
  const { ctx, w, h } = fit(canvas);
  const r = frame(ctx, w, h, 'WFC  Shannon entropy of the solver');
  const cells = Array.from(w0.cells());
  const cw = r.w / w0.width();
  const ch = r.h / w0.height();
  for (let y = 0; y < w0.height(); y++) {
    for (let x = 0; x < w0.width(); x++) {
      const v = cells[y * w0.width() + x];
      const px = r.x0 + x * cw;
      const py = r.y0 + y * ch;
      if (v < 0) {
        // Still in superposition: shade by how many options remain.
        const opts = -v - 1;
        ctx.fillStyle = `rgba(127,178,217,${0.05 + 0.05 * opts})`;
        ctx.fillRect(px, py, cw, ch);
        continue;
      }
      const t = tiles.slice(v * 4, v * 4 + 4);
      ctx.strokeStyle = '#7fb2d9';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      if (t[0]) { ctx.moveTo(px, py + ch / 2); ctx.lineTo(px + cw / 2, py + ch / 2); }
      if (t[1]) { ctx.moveTo(px + cw, py + ch / 2); ctx.lineTo(px + cw / 2, py + ch / 2); }
      if (t[2]) { ctx.moveTo(px + cw / 2, py); ctx.lineTo(px + cw / 2, py + ch / 2); }
      if (t[3]) { ctx.moveTo(px + cw / 2, py + ch); ctx.lineTo(px + cw / 2, py + ch / 2); }
      ctx.stroke();
    }
  }
  readout(ctx, `S = ${fmt(w0.total_entropy(), 2)}`, r.x1, 3, INK);
}
