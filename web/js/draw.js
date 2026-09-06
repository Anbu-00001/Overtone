// Canvas primitives. Nothing here knows any physics; it draws numbers it is handed.

import { ACCENT, GRID, INK, INK_DIM, INSTRUMENT_BG } from './color.js';

export const MONO = '11px "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace';
export const MONO_SM = '10px "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace';

// Size a canvas to its CSS box at device resolution, and return a ready context.
export function fit(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = INSTRUMENT_BG;
  ctx.fillRect(0, 0, w, h);
  return { ctx, w, h };
}

// A plot frame with a title and inset margins.
export function frame(ctx, w, h, title, pad = { l: 34, r: 8, t: 18, b: 20 }) {
  ctx.fillStyle = INK_DIM;
  ctx.font = MONO;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText(title, pad.l, 3);
  return { x0: pad.l, y0: pad.t, x1: w - pad.r, y1: h - pad.b, w: w - pad.l - pad.r, h: h - pad.t - pad.b };
}

export function axes(ctx, r) {
  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(r.x0, r.y0);
  ctx.lineTo(r.x0, r.y1);
  ctx.lineTo(r.x1, r.y1);
  ctx.stroke();
}

export function label(ctx, text, x, y, align = 'left', color = INK_DIM, font = MONO_SM) {
  ctx.fillStyle = color;
  ctx.font = font;
  ctx.textAlign = align;
  ctx.textBaseline = 'alphabetic';
  ctx.fillText(text, x, y);
}

// A readout: numbers use tabular figures so digits do not jitter as they update.
export function readout(ctx, text, x, y, color = INK) {
  ctx.fillStyle = color;
  ctx.font = MONO;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'top';
  ctx.fillText(text, x, y);
}

export function polyline(ctx, r, values, lo, hi, color, width = 1.5) {
  if (values.length < 2) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  for (let i = 0; i < values.length; i++) {
    const x = r.x0 + (i / (values.length - 1)) * r.w;
    const y = r.y1 - ((values[i] - lo) / (hi - lo)) * r.h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

// A step function, for a staircase that must not be interpolated between its steps.
export function stepline(ctx, r, values, lo, hi, color, dash = []) {
  if (!values.length) return;
  ctx.save();
  ctx.setLineDash(dash);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < values.length; i++) {
    const xa = r.x0 + (i / values.length) * r.w;
    const xb = r.x0 + ((i + 1) / values.length) * r.w;
    const y = r.y1 - ((values[i] - lo) / (hi - lo)) * r.h;
    ctx.moveTo(xa, y);
    ctx.lineTo(xb, y);
  }
  ctx.stroke();
  ctx.restore();
}

export function vrule(ctx, r, t, color = ACCENT, dash = [3, 3]) {
  ctx.save();
  ctx.setLineDash(dash);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  const x = r.x0 + t * r.w;
  ctx.beginPath();
  ctx.moveTo(x, r.y0);
  ctx.lineTo(x, r.y1);
  ctx.stroke();
  ctx.restore();
}

export function bars(ctx, r, values, max, colorFor) {
  const n = values.length;
  const bw = Math.max(1, (r.w / n) - 1);
  for (let i = 0; i < n; i++) {
    const x = r.x0 + (i / n) * r.w;
    const hgt = max > 0 ? (values[i] / max) * r.h : 0;
    ctx.fillStyle = colorFor(i, values[i]);
    ctx.fillRect(x, r.y1 - hgt, bw, hgt);
  }
}

export function scatter(ctx, r, xs, ys, lo, hi, color, radius = 2) {
  ctx.fillStyle = color;
  for (let i = 0; i < xs.length; i++) {
    const x = r.x0 + ((xs[i] - lo) / (hi - lo)) * r.w;
    const y = r.y1 - ((ys[i] - lo) / (hi - lo)) * r.h;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fill();
  }
}

export function fmt(x, places = 4) {
  if (!isFinite(x)) return '--';
  return x.toFixed(places);
}

export function sci(x) {
  if (!isFinite(x) || x === 0) return '0';
  return x.toExponential(1);
}
