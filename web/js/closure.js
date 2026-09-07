// The Closure chapter: C1, the closure animated, and C2, the prediction landing.
//
// Part III 6 asks for a combinatorial explosion resolving into a finite structure. The
// strings are laid out as a sorted lattice in creation order, each new one connected to the
// pair whose commutator produced it, and the counter freezes when the set stops growing.

import { GRID, INK, INK_DIM, ACCENT } from './color.js';
import { fit, fmt, frame, label, readout } from './draw.js';

const GLYPH = ['#2a2f38', '#7fb2d9', '#d98a6a', '#8fc7a8']; // I X Y Z

export function lattice(canvas, view) {
  const { ctx, w, h } = fit(canvas);
  const r = frame(ctx, w, h, 'dynamical Lie algebra  g');
  const { glyphs, parents, n, dim, shown, growth, generators } = view;
  // Largest glyph that still fits every string in the panel: the lattice should fill the
  // canvas at dim(g) = 12 and stay legible at 500, which is the range Part III 6 asks for.
  let gw = 2;
  let cw = n * gw + 4;
  let ch = 6;
  let cols = Math.max(1, Math.floor(r.w / cw));
  for (let g = 14; g >= 2; g--) {
    const w2 = n * g + 4;
    const h2 = Math.round(g * 1.5) + 3;
    const maxCols = Math.max(1, Math.floor(r.w / w2));
    const maxRows = Math.max(1, Math.floor(r.h / h2));
    if (maxCols * maxRows >= dim) {
      // Use the full height, then only as much width as that needs.
      [gw, cw, ch, cols] = [g, w2, h2, Math.min(maxCols, Math.ceil(dim / maxRows))];
      break;
    }
  }
  const at = (i) => [r.x0 + (i % cols) * cw, r.y0 + Math.floor(i / cols) * ch];

  // Edges first, so the strings sit on top of them.
  ctx.strokeStyle = 'rgba(62,111,217,0.22)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = generators; i < shown; i++) {
    const [x, y] = at(i);
    for (const p of [parents[2 * i], parents[2 * i + 1]]) {
      const [px, py] = at(p);
      ctx.moveTo(px + cw / 2 - 2, py + ch / 2);
      ctx.lineTo(x + cw / 2 - 2, y + ch / 2);
    }
  }
  ctx.stroke();

  for (let i = 0; i < shown; i++) {
    const [x, y] = at(i);
    if (y > r.y1) break;
    for (let q = 0; q < n; q++) {
      const g = glyphs[i * n + q];
      if (g === 0 && gw < 4) continue;
      ctx.fillStyle = GLYPH[g];
      ctx.fillRect(x + q * gw, y + 1, gw - 1, ch - 3);
    }
  }

  // The growth curve. Its plateau is the closure.
  if (growth.length > 1 && r.w > 220) {
    const iw = 74;
    const ih = 34;
    const x0 = r.x1 - iw;
    const y0 = r.y1 - ih - 12;
    const y1 = r.y1 - 12;
    const max = growth[growth.length - 1];
    ctx.fillStyle = 'rgba(18,21,26,0.86)';
    ctx.fillRect(x0 - 4, y0 - 4, iw + 8, ih + 8);
    ctx.strokeStyle = GRID;
    ctx.strokeRect(x0 - 4, y0 - 4, iw + 8, ih + 8);
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    growth.forEach((v, i) => {
      const x = x0 + (i / (growth.length - 1)) * iw;
      const y = y1 - (v / max) * ih;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    label(ctx, '|S| vs round', x0 - 4, y1 + 12, 'left', INK_DIM);
  }

  readout(ctx, `dim(g) = ${shown}${shown < dim ? '' : ' — closed'}`, r.x1, 3,
    shown < dim ? INK_DIM : ACCENT);
}

// C2. The predicted line is drawn first and the measurement lands on it. Part III 13: never
// render them simultaneously, so `shown` walks the measured points in one at a time.
export function landing(canvas, data, shown) {
  const { ctx, w, h } = fit(canvas);
  const r = frame(ctx, w, h, 'Var[loss]  predicted, then measured');
  const { widths, predicted, measured } = data;
  if (!widths.length) return;
  const all = [...predicted, ...measured.slice(0, Math.max(1, shown))].filter((v) => v > 0);
  const hi = Math.log2(Math.max(...all)) + 0.6;
  const lo = Math.log2(Math.min(...all)) - 0.6;
  const px = (i) => r.x0 + (widths.length === 1 ? 0.5 : i / (widths.length - 1)) * r.w;
  const py = (v) => r.y1 - ((Math.log2(Math.max(v, 1e-300)) - lo) / (hi - lo)) * r.h;

  ctx.strokeStyle = ACCENT;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  predicted.forEach((v, i) => (i === 0 ? ctx.moveTo(px(i), py(v)) : ctx.lineTo(px(i), py(v))));
  ctx.stroke();

  ctx.fillStyle = '#7fb2d9';
  for (let i = 0; i < Math.min(shown, measured.length); i++) {
    ctx.beginPath();
    ctx.arc(px(i), py(measured[i]), 3, 0, 2 * Math.PI);
    ctx.fill();
  }

  label(ctx, `n=${widths[0]}`, r.x0, r.y1 + 11);
  label(ctx, `n=${widths[widths.length - 1]}`, r.x1, r.y1 + 11, 'right');
  label(ctx, 'predicted from the algebra', r.x0 + 4, r.y0 + 11, 'left', ACCENT);
  label(ctx, 'measured from random circuits', r.x0 + 4, r.y0 + 23, 'left', '#7fb2d9');
  if (shown >= measured.length) {
    readout(ctx, `worst ${fmt(100 * data.worst, 1)}%`, r.x1, 3);
  }
}
