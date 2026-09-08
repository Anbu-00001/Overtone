// The Lattice: proto-value functions, the two exponents, and policies that add.
//
// Part V 1.3 and 2.3. Every field drawn here arrives from Rust already normalised to
// [-1, 1] or [0, 1]; this file decides where pixels go and nothing else.

import { ACCENT, GRID, INK, INK_DIM, INSTRUMENT_BG } from './color.js';
import { fit, fmt, frame, label, sci } from './draw.js';

// A signed field is drawn on one diverging ramp so that the sign is visible without a
// legend: cold below zero, warm above, and the maze's own background at zero.
function signedColour(v) {
  const t = Math.max(-1, Math.min(1, v));
  if (t >= 0) return `rgb(${Math.round(18 + 200 * t)},${Math.round(21 + 120 * t)},${Math.round(26 + 60 * t)})`;
  return `rgb(${Math.round(18 - 8 * t)},${Math.round(21 + 90 * -t)},${Math.round(26 + 190 * -t)})`;
}

function cells(lat) {
  const c = Array.from(lat.coords());
  const out = [];
  for (let i = 0; i < c.length; i += 2) out.push([c[i], c[i + 1]]);
  return out;
}

// One maze, one field over its vertices.
function field(canvas, lat, geometry, values, title, note) {
  const { ctx, w, h } = fit(canvas);
  const r = frame(ctx, w, h, title);
  const { coords, gw, gh } = geometry;
  const s = Math.min((r.x1 - r.x0) / gw, (r.y1 - r.y0) / gh);
  const ox = r.x0 + ((r.x1 - r.x0) - s * gw) / 2;
  const oy = r.y0 + ((r.y1 - r.y0) - s * gh) / 2;

  ctx.fillStyle = INSTRUMENT_BG;
  ctx.fillRect(r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0);
  const edges = geometry.edges;
  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < edges.length; i += 2) {
    const a = coords[edges[i]];
    const b = coords[edges[i + 1]];
    ctx.moveTo(ox + (a[0] + 0.5) * s, oy + (a[1] + 0.5) * s);
    ctx.lineTo(ox + (b[0] + 0.5) * s, oy + (b[1] + 0.5) * s);
  }
  ctx.stroke();

  for (let i = 0; i < coords.length; i++) {
    ctx.fillStyle = signedColour(values[i]);
    ctx.beginPath();
    ctx.arc(ox + (coords[i][0] + 0.5) * s, oy + (coords[i][1] + 0.5) * s, s * 0.34, 0, 2 * Math.PI);
    ctx.fill();
  }
  const mark = (v, colour, text) => {
    ctx.strokeStyle = colour;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(ox + (coords[v][0] + 0.5) * s, oy + (coords[v][1] + 0.5) * s, s * 0.5, 0, 2 * Math.PI);
    ctx.stroke();
    label(ctx, text, ox + (coords[v][0] + 0.5) * s + s * 0.6, oy + (coords[v][1] + 0.5) * s + 3, 'left', colour);
  };
  mark(geometry.source, INK, 'start');
  if (geometry.showExits) {
    mark(geometry.exitA, ACCENT, 'A');
    mark(geometry.exitB, '#d98a6a', 'B');
  }
  if (note) label(ctx, note, r.x0 + 2, r.y1 - 4, 'left', INK_DIM);
}

export function geometryOf(lat) {
  return {
    coords: cells(lat),
    edges: Array.from(lat.edges()),
    gw: lat.grid_width(),
    gh: lat.grid_height(),
    source: lat.source(),
    exitA: lat.exit_a(),
    exitB: lat.exit_b(),
    showExits: false,
  };
}

export function drawAll(lat, geometry, state) {
  const times = Array.from(lat.time_grid(121));
  const t = times[Math.max(0, Math.min(times.length - 1, state.timeIndex))];
  const eigenvalue = Array.from(lat.eigenvalues())[state.mode];

  field(
    document.getElementById('c-mode'),
    lat,
    geometry,
    Array.from(lat.mode(state.mode)),
    `proto-value function ${state.mode}`,
    `lambda = ${fmt(eigenvalue, 5)}`,
  );

  const quantum = state.exponent === 1;
  field(
    document.getElementById('c-walk'),
    lat,
    geometry,
    Array.from(quantum ? lat.interfere(t) : lat.diffuse(t)),
    quantum ? 'e^(-iLt)  interference' : 'e^(-Lt)  diffusion',
    `t = ${fmt(t, 2)}`,
  );

  const alpha = state.alpha;
  field(
    document.getElementById('c-compose'),
    lat,
    { ...geometry, showExits: true },
    Array.from(lat.composed(alpha, 1 - alpha)),
    `z = ${fmt(alpha, 2)} z_A + ${fmt(1 - alpha, 2)} z_B`,
    'added, not solved',
  );

  const spread = Array.from(lat.spread(t));
  document.getElementById('lt-eigenvalue').textContent = fmt(eigenvalue, 5);
  document.getElementById('lt-classical').textContent = fmt(spread[0], 3);
  document.getElementById('lt-quantum').textContent = fmt(spread[1], 3);
  document.getElementById('lt-error').textContent = sci(lat.composition_error(alpha, 1 - alpha));
  document.getElementById('lt-time-value').textContent = fmt(t, 2);
  document.getElementById('lt-mode-value').textContent = state.mode;
  document.getElementById('lt-alpha-value').textContent = fmt(alpha, 2);
  document.getElementById('lt-note').textContent =
    `${lat.order()} vertices, diameter ${lat.diameter()}, regular ${lat.is_regular()}` +
    ` -- both exponents use L, because they differ on an irregular graph`;
}
