// Colour comes from the physics (Part I 8.2). Do not choose a palette; derive one.
//
// Phase is a cyclic quantity, so phase is hue and the map must be cyclic and perceptually
// uniform. These are matplotlib's `twilight` anchors, sampled at 17 points; the first and
// last are the same colour, so wrapping produces no seam. A linear ramp here would create a
// visible discontinuity where the physics is smooth, and physicists notice.
//
// Amplitude is a magnitude, so amplitude is lightness.

const TWILIGHT = [
  [226, 217, 226], [196, 206, 212], [149, 181, 199], [114, 151, 193],
  [98, 118, 186], [94, 81, 173], [89, 42, 143], [69, 19, 92],
  [47, 20, 54], [74, 19, 66], [116, 30, 79], [152, 53, 80],
  [178, 86, 82], [194, 124, 99], [204, 163, 137], [216, 199, 190],
  [226, 217, 226],
];

// Exactly one non-derived accent, used only for the human's own interactions.
export const ACCENT = '#3e6fd9';
export const INK = '#e8e6e3';
export const INK_DIM = '#8a8f98';
export const GRID = '#252a33';
export const INSTRUMENT_BG = '#12151a';

// A phase in radians to an rgb triple, wrapping cleanly.
function twilightRgb(phase) {
  let t = (phase / (2 * Math.PI)) % 1;
  if (t < 0) t += 1;
  const x = t * (TWILIGHT.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = TWILIGHT[i];
  const b = TWILIGHT[Math.min(i + 1, TWILIGHT.length - 1)];
  return [
    a[0] + (b[0] - a[0]) * f,
    a[1] + (b[1] - a[1]) * f,
    a[2] + (b[2] - a[2]) * f,
  ];
}

// Hue from phase, lightness from amplitude. `mag` is already normalised to [0, 1].
export function phaseColor(phase, mag = 1) {
  const [r, g, b] = twilightRgb(phase);
  const k = 0.25 + 0.75 * Math.min(1, Math.max(0, mag));
  return `rgb(${Math.round(r * k)},${Math.round(g * k)},${Math.round(b * k)})`;
}

// Hue is never the only carrier of information (Part IV 5.5): roughly one man in twelve has
// a colour-vision deficiency. Phase gets a redundant channel as a short tick whose angle
// encodes it, drawn by the caller wherever a phase colour is used at a readable size.
export function phaseTick(ctx, x, y, phase, radius) {
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + radius * Math.cos(-phase), y + radius * Math.sin(-phase));
  ctx.stroke();
}

// The legend for the phase map is literally the complex unit disc.
export function drawPhaseDisc(ctx, cx, cy, r) {
  for (let i = 0; i < 180; i++) {
    const a0 = (i / 180) * 2 * Math.PI;
    const a1 = ((i + 1) / 180) * 2 * Math.PI;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, r, -a1, -a0);
    ctx.closePath();
    ctx.fillStyle = phaseColor(a0, 1);
    ctx.fill();
  }
  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, 2 * Math.PI);
  ctx.stroke();
}
