// Sonification (Part IV 5.1). The project is called Overtone and the policy is a Fourier
// series; mapping its reachable frequencies to audible pitches is the smallest change that
// makes the name mean something.
//
// As lambda trains, the pitch slides. Near the environment's frequency you hear beating,
// and the beat rate falls to zero as it locks on -- audible before it is legible, because
// pitch discrimination beats reading a bar chart.
//
// Part IV 7: never autoplay. Off by default, one toggle, and the context is not even
// created until a human asks for it, since browsers require a gesture anyway.

let ctx = null;
let env = null;
let policy = null;

const BASE = 110; // A2, so a frequency of 3 lands near an audible A4.

export function enabled() {
  return ctx !== null;
}

export function toggle() {
  if (ctx) {
    stop();
    return false;
  }
  ctx = new (window.AudioContext || window.webkitAudioContext)();
  const gain = ctx.createGain();
  gain.gain.value = 0.06;
  gain.connect(ctx.destination);
  const mk = (f) => {
    const o = ctx.createOscillator();
    o.type = 'sine';
    o.frequency.value = f;
    o.connect(gain);
    o.start();
    return o;
  };
  env = mk(BASE);
  policy = mk(BASE);
  return true;
}

export function stop() {
  if (!ctx) return;
  ctx.close();
  ctx = null;
  env = null;
  policy = null;
}

// Two tones: the environment's frequency k, and the policy's reachable frequency lambda*C.
// When they differ you hear the beat; when they coincide the beat stops.
export function update(k, reach) {
  if (!ctx) return;
  const now = ctx.currentTime;
  env.frequency.setTargetAtTime(BASE * k, now, 0.05);
  policy.frequency.setTargetAtTime(BASE * Math.max(reach, 0.2), now, 0.05);
}
