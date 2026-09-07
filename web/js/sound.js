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
  env = mk(220);
  policy = mk(220);
  return true;
}

export function stop() {
  if (!ctx) return;
  ctx.close();
  ctx = null;
  env = null;
  policy = null;
}

// Two tones: the environment's frequency, and the policy's input scaling. When they differ
// you hear the beat; when they coincide the beat stops. Both frequencies come from Rust --
// the mapping carries a claim about what a listener hears, and a claim in the renderer is a
// claim nothing can test.
export function update(tones) {
  if (!ctx) return;
  const now = ctx.currentTime;
  env.frequency.setTargetAtTime(tones[0], now, 0.05);
  policy.frequency.setTargetAtTime(tones[1], now, 0.05);
}
