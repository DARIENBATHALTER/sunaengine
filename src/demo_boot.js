// SunaEngine — thin demo harness over the real SunaBox (aether) engine.
// This file is ORIGINAL to the sunaengine repo (engine files above it are
// extracted unmodified from aether commit e41e06f — see their headers).
// Copyright (C) 2026 Darien Bathalter
// SPDX-License-Identifier: AGPL-3.0-only — see LICENSE. NO WARRANTY.
//
// ============================================================================
// What this is: the ONLY glue the demo pages are allowed to need. It boots the
// real engine (src/engine.js — the same 7k-line module the SunaBox toy runs)
// on a WATER-ONLY scene, steps it deterministically, and exposes exactly the
// surface the two demos require:
//
//   * state digest + per-substep hash chain      (Demo 1's live equality proof)
//   * particle counts
//   * injectWater(schedule)                       — a substep-indexed injection
//     schedule, so two sims fed the same schedule stay bit-exact (§26.C shape)
//   * .sunar record / replay hooks                (Demo 2's record→replay gate)
//   * snapshot()/restore()                        — keyframes for scrubbing
//
// There is no game UI here and no rendering. Determinism rules are inherited,
// not reimplemented: nothing in this file reads a clock, uses Math.random, or
// derives a substep count from elapsed time. All injection happens at substep
// boundaries identified by INTEGER substep indices.
// ============================================================================

import {
  createEngine, makeScene, digestState,
  MAT_WATER, ONE, PARTICLE_WORDS, HASHED_WORDS, FLAG_MASK, SPEC_VERSION,
} from './engine.js';
import { encodeSunar, decodeSunar } from './sunar.js';

/** Hard cap for every website demo sim (brief: max 3000 particles). The cap is
 *  structural: it is passed as the engine's maxParticles, and spawnFluid
 *  refuses (deterministically) to grow past it. */
export const DEMO_PARTICLE_CAP = 3000;
export const DEMO_SUBMIT_BATCH = 25;

const int = (x, label) => {
  if (!Number.isInteger(x)) throw new Error(`[demo_boot] ${label} must be an integer, got ${x}`);
  return x;
};

/** World-unit helper: wu(3.5) -> Q16.16 ticks. Demo pages should build every
 *  injection coordinate through this so schedules stay integer-exact. */
export const wu = (x) => Math.round(x * ONE) | 0;

/** A small square droplet of water particles centred at (cx, cy) ticks, at
 *  1 wu pitch. Deterministic layout — the identical call is the identical
 *  particle list, which is what lets two sims share one schedule. */
export function makeDroplet(cx, cy, w = 3, h = 3, vx = 0, vy = 0) {
  int(cx, 'cx'); int(cy, 'cy');
  const out = [];
  const x0 = cx - ((w - 1) * ONE >> 1), y0 = cy - ((h - 1) * ONE >> 1);
  for (let gy = 0; gy < h; gy++) {
    for (let gx = 0; gx < w; gx++) {
      out.push({ x: (x0 + gx * ONE) | 0, y: (y0 + gy * ONE) | 0, vx: vx | 0, vy: vy | 0 });
    }
  }
  return out;
}

/**
 * Boot a water-only demo sim on the real engine.
 *
 * @param {object} opts
 *   canvas        — optional; null boots headless (the extraction gate's mode)
 *   scene         — 'pool' (default), 'empty', or a raw engine scene object
 *   poolW, poolH  — pool_v1 dimensions in wu (default 60 x 20 = 1200 fluid)
 *   maxParticles  — defaults to (and may not exceed) DEMO_PARTICLE_CAP
 */
export async function bootWaterDemo(opts = {}) {
  const {
    canvas = null,
    scene = 'pool',
    poolW = 60, poolH = 20,
    maxParticles = DEMO_PARTICLE_CAP,
    submitBatch = DEMO_SUBMIT_BATCH,
    chain = true,
  } = opts;
  if (maxParticles > DEMO_PARTICLE_CAP) {
    throw new Error(`[demo_boot] maxParticles ${maxParticles} exceeds the demo cap ${DEMO_PARTICLE_CAP}`);
  }
  const sc = (typeof scene === 'object' && scene !== null) ? scene
    : scene === 'empty' ? makeScene('empty_v1')
    : scene === 'pool' ? makeScene('pool_v1', { w: poolW, h: poolH, mat: MAT_WATER })
    : makeScene(scene, opts.sceneOpts ?? {});
  if (sc.n > maxParticles) {
    throw new Error(`[demo_boot] scene has ${sc.n} particles; the demo cap is ${maxParticles}`);
  }
  for (let i = 0; i < sc.n; i++) {
    if ((sc.matId[i] >>> 0) !== MAT_WATER) {
      throw new Error('[demo_boot] water-only harness: scene contains a non-WATER particle');
    }
  }
  const eng = await createEngine({ canvas, maxParticles, chain, submitBatch });
  eng.loadScene(sc);
  return new WaterDemo(eng, maxParticles);
}

/** Rebuild a demo sim from a `.sunar` replay (text or bytes) produced by
 *  WaterDemo.stopRecording(). The returned demo starts at the recording's
 *  initial state with the recorded injection schedule pre-armed; stepping it
 *  `doc.durationSubsteps` substeps reproduces the recorded run bit-for-bit. */
export async function replayFromSunar(source, opts = {}) {
  const doc = decodeSunar(source);
  const cap = doc.app?.demoParticleCap ?? DEMO_PARTICLE_CAP;
  const eng = await createEngine({
    canvas: opts.canvas ?? null,
    maxParticles: cap,
    chain: opts.chain ?? true,
    submitBatch: opts.submitBatch ?? DEMO_SUBMIT_BATCH,
  });
  const demo = new WaterDemo(eng, cap);
  demo._loadPayload(doc);
  // Recorded frames are substep offsets from the recording start (frame ==
  // substep: the recorder writes substepsPerFrame = 1).
  for (const e of doc.events) {
    if (e.type === 'inject') demo.injectWater({ substep: e.frame, particles: e.particles });
    else if (e.type === 'push') {
      demo.pushAt({ substep: e.frame, x: e.x, y: e.y, r: e.r, ix: e.ix, iy: e.iy });
    }
  }
  demo.replayDoc = doc;
  return demo;
}

class WaterDemo {
  constructor(engine, cap) {
    this.engine = engine;
    this.cap = cap;
    this.substep = 0;             // absolute substep index since (re)load
    this._schedule = new Map();   // substep -> [{x,y,vx,vy}, ...]
    this._pushes = new Map();     // substep -> [{x,y,r,ix,iy}, ...]
    this.pushHits = 0;            // total particles moved by pushes (gate read)
    this._rec = null;
  }

  get n() { return this.engine.n; }
  get nFluid() { return this.engine.nFluid; }

  /** 64-hex-char digest of the full particle state buffer (8 words/particle). */
  async digest() { return (await this.engine.digest()).hex; }
  /** Per-substep folded hash chain (resets on load/restore; digest is the
   *  cross-run comparator, the chain is the per-run comparator). */
  async chainHex() { return this.engine.chainHex(); }
  async debugCounters() { return [...(await this.engine.debugCounters())]; }
  async checkErrors() { return this.engine.checkErrors(); }
  destroy() { this.engine.destroy(); }

  /**
   * Arm a deterministic injection schedule. Accepts one event or an array of
   * events: { substep, particles: [{x, y, vx?, vy?}, ...] } with `substep` an
   * ABSOLUTE substep index >= the current one (ticks are Q16.16 integers —
   * build them with wu()/makeDroplet()). The particles are spawned as WATER
   * immediately BEFORE that substep advances. Feeding two sims the same
   * schedule keeps them bit-exact; that is Demo 1's whole mechanism.
   */
  injectWater(scheduleOrEvent) {
    const list = Array.isArray(scheduleOrEvent) ? scheduleOrEvent : [scheduleOrEvent];
    for (const ev of list) {
      const s = int(ev.substep, 'event.substep');
      if (s < this.substep) {
        throw new Error(`[demo_boot] injection at substep ${s} is in the past (now ${this.substep})`);
      }
      const ps = (ev.particles ?? []).map((p) => ({
        x: int(p.x, 'particle.x'), y: int(p.y, 'particle.y'),
        vx: (p.vx ?? 0) | 0, vy: (p.vy ?? 0) | 0,
      }));
      if (!ps.length) continue;
      const bucket = this._schedule.get(s);
      if (bucket) bucket.push(...ps); else this._schedule.set(s, ps.slice());
      if (this._rec) {
        this._rec.events.push({
          frame: s - this._rec.startSubstep,
          order: this._rec.order++,
          type: 'inject',
          particles: ps,
        });
      }
    }
  }

  /**
   * Arm a deterministic cursor push: at `substep`, every fluid particle within
   * `r` of (x, y) gets (ix, iy) added to its velocity (VMAX-clamped by the
   * engine). Applied from a mirror synced AT that exact substep boundary, so
   * two lockstep sims fed the same push select identical particle sets and
   * stay bit-exact — the same contract as injectWater, extended to touch.
   */
  pushAt(ev) {
    const s = int(ev.substep, 'push.substep');
    if (s < this.substep) {
      throw new Error(`[demo_boot] push at substep ${s} is in the past (now ${this.substep})`);
    }
    const p = { x: int(ev.x, 'push.x'), y: int(ev.y, 'push.y'), r: int(ev.r, 'push.r'),
                ix: (ev.ix ?? 0) | 0, iy: (ev.iy ?? 0) | 0 };
    if (!p.ix && !p.iy) return;
    const bucket = this._pushes.get(s);
    if (bucket) bucket.push(p); else this._pushes.set(s, [p]);
    if (this._rec) {
      this._rec.events.push({ frame: s - this._rec.startSubstep,
                              order: this._rec.order++, type: 'push', ...p });
    }
  }

  /** Advance exactly `count` substeps, applying due injections at their exact
   *  substep boundaries. Chunked between events so submit batching stays hot.
   *  Pushes need a GPU sync, so this synchronous path REFUSES them — silently
   *  skipping one would be a determinism bug wearing a convenience. */
  step(count = 1) {
    const end = this.substep + int(count, 'count');
    for (const s of this._pushes.keys()) {
      if (s >= this.substep && s < end) {
        throw new Error(`[demo_boot] push armed at substep ${s}: use stepThrough()`);
      }
    }
    return this._stepInjectOnly(count);
  }

  _stepInjectOnly(count) {
    let remaining = int(count, 'count');
    while (remaining > 0) {
      const due = this._schedule.get(this.substep);
      if (due) {
        this._schedule.delete(this.substep);
        // spawnFluid refuses past maxParticles — the cap, enforced by the engine.
        this.engine.spawnFluid(due.map((p) => ({ ...p, mat: MAT_WATER })), { noWeld: true });
      }
      // run uninterrupted until the next scheduled substep (or the end)
      let chunk = remaining;
      for (const s of this._schedule.keys()) {
        if (s > this.substep && s - this.substep < chunk) chunk = s - this.substep;
      }
      this.engine.step(chunk);
      this.substep += chunk;
      remaining -= chunk;
    }
    return this.substep;
  }

  /** step(), but push-capable: at a substep with armed pushes the mirror is
   *  synced on the exact boundary (queue drained -> mirror IS the state) and
   *  the impulses applied in arming order. Deterministic per (schedule, state). */
  async stepThrough(count = 1) {
    let remaining = int(count, 'count');
    while (remaining > 0) {
      const duePush = this._pushes.get(this.substep);
      if (duePush) {
        this._pushes.delete(this.substep);
        await this.engine.syncMirror();
        for (const p of duePush) {
          this.pushHits += this.engine.applyImpulse(p.x, p.y, p.r, p.ix, p.iy);
        }
      }
      let chunk = remaining;
      for (const s of this._pushes.keys()) {
        if (s > this.substep && s - this.substep < chunk) chunk = s - this.substep;
      }
      // _stepInjectOnly stops early at its own scheduled substeps; re-check
      // push boundaries after it returns rather than pre-slicing both maps.
      const target = this.substep + chunk;
      while (this.substep < target) {
        let inj = target - this.substep;
        for (const s of this._schedule.keys()) {
          if (s > this.substep && s - this.substep < inj) inj = s - this.substep;
        }
        this._stepInjectOnly(inj);
      }
      remaining -= chunk;
    }
    return this.substep;
  }

  // ---- snapshots (Demo 2's scrub keyframes) --------------------------------

  /**
   * Exact keyframe. Captures the hashed particle state AND `buf.derived` —
   * the engine's cross-substep scratch (XSPH/normal warm data): two worlds
   * with identical state digests but different derived contents evolve
   * differently, so a scrub keyframe restored without it would not rejoin its
   * own timeline. Verified: state-only restore diverges on the next substep;
   * state+derived restore continues bit-exactly. `derived` is in-memory only —
   * a serialized `.sunar` initial is a load boundary instead (see
   * startRecording), because `.suna` v1 deliberately does not carry scratch.
   */
  async snapshot({ exact = true } = {}) {
    const eng = this.engine;
    const state = await eng.readState();
    const tints = await eng.readTints(eng.n);
    return {
      app: { name: 'SunaEngine demo', specVersion: SPEC_VERSION, demoParticleCap: this.cap },
      scene: {
        n: eng.n, nFluid: eng.nFluid, domW: eng.domW, domH: eng.domH,
        cellShift: eng.cellShift, wallsOn: eng.wallsOn,
        substepCount: eng.substepCount, formationClock: eng.formationClock,
      },
      state, tints,
      derived: exact ? await eng.readDerived() : null,
      substep: this.substep,
    };
  }

  /** Restore a snapshot() (or decoded .sunar initial) into this engine.
   *  With `derived` present the continuation is bit-exact against the timeline
   *  the snapshot was taken from; without it, this is a load boundary (still
   *  fully deterministic from here on). Clears any armed schedule; the chain
   *  restarts (digest comparisons hold across runs, the chain within one). */
  restore(snap) {
    this._loadPayload(snap);
    if (snap.derived) {
      const eng = this.engine;
      // loadScene reconstructed the same state words (water-only: no flag bits
      // outside FLAG_MASK exist); rewrite both parity copies with the captured
      // words and put the scratch back, exactly like the timeline never broke.
      eng.device.queue.writeBuffer(eng.buf.stateA, 0, snap.state);
      eng.device.queue.writeBuffer(eng.buf.stateB, 0, snap.state);
      eng.device.queue.writeBuffer(eng.buf.derived, 0, snap.derived);
    }
    this.substep = snap.substep ?? 0;
  }

  _loadPayload(p) {
    const s = p.scene ?? p.initial?.scene;
    const state = p.state ?? p.initial?.state;
    const tints = p.tints ?? p.initial?.tints;
    if (!s || !state) throw new Error('[demo_boot] payload has no scene/state');
    const n = s.n | 0;
    const su = new Uint32Array(state.buffer, state.byteOffset, state.length);
    const pos = new Int32Array(n * 2), vel = new Int32Array(n * 2);
    const matId = new Uint32Array(n), flags = new Uint32Array(n);
    const temp = new Int32Array(n), fuel = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const o = i * PARTICLE_WORDS;
      pos[i * 2] = state[o]; pos[i * 2 + 1] = state[o + 1];
      vel[i * 2] = state[o + 2]; vel[i * 2 + 1] = state[o + 3];
      matId[i] = su[o + 4]; flags[i] = su[o + 5] & FLAG_MASK;
      temp[i] = state[o + 6]; fuel[i] = state[o + 7];
    }
    this.engine.loadScene({
      n, nFluid: s.nFluid, pos, vel, matId, flags, temp, fuel,
      domW: s.domW, domH: s.domH, cellShift: s.cellShift,
    });
    if (s.wallsOn === false) this.engine.setWalls(false);
    if (tints) this.engine.setTints(0, tints);
    this._schedule.clear();
    this._pushes.clear();
    this._rec = null;
    this.substep = 0;
  }

  // ---- .sunar record / replay (Demo 2) -------------------------------------

  /**
   * Begin recording: captures the current state as the replay's initial world
   * and journals every injectWater() from now on.
   *
   * THE BOUNDARY IS NORMALIZED, DELIBERATELY. A `.sunar` initial cannot carry
   * the engine's derived scratch (`.suna` v1 has no field for it), so a replay
   * always starts from a clean load boundary. If the live run kept its warm
   * scratch, live and replay would diverge on the very first substep (measured:
   * every fluid particle, pos+vel, within 1 substep). So recording starts by
   * restoring the live engine onto the SAME load boundary the replay will
   * reconstruct — the visible state is untouched (state words are identical;
   * only unhashed scratch resets), and from that instant live and replay are
   * one timeline. This is the engine's own restore contract
   * (restoreAuthoringSnapshot: "exactly like a fresh scene"), §26.C's shape.
   */
  async startRecording() {
    const startSubstep = this.substep;
    const initial = await this.snapshot({ exact: false });
    // deep-copy: _loadPayload clears the schedule Map it finds on the instance
    const pendingSchedule = new Map(
      [...this._schedule.entries()].map(([s, ps]) => [s, ps.slice()]));
    const pendingPushes = new Map(
      [...this._pushes.entries()].map(([s, ps]) => [s, ps.map((p) => ({ ...p }))]));
    this._loadPayload(initial);
    this._schedule = pendingSchedule;
    this._pushes = pendingPushes;
    this.substep = startSubstep;
    this._rec = { startSubstep, initial, events: [], order: 0 };
    // journal anything already armed for the future so the replay gets it too
    for (const [s, ps] of [...this._schedule.entries()].sort((a, b) => a[0] - b[0])) {
      this._rec.events.push({
        frame: s - startSubstep, order: this._rec.order++, type: 'inject',
        particles: ps.map((p) => ({ ...p })),
      });
    }
    for (const [s, ps] of [...this._pushes.entries()].sort((a, b) => a[0] - b[0])) {
      for (const p of ps) {
        this._rec.events.push({ frame: s - startSubstep, order: this._rec.order++,
                                type: 'push', ...p });
      }
    }
  }

  /** Stop recording and return { text, doc }: `text` is the `.sunar` file
   *  (feed it to replayFromSunar), `doc.final.digest` is the recorded run's
   *  closing state digest — the replay gate's expected value. */
  async stopRecording() {
    if (!this._rec) throw new Error('[demo_boot] stopRecording without startRecording');
    const rec = this._rec;
    this._rec = null;
    // A page may arm future substeps out of order; the journal is canonically
    // frame-sorted (stable), then renumbered, so encodeSunar's ordering
    // invariant holds and identical schedules give identical journals.
    rec.events.sort((a, b) => (a.frame - b.frame) || (a.order - b.order));
    rec.events.forEach((e, i) => { e.order = i; });
    const durationSubsteps = this.substep - rec.startSubstep;
    const doc = {
      app: rec.initial.app,
      initial: rec.initial,
      events: rec.events,
      substepsPerFrame: 1,           // frame == substep offset from start
      durationFrames: durationSubsteps,
      durationSubsteps,
      final: { digest: await this.digest(), n: this.n, nFluid: this.nFluid },
    };
    return { text: encodeSunar(doc), doc };
  }
}

export { digestState, HASHED_WORDS, PARTICLE_WORDS, ONE, MAT_WATER };
