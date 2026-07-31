// SunaEngine — real SunaBox (aether) engine code, extracted UNMODIFIED below this header
// from aether commit e41e06f. Verify: diff <(tail -n +6 src/sunar.js) <aether>/src/sunar.js
// Copyright (C) 2026 Darien Bathalter
// SPDX-License-Identifier: AGPL-3.0-only — see LICENSE. NO WARRANTY.

// SunaBox deterministic interaction replay v1.
//
// A replay is not a video. It is one exact `.suna` starting world plus an
// ordered, frame/substep-indexed journal of normalized user inputs. The solver
// advances a fixed two substeps per animation frame, so the same journal
// reconstructs the same world without storing rendered frames.

import { encodeSuna, decodeSuna } from './suna.js';

export const SUNAR_MAGIC = 'SunaBox Replay';
export const SUNAR_FORMAT_VERSION = 1;
export const SUNAR_EXTENSION = '.sunar';
export const SUNAR_MIME = 'application/vnd.sunabox.replay+json';

const integer = (x, label, lo = 0, hi = 0x7fffffff) => {
  if (!Number.isInteger(x) || x < lo || x > hi) {
    throw new Error(`${label} is outside ${lo}..${hi}`);
  }
  return x;
};

export function encodeSunar(replay) {
  if (!replay?.initial) throw new Error('Suna replay needs an initial scene');
  const events = Array.isArray(replay.events) ? replay.events : [];
  let prevFrame = -1, prevOrder = -1;
  for (let i = 0; i < events.length; i++) {
    const e = events[i] ?? {};
    const frame = integer(e.frame, `events[${i}].frame`);
    const order = integer(e.order, `events[${i}].order`);
    if (frame < prevFrame || (frame === prevFrame && order <= prevOrder)) {
      throw new Error(`events[${i}] is not in stable frame/order sequence`);
    }
    if (typeof e.type !== 'string' || !e.type) throw new Error(`events[${i}].type is empty`);
    prevFrame = frame; prevOrder = order;
  }
  const initial = JSON.parse(encodeSuna(replay.initial));
  const doc = {
    magic: SUNAR_MAGIC,
    formatVersion: SUNAR_FORMAT_VERSION,
    app: replay.app ?? initial.app,
    timing: {
      substepsPerFrame: integer(replay.substepsPerFrame ?? 2, 'substepsPerFrame', 1, 32),
      durationFrames: integer(replay.durationFrames ?? 0, 'durationFrames'),
      durationSubsteps: integer(replay.durationSubsteps ?? 0, 'durationSubsteps'),
    },
    initial,
    events,
    final: replay.final ?? null,
  };
  return JSON.stringify(doc, null, 2) + '\n';
}

export function decodeSunar(source) {
  const text = typeof source === 'string' ? source
    : new TextDecoder().decode(source instanceof Uint8Array ? source : new Uint8Array(source));
  let doc;
  try { doc = JSON.parse(text); } catch (e) { throw new Error(`not valid JSON: ${e.message}`); }
  if (doc?.magic !== SUNAR_MAGIC) throw new Error(`not a ${SUNAR_MAGIC} file`);
  if (doc.formatVersion !== SUNAR_FORMAT_VERSION) {
    throw new Error(`unsupported .sunar formatVersion ${doc.formatVersion}; this build reads 1`);
  }
  const timing = doc.timing ?? {};
  integer(timing.substepsPerFrame, 'timing.substepsPerFrame', 1, 32);
  integer(timing.durationFrames, 'timing.durationFrames');
  integer(timing.durationSubsteps, 'timing.durationSubsteps');
  const events = Array.isArray(doc.events) ? doc.events : [];
  let prevFrame = -1, prevOrder = -1;
  for (let i = 0; i < events.length; i++) {
    const e = events[i] ?? {};
    const frame = integer(e.frame, `events[${i}].frame`);
    const order = integer(e.order, `events[${i}].order`);
    if (frame < prevFrame || (frame === prevFrame && order <= prevOrder)) {
      throw new Error(`events[${i}] is not in stable frame/order sequence`);
    }
    if (typeof e.type !== 'string' || !e.type) throw new Error(`events[${i}].type is empty`);
    prevFrame = frame; prevOrder = order;
  }
  return {
    app: doc.app,
    substepsPerFrame: timing.substepsPerFrame,
    durationFrames: timing.durationFrames,
    durationSubsteps: timing.durationSubsteps,
    initial: decodeSuna(JSON.stringify(doc.initial)),
    events,
    final: doc.final ?? null,
  };
}
