// SunaEngine — real SunaBox (aether) engine code, extracted UNMODIFIED below this header
// from aether commit e41e06f. Verify: diff <(tail -n +6 src/suna.js) <aether>/src/suna.js
// Copyright (C) 2026 Darien Bathalter
// SPDX-License-Identifier: AGPL-3.0-only — see LICENSE. NO WARRANTY.

// SunaBox scene container v1.
//
// `.suna` is intentionally JSON: the header, settings, devices and body map are
// inspectable in any editor. Large exact integer buffers are base64-encoded
// little-endian words so a 100k-particle scene does not become millions of
// decimal characters. This is our format; `.oec` remains import-only.

export const SUNA_MAGIC = 'SunaBox Scene';
export const SUNA_FORMAT_VERSION = 1;
export const SUNA_EXTENSION = '.suna';
export const SUNA_MIME = 'application/vnd.sunabox.scene+json';

const bytesToB64 = (bytes) => {
  let s = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    s += String.fromCharCode(...bytes.subarray(i, Math.min(bytes.length, i + CHUNK)));
  }
  return btoa(s);
};
const b64ToBytes = (s) => {
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
};
const packWords = (a) => bytesToB64(new Uint8Array(a.buffer, a.byteOffset, a.byteLength));
const unpackWords = (s, Type, words, label) => {
  const bytes = b64ToBytes(s);
  if (bytes.byteLength !== words * 4) {
    throw new Error(`${label} has ${bytes.byteLength} bytes; expected ${words * 4}`);
  }
  const copy = bytes.slice();
  return new Type(copy.buffer);
};
const integer = (x, label, lo = -2147483648, hi = 2147483647) => {
  if (!Number.isInteger(x) || x < lo || x > hi) throw new Error(`${label} is outside ${lo}..${hi}`);
  return x;
};

export function encodeSuna(payload) {
  const { state, tints, bonds, bondCand, bodyRecords, restOffsets } = payload;
  if (!(state instanceof Int32Array)) throw new Error('Suna save needs an Int32Array state');
  if (!(tints instanceof Uint32Array)) throw new Error('Suna save needs a Uint32Array tint buffer');
  const n = integer(payload.scene?.n, 'scene.n', 0, 131072);
  integer(payload.scene?.nFluid, 'scene.nFluid', 0, n);
  if (state.length !== n * 8) throw new Error(`state has ${state.length} words; expected ${n * 8}`);
  if (tints.length !== n) throw new Error(`tints has ${tints.length} words; expected ${n}`);
  const exact = bonds instanceof Uint32Array && bondCand instanceof Uint32Array &&
    bodyRecords instanceof Int32Array && restOffsets instanceof Int32Array;
  if (exact) {
    if (bonds.length !== bondCand.length) throw new Error('bond and candidate tables differ in size');
    if (restOffsets.length !== n * 2) {
      throw new Error(`rest offsets have ${restOffsets.length} words; expected ${n * 2}`);
    }
  }
  const doc = {
    magic: SUNA_MAGIC,
    formatVersion: SUNA_FORMAT_VERSION,
    app: payload.app,
    scene: {
      ...payload.scene,
      state: { encoding: 'i32le/base64', words: state.length, data: packWords(state) },
      tints: { encoding: 'u32le/base64', words: tints.length, data: packWords(tints) },
      ...(exact ? {
        bonds: { encoding: 'u32le/base64', words: bonds.length, data: packWords(bonds) },
        bondCandidates: { encoding: 'u32le/base64', words: bondCand.length, data: packWords(bondCand) },
        bodyRecords: { encoding: 'i32le/base64', words: bodyRecords.length, data: packWords(bodyRecords) },
        restOffsets: { encoding: 'i32le/base64', words: restOffsets.length, data: packWords(restOffsets) },
      } : {}),
    },
    materials: payload.materials ?? { events: [] },
    bodies: payload.bodies ?? [],
    drivers: payload.drivers ?? [],
    devices: payload.devices ?? [],
    settings: payload.settings ?? {},
  };
  return JSON.stringify(doc, null, 2) + '\n';
}

export function decodeSuna(source) {
  const text = typeof source === 'string' ? source
    : new TextDecoder().decode(source instanceof Uint8Array ? source : new Uint8Array(source));
  let doc;
  try { doc = JSON.parse(text); } catch (e) { throw new Error(`not valid JSON: ${e.message}`); }
  if (doc?.magic !== SUNA_MAGIC) throw new Error(`not a ${SUNA_MAGIC} file`);
  if (doc.formatVersion !== 1) {
    throw new Error(`unsupported .suna formatVersion ${doc.formatVersion}; this build reads 1`);
  }
  const s = doc.scene ?? {};
  const n = integer(s.n, 'scene.n', 0, 131072);
  integer(s.nFluid, 'scene.nFluid', 0, n);
  integer(s.domW, 'scene.domW', 1);
  integer(s.domH, 'scene.domH', 1);
  integer(s.cellShift, 'scene.cellShift', 1, 27);
  const state = unpackWords(s.state?.data ?? '', Int32Array, n * 8, 'scene.state');
  const tints = unpackWords(s.tints?.data ?? '', Uint32Array, n, 'scene.tints');
  const hasExact = !!(s.bonds && s.bondCandidates && s.bodyRecords && s.restOffsets);
  const bonds = hasExact
    ? unpackWords(s.bonds.data ?? '', Uint32Array, integer(s.bonds.words, 'scene.bonds.words', 0), 'scene.bonds')
    : null;
  const bondCand = hasExact
    ? unpackWords(s.bondCandidates.data ?? '', Uint32Array,
      integer(s.bondCandidates.words, 'scene.bondCandidates.words', 0), 'scene.bondCandidates')
    : null;
  const bodyRecords = hasExact
    ? unpackWords(s.bodyRecords.data ?? '', Int32Array,
      integer(s.bodyRecords.words, 'scene.bodyRecords.words', 0), 'scene.bodyRecords')
    : null;
  const restOffsets = hasExact
    ? unpackWords(s.restOffsets.data ?? '', Int32Array,
      integer(s.restOffsets.words, 'scene.restOffsets.words', 0), 'scene.restOffsets')
    : null;
  const scene = { ...s };
  for (const k of ['state', 'tints', 'bonds', 'bondCandidates', 'bodyRecords', 'restOffsets']) {
    delete scene[k];
  }
  return { ...doc, scene, state, tints, bonds, bondCand, bodyRecords, restOffsets };
}
