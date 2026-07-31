// SunaEngine — real SunaBox (aether) engine code, extracted UNMODIFIED below this header
// from aether commit e41e06f. Verify: diff <(tail -n +6 src/hash/statehash.js) <aether>/src/hash/statehash.js
// Copyright (C) 2026 Darien Bathalter
// SPDX-License-Identifier: AGPL-3.0-only — see LICENSE. NO WARRANTY.

// Aether canonical state hash — JS reference implementation.
// MUST stay bit-identical to src/hash/statehash.wgsl. Any edit here needs the
// same edit there, and test/pages/hash_parity.html must be re-run.
//
// Algorithm: MurmurHash3_x86_32 body/finalizer, run as 4 independent salted
// lanes over each particle's words, combined ACROSS particles by u32 addition
// (commutative + associative => dispatch-order independent => GPU-safe).
//
// Two digests fall out of the same pass:
//   H_set  (lanes 0..3) : payload only. Invariant under particle permutation.
//   H_slot (lanes 4..7) : payload + storage slot index. Detects any reordering.
// Compare H_slot as the gate. If H_set matches but H_slot differs, the
// divergence is a sort/ordering instability, not a physics divergence.

export const MM_C1 = 0xcc9e2d51;
export const MM_C2 = 0x1b873593;

// Lane seeds (initial states) and lane salts (XORed into every key word so the
// four lanes are not merely the same chain from different start states).
export const LANE_SEED = [0x9e3779b1, 0x85ebca77, 0xc2b2ae3d, 0x27d4eb2f];
export const LANE_SALT = [0x00000000, 0x51633e2d, 0x9c8f2a17, 0xd0e5b943];

export const CHAIN_INIT = [
  0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344,
  0xa4093822, 0x299f31d0, 0x082efa98, 0xec4e6c89,
];

const rotl32 = (x, r) => (((x << (r & 31)) | (x >>> ((32 - r) & 31))) >>> 0);

/** One MurmurHash3_x86_32 body step. h and k are u32, result u32. */
export function mmStep(h, k) {
  k = Math.imul(k, MM_C1) >>> 0;
  k = rotl32(k, 15);
  k = Math.imul(k, MM_C2) >>> 0;
  h = (h ^ k) >>> 0;
  h = rotl32(h, 13);
  h = (Math.imul(h, 5) + 0xe6546b64) >>> 0;
  return h;
}

/** MurmurHash3 fmix32 avalanche finalizer. */
export function fmix32(h) {
  h = (h ^ (h >>> 16)) >>> 0;
  h = Math.imul(h, 0x85ebca6b) >>> 0;
  h = (h ^ (h >>> 13)) >>> 0;
  h = Math.imul(h, 0xc2b2ae35) >>> 0;
  h = (h ^ (h >>> 16)) >>> 0;
  return h;
}

/**
 * Digest a packed particle state buffer.
 *
 * @param {Uint32Array|Int32Array|ArrayBuffer} state raw sim state. Read as u32
 *        BITS — never as numeric i32 values (see WGSL bitcast note).
 * @param {{particleCount:number, strideWords:number, hashedWords:number,
 *          nFluid?:number, bond?:Uint32Array|Int32Array, bondCap?:number}} p
 *        strideWords = words per particle in the buffer (may include padding).
 *        hashedWords = leading words that participate (<= strideWords). Words
 *        in [hashedWords, strideWords) are EXCLUDED: this is how padding and
 *        scratch fields are kept out of the digest.
 *
 *        §18: `bond` is the packed bond table (nFluid rows of bondCap u32).
 *        WHEN `p.bond` IS ABSENT THE FOLD DOES NOT RUN, so every pre-v8 caller
 *        and every existing parity vector keeps its exact behaviour and its
 *        exact digest. `Engine.digestCPU()` MUST pass it, or
 *        `cpuDigestOfFinalReadback` silently diverges from the GPU digest on
 *        every bonded scene — a one-line omission that would look exactly like
 *        a real divergence.
 * @returns {{acc:Uint32Array, hex:string, setHex:string, slotHex:string, n:number}}
 */
export const BOND_SALT = 0x5bd1e995;
export const BOND_SALT2 = 0x2f5b3c17;

export function digestState(state, p) {
  const u32 =
    state instanceof Uint32Array ? state
    : state instanceof Int32Array ? new Uint32Array(state.buffer, state.byteOffset, state.length)
    : new Uint32Array(state);

  const { particleCount, strideWords, hashedWords } = p;
  if (hashedWords > strideWords) throw new Error('hashedWords > strideWords');
  if (particleCount * strideWords > u32.length) throw new Error('state buffer too small');

  // §18 bond fold, opt-in by the presence of `p.bond`.
  const bondRaw = p.bond ?? null;
  const bondU32 = bondRaw == null ? null
    : (bondRaw instanceof Uint32Array ? bondRaw
      : new Uint32Array(bondRaw.buffer, bondRaw.byteOffset, bondRaw.length));
  const bondCap = (p.bondCap ?? 0) >>> 0;
  const nFluid = (p.nFluid ?? particleCount) >>> 0;
  if (bondU32 && bondCap === 0) throw new Error('digestState: p.bond given without p.bondCap');

  const acc = new Uint32Array(8);
  const n = hashedWords >>> 0;

  for (let i = 0; i < particleCount; i++) {
    const base = i * strideWords;
    let h0 = LANE_SEED[0], h1 = LANE_SEED[1], h2 = LANE_SEED[2], h3 = LANE_SEED[3];
    for (let w = 0; w < hashedWords; w++) {
      const k = u32[base + w];
      h0 = mmStep(h0, (k ^ LANE_SALT[0]) >>> 0);
      h1 = mmStep(h1, (k ^ LANE_SALT[1]) >>> 0);
      h2 = mmStep(h2, (k ^ LANE_SALT[2]) >>> 0);
      h3 = mmStep(h3, (k ^ LANE_SALT[3]) >>> 0);
    }
    // §18. TWO LANE SETS: `s` never sees a partner slot (H_set must stay
    // permutation-invariant or §7.6's triage signal is gone), `l` folds the
    // ordered words including the partner. The `i < nFluid` guard is what keeps
    // the digest independent of rows nothing ever writes.
    let s0 = h0, s1 = h1, s2 = h2, s3 = h3;
    let l0 = h0, l1 = h1, l2 = h2, l3 = h3;
    // BOTH FOLDS ARE COMMUTATIVE — see the note in statehash.wgsl. A bond row's
    // ORDER is the neighbour traversal order; the bond SET is not, and neither
    // is any force. An ordered fold moves the chain when `canonicalize` is
    // defeated on a bonded scene, which is the exact order-independence contract
    // the determinism tripwire protects.
    if (bondU32 && i < nFluid) {
      let bn = 0, com = 0, scom = 0;
      for (let k = 0; k < bondCap; k++) {
        const w = bondU32[i * bondCap + k] >>> 0;
        if (w === 0) break;                        // dense row; the tail is zero
        com  = (com  + fmix32((((w >>> 17) & 0x1fff) ^ BOND_SALT) >>> 0)) >>> 0;
        scom = (scom + fmix32((w ^ BOND_SALT2) >>> 0)) >>> 0;
        bn++;
      }
      if (bn !== 0) {
        s0 = (mmStep(s0, bn >>> 0) ^ com) >>> 0; s1 = (mmStep(s1, bn >>> 0) ^ com) >>> 0;
        s2 = (mmStep(s2, bn >>> 0) ^ com) >>> 0; s3 = (mmStep(s3, bn >>> 0) ^ com) >>> 0;
        l0 = (mmStep(l0, bn >>> 0) ^ scom) >>> 0; l1 = (mmStep(l1, bn >>> 0) ^ scom) >>> 0;
        l2 = (mmStep(l2, bn >>> 0) ^ scom) >>> 0; l3 = (mmStep(l3, bn >>> 0) ^ scom) >>> 0;
      }
    }
    // H_set: payload only.
    acc[0] = (acc[0] + fmix32((s0 ^ n) >>> 0)) >>> 0;
    acc[1] = (acc[1] + fmix32((s1 ^ n) >>> 0)) >>> 0;
    acc[2] = (acc[2] + fmix32((s2 ^ n) >>> 0)) >>> 0;
    acc[3] = (acc[3] + fmix32((s3 ^ n) >>> 0)) >>> 0;
    // H_slot: payload with the storage slot index appended.
    const m = (n + 1) >>> 0;
    acc[4] = (acc[4] + fmix32((mmStep(l0, i >>> 0) ^ m) >>> 0)) >>> 0;
    acc[5] = (acc[5] + fmix32((mmStep(l1, i >>> 0) ^ m) >>> 0)) >>> 0;
    acc[6] = (acc[6] + fmix32((mmStep(l2, i >>> 0) ^ m) >>> 0)) >>> 0;
    acc[7] = (acc[7] + fmix32((mmStep(l3, i >>> 0) ^ m) >>> 0)) >>> 0;
  }

  return {
    acc,
    hex: hex8(acc),
    setHex: hex8(acc.subarray(0, 4)),
    slotHex: hex8(acc.subarray(4, 8)),
    n: particleCount,
  };
}

/** Fold one frame's 8-lane digest into a persistent 8-lane chain. In place. */
export function foldChain(chain, acc) {
  const next = new Uint32Array(8);
  for (let j = 0; j < 8; j++) {
    next[j] = mmStep(chain[j], (acc[j] ^ rotl32(acc[(j + 1) & 7], 7)) >>> 0);
  }
  chain.set(next);
  return chain;
}

export function newChain() { return Uint32Array.from(CHAIN_INIT); }

export const hex8 = (a) =>
  Array.from(a, (v) => (v >>> 0).toString(16).padStart(8, '0')).join('');
