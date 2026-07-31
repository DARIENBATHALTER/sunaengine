// SunaEngine — real SunaBox (aether) engine code, extracted UNMODIFIED below this header
// from aether commit e41e06f. Verify: diff <(tail -n +6 src/hash/statehash.wgsl) <aether>/src/hash/statehash.wgsl
// Copyright (C) 2026 Darien Bathalter
// SPDX-License-Identifier: AGPL-3.0-only — see LICENSE. NO WARRANTY.

// Aether canonical state hash — WGSL implementation.
// MUST stay bit-identical to src/hash/statehash.js. Proven by
// test/pages/hash_parity.html (run via test/run.mjs hash-parity).
//
// NOTE ON BITCAST: `state` is bound as array<u32> deliberately. If you bind sim
// state as array<i32> you MUST use bitcast<u32>(x), never u32(x). WGSL's u32(x)
// on a negative i32 is a value conversion whose result is unspecified for
// out-of-range inputs; bitcast is the only defined reinterpretation.

const MM_C1 : u32 = 0xcc9e2d51u;
const MM_C2 : u32 = 0x1b873593u;

const LANE_SEED0 : u32 = 0x9e3779b1u;
const LANE_SEED1 : u32 = 0x85ebca77u;
const LANE_SEED2 : u32 = 0xc2b2ae3du;
const LANE_SEED3 : u32 = 0x27d4eb2fu;

const LANE_SALT0 : u32 = 0x00000000u;
const LANE_SALT1 : u32 = 0x51633e2du;
const LANE_SALT2 : u32 = 0x9c8f2a17u;
const LANE_SALT3 : u32 = 0xd0e5b943u;

struct HashParams {
  particle_count : u32,
  stride_words   : u32,   // words per particle in the buffer (may include padding)
  hashed_words   : u32,   // leading words that participate; <= stride_words
  // §18. WAS `_pad`. The bond fold's guard: rows in [n_fluid, particle_count)
  // are never written by any pass (bondForm1/2 are fluidOnly) and may hold a
  // PREVIOUS SCENE'S BYTES. Folding them makes the digest a function of memory
  // nothing writes — PLATFORM_NOTES trap #7 wearing a different hat. Measured:
  // without the guard, a stale boundary tail CHANGES the digest.
  n_fluid        : u32,
  bond_cap       : u32,   // §18, so BOND_CAP is not a third copy of a constant
  _pad0 : u32, _pad1 : u32, _pad2 : u32,
};

@group(0) @binding(0) var<storage, read>       state : array<u32>;
@group(0) @binding(1) var<storage, read_write> acc   : array<atomic<u32>, 8>;
@group(0) @binding(2) var<storage, read_write> chain : array<u32, 8>;
@group(0) @binding(3) var<uniform>             P     : HashParams;
// §18. The bond table is persistent solver state that determines future
// positions. If it diverged while positions had not yet diverged, the chain
// would be blind until it showed up.
@group(0) @binding(4) var<storage, read>       bond  : array<u32>;
const BOND_SALT  : u32 = 0x5bd1e995u;
const BOND_SALT2 : u32 = 0x2f5b3c17u;

// Masked so r == 0 is well defined (WGSL shift by >= 32 is undefined).
fn rotl32(x : u32, r : u32) -> u32 {
  return (x << (r & 31u)) | (x >> ((32u - r) & 31u));
}

fn mm_step(h_in : u32, k_in : u32) -> u32 {
  var k : u32 = k_in * MM_C1;
  k = rotl32(k, 15u);
  k = k * MM_C2;
  var h : u32 = h_in ^ k;
  h = rotl32(h, 13u);
  h = h * 5u + 0xe6546b64u;
  return h;
}

fn fmix32(h_in : u32) -> u32 {
  var h : u32 = h_in;
  h = h ^ (h >> 16u);
  h = h * 0x85ebca6bu;
  h = h ^ (h >> 13u);
  h = h * 0xc2b2ae35u;
  h = h ^ (h >> 16u);
  return h;
}

@compute @workgroup_size(8)
fn clear_acc(@builtin(global_invocation_id) gid : vec3<u32>) {
  atomicStore(&acc[gid.x], 0u);
}

@compute @workgroup_size(256)
fn digest(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i : u32 = gid.x;
  if (i >= P.particle_count) { return; }

  let base : u32 = i * P.stride_words;
  var h0 : u32 = LANE_SEED0;
  var h1 : u32 = LANE_SEED1;
  var h2 : u32 = LANE_SEED2;
  var h3 : u32 = LANE_SEED3;

  for (var w : u32 = 0u; w < P.hashed_words; w = w + 1u) {
    let k : u32 = state[base + w];
    h0 = mm_step(h0, k ^ LANE_SALT0);
    h1 = mm_step(h1, k ^ LANE_SALT1);
    h2 = mm_step(h2, k ^ LANE_SALT2);
    h3 = mm_step(h3, k ^ LANE_SALT3);
  }

  let n : u32 = P.hashed_words;
  let m : u32 = n + 1u;

  // ── §18 BOND FOLD. TWO LANE SETS, AND THE SPLIT IS THE POINT. ─────────────
  // `s` is the SET digest and MAY NEVER SEE A PARTNER SLOT — §7.6's whole
  // triage signal is "H_set matched but H_slot moved => this is a reordering,
  // not a physics divergence", and a partner index is a slot index. The
  // previous draft fed the ordered partner-inclusive words into h0..h3 and then
  // into BOTH accumulators, which was measured to MOVE H_set under a
  // permutation of the fluid block — i.e. it broke the exact property it
  // claimed to preserve. Split lanes: H_set folds only the bond COUNT and the
  // commutative multiset of rest lengths; H_slot folds the ordered words.
  var s0 = h0; var s1 = h1; var s2 = h2; var s3 = h3;
  var l0 = h0; var l1 = h1; var l2 = h2; var l3 = h3;
  // BOTH FOLDS ARE COMMUTATIVE, and that is a correctness requirement, not a
  // stylistic one. A bond row's ORDER is the order bondForm1 walked nbr[i] in,
  // so it is a function of the neighbour TRAVERSAL ORDER — while the bond SET,
  // and therefore every force, is not (the gather is a per-term-clamped sum).
  // An ordered fold would make the CHAIN move when `canonicalize` is defeated on
  // a bonded scene, which is precisely the order-independence contract §5.4 and
  // the determinism tripwire exist to protect. MEASURED: elements.html E15's
  // tripwire went red on a bonded scene with an ordered fold and green with this
  // one. u32 addition is commutative and associative, exactly as `acc` is across
  // particles.
  if (i < P.n_fluid) {
    var bn : u32 = 0u;
    var com : u32 = 0u;       // SET lane: L0q multiset only — NEVER a slot index
    var scom : u32 = 0u;      // SLOT lane: the whole word, partner included
    for (var k : u32 = 0u; k < P.bond_cap; k = k + 1u) {
      let w = bond[i * P.bond_cap + k];
      if (w == 0u) { break; }                     // dense row; the tail is zero
      com  = com  + fmix32(((w >> 17u) & 0x1fffu) ^ BOND_SALT);
      scom = scom + fmix32(w ^ BOND_SALT2);
      bn = bn + 1u;
    }
    // bn == 0 skips the whole fold, so a scene with NO bonds digests
    // byte-identically to no bond fold at all. That is what the all-WATER
    // golden chain rests on, and it is asserted by RUNNING an inert table.
    if (bn != 0u) {
      s0 = mm_step(s0, bn) ^ com;  s1 = mm_step(s1, bn) ^ com;
      s2 = mm_step(s2, bn) ^ com;  s3 = mm_step(s3, bn) ^ com;
      l0 = mm_step(l0, bn) ^ scom; l1 = mm_step(l1, bn) ^ scom;
      l2 = mm_step(l2, bn) ^ scom; l3 = mm_step(l3, bn) ^ scom;
    }
  }

  // H_set — payload only, permutation invariant.
  atomicAdd(&acc[0], fmix32(s0 ^ n));
  atomicAdd(&acc[1], fmix32(s1 ^ n));
  atomicAdd(&acc[2], fmix32(s2 ^ n));
  atomicAdd(&acc[3], fmix32(s3 ^ n));

  // H_slot — payload with storage slot index appended.
  atomicAdd(&acc[4], fmix32(mm_step(l0, i) ^ m));
  atomicAdd(&acc[5], fmix32(mm_step(l1, i) ^ m));
  atomicAdd(&acc[6], fmix32(mm_step(l2, i) ^ m));
  atomicAdd(&acc[7], fmix32(mm_step(l3, i) ^ m));
}

// Fold this frame's digest into the persistent chain. Dispatch AFTER digest in
// the same pass (WebGPU orders dispatches within a pass, so acc is complete).
// Reads acc[j] and acc[j+1] but writes only chain[j] => no intra-dispatch race.
@compute @workgroup_size(8)
fn fold_chain(@builtin(global_invocation_id) gid : vec3<u32>) {
  let j : u32 = gid.x;
  let a : u32 = atomicLoad(&acc[j]);
  let b : u32 = atomicLoad(&acc[(j + 1u) & 7u]);
  chain[j] = mm_step(chain[j], a ^ rotl32(b, 7u));
}
