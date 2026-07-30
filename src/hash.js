// sunaEngine — deterministic state hash (JS reference)
// Bit-identical to the GPU implementation in the shader.
// Based on MurmurHash3_x86_32 with 4 salted lanes.

const MM_C1 = 0xcc9e2d51;
const MM_C2 = 0x1b873593;
const LANE_SEED = [0x9e3779b1, 0x85ebca77, 0xc2b2ae3d, 0x27d4eb2f];
const LANE_SALT = [0x00000000, 0x51633e2d, 0x9c8f2a17, 0xd0e5b943];

const rotl32 = (x, r) => (((x << (r & 31)) | (x >>> ((32 - r) & 31))) >>> 0);

function mmStep(h, k) {
  k = Math.imul(k, MM_C1) >>> 0;
  k = rotl32(k, 15);
  k = Math.imul(k, MM_C2) >>> 0;
  h = (h ^ k) >>> 0;
  h = rotl32(h, 13);
  h = (Math.imul(h, 5) + 0xe6546b64) >>> 0;
  return h;
}

function fmix32(h) {
  h = (h ^ (h >>> 16)) >>> 0;
  h = Math.imul(h, 0x85ebca6b) >>> 0;
  h = (h ^ (h >>> 13)) >>> 0;
  h = Math.imul(h, 0xc2b2ae35) >>> 0;
  h = (h ^ (h >>> 16)) >>> 0;
  return h;
}

export const hex8 = (a) =>
  Array.from(a, (v) => (v >>> 0).toString(16).padStart(8, '0')).join('');

/**
 * Digest particle state. Returns 64-char hex string.
 * @param {Uint32Array} state — raw particle buffer
 * @param {number} n — number of particles
 * @param {number} strideWords — words per particle in buffer
 */
export function digestState(state, n, strideWords) {
  const acc = new Uint32Array(8);
  for (let i = 0; i < n; i++) {
    const base = i * strideWords;
    let h0 = LANE_SEED[0], h1 = LANE_SEED[1],
        h2 = LANE_SEED[2], h3 = LANE_SEED[3];
    // Hash all words (pos, vel, mat, flags)
    for (let w = 0; w < strideWords; w++) {
      const k = state[base + w];
      h0 = mmStep(h0, (k ^ LANE_SALT[0]) >>> 0);
      h1 = mmStep(h1, (k ^ LANE_SALT[1]) >>> 0);
      h2 = mmStep(h2, (k ^ LANE_SALT[2]) >>> 0);
      h3 = mmStep(h3, (k ^ LANE_SALT[3]) >>> 0);
    }
    const m = strideWords;
    acc[0] = (acc[0] + fmix32((h0 ^ m) >>> 0)) >>> 0;
    acc[1] = (acc[1] + fmix32((h1 ^ m) >>> 0)) >>> 0;
    acc[2] = (acc[2] + fmix32((h2 ^ m) >>> 0)) >>> 0;
    acc[3] = (acc[3] + fmix32((h3 ^ m) >>> 0)) >>> 0;
    acc[4] = (acc[4] + fmix32((mmStep(h0, i >>> 0) ^ (m + 1)) >>> 0)) >>> 0;
    acc[5] = (acc[5] + fmix32((mmStep(h1, i >>> 0) ^ (m + 1)) >>> 0)) >>> 0;
    acc[6] = (acc[6] + fmix32((mmStep(h2, i >>> 0) ^ (m + 1)) >>> 0)) >>> 0;
    acc[7] = (acc[7] + fmix32((mmStep(h3, i >>> 0) ^ (m + 1)) >>> 0)) >>> 0;
  }
  return { acc, hex: hex8(acc), n };
}
