// SunaEngine — real SunaBox (aether) engine code, extracted UNMODIFIED below this header
// from aether commit e41e06f. Verify: diff <(tail -n +6 src/tables.js) <aether>/src/tables.js
// Copyright (C) 2026 Darien Bathalter
// SPDX-License-Identifier: AGPL-3.0-only — see LICENSE. NO WARRANTY.

// ============================================================================
// aether — src/tables.js  (implementer F, per BINDING SPEC §9.1)
// ----------------------------------------------------------------------------
// OFFLINE kernel-table generator. Runs ONCE, in f64, and its output is FROZEN
// into src/tables.json (base64 Int32Array) which is what ships.
//
// buildTables() MUST NEVER be called at runtime in a shipped build: the adhesion
// kernel contains x^0.25, so a different libm would change every entry and
// therefore every state hash thousands of substeps later. That is the silent
// cross-machine failure mode all four design reports flagged independently.
//
// Regenerate with:   node src/tables.js --write
// ============================================================================

// ── pinned geometry (spec §4 / §3.4) ────────────────────────────────────────
export const ONE  = 65536;
export const H    = 163840;          // h = 2.5 wu
export const DX0  = 65536;           // rest spacing 1.0 wu
export const RHO0 = 1 << 20;
export const PRE  = 4;
export const HS   = H >> PRE;        // 10240
export const H2S  = HS * HS;         // 104857600
export const KSH  = 15;
export const TN   = (H2S >> KSH) + 2;  // 3202
export const GF   = 22;
export const CF   = 22;

export const OFF_W = 0;
export const OFF_G = TN;
export const OFF_C = 2 * TN;
export const OFF_A = 3 * TN;
export const OFF_S = 4 * TN;

// ── frozen generation parameters (spec §6.4) ────────────────────────────────
export const TABLE_PARAMS = Object.freeze({
  h: 2.5, rmin: 0.1, scorrK: 0, scorrN: 4, scorrDq: 0.2,
});

const POW = new Float64Array(64);
for (let i = 0; i < 64; i++) POW[i] = Math.pow(2, i);

/** Centre of LUT bin i, in wu. */
export const rOfIdx = (i) => Math.sqrt((i + 0.5) * POW[KSH]) * POW[PRE] / ONE;

/**
 * Build the five kernel tables in f64 and round to the pinned integer formats.
 * The WT rescale is computed THROUGH the real lut_index pre-shift and bin
 * quantisation (not through the analytic kernel), so what is calibrated is
 * exactly what is baked.
 */
export function buildTables(cfg = TABLE_PARAMS) {
  const h = cfg.h;
  const WT = new Int32Array(TN);
  const GT = new Int32Array(TN);
  const CT = new Int32Array(TN);
  const AT = new Int32Array(TN);
  const SC = new Int32Array(TN);

  const poly6   = (r) => (r >= h ? 0 : (4 / (Math.PI * h ** 8)) * (h * h - r * r) ** 3);
  const spikyDr = (r) => (r >= h ? 0 : -(30 / (Math.PI * h ** 5)) * (h - r) ** 2);
  const cohRaw  = (r) => {
    if (r >= h || r <= 0) return 0;
    const a = (h - r) ** 3 * r ** 3;
    return (2 * r > h) ? a : 2 * a - h ** 6 / 64;
  };
  const cohPeak = cohRaw(h / 2);
  const adhRaw  = (r) => (2 * r > h && r < h ? (-4 * r * r / h + 6 * r - 2 * h) ** 0.25 : 0);
  const adhPeak = adhRaw(0.75 * h);
  const RMIN = cfg.rmin * h;

  // Discrete-lattice rest-density calibration. The Riemann sum of a normalised
  // kernel over a rest lattice is NOT 1 (measured 0.99335), so an authored block
  // would immediately implode. Rescaling here keeps RHO0 = 2^20 exact, so
  // C = fp_shr_rne(rho - RHO0, 4) stays a pure shift.
  const binOf = (dx, dy) => {
    const ax = (Math.abs(dx) + 8) >> PRE, ay = (Math.abs(dy) + 8) >> PRE;
    if (ax > HS || ay > HS) return -1;
    const r2 = ax * ax + ay * ay;
    return r2 >= H2S ? -1 : (r2 >> KSH);
  };
  const reach = Math.ceil(H / DX0);
  let latt = poly6(rOfIdx(0));                       // self term == WT[0]
  for (let jy = -reach; jy <= reach; jy++) {
    for (let jx = -reach; jx <= reach; jx++) {
      if (jx === 0 && jy === 0) continue;
      const b = binOf(jx * DX0, jy * DX0);
      if (b >= 0) latt += poly6(rOfIdx(b));
    }
  }
  const WCAL = 1 / latt;

  for (let i = 0; i < TN; i++) {
    const r = rOfIdx(i), rc = Math.max(r, RMIN);
    WT[i] = Math.round(RHO0 * WCAL * poly6(r));
    GT[i] = Math.round(RHO0 * (spikyDr(rc) / rc) * POW[GF - 16]);
    CT[i] = Math.round((cohRaw(rc) / cohPeak / rc) * POW[CF]);
    AT[i] = Math.round((adhRaw(rc) / adhPeak / rc) * POW[CF]);
  }
  const Wdq = poly6(cfg.scorrDq * h);
  for (let i = 0; i < TN; i++) {
    const ratio = Wdq > 0 ? poly6(rOfIdx(i)) / Wdq : 0;
    SC[i] = -Math.round(cfg.scorrK * ratio ** cfg.scorrN);
  }
  return { WT, GT, CT, AT, SC, WCAL, latt };
}

/** Pack the five tables into the single 5*TN i32 `luts` buffer image. */
export function packTables(t) {
  const out = new Int32Array(5 * TN);
  out.set(t.WT, OFF_W);
  out.set(t.GT, OFF_G);
  out.set(t.CT, OFF_C);
  out.set(t.AT, OFF_A);
  out.set(t.SC, OFF_S);
  return out;
}

// ── base64 <-> bytes, environment agnostic ──────────────────────────────────
export function bytesToB64(u8) {
  if (typeof Buffer !== 'undefined') return Buffer.from(u8).toString('base64');
  let s = '';
  for (let i = 0; i < u8.length; i++) s += String.fromCharCode(u8[i]);
  return btoa(s);
}
export function b64ToBytes(b64) {
  if (typeof Buffer !== 'undefined') return new Uint8Array(Buffer.from(b64, 'base64'));
  const s = atob(b64);
  const u8 = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u8[i] = s.charCodeAt(i);
  return u8;
}

// ── table hash: Murmur3, 4 salted lanes, set+slot, 64 hex chars ─────────────
// Same primitive as the state hash (src/hash/statehash.js) so there is exactly
// one hash implementation in the project.
const MM_C1 = 0xcc9e2d51, MM_C2 = 0x1b873593;
const LANE_SEED = [0x9e3779b1, 0x85ebca77, 0xc2b2ae3d, 0x27d4eb2f];
const LANE_SALT = [0x00000000, 0x51633e2d, 0x9c8f2a17, 0xd0e5b943];
const rotl32 = (x, r) => (((x << (r & 31)) | (x >>> ((32 - r) & 31))) >>> 0);
function mmStep(h, k) {
  k = Math.imul(k, MM_C1) >>> 0; k = rotl32(k, 15); k = Math.imul(k, MM_C2) >>> 0;
  h = (h ^ k) >>> 0; h = rotl32(h, 13); h = (Math.imul(h, 5) + 0xe6546b64) >>> 0;
  return h;
}
function fmix32(h) {
  h = (h ^ (h >>> 16)) >>> 0; h = Math.imul(h, 0x85ebca6b) >>> 0;
  h = (h ^ (h >>> 13)) >>> 0; h = Math.imul(h, 0xc2b2ae35) >>> 0;
  h = (h ^ (h >>> 16)) >>> 0; return h;
}
const hex8 = (a) => Array.from(a, (v) => (v >>> 0).toString(16).padStart(8, '0')).join('');

/** 64 hex chars over an Int32Array table image. Order sensitive (slot lanes). */
export function tableHash(int32) {
  const u32 = new Uint32Array(int32.buffer, int32.byteOffset, int32.length);
  const acc = new Uint32Array(8);
  for (let i = 0; i < u32.length; i++) {
    const k = u32[i];
    let h0 = LANE_SEED[0], h1 = LANE_SEED[1], h2 = LANE_SEED[2], h3 = LANE_SEED[3];
    h0 = mmStep(h0, (k ^ LANE_SALT[0]) >>> 0);
    h1 = mmStep(h1, (k ^ LANE_SALT[1]) >>> 0);
    h2 = mmStep(h2, (k ^ LANE_SALT[2]) >>> 0);
    h3 = mmStep(h3, (k ^ LANE_SALT[3]) >>> 0);
    acc[0] = (acc[0] + fmix32((h0 ^ 1) >>> 0)) >>> 0;
    acc[1] = (acc[1] + fmix32((h1 ^ 1) >>> 0)) >>> 0;
    acc[2] = (acc[2] + fmix32((h2 ^ 1) >>> 0)) >>> 0;
    acc[3] = (acc[3] + fmix32((h3 ^ 1) >>> 0)) >>> 0;
    acc[4] = (acc[4] + fmix32((mmStep(h0, i >>> 0) ^ 2) >>> 0)) >>> 0;
    acc[5] = (acc[5] + fmix32((mmStep(h1, i >>> 0) ^ 2) >>> 0)) >>> 0;
    acc[6] = (acc[6] + fmix32((mmStep(h2, i >>> 0) ^ 2) >>> 0)) >>> 0;
    acc[7] = (acc[7] + fmix32((mmStep(h3, i >>> 0) ^ 2) >>> 0)) >>> 0;
  }
  return hex8(acc);
}

/** Load, verify and return the frozen table image from a tables.json object. */
export function loadFrozen(json) {
  const bytes = b64ToBytes(json.b64);
  const img = new Int32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  const h = tableHash(img);
  if (h !== json.tableHash) throw new Error(`TABLE_HASH mismatch: ${h} != ${json.tableHash}`);
  if (img.length !== 5 * TN) throw new Error(`table length ${img.length} != ${5 * TN}`);
  return img;
}

// ── node entry point: regenerate src/tables.json ────────────────────────────
if (typeof process !== 'undefined' && process.argv && process.argv[1]
    && process.argv[1].endsWith('tables.js')) {
  const t = buildTables();
  const img = packTables(t);
  const json = {
    // The LUT BYTES have not changed since v1 (`tableHash` is identical across
    // every version), but engine.js asserts json.specVersion === SPEC_VERSION so
    // that a stale tables.json cannot be loaded against a newer solver. Bump it
    // in the same commit as engine.js's SPEC_VERSION. v9 = §18.22 one gesture,
    // one weld group; v10 = §20 TRUE RIGID BODIES + draw-while-paused;
    // v11 = §19 anchored matter + axis bodies; v12 = §18.23 RUBBER/PHOTO bonds
    // (a MATERIAL-TABLE change: `tableHash` below is byte-identical across it,
    // which is precisely why `matsHash` had to be invented — see
    // test/determinism.html). v13 = §23 devices (a filtering drain, a jet
    // thrust primitive and the String pivot) — `tableHash` AND `matsHash` are
    // both byte-identical across it, because §23 adds no material row at all.
    // v14 = §24 USER: arrow-key matter. Same story as v13 and for the same
    // reason: one §20 body-record word and one uniform vector, no material row,
    // so `tableHash` and `matsHash` are BOTH byte-identical across it too and
    // the v13 -> v14 golden diff is the single field `specVersion`.
    // v15 = §26 the feedback pass. `tableHash` is byte-identical AGAIN (the LUTs
    // still have nothing to do with the material table) but `matsHash` and
    // `matCount` both move, because §26.F appends BEDROCK — and §26 touches
    // `src/sim.wgsl` not at all, so those three fields are the whole diff.
    // v16 = §25 composable property bits. `tableHash` byte-identical (still no
    // LUT change) and `matsHash`/`matCount` DO NOT move either — MAT_MAX is a
    // capacity constant, interned rows are scene state outside the authored
    // table, and every authored row ships tempDrive 0 — so the v15 -> v16
    // golden diff is the single field `specVersion` again.
    // v17 = §29 pressurized containment. The kernel LUT bytes still do not
    // move; authored material words 46/47 now carry burstP/sealMode, so
    // matsHash moves while the all-WATER state chain remains byte-identical.
    // v18 = §30 host-authored Inflow/Outflow matter. No LUT or authored
    // material coefficient changes; this is the recorded capability version.
    // v19 = §31 OE-CAKE structural import fidelity. LUT bytes remain unchanged;
    // WOOD's authored sealMode and the simulation capability version move.
    // v20 = §32 v10 module constraints. Body-driver sidecars do not touch LUT
    // bytes or authored material rows.
    // v21 = §33 thin imported rigid/anchored collision sealing. The change is
    // entirely in the solver; LUT bytes and authored material rows are stable.
    // v22 = §34 full swept-grid hard body contact plus staggered congealing
    // fusion. LUT bytes and authored material rows remain stable.
    specVersion: 22,
    tn: TN,
    offsets: { W: OFF_W, G: OFF_G, C: OFF_C, A: OFF_A, S: OFF_S },
    params: TABLE_PARAMS,
    lattSum: t.latt,
    wcal: t.WCAL,
    tableHash: tableHash(img),
    b64: bytesToB64(new Uint8Array(img.buffer, img.byteOffset, img.byteLength)),
  };
  const text = JSON.stringify(json, null, 1) + '\n';
  if (process.argv.includes('--write')) {
    const { writeFileSync } = await import('node:fs');
    const { fileURLToPath } = await import('node:url');
    const out = fileURLToPath(new URL('./tables.json', import.meta.url));
    writeFileSync(out, text);
    console.log('wrote', out, json.tableHash);
  }
  console.log(JSON.stringify({
    TN, lattSum: t.latt, WCAL: t.WCAL, tableHash: json.tableHash,
    WT0: t.WT[0], WTmax: Math.max(...t.WT), GTmin: Math.min(...t.GT),
    CTmin: Math.min(...t.CT), CTmax: Math.max(...t.CT), ATmax: Math.max(...t.AT),
    SCnonzero: t.SC.some((v) => v !== 0),
  }, null, 1));
}
