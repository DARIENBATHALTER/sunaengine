// ============================================================================
// sunaEngine - stripped deterministic PBF solver (WGSL)
// ============================================================================
//
// Binding budget note (2026-07-30): WebGPU guarantees only 8 storage buffers
// per compute stage. The first public draft consumed 12 — above the default
// limit, so `createComputePipeline` marked every pipeline invalid and every
// dispatch became a silent no-op. The demo page looked frozen (it was). The
// scratch arrays are now merged into three packed buffers:
//   scratchA = cellCount[cellTotal] | cellStart[cellTotal+1]
//   scratchB = cellOf[i16 as u32..]    | bucketIds            | sortedIds(unused)
//   nbrBlk   = nbr[n*MAXNBR]         | nbrN[n]
//
// Determinism note: `gridSort` scatters via atomic tickets, so the order of
// ids inside one cell varies run-to-run. Every consumer that ACCUMULATES over
// neighbours sorts the neighbour index list first (insertion sort in-register),
// so the accumulation order is a pure function of particle ids, not of ticket
// order. That is what makes the hash reproducible across runs and devices.

// ---------------------------------- fixed-point arithmetic core ------------
const I32_MAX: i32 = 2147483647;
const I32_MIN: i32 = -2147483647 - 1;
const ONE16: i32 = 65536;
const FP_R30: i32 = 1073741824;

struct I64 { hi: u32, lo: u32 }

fn u64_shl(v: I64, s: u32) -> I64 {
  if (s == 0u) { return v; }
  if (s >= 32u) { return I64(v.lo << (s - 32u), 0u); }
  return I64((v.hi << s) | (v.lo >> (32u - s)), v.lo << s);
}
fn u64_add(a: I64, b: I64) -> I64 {
  let lo = a.lo + b.lo;
  let c = select(0u, 1u, lo < a.lo);
  return I64(a.hi + b.hi + c, lo);
}
fn u64_mul_u32(a: u32, b: u32) -> I64 {
  let a0 = a & 0xFFFFu; let a1 = a >> 16u;
  let b0 = b & 0xFFFFu; let b1 = b >> 16u;
  let p00 = a0 * b0; let p01 = a0 * b1; let p10 = a1 * b0; let p11 = a1 * b1;
  let mid = p01 + p10;
  let midCarry = select(0u, 1u, mid < p01);
  let lo = p00 + (mid << 16u);
  let loCarry = select(0u, 1u, lo < p00);
  let hi = p11 + (mid >> 16u) + (midCarry << 16u) + loCarry;
  return I64(hi, lo);
}
fn i64_mul_i32(a: i32, b: i32) -> I64 {
  let ua = bitcast<u32>(a); let ub = bitcast<u32>(b);
  var r = u64_mul_u32(ua, ub);
  if (a < 0) { r.hi = r.hi - ub; }
  if (b < 0) { r.hi = r.hi - ua; }
  return r;
}
fn i64_shr_rne_sat(v: I64, s: u32) -> i32 {
  let hiS = bitcast<u32>(bitcast<i32>(v.hi) >> s);
  let loS = (v.lo >> s) | (v.hi << (32u - s));
  let mask = (1u << s) - 1u; let rem = v.lo & mask; let half = 1u << (s - 1u);
  var inc: u32 = 0u;
  if (rem > half) { inc = 1u; } else if (rem == half) { inc = loS & 1u; }
  let lo2 = loS + inc; var hi2 = hiS;
  if (inc == 1u && lo2 == 0u) { hi2 = hi2 + 1u; }
  let sgn = lo2 >> 31u;
  if (hi2 == 0u && sgn == 0u) { return bitcast<i32>(lo2); }
  if (hi2 == 0xFFFFFFFFu && sgn == 1u) { return bitcast<i32>(lo2); }
  if (bitcast<i32>(v.hi) < 0) { return I32_MIN; } return I32_MAX;
}
fn fp_mul_s(a: i32, b: i32, s: u32) -> i32 { return i64_shr_rne_sat(i64_mul_i32(a, b), s); }
fn fp_add_sat(a: i32, b: i32) -> i32 {
  let s = bitcast<i32>(bitcast<u32>(a) + bitcast<u32>(b));
  if (((a ^ s) & (b ^ s)) < 0) { if (a < 0) { return I32_MIN; } return I32_MAX; }
  return s;
}
fn fp_neg_sat(a: i32) -> i32 { if (a == I32_MIN) { return I32_MAX; } return -a; }
fn fp_sub_sat(a: i32, b: i32) -> i32 {
  let s = bitcast<i32>(bitcast<u32>(a) - bitcast<u32>(b));
  if (((a ^ b) & (a ^ s)) < 0) { if (a < 0) { return I32_MIN; } return I32_MAX; }
  return s;
}
fn fp_abs(a: i32) -> i32 { if (a < 0) { return fp_neg_sat(a); } return a; }
fn fp_shr_rne(x: i32, s: u32) -> i32 {
  let q = x >> s; let rem = bitcast<u32>(x) & ((1u << s) - 1u); let half = 1u << (s - 1u);
  var inc: i32 = 0;
  if (rem > half) { inc = 1; } else if (rem == half && (q & 1) == 1) { inc = 1; }
  return q + inc;
}
fn fp_divshift(num: i32, den: i32, k: u32) -> i32 {
  if (den == 0) { if (num == 0) { return 0; } if (num > 0) { return I32_MAX; } return I32_MIN; }
  var d = den; var nn = num;
  if (d < 0) { d = fp_abs(d); nn = fp_neg_sat(nn); }
  let e = 16 - i32(countLeadingZeros(bitcast<u32>(d)));
  var dn: i32;
  if (e >= 0) { dn = d >> bitcast<u32>(e); } else { dn = d << bitcast<u32>(-e); }
  let Rn = FP_R30 / dn; let s = 30 + e - i32(k);
  if (s > 31) { return 0; }
  if (s < 1) { if (nn == 0) { return 0; } if (nn > 0) { return I32_MAX; } return I32_MIN; }
  return i64_shr_rne_sat(i64_mul_i32(nn, Rn), bitcast<u32>(s));
}

// ---------------------------------- PBF solver constants ------------------
const ONE    : i32 = 65536;
const H      : i32 = 163840;
const RHO0   : i32 = 1048576;
const TWO_RHO0 : i32 = 2097152;
const PRE    : u32 = 4u;
const HS     : i32 = 10240;
const H2S    : i32 = 104857600;
const KSH    : u32 = 15u;
const GF     : u32 = 22u;
const CF     : u32 = 22u;
const MAXNBR : u32 = 48u;
const WALL   : i32 = 32768;
const VMAX   : i32 = 65536;

const LAM_MAX  : i32 =  4194304;
const DP_TERM  : i32 =  2097152;
const RINV_MAX : i32 =   262144;
const C_MAX    : i32 =    58000;
const RHO_FLOOR_RINV : i32 = 32768;
const RHO_FLOOR_K    : i32 = 65536;
const NEG_LAM_MAX : i32 =  -4194304;
const NEG_DP_TERM : i32 =  -2097152;

// Water material
const W_EPS      : i32 =   40000;
const W_XSPH     : i32 =    6000;
const W_GAMMACOH : i32 =     240;
const W_GAMMACUR : i32 =      60;
const W_GRAV     : i32 =    1200;

// LUT offsets
const OFF_W : u32 = 0u;
const OFF_G : u32 = 3202u;
const OFF_C : u32 = 6404u;

// ---------------------------------- structures -----------------------------
struct Particle {
  pos   : vec2<i32>,
  vel   : vec2<i32>,
  _pad0 : u32,
  _pad1 : u32,
}
struct Derived {
  pred  : vec2<i32>,
  dp    : vec2<i32>,
  vtmp  : vec2<i32>,
  _pad0 : vec2<i32>,
  stAcc : vec2<i32>,
  rho   : i32,
  lam   : i32,
  rinv  : i32,
  _pad1 : i32,
  _pad2 : i32,
  _pad3 : u32,
}
struct Params {
  n         : u32,
  nFluid    : u32,
  gridW     : u32,
  gridH     : u32,
  cellTotal : u32,
  cellShift : u32,
  originX   : i32,
  originY   : i32,
  domW      : i32,
  domH      : i32,
  vmax      : i32,
  maxNbr    : u32,
}

// ---------------------------------- bindings --------------------------------
// 1 uniform + 7 storage = inside the WebGPU default maxStorageBuffersPerShaderStage (8).
@group(0) @binding(0)  var<uniform>             P         : Params;
@group(0) @binding(1)  var<storage, read>       state_in  : array<Particle>;
@group(0) @binding(2)  var<storage, read_write> state_out : array<Particle>;
@group(0) @binding(3)  var<storage, read_write> derived   : array<Derived>;
// scratchA: [0, cellTotal) = cellCount (atomic); [cellTotal, 2*cellTotal+1) = cellStart
@group(0) @binding(4)  var<storage, read_write> scratchA  : array<atomic<u32>>;
// scratchB: [0, n) = cellOf; [n, 2n) = bucketIds
@group(0) @binding(5)  var<storage, read_write> scratchB  : array<u32>;
// nbrBlk: [0, n*MAXNBR) = nbr ids; [n*MAXNBR, n*MAXNBR+n) = nbrN counts
@group(0) @binding(6)  var<storage, read_write> nbrBlk    : array<u32>;
@group(0) @binding(7)  var<storage, read>       luts      : array<i32>;

fn cellCountPtr(c: u32) -> u32 { return c; }
fn cellStartPtr(c: u32) -> u32 { return P.cellTotal + c; }
fn cellOfPtr(i: u32) -> u32 { return i; }
fn bucketPtr(k: u32) -> u32 { return P.n + k; }
fn nbrPtr(fi: u32, k: u32) -> u32 { return fi * P.maxNbr + k; }
fn nbrNPtr(fi: u32) -> u32 { return P.n * P.maxNbr + fi; }

// ---------------------------------- helpers ---------------------------------
fn cell_coord(p : vec2<i32>) -> vec2<i32> {
  let cx = (p.x - P.originX) >> P.cellShift;
  let cy = (p.y - P.originY) >> P.cellShift;
  return vec2<i32>(clamp(cx, 0, i32(P.gridW) - 1), clamp(cy, 0, i32(P.gridH) - 1));
}
fn cell_index(c : vec2<i32>) -> u32 { return u32(c.y) * P.gridW + u32(c.x); }

fn lut_index(d : vec2<i32>) -> i32 {
  let ax = (fp_abs(d.x) + 8) >> PRE;
  let ay = (fp_abs(d.y) + 8) >> PRE;
  if (ax > HS || ay > HS) { return -1; }
  let r2 = ax * ax + ay * ay;
  if (r2 >= H2S) { return -1; }
  return r2 >> KSH;
}
fn grad_q16(gt : i32, d : i32) -> i32 { return fp_mul_s(gt, d, GF + 4u); }
fn wall_clamp(p : vec2<i32>) -> vec2<i32> {
  return clamp(p, vec2<i32>(WALL, WALL), vec2<i32>(P.domW - WALL, P.domH - WALL));
}
fn sym_k(rhoi : i32, rhoj : i32) -> i32 {
  let den = max(RHO_FLOOR_K, fp_add_sat(rhoi, rhoj));
  return clamp(fp_divshift(TWO_RHO0, den, 16u), 0, RINV_MAX);
}

// ---------------------------------- GRID PASSES -----------------------------
@compute @workgroup_size(256)
fn gridCount(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let p = state_in[i].pos;
  let c = cell_coord(p);
  scratchB[cellOfPtr(i)] = cell_index(c);
  atomicAdd(&scratchA[cellCountPtr(cell_index(c))], 1u);
}

// Prefix sum over cellCount -> cellStart. Deterministic: the accumulation
// order is the cell index order, always. One workgroup; cellTotal is capped at
// 1024 by the host, so four sequential 256-wide Hillis-Steele chunks with an
// explicit scalar carry suffice. Thread 0 owns the carry; the whole scan is a
// pure function of cellCount and never of ticket timing.
var<workgroup> scanTmp: array<u32, 256>;
var<workgroup> carry: array<u32, 1>;
@compute @workgroup_size(256)
fn gridScan(@builtin(local_invocation_id) lid : vec3<u32>) {
  let t = lid.x;
  if (t == 0u) { carry[0] = 0u; }
  workgroupBarrier();
  for (var base = 0u; base < P.cellTotal; base = base + 256u) {
    let i = base + t;
    var v = 0u;
    if (i < P.cellTotal) { v = atomicLoad(&scratchA[cellCountPtr(i)]); }
    scanTmp[t] = v;
    workgroupBarrier();
    for (var off = 1u; off < 256u; off = off << 1u) {
      var add = 0u;
      if (t >= off) { add = scanTmp[t - off]; }
      workgroupBarrier();
      scanTmp[t] = scanTmp[t] + add;
      workgroupBarrier();
    }
    let total = scanTmp[255];
    if (i < P.cellTotal) {
      atomicStore(&scratchA[cellStartPtr(i + 1u)], carry[0] + scanTmp[t]);
    }
    workgroupBarrier();
    if (t == 0u) { carry[0] = carry[0] + total; }
    workgroupBarrier();
    // cellStart[0] must be 0 for the first chunk; write it once.
    if (base == 0u && t == 0u) { atomicStore(&scratchA[cellStartPtr(0u)], 0u); }
    workgroupBarrier();
  }
}

@compute @workgroup_size(256)
fn gridSort(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let cidx = scratchB[cellOfPtr(i)];
  // Ticket order within a cell is unspecified; consumers SORT before summing.
  let slot = atomicSub(&scratchA[cellCountPtr(cidx)], 1u) - 1u;
  let idx  = atomicLoad(&scratchA[cellStartPtr(cidx)]) + slot;
  scratchB[bucketPtr(idx)] = i;
}

@compute @workgroup_size(256)
fn buildNbr(@builtin(global_invocation_id) gid : vec3<u32>) {
  let wallCount = P.n - P.nFluid;
  let fi = gid.x;
  if (fi >= P.nFluid) { return; }
  let i = wallCount + fi;
  var pi = state_in[i].pos;
  let ci = cell_coord(pi);

  // CANONICAL SELECTION: the neighbour list is the MAXNBR smallest particle
  // ids inside the 3x3 cell neighbourhood within kernel reach, kept as a
  // sorted register array as we scan. `gridSort`'s atomic-ticket order varies
  // between runs AND between devices, so any rule that depends on bucket
  // order — which candidates make the 48-cap, or which order they accumulate
  // in — would make the hash unreproducible. (Confirmed when twin engines
  // diverged the frame the water got crowded: sorted-but-arrival-capped lists
  // still held different candidate sets.)
  var keep: array<u32, 48>;
  var nn: u32 = 0u;

  for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
    for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
      let cy = ci.y + dy; let cx = ci.x + dx;
      if (cx < 0 || cy < 0 || cx >= i32(P.gridW) || cy >= i32(P.gridH)) { continue; }
      let ci2 = u32(cy) * P.gridW + u32(cx);
      let start = atomicLoad(&scratchA[cellStartPtr(ci2)]);
      let end   = select(P.n, atomicLoad(&scratchA[cellStartPtr(ci2 + 1u)]), ci2 + 1u < P.cellTotal);
      for (var k: u32 = start; k < end; k = k + 1u) {
        let j = scratchB[bucketPtr(k)];
        if (j == i) { continue; }
        let d = vec2<i32>(fp_sub_sat(pi.x, state_in[j].pos.x), fp_sub_sat(pi.y, state_in[j].pos.y));
        let li = lut_index(d);
        if (li < 0) { continue; }
        // Insert j into the ascending `keep` array, evicting the current max
        // when full. Pure function of the candidate SET, never of scan order.
        if (nn < MAXNBR) {
          var b = nn;
          loop {
            if (b == 0u) { break; }
            if (keep[b - 1u] <= j) { break; }
            keep[b] = keep[b - 1u];
            b = b - 1u;
          }
          keep[b] = j;
          nn = nn + 1u;
        } else if (j < keep[MAXNBR - 1u]) {
          var b = MAXNBR - 1u;
          loop {
            if (b == 0u) { break; }
            if (keep[b - 1u] <= j) { break; }
            keep[b] = keep[b - 1u];
            b = b - 1u;
          }
          keep[b] = j;
        }
      }
    }
  }

  for (var a: u32 = 0u; a < nn; a = a + 1u) {
    nbrBlk[nbrPtr(fi, a)] = keep[a];
  }
  nbrBlk[nbrNPtr(fi)] = nn;
}

// ---------------------------------- PREDICT ---------------------------------
@compute @workgroup_size(256)
fn predict(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  let wallCount = P.n - P.nFluid;
  if (i < wallCount) {
    derived[i].pred = state_in[i].pos;
    derived[i].vtmp = vec2<i32>(0, 0);
    return;
  }
  if (i >= wallCount + P.nFluid) { return; }
  let s = state_in[i];
  let grav_vx = s.vel.x;
  let grav_vy = fp_add_sat(s.vel.y, -W_GRAV);
  let px = fp_add_sat(s.pos.x, grav_vx);
  let py = fp_add_sat(s.pos.y, grav_vy);
  derived[i].pred = wall_clamp(vec2<i32>(px, py));
  derived[i].vtmp = s.vel;
}

// ---------------------------------- SOLVE A ---------------------------------
@compute @workgroup_size(256)
fn solveA(@builtin(global_invocation_id) gid : vec3<u32>) {
  let wallCount = P.n - P.nFluid;
  let fi = gid.x;
  if (fi >= P.nFluid) { return; }
  let i = wallCount + fi;
  let pi = derived[i].pred;
  var rho: i32 = luts[OFF_W];
  let nn = nbrBlk[nbrNPtr(fi)];

  for (var k: u32 = 0u; k < nn; k = k + 1u) {
    let j = nbrBlk[nbrPtr(fi, k)];
    let pj = select(state_in[j].pos, derived[j].pred, j >= wallCount);
    let d = vec2<i32>(fp_sub_sat(pi.x, pj.x), fp_sub_sat(pi.y, pj.y));
    let li = lut_index(d);
    if (li < 0) { continue; }
    rho = fp_add_sat(rho, luts[OFF_W + u32(li)]);
  }
  derived[i].rho = rho;

  let C = fp_shr_rne(fp_sub_sat(rho, RHO0), 4);
  if (C <= 0) { derived[i].lam = 0; return; }
  let cc = clamp(C, -C_MAX, C_MAX);
  let lam = fp_divshift(cc, W_EPS, 20u);
  derived[i].lam = clamp(lam, NEG_LAM_MAX, LAM_MAX);
}

// ---------------------------------- SOLVE B ---------------------------------
@compute @workgroup_size(256)
fn solveB(@builtin(global_invocation_id) gid : vec3<u32>) {
  let wallCount = P.n - P.nFluid;
  let fi = gid.x;
  if (fi >= P.nFluid) { return; }
  let i = wallCount + fi;
  let lam_i = derived[i].lam;
  let pi = derived[i].pred;
  let rhoi = derived[i].rho;
  var dpx: i32 = 0; var dpy: i32 = 0;
  let nn = nbrBlk[nbrNPtr(fi)];

  for (var k: u32 = 0u; k < nn; k = k + 1u) {
    let j = nbrBlk[nbrPtr(fi, k)];
    let pj = select(state_in[j].pos, derived[j].pred, j >= wallCount);
    let d = vec2<i32>(fp_sub_sat(pi.x, pj.x), fp_sub_sat(pi.y, pj.y));
    let li = lut_index(d);
    if (li < 0) { continue; }
    let lam_j = derived[j].lam;
    if (lam_i == 0 && lam_j == 0) { continue; }

    let rhoj = derived[j].rho;
    let lam = fp_shr_rne(fp_add_sat(lam_i, lam_j), 1);
    let gt = luts[OFF_G + u32(li)];
    let kij = sym_k(rhoi, rhoj);
    let klam = fp_mul_s(kij, lam, 16u);
    dpx = fp_add_sat(dpx, fp_mul_s(klam, grad_q16(gt, d.x), GF + 4u));
    dpy = fp_add_sat(dpy, fp_mul_s(klam, grad_q16(gt, d.y), GF + 4u));

    let ct = luts[OFF_C + u32(li)];
    let cohW = (W_GAMMACOH + W_GAMMACOH) >> 1;
    dpx = fp_add_sat(dpx, fp_mul_s(fp_mul_s(cohW, ct, CF), d.x, CF));
    dpy = fp_add_sat(dpy, fp_mul_s(fp_mul_s(cohW, ct, CF), d.y, CF));

    let curW = (W_GAMMACUR + W_GAMMACUR) >> 1;
    let cur = fp_mul_s(curW, ct, CF);
    dpx = fp_add_sat(dpx, fp_mul_s(-cur, d.x, CF));
    dpy = fp_add_sat(dpy, fp_mul_s(-cur, d.y, CF));
  }

  derived[i].dp.x = clamp(dpx, NEG_DP_TERM, DP_TERM);
  derived[i].dp.y = clamp(dpy, NEG_DP_TERM, DP_TERM);
}

// ---------------------------------- APPLY DP -------------------------------
@compute @workgroup_size(256)
fn applyDp(@builtin(global_invocation_id) gid : vec3<u32>) {
  let wallCount = P.n - P.nFluid;
  let fi = gid.x;
  if (fi >= P.nFluid) { return; }
  let i = wallCount + fi;
  let dp = derived[i].dp;
  var px = fp_add_sat(derived[i].pred.x, dp.x);
  var py = fp_add_sat(derived[i].pred.y, dp.y);
  derived[i].pred = wall_clamp(vec2<i32>(px, py));
  derived[i].dp = vec2<i32>(0, 0);
}

// ---------------------------------- FINALIZE -------------------------------
@compute @workgroup_size(256)
fn finalize(@builtin(global_invocation_id) gid : vec3<u32>) {
  let wallCount = P.n - P.nFluid;
  let fi = gid.x;
  if (fi >= P.nFluid) { return; }
  let i = wallCount + fi;
  let pred = derived[i].pred;
  let old  = state_in[i].pos;
  var vel  = vec2<i32>(fp_sub_sat(pred.x, old.x), fp_sub_sat(pred.y, old.y));

  let nn = nbrBlk[nbrNPtr(fi)];
  var dvx: i32 = 0; var dvy: i32 = 0;
  for (var k: u32 = 0u; k < nn; k = k + 1u) {
    let j = nbrBlk[nbrPtr(fi, k)];
    let d = vec2<i32>(fp_sub_sat(pred.x, derived[j].pred.x), fp_sub_sat(pred.y, derived[j].pred.y));
    let li = lut_index(d);
    if (li < 0) { continue; }
    let w = luts[OFF_W + u32(li)];
    let dv = vec2<i32>(
      fp_sub_sat(derived[j].vtmp.x, derived[i].vtmp.x),
      fp_sub_sat(derived[j].vtmp.y, derived[i].vtmp.y)
    );
    let xs = fp_mul_s(W_XSPH, w, 16u);
    dvx = fp_add_sat(dvx, fp_mul_s(xs, dv.x, 16u));
    dvy = fp_add_sat(dvy, fp_mul_s(xs, dv.y, 16u));
  }
  vel.x = fp_add_sat(vel.x, dvx);
  vel.y = fp_add_sat(vel.y, dvy);
  vel = clamp(vel, vec2<i32>(-P.vmax, -P.vmax), vec2<i32>(P.vmax, P.vmax));

  state_out[i].pos = pred;
  state_out[i].vel = vel;
  state_out[i]._pad0 = state_in[i]._pad0;
  state_out[i]._pad1 = state_in[i]._pad1;
}

// ---------------------------------- COPY BOUNDARY --------------------------
@compute @workgroup_size(256)
fn copyBoundary(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  let wallCount = P.n - P.nFluid;
  if (i >= P.n || i >= wallCount) { return; }
  state_out[i] = state_in[i];
}
