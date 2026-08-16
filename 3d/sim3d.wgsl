// ============================================================================
// aether-3d — deterministic 3D fluid, integer fixed point
// ----------------------------------------------------------------------------
// Concatenated AFTER src/fixed.wgsl (verbatim from the SunaBox engine), which
// supplies every arithmetic primitive used below. No f32 appears in this file.
//
// The method is SunaBox's, one dimension up:
//   predict -> gridCount -> scan -> scatter -> CANONICALIZE -> buildNbr
//           -> (density -> dp -> applyDp) x ITERS -> integrate
//
// The load-bearing pass is canonicalize. `scatter` allocates bucket slots with
// the RETURN VALUE of an atomicAdd, which is scheduler order and therefore
// vendor- and run-dependent. canonicalize re-ranks each cell into ascending
// particle id before the neighbour list is built, so nothing downstream can
// ever observe scatter order. Defeating it is exposed as a control in the UI.
//
// Order-independence rules inherited from the 2D solver and kept here:
//   - integer accumulation only (associative, commutative, wrapping-free here)
//   - PER-TERM clamps, never a clamp on a running total
//   - neighbour truncation drops the tail of the CANONICAL order, not of an
//     arbitrary one, so an overflowing cell is still bit-exact
// ============================================================================

struct Params {
  n           : u32,
  cellTotal   : u32,
  gx          : u32,
  gy          : u32,
  gz          : u32,
  offCellOf   : u32,
  offBucket   : u32,
  offSorted   : u32,
  offBlock    : u32,
  nBlocks     : u32,
  defeatCanon : u32,
  frame       : u32,
  gravStep    : i32,
  damp        : i32,
  stirPhase   : i32,
  stirAmp     : i32,
  ballPos     : vec3<i32>,   // vec3 forces 16-byte alignment; byte 64 already is
  ballReach   : i32,
  ballOn      : u32,
};

struct Particle {
  p  : vec4<i32>,   // xyz world position, Q16.16
  v  : vec4<i32>,   // xyz displacement per substep, Q16.16
  pr : vec4<i32>,   // xyz predicted position; w = lambda
  dp : vec4<i32>,   // xyz position correction; w = density
};

@group(0) @binding(0) var<uniform>                        P     : Params;
@group(0) @binding(1) var<storage, read_write> parts : array<Particle>;
@group(0) @binding(2) var<storage, read_write> cellCount : array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> idx   : array<u32>;
@group(0) @binding(4) var<storage, read_write> nbr   : array<u32>;
@group(0) @binding(5) var<storage, read>       luts  : array<i32>;
@group(0) @binding(6) var<storage, read_write> digest : array<u32>;
// binding 15 is fp_dbg, declared by fixed.wgsl.

// ---------------------------------------------------------------- constants
const MAXNBR    : u32 = 48u;
const H         : i32 = 131072;      // smoothing radius, 2.0 wu in Q16.16
const H2Q       : i32 = 262144;      // (H >> 8)^2 — squared radius in reduced units
const LUT_SHIFT : u32 = 7u;          // r2q >> 7 indexes the kernel tables
const LUT_N     : u32 = 2049u;
const OFF_W     : u32 = 0u;
const OFF_G     : u32 = 2049u;
const OFF_P     : u32 = 4098u;       // wall density, indexed by (distance >> 8)
const OFF_PG    : u32 = 4611u;       // wall gradient magnitude, same index
const OFF_BW    : u32 = 5124u;       // ball density, indexed by (r2q >> 12)
const OFF_BG    : u32 = 6148u;       // ball gradient coefficient, same index
const BALL_R2MAX: i32 = 4194304;     // 1024 << 12, the ball table's domain
const OFF_MAT   : u32 = 7172u;       // material rows, MAT_WORDS each
const MAT_WORDS : u32 = 4u;

// Material id rides in p.w, which the solver never otherwise uses.
fn mat_of(i: u32) -> u32 { return u32(parts[i].p.w); }
fn mat_grav(m: u32) -> i32 { return luts[OFF_MAT + m * MAT_WORDS + 0u]; }
fn mat_rhoW(m: u32) -> i32 { return luts[OFF_MAT + m * MAT_WORDS + 1u]; }
fn mat_damp(m: u32) -> i32 { return luts[OFF_MAT + m * MAT_WORDS + 2u]; }

const RHO0      : i32 = 1048576;     // 2^20, the rest density the LUTs are scaled to
const C_MAX     : i32 = 1048576;
const LAM_MAX   : i32 = 4194304;     // 2^22 == 64.0 in Q16.16; a bound, not a working range
const EPS_S     : i32 = 524288;      // PBF relaxation, ~1% of a typical S

// SCALE DERIVATION, so these are checkable rather than tuned:
//   Cq   = (rho - RHO0) >> 4  is the dimensionless constraint C in Q16.16,
//          because RHO0 is 2^20 and 2^20 >> 4 == 2^16.
//   |dC| = g / RHO0  where g is a gradient-table term, so
//   S    = sum |dC|^2 wants g*g >> 40, and we want it carrying 2^24 of headroom
//          for the divide below, hence the accumulator shift of 16.
//   lam  = -Cq / S_normalised  is fp_divshift(Cq, S + eps, 24).
//   dp   = lam * g / RHO0      is fp_mul_s(lam, g, 20).
const GRAD_S    : u32 = 16u;         // gradient coefficient x delta, delta is Q16.16
const S_ACC_S   : u32 = 16u;         // squared-gradient accumulator
const LAM_S     : u32 = 24u;         // the divide's shift, paired with S_ACC_S
const DP_S      : u32 = 20u;         // lambda x gradient -> a Q16.16 displacement

// Per-term ceiling. |g| peaks at SG*h^2 where the spiky shape is steepest, so a
// single term cannot exceed ~7.2e6; 2^23 covers it, and 147 terms of 2^23 stay
// clear of i32. Bounding the TERM is what keeps the sum order-independent.
const S_TERM    : i32 = 8388608;

const MARGIN    : i32 = 32768;       // half a world unit of wall standoff

fn world_max(axis: u32) -> i32 {
  if (axis == 0u) { return i32(P.gx) * H; }
  if (axis == 1u) { return i32(P.gy) * H; }
  return i32(P.gz) * H;
}

fn cell_coord(p: vec3<i32>) -> vec3<i32> {
  return vec3<i32>(p.x / H, p.y / H, p.z / H);
}

fn cell_index(c: vec3<i32>) -> u32 {
  return (u32(c.z) * P.gy + u32(c.y)) * P.gx + u32(c.x);
}

// Distance to wall `w` (axis = w/2, high side when bit 0 set), and the unit
// direction TOWARD that wall. Density rises as you approach a solid, so the
// gradient of rho points into the wall — the same convention fluid neighbours
// use, where g points toward the neighbour.
struct Wall { dist: i32, dir: vec3<i32> };

fn wall_of(p: vec3<i32>, w: u32, gm: i32) -> Wall {
  let axis = w >> 1u;
  let hi   = (w & 1u) == 1u;
  var pos: i32 = p.x;
  if (axis == 1u) { pos = p.y; } else if (axis == 2u) { pos = p.z; }
  let wmax = world_max(axis);
  var d = pos;
  if (hi) { d = wmax - pos; }

  let s = select(fp_neg_sat(gm), gm, hi);
  var dir = vec3<i32>(0, 0, 0);
  if (axis == 0u) { dir.x = s; } else if (axis == 1u) { dir.y = s; } else { dir.z = s; }
  return Wall(max(d, 0), dir);
}

// The cursor ball: a solid sphere, handled with the same machinery as a wall.
// Its table is indexed by CENTRE distance squared, so no square root is needed
// here. g points toward the centre, matching the fluid convention that the
// gradient points the way density increases; lambda is negative under
// compression, so the correction comes back out along -g.
//
// Nothing about this is a special "push force". A moving solid displaces
// particles, and position-based dynamics reads velocity back off (pred - p),
// so the momentum transfer falls out of the constraint solve for free.
struct BallTerm { hit: bool, w: i32, g: vec3<i32> };

fn ball_term(pi: vec3<i32>) -> BallTerm {
  var t = BallTerm(false, 0, vec3<i32>(0, 0, 0));
  if (P.ballOn != 1u) { return t; }

  let dc = P.ballPos - pi;
  if (fp_abs(dc.x) >= P.ballReach ||
      fp_abs(dc.y) >= P.ballReach ||
      fp_abs(dc.z) >= P.ballReach) { return t; }

  let q = dc >> vec3<u32>(8u, 8u, 8u);
  let r2q = q.x * q.x + q.y * q.y + q.z * q.z;
  if (r2q >= BALL_R2MAX) { return t; }

  let bi = u32(r2q) >> 12u;
  let gt = luts[OFF_BG + bi];
  t.hit = true;
  t.w = luts[OFF_BW + bi];
  t.g = vec3<i32>(fp_mul_s(gt, dc.x, GRAD_S),
                  fp_mul_s(gt, dc.y, GRAD_S),
                  fp_mul_s(gt, dc.z, GRAD_S));
  return t;
}

fn clamp_world(p: vec3<i32>) -> vec3<i32> {
  return vec3<i32>(
    clamp(p.x, MARGIN, world_max(0u) - MARGIN),
    clamp(p.y, MARGIN, world_max(1u) - MARGIN),
    clamp(p.z, MARGIN, world_max(2u) - MARGIN));
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 1 — predict
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn predict(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }

  let mi = mat_of(i);
  var v = parts[i].v.xyz;
  v.y = fp_sub_sat(v.y, mat_grav(mi));   // negative grav rises

  // A slow horizontal stir so the surface keeps moving for the camera. Integer
  // sinusoid from the shared BAM table would need fixed.wgsl's rg_cos; a plain
  // triangle wave is enough and is exactly representable.
  if (P.stirAmp != 0) {
    let tri = abs(((P.stirPhase + 32768) & 131071) - 65536) - 32768;   // -32768..32768
    v.x = fp_add_sat(v.x, fp_mul_s(tri, P.stirAmp, 16u));
  }

  let dmp = mat_damp(mi);
  v = vec3<i32>(fp_mul_s(v.x, dmp, 16u),
                fp_mul_s(v.y, dmp, 16u),
                fp_mul_s(v.z, dmp, 16u));

  let pr = clamp_world(vec3<i32>(fp_add_sat(parts[i].p.x, v.x),
                                 fp_add_sat(parts[i].p.y, v.y),
                                 fp_add_sat(parts[i].p.z, v.z)));
  parts[i].v  = vec4<i32>(v, 0);
  parts[i].pr = vec4<i32>(pr, 0);
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 2 — clearCells
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn clearCells(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c >= P.cellTotal) { return; }
  atomicStore(&cellCount[c], 0u);
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 3 — gridCount. Pure counting reduction: exact and order-independent.
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn gridCount(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let c = cell_index(cell_coord(parts[i].pr.xyz));
  idx[P.offCellOf + i] = c;
  atomicAdd(&cellCount[c], 1u);
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 4-6 — exclusive prefix sum over cellCount into idx[0 .. cellTotal]
// ════════════════════════════════════════════════════════════════════════════
var<workgroup> sdata : array<u32, 256>;

fn wg_scan_inclusive(tid: u32) {
  for (var off: u32 = 1u; off < 256u; off = off << 1u) {
    var add: u32 = 0u;
    if (tid >= off) { add = sdata[tid - off]; }
    workgroupBarrier();
    if (tid >= off) { sdata[tid] = sdata[tid] + add; }
    workgroupBarrier();
  }
}

@compute @workgroup_size(256)
fn scanBlock(@builtin(workgroup_id) wg: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid  = lid.x;
  let ct   = P.cellTotal;
  let base = wg.x * 1024u + tid * 4u;

  var v0: u32 = 0u; var v1: u32 = 0u; var v2: u32 = 0u; var v3: u32 = 0u;
  if (base + 0u < ct) { v0 = atomicLoad(&cellCount[base + 0u]); }
  if (base + 1u < ct) { v1 = atomicLoad(&cellCount[base + 1u]); }
  if (base + 2u < ct) { v2 = atomicLoad(&cellCount[base + 2u]); }
  if (base + 3u < ct) { v3 = atomicLoad(&cellCount[base + 3u]); }

  sdata[tid] = v0 + v1 + v2 + v3;
  workgroupBarrier();
  wg_scan_inclusive(tid);

  var run = sdata[tid] - (v0 + v1 + v2 + v3);   // exclusive prefix for this thread
  if (base + 0u < ct) { idx[base + 0u] = run; run = run + v0; }
  if (base + 1u < ct) { idx[base + 1u] = run; run = run + v1; }
  if (base + 2u < ct) { idx[base + 2u] = run; run = run + v2; }
  if (base + 3u < ct) { idx[base + 3u] = run; }

  if (tid == 255u) { idx[P.offBlock + wg.x] = sdata[255u]; }
}

@compute @workgroup_size(256)
fn scanTop(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  var v: u32 = 0u;
  if (tid < P.nBlocks) { v = idx[P.offBlock + tid]; }
  sdata[tid] = v;
  workgroupBarrier();
  wg_scan_inclusive(tid);
  if (tid < P.nBlocks) { idx[P.offBlock + tid] = sdata[tid] - v; }
}

@compute @workgroup_size(256)
fn scanAdd(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c > P.cellTotal) { return; }
  if (c == P.cellTotal) { idx[c] = P.n; return; }     // the tail sentinel
  idx[c] = idx[c] + idx[P.offBlock + (c / 1024u)];
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 7 — scatter. Counting-sort placement.
// atomicAdd's RETURN VALUE is the slot, so this order is scheduler order.
// Nothing downstream is allowed to see it — see canonicalize.
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let c   = idx[P.offCellOf + i];
  let off = atomicAdd(&cellCount[c], 1u);
  idx[P.offBucket + idx[c] + off] = i;
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 8 — canonicalize. THE DETERMINISM KEYSTONE.
// Re-ranks every cell's bucket into ascending particle id, which is a total
// order fixed by identity alone and so is identical on every device and every
// run. A different-but-deterministic order (descending) would be equally
// self-consistent and would yield a DIFFERENT chain, so ascending is wire
// format, not an implementation detail.
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn canonicalize(@builtin(global_invocation_id) gid: vec3<u32>) {
  let s = gid.x;
  if (s >= P.n) { return; }
  let me = idx[P.offBucket + s];
  let c  = idx[P.offCellOf + me];
  let lo = idx[c];
  let hi = idx[c + 1u];
  var rank: u32 = 0u;
  for (var t = lo; t < hi; t = t + 1u) {
    if (idx[P.offBucket + t] < me) { rank = rank + 1u; }
  }
  idx[P.offSorted + lo + rank] = me;
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 9 — buildNbr. Fixed 3x3x3 stencil, dz outer, dy, dx inner, and
// ascending order within each cell. Truncation at MAXNBR drops the tail of
// THIS order and raises dbg[6].
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn buildNbr(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }

  let pi = parts[i].pr.xyz;
  let c  = cell_coord(pi);
  let base = i * MAXNBR;
  var cnt: u32 = 0u;

  for (var dz = -1; dz <= 1; dz = dz + 1) {
    let cz = c.z + dz;
    if (cz < 0 || cz >= i32(P.gz)) { continue; }
    for (var dy = -1; dy <= 1; dy = dy + 1) {
      let cy = c.y + dy;
      if (cy < 0 || cy >= i32(P.gy)) { continue; }
      for (var dx = -1; dx <= 1; dx = dx + 1) {
        let cx = c.x + dx;
        if (cx < 0 || cx >= i32(P.gx)) { continue; }

        let cc = cell_index(vec3<i32>(cx, cy, cz));
        let lo = idx[cc];
        let hi = idx[cc + 1u];
        for (var s = lo; s < hi; s = s + 1u) {
          // The control: reading the raw bucket here reintroduces scatter order.
          var j: u32;
          if (P.defeatCanon == 1u) { j = idx[P.offBucket + s]; }
          else                     { j = idx[P.offSorted + s]; }
          if (j == i) { continue; }

          // Axis-reject BEFORE squaring (fixed.wgsl caller rule 1). fp_abs, not
          // bare abs(), per rule 4 — positions are clamped well inside i32 here,
          // but the discipline is cheap and the exception would have to be argued.
          let d = parts[j].pr.xyz - pi;
          if (fp_abs(d.x) >= H || fp_abs(d.y) >= H || fp_abs(d.z) >= H) { continue; }
          // Whitelisted signed >>: this is FLOOR BUCKETING for a table index, not
          // a rounded rescale. RNE here would be wrong, not merely different.
          let q = d >> vec3<u32>(8u, 8u, 8u);
          let r2q = q.x * q.x + q.y * q.y + q.z * q.z;
          if (r2q >= H2Q) { continue; }

          if (cnt >= MAXNBR) { fp_flag(6u); continue; }
          nbr[base + cnt] = j;
          cnt = cnt + 1u;
        }
      }
    }
  }
  nbr[P.n * MAXNBR + i] = cnt;
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 10 — density and lambda.
// Every clamp here is PER TERM. Clamping a running total would make the sum
// order-dependent and destroy bit-exactness even with a canonical order.
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn density(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }

  let pi   = parts[i].pr.xyz;
  let cnt  = nbr[P.n * MAXNBR + i];
  let base = i * MAXNBR;

  // §37, ported: every kernel term is weighted by ITS OWN particle's volume
  // claim, (W * (256 + rhoW)) >> 8. A swollen bead raises its neighbours'
  // density, the constraint decompresses, and the bead really takes more room.
  // Max term is ~203k * 512 < 2^27, so the shift cannot overflow.
  var rho: i32 = (luts[OFF_W] * (256 + mat_rhoW(mat_of(i)))) >> 8u;
  var S:   i32 = 0;
  var sg = vec3<i32>(0, 0, 0);

  for (var k: u32 = 0u; k < cnt; k = k + 1u) {
    let j = nbr[base + k];
    let d = parts[j].pr.xyz - pi;
    let q = d >> vec3<u32>(8u, 8u, 8u);
    let r2q = q.x * q.x + q.y * q.y + q.z * q.z;
    if (r2q >= H2Q) { continue; }
    let ui = u32(r2q) >> LUT_SHIFT;

    rho = rho + ((luts[OFF_W + ui] * (256 + mat_rhoW(mat_of(j)))) >> 8u);

    let gt = luts[OFF_G + ui];
    let g  = vec3<i32>(fp_mul_s(gt, d.x, GRAD_S),
                       fp_mul_s(gt, d.y, GRAD_S),
                       fp_mul_s(gt, d.z, GRAD_S));
    sg = sg + g;

    S = S + clamp(fp_mul_s(g.x, g.x, S_ACC_S), 0, S_TERM)
          + clamp(fp_mul_s(g.y, g.y, S_ACC_S), 0, S_TERM)
          + clamp(fp_mul_s(g.z, g.z, S_ACC_S), 0, S_TERM);
  }

  // ---- solid walls, as a half-space of lattice matter rather than a clamp
  for (var w: u32 = 0u; w < 6u; w = w + 1u) {
    let probe = wall_of(pi, w, 0);
    if (probe.dist >= H) { continue; }
    let b = u32(probe.dist) >> 8u;
    rho = rho + luts[OFF_P + b];
    let gv = wall_of(pi, w, luts[OFF_PG + b]).dir;
    sg = sg + gv;
    S = S + clamp(fp_mul_s(gv.x, gv.x, S_ACC_S), 0, S_TERM)
          + clamp(fp_mul_s(gv.y, gv.y, S_ACC_S), 0, S_TERM)
          + clamp(fp_mul_s(gv.z, gv.z, S_ACC_S), 0, S_TERM);
  }

  let bt = ball_term(pi);
  if (bt.hit) {
    rho = rho + bt.w;
    sg = sg + bt.g;
    S = S + clamp(fp_mul_s(bt.g.x, bt.g.x, S_ACC_S), 0, S_TERM)
          + clamp(fp_mul_s(bt.g.y, bt.g.y, S_ACC_S), 0, S_TERM)
          + clamp(fp_mul_s(bt.g.z, bt.g.z, S_ACC_S), 0, S_TERM);
  }

  S = S + clamp(fp_mul_s(sg.x, sg.x, S_ACC_S), 0, S_TERM)
        + clamp(fp_mul_s(sg.y, sg.y, S_ACC_S), 0, S_TERM)
        + clamp(fp_mul_s(sg.z, sg.z, S_ACC_S), 0, S_TERM);

  // Compression only. A free-surface particle has S near zero; without this
  // clamp its lambda detonates the surface.
  let Cq = clamp(fp_shr_rne(rho - RHO0, 4u), 0, C_MAX);

  parts[i].dp.w = rho;
  parts[i].pr.w = clamp(fp_neg_sat(fp_divshift(Cq, S + EPS_S, LAM_S)),
                        fp_neg_sat(LAM_MAX), LAM_MAX);
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 11 — position correction, GATHERED.
// Each particle sums its own correction from its own neighbour list, so there
// is no scatter and no intra-dispatch race. dp is written, never pred: pred[j]
// is read here.
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn solveDp(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }

  let pi   = parts[i].pr.xyz;
  let lami = parts[i].pr.w;
  let cnt  = nbr[P.n * MAXNBR + i];
  let base = i * MAXNBR;

  var acc = vec3<i32>(0, 0, 0);

  for (var k: u32 = 0u; k < cnt; k = k + 1u) {
    let j = nbr[base + k];
    let d = parts[j].pr.xyz - pi;
    let q = d >> vec3<u32>(8u, 8u, 8u);
    let r2q = q.x * q.x + q.y * q.y + q.z * q.z;
    if (r2q >= H2Q) { continue; }
    let ui = u32(r2q) >> LUT_SHIFT;

    let gt = luts[OFF_G + ui];
    let g  = vec3<i32>(fp_mul_s(gt, d.x, GRAD_S),
                       fp_mul_s(gt, d.y, GRAD_S),
                       fp_mul_s(gt, d.z, GRAD_S));

    let lsum = fp_add_sat(lami, parts[j].pr.w);
    acc = vec3<i32>(fp_add_sat(acc.x, fp_mul_s(lsum, g.x, DP_S)),
                    fp_add_sat(acc.y, fp_mul_s(lsum, g.y, DP_S)),
                    fp_add_sat(acc.z, fp_mul_s(lsum, g.z, DP_S)));
  }

  // Walls push back with the particle's own lambda: a static solid carries no
  // lambda of its own, so lsum here is lami rather than lami + lamj.
  for (var w: u32 = 0u; w < 6u; w = w + 1u) {
    let probe = wall_of(pi, w, 0);
    if (probe.dist >= H) { continue; }
    let b  = u32(probe.dist) >> 8u;
    let gv = wall_of(pi, w, luts[OFF_PG + b]).dir;
    acc = vec3<i32>(fp_add_sat(acc.x, fp_mul_s(lami, gv.x, DP_S)),
                    fp_add_sat(acc.y, fp_mul_s(lami, gv.y, DP_S)),
                    fp_add_sat(acc.z, fp_mul_s(lami, gv.z, DP_S)));
  }

  let bt = ball_term(pi);
  if (bt.hit) {
    acc = vec3<i32>(fp_add_sat(acc.x, fp_mul_s(lami, bt.g.x, DP_S)),
                    fp_add_sat(acc.y, fp_mul_s(lami, bt.g.y, DP_S)),
                    fp_add_sat(acc.z, fp_mul_s(lami, bt.g.z, DP_S)));
  }

  parts[i].dp = vec4<i32>(acc, parts[i].dp.w);
}

@compute @workgroup_size(256)
fn applyDp(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let pr = clamp_world(vec3<i32>(fp_add_sat(parts[i].pr.x, parts[i].dp.x),
                                 fp_add_sat(parts[i].pr.y, parts[i].dp.y),
                                 fp_add_sat(parts[i].pr.z, parts[i].dp.z)));
  parts[i].pr = vec4<i32>(pr, parts[i].pr.w);
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 12 — integrate
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let pr = parts[i].pr.xyz;
  let p  = parts[i].p.xyz;
  parts[i].v = vec4<i32>(fp_sub_sat(pr.x, p.x),
                         fp_sub_sat(pr.y, p.y),
                         fp_sub_sat(pr.z, p.z), 0);
  parts[i].p = vec4<i32>(pr, parts[i].p.w);   // p.w is the material id — keep it
}

// ════════════════════════════════════════════════════════════════════════════
// DIGEST — striped FNV-1a. Each of 256 lanes owns a FIXED contiguous stripe and
// writes one word; the host folds the 256 words in index order. A tree or
// atomic reduction would be order-dependent and would defeat the purpose.
// ════════════════════════════════════════════════════════════════════════════
const FNV_OFF : u32 = 2166136261u;
const FNV_PR  : u32 = 16777619u;

fn fnv(h0: u32, w: u32) -> u32 {
  var h = h0;
  h = (h ^ ( w        & 0xFFu)) * FNV_PR;
  h = (h ^ ((w >>  8u) & 0xFFu)) * FNV_PR;
  h = (h ^ ((w >> 16u) & 0xFFu)) * FNV_PR;
  h = (h ^ ((w >> 24u) & 0xFFu)) * FNV_PR;
  return h;
}

@compute @workgroup_size(256)
fn digestStripe(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= 256u) { return; }
  let per = (P.n + 255u) / 256u;
  let lo  = k * per;
  var hi  = lo + per;
  if (hi > P.n) { hi = P.n; }

  var h: u32 = FNV_OFF;
  for (var i = lo; i < hi; i = i + 1u) {
    h = fnv(h, bitcast<u32>(parts[i].p.x));
    h = fnv(h, bitcast<u32>(parts[i].p.y));
    h = fnv(h, bitcast<u32>(parts[i].p.z));
    h = fnv(h, bitcast<u32>(parts[i].v.x));
    h = fnv(h, bitcast<u32>(parts[i].v.y));
    h = fnv(h, bitcast<u32>(parts[i].v.z));
  }
  digest[k] = h;
}
