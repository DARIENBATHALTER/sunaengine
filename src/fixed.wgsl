// SunaEngine — real SunaBox (aether) engine code, extracted UNMODIFIED below this header
// from aether commit e41e06f. Verify: diff <(tail -n +6 src/fixed.wgsl) <aether>/src/fixed.wgsl
// Copyright (C) 2026 Darien Bathalter
// SPDX-License-Identifier: AGPL-3.0-only — see LICENSE. NO WARRANTY.

// ============================================================================
// aether — fixed-point numeric core (WGSL)                     [implementer E]
// ----------------------------------------------------------------------------
// BINDING SPEC v1, section 2. This file is the ONLY source of arithmetic for
// src/sim.wgsl. It is concatenated AHEAD of sim.wgsl by engine.js (WGSL has no
// #include), so it must be self-contained and must declare exactly ONE binding
// (fp_dbg at 15, strippable).
//
// ALL simulation state is signed 32-bit fixed point. No floating point below.
// Rounding rule everywhere: ROUND-HALF-TO-EVEN (RNE). Overflow rule: SATURATE.
// RNE is symmetric — RNE(-x) == -RNE(x) — which is what makes fp_mul_s(-a,b,s)
// == -fp_mul_s(a,b,s) and therefore makes pair forces exactly antisymmetric.
//
// Pinned formats (spec 1.1):
//   Q16.16  ONE16 = 65536      all geometry and kinematics
//   Q8.24   ONE24 = 16777216   reserved; NOT used by the v1 solver
//   Q4.28   ONE28 = 268435456  internal to fp_rsqrt16 only, never stored
//
// Primitive that everything is built on: exact 32x32->64 signed multiply, then
// arithmetic shift right by s with RNE, then saturate to i32.
//
// CALLER RULES (spec 2.7) — not enforceable by the type system:
//   1. axis-reject before squaring
//   2. `s` in every fp_*_s call is a COMPILE-TIME CONSTANT in 1..31
//   3. never negate with unary `-`; use fp_neg_sat
//   4. never call bare abs() on a value that could be I32_MIN; use fp_abs
//   5. every signed >> is fp_shr_rne except the three whitelisted sites
//   6. no float scalars, float vectors or float bitcasts in this file or sim.wgsl
//      (spec 8.4 greps for those tokens; this file must contain none, even in comments)
//   7. no wave/lane-collective builtins of any kind (that whole family is banned;
//      lane counts are not architecturally guaranteed)
//   8. bitcast<u32>(x), never u32(x), when reinterpreting an i32 bit pattern
// ============================================================================

const I32_MAX: i32 = 2147483647;
const I32_MIN: i32 = -2147483647 - 1;   // literal -2147483648 does not parse
const ONE16: i32 = 65536;
const ONE24: i32 = 16777216;
const ONE28: i32 = 268435456;

// 2^30, the numerator of the fp_divshift reciprocal.
const FP_R30: i32 = 1073741824;

// ------------------------------------------------------------------- debug
// RELEASE STRIP — the exact string operation engine.js must perform (spec 2.5):
//
//   src = src.replace(/\/\/@DBG_BEGIN[\s\S]*?\/\/@DBG_END/, 'fn fp_flag(slot: u32) { }');
//
// i.e. replace the marked block INCLUSIVE with an empty fp_flag stub, and drop
// binding 15 from every bind group layout. A naive delete would leave the
// fp_flag call sites undefined and fail compilation — the stub is required.
// test/fixed_test.html compiles the stripped source and asserts it produces
// byte-identical output to the instrumented build.
//
// The goldens must be recorded with debug ENABLED and re-verified with it
// STRIPPED; the two must produce identical hashes.
//@DBG_BEGIN
const FP_DEBUG: bool = true;
// slot 0 = mul/shift saturation          slot 4 = rsqrt of non-positive
// slot 1 = divide saturation / bad shift slot 5 = add/sub/neg saturation
// slot 2 = divide by zero                slot 6 = neighbour-list truncation (sim)
// slot 3 = sqrt of negative              slot 7 = cell/scan capacity overflow (sim)
// slot 8 = bond row saturation (sim, §18)    slots 9..15 = RESERVED
//
// §18 WIDENED THIS 8 -> 16 AND TOOK SLOT 8. It did NOT take slot 7: that is
// scanTop's cell/scan capacity detector, and `engine.js assertSpecValidRegime()`
// hard-codes the names POSITIONALLY, so a collision there would corrupt the one
// function that decides whether a run is a valid keystream. Slots 0..7 keep
// their exact meaning AND their exact position. Keep the two lists in step.
@group(0) @binding(15) var<storage, read_write> fp_dbg: array<atomic<u32>, 16>;
fn fp_flag(slot: u32) { if (FP_DEBUG) { atomicAdd(&fp_dbg[slot], 1u); } }
//@DBG_END

// ---------------------------------------------------------------- 64-bit core
// Two's-complement 64-bit held as (hi, lo). Interpreted signed or unsigned by
// the consuming function; the bit pattern is identical either way.
struct I64 { hi: u32, lo: u32 };

// Shift left by s in 0..=63. Larger s is not defined.
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
fn u64_sub(a: I64, b: I64) -> I64 {
  let lo = a.lo - b.lo;
  let br = select(0u, 1u, a.lo < b.lo);
  return I64(a.hi - b.hi - br, lo);
}
fn u64_ge(a: I64, b: I64) -> bool {
  if (a.hi != b.hi) { return a.hi > b.hi; }
  return a.lo >= b.lo;
}

// |v| as an exact u32 magnitude. Correct for I32_MIN (yields 2^31).
fn u32abs(v: i32) -> u32 {
  if (v < 0) { return (~bitcast<u32>(v)) + 1u; }
  return bitcast<u32>(v);
}

// Exact unsigned 32x32 -> 64. Split into 16-bit halves; every partial product
// fits in u32, carries propagated explicitly. u32 arithmetic in WGSL is defined
// to wrap mod 2^32, which is exactly what the carry detection relies on.
fn u64_mul_u32(a: u32, b: u32) -> I64 {
  let a0 = a & 0xFFFFu; let a1 = a >> 16u;
  let b0 = b & 0xFFFFu; let b1 = b >> 16u;
  let p00 = a0 * b0;
  let p01 = a0 * b1;
  let p10 = a1 * b0;
  let p11 = a1 * b1;
  let mid = p01 + p10;                          // may wrap; bit 32 recovered below
  let midCarry = select(0u, 1u, mid < p01);
  let lo = p00 + (mid << 16u);
  let loCarry = select(0u, 1u, lo < p00);
  let hi = p11 + (mid >> 16u) + (midCarry << 16u) + loCarry;
  return I64(hi, lo);
}

// Exact signed 32x32 -> 64.
// a = ua - 2^32*sa, b = ub - 2^32*sb  =>  a*b == ua*ub - 2^32*(sa*ub + sb*ua)  (mod 2^64)
fn i64_mul_i32(a: i32, b: i32) -> I64 {
  let ua = bitcast<u32>(a);
  let ub = bitcast<u32>(b);
  var r = u64_mul_u32(ua, ub);
  if (a < 0) { r.hi = r.hi - ub; }
  if (b < 0) { r.hi = r.hi - ua; }
  return r;
}

// Arithmetic shift right by s (1..=31 ONLY) of a signed 64-bit value, rounded
// half-to-even, saturated into i32.
//
// Note: an arithmetic shift is floor(). The remainder `rem` is therefore always
// in [0, 2^s) even for negative inputs, so the same tie-break code yields the
// symmetric RNE(-x) == -RNE(x) behaviour required to match the BigInt reference.
fn i64_shr_rne_sat(v: I64, s: u32) -> i32 {
  let hiS = bitcast<u32>(bitcast<i32>(v.hi) >> s);
  let loS = (v.lo >> s) | (v.hi << (32u - s));
  let mask = (1u << s) - 1u;
  let rem  = v.lo & mask;
  let half = 1u << (s - 1u);
  var inc: u32 = 0u;
  if (rem > half) { inc = 1u; }
  else if (rem == half) { inc = loS & 1u; }
  let lo2 = loS + inc;
  var hi2 = hiS;
  if (inc == 1u && lo2 == 0u) { hi2 = hi2 + 1u; }
  let sgn = lo2 >> 31u;
  if (hi2 == 0u && sgn == 0u) { return bitcast<i32>(lo2); }
  if (hi2 == 0xFFFFFFFFu && sgn == 1u) { return bitcast<i32>(lo2); }
  fp_flag(0u);
  if (bitcast<i32>(v.hi) < 0) { return I32_MIN; }
  return I32_MAX;
}

// ------------------------------------------------------------------ multiply
// s MUST be a compile-time constant in 1..31 at every call site (spec 2.7 r2).
fn fp_mul_s(a: i32, b: i32, s: u32) -> i32 { return i64_shr_rne_sat(i64_mul_i32(a, b), s); }
fn fp_mul16(a: i32, b: i32) -> i32 { return fp_mul_s(a, b, 16u); }   // Q16.16 x Q16.16 -> Q16.16
fn fp_mul24(a: i32, b: i32) -> i32 { return fp_mul_s(a, b, 24u); }   // Q8.24  x Q8.24  -> Q8.24
// Cross-format: Q8.24 x Q16.16 -> Q16.16 is fp_mul_s(q24, q16, 24u).
// Cross-format: Q16.16 x Q16.16 -> Q8.24 is fp_mul_s(a, b, 8u).

// ------------------------------------------------------------ add / subtract
fn fp_add_sat(a: i32, b: i32) -> i32 {
  let s = bitcast<i32>(bitcast<u32>(a) + bitcast<u32>(b));
  if (((a ^ s) & (b ^ s)) < 0) {
    fp_flag(5u);
    if (a < 0) { return I32_MIN; }
    return I32_MAX;
  }
  return s;
}
// WGSL unary `-` on i32 WRAPS: -(I32_MIN) == I32_MIN. That silently flips the
// sign of a saturated force and breaks Newton's third law. Always negate with this.
fn fp_neg_sat(a: i32) -> i32 {
  if (a == I32_MIN) { fp_flag(5u); return I32_MAX; }
  return -a;
}
fn fp_sub_sat(a: i32, b: i32) -> i32 {
  let s = bitcast<i32>(bitcast<u32>(a) - bitcast<u32>(b));
  if (((a ^ b) & (a ^ s)) < 0) {
    fp_flag(5u);
    if (a < 0) { return I32_MIN; }
    return I32_MAX;
  }
  return s;
}
// Bare abs() on I32_MIN is UB in WGSL. This is the only legal absolute value.
fn fp_abs(a: i32) -> i32 { if (a < 0) { return fp_neg_sat(a); } return a; }

// ------------------------------------------------------ RNE right-shift of i32
// The ONLY legal signed >> in simulation code, apart from the three whitelisted
// sites in spec 2.7 rule 5. s in 1..31.
fn fp_shr_rne(x: i32, s: u32) -> i32 {
  let q    = x >> s;                                  // arithmetic shift == floor
  let rem  = bitcast<u32>(x) & ((1u << s) - 1u);      // floor remainder, always in [0, 2^s)
  let half = 1u << (s - 1u);
  var inc: i32 = 0;
  if (rem > half) { inc = 1; }
  else if (rem == half && (q & 1) == 1) { inc = 1; }
  return q + inc;                                     // |q| <= 2^30, cannot overflow
}

// ------------------------------------------------------------------- divide
// Exact, correctly-rounded (RNE) fixed-point divide: result = RNE(a * 2^s / b).
// 64/32 restoring division, fixed 64 iterations, input-independent cost.
// BANNED in pair loops (spec C8) — ~7.9x the cost of a multiply. Offline/tooling.
fn fp_div_s(a: i32, b: i32, s: u32) -> i32 {
  if (b == 0) {
    fp_flag(2u);
    if (a == 0) { return 0; }
    if (a > 0) { return I32_MAX; }
    return I32_MIN;
  }
  let neg = (a < 0) != (b < 0);
  let ua = u32abs(a);
  let ub = u32abs(b);
  let n = u64_shl(I64(0u, ua), s);

  var rem: u32 = 0u;
  var q: u32 = 0u;
  var ovf: bool = false;
  for (var i: i32 = 63; i >= 0; i = i - 1) {
    var bit: u32;
    if (i >= 32) { bit = (n.hi >> u32(i - 32)) & 1u; }
    else         { bit = (n.lo >> u32(i)) & 1u; }
    let carry = rem >> 31u;                    // bit shifted out of the 32-bit remainder
    rem = (rem << 1u) | bit;
    var one: u32 = 0u;
    if (carry == 1u || rem >= ub) { rem = rem - ub; one = 1u; }
    if (i >= 32) { if (one == 1u) { ovf = true; } }   // quotient bit above 2^32
    else         { q = (q << 1u) | one; }
  }
  // RNE on the exact remainder: compare 2*rem against ub without overflowing.
  let other = ub - rem;                        // rem < ub, so this is >= 1
  var incq: bool = false;
  if (rem > other) { incq = true; }
  else if (rem == other && (q & 1u) == 1u) { incq = true; }
  if (incq) {
    if (q == 0xFFFFFFFFu) { ovf = true; q = 0u; } else { q = q + 1u; }
  }
  if (ovf) { fp_flag(1u); if (neg) { return I32_MIN; } return I32_MAX; }
  if (neg) {
    if (q > 2147483648u) { fp_flag(1u); return I32_MIN; }
    if (q == 2147483648u) { return I32_MIN; }
    return -bitcast<i32>(q);
  }
  if (q > 2147483647u) { fp_flag(1u); return I32_MAX; }
  return bitcast<i32>(q);
}
fn fp_div16(a: i32, b: i32) -> i32 { return fp_div_s(a, b, 16u); }
fn fp_div24(a: i32, b: i32) -> i32 { return fp_div_s(a, b, 24u); }

// THE solver divide: approximately floor(num * 2^k / den) via a clz-normalised
// 15-bit reciprocal. One i32 divide plus one 32x32->64 multiply, fixed cost, no
// data-dependent iteration count. Relative error <= 2^-14.
//
// Callers guarantee den >= 2^15 (eps >= 32768, rho clamped at 1<<15, etc), which
// keeps the internal shift s = 30 + e - k inside 1..31 for every solver call.
// Out-of-contract inputs are handled deterministically and flagged on dbg[1]:
//   den == 0        -> dbg[2], sign-saturated
//   den <  0        -> dbg[1], computed on |den| with the numerator negated
//   s   >  31       -> dbg[1], returns 0        (|true result| < 1)
//   s   <   1       -> dbg[1], returns sign-saturated
//
// NOTE (deviation from the spec pseudocode, deliberate and required):
//  * the spec writes `select(den << u32(-e), den >> u32(e), e >= 0)`. select()
//    evaluates BOTH arms, so for e >= 0 the dead arm computes u32(-e) which is a
//    huge unsigned value and makes the shift amount >= 32 — INDETERMINATE in
//    WGSL, not an error. An if/else is mandatory here.
//  * the spec writes fp_mul_s(num, Rn, u32(30 + e - k)) with a RUNTIME shift,
//    which violates caller rule 2. The i64_shr_rne_sat call is made directly
//    with a range-validated runtime shift instead.
fn fp_divshift(num: i32, den: i32, k: u32) -> i32 {
  if (den == 0) {
    fp_flag(2u);
    if (num == 0) { return 0; }
    if (num > 0) { return I32_MAX; }
    return I32_MIN;
  }
  var d = den;
  var nn = num;
  if (d < 0) { fp_flag(1u); d = fp_abs(d); nn = fp_neg_sat(nn); }
  // d > 0. e = (index of top set bit) - 15, so dn is always in [2^15, 2^16).
  let e = 16 - i32(countLeadingZeros(bitcast<u32>(d)));
  var dn: i32;
  if (e >= 0) { dn = d >> bitcast<u32>(e); } else { dn = d << bitcast<u32>(-e); }
  let Rn = FP_R30 / dn;                       // exact i32 divide; Rn in (2^14, 2^15]
  let s  = 30 + e - i32(k);
  if (s > 31) { fp_flag(1u); return 0; }
  if (s < 1) {
    fp_flag(1u);
    if (nn == 0) { return 0; }
    if (nn > 0) { return I32_MAX; }
    return I32_MIN;
  }
  return i64_shr_rne_sat(i64_mul_i32(nn, Rn), bitcast<u32>(s));
}

// --------------------------------------------------------------------- sqrt
// NONE of the root functions below appear in the v1 solver. They are retained
// because they are validated and because v2 will need them. Their presence in
// sim.wgsl is a review failure.

// Exact integer sqrt of a 64-bit unsigned value, digit-recurrence (2 bits per
// step), fixed 32 iterations. Returns floor(sqrt(n)).
fn isqrt64(n: I64) -> u32 {
  var rem = I64(0u, 0u);
  var root: u32 = 0u;
  for (var i: i32 = 31; i >= 0; i = i - 1) {
    let sh = u32(i) * 2u;
    var bits: u32;
    if (sh >= 32u) { bits = (n.hi >> (sh - 32u)) & 3u; }
    else           { bits = (n.lo >> sh) & 3u; }
    rem = u64_add(u64_shl(rem, 2u), I64(0u, bits));
    let trial = u64_add(u64_shl(I64(0u, root), 2u), I64(0u, 1u));  // 4*root + 1
    root = root << 1u;
    if (u64_ge(rem, trial)) { rem = u64_sub(rem, trial); root = root | 1u; }
  }
  return root;
}

// Round floor(sqrt(n)) to nearest. Exact ties are impossible for integer n
// (since (r+1/2)^2 is never an integer), so no tie-break is needed.
fn isqrt64_round(n: I64) -> u32 {
  let r = isqrt64(n);
  if (r == 0xFFFFFFFFu) { return r; }
  let d = u64_sub(n, u64_mul_u32(r, r));
  if (u64_ge(d, I64(0u, r + 1u))) { return r + 1u; }
  return r;
}

// sqrt in format Qs: sqrt(x/2^s) * 2^s == sqrt(x * 2^s). Correctly rounded.
fn fp_sqrt_s(x: i32, s: u32) -> i32 {
  if (x < 0) { fp_flag(3u); return 0; }
  if (x == 0) { return 0; }
  let res = isqrt64_round(u64_shl(I64(0u, bitcast<u32>(x)), s));
  if (res > 2147483647u) { return I32_MAX; }
  return bitcast<i32>(res);
}
fn fp_sqrt16(x: i32) -> i32 { return fp_sqrt_s(x, 16u); }
fn fp_sqrt24(x: i32) -> i32 { return fp_sqrt_s(x, 24u); }

// Exact hypotenuse in Q16.16, overflow-free for every i32 input pair:
// |(ax,ay)| raw == round(sqrt(ax_raw^2 + ay_raw^2)); the squares are kept in 64 bits.
fn fp_length16(ax: i32, ay: i32) -> i32 {
  let n = u64_add(i64_mul_i32(ax, ax), i64_mul_i32(ay, ay));
  let res = isqrt64_round(n);
  if (res > 2147483647u) { fp_flag(0u); return I32_MAX; }
  return bitcast<i32>(res);
}

// -------------------------------------------------------------------- rsqrt
// Seed table for Newton-Raphson, Q4.28, indexed by the top 6 bits of a mantissa
// normalized into [0.25, 1). Entries 0..15 are unreachable by construction.
// FROZEN LITERAL: these are the exact values produced by fixed_ref.mjs's
// buildRsqrtSeed() (round(2^28 / sqrt((i+0.5)/64)) computed in BigInt). They are
// embedded rather than injected so that fixed.wgsl is a self-contained string
// that engine.js can concatenate verbatim.
var<private> RSQRT_SEED: array<i32, 64> = array<i32, 64>(
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  528673928, 513346779, 499279807, 486309283, 474299793, 463138234, 452729304, 442992052,
  433857214, 425265132, 417164101, 409509064, 402260543, 395383788, 388848068, 382626098,
  376693551, 371028664, 365611896, 360425649, 355454023, 350682611, 346098328, 341689252,
  337444500, 333354112, 329408954, 325600630, 321921409, 318364157, 314922282, 311589677,
  308360681, 305230035, 302192844, 299244551, 296380901, 293597920, 290891891, 288259331,
  285696975, 283201757, 280770796, 278401379, 276090955, 273837113, 271637583, 269490216
);

// 1/sqrt(x) for x in Q16.16, result in Q16.16.
// Fixed cost: 1 countLeadingZeros, 1 table lookup, exactly 3 NR iterations.
// Range-reduce x = m * 4^e with m in [0.25,1)  =>  rsqrt(x) = rsqrt(m) * 2^-e.
fn fp_rsqrt16(x: i32) -> i32 {
  if (x <= 0) { fp_flag(4u); return I32_MAX; }
  let ux = bitcast<u32>(x);
  let p = i32(31u - countLeadingZeros(ux));   // index of the top set bit, 0..30
  let e = (p - 14) >> 1;                       // arithmetic shift == floor div 2
  let sh = 12 - 2 * e;
  var xn: u32;
  if (sh >= 0) { xn = ux << u32(sh); } else { xn = ux >> u32(-sh); }
  // xn is now the mantissa in Q4.28, guaranteed in [2^26, 2^28)  ==  [0.25, 1).
  var y = RSQRT_SEED[xn >> 22u];               // index in [16,64), ~6 good bits
  for (var k: i32 = 0; k < 3; k = k + 1) {
    let y2  = fp_mul_s(y, y, 28u);                       // y^2   in (1,4]
    let xy2 = fp_mul_s(bitcast<i32>(xn), y2, 28u);       // x*y^2 in (0.25,4]
    let t   = 3 * ONE28 - xy2;                           // 3-x*y^2 in [-1,2.75]
    y = fp_mul_s(y, t, 29u);                             // y*(3-x*y^2)/2
  }
  return i64_shr_rne_sat(I64(0u, bitcast<u32>(y)), u32(12 + e));
}
