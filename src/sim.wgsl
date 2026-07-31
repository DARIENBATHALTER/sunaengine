// SunaEngine — real SunaBox (aether) engine code, extracted UNMODIFIED below this header
// from aether commit e41e06f. Verify: diff <(tail -n +6 src/sim.wgsl) <aether>/src/sim.wgsl
// Copyright (C) 2026 Darien Bathalter
// SPDX-License-Identifier: AGPL-3.0-only — see LICENSE. NO WARRANTY.

// ============================================================================
// aether — src/sim.wgsl  (implementer F)
// ----------------------------------------------------------------------------
// The 2D deterministic PBF solver: 25 compute entry points, 58 dispatches per
// substep (see BINDING SPEC §5).
//
// v7 (2026-07-26): GAS VENTING / RETIREMENT IN PLACE (§17). EIGHT LINES, no new
// pass, no new buffer, no new dispatch — 58 stays 58. `struct Mat`'s last
// reserved word (3, `vort`) becomes `phantom`; `Particle.flags` gains bit 1
// (FLAG_DEAD); `buildNbr` gains binding 1 (state_in), 6 storage buffers -> 7
// against the device cap of 8 (`solveB_c*` is AT 8 and is untouched).
// A retired particle KEEPS ITS SLOT FOREVER. Nothing here compacts or
// renumbers, `P.n` and `P.nFluid` never change, and every line added is
// self-indexed — design/bonds/prove_slot_stability.mjs is the gate for that and
// it must still exit 0. The three edits are marked `§17` in `predict`,
// `buildNbr` (twice) and `thermal`.
//
// v4 (2026-07-26): TEMPERATURE + REACTIONS (§12). Two new passes at the tail of
// the substep — `conduct` (an order-independent per-term-clamped gather, like
// every other pair loop here) and `thermal` (purely elementwise threshold tests
// against the material table). `Particle.temp` and `Particle.pad0` stop being
// reserved and become `temp` / `fuel`; the struct layout does not move, but the
// state hash widens from 6 words to 8. `struct Mat` grows from 8 words to 32.
//
// v3 (2026-07-25): the solver loop is 4-COLOUR BLOCK GAUSS-SEIDEL (§5.1b).
// Per iteration: one shared `solveA`, then four (`solveB_cK`, `applyDp_cK`)
// pairs in colour order 0,1,2,3. Colour is a property of the particle's CELL,
// read from the frozen per-substep `cellOf`. This is Gauss-Seidel BETWEEN colour
// classes and Jacobi WITHIN one; see §5.1b for why "no two same-colour particles
// interact" is unattainable and why that matters.
//
// This file is CONCATENATED AFTER src/fixed.wgsl by engine.js (WGSL has no
// #include). It therefore declares NO arithmetic helpers of its own and calls
// only the §2 API: fp_mul_s / fp_add_sat / fp_sub_sat / fp_neg_sat / fp_abs /
// fp_shr_rne / fp_divshift / fp_flag.
//
// Hard rules honoured here:
//   * integer only — no floating-point type or reinterpretation anywhere
//   * no timestep uniform, no clock, no host-side randomness
//   * fixed iteration counts; the only data-dependent `continue` is
//     `lut_index(d) < 0`, which skips a term that is identically zero
//   * every signed right shift is fp_shr_rne EXCEPT the three whitelisted in
//     §2.7 rule 5 (cell_coord, lut_index pre-shift, lut_index r2>>KSH)
//   * per-term clamps only; never clamp a running total
//   * no intra-dispatch read-of-another-invocation's-write (see §5.1)
//   * no cross-lane/ballot builtins, no atomics in the physics (gather only)
// ============================================================================

// ─────────────────────────────────── world constants (§4) ───────────────────
const ONE    : i32 = 65536;         // 1 wu in ticks (Q16.16)
const H      : i32 = 163840;        // kernel radius h = 2.5 wu
const DX0    : i32 = 65536;         // rest spacing 1.0 wu  => h/dx = 2.5
const RHO0   : i32 = 1048576;       // 2^20 exactly
const TWO_RHO0 : i32 = 2097152;     // 2 * RHO0
const PRE    : u32 = 4u;            // delta pre-shift before squaring
const HS     : i32 = 10240;         // H >> PRE
const H2S    : i32 = 104857600;     // HS * HS
const KSH    : u32 = 15u;           // r2 -> LUT index shift
const GF     : u32 = 22u;           // gradient LUT Q
const CF     : u32 = 22u;           // cohesion / adhesion LUT Q
const ACC    : u32 = 8u;            // extra fractional bits in accumulators
const MAXNBR : u32 = 48u;
const ITERS  : u32 = 4u;            // host-driven; here for documentation only
const SUBSTEPS_PER_FRAME : u32 = 2u;

// ─────────────────────────────────── LUT geometry (§3.4) ────────────────────
const TN    : u32 = 3202u;
const OFF_W : u32 = 0u;
const OFF_G : u32 = 3202u;
const OFF_C : u32 = 6404u;
const OFF_A : u32 = 9606u;
const OFF_S : u32 = 12808u;

// ─────────────────────────────────── accumulator bounds (§6.1) ──────────────
const LAM_MAX  : i32 =  4194304;    // 1<<22   lambda, Q20
const S_TERM   : i32 = 16777216;    // 1<<24   |grad C|^2 per term, Q21
const DP_TERM  : i32 =  2097152;    // 1<<21   delta-p per term, ticks<<ACC
const XS_TERM  : i32 =  2097152;    // 1<<21   XSPH per term
const ST_TERM  : i32 =  4194304;    // 1<<22   surface tension per term
const RINV_MAX : i32 =   262144;    // 1<<18
const WALL     : i32 =    32768;    // 0.5 wu
const CUR_SIGN : i32 =       -1;    // Akinci as published
const C_MAX    : i32 =    58000;    // compression clamp on C, Q16
const RHO_FLOOR_RINV : i32 = 32768; // 1<<15, fp_divshift denominator floor
const RHO_FLOOR_K    : i32 = 65536; // 1<<16, K_ij denominator floor

// ─────────────────────── §12 THERMAL / REACTION CONSTANTS (v4) ───────────────
// Temperature is Q16.16 "aether degrees", the SAME tick unit as geometry:
// ONE = 65536 ticks = 1 degree. Water freezes at 0 and boils at 100, so the
// numbers read like Celsius; nothing in the solver depends on that reading.
//
// STORAGE RANGE. temp is clamped to [TEMP_MIN, TEMP_MAX] every substep, so the
// bound below is structural, not aspirational:
//   TEMP_MIN = -2^25 =    -512 deg   (below absolute zero; a floor, not physics)
//   TEMP_MAX =  2^29 =    8192 deg   (plasma has headroom: fire ~900, lava ~1200)
// Headroom to i32: 4x on the high side. |T_j - T_i| <= 2^29 + 2^25 < 2^30, so
// fp_sub_sat on a temperature pair provably cannot saturate.
const AMBIENT  : i32 =    1310720;  // 20 deg — the room the box sits in
const TEMP_MIN : i32 =  -33554432;  // -2^25
const TEMP_MAX : i32 =  536870912;  //  2^29

// CONDUCTION FLUX CAP. |T_j - T_i| is clamped to DT_CLAMP before the multiply.
// clamp() is symmetric about zero, so clamping the DIFFERENCE keeps the pair
// term exactly antisymmetric and therefore keeps total heat conserved — it is a
// saturating flux, not a leak. It exists to bound the multiply: with
// |dT| <= 2^24 and max WT = 214942 (measured over the frozen tables), the
// per-term product is <= 8.8e8, so fp_mul_s can never saturate and dbg[0] can
// never fire from this pass.
const DT_CLAMP     : i32 =  16777216;   //  2^24 = 256 deg
const NEG_DT_CLAMP : i32 = -16777216;
// Per-term clamp on the heat accumulator, in temp ticks << ACC. 2^24 << ACC=8
// is 1.0 deg of transfer from ONE neighbour in ONE substep. Sum bound:
// MAXNBR * HT_TERM = 48 * 2^24 = 8.05e8, i.e. 2.7x headroom to 2^31.
const HT_TERM      : i32 =  16777216;   //  2^24
const NEG_HT_TERM  : i32 = -16777216;

// §13 CORROSION. The dose accumulator is non-negative by construction (`corrode`
// is validated >= 0 and the poly6 weight is >= 0), so it needs a per-term cap
// but no negative bound. CORR_TERM = 2^24 << nothing: at ACC = 8 that is 65536
// dose units (= 1.0 Q16) from ONE neighbour in ONE substep, which is already
// more than any material's corrPt. Sum bound MAXNBR * CORR_TERM = 8.05e8, the
// same 2.7x headroom to 2^31 the heat accumulator has.
const CORR_TERM    : i32 =  16777216;   //  2^24
// The immunity sentinel, shared with the phase thresholds: a dose is a sum of at
// most 48 terms of <= 2^24, so it can never reach i32 max.
const CORR_NEVER   : i32 =  2147483647;
// Corrosion strength ceiling, Q16. engine.js asserts it; it is here so that the
// sum bound above is a property of the table, not a hope.
const CORR_MAX     : i32 =     262144;  //  4.0 in Q16

// Particle.flags bits. Authored scenes may only set the public low bits;
// engine-owned lifecycle, weld, rigid, containment, and seal-cache bits are
// validated separately in engine.js so future assignments cannot collide.
const FLAG_BURNING : u32 = 1u;
// §17, v7. RETIRED MATTER. Set by `thermal` from mats[mat].phantom, so it is a
// pure function of matId and can never go stale — no shader clears it, and a
// slot recycled for live matter loses it automatically on the substep after the
// material changes. It is read in exactly two places (`buildNbr`, twice) and
// written in exactly one (`thermal`). No material ID appears anywhere in this
// file; retirement is entirely a property of the TABLE.
const FLAG_DEAD    : u32 = 2u;
// §18.25 — engine-assigned and transient. `thermal` marks a particle that just
// froze into a bonded solid; the next formation pass welds simultaneously
// frozen same-material neighbours, then `bondForm2` clears the mark.
const FLAG_CONGEAL : u32 = 32u;
// §29 — a finite seal endpoint released by pressure. Engine-owned like
// FLAG_CONGEAL; authored scenes cannot set it.
const FLAG_BURST   : u32 = 64u;
// §29 — pure row-derived cache. buildNbr already reads Particle.flags, so this
// avoids adding the mats buffer and a 48-word-row load to its hottest loop.
const FLAG_SEAL    : u32 = 128u;

// The material-id ceiling. mats[] is allocated at MAT_MAX rows by engine.js and
// loadScene rejects any matId >= the live material count, so an out-of-range
// index is impossible by construction rather than by luck.
// v16 (§25): 64, was 32 through v15. The tail rows are the intern cache for
// runtime-synthesised composed materials; a capacity constant, not a solver
// change — no golden scene carries a matId past the authored table.
const MAT_MAX : u32 = 64u;

// Negated bounds as literals: WGSL unary `-` wraps on I32_MIN, and §2.7 rule 3
// bans bare negation in sim code. These are compile-time constants, so writing
// them out is both compliant and free.
const NEG_LAM_MAX : i32 =  -4194304;
const NEG_DP_TERM : i32 =  -2097152;
const NEG_XS_TERM : i32 =  -2097152;
const NEG_ST_TERM : i32 =  -4194304;

// ─────────────────────────────────── host-shared structs (§3.1) ─────────────
// Particle: align 8, size 32, stride 32, ZERO implicit padding (MEASURED).
// v4 (2026-07-26): `temp` and `pad0` STOP being reserved. The offsets, the
// stride and the measured layout are UNCHANGED — only the meaning of words 6
// and 7 moves — but they now carry live simulation state, so the state hash
// MUST cover 8 words instead of 6. engine.js writes hashedWords = PARTICLE_WORDS
// and determinism.html asserts the two agree; see §7.6.
struct Particle {
  pos   : vec2<i32>,        // offset  0   Q16.16 ticks
  vel   : vec2<i32>,        // offset  8   Q16.16 ticks/substep
  matId : u32,              // offset 16   index into mats[], < MAT_MAX
  flags : u32,              // offset 20   bit0 FLAG_BURNING, bit1 FLAG_DEAD (§17)
  temp  : i32,              // offset 24   Q16.16 degrees, [TEMP_MIN, TEMP_MAX]
  fuel  : i32,              // offset 28   Q16.16 fuel remaining, >= 0
}

// Derived: align 8, size 64, stride 64.
// v5 appends `corr` (§13). vec2<i32> forces align 8, so a struct of 60 bytes
// would be padded to 64 IMPLICITLY — and an implicit tail pad is exactly the
// layout hazard PLATFORM_NOTES records for vec3. `pad1` makes it explicit so the
// host packer and test/determinism.html's layout probe agree by construction.
struct Derived {
  pred  : vec2<i32>,        // offset  0
  dp    : vec2<i32>,        // offset  8   ticks << ACC during accumulation
  vtmp  : vec2<i32>,        // offset 16   read-only velocity snapshot for XSPH
  nrm   : vec2<i32>,        // offset 24   Q16 colour-field normal
  stAcc : vec2<i32>,        // offset 32   applied one substep LATE
  rho   : i32,              // offset 40   RHO0-scaled
  lam   : i32,              // offset 44   Q20
  rinv  : i32,              // offset 48   Q16
  heat  : i32,              // offset 52   temp ticks << ACC (was pad0)
  corr  : i32,              // offset 56   §13 corrosion dose << ACC
  pad1  : u32,              // offset 60   explicit tail pad, always 0
}

// §23.B. HOW MANY JETS THE UNIFORM CARRIES. 8, against `index.html`'s
// MAX_DEVICES of 12: a jet is a device and the device cap is the user-facing
// one, but the shader's array is fixed at compile time and 8 jets in one tank
// is already more thrust than a 128 x 72 wu box has room for. The HOST clamps
// to this and says so; nothing here silently drops one.
const JET_MAX : u32 = 8u;
// The circle test squares a difference of TICKS, so it is shifted down first.
// Bound: the domain is 128 x 72 wu = 8 388 608 x 4 718 592 ticks, and a rigid
// member may sit up to rMax (~13 wu) outside the box because `rg_place` does
// not clamp members. Worst case |dx| < 141 wu = 9.24e6 ticks; >> 9 gives
// 18 048, squared 3.26e8, and dx^2 + dy^2 stays under 4e8 — an order of
// magnitude inside i32. Resolution is 512 ticks = 1/128 wu, which is 1/128th of
// one particle spacing.
const JET_SH : u32 = 9u;

// Params: uniform. Scalars at 0..60, then two align(16) jet arrays.
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
  // §18.22 — THE ONE WELD TAG THAT IS CURRENTLY HELD OPEN, already shifted into
  // WELD_TAG_MASK position; 0 means "no gesture is open". Written by the host at
  // beginWeldGroup/endWeldGroup and read by `bondForm2` and nowhere else.
  //
  // It is host state exactly like `n` and `nFluid`: a pure function of the call
  // sequence, never of a clock. It is 0 for every scene the determinism gate,
  // the goldens or `loadScene` ever produce, and `bondForm2`'s use of it is
  // bit-for-bit inert at 0 (see the note there), which is why v9 does NOT move
  // the chain.
  weldHold  : u32,
  // §20 — HOW MANY RIGID BODY RECORDS ARE LIVE. The four §20 passes dispatch
  // ONE WORKGROUP PER BODY, so this is the grid size, not a loop bound. It is 0
  // for every scene the determinism gate, the goldens or `loadScene` produce
  // without an explicit body, and at 0 the §20 passes launch zero workgroups —
  // which is why §20 does NOT move the chain.
  nBodies   : u32,
  // §23.B — HOW MANY JET DEVICES ARE LIVE, and the jets themselves. A jet is a
  // circular region that adds a constant acceleration to whatever matter is
  // inside it. It lives HERE, in the uniform, and not in a storage buffer:
  // `friction` and `bondForm1` already sit at this device's cap of 8 storage
  // buffers per stage (PLATFORM_NOTES) and a thrust primitive is not worth
  // spending the wall on. It is host state exactly like `weldHold` and
  // `nBodies` — a pure function of the call sequence, never of a clock — and at
  // `nJets == 0` `jet_accel` returns (0, 0), which `fp_add_sat` absorbs
  // exactly. That is why v13 does NOT move the chain.
  nJets     : u32,
  _padJet   : u32,          // the arrays below are align(16) in uniform space
  // (cx, cy, r) in TICKS, w unused. r is a hard edge: uniform inside, nothing
  // outside. A falloff would need a divide per particle per jet and would make
  // "is this jet doing anything" a judgement instead of a yes/no.
  jetPos    : array<vec4<i32>, JET_MAX>,
  // (fx, fy) in TICKS PER SUBSTEP SQUARED — the same units as `Mat.grav`, so a
  // jet's strength is legible against a material's weight and the per-term
  // clamp in `rigidStep` keeps exactly the bound §20 proved.
  jetF      : array<vec4<i32>, JET_MAX>,
  // §24 — THE PLAYER'S DRIVE VECTOR, in TICKS PER SUBSTEP SQUARED (the same
  // units as `Mat.grav` and as `jetF`, so "half a gravity of push" is a number
  // you can read). It is KEY STATE, and key state is an INPUT: the host samples
  // it ONCE PER FRAME, before `step()`, exactly where pointer events are already
  // applied — never on a clock, never inside the substep loop. Two words after
  // the jet arrays because a uniform struct's size rounds up to 16 and its
  // ARRAY members must start on 16; appending scalars costs one 16-byte tail and
  // moves nothing that already works.
  //
  // It is read in exactly ONE place — `rigidStep`'s free-body velocity fold —
  // and only for a body whose record says RB_USER != 0. At (0, 0) the addition
  // is `fp_add_sat(dv, 0) == dv` exactly, which is why §24 does NOT move the
  // chain: the golden scene has no bodies at all, and a body that is not flagged
  // never reads these words.
  userAx    : i32,
  userAy    : i32,
  // §18.24 — THE ONE GESTURE WHOSE FUSION IS ARMED, already shifted into
  // WELD_TAG_MASK position; 0 means "no gesture fuses". Written by the host at
  // armFuse (beginWeldGroup disarms, step() clears it after the formation pass
  // that follows the gesture's release) and read by `bondForm1` and nowhere
  // else. It is host state exactly like `weldHold`: a pure function of the
  // call sequence, never of a clock. It is 0 for every scene the determinism
  // gate, the goldens or `loadScene` ever produce, and every §18.24 branch is
  // bit-for-bit inert at 0 — which is why fusion does NOT move the chain.
  // This word was `_padUsr0`, always 0, so the layout did not move.
  fuseTag   : u32,
  _padUsr1  : u32,
}

// ─────────────────────────── THE MATERIAL TABLE (§6.2, v4) ───────────────────
// 32 i32 words, align 4, size 128, stride 128. Rows are DATA: adding a material
// is one row in engine.js MATS, never a shader edit. Words 0..7 are the v3
// mechanics block and their ORDER IS FROZEN — v3 rows still read correctly.
//
// A transition is a THRESHOLD TEST ON AN INTEGER, evaluated in a fixed order in
// `thermal`. There is no randomness anywhere: the only variation is `splitPct`,
// resolved by a pure integer hash of the particle id.
//
// SENTINELS. A transition this material does not have is disabled by putting
// the threshold out of reach, NOT by a flag:
//   never-on-heating  ->  2147483647  (i32 max; T <= TEMP_MAX = 2^29 always)
//   never-on-cooling  -> -2147483648  (i32 min; T >= TEMP_MIN = -2^25 always)
struct Mat {
  // ---- mechanics (v3 block, order frozen) ----------------------------------
  eps      : i32,           //  0  Q21 CFM relaxation, >= 32768 (§2.3 contract)
  dpMax    : i32,           //  1  ticks, max |dp| per solver iteration
  xsph     : i32,           //  2  Q16 XSPH viscosity coefficient
  // §17, v7. 0 or 1. 1 means THIS ROW IS NOT MATTER — the particle is retired:
  // frozen where it fell, absent from every neighbour list, invisible. It was
  // `vort`, RESERVED-must-be-0 since v3 and never implemented, so every v3..v6
  // row already reads as `phantom = 0` and rows 0-23 keep their exact bytes.
  // Word 10 `heatCap` is still reserved and is the better home for vorticity
  // confinement anyway: both are mechanics, only one is a solver-loop term.
  phantom  : i32,           //  3  0 = matter, 1 = retired (§17)
  gammaCoh : i32,           //  4  ticks/substep^2 cohesion
  gammaCur : i32,           //  5  ticks/substep^2 curvature (~gammaCoh/4)
  adhesion : i32,           //  6  ticks/substep^2 wall wetting (NOT pair-mixed)
  grav     : i32,           //  7  ticks/substep^2; MAY BE NEGATIVE (buoyancy)
  // ---- thermal -------------------------------------------------------------
  cond     : i32,           //  8  Q16 conductivity, [0, 32768]; 0 = insulator
  coolRate : i32,           //  9  Q16 relaxation toward AMBIENT per substep
  heatCap  : i32,           // 10  RESERVED, must be 0 (v5: per-material capacity)
  // ---- phase change --------------------------------------------------------
  meltPt   : i32,           // 11  T >= meltPt   -> meltTo
  meltTo   : u32,           // 12
  freezePt : i32,           // 13  T <= freezePt -> freezeTo
  freezeTo : u32,           // 14
  boilPt   : i32,           // 15  T >= boilPt   -> boilTo
  boilTo   : u32,           // 16
  condPt   : i32,           // 17  T <= condPt   -> condTo
  condTo   : u32,           // 18
  // ---- combustion ----------------------------------------------------------
  ignitePt : i32,           // 19  T >= ignitePt && fuel > 0 -> FLAG_BURNING
  fuel0    : i32,           // 20  fuel a fresh particle of this material carries
  burnRate : i32,           // 21  fuel consumed per substep while burning
  burnHeat : i32,           // 22  temp ticks added to SELF per substep, burning
  burnTo   : u32,           // 23  material once fuel is exhausted
  // ---- product split (applies to WHICHEVER transition fires) ---------------
  splitTo  : u32,           // 24  secondary product
  splitPct : u32,           // 25  0..256 out of 256; 0 = never split
  // ---- host-only authoring metadata; the SHADER NEVER READS THESE ----------
  spawnTemp: i32,           // 26  temp a UI-painted particle starts at
  tint     : u32,           // 27  0x00RRGGBB hint for render.js (optional)
  // ---- corrosion (§13, v5). CHEMISTRY, not temperature ----------------------
  // The one interaction in the roster that is NOT a function of temperature: a
  // material transforms because of WHAT is touching it, not how hot it is.
  // It cannot be expressed on the thermal axis — a hot corrosive is LAVA and a
  // cold one is CRYO, and both already exist.
  corrode  : i32,           // 28  Q16 attack this material inflicts on any
                            //     DIFFERENT material. 0 = not corrosive, and 0
                            //     makes the whole branch bit-for-bit inert.
  corrPt   : i32,           // 29  dose (Q16, per substep) at which this material
                            //     becomes corrTo. NEVER_HOT sentinel = immune.
  corrTo   : u32,           // 30  product
  // ---- granular friction (§16, v6) ----------------------------------------
  // Q16 Coulomb coefficient, [0, 65536]. 0 = frictionless and makes the whole
  // `friction` pass bit-for-bit inert for this material — which is what WATER
  // and GOO ship at, and what keeps every v5 hash reproducible.
  fric     : i32,           // 31  see the `friction` pass
  // ---- bonded rigid bodies (§18, v8) --------------------------------------
  // The block APPENDS at word 32, so words 0..31 of every existing row keep
  // their exact bytes and a v7 row still reads correctly for everything that
  // is not a bond. Stride goes 32 -> 48 words (128 -> 192 B); test/layout.html
  // asserts it, because an implicit pad here would be silent (PLATFORM_NOTES
  // records vec3<T> as size 12 / align 16 — every word below is a 4-byte
  // scalar precisely so offsets stay 4*index with zero implicit padding).
  bondK      : i32,         // 32  Q16 stiffness at ramp 1. 0 => this row NEVER
                            //     bonds, and 0 is what makes the whole §18 term
                            //     bit-for-bit inert for it (WATER, GOO, VOID…).
  bondBreak  : i32,         // 33  Q16 (1 + eps_break)^2; 0 = unbreakable
  bondReform : i32,         // 34  0 = spawn-weld only, 1 = also re-forms on
                            //     contact. A `bondReform == 0` material cannot
                            //     be put on an emitter (§18.12).
  bondFormV  : i32,         // 35  ticks: max |relative velocity| for a contact bond
  bondFormP  : i32,         // 36  Q20:   min |lambda| (compression) for a contact bond
  bondMu     : i32,         // 37  Q16 Mohr-Coulomb slope; 0 = pure cohesion, inert
  bondGroup  : i32,         // 38  only EQUAL groups bond; -1 = never bonds.
                            //     NOTE: `min` mixing is provably free for the
                            //     ROW coefficients ONLY while every bonded
                            //     material has a UNIQUE group. Share a group and
                            //     every number in SPEC §18 is invalid.
  rampLo     : i32,         // 39  temp ticks, K(T) ramp low edge
  rampHi     : i32,         // 40  temp ticks, K(T) ramp high edge; MUST be > rampLo
  rampInv    : i32,         // 41  Q16, BAKED by createEngine = floor(2^32/(hi-lo))
  rampSign   : i32,         // 42  -1 softens with heat, +1 stiffens, 0 = INERT BRANCH
  // ---- §20, v10. THE SOLID AXIS. ONE FIELD, THREE MUTUALLY EXCLUSIVE VALUES.
  // 0 SOLID_NONE  — fluids and grains, no cohesion mechanism.
  // 1 SOLID_BOND  — §18 distance bonds: cohesive and DEFORMABLE (RICE, and the
  //                 queued elastic RUBBER/PHOTO). A network of springs is a
  //                 spring, which is the right answer for a bouncy ball and the
  //                 wrong one for a wrench.
  // 2 SOLID_RIGID — §20 rigid bodies: NOT deformable, at all. A body's members
  //                 are PLACED at c + R(theta)*o, so flexion is not
  //                 representable.
  // The exclusivity IS the design: rigid and elastic are opposite properties and
  // a row claiming both is a bug rather than a blend. It was `rsv4` (RESERVED,
  // must be 0) through v9, so a v9 row reads as SOLID_NONE and the field can
  // only change behaviour for a row that opts in.
  //
  // THE SHADER READS IT IN EXACTLY ONE PLACE — `rigidMember`'s leave test — so a
  // material that transmutes, melts, burns, corrodes or is retired LEAVES its
  // body through one rule and no new machinery (§20.4.1). The HOST reads it to
  // decide whether a finished gesture is promoted to a body at all.
  solidMode  : i32,         // 43  0 NONE / 1 BOND / 2 RIGID  (§20.6)
  // ---- §25, v16. THE THERMAL DRIVE. Words 44/45 were rsv5/rsv6 (RESERVED,
  // must be 0) through v15, so every pre-v16 row reads as tempDrive = 0 and
  // the one branch that reads them (thermal step 2.5) is bit-for-bit inert on
  // the whole authored table. A driven row relaxes its OWN temperature toward
  // tempTarget exactly as coolRate relaxes toward AMBIENT — deliberately
  // non-conserving, which is what makes a heater a heater: a merely born-hot
  // row spends its heat (conduction conserves sum(T), §12.2) and stops.
  tempTarget : i32,         // 44  temp ticks the row drives itself toward
  tempDrive  : i32,         // 45  Q16 relaxation rate, [0, 65536]; 0 = inert
  // ---- §29, v17. CONTAINMENT. Words 46/47 were rsv7/rsv8 (RESERVED,
  // must-be-zero) through v16. Neither word grants sealing: seal_class()
  // derives that from the row's existing solid mechanics.
  burstP     : i32,         // 46  intrusion threshold; 0 = never bursts
  sealMode   : i32,         // 47  0 = sealed, 1 = liquid-permeable membrane
}

// ─────────────────────────────────── global binding map (§3.3) ──────────────
// Group 0 only. A buffer keeps the SAME index in every shader that uses it.
// Binding 14 is reserved for the state-hash alias; binding 15 (fp_dbg) is
// declared by fixed.wgsl.
@group(0) @binding(0)  var<uniform>             P         : Params;
@group(0) @binding(1)  var<storage, read>       state_in  : array<Particle>;
@group(0) @binding(2)  var<storage, read_write> state_out : array<Particle>;
@group(0) @binding(3)  var<storage, read_write> derived   : array<Derived>;
@group(0) @binding(4)  var<storage, read_write> cellCount : array<atomic<u32>>;
@group(0) @binding(5)  var<storage, read_write> cellStart : array<u32>;
@group(0) @binding(6)  var<storage, read_write> blockSums : array<u32>;
@group(0) @binding(7)  var<storage, read_write> cellOf    : array<u32>;
@group(0) @binding(8)  var<storage, read_write> bucketIds : array<u32>;
@group(0) @binding(9)  var<storage, read_write> sortedIds : array<u32>;
@group(0) @binding(10) var<storage, read_write> nbr       : array<u32>;
@group(0) @binding(11) var<storage, read_write> nbrN      : array<u32>;
@group(0) @binding(12) var<storage, read>       luts      : array<i32>;
@group(0) @binding(13) var<storage, read>       mats      : array<Mat>;
// §18 bonds. Two buffers, and the PING-PONG IS MANDATORY, not stylistic:
// `bondForm2` reads bondCand[j] (another invocation's row) while writing
// bond[i]. That is §5.1-legal precisely because they are DIFFERENT buffers, so
// no invocation reads a value another invocation wrote in this dispatch.
// Updating in place instead silently re-forms already-bonded pairs and resets
// their rest lengths, invisibly to every structural check.
// `w == 0` means EMPTY, so an all-zero buffer is a valid empty table and there
// is no initialisation pass.
@group(0) @binding(16) var<storage, read_write> bond      : array<u32>;
@group(0) @binding(17) var<storage, read_write> bondCand  : array<u32>;
// §20 TRUE RIGID BODIES. ONE buffer, three regions, laid out by the constants
// below. It carries the Q22 quarter-turn cosine table (baked by the host, like
// the kernel LUTs), the body records, and the per-slot rest offsets.
//
// It is NOT atomic and it needs no atomics: every §20 pass dispatches ONE
// WORKGROUP PER BODY and reduces in workgroup memory with a fixed barrier
// schedule, so no two invocations ever write the same word. That is also what
// keeps the reduction order-independent for free — integer addition is exact
// and associative (PLATFORM_NOTES), and the tree is fixed.
@group(0) @binding(18) var<storage, read_write> rigid     : array<i32>;

// ─────────────────────────────────── shared helpers ─────────────────────────

// WHITELISTED plain shift (§2.7 rule 5): floor bucketing is the CORRECT rule.
// An integer divide truncates toward zero and would make the cell straddling
// the origin double-width.
fn cell_coord(p : vec2<i32>) -> vec2<i32> {
  let cx = (p.x - P.originX) >> P.cellShift;
  let cy = (p.y - P.originY) >> P.cellShift;
  return vec2<i32>(clamp(cx, 0, i32(P.gridW) - 1), clamp(cy, 0, i32(P.gridH) - 1));
}

fn cell_index(c : vec2<i32>) -> u32 { return u32(c.y) * P.gridW + u32(c.x); }

// ─────────────────────────────── §5.1b THE SOLVER COLOURING ──────────────────
// 4 colours on the 4.0 wu simulation grid. Same-colour cells are 2 cells apart,
// i.e. >= 4.0 wu, against a maximum possible interaction distance of 2.50015 wu
// (h = 2.5 wu plus lut_index's `+8` pre-shift rounding). Margin 1.60x. 9 colours
// was measured to remove exactly ZERO additional interactions and is pure cost;
// 2 colours fails, because diagonally adjacent cells share a parity.
//
// *** THE COLOUR COMES FROM cellOf, NOT FROM derived[i].pred. ***
// `gridCount` writes cellOf once per substep and no solver pass touches it, so a
// particle CANNOT change colour between colour rounds. Recomputing the colour
// from `pred` would be a silent correctness bug: `applyDp` moves pred by up to
// dpMax = 16384 ticks = 0.25 wu per iteration against a 4.0 wu cell, so a
// particle near a cell edge would change colour mid-iteration and be corrected
// twice or not at all. That failure is DETERMINISTIC, so no gate would ever see
// it — it would just be quietly wrong physics.
//
// *** THIS IS NOT "SAME-COLOUR PARTICLES NEVER INTERACT". *** No cell colouring
// can achieve that, because two particles in the SAME cell share a colour by
// construction, and on a settled pool ~51% of interacting fluid-fluid pairs are
// intra-cell. What this buys is Gauss-Seidel BETWEEN colour classes and Jacobi
// WITHIN one — which is enough to melt the lattice (§C17) — and it is exactly
// why `solveB_cK` must NOT be fused into `applyDp_cK`. See §5.1b.
fn cell_colour(i : u32) -> u32 {
  let c  = cellOf[i];
  let cx = c % P.gridW;
  let cy = c / P.gridW;
  return (cx & 1u) | ((cy & 1u) << 1u);
}

// THE pair-loop index. Returns -1 when the pair is out of kernel range.
// The axis reject on the PRE-SHIFTED value is what keeps r2 inside i32; it must
// come before the squares (§2.7 rule 1). Both shifts here are whitelisted:
// the operands are provably non-negative, so floor == truncate, and `+8` makes
// the first a round-half-up of a non-negative value — symmetric in d, so both
// ends of a pair land in the same bin. That symmetry is load-bearing.
fn lut_index(d : vec2<i32>) -> i32 {
  let ax = (fp_abs(d.x) + 8) >> PRE;
  let ay = (fp_abs(d.y) + 8) >> PRE;
  if (ax > HS || ay > HS) { return -1; }
  let r2 = ax * ax + ay * ay;
  if (r2 >= H2S) { return -1; }
  return r2 >> KSH;
}

// grad W / RHO0 in Q16 units of 1/wu. s = GF + 4 = 26, a compile-time constant.
fn grad_q16(gt : i32, d : i32) -> i32 { return fp_mul_s(gt, d, GF + 4u); }

fn is_boundary(j : u32) -> bool { return j >= P.nFluid; }
const NBR_SEAL_BIT : u32 = 0x80000000u;
const NBR_EXT_SEAL_BIT : u32 = 0x40000000u;
const NBR_CONGEAL_BIT : u32 = 0x20000000u;
fn nbr_count(i : u32) -> u32 {
  return nbrN[i] & ~(NBR_SEAL_BIT | NBR_EXT_SEAL_BIT | NBR_CONGEAL_BIT);
}
fn nbr_has_seal(i : u32) -> bool { return (nbrN[i] & NBR_SEAL_BIT) != 0u; }
fn nbr_has_external_seal(i : u32) -> bool { return (nbrN[i] & NBR_EXT_SEAL_BIT) != 0u; }
fn nbr_has_congeal(i : u32) -> bool { return (nbrN[i] & NBR_CONGEAL_BIT) != 0u; }

// pred_i - pred_j, saturating. |d| <= 2^27 by the domain limit, so fp_sub_sat
// can never actually saturate — it is used for the discipline, not the guard.
fn pred_delta(pi : vec2<i32>, j : u32) -> vec2<i32> {
  let pj = derived[j].pred;
  return vec2<i32>(fp_sub_sat(pi.x, pj.x), fp_sub_sat(pi.y, pj.y));
}

fn wall_clamp(p : vec2<i32>) -> vec2<i32> {
  return clamp(p, vec2<i32>(WALL, WALL),
                  vec2<i32>(P.domW - WALL, P.domH - WALL));
}

// §6.2 cross-material coefficient mixing. WHITELISTED plain shift (§2.7 rule 5
// addendum): both operands are non-negative material coefficients read from the
// frozen mats table (max 30000), so the sum cannot overflow and >> 1 is an exact
// floor. It is a SYMMETRIC function of (a, b), which is the entire point — both
// ends of a pair must compute the identical coefficient or the pair force stops
// obeying Newton's third law. Note (a + a) >> 1 == a exactly, so same-material
// pairs are bit-for-bit unchanged by this mixing.
fn mat_mix(a : i32, b : i32) -> i32 { return (a + b) >> 1; }

// K_ij = 2*RHO0 / (rho_i + rho_j), Q16. Symmetric in (i, j) by construction.
// Clamped to RINV_MAX (4.0) exactly as `rinv` is, which keeps the per-term
// bound analysis in §6.1 unchanged.
fn sym_k(rhoi : i32, rhoj : i32) -> i32 {
  let den = max(RHO_FLOOR_K, fp_add_sat(rhoi, rhoj));
  return clamp(fp_divshift(TWO_RHO0, den, 16u), 0, RINV_MAX);
}

// ─────────────────────── §12.3 VARIETY WITHOUT RANDOMNESS ────────────────────
// splitmix32's finalizer over the particle id. A PURE FUNCTION of i: no state,
// no counter, no clock, and identical on the CPU mirror (`idHash` there). This
// is the ONLY source of per-particle variety in the reaction system, and it is
// deliberately not an RNG — an RNG would need per-particle state that evolves,
// which is one more thing that has to survive a bit-exact contract for nothing.
// u32 arithmetic wraps by definition in WGSL, so every step here is exact.
fn id_hash(i : u32) -> u32 {
  var x : u32 = i + 0x9e3779b9u;
  x = (x ^ (x >> 16u)) * 0x21f0aaadu;
  x = (x ^ (x >> 15u)) * 0x735a2d97u;
  return x ^ (x >> 15u);
}

// Resolve a transition's product. `splitPct` out of 256 particles take the
// secondary product instead of the primary — that is how wood becomes ash AND
// smoke without the solver ever allocating a particle. splitPct = 0 disables it
// and the branch is then bit-for-bit inert.
fn pick_product(primary : u32, splitTo : u32, splitPct : u32, i : u32) -> u32 {
  if (splitPct == 0u) { return primary; }
  if ((id_hash(i) & 255u) < splitPct) { return splitTo; }
  return primary;
}

// ════════════════════════════════════════════════════════════════════════════
// §20 — TRUE RIGID BODIES. Constants and the two helpers the ORDINARY passes
// need. The four §20 passes themselves are at the tail of this file.
//
// THE ONE SENTENCE: a drawn solid is ONE object with four degrees of freedom.
// Per body, per fold, the body's own (c, theta) integrate; then every member is
// PLACED at  pos_i = c + R(theta) * o_i.  There is no per-member degree of
// freedom for flexion to live in, so flexion is NOT REPRESENTABLE — which is
// the difference between this and §18. A distance constraint resists stretch
// ALONG the bond and has no opinion about BENDING, so a thin §18 stroke is a
// chain of joints each free to rotate. Measured on identical staging: a loaded
// 2x20 METAL bar deforms 5.192 wu with §18 bonds and 0.000 with §20.
//
// ABSOLUTE PLACEMENT ONLY. `theta` is an i32 in BAM (2^32 BAM = one turn) so
// wraparound IS the topology — no modulo, no range reduction, no accumulated
// error — and R(theta) is RE-READ from the table every placement. NOTHING ever
// accumulates a rotation matrix. Measured (design/rigid/OUT_r4_drift.txt): a
// Q16 incremental matrix has determinant 1.0000151, so a 10 wu wheel grows
// 3.50 % by substep 10 000 — the exact depth test/determinism.html certifies —
// and 39.72 % by 100 000. Renormalising every substep fixes the RADIUS and
// leaves 961 ticks of placement error against 1.09 for absolute placement, an
// 880x margin. This is the single most important constraint in §20.
// ════════════════════════════════════════════════════════════════════════════
// Particle.flags bits 8..19 — the §18.22 spawn-event tag, and §20's BODY ID.
// One gesture is one weld group is one body, so §20 spends NO new per-particle
// bits: bits 20..31 stay free for §17 vent phase 2's generation counter.
// (Declared here rather than in the §18 block below because `solveB_at` and
// `friction` — which are earlier in this file — now need it.)
const WELD_TAG_MASK: u32 = 0x000fff00u;
fn weld_tag(f : u32) -> u32 { return f & WELD_TAG_MASK; }
// Particle.flags bit 4. Bit 3 is left for a future per-particle pivot mark;
// §19.B did not need one (see the RB_AXIS note below).
const FLAG_RIGID   : u32 = 16u;

// ════════════════════════════════════════════════════════════════════════════
// §19.A — ANCHORED MATTER. Particle.flags bit 2.
//
// Matter you PLACED that nothing in the world can move. It is not a boundary
// particle and the difference is the whole feature: a boundary particle carries
// matId = WATER as a placeholder (see `addWalls`), so today everything sliding
// on the world floor feels WATER friction and WATER density. An anchored STONE
// ledge carries the STONE row into every pair term it takes part in, so sand
// piles differently on stone than on ice. ANCHORED IS A MATERIALS FEATURE
// WEARING AN IMMOBILITY COSTUME.
//
// THREE EARLY RETURNS, and §17's retirement line is the template for the first:
//   predict     pred = pos  =>  finalize recovers v = pred - pos = 0 and writes
//               pos back unchanged, so position AND velocity are constant
//               forever with no further code.
//   applyDp_at  §17 gets away without this because a corpse is dropped from
//               every neighbour list. ANCHORED MATTER CANNOT BE DROPPED — being
//               seen is the point of it — so it does accumulate dp and would
//               otherwise be pushed.
//   xsph        runs AFTER finalize and writes state_out[i].vel. Without the
//               guard an anchored particle would end the substep holding a
//               velocity it never uses. Deterministic either way, but `vel` is
//               hashed and the renderer reads it, so "anchored matter is
//               bit-for-bit frozen" would be false as a stated invariant.
//
// It deliberately breaks momentum conservation, exactly as the domain boundary
// has since v1: an anchored particle is a momentum SINK. Any Newton's-third-law
// gate must exclude anchored matter by construction rather than be weakened.
//
// It is a pure function of one particle's own hashed word, so it cannot vary
// with scheduling, and it is written by the HOST at spawn — no shader sets it.
// One shader CLEARS it: `thermal`, when the row transmutes (§19.A.5), because an
// anchored ICE shelf that melts into an immovable puddle is absurd.
const FLAG_ANCHOR  : u32 = 4u;

// mats[].solidMode
const SOLID_NONE   : i32 = 0;
const SOLID_BOND   : i32 = 1;
const SOLID_RIGID  : i32 = 2;

// ---- the rigid buffer's four regions, in i32 words --------------------------
// engine.js RG_* must agree with every one of these; test/rigid.html R0 asserts
// it by comparing the two lists, because a silent disagreement here would place
// members from another body's record.
const RG_COS_BASE   : u32 = 0u;         // 1026 entries, Q22 quarter turn
const RG_BODY_BASE  : u32 = 1088u;      // 1026 rounded up to 64
const RG_BODY_WORDS : u32 = 32u;
const RG_MAX_BODIES : u32 = 1024u;
// Rung 4: mask + nine imported-module values. A sidecar preserves the frozen
// 32-word body record and native snapshot layout.
const RG_DRIVER_WORDS : u32 = 10u;
const RG_DRIVER_BASE  : u32 = 33856u;    // RG_BODY_BASE + RG_MAX_BODIES*32
const RG_REST_BASE    : u32 = 44096u;    // RG_DRIVER_BASE + RG_MAX_BODIES*10
const RD_MASK : u32 = 0u;
const RD_PX   : u32 = 1u;
const RD_PY   : u32 = 2u;
const RD_A    : u32 = 3u;
const RD_VX   : u32 = 4u;
const RD_VY   : u32 = 5u;
const RD_W    : u32 = 6u;
const RD_FX   : u32 = 7u;
const RD_FY   : u32 = 8u;
const RD_TQ   : u32 = 9u;
const RD_HAS_PX : i32 = 1;
const RD_HAS_PY : i32 = 2;
const RD_HAS_A  : i32 = 4;
const RD_HAS_VX : i32 = 8;
const RD_HAS_VY : i32 = 16;
const RD_HAS_W  : i32 = 32;
const RD_HAS_FX : i32 = 64;
const RD_HAS_FY : i32 = 128;
const RD_HAS_TQ : i32 = 256;

// ---- body record words ------------------------------------------------------
const RB_LIVE   : u32 =  0u;   // 1 = this record is a live body
const RB_TAG    : u32 =  1u;   // weldTag, ALREADY SHIFTED into WELD_TAG_MASK
const RB_FIRST  : u32 =  2u;   // lowest slot the body can own   (inclusive)
const RB_LAST   : u32 =  3u;   // highest slot the body can own  (inclusive)
const RB_CX     : u32 =  4u;   // centre, ticks                   STATE
const RB_CY     : u32 =  5u;
const RB_VX     : u32 =  6u;   // linear velocity, ticks/substep  STATE
const RB_VY     : u32 =  7u;
const RB_THETA  : u32 =  8u;   // BAM                             STATE
const RB_OMEGA  : u32 =  9u;   // BAM/substep                     STATE
const RB_RESX   : u32 = 10u;   // carried remainder, POSITION fold  STATE §20.3.4
const RB_RESY   : u32 = 11u;
const RB_RESVX  : u32 = 12u;   // carried remainder, VELOCITY fold  STATE
const RB_RESVY  : u32 = 13u;
const RB_N      : u32 = 14u;   // live member count               derived
const RB_SUMIS2 : u32 = 15u;   // SUM (o >> ish)^2 over live members
const RB_GSUM   : u32 = 16u;   // SUM grav over live members
const RB_INVI   : u32 = 17u;   // Q(iq) reciprocal inertia
const RB_IQ     : u32 = 18u;
const RB_RSH    : u32 = 19u;   // moment-arm pre-shift, CONSTANT from weld time
const RB_ISH    : u32 = 20u;   // inertia pre-shift,    CONSTANT from weld time
const RB_OMEGAM : u32 = 21u;   // rim-speed cap, BAM/substep
const RB_RMAX   : u32 = 22u;
const RB_OSH    : u32 = 23u;   // centroid-sum pre-shift, CONSTANT from weld time
const RB_CQX    : u32 = 24u;   // within-substep predicted centre   scratch
const RB_CQY    : u32 = 25u;
const RB_THETAQ : u32 = 26u;   // within-substep predicted angle    scratch
// ════════════════════════════════════════════════════════════════════════════
// §19.B — AXIS BODIES. Word 27, and it is the WHOLE of the feature's state.
//
// An axis body is a §20 rigid body with the two translational degrees of
// freedom PINNED. That is the entire design, and it is why this costs ZERO new
// dispatches, ZERO new buffers, ZERO new bindings and ZERO per-particle bits:
// §20 already carries (c, v, theta, omega) per body, already places members
// absolutely at c + R(theta)*o from EXACT rest offsets, already reduces torque
// in workgroup memory with a fixed barrier schedule, and already owns the Q22
// quarter-turn LUT. SPEC19_DRAFT priced this feature at +5 or +10 dispatches and
// a new storage buffer because it was written before §20 existed.
//
// Three differences from a free body, and each is one branch:
//   1. THE FRAME ORIGIN IS THE PIVOT, NOT THE CENTROID. The user pressed a
//      point; the body's `c` is that point and never moves. Rest offsets are
//      measured from it, so RB_SUMIS2 is the moment of inertia ABOUT THE PIVOT
//      with no extra term, and RB_RMAX is the true rim radius.
//   2. GRAVITY PRODUCES TORQUE, and for a free body it must not. A free body's
//      frame origin is its centroid, so SUM r_i = 0 and gravity is folded as the
//      single per-body constant `gSum` into the FORCE with no torque term at all
//      (that is what keeps a symmetric body exactly still). Pin the frame at a
//      point the user chose and SUM r_i is no longer 0: an unbalanced wheel
//      swings like a pendulum, which is correct and is what makes an axis body
//      read as a machine part. It is folded per-member into the SAME clamped
//      torque term, so the accumulator bound is byte-for-byte the one §20 proved
//      (|s| <= 2^15 per member, |raw| <= 2^27) and nothing new needs bounding.
//   3. THE AXLE ABSORBS THE LINEAR CORRECTION. `rigidStep` does not integrate
//      velocity and `rigidSolve` does not translate the centre; the sum of dp
//      the solver wanted to apply is exactly the constraint force the bearing
//      supplies. `rigidMember` also skips the centroid RE-BASE, because re-basing
//      moves the frame origin to the live centroid — which for a free body is
//      the point, and for an axis body would silently walk the axle.
//
// A DRY SYMMETRIC WHEEL IS STILL EXACTLY STILL, and the proof survives the new
// term. Members pair off at +/- r about the pivot; `fp_shr_rne` and `clamp` are
// both sign-symmetric, so the pair's raw torques are exact negatives t and -t.
// The hi/lo split does NOT cancel them term by term (hi = t >> 15 floors, so a
// pair contributes hi = -1 and lo = +2^15) but it cancels them EXACTLY in the
// rejoin: the two arguments become Shi = -K at shift iq-15 and Slo = K*2^15 at
// shift iq, whose exact products are equal and opposite, and `i64_shr_rne_sat`
// is round-half-to-EVEN, hence sign-symmetric. dtheta is 0 to the BAM.
const RB_AXIS   : u32 = 27u;   // 1 = the centre is a PINNED PIVOT (§19.B)

// §24 — USER: ARROW-KEY MATTER. 1 = the player drives this body.
//
// WHY A BODY WORD AND NOT A MATERIAL OR A PARTICLE BIT. Measured over the 154
// real .oec files: 2 840 of 3 069 User particles (92.5 %) sit on a Rigid
// particle. OE-CAKE's User bit is not a substance anyone pours, it is a label
// on a machine you built — so it belongs on the object that already has a
// centre, a velocity and a mass, which is the §20 body record. Word 28 of 32;
// 29..31 stay free, and NO per-particle flag bit is spent (bit 3 is still
// reserved for a future pivot mark, exactly as the FLAG_RIGID note says).
//
// WHAT IT DOES: adds `P.userA` to the body's velocity fold, ONCE per substep,
// after the force sum has been divided by N — so it is an ACCELERATION of the
// whole body and not a force summed over members. That is deliberate and it is
// the same choice §20 made for gravity: a term folded per BODY cannot torque it
// (`SUM r_i = 0` only up to the rounding of the shifts, which is exactly why
// gravity is not folded per member for a free body). A car you drive with the
// arrow keys must not spin because you pressed right.
//
// AN AXIS BODY IS NOT DRIVEN, and that is not an omission. §19.B's rule is that
// the axle supplies whatever force holds the centre, so the linear fold is not
// computed at all for a pinned body — gravity is absorbed by the axle and so is
// the player. A linear drive cannot spin a balanced wheel any more than gravity
// can (a wheel pinned at its own centroid reads theta EXACTLY 0 after 10 000
// substeps of gravity; test/axis.html X1). Driving a wheel needs a TORQUE
// input, which is a different quantity from an arrow key, and inventing one
// here would be inventing physics to make a gate green. Measured and gated
// instead: a user-flagged axis body reads omega 0 and theta 0 exactly.
const RB_USER   : u32 = 28u;   // 1 = the player's keys drive this body (§24)

// ---- BAM / the Q22 quarter-turn cosine table -------------------------------
const BAM_QUARTER : i32 = 1073741824;   // 2^30 == pi/2
const BAM_PER_RAD : i32 =  683565276;   // 2^32 / 2pi, rounded
const RG_QSH      : u32 = 20u;          // 30 - QBITS(10): BAM per table step
const RG_QMASK    : i32 = 1048575;      // (1 << RG_QSH) - 1
const COS_Q       : u32 = 22u;
const COS_ONE     : i32 = 4194304;

// ---- accumulator bounds ----------------------------------------------------
// |r >> rsh| <= RS_TARGET, so ONE raw torque term is at most
// 2 * 2048 * dpMax(16384) = 2^26 and TQ_TERM_R is 2x that.
const RS_TARGET   : i32 = 2048;
const TQ_TERM_R   : i32 = 134217728;    // 2^27
const NEG_TQ_TERM : i32 = -134217728;
const TQ_HI_SH    : u32 = 15u;          // hi/lo split so a 16384-member body fits
const NMEM_MAX_R  : i32 = 16384;
const RIGID_MIN_MEMBERS : i32 = 4;      // below this a body DISSOLVES (§20.4.4)
const SUMI_MAX    : i32 = 1073741824;   // 2^30

/**
 * cos(a) in Q22 for a BAM angle: quarter-turn fold + linear interpolation, all
 * integer. The three folds are exact negations / complements of the first, so
 * cos(-a) == cos(a) and sin(-a) == -sin(a) BIT-FOR-BIT — which is what stops a
 * symmetric body creeping.
 *
 * Q16 IS NOT ENOUGH and that cost a measurement pass to see: at 1024 steps per
 * quarter turn the interpolation error is h^2/8 = 2.94e-7 but a Q16 entry is
 * quantised at 1.53e-5, FIFTY TIMES LARGER. Q22 puts the entry quantisation
 * (2.4e-7) just under the interpolation bound, which is where a table belongs.
 *
 * ONE i64 INTERMEDIATE is mandatory: (c1-c0)*frac reaches 6434 * 2^20 = 6.7e9,
 * i.e. past i32. fp_mul_s is that primitive.
 */
fn rg_cos(a : i32) -> i32 {
  let q = (bitcast<u32>(a) >> 30u) & 3u;
  let w = a & 1073741823;                       // within-quadrant BAM, [0, 2^30)
  var u : i32;
  var neg : bool;
  if (q == 0u)      { u = w;                neg = false; }
  else if (q == 1u) { u = BAM_QUARTER - w;  neg = true;  }
  else if (q == 2u) { u = w;                neg = true;  }
  else              { u = BAM_QUARTER - w;  neg = false; }
  let idx  = bitcast<u32>(u) >> RG_QSH;         // 0..1024
  let frac = u & RG_QMASK;
  let c0 = rigid[RG_COS_BASE + idx];
  let c1 = rigid[RG_COS_BASE + idx + 1u];
  let v  = c0 + fp_mul_s(c1 - c0, frac, RG_QSH);
  if (neg) { return -v; }
  return v;
}
fn rg_sin(a : i32) -> i32 { return rg_cos(a - BAM_QUARTER); }

/**
 * Round-half-to-EVEN integer divide. SIGN-SYMMETRIC by construction:
 * rg_idiv_rne(-a, n) == -rg_idiv_rne(a, n) for every a. That symmetry is what
 * keeps a dry symmetric body EXACTLY still (gate G6), and the remainder
 * companion below is what makes the whole fold exactly momentum-conserving.
 * One i32 divide; callers guarantee n >= 1.
 */
fn rg_idiv_rne(a : i32, n : i32) -> i32 {
  if (n <= 0) { return 0; }
  let neg = a < 0;
  var x : i32;
  if (neg) { x = -a; } else { x = a; }
  var q = x / n;
  let r2 = 2 * (x - q * n);
  if (r2 > n || (r2 == n && (q & 1) == 1)) { q = q + 1; }
  if (neg) { return -q; }
  return q;
}

/**
 * THE SAME-BODY TEST. Two particles are in the same rigid body iff both wear
 * FLAG_RIGID and carry the same weld tag — and `flags` is a word of Particle, so
 * this needs NO new buffer in `solveB_c*` or in `friction`. That is the whole
 * storage argument of §20: `friction` and `bondForm1` are the two pipelines
 * already AT the 8-buffer device cap and neither gains a binding.
 *
 * WHY MEMBERS SKIP EACH OTHER AT ALL. It changes nothing for the fluid: this
 * solver is gather-only and self-indexed, so particle i's dp is computed by i's
 * own lane and affects only i; a fluid neighbour j computes its own dp from the
 * member's PLACED position and is untouched. Internal DENSITY force is central
 * and cancels in the torque anyway. Internal FRICTION is tangential and does
 * NOT cancel — measured in prior work as a phantom brake that bled omega from
 * -7.0e6 to -1.1e5 in twenty substeps in a scene with no water at all. A shader
 * that omits this rule ships a solid with a mystery drag that gets worse the
 * more members it has.
 */
fn rg_same_body(fi : u32, fj : u32) -> bool {
  return (fi & FLAG_RIGID) != 0u && (fj & FLAG_RIGID) != 0u &&
         weld_tag(fi) == weld_tag(fj) && weld_tag(fi) != 0u;
}

// Workgroup scratch shared by scanBlock and scanTop (both are 256 threads).
// 256 * 4 = 1024 B, well under maxComputeWorkgroupStorageSize = 16384.
var<workgroup> sdata : array<u32, 256>;

// Inclusive Hillis-Steele scan of sdata over 256 lanes. Atomic-free, fixed
// barrier schedule of integer adds => deterministic by construction.
// The read / barrier / write / barrier shape avoids the WAR hazard.
// MUST be called from uniform control flow by all 256 invocations.
fn wg_scan_inclusive(tid : u32) {
  for (var off : u32 = 1u; off < 256u; off = off << 1u) {
    var add : u32 = 0u;
    if (tid >= off) { add = sdata[tid - off]; }
    workgroupBarrier();
    if (tid >= off) { sdata[tid] = sdata[tid] + add; }
    workgroupBarrier();
  }
}

// ════════════════════════════════════════════════════════════════════════════
// §23.B — THE JET FIELD AT A POINT.
//
// OE-CAKE's Jet "adds force" and 3 918 of the corpus's 4 182 Jet particles
// (93.7 %) sit on a Rigid or a Wall particle — it is a FIXTURE that pushes,
// not a substance. So it is a device, and this is the whole of it: a sum of
// constant accelerations over the jets whose disc contains `p`.
//
// EVERY PROPERTY THE SIM CONTRACT ASKS FOR FALLS OUT OF THAT SHAPE:
//   * ORDER-INDEPENDENT — it reads only the uniform and the particle's own
//     position. There is no pair, no accumulation, no atomic.
//   * DETERMINISTIC — integer only, no divide, no RNG, no clock. `P.nJets` is
//     host state written by `setJets`, exactly like `weldHold` and `nBodies`.
//   * NOT AN EARLY EXIT — the loop bound is uniform across the whole dispatch,
//     so no lane's control flow depends on another lane's data.
//   * BIT-INERT AT nJets == 0 — the loop never runs, the result is (0, 0), and
//     both call sites add it with `fp_add_sat`, for which x + 0 == x exactly.
//
// The sum saturates rather than wraps, so overlapping jets are additive up to
// the i32 rail and never flip sign.
// ════════════════════════════════════════════════════════════════════════════
fn jet_accel(p : vec2<i32>) -> vec2<i32> {
  var a = vec2<i32>(0, 0);
  for (var k : u32 = 0u; k < P.nJets; k = k + 1u) {
    let jp = P.jetPos[k];
    let dx = fp_sub_sat(p.x, jp.x) >> JET_SH;
    let dy = fp_sub_sat(p.y, jp.y) >> JET_SH;
    let rr = jp.z >> JET_SH;
    if (dx * dx + dy * dy <= rr * rr) {
      let jf = P.jetF[k];
      a.x = fp_add_sat(a.x, jf.x);
      a.y = fp_add_sat(a.y, jf.y);
    }
  }
  return a;
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 1 — predict
// v += stAcc + gravity ; clamp per component ; pred = pos + v ; wall clamp.
// v is NOT written back: finalize recovers it as pred - pos, which also folds
// in the wall clamp for free.
// bindings: 0 (uniform), 1, 3, 13 [, 15 debug]
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn predict(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.nFluid) { return; }

  let M  = mats[state_in[i].matId];
  // §17. RETIRED MATTER IS FROZEN WHERE IT FELL. `M` is already loaded, so this
  // costs no new memory traffic. pred = pos means `finalize` recovers
  // v = pred - pos = 0 and writes pos back unchanged, so a corpse's pos and vel
  // are constant forever from the substep after it dies — which is the property
  // the state hash rests on (a corpse that coasted would move the chain, and
  // that is exactly what we want a divergence to do).
  if (M.phantom != 0) { derived[i].pred = state_in[i].pos; return; }
  // §19.A. ANCHORED MATTER NEVER MOVES. Same shape, same consequence as the
  // line above: pred == pos makes `finalize` recover v = 0 and write pos back.
  if ((state_in[i].flags & FLAG_ANCHOR) != 0u) { derived[i].pred = state_in[i].pos; return; }
  // §20. A MEMBER OF A RIGID BODY IS ALREADY PREDICTED. `rigidStep` ran first in
  // this substep and PLACED it at c + R(theta)*o; integrating it again here
  // would give it a second, independent trajectory and is exactly the flexion
  // §20 exists to remove. Its surface tension has already been folded into the
  // body's force and torque, so stAcc is consumed, not dropped.
  if ((state_in[i].flags & FLAG_RIGID) != 0u) { return; }
  let st = derived[i].stAcc;
  let vmaxN = fp_neg_sat(P.vmax);

  // §23.B. A jet is an ACCELERATION, so it enters exactly where gravity does
  // and in the same units. It is read at the particle's CURRENT position, one
  // substep late, which is the same convention `stAcc` already uses (§5.2).
  // Note what does NOT reach this line: retired matter and anchored matter both
  // returned above, so a jet cannot move a corpse and cannot blow a wall over —
  // and a rigid member returned too, because its jet is applied to the BODY in
  // `rigidStep`, where it produces torque as well as force.
  let jf = jet_accel(state_in[i].pos);

  var v = state_in[i].vel;
  v.x = fp_add_sat(fp_add_sat(v.x, st.x), jf.x);
  v.y = fp_add_sat(fp_add_sat(fp_add_sat(v.y, st.y), M.grav), jf.y);
  v = clamp(v, vec2<i32>(vmaxN, vmaxN), vec2<i32>(P.vmax, P.vmax));

  let pos = state_in[i].pos;
  let pr  = vec2<i32>(fp_add_sat(pos.x, v.x), fp_add_sat(pos.y, v.y));
  derived[i].pred = wall_clamp(pr);
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 2 — gridCount
// Histogram of predicted positions into cells. Processes ALL n: boundary
// particles must be in the grid or the fluid cannot see the walls.
// The atomicAdd is a pure counting reduction — exact and order-independent.
// bindings: 0, 3, 4, 7
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn gridCount(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let c = cell_index(cell_coord(derived[i].pred));
  cellOf[i] = c;
  atomicAdd(&cellCount[c], 1u);
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 3 — scanBlock
// Exclusive prefix sum of cellCount within blocks of 1024 cells
// (256 threads x 4 elements). Emits the block total into blockSums.
// Dispatch BG = ceil(cellTotal / 1024) workgroups.
// bindings: 0, 4, 5, 6
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn scanBlock(@builtin(workgroup_id) wg : vec3<u32>,
             @builtin(local_invocation_id) lid : vec3<u32>) {
  let tid = lid.x;
  let ct  = P.cellTotal;
  let base = wg.x * 1024u + tid * 4u;

  var v0 : u32 = 0u; var v1 : u32 = 0u; var v2 : u32 = 0u; var v3 : u32 = 0u;
  if (base + 0u < ct) { v0 = atomicLoad(&cellCount[base + 0u]); }
  if (base + 1u < ct) { v1 = atomicLoad(&cellCount[base + 1u]); }
  if (base + 2u < ct) { v2 = atomicLoad(&cellCount[base + 2u]); }
  if (base + 3u < ct) { v3 = atomicLoad(&cellCount[base + 3u]); }
  let s = v0 + v1 + v2 + v3;

  sdata[tid] = s;
  workgroupBarrier();
  wg_scan_inclusive(tid);

  let excl = sdata[tid] - s;              // exclusive prefix for this thread
  let total = sdata[255u];
  workgroupBarrier();                     // sdata dead after this point

  if (base + 0u < ct) { cellStart[base + 0u] = excl; }
  if (base + 1u < ct) { cellStart[base + 1u] = excl + v0; }
  if (base + 2u < ct) { cellStart[base + 2u] = excl + v0 + v1; }
  if (base + 3u < ct) { cellStart[base + 3u] = excl + v0 + v1 + v2; }
  if (tid == 0u) { blockSums[wg.x] = total; }
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 4 — scanTop
// Exclusive prefix sum over the block totals, in place, in ONE workgroup.
// 256 threads x 4 => at most 1024 blocks => at most 1048576 cells. engine.js
// throws above that at init; dbg[7] is the belt-and-braces detector because a
// silent wrong offset here corrupts every downstream result.
// bindings: 0, 6 [, 15 debug]
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn scanTop(@builtin(local_invocation_id) lid : vec3<u32>) {
  let tid = lid.x;
  let nb  = (P.cellTotal + 1023u) / 1024u;      // number of blocks

  //@DBG_BEGIN
  if (tid == 0u && nb > 1024u) { fp_flag(7u); }
  //@DBG_END

  let base = tid * 4u;
  var v0 : u32 = 0u; var v1 : u32 = 0u; var v2 : u32 = 0u; var v3 : u32 = 0u;
  if (base + 0u < nb) { v0 = blockSums[base + 0u]; }
  if (base + 1u < nb) { v1 = blockSums[base + 1u]; }
  if (base + 2u < nb) { v2 = blockSums[base + 2u]; }
  if (base + 3u < nb) { v3 = blockSums[base + 3u]; }
  let s = v0 + v1 + v2 + v3;

  sdata[tid] = s;
  workgroupBarrier();
  wg_scan_inclusive(tid);

  let excl = sdata[tid] - s;
  workgroupBarrier();

  if (base + 0u < nb) { blockSums[base + 0u] = excl; }
  if (base + 1u < nb) { blockSums[base + 1u] = excl + v0; }
  if (base + 2u < nb) { blockSums[base + 2u] = excl + v0 + v1; }
  if (base + 3u < nb) { blockSums[base + 3u] = excl + v0 + v1 + v2; }
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 5 — scanAdd
// Add each block's base offset to its cells and terminate the prefix sum with
// cellStart[cellTotal] = n. Every particle lands in exactly one cell, so the
// total is exactly n by construction.
// Dispatch CG = ceil(cellTotal / 256).
// bindings: 0, 5, 6
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn scanAdd(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i == 0u) { cellStart[P.cellTotal] = P.n; }
  if (i >= P.cellTotal) { return; }
  cellStart[i] = cellStart[i] + blockSums[i >> 10u];
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 6 — scatter
// Counting-sort placement. cellCount has been cleared and is reused as a
// per-cell cursor. Ranges come from the exclusive prefix sum, so they are
// EXACTLY sized: there is no cell capacity and nothing can ever be dropped.
// The order inside a cell here is atomic-arrival order and is therefore NOT
// deterministic — canonicalize fixes that next.
// bindings: 0, 4, 5, 7, 8
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let c = cellOf[i];
  let off = atomicAdd(&cellCount[c], 1u);
  bucketIds[cellStart[c] + off] = i;
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 7 — canonicalize  ← THE DETERMINISM CRUX
// rank(p) = |{q in cell(p) : q < p}| is an order-INDEPENDENT counting
// reduction, and particle ids are unique so there are no ties. After this the
// occupants of every cell sit in ascending id order, which makes the neighbour
// list a pure function of the particle set — independent of atomic arrival
// order, thread scheduling and submit batching.
// bindings: 0, 5, 7, 8, 9
//
// ─── DO NOT DELETE THIS PASS ON THE STRENGTH OF A PROFILE. Measured 2026-07-25.
// This kernel is bit-for-bit INERT on any scene where dbg[6] == 0, because every
// consumer of the neighbour list is a per-term-clamped plain i32 sum and is
// therefore commutative. So NO benchmark of the shipped toy can ever show it
// earning its O(k^2)-per-cell cost (spec §11 R5 flags exactly that cost), and
// deleting it will look free.
//
// It is not free. The MAXNBR truncation in buildNbr below is the one
// order-sensitive consumer in the pipeline — it decides WHICH neighbours get
// dropped — and the moment local density passes ~2.5x rest, defeating this pass
// gives 5 unique chains in 5 runs, diverging at substep 1. Measured on
// overdense_v1: shipped 1 unique / 5 runs, defeated 5 unique / 5 runs. A
// different-but-deterministic order (descending id) is self-consistent and still
// yields a DIFFERENT chain, so the ascending-id order here is load-bearing wire
// format for product 2, not an implementation detail.
//
// Any future removal must be argued against test/determinism.html stage C, which
// exists solely to make this measurable. Do not "fix" a red stage C by deleting
// stage C.
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn canonicalize(@builtin(global_invocation_id) gid : vec3<u32>) {
  let s = gid.x;
  if (s >= P.n) { return; }
  let me = bucketIds[s];
  let c  = cellOf[me];
  let lo = cellStart[c];
  let hi = cellStart[c + 1u];
  var rank : u32 = 0u;
  for (var t = lo; t < hi; t = t + 1u) {
    if (bucketIds[t] < me) { rank = rank + 1u; }
  }
  sortedIds[lo + rank] = me;
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 8 — buildNbr
// Fixed 3x3 stencil traversal: dy = -1,0,+1 outer, dx = -1,0,+1 inner, and
// ascending sortedIds within each cell. The kernel-range test lives here so the
// solver's seven pair loops never re-test it, and so the list holds ~20 real
// neighbours instead of ~144 raw candidates.
// Truncation at MAXNBR is canonical (it drops the tail of THIS fixed order) and
// raises dbg[6]; CI asserts dbg[6] == 0.
//
// §17 — THIS PASS IS WHERE RETIREMENT ACTUALLY HAPPENS, in two tests.
// Everything downstream (solveA, solveB_at, friction, xsph, normals,
// surfaceTension, conduct — all seven pair loops) is a gather over
// nbr[base .. base+cnt), so a particle that is absent from every list exerts no
// force on anyone and a particle with cnt == 0 receives none. It is sufficient
// BY THE SHAPE OF THE SOLVER rather than by tuning: solveA with cnt == 0 reads
// rho = luts[OFF_W] = 214942 against RHO0 = 1048576, and the density constraint
// is COMPRESSION-ONLY, so Cq = clamp((rho - RHO0) >> 4, 0, C_MAX) = 0, lam = 0
// and dp = (0,0) exactly.
// bindings: 0, 1, 3, 5, 9, 10, 11 [, 15 debug]   (state_in @1 is new in v7)
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn buildNbr(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  // §17 a) a corpse has no neighbours. (This is NOT the fluid-only guard: it is
  // deliberately after the P.n test so buildNbr keeps dispatching over n —
  // test/perf/dispatch.html asserts that classification.)
  if ((state_in[i].flags & FLAG_DEAD) != 0u) { nbrN[i] = 0u; return; }

  let pi   = derived[i].pred;
  let cc   = cell_coord(pi);
  let base = i * MAXNBR;
  var cnt : u32 = 0u;
  var sealSeen = false;
  var externalSealSeen = false;
  var congealSeen = false;

  for (var dy : i32 = -1; dy <= 1; dy = dy + 1) {
    let gy = cc.y + dy;
    if (gy < 0 || gy >= i32(P.gridH)) { continue; }
    for (var dx : i32 = -1; dx <= 1; dx = dx + 1) {
      let gx = cc.x + dx;
      if (gx < 0 || gx >= i32(P.gridW)) { continue; }
      let c  = u32(gy) * P.gridW + u32(gx);
      let lo = cellStart[c];
      let hi = cellStart[c + 1u];
      for (var s = lo; s < hi; s = s + 1u) {
        let j = sortedIds[s];
        if (j == i) { continue; }
        // §17 b) nobody can see a corpse. THIS TEST MUST COME BEFORE THE RANGE
        // TEST AND BEFORE `cnt`: skipping a dead candidate after the count would
        // let it consume a MAXNBR slot, and then WHERE the corpses are would
        // change WHICH live neighbours got truncated away. Placed here, a
        // retired particle is exactly as absent as one that was never there —
        // which is what makes the §17 exactness gate (elements.html E18) an
        // equality on the live particles' hashed words rather than a tolerance.
        if ((state_in[j].flags & FLAG_DEAD) != 0u) { continue; }
        if (lut_index(pred_delta(pi, j)) < 0) { continue; }
        if (j < P.nFluid && (state_in[j].flags & FLAG_SEAL) != 0u) {
          sealSeen = true;
          if ((state_in[i].flags & FLAG_RIGID) != 0u &&
              (state_in[j].flags & FLAG_ANCHOR) != 0u &&
              !rg_same_body(state_in[i].flags, state_in[j].flags)) {
            externalSealSeen = true;
          }
        }
        if (j < P.nFluid && (state_in[j].flags & FLAG_CONGEAL) != 0u) {
          congealSeen = true;
        }
        if (cnt >= MAXNBR) {
          //@DBG_BEGIN
          fp_flag(6u);
          //@DBG_END
          continue;
        }
        nbr[base + cnt] = j;
        cnt = cnt + 1u;
      }
    }
  }
  nbrN[i] = cnt |
    select(0u, NBR_SEAL_BIT, sealSeen) |
    select(0u, NBR_EXT_SEAL_BIT, externalSealSeen) |
    select(0u, NBR_CONGEAL_BIT, congealSeen);
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 9,18,27,36 — solveA : density, |grad C|^2, lambda  (gather)
// No atomics: every term is read by the owning invocation and summed locally,
// so there is no scatter hazard and no saturating-atomic problem.
// Boundary neighbours ARE included — that is the entire point of them.
//
// §5.1b: solveA is NOT coloured. lambda stays JACOBI (one shared solve per
// iteration, evaluated on the pred at the start of the iteration); only the
// position update is Gauss-Seidel. That is not a shortcut, it is the measured
// better configuration: colouring solveA as well costs 3 more dispatches per
// iteration AND lands a HIGHER psi6 (measured on the §C17 scene at 5000
// substeps: shared-lambda 0.6127, fully coloured 0.6364, shipped Jacobi 0.8036).
// bindings: 0, 1, 3, 10, 11, 12, 13 [, 15 debug]
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn solveA(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.nFluid) { return; }

  let M    = mats[state_in[i].matId];
  let pi   = derived[i].pred;
  let cnt  = nbr_count(i);
  let base = i * MAXNBR;

  var rho : i32 = luts[OFF_W];               // self term, WT[0]
  var S   : i32 = 0;                         // Q21
  var sgx : i32 = 0;
  var sgy : i32 = 0;

  for (var k : u32 = 0u; k < cnt; k = k + 1u) {
    let j  = nbr[base + k];
    let d  = pred_delta(pi, j);
    let id = lut_index(d);
    if (id < 0) { continue; }
    let ui = u32(id);

    rho = rho + luts[OFF_W + ui];

    let gt = luts[OFF_G + ui];
    let gx = grad_q16(gt, d.x);
    let gy = grad_q16(gt, d.y);
    sgx = sgx + gx;
    sgy = sgy + gy;

    // PER-TERM clamp: order-independent AND provably bounds the sum. Clamping a
    // RUNNING TOTAL would be order-dependent and would destroy bit-exactness.
    S = S + clamp(fp_mul_s(gx, gx, 11u), 0, S_TERM)
          + clamp(fp_mul_s(gy, gy, 11u), 0, S_TERM);
  }
  // the k == i term
  S = S + clamp(fp_mul_s(sgx, sgx, 11u), 0, S_TERM)
        + clamp(fp_mul_s(sgy, sgy, 11u), 0, S_TERM);

  // COMPRESSION ONLY. Unclamped, a particle with 1-3 neighbours has S ~ 0 and
  // gets lambda ~ 4.4e7 (42 wu^2), which detonates free surfaces. RHO0 = 2^20
  // exactly, so this is a pure rounded shift.
  let Cq = clamp(fp_shr_rne(rho - RHO0, 4u), 0, C_MAX);      // Q16

  derived[i].rho = rho;
  // lambda = -C/(S+eps): C is Q16, den is Q21, want Q20 -> shift 25.
  derived[i].lam = clamp(fp_neg_sat(fp_divshift(Cq, S + M.eps, 25u)),
                         NEG_LAM_MAX, LAM_MAX);
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 10-11,14-15,… — solveB : position correction, ONE COLOUR AT A TIME
//
// Writes derived.dp (never read in this dispatch). It must NOT write pred:
// pred[j] is read here, so writing pred[i] would be a genuine intra-dispatch
// data race regardless of how stable it looks on any one driver (§5.1).
//
// THE COLOURING DOES NOT CHANGE THAT. The tempting saving is to have the
// coloured pass write pred[i] directly and drop applyDp — 4 dispatches per
// iteration. It is broken: ~51% of interacting pairs are INTRA-CELL and so share
// a colour, which makes the fused form a live intra-dispatch race. Measured on a
// settled 5,500-particle pool at 300 substeps: deferred-dp 1 unique chain in 3
// runs, fused 3 unique chains in 3 runs. Do not fuse these two passes.
//
// The single shared body lives in solveB_at so the four colours cannot drift.
// bindings: 0, 1, 3, 7, 10, 11, 12, 13 [, 15 debug]  ← 8 storage, AT THE CAP
// ════════════════════════════════════════════════════════════════════════════
fn solveB_at(i : u32) {
  let pi   = derived[i].pred;
  let li   = derived[i].lam;
  let cnt  = nbr_count(i);
  let base = i * MAXNBR;
  // §20. `flags` arrives with the Particle struct, so the same-body test costs
  // ONE new binding on this pass (state_in, 6 -> 7 storage) and none at all on
  // the two passes that are at the device cap.
  let fi = state_in[i].flags;
  let rigidI = (fi & FLAG_RIGID) != 0u;

  var dpx : i32 = 0;                          // carried at ticks << ACC
  var dpy : i32 = 0;

  for (var k : u32 = 0u; k < cnt; k = k + 1u) {
    let j  = nbr[base + k];
    // §20. A member does not push on its own body. solveA is UNTOUCHED, so the
    // member still has a correct rho and lambda and the fluid cannot walk
    // through it; only the member's OWN correction drops the internal terms.
    if (rigidI && !is_boundary(j) && rg_same_body(fi, state_in[j].flags)) { continue; }
    let d  = pred_delta(pi, j);
    let id = lut_index(d);
    if (id < 0) { continue; }
    let ui = u32(id);

    // Mirrored wall lambda: a boundary particle answers with the fluid
    // particle's own lambda, which makes the wall push back exactly as hard as
    // the fluid pushes in. (Boundary lam is never written, so this also avoids
    // consuming an undefined value.)
    var lj = li;
    if (!is_boundary(j)) { lj = derived[j].lam; }

    let co = fp_add_sat(fp_add_sat(li, lj), luts[OFF_S + ui]);      // Q20
    let gt = luts[OFF_G + ui];
    dpx = dpx + clamp(fp_mul_s(co, grad_q16(gt, d.x), 20u - ACC), NEG_DP_TERM, DP_TERM);
    dpy = dpy + clamp(fp_mul_s(co, grad_q16(gt, d.y), 20u - ACC), NEG_DP_TERM, DP_TERM);
  }
  derived[i].dp = vec2<i32>(dpx, dpy);
}

// The four colour entry points. Each is a guard plus a call — the physics is in
// solveB_at, once. The `i >= P.nFluid` test comes FIRST so out-of-range lanes
// never index cellOf past n.
@compute @workgroup_size(256)
fn solveB_c0(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.nFluid) { return; }
  if (cell_colour(i) != 0u) { return; }
  solveB_at(i);
}
@compute @workgroup_size(256)
fn solveB_c1(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.nFluid) { return; }
  if (cell_colour(i) != 1u) { return; }
  solveB_at(i);
}
@compute @workgroup_size(256)
fn solveB_c2(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.nFluid) { return; }
  if (cell_colour(i) != 2u) { return; }
  solveB_at(i);
}
@compute @workgroup_size(256)
fn solveB_c3(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.nFluid) { return; }
  if (cell_colour(i) != 3u) { return; }
  solveB_at(i);
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 12-13,16-17,… — applyDp, ONE COLOUR AT A TIME
// Trivially elementwise: reads only its own dp, writes only its own pred.
// This is the cheapest possible way to be unambiguously race-free, and it is
// what makes the coloured sweep Gauss-Seidel: colour K's positions are already
// updated when colour K+1's solveB reads them.
// The safety clamp is applied AFTER the full sum, so it stays order-independent.
// bindings: 0, 1, 3, 7, 13 [, 15 debug]
// ════════════════════════════════════════════════════════════════════════════
fn applyDp_at(i : u32) {
  // §20. A MEMBER'S dp IS NOT ITS OWN. It is scattered to the body by
  // `rigidSolve` at the end of this iteration and comes back as a rigid motion
  // of the WHOLE body. Applying it here as well would move the member
  // independently, which is the definition of deforming.
  //
  // This early return is also what makes §20 safe under §5.1b's 4-colour block
  // Gauss-Seidel: a member is never WRITTEN inside a colour pass, so within one
  // colour it is a PURE READER, and §5.1a only forbids two neighbouring
  // particles being WRITTEN in the same pass. A body's members may therefore
  // span all four colours and the colouring never has to know bodies exist.
  if ((state_in[i].flags & FLAG_RIGID) != 0u) { return; }
  // §19.A. ANCHORED MATTER ACCUMULATES dp AND IGNORES IT. It cannot be dropped
  // from the neighbour lists the way §17's corpses are — every pair term must
  // still see it, carrying its own material row — so this guard is what makes
  // it immovable, and it covers all four colour variants and applyFric at once.
  if ((state_in[i].flags & FLAG_ANCHOR) != 0u) { return; }
  let M  = mats[state_in[i].matId];
  let dm = M.dpMax;
  let dmN = fp_neg_sat(dm);
  let dp = derived[i].dp;

  let d = vec2<i32>(clamp(fp_shr_rne(dp.x, ACC), dmN, dm),
                    clamp(fp_shr_rne(dp.y, ACC), dmN, dm));
  let pr = derived[i].pred;
  let np = vec2<i32>(fp_add_sat(pr.x, d.x), fp_add_sat(pr.y, d.y));
  derived[i].pred = wall_clamp(np);
}

@compute @workgroup_size(256)
fn applyDp_c0(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.nFluid) { return; }
  if (cell_colour(i) != 0u) { return; }
  applyDp_at(i);
}
@compute @workgroup_size(256)
fn applyDp_c1(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.nFluid) { return; }
  if (cell_colour(i) != 1u) { return; }
  applyDp_at(i);
}
@compute @workgroup_size(256)
fn applyDp_c2(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.nFluid) { return; }
  if (cell_colour(i) != 2u) { return; }
  applyDp_at(i);
}
@compute @workgroup_size(256)
fn applyDp_c3(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.nFluid) { return; }
  if (cell_colour(i) != 3u) { return; }
  applyDp_at(i);
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 45 — friction  (§16 THE GRANULAR TERM — "sand must pile")
//
// THE PROBLEM THIS EXISTS TO SOLVE. PBF's only stress is a density constraint
// that is COMPRESSION-ONLY, so at rest it can push two particles apart and can
// do NOTHING to stop them sliding past each other. That is the correct model of
// an inviscid liquid and it is the WRONG model of everything else: measured on
// the shipped v5 build, an identical 113-particle blob of every material in the
// roster spread from 12 wu to 109-127 wu in a 128 wu tank within 2-4 s and then
// stopped dead. Wood and stone landed on IDENTICAL numbers. Twenty-four
// materials, twenty-four differently-coloured puddles, and an imported picture
// was a picture for less than one second.
//
// No amount of table tuning fixes that, because there is no table entry for the
// missing physics: a solid needs to resist SHEAR AT ZERO VELOCITY, and nothing
// in §5/§6 can. Viscosity (xsph) only slows the collapse; cohesion (gammaCoh)
// pulls toward a sphere, not toward the shape you drew.
//
//   Δx_i = pred_i - pos_i                     this substep's total displacement
//   rel  = Δx_i - Δx_j                        relative displacement of the pair
//   t_ij = rel - d̂(rel·d̂)                     its TANGENTIAL part
//   corr_i = -Σ_j  μ_ij · (W_ij/ρ0) · t_ij
//
// i.e. Coulomb friction as a POSITION correction (Macklin 2014 §6), applied
// once after the solver loop and before `finalize` — so `finalize`'s
// v = pred - pos picks it up and the velocity follows the position for free.
//
// WHY TANGENTIAL AND NOT THE WHOLE RELATIVE DISPLACEMENT. Damping `rel` itself
// is one line shorter and it is wrong: it fights the density constraint in the
// NORMAL direction, undoing the separation the solver just computed, and at
// μ→1 the density solve stops working entirely. Removing the normal component
// is what makes this a friction term rather than a glue term.
//
// THE PROJECTION IS DONE WITHOUT A SQUARE ROOT. d̂(rel·d̂) = d·(rel·d)/|d|², so
// one fp_divshift replaces a normalise. That matters for more than speed: it
// keeps every step an exact integer operation whose sign-symmetry is provable.
//
// WHY THIS IS ORDER-INDEPENDENT — the only property that matters for the
// determinism contract. It is a per-particle GATHER whose accumulator is a
// per-term-clamped plain i32 sum: exactly the shape solveA, xsph, normals,
// surfaceTension and conduct already have, and exactly what determinism.html
// stage C2 and test/elements.html e10 exist to protect. No running-total clamp,
// no min/max reduction, no early break, no atomic.
//
// WHY IT OBEYS NEWTON'S THIRD LAW EXACTLY (fluid-fluid pairs):
//   d_ji  = -d_ij           exactly; fp_sub_sat cannot saturate at |d| <= H
//   rel_ji= -rel_ij         exactly
//   dot   = (-d)·(-rel) = dot        SYMMETRIC (exact integer products)
//   d2                                SYMMETRIC
//   proj_ji = fp_mul_s(-d4, q, 16) = -proj_ij     (fp_mul_s is sign-symmetric)
//   t_ji  = rel_ji - proj_ji = -t_ij  ANTISYMMETRIC
//   W_ij  symmetric (lut_index rounds |d| symmetrically — load-bearing)
//   μ_ij  symmetric (mat_mix is the arithmetic mean, §6.2)
//   clamp(x,-k,+k) symmetric about zero
// therefore term_ij == -term_ji EXACTLY. The pre-shifts use fp_shr_rne, NOT the
// fp_abs form lut_index uses, precisely because this term must keep its sign
// symmetry and fp_abs would destroy it.
//
// WALLS ARE THE EXCEPTION AND MUST BE. A boundary particle does not move, so
// Δx_j = 0 and the reaction has nowhere to go — this is a genuine momentum SINK
// against an immovable object, exactly like `adhesion` (§6.2), and it is what
// stops a pile from sliding along the floor it is standing on. Wall friction
// therefore uses the material's OWN μ and is NOT pair-mixed, for the same
// reason adhesion is not: boundary particles carry matId = WATER as a
// placeholder, so mixing would silently halve every solid's grip on the floor.
//
// μ = 0 MAKES THIS PASS BIT-FOR-BIT INERT. Every term multiplies by μ, so a
// material with fric = 0 accumulates exactly 0 and `applyDp_at` then adds
// clamp(0) to pred. WATER and GOO ship at 0, which is what keeps the §8.5 feel
// suite and the 10 000-substep golden chain comparable across v5 -> v6 —
// asserted in test/granular.html G0, not assumed.
//
// bindings: 0, 1, 3, 10, 11, 12, 13 [, 15 debug]
// ════════════════════════════════════════════════════════════════════════════

// Pre-shift on the pair separation, mirroring lut_index's PRE — but SIGNED and
// round-to-nearest-even, so d4(-x) == -d4(x) exactly.
// THE TWO PRE-SHIFTS ARE A PRECISION BUDGET, NOT A FORMALITY. They were the one
// thing this pass got wrong on the first attempt and the failure mode was
// silent: at RPRE = 5 a settled particle's relative displacement (~300 ticks)
// pre-shifted to r4 = 9, its tangential part rounded to ~3, and the very next
// multiply (tx * wq >> 16, with wq ~ 2621 for a neighbour at rest spacing)
// rounded THAT to 0. The pass ran, the chain moved, dbg stayed clean, and a
// 12x12 sand block still spread 119.09 -> 117.30 wu across the whole mu sweep
// — i.e. friction measurably did nothing while looking alive.
//   FPRE 5: |d4| <= H >> 5 = 5120, so d2 <= 2 * 5120^2 = 5.2e7 and
//           dot <= 2 * 5120 * 65536 = 6.7e8 — 3.2x headroom to 2^31.
//           At rest spacing (1 wu) d4 = 2048, which is ample resolution for d̂.
//   RPRE 3: |rel| <= 2^19 (vmax 65536 + iters * dpMax, both ends) so
//           |r4| <= 65536, and a settled 300-tick relative displacement keeps
//           ~6 bits instead of ~3.
// The multiply ORDER matters for the same reason and is fixed below: the weight
// is applied at (16 - ACC) so the ACC guard bits are carried THROUGH it, and mu
// is applied last.
const FPRE   : u32 = 5u;
const RPRE   : u32 = 3u;
//
// D2_FLOOR is the whole reason `q` cannot blow up. q = dot/d2 has magnitude
// |rel4| / |d4|, so a pair at zero separation would divide by zero. Below
// 0.25 wu of separation the pair is skipped: |d4| >= 512 => d2 >= 262144, so
// |q| <= 65536/512 = 128, i.e. |q| in Q16 <= 2^23. Skipping is safe because a
// pair that close is already deep inside the density constraint's job.
const D2_FLOOR : i32 = 262144;    // (0.25 wu >> 5)^2
// Per-term clamp, ticks<<(ACC-RPRE). Sum bound MAXNBR * FR_TERM = 48 * 2^24 =
// 8.05e8 — the same 2.7x headroom to 2^31 the heat and corrosion accumulators
// have, and for the same reason.
const FR_TERM     : i32 =  16777216;    //  2^24
const NEG_FR_TERM : i32 = -16777216;

// ════════════════════════════════════════════════════════════════════════════
// §18 — BONDED RIGID BODIES (v8)
// ----------------------------------------------------------------------------
// A bond is a persistent distance constraint between two particle SLOTS, stored
// in a dense zero-terminated row of BOND_CAP words per particle. It is the only
// term in the solver that remembers anything about the past, which is why it is
// hashed (§7.6) and why eraseWhere has to remap it.
//
// THE PACKED WORD (§18.5):
//   bit  31   | 30   | 29 ................ 17 | 16 ................. 0
//        RSV  | TOMB | L0q  (13 bits, <=5120) | partner slot (17 bits, <=131071)
//
//   w == 0                 => EMPTY SLOT, and it TERMINATES the dense row
//   (w & BOND_TOMB) != 0   => BROKEN: occupied, inert, dropped by next bondForm1
//   bit 31                 => RESERVED, MUST BE ZERO
//
// 17 + 13 + 1 + 1 = 32 exactly. The tombstone NEEDS its own bit and cannot be
// `L0q == 0`: a tombstoned bond to slot 0 would then be the word 0, and the
// dense-row `break` would orphan every live bond after it. With bit 30 it is
// 0x44000000 — non-zero, verified over 411 652 exhaustive round-trips.
// ════════════════════════════════════════════════════════════════════════════
const BOND_CAP     : u32 = 48u;          // == MAXNBR. Also the ROW STRIDE.
const BPRE         : u32 = 5u;           // identical to FPRE, deliberately: bond
                                         // and friction pre-shifts then share one
                                         // meaning and L0 needs no dequantisation
const BOND_R2      : i32 = 17623204;     // (2.05 wu >> BPRE)^2 = 4198^2
const HARD_BREAK2  : i32 = 26214400;     // (H >> BPRE)^2 = 5120^2 — UNCONDITIONAL
const L0_MAX       : i32 = 5120;         // 13 bits, AND the per-axis reject bound
const L0_MIN       : i32 = 512;          // isqrt(D2_FLOOR); a live L0q is never below it
const BOND_J_MASK  : u32 = 0x0001ffffu;  // 17 bits
const BOND_L0_SH   : u32 = 17u;
const BOND_L0_MASK : u32 = 0x3ffe0000u;
const BOND_TOMB    : u32 = 0x40000000u;  // bit 30: broken, occupied, inert
// WELD_TAG_MASK and weld_tag() MOVED UP to the shared-helper block (§20) —
// `solveB_at` and `friction` are earlier in this file and now need them for the
// same-body test. Particle.flags bits 8..19; bits 20..31 stay free ON PURPOSE
// for §17 vent phase 2's generation counter (§18.7 R5).
// Per-term clamp on the bond contribution, ticks << ACC. 2^24 and NOT 2^23:
// the measured peak per-term contribution on a 12x12 STONE block dropped 18 wu
// is 11 954 817, which is 1.43x over 2^23 — a 2^23 clamp would silently limit a
// real impact. The FUSED accumulator bound is therefore
//   BOND_CAP*BD_TERM + MAXNBR*FR_TERM = 48*2^24 + 48*2^24 = 1 610 612 736,
// i.e. 1.33x headroom to 2^31 — the TIGHTEST accumulator in the repo. MAXNBR,
// FRIC_MAX and BD_TERM are now coupled and none may be raised alone (§C6).
// The clamp is sign-symmetric (clamp(-x) == -clamp(x)), so unlike dpMax — which
// clamps the SUM and is what broke Newton's third law in §16.5 — a binding
// BD_TERM cannot break pair antisymmetry.
const BD_TERM      : i32 =  16777216;    //  2^24
const NEG_BD_TERM  : i32 = -16777216;

fn bond_j(w : u32)  -> u32 { return w & BOND_J_MASK; }
// L0q is isqrt(d2) with D2_FLOOR <= d2 <= BOND_R2, so it is in [512, 4198] and
// NEVER negative — which is why the u32 cast below is safe. PLATFORM_NOTES
// flags u32() on a possibly-negative i32 as a cross-vendor hazard; this one is
// guarded by the D2_FLOOR test at the ONE site that packs a word.
fn bond_l0(w : u32) -> i32 { return i32((w >> BOND_L0_SH) & 0x1fffu); }
fn bond_pack(j : u32, l0 : i32) -> u32 {
  return ((u32(l0) & 0x1fffu) << BOND_L0_SH) | (j & BOND_J_MASK);
}

// §18.24 — WHICH ROWS MAY FUSE. A fusion end must be a bonded family that
// holds its shape: bondK != 0 (loose WATER/GOO can never bond at all),
// bondReform == 0 (SAND and cooked RICE are poured heaps, not bodies — the
// brief's "loose matter" exclusion, row-valued), a live group, and never
// PHOTO (an imported picture cutout is excluded by decision; row 15 is the
// authored PHOTO row — engine.js MAT_PHOTO, asserted there against this
// constant's value). Interned §25 rows inherit all four words from their
// recipe, so a composed elastic fuses and a composed rice does not, with no
// new table word and no new binding — bondForm1 is AT the 8-buffer cap.
const MAT_PHOTO_ROW : u32 = 15u;
fn fuse_row(m : Mat, id : u32) -> bool {
  return m.bondK != 0 && m.bondReform == 0 && m.bondGroup >= 0 && id != MAT_PHOTO_ROW;
}

// K(T): one multiply, one shift, NO divide and no transcendental. `rampInv` is
// baked by createEngine as floor(2^32 / (rampHi - rampLo)), never at runtime.
// rampSign == 0 makes the branch bit-for-bit inert, which is what keeps every
// bondK == 0 row — and therefore the all-WATER golden chain — unchanged.
fn bond_k(m : Mat, T : i32) -> i32 {
  if (m.bondK == 0) { return 0; }
  if (m.rampSign == 0) { return m.bondK; }
  let span = fp_sub_sat(m.rampHi, m.rampLo);
  let dT   = clamp(fp_sub_sat(T, m.rampLo), 0, span);      // clamp BEFORE the multiply
  var u    = clamp(fp_mul_s(dT, m.rampInv, 16u), 0, 65536);
  if (m.rampSign < 0) { u = 65536 - u; }
  return fp_mul_s(m.bondK, u, 16u);
}

@compute @workgroup_size(256)
fn friction(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.nFluid) { return; }

  let Mi = mats[state_in[i].matId];
  let fi = Mi.fric;
  // §20. A member ACCUMULATES into derived[i].dp rather than overwriting it:
  // `applyDp_at` early-returned for it, so its solver correction is still
  // sitting there waiting for `rigidSolve` to gather both together. It also
  // gathers NO bonds — a rigid body has no strain, so §18 has nothing to do —
  // and `bondForm1` empties its row on the next formation pass, which drops the
  // other half of any cross-body bond through bondForm2's mutual test.
  let flagsI = state_in[i].flags;
  let rigidI = (flagsI & FLAG_RIGID) != 0u;
  // The zero-work fast path is not an optimisation — it is the guarantee that
  // WATER and GOO are bit-for-bit unchanged from v5. Writing dp = (0,0)
  // explicitly is what makes `applyDp_at` a no-op instead of re-applying the
  // solver's last dp a second time.
  //
  // §18: IT NOW NEEDS BOTH TERMS, and the bond term uses the TABLE value, not
  // the ramped one. A bonded material whose stiffness is momentarily zero (raw
  // RICE at room temperature; STONE mid-melt) must still reach its HARD BREAK,
  // or its bonds become IMMORTAL — measured on the shipped table, 1004 bonds
  // stretched to 316 wu, 754 i32 overflows and 400 silent break failures.
  if (fi == 0 && Mi.bondK == 0) {
    // §20: a member must NOT have its accumulated solver dp wiped here.
    if (!rigidI) { derived[i].dp = vec2<i32>(0, 0); }
    return;
  }

  let pi   = derived[i].pred;
  let dxi  = vec2<i32>(fp_sub_sat(pi.x, state_in[i].pos.x),
                       fp_sub_sat(pi.y, state_in[i].pos.y));
  let cnt  = nbr_count(i);
  let base = i * MAXNBR;

  var fx : i32 = 0;                        // ticks << (ACC - RPRE)
  var fy : i32 = 0;

  // GUARDING THE LOOP AS WELL AS THE EARLY-OUT is load-bearing, not tidiness.
  // No shipped bonded material has fric == 0 today (STONE/METAL 131072,
  // WOOD/ICE/RICE 114688, SAND 98304), so this is defensive — but RUBBER.fric
  // is deliberately 0 and the next bonded material could be, and the failure
  // would be SILENT: the bonds would exist, be hashed, and never apply a force
  // or break.
  if (fi != 0) {
  for (var k : u32 = 0u; k < cnt; k = k + 1u) {
    let j  = nbr[base + k];
    // §20. INTERNAL FRICTION IS NOT OPTIONAL TO EXCLUDE — it is tangential, so
    // unlike the central density force it does not cancel over the body, and it
    // reads as a phantom brake that gets worse the more members there are.
    if (rigidI && !is_boundary(j) && rg_same_body(flagsI, state_in[j].flags)) { continue; }
    let d  = pred_delta(pi, j);
    let id = lut_index(d);
    if (id < 0) { continue; }
    let ui = u32(id);

    let d4x = fp_shr_rne(d.x, FPRE);
    let d4y = fp_shr_rne(d.y, FPRE);
    let d2  = d4x * d4x + d4y * d4y;
    if (d2 < D2_FLOOR) { continue; }

    // Relative displacement. A wall does not move, so the pair term is the
    // fluid particle's own displacement and the wall absorbs the reaction.
    var rel = dxi;
    var mu  = fi;
    if (!is_boundary(j)) {
      let pj = derived[j].pred;
      rel = vec2<i32>(fp_sub_sat(dxi.x, fp_sub_sat(pj.x, state_in[j].pos.x)),
                      fp_sub_sat(dxi.y, fp_sub_sat(pj.y, state_in[j].pos.y)));
      mu  = mat_mix(fi, mats[state_in[j].matId].fric);    // §6.2, symmetric
    }
    if (mu == 0) { continue; }

    let r4x = fp_shr_rne(rel.x, RPRE);
    let r4y = fp_shr_rne(rel.y, RPRE);

    // tangential = rel - d*(rel·d)/|d|^2, all in the >>RPRE unit.
    let dot = d4x * r4x + d4y * r4y;
    let q   = fp_divshift(dot, d2, 16u);                   // Q16, |q| <= 2^20
    let tx  = fp_sub_sat(r4x, fp_mul_s(d4x, q, 16u));
    let ty  = fp_sub_sat(r4y, fp_mul_s(d4y, q, 16u));

    // weight W/RHO0 in Q16. RHO0 = 2^20 exactly, so this is a pure shift.
    // ORDER IS LOAD-BEARING: the weight is applied at (16 - ACC) so the ACC
    // guard bits survive it, and mu comes last. Doing it the other way round
    // rounds the whole term to zero at rest — see the FPRE/RPRE note above.
    //
    // THE SHIFT IS (16 - ACC - RPRE) AND ALL THREE TERMS ARE LOAD-BEARING:
    //   -16    undoes wq's Q16
    //   +ACC   keeps the accumulator's guard bits, so a rest-spacing term
    //          (tx ~ 11, wq ~ 2621) survives instead of rounding to zero
    //   +RPRE  converts OUT of the >>RPRE unit tx is carried in, so the sum
    //          lands directly in applyDp_at's unit — ticks << ACC — and NOT in
    //          ticks.
    // The RPRE term is the one that was missing and it cost a whole sweep:
    // storing dp in ticks meant applyDp_at's own fp_shr_rne(dp, ACC) divided the
    // entire correction by 256, so a 12x12 sand block still spread 119.09 ->
    // 115.95 wu at mu = 1.0 and friction looked like the wrong IDEA rather than
    // a wrong UNIT. NOTE THE MIRROR CANNOT CATCH THIS CLASS: cpu_mirror.mjs
    // agreed bit-for-bit throughout, because a shared unit error is not an
    // implementation divergence. Only a physical measurement finds it.
    let wq = fp_shr_rne(luts[OFF_W + ui], 4u);             // Q16
    let sx = fp_mul_s(fp_mul_s(tx, wq, 16u - ACC - RPRE), mu, 16u);
    let sy = fp_mul_s(fp_mul_s(ty, wq, 16u - ACC - RPRE), mu, 16u);
    fx = fx - clamp(sx, NEG_FR_TERM, FR_TERM);
    fy = fy - clamp(sy, NEG_FR_TERM, FR_TERM);
  }
  }

  // ── §18 BOND GATHER. Same accumulator, same applicator, no new dispatch. ──
  // Fused into `friction` rather than given its own pass because PLATFORM_NOTES
  // is unambiguous that the substep is DISPATCH-BOUND: one dispatch is worth
  // ~2 % of the substep at 3 600 particles, so a separate bondSolve at one
  // dispatch per solver ITERATION would cost +4 dispatches ~= 8 %.
  //
  // The loop is entered on the TABLE value Mi.bondK, never on the ramped `ki`
  // (§18.3(a)). A bondK == 0 row — WATER, GOO, §17's VOID, 20 of the 26 rows —
  // has an empty bond row, so it costs one load and a `break`.
  let ki = bond_k(Mi, state_in[i].temp);
  var bx : i32 = 0;                        // ticks << ACC, like fx/fy
  var by : i32 = 0;
  if (Mi.bondK != 0 && !rigidI && (flagsI & FLAG_BURST) == 0u) {
    let bb = i * BOND_CAP;
    for (var k : u32 = 0u; k < BOND_CAP; k = k + 1u) {
      let w = bond[bb + k];
      if (w == 0u) { break; }              // rows are DENSE; the tail is zero
      if ((w & BOND_TOMB) != 0u) { continue; }   // broken this substep, inert

      let j  = bond_j(w);
      let Mj = mats[state_in[j].matId];
      if ((state_in[j].flags & FLAG_BURST) != 0u) {
        bond[bb + k] = w | BOND_TOMB; continue;
      }
      // ROW-VALUED VETO, PERMANENT. A melted, transmuted or RETIRED partner
      // (§17's VOID row has bondK == 0) can carry no bond, ever. This one line
      // is §18.7 R1 — without it, three particles retired inside a 14x14 STONE
      // blob leaked 20 221 223 226 ticks of unbalanced impulse over 44 800
      // evaluations and the rows NEVER cleaned up. It costs nothing: Mj is
      // loaded anyway, and the write is SELF-INDEXED (row i).
      //
      // A merely-COLD partner (bondK != 0, kEff == 0) is NOT vetoed here.
      // Vetoing it is what would destroy raw RICE's spawn weld before it could
      // ever be cooked; it keeps its bond, contributes zero force, and is still
      // break-checked below.
      //
      // §18.24 DROPPED THE GROUP HALF OF THIS VETO, and the drop is provably
      // inert on every pre-fusion scene: a bond only ever FORMED between equal
      // groups, and no transmutation on the shipped roster turns a bonded row
      // into a DIFFERENT bonded row (every melt/burn/corrode target has
      // bondK == 0, WATER->ICE starts from an unbondable row), so a live
      // cross-group bond was unreachable and the test was dead code. Fusion
      // makes cross-group bonds REAL — a RUBBER band welded onto a STONE pole
      // — and they must survive this loop. §18.7 R1's actual defect (retired
      // partners leaking impulse) is entirely the bondK == 0 half, which stays.
      if (Mj.bondK == 0) {
        bond[bb + k] = w | BOND_TOMB; continue;
      }

      let L0  = bond_l0(w);
      let L02 = L0 * L0;
      let d   = pred_delta(pi, j);          // the SAME helper every pair loop uses
      let d4x = fp_shr_rne(d.x, BPRE);
      let d4y = fp_shr_rne(d.y, BPRE);

      // ══ THE BOUND. PER-AXIS REJECT BEFORE SQUARING. ══════════════════════
      // Every other pair loop in this shader reaches its squares through
      // lut_index(), which rejects per axis first. The bond row is NOT
      // distance-bounded — a bond survives until the hard break, wherever its
      // partner has got to — so without this, d2 overflows i32 at 22.63 wu on
      // one axis and 16.00 wu on the diagonal, MEASURED on the shipped table
      // with zero edits: 754 overflows, 400 SILENT hard-break failures (the
      // wrapped d2 came out BELOW HARD_BREAK2), 2 forces applied on garbage.
      // The break is evaluated on the number this line bounds, which is why it
      // has to come first.
      // After this line |d4| <= 5120, so d2 <= 52 428 800 and 4*d2 <= 209 715 200
      // — 10.24x headroom on the tightest i32 in the term, and it holds for
      // EVERY input, adversarial or not. abs(d4x) itself cannot overflow:
      // d.x is fp_sub_sat's saturating i32, so |d4x| <= 2^31/32 = 67 108 864.
      // It is ANTISYMMETRIC (fp_shr_rne is sign-symmetric), so it fires at both
      // ends in the same iteration — a one-sided reject would be a one-sided
      // break.
      if (abs(d4x) > L0_MAX || abs(d4y) > L0_MAX) {
        bond[bb + k] = w | BOND_TOMB; continue;
      }
      let d2 = d4x * d4x + d4y * d4y;

      // HARD BREAK AT THE KERNEL RADIUS, UNCONDITIONAL. With bondBreak == 0
      // meaning literally unbreakable, a STONE bar stretched to 11.18 wu of
      // extension with ZERO bonds broken. It is reached on EVERY evaluation
      // because the loop is no longer gated on kEff.
      if (d2 >= HARD_BREAK2) { bond[bb + k] = w | BOND_TOMB; continue; }

      // MOHR-COULOMB. A bare strain limit is COHESION: a fixed shear strength
      // independent of the load above. Adding a term proportional to the local
      // pressure -- which is what lambda IS -- makes the yield stress grow with
      // depth. bondMu == 0 recovers the pure-cohesion bond bit-for-bit.
      var mb  = min(Mi.bondBreak, Mj.bondBreak);
      let mub = min(Mi.bondMu, Mj.bondMu);
      if (mb != 0 && mub != 0) {
        let lp = fp_neg_sat(fp_add_sat(derived[i].lam, derived[j].lam));
        var Pq : i32 = 0;
        if (lp > 0) { Pq = fp_shr_rne(lp, 4u); }              // Q20 -> Q16
        mb = 65536 + (mb - 65536) + fp_mul_s(Pq, mub, 16u);
      }
      if (mb != 0 && d2 > fp_mul_s(L02, mb, 16u)) {
        bond[bb + k] = w | BOND_TOMB; continue;
      }
      if (d2 < D2_FLOOR) { continue; }     // shared with §16; do not redeclare

      // THE FORCE IS GATED SEPARATELY, AFTER EVERY BREAK HAS RUN. min(), not
      // mat_mix(): `min` is the ONLY mixing rule that can express "either end
      // vetoes", and the veto is load-bearing twice over — §18.7 R1 IS the
      // statement "kEff == 0 carries no force", and RICE's whole identity is
      // that a raw grain bonded to a cooked one is not half-set.
      // mat_mix(16384, 0) = 8192 would leave HALF STRENGTH on a partner that is
      // retired, melting or raw. NOT bit-for-bit equivalent: measured across
      // STONE's 700-950 C ramp, 1282 bonds under mat_mix vs 1298 under min,
      // different hashes. Bit-identical isothermally, which is where the
      // "inert" claim was originally measured and the only place it holds.
      let kj = bond_k(Mj, state_in[j].temp);
      let kp = min(ki, kj);
      if (kp == 0) { continue; }

      // C = |d|^2 - L0^2, then q = -C/(4|d|^2). ONE fp_divshift, no sqrt.
      // |q| <= L0^2 / (4*D2_FLOOR) <= 16.0 in Q16 => 1 048 064.
      let C  = fp_sub_sat(d2, L02);
      let q  = fp_divshift(C, d2 * 4, 16u);
      let qk = fp_mul_s(q, kp, 16u);

      // (16 - BPRE - ACC) lands the term directly in applyDp_at's unit —
      // ticks << ACC — exactly as §16.8's (16 - ACC - RPRE) does for friction.
      // GETTING THIS SHIFT WRONG IS THE ONE UNIT BUG THE CPU MIRROR CANNOT
      // CATCH, because a shared unit error is not an implementation divergence.
      // §16 was bitten by exactly this and it cost a whole sweep. The mirror
      // gate therefore carries an assertion on the numeric EFFECT (a bonded
      // block's measured width), not only on the two implementations agreeing.
      let gx = fp_mul_s(d4x, qk, 16u - BPRE - ACC);
      let gy = fp_mul_s(d4y, qk, 16u - BPRE - ACC);
      bx = bx - clamp(gx, NEG_BD_TERM, BD_TERM);
      by = by - clamp(gy, NEG_BD_TERM, BD_TERM);
    }
  }

  // ACC extra bits minus the RPRE the accumulator is carried in.
  // Already in ticks << ACC — the unit applyDp_at reads. It applies the single
  // fp_shr_rne(dp, ACC) and the per-material dpMax clamp, exactly as it does for
  // the solver's own dp, so friction inherits one rounding and one bound rather
  // than inventing a second pair. That single rounding is also the only place
  // the pair antisymmetry is lost: <= 0.5 tick per particle per substep, the
  // same class and bound as XSPH's momentum residue (§11 R11).
  if (rigidI) {
    // ACCUMULATE. The solver's own correction for this member is untouched in
    // derived[i].dp (applyDp_at early-returned), and `rigidSolve` reads the sum
    // of both exactly once per iteration.
    let d0 = derived[i].dp;
    derived[i].dp = vec2<i32>(fp_add_sat(d0.x, fx), fp_add_sat(d0.y, fy));
  } else {
    derived[i].dp = vec2<i32>(fp_add_sat(fx, bx), fp_add_sat(fy, by));
  }
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 46 — applyFric
// The elementwise half, exactly as applyDp is the elementwise half of solveB:
// reads only its own dp, writes only its own pred, so there is no hazard of any
// kind. It reuses applyDp_at unchanged, which means the friction correction is
// bounded by the SAME per-material dpMax the solver is bounded by — one clamp,
// one meaning, and nothing new to prove.
// bindings: 0, 1, 3, 7, 13 [, 15 debug]
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn applyFric(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.nFluid) { return; }
  applyDp_at(i);
}

// ════════════════════════════════════════════════════════════════════════════
// §29 — PRESSURIZED CONTAINMENT (v17)
//
// test/pressure/diagnose.mjs separates three possible leaks. The shipped
// failure is pressure-driven lattice permeation: particles spend several
// substeps inside a two-row wall, with no shell motion, no bond loss and no
// single-step tunnel. A finite repulsion can always be outspent by more
// pressure, so sealed matter is represented geometrically instead: adjacent
// seal-class particles form impermeable line segments. If a non-seal
// particle's final substep motion crosses one, the motion is reverted. The
// segment disappears naturally when a burst moves its endpoints apart.
//
// The pass is after the final constraint/rigid solve and before finalize:
// every correction is already in pred, and the revert becomes ordinary
// zero-velocity state through finalize's existing v = pred - pos rule.
// It writes only the intruder's pred and reads seal particles, so the read and
// write sets are disjoint.
// ════════════════════════════════════════════════════════════════════════════
// Segment geometry is evaluated after shifting Q16.16 positions down by
// BPRE, so one world unit is 2048 here.  Keep these squared thresholds in
// that same coordinate system.
const SEAL_SEG_MAX2 : i32 = 8820900; // round(1.45 * 2048)^2
const SEAL_SEG_MIN2 : i32 = 670761;  // round(0.40 * 2048)^2
const SEAL_KEEP4    : i32 = 717;     // round(0.35 * 2048)
const SEAL_REACH    : i32 = 95027;   // round(1.45 * 65536)

fn seal_class(m : Mat) -> bool {
  return m.bondK != 0 && m.bondReform == 0 &&
         m.solidMode != SOLID_NONE && m.phantom == 0;
}

fn seal_liquid(m : Mat) -> bool {
  return m.bondK == 0 && m.fric == 0 && m.grav > 0 && m.phantom == 0;
}

// Exact sign of a 2-D cross product using the fixed core's signed 64-bit
// product. Only its two's-complement sign and zero-ness are observed.
fn seal_cross_sign(ax : i32, ay : i32, bx : i32, by : i32) -> i32 {
  let d = u64_sub(i64_mul_i32(ax, by), i64_mul_i32(ay, bx));
  if ((d.hi | d.lo) == 0u) { return 0; }
  if ((d.hi >> 31u) != 0u) { return -1; }
  return 1;
}

// A rigid member cannot use contain()'s final per-particle position revert:
// rigidMember would place it back from the body record, and moving one member
// independently would violate rigidity anyway. Instead return the swept
// segment correction here so rigidSolve folds the contact into the body's
// translation/rotation exactly like density and friction corrections.
fn rigid_seal_delta(i : u32, bodyTag : i32) -> vec2<i32> {
  let pos = state_in[i].pos;
  let pr = derived[i].pred;
  let mx4 = fp_shr_rne(fp_sub_sat(pr.x, pos.x), BPRE);
  let my4 = fp_shr_rne(fp_sub_sat(pr.y, pos.y), BPRE);
  let ml2 = mx4 * mx4 + my4 * my4;
  const R4 : i32 = 1126; // 0.55 wu: overlapping capsules close a <=1.10 wu rail
  // A proximity neighbour list cannot prove a swept collision: at VMAX a
  // member can begin outside the 1.45-wu kernel, cross a one-particle rail and
  // finish outside it on the other side. Search the canonical grid cells
  // touched by the segment's expanded AABB instead. Cells are 4 wu and VMAX is
  // 1 wu, so this is normally 1-4 cells; unlike a private neighbour row it
  // cannot drop the rail because fluid filled 48 nearer slots.
  let lo = cell_coord(vec2<i32>(
    fp_sub_sat(min(pos.x, pr.x), SEAL_REACH),
    fp_sub_sat(min(pos.y, pr.y), SEAL_REACH)));
  let hi = cell_coord(vec2<i32>(
    fp_add_sat(max(pos.x, pr.x), SEAL_REACH),
    fp_add_sat(max(pos.y, pr.y), SEAL_REACH)));
  for (var gy = lo.y; gy <= hi.y; gy = gy + 1) {
    for (var gx = lo.x; gx <= hi.x; gx = gx + 1) {
      let c = u32(gy) * P.gridW + u32(gx);
      for (var t = cellStart[c]; t < cellStart[c + 1u]; t = t + 1u) {
        let j = sortedIds[t];
        if (j == i || is_boundary(j)) { continue; }
        let fj = state_in[j].flags;
        if ((fj & (FLAG_ANCHOR | FLAG_SEAL)) != (FLAG_ANCHOR | FLAG_SEAL) ||
            !seal_class(mats[state_in[j].matId])) { continue; }
        if ((fj & FLAG_RIGID) != 0u && i32(weld_tag(fj)) == bodyTag) { continue; }
        let pj = derived[j].pred;
        let px4 = fp_shr_rne(fp_sub_sat(pj.x, pos.x), BPRE);
        let py4 = fp_shr_rne(fp_sub_sat(pj.y, pos.y), BPRE);
        var dx4 = px4;
        var dy4 = py4;
        if (ml2 > 0) {
          let dot = px4 * mx4 + py4 * my4;
          if (dot >= ml2) {
            dx4 = px4 - mx4; dy4 = py4 - my4;
          } else if (dot > 0) {
            let tq = fp_div_s(dot, ml2, 16u);
            dx4 = px4 - fp_mul_s(mx4, tq, 16u);
            dy4 = py4 - fp_mul_s(my4, tq, 16u);
          }
        }
        if (dx4 * dx4 + dy4 * dy4 < R4 * R4) {
          return vec2<i32>(fp_sub_sat(pos.x, pr.x), fp_sub_sat(pos.y, pr.y));
        }
      }
    }
  }
  return vec2<i32>(0, 0);
}

// The normal-density path stays on the compact canonical neighbour list and,
// critically, owns no private 48-word array. Keeping it in its own pipeline
// avoids register/spill cost on every ordinary substep. containOverflow below
// pays for an uncapped cell gather only when buildNbr actually filled.
@compute @workgroup_size(256)
fn contain(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.nFluid) { return; }
  if (!nbr_has_seal(i)) { return; }
  if (nbr_count(i) >= MAXNBR) { return; }
  let Mi = mats[state_in[i].matId];
  if (seal_class(Mi) || Mi.phantom != 0) { return; }
  if ((state_in[i].flags & (FLAG_ANCHOR | FLAG_RIGID)) != 0u) { return; }

  let pos = state_in[i].pos;
  var pr = derived[i].pred;
  let liquidI = seal_liquid(Mi);
  let cnt = nbr_count(i);
  let base = i * MAXNBR;
  var touchedSeal = false;

  for (var k : u32 = 0u; k < cnt; k = k + 1u) {
    let j = nbr[base + k];
    if (is_boundary(j) || !seal_class(mats[state_in[j].matId])) { continue; }
    touchedSeal = true;
    let pj = derived[j].pred;
    let dx4 = fp_shr_rne(fp_sub_sat(pr.x, pj.x), BPRE);
    let dy4 = fp_shr_rne(fp_sub_sat(pr.y, pj.y), BPRE);
    let d2 = dx4 * dx4 + dy4 * dy4;
    if (d2 > 0 && d2 < SEAL_KEEP4 * SEAL_KEEP4) {
      let d4 = i32(isqrt64(I64(0u, bitcast<u32>(d2))));
      if (d4 > 0) {
        // d4 is deliberately sub-unit (< 2^15), outside fp_divshift's solver
        // reciprocal contract. Use the exact signed divide for this rare
        // endpoint case or the reciprocal helper correctly sign-saturates.
        let ux = fp_div_s(dx4, d4, 16u);
        let uy = fp_div_s(dy4, d4, 16u);
        let amt = (SEAL_KEEP4 - d4) << BPRE;
        pr.x = fp_add_sat(pr.x, fp_mul_s(ux, amt, 16u));
        pr.y = fp_add_sat(pr.y, fp_mul_s(uy, amt, 16u));
      }
    }
  }
  if (!touchedSeal) { derived[i].pred = pr; return; }

  for (var k : u32 = 0u; k < cnt; k = k + 1u) {
    let j = nbr[base + k];
    if (is_boundary(j)) { continue; }
    let Mj = mats[state_in[j].matId];
    if (!seal_class(Mj)) { continue; }
    let pj = derived[j].pred;
    for (var q : u32 = k + 1u; q < cnt; q = q + 1u) {
      let k2 = nbr[base + q];
      if (is_boundary(k2)) { continue; }
      let Mk = mats[state_in[k2].matId];
      if (!seal_class(Mk)) { continue; }
      if (liquidI && Mj.sealMode != 0 && Mk.sealMode != 0) { continue; }
      let pk = derived[k2].pred;
      let ex4 = fp_shr_rne(fp_sub_sat(pk.x, pj.x), BPRE);
      let ey4 = fp_shr_rne(fp_sub_sat(pk.y, pj.y), BPRE);
      let e2 = ex4 * ex4 + ey4 * ey4;
      if (e2 > SEAL_SEG_MAX2 || e2 < SEAL_SEG_MIN2) { continue; }
      if (min(pj.x, pk.x) > max(pos.x, pr.x) ||
          max(pj.x, pk.x) < min(pos.x, pr.x) ||
          min(pj.y, pk.y) > max(pos.y, pr.y) ||
          max(pj.y, pk.y) < min(pos.y, pr.y)) { continue; }

      let ex = fp_sub_sat(pk.x, pj.x);
      let ey = fp_sub_sat(pk.y, pj.y);
      let s0 = seal_cross_sign(ex, ey,
        fp_sub_sat(pos.x, pj.x), fp_sub_sat(pos.y, pj.y));
      let s1 = seal_cross_sign(ex, ey,
        fp_sub_sat(pr.x, pj.x), fp_sub_sat(pr.y, pj.y));
      if (s0 == 0 || s0 == s1) { continue; }
      let mx = fp_sub_sat(pr.x, pos.x);
      let my = fp_sub_sat(pr.y, pos.y);
      let t0 = seal_cross_sign(mx, my,
        fp_sub_sat(pj.x, pos.x), fp_sub_sat(pj.y, pos.y));
      let t1 = seal_cross_sign(mx, my,
        fp_sub_sat(pk.x, pos.x), fp_sub_sat(pk.y, pos.y));
      if (t0 == t1 && t0 != 0) { continue; }
      pr = pos;
    }
  }
  derived[i].pred = pr;
}

// Same geometry as contain, but only for an actually full neighbour row. The
// uncapped canonical gather prevents crowded fluid from hiding every wall node.
@compute @workgroup_size(256)
fn containOverflow(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.nFluid) { return; }
  if (!nbr_has_seal(i)) { return; }
  if (nbr_count(i) < MAXNBR) { return; }
  let Mi = mats[state_in[i].matId];
  if (seal_class(Mi) || Mi.phantom != 0) { return; }
  if ((state_in[i].flags & (FLAG_ANCHOR | FLAG_RIGID)) != 0u) { return; }

  let pos = state_in[i].pos;
  var pr = derived[i].pred;
  let liquidI = seal_liquid(Mi);
  var seals : array<u32, 48>;
  var sealN : u32 = 0u;
  let loX = fp_sub_sat(min(pos.x, pr.x), SEAL_REACH);
  let hiX = fp_add_sat(max(pos.x, pr.x), SEAL_REACH);
  let loY = fp_sub_sat(min(pos.y, pr.y), SEAL_REACH);
  let hiY = fp_add_sat(max(pos.y, pr.y), SEAL_REACH);
  let cc = cell_coord(pr);

  // Overflow is rare by definition. Gather only seal endpoints into a private
  // list so the pressure path scans crowded cells once rather than once per
  // segment endpoint. This lives in its own pipeline: ordinary scenes execute
  // contain's compact-list kernel without reserving this array.
  for (var gy = cc.y - 1; gy <= cc.y + 1; gy = gy + 1) {
    if (gy < 0 || gy >= i32(P.gridH)) { continue; }
    for (var gx = cc.x - 1; gx <= cc.x + 1; gx = gx + 1) {
      if (gx < 0 || gx >= i32(P.gridW)) { continue; }
      let c = u32(gy) * P.gridW + u32(gx);
      for (var t = cellStart[c]; t < cellStart[c + 1u]; t = t + 1u) {
        let j = sortedIds[t];
        if (is_boundary(j) || !seal_class(mats[state_in[j].matId])) { continue; }
        let pj = derived[j].pred;
        if (pj.x < loX || pj.x > hiX || pj.y < loY || pj.y > hiY) { continue; }
        if (sealN < 48u) { seals[sealN] = j; sealN = sealN + 1u; }
      }
    }
  }

  for (var k : u32 = 0u; k < sealN; k = k + 1u) {
    let pj = derived[seals[k]].pred;
    let dx4 = fp_shr_rne(fp_sub_sat(pr.x, pj.x), BPRE);
    let dy4 = fp_shr_rne(fp_sub_sat(pr.y, pj.y), BPRE);
    let d2 = dx4 * dx4 + dy4 * dy4;
    if (d2 > 0 && d2 < SEAL_KEEP4 * SEAL_KEEP4) {
      let d4 = i32(isqrt64(I64(0u, bitcast<u32>(d2))));
      if (d4 > 0) {
        let ux = fp_div_s(dx4, d4, 16u);
        let uy = fp_div_s(dy4, d4, 16u);
        let amt = (SEAL_KEEP4 - d4) << BPRE;
        pr.x = fp_add_sat(pr.x, fp_mul_s(ux, amt, 16u));
        pr.y = fp_add_sat(pr.y, fp_mul_s(uy, amt, 16u));
      }
    }
  }

  for (var k : u32 = 0u; k < sealN; k = k + 1u) {
    let j = seals[k];
    let Mj = mats[state_in[j].matId];
    let pj = derived[j].pred;
    for (var q : u32 = k + 1u; q < sealN; q = q + 1u) {
      let k2 = seals[q];
      let Mk = mats[state_in[k2].matId];
      if (liquidI && Mj.sealMode != 0 && Mk.sealMode != 0) { continue; }
      let pk = derived[k2].pred;
      let ex4 = fp_shr_rne(fp_sub_sat(pk.x, pj.x), BPRE);
      let ey4 = fp_shr_rne(fp_sub_sat(pk.y, pj.y), BPRE);
      let e2 = ex4 * ex4 + ey4 * ey4;
      if (e2 > SEAL_SEG_MAX2 || e2 < SEAL_SEG_MIN2) { continue; }
      if (min(pj.x, pk.x) > max(pos.x, pr.x) ||
          max(pj.x, pk.x) < min(pos.x, pr.x) ||
          min(pj.y, pk.y) > max(pos.y, pr.y) ||
          max(pj.y, pk.y) < min(pos.y, pr.y)) { continue; }
      let ex = fp_sub_sat(pk.x, pj.x);
      let ey = fp_sub_sat(pk.y, pj.y);
      let s0 = seal_cross_sign(ex, ey,
        fp_sub_sat(pos.x, pj.x), fp_sub_sat(pos.y, pj.y));
      let s1 = seal_cross_sign(ex, ey,
        fp_sub_sat(pr.x, pj.x), fp_sub_sat(pr.y, pj.y));
      if (s0 == 0 || s0 == s1) { continue; }
      let mx = fp_sub_sat(pr.x, pos.x);
      let my = fp_sub_sat(pr.y, pos.y);
      let t0 = seal_cross_sign(mx, my,
        fp_sub_sat(pj.x, pos.x), fp_sub_sat(pj.y, pos.y));
      let t1 = seal_cross_sign(mx, my,
        fp_sub_sat(pk.x, pos.x), fp_sub_sat(pk.y, pos.y));
      if (t0 == t1 && t0 != 0) { continue; }
      pr = pos;
    }
  }
  derived[i].pred = pr;
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 47 — finalize
// v = pred - pos (one substep IS the time unit, so no divide) ; pos = pred ;
// snapshot v into vtmp for XSPH ; compute rinv.
// Processes ALL n so BOTH ping-pong buffers stay consistent: boundary
// particles get pos copied through unchanged and vel forced to zero.
// bindings: 0, 1, 2, 3 [, 15 debug]
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn finalize(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }

  var outP = state_in[i];
  if (i < P.nFluid) {
    let vmaxN = fp_neg_sat(P.vmax);
    let pr = derived[i].pred;
    var v = vec2<i32>(fp_sub_sat(pr.x, outP.pos.x), fp_sub_sat(pr.y, outP.pos.y));
    v = clamp(v, vec2<i32>(vmaxN, vmaxN), vec2<i32>(P.vmax, P.vmax));
    outP.pos = pr;
    outP.vel = v;
    derived[i].vtmp = v;
    derived[i].rinv = clamp(
        fp_divshift(RHO0, max(derived[i].rho, RHO_FLOOR_RINV), 16u), 0, RINV_MAX);
  } else {
    outP.vel = vec2<i32>(0, 0);
  }
  state_out[i] = outP;
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 22 — xsph  (viscosity)
// v += sum_j c_ij * K_ij * W_ij * (v_j - v_i),  K_ij = 2*RHO0/(rho_i + rho_j).
//
// MONAGHAN's XSPH, not the naive one. The weight MUST be symmetric in (i, j):
// the earlier form used (m/rho_j)·W, i.e. `derived[j].rinv`, which depends on j
// alone, so term_ij = -term_ji held only when rho_i == rho_j. In a free blob the
// measured density spread is 0.58..1.01, so surface pairs were strongly
// asymmetric and XSPH injected net momentum — a gravity-free GOO blob
// self-propelled 52 wu. See §11 R11. K_ij is symmetric by construction, so
// term_ij == -term_ji exactly (fp_mul_s is exactly sign-symmetric when it does
// not saturate, and the per-term clamp is symmetric).
//
// c_ij is the §6.2 arithmetic mean of the two materials' xsph, so it too is
// symmetric; it therefore has to be applied INSIDE the loop, not once to the
// finished sum. ACC = 8 fractional bits absorb the extra rounding.
//
// The only residual momentum leak is the single per-particle fp_shr_rne(ax, ACC)
// at the end (<= 0.5 tick per particle per substep, 1 tick = 1.5e-5 wu). It is
// asserted-bounded, not eliminated; eliminating it would require carrying
// velocity at sub-tick precision. Measured in test/acceptance.html A10.
//
// This is NOT the "convex blend so it can only damp" the old comment claimed.
// K_ij can exceed 1 at a free surface (bounded by RINV_MAX = 4.0), so the blend
// weight sum can exceed 1 in principle; at the shipped c (0.0916 water /
// 0.4578 goo) it does not, but the guarantee is empirical, not structural.
//
// Reads vtmp (read-only this pass) and writes state_out.vel — reading vel[j]
// while writing vel[i] in one dispatch would be the same race applyDp exists
// to avoid.
// Fluid neighbours only: a wall must not drag velocity out of the fluid.
// bindings: 0, 2, 3, 10, 11, 12, 13 [, 15 debug]
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn xsph(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.nFluid) { return; }
  // §20. A member's velocity IS the placement delta `finalize` just wrote —
  // v = pred - pos with pred = c + R(theta)*o. Blending it toward its
  // neighbours' velocities would make the recorded velocity disagree with the
  // motion the body actually performed, and next substep's `finalize` would
  // silently undo the blend anyway. The water still feels the solid, through
  // the ordinary density and friction pair terms.
  if ((state_out[i].flags & FLAG_RIGID) != 0u) { return; }
  // §19.A. `xsph` runs AFTER `finalize`, so it is the one pass that could leave
  // an anchored particle holding a non-zero `vel` it never uses. `vel` is
  // hashed and the renderer reads it, so this guard is what makes "anchored
  // matter is bit-for-bit frozen" true as a stated invariant rather than nearly.
  if ((state_out[i].flags & FLAG_ANCHOR) != 0u) { return; }

  let ci   = mats[state_out[i].matId].xsph;
  let pi   = derived[i].pred;
  let vi   = derived[i].vtmp;
  let rhoi = derived[i].rho;
  let cnt  = nbr_count(i);
  let base = i * MAXNBR;

  var ax : i32 = 0;                            // ticks << ACC
  var ay : i32 = 0;

  for (var k : u32 = 0u; k < cnt; k = k + 1u) {
    let j = nbr[base + k];
    if (is_boundary(j)) { continue; }
    let d  = pred_delta(pi, j);
    let id = lut_index(d);
    if (id < 0) { continue; }
    let ui = u32(id);

    let K   = sym_k(rhoi, derived[j].rho);                         // Q16, symmetric
    let w   = fp_mul_s(luts[OFF_W + ui], K, 16u);                  // 2m/(rho_i+rho_j) W, Q20
    let cij = mat_mix(ci, mats[state_out[j].matId].xsph);          // Q16, symmetric
    let vj  = derived[j].vtmp;
    ax = ax + clamp(fp_mul_s(fp_mul_s(fp_sub_sat(vj.x, vi.x), w, 20u - ACC), cij, 16u),
                    NEG_XS_TERM, XS_TERM);
    ay = ay + clamp(fp_mul_s(fp_mul_s(fp_sub_sat(vj.y, vi.y), w, 20u - ACC), cij, 16u),
                    NEG_XS_TERM, XS_TERM);
  }

  let v = state_out[i].vel;
  state_out[i].vel = vec2<i32>(fp_add_sat(v.x, fp_shr_rne(ax, ACC)),
                               fp_add_sat(v.y, fp_shr_rne(ay, ACC)));
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 23 — normals
// Akinci colour-field normal n_i = h * sum (m/rho_j) gradW, Q16, scaled by h.
// Fluid neighbours only: the colour field is a fluid/not-fluid indicator, and
// counting walls as fluid would erase the free surface next to a wall.
// bindings: 0, 3, 10, 11, 12 [, 15 debug]
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn normals(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.nFluid) { return; }

  let pi   = derived[i].pred;
  let cnt  = nbr_count(i);
  let base = i * MAXNBR;

  var sx : i32 = 0;
  var sy : i32 = 0;

  for (var k : u32 = 0u; k < cnt; k = k + 1u) {
    let j = nbr[base + k];
    if (is_boundary(j)) { continue; }
    let d  = pred_delta(pi, j);
    let id = lut_index(d);
    if (id < 0) { continue; }
    let ui = u32(id);

    let gt = luts[OFF_G + ui];
    let rj = derived[j].rinv;
    sx = sx + fp_mul_s(grad_q16(gt, d.x), rj, 16u);
    sy = sy + fp_mul_s(grad_q16(gt, d.y), rj, 16u);
  }
  // |term| <= ~1.3e6 and MAXNBR = 48 => |sum| <= 6.3e7; * 2.5 => 1.6e8 < 2^31.
  derived[i].nrm = vec2<i32>(fp_mul_s(sx, H, 16u), fp_mul_s(sy, H, 16u));
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 24 — surfaceTension  (Akinci 2013) — the term that makes it look WET
// Cohesion + curvature against fluid neighbours, adhesion against boundary
// neighbours. K_ij = 2*RHO0/(rho_i + rho_j) is the free-surface symmetry
// factor. gammaCur MUST stay ~gammaCoh/4: at equal values a RESTING disc
// expands 5.7x (measured). CUR_SIGN = -1 is Akinci as published; +1 diverges
// above gammaCur = 8 (measured).
// The result is consumed by predict on the NEXT substep. That one-substep lag
// saves an entire density pass and IS PART OF THE BIT-EXACT CONTRACT.
//
// §6.2 CROSS-MATERIAL MIXING: gammaCoh and gammaCur are mixed pairwise with
// mat_mix so a WATER-GOO pair computes the SAME coefficient at both ends and
// f_ij == -f_ji survives. Without it an isolated mixed pair gained net momentum
// out of nothing (measured: SUM=(54,0) ticks after 6 substeps on a 2-particle
// scene; the sign flipped when the two materials were swapped). Because
// (a + a) >> 1 == a, this is bit-for-bit inert on single-material scenes.
//
// `adhesion` is DELIBERATELY NOT mixed. It is a fluid-to-wall property and the
// wall is immovable, so there is no third-law pair to conserve; boundary
// particles also carry matId = WATER purely as a placeholder, so mixing would
// silently halve GOO's wall wetting against a wall that has no material.
// bindings: 0, 2, 3, 10, 11, 12, 13 [, 15 debug]
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn surfaceTension(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.nFluid) { return; }

  let M    = mats[state_out[i].matId];
  let pi   = derived[i].pred;
  let ni   = derived[i].nrm;
  let rhoi = derived[i].rho;
  let cnt  = nbr_count(i);
  let base = i * MAXNBR;
  // §19.B — THE THIRD INTERNAL FORCE, AND THE ONE §20 MISSED.
  //
  // §20 already skips same-body neighbours in `solveB_at` and in `friction`, and
  // states the rule those two obey: the internal DENSITY force is central, so
  // cross(r_i, f) + cross(r_j, -f) = cross(r_i - r_j, f) = 0 and it cannot torque
  // the body; internal FRICTION is tangential and does not cancel. Surface
  // tension is BOTH at once and only half of it was safe. The Akinci COHESION
  // term is central (it is a multiple of `d`) and cancels. THE CURVATURE TERM IS
  // NOT — it is a multiple of (ni - nj), which points nowhere in particular —
  // and between two members of one body it is a PHANTOM DRIVE TORQUE.
  //
  // MEASURED, and this is how it was found. A dry 6-blade METAL wheel in vacuum,
  // pinned on an axis, 10 000 substeps with nothing whatever to push it:
  //   shipped         theta = -1 943 239 537  (0.45 of a turn, and accelerating)
  //   gammaCoh = 0    theta = -1 858 ... unchanged in magnitude   <- not this one
  //   gammaCur = 0    theta = -936                                <- this one
  //   both     = 0    theta = 0 exactly
  // A 4- or 8-blade wheel reads exactly 0 either way, because the union of its
  // blades has the square lattice's 90-degree symmetry and the phantom torques
  // cancel by shape. THAT is why §20 shipped green: its symmetric-body gate used
  // a block and a ring. Same class as the §18 gate measuring a block while the
  // user draws a bar.
  //
  // The fix is the rule §20 already wrote down, applied to the term it missed.
  // It changes NOTHING for the fluid: this solver is gather-only and
  // self-indexed, so skipping same-body neighbours in i's own gather changes only
  // derived[i].stAcc; the water neighbour j computes its own stAcc from the
  // member's placed position and is bit-for-bit untouched. It costs no binding —
  // this pass already reads state_out[j].matId — and it does move the chain of
  // any scene containing a rigid body, which is deliberate and announced.
  let stRigidI = (state_out[i].flags & FLAG_RIGID) != 0u;

  var ax : i32 = 0;                            // ticks/substep^2 << ACC
  var ay : i32 = 0;

  for (var k : u32 = 0u; k < cnt; k = k + 1u) {
    let j  = nbr[base + k];
    if (stRigidI && !is_boundary(j) && rg_same_body(state_out[i].flags, state_out[j].flags)) { continue; }
    let d  = pred_delta(pi, j);
    let id = lut_index(d);
    if (id < 0) { continue; }
    let ui = u32(id);

    if (is_boundary(j)) {
      // Wall wetting. adhesion == 0 (WATER) makes both terms exactly zero, so
      // this needs no branch of its own.
      let at = luts[OFF_A + ui];
      ax = ax - clamp(fp_mul_s(fp_mul_s(at, d.x, CF), M.adhesion, 16u - ACC),
                      NEG_ST_TERM, ST_TERM);
      ay = ay - clamp(fp_mul_s(fp_mul_s(at, d.y, CF), M.adhesion, 16u - ACC),
                      NEG_ST_TERM, ST_TERM);
      continue;
    }

    let K   = sym_k(rhoi, derived[j].rho);                        // Q16, symmetric
    let ct  = luts[OFF_C + ui];
    let Mj  = mats[state_out[j].matId];
    let gCoh = mat_mix(M.gammaCoh, Mj.gammaCoh);                  // §6.2, symmetric
    let gCur = mat_mix(M.gammaCur, Mj.gammaCur);                  // §6.2, symmetric

    var fx = fp_neg_sat(fp_mul_s(fp_mul_s(ct, d.x, CF), gCoh, 16u));
    var fy = fp_neg_sat(fp_mul_s(fp_mul_s(ct, d.y, CF), gCoh, 16u));

    let nj  = derived[j].nrm;
    let cx  = fp_mul_s(fp_sub_sat(ni.x, nj.x), gCur, 16u);
    let cy  = fp_mul_s(fp_sub_sat(ni.y, nj.y), gCur, 16u);
    fx = fp_add_sat(fx, select(cx, fp_neg_sat(cx), CUR_SIGN < 0));
    fy = fp_add_sat(fy, select(cy, fp_neg_sat(cy), CUR_SIGN < 0));

    ax = ax + clamp(fp_mul_s(fx, K, 16u - ACC), NEG_ST_TERM, ST_TERM);
    ay = ay + clamp(fp_mul_s(fy, K, 16u - ACC), NEG_ST_TERM, ST_TERM);
  }

  derived[i].stAcc = vec2<i32>(fp_shr_rne(ax, ACC), fp_shr_rne(ay, ACC));
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 49 — conduct  (§12.1 heat transfer)
//
//   q_i = sum_j  clamp( c_ij * (W_ij / RHO0) * clamp(T_j - T_i) )   ticks << ACC
//
// WHY THIS IS ORDER-INDEPENDENT, which is the only property that matters here:
// it is a per-particle GATHER whose accumulator is a per-term-clamped plain i32
// sum — exactly the shape every other pair loop in this file already has, and
// exactly what test/determinism.html stage C2 exists to protect. There is no
// running-total clamp, no min/max reduction, no early break, no atomic.
//
// WHY TOTAL HEAT IS CONSERVED EXACTLY:
//   W_ij         symmetric  (lut_index rounds |d| symmetrically — load-bearing)
//   c_ij         symmetric  (mat_mix is the arithmetic mean, §6.2)
//   T_j - T_i    ANTIsymmetric, and fp_sub_sat cannot saturate here because
//                |T| <= 2^29 so |dT| < 2^30
//   clamp(x, -k, +k)  symmetric about zero, so clamping the DIFFERENCE preserves
//                antisymmetry — it is a saturating flux, not a leak
//   fp_mul_s     exactly sign-symmetric when it does not saturate, and the
//                magnitudes above prove it cannot
// therefore term_ij == -term_ji EXACTLY and sum_i q_i == 0 EXACTLY.
// The ONLY loss is the single fp_shr_rne(q, ACC) per particle in `thermal`
// (<= 0.5 tick = 7.6e-6 deg per particle per substep). Same class and same
// bound as XSPH's momentum residue (§11 R11); measured in test/thermal.html T2.
//
// FLUID ONLY, both ends. Walls are perfect insulators in v4: a boundary
// particle's temp is never written by anything, so letting fluid exchange heat
// with one would be a free source/sink and would destroy the conservation proof
// above. A "hot wall" is a fluid material with coolRate = 0 and a high
// spawnTemp, not a boundary particle.
// bindings: 0, 2, 3, 10, 11, 12, 13 [, 15 debug]
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn conduct(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.nFluid) { return; }

  let mi   = state_out[i].matId;
  let Mi   = mats[mi];
  let ki   = Mi.cond;
  let Ti   = state_out[i].temp;
  let pi   = derived[i].pred;
  let cnt  = nbr_count(i);
  let base = i * MAXNBR;

  var q : i32 = 0;                             // temp ticks << ACC
  var c : i32 = 0;                             // §13 corrosion dose << ACC

  for (var k : u32 = 0u; k < cnt; k = k + 1u) {
    let j = nbr[base + k];
    if (is_boundary(j)) { continue; }
    let d  = pred_delta(pi, j);
    let id = lut_index(d);
    if (id < 0) { continue; }
    let ui = u32(id);
    let W  = luts[OFF_W + ui];
    let mj = state_out[j].matId;
    let Mj = mats[mj];

    let cij = mat_mix(ki, Mj.cond);                              // Q16, symmetric
    let wk  = fp_mul_s(W, cij, 16u);                             // W * c, RHO0-scaled
    let dT  = clamp(fp_sub_sat(state_out[j].temp, Ti), NEG_DT_CLAMP, DT_CLAMP);
    q = q + clamp(fp_mul_s(dT, wk, 20u - ACC), NEG_HT_TERM, HT_TERM);

    // ---- §13 corrosion, same gather, no extra pass and no extra binding -----
    // Two terms, and a material is only ever eligible for one of them (engine.js
    // asserts an attacker is itself immune, so `corrode > 0` and `corrPt` being
    // reachable are mutually exclusive):
    //   ATTACK  j is corrosive and i is a different material -> i takes a dose.
    //   SPEND   i is corrosive and j is something i can actually eat -> i takes
    //           a dose of its OWN strength. This is what consumes the acid: it
    //           is spent in proportion to how much dissolvable matter it touches,
    //           so a puddle on stone runs out and a puddle on water does not.
    // Same-material pairs are skipped so acid does not eat acid.
    if (mj != mi) {
      let attack = Mj.corrode;
      let spend  = select(0, Mi.corrode, Mj.corrPt != CORR_NEVER);
      let str    = attack + spend;             // both >= 0, both <= CORR_MAX
      if (str != 0) {
        c = c + clamp(fp_mul_s(W, str, 20u - ACC), 0, CORR_TERM);
      }
    }
  }

  derived[i].heat = q;
  derived[i].corr = c;
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 50 — thermal  (§12.2 state change and combustion)
//
// Purely elementwise: reads and writes ONLY its own particle plus the frozen
// mats table. No neighbour is touched, so there is no hazard of any kind and
// nothing here can depend on execution order.
//
// It runs LAST in the substep, after every mechanical pass, so a material
// change takes effect on the NEXT substep — the same one-substep lag `stAcc`
// already has (§5.2), and for the same reason: every pass in this substep
// already read the OLD matId, and rewriting it mid-substep would make the pair
// forces asymmetric for one step.
//
// THE TRANSITION ORDER IS WIRE FORMAT (§11 R7). It is a fixed if/else chain, so
// exactly one transition can fire per substep:
//     1 burnout   (burning && fuel exhausted)   -> burnTo
//     2 corrode   (dose >= corrPt)              -> corrTo      (v5, §13)
//     3 boil      (T >= boilPt)                 -> boilTo
//     4 melt      (T >= meltPt)                 -> meltTo
//     5 condense  (T <= condPt)                 -> condTo
//     6 freeze    (T <= freezePt)               -> freezeTo
// Corrosion sits directly under burnout because it is a destructive CHEMICAL
// event: a stone being eaten by acid should not spend that substep deciding
// whether it is warm enough to melt.
// A material that lacks a transition disables it with the out-of-reach sentinel
// (see struct Mat), so the branch costs a compare and nothing else.
// bindings: 0, 2, 3, 13 [, 15 debug]
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn thermal(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.nFluid) { return; }

  let mat0 = state_out[i].matId;
  let M    = mats[mat0];

  // ---- 1. conduction, then ambient relaxation ------------------------------
  // The ambient term is a per-particle relaxation toward the room, NOT a pair
  // term, so it deliberately does NOT conserve heat — it is the box losing
  // energy to the world. coolRate = 0 makes it exactly zero, which is what the
  // conservation test in test/thermal.html uses.
  var T = fp_add_sat(state_out[i].temp, fp_shr_rne(derived[i].heat, ACC));
  T = fp_add_sat(T, fp_mul_s(fp_sub_sat(AMBIENT, T), M.coolRate, 16u));

  // ---- 2. combustion -------------------------------------------------------
  var fuel    = state_out[i].fuel;
  var burning = (state_out[i].flags & FLAG_BURNING) != 0u;
  if (!burning && fuel > 0 && T >= M.ignitePt) { burning = true; }
  if (burning) {
    T    = fp_add_sat(T, M.burnHeat);
    fuel = max(0, fp_sub_sat(fuel, M.burnRate));
  }
  T = clamp(T, TEMP_MIN, TEMP_MAX);

  // ---- 2.5. the §25 fixed-target drive (v16) -------------------------------
  // coolRate's exact shape pointed at a per-row target: a heater/cooler row
  // pulls its own T toward tempTarget every substep, forever. tempDrive = 0
  // (every authored row) makes this bit-for-bit inert. The result stays inside
  // [TEMP_MIN, TEMP_MAX] because T is in range after the clamp above, the
  // validator pins tempTarget inside the range whenever tempDrive != 0, and a
  // Q16 rate <= ONE moves T toward the target by at most the full gap.
  T = fp_add_sat(T, fp_mul_s(fp_sub_sat(M.tempTarget, T), M.tempDrive, 16u));

  // ---- 3. transitions ------------------------------------------------------
  // `dose` is the same shape as the conduction term: an accumulator in << ACC,
  // rounded once. With every material's `corrode` at 0 the dose is exactly 0 and
  // corrPt is the CORR_NEVER sentinel, so this branch is bit-for-bit inert — the
  // v4 chain is reproduced unchanged, which is asserted, not assumed.
  let dose = fp_shr_rne(derived[i].corr, ACC);
  var mat = mat0;
  var froze = false;
  if (burning && fuel <= 0) {
    mat = pick_product(M.burnTo,   M.splitTo, M.splitPct, i);
  } else if (dose >= M.corrPt) {
    mat = pick_product(M.corrTo,   M.splitTo, M.splitPct, i);
  } else if (T >= M.boilPt) {
    mat = pick_product(M.boilTo,   M.splitTo, M.splitPct, i);
  } else if (T >= M.meltPt) {
    mat = pick_product(M.meltTo,   M.splitTo, M.splitPct, i);
  } else if (T <= M.condPt) {
    mat = pick_product(M.condTo,   M.splitTo, M.splitPct, i);
  } else if (T <= M.freezePt) {
    mat = pick_product(M.freezeTo, M.splitTo, M.splitPct, i);
    froze = true;
  }

  if (mat != mat0) {
    // A fresh particle of the product material: it carries the product's fuel
    // load and is not on fire until it re-ignites on its own merits.
    fuel    = mats[mat].fuel0;
    burning = false;
  }

  var fl = state_out[i].flags & (~(FLAG_BURNING | FLAG_DEAD | FLAG_SEAL));
  // §19.A. ANCHORED MATTER THAT TRANSMUTES STOPS BEING ANCHORED. You anchored
  // an ICE shelf, not "whatever this slot becomes": an immovable puddle, an
  // immovable flame or an immovable pool of lava is absurd, and the alternative
  // rule — refuse the anchor toggle on materials with a reachable transition —
  // needs a typed list of which transitions are "reachable" and would forbid
  // anchoring STONE. This is the table-derived version: the row changed, so the
  // matter you placed is gone. It is the ONLY place any shader writes this bit.
  if (mat != mat0) { fl = fl & (~(FLAG_ANCHOR | FLAG_CONGEAL | FLAG_BURST)); }
  // Solidification is the spawn event a cooling pool never had. Mark only the
  // actual freeze transition and only when its product is a bonded solid row.
  // Unlike the abandoned v17 attempt, formation scans ONLY marked particles
  // and bonds marked-to-marked: ordinary stone/metal rows retain their old
  // O(1) idle path, avoiding the catastrophic all-solid neighbour scan.
  if (froze && mat != mat0 && fuse_row(mats[mat], mat)) {
    fl = fl | FLAG_CONGEAL;
  }
  if (burning) { fl = fl | FLAG_BURNING; }
  // §17. FLAG_DEAD is a PURE FUNCTION of the (possibly just-changed) matId, read
  // out of the table — no material id is hard-coded here, and nothing else in
  // this file mentions retirement. Recomputing it every substep (rather than
  // latching it) is what makes it impossible to leave a stale corpse flag on a
  // slot that has been reused for live matter.
  if (mats[mat].phantom != 0) { fl = fl | FLAG_DEAD; }
  if (seal_class(mats[mat])) { fl = fl | FLAG_SEAL; }

  state_out[i].matId = mat;
  state_out[i].flags = fl;
  state_out[i].temp  = T;
  state_out[i].fuel  = fuel;
}

// Solidification happens after buildNbr, but mutual bond formation needs an
// already-solid neighbour to know that a newly frozen partner is beside it.
// Refresh one transient metadata bit from the post-thermal flags. Each
// invocation writes only its own nbrN word and reads the immutable neighbour
// row, so this is the same race-free gather shape as every other pair pass.
@compute @workgroup_size(256)
fn markCongealNbr(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.nFluid) { return; }
  let cnt = nbr_count(i);
  let base = i * MAXNBR;
  var seen = false;
  for (var k : u32 = 0u; k < cnt; k = k + 1u) {
    let j = nbr[base + k];
    if (j < P.nFluid && (state_out[j].flags & FLAG_CONGEAL) != 0u) {
      seen = true;
      break;
    }
  }
  nbrN[i] = (nbrN[i] & ~NBR_CONGEAL_BIT) |
    select(0u, NBR_CONGEAL_BIT, seen);
}

// ════════════════════════════════════════════════════════════════════════════
// §29 — VISIBLE BURST. A held seal particle sums nearby intrusion. Past its
// row threshold it loses the constraint that held it; rigidMember immediately
// recounts a rigid body without it, while an anchored particle becomes free.
// The geometric seal persists only while adjacent endpoints remain close, so
// the freed particle must visibly move before matter can escape through the
// resulting gap. burstP == 0 means absolute containment (BEDROCK).
// ════════════════════════════════════════════════════════════════════════════
const SEAL_STRESS4 : i32 = 2560;     // 1.25 wu >> BPRE

@compute @workgroup_size(256)
fn containStress(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.nFluid) { return; }
  let Mi = mats[state_out[i].matId];
  if (!seal_class(Mi) || Mi.burstP <= 0) { return; }
  let fl = state_out[i].flags;
  if ((fl & (FLAG_ANCHOR | FLAG_RIGID)) == 0u) { return; }

  let pi = state_out[i].pos;
  let cnt = nbr_count(i);
  let base = i * MAXNBR;
  var stress : i32 = 0;
  for (var k : u32 = 0u; k < cnt; k = k + 1u) {
    let j = nbr[base + k];
    if (is_boundary(j)) { continue; }
    let Mj = mats[state_out[j].matId];
    if (seal_class(Mj) || Mj.phantom != 0) { continue; }
    let dx4 = fp_shr_rne(fp_sub_sat(pi.x, state_out[j].pos.x), BPRE);
    let dy4 = fp_shr_rne(fp_sub_sat(pi.y, state_out[j].pos.y), BPRE);
    let d2 = dx4 * dx4 + dy4 * dy4;
    if (d2 >= SEAL_STRESS4 * SEAL_STRESS4) { continue; }
    let d4 = i32(isqrt64(I64(0u, bitcast<u32>(d2))));
    stress = stress + (SEAL_STRESS4 - d4);
  }
  if ((stress << BPRE) > Mi.burstP) {
    state_out[i].flags =
      (fl & (~(FLAG_ANCHOR | FLAG_RIGID | WELD_TAG_MASK))) | FLAG_BURST;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 59 — bondForm1  (§18)  PROPOSE
// bindings: 0, 2, 3, 10, 11, 13, 15, 16, 17
//
// Runs at the TAIL of the substep, after `thermal`, and that is LOAD-BEARING:
// after `finalize`, state_out[i].pos == derived[i].pred and `nbr` was built
// from `pred`, so the neighbour list and the positions this pass reads are the
// SAME positions; and a particle that melted or transmuted this substep already
// carries its new matId.
//
// Two kinds of proposal, and the difference is the whole of §18.2:
//   WELD   — same non-zero weldTag. This is the SPAWN EVENT. It ignores
//            bondReform and both contact gates, and it is the ONLY way a
//            bondReform == 0 solid (STONE, METAL, WOOD, ICE) ever bonds.
//            Gating formation on bondReform, as the previous draft did, gives
//            those four materials ZERO bonds — measured, and byte-identical to
//            the bonds-off build.
//   REFORM — bondReform != 0 and we are stiff: settled-contact bonding, which
//            is the mechanism for SAND and cooked RICE, where pouring IS the
//            gesture.
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn bondForm1(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x; if (i >= P.nFluid) { return; }
  let cb = i * BOND_CAP;

  // §20. A MEMBER OF A RIGID BODY PROPOSES NOTHING AND KEEPS NOTHING. Emptying
  // its candidate row is not merely tidy: it is how the OTHER half of a bond to
  // a non-member gets dropped, because bondForm2 keeps bond[j][k] only if
  // bondCand[i] still names j. So a stroke that §18-welded to its neighbours
  // during the gesture is cleanly released the pass after it is promoted, with
  // no one-sided bonds and no host readback. Costs no new binding: `flags` is a
  // word of Particle and this pass already binds state_out — which matters,
  // because `bondForm1` is one of the two pipelines AT the 8-buffer device cap.
  //
  // §19.A used to add ANCHORED matter to the same early-out — an
  // anchored<->anchored bond can never strain and §18 bonds only ever formed
  // inside ONE spawn event, so an anchored row was pure waste of a gather slot.
  // §18.24 AMENDS THAT: an anchored pole is exactly what a slingshot's band
  // fuses TO, and a fused bond lives in BOTH rows (bondForm2 keeps bond[i][k]
  // only if bondCand[j] still names i), so an anchored particle now (1) carries
  // its existing bonds forward and (2) proposes FUSION — and nothing else: no
  // weld, no reform, so an anchored gesture still never self-welds and
  // anchored<->anchored stays impossible (the fusion rule below requires a free
  // end). At fuseTag == 0 an anchored row has no bonds to carry and proposes
  // nothing — bit-for-bit the old empty row.
  if ((state_out[i].flags & FLAG_RIGID) != 0u) {
    for (var k : u32 = 0u; k < BOND_CAP; k = k + 1u) { bondCand[cb + k] = 0u; }
    return;
  }
  let anchoredI = (state_out[i].flags & FLAG_ANCHOR) != 0u;

  let Mi = mats[state_out[i].matId];
  // A bondK == 0 row (WATER, GOO, ... and §17's VOID) proposes NOTHING and its
  // candidate list stays empty. That is what reaps every bond naming a retired
  // particle in phase 2, with no FLAG_DEAD test anywhere in the bond code.
  var cn : u32 = 0u;
  if (Mi.bondK != 0 && Mi.bondGroup >= 0) {
    // 1. carry existing bonds forward, COMPACTING: tombstones are dropped here
    //    and nowhere else, which is what keeps a row dense for the solve loop.
    for (var k : u32 = 0u; k < BOND_CAP; k = k + 1u) {
      let w = bond[cb + k];
      if (w == 0u) { break; }
      if ((w & BOND_TOMB) != 0u) { continue; }
      // GUARDED, and the guard is NOT redundant. The premise "cn <= nbrN[i] <=
      // MAXNBR == BOND_CAP" is FALSE: carried bonds are not a subset of nbr[i],
      // because the neighbour list is rebuilt every substep from `pred` while a
      // bond survives until the hard break. Measured on the REST lattice — the
      // quietest scene in the suite — 18 carried bonds named a partner that was
      // not in this substep's nbr[i]. The true bound is
      // BOND_CAP + MAXNBR = 96, so an unguarded write reaches index 95 in a row
      // of 48 and lands in the NEXT PARTICLE'S ROW, silently.
      if (cn >= BOND_CAP) {
        //@DBG_BEGIN
        fp_flag(8u);
        //@DBG_END
        break;
      }
      bondCand[cb + cn] = w;
      cn = cn + 1u;
    }
    // 2. propose new partners.
    let tagI      = weld_tag(state_out[i].flags);
    let ki        = bond_k(Mi, state_out[i].temp);
    let canReform = (!anchoredI && Mi.bondReform != 0 && ki != 0);
    // §18.24. Fusion is live for this row exactly while a gesture is armed.
    let fuseArmed = (P.fuseTag != 0u && fuse_row(Mi, state_out[i].matId));
    let congealI = ((state_out[i].flags & FLAG_CONGEAL) != 0u &&
                    fuse_row(Mi, state_out[i].matId));
    // A settled Stone/Metal neighbour normally takes the O(1) idle path.
    // buildNbr's transient bit wakes it for exactly the one formation pass in
    // which adjacent Lava/Molten has frozen, allowing the proposal to be
    // mutual without making every permanent solid scan forever.
    let congealNear = (nbr_has_congeal(i) && fuse_row(Mi, state_out[i].matId));
    if (canReform || (!anchoredI && tagI != 0u) || fuseArmed ||
        congealI || congealNear) {
      let cnt = nbr_count(i);
      let nb  = i * MAXNBR;
      for (var k : u32 = 0u; k < cnt; k = k + 1u) {
        if (cn >= BOND_CAP) {
          //@DBG_BEGIN
          fp_flag(8u);
          //@DBG_END
          break;
        }
        let j = nbr[nb + k];
        if (j >= P.nFluid) { continue; }               // no bonds to boundary
        let Mj = mats[state_out[j].matId];
        let fJ   = state_out[j].flags;
        let tagJ = weld_tag(fJ);
        // WELD — the spawn event (§18.2/§18.22): same non-zero tag. Never for
        // an anchored end: §19.A's rule that an anchored gesture makes scenery,
        // not a body, is unchanged — before §18.24 an anchored row simply never
        // reached this line.
        let weld = (!anchoredI && tagI != 0u && tagI == tagJ);
        // §18.24/§31 FUSION — DRAW-TIME OR IMPORT CONTACT, the one
        // table-scoped exception to
        // §18.22's cross-event prohibition. Exactly ONE end is the armed
        // gesture and the other is EXISTING settled matter (tag 0 — so an
        // emitter tick's own group, and any second concurrent event, never
        // fuses); both rows pass fuse_row (bonded families that hold shape,
        // never PHOTO); the existing end is not a §20 body member (bonds into
        // a body's force/torque path are Phase B); and at least one end is
        // FREE, because an anchored<->anchored bond could never strain. The
        // whole classification sits behind fuseArmed, so at fuseTag == 0 this
        // is bit-for-bit the pre-§18.24 pass.
        var fuse = false;
        if (fuseArmed && (fJ & FLAG_RIGID) == 0u
            && fuse_row(Mj, state_out[j].matId)
            && (!anchoredI || (fJ & FLAG_ANCHOR) == 0u)) {
          fuse = ((tagI == P.fuseTag && tagJ == 0u) ||
                  (tagJ == P.fuseTag && tagI == 0u) ||
                  // An OE-CAKE import is one armed event. Touching bonded
                  // families in that save may cross material/group boundaries,
                  // including free Elastic against anchored Wall.
                  (tagI == P.fuseTag && tagJ == P.fuseTag));
        }
        // §18.25 CONGEAL — at least one particle froze since the previous
        // formation pass and both are now the same bonded product. This bonds
        // a cooling pool to itself even when its cells cross the threshold on
        // different substeps, and welds Lava→Stone / Molten→Metal onto an
        // existing non-rigid or anchored matching solid. A §20 rigid member is
        // still excluded: joining a live body requires body-membership fusion,
        // not a one-sided distance bond.
        let congealJ = ((fJ & FLAG_CONGEAL) != 0u &&
                        fuse_row(Mj, state_out[j].matId));
        let congeal = ((congealI || congealJ) && !weld && !fuse &&
                       state_out[j].matId == state_out[i].matId &&
                       (fJ & FLAG_RIGID) == 0u);
        // The group law is UNTOUCHED for weld and reform. A FUSED pair is the
        // one legal cross-group — and therefore cross-material — bond, which
        // is what makes §18.4 F1's forward constraint fire: the gather's
        // min() coefficient mixing is exercised across rows for the first time.
        if (!fuse && !congeal && Mj.bondGroup != Mi.bondGroup) { continue; }

        if (!weld && !fuse && !congeal) {
          if (!canReform) { continue; }
          if (bond_k(Mj, state_out[j].temp) == 0) { continue; }
        }

        var dup : bool = false;
        for (var q : u32 = 0u; q < cn; q = q + 1u) {
          if (bond_j(bondCand[cb + q]) == j) { dup = true; break; }
        }
        if (dup) { continue; }

        let d   = vec2<i32>(fp_sub_sat(state_out[i].pos.x, state_out[j].pos.x),
                            fp_sub_sat(state_out[i].pos.y, state_out[j].pos.y));
        let d4x = fp_shr_rne(d.x, BPRE);
        let d4y = fp_shr_rne(d.y, BPRE);
        // the same per-axis reject as the gather. Here it is cheap insurance:
        // BOND_R2 would reject anyway, but only AFTER the square.
        if (abs(d4x) > L0_MAX || abs(d4y) > L0_MAX) { continue; }
        let d2  = d4x * d4x + d4y * d4y;
        if (d2 < D2_FLOOR || d2 > BOND_R2) { continue; }

        // §18.24: fusion is WELD-CLASS — draw-time contact, not settled
        // contact — so it skips both gates exactly as the spawn weld does.
        if (!weld && !fuse && !congeal) {
          // STATIC-CONTACT GATE: a grain still sliding is not in static contact.
          // THE AXIS REJECT IS LOAD-BEARING, NOT AN OPTIMISATION. |rel| can reach
          // 2^19 (vmax on both ends plus iters*dpMax), so r4 reaches 65536 and
          // r4x*r4x + r4y*r4y would be 4.3e9 — PAST 2^31. Rejecting per axis
          // first bounds |r4| by vl4 (225 for SAND, 325 for RICE), so the sum of
          // squares is at most 2*325^2 = 211250.
          let vl  = min(Mi.bondFormV, Mj.bondFormV);
          let vl4 = fp_shr_rne(vl, RPRE);
          let r4x = fp_shr_rne(fp_sub_sat(state_out[i].vel.x, state_out[j].vel.x), RPRE);
          let r4y = fp_shr_rne(fp_sub_sat(state_out[i].vel.y, state_out[j].vel.y), RPRE);
          if (abs(r4x) > vl4 || abs(r4y) > vl4) { continue; }
          if (r4x * r4x + r4y * r4y > vl4 * vl4) { continue; }

          // COMPRESSION GATE: grains stick when PRESSED, not when pulled.
          // lambda <= 0 always; lambda == 0 means "in tension or at a free
          // surface", so this is one compare on a value the solver already has.
          let pl = min(Mi.bondFormP, Mj.bondFormP);
          if (pl != 0 && (fp_neg_sat(derived[i].lam) < pl ||
                          fp_neg_sat(derived[j].lam) < pl)) { continue; }
        }

        // THE ONLY sqrt IN ALL OF §18, AND IT IS ALREADY IN THE REPO.
        // fixed.wgsl's `isqrt64` is the validated exact digit-recurrence
        // (32 iterations, floor) and — verified by tracing the call graph — it
        // does NOT touch fp_dbg, unlike every fp_* wrapper. d2 >= D2_FLOOR > 0
        // here, so the u32 cast is safe. fp_sqrt_s / fp_sqrt16 / fp_rsqrt still
        // appear NOWHERE, which is what keeps §18 off the cross-vendor risk.
        // FLOOR, not isqrt64_round: worst relative bias 0.1838 % at |d| = 0.256
        // wu and EXACTLY 0 at every lattice spacing a scene actually forms at.
        let L0 = i32(isqrt64(I64(0u, u32(d2))));
        if (L0 < L0_MIN || L0 > L0_MAX) { continue; }
        bondCand[cb + cn] = bond_pack(j, L0);
        cn = cn + 1u;
      }
    }
  }
  for (var k : u32 = cn; k < BOND_CAP; k = k + 1u) { bondCand[cb + k] = 0u; }
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 60 — bondForm2  (§18)  MUTUALISE, REAP, AND CLEAR THE WELD TAG
// bindings: 0, 2, 16, 17
//
// Reads bondCand[j] — another invocation's row — while writing bond[i]. That is
// §5.1-legal precisely BECAUSE THEY ARE DIFFERENT BUFFERS: no invocation reads a
// value another invocation wrote in this dispatch. Same argument as
// solveB/applyDp, and it is why the ping-pong is mandatory rather than
// stylistic.
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn bondForm2(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x; if (i >= P.nFluid) { return; }
  // NO material early-out. This pass runs for EVERY fluid particle, which is
  // what zero-fills a retired particle's row (§18.7 R2) and what clears the
  // weld tag (§18.2 W3).
  let cb = i * BOND_CAP;
  var n : u32 = 0u;
  for (var k : u32 = 0u; k < BOND_CAP; k = k + 1u) {
    let w = bondCand[cb + k];
    if (w == 0u) { break; }
    let j  = bond_j(w);
    let jb = j * BOND_CAP;
    // A proposal survives ONLY if the partner proposed it too, WITH THE SAME L0.
    // Both ends computed L0 = isqrt(d2) from the identical d2, so agreement is
    // exact. This is what makes the graph symmetric with or without a cap, and
    // it is also what drops a bond whose partner did not see us because the
    // neighbour list truncated (dbg[6] > 0) — so §5.4's "invalid run" is a
    // precondition for NEWTON'S THIRD LAW here, not merely for accumulator
    // bounds. Phase 2 converts it from a wrong force into a MISSING bond: the
    // right failure mode, still a failure mode.
    var ok : bool = false;
    for (var q : u32 = 0u; q < BOND_CAP; q = q + 1u) {
      let v = bondCand[jb + q];
      if (v == 0u) { break; }
      if (bond_j(v) == i) { ok = ((v & BOND_L0_MASK) == (w & BOND_L0_MASK)); break; }
    }
    if (ok) { bond[cb + n] = w & ~BOND_TOMB; n = n + 1u; }
  }
  for (var k : u32 = n; k < BOND_CAP; k = k + 1u) { bond[cb + k] = 0u; }

  // §18.2 W3 / §18.22 — THE TAG SURVIVES THE GESTURE, NOT THE PASS.
  //
  // Through v8 this was an unconditional clear, and it is the whole of the bug
  // Darien reported: index.html calls spawnFluid EVERY FRAME while the pointer
  // is held, so a one-second drag is ~120 spawn events, each welding into its
  // own sliver and bonded to nothing else. MEASURED through the real UI: a 30 wu
  // STONE drag laid 180 particles as 39 bodies + 20 loose grains, largest 13.
  //
  // `P.weldHold` is the ONE tag the host has declared still open. A particle
  // wearing it keeps it, so material laid on frame 60 of a drag still welds to
  // material laid on frame 1 — bonds still only form within BOND_R, so a stroke
  // welds to itself where it touches itself and nowhere else. Everything else
  // is cleared exactly as before, which is what keeps §18.2's cross-event rule
  // (two strokes never weld; an emitter never welds) intact.
  //
  // AT weldHold == 0 THIS IS BIT-FOR-BIT THE v8 WRITE: a particle with a tag
  // takes the same clear, and a particle without one is skipped by a branch
  // whose write would have stored the value already there. Every scene the
  // determinism gate, the goldens and `loadScene` produce holds nothing.
  //
  // This write is SELF-INDEXED and this pass reads no other slot's state_out, so
  // it is §5.1-legal. It CANNOT be done in bondForm1, which reads
  // state_out[j].flags for the weld test — that would be a foreign read racing
  // a self write.
  //
  // §20 adds ONE condition: a rigid member KEEPS its tag, because for a member
  // the tag is not a transient gesture id — IT IS THE BODY ID. `rigidStep`,
  // `rigidSolve` and `rigidMember` all enumerate a body's members by it, and
  // `rg_same_body` is the whole storage argument of §20. Clearing it would
  // dissolve every body on the first formation pass after promotion. Inert
  // wherever no member exists, which is every golden scene.
  var fOut = state_out[i].flags & ~FLAG_CONGEAL;
  if ((fOut & FLAG_RIGID) == 0u && weld_tag(fOut) != P.weldHold) {
    fOut = fOut & ~WELD_TAG_MASK;
  }
  state_out[i].flags = fOut;
}

// ════════════════════════════════════════════════════════════════════════════
// §20 — THE THREE RIGID-BODY PASSES
//
// ONE WORKGROUP PER BODY. That single choice answers the open GPU question
// SPEC19_DRAFT §19.B.6 and design/rigid/SPEC20.md §20.2.8 both name — "can a
// workgroup atomic reduction plus a single-lane per-body update live in one
// dispatch?" — with YES, and without any atomics at all:
//
//   * the reduction is a fixed tree in workgroup memory with a fixed barrier
//     schedule, so it is deterministic by construction AND order-independent
//     for free (integer addition is exact and associative — PLATFORM_NOTES);
//   * lane 0 does the per-body update between two barriers;
//   * every lane then places its share of the members.
//
// So §20 costs +6 dispatches (58 -> 64, +10.3 %), not the +10 the unfused
// arrangement would have cost, and no invocation ever writes a word another
// invocation writes: a workgroup only touches slots that pass ITS OWN
// (FLAG_RIGID && weldTag == tag) test.
//
// MEMBERS ARE ENUMERATED BY A SLOT RANGE PLUS THE TAG. `spawnFluid` appends,
// so one gesture's particles are contiguous in slot space; `eraseWhere` is
// order-preserving, so they stay contiguous. [first, last] is a conservative
// window and the tag test is the truth — an emitter firing mid-gesture can
// interleave foreign slots into the window and they are simply skipped.
// ════════════════════════════════════════════════════════════════════════════

// Seven reduction lanes. 7 * 256 * 4 = 7168 B, well under the 16384 B cap.
// F/G keep hard swept-contact translation separate from the ordinary averaged
// PBF correction: averaging one rim contact over a 450-member wheel makes a
// one-particle rail effectively disappear.
var<workgroup> rgA : array<i32, 256>;
var<workgroup> rgB : array<i32, 256>;
var<workgroup> rgC : array<i32, 256>;
var<workgroup> rgD : array<i32, 256>;
var<workgroup> rgE : array<i32, 256>;
var<workgroup> rgF : array<i32, 256>;
var<workgroup> rgG : array<i32, 256>;

// Fixed-schedule tree reduction over all seven lanes. MUST be called from
// uniform control flow by all 256 invocations. barrier-then-add: every lane has
// finished level k's writes before any lane reads them at level k+1, and the
// trailing barrier publishes [0] to the whole workgroup.
fn rg_reduce(tid : u32) {
  for (var s : u32 = 128u; s > 0u; s = s >> 1u) {
    workgroupBarrier();
    if (tid < s) {
      rgA[tid] = rgA[tid] + rgA[tid + s];
      rgB[tid] = rgB[tid] + rgB[tid + s];
      rgC[tid] = rgC[tid] + rgC[tid + s];
      rgD[tid] = rgD[tid] + rgD[tid + s];
      rgE[tid] = rgE[tid] + rgE[tid + s];
      rgF[tid] = rgF[tid] + rgF[tid + s];
      rgG[tid] = rgG[tid] + rgG[tid + s];
    }
  }
  workgroupBarrier();
}

/**
 * invI and iq from the live inertia sum. Pure function of (s, rsh, ish), so
 * running it every substep is idempotent and costs one fp_div_s on one lane.
 *
 *   dtheta = acc * 2^rsh / (s * 2^(2*ish)) * BAM_PER_RAD
 *          = fp_mul_s(acc, invI, iq)      with invI = 2^E * BAM_PER_RAD / s
 *          and E = iq + rsh - 2*ish
 *
 * E is chosen from the BIT LENGTH of s so invI always lands near 2^28 — inside
 * i32 with room, and never near 0 where the quotient would lose every
 * significant bit. fp_div_s (the exact 64-iteration divide) is used rather than
 * fp_divshift precisely because it carries no den >= 2^15 caller contract: a
 * small body's inertia sum can be far below that, and a contract violation
 * would tick dbg[1] and turn a spec-valid run red for no physical reason.
 *
 * `iq` IS CLAMPED INTO [16, 31], AND THAT WINDOW IS NOT A TASTE CHOICE — it is
 * the domain `i64_shr_rne_sat` is DEFINED on. It reads `1u << (s - 1u)` and
 * `v.hi << (32u - s)`, both of which are indeterminate in WGSL outside 1..31,
 * and `iq - TQ_HI_SH` must land in the same window, which sets the lower edge at
 * 16. The natural value is 30-40 on every shape measured (a 2x20 METAL bar
 * reads 33), so the UPPER clamp is what binds; it costs invI two to nine bits of
 * a 29-bit quantity and nothing measurable of the motion. THE LOWER CLAMP IS
 * UNREACHABLE — it needs four particles inside 0.016 wu of the centroid — so it
 * is not clamped at all: it returns invI = 0 (translate, do not rotate) and
 * TICKS dbg[10], because a silent clamp there would be a wrong angular rate
 * rather than a visibly missing one.
 */
const RG_IQ_MIN : i32 = 16;
const RG_IQ_MAX : i32 = 31;
fn rg_inv_inertia(s : i32, rsh : i32, ish : i32) -> vec2<i32> {
  if (s <= 0) { return vec2<i32>(0, 0); }
  let bl = 32 - i32(countLeadingZeros(bitcast<u32>(s)));   // 1..31
  let iqNat = bl - 1 + 2 * ish - rsh;
  let iq = min(iqNat, RG_IQ_MAX);
  let E  = iq + rsh - 2 * ish;
  if (iqNat < RG_IQ_MIN || E < 1 || E > 62) {
    //@DBG_BEGIN
    fp_flag(10u);
    //@DBG_END
    return vec2<i32>(0, 0);
  }
  return vec2<i32>(fp_div_s(BAM_PER_RAD, s, bitcast<u32>(E)), iq);
}

/**
 * a*b >> s, RNE, saturating, with a RUNTIME s.
 *
 * `fp_mul_s`'s caller contract (spec 2.7 rule 2) is a COMPILE-TIME CONSTANT
 * shift, and `iq` is per-body state, so this takes exactly the route `fp_divshift`
 * documents for the same problem: call `i64_shr_rne_sat` directly with a shift
 * that is RANGE-VALIDATED rather than assumed. Outside 1..31 the helper's own
 * `1u << (s - 1u)` and `v.hi << (32u - s)` are indeterminate in WGSL — which is
 * a portability bug, not a rounding one — so the clamp is load-bearing even
 * though `rg_inv_inertia` guarantees it never binds.
 */
fn rg_mul_shr(a : i32, b : i32, s : i32) -> i32 {
  return i64_shr_rne_sat(i64_mul_i32(a, b), bitcast<u32>(clamp(s, 1, 31)));
}

/** true iff slot i is a live member of the body wearing `tag`. */
fn rg_is_member(i : u32, tag : i32) -> bool {
  let f = state_in[i].flags;
  return (f & FLAG_RIGID) != 0u && i32(weld_tag(f)) == tag;
}
fn rg_is_member_out(i : u32, tag : i32) -> bool {
  let f = state_out[i].flags;
  return (f & FLAG_RIGID) != 0u && i32(weld_tag(f)) == tag;
}

/**
 * PLACE every member of one body: pos_i = c + R(theta) * o_i.
 *
 * (cos, sin) is read from the table ONCE PER BODY, not per member — which is
 * why cartesian rest offsets beat the polar packing a pivot body uses: polar
 * needs two LUT interpolations PER MEMBER, cartesian needs two per BODY plus
 * four fp_mul_s per member. Rest offsets are EXACT i32 ticks with no packing
 * and no quantisation, which is also what makes the membership re-base a
 * lossless integer subtraction rather than a re-capture (§20.4.2).
 *
 * Members are NOT wall-clamped, deliberately: clamping one member and not
 * another IS deformation. The body's centre is clamped instead, and the
 * boundary ring is what actually stops a solid.
 */
fn rg_place(base : u32, tid : u32, tag : i32, first : u32, last : u32,
            cx : i32, cy : i32, th : i32) {
  let c = rg_cos(th);
  let s = rg_sin(th);
  for (var i = first + tid; i <= last; i = i + 256u) {
    if (i >= P.nFluid) { break; }
    if (!rg_is_member(i, tag)) { continue; }
    let ox = rigid[RG_REST_BASE + i * 2u];
    let oy = rigid[RG_REST_BASE + i * 2u + 1u];
    derived[i].pred = vec2<i32>(
      fp_add_sat(cx, fp_sub_sat(fp_mul_s(ox, c, COS_Q), fp_mul_s(oy, s, COS_Q))),
      fp_add_sat(cy, fp_add_sat(fp_mul_s(ox, s, COS_Q), fp_mul_s(oy, c, COS_Q))));
  }
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 0 — rigidStep   (runs BEFORE `predict`, which then skips members)
//
//   V += (gSum + SUM stAcc) / N     with the remainder CARRIED  (§20.3.4)
//   W += (SUM cross(r, stAcc)) * invI
//   C = c + V ;  Q = theta + W ;  PLACE
//
// GRAVITY PRODUCES EXACTLY ZERO TORQUE, and the way it does so is TO NOT BE
// ACCUMULATED AT ALL. The frame origin is the members' centroid and every
// particle here has the same mass, so SUM r_i = 0 and SUM cross(r_i, g) =
// cross(SUM r_i, g) = 0 for a uniform grav. Folding gravity as the single
// per-body constant `gSum` removes an entire reduction AND removes the only
// place a rounding could give a symmetric body a phantom spin (gate G6).
// The constraint that buys: A BODY'S MEMBERS MUST SHARE A `grav`. One gesture
// is one material, so it is free today, and the membership rule MAINTAINS it.
//
// Surface tension gets no such exemption — stAcc differs per member — so it is
// folded into both the force and the torque here, one substep late exactly as
// it is for a free particle (§5.2).
// bindings: 0, 1, 3, 13, 18 [, 15 debug]
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn rigidStep(@builtin(workgroup_id) wg : vec3<u32>,
             @builtin(local_invocation_id) lid : vec3<u32>) {
  if (wg.x >= P.nBodies) { return; }
  let base = RG_BODY_BASE + wg.x * RG_BODY_WORDS;
  let tid  = lid.x;

  let live  = rigid[base + RB_LIVE];
  let tag   = rigid[base + RB_TAG];
  let first = bitcast<u32>(rigid[base + RB_FIRST]);
  let last  = bitcast<u32>(rigid[base + RB_LAST]);
  let n     = rigid[base + RB_N];
  let rsh   = bitcast<u32>(rigid[base + RB_RSH]);
  let drive = RG_DRIVER_BASE + wg.x * RG_DRIVER_WORDS;
  let dmask = rigid[drive + RD_MASK];
  var cx    = rigid[base + RB_CX];
  var cy    = rigid[base + RB_CY];
  let axis  = rigid[base + RB_AXIS];
  let usr   = rigid[base + RB_USER];
  let ok    = (live != 0 && n >= RIGID_MIN_MEMBERS);
  // Position/angle slots are kinematic constraints. They replace the body's
  // degree of freedom before prediction; the solver branches below also refuse
  // to move a held axis, so the target is not merely a one-frame teleport.
  if ((dmask & RD_HAS_PX) != 0) { cx = rigid[drive + RD_PX]; }
  if ((dmask & RD_HAS_PY) != 0) { cy = rigid[drive + RD_PY]; }

  var fx : i32 = 0;
  var fy : i32 = 0;
  var thi : i32 = 0;
  var tlo : i32 = 0;
  if (ok) {
    for (var i = first + tid; i <= last; i = i + 256u) {
      if (i >= P.nFluid) { break; }
      if (!rg_is_member(i, tag)) { continue; }
      let st0 = derived[i].stAcc;
      // §19.B. GRAVITY IS A PER-MEMBER FORCE FOR AN AXIS BODY AND A PER-BODY
      // CONSTANT FOR A FREE ONE, and the difference is the frame origin. `axis`
      // is a body record word, so this branch is uniform across the workgroup
      // and gI is EXACTLY ZERO for every body that existed before §19.B — the
      // free-body arithmetic is untouched to the bit.
      var gI : i32 = 0;
      if (axis != 0) { gI = mats[state_in[i].matId].grav; }
      let sy0 = fp_add_sat(st0.y, gI);
      // §23.B. THE JET IS A PER-MEMBER FORCE AND IT IS SUPPOSED TO PRODUCE
      // TORQUE. Gravity is folded as the per-body constant `gSum` precisely so
      // that it CANNOT torque a body (SUM r_i = 0 about the centroid); a jet is
      // the opposite case — it is not uniform over the body, and a jet playing
      // on one flank of a wheel must turn it. So it goes in here, per member,
      // where the same loop that sums the force also sums cross(r, f).
      // At nJets == 0 `jet_accel` is (0, 0) and `fp_add_sat(x, 0) == x`, so
      // every existing body's arithmetic is untouched to the bit.
      let jm = jet_accel(state_in[i].pos);
      let jx0 = fp_add_sat(st0.x, jm.x);
      let jy0 = fp_add_sat(sy0, jm.y);
      // PER-TERM clamp — never the running total. 2^15 per member against
      // NMEM_MAX_R = 2^14 bounds the sum at 2^29; a surface-tension impulse
      // this large is 0.5 wu/substep^2 and has never been observed. Folding
      // gravity AND the jet in BEFORE the clamp is what keeps that bound
      // exactly as proved — a jet cannot widen it.
      let sx = clamp(jx0, -32768, 32768);
      let sy = clamp(jy0, -32768, 32768);
      //@DBG_BEGIN
      if (sx != jx0 || sy != jy0) { fp_flag(11u); }
      //@DBG_END
      fx = fx + sx;
      fy = fy + sy;
      let p = state_in[i].pos;
      let rx = fp_shr_rne(fp_sub_sat(p.x, cx), rsh);
      let ry = fp_shr_rne(fp_sub_sat(p.y, cy), rsh);
      let raw = rx * sy - ry * sx;
      //@DBG_BEGIN
      if (raw > TQ_TERM_R || raw < NEG_TQ_TERM) { fp_flag(9u); }
      //@DBG_END
      let t = clamp(raw, NEG_TQ_TERM, TQ_TERM_R);
      let hi = t >> TQ_HI_SH;
      thi = thi + hi;
      tlo = tlo + (t - (hi << TQ_HI_SH));
    }
  }
  rgA[tid] = fx; rgB[tid] = fy; rgC[tid] = thi; rgD[tid] = tlo; rgE[tid] = 0;
  rgF[tid] = 0; rgG[tid] = 0;
  rg_reduce(tid);

  if (tid == 0u && ok) {
    let invI = rigid[base + RB_INVI];
    let iq   = rigid[base + RB_IQ];
    let gS   = rigid[base + RB_GSUM];
    // THE CARRIED REMAINDER. acc = res + SUM ; d = idivRne(acc, N) ;
    // res = acc - N*d. The impulse is DEFERRED, never destroyed: the system
    // momentum is exactly -SUM_bodies res away from its exact value and
    // |res| <= N/2 forever. Measured at N = 10 000 x 10 000 substeps: leak
    // +4717 against res -4717, EXACTLY, where a naive fp_divshift leaked
    // -425 283. The VELOCITY fold and the POSITION fold carry SEPARATE
    // remainders because they are dimensionally different — sharing one would
    // be a unit bug of exactly the §16.8 kind.
    // §19.B. AN AXIS BODY DOES NOT INTEGRATE VELOCITY AT ALL. The axle supplies
    // whatever force holds the centre — that is what a bearing IS — so the
    // linear fold is not computed, not carried and not deferred. `gSum` goes
    // unread for the same reason: for an axis body gravity has already been
    // folded per member into the TORQUE above, where it belongs.
    // §24. The axle absorbs the PLAYER for exactly the same reason it absorbs
    // gravity: there is no linear fold here to add a drive to. A user-flagged
    // axis body therefore reads omega 0 and theta 0 exactly, and that is gated
    // rather than hidden (test/user.html U6).
    let holdX = axis != 0 || (dmask & RD_HAS_PX) != 0;
    let holdY = axis != 0 || (dmask & RD_HAS_PY) != 0;
    var dvx : i32 = 0;
    var dvy : i32 = 0;
    if (axis == 0) {
      let ax = rigid[base + RB_RESVX] + rgA[0];
      let ay = rigid[base + RB_RESVY] + rgB[0] + gS;
      dvx = rg_idiv_rne(ax, n);
      dvy = rg_idiv_rne(ay, n);
      rigid[base + RB_RESVX] = ax - dvx * n;
      rigid[base + RB_RESVY] = ay - dvy * n;
      // §24 — THE PLAYER'S DRIVE. It enters AFTER the divide, so it is an
      // acceleration of the body and NOT a force to be shared out among
      // members: adding `userA * n` to the numerator would have been the same
      // number only when n divides it, and it would have contaminated the
      // carried remainder — which exists to make the deferred impulse exactly
      // recoverable and must stay a pure function of the FORCE sum.
      //
      // BIT-INERT TWICE OVER: `usr` is 0 for every body record ever written
      // before §24 (word 28 of a zero-filled record), and `P.userA` is (0, 0)
      // until a host calls `setUserDrive`. `fp_add_sat(dv, 0) == dv` exactly.
      var uax : i32 = 0;
      var uay : i32 = 0;
      if (usr != 0) { uax = P.userAx; uay = P.userAy; }
      dvx = fp_add_sat(dvx, uax);
      dvy = fp_add_sat(dvy, uay);
    }
    if ((dmask & RD_HAS_FX) != 0) { dvx = fp_add_sat(dvx, rigid[drive + RD_FX]); }
    if ((dmask & RD_HAS_FY) != 0) { dvy = fp_add_sat(dvy, rigid[drive + RD_FY]); }
    var vx0 = rigid[base + RB_VX];
    var vy0 = rigid[base + RB_VY];
    if ((dmask & RD_HAS_VX) != 0) { vx0 = rigid[drive + RD_VX]; }
    if ((dmask & RD_HAS_VY) != 0) { vy0 = rigid[drive + RD_VY]; }
    let vmaxN = fp_neg_sat(P.vmax);
    let vx = clamp(fp_add_sat(vx0, dvx), vmaxN, P.vmax);
    let vy = clamp(fp_add_sat(vy0, dvy), vmaxN, P.vmax);
    if (holdX) {
      rigid[base + RB_VX] = 0;
      rigid[base + RB_RESVX] = 0;
      rigid[base + RB_CQX] = cx;
      rigid[base + RB_CX] = cx;
    } else {
      rigid[base + RB_VX] = vx;
      rigid[base + RB_CQX] = clamp(fp_add_sat(cx, vx), WALL, P.domW - WALL);
    }
    if (holdY) {
      rigid[base + RB_VY] = 0;
      rigid[base + RB_RESVY] = 0;
      rigid[base + RB_CQY] = cy;
      rigid[base + RB_CY] = cy;
    } else {
      rigid[base + RB_VY] = vy;
      // The centre is clamped into the box; MEMBERS ARE NOT (see rg_place).
      rigid[base + RB_CQY] = clamp(fp_add_sat(cy, vy), WALL, P.domH - WALL);
    }

    // THE HI/LO SPLIT REJOINS HERE. hi*2^15 + lo == term EXACTLY, so multiplying
    // each half by invI at shifts (iq - 15) and iq and adding is the same number
    // as one multiply would have been, computed without ever needing a 41-bit
    // accumulator. Two roundings replace one; 1 BAM is 1.46e-9 rad.
    var dth = 0;
    if (invI != 0) {
      dth = fp_add_sat(rg_mul_shr(rgC[0], invI, iq - i32(TQ_HI_SH)),
                       rg_mul_shr(rgD[0], invI, iq));
    }
    if ((dmask & RD_HAS_TQ) != 0) {
      dth = fp_add_sat(dth, rigid[drive + RD_TQ]);
    }
    var theta = rigid[base + RB_THETA];
    var w0 = rigid[base + RB_OMEGA];
    if ((dmask & RD_HAS_A) != 0) { theta = rigid[drive + RD_A]; }
    if ((dmask & RD_HAS_W) != 0) { w0 = rigid[drive + RD_W]; }
    let om = rigid[base + RB_OMEGAM];
    let w  = clamp(fp_add_sat(w0, dth), -om, om);
    if ((dmask & RD_HAS_A) != 0) {
      // A direct angle owns the rotational degree of freedom. Spin/torque may
      // still be present in malformed authored data, but cannot move a held
      // angle; this precedence is reported by the importer.
      rigid[base + RB_THETA] = theta;
      rigid[base + RB_THETAQ] = theta;
      rigid[base + RB_OMEGA] = 0;
    } else {
      rigid[base + RB_OMEGA] = w;
      // theta wraps. BAM wraparound IS the topology: no modulo, no reduction.
      rigid[base + RB_THETAQ] = theta + w;
    }
  }
  workgroupBarrier();
  if (ok) {
    rg_place(base, tid, tag, first, last,
             rigid[base + RB_CQX], rigid[base + RB_CQY], rigid[base + RB_THETAQ]);
  }
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 20/25/30/35 — rigidSolve   (once per solver ITERATION, after
// `applyFric`, which is where every external correction for this substep's
// iteration has landed in derived[i].dp)
//
//   C += (SUM dp) / N        remainder carried
//   Q += (SUM cross(r, dp)) * invI
//   PLACE
//
// The member's dp is the SUM of the density solve (solveB wrote it; applyDp
// early-returned) and friction + wall contacts (friction ADDED to it). It is
// shifted out of ticks<<ACC and clamped by the member's OWN material dpMax
// here, exactly as applyDp_at would have done — one clamp, one meaning, and
// nothing new to prove.
// bindings: 0, 1, 3, 13, 18 [, 15 debug]
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn rigidSolve(@builtin(workgroup_id) wg : vec3<u32>,
              @builtin(local_invocation_id) lid : vec3<u32>) {
  if (wg.x >= P.nBodies) { return; }
  let base = RG_BODY_BASE + wg.x * RG_BODY_WORDS;
  let tid  = lid.x;

  let live  = rigid[base + RB_LIVE];
  let tag   = rigid[base + RB_TAG];
  let first = bitcast<u32>(rigid[base + RB_FIRST]);
  let last  = bitcast<u32>(rigid[base + RB_LAST]);
  let n     = rigid[base + RB_N];
  let rsh   = bitcast<u32>(rigid[base + RB_RSH]);
  let cqx   = rigid[base + RB_CQX];
  let cqy   = rigid[base + RB_CQY];
  let axis  = rigid[base + RB_AXIS];
  let drive = RG_DRIVER_BASE + wg.x * RG_DRIVER_WORDS;
  let dmask = rigid[drive + RD_MASK];
  let ok    = (live != 0 && n >= RIGID_MIN_MEMBERS);

  var fx : i32 = 0;
  var fy : i32 = 0;
  var sx : i32 = 0;
  var sy : i32 = 0;
  var sn : i32 = 0;
  var thi : i32 = 0;
  var tlo : i32 = 0;
  if (ok) {
    for (var i = first + tid; i <= last; i = i + 256u) {
      if (i >= P.nFluid) { break; }
      if (!rg_is_member(i, tag)) { continue; }
      let dm  = mats[state_in[i].matId].dpMax;
      let dmN = fp_neg_sat(dm);
      let dp  = derived[i].dp;
      let seal = rigid_seal_delta(i, tag);
      let dx  = clamp(fp_shr_rne(dp.x, ACC), dmN, dm);
      let dy  = clamp(fp_shr_rne(dp.y, ACC), dmN, dm);
      fx = fx + dx;
      fy = fy + dy;
      if (seal.x != 0 || seal.y != 0) {
        sx = sx + seal.x;
        sy = sy + seal.y;
        sn = sn + 1;
      }
      let tx = fp_add_sat(dx, seal.x);
      let ty = fp_add_sat(dy, seal.y);
      let p  = derived[i].pred;
      let rx = fp_shr_rne(fp_sub_sat(p.x, cqx), rsh);
      let ry = fp_shr_rne(fp_sub_sat(p.y, cqy), rsh);
      let raw = rx * ty - ry * tx;
      //@DBG_BEGIN
      if (raw > TQ_TERM_R || raw < NEG_TQ_TERM) { fp_flag(9u); }
      //@DBG_END
      let t = clamp(raw, NEG_TQ_TERM, TQ_TERM_R);
      let hi = t >> TQ_HI_SH;
      thi = thi + hi;
      tlo = tlo + (t - (hi << TQ_HI_SH));
    }
  }
  rgA[tid] = fx; rgB[tid] = fy; rgC[tid] = thi; rgD[tid] = tlo; rgE[tid] = sn;
  rgF[tid] = sx; rgG[tid] = sy;
  rg_reduce(tid);

  if (tid == 0u && ok) {
    let invI = rigid[base + RB_INVI];
    let iq   = rigid[base + RB_IQ];
    // §19.B. THE AXLE ABSORBS IT. SUM dp is exactly the net external push the
    // solver wanted to give the body this iteration; a pinned body answers it
    // with an equal constraint force from the bearing, so the centre does not
    // move and NOTHING is carried — the remainder words stay 0 because there is
    // no deferred impulse, only a reaction. (This is the same momentum-sink
    // licence the domain boundary and §19.A anchored matter already hold, and it
    // is why a Newton's-third-law gate must exclude pinned bodies by
    // construction rather than be weakened to accommodate them.)
    if (axis == 0) {
      let ax = rigid[base + RB_RESX] + rgA[0];
      let ay = rigid[base + RB_RESY] + rgB[0];
      var dcx = rg_idiv_rne(ax, n);
      var dcy = rg_idiv_rne(ay, n);
      rigid[base + RB_RESX] = ax - dcx * n;
      rigid[base + RB_RESY] = ay - dcy * n;
      // A hard swept contact is one body constraint, not a force distributed
      // over every member. Average simultaneous rim contacts among themselves
      // and add that body-level reversion after the ordinary N-member fold.
      // This is what lets a sparse 453-point tire land on a one-particle rail.
      if (rgE[0] > 0) {
        dcx = fp_add_sat(dcx, rg_idiv_rne(rgF[0], rgE[0]));
        dcy = fp_add_sat(dcy, rg_idiv_rne(rgG[0], rgE[0]));
      }
      if ((dmask & RD_HAS_PX) == 0) {
        rigid[base + RB_CQX] = clamp(fp_add_sat(cqx, dcx), WALL, P.domW - WALL);
      } else {
        rigid[base + RB_RESX] = 0;
      }
      if ((dmask & RD_HAS_PY) == 0) {
        rigid[base + RB_CQY] = clamp(fp_add_sat(cqy, dcy), WALL, P.domH - WALL);
      } else {
        rigid[base + RB_RESY] = 0;
      }
    }
    var dth = 0;
    if (invI != 0) {
      dth = fp_add_sat(rg_mul_shr(rgC[0], invI, iq - i32(TQ_HI_SH)),
                       rg_mul_shr(rgD[0], invI, iq));
    }
    if ((dmask & RD_HAS_A) == 0) {
      rigid[base + RB_THETAQ] = rigid[base + RB_THETAQ] + dth;
    }
  }
  workgroupBarrier();
  if (ok) {
    rg_place(base, tid, tag, first, last,
             rigid[base + RB_CQX], rigid[base + RB_CQY], rigid[base + RB_THETAQ]);
  }
}

// ════════════════════════════════════════════════════════════════════════════
// DISPATCH 63 — rigidMember   (after `thermal`, before `bondForm1`)
//
// Three jobs, in order, and it is IDEMPOTENT so it needs no dirty flag and no
// data-dependent dispatch:
//   1. CLOSE THE SUBSTEP.  V = C - c ; W = Q - theta ; c = C ; theta = Q.
//   2. LEAVE.  A member whose row is no longer SOLID_RIGID (it burned, melted,
//      corroded, transmuted or was retired by §17) becomes an ORDINARY FREE
//      PARTICLE, keeping the position and velocity it already had — `finalize`
//      has been writing it a correct velocity all along, so there is nothing to
//      hand over. A member with no live same-body neighbour within BOND_R SHEDS.
//   3. RE-BASE AND RECOUNT.  NOTHING IS EVER RE-CAPTURED:
//          o'  = rne(SUM_live o / N')          (in units of 2^osh)
//          c'  = c + R(theta) * o'
//          o_i = o_i - o'                      EXACT integer subtraction
//      so c' + R(o_i - o') == c + R(o_i) for every member, exactly, up to ONE
//      rounding of R*o'. Re-capturing rest offsets mid-motion is what makes
//      free rigid bodies hard (FINDINGS §1 precondition 2); this never does it.
//      Measured with the OUTERMOST members removed mid-tumble: the largest
//      survivor displacement is 2.24 ticks = 3.4e-5 wu, INDEPENDENT of whether
//      1 or 20 members leave.
//
// A body that falls below RIGID_MIN_MEMBERS DISSOLVES: every survivor becomes a
// free particle with the velocity it had. n = 1 is the hard floor (sumIs2 = 0,
// invI = 0 — a one-particle "rigid body" has neither inertia nor shape); 4 is a
// FEEL constant, because a 2- or 3-member remnant of a burned plank picks up
// 286x / 71x the angular rate of a 12-member one from the same nudge.
// bindings: 0, 2, 10, 11, 13, 18 [, 15 debug]
// ════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn rigidMember(@builtin(workgroup_id) wg : vec3<u32>,
               @builtin(local_invocation_id) lid : vec3<u32>) {
  if (wg.x >= P.nBodies) { return; }
  let base = RG_BODY_BASE + wg.x * RG_BODY_WORDS;
  let tid  = lid.x;

  let live  = rigid[base + RB_LIVE];
  let tag   = rigid[base + RB_TAG];
  let first = bitcast<u32>(rigid[base + RB_FIRST]);
  let last  = bitcast<u32>(rigid[base + RB_LAST]);
  let ish   = bitcast<u32>(rigid[base + RB_ISH]);
  let osh   = bitcast<u32>(rigid[base + RB_OSH]);
  let axis  = rigid[base + RB_AXIS];
  let ok0   = (live != 0);

  // ---- 1. close the substep ------------------------------------------------
  if (tid == 0u && ok0) {
    let cx = rigid[base + RB_CX];
    let cy = rigid[base + RB_CY];
    let cqx = rigid[base + RB_CQX];
    let cqy = rigid[base + RB_CQY];
    let vmaxN = fp_neg_sat(P.vmax);
    rigid[base + RB_VX] = clamp(fp_sub_sat(cqx, cx), vmaxN, P.vmax);
    rigid[base + RB_VY] = clamp(fp_sub_sat(cqy, cy), vmaxN, P.vmax);
    rigid[base + RB_OMEGA] = rigid[base + RB_THETAQ] - rigid[base + RB_THETA];
    rigid[base + RB_CX] = cqx;
    rigid[base + RB_CY] = cqy;
    rigid[base + RB_THETA] = rigid[base + RB_THETAQ];
  }
  workgroupBarrier();

  // ---- 2. leave ------------------------------------------------------------
  if (ok0) {
    for (var i = first + tid; i <= last; i = i + 256u) {
      if (i >= P.nFluid) { break; }
      if (!rg_is_member_out(i, tag)) { continue; }
      let M = mats[state_out[i].matId];
      var leave = (M.solidMode != SOLID_RIGID || M.phantom != 0);
      if (!leave) {
        // SHED. An isolated leftover — the tip of a plank that has burned
        // through — drops off instead of orbiting a hole. One local gather,
        // per-particle, order-independent by construction.
        let pi = state_out[i].pos;
        let cnt = nbr_count(i);
        let nb = i * MAXNBR;
        var touched = false;
        for (var k : u32 = 0u; k < cnt; k = k + 1u) {
          let j = nbr[nb + k];
          if (j >= P.nFluid) { continue; }
          if (!rg_is_member_out(j, tag)) { continue; }
          let dx = fp_shr_rne(fp_sub_sat(pi.x, state_out[j].pos.x), BPRE);
          let dy = fp_shr_rne(fp_sub_sat(pi.y, state_out[j].pos.y), BPRE);
          if (abs(dx) > L0_MAX || abs(dy) > L0_MAX) { continue; }
          if (dx * dx + dy * dy <= BOND_R2) { touched = true; break; }
        }
        leave = !touched;
      }
      if (leave) { state_out[i].flags = state_out[i].flags & ~FLAG_RIGID; }
    }
  }
  workgroupBarrier();

  // ---- 3. recount ----------------------------------------------------------
  var cn : i32 = 0;
  var sox : i32 = 0;
  var soy : i32 = 0;
  var si2 : i32 = 0;
  var gsm : i32 = 0;
  if (ok0) {
    for (var i = first + tid; i <= last; i = i + 256u) {
      if (i >= P.nFluid) { break; }
      if (!rg_is_member_out(i, tag)) { continue; }
      let ox = rigid[RG_REST_BASE + i * 2u];
      let oy = rigid[RG_REST_BASE + i * 2u + 1u];
      cn = cn + 1;
      // WHITELISTED plain shift: an exact floor is what makes the re-base
      // idempotent — (o - (d << osh)) >> osh == (o >> osh) - d EXACTLY.
      sox = sox + (ox >> osh);
      soy = soy + (oy >> osh);
      let a = ox >> ish;
      let b = oy >> ish;
      si2 = si2 + a * a + b * b;
      gsm = gsm + mats[state_out[i].matId].grav;
    }
  }
  rgA[tid] = cn; rgB[tid] = sox; rgC[tid] = soy; rgD[tid] = si2; rgE[tid] = gsm;
  rgF[tid] = 0; rgG[tid] = 0;
  rg_reduce(tid);

  let nLive = rgA[0];
  let dissolve = ok0 && nLive < RIGID_MIN_MEMBERS;
  var dx : i32 = 0;
  var dy : i32 = 0;
  // §19.B. AN AXIS BODY IS NEVER RE-BASED. The re-base moves the frame origin to
  // the LIVE centroid, which is exactly right for a free body — its frame IS its
  // centroid, and losing a member moves it — and exactly wrong for a pinned one,
  // where the origin is a point the user placed. Burning half a paddle off a
  // waterwheel must not walk the axle across the tank. Skipping it is also why
  // an axis body needs no re-capture argument at all: its rest offsets are the
  // ones taken at weld time, forever, and SUM|o|^2 stays the moment of inertia
  // about the SAME pivot as the live member set shrinks.
  if (ok0 && !dissolve && axis == 0) {
    dx = rg_idiv_rne(rgB[0], nLive) << osh;
    dy = rg_idiv_rne(rgC[0], nLive) << osh;
  }
  if (tid == 0u && ok0) {
    if (dissolve) {
      rigid[base + RB_LIVE] = 0;
      rigid[base + RB_N] = 0;
    } else {
      if (dx != 0 || dy != 0) {
        let th = rigid[base + RB_THETA];
        let c = rg_cos(th);
        let s = rg_sin(th);
        rigid[base + RB_CX] = fp_add_sat(rigid[base + RB_CX],
          fp_sub_sat(fp_mul_s(dx, c, COS_Q), fp_mul_s(dy, s, COS_Q)));
        rigid[base + RB_CY] = fp_add_sat(rigid[base + RB_CY],
          fp_add_sat(fp_mul_s(dx, s, COS_Q), fp_mul_s(dy, c, COS_Q)));
      }
      rigid[base + RB_N] = nLive;
      rigid[base + RB_SUMIS2] = rgD[0];
      rigid[base + RB_GSUM] = rgE[0];
      let ii = rg_inv_inertia(rgD[0], rigid[base + RB_RSH], rigid[base + RB_ISH]);
      rigid[base + RB_INVI] = ii.x;
      rigid[base + RB_IQ] = ii.y;
    }
  }
  workgroupBarrier();
  if (ok0) {
    for (var i = first + tid; i <= last; i = i + 256u) {
      if (i >= P.nFluid) { break; }
      if (!rg_is_member_out(i, tag)) { continue; }
      if (dissolve) {
        state_out[i].flags = state_out[i].flags & ~FLAG_RIGID;
      } else if (dx != 0 || dy != 0) {
        rigid[RG_REST_BASE + i * 2u]      = rigid[RG_REST_BASE + i * 2u] - dx;
        rigid[RG_REST_BASE + i * 2u + 1u] = rigid[RG_REST_BASE + i * 2u + 1u] - dy;
      }
    }
  }
}
