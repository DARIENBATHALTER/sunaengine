// SunaEngine — real SunaBox (aether) engine code, extracted UNMODIFIED below this header
// from aether commit e41e06f. Verify: diff <(tail -n +6 src/render.js) <aether>/src/render.js
// Copyright (C) 2026 Darien Bathalter
// SPDX-License-Identifier: AGPL-3.0-only — see LICENSE. NO WARRANTY.

// ============================================================================
// aether — renderer  (implementer G)
// ----------------------------------------------------------------------------
// SIX views live in this file, selectable at runtime so the look can be
// A/B'd by hand:
//
//   'water'    screen-space fluid rendering — Van der Laan / Green / Sainz,
//              "Screen Space Fluid Rendering with Curvature Flow" (I3D 2009),
//              adapted to a 2D sim viewed face-on. THE DEFAULT.
//   'metaball' the original threshold-metaball resolve, mathematics unchanged.
//   'points'   one shaded disc per particle.
//   'flat'     (post-landing patch) the metaball threshold with NO shading —
//              OE-CAKE's own blob look.
//   'point'    (post-landing patch) one DEVICE PIXEL per particle — OE-CAKE's
//              Points, distinct from our filled-disc 'points'.
//   'crosses'  (post-landing patch) a "+" per particle; §19/§20 body members
//              rotate from the OWNING BODY's angle, moving free particles from
//              their velocity heading.
//
// Both composite over the SAME background, so the A/B is a comparison of
// rendering technique and not of lighting-vs-background.
//
// f32 is legal HERE AND ONLY HERE. Hard rules (spec §7.8):
//   * Every simulation buffer is bound READ-ONLY. WebGPU validation then
//     enforces the "no float feedback into sim state" ban mechanically, and
//     the error-scope wrapper turns a violation into a thrown exception rather
//     than a silent invalidation (PLATFORM_NOTES trap #2).
//   * The renderer NEVER calls Engine.step. The page's animation loop drives
//     step and draw separately, and the substeps-per-tick count is a fixed
//     integer, never derived from elapsed time.
//   * No value computed in this file is ever written into a sim buffer. There
//     is no writable storage binding anywhere below; test/run.mjs greps this
//     file for a writable-storage buffer binding type and for a renderer-side
//     call into step, and both greps must keep returning nothing.
//
// Pipeline for 'water', per frame:
//   0. background -> bgTex     (once, on resize — a 2D-canvas grid + wordmark)
//   1. splat      -> fieldTex  additive poly6 kernels: coverage, material, speed
//   2. smooth     -> ping/pong separable bilateral Gaussian (or curvature flow)
//   3. thickness  -> thickTex  coverage mask + wide blur at 1/8 resolution
//   4. composite  -> canvas    normals from central differences of the smoothed
//                              height, Fresnel(refraction, sky reflection),
//                              per-channel Beer-Lambert, two-lobe Blinn-Phong,
//                              rim, chromatic dispersion. The optics come from a
//                              32-slot material table indexed by the material id
//                              carried in the splat field's alpha channel.
//   5. walls      -> canvas    the container, drawn on top
// ============================================================================

// RUNG 2.6 — the renderer's ONE import, and it is read-only metadata: the Mat
// field index list, so derivedLook() can read an interned row's tint /
// solidMode / bondK words out of the entry the engine hands it without
// hard-coding word offsets that would rot silently when the schema grows.
// This does not couple render into sim any deeper than draw() already is
// (engine.n, engine.bodyCount, engine.internEntry — host mirrors, all reads).
import { MAT_FIELDS } from './engine.js';
import { DIAG_WGSL } from './render_diag.wgsl.js';
const TINT_WORD = MAT_FIELDS.indexOf('tint');
const SOLIDMODE_WORD = MAT_FIELDS.indexOf('solidMode');
const BONDK_WORD = MAT_FIELDS.indexOf('bondK');

// ---------------------------------------------------------------------------
// 1. SPLAT — coverage / material / speed field, and the container walls
// ---------------------------------------------------------------------------
const SPLAT_WGSL = /* wgsl */`
// Mirrors src/sim.wgsl's Particle exactly (measured layout: offsets 0,4,8,12,
// 16,20,24,28, stride 32). Words 6 and 7 became live state in v4 — 'temp' is
// Q16.16 degrees and 'fuel' is what is left to burn — and the renderer READS
// them, read-only, to decide what glows.
struct Particle {
  pos   : vec2<i32>,
  vel   : vec2<i32>,
  matId : u32,
  flags : u32,      // bit0 = FLAG_BURNING
  temp  : i32,      // Q16.16 degrees
  fuel  : i32,
};

struct SU {
  invDom   : vec2<f32>,   //  0
  scale    : vec2<f32>,   //  8
  radius   : vec2<f32>,   // 16   NDC radius (x, y) — kept round on screen
  count    : u32,         // 24
  nFluid   : u32,         // 28
  velScale : f32,         // 32
  radScale : f32,         // 36   1.0 for fluid, smaller for container walls
  glowLo   : f32,         // 40   temperature (degrees) where a body starts to glow
  glowInv  : f32,         // 44   1 / (glowHi - glowLo)
  // §21. 0 = MATERIAL COLOUR (every particle wears its row's albedo, exactly
  // the pre-§21 picture), 1 = IMAGE COLOUR (a particle with a non-zero tint
  // alpha wears its own). Both are useful — the material view is how you read
  // the physics, the image view is how you recognise the photograph — so this
  // is a toggle and not a migration.
  tintMode : f32,         // 48
  // THREE SCALARS, NOT A vec3. PLATFORM_NOTES' measured rule: vec3<T> is size 12
  // and ALIGN 16, so a vec3 pad here would land at 64 instead of 52 and push
  // look to 80 — the struct grows 1088 -> 1104 and the host's uniform buffer
  // is silently too small. That is not hypothetical: it is what this edit did on
  // its first run, and WebGPU reported it as a render pass that produced nothing.
  //
  // The post-landing patch spent all three former pads, offsets unchanged:
  //   viewW/viewH  canvas size in DEVICE PIXELS — vs_dot snaps to a pixel
  //                centre, which is what makes a "1px dot" exactly one pixel.
  //   nBodies      the §20 compact body count — vs_cross searches records
  //                [0, nBodies) for the tag in flags bits 8..19.
  viewW    : f32,         // 52
  viewH    : f32,         // 56
  nBodies  : f32,         // 60
  // Per-material LOOK, indexed by matId (clamped to 31). This is the whole
  // hybrid direction in one table:
  //   look[m].rgb  flat graphic albedo — what a SOLID or a GAS of this material
  //                simply IS on screen. Liquids ignore it (they are refractive
  //                and their colour comes from the optics table in the
  //                composite) but still carry one, so a material that changes
  //                class later cannot become invisible.
  //   look[m].w    class code: 0 LIQUID, 1 SOLID, 2 EMISSIVE, 3 GAS.
  //   emis[m].x    glow floor. 1.0 = always alight (FIRE) regardless of temp.
  //   emis[m].y    glow gain multiplier — how hard this material blooms.
  //   emis[m].z    opacity scale for the GAS class (steam thin, smoke thick).
  //   emis[m].w    reserved.
  // RUNG 2.6: 32 -> 64 slots. Slots 32..63 are the §25 INTERNED rows, filled
  // per frame by the host from the engine's own mint list: the base row's look
  // pulled toward the composed row's tint word (the composeTint algebra), so a
  // composed material is visually DISTINCT instead of clamping onto row 31's
  // procedural optic. Render-only: nothing here can reach sim state.
  look     : array<vec4<f32>, 64>,   //   64
  emis     : array<vec4<f32>, 64>,   // 1088
};

// §21. Straight-alpha RGBA8 -> linear-ish colour, and the "do I have one at
// all" test in the same place. The renderer's albedo table is authored in the
// same non-linear 0..1 space rgb() produces (a plain /255), so an imported
// pixel goes through exactly the same door a hand-tuned albedo does — that is
// what makes "material colour" and "image colour" comparable rather than one
// of them being systematically darker.
struct Tint { rgb : vec3<f32>, has : f32 };
fn tintOf(i : u32) -> Tint {
  var t : Tint;
  t.rgb = vec3<f32>(0.0, 0.0, 0.0);
  t.has = 0.0;
  let w = tints[i];
  // The opening enclosure is simulated as anchored BEDROCK, but is UI chrome,
  // not a purple material layer. Its reserved sidecar word makes it black in
  // every render mode without enabling the user's optional image-colour mode.
  // It is deliberately an exact word (not "alpha < 1") so semi-transparent
  // imported pixels retain their normal colour semantics.
  if (w == 0xfe000000u) {
    t.has = 1.0;
    return t;
  }
  if (S.tintMode < 0.5) { return t; }
  let a = (w >> 24u) & 255u;
  if (a == 0u) { return t; }
  t.rgb = vec3<f32>(f32(w & 255u), f32((w >> 8u) & 255u), f32((w >> 16u) & 255u)) * (1.0 / 255.0);
  t.has = 1.0;
  return t;
}

@group(0) @binding(0) var<uniform> S : SU;
@group(0) @binding(1) var<storage, read> parts : array<Particle>;
// §21 PER-PARTICLE COLOUR. One RGBA8 word per SLOT, straight alpha, and
// **a == 0 means "no colour of my own"**. The engine owns and permutes it (it
// is indexed by slot and only spawnFluid/eraseWhere renumber a slot); this
// shader only ever READS it, like every other sim access in this file.
//
// IT IS A SEPARATE BUFFER AND THAT IS THE ENTIRE SAFETY ARGUMENT. No sim
// pipeline binds it, so no colour can reach a trajectory or a digest. The cost
// here is one extra read-only storage buffer in the VERTEX stage — 1 -> 2
// against this device's cap of 8 (PLATFORM_NOTES), and the fragment stage still
// binds none.
@group(0) @binding(2) var<storage, read> tints : array<u32>;
// POST-LANDING PATCH: the §20 rigid buffer, READ-ONLY, vertex stage only — the
// third sim buffer this stage binds against the device cap of 8. Only vs_cross
// reads it (a "+" on a body member rotates with the OWNING BODY's angle, which
// is real orientation — tension and compression made visible); every other
// entry point ignores it, which the explicit bind group layout makes safe
// (PLATFORM_NOTES trap #6 is about layout:'auto', which this file never uses).
// No render pass can write it: the binding is declared read-only both in WGSL
// and in the bind group layout, which the static gate enforces.
@group(0) @binding(3) var<storage, read> rigidBuf : array<i32>;

// §20's rigid-buffer geometry, mirrored from src/engine.js RG_*/RB. rigid.html
// R0 already proves engine.js and sim.wgsl agree on these; the renderer only
// READS words 0/1/8 of a record, and only in the crosses view.
const CR_BODY_BASE  : u32 = 1088u;   // RG_BODY_BASE
const CR_BODY_WORDS : u32 = 32u;     // RG_BODY_WORDS
const CR_MAX_BODIES : u32 = 1024u;   // RG_MAX_BODIES
const CR_LIVE  : u32 = 0u;           // RB.LIVE
const CR_TAG   : u32 = 1u;           // RB.TAG — ALREADY SHIFTED into the mask
const CR_THETA : u32 = 8u;           // RB.THETA — i32 BAM, 2^32 BAM = one turn
const CR_TAG_MASK : u32 = 0x000fff00u;   // WELD_TAG_MASK
const CR_FLAG_RIGID : u32 = 16u;         // FLAG_RIGID
// BAM -> radians for DISPLAY only: 2*pi / 2^32. f32 loses low bits of a large
// BAM, which at worst is ~1e-7 of a turn of cross rotation — invisible.
const CR_BAM_TO_RAD : f32 = 1.4629180792671596e-9;

struct VOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) uv  : vec2<f32>,
  // x = non-water flag, y = speed01, z = material id (as a float), w = hot01
  @location(1) aux : vec4<f32>,
  // rgb = flat albedo (zero for a LIQUID), w = class code
  @location(2) look : vec4<f32>,
  // xy = signed particle velocity, z = temperature relative to room
  // temperature, w = unused. These are renderer-only values used by the
  // diagnostic-field splat; no simulation pipeline can read them back.
  @location(3) diag : vec4<f32>,
};

// §36 ORBIES. Shader-mode fluids deliberately cohere into one smooth screen
// surface. That is right for water and Gel, but wrong for a handful of tiny
// glass gumballs. Orbies therefore bypass the field and are drawn as individual
// translucent spheres after the composite. This is render-only: the hue comes
// from a stable slot/position hash and can never feed back into simulation.
const ORBIE_ID : u32 = 27u;
struct BeadOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) uv : vec2<f32>,
  @location(1) hue : f32,
};

fn orbie_rgb(h : f32) -> vec3<f32> {
  let q = abs(fract(vec3<f32>(h, h + 0.3333333, h + 0.6666667)) * 6.0 - 3.0);
  return clamp(q - 1.0, vec3<f32>(0.0), vec3<f32>(1.0));
}

@vertex
fn vs_bead(@builtin(vertex_index) vi : u32, @builtin(instance_index) ii : u32) -> BeadOut {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
  var o : BeadOut;
  o.uv = vec2<f32>(0.0); o.hue = 0.0;
  if (ii >= S.nFluid) { o.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0); return o; }
  let p = parts[ii];
  if (p.matId != ORBIE_ID) { o.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0); return o; }
  let unit = vec2<f32>(f32(p.pos.x), f32(p.pos.y)) * S.invDom;
  let base = vec2<f32>((unit.x * 2.0 - 1.0) * S.scale.x, (1.0 - unit.y * 2.0) * S.scale.y);
  // A small overlap leaves a satisfying packed volume without merging the beads.
  o.clip = vec4<f32>(base + corners[vi] * S.radius * 0.68, 0.0, 1.0);
  o.uv = corners[vi];
  // Slot order is stable while the bead lives, so the colour does not flicker
  // as it rolls. (Erase/compaction may re-seat a bead, which is an acceptable
  // little shuffle for a deliberately playful material.)
  var hash = ii * 747796405u + 2891336453u;
  hash = (hash ^ (hash >> 16u)) * 2246822519u;
  o.hue = f32(hash & 1023u) * (1.0 / 1024.0);
  return o;
}

@fragment
fn fs_bead(in : BeadOut) -> @location(0) vec4<f32> {
  let d = length(in.uv);
  if (d > 1.0) { discard; }
  let z = sqrt(max(0.0, 1.0 - d * d));
  let n = normalize(vec3<f32>(in.uv.x, in.uv.y, z));
  let l = normalize(vec3<f32>(-0.45, 0.55, 0.70));
  let glint = pow(max(0.0, dot(reflect(-l, n), vec3<f32>(0.0, 0.0, 1.0))), 72.0);
  let rim = pow(1.0 - z, 1.45);
  // A gentle radial hue shift is an intentional diffraction cue: each bead is
  // rainbow on its own, while the hash keeps a packed collection varied.
  let core = orbie_rgb(in.hue);
  let edge = orbie_rgb(in.hue + 0.13 + d * 0.18);
  let col = mix(core * (0.32 + 0.48 * z), edge * 1.18, rim)
          + vec3<f32>(1.0) * (glint * 1.25 + rim * 0.18);
  let alpha = smoothstep(1.0, 0.78, d) * (0.72 + 0.23 * z);
  return vec4<f32>(min(col, vec3<f32>(1.0)), alpha);
}

@vertex
fn vs(@builtin(vertex_index) vi : u32, @builtin(instance_index) ii : u32) -> VOut {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0,  1.0), vec2<f32>(1.0, -1.0), vec2<f32>( 1.0, 1.0));

  var o : VOut;
  o.uv = vec2<f32>(0.0, 0.0);
  o.aux = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  o.look = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  o.diag = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  if (ii >= S.count) { o.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0); return o; }

  let p = parts[ii];
  let c = corners[vi];
  // +y is the gravity direction in sim space, so y is flipped for display.
  let unit = vec2<f32>(f32(p.pos.x), f32(p.pos.y)) * S.invDom;
  let base = vec2<f32>((unit.x * 2.0 - 1.0) * S.scale.x, (1.0 - unit.y * 2.0) * S.scale.y);
  o.clip = vec4<f32>(base + c * S.radius * S.radScale, 0.0, 1.0);
  o.uv = c;

  let sp = length(vec2<f32>(f32(p.vel.x), f32(p.vel.y))) * S.velScale;
  // matId is CLAMPED, not assumed: this renderer must survive material ids that
  // did not exist when it was written. 63 keeps the id-weighted accumulation
  // below f16's exactly-representable range for any plausible overlap count.
  let mid = f32(min(p.matId, 63u));
  let slot = min(p.matId, 63u);   // RUNG 2.6: interned rows have their own look

  // Temperature -> glow. 'temp' is Q16.16 degrees and is REAL SIMULATION STATE
  // as of v4, so a metal bar carrying a flame's heat lights up on its own with
  // nothing else to tell the renderer about it. Reading it is read-only, like
  // every other sim access in this file.
  let degC = f32(p.temp) * (1.0 / 65536.0);
  var hot = clamp((degC - S.glowLo) * S.glowInv, 0.0, 1.0);
  hot = max(hot, S.emis[slot].x);
  // A particle that is actively BURNING glows even if its own temperature has
  // not caught up yet — that is the ignition frame, and it is the frame the eye
  // is waiting for.
  if ((p.flags & 1u) != 0u) { hot = max(hot, 0.55); }

  o.aux = vec4<f32>(select(0.0, 1.0, p.matId != 0u), clamp(sp, 0.0, 1.0), mid, hot);
  o.diag = vec4<f32>(
    clamp(f32(p.vel.x) * S.velScale, -1.0, 1.0),
    clamp(f32(p.vel.y) * S.velScale, -1.0, 1.0),
    clamp((degC - 20.0) * (1.0 / 1000.0), -1.0, 2.0),
    0.0);
  o.look = S.look[slot];
  // §21. THE COLOUR REPLACES THE ALBEDO AND NOTHING ELSE. o.look.w — the
  // class code — is deliberately left alone: a red pixel of STONE is still a
  // SOLID, a red pixel of SMOKE is still a GAS, and the retirement cull below
  // still fires. If colour could move a particle between classes it would
  // change how the composite lights it, which is a physics-shaped decision
  // being made by a paint bucket.
  let tn = tintOf(ii);
  o.look = vec4<f32>(mix(o.look.rgb, tn.rgb, tn.has), o.look.w);
  // §17. Look class 4 = INVISIBLE: retired matter. Pushed behind the far plane
  // and clipped, which is the same mechanism the "ii >= S.count" early-out above
  // already uses — so a corpse costs one vertex invocation and ZERO fragments,
  // contributes nothing to the coverage field and nothing to the class field.
  // Doing it here rather than in fs_splat is what makes it free; a discard
  // would still rasterise the quad. Table-driven: no material id appears here.
  if (o.look.w > 3.5) { o.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0); }
  return o;
}

// Poly6-shaped radial falloff. Smooth to second order at the rim, so the summed
// field has no kernel-boundary creases to become fake normals.
//
// The four channels of fieldTex, all accumulated with plain additive blending:
//   r  coverage           sum(w)
//   g  non-water coverage sum(w * (matId != 0))
//   b  speed              sum(w * speed01)
//   a  id moment          sum(w * matId)
// so g/r is "how much of this pixel is not water" and a/g is the mean material
// id AMONG THE NON-WATER contributors — which is stable across a water/X
// interface, where a/r would sweep through every id between 0 and X and paint a
// rainbow fringe of materials that are not there.
@fragment
fn fs_splat(in : VOut) -> @location(0) vec4<f32> {
  if (u32(in.aux.z + 0.5) == ORBIE_ID) { discard; }
  let d2 = dot(in.uv, in.uv);
  if (d2 > 1.0) { discard; }
  let t = 1.0 - d2;
  let w = t * t * t;
  return vec4<f32>(w, w * in.aux.x, w * in.aux.y, w * in.aux.z);
}

// A second, low-resolution vector field for visual instruments. Unlike
// fieldTex this keeps the two signed velocity components and temperature.
// Alpha is the kernel mass used to recover a weighted mean after blur.
@fragment
fn fs_diag(in : VOut) -> @location(0) vec4<f32> {
  let d2 = dot(in.uv, in.uv);
  if (d2 > 1.0) { discard; }
  let t = 1.0 - d2;
  let w = t * t * t;
  return vec4<f32>(in.diag.xyz * w, w);
}

// ---------------------------------------------------------------------------
// THE CLASS FIELD — the hybrid direction's data path.
//
// A second, half-resolution splat of the same particles into two attachments.
// It exists because "18 materials all rendered as glass blur into mush": the
// composite has to know, per pixel, how much of that pixel is refractive
// LIQUID, how much is flat SOLID, how much is glowing EMISSIVE and how much is
// soft GAS — and what colour the non-liquid part is.
//
// Why a second pass instead of more channels on the first: the first field is
// blurred four times at full resolution and its parameters are MEASURED (grain
// 2.460 % -> 0.210 %); widening it triples that cost. These two attachments are
// only ever used as RATIOS against each other, so they can live at half
// resolution behind a single cheap blur, and the sharp silhouette still comes
// from the full-resolution field.
//
//   mat.rgb  sum(w * albedo) over NON-LIQUID particles only, so a stone under
//            water is stone-coloured and not a wash of stone and water.
//   mat.a    sum(w)          over EVERY particle — the denominator.
//   cls.rgb  sum(w) for SOLID / EMISSIVE / GAS. LIQUID is the remainder,
//            mat.a - (r+g+b), which is exactly zero for a scene with no
//            liquid in it and exactly mat.a for water on its own.
//   cls.a    sum(w * hot01) — the glow mass, and the only channel the halo
//            blur needs.
// ---------------------------------------------------------------------------
struct ClsOut {
  @location(0) mat : vec4<f32>,
  @location(1) cls : vec4<f32>,
};

@fragment
fn fs_cls(in : VOut) -> ClsOut {
  if (u32(in.aux.z + 0.5) == ORBIE_ID) { discard; }
  let d2 = dot(in.uv, in.uv);
  if (d2 > 1.0) { discard; }
  let t = 1.0 - d2;
  let w = t * t * t;
  let k = i32(in.look.w + 0.5);
  let isSolid = select(0.0, 1.0, k == 1);
  let isEmis  = select(0.0, 1.0, k == 2);
  let isGas   = select(0.0, 1.0, k == 3);
  let nonLiquid = isSolid + isEmis + isGas;

  var o : ClsOut;
  o.mat = vec4<f32>(in.look.rgb * (w * nonLiquid), w);
  o.cls = vec4<f32>(w * isSolid, w * isEmis, w * isGas, w * in.aux.w);
  return o;
}

// Container walls, drawn on top of the composite.
@fragment
fn fs_wall(in : VOut) -> @location(0) vec4<f32> {
  let d = length(in.uv);
  if (d > 1.0) { discard; }
  let z = sqrt(max(0.0, 1.0 - d * d));
  let n = normalize(vec3<f32>(in.uv.x, in.uv.y, z));
  let l = normalize(vec3<f32>(-0.45, 0.55, 0.70));
  let diff = 0.34 + 0.42 * max(0.0, dot(n, l));
  let edge = smoothstep(1.0, 0.72, d);
  let col = vec3<f32>(0.115, 0.125, 0.165) * diff * 2.0;
  return vec4<f32>(col * edge, edge);
}

// ---- the ORIGINAL point / metaball-field shaders, mathematics unchanged -----
fn matColor(m : u32, speed : f32) -> vec3<f32> {
  // Every material gets its OWN colour here, not just WATER and GOO.
  //
  // The per-material albedo and class code already ride in the uniform for the
  // hybrid water renderer (S.look / S.emis, see the SU comment above). The two
  // legacy views simply never read them, so all 25 materials collapsed onto a
  // two-way select between WATER and GOO — every solid rendered as water, every
  // gas as water, fire as water. Reading the SAME table keeps Metaball and Points
  // consistent with Water mode by construction: retune a material's look once
  // and all three renderers follow.
  let mi  = min(m, 63u);   // RUNG 2.6: composed rows keep their derived look here too
  let L   = S.look[mi];
  let E   = S.emis[mi];
  let cls = u32(L.w + 0.5);          // 0 LIQUID, 1 SOLID, 2 EMISSIVE, 3 GAS
  let s   = clamp(speed, 0.0, 1.0);

  // Speed still brightens, as it always did — fast matter reads lighter. Kept
  // gentler than the old hardcoded ramp because a real albedo is already the
  // right value, where the old one started from an arbitrary dark teal.
  var col = mix(L.rgb, min(L.rgb * 1.9 + vec3<f32>(0.16), vec3<f32>(1.0)), s * 0.8);

  // EMISSIVE must read as ALIGHT in a debug view too, or fire and lava look
  // like dull orange dust here while glowing in Water mode.
  if (cls == 2u) {
    col = min(col * (1.0 + E.y * 0.9) + vec3<f32>(E.x * 0.30), vec3<f32>(1.45));
  }
  // GAS reads wispy rather than solid: scale by the same per-material opacity
  // the composite uses, so steam stays thin and smoke stays thick.
  if (cls == 3u) {
    col = col * mix(0.55, 1.0, clamp(E.z, 0.0, 1.0));
  }
  return col;
}

struct LOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) uv   : vec2<f32>,
  @location(1) tint : vec4<f32>,
};

@vertex
fn vs_legacy(@builtin(vertex_index) vi : u32, @builtin(instance_index) ii : u32) -> LOut {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0,  1.0), vec2<f32>(1.0, -1.0), vec2<f32>( 1.0, 1.0));

  var out : LOut;
  out.uv = vec2<f32>(0.0, 0.0);
  out.tint = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  if (ii >= S.count) {
    out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);   // behind the far plane: clipped
    return out;
  }
  let p = parts[ii];
  if (p.matId == ORBIE_ID) { out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0); return out; }
  let c = corners[vi];
  let unit = vec2<f32>(f32(p.pos.x), f32(p.pos.y)) * S.invDom;
  let base = vec2<f32>((unit.x * 2.0 - 1.0) * S.scale.x, (1.0 - unit.y * 2.0) * S.scale.y);
  let isBoundary = ii >= S.nFluid;
  let rad = S.radius * select(S.radScale, 0.62, isBoundary);

  out.clip = vec4<f32>(base + c * rad, 0.0, 1.0);
  out.uv = c;

  let v = vec2<f32>(f32(p.vel.x), f32(p.vel.y)) / 5000.0;
  var col = matColor(p.matId, length(v));
  // §21. Points and Metaball read the SAME sidecar the hybrid path does, for the
  // same reason matColor was made to read the same look table: a debug view that
  // disagrees with the beauty view is worse than no debug view. This is also the
  // mode where an imported photograph is most literally itself — one particle,
  // one disc, one pixel's colour, which is exactly OE-CAKE's "texture remapping"
  // degrade path.
  let tn = tintOf(ii);
  col = mix(col, tn.rgb, tn.has);
  if (isBoundary) { col = vec3<f32>(0.15, 0.16, 0.21); }
  out.tint = vec4<f32>(col, 1.0);
  // §17, same cull as the hybrid vs. The legacy point renderer only knew about
  // WATER and GOO, so retired matter would otherwise render as teal water —
  // the single worst way for this to fail, because it looks like a bug in the
  // solver rather than a missing case in a debug view.
  if (S.look[min(p.matId, 63u)].w > 3.5) { out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0); }
  return out;
}

@fragment
fn fs_points(in : LOut) -> @location(0) vec4<f32> {
  let d = length(in.uv);
  if (d > 1.0) { discard; }
  // Cheap sphere shading: normal from the disc, one key light up-left.
  let z = sqrt(max(0.0, 1.0 - d * d));
  let n = normalize(vec3<f32>(in.uv.x, in.uv.y, z));
  let l = normalize(vec3<f32>(-0.45, 0.55, 0.70));
  let diff = 0.45 + 0.55 * max(0.0, dot(n, l));
  let spec = pow(max(0.0, dot(reflect(-l, n), vec3<f32>(0.0, 0.0, 1.0))), 26.0);
  let edge = smoothstep(1.0, 0.70, d);
  let col = in.tint.rgb * diff + vec3<f32>(spec * 0.5, spec * 0.5, spec * 0.5);
  return vec4<f32>(col * edge, edge);
}

@fragment
fn fs_field(in : LOut) -> @location(0) vec4<f32> {
  let d = length(in.uv);
  if (d > 1.0) { discard; }
  let w = pow(max(0.0, 1.0 - d * d), 3.0);
  return vec4<f32>(in.tint.rgb * w, w);
}

// ---- POST-LANDING PATCH: the OE-CAKE views — Point (1px) and Crosses --------
// Both reuse vs_legacy's colour path exactly (matColor + §21 tint + boundary
// grey + the §17 retired cull), so a material reads the same in every view.

// The shared prologue, factored: position in NDC, colour, and the culls.
// Returns tint.a == 0 for "cull me" (the caller writes the far-plane clip).
fn legacyColor(ii : u32) -> vec4<f32> {
  let p = parts[ii];
  let v = vec2<f32>(f32(p.vel.x), f32(p.vel.y)) / 5000.0;
  var col = matColor(p.matId, length(v));
  let tn = tintOf(ii);
  col = mix(col, tn.rgb, tn.has);
  if (ii >= S.nFluid) { col = vec3<f32>(0.15, 0.16, 0.21); }
  return vec4<f32>(col, 1.0);
}

// POINT — OE-CAKE's point view, deliberately distinct from our shaded Circles.
// The centre is snapped to a device pixel, but the sprite is 2.7px and brighter
// than its material so it stays visible against the grid on high-DPR displays.
@vertex
fn vs_dot(@builtin(vertex_index) vi : u32, @builtin(instance_index) ii : u32) -> LOut {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0,  1.0), vec2<f32>(1.0, -1.0), vec2<f32>( 1.0, 1.0));
  var out : LOut;
  out.uv = vec2<f32>(0.0, 0.0);
  out.tint = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  if (ii >= S.count) { out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0); return out; }
  let p = parts[ii];
  if (p.matId == ORBIE_ID || S.look[min(p.matId, 63u)].w > 3.5) { out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0); return out; }
  let c = corners[vi];
  let unit = vec2<f32>(f32(p.pos.x), f32(p.pos.y)) * S.invDom;
  let base = vec2<f32>((unit.x * 2.0 - 1.0) * S.scale.x, (1.0 - unit.y * 2.0) * S.scale.y);
  // snap to the centre of the device pixel the particle lands in
  let px = floor((base.x * 0.5 + 0.5) * S.viewW) + 0.5;
  let py = floor((0.5 - base.y * 0.5) * S.viewH) + 0.5;
  let sx = px / S.viewW * 2.0 - 1.0;
  let sy = 1.0 - py / S.viewH * 2.0;
  out.clip = vec4<f32>(sx + c.x * (2.7 / S.viewW), sy + c.y * (2.7 / S.viewH), 0.0, 1.0);
  out.tint = legacyColor(ii);
  return out;
}

@fragment
fn fs_dot(in : LOut) -> @location(0) vec4<f32> {
  // No disc test: a crisp, bright square sprite, rather than a dull lone pixel.
  return vec4<f32>(min(in.tint.rgb * 1.65 + vec3<f32>(0.12), vec3<f32>(1.0)), 1.0);
}

// CROSSES — a "+" per particle, OE-CAKE's cross view. HONESTY NOTE, decided in
// the patch brief: OE-CAKE's crosses rotated because their particles carried
// orientation; ours do not (bond bit 31 is still reserved). A §19/§20 BODY
// MEMBER rotates from its OWNING BODY's angle — that is real orientation and
// the interesting case — found by matching the weld tag in flags bits 8..19
// against the compact body records. A FREE particle takes its VELOCITY HEADING
// when it is really moving and stays axis-aligned at rest (decided by eye:
// heading makes falling spray read as motion streaks while a settled pool
// stays a calm grid; an always-aligned build read dead, an always-heading one
// jittered at rest on solver noise).
@vertex
fn vs_cross(@builtin(vertex_index) vi : u32, @builtin(instance_index) ii : u32) -> LOut {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0,  1.0), vec2<f32>(1.0, -1.0), vec2<f32>( 1.0, 1.0));
  var out : LOut;
  out.uv = vec2<f32>(0.0, 0.0);
  out.tint = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  if (ii >= S.count) { out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0); return out; }
  let p = parts[ii];
  if (p.matId == ORBIE_ID || S.look[min(p.matId, 63u)].w > 3.5) { out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0); return out; }
  let c = corners[vi];
  let unit = vec2<f32>(f32(p.pos.x), f32(p.pos.y)) * S.invDom;
  let base = vec2<f32>((unit.x * 2.0 - 1.0) * S.scale.x, (1.0 - unit.y * 2.0) * S.scale.y);

  // Screen angle. Sim y points DOWN and the display flips it, so a positive
  // sim-space rotation appears NEGATED on screen — both branches negate.
  var ang = 0.0;
  if ((p.flags & CR_FLAG_RIGID) != 0u) {
    let tag = i32(p.flags & CR_TAG_MASK);
    let nB = min(u32(S.nBodies + 0.5), CR_MAX_BODIES);
    for (var b = 0u; b < nB; b = b + 1u) {
      let rb = CR_BODY_BASE + b * CR_BODY_WORDS;
      if (rigidBuf[rb + CR_LIVE] != 0 && rigidBuf[rb + CR_TAG] == tag) {
        ang = -f32(rigidBuf[rb + CR_THETA]) * CR_BAM_TO_RAD;
        break;
      }
    }
  } else {
    let vx = f32(p.vel.x);
    let vy = f32(p.vel.y);
    // ~0.02 wu/substep. Below it a settled pool's residual jitter would spin
    // the crosses; above it a heading is real information. Eye-tuned.
    if (vx * vx + vy * vy > 1310.72 * 1310.72) { ang = atan2(-vy, vx); }
  }
  let ca = cos(ang);
  let sa = sin(ang);
  // Rotate in the isotropic corner space, THEN scale per axis. S.radius keeps
  // discs round on screen (ry = rx * aspect), so a rotation of c is a rigid
  // rotation of the mark on screen.
  let rc = vec2<f32>(c.x * ca - c.y * sa, c.x * sa + c.y * ca);
  let isBoundary = ii >= S.nFluid;
  let rad = S.radius * select(S.radScale, 0.62, isBoundary);
  out.clip = vec4<f32>(base + rc * rad, 0.0, 1.0);
  out.uv = c;
  out.tint = legacyColor(ii);
  return out;
}

@fragment
fn fs_cross(in : LOut) -> @location(0) vec4<f32> {
  // A "+" in quad space: two perpendicular arms, flat colour, no shading. The
  // quad's own rotation (vs_cross rotates the CORNERS, uv stays unrotated) is
  // what turns the mark on screen.
  let t = 0.28;
  if (abs(in.uv.x) > t && abs(in.uv.y) > t) { discard; }
  return vec4<f32>(in.tint.rgb, 1.0);
}
`;

// ---------------------------------------------------------------------------
// 2/3. SMOOTHING + THICKNESS + the background blit
// ---------------------------------------------------------------------------
const BLUR_WGSL = /* wgsl */`
struct BU {
  p : vec4<f32>,   // stepX, stepY (uv units), sigma (taps), bilateral falloff
  q : vec4<f32>,   // threshold, softness, curvature gradient scale, curvature rate
};
@group(0) @binding(0) var<uniform> B : BU;
@group(0) @binding(1) var src : texture_2d<f32>;
@group(0) @binding(2) var samp : sampler;

struct VOut { @builtin(position) clip : vec4<f32>, @location(0) uv : vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) vi : u32) -> VOut {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var o : VOut;
  let q = p[vi];
  o.clip = vec4<f32>(q, 0.0, 1.0);
  o.uv = vec2<f32>((q.x + 1.0) * 0.5, 1.0 - (q.y + 1.0) * 0.5);
  return o;
}

// Separable bilateral Gaussian, 9 taps. The bilateral term (B.p.w > 0) stops the
// blur at the fluid's silhouette so a droplet does not smear into the
// background, while still low-passing the interior — which is where the
// hexagonal lattice lives.
@fragment
fn fs_gauss(in : VOut) -> @location(0) vec4<f32> {
  let dir = B.p.xy;
  let sigma = max(0.35, B.p.z);
  let inv2s2 = 1.0 / (2.0 * sigma * sigma);
  let c0 = textureSample(src, samp, in.uv);
  var sum = vec4<f32>(0.0);
  var wsum = 0.0;
  for (var i : i32 = -4; i <= 4; i = i + 1) {
    let fi = f32(i);
    let ws = exp(-fi * fi * inv2s2);
    let s = textureSample(src, samp, in.uv + dir * fi);
    let dr = s.r - c0.r;
    var we = 1.0;
    if (B.p.w > 0.0) { we = exp(-dr * dr * B.p.w); }
    sum = sum + s * (ws * we);
    wsum = wsum + ws * we;
  }
  return sum / max(wsum, 1e-6);
}

// Curvature flow (Van der Laan §3.2): evolve the height field along its mean
// curvature. Preserves the silhouette better than a Gaussian of equivalent
// smoothing power, because flat regions have H = 0 and do not move. Measured 3x
// worse than the Gaussian on the lattice metric, so it is NOT the default; it is
// kept because it wins on droplet-heavy scenes.
@fragment
fn fs_curv(in : VOut) -> @location(0) vec4<f32> {
  let ex = vec2<f32>(B.p.x, 0.0);
  let ey = vec2<f32>(0.0, B.p.y);
  let c  = textureSample(src, samp, in.uv);
  let l  = textureSample(src, samp, in.uv - ex);
  let r  = textureSample(src, samp, in.uv + ex);
  let u  = textureSample(src, samp, in.uv - ey);
  let d  = textureSample(src, samp, in.uv + ey);
  let lu = textureSample(src, samp, in.uv - ex - ey);
  let ru = textureSample(src, samp, in.uv + ex - ey);
  let ld = textureSample(src, samp, in.uv - ex + ey);
  let rd = textureSample(src, samp, in.uv + ex + ey);

  let k = max(B.q.z, 1e-4);          // field units -> height units, per texel
  let zx  = (r.r - l.r) * 0.5 * k;
  let zy  = (d.r - u.r) * 0.5 * k;
  let zxx = (r.r - 2.0 * c.r + l.r) * k;
  let zyy = (d.r - 2.0 * c.r + u.r) * k;
  let zxy = (rd.r - ld.r - ru.r + lu.r) * 0.25 * k;
  let g   = 1.0 + zx * zx + zy * zy;
  let H   = ((1.0 + zx * zx) * zyy - 2.0 * zx * zy * zxy + (1.0 + zy * zy) * zxx)
          / max(pow(g, 1.5), 1e-4);

  var o = c;
  o.r = c.r + B.q.w * H / k;
  // The auxiliary channels get a plain 5-tap so material/speed track the height.
  // ALPHA IS THE MATERIAL-ID MOMENT and must be averaged like the others: the
  // old 'o.a = o.r' was harmless when alpha duplicated coverage and would now
  // silently rewrite every pixel's material to WATER in curvature-flow mode.
  let aux = (c + l + r + u + d) * 0.2;
  o.g = aux.g;
  o.b = aux.b;
  o.a = aux.a;
  return o;
}

// Coverage mask -> thickness seed. A non-water material seeds a larger
// thickness than water at the same coverage, which is most of what makes goo
// read as a thicker liquid; per-material fine tuning is thicknessGain, applied
// in the composite where the material is actually known.
@fragment
fn fs_mask(in : VOut) -> @location(0) vec4<f32> {
  let c = textureSample(src, samp, in.uv);
  let a = smoothstep(B.q.x - B.q.y, B.q.x + B.q.y, c.r);
  let nw = clamp(c.g / max(c.r, 1e-4), 0.0, 1.0);
  return vec4<f32>(a * (1.0 + 0.7 * nw), nw * a, 0.0, 1.0);
}

// Wide separable Gaussian for thickness. Thickness in a 2D sim is "how much
// liquid is between the eye and the background", which for a face-on slab is a
// low-frequency function of how BIG the body of liquid is here — hence a very
// wide blur of the coverage mask, not the coverage mask itself.
@fragment
fn fs_wide(in : VOut) -> @location(0) vec4<f32> {
  let dir = B.p.xy;
  let sigma = max(0.35, B.p.z);
  let inv2s2 = 1.0 / (2.0 * sigma * sigma);
  var sum = vec4<f32>(0.0);
  var wsum = 0.0;
  for (var i : i32 = -6; i <= 6; i = i + 1) {
    let fi = f32(i);
    let w = exp(-fi * fi * inv2s2);
    sum = sum + textureSample(src, samp, in.uv + dir * fi) * w;
    wsum = wsum + w;
  }
  return sum / max(wsum, 1e-6);
}

// Straight copy — used to lay the background down under the 'points' renderer.
@fragment
fn fs_blit(in : VOut) -> @location(0) vec4<f32> {
  return vec4<f32>(textureSample(src, samp, in.uv).rgb, 1.0);
}
`;

// ---------------------------------------------------------------------------
// 4. COMPOSITE
// ---------------------------------------------------------------------------
const COMPOSITE_WGSL = /* wgsl */`
struct Mat {
  absorb  : vec4<f32>,   // rgb extinction, a = absorption scale
  scatter : vec4<f32>,   // rgb in-scatter colour, a = scatter gain
  optics  : vec4<f32>,   // F0, specPower, specGain, refractScale
  extra   : vec4<f32>,   // rimGain, rimPower, thicknessGain, envGain
  extra2  : vec4<f32>,   // specBroad, refractBase, gasOpacity, emissiveGain
};
struct CU {
  texel : vec4<f32>,     // 1/fieldW, 1/fieldH, 1/thickW, 1/thickH
  surf  : vec4<f32>,     // threshold, softness, normalScale, interiorGain
  light : vec4<f32>,     // Lx, Ly, Lz, chromatic dispersion
  misc  : vec4<f32>,     // debugMode, exposure, foamGain, height-field debug scale
  cls   : vec4<f32>,     // solidHard, gasThreshold, gasSoftness, hybrid on/off
  glow  : vec4<f32>,     // haloGain, haloWhitePoint, solidEdge, solidTopLight
  glow2 : vec4<f32>,     // emissiveLift, coreWhitePoint, haloInsideCut, -
  // Slot 0 is WATER and slot 1 is GOO — the two hand-tuned looks. Every other
  // slot is filled by the host with a procedurally derived optic so a material
  // id this renderer has never heard of still lands on something lit, coloured
  // and unlike its neighbours. See MAT_SLOTS in this file.
  mats  : array<Mat, 32>,
};

@group(0) @binding(0) var<uniform> C : CU;
@group(0) @binding(1) var fieldTex : texture_2d<f32>;
@group(0) @binding(2) var thickTex : texture_2d<f32>;
@group(0) @binding(3) var bgTex    : texture_2d<f32>;
@group(0) @binding(4) var samp     : sampler;
@group(0) @binding(5) var matTex   : texture_2d<f32>;
@group(0) @binding(6) var clsTex   : texture_2d<f32>;
@group(0) @binding(7) var glowTex  : texture_2d<f32>;

struct VOut { @builtin(position) clip : vec4<f32>, @location(0) uv : vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) vi : u32) -> VOut {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var o : VOut;
  let q = p[vi];
  o.clip = vec4<f32>(q, 0.0, 1.0);
  o.uv = vec2<f32>((q.x + 1.0) * 0.5, 1.0 - (q.y + 1.0) * 0.5);
  return o;
}

// Surface elevation from the smoothed field. Van der Laan smooths an eye-space
// DEPTH buffer; a 2D sim viewed face-on has no depth, so the surface is
// reconstructed from coverage instead. sqrt() is the whole trick: the slope goes
// to infinity at the silhouette, which is exactly the steep glassy meniscus that
// makes flat 2D liquid read as 3D. interiorGain adds a small LINEAR response to
// the field inside the body, so interior structure is visible at all — with it
// at 0 the interior is flat by construction.
fn heightAt(uv : vec2<f32>) -> f32 {
  let f = textureSampleLevel(fieldTex, samp, uv, 0.0).r;
  let a = smoothstep(C.surf.x - C.surf.y, C.surf.x + C.surf.y, f);
  return sqrt(clamp(a, 0.0, 1.0)) + C.surf.w * max(f - C.surf.x, 0.0);
}

fn envColor(r : vec3<f32>) -> vec3<f32> {
  let t = clamp(r.y * 0.5 + 0.5, 0.0, 1.0);
  let horizon = vec3<f32>(0.30, 0.36, 0.45);
  let zenith  = vec3<f32>(0.64, 0.76, 0.94);
  var c = mix(horizon, zenith, t * t);
  let L = normalize(C.light.xyz);
  let s = max(dot(normalize(r), L), 0.0);
  c = c + vec3<f32>(1.00, 0.95, 0.86) * pow(s, 90.0) * 1.4;
  c = c + vec3<f32>(0.35, 0.42, 0.55) * pow(s, 6.0) * 0.25;
  return c;
}

@fragment
fn fs(in : VOut) -> @location(0) vec4<f32> {
  let bgDirect = textureSampleLevel(bgTex, samp, in.uv, 0.0).rgb;
  let c = textureSampleLevel(fieldTex, samp, in.uv, 0.0);
  let f = c.r;
  let a = smoothstep(C.surf.x - C.surf.y, C.surf.x + C.surf.y, f);

  let mode = i32(C.misc.x + 0.5);
  if (mode == 1) {                                   // raw/smoothed height field
    // Linear and unclipped in the interior: this view is MEASURED, so a display
    // curve that saturates would silently erase the signal being measured.
    let v = clamp(f * C.misc.w, 0.0, 1.0);
    return vec4<f32>(v, v, v, 1.0);
  }

  // ---- the class field: how much of this pixel is which KIND of matter -----
  // hybrid == 0 collapses every one of these to "all liquid", which reproduces
  // the pre-hybrid renderer exactly — that is the A/B control, and it is what
  // makes the transparency measurement independent of this whole section.
  let hybrid = C.cls.w;
  let mt = textureSampleLevel(matTex, samp, in.uv, 0.0);
  let cl = textureSampleLevel(clsTex, samp, in.uv, 0.0);
  let wB = mt.a;
  var fS = 0.0; var fE = 0.0; var fG = 0.0; var fL = 1.0;
  var hot = 0.0;
  var albedo = vec3<f32>(0.5, 0.5, 0.5);
  if (hybrid > 0.5 && wB > 1.0e-4) {
    let inv = 1.0 / wB;
    fS = clamp(cl.r * inv, 0.0, 1.0);
    fE = clamp(cl.g * inv, 0.0, 1.0);
    fG = clamp(cl.b * inv, 0.0, 1.0);
    fL = clamp(1.0 - fS - fE - fG, 0.0, 1.0);
    // A solid under liquid is an interface between two materials, not a free
    // liquid silhouette. Preserve the amount assigned to those two classes but
    // steepen their split so a submerged edge does not inherit the broad,
    // glassy meniscus treatment. Gas and emissive weights are untouched.
    let sl = fS + fL;
    if (sl > 1.0e-4) {
      let solidShare = smoothstep(0.18, 0.82, fS / sl);
      fS = sl * solidShare;
      fL = sl * (1.0 - solidShare);
    }
    hot = clamp(cl.a * inv, 0.0, 1.0);
    albedo = mt.rgb / max(cl.r + cl.g + cl.b, 1.0e-4);
  }

  // The halo. A wide blur of the glow mass, added over EVERYTHING including the
  // bare background, so fire spills light onto what is next to it — the one
  // thing flat colour provably cannot do and the reason fire is its own class.
  let gw = textureSampleLevel(glowTex, samp, in.uv, 0.0).a * hybrid;
  let glowT = clamp(gw * C.glow.y, 0.0, 1.0);
  let glowRGB = mix(vec3<f32>(1.00, 0.28, 0.04), vec3<f32>(1.00, 0.94, 0.80), glowT)
              * clamp(gw * C.glow.x, 0.0, 2.5);

  // Gas has its own, lower coverage threshold: a wisp of smoke is thinner than
  // the liquid surface threshold and would otherwise not exist at all.
  let aGasCov = smoothstep(C.cls.y - C.cls.z, C.cls.y + C.cls.z, f);

  if (mode == 0 && a <= 0.0015 && fG * aGasCov <= 0.002) {
    return vec4<f32>(bgDirect + glowRGB, 1.0);
  }

  // ---- normal from central differences of the smoothed height --------------
  let ex = vec2<f32>(C.texel.x, 0.0);
  let ey = vec2<f32>(0.0, C.texel.y);
  let zx = (heightAt(in.uv + ex) - heightAt(in.uv - ex)) * 0.5;
  let zy = (heightAt(in.uv + ey) - heightAt(in.uv - ey)) * 0.5;
  // uv.y runs DOWN the screen; n.y is expressed with y UP, hence the sign.
  let n = normalize(vec3<f32>(-zx * C.surf.z, zy * C.surf.z, 1.0));
  let V = vec3<f32>(0.0, 0.0, 1.0);

  if (mode == 2) { return vec4<f32>(n * 0.5 + 0.5, 1.0); }

  // ---- material blend ------------------------------------------------------
  // 'nw' is the fraction of this pixel that is NOT water; 'idq' is the mean id
  // among those non-water contributors, rounded. For the WATER/GOO pair this is
  // exactly the old 'goo' blend. For an id nobody told this renderer about it
  // picks that id's slot, and ids beyond the table fold back into 2..23 rather
  // than collapsing onto WATER — an unknown material must never render as
  // nothing.
  let nw  = clamp(c.g / max(f, 1e-4), 0.0, 1.0);
  let spd = clamp(c.b / max(f, 1e-4), 0.0, 1.0);
  var idx = i32(round(c.a / max(c.g, 1e-4)));
  if (idx >= 32) { idx = 2 + (idx - 32) % 30; }
  idx = clamp(idx, 0, 31);
  let mW = C.mats[0];
  let mX = C.mats[idx];
  let absorb   = mix(mW.absorb,  mX.absorb,  nw);
  let scatter  = mix(mW.scatter, mX.scatter, nw);
  let optics   = mix(mW.optics,  mX.optics,  nw);
  let extra    = mix(mW.extra,   mX.extra,   nw);
  let extra2   = mix(mW.extra2,  mX.extra2,  nw);

  if (mode == 5) {                                   // material-id readout
    let h = fract(f32(idx) * 0.6180339887);
    let k = abs(fract(vec3<f32>(h) + vec3<f32>(1.0, 0.6666667, 0.3333333)) * 6.0 - vec3<f32>(3.0));
    return vec4<f32>(clamp(k - vec3<f32>(1.0), vec3<f32>(0.0), vec3<f32>(1.0)) * a, 1.0);
  }
  if (mode == 6) {                                   // class readout
    // red SOLID, green EMISSIVE, blue GAS, grey LIQUID — one look tells you
    // whether the class field agrees with what the sim actually contains.
    return vec4<f32>(fS + fL * 0.35, fE + fL * 0.35, fG + fL * 0.35, 1.0) * a;
  }
  if (mode == 7) { return vec4<f32>(albedo * a, 1.0); }   // flat albedo readout

  let tRaw = textureSampleLevel(thickTex, samp, in.uv, 0.0).r;
  let T = clamp(tRaw * extra.z, 0.0, 4.0);
  if (mode == 3) { let v = clamp(T * 0.5, 0.0, 1.0); return vec4<f32>(v, v * 0.85, v * 0.6, 1.0); }

  // ---- refraction ----------------------------------------------------------
  // Offset the background sample along the surface normal, scaled by thickness:
  // a thin film bends a little, a deep body bends a lot. This is the term that
  // makes the fluid read as GLASS rather than as a coloured overlay.
  // refractBase (extra2.y) is the floor of that ramp — how much a FILM bends.
  // Raising it is most of what makes thin water legible as water once the body
  // colour has been taken away.
  let rb = extra2.y;
  let bend = optics.w * (rb + (1.0 - rb) * clamp(T, 0.0, 1.0));
  let off = vec2<f32>(n.x, -n.y) * bend;
  let rUv = clamp(in.uv + off, vec2<f32>(0.002), vec2<f32>(0.998));
  // Chromatic dispersion: blue bends further than red. Three background taps
  // instead of one, and the single cheapest cue that says GLASS.
  let disp = C.light.w;
  let rUvR = clamp(in.uv + off * (1.0 - disp), vec2<f32>(0.002), vec2<f32>(0.998));
  let rUvB = clamp(in.uv + off * (1.0 + disp), vec2<f32>(0.002), vec2<f32>(0.998));
  let refr = vec3<f32>(
    textureSampleLevel(bgTex, samp, rUvR, 0.0).r,
    textureSampleLevel(bgTex, samp, rUv,  0.0).g,
    textureSampleLevel(bgTex, samp, rUvB, 0.0).b);

  // ---- Beer-Lambert --------------------------------------------------------
  // Transmission is spectral (that is the whole point: red dies first, so deep
  // water goes cyan). The IN-SCATTER term deliberately is NOT weighted by the
  // per-channel (1 - trans): doing that ties the scatter hue to the absorption
  // spectrum and turns clean blue water muddy green. It is driven by a scalar
  // optical depth instead, so the artist keeps control of the body colour.
  let trans = exp(-absorb.rgb * (T * absorb.w));
  let am = (absorb.r + absorb.g + absorb.b) * 0.3333333;
  let sf = 1.0 - exp(-am * T * absorb.w);
  var body = refr * trans + scatter.rgb * (scatter.w * sf);

  // ---- Fresnel reflection --------------------------------------------------
  let cosT = clamp(dot(n, V), 0.0, 1.0);
  let F0 = optics.x;
  let fres = clamp(F0 + (1.0 - F0) * pow(1.0 - cosT, 5.0), 0.0, 1.0);
  let refl = envColor(reflect(-V, n)) * extra.w;

  var col = mix(body, refl, fres);

  // ---- specular: a tight lobe for the glint, a broad one for the sheen -----
  // The BROAD lobe is the expensive one perceptually: at the interior normal
  // (n ~ +z) nh is already ~0.92, so a broad exponent lights the entire body
  // uniformly and reads as haze laid over the liquid rather than as a highlight
  // on it. specBroad (extra2.x) is its weight, per material, and it is the
  // difference between a wet glint and a chrome stripe.
  let L = normalize(C.light.xyz);
  let H = normalize(L + V);
  let nh = max(dot(n, H), 0.0);
  let spec = pow(nh, optics.y) * optics.z
           + pow(nh, max(optics.y * 0.12, 2.0)) * optics.z * extra2.x;
  col = col + vec3<f32>(1.0, 0.98, 0.94) * spec;

  // ---- rim: steep surface at the silhouette catches the sky ----------------
  let rim = pow(clamp(1.0 - n.z, 0.0, 1.0), extra.y) * extra.x;
  col = col + vec3<f32>(0.62, 0.80, 1.00) * rim;

  // ---- whitewater: fast + thin ---------------------------------------------
  let foam = clamp(spd * 2.2 - 0.35, 0.0, 1.0) * (1.0 - clamp(T * 0.9, 0.0, 1.0)) * C.misc.z;
  col = mix(col, vec3<f32>(0.92, 0.96, 1.0), foam * 0.55);

  col = col * C.misc.y;
  if (mode == 4) { return vec4<f32>(a, a, a, 1.0); }

  // ==========================================================================
  // HYBRID: four shading models, weighted by what is actually in this pixel.
  //
  // 'col' above is the LIQUID model and it is untouched — that is the point.
  // Water stays glass; everything else stops pretending to be glass.
  // ==========================================================================
  if (hybrid < 0.5 || fL > 0.998) {
    return vec4<f32>(mix(bgDirect, col, a) + glowRGB * (1.0 - C.glow2.z * a), 1.0);
  }

  // ---- SOLID: flat graphic colour, hard edge ------------------------------
  // A quarter of the liquid softness, so the silhouette lands inside ~1-2 px
  // instead of the 6-8 px meniscus that makes a liquid look wet. No refraction,
  // no Fresnel, no environment: a solid is a SHAPE, and the eye reads shape
  // fastest when nothing is happening inside it.
  let aSolid = smoothstep(C.surf.x - C.surf.y * C.cls.x, C.surf.x + C.surf.y * C.cls.x, f);
  // Two departures from perfectly flat, both deliberate and both small:
  //   * the silhouette is darkened, which separates two solids of similar
  //     colour that are touching (sand on stone) without a lighting model;
  //   * upward-facing surfaces are lifted a few percent, which is the entire
  //     "graphic illustration" trick for reading a pile as a pile.
  let edgeF = pow(clamp(1.0 - n.z, 0.0, 1.0), 1.5);
  var colSolid = albedo * ((1.0 - C.glow.z * edgeF) + C.glow.w * clamp(n.y, 0.0, 1.0));

  // ---- EMISSIVE: it is its own light source -------------------------------
  // A hot body's colour comes from its TEMPERATURE, not from what it is made
  // of: dull red, then orange, then white. That is why a metal bar heated in a
  // fire reads as hot metal and not as bright grey metal — which is exactly
  // what it did when this ramp was a mix toward white.
  //
  // The material's own colour survives at the bottom of the ramp, so a barely
  // warm solid is still its own colour and only committed heat changes it.
  // The whole thing is then lifted well above 1.0 so the core clips on an
  // 8-bit canvas; that clipped core plus the halo IS what "glowing" looks like.
  let hotC = clamp(hot * C.glow2.y, 0.0, 1.0);
  let ember = mix(vec3<f32>(0.85, 0.07, 0.01), vec3<f32>(1.00, 0.34, 0.03),
                  smoothstep(0.0, 0.55, hotC));
  let blackbody = mix(ember, vec3<f32>(1.00, 0.92, 0.76), smoothstep(0.80, 1.10, hotC));
  let colEmis = mix(albedo, blackbody, smoothstep(0.0, 0.30, hotC))
              * (1.0 + C.glow2.x * extra2.w * hot);

  // A SOLID that gets hot crosses into the emissive model on its own. Nothing
  // in this file knows that a metal bar is lying in a fire; it reads
  // Particle.temp, which the solver conducted along the bar.
  colSolid = mix(colSolid, colEmis, smoothstep(0.02, 0.55, hot));

  // ---- GAS: soft, translucent, no surface ---------------------------------
  // Gas gets no normal-based shading at all; a puff of smoke with a specular
  // highlight on it looks like a balloon. Only a mild thickness ramp, so a
  // dense column reads denser than a wisp.
  let gasT = clamp(f * 0.55, 0.0, 1.0);
  let colGas = albedo * (0.72 + 0.55 * gasT);
  let aGas = aGasCov * extra2.z;

  // ---- resolve ------------------------------------------------------------
  let wL = fL * a;
  let wS = fS * aSolid;
  let wE = fE * a;
  let wG = fG * aGas;
  let wSum = wL + wS + wE + wG;
  if (wSum <= 1.0e-4) { return vec4<f32>(bgDirect + glowRGB, 1.0); }
  let mixed = (col * wL + colSolid * wS + colEmis * wE + colGas * wG) / wSum;
  let alpha = clamp(wSum, 0.0, 1.0);
  // The halo is light LEAVING the body, so most of it is cut where the body
  // itself already is. Without this the halo lands on top of an emissive body
  // that is already clipping and washes lava from orange to white — measured:
  // the lava swatch went (201, 190, 96) with the halo full-strength inside.
  return vec4<f32>(mix(bgDirect, mixed, alpha) + glowRGB * (1.0 - C.glow2.z * alpha), 1.0);
}
`;

// ---------------------------------------------------------------------------
// The metaball resolve. Mathematics unchanged from the original renderer; the
// only edit is that it composites over bgTex instead of over a cleared colour,
// so the A/B toggle compares rendering technique and not backgrounds.
// ---------------------------------------------------------------------------
const RESOLVE_WGSL = /* wgsl */`
struct RS { texel : vec2<f32>, threshold : f32, softness : f32 };
@group(0) @binding(0) var<uniform> S : RS;
@group(0) @binding(1) var fieldTex : texture_2d<f32>;
@group(0) @binding(2) var samp : sampler;
@group(0) @binding(3) var bgTex : texture_2d<f32>;

struct VOut { @builtin(position) clip : vec4<f32>, @location(0) uv : vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) vi : u32) -> VOut {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var o : VOut;
  let q = p[vi];
  o.clip = vec4<f32>(q, 0.0, 1.0);
  o.uv = vec2<f32>((q.x + 1.0) * 0.5, 1.0 - (q.y + 1.0) * 0.5);
  return o;
}

@fragment
fn fs(in : VOut) -> @location(0) vec4<f32> {
  let bg = textureSampleLevel(bgTex, samp, in.uv, 0.0).rgb;
  let c = textureSampleLevel(fieldTex, samp, in.uv, 0.0);
  let f = c.a;
  let a = smoothstep(S.threshold - S.softness, S.threshold + S.softness, f);
  if (a <= 0.002) { return vec4<f32>(bg, 1.0); }

  let fx = textureSampleLevel(fieldTex, samp, in.uv + vec2<f32>(S.texel.x, 0.0), 0.0).a
         - textureSampleLevel(fieldTex, samp, in.uv - vec2<f32>(S.texel.x, 0.0), 0.0).a;
  let fy = textureSampleLevel(fieldTex, samp, in.uv + vec2<f32>(0.0, S.texel.y), 0.0).a
         - textureSampleLevel(fieldTex, samp, in.uv - vec2<f32>(0.0, S.texel.y), 0.0).a;
  let n = normalize(vec3<f32>(-fx * 1.8, fy * 1.8, 1.0));
  let l = normalize(vec3<f32>(-0.45, 0.55, 0.72));

  let base = c.rgb / max(f, 1e-4);
  let diff = 0.42 + 0.58 * max(0.0, dot(n, l));
  let spec = pow(max(0.0, dot(reflect(-l, n), vec3<f32>(0.0, 0.0, 1.0))), 22.0);
  let rim  = pow(1.0 - clamp(a, 0.0, 1.0), 2.0) * 0.30;
  let col  = base * diff + vec3<f32>(spec * 0.40, spec * 0.44, spec * 0.50) + base * rim;
  return vec4<f32>(mix(bg, col, a), 1.0);
}

// POST-LANDING PATCH: FLAT — the same metaball threshold, NO shading. This is
// OE-CAKE's own blob look: inside the isocontour a pixel simply IS the local
// field colour, with no normal, no light, no specular and no rim. The
// threshold mathematics above is reused untouched so Flat and Metaball always
// agree about WHERE the surface is and differ only in what a surface pixel
// wears.
@fragment
fn fs_flat(in : VOut) -> @location(0) vec4<f32> {
  let bg = textureSampleLevel(bgTex, samp, in.uv, 0.0).rgb;
  let c = textureSampleLevel(fieldTex, samp, in.uv, 0.0);
  let a = smoothstep(S.threshold - S.softness, S.threshold + S.softness, c.a);
  if (a <= 0.002) { return vec4<f32>(bg, 1.0); }
  let base = c.rgb / max(c.a, 1e-4);
  return vec4<f32>(mix(bg, base, a), 1.0);
}
`;

// ---------------------------------------------------------------------------
// Material optics.
//
// EVERY constant below is a RENDERER constant. Not one of them is a simulation
// parameter, and the §6.2 material table was not read, not re-tuned and not
// touched by this file. These were hand-tuned against contact sheets; they are
// exactly the kind of hand-tuned polish PROTECTED.md exists to guard.
// ---------------------------------------------------------------------------

// The look as it stood at commit 257b8a4, kept EXACTLY, so the transparency
// change is falsifiable: test/render/glass.html renders one sim state through
// both tables in the same frame and measures the difference. `specBroad` and
// `refractBase` are the two constants that used to be hard-coded in the shader
// (0.16 and 0.25); with these values the shader reproduces the old image.
export const PRESETS_LEGACY = {
  water: {
    absorb: [1.05, 0.38, 0.17], absorbScale: 0.85,
    scatter: [0.10, 0.45, 0.62], scatterGain: 0.20,
    F0: 0.020, specPower: 90, specGain: 2.20, refract: 0.100,
    rimGain: 0.90, rimPower: 2.2, thicknessGain: 1.55, envGain: 1.15,
    specBroad: 0.16, refractBase: 0.25,
  },
  goo: {
    absorb: [0.40, 1.30, 2.90], absorbScale: 2.20,
    scatter: [0.95, 0.66, 0.22], scatterGain: 0.62,
    F0: 0.045, specPower: 20, specGain: 0.55, refract: 0.022,
    rimGain: 0.40, rimPower: 1.7, thicknessGain: 2.00, envGain: 0.55,
    specBroad: 0.16, refractBase: 0.25,
  },
};

export const PRESETS = {
  water: {
    // GLASS WATER (2026-07-26). The brief was one sentence — "the water ought to
    // be more transparent in transparent mode" — and the previous table read as
    // a swimming pool: measured over a 25 wu column it kept only 46 % of the
    // background's local contrast and pulled the red channel from 198 to 97.
    //
    // Four changes, in the order they matter:
    //  1. extinction down 2.0x (absorb x absorbScale: 0.892 -> 0.229 in red)
    //     while thicknessGain goes UP 1.55 -> 2.70. Net: a thin film is very
    //     nearly air, and the depth at which water starts to tint moves out to
    //     where a body is actually deep. Absorption per unit thickness is what
    //     makes water read as water; the old table simply had too much of it
    //     and too little dynamic range.
    //  2. in-scatter gain 0.20 -> 0.075. In-scatter is the milky term; it is
    //     what stops you seeing through, and it does not vanish with depth.
    //  3. refraction 0.100 -> 0.140 with refractBase 0.25 -> 0.50, so the
    //     surface is now carried by DISPLACEMENT of the background rather than
    //     by body colour. Refraction got MORE legible, not less.
    //  4. specular: gain 2.20 -> 0.80, broad-lobe weight 0.16 -> 0.045, power
    //     90 -> 130. The old broad lobe added ~36/255 of white to every
    //     interior pixel — a chrome stripe wearing a highlight's clothes.
    // rimGain is up slightly (0.90 -> 1.05): with the body colour gone, the
    // silhouette is where the surface has to be read.
    absorb: [0.62, 0.225, 0.101], absorbScale: 0.37,
    scatter: [0.16, 0.50, 0.66], scatterGain: 0.075,
    F0: 0.020, specPower: 130, specGain: 0.80, refract: 0.140,
    rimGain: 1.05, rimPower: 2.2, thicknessGain: 2.70, envGain: 1.15,
    specBroad: 0.045, refractBase: 0.50,
  },
  goo: {
    // Slot 1 is GEL. The historical renderer key remains `goo` so existing
    // render integrations keep working, but no user-facing surface calls it
    // Goo. Gel inherits Goo's thermal isolation in engine.js and the former
    // Orbies visual/mechanical character: clear, dense and satisfyingly soft.
    absorb: [0.16, 0.08, 0.035], absorbScale: 0.20,
    scatter: [0.30, 0.78, 1.00], scatterGain: 0.055,
    F0: 0.075, specPower: 180, specGain: 1.40, refract: 0.205,
    rimGain: 1.65, rimPower: 1.45, thicknessGain: 1.05, envGain: 1.22,
    specBroad: 0.035, refractBase: 0.62,
  },
};

// ===========================================================================
// THE HYBRID DIRECTION — one shading model per material CLASS.
//
// Darien's call, 2026-07-26, verbatim: glassy fluids, flat solids, emissive
// fire and lava, soft translucent smoke and steam. The reasoning he accepted
// was that eighteen materials all rendered as refractive glass blur into mush;
// each material CLASS has to read instantly, and that costs art direction per
// class. This table is that art direction.
//
//   LIQUID    the screen-space glass path — refraction, Beer-Lambert, Fresnel,
//             rim. Water, goo, oil. The look this renderer was built for.
//   SOLID     flat graphic colour, hard silhouette, no refraction, no
//             environment. Ice, wood, ash, stone, sand, metal, an imported
//             photo. Reads as a SHAPE at a glance, which is the whole job.
//   EMISSIVE  its own light source: a body that ramps to white with
//             temperature and a wide additive halo that spills onto whatever
//             is next to it. Fire, lava, molten metal — and any SOLID that
//             gets hot enough, which is how a metal bar carrying a flame's
//             heat lights up without being a different material.
//   GAS       soft, translucent, no surface shading at all. Steam, smoke, the
//             flammable gas.
//
// `albedo` is a display-space colour, 0..1, and is what a SOLID or GAS simply
// IS. The hue of each one is taken from the material row's `tint` hint in
// engine.js where that hint reads well on a dark field, and lifted where it
// did not (ash and stone are both a few percent brighter than their tint: at
// tint value they sat within 12/255 of the dark background).
//
// `glowFloor` 1.0 means "alight regardless of temperature" and is FIRE only.
// Everything else glows because the SIMULATION says it is hot, so lava that
// cools toward stone dims on its own and nothing in this file has to know.
// ===========================================================================
// INVISIBLE (4) is not a look, it is the ABSENCE of one — §17 retired matter.
// It is culled in the VERTEX shader (the quad is pushed behind the far plane,
// exactly the trick `ii >= S.count` already used), so it costs no fragment work
// at all and it contributes to neither the coverage field nor the class field.
// Every other class is a shading style; this one is "there is nothing here".
export const CLASS = { LIQUID: 0, SOLID: 1, EMISSIVE: 2, GAS: 3, INVISIBLE: 4 };
export const CLASS_NAMES = ['LIQUID', 'SOLID', 'EMISSIVE', 'GAS', 'INVISIBLE'];

/** Degrees C at which a hot body starts / finishes glowing. */
export const GLOW_LO = 450;
export const GLOW_HI = 1300;

const rgb = (hex) => [((hex >> 16) & 255) / 255, ((hex >> 8) & 255) / 255, (hex & 255) / 255];

/**
 * Per-material look, indexed by engine.js MAT_* ids. Ids past the end fall back
 * to proceduralMaterial(), which is SOLID and therefore always visible.
 *
 * `freezeDraw` (post-landing patch, Darien 2026-07-27) — 1 = TIME STOPS WHILE
 * YOU DRAW THIS ROW (index.html §20.7 reads it at `beginStroke`). It is an
 * IDENTITY flag, not derivable from the physics columns: RICE has bonds and
 * lays LIVE ("honestly fine with no-freeze" — that pourable-heap feel IS rice),
 * while RUBBER and PHOTO are not rigid and freeze ("the freeze-time-while-
 * drawing feature needs to apply to [elastic] as well"). It is UI-side material
 * METADATA in the MATERIAL_LOOKS pattern — not a sim `Mat` word, so it moves
 * neither `matsHash` nor any golden: the freeze is authoring-time behaviour,
 * captured naturally in the recorded input stream. Values as decided: every
 * SOLID_RIGID row = 1, RUBBER = 1, PHOTO = 1, RICE = 0, everything else 0.
 * MOCHI freezes the day it exists ("frozen until released"); a String row
 * ships with freezeDraw 1 the day §25 mints one (same message; the §25 String
 * bit implies freeze). Omitted means 0.
 */
export const MATERIAL_LOOKS = [
  { name: 'WATER',  cls: CLASS.LIQUID,   albedo: rgb(0x2a6fb0) },
  { name: 'GEL',    cls: CLASS.LIQUID,   albedo: rgb(0x8cecff),
    absorb: [0.16, 0.08, 0.035], absorbScale: 0.20,
    scatter: [0.30, 0.78, 1.00], scatterGain: 0.055,
    F0: 0.075, specPower: 180, specGain: 1.40, refract: 0.205,
    rimGain: 1.65, rimPower: 1.45, thicknessGain: 1.05, envGain: 1.22,
    specBroad: 0.035, refractBase: 0.62 },
  { name: 'ICE',    cls: CLASS.SOLID,    albedo: rgb(0xbfe7ff), freezeDraw: 1 },
  { name: 'STEAM',  cls: CLASS.GAS,      albedo: rgb(0xe6f1f8), gasOpacity: 0.40 },
  // OIL is the third liquid and it has to be TOLD APART from water at a
  // glance: heavy absorption in blue and green, so a film is amber and a body
  // is deep amber, a much glossier tight highlight, and half water's
  // refraction.
  // RUNG 2.7 item 6 — DARIEN'S TASTE CALL, RECORDED: "much more transparent"
  // (absorbScale 1.55 -> 0.55, scatterGain 0.40 -> 0.12, thickness ramp
  // 2.20 -> 2.00). A body used to go nearly black; now it goes deep amber
  // with the backdrop legible through it. The amber RATIOS and the tight
  // glossy highlight — oil's identity against water and nitro — are
  // untouched. His values, for the future PROTECTED.md.
  { name: 'OIL',    cls: CLASS.LIQUID,   albedo: rgb(0x6b4a1f),
    absorb: [1.05, 2.35, 3.70], absorbScale: 0.55,
    scatter: [0.62, 0.33, 0.07], scatterGain: 0.12,
    F0: 0.035, specPower: 55, specGain: 1.30, refract: 0.075,
    rimGain: 0.55, rimPower: 2.0, thicknessGain: 2.00, envGain: 0.80,
    specBroad: 0.09, refractBase: 0.35 },
  // FIRE's floor is 0.50, not 1.0: at 1.0 every flame pixel sits at the top of
  // the blackbody ramp and the whole fire is white. 0.50 keeps a cool flame
  // orange and lets the equilibrium temperature (~1000 C, so hot ~= 0.65) do
  // the rest, which is what makes one fire hotter-looking than another.
  { name: 'FIRE',   cls: CLASS.EMISSIVE, albedo: rgb(0xff8a2a), glowFloor: 0.62, emissiveGain: 1.9 },
  { name: 'SMOKE',  cls: CLASS.GAS,      albedo: rgb(0x4a4a54), gasOpacity: 0.80 },
  { name: 'WOOD',   cls: CLASS.SOLID,    albedo: rgb(0x96602f), freezeDraw: 1 },
  { name: 'ASH',    cls: CLASS.SOLID,    albedo: rgb(0x413f3c) },
  { name: 'LAVA',   cls: CLASS.EMISSIVE, albedo: rgb(0xff5a10), emissiveGain: 0.85 },
  { name: 'STONE',  cls: CLASS.SOLID,    albedo: rgb(0x6f7a8c), freezeDraw: 1 },
  { name: 'SAND',   cls: CLASS.SOLID,    albedo: rgb(0xdcc78d) },
  { name: 'METAL',  cls: CLASS.SOLID,    albedo: rgb(0x9aa7b8), freezeDraw: 1 },
  { name: 'MOLTEN', cls: CLASS.EMISSIVE, albedo: rgb(0xffd07a), emissiveGain: 1.10 },
  { name: 'GAS',    cls: CLASS.GAS,      albedo: rgb(0xa8ff9a), gasOpacity: 0.42 },
  // PHOTO is the imported picture. Near-white and flat, so the SHAPE that came
  // out of the image is what you see, and so the char marks read when it burns.
  { name: 'PHOTO',  cls: CLASS.SOLID,    albedo: rgb(0xf4eee2), freezeDraw: 1 },

  // ==========================================================================
  // §13 / §15 — the eight materials that had NO look and fell through to
  // proceduralMaterial(). Every one of them rendered as a FLAT SOLID in a
  // golden-ratio hue that had nothing to do with the material: tar was pale
  // yellow, mercury purple, acid dull red, cryo magenta, and BLAST — a
  // detonation — was a dark grey blob at luminance 74.6.
  //
  // That is not a small miss. It broke taste decision #1 in both directions
  // (four LIQUIDS shaded as opaque solids, the brightest event in the game
  // shaded as the dullest thing on screen), it disagreed with the picker, which
  // files CRYO/TAR/MERCURY/ACID under "liquid", and it collapsed render
  // distinctness: the closest opaque pair on the material sheet was 7.5 RGB
  // (FIRE vs BLAST) against 23.8 when the roster was 16.
  //
  // The engine row already carried the right colour in `tint`. These entries
  // are that hue, given a CLASS and the optics its class needs.
  // ==========================================================================

  // CRYO — the coldest thing in the box, and it must not be mistaken for water.
  // Same glass path, but almost no absorption (a cryogen is water-clear), a
  // hard tight highlight, and a strong pale-cyan rim: it reads as *colder* than
  // water rather than merely bluer. refract slightly above water's, because a
  // thin cold liquid boiling in a warm room should shimmer.
  { name: 'CRYO',   cls: CLASS.LIQUID,   albedo: rgb(0x7fd8ff),
    absorb: [0.62, 0.30, 0.14], absorbScale: 0.72,
    scatter: [0.42, 0.78, 1.00], scatterGain: 0.42,
    F0: 0.030, specPower: 70, specGain: 1.35, refract: 0.115,
    rimGain: 1.05, rimPower: 1.5, thicknessGain: 1.30, envGain: 0.95,
    specBroad: 0.07, refractBase: 0.30 },

  // RUBBER — flat, matte, near-black. The one thing it must never look like is
  // glass, because its whole identity is that it BOUNCES. Lifted well off its
  // 0x33333a tint: at tint value it sat within 10/255 of the night background
  // and a rubber ball was invisible until it moved.
  { name: 'RUBBER', cls: CLASS.SOLID,    albedo: rgb(0x53535e), freezeDraw: 1 },

  // TAR — the second black liquid, and it has to be told apart from OIL at a
  // glance. Oil is amber in a film; tar is brown-black in a film and dead black
  // in a body: absorption is 2.4x oil's and biased so the little that gets
  // through is red. It is also the glossiest surface in the roster (specPower
  // 90) — wet tar has a sharp, oily sheen that a matte solid cannot fake.
  { name: 'TAR',    cls: CLASS.LIQUID,   albedo: rgb(0x2b241d),
    absorb: [2.60, 5.20, 7.40], absorbScale: 2.60,
    scatter: [0.40, 0.24, 0.12], scatterGain: 0.16,
    F0: 0.048, specPower: 90, specGain: 1.60, refract: 0.045,
    rimGain: 0.40, rimPower: 2.4, thicknessGain: 3.00, envGain: 0.70,
    specBroad: 0.05, refractBase: 0.42 },

  // MERCURY — a liquid METAL, which is the one optic this renderer did not
  // already have. It is on the LIQUID path so it beads and flows like one, but
  // it is tuned to be OPAQUE and MIRRORED rather than transmissive: F0 0.62
  // (metallic Fresnel, ~15x water's), absorption high enough that nothing shows
  // through even a thin film, and envGain 1.8 so the background it reflects is
  // what you actually see. refract is near zero — you do not see THROUGH
  // mercury, you see the room IN it.
  { name: 'MERCURY',cls: CLASS.LIQUID,   albedo: rgb(0xc9ccd6),
    absorb: [7.00, 7.00, 7.00], absorbScale: 3.20,
    scatter: [0.86, 0.88, 0.94], scatterGain: 0.72,
    F0: 0.620, specPower: 120, specGain: 2.10, refract: 0.012,
    rimGain: 0.95, rimPower: 1.3, thicknessGain: 3.40, envGain: 1.80,
    specBroad: 0.04, refractBase: 0.55 },

  // DUST — a hanging cloud you can see through, not a solid. GAS, at the
  // heaviest opacity in the gas family: steam 0.40, gas 0.42, smoke 0.80,
  // dust 0.66. Warm grey-tan so it never reads as smoke.
  { name: 'DUST',   cls: CLASS.GAS,      albedo: rgb(0xc2b69c), gasOpacity: 0.66 },

  // ACID — the one material whose colour IS the warning. Transmits its own
  // green hard (absorption in red and blue, almost none in green) and
  // in-scatters green, so even a one-particle film is unmistakably
  // acid-green rather than a pale wash. Fizzy: a broad specular and a
  // strong rim.
  // RUNG 2.7 item 6 — DARIEN'S TASTE CALL, RECORDED: "much more transparent"
  // (absorbScale 1.35 -> 0.50, scatterGain 0.78 -> 0.25). A slab used to
  // read as opaque mustard; now it is green GLASS with the warning colour
  // carried by transmission — the red/blue-eating ratios, the fizzy broad
  // specular and the rim are untouched. His values, for the future
  // PROTECTED.md.
  { name: 'ACID',   cls: CLASS.LIQUID,   albedo: rgb(0x8dff3a),
    absorb: [2.30, 0.22, 2.90], absorbScale: 0.50,
    scatter: [0.42, 1.00, 0.20], scatterGain: 0.25,
    F0: 0.036, specPower: 34, specGain: 1.05, refract: 0.085,
    rimGain: 0.90, rimPower: 1.7, thicknessGain: 1.70, envGain: 0.85,
    specBroad: 0.20, refractBase: 0.28 },

  // NITRO — a charge has to look like a charge. Amber-yellow glass, dense and
  // syrupy (still the highest thicknessGain of the four), with a hot rim so a
  // settled pool of it glints in the dark and you can find it before you
  // light it.
  // RUNG 2.7 item 6 — DARIEN'S TASTE CALL, RECORDED: "much more transparent"
  // (absorbScale 1.60 -> 0.55, scatterGain 0.60 -> 0.15, thickness ramp
  // 2.60 -> 2.30). It used to bloom into an opaque yellow glare; now it is
  // yellow glass. The blue-eating ratios (yellow transmission), the warm
  // rim and the syrup ramp — nitro's identity against oil — are untouched.
  // His values, for the future PROTECTED.md.
  { name: 'NITRO',  cls: CLASS.LIQUID,   albedo: rgb(0xe8e030),
    absorb: [0.30, 0.70, 3.40], absorbScale: 0.55,
    scatter: [1.00, 0.88, 0.18], scatterGain: 0.15,
    F0: 0.042, specPower: 60, specGain: 1.30, refract: 0.070,
    rimGain: 0.80, rimPower: 1.8, thicknessGain: 2.30, envGain: 0.80,
    specBroad: 0.09, refractBase: 0.34 },

  // BLAST — THE BRIGHTEST THING IN THE GAME, and it was a dark grey blob.
  // glowFloor 1.0 (alight regardless of what the thermometer says — it is the
  // pressure wave itself, not something that got hot) and the largest
  // emissiveGain in the table by 1.4x, so a detonation blows out the frame and
  // spills light onto everything near it. This is the ONE material where the
  // halo is the point.
  { name: 'BLAST',  cls: CLASS.EMISSIVE, albedo: rgb(0xfff3c0), glowFloor: 1.0, emissiveGain: 2.70 },

  // VOID (§17) — retired matter. The only entry in this table that is not a
  // look. Everything else here answers "what does this material look like";
  // this one answers "it does not". The vertex shader culls it before it can
  // splat, so the albedo below is never sampled — it is black so that a future
  // code path which somehow reaches it fails toward invisible rather than
  // toward a magenta rectangle in the middle of the toy.
  { name: 'VOID',   cls: CLASS.INVISIBLE, albedo: rgb(0x000000) },

  // RICE (§18) — and this entry is a BUG FIX, not a new material's look.
  //
  // v8 added row 25 to the roster and did not add a look, so RICE fell straight
  // through to proceduralMaterial() and rendered at hue (25 * 0.618) mod 1 —
  // #15d49b, BRIGHT SPRING GREEN — while its own engine row declares
  // tint #eee4c6, cream. The newest material in the toy shipped looking like
  // radioactive slime, and it landed in the picker's "other" group as well,
  // because index.html's phaseOf() also reads MATERIAL_LOOKS.
  //
  // This is the SAME defect class that was closed for CRYO/RUBBER/TAR/MERCURY/
  // DUST/ACID/NITRO/BLAST, reopened by adding a row. Adding a material to
  // engine.js MATS could not fail any gate, because nothing compared the two
  // lists. It can now: test/run.mjs's `static` gate asserts that every live
  // (non-phantom) row has a hand-tuned entry here whose `name` matches the
  // engine's, and that entry's albedo is near its declared tint. A row added
  // without a look is a RED BUILD, not a green one with a green blob in it.
  //
  // The look itself: a flat, slightly warm off-white solid. Cooked rice is not
  // glossy and it is not translucent at this scale, so it takes the plain SOLID
  // path — the interesting thing about rice is its SHAPE (a pourable heap that
  // sets into one lump), and a busy optic would fight that rather than show it.
  { name: 'RICE',   cls: CLASS.SOLID,    albedo: rgb(0xeee4c6), freezeDraw: 0 },

  // BEDROCK (v15, §26.F) — the row nothing happens to, and the look has one
  // job: to be UNMISTAKABLE. It is the only material in the toy that survives
  // every experiment, so a player has to be able to see at a glance which parts
  // of a machine are the apparatus and which are the sample.
  //
  // The colour is the one hole left in a 26-row palette. MEASURED across every
  // shipped entry, max channel distance: this ALBEDO's nearest neighbour of any
  // class is STONE at 48, where the closest pair of SOLID looks already
  // shipping is PHOTO~RICE at 28. Its TINT's nearest is SMOKE at 42 (a
  // translucent gas) and then WATER at 53. It is also the only colour here that
  // nothing in the tank can turn INTO, which is the point: bedrock never
  // becomes anything else, so nothing ever arrives wearing it by accident.
  //
  // Lifted off its 0x4a3a8c tint the same way RUBBER's is, for the same reason
  // — at tint value a dark solid sits too near the night background to read as
  // an object until it moves. Flat SOLID optics: the interesting thing about
  // this material is that nothing happens to it, and a busy optic would be a
  // claim that something might.
  { name: 'BEDROCK', cls: CLASS.SOLID,   albedo: rgb(0x5b4aa8), freezeDraw: 1 },

  // ORBIES are rendered through the dedicated bead pass in Shader mode so a
  // volume remains a collection of colourful gumballs, not one metaball. This
  // fallback look covers the other render modes.
  { name: 'ORBIES', cls: CLASS.LIQUID,   albedo: rgb(0xff84e8),
    absorb: [0.08, 0.04, 0.02], absorbScale: 0.12,
    scatter: [1.00, 0.38, 0.85], scatterGain: 0.08,
    F0: 0.090, specPower: 190, specGain: 1.55, refract: 0.220,
    rimGain: 1.85, rimPower: 1.35, thicknessGain: 0.85, envGain: 1.30,
    specBroad: 0.025, refractBase: 0.66 },
];

// ---------------------------------------------------------------------------
// Unknown materials.
//
// Other people are adding materials to this engine and they cannot add looks,
// because this file is the only place a look can live. So the renderer refuses
// to have an opinion it cannot have: any material id without a hand-tuned entry
// gets an optic DERIVED FROM THE ID, and the derivation is chosen so that the
// result is always (a) visible against both backgrounds and (b) unlike its
// neighbours in the id sequence.
//
// Hue walks by the golden ratio, which is the standard way to get a sequence
// whose consecutive entries are maximally far apart on a circle; value and
// saturation then walk on two short coprime cycles so that ids whose hues
// happen to land close still separate on a second axis.
// ---------------------------------------------------------------------------
// The OPTICS table (the composite shader's `array<Mat, 32>`) — not the roster
// size. It deliberately does NOT grow for §17's VOID row: retired matter is
// culled in the vertex shader, so it never reaches the field, never reaches the
// composite, and has no optic to look up. `lookTable` (all 32 ids) is the one
// that has to cover every material id the SPLAT shader can see.
export const MAT_SLOTS = 32;

function hsv(h, s, v) {
  const f = (n) => {
    const k = (n + h * 6) % 6;
    return v - v * s * Math.max(0, Math.min(k, 4 - k, 1));
  };
  return [f(5), f(3), f(1)];
}

/** The look this renderer falls back to for a material id it was never told about. */
export function proceduralMaterial(id) {
  const i = id >>> 0;
  const h = (i * 0.6180339887) % 1;
  const s = 0.60 + 0.30 * (((i * 5) % 3) / 2);
  const v = 1.0 - 0.17 * ((i * 3) % 2);
  const rgb = hsv(h, s, v);
  return {
    // Absorb the COMPLEMENT of the hue, so the body transmits its own colour
    // instead of turning into a silhouette.
    absorb: rgb.map((c) => 0.34 + 1.55 * (1 - c)), absorbScale: 1.15,
    // In-scatter in the hue itself, with real gain: this is the term that
    // guarantees a floor on brightness. An unknown material can be dense, but
    // it can never be black.
    scatter: rgb.map((c) => 0.20 + 0.80 * c), scatterGain: 0.55,
    F0: 0.042, specPower: 40, specGain: 0.85, refract: 0.045,
    rimGain: 0.62, rimPower: 1.9, thicknessGain: 1.90, envGain: 0.70,
    specBroad: 0.16, refractBase: 0.30,
    // An id nobody hand-tuned renders as a FLAT SOLID in its own hue. That is
    // the safest possible default: a flat colour cannot be invisible, cannot be
    // black-on-black, and cannot be confused with the glass the liquids use.
    cls: 1, albedo: rgb,
    procedural: true, hue: +h.toFixed(4),
  };
}

/** The full look for a material id: hand-tuned where one exists, procedural otherwise. */
export function materialLook(id) {
  const i = id >>> 0;
  const base = proceduralMaterial(i);
  const hand = MATERIAL_LOOKS[i];
  return hand ? { ...base, ...hand } : base;
}

export const MAT_WORDS_RENDER = 20;   // floats per Mat in the uniform

function packMat(m, out, o) {
  out[o + 0] = m.absorb[0]; out[o + 1] = m.absorb[1]; out[o + 2] = m.absorb[2]; out[o + 3] = m.absorbScale;
  out[o + 4] = m.scatter[0]; out[o + 5] = m.scatter[1]; out[o + 6] = m.scatter[2]; out[o + 7] = m.scatterGain;
  out[o + 8] = m.F0; out[o + 9] = m.specPower; out[o + 10] = m.specGain; out[o + 11] = m.refract;
  out[o + 12] = m.rimGain; out[o + 13] = m.rimPower; out[o + 14] = m.thicknessGain; out[o + 15] = m.envGain;
  // Defaulted, so a preset written before these four existed still packs to the
  // shader's previous hard-coded behaviour instead of to zero.
  out[o + 16] = m.specBroad ?? 0.16; out[o + 17] = m.refractBase ?? 0.25;
  out[o + 18] = m.gasOpacity ?? 0.55; out[o + 19] = m.emissiveGain ?? 1.0;
}

async function makeModule(device, code, label) {
  const m = device.createShaderModule({ code, label });
  const ci = await m.getCompilationInfo();
  const bad = ci.messages.filter((x) => x.type !== 'info');
  if (bad.length) {
    throw new Error(`[render] shader "${label}":\n` +
      bad.map((x) => `  ${x.type} L${x.lineNum}:${x.linePos} ${x.message}`).join('\n'));
  }
  return m;
}

// ---------------------------------------------------------------------------
// Background.
//
// Refraction against a flat colour is invisible; refraction that warps legible
// TYPE is unmistakable. So the background is drawn on a 2D canvas — gradient,
// broad colour fields for the absorption to eat, a fine grid, and a wordmark — and
// uploaded once per resize. Nothing about it moves between frames, so every
// distortion visible in the fluid is refraction.
//
// 'night' (a DARK GRID) is the default. Darien's call, 2026-07-26: the light
// studio background became the default during the renderer work without his
// sign-off, and the toy is a dark-field toy. The constraint that came with the
// call is the interesting part — refraction and absorption must still READ
// against a dark field, which is exactly why 'studio' had been chosen — so the
// night palette below is not the old near-black one. It carries:
//   * a BRIGHT grid (the fine grid is the refraction read-out, and a bent
//     bright line on a dark field is more legible than a bent dark line on a
//     light one, not less),
//   * two calm, saturated colour fields, because Beer-Lambert absorption has nothing to
//     eat on a neutral field and deep water would tint invisibly,
//   * type, which is the only unarguable proof of refraction.
// 'studio' is kept and is one keypress away.
//
// 'image' is the user's own picture, cover-fitted. OE-CAKE let you do this and
// he wants it back. It is dimmed and gridded by default so a busy photograph
// does not fight the fluid; both are switchable.
// ---------------------------------------------------------------------------
export function drawBackground(w, h, opts = {}) {
  const cv = (typeof OffscreenCanvas !== 'undefined')
    ? new OffscreenCanvas(w, h)
    : Object.assign(document.createElement('canvas'), { width: w, height: h });
  const g = cv.getContext('2d');
  const s = h / 900;

  const style = opts.style ?? 'night';
  if (style === 'flat') {
    g.fillStyle = opts.flatColor || '#0a0d14';
    g.fillRect(0, 0, w, h);
    return cv;
  }

  // ---- the user's own picture ---------------------------------------------
  if (style === 'image' && opts.image) {
    const img = opts.image;
    const iw = img.width || img.videoWidth || w;
    const ih = img.height || img.videoHeight || h;
    g.fillStyle = '#05070c';
    g.fillRect(0, 0, w, h);
    // cover-fit: fill the frame, crop the overflow, never distort the picture.
    const sc = Math.max(w / iw, h / ih);
    const dw = iw * sc, dh = ih * sc;
    g.drawImage(img, (w - dw) * 0.5, (h - dh) * 0.5, dw, dh);
    const dim = opts.imageDim ?? 0.35;
    if (dim > 0) {
      g.fillStyle = `rgba(4, 6, 12, ${Math.min(1, dim)})`;
      g.fillRect(0, 0, w, h);
    }
    if (opts.imageGrid ?? true) {
      const fine = 26 * s;
      g.lineWidth = Math.max(1, 1 * s);
      g.strokeStyle = 'rgba(200, 225, 255, 0.10)';
      g.beginPath();
      for (let x = 0; x <= w; x += fine) { g.moveTo(x + 0.5, 0); g.lineTo(x + 0.5, h); }
      for (let y = 0; y <= h; y += fine) { g.moveTo(0, y + 0.5); g.lineTo(w, y + 0.5); }
      g.stroke();
    }
    return cv;
  }

  // Two full-bleed linear colour fields per backdrop. Unlike the old radial
  // swatches, these have no visible "ball" edge at any aspect ratio or rotation.
  // They still give water and glass something chromatic to refract and absorb.
  const P = style === 'studio'
    ? { top: '#eef2f7', mid: '#dde4ee', bot: '#c9d3e2',
        fine: 'rgba(60, 92, 140, 0.20)', heavy: 'rgba(40, 70, 115, 0.38)',
        type: 'rgba(30, 55, 95, 0.30)', sub: 'rgba(35, 65, 105, 0.45)',
        sweeps: [
          [[-0.08, 0.04, 1.06, 0.94], [[0, 'rgba(74, 177, 222, 0.34)'], [0.56, 'rgba(116, 180, 230, 0.12)'], [1, 'rgba(255,255,255,0)']]],
          [[0.12, 1.02, 0.94, -0.06], [[0, 'rgba(238, 177, 105, 0.26)'], [0.52, 'rgba(195, 161, 231, 0.12)'], [1, 'rgba(255,255,255,0)']]],
        ] }
    : style === 'grey'
      ? { top: '#3b414b', mid: '#2d323b', bot: '#20242b',
          fine: 'rgba(210, 220, 234, 0.19)', heavy: 'rgba(228, 235, 246, 0.34)',
          type: 'rgba(235, 241, 250, 0.31)', sub: 'rgba(222, 232, 246, 0.43)',
          sweeps: [
            [[-0.04, 0.12, 1.04, 0.88], [[0, 'rgba(93, 151, 184, 0.25)'], [0.58, 'rgba(100, 121, 157, 0.10)'], [1, 'rgba(0,0,0,0)']]],
            [[0.12, 1.04, 0.94, -0.04], [[0, 'rgba(177, 139, 106, 0.20)'], [0.54, 'rgba(139, 116, 157, 0.10)'], [1, 'rgba(0,0,0,0)']]],
          ] }
      // The dark field, re-lit. Base is darker than the old night palette
      // (#080b14 vs #0b1220) and everything ON it is brighter, so the CONTRAST
      // the refraction has to bend is higher in both directions.
      : { top: '#060914', mid: '#03050b', bot: '#010207',
          fine: 'rgba(150, 200, 255, 0.30)', heavy: 'rgba(190, 225, 255, 0.62)',
          type: 'rgba(205, 232, 255, 0.34)', sub: 'rgba(170, 215, 255, 0.52)',
          sweeps: [
            [[-0.10, 0.06, 1.08, 0.92], [[0, 'rgba(38, 110, 185, 0.075)'], [0.56, 'rgba(58, 95, 168, 0.025)'], [1, 'rgba(0,0,0,0)']]],
            [[0.08, 1.04, 0.98, -0.06], [[0, 'rgba(22, 138, 128, 0.045)'], [0.50, 'rgba(105, 70, 157, 0.038)'], [1, 'rgba(0,0,0,0)']]],
          ] };

  const grad = g.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, P.top);
  grad.addColorStop(0.55, P.mid);
  grad.addColorStop(1, P.bot);
  g.fillStyle = grad;
  g.fillRect(0, 0, w, h);

  // Fullscreen chroma fields — enough colour for absorption, with no local blobs.
  for (const [[x0, y0, x1, y1], stops] of P.sweeps) {
    const sweep = g.createLinearGradient(x0 * w, y0 * h, x1 * w, y1 * h);
    for (const [at, colour] of stops) sweep.addColorStop(at, colour);
    g.fillStyle = sweep;
    g.fillRect(0, 0, w, h);
  }

  // Fine grid + heavy grid. The fine grid is the refraction read-out.
  const fine = 26 * s, heavy = fine * 4;
  g.lineWidth = Math.max(1, 1 * s);
  g.strokeStyle = P.fine;
  g.beginPath();
  for (let x = 0; x <= w; x += fine) { g.moveTo(x + 0.5, 0); g.lineTo(x + 0.5, h); }
  for (let y = 0; y <= h; y += fine) { g.moveTo(0, y + 0.5); g.lineTo(w, y + 0.5); }
  g.stroke();
  g.lineWidth = Math.max(1, 1.6 * s);
  g.strokeStyle = P.heavy;
  g.beginPath();
  for (let x = 0; x <= w; x += heavy) { g.moveTo(x + 0.5, 0); g.lineTo(x + 0.5, h); }
  for (let y = 0; y <= h; y += heavy) { g.moveTo(0, y + 0.5); g.lineTo(w, y + 0.5); }
  g.stroke();

  // ---- THE CALIBRATION OVERLAY — OPT IN, AND OFF IN THE SHIPPED TOY ---------
  // A giant wordmark, two lines of copy addressed to a tester, and a numbered
  // ruler. Every one of those is a MEASUREMENT INSTRUMENT: warped letters are
  // the unarguable proof that the fluid refracts, and the ruler makes vertical
  // displacement readable by eye. They belong in test/render/*.html, which is
  // where they are now switched on.
  //
  // They were baked unconditionally into `night` — the background the toy boots
  // with — so the first thing a player saw was a tech demo with the debug view
  // left on. Darien asked for "dark grid by default"; the grid was right and the
  // copy on top of it was never part of that. `calibration: true` brings it back
  // for the pages that use it as an instrument.
  if (opts.calibration) {
    g.save();
    g.translate(w * 0.5, h * 0.40);
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.font = `700 ${Math.round(150 * s)}px ui-sans-serif, -apple-system, "SF Pro Display", Helvetica, sans-serif`;
    g.fillStyle = P.type;
    g.fillText('AETHER', 0, 0);
    g.font = `500 ${Math.round(30 * s)}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    g.fillStyle = P.sub;
    g.fillText('READ THIS THROUGH THE WATER', 0, 108 * s);
    g.font = `500 ${Math.round(26 * s)}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    g.fillText('THE GRID IS STRAIGHT. THE FLUID BENDS IT.', 0, 300 * s);
    g.restore();

    g.font = `500 ${Math.round(15 * s)}px ui-monospace, Menlo, monospace`;
    g.fillStyle = P.sub;
    for (let i = 1; i * heavy < h; i++) g.fillText(String(i * 10), 10 * s, i * heavy + 5 * s);
  }

  return cv;
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------
export async function createRenderer(device, canvas, engine, opts = {}) {
  const ctx = canvas.getContext('webgpu');
  if (!ctx) throw new Error('[render] canvas.getContext("webgpu") returned null');
  const format = navigator.gpu.getPreferredCanvasFormat();
  // COPY_SRC so a headless run can read the frame off the GPU without toDataURL.
  ctx.configure({
    device, format, alphaMode: 'opaque',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });

  const FIELD_FORMAT = 'rgba16float';
  const THICK_FORMAT = 'rgba16float';
  const BG_FORMAT = 'rgba8unorm';

  device.pushErrorScope('validation');

  const splatMod = await makeModule(device, SPLAT_WGSL, 'render-splat');
  const blurMod = await makeModule(device, BLUR_WGSL, 'render-blur');
  const compMod = await makeModule(device, COMPOSITE_WGSL, 'render-composite');
  const resolveMod = await makeModule(device, RESOLVE_WGSL, 'render-resolve');
  const diagMod = await makeModule(device, DIAG_WGSL, 'render-diagnostic');

  // ---- bind group layouts. Every sim binding is read-only, deliberately. -----
  const bglSplat = device.createBindGroupLayout({
    label: 'bgl:render-splat',
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      // §21. The colour sidecar. VERTEX only, read-only, and it takes the
      // vertex stage from 1 storage buffer to 2 against this device's cap of 8.
      // Explicit layout (not `layout:'auto'`) is what stops PLATFORM_NOTES trap
      // #6 deleting it the moment a shader variant stops reading it.
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      // POST-LANDING PATCH. The §20 rigid buffer for the crosses view: 2 -> 3
      // storage buffers in the vertex stage against the cap of 8, read-only
      // like every sim binding here. Declared in the ONE splat layout (explicit
      // layouts may carry bindings a shader ignores) so every splat pipeline
      // shares the same bind groups and switching views stays free.
      { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ],
  });
  const bglBlur = device.createBindGroupLayout({
    label: 'bgl:render-blur',
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  });
  const bglComp = device.createBindGroupLayout({
    label: 'bgl:render-composite',
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 7, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
    ],
  });
  const bglResolve = device.createBindGroupLayout({
    label: 'bgl:render-resolve',
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
    ],
  });
  const bglDiag = device.createBindGroupLayout({
    label: 'bgl:render-diagnostic',
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  });

  const blendAdd = {
    color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
  };
  const blendAlpha = {
    color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  };

  const plSplat = device.createPipelineLayout({ bindGroupLayouts: [bglSplat] });
  const mkSplat = (vsEntry, fsEntry, fmt, blend, label) => device.createRenderPipeline({
    label, layout: plSplat,
    vertex: { module: splatMod, entryPoint: vsEntry },
    fragment: { module: splatMod, entryPoint: fsEntry, targets: [{ format: fmt, blend }] },
    primitive: { topology: 'triangle-list' },
  });
  const splatPipe = mkSplat('vs', 'fs_splat', FIELD_FORMAT, blendAdd, 'rp:splat');
  const diagSplatPipe = mkSplat('vs', 'fs_diag', FIELD_FORMAT, blendAdd, 'rp:diagnostic-splat');
  const wallPipe = mkSplat('vs', 'fs_wall', format, blendAlpha, 'rp:wall');
  const beadPipe = mkSplat('vs_bead', 'fs_bead', format, blendAlpha, 'rp:orbies');
  // The class field: one draw, two additive attachments (see fs_cls).
  const clsPipe = device.createRenderPipeline({
    label: 'rp:class', layout: plSplat,
    vertex: { module: splatMod, entryPoint: 'vs' },
    fragment: {
      module: splatMod, entryPoint: 'fs_cls',
      targets: [{ format: FIELD_FORMAT, blend: blendAdd }, { format: FIELD_FORMAT, blend: blendAdd }],
    },
    primitive: { topology: 'triangle-list' },
  });
  const fieldPipe = mkSplat('vs_legacy', 'fs_field', FIELD_FORMAT, blendAdd, 'rp:field');
  const pointsPipe = mkSplat('vs_legacy', 'fs_points', format, blendAlpha, 'rp:points');
  // POST-LANDING PATCH: the two new particle views.
  const dotPipe = mkSplat('vs_dot', 'fs_dot', format, blendAlpha, 'rp:dot');
  const crossPipe = mkSplat('vs_cross', 'fs_cross', format, blendAlpha, 'rp:cross');

  const plBlur = device.createPipelineLayout({ bindGroupLayouts: [bglBlur] });
  const mkBlur = (entry, fmt, label) => device.createRenderPipeline({
    label, layout: plBlur,
    vertex: { module: blurMod, entryPoint: 'vs' },
    fragment: { module: blurMod, entryPoint: entry, targets: [{ format: fmt }] },
    primitive: { topology: 'triangle-list' },
  });
  const gaussPipe = mkBlur('fs_gauss', FIELD_FORMAT, 'rp:gauss');
  const curvPipe = mkBlur('fs_curv', FIELD_FORMAT, 'rp:curv');
  const maskPipe = mkBlur('fs_mask', THICK_FORMAT, 'rp:mask');
  const widePipe = mkBlur('fs_wide', THICK_FORMAT, 'rp:wide');
  const blitPipe = mkBlur('fs_blit', format, 'rp:blit');

  const compPipe = device.createRenderPipeline({
    label: 'rp:composite',
    layout: device.createPipelineLayout({ bindGroupLayouts: [bglComp] }),
    vertex: { module: compMod, entryPoint: 'vs' },
    fragment: { module: compMod, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
  const resolvePipe = device.createRenderPipeline({
    label: 'rp:resolve',
    layout: device.createPipelineLayout({ bindGroupLayouts: [bglResolve] }),
    vertex: { module: resolveMod, entryPoint: 'vs' },
    fragment: { module: resolveMod, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
  // POST-LANDING PATCH: Flat is the metaball path with fs_flat — same layout,
  // same uniform, same field texture, no shading.
  const resolveFlatPipe = device.createRenderPipeline({
    label: 'rp:resolve-flat',
    layout: device.createPipelineLayout({ bindGroupLayouts: [bglResolve] }),
    vertex: { module: resolveMod, entryPoint: 'vs' },
    fragment: { module: resolveMod, entryPoint: 'fs_flat', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
  const diagResolvePipe = device.createRenderPipeline({
    label: 'rp:diagnostic-resolve',
    layout: device.createPipelineLayout({ bindGroupLayouts: [bglDiag] }),
    vertex: { module: diagMod, entryPoint: 'vs' },
    fragment: { module: diagMod, entryPoint: 'fs', targets: [{ format: BG_FORMAT }] },
    primitive: { topology: 'triangle-list' },
  });

  // 16 header floats (48 B of payload rounded up to 64: `look` is an array of
  // vec4 and wants align 16) + 32 look vec4s + 32 emis vec4s. §21 took word 12.
  const SPLAT_HEADER = 16;
  // RUNG 2.6: 64 look + 64 emis slots (32..63 are the interned §25 rows).
  const SPLAT_FLOATS = SPLAT_HEADER + 64 * 4 + 64 * 4;
  const uSplat = device.createBuffer({ size: SPLAT_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const uWall = device.createBuffer({ size: SPLAT_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  // 7 header vec4s + MAT_SLOTS x Mat(5 vec4s).
  const COMP_HEADER = 28;
  const COMP_FLOATS = COMP_HEADER + MAT_SLOTS * MAT_WORDS_RENDER;
  const uComp = device.createBuffer({ size: COMP_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const compUni = new Float32Array(COMP_FLOATS);
  const splatUniBuf = new ArrayBuffer(SPLAT_FLOATS * 4);
  const splatUni = new Float32Array(splatUniBuf);
  const splatUniU = new Uint32Array(splatUniBuf);
  const uResolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const uDiag = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  // One uniform slot per blur invocation, so a frame can queue several blur
  // passes with different parameters without stomping its own uniforms.
  // 256 is the minUniformBufferOffsetAlignment on this platform.
  const MAX_BLUR = 40;
  const uBlur = device.createBuffer({
    size: 256 * MAX_BLUR, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const sampler = device.createSampler({
    magFilter: 'linear', minFilter: 'linear',
    addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
  });

  // §21. Loudly, not silently. A missing sidecar would otherwise surface as
  // "[Invalid BindGroup] is invalid due to a previous error" three calls later
  // (PLATFORM_NOTES trap #6's failure signature), or worse as an async
  // validation error that reads back as a black canvas.
  if (!engine.buf || !engine.buf.tint) {
    throw new Error('[render] engine has no buf.tint — §21 per-particle colour needs an engine ' +
                    'from src/engine.js at v12 or later.');
  }
  const partGroups = [0, 1].map((parity) => device.createBindGroup({
    layout: bglSplat,
    entries: [
      { binding: 0, resource: { buffer: uSplat } },
      { binding: 1, resource: { buffer: parity === 0 ? engine.buf.stateA : engine.buf.stateB } },
      { binding: 2, resource: { buffer: engine.buf.tint } },
      { binding: 3, resource: { buffer: engine.buf.rigid } },
    ],
  }));
  const wallGroups = [0, 1].map((parity) => device.createBindGroup({
    layout: bglSplat,
    entries: [
      { binding: 0, resource: { buffer: uWall } },
      { binding: 1, resource: { buffer: parity === 0 ? engine.buf.stateA : engine.buf.stateB } },
      { binding: 2, resource: { buffer: engine.buf.tint } },
      { binding: 3, resource: { buffer: engine.buf.rigid } },
    ],
  }));

  const err0 = await device.popErrorScope();
  if (err0) throw new Error('[render] pipeline setup failed validation: ' + err0.message);

  // ---- targets --------------------------------------------------------------
  let T = null;
  let readback = null, readbackBytes = 0, viewId = 0;

  function destroyTargets() {
    if (!T) return;
    for (const k of ['field', 'ping', 'pong', 'thickA', 'thickB', 'bg',
                     'diagA', 'diagB', 'diagBg',
                     'matA', 'matB', 'clsA', 'clsB', 'glowA', 'glowB']) {
      try { T[k].destroy(); } catch { /* already gone */ }
    }
    T = null;
  }

  function ensureTargets(w, h) {
    // The metaball path keeps its SHIPPED half-resolution field, so the A/B
    // toggle shows the old renderer as it actually shipped and not a sharpened
    // variant of it.
    const fs = R.mode === 'water' ? R.fieldScale : R.metaballFieldScale;
    const fw = Math.max(1, Math.round(w * fs));
    const fh = Math.max(1, Math.round(h * fs));
    const tw = Math.max(1, Math.round(w * R.thickScale));
    const th = Math.max(1, Math.round(h * R.thickScale));
    // The class field lives at half resolution and the glow at an eighth: they
    // are only ever read as ratios and as a wide blur, so full resolution would
    // be paid for nothing. See fs_cls.
    const cw = Math.max(1, Math.round(w * R.classScale));
    const chh = Math.max(1, Math.round(h * R.classScale));
    const gw = Math.max(1, Math.round(w * R.glowScale));
    const gh = Math.max(1, Math.round(h * R.glowScale));
    if (T && T.w === w && T.h === h && T.fw === fw && T.fh === fh
        && T.tw === tw && T.th === th && T.cw === cw && T.chh === chh
        && T.gw === gw && T.gh === gh
        && T.bgStyle === R.bgStyle && T.bgVersion === R.bgVersion) return;
    destroyTargets();

    const mk = (fmt, ww, hh, extra = 0) => device.createTexture({
      size: [ww, hh], format: fmt,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | extra,
    });
    const field = mk(FIELD_FORMAT, fw, fh);
    const ping = mk(FIELD_FORMAT, fw, fh);
    const pong = mk(FIELD_FORMAT, fw, fh);
    const thickA = mk(THICK_FORMAT, tw, th);
    const thickB = mk(THICK_FORMAT, tw, th);
    const matA = mk(FIELD_FORMAT, cw, chh);
    const matB = mk(FIELD_FORMAT, cw, chh);
    const clsA = mk(FIELD_FORMAT, cw, chh);
    const clsB = mk(FIELD_FORMAT, cw, chh);
    const glowA = mk(FIELD_FORMAT, gw, gh);
    const glowB = mk(FIELD_FORMAT, gw, gh);
    const bg = mk(BG_FORMAT, w, h, GPUTextureUsage.COPY_DST);
    const diagA = mk(FIELD_FORMAT, cw, chh);
    const diagB = mk(FIELD_FORMAT, cw, chh);
    const diagBg = mk(BG_FORMAT, w, h);

    const src = drawBackground(w, h, {
      style: R.bgStyle, flatColor: R.flatColor, calibration: R.bgCalibration,
      image: R.bgImage, imageDim: R.bgImageDim, imageGrid: R.bgImageGrid,
    });
    device.queue.copyExternalImageToTexture({ source: src }, { texture: bg }, [w, h]);

    T = {
      w, h, fw, fh, tw, th, cw, chh, gw, gh,
      field, ping, pong, thickA, thickB, matA, matB, clsA, clsB, glowA, glowB, bg,
      diagA, diagB, diagBg,
      bgStyle: R.bgStyle, bgVersion: R.bgVersion,
      fieldV: field.createView(), pingV: ping.createView(), pongV: pong.createView(),
      thickAV: thickA.createView(), thickBV: thickB.createView(), bgV: bg.createView(),
      matAV: matA.createView(), matBV: matB.createView(),
      clsAV: clsA.createView(), clsBV: clsB.createView(),
      glowAV: glowA.createView(), glowBV: glowB.createView(),
      diagAV: diagA.createView(), diagBV: diagB.createView(), diagBgV: diagBg.createView(),
      groups: new Map(),
    };
    const idOf = (v) => (v.__id || (v.__id = ++viewId));
    T.blurGroupOf = (texView, slot) => {
      const key = `b${slot}|${idOf(texView)}`;
      let g = T.groups.get(key);
      if (!g) {
        g = device.createBindGroup({
          layout: bglBlur,
          entries: [
            { binding: 0, resource: { buffer: uBlur, offset: slot * 256, size: 32 } },
            { binding: 1, resource: texView },
            { binding: 2, resource: sampler },
          ],
        });
        T.groups.set(key, g);
      }
      return g;
    };
    T.compGroupOf = (fieldView, matView, clsView) => {
      const bgView = R.diagnosticMode ? T.diagBgV : T.bgV;
      const key = `c${idOf(fieldView)}|${idOf(matView)}|${idOf(clsView)}|${idOf(bgView)}`;
      let g = T.groups.get(key);
      if (!g) {
        g = device.createBindGroup({
          layout: bglComp,
          entries: [
            { binding: 0, resource: { buffer: uComp } },
            { binding: 1, resource: fieldView },
            { binding: 2, resource: T.thickAV },
            { binding: 3, resource: bgView },
            { binding: 4, resource: sampler },
            { binding: 5, resource: matView },
            { binding: 6, resource: clsView },
            { binding: 7, resource: T.glowAV },
          ],
        });
        T.groups.set(key, g);
      }
      return g;
    };
    T.resolveGroupOf = () => device.createBindGroup({
        layout: bglResolve,
        entries: [
          { binding: 0, resource: { buffer: uResolve } },
          { binding: 1, resource: T.fieldV },
          { binding: 2, resource: sampler },
          { binding: 3, resource: R.diagnosticMode ? T.diagBgV : T.bgV },
        ],
      });
    T.diagGroupOf = (fieldView) => device.createBindGroup({
      layout: bglDiag,
      entries: [
        { binding: 0, resource: { buffer: uDiag } },
        { binding: 1, resource: fieldView },
        { binding: 2, resource: T.bgV },
        { binding: 3, resource: sampler },
      ],
    });
  }

  // ---------------------------------------------------------------------------
  const R = {
    // --- what to draw -------------------------------------------------------
    // 'water' | 'metaball' | 'points' | 'flat' | 'point' | 'crosses'
    // The last three are the post-landing patch's OE-CAKE views: Flat is the
    // metaball threshold with no shading, Point is one device pixel per
    // particle, Crosses is a "+" per particle rotated from the owning body's
    // angle (velocity heading for a moving free particle). All render-only.
    mode: opts.mode ?? 'water',
    // 0 beauty 1 field 2 normals 3 thickness 4 coverage 5 material id
    // 6 material CLASS 7 flat albedo
    debugMode: opts.debugMode ?? 0,
    // 0 off; 1 compression; 2 compression+flow; 3 temperature; 4 curl;
    // 5 vector composite. This is view state only.
    diagnosticMode: opts.diagnosticMode ?? 0,
    diagnosticStrength: opts.diagnosticStrength ?? 1.15,
    // DARK GRID BY DEFAULT — Darien's call, 2026-07-26. See drawBackground.
    bgStyle: opts.bgStyle ?? 'night',            // 'night' | 'studio' | 'grey' | 'image' | 'flat'
    // 'flat' reproduces the ORIGINAL renderer's background exactly: the old
    // clearValue (0.030, 0.038, 0.055) in an 8-bit unorm target is (8, 10, 14).
    flatColor: opts.flatColor ?? '#080a0e',
    // The wordmark / instruction copy / ruler. OFF unless a test page asks for
    // it: it is an instrument, not a backdrop. See drawBackground.
    bgCalibration: opts.bgCalibration ?? false,
    drawWalls: opts.drawWalls ?? true,
    // `contain` is the renderer-library default: preserve world aspect for
    // captures and diagnostics. The interactive toy opts into `stretch`, where
    // the domain's four edges map to the viewport's four edges so there is no
    // letterboxed region that looks like usable simulation space.
    fitMode: opts.fitMode ?? 'contain',           // 'contain' | 'stretch'

    // --- geometry -----------------------------------------------------------
    radiusWu: opts.radiusWu ?? 2.3,
    wallRadScale: opts.wallRadScale ?? 0.60,
    velScale: opts.velScale ?? (1 / 6000),

    // --- surface ------------------------------------------------------------
    threshold: opts.threshold ?? 1.30,
    softness: opts.softness ?? 0.45,
    fieldScale: opts.fieldScale ?? 1.0,          // quality knob for high-DPR displays
    thickScale: opts.thickScale ?? 0.125,

    // --- smoothing. THE MEASURED DEFAULT is gauss x4, sigma 1.9 texels: -----
    // grain in the particle-spacing band drops from 2.460 % of mean luminance
    // to 0.210 % on the psi6 = 0.888 settled pool (8-bit measurement floor for
    // that band: 0.097 %). Curvature flow measured 3x worse and costs 26 % more.
    smoothMode: opts.smoothMode ?? 'gauss',      // 'gauss' | 'curv' | 'none'
    smoothIters: opts.smoothIters ?? 4,
    smoothSigma: opts.smoothSigma ?? 1.9,        // TEXELS — see sigmaAutoScale
    bilateral: opts.bilateral ?? 0.9,            // 0 disables the silhouette stop
    curvDt: opts.curvDt ?? 0.30,
    curvGrad: opts.curvGrad ?? 1.0,
    thickIters: opts.thickIters ?? 4,
    thickSigma: opts.thickSigma ?? 5.5,
    // The blur is specified in TEXELS while the lattice pitch in texels depends
    // on pixels-per-world-unit, so the grain mitigation is scale-relative: it
    // improves as you zoom out and degrades as you zoom in. Turning this on ties
    // sigma to px/wu against the reference at which 0.210 % was measured. OFF by
    // default so the shipped configuration is the measured one.
    sigmaAutoScale: opts.sigmaAutoScale ?? false,
    sigmaRefPxPerWu: opts.sigmaRefPxPerWu ?? 11.25,

    // --- optics -------------------------------------------------------------
    normalScale: opts.normalScale ?? 26.0,
    interiorGain: opts.interiorGain ?? 0.055,
    light: opts.light ?? [-0.42, 0.58, 0.70],
    dispersion: opts.dispersion ?? 0.45,
    exposure: opts.exposure ?? 1.0,
    foamGain: opts.foamGain ?? 0.9,
    fieldDebugScale: opts.fieldDebugScale ?? 0.085,
    mats: { water: { ...PRESETS.water }, goo: { ...PRESETS.goo } },
    // Slots 2..MAT_SLOTS-1 start from the hand-tuned look where one exists and
    // fall back to procedural where it does not. Slots 0 and 1 are overwritten
    // from R.mats every frame, so the two originally hand-tuned looks stay
    // editable the way they always were.
    matTable: Array.from({ length: MAT_SLOTS }, (_, i) => materialLook(i)),
    // Every id the SPLAT shader can see, not just the 24 optic slots: the class
    // and the flat albedo are per-particle data and must exist for ids 24..31
    // too, or a late-arriving material becomes invisible instead of ugly.
    lookTable: Array.from({ length: 32 }, (_, i) => materialLook(i)),

    // --- hybrid rendering (per-class shading) -------------------------------
    hybrid: opts.hybrid ?? true,           // false = the pre-hybrid all-glass look
    classScale: opts.classScale ?? 0.5,
    glowScale: opts.glowScale ?? 0.125,
    classIters: opts.classIters ?? 1,
    // This field separates shading CLASSES inside one body. At 2.2 texels a
    // submerged solid/liquid boundary read as a soft fog bank; 1.6 keeps a
    // little antialiasing while making the material interface visibly firmer.
    // It does not touch the liquid surface field or any simulation state.
    classSigma: opts.classSigma ?? 1.6,
    glowIters: opts.glowIters ?? 2,
    glowSigma: opts.glowSigma ?? 3.2,
    solidHard: opts.solidHard ?? 0.26,     // silhouette softness vs the liquid's
    solidEdge: opts.solidEdge ?? 0.42,     // how much the silhouette darkens
    solidTop: opts.solidTop ?? 0.10,       // lift on upward-facing surfaces
    gasThreshold: opts.gasThreshold ?? 0.72,
    gasSoftness: opts.gasSoftness ?? 0.62,
    glowGain: opts.glowGain ?? 0.90,       // halo strength
    glowWhite: opts.glowWhite ?? 0.55,     // halo mass at which it goes white-hot
    emissiveLift: opts.emissiveLift ?? 0.55,
    coreWhite: opts.coreWhite ?? 0.85,     // body temperature at which it clips
    haloInsideCut: opts.haloInsideCut ?? 0.78,   // how much halo is cut inside a body
    glowLoDeg: opts.glowLoDeg ?? GLOW_LO,
    glowHiDeg: opts.glowHiDeg ?? GLOW_HI,

    // --- §21 per-particle colour --------------------------------------------
    // OFF by default. An imported photograph is the reason this exists, so the
    // import path turns it on when it actually supplies colour — but a session
    // that never imports anything must render byte-for-byte the picture Darien
    // has already art-directed. Defaulting this ON would silently re-tune the
    // look of every scene, which is exactly what the no-PROTECTED.md rule says
    // not to do.
    tintMode: opts.tintMode ?? false,

    // --- user background image ----------------------------------------------
    bgImage: opts.bgImage ?? null,
    bgImageDim: opts.bgImageDim ?? 0.35,
    bgImageGrid: opts.bgImageGrid ?? true,
    bgVersion: 0,

    // --- the metaball path, at its shipped settings -------------------------
    metaballThreshold: opts.metaballThreshold ?? 1.9,
    metaballFieldScale: opts.metaballFieldScale ?? 0.5,

    stats: { passes: 0, pxPerWu: 0 },

    // --- controls -----------------------------------------------------------
    setMode(m) { R.mode = m; },
    setRadius(r) { R.radiusWu = r; },
    setThreshold(t) { R.threshold = t; },
    setDebug(d) { R.debugMode = d | 0; },
    setDiagnostic(d) { R.diagnosticMode = Math.max(0, Math.min(5, d | 0)); return R.diagnosticMode; },
    /** §21. true = image colour, false = material colour. One uniform word. */
    setTintMode(on) { R.tintMode = !!on; return R.tintMode; },
    tintModeOn() { return !!R.tintMode; },
    setBackground(style) {
      R.bgStyle = style;
      ensureTargets(canvas.width | 0, canvas.height | 0);
      return R;
    },
    /**
     * THE USER'S OWN BACKGROUND PICTURE. `src` is anything
     * copyExternalImageToTexture-adjacent that a 2D canvas can draw: an
     * ImageBitmap, an HTMLImageElement, a canvas.
     *
     * This is ONE of the two image-drop paths and it is deliberately a
     * different call from the other one. Dropping a picture in can mean "this
     * image becomes the BACKDROP" (here) or "this image becomes MATTER" (the
     * importer, which turns pixels into particles). The renderer owns only the
     * first; nothing in this file guesses which the user meant.
     *
     *   renderer.setBackgroundImage(bitmap);              // cover-fit, dimmed
     *   renderer.setBackgroundImage(bitmap, { dim: 0 });  // full strength
     *   renderer.setBackgroundImage(null);                // back to the grid
     */
    setBackgroundImage(src, { dim, grid } = {}) {
      R.bgImage = src || null;
      if (dim !== undefined) R.bgImageDim = dim;
      if (grid !== undefined) R.bgImageGrid = grid;
      R.bgStyle = src ? 'image' : (R.bgStyle === 'image' ? 'night' : R.bgStyle);
      R.bgVersion++;
      ensureTargets(canvas.width | 0, canvas.height | 0);
      return R;
    },
    clearBackgroundImage() { return R.setBackgroundImage(null); },
    /**
     * Give a material id a hand-tuned look. ids 0/1 are WATER/GOO and live in
     * R.mats; anything else lands in the optic table. Pass null to go back to
     * the default. Unspecified fields inherit, so
     * `setMaterialPreset(4, { scatter: [1,0.3,0.1] })` is a legal, useful call.
     *
     * `cls` and `albedo` are part of a preset too, and unlike the optics they
     * apply to EVERY id the splat shader can see (0..31), not only the 24 optic
     * slots — that is what keeps a brand-new material visible.
     */
    setMaterialPreset(id, preset) {
      const i = id >>> 0;
      if (i >= 32) throw new Error(`[render] material id ${id} >= 32`);
      const base = materialLook(i);
      const full = preset ? { ...base, ...preset } : base;
      R.lookTable[i] = full;
      if (i < MAT_SLOTS) {
        if (i < 2) R.mats[i === 0 ? 'water' : 'goo'] = { ...(preset ? { ...PRESETS[i === 0 ? 'water' : 'goo'], ...preset } : PRESETS[i === 0 ? 'water' : 'goo']) };
        else R.matTable[i] = full;
      }
      return R;
    },
    materialPreset(id) { return id === 0 ? R.mats.water : id === 1 ? R.mats.goo : R.matTable[id >>> 0]; },
    /**
     * RUNG 2.6 — the look an INTERNED §25 row (id 32..63) wears: the BASE
     * row's look pulled 60 % toward the composed row's own `tint` word (the
     * composeTint algebra — base hue shifted per modifier), so "brittle
     * elastic goo" is visibly kin to GOO and visibly not GOO. The class is
     * taken from the composed row's PHYSICS (a bond template or a rigid
     * solidMode makes it a SOLID on screen, whatever the base was); glow and
     * gas opacity ride with the base. Reads the engine's live mint list every
     * call — a recycled id re-derives on the next frame, no cache. Render-only
     * METADATA: nothing here writes anything, and the render-ui replay arm
     * asserts the particle bytes cannot move.
     */
    derivedLook(id) {
      const i = id >>> 0;
      const e = engine.internEntry ? engine.internEntry(i) : null;
      if (!e) return R.lookTable[i & 31] || materialLook(i);
      const baseLook = R.lookTable[e.base & 31] || materialLook(e.base);
      const t = e.words[TINT_WORD] >>> 0;
      const tr = ((t >>> 16) & 255) / 255, tg = ((t >>> 8) & 255) / 255, tb = (t & 255) / 255;
      const ba = baseLook.albedo || [0.5, 0.5, 0.5];
      const K = 0.6;
      const albedo = [ba[0] + (tr - ba[0]) * K, ba[1] + (tg - ba[1]) * K, ba[2] + (tb - ba[2]) * K];
      const solid = e.words[SOLIDMODE_WORD] !== 0 || e.words[BONDK_WORD] !== 0;
      const cls = solid && (baseLook.cls ?? 0) === 0 ? 1 : (baseLook.cls ?? 0);
      return { ...baseLook, albedo, cls, derived: true, base: e.base };
    },
    materialLook(id) { return R.derivedLook(id >>> 0); },
    set(k, v) { R[k] = v; return R; },

    // --- the frame ----------------------------------------------------------
    draw({ capture = false } = {}) {
      const w = canvas.width | 0, h = canvas.height | 0;
      if (w === 0 || h === 0 || engine.n === 0) return null;
      ensureTargets(w, h);

      // Standalone captures preserve aspect by default. The interactive toy
      // asks for full-bleed `stretch`, making the viewport itself the domain
      // boundary even when the browser is not the world's native aspect.
      const domW = engine.domW, domH = engine.domH;
      const canvasAspect = w / h, domAspect = domW / domH;
      let sx = 1, sy = 1;
      if (R.fitMode !== 'stretch') {
        if (domAspect > canvasAspect) sy = canvasAspect / domAspect;
        else sx = domAspect / canvasAspect;
      }
      // Radius in NDC. A stretched full-bleed view scales the complete world
      // transform, including particle kernels; contain keeps discs round.
      const rx = (R.radiusWu * 65536 / domW) * 2 * sx;
      const ry = R.fitMode === 'stretch'
        ? (R.radiusWu * 65536 / domH) * 2
        : rx * canvasAspect;
      const pxPerWu = (w * sx * 65536) / domW;
      R.stats.pxPerWu = pxPerWu;

      const suf = splatUni, suu = splatUniU;
      // POST-LANDING PATCH: Points dot radius -50% (Darien, verbatim). Crosses
      // share the halved size so flipping between the two particle views keeps
      // marks the same scale. Water/metaball splat radii are untouched, so the
      // beauty view and the metaball surface cannot move.
      const modeRad = (R.mode === 'points' || R.mode === 'crosses') ? 0.5 : 1;
      suf[0] = 1 / domW; suf[1] = 1 / domH;
      suf[2] = sx; suf[3] = sy;
      suf[4] = rx * modeRad; suf[5] = ry * modeRad;
      suu[6] = engine.n; suu[7] = engine.nFluid;
      suf[8] = R.velScale; suf[9] = 1.0;
      suf[10] = R.glowLoDeg;
      suf[11] = 1 / Math.max(1, R.glowHiDeg - R.glowLoDeg);
      // §21. The toggle. It reaches the GPU as a uniform word and nothing else:
      // no pipeline swap, no second bind group, no re-upload of the sidecar. The
      // buffer is bound whether or not it is being read, so switching views is
      // free and cannot fail halfway.
      suf[12] = R.tintMode ? 1 : 0;
      // POST-LANDING PATCH: the former pads. Canvas device-pixel size for
      // vs_dot's pixel snap; the compact §20 body count for vs_cross's tag
      // search. Read every frame from the engine's own host mirror, never
      // cached, so a body promoted or dissolved this frame is what the crosses
      // rotate from.
      suf[13] = w; suf[14] = h; suf[15] = engine.bodyCount || 0;
      for (let i = 0; i < 64; i++) {
        // RUNG 2.6: any slot currently held by an interned §25 row wears its
        // DERIVED look — and the mats TAIL STARTS AT THE AUTHORED COUNT (27),
        // not at 32, so slots 27..31 derive too (they used to get an arbitrary
        // procedural hue; 32..63 used to clamp onto row 31). Recycled rows
        // re-derive on the next frame for free, because derivedLook reads the
        // engine's live mint list rather than caching anything, and it falls
        // back to the authored look table for every non-interned id.
        const L = R.derivedLook(i);
        const al = L.albedo || [0.5, 0.5, 0.5];
        const o = SPLAT_HEADER + i * 4;
        suf[o] = al[0]; suf[o + 1] = al[1]; suf[o + 2] = al[2];
        suf[o + 3] = L.cls ?? 0;
        const e = SPLAT_HEADER + 256 + i * 4;
        suf[e] = L.glowFloor ?? 0; suf[e + 1] = L.glowGain ?? 1;
        suf[e + 2] = L.gasOpacity ?? 0.55; suf[e + 3] = 0;
      }
      device.queue.writeBuffer(uSplat, 0, suf);
      suf[9] = R.wallRadScale;
      device.queue.writeBuffer(uWall, 0, suf);

      const enc = device.createCommandEncoder({ label: 'render' });
      const nFluid = engine.nFluid, nWall = engine.n - engine.nFluid;
      const parity = engine.parity | 0;
      const water = R.mode === 'water';
      let passes = 0;

      // ---- 0a. renderer-only diagnostic background ------------------------
      // Reconstruct a smooth vector/temperature field from particle state,
      // then resolve it over the authored background. It is never copied into
      // an engine buffer, so toggling it cannot affect determinism.
      if (R.diagnosticMode) {
        {
          const p = enc.beginRenderPass({
            colorAttachments: [{
              view: T.diagAV, clearValue: { r: 0, g: 0, b: 0, a: 0 },
              loadOp: 'clear', storeOp: 'store',
            }],
          });
          if (nFluid > 0) {
            p.setPipeline(diagSplatPipe);
            p.setBindGroup(0, partGroups[parity]);
            p.draw(6, nFluid, 0, 0);
          }
          p.end(); passes++;
        }
        const diagBlur = new Float32Array([1 / T.cw, 0, 4.2, 0, 0, 0, 0, 0]);
        // Slots 38/39 are reserved for this pre-pass. The beauty path starts
        // at slot 0 and may queue many differently parameterised blurs before
        // submit; sharing a slot would make the last queue.writeBuffer silently
        // retune an earlier draw.
        device.queue.writeBuffer(uBlur, 38 * 256, diagBlur);
        {
          const p = enc.beginRenderPass({
            colorAttachments: [{ view: T.diagBV, loadOp: 'clear',
              clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store' }],
          });
          p.setPipeline(gaussPipe); p.setBindGroup(0, T.blurGroupOf(T.diagAV, 38)); p.draw(3); p.end(); passes++;
        }
        diagBlur[0] = 0; diagBlur[1] = 1 / T.chh;
        device.queue.writeBuffer(uBlur, 39 * 256, diagBlur);
        {
          const p = enc.beginRenderPass({
            colorAttachments: [{ view: T.diagAV, loadOp: 'clear',
              clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store' }],
          });
          p.setPipeline(gaussPipe); p.setBindGroup(0, T.blurGroupOf(T.diagBV, 39)); p.draw(3); p.end(); passes++;
        }
        device.queue.writeBuffer(uDiag, 0, new Float32Array([
          R.diagnosticMode, R.diagnosticStrength, 1 / T.cw, 1 / T.chh,
          (engine.frameNo || 0) * 0.12, 0.72, 0.30, 0,
        ]));
        {
          const p = enc.beginRenderPass({
            colorAttachments: [{ view: T.diagBgV, loadOp: 'clear',
              clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store' }],
          });
          p.setPipeline(diagResolvePipe); p.setBindGroup(0, T.diagGroupOf(T.diagAV)); p.draw(3); p.end(); passes++;
        }
      }
      const backgroundView = R.diagnosticMode ? T.diagBgV : T.bgV;

      // ---- 0. the per-particle modes: background blit, particles, done -----
      // 'points' (shaded discs), 'point' (1px dots) and 'crosses' ("+" marks)
      // share one path and differ only in which splat pipeline draws.
      if (R.mode === 'points' || R.mode === 'point' || R.mode === 'crosses') {
        const view = ctx.getCurrentTexture().createView();
        const p = enc.beginRenderPass({
          colorAttachments: [{ view, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store' }],
        });
        device.queue.writeBuffer(uBlur, 0, new Float32Array(8));
        p.setPipeline(blitPipe);
        p.setBindGroup(0, T.blurGroupOf(backgroundView, 0));
        p.draw(3, 1);
        p.setPipeline(R.mode === 'point' ? dotPipe : R.mode === 'crosses' ? crossPipe : pointsPipe);
        p.setBindGroup(0, partGroups[parity]);
        // `drawWalls` is a view policy, not just a beauty-renderer policy.
        // The interactive viewport uses an invisible collision container, so
        // Points/Crosses must not resurrect its boundary particles as dark
        // discs after a mode switch.
        p.draw(6, R.drawWalls ? engine.n : nFluid, 0, 0);
        // Orbies retain their individual rainbow/glass treatment in every
        // render view, not just the default Shader view.
        p.setPipeline(beadPipe);
        p.setBindGroup(0, partGroups[parity]);
        p.draw(6, nFluid, 0, 0);
        p.end(); passes++;
        return finish(enc, w, h, capture, passes);
      }

      // ---- 1. splat --------------------------------------------------------
      {
        const p = enc.beginRenderPass({
          colorAttachments: [{
            view: T.fieldV, clearValue: { r: 0, g: 0, b: 0, a: 0 },
            loadOp: 'clear', storeOp: 'store',
          }],
        });
        if (nFluid > 0) {
          p.setPipeline(water ? splatPipe : fieldPipe);
          p.setBindGroup(0, partGroups[parity]);
          p.draw(6, nFluid, 0, 0);
        }
        p.end(); passes++;
      }

      // ---- 2. smoothing ----------------------------------------------------
      let srcView = T.fieldV;
      let slot = 0;
      const blurUni = new Float32Array(8);
      const writeBlur = (stepX, stepY, sigma, bil, thr, soft, k, rate) => {
        blurUni[0] = stepX; blurUni[1] = stepY; blurUni[2] = sigma; blurUni[3] = bil;
        blurUni[4] = thr; blurUni[5] = soft; blurUni[6] = k; blurUni[7] = rate;
        device.queue.writeBuffer(uBlur, slot * 256, blurUni);
        return slot++;
      };
      const runBlur = (pipe, inView, outView, args) => {
        const s = writeBlur(...args);
        const p = enc.beginRenderPass({
          colorAttachments: [{ view: outView, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store' }],
        });
        p.setPipeline(pipe);
        p.setBindGroup(0, T.blurGroupOf(inView, s));
        p.draw(3, 1);
        p.end(); passes++;
      };

      if (water && R.smoothMode !== 'none' && R.smoothIters > 0) {
        const tx = 1 / T.fw, ty = 1 / T.fh;
        const sigma = R.sigmaAutoScale && pxPerWu > 0
          ? R.smoothSigma * (pxPerWu * R.fieldScale) / R.sigmaRefPxPerWu
          : R.smoothSigma;
        for (let i = 0; i < R.smoothIters && slot < MAX_BLUR - 10; i++) {
          // Two scratch targets, neither of which is the texture being read.
          const A = (srcView === T.pingV) ? T.pongV : T.pingV;
          if (R.smoothMode === 'gauss') {
            const B = (srcView === T.fieldV) ? T.pongV : srcView;
            runBlur(gaussPipe, srcView, A, [tx, 0, sigma, R.bilateral, 0, 0, 0, 0]);
            runBlur(gaussPipe, A, B, [0, ty, sigma, R.bilateral, 0, 0, 0, 0]);
            srcView = B;
          } else {
            runBlur(curvPipe, srcView, A, [tx, ty, 0, 0, R.threshold, R.softness, R.curvGrad, R.curvDt]);
            srcView = A;
          }
        }
      }

      // ---- 3. thickness ----------------------------------------------------
      // Ends in thickAV after an even number of wide passes, which is what the
      // composite bind group reads.
      if (water) {
        const ttx = 1 / T.tw, tty = 1 / T.th;
        runBlur(maskPipe, srcView, T.thickAV, [0, 0, 0, 0, R.threshold, R.softness, 0, 0]);
        for (let i = 0; i < R.thickIters && slot < MAX_BLUR - 2; i++) {
          runBlur(widePipe, T.thickAV, T.thickBV, [ttx, 0, R.thickSigma, 0, 0, 0, 0, 0]);
          runBlur(widePipe, T.thickBV, T.thickAV, [0, tty, R.thickSigma, 0, 0, 0, 0, 0]);
        }
      }

      // ---- 3b. the class field: what KIND of matter is in each pixel -------
      // Half resolution, one draw, one separable blur. Then a wide blur of the
      // glow channel alone, at an eighth resolution, which is the halo.
      let matView = T.matAV, clsView = T.clsAV;
      if (water && R.hybrid) {
        {
          const p = enc.beginRenderPass({
            colorAttachments: [
              { view: T.matAV, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
              { view: T.clsAV, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
            ],
          });
          if (nFluid > 0) {
            p.setPipeline(clsPipe);
            p.setBindGroup(0, partGroups[parity]);
            p.draw(6, nFluid, 0, 0);
          }
          p.end(); passes++;
        }
        const ctx1 = 1 / T.cw, cty = 1 / T.chh;
        for (let i = 0; i < R.classIters && slot < MAX_BLUR - 8; i++) {
          runBlur(gaussPipe, matView, T.matBV, [ctx1, 0, R.classSigma, 0, 0, 0, 0, 0]);
          runBlur(gaussPipe, T.matBV, T.matAV, [0, cty, R.classSigma, 0, 0, 0, 0, 0]);
          runBlur(gaussPipe, clsView, T.clsBV, [ctx1, 0, R.classSigma, 0, 0, 0, 0, 0]);
          runBlur(gaussPipe, T.clsBV, T.clsAV, [0, cty, R.classSigma, 0, 0, 0, 0, 0]);
          matView = T.matAV; clsView = T.clsAV;
        }
        // The halo: a wide blur of the class field, of which the composite reads
        // only the glow-mass channel. The seed pair does the 4x downsample with
        // real taps (steps in SOURCE texels horizontally, DEST texels
        // vertically) instead of point-sampling it and aliasing the glow.
        // Ends in glowA after an even number of passes.
        const gtx = 1 / T.gw, gty = 1 / T.gh;
        runBlur(widePipe, clsView, T.glowBV, [ctx1, 0, 2.2, 0, 0, 0, 0, 0]);
        runBlur(widePipe, T.glowBV, T.glowAV, [0, gty, 2.2, 0, 0, 0, 0, 0]);
        for (let i = 0; i < R.glowIters && slot < MAX_BLUR - 2; i++) {
          runBlur(widePipe, T.glowAV, T.glowBV, [gtx, 0, R.glowSigma, 0, 0, 0, 0, 0]);
          runBlur(widePipe, T.glowBV, T.glowAV, [0, gty, R.glowSigma, 0, 0, 0, 0, 0]);
        }
      }

      // ---- 4. composite ----------------------------------------------------
      const view = ctx.getCurrentTexture().createView();
      if (water) {
        const cu = compUni;
        cu[0] = 1 / T.fw; cu[1] = 1 / T.fh; cu[2] = 1 / T.tw; cu[3] = 1 / T.th;
        cu[4] = R.threshold; cu[5] = R.softness; cu[6] = R.normalScale; cu[7] = R.interiorGain;
        cu[8] = R.light[0]; cu[9] = R.light[1]; cu[10] = R.light[2]; cu[11] = R.dispersion;
        cu[12] = R.debugMode; cu[13] = R.exposure; cu[14] = R.foamGain; cu[15] = R.fieldDebugScale;
        cu[16] = R.solidHard; cu[17] = R.threshold * R.gasThreshold;
        cu[18] = R.softness * R.gasSoftness; cu[19] = R.hybrid ? 1 : 0;
        cu[20] = R.glowGain; cu[21] = R.glowWhite; cu[22] = R.solidEdge; cu[23] = R.solidTop;
        cu[24] = R.emissiveLift; cu[25] = R.coreWhite; cu[26] = R.haloInsideCut; cu[27] = 0;
        R.matTable[0] = R.mats.water;
        R.matTable[1] = R.mats.goo;
        for (let i = 0; i < MAT_SLOTS; i++) {
          packMat(R.matTable[i] || materialLook(i), cu, COMP_HEADER + i * MAT_WORDS_RENDER);
        }
        device.queue.writeBuffer(uComp, 0, cu);

        const p = enc.beginRenderPass({
          colorAttachments: [{ view, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store' }],
        });
        p.setPipeline(compPipe);
        p.setBindGroup(0, T.compGroupOf(srcView, matView, clsView));
        p.draw(3, 1);
        p.setPipeline(beadPipe);
        p.setBindGroup(0, partGroups[parity]);
        p.draw(6, nFluid, 0, 0);
        if (R.drawWalls && nWall > 0 && R.debugMode === 0) {
          p.setPipeline(wallPipe);
          p.setBindGroup(0, wallGroups[parity]);
          p.draw(6, nWall, 0, nFluid);
        }
        p.end(); passes++;
      } else {
        device.queue.writeBuffer(uResolve, 0, new Float32Array(
          [1 / T.fw, 1 / T.fh, R.metaballThreshold, R.metaballThreshold * 0.34]));
        const p = enc.beginRenderPass({
          colorAttachments: [{ view, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store' }],
        });
        // 'metaball' shades the isocontour; 'flat' (post-landing patch) is the
        // identical field and threshold with the shading removed.
        p.setPipeline(R.mode === 'flat' ? resolveFlatPipe : resolvePipe);
        p.setBindGroup(0, T.resolveGroupOf());
        p.draw(3, 1);
        p.setPipeline(beadPipe);
        p.setBindGroup(0, partGroups[parity]);
        p.draw(6, nFluid, 0, 0);
        if (R.drawWalls && nWall > 0) {   // container drawn on top so the walls read
          p.setPipeline(pointsPipe);
          p.setBindGroup(0, partGroups[parity]);
          p.draw(6, nWall, 0, nFluid);
        }
        p.end(); passes++;
      }

      return finish(enc, w, h, capture, passes);
    },

    /** Render one frame and return {w, h, rgba:Uint8ClampedArray} in RGBA order. */
    async capture() {
      const cap = R.draw({ capture: true });
      if (!cap) return null;
      await cap.buffer.mapAsync(GPUMapMode.READ);
      const src = new Uint8Array(cap.buffer.getMappedRange().slice(0));
      cap.buffer.unmap();
      const out = new Uint8ClampedArray(cap.w * cap.h * 4);
      const bgra = cap.format.startsWith('bgra');
      for (let y = 0; y < cap.h; y++) {
        let s = y * cap.bpr, d = y * cap.w * 4;
        for (let x = 0; x < cap.w; x++, s += 4, d += 4) {
          if (bgra) { out[d] = src[s + 2]; out[d + 1] = src[s + 1]; out[d + 2] = src[s]; }
          else { out[d] = src[s]; out[d + 1] = src[s + 1]; out[d + 2] = src[s + 2]; }
          out[d + 3] = 255;
        }
      }
      return { w: cap.w, h: cap.h, rgba: out };
    },

    destroy() {
      destroyTargets();
      uSplat.destroy(); uWall.destroy(); uComp.destroy(); uBlur.destroy(); uResolve.destroy(); uDiag.destroy();
      if (readback) { readback.destroy(); readback = null; }
    },
  };

  // Optional readback, then submit. Shared by every mode.
  function finish(enc, w, h, capture, passes) {
    let cap = null;
    if (capture) {
      const bpr = Math.ceil((w * 4) / 256) * 256;
      const need = bpr * h;
      if (!readback || readbackBytes < need) {
        if (readback) readback.destroy();
        readback = device.createBuffer({ size: need, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        readbackBytes = need;
      }
      enc.copyTextureToBuffer(
        { texture: ctx.getCurrentTexture() },
        { buffer: readback, bytesPerRow: bpr, rowsPerImage: h },
        [w, h, 1]);
      cap = { w, h, bpr, buffer: readback, format };
    }
    device.queue.submit([enc.finish()]);
    R.stats.passes = passes;
    return cap;
  }

  return R;
}
