// SunaEngine — real SunaBox (aether) engine code, extracted UNMODIFIED below this header
// from aether commit e41e06f. Verify: diff <(tail -n +6 src/engine.js) <aether>/src/engine.js
// Copyright (C) 2026 Darien Bathalter
// SPDX-License-Identifier: AGPL-3.0-only — see LICENSE. NO WARRANTY.

// ============================================================================
// aether — host engine (implementer G)
// ----------------------------------------------------------------------------
// The ONLY module that talks to WebGPU for simulation. render.js gets a
// read-only view; nothing else creates pipelines or bind groups.
//
// HARD RULES ENFORCED HERE (Binding Spec v1):
//   * Fixed integer timestep. dt does not exist. `step(count)` advances exactly
//     `count` substeps. Nothing in this file reads a clock, and nothing derives
//     a substep count from elapsed real time.
//   * Seeded integer PRNG only (xorshift32 / splitmix32). No float randomness.
//   * Explicit GPUBindGroupLayouts for every pipeline. Never the automatic
//     reflected layout (it silently deletes unused bindings — trap #6).
//   * Every shader module is compile-checked via getCompilationInfo().
//   * All GPU work runs inside a validation error scope; errors are hard fails.
// ============================================================================

import {
  digestState, foldChain, newChain, hex8, CHAIN_INIT,
} from './hash/statehash.js';

// ---------------------------------------------------------------- §7.1 exports
// v2 (2026-07-25): SOLVER CHANGE — xsph now uses Monaghan's symmetric
// K_ij = 2*RHO0/(rho_i+rho_j) instead of rinv_j, and §6.2 cross-material
// coefficient mixing is implemented. Both change the keystream (§11 R7), so
// this MUST be bumped, and every v1 golden and glyph is invalid under it.
//
// v3 (2026-07-25): SOLVER CLASS CHANGE — the position-correction loop is now
// 4-COLOUR BLOCK GAUSS-SEIDEL (§5.1b) instead of pure Jacobi. Per iteration: one
// shared `solveA` (lambda stays Jacobi), then four (`solveB_cK`, `applyDp_cK`)
// pairs in colour order 0,1,2,3. 24 dispatches/substep -> 48. This is the fix
// for the crystallisation recorded in §11 R10 / §C17: measured psi6 on a settled
// 3600-particle pool at 10000 substeps fell 0.894 -> 0.695, and on the §C17
// reference scene 0.804 -> 0.613 against the Gauss-Seidel prototype's 0.5256.
// It re-versions every hash and every golden. The material table is UNCHANGED.
//
// v4 (2026-07-26): TEMPERATURE + REACTIONS (§12). `Particle.temp` and
// `Particle.pad0` become live state (`temp`, `fuel`), so the state hash widens
// from 6 words to 8 — that alone re-versions every chain even on a thermally
// quiescent scene. Two passes are appended to the substep (`conduct`,
// `thermal`), 48 dispatches -> 50, and `struct Mat` grows 8 words -> 32.
// The MECHANICAL solver is untouched: on a scene held at AMBIENT the 6-word
// digest is bit-identical to v3 (verified — see test/thermal.html T0).
// SPEC_VERSION 6 (2026-07-26) — §16 GRANULAR FRICTION. Two dispatches are added
// PER SOLVER ITERATION (`friction`, `applyFric`), 50 -> 58 per substep, and
// `struct Mat`'s reserved word 31 becomes `fric`. It is the first term in this
// solver that resists SHEAR, and it is the only reason any material keeps a
// shape: before it, all 24 spread to a one-particle film in 2-4 s.
// THE CHAIN DID NOT MOVE. fric = 0 takes an early exit that writes dp = (0,0),
// WATER's fric is 0, and the golden scene is 100% WATER — so the v6 chain on
// dambreak_v1 is byte-identical to v5's. Only tableHash changed, which is why a
// v6 golden had to be recorded at all. Asserted, not assumed: granular.html G0.
//
// SPEC_VERSION 7 (2026-07-26) — §17 GAS VENTING AND RETIREMENT IN PLACE. The
// first way a particle can STOP EXISTING without another particle taking its
// place. `struct Mat`'s last reserved word 3 (`vort`) becomes `phantom`, a new
// row 24 `VOID` carries phantom = 1, and `Particle.flags` gains bit 1
// (FLAG_DEAD). SMOKE stops condensing into ASH — it condenses into VOID and
// splits 6/256 into ASH, which is what takes wood's ash yield from 100 % of
// burned mass to 12.3 %.
// RETIREMENT IS IN PLACE. A retired particle keeps its storage slot forever:
// nothing is compacted, `n` and `nFluid` do not change, and every line the
// solver gained is self-indexed. That is a hard requirement, not a convenience
// — design/bonds/prove_slot_stability.mjs is the gate, and the concurrent
// bonded-body design stores bonds as raw slot indices on the strength of it.
// THE CHAIN DID NOT MOVE. dambreak_v1 is 100 % WATER at AMBIENT, WATER.phantom
// is 0, no particle ever carries FLAG_DEAD, and rows 0-23 keep their exact
// bytes for every word the golden scene reads — so the v7 chain on the gate
// scene is byte-identical to v6's. `matsHash` DOES move (ASH's row is retuned
// and row 24 is appended), which is exactly what matsHash was added to catch.
// Asserted, not assumed: test/elements.html E13.v7InertVsV6.
//
// SPEC_VERSION 9 (2026-07-26) — §18.22 ONE GESTURE, ONE WELD GROUP. `Params`
// gains a 13th word, `weldHold`: the ONE weld tag the host has declared still
// open. `bondForm2` keeps that tag instead of clearing it, so every stamp a
// brush stroke lays welds into ONE body — which is the difference between
// painting stone and painting drops of stone. Nothing else in the substep moves:
// no dispatch is added (58 stays 58), no buffer is added, `struct Mat` and
// `struct Particle` are untouched, and the material table is byte-identical.
// THE CHAIN DID NOT MOVE, AND NEITHER DID matsHash. `weldHold` is 0 for every
// scene `loadScene` produces, and at 0 the new branch writes exactly the word
// the v8 unconditional write stored. So the v9 chain on dambreak_v1 is
// byte-identical to v8's — which is byte-identical to v4's. `tableHash` is
// unchanged too; only `specVersion` moved, which is why a v9 golden had to be
// recorded at all. Asserted by RUNNING, not assumed: bonds.html B0 re-runs the
// all-bond-inert table, and the recorded golden carries the same chain.
//
// SPEC_VERSION 5 (2026-07-26) — §13 CORROSION. `struct Derived` gains `corr`
// (56 -> 64 bytes) and `struct Mat` spends three of its four reserved words on
// (corrode, corrPt, corrTo). No Particle word changed, no mechanical pass
// changed, and with every material's `corrode` at 0 the new branch is
// bit-for-bit inert — so v5 reproduces the v4 CHAIN EXACTLY on the gate scene.
// That is asserted, not assumed: test/golden/determinism.v5.json carries the
// same chain as v4 and test/elements.html E0 gates the equality.
// SPEC_VERSION 12 (2026-07-27) — §18.23 RUBBER AND PHOTO GET BONDS. TWO TABLE
// ROWS AND NOTHING ELSE: no dispatch is added (58 stays 58), no buffer, no
// binding, no bond bit, no `struct Mat` field, and `src/sim.wgsl` is
// byte-identical. RUBBER goes from bondK 0 — a ball that spread to x12.51 its
// own radius of gyration, i.e. a puddle — to a soft, very-high-break-strain
// distance network that holds a shape (x1.01), squashes to 2.13:1 on impact and
// gives 78 % of it back. PHOTO goes from being held by `fric: 1.75` alone
// (x2.41) to being held by structure (x1.01).
// THE CHAIN DID NOT MOVE. dambreak_v1 is 100 % WATER, WATER's row is untouched,
// and every §18 branch keys off `bondK == 0`, so the v12 chain on the gate scene
// is byte-identical to v11's — which is byte-identical to v4's. `tableHash` is
// unchanged (the kernel LUTs are not a function of the material table); only
// `specVersion` and `matsHash` moved, and matsHash moving is exactly what
// matsHash exists to catch. Asserted by RUNNING, not assumed: bonds.html B0
// re-runs the all-bond-inert table and elements.html E0 compares the goldens.
// SPEC_VERSION 13 (2026-07-27) — §23 DEVICES: THE FILTERING DRAIN, THE JET AND
// THE STRING PIVOT. Only ONE of the three touches the solver. §23.A (Outflow as
// a FILTER) is one predicate inside `index.html`'s existing collector circle
// test; §23.C (a String particle on a rigid body is a PIVOT) is one line in
// `src/oec.js`'s recipe mapper spending §19.B, which already shipped. §23.B is
// the shader change: `Params` gains `nJets` and two align(16) arrays of eight
// jets each (the uniform grows 64 -> 320 B), and `jet_accel` — a sum of
// constant accelerations over the discs containing a point — is added in
// `predict` beside gravity and inside `rigidStep`'s existing per-member force
// fold, where it produces torque as well as force.
// NO NEW DISPATCH (58 stays 58), NO NEW STORAGE BUFFER, NO NEW BINDING, no bond
// bit, no `struct Mat` field, no `struct Particle` word. The 8-storage-buffers-
// per-stage cap `friction` and `bondForm1` sit at is untouched: a jet lives in
// the UNIFORM.
// THE CHAIN DID NOT MOVE, AND NEITHER DID matsHash OR tableHash. At `nJets == 0`
// the accumulation loop never runs, `jet_accel` returns (0, 0), and both call
// sites add it with `fp_add_sat`, for which x + 0 == x exactly — so every scene
// the determinism gate, the goldens or `loadScene` produce is bit-for-bit the
// v12 run. The v12 and v13 goldens differ in EXACTLY ONE FIELD, `specVersion`.
// Asserted by RUNNING, not assumed: elements.html E0 compares the goldens field
// by field and test/jet.html J0 runs the same scene with jets set and cleared.
// SPEC_VERSION 14 (2026-07-27) — §24 USER: ARROW-KEY MATTER. The last of
// PARITY.md's six named gaps that was neither closed nor blocked on provenance.
// ONE BODY RECORD WORD (`RB_USER`, word 28 of 32) and ONE UNIFORM VECTOR
// (`userAx`/`userAy` at offset 320; the uniform grows 320 -> 336 B), read in
// exactly one place: `rigidStep`'s free-body velocity fold, where the drive is
// added AFTER the force sum is divided by N — so it is an acceleration of the
// BODY and cannot torque it, which is the same choice §20 made for gravity.
// NO NEW DISPATCH (58 stays 58), NO NEW STORAGE BUFFER, NO NEW BINDING, no bond
// bit, no `struct Mat` field, no `struct Particle` word, NO PARTICLE FLAG BIT
// (bit 3 is still reserved for a future pivot mark). Bond bit 31 and the
// BOND_CAP row stride are both still unspent.
// KEY STATE IS AN INPUT. The host samples it once per frame at a fixed point in
// its own schedule and calls `setUserDrive`; nothing in the sim path reads a
// clock. A replay of the same input sequence is byte-identical, and that is
// asserted by RUNNING a recorded sequence twice (test/user.html U7).
// THE CHAIN DID NOT MOVE, AND NEITHER DID matsHash OR tableHash. `RB_USER` is
// word 28 of a zero-filled record, so it reads 0 for every body ever written
// before §24; `P.userA` is (0, 0) until a host calls `setUserDrive`; and
// `fp_add_sat(x, 0) == x` exactly. The v13 and v14 goldens differ in EXACTLY
// ONE FIELD, `specVersion`. Asserted by RUNNING: elements.html E0 compares the
// goldens field by field and test/user.html U0 runs the same scene with the
// drive set, cleared, and set with no body flagged.
// SPEC_VERSION 15 (2026-07-27) — §26 THE FEEDBACK PASS. Six things Darien asked
// for after playing it, and exactly ONE of them is a table change: §26.F adds
// BEDROCK, the material nothing happens to, so that a machine can survive its
// own experiment. Everything else in §26 is HOST-SIDE: anchoring gated on
// `solidMode` (§26.A), an imported picture promoted through the SAME
// `promoteWeldGroup` a gesture takes (§26.B), the Scene box (§26.C), the pencil
// and the Controls modal (§26.D) and the tooltip's placement and length (§26.E).
// NO SHADER CHANGE AT ALL. `src/sim.wgsl` is byte-identical across this bump —
// no new dispatch (58 stays 58), no storage buffer, no binding, no bond bit, no
// `struct Mat` field, no `struct Particle` word, no uniform growth (`Params`
// stays 336 B). §26.C's world gravity is a re-upload of the material image's
// `grav` COLUMN and §26.C's drain is `n` moving back to `nFluid`; the solver
// reads the fields it always read and cannot tell either happened.
// THE CHAIN DID NOT MOVE. The v14 and v15 goldens differ in EXACTLY THREE
// FIELDS — `specVersion`, `matsHash` and `matCount` — all three of which are
// the appended row announcing itself, which is the entire reason `matsHash` and
// `matCount` are in a golden. `tableHash` does not move (the kernel LUTs are
// not a function of the material table) and neither does the all-WATER
// `dambreak_v1` chain, which has now survived a TENTH version bump: BEDROCK is
// row 26 and no golden scene contains one. Asserted by RUNNING: elements.html
// E0 compares the goldens field by field.
// SPEC_VERSION 16 (2026-07-27) — §25 COMPOSABLE PROPERTY BITS. The eleventh
// bump, and it buys TWO capacities in one stride change and one branch:
//   · MAT_MAX 32 -> 64. `mats` is a STORAGE buffer (6 144 -> 12 288 B), not a
//     uniform: zero new buffers, zero new bindings, zero new dispatches. The
//     tail rows are the INTERN CACHE — runtime-synthesised material rows keyed
//     by (base, modifier bitset), recorded as SCENE STATE (§25.2), never part
//     of the authored table. `matsHash` keeps covering exactly the authored
//     MATS and does not move; interned rows are covered by `sceneRowsDigest`.
//   · words 44/45 stop being reserved and become `tempTarget`/`tempDrive` —
//     the §12 spine's `coolRate` shape pointed at a per-row target instead of
//     AMBIENT: T += (tempTarget - T) * tempDrive >> 16 at thermal step 2.5.
//     This is the ONE new shader code path in the whole feature, and it is the
//     same field pair the §23-style Heater/Cooler devices need (two front
//     doors, one mechanism — design/staged/HEATER_COOLER.md).
// THE CHAIN MUST NOT MOVE: every authored row ships `tempDrive 0` (the branch
// is bit-for-bit inert) and MAT_MAX is a capacity constant (no golden scene
// carries matId >= 27). Expected golden diff: EXACTLY `specVersion` — matsHash
// keeps its `27x48:` prefix because the FNV runs over the authored MATS array,
// not the zeroed tail. Asserted by RUNNING: the determinism gate's golden
// comparison, and its new `internCount == 0` assertion on golden scenes.
// SPEC_VERSION 17 (2026-07-28) — §29 PRESSURIZED CONTAINMENT. `contain`
// geometrically closes adjacent sealed-solid segments after the final
// constraint solve; `containStress` releases finite-strength held wall
// particles under crush so METAL/STONE fail visibly rather than seep. Words
// 46/47 (reserved-must-be-zero through v16) become burstP/sealMode.
// The all-WATER golden takes both passes' inert branch; its state chain must
// remain byte-identical, with only specVersion and matsHash deliberately moving.
// SPEC_VERSION 18 (2026-07-28) — §30 INFLOW/OUTFLOW MATTER. The authored
// table and shader layout do not move. Interned rows may now retain I/O recipe
// bits and the host-owned matter-I/O service turns those simulated particles
// into contact sources/sinks. The golden state remains byte-identical; only
// the version field moves.
// SPEC_VERSION 19 (2026-07-28) — §31 OE-CAKE STRUCTURAL IMPORT FIDELITY.
// Imported Elastic now wears a SYNTH_VERSION 2 E row whose spring is 1.5x the
// native soft RUBBER row, matching a dense rubber lump / tight thin band rather
// than loose jelly. An import is also one armed fusion event, so touching
// bonded families — including Elastic against anchored Wall — form
// cross-material contact bonds on their first formation pass. WOOD is no
// longer a magic liquid-permeable membrane: every intact solid row seals, and
// its existing finite burst threshold remains the honest way pressure opens it.
// The all-WATER golden state chain is unchanged; specVersion and matsHash move.
// SPEC_VERSION 20 (2026-07-29) — §32 V10 MODULE CONSTRAINTS. The frozen
// 32-word body record stays fixed; a mask + nine-value driver sidecar is
// inserted before rest offsets in the rigid allocation. A zero mask is inert,
// and the all-WATER golden has no bodies, so its state chain, matsHash and
// tableHash remain byte-identical. The expected v19 -> v20 golden diff is
// exactly `specVersion`.
// SPEC_VERSION 21 (2026-07-29) — §33 SWEPT RIGID/SEAL CONTACT. Rigid members
// now fold crossings of adjacent sealed-solid segments into rigidSolve, so a
// one-particle rail is not a temporal/per-particle sieve. The all-WATER golden
// has no rigid body or seal-class matter; its state chain remains byte-identical.
// SPEC_VERSION 22 (2026-07-29) — §34 HARD BODY CONTACT + CONGEALING FUSION.
// Swept rail contacts are gathered from every grid cell the member traversed,
// and their hard correction is reduced over contacting members rather than
// diluted across the whole body. A newly cooled LAVA/MOLTEN particle can now
// fuse to a matching STONE/METAL neighbour even when that neighbour cooled on
// an earlier substep. The all-WATER golden state chain remains byte-identical.
// SPEC_VERSION 23 (2026-07-29) — ORBIES. Appended a first cohesive glass-bead
// material; this is retained as history because v24 deliberately reassigns it.
// SPEC_VERSION 24 (2026-07-29) — GEL + ORBIES. Goo's inert slot is now Gel and
// carries the former soft, contact-reforming bead mechanics. ORBIES becomes a
// separate non-bonding bead material with its own renderer path. The all-WATER
// chain is inert; matsHash deliberately moves because the authored table moved.
export const SPEC_VERSION       = 24;
export const SUBSTEPS_PER_FRAME = 2;
export const MAXNBR             = 48;
export const ITERS              = 4;
// ---- §18 bonded rigid bodies (v8) ------------------------------------------
// BOND_CAP serves THREE roles at once and it is worth knowing all three before
// changing it: the maximum bonds per particle, the bond ROW STRIDE (`i*BOND_CAP`
// at ~10 WGSL sites, in eraseWhere and in the hash fold), and it is deliberately
// == MAXNBR so one number cannot drift from two. Worst demand measured on a
// spec-valid scene is 30 (the §5.4 density cliff) => 1.60x headroom.
export const BOND_CAP           = 48;
// Formation runs on every BOND_PERIOD-th substep, or immediately after any
// spawn (`_bondDirty`). WIRE FORMAT: STONE is bit-identical at 1/2/4/8 but SAND
// is not, so this is part of the keystream exactly like ITERS and NCOLOUR.
export const BOND_PERIOD        = 4;
// 17-bit partner field (Darien's call). 131 071 is 1.64x past the measured
// ~80 000-particle 60 fps ceiling, so the field is never the binding limit —
// which is the property that was asked for. The assert is what turns a silently
// truncated partner index into a load-time error.
export const MAX_PARTICLES_BONDED = 131071;
export const BOND_J_MASK        = 0x0001ffff;
export const BOND_L0_MASK       = 0x3ffe0000;
export const BOND_TOMB          = 0x40000000;   // bit 30
// Q16 ceiling on bondK. At 1.0 the dpMax clamp binds on impact; at 2.0 the
// Jacobi bond solve over-relaxes (rg 5.70 -> 18.29).
export const BOND_K_MAX         = 32768;
// Particle.flags bits 8..19 — a 12-bit spawn-event id (§18.2). Bits 20..31 are
// left free for §17 vent phase 2's generation counter; that is a real coupling
// and a collision would silently weld two spawn events together.
export const WELD_TAG_SHIFT     = 8;
export const WELD_TAG_MASK      = 0x000fff00;
// The 12-bit id space, minus 0 (which means "no tag"). See §18.22 for the wrap
// argument, which a HELD tag changes and which `_nextWeldSeq` repairs.
export const WELD_SEQ_MAX       = 4095;
// The tag every `loadScene` particle is born with. One scene is one group.
export const SCENE_WELD_TAG     = (1 << WELD_TAG_SHIFT) >>> 0;
/**
 * §18.2 W1 — DOES THIS ROW WELD ON THE SPAWN EVENT? Table-derived, never a name
 * list, and it is the same predicate `index.html`'s emitter exclusion inverts.
 *
 * A `bondReform == 0` material has NO OTHER WAY to bond: the spawn weld is the
 * whole mechanism, and without it STONE/METAL/WOOD/ICE form exactly zero bonds
 * (measured: STONE drop width 42.40 wu, byte-identical to the bonds-off build).
 *
 * A `bondReform == 1` material — SAND, RICE — ALREADY has a mechanism, and it is
 * the one the material is for: grains stick when they come to rest against each
 * other, under compression, below a relative-velocity threshold. Giving those a
 * spawn weld as well is redundant, and it is not free: it makes a PAINTED blob
 * of sand a rigid body for as long as the weld holds. SPEC §18.21 Q1 parked that
 * as a taste call and measured the poured heap UNAFFECTED either way; three
 * shipped gates decide it. With the weld on every bondable row,
 * `elements.html` E2 measured painted SAND retaining 100 % of itself on a
 * pedestal against STONE's 33 % (sand must be the one that FLOWS), E2's
 * sand-through-water separation inverted, and E15's blast could no longer push
 * its own sand walls. Restricting the weld keeps all three, keeps the headline
 * (a painted STONE block IS a rigid body), and costs nothing measured.
 *
 * @param {number[]} row a MATS row
 */
export function spawnWelds(row) {
  return row[MAT_FIELDS.indexOf('bondGroup')] >= 0
      && row[MAT_FIELDS.indexOf('bondReform')] === 0;
}
// §5.1b. The number of solver colours, and — because the colour sweep order is
// part of the keystream (§11 R7) — a wire-format constant. 4 is the minimum
// that separates every INTER-cell interaction on the 4.0 wu grid; 9 was measured
// to remove exactly zero more and is pure cost. See SPEC §5.1b / §C17.
export const NCOLOUR            = 4;
export const ONE                = 65536;
export const RHO0               = 1048576;
export const PARTICLE_WORDS     = 8;   // 32 bytes
// v5: 16 words / 64 bytes. `corr` at word 14 pushes the struct to 60 bytes and
// vec2<i32>'s align-8 rounds that to 64, so word 15 is an EXPLICIT pad — see the
// note on `struct Derived` in sim.wgsl. Every host size derives from this
// constant; there are no bare 56s left.
export const DERIVED_WORDS      = 16;  // 64 bytes
export const DERIVED_BYTES      = DERIVED_WORDS * 4;
/**
 * Leading Particle words that participate in the state digest (§7.6).
 * v1-v3: 6 (temp and pad0 were reserved and provably never written).
 * v4:    8 — `temp` and `fuel` are live state. THIS MUST EQUAL PARTICLE_WORDS
 * for as long as every word carries meaning; determinism.html asserts it.
 */
export const HASHED_WORDS       = 8;
// Particle.flags bits (§12, §17). Bits 2..31 must be clear.
export const FLAG_BURNING       = 1;
// §17. Set on every particle whose material row has `phantom != 0`. It is a
// PURE FUNCTION of matId, recomputed from the table every substep by `thermal`,
// so it can never go stale and a slot that is reused for live matter clears it
// automatically. It is what the solver reads; `phantom` is what the table says.
export const FLAG_DEAD          = 2;
// §18.25. Engine-assigned, transient: thermal marks a particle that has just
// frozen into a bonded solid; the next bond formation pass consumes and clears
// it. It is deliberately absent from FLAG_MASK/SPAWN_FLAG_MASK, so scenes and
// callers cannot author a fake solidification event.
export const FLAG_CONGEAL       = 32;
// §29, v17. Engine-assigned fracture mark. A finite seal endpoint receives it
// when pressure releases its anchor/body constraint. Bond solving treats the
// marked endpoint as severed until it changes material; scenes and spawn calls
// cannot author it.
export const FLAG_BURST         = 64;
// §29. Pure row-derived cache used by buildNbr so its hot loop can record
// "this neighbour row contains a seal endpoint" without binding/loading the
// 48-word material table for every pair.
export const FLAG_SEAL          = 128;
// §20, v10. Bit 4: this particle is a MEMBER of a rigid body, and the weld tag
// in bits 8..19 names WHICH body. Bits 2 and 3 are reserved for SPEC19's
// FLAG_ANCHOR and FLAG_PIVOT so the three solid mechanisms can share one word.
//
// It is ENGINE-ASSIGNED and never authored: FLAG_MASK — the mask `loadScene`
// enforces on scene.flags — deliberately STAYS 3, so a scene cannot hand the
// solver a member of a body that has no record. Bodies are made by
// `makeRigidBody` / `promoteWeldGroup`, which write the flag and the record in
// the same call and can therefore never leave one without the other.
export const FLAG_RIGID         = 16;
// §19.A, v11. Bit 2: matter the USER PLACED that nothing can move. Unlike
// FLAG_RIGID this one IS authored — `loadScene` and `spawnFluid` both accept it
// — because an anchored particle needs no record anywhere: it is one bit and
// three early returns in the shader, with no body, no rest offset and no
// per-body state to get out of step with.
//
// Its value is NOT immobility (a boundary particle already has that) but
// MATERIALITY: `addWalls` gives every boundary particle matId = WATER, so the
// world floor makes sand and stone feel water friction. Anchored matter carries
// its own row into every pair term. See SPEC §19.A.
export const FLAG_ANCHOR        = 4;
export const FLAG_MASK          = 7;   // every currently-legal AUTHORED bit
/** The bits `spawnFluid` will take from a caller's `flags`. Deliberately a
 *  WHITELIST: FLAG_DEAD is derived from the row, the weld tag is owned by the
 *  gesture, and FLAG_RIGID must only ever be set together with a body record. */
export const SPAWN_FLAG_MASK    = FLAG_BURNING | FLAG_ANCHOR;

// §21 PER-PARTICLE COLOUR — the word format, in one place so the host, the
// importer, the renderer's WGSL and the gates cannot drift apart.
//
// RGBA8 little-endian: r | g<<8 | b<<16 | a<<24, STRAIGHT alpha (the importer
// un-premultiplies before it gets here). **a == 0 is the sentinel for "this
// particle has no colour of its own"**, which is why an all-zero buffer is
// exactly the pre-§21 picture and why the feature costs nothing when unused.
//
// This lives in engine.js and not render.js for one reason: colour is indexed
// by SLOT, and slots are the engine's to renumber.
export const TINT_NONE = 0;
export function packTint(r, g, b, a = 255) {
  const c = (v) => Math.max(0, Math.min(255, v | 0));
  return (c(r) | (c(g) << 8) | (c(b) << 16) | (c(a) << 24)) >>> 0;
}
export function unpackTint(w) {
  const u = w >>> 0;
  return { r: u & 255, g: (u >>> 8) & 255, b: (u >>> 16) & 255, a: (u >>> 24) & 255 };
}

// §20 — the solid axis. ONE material-table field, three mutually exclusive
// values. The exclusivity IS the design: rigid and elastic are opposite
// properties, and a row claiming both is a bug rather than a blend.
//   A RIGID BODY CAN LOSE MEMBERS BUT CANNOT DEFORM.
//   IF A MATERIAL SHOULD DEFORM, IT MUST NOT BE RIGID.
export const SOLID_NONE  = 0;
export const SOLID_BOND  = 1;
export const SOLID_RIGID = 2;

// §20 — the rigid buffer's three regions, in i32 WORDS. src/sim.wgsl's RG_*
// constants must agree with every one of these; test/rigid.html R0 compares the
// two lists by parsing the shader, because a silent disagreement here would
// place one body's members from another body's record.
export const RG_COS_BASE   = 0;        // 1026 entries, Q22 quarter turn
export const RG_COS_N      = 1026;
export const RG_BODY_BASE  = 1088;     // RG_COS_N rounded up to 64
export const RG_BODY_WORDS = 32;
export const RG_MAX_BODIES = 1024;
// Rung 4. Constraint drivers are a sidecar, not body-record growth: the body
// record remains 32 words, preserving native-scene snapshots and every §20
// offset. One ten-word record per body: mask + the nine clean-room slots.
export const RG_DRIVER_WORDS = 10;
export const RG_DRIVER_BASE = RG_BODY_BASE + RG_MAX_BODIES * RG_BODY_WORDS;
export const RG_REST_BASE = RG_DRIVER_BASE + RG_MAX_BODIES * RG_DRIVER_WORDS;
export const BODY_DRIVER = Object.freeze({
  MASK: 0, POSITION_X: 1, POSITION_Y: 2, ANGLE: 3,
  VELOCITY_X: 4, VELOCITY_Y: 5, SPIN: 6,
  FORCE_X: 7, FORCE_Y: 8, TORQUE: 9,
});
// body record words — the WHOLE dynamic state of a rigid body of any size
export const RB = {
  LIVE: 0, TAG: 1, FIRST: 2, LAST: 3, CX: 4, CY: 5, VX: 6, VY: 7,
  THETA: 8, OMEGA: 9, RESX: 10, RESY: 11, RESVX: 12, RESVY: 13,
  N: 14, SUMIS2: 15, GSUM: 16, INVI: 17, IQ: 18, RSH: 19, ISH: 20,
  OMEGAM: 21, RMAX: 22, OSH: 23, CQX: 24, CQY: 25, THETAQ: 26,
  // §19.B, v11. 1 = the centre is a PINNED PIVOT the user placed. See the long
  // note at src/sim.wgsl's RB_AXIS: an axis body is a §20 body with the two
  // translational DOFs removed, so it costs no dispatch, no buffer, no binding
  // and no per-particle bit — ONE record word and three branches.
  AXIS: 27,
  // §24, v14. 1 = the player's arrow keys drive this body. See the long note at
  // src/sim.wgsl's RB_USER: 92.5 % of the corpus's User particles sit on a Rigid
  // particle, so User is a label on a MACHINE, not a substance — one record word
  // and one addition inside the velocity fold that already exists.
  USER: 28,
};
export const RS_TARGET  = 2048;
export const SUMI_MAX   = 1 << 30;
export const NMEM_MAX_R = 16384;
/** below this many live members a body DISSOLVES into free particles (§20.4.4).
 *  The HARD floor is 2 (n = 1 gives sumIs2 = 0, invI = 0 — a one-particle rigid
 *  body has neither inertia nor shape). 4 is a FEEL constant: a 2- or 3-member
 *  remnant of a burned plank picks up 286x / 71x the angular rate of a
 *  12-member one from the same nudge, so the last splinters spin like sparks. */
export const RIGID_MIN_MEMBERS = 4;
// ---- §23.B — JETS ---------------------------------------------------------
/** How many jet devices the Params UNIFORM carries. Must equal `JET_MAX` in
 *  src/sim.wgsl. The host clamps to it and reports the clamp; nothing silently
 *  drops a jet. */
export const JET_MAX = 8;
/** Byte offsets inside Params. Scalars 0..56, `_padJet` at 60, then the two
 *  align(16) arrays. `PARAMS_BYTES` is what the buffer is allocated at. */
export const JET_N_OFF   = 56;
export const JET_POS_OFF = 64;
export const JET_F_OFF   = JET_POS_OFF + JET_MAX * 16;
/** The shift `jet_accel` applies before squaring a tick difference. Mirrored in
 *  src/sim.wgsl and test/mirror/cpu_mirror.mjs; all three must agree or the
 *  circle has a different edge on each. */
export const JET_SH = 9;
// ---- §24 — USER: ARROW-KEY MATTER -----------------------------------------
/** Byte offset of the player's drive vector (two i32) inside Params. It sits
 *  AFTER §23.B's two arrays because a uniform struct's ARRAY members must start
 *  on a 16-byte boundary while trailing scalars need only 4 — so appending is
 *  the change that moves nothing already working. Two pad words follow so the
 *  struct's size stays a multiple of 16 (PLATFORM_NOTES §23.B rule 2: a size
 *  mismatch is an ASYNCHRONOUS validation error that reads back as stable
 *  zeros, i.e. as a sim that quietly stopped). */
export const USER_A_OFF = JET_F_OFF + JET_MAX * 16;     // 320
export const PARAMS_BYTES = USER_A_OFF + 16;            // 336
/** The drive is clamped to this magnitude, in ticks/substep^2 — the same units
 *  as `Mat.grav` (WATER is 1200) and as a jet's `fx`/`fy`. The bound is §20's
 *  own per-term clamp: `rigidStep` clamps every member force to +/-32768 before
 *  summing, and the drive is added to the same velocity a force sum feeds, so
 *  keeping it inside the same envelope keeps §20's overflow proof exactly as
 *  proved. Clamped HERE and not in the shader, because a host that can write a
 *  fractional or unbounded uniform is a determinism hazard and this is the only
 *  place it can be caught. */
export const USER_A_MAX = 32768;
export const BAM_QUARTER = 1073741824;
export const BAM_PER_RAD = 683565276;
export const RG_QBITS = 10;
export const RG_QN    = 1 << RG_QBITS;
export const RG_QSH   = 30 - RG_QBITS;
export const COS_Q    = 22;
export const COS_ONE  = 1 << COS_Q;
/** §20. The window `i64_shr_rne_sat` is DEFINED on, minus the hi/lo split's 15.
 *  Not a taste choice — outside 1..31 that helper's own shifts are
 *  indeterminate in WGSL. src/sim.wgsl's RG_IQ_MIN / RG_IQ_MAX must match. */
export const RG_IQ_MIN = 16;
export const RG_IQ_MAX = 31;

// The frozen-table hash: 4-lane Murmur3 over the decoded bytes of
// src/tables.json. Verified at every load; a mismatch is a hard throw, because
// a changed table entry silently changes every hash thousands of substeps later
// and would invalidate every glyph ever encoded (spec §3.4, risk R7).
// This value was computed independently of src/tables.json's own `tableHash`
// field and the two agree, so the file is self-consistent.
export const TABLE_HASH = '31ebf1f3e1b168a800f0f180007dd390d0dc182fd06e398adfeec08e8a62908b';

// ---------------------------------------------------------------- world consts
export const H              = 163840;      // kernel radius, ticks (2.5 wu)
export const WALL           = 32768;       // 0.5 wu
export const CELL_SHIFT     = 18;          // 4.0 wu cells
export const TN             = 3202;        // entries per kernel table
export const LUT_TABLES     = 5;
export const LUT_WORDS      = LUT_TABLES * TN;   // 16010 i32 = 64040 bytes
export const MAX_CELL_TOTAL = 1048576;     // hard limit of the 2-level scan
export const VMAX           = 65536;       // ticks/substep, per component
export const MAX_POS        = 134217728;   // 2^27

// ============================================================================
// §12 THERMAL CONSTANTS — the host mirror of sim.wgsl's block. Both files
// declare these; test/thermal.html asserts they agree, because a silent
// disagreement here is a wrong-physics bug with no symptom.
// ============================================================================
/** Temperature is Q16.16 "aether degrees": ONE tick-scale, 1 degree = 65536. */
export const TEMP_ONE  = 65536;
export const AMBIENT   = 20 * TEMP_ONE;       //   1310720
export const TEMP_MIN  = -33554432;           //  -2^25  = -512 deg
export const TEMP_MAX  = 536870912;           //   2^29  = 8192 deg
export const DT_CLAMP  = 16777216;            //   2^24  =  256 deg (flux cap)
export const HT_TERM   = 16777216;            //   2^24  per-term heat clamp
/** Conductivity ceiling. See MATS below for why it is 0.5 and not 1.0. */
export const COND_MAX  = 32768;
// §16 granular friction ceiling: 2.0 in Q16. NOT 1.0 and NOT 4.0, and both
// bounds are measured.
//
// ABOVE 1.0 IS NECESSARY. One Jacobi friction pass weighted by W/rho0 sums to
// ~0.8 over a full neighbourhood, so mu = 1.0 cancels only ~80% of the pair's
// tangential relative displacement and the rest creeps. The shape-retention
// sweep keeps improving past 1.0 (see the MATS header).
//
// ABOVE 2.0 IS NOT SAFE, and the reason is the dpMax clamp, not the friction
// term itself. MEASURED — 196 SAND particles, gravity 0, no walls, 800
// substeps, drift of sum(v) against the single-rounding random-walk floor of
// sqrt(196*800)*0.5 = 198:
//     mu    0     0.5    1.0    2.0    3.0
//   drift  169    719    941   1216  23883      (dpMax 16384, the shipped value)
//   drift  169    719    941   1216    439      (dpMax 65536, 4x)
//   drift  169    719    941   1216  45824      (dpMax  4096, 1/4x)
// At mu = 3.0 the friction correction routinely EXCEEDS dpMax, and a clamp that
// binds on one end of a pair and not the other destroys f_ij == -f_ji: the leak
// jumps 20x and scales inversely with dpMax, which is the signature of the
// clamp and not of the accumulator. mu = 3.0 is also where a low-shear blob
// stopped being stable (it grew 15 -> 56 wu instead of holding).
// Up to 2.0 the residue is 7x the floor, bounded, monotone, and of the same
// class as XSPH's known one (§11 R11). test/granular.html G4 gates it with the
// frictionless control run on the identical scene.
export const FRIC_MAX  = 131072;
export const XSPH_MAX  = 45000;
/** §13 corrosion strength ceiling, Q16. Mirrors CORR_MAX in sim.wgsl. */
export const CORR_MAX  = 262144;
/** Degrees -> ticks. Integer degrees only, so this is exact on every host. */
export const DEG = (d) => d * TEMP_ONE;
/** A transition this material does not have. Out of reach, not a flag. */
export const NEVER_HOT  =  2147483647;        // T >= this is impossible (T <= 2^29)
export const NEVER_COLD = -2147483648;        // T <= this is impossible (T >= -2^25)

// ============================================================================
// THE MATERIAL TABLE (§6.2 / §12) — 32 i32 words per row, MAT_MAX = 32 rows.
// ----------------------------------------------------------------------------
// ADDING A MATERIAL IS ONE ROW HERE. Nothing in sim.wgsl, the CPU mirror or the
// engine needs to know it exists.
//
// Rows are packed BY POSITION into `struct Mat`, so the buffer is an array of
// integers — but authoring 32 bare numbers per row is how you get a table where
// `gammaCur` is silently in `adhesion`'s slot. `mat({...})` takes named fields,
// fills every default, rejects an unknown key, and returns the positional
// Array the packer wants. Use it. A raw array still works and is still checked.
//
// FIELD REFERENCE (index: name — meaning)
//   MECHANICS (v3 block, order frozen)
//     0 eps        Q21 CFM relaxation. MUST be >= 32768: it is the only term
//                  guaranteeing fp_divshift's den >= 2^15 when S collapses (§2.3).
//     1 dpMax      ticks, max |dp| per solver iteration. 16384 = 0.25 wu.
//     2 xsph       Q16 viscosity, [0, XSPH_MAX]. water 6000, goo 30000.
//     3 phantom    §17, v7. 0 or 1. 1 means THIS ROW IS NOT MATTER: a particle
//                  wearing it is retired — it is skipped by every pair loop, it
//                  is frozen where it fell, and it is invisible. It was `vort`,
//                  RESERVED-must-be-0 since v3 and never implemented, so every
//                  v3..v6 row already reads as `phantom = 0 = real matter` and
//                  the mats buffer bytes for rows 0-23 are unchanged.
//                  createEngine REFUSES a phantom row that carries any force,
//                  any conduction or any transition — a phantom with gravity is
//                  a ghost, and that throw is what stops it.
//     4 gammaCoh   ticks/substep^2 cohesion — "how badly it wants to be a drop".
//     5 gammaCur   ticks/substep^2 curvature. KEEP AT ~gammaCoh/4: at equal
//                  values a resting disc expands 5.7x (measured, §6.2).
//     6 adhesion   ticks/substep^2 wall wetting. NOT pair-mixed (§6.2).
//     7 grav       ticks/substep^2. MAY BE NEGATIVE — this is the buoyancy knob.
//   THERMAL
//     8 cond       Q16 conductivity, [0, COND_MAX = 32768].
//                  WHY 0.5 AND NOT 1.0: conduction is an EXPLICIT diffusion and
//                  its per-substep coefficient is cond * sum_j(W_ij/RHO0). At
//                  rest sum_j W_ij/RHO0 = 0.795 (= 1 - W(0)/RHO0, measured from
//                  the frozen tables), but under 2x compression it reaches ~1.8.
//                  cond = 0.5 keeps the coefficient at 0.9 < 1 even there, so
//                  the scheme cannot ring. cond = 1.0 could.
//     9 coolRate   Q16 relaxation toward AMBIENT per substep. Time constant in
//                  substeps = 65536 / coolRate. 300 -> 218 substeps ~ 1.8 s.
//                  This is the box leaking heat to the room; it deliberately
//                  does NOT conserve energy. Set 0 for a thermally sealed test.
//    10 heatCap    RESERVED, must be 0. v4 assumes uniform heat capacity, which
//                  is exactly what makes conduction conserve sum(T) to the bit.
//   PHASE CHANGE  (threshold, product) pairs — sentinel = no such transition
//    11 meltPt   12 meltTo      T >= meltPt   (NEVER_HOT  to disable)
//    13 freezePt 14 freezeTo    T <= freezePt (NEVER_COLD to disable)
//    15 boilPt   16 boilTo      T >= boilPt   (NEVER_HOT  to disable)
//    17 condPt   18 condTo      T <= condPt   (NEVER_COLD to disable)
//   COMBUSTION
//    19 ignitePt   T >= ignitePt && fuel > 0 sets FLAG_BURNING. TEMP_MIN = always.
//    20 fuel0      fuel a fresh particle carries. 0 = non-flammable, and it is
//                  the ONLY thing that makes a material non-flammable.
//    21 burnRate   fuel consumed per substep. Burn time = fuel0 / burnRate.
//    22 burnHeat   temp ticks added to SELF per substep while burning.
//                  FLAME TEMPERATURE IS AN EQUILIBRIUM, not a setting:
//                     T_flame = AMBIENT + burnHeat * 65536 / coolRate
//                  (conduction to neighbours pulls it below that). Pick the
//                  flame temperature you want, then solve for burnHeat.
//    23 burnTo     material once fuel is exhausted.
//   PRODUCT SPLIT — applies to WHICHEVER transition fires
//    24 splitTo    secondary product.
//    25 splitPct   0..256 out of 256 take splitTo instead. 0 disables it and
//                  the branch is bit-for-bit inert. The choice is a pure
//                  integer hash of the particle id — never an RNG.
//   HOST-ONLY (the shader reads the words and ignores them)
//    26 spawnTemp  temperature a UI-painted particle of this material starts at.
//    27 tint       0x00RRGGBB hint for render.js. Optional; ignore it freely.
//   CORROSION (§13, v5) — the one axis that is NOT temperature
//    28 corrode    Q16, [0, CORR_MAX]. Attack this material inflicts on any
//                  DIFFERENT material in kernel range. 0 = not corrosive, and 0
//                  is what makes the whole branch bit-for-bit inert.
//    29 corrPt     dose at which this material becomes corrTo. NEVER_HOT = immune.
//                  Dose scale: a particle COMPLETELY surrounded by an attacker of
//                  strength S accumulates ~0.795 * S per substep (sum_j W_ij/RHO0
//                  at rest density, measured from the frozen tables), so a
//                  corrPt of 0.2 * S means "roughly a quarter of my neighbourhood
//                  is acid". Thresholds are per-substep RATES, not doses over
//                  time: there is no per-particle corrosion accumulator, by
//                  design — one would need a ninth Particle word.
//    30 corrTo     product.
//    31 fric       Q16 Coulomb friction, [0, FRIC_MAX = 65536]. §16, v6.
//                  THE ONLY THING IN THE TABLE THAT MAKES A MATERIAL HOLD A
//                  SHAPE. PBF's density constraint is compression-only: it can
//                  push two particles apart and can do NOTHING to stop them
//                  sliding past each other, so without this every material in
//                  the roster spreads to a one-particle film and stops.
//                  MEASURED, an identical 113-particle blob per material,
//                  p05-p95 width after 4 s in a 128 wu tank, fric = 0
//                  everywhere: 24 of 24 materials past 109 wu, WOOD and STONE
//                  landing on identical numbers.
//                  0 = frictionless AND makes the `friction` pass bit-for-bit
//                  inert for this material — which is why WATER and GOO sit at
//                  0 and every v5 hash still reproduces.
//
//   THE ONE CORROSION INVARIANT createEngine ENFORCES: an attacker must itself be
//   immune (corrode > 0 => corrPt == NEVER_HOT). Two mutually-corrosive materials
//   would make the SPEND term above fire on both sides of the same pair, which is
//   defined but confusing, and it is not a thing any material in the roster wants.
// ============================================================================
// v8 (§18): 48. The bond block appends at word 32, so words 0..31 of every row
// keep their exact bytes; stride 128 -> 192 B. test/layout.html asserts the
// stride and the zero implicit padding, because getting it wrong would be
// silent (PLATFORM_NOTES: vec3<T> is size 12 / align 16 — every word in the
// block is a 4-byte scalar precisely so offsets stay 4*index).
export const MAT_WORDS = 48;
// v16 (§25): 64, was 32 through v15. The widening is the INTERN CACHE's sizing
// knob: rows [MATS.length, MAT_MAX) hold runtime-synthesised composed
// materials (36 spare at 28 authored rows >= 21, the most mixed recipes any
// real .oec ever used, so corpus import NEVER projects). `mats` is a storage
// buffer, so this is +6 KiB and zero new bindings; `matsHash` is over the
// authored array and does not move.
export const MAT_MAX   = 64;

export const MAT_FIELDS = [
  'eps', 'dpMax', 'xsph', 'phantom', 'gammaCoh', 'gammaCur', 'adhesion', 'grav',
  'cond', 'coolRate', 'heatCap',
  'meltPt', 'meltTo', 'freezePt', 'freezeTo', 'boilPt', 'boilTo', 'condPt', 'condTo',
  'ignitePt', 'fuel0', 'burnRate', 'burnHeat', 'burnTo',
  'splitTo', 'splitPct', 'spawnTemp', 'tint',
  'corrode', 'corrPt', 'corrTo', 'fric',
  // §18 bonds. APPENDED at word 32 — words 0..31 of every existing row keep
  // their exact bytes, so a v7 row still reads correctly for everything that is
  // not a bond. `rampInv` is BAKED by createEngine, never authored.
  'bondK', 'bondBreak', 'bondReform', 'bondFormV', 'bondFormP', 'bondMu',
  'bondGroup', 'rampLo', 'rampHi', 'rampInv', 'rampSign',
  // §20, v10. Word 43 was `rsv4` (RESERVED, must be 0) through v9, so every
  // v9 row reads as SOLID_NONE and the field can only change behaviour for a
  // row that opts in. The SHADER reads it in exactly one place — the leave
  // test in `rigidMember` — and the HOST reads it to decide whether a
  // finished gesture becomes a body at all.
  // §25, v16. Words 44/45 were `rsv5`/`rsv6` (RESERVED, must be 0) through
  // v15, so every v15 row reads as `tempDrive 0` and the drive branch is
  // bit-for-bit inert on all of them — the same landing pattern as `phantom`
  // (word 3) and `solidMode` (word 43). `tempTarget`/`tempDrive` are the §12
  // spine's `coolRate` shape pointed at a per-row target instead of AMBIENT:
  // a heater is a row that pulls its own temperature toward DEG(900) forever
  // (a born-hot row merely SPENDS its heat — conduction conserves sum(T)).
  // §29, v17. Words 46/47 were reserved through v16. They modify a row already
  // classified as seal-capable; they never grant sealing by themselves.
  'solidMode', 'tempTarget', 'tempDrive', 'burstP', 'sealMode',
];
export function sealRow(row) {
  return row[MAT_FIELDS.indexOf('bondK')] !== 0 &&
         row[MAT_FIELDS.indexOf('bondReform')] === 0 &&
         row[MAT_FIELDS.indexOf('solidMode')] !== SOLID_NONE &&
         row[MAT_FIELDS.indexOf('phantom')] === 0;
}

const MAT_DEFAULTS = {
  eps: 40000, dpMax: 16384, xsph: 6000, phantom: 0,
  gammaCoh: 240, gammaCur: 60, adhesion: 0, grav: 1200,
  cond: 0, coolRate: 0, heatCap: 0,
  meltPt: NEVER_HOT, meltTo: 0, freezePt: NEVER_COLD, freezeTo: 0,
  boilPt: NEVER_HOT, boilTo: 0, condPt: NEVER_COLD, condTo: 0,
  ignitePt: NEVER_HOT, fuel0: 0, burnRate: 0, burnHeat: 0, burnTo: 0,
  splitTo: 0, splitPct: 0, spawnTemp: AMBIENT, tint: 0x8090a0,
  corrode: 0, corrPt: NEVER_HOT, corrTo: 0, fric: 0,
  // §18. bondK = 0 makes the WHOLE bond term bit-for-bit inert for a row — the
  // gather never enters, the formation pass proposes nothing, the hash fold sees
  // an empty row. 20 of the 26 rows ship at 0 and that is what keeps the
  // all-WATER golden chain reproducible. bondGroup = -1 is "never bonds"; the
  // validator refuses bondK != 0 with bondGroup < 0.
  bondK: 0, bondBreak: 0, bondReform: 0, bondFormV: 0, bondFormP: 0, bondMu: 0,
  bondGroup: -1, rampLo: 0, rampHi: 65536, rampInv: 0, rampSign: 0,
  solidMode: SOLID_NONE, tempTarget: 0, tempDrive: 0, burstP: 0, sealMode: 0,
};

/** Named fields -> the positional Array the GPU packer needs. */
export function mat(o = {}) {
  for (const k of Object.keys(o)) {
    if (!(k in MAT_DEFAULTS)) {
      throw new Error(`[aether] mat(): unknown material field "${k}". Known: ${MAT_FIELDS.join(', ')}`);
    }
  }
  const row = MAT_FIELDS.map((f) => (o[f] !== undefined ? o[f] : MAT_DEFAULTS[f]));
  // gammaCur defaults to gammaCoh/4 when only gammaCoh was given — the ratio
  // §6.2 says must hold, applied by default so it cannot be forgotten.
  if (o.gammaCoh !== undefined && o.gammaCur === undefined) {
    row[MAT_FIELDS.indexOf('gammaCur')] = (o.gammaCoh / 4) | 0;
  }
  // §18. `rampInv` is DERIVED, never authored — bond_k(M, T) is one multiply and
  // one shift precisely because this divide happens once, here, on the host.
  // Whatever the caller passed is overwritten, which is the point: two rows with
  // the same ramp edges cannot disagree about the reciprocal.
  {
    const iLo = MAT_FIELDS.indexOf('rampLo'), iHi = MAT_FIELDS.indexOf('rampHi');
    const span = row[iHi] - row[iLo];
    row[MAT_FIELDS.indexOf('rampInv')] = span > 0 ? Math.floor(2 ** 32 / span) : 0;
  }
  return row;
}

// ---- material ids. Index into MATS; also the value stored in Particle.matId.
export const MAT_WATER  = 0;
// GEL keeps Goo's wire id so old scenes remain mechanically addressable. Keep
// MAT_GOO as a source-compatible alias for the .oec importer and old helpers;
// user-facing names, palettes and manifests say GEL exclusively.
export const MAT_GEL    = 1;
export const MAT_GOO    = MAT_GEL;
export const MAT_ICE    = 2;
export const MAT_STEAM  = 3;
export const MAT_OIL    = 4;
export const MAT_FIRE   = 5;
export const MAT_SMOKE  = 6;
export const MAT_WOOD   = 7;
export const MAT_ASH    = 8;
export const MAT_LAVA   = 9;
export const MAT_STONE  = 10;
export const MAT_SAND   = 11;
export const MAT_METAL  = 12;
export const MAT_MOLTEN = 13;
export const MAT_GAS    = 14;
export const MAT_PHOTO  = 15;
// v5 (§13). Ids are APPENDED, never inserted: matId is wire format (§11 R7) and
// every saved scene, every golden and every glyph payload would re-version if a
// row moved. 22 of MAT_MAX = 32 rows are live.
export const MAT_CRYO    = 16;
export const MAT_RUBBER  = 17;
export const MAT_TAR     = 18;
export const MAT_MERCURY = 19;
export const MAT_DUST    = 20;
export const MAT_ACID    = 21;
// v6 (SPEC §15). The reactor pair — see their rows below. 24 of MAT_MAX = 32 live.
export const MAT_NITRO   = 22;
export const MAT_BLAST   = 23;
// v7 (SPEC §17). NOT A MATERIAL — the absence of one. It is the only row in the
// table whose job is to not be there, it is not paintable, it has no keyboard
// letter and no look. Everything that presents materials to a player filters on
// `phantom`, so this row never has to be named in a list anywhere.
// 24 live rows + 1 phantom of MAT_MAX = 32.
export const MAT_VOID    = 24;
// v8 (SPEC §18). RICE — raw it pours, cooked it sets. The first material whose
// entire thermal identity lives in the BOND block rather than in a phase
// transition. 25 live rows + 1 phantom of MAT_MAX = 32.
export const MAT_RICE    = 25;
// v15 (SPEC §26.F). BEDROCK — the row nothing happens to. Darien: "Do we maybe
// need a completely inert material to build indestructible walls/wheels/rigid
// bodies?" 26 live rows + 1 phantom of MAT_MAX = 32.
export const MAT_BEDROCK = 26;
// v23 (§35). ORBIES — clear, sticky, elastic beads. Appended so existing
// material IDs remain their wire-format values; composed rows begin at 28.
export const MAT_ORBIES  = 27;

export const MAT_NAMES = [
  'WATER', 'GEL', 'ICE', 'STEAM', 'OIL', 'FIRE', 'SMOKE', 'WOOD',
  'ASH', 'LAVA', 'STONE', 'SAND', 'METAL', 'MOLTEN', 'GAS', 'PHOTO',
  'CRYO', 'RUBBER', 'TAR', 'MERCURY', 'DUST', 'ACID', 'NITRO', 'BLAST',
  'VOID', 'RICE', 'BEDROCK', 'ORBIES',
];

/** True if material id `m` is retired matter rather than a material (§17). */
export function isPhantom(m, mats = MATS) {
  const row = mats[m];
  return !!row && row[MAT_FIELDS.indexOf('phantom')] !== 0;
}
/** The ids a player may paint / see / be told about: everything that is matter. */
export function liveMaterialIds(mats = MATS) {
  const out = [];
  for (let i = 0; i < mats.length; i++) if (!isPhantom(i, mats)) out.push(i);
  return out;
}

// ============================================================================
// §25 (v16) — COMPOSABLE PROPERTY BITS: synth(), the interned-row algebra
// ----------------------------------------------------------------------------
// A composed material ("viscous water", "brittle elastic goo") is NOT a new
// authored row and NOT a per-particle word (measured 1.57-1.74x on the pair
// loop — SPEC23_DRAFT §25.1c). It is an INTERNED ROW: a runtime-synthesised
// 48-word row living in the mats tail [MATS.length, MAT_MAX), keyed by
// (baseMatId, modifier bitset), derived by the PURE function `synth()` below.
// Purity is the whole determinism story: a replay records the mint list and
// re-derives every row at the recorded SYNTH_VERSION, refusing the load on any
// disagreement. `matsHash` keeps covering exactly the authored table; interned
// rows are covered by `sceneRowsDigest` (a different instrument for a
// different thing: scene state, not authorship).
//
// SYNTH_VERSION is WIRE FORMAT (§11 R7 pattern): any constant below moving is
// a SYNTH_VERSION bump, and old versions stay executable forever.
// ============================================================================
export const SYNTH_VERSION = 4;

// Modifier/recipe bits. Mechanical matter-algebra bits live here, plus §30's
// I/O behavior bits because Inflow/Outflow are now properties of an interned
// matter recipe. W/X/R/U/J/Z remain body/device paths and are deliberately not
// row algebra. YPROD is the products-closure bit for
// Snow's one-way melt ("snow-water cannot be frozen back" — wiki 'Snow'); it
// is minted BY the closure, never offered in any UI.
export const MOD_BITS = Object.freeze({
  V: 1 << 0,   // Viscous — xsph x4, clamped; Rung 3 real-GPU bulk/film tuning
  T: 1 << 1,   // Tensile   — gammaCoh x4, adhesion 0
  D: 1 << 2,   // Dense     — grav +600 (the gravity substitution, §25.8 #6 ACCEPTED)
  L: 1 << 3,   // Light     — grav -600 (D+L cancels EXACTLY, the wiki law)
  E: 1 << 4,   // Elastic   — the §18.23 RUBBER bond block verbatim
  B: 1 << 5,   // Brittle   — bondBreak -> eps 0.10; inert without a bond template
  M: 1 << 6,   // Mochi     — contact re-forming sticky bonds
  K: 1 << 7,   // Rice      — RICE's block, thermal gate OFF (always cooked)
  S: 1 << 8,   // String    — weak distance net (ours linear; theirs exponential, declared)
  H: 1 << 9,   // Heater    — tempTarget DEG(900), tempDrive 300
  C: 1 << 10,  // Cooler    — tempTarget DEG(-150), tempDrive 300
  Y: 1 << 11,  // Snow      — ICE-like row whose melt is ONE-WAY (products closure)
  YPROD: 1 << 12, // snow-water: WATER that never freezes. Closure-only.
  I: 1 << 13,  // Inflow    — particle source; mechanics are host-authored
  O: 1 << 14,  // Outflow   — contact sink with shared-ingredient protection
  IO_NULL: 1 << 15, // pure-I payload marker. Internal; never a palette toggle.
  FLOAT: 1 << 16,   // import-only textured rigid shell: tuned buoyant weight
});
export const MOD_LETTERS = 'VTDLEBMKSHCYIO';   // UI-offerable, in bit order

/** Stable A..Z ingredient bit. OE-CAKE recipes are 26-bit sets, which fit in
 * one unsigned host word and make "shares any ingredient" an exact AND. */
export const ELEMENT_BITS = Object.freeze(Object.fromEntries(
  Array.from({ length: 26 }, (_, i) => [String.fromCharCode(65 + i), (1 << i) >>> 0])));
export function elementMask(letters) {
  let out = 0;
  for (const c of letters ?? []) if (ELEMENT_BITS[c] !== undefined) out |= ELEMENT_BITS[c];
  return out >>> 0;
}

// The standard ingredient recipe represented by each authored row. Rows with
// no honest OE-CAKE ingredient analogue deliberately map to zero: a fallback
// STONE body used to carry WO must not accidentally claim a hidden R/Stone
// ingredient and protect unrelated stone from that Outflow.
const AUTHORED_ELEMENT_RECIPES = [
  'Q', 'V', 'QY', 'GLQ', 'F', 'FH', 'G', 'F', '', 'H', '', 'P', 'D', 'DH',
  'GL', 'A', 'C', 'E', 'M', 'T', 'GP', '', 'FP', 'FP', '', 'KE', '',
];
export function authoredElementMask(matId) {
  return elementMask(AUTHORED_ELEMENT_RECIPES[matId] ?? '');
}

/** "QV"-style canonical letters for a bitset, closure bits in lowercase. */
export function modLetters(bits) {
  let s = '';
  for (const c of MOD_LETTERS) if (bits & MOD_BITS[c]) s += c;
  if (bits & MOD_BITS.YPROD) s += 'y';
  if (bits & MOD_BITS.IO_NULL) s += '∅';
  if (bits & MOD_BITS.FLOAT) s += 'f';
  return s;
}

// THE SYNTH_VERSION 1 CONSTANTS. Read out of this source text by the gate
// (test/properties.html A1, the T0 pattern) so a retyped copy cannot drift.
// Every number is either measured, derived from a shipped row, or graded OURS
// in SPEC23_DRAFT §25.4 (the wiki has no number to give).
export const SYNTH1 = Object.freeze({
  // Rung 3, real shader, N=8 relative 0..7-tick phases. WATER bulk-shear
  // memory: 0.727 base, 0.439 at ×3 (miss), 0.321 at ×4 (pass). A one-particle
  // V film between zero-xsph surfaces retains 1.000 / 0.284 / 0.207 of slide
  // at 0 / WATER×3 / WATER×4 (×3 misses the <=0.25 band, ×4 passes). Every sd
  // < 0.001, no clamps, no monotonic inversions. `viscous_sweep.html` owns the
  // instrument and its negative controls. Wiki supplies behavior, not numbers.
  V_XSPH_MUL: 4, V_XSPH_CLAMP: 45000,      // = XSPH_MAX; GOO x4 clamps here
  T_COH_MUL: 4,                             // WATER 240 -> 960 ~ MERCURY's 950
  DL_GRAV: 600,                             // +-600; GAS = pureG(0) + L(-600) derives the table
  E_BONDK: 16384, E_BONDBREAK: 409600,      // RUBBER's block verbatim (eps 1.50)
  B_BONDBREAK: 79299,                       // eps 0.10 (§25.8 #7 feel-check pending)
  M_BONDK: 8192, M_ADHESION_MIN: 260,       // contact re-form is what mochi IS
  S_BONDK: 8192, S_BONDBREAK: 165000,       // eps ~0.59, ours to tune
  H_TARGET: DEG(900), C_TARGET: DEG(-150), HC_DRIVE: 300,   // tau ~218 substeps
});

// v2 changes one import-visible property and nothing else: E is a tight rubber
// spring rather than a byte-for-byte alias of the deliberately soft palette
// RUBBER row. SYNTH1 remains executable for recorded scene rows.
export const SYNTH2 = Object.freeze({
  ...SYNTH1,
  E_BONDK: 24576,
});
// v3 adds one import-only structural property. FLOAT is not a palette element:
// OE-CAKE v0 textured rigid props store appearance/mass outside the particle
// recipe, and duck.oec is the canonical buoyant example. v4 corrects the first
// approximation from weightless (which made a body hang in air and then rise
// too eagerly) to the measured ICE-like 1050/1200 gravity ratio: it falls
// freely in air but remains lighter than water.
export const SYNTH3 = Object.freeze({ ...SYNTH2 });
export const SYNTH4 = Object.freeze({ ...SYNTH3, FLOAT_GRAV: 1050 });

// Bond-template precedence when several of E/M/K/S are present (EKNV is a
// top-20 recipe): K > M > E > S, then B applies. FROZEN at SYNTH_VERSION 1 —
// replays depend on it (SPEC23_DRAFT §25.4).
const BOND_TEMPLATE_PRECEDENCE = ['K', 'M', 'E', 'S'];

// The one composed-tint algebra, host-only (render.js reads the word, the
// shader never does). Deliberately simple; the LOOK is parked for Darien's
// /taste pass — this only has to make a composed swatch visibly not-the-base.
function composeTint(tint, bits) {
  // Classic's bare Cooler preset is an instrument, not a subtle temperature
  // hint: it should lay unmistakable solid blue matter in the world.
  if (bits === MOD_BITS.C) return 0x0000ff;
  let r = (tint >>> 16) & 255, g = (tint >>> 8) & 255, b = tint & 255;
  if (bits & MOD_BITS.V) { g += 24; }
  if (bits & MOD_BITS.T) { r += 16; g += 16; b += 16; }
  if (bits & MOD_BITS.D) { r -= 24; g -= 24; b -= 24; }
  if (bits & MOD_BITS.L) { r += 24; g += 24; b += 24; }
  if (bits & MOD_BITS.E) { r += 20; b += 20; }
  if (bits & MOD_BITS.B) { r += 20; g += 10; }
  if (bits & MOD_BITS.M) { r += 24; g += 12; b -= 12; }
  if (bits & MOD_BITS.K) { r += 12; g += 12; }
  if (bits & MOD_BITS.S) { b += 28; }
  if (bits & MOD_BITS.H) { r += 48; b -= 24; }
  if (bits & MOD_BITS.C) { b += 48; r -= 24; }
  if (bits & (MOD_BITS.Y | MOD_BITS.YPROD)) { r += 16; g += 24; b += 24; }
  if (bits & MOD_BITS.I) { g += 42; b += 18; }
  if (bits & MOD_BITS.O) { r += 42; b += 18; }
  const cl = (x) => Math.max(0, Math.min(255, x | 0));
  return (cl(r) << 16) | (cl(g) << 8) | cl(b);
}

/**
 * §25.4 — THE COMPOSITION ALGEBRA. Pure: (authored table, baseId, bits,
 * version) -> { words, needs, dropped }. Never touches an engine, never
 * allocates a rowId, never writes a buffer — `Engine.internMaterial` does the
 * impure half. Application order is wire format: bond template -> Brittle ->
 * scalars (V, T, D, L) -> thermal (H, C) -> products closure.
 *
 * `words[F('bondGroup')]` is left at -2 as a SENTINEL when the row bonds: the
 * interner replaces it with the fresh unique group 64 + rowId (§25.3 — the F1
 * invariant extends to interned rows; sharing a group would invalidate every
 * §18 number).
 *
 * `needs` lists product rows the closure requires: [{ field, base, bits }].
 * `dropped` names bits that were accepted-but-annulled (H+C) for the manifest.
 *
 * Throws on a row that would destroy itself (a heater above its own meltPt) —
 * the same contract createEngine enforces on authored rows.
 */
export function synth(baseId, bits, mats = MATS, version = SYNTH_VERSION) {
  if (version !== 1 && version !== 2 && version !== 3 && version !== 4) {
    throw new Error(`[aether] synth: unknown SYNTH_VERSION ${version} (this build speaks 1..4)`);
  }
  const C = version === 1 ? SYNTH1 : version === 2 ? SYNTH2 :
    version === 3 ? SYNTH3 : SYNTH4;
  const F = (f) => MAT_FIELDS.indexOf(f);
  assertFn(Number.isInteger(baseId) && baseId >= 0 && baseId < mats.length,
    `synth: baseId ${baseId} is not an authored material`);
  assertFn(!isPhantom(baseId, mats), `synth: base ${MAT_NAMES[baseId] ?? baseId} is phantom`);
  assertFn(Number.isInteger(bits) && bits > 0 && bits < (1 << 17),
    `synth: bits ${bits} outside the modifier bitset`);
  const has = (letter) => (bits & MOD_BITS[letter]) !== 0;

  // Y swaps the MECHANICAL base to ICE before everything else: Snow is an
  // ICE-like solid whose melt is one-way (wiki 'Snow', verbatim). The recipe's
  // own base still names the key — (WATER, Y) and (SAND, Y) are different
  // rows on purpose, they just share ICE's mechanics at v1.
  const mech = (bits & MOD_BITS.Y) ? MAT_ICE : baseId;
  const row = mats[mech].slice();
  const dropped = [];
  const needs = [];

  // ---- 1. bond template (K > M > E > S), fresh group sentinel ---------------
  const bondBits = BOND_TEMPLATE_PRECEDENCE.filter(has);
  if (bondBits.length) {
    const t = bondBits[0];
    const rice = mats[MAT_RICE], rubber = mats[MAT_RUBBER];
    const copyBond = (src) => {
      for (const f of ['bondK', 'bondBreak', 'bondReform', 'bondFormV', 'bondFormP',
                       'bondMu', 'rampLo', 'rampHi', 'rampSign']) row[F(f)] = src[F(f)];
    };
    if (t === 'K') {
      // RICE's block with the thermal gate OFF: rampSign 0 makes kEff always
      // full — "Rice adds Novel every step" (wiki 'Novel'), always-on joints.
      copyBond(rice);
      row[F('rampSign')] = 0;
      row[F('solidMode')] = SOLID_BOND;
    } else if (t === 'M') {
      // Mochi: contact re-forming sticky glob. RICE's formation numbers with a
      // softer spring; adhesion floor is what makes it a glob.
      copyBond(rice);
      row[F('rampSign')] = 0;
      row[F('bondK')] = C.M_BONDK;
      row[F('bondReform')] = 1;
      row[F('adhesion')] = Math.max(row[F('adhesion')], C.M_ADHESION_MIN);
      row[F('solidMode')] = SOLID_BOND;
    } else if (t === 'E') {
      // v1 copied the §18.23 RUBBER block verbatim. v2 keeps its break strain
      // but tightens the spring for OE-CAKE Elastic / the E property.
      copyBond(rubber);
      row[F('bondK')] = C.E_BONDK;
      row[F('solidMode')] = SOLID_BOND;
    } else {
      // String: a weak distance-only net. Spawn-weld forms it (bondReform 0),
      // so the painted stroke IS the chain. Ours is linear; theirs is
      // exponential — declared, not chased (wiki 'String').
      copyBond(rubber);
      row[F('bondK')] = C.S_BONDK;
      row[F('bondBreak')] = C.S_BONDBREAK;
      row[F('bondReform')] = 0;
      row[F('solidMode')] = SOLID_BOND;
    }
    row[F('bondGroup')] = -2;              // sentinel: interner assigns 64 + rowId
    const others = bondBits.slice(1);
    if (others.length) dropped.push(`bond template ${t} wins over ${others.join(',')} (K>M>E>S)`);
  } else if (has('B') && row[F('bondK')] === 0) {
    // B without any bond mechanism "acts exactly like Null" (wiki 'Material').
    dropped.push('B inert without a bond template');
  }

  // ---- 2. Brittle -----------------------------------------------------------
  // Applies to the template the row now wears. Their brittlenessCoefficient is
  // a per-SAVE global; per-row is strictly MORE expressive (wiki 'Parameters').
  if (has('B') && bondBits.length) row[F('bondBreak')] = C.B_BONDBREAK;

  // ---- 3. scalars V, T, D, L ------------------------------------------------
  // Symmetric XSPH carries BOTH measured V observables: slow bulk shear and
  // notable damping through a film one particle thick. OE-CAKE stores the
  // intermediary as particle spin; Suna represents the visible consequence
  // directly without adding invisible per-particle rotation state.
  if (has('V')) row[F('xsph')] = Math.min(row[F('xsph')] * C.V_XSPH_MUL, C.V_XSPH_CLAMP);
  if (has('T')) {
    row[F('gammaCoh')] = row[F('gammaCoh')] * C.T_COH_MUL;
    row[F('gammaCur')] = (row[F('gammaCoh')] / 4) | 0;   // the §12.4 ratio, kept by hand here
    row[F('adhesion')] = 0;
  }
  // D and L are a GRAVITY SUBSTITUTION (accepted, §25.8 #6): theirs is a
  // neighbour-push that dies at zero-g and scales with g — ours is +-600 grav,
  // which under setGravityScale dies and scales EXACTLY the same way, and
  // D+L cancels to the base by construction (the wiki 'LightDense' law).
  if (has('D')) row[F('grav')] = row[F('grav')] + C.DL_GRAV;
  if (has('L')) row[F('grav')] = row[F('grav')] - C.DL_GRAV;
  if (bits & MOD_BITS.FLOAT) row[F('grav')] =
    version >= 4 ? C.FLOAT_GRAV : 0;

  // ---- 4. thermal H, C ------------------------------------------------------
  if (has('H') && has('C')) {
    // Cool Heat: per-interaction hot/cold duality that one scalar cannot
    // express (wiki 'Cool Heat'). QUIET ANNUL — Darien's call (§25.8 #9,
    // locked): tempDrive 0, both HUD bits stay lit, the manifest names it.
    row[F('tempTarget')] = 0; row[F('tempDrive')] = 0;
    dropped.push('H+C annul (Cool Heat is per-interaction duality; one tempTarget cannot say it)');
  } else if (has('H')) {
    row[F('tempTarget')] = C.H_TARGET; row[F('tempDrive')] = C.HC_DRIVE;
    row[F('spawnTemp')] = C.H_TARGET;
  } else if (has('C')) {
    row[F('tempTarget')] = C.C_TARGET; row[F('tempDrive')] = C.HC_DRIVE;
    row[F('spawnTemp')] = C.C_TARGET;
  }

  // ---- 5. products closure --------------------------------------------------
  // A composed row's transitions default to the base row's products (already
  // copied). Overrides mint PRODUCT rows too — Snow is the exemplar: melt must
  // yield a water that NEVER freezes back, which is a second interned row.
  if (has('Y')) {
    needs.push({ field: 'meltTo', base: MAT_WATER, bits: MOD_BITS.YPROD });
  }
  if (bits & MOD_BITS.YPROD) {
    // Snow-water: the base row with freezing disabled. One-way, forever.
    row[F('freezePt')] = NEVER_COLD; row[F('freezeTo')] = 0;
  }

  // ---- EXACT CANCELLATION COLLAPSES TO THE BASE ROW -------------------------
  // D+L is the canonical case: the wiki law says they neutralise EXACTLY, and
  // the ±600 algebra satisfies it by construction — so the synthesised row IS
  // the base row, and minting a duplicate would waste a cache slot on a
  // distinction with no physics in it. PRESET_TABLE §2 reads LightDense the
  // same way ("base row unchanged"). `identity: true` tells the interner to
  // answer with the base rowId and record NOTHING.
  const ioBits = bits & (MOD_BITS.I | MOD_BITS.O | MOD_BITS.IO_NULL);
  if (ioBits === 0 && mech === baseId && needs.length === 0
      && row.every((w, i) => i === F('tint') || w === mats[baseId][i])) {
    return { words: mats[baseId].slice(), needs: [], dropped, identity: true };
  }

  // ---- §25.3 THE F1 INVARIANT, EXTENDED -------------------------------------
  // EVERY interned row with a bond block gets a FRESH unique group — not just
  // template rows: a composed ICE (bondGroup 5 inherited) sharing plain ICE's
  // group would put two different materials on the two ends of one bond, and
  // §18.11 MEASURED that min-vs-mat_mix stops being inert exactly there. The
  // -2 sentinel is replaced by 64 + rowId at mint time (the field is an i32;
  // authored groups live in [-1, 63], so 64 + rowId collides with nothing).
  if (row[F('bondK')] !== 0) row[F('bondGroup')] = -2;

  // ---- host-only words ------------------------------------------------------
  row[F('tint')] = composeTint(mats[mech][F('tint')], bits);

  // ---- the self-destruction contract (mirrors createEngine's §25 checks) ----
  if (row[F('tempDrive')] !== 0) {
    const tt = row[F('tempTarget')];
    if (!(tt < row[F('meltPt')] && tt < row[F('boilPt')])) {
      throw new Error(`[aether] synth: ${modLetters(bits)} on ${MAT_NAMES[baseId]} refused — ` +
        `tempTarget ${tt} >= the row's own meltPt/boilPt; a heater must not sit above its own ` +
        `transition thresholds (it would melt itself on the substep it is painted)`);
    }
    if (!(tt > row[F('freezePt')] && tt > row[F('condPt')])) {
      throw new Error(`[aether] synth: ${modLetters(bits)} on ${MAT_NAMES[baseId]} refused — ` +
        `tempTarget ${tt} <= the row's own freezePt/condPt; a cooler must not sit below its own ` +
        `transition thresholds`);
    }
  }
  return { words: row, needs, dropped };
}

// module-scope assert used by synth (the engine-instance assert lives in
// createEngine's closure).
function assertFn(cond, msg) { if (!cond) throw new Error('[aether] ' + msg); }

// ============================================================================
// §16 — WHICH MATERIALS GET FRICTION, AND THE SWEEP THAT CHOSE THE NUMBERS
// ----------------------------------------------------------------------------
// `fric` is the only field in the table that resists SHEAR, and it is the only
// reason anything in this roster keeps a shape. Eight rows carry it; the other
// sixteen are 0, which makes the whole `friction` pass bit-for-bit inert for
// them (and is what keeps every WATER/GOO hash from v5 reproducing exactly).
//
// THE SWEEP. A 12x12 SAND block dropped onto the floor and a 24x16 PHOTO sheet
// resting on it, 3000 / 2400 substeps, p05-p95 width as a multiple of the
// starting width. Friction runs once per solver iteration (see the encoder):
//
//   mu (Q16)    0     0.5    1.0    1.5    2.0    3.0    4.0
//   sand      10.83   6.25   5.21   4.90   4.75   4.07   4.41
//   photo      4.55   4.08   3.81   3.51   3.28   2.85   3.33
//
// SO WHY DOES NOTHING SHIP ABOVE 2.0, WHEN 3.0 RETAINS SHAPE BEST? Because a
// SECOND measurement disagreed with the first, and it is the one that decides.
// Above 2.0 the friction correction routinely exceeds dpMax, and a clamp that
// binds on one end of a pair and not the other breaks f_ij == -f_ji: the
// momentum leak jumps 20x (169 -> 1216 -> 23883 ticks at mu 0 / 2.0 / 3.0) and
// scales inversely with dpMax, which is the clamp's signature. mu = 3.0 also
// stopped being stable on a low-shear blob, which grew instead of holding.
// The full numbers are in the FRIC_MAX comment; the gate is granular.html G4.
// 2.0 is therefore a STABILITY ceiling, not a taste one, and it costs the
// difference between 4.75x and 4.07x on the sand column.
//
// AND THE HONEST PART: 4.75x is not a pile. Friction cuts sand's spread by
// ~2.3x and gives the roster a real second mechanical axis, but this solver
// still has no bulk yield stress (§13.7), so a heap slumps — slowly now instead
// of instantly. See test/granular.html G1 for the shipped numbers and RESUME.md
// for what would actually close it.
export const MATS = [
  // 0 WATER — the reference material. Mechanics IDENTICAL to v3.
  mat({ xsph: 6000, gammaCoh: 240, gammaCur: 60, adhesion: 0, grav: 1200,
        cond: 9000, coolRate: 40,
        freezePt: DEG(0),   freezeTo: MAT_ICE,
        boilPt:   DEG(100), boilTo:   MAT_STEAM,
        spawnTemp: DEG(20), tint: 0x2a6fb0 }),

  // 1 GEL — the successor to Goo. It retains Goo's deliberate isolation from
  // heat and reactions (cond 0, coolRate 0, no phase/reaction fields), while
  // using the former ORBIES soft, reforming contact net. This is the material
  // that holds a squishy, pushable lump instead of reading as thick water.
  mat({ fric: 49152,
        xsph: 45000, gammaCoh: 1700, gammaCur: 425, adhesion: 300, grav: 1080,
        cond: 0, coolRate: 0,
        spawnTemp: DEG(20), tint: 0x8cecff,
        bondK: 12288, bondBreak: 262144, bondReform: 1,
        bondFormV: 2600, bondFormP: 2000, bondMu: 0, bondGroup: 10,
        rampSign: 0,
        solidMode: SOLID_BOND }),

  // 2 ICE — pseudo-solid (no rigid bonds exist; "solid" here means very viscous
  // and very cohesive). Melts in a 20 deg room, which is correct and is also the
  // easiest phase change to see by eye.
  //
  // meltPt IS 2 deg, NOT 0, AND THAT GAP IS MANDATORY. §12 has no latent heat, so
  // a transition is a bare threshold test on an integer. With ICE.meltPt equal to
  // WATER.freezePt, a particle sitting at exactly that temperature satisfies BOTH
  // every substep and flips phase forever. MEASURED before this line existed: 36
  // particles at exactly 0 deg, sealed (cond 0, coolRate 0), alternated
  // ICE/WATER/ICE/WATER on 11 of 12 consecutive substeps — a perfect period-2
  // limit cycle. It is deterministic, so no determinism gate would ever see it;
  // it reads in the toy as strobing, and it swaps xsph 6000 <-> 36000 and
  // gammaCoh 240 <-> 700 every substep, so it is a mechanics defect, not a
  // cosmetic one. 2 deg of hysteresis removes it and keeps "water freezes at 0".
  // test/thermal.html T0.phasePairHysteresis gates every pair in the table.
  //
  // grav IS 1050 AND NOT 1200 BECAUSE ICE FLOATS. Real ice is 0.917x water, i.e.
  // grav 1100 — and 1100 is NOT what shipped, because PBF buoyancy is not a
  // density ratio. Every particle has the same mass and the same rest density;
  // the only separating force is the gravity DIFFERENCE against the drag, so the
  // number that looks like real ice is not 0.917. MEASURED, a 10x6 slab held
  // under an 18-wide pool for 2000 substeps on a thermally-held copy of the table
  // (ICE/WATER cond and coolRate zeroed so nothing melts), settled lift of the
  // ice above the water, +y is down:
  //   grav 1200 (= water)  -0.55 wu   (it SINKS — this is the pre-v6 behaviour)
  //   grav 1150            +0.89 wu
  //   grav 1100            +2.00 wu   (0.917x, "physical")
  //   grav 1050            +3.50 wu   <- SHIPPED
  //   grav 1000            +3.01 wu
  //   grav  900            +6.44 wu
  // 1050 is the cheapest value that clears the 1.0 wu margin E12 uses (the same
  // margin E1's density ladder uses) by a factor of 3. test/elements.html E12.
  //
  // §18 BONDS. The ramp is -8..2 deg with sign -1, which puts kEff at ZERO for
  // any ice warmer than 2 deg — i.e. ice that is ABOUT to melt has already
  // stopped being rigid, and the bond term hands over to the phase change
  // instead of fighting it. Painted at its spawnTemp of -20 it is fully bonded.
  mat({ xsph: 36000, gammaCoh: 700, adhesion: 120, grav: 1050,
        cond: 16000, coolRate: 40,
        meltPt: DEG(2), meltTo: MAT_WATER,
        corrPt: 9000, corrTo: MAT_WATER,
        fric: 114688,
        spawnTemp: DEG(-20), tint: 0xa8e0ff,
        bondK: 32768, bondBreak: 102400, bondReform: 0, bondMu: 0, bondGroup: 5,
        burstP: 98304,
        rampLo: DEG(-8), rampHi: DEG(2), rampSign: -1,
        // §20: RIGID while frozen. MELTING IS MEMBERSHIP and it costs no new
        // rule — `rampAt(ICE, T) == 0` above 2 deg already says the slab has
        // stopped being rigid, and a member whose row is no longer SOLID_RIGID
        // leaves the body and becomes an ordinary free particle.
        solidMode: SOLID_RIGID }),

  // 3 STEAM — buoyant gas. PBF has one rest density for everything, so a "gas"
  // is a light, weakly-cohesive fluid with NEGATIVE gravity, not a real gas.
  mat({ xsph: 3000, gammaCoh: 40, adhesion: 0, grav: -700,
        cond: 4000, coolRate: 110,
        condPt: DEG(55), condTo: MAT_WATER,
        spawnTemp: DEG(120), tint: 0xd8e8f0 }),

  // 4 OIL — floats on water because its gravity is lower, and burns.
  mat({ xsph: 12000, gammaCoh: 300, adhesion: 180, grav: 900,
        cond: 5000, coolRate: 300,
        ignitePt: DEG(180), fuel0: DEG(100), burnRate: 21000, burnHeat: 264000,
        burnTo: MAT_SMOKE,
        spawnTemp: DEG(20), tint: 0x6b4a1f }),

  // 5 FIRE — ignitePt = TEMP_MIN means "always alight". Burns for ~150 substeps
  // (1.3 s) and becomes smoke. Flame equilibrium ~1000 deg (see burnHeat above).
  //
  // grav IS -350, NOT -800, AND THAT IS AN IGNITION PARAMETER, NOT A LOOK.
  // Conduction is the only way a flame delivers heat, so what a flame can light
  // is decided by how long it stays in kernel range of it. MEASURED, fire seeded
  // on top of a wood slab, peak wood temperature reached over 2400 substeps:
  //   grav -800 ->  45 deg      grav -350 -> 169 deg
  // and with a seed embedded in the slab, wood consumed at ignitePt 200:
  //   grav -800 ->  65/102      grav -350 ->  98/102
  // It still rises — it is still the second most buoyant material in the table —
  // it just does not leave before it has done anything. test/thermal.html T4.
  mat({ xsph: 3000, gammaCoh: 30, adhesion: 0, grav: -350,
        cond: 12000, coolRate: 900,
        ignitePt: TEMP_MIN, fuel0: DEG(60), burnRate: 26000, burnHeat: 882000,
        burnTo: MAT_SMOKE,
        spawnTemp: DEG(900), tint: 0xff7a1e }),

  // 6 SMOKE — rises, cools, and VENTS. It is the only material whose job is to
  // stop existing, and as of v7 it actually does.
  //
  // condPt DEG(22) IS THE BOUNDED LIFETIME, and it is built out of the §12
  // machinery rather than a despawn pass. There is no despawn in the SOLVER and
  // there cannot be one: GPU compaction needs an atomic allocation whose arrival
  // order is not reproducible on this device, which is the exact thing
  // `canonicalize` exists to absorb. So instead a smoke particle is on a
  // THERMAL clock — it is born hot, coolRate 600 relaxes it toward the 20 deg
  // room with a time constant of 65536/600 = 109 substeps, then at 22 deg it
  // retires. This is a modestly brisker fade than the old 5-7 s behavior:
  // smoke should give a fire atmosphere, not accumulate into a permanent ceiling.
  // Smoke sitting over a live fire is reheated by
  // conduction and does NOT fade, which is the behaviour you want and is a
  // consequence of the mechanism rather than a case in it.
  //
  // WHAT CHANGED IN v7 IS ONLY THE PRODUCT, NOT THE CLOCK. Through v6 the
  // product was ASH, so smoke's bounded lifetime WAS a 1:1 conversion into ash:
  // one smoke particle in, one ash particle out, forever. That is the whole of
  // Darien's complaint and it was structural, not a tuning error. Now:
  //     condTo = VOID (it is gone), splitTo = ASH, splitPct = 6
  // so 6/256 of retiring smoke falls out as soot and 250/256 simply leaves. The
  // existing seeded-integer-hash `pick_product` does the entire split — no new
  // pass, no new field, no randomness. Realised over the real id_hash: 2.33 %.
  //
  // THE THRESHOLD IS ONE-SIDED, AND THAT CONSTRAINS THE WHOLE TABLE — and in v7
  // it constrains it MUCH harder, because a cold-born smoke particle no longer
  // becomes soot one substep later, it VANISHES. Any row that produces SMOKE at
  // or below 22 deg silently destroys matter. Four rows produced smoke at
  // AMBIENT before v6 (CRYO.boilTo, METAL.corrTo, PHOTO.corrTo, ACID.corrTo) and
  // were re-pointed; WOOD and PHOTO's CORRODE paths were still doing it in v6
  // and E13's audit did not look at them (it read only the four phase paths).
  // Both are fixed below and the audit is widened. E13.noSmokeIsBornCold now
  // reads burnTo / splitTo / corrTo as well.
  mat({ xsph: 2000, gammaCoh: 25, adhesion: 0, grav: -420,
        cond: 3000, coolRate: 600,
        condPt: DEG(22), condTo: MAT_VOID, splitTo: MAT_ASH, splitPct: 6,
        spawnTemp: DEG(120), tint: 0x5a5a62 }),

  // 7 WOOD — the thing you set on fire. Insulating, so it heats locally and the
  // burn front creeps instead of flashing over. Burns ~9 s to ash + smoke.
  //
  // ignitePt IS 200, NOT 280, BECAUSE 280 IS ABOVE WHAT A BURNING NEIGHBOUR CAN
  // DELIVER. This is a table INVARIANT, not a taste knob: a combustible whose
  // ignition point sits above the temperature its own burning neighbours reach
  // cannot sustain a fire — it can only be lit particle by particle by an
  // external source, and the front dies as soon as the source is gone. MEASURED,
  // fire seeded inside a 102-particle slab, particles consumed in 2400 substeps:
  //   ignitePt 280 -> 19/102 (front self-extinguishes)
  //   ignitePt 240 -> 65/102
  //   ignitePt 200 -> 98/102 (self-sustaining)
  // Two hard caps set that ceiling and neither is a material property: DT_CLAMP
  // limits the driving temperature difference to 256 deg, and HT_TERM limits ONE
  // neighbour to 1.0 deg of transfer per substep. Making the flame hotter buys
  // nothing above those; raising FIRE.cond bought nothing either (measured).
  //
  // THE ASH YIELD, v7 (§17). Through v6 this row read
  //     burnTo: ASH, splitTo: SMOKE, splitPct: 110
  // which is 57 % ash directly and 43 % smoke — and smoke then became ash too,
  // so ONE WOOD PARTICLE LEFT EXACTLY ONE ASH PARTICLE. 100 % of burned mass,
  // against a physical anchor of 1-3 %. It is now inverted:
  //     burnTo: SMOKE, splitTo: ASH, splitPct: 26
  // 26/256 = 10.16 % of the wood is left as a bed of ash where it stood, the
  // other 89.8 % rises as smoke, and 2.34 % of THAT falls back out as soot —
  // 12.26 % of burned mass in total, 8.2x less than v6. Still 4-12x the physical
  // number, deliberately: a 132-particle photograph leaving 1-4 particles is
  // indistinguishable from nothing, and a visible dusting beats a physically
  // perfect invisible one. design/vent/yield.mjs computes all of this against
  // the REAL id_hash rather than against the nominal fraction.
  //
  // AND THE ELEGANT HALF: `splitPct` is shared by EVERY transition a row has, so
  // inverting which product is primary fixes the corrode path for free.
  // corrTo was ASH and splitTo was SMOKE, so acid on wood produced 43 % SMOKE
  // AT AMBIENT — cold-born smoke, which in v6 became ash one substep later
  // (harmless) and in v7 would VANISH (43 % of any acid-dissolved wood silently
  // destroyed). With splitTo == corrTo == ASH the split is a no-op on that path
  // and acid pulps wood into 100 % ash, which is what it always meant to do.
  //
  // §18 BONDS. Spawn-weld only, and BREAKABLE at 30 % strain — wood splinters,
  // stone does not. eps_break of 2-5 % is a trap and was measured as one: ICE at
  // 2.4 % and WOOD at 4.6 % went 652 bonds -> 8 in 200 substeps on a 1 wu drop
  // while still reporting K = 1.0. Solids want 25-35 %.
  // The ramp softens it across 260..420 deg, i.e. a plank stops being rigid as
  // it chars, before `ignitePt` has anything to say about it.
  mat({ fric: 114688,
        xsph: 40000, gammaCoh: 800, adhesion: 400, grav: 1200,
        cond: 3000, coolRate: 120,
        ignitePt: DEG(200), fuel0: DEG(200), burnRate: 12000, burnHeat: 93600,
        burnTo: MAT_SMOKE, splitTo: MAT_ASH, splitPct: 26,
        corrPt: 16000, corrTo: MAT_ASH,
        spawnTemp: DEG(20), tint: 0x8a5a2b,
        bondK: 26214, bondBreak: 110756, bondReform: 0, bondMu: 0, bondGroup: 4,
        // Intact solids are impermeable. Pressure opens WOOD through the
        // existing visible burst path; it never ghosts through an intact plank.
        burstP: 49152, sealMode: 0,
        rampLo: DEG(260), rampHi: DEG(420), rampSign: -1,
        // §20: a plank is a plank. Its destruction mode is BURNING, which is
        // member loss, and member loss is what §20 is good at.
        // STATED PLAINLY: a rigid plank has no strain, so it cannot SNAP under
        // load — it burns, and it splits when you erase it. If 'wood snaps' is
        // wanted it is a fracture rule, which is the same machinery as the
        // split running on a different trigger. Darien's call, not a blocker.
        solidMode: SOLID_RIGID }),

  // 8 ASH — LIGHT, AND EASILY DISTURBED. Darien: "the ash just accumulates kind
  // of quickly and isn't very reactive."  §17 answers the first half by making
  // 8.2x less of it; this row answers the second half, and the non-obvious part
  // is WHICH four numbers were wrong.
  //
  // It was not cohesion. v6's ash sat in a heap because of `xsph: 20000` and
  // `fric: 65536`: a shove was viscously damped before it could propagate, and
  // the §16 shear term held the pile together against it. That is soot with the
  // rheology of wet clay. The retune, with what each number buys:
  //   fric     65536 -> 16384  (1.00 -> 0.25) it flows and puffs instead of
  //                            holding a heap. Still > 0, so it is still on the
  //                            frictional side of granular.html G2's ordering.
  //   xsph     20000 ->  2500  a shove PROPAGATES through the layer instead of
  //                            being absorbed by the first two particles.
  //   gammaCoh   120 ->    24  it layers and drifts instead of beading.
  //   adhesion    60 ->     0  it does not cling to walls.
  //   grav       800 ->   520  it still SETTLES (positive gravity, no buoyancy)
  //                            so the dusting is visible — but anything can loft
  //                            it, and 520 < WATER's 1200 means a burned
  //                            photograph leaves a SCUM ON A POOL instead of
  //                            sinking out of sight. That float is the one
  //                            genuinely new material-on-material interaction
  //                            this row gains, and elements.html E16 gates it.
  // E0's no-filler check stays green: ASH still differs from DUST — its nearest
  // rival — on all five axes (density 520 vs 260, viscosity 2500 vs 1000,
  // cohesion 24 vs 8, conductivity 2000 vs 5000, reactions).
  mat({ fric: 16384,
        xsph: 2500, gammaCoh: 24, gammaCur: 6, adhesion: 0, grav: 520,
        cond: 2000, coolRate: 300,
        spawnTemp: DEG(20), tint: 0x4a4844 }),

  // 9 LAVA — heavy, viscous, and hot enough to ignite everything it touches.
  mat({ xsph: 38000, gammaCoh: 500, adhesion: 300, grav: 1500,
        cond: 14000, coolRate: 60,
        freezePt: DEG(700), freezeTo: MAT_STONE,
        spawnTemp: DEG(1200), tint: 0xff5a10 }),

  // 10 STONE
  // corrPt/corrTo (v5): acid eats stone into SAND, and sand is immune — silica
  // is what you keep acid in. The reaction therefore stops itself once the face
  // it was eating has turned to grit, which is a consequence of the table, not a
  // special case in the solver.
  //
  // §18 BONDS. STONE is the headline: `bondReform: 0` means it welds ONLY on the
  // spawn event (one brush stroke, one PNG import, one loadScene) and never
  // re-forms on contact, which is what makes a painted block a rigid body and a
  // poured heap NOT one. `bondBreak: 0` is literally unbreakable — the hard
  // break at the kernel radius is the only thing that severs it. The ramp
  // softens it to nothing across its own 700..950 melt window, so a stone that
  // is turning to lava stops being a solid BEFORE the transition fires.
  //
  // dpMax STAYS AT 16384, AND THAT IS A MEASUREMENT, NOT AN OVERSIGHT.
  // design/bonds/SPEC18.md §18.10 pins it at 65536 for the four rigid solids on
  // a CPU-prototype measurement of momentum leakage (clamp hits 996 452 -> 357).
  // On the SHIPPED solver it does the opposite of what it was for: a 1.0 wu
  // single-iteration correction is larger than the gap between the 2.05 wu bond
  // radius and the 2.5 wu hard break, so it TEARS THE BODY APART. Measured here,
  // 12x12 STONE dropped 18 wu, live bonds at substeps 1 / 8 / 2400 and the final
  // p05-p95 width:
  //     dpMax 16384 (SHIPPED)  1492 -> 1486 -> 1152    width 12.39 wu
  //     dpMax 32768            1492 -> 1138 ->  512    width 22.14 wu
  //     dpMax 65536            1492 ->  798 ->  480    width 20.37 wu
  // Two thirds of the body's bonds are gone inside EIGHT substeps at 65536.
  // METRIC.md §9.2 says it in general terms — "the band widths transfer, the
  // centres may not" — and this is the case where a prototype centre was not
  // merely off, it was pointing the wrong way. The frozen v3 mechanics block is
  // therefore UNCHANGED by §18 on every row, which also retires SPEC18 §18.21 Q3.
  mat({ fric: 131072,
        xsph: 40000, gammaCoh: 900, adhesion: 500, grav: 1600,
        cond: 6000, coolRate: 60,
        meltPt: DEG(950), meltTo: MAT_LAVA,
        corrPt: 14000, corrTo: MAT_SAND,
        spawnTemp: DEG(20), tint: 0x6e6a66,
        bondK: 32768, bondBreak: 0, bondReform: 0, bondMu: 0, bondGroup: 2,
        burstP: 163840,
        rampLo: DEG(700), rampHi: DEG(950), rampSign: -1,
        // §20: RIGID. It is what the paint tool defaults to for solids.
        solidMode: SOLID_RIGID }),

  // 11 SAND — heavy and barely cohesive, so it piles and pours.
  //
  // §18: SAND DOES NOT BOND, AND THAT IS A MEASUREMENT.
  // design/bonds/SPEC18.md §18.10 gives SAND a full contact-bond row
  // (bondK 13107, bondReform 1, bondFormV 1800, bondFormP 6000, bondMu 65536,
  // bondGroup 1) on the theory that settled-contact bonding is what makes a
  // heap. On the SHIPPED solver it makes WELDED sand, and it breaks two roster
  // claims Darien signed off in §16. Measured, elements.html E2, all three arms
  // in this tree:
  //     SAND retained on the pedestal   v7 37 %  ->  99 % at bondFormP 6000
  //                                              ->  83 % at 24000
  //                                              ->  43 % at 60000
  //     (E2's own comment: "above 90 % means it welded")
  //     sand through a water pool       v7 +4.82 wu BELOW the water
  //                                     -> -2.99 (it FLOATS as a bonded raft)
  //                                     -> +2.19 / +0.71 / +4.82 at
  //                                        bondFormP 24000 / 60000 / 150000
  // The sink number is NON-MONOTONIC across a 6x sweep of the one constant that
  // controls it, i.e. it is exactly the kind of number design/bonds/METRIC.md
  // proved may not be tuned against. There is no value that makes sand both
  // pile and sink; at 150000 no bonds form at all and the row is v7 again.
  //
  // So §18 ships on the four RIGID solids plus RICE, and sand stays the §16
  // granular material it was. The `bondReform` contact path is NOT untested by
  // this — RICE exercises it, and test/bonds.html gates it there. Re-enabling
  // sand is one line plus a re-derivation of E2, and the tuned constants are
  // kept above so nobody has to find them again. It is a FEEL call and it is
  // parked for Darien, not closed.
  mat({ fric: 98304,
        xsph: 8000, gammaCoh: 60, adhesion: 0, grav: 1800,
        cond: 4000, coolRate: 100,
        meltPt: DEG(1200), meltTo: MAT_LAVA,
        spawnTemp: DEG(20), tint: 0xd8c48a }),

  // 12 METAL — the conductor. cond 30000 (0.458) is the highest in the table and
  // is what makes a metal bar carry a flame's heat across the screen.
  //
  // corrTo IS GAS AND NOT SMOKE (changed with the v6 smoke fade). Two reasons and
  // both are load-bearing: (1) a corroding metal particle sits at room
  // temperature, and SMOKE born at room temperature is ASH one substep later, so
  // the fizz would have been invisible; (2) acid on metal releasing a FLAMMABLE
  // gas is a chain worth having — the gas rises, pools, and goes off if anything
  // lights it. test/elements.html E14 gates that chain end to end.
  mat({ fric: 131072,
        xsph: 40000, gammaCoh: 900, adhesion: 400, grav: 2000,
        cond: 30000, coolRate: 80,
        meltPt: DEG(1200), meltTo: MAT_MOLTEN,
        corrPt: 12000, corrTo: MAT_GAS,
        spawnTemp: DEG(20), tint: 0xb9c2cc,
        bondK: 32768, bondBreak: 0, bondReform: 0, bondMu: 0, bondGroup: 3,
        burstP: 294912,
        rampLo: DEG(800), rampHi: DEG(1200), rampSign: -1,
        // §20: RIGID — THE MATERIAL DARIEN NAMED. A drawn wrench must not go
        // rhomboid.
        solidMode: SOLID_RIGID }),

  // 13 MOLTEN metal
  mat({ xsph: 20000, gammaCoh: 700, adhesion: 200, grav: 2000,
        cond: 30000, coolRate: 90,
        freezePt: DEG(1050), freezeTo: MAT_METAL,
        spawnTemp: DEG(1400), tint: 0xffd07a }),

  // 14 GAS — flashes over in ~0.34 s and becomes FIRE, which then becomes smoke.
  // This is the reactor-detonation material.
  mat({ xsph: 1500, gammaCoh: 15, adhesion: 0, grav: -600,
        cond: 6000, coolRate: 400,
        ignitePt: DEG(120), fuel0: DEG(25), burnRate: 40000, burnHeat: 432000,
        burnTo: MAT_FIRE,
        spawnTemp: DEG(20), tint: 0xa8ff9a }),

  // 15 PHOTO — the material an imported PNG becomes. Pseudo-solid so the picture
  // holds its shape, and FLAMMABLE, because the entire point is to set it alight.
  // The importer owns the density guard (SPEC §5.4); this row owns the feel.
  //
  // ignitePt IS 140 — THE LOWEST IN THE TABLE, DELIBERATELY. This is the one
  // material whose product requirement is "easy to light": import a picture, drop
  // fire on it, watch it burn. It is paper, and paper should be the easiest thing
  // in the roster to set alight. MEASURED, particles consumed in 2400 substeps
  // with fire seeded ON TOP of the slab / EMBEDDED in it:
  //   ignitePt 240 ->   0/132  |  52/102
  //   ignitePt 200 ->   0/132  |  80/102
  //   ignitePt 170 ->   1/132  | 102/102
  //   ignitePt 140 ->  46/132  | 102/102   <- fire DROPPED ON IT lights it
  // 140 is the only value tested at which the nostalgic gesture works at all.
  // WOOD stays at 200 on purpose, so "paper catches, wood needs to be got going"
  // is a legible difference between two materials rather than an accident.
  mat({ fric: 114688,
        xsph: 40000, gammaCoh: 700, adhesion: 350, grav: 1200,
        cond: 2500, coolRate: 120,
        ignitePt: DEG(140), fuel0: DEG(120), burnRate: 10000, burnHeat: 88800,
        // v7 (§17): inverted exactly as WOOD's was, and for both of the same
        // reasons — the ash yield, and the acid path. It was
        // `burnTo: ASH, splitTo: SMOKE, splitPct: 96` (62.5 % ash direct, and
        // the rest became ash too), which made a burned photograph leave a
        // photograph-shaped pile of soot. 26/256 leaves a legible smear
        // instead: 132 particles -> ~13 of ash rather than 132.
        burnTo: MAT_SMOKE, splitTo: MAT_ASH, splitPct: 26,
        // corrTo ASH, not SMOKE: acid pulps paper, and a smoke particle born at
        // room temperature vents into nothing (see SMOKE). splitTo is now also
        // ASH, so the split is a no-op here and acid turns the whole sheet to
        // pulp — in v6 the shared splitPct sent 37.5 % of it to cold SMOKE.
        corrPt: 10000, corrTo: MAT_ASH,
        spawnTemp: DEG(20), tint: 0xffffff,
        // ── v12 (§18.23): PHOTO GETS BONDS — THE PICTURE IS HELD BY STRUCTURE.
        // Until v12 an imported photograph was held together by `fric: 1.75`
        // alone, and friction is a tangential term: it slows a sheet coming
        // apart, it does not stop it. Measured on the same disc rig as RUBBER,
        // a PHOTO ball dropped 40 wu: radius of gyration 3.59 -> 8.66 (x2.41).
        // With bonds: 3.60 -> 3.63 (x1.01). The picture arrives as the picture
        // and STAYS the picture.
        //
        // bondK 8192 IS THE SOFTEST BOND IN THE ROSTER, and that is the point:
        // this is PAPER. It holds, and it crumples — peak squash 1.57:1 against
        // 1.34 at k 16384 and 1.24 at k 32768, all of which hold shape equally
        // well (rg x1.00-1.01). The flimsiest value that holds is the right one
        // for a sheet.
        //
        // bondBreak 262144 IS ε_break 1.00 — deliberately BELOW rubber's 1.50,
        // so the roster reads as an ordering a person would predict from the
        // names: ICE 0.25 < WOOD 0.30 < RICE 0.50 < PHOTO 1.00 < RUBBER 1.50 <
        // STONE/METAL unbreakable. Paper tears; rubber stretches.
        //
        // THE BURN TRADE WENT THE OTHER WAY AND IT IS REPORTED, NOT HIDDEN.
        // QUEUE #1 warned that "the sheet holding together makes it burn 58 %
        // not 100 %". On a 22x6 sheet with FIRE embedded in it, 2400 substeps:
        // bondK 0 consumed 109 of 124 = 87.90 %; bonded consumed 124 of 124 =
        // 100.00 %. Bonds hold the burning region against its own fuel instead
        // of letting the fire blow the sheet apart, so a bonded photograph
        // burns MORE completely, not less. Nothing was traded away.
        bondK: 8192, bondBreak: 262144, bondReform: 0, bondMu: 0, bondGroup: 8,
        solidMode: SOLID_BOND }),

  // ==========================================================================
  // v5 (§13). Six rows, and each one had to earn its place: the rule for this
  // roster is that a material must differ MEASURABLY from every existing one on
  // at least two of {density, viscosity, cohesion, conductivity, reactions}. The
  // per-row comment names which two. Materials that failed the rule were cut and
  // the cuts are recorded in SPEC §13.6, not silently shipped as near-duplicates.
  // ==========================================================================

  // 16 CRYO — the COLD SOURCE, and the reason it exists is structural, not
  // decorative: AMBIENT is a compile-time 20 deg, so ambient relaxation only ever
  // pulls UP. Before this row there was no way to make anything cold except by
  // seeding it cold in the scene file. CRYO is the tool that makes every
  // FREEZING transition in the table reachable from the UI: water -> ice, lava ->
  // stone, molten -> metal, mercury -> metal.
  //
  // It is a liquid at grav 950, which puts it BETWEEN oil (900) and water (1200)
  // — pour all three and they stack in that order, measured in test/elements.html.
  // It boils away on its own (boilPt -60 in a 20 deg room), so it is a consumable
  // and not a permanent world-freezer.
  // DISTINCT ON: conductivity (26000, second only to metal/mercury) and reactions
  // (the only material whose transition is triggered by being TOO WARM at a
  // temperature below zero).
  mat({ xsph: 2200, gammaCoh: 120, adhesion: 0, grav: 950,
        cond: 26000, coolRate: 220,
        // boilTo IS WATER, NOT SMOKE (changed with the v6 smoke fade). Cryo
        // vapour is born at -60 deg and SMOKE now becomes ASH at 22 deg, so
        // routing it through smoke would have made a cryogen boil off into a
        // little pile of soot. It evaporates into water instead — which, at
        // -60 deg, freezes on the same substep, so a cryo puddle boiling away
        // leaves FROST. That is better than what it replaced.
        boilPt: DEG(-60), boilTo: MAT_WATER,
        spawnTemp: DEG(-180), tint: 0x7fd8ff }),

  // 17 RUBBER — the ELASTIC solid. PBF has no rest-shape bonds, so "springy" here
  // is built out of the three knobs that decide how much kinetic energy survives
  // a collision: xsph 400 is the LOWEST viscosity in the table by 3.75x (nothing
  // damps the rebound), eps sits at the 32768 floor (the stiffest constraint
  // §2.3 permits, so the density correction pushes back hardest), and adhesion 0
  // means it does not wet the wall it hits. Rebound is measured in
  // test/elements.html E4 against the same blob made of WATER and GOO.
  //
  // NON-FLAMMABLE ON PURPOSE, and it still burns: at 220 deg it MELTS INTO TAR,
  // and tar is what catches fire. That chain is why rubber does not need fuel of
  // its own, and it dodges the §12.5(b) self-sustaining-front invariant entirely.
  // DISTINCT ON: viscosity (400 vs the next lowest 700) and reactions (melts to a
  // different material rather than burning).
  // fric 0, AND THAT IS NOT AN OVERSIGHT. Rubber is the ELASTIC row: its whole
  // identity is that a blob of it rebounds higher than the same blob of water or
  // goo, built out of the lowest viscosity in the table (400, 1.75x below the
  // next) and the stiffest permitted eps. §16 friction is the exact opposite of
  // that — it removes the tangential energy the rebound is made of. MEASURED at
  // fric 0.75: rubber's rebound fell to 0.03 wu against goo's 0.05 and
  // elements.html E4 went red on both of its checks, plus E8's shear check.
  // Rubber holds its shape through cohesion (gammaCoh 900) instead, which is
  // what it always did. E4 is the gate that keeps this at 0.
  //
  // ── v12 (§18.23): RUBBER GETS BONDS, AND THAT IS THE WHOLE OF QUEUE #1. ────
  // Until v12 this row had NEITHER elasticity mechanism — bondK 0 AND fric 0 —
  // so the paragraph above was the entire design and it was not enough. What it
  // bought was a REBOUND number; what it could not buy is a SHAPE. Measured on
  // test/pages/rubber_sweep.html, a RUBBER ball of radius 5 dropped 40 wu:
  //
  //     RUBBER v11 (bondK 0)   radius of gyration 3.58 -> 44.75  = x12.51
  //     WATER                                     3.59 -> 47.50  = x13.23
  //     GOO                                       3.58 -> 47.41  = x13.23
  //
  // i.e. a rubber ball ended the run indistinguishable from a puddle of water,
  // spread across the whole tank floor. Its "rebound" of 2.13 wu of mean-y
  // recovery was a SPLASH, and the honest instrument is how far the DEEPEST
  // particle in the body leaves the floor: v11 rubber reads 0.00 wu, exactly
  // like water and goo. A ball that never leaves the ground is not bouncy.
  //
  // With bonds, same staging: rg 3.60 -> 3.64 (x1.01), and 0.83 wu of clean
  // liftoff — the WHOLE ball in the air. It squashes to 2.13:1 at the deepest
  // point of the impact and comes back to 1.25:1, i.e. 78 % of the deformation
  // is returned; a §18-bonded STONE disc on the identical rig squashes to
  // 1.51:1 and gives back 35 %.
  //
  // bondK 16384 IS THE SOFTEST VALUE THAT STILL HOLDS A SHAPE, and that is why
  // it is the value. Same rig, sweeping k at bondBreak 409600:
  //     k  4096  rg x1.61   (tears: 788 -> 384 bonds)
  //     k  8192  rg x1.48   (tears: 788 -> 534)
  //     k 12288  rg x1.02   <- the cliff
  //     k 16384  rg x1.01   squash 2.13, 78 % returned, liftoff 0.83
  //     k 20480  rg x1.00   squash 1.46 — LESS deformable than bonded STONE
  //     k 24576  rg x1.00   squash 1.24
  // Above 16384 the bounce keeps rising (liftoff 1.70 at 20480, 2.41 at 24576)
  // but the ball stops DEFORMING, and elastic that does not deform is just a
  // rock with a good coefficient of restitution. QUEUE #1 asks for "a soft
  // bondK... it should deform hugely and not tear", and 16384 is the softest
  // row that satisfies both halves.
  //
  // bondBreak 409600 IS ε_break 1.50, THE HIGHEST IN THE ROSTER BY 3x, and it
  // is a derived number rather than a taste one: the §18 hard break is at the
  // kernel radius H = 2.5 wu, so for a bond formed on the 1.0 wu rest lattice
  // the soft break can never fire below (2.5/1.0)^2 = 6.25 = 409600 in Q16.
  // 409600 therefore means "as unbreakable as a lattice bond can be" while
  // still tearing an anomalously SHORT bond that has been stretched 6.25x in
  // area — a real safety valve, unlike STONE's literal `bondBreak: 0`. It also
  // has to be this high: at RICE's ε_break 0.50 the ball comes apart on
  // landing (rg x2.68, 788 -> 504 bonds) and at ε_break 1.00 it still loses
  // liftoff (0.83 -> 0.26).
  //
  // bondReform 0, AGAINST design/QUEUE.md's OWN HINT ("bondReform likely 1 so a
  // squashed ball re-knits"), AND THE MEASUREMENT IS WHY. Two balls, one
  // dropped onto the other, counting bonds whose endpoints started in different
  // balls: bondReform 0 gives 0 of 1366; bondReform 1 gives 188 of 1846. Two
  // rubber balls that touch must not become one rubber blob. And the hint's
  // premise does not hold: a bond breaks on EXTENSION (d2 > L0^2 * mb) and
  // never on compression, so SQUASHING a ball breaks nothing — there is nothing
  // for the re-knit to repair. What reform bought was 83 % of the deformation
  // returned instead of 78 %, which is not worth welding the world together.
  // It also keeps RUBBER out of the emitter (§18.12's table-derived rule), and
  // that is correct rather than incidental: an emitter batch is its own spawn
  // event, so a poured rubber stream would be a heap of 2- and 3-grain clumps.
  //
  // solidMode SOLID_BOND, NEVER SOLID_RIGID. Elastic is the OPPOSITE property
  // to rigid; §20 places every member at c + R(theta)*o and flexion is not
  // representable there. `index.html` already reads this field to decide
  // whether to freeze time during a stroke, and rubber must NOT freeze.
  mat({ fric: 0,
        eps: 32768, dpMax: 24576, xsph: 400, gammaCoh: 900, gammaCur: 225,
        adhesion: 0, grav: 1100,
        cond: 1200, coolRate: 60,
        meltPt: DEG(220), meltTo: MAT_TAR,
        spawnTemp: DEG(20), tint: 0x33333a,
        bondK: 16384, bondBreak: 409600, bondReform: 0, bondMu: 0, bondGroup: 7,
        solidMode: SOLID_BOND }),

  // 18 TAR — the SLOW fire. Oil is the fast one: it lights at 180 deg and is gone
  // in 312 substeps. Tar lights at 230 and burns for 2457 substeps — 7.9x longer,
  // the longest burn in the roster — and it SINKS in water (grav 1450) where oil
  // floats (900). It is also the stickiest thing in the table (adhesion 520,
  // above stone's 500) and sits at the XSPH_MAX ceiling of 45000, so it crawls
  // rather than flows.
  //
  // grav 1450 and cond 1600 ARE NOT ARBITRARY: at the first values I wrote
  // (1350 / 2500) test/elements.html E0's no-filler check went RED against PHOTO,
  // which sits at 1200 / 2500 / xsph 40000 / coh 700 — the two rows differed on
  // NOTHING but their reaction lists, which is exactly the near-duplicate the
  // roster rule exists to reject. Pushed apart to 17% on density and 36% on
  // conductivity, and E0 is the gate that keeps them apart.
  // DISTINCT ON: viscosity (3.8x oil), density (sinks vs floats) and reactions
  // (7.9x burn time).
  mat({ fric: 16384,
        xsph: 45000, gammaCoh: 650, adhesion: 520, grav: 1450,
        cond: 4200, coolRate: 150,
        ignitePt: DEG(230), fuel0: DEG(300), burnRate: 8000, burnHeat: 102000,
        // v7 (§17): splitPct 64 -> 26. This row was ALREADY the right shape
        // (primary SMOKE, split ASH) — WOOD and PHOTO were inverted to match it.
        // Only the fraction moves, onto the one number the whole roster now
        // shares for "how much of a burned solid is left behind".
        burnTo: MAT_SMOKE, splitTo: MAT_ASH, splitPct: 26,
        spawnTemp: DEG(20), tint: 0x241f1a }),

  // 19 MERCURY — the HEAVY bead. Four table extremes at once: the densest row
  // (grav 2600, 30% above metal), the most cohesive (gammaCoh 950), the most
  // conductive (cond 32768 = COND_MAX exactly), and adhesion 0 so it wets
  // nothing. The combination is the point — it sinks under everything, refuses to
  // spread, and carries heat across a scene faster than a metal bar.
  //
  // freezePt -39 deg is the real number and it is REACHABLE now that CRYO exists;
  // it is the only material in the roster that freezes below zero.
  // DISTINCT ON: density (highest), cohesion (highest), conductivity (highest).
  mat({ xsph: 700, gammaCoh: 950, gammaCur: 237, adhesion: 0, grav: 2600,
        cond: 32768, coolRate: 70,
        freezePt: DEG(-39), freezeTo: MAT_METAL,
        boilPt: DEG(357), boilTo: MAT_SMOKE,
        spawnTemp: DEG(20), tint: 0xc9ccd6 }),

  // 20 DUST — the HANGING cloud that goes off. GAS is the other flashover
  // material and the difference is the sign of gravity: gas rises to the ceiling
  // (-600), dust drifts DOWN (+260) and settles into a layer you can disturb.
  // gammaCoh 8 is the lowest in the table, so a dust cloud disperses instead of
  // beading, and fuel0/burnRate gives a 20-substep flash — 0.16 s, the fastest
  // burn in the roster, 2x faster than gas.
  // DISTINCT ON: density (settles vs rises — a sign flip, not a magnitude) and
  // cohesion (8, half of gas's 15 and 7x below sand).
  //
  // burnHeat 700000 IS NOT AN EQUILIBRIUM FLAME TEMPERATURE and the §12 formula
  // in the field reference does not apply to it. That formula assumes the
  // particle burns for longer than its thermal time constant (65536/coolRate =
  // 262 substeps here); dust burns for 29. What matters for a flash is the TOTAL
  // heat released over the whole burn, 29 * 700000 = 310 deg.
  //
  // ignitePt 60 IS THE LOWEST IN THE ROSTER AND IT HAD TO BE. §12.5(b) says a
  // combustible's ignition point must sit below what its own burning neighbours
  // deliver, and for a SHORT burn that ceiling is brutal: HT_TERM caps ONE
  // neighbour at 1.0 deg per substep, so a 29-substep burn can hand a neighbour
  // at most 29 deg however hot it gets. MEASURED, 20 FIRE seeded in a 120-particle
  // dust slab, particles consumed in 4000 substeps:
  //     ignitePt 150, burnHeat 270000  ->  10/120   (front dies at the seed)
  //     ignitePt 130, burnHeat 700000  ->  10/120   (more heat does not help)
  //     ignitePt  60, burnHeat 700000  ->  82/120   (flashover, peak 46 alight)
  // Raising `cond` from 1800 to 7000 changed the result by ZERO particles, which
  // is the §12.5 note about DT_CLAMP/HT_TERM saturating the flame's transfer
  // reproducing itself on a second material. 60 deg is also right for the FEEL:
  // dust is the thing that goes off when anything warm gets near it.
  mat({ xsph: 1000, gammaCoh: 8, gammaCur: 2, adhesion: 0, grav: 260,
        cond: 5000, coolRate: 250,
        ignitePt: DEG(60), fuel0: DEG(18), burnRate: 40000, burnHeat: 700000,
        burnTo: MAT_FIRE, splitTo: MAT_SMOKE, splitPct: 80,
        spawnTemp: DEG(20), tint: 0xb0a58c }),

  // 21 ACID — the only material in the roster that acts through CHEMISTRY rather
  // than temperature (§13). corrode 65536 = 1.0 in Q16 is the reference strength
  // the victims' corrPt values are quoted against: a particle completely
  // surrounded by acid takes a dose of ~0.795 * corrode per substep, so a corrPt
  // of 14000 means "about a fifth of my neighbourhood is acid".
  //
  // IT IS CONSUMED, and by the same accumulator: the SPEND term doses an acid
  // particle in proportion to how much DISSOLVABLE matter it touches. So a puddle
  // on stone runs out, a puddle on water does not, and once the stone it was
  // eating has turned to SAND — which is immune, because silica is what you keep
  // acid in — the reaction stops on its own. That self-limiting behaviour is a
  // consequence of the mechanism, not a special case in it.
  //
  // Slightly denser than water (1250 vs 1200) so it sinks through a pool.
  // DISTINCT ON: reactions (the only chemical agent) and conductivity/density
  // against every other thin liquid.
  mat({ xsph: 4000, gammaCoh: 180, gammaCur: 45, adhesion: 60, grav: 1250,
        cond: 7000, coolRate: 120,
        boilPt: DEG(130), boilTo: MAT_SMOKE,
        // corrTo WATER: spent acid neutralises. It used to be SMOKE, but a
        // spent acid particle is at room temperature and SMOKE now falls out as
        // soot at 22 deg, so it would have left grit instead of a puddle.
        // WATER is also immune to acid, so a neutralised particle is inert —
        // which is what makes the reaction stop itself twice over.
        corrode: 65536, corrPt: 20000, corrTo: MAT_WATER,
        spawnTemp: DEG(20), tint: 0x8dff3a }),

  // ==========================================================================
  // v6 (SPEC §15). THE REACTOR PAIR. Two rows, and they only make sense together:
  // NITRO is a charge that does nothing until something warm touches it, and
  // BLAST is the pressure wave it turns into. Every step is a table row.
  //
  //   NITRO --ignite(40 deg)--> BLAST --33 substeps--> FIRE --151--> SMOKE --> ASH
  //
  // five materials, five transitions, no code. Nothing in sim.wgsl knows either
  // of them exists.
  // ==========================================================================

  // 22 NITRO — the charge. A dense, thin, strongly-beading liquid that is the
  // most ignition-sensitive thing in the roster: ignitePt 40 deg, twenty above
  // the room, below even DUST's 60.
  //
  // WHY 40 AND NOT 200: exactly the §12.5(b) ceiling every combustible in this
  // table has run into. HT_TERM caps ONE neighbour at 1.0 deg of transfer per
  // substep, so a burning neighbour that lives L substeps can hand its
  // neighbour at most L degrees however hot it is — and BLAST lives 33. A
  // detonation front therefore has ~33 neighbour-degrees to work with per
  // contact. MEASURED, 12x11 charge with a 12-particle FIRE seed in a carved
  // pocket, particles of the charge consumed in 800 substeps:
  //     ignitePt 200 ->   0/168      ignitePt 90 ->  14/156
  //     ignitePt 140 ->   0/168      ignitePt 60 ->  63/156
  //     ignitePt  45 -> 119/156      ignitePt 40 -> 119/120   <- SHIPPED
  //
  // gammaCoh 900 IS PART OF THE DETONATION, not a look. At the cohesion I first
  // wrote (150) the blast blew the unreacted charge apart faster than the front
  // could cross it and 37 of 156 particles survived; at 900 the charge holds
  // itself together long enough to be consumed. Measured at ignitePt 45:
  //     gammaCoh 150 -> 119/156 consumed      gammaCoh 900 -> 129/156
  //     gammaCoh 900 + BLAST.gammaCoh -1200 -> 150/156
  // DISTINCT ON: density (1700, between stone and metal), cohesion (900 with
  // xsph 900 — nothing else is that sticky AND that thin) and reactions.
  mat({ xsph: 900, gammaCoh: 900, gammaCur: 225, adhesion: 40, grav: 1700,
        cond: 9000, coolRate: 150,
        ignitePt: DEG(40), fuel0: DEG(4), burnRate: 260000, burnHeat: 1500000,
        burnTo: MAT_BLAST,
        spawnTemp: DEG(20), tint: 0xe8e030 }),

  // 23 BLAST — the pressure wave, and the ONLY NEGATIVE COHESION IN THE TABLE.
  //
  // §6.2's surface-tension term is  f_ij = -gammaCoh * C(|d|) * d_ij , so a
  // NEGATIVE gammaCoh is a pair-symmetric REPULSION: every blast particle pushes
  // every neighbour away, which is what "expands and pressurises" means when the
  // only pressure the solver has is a density constraint that (by construction)
  // pushes apart under compression and does nothing under rarefaction. It is
  // still the same clamped, symmetric, order-independent accumulator — the
  // clamp is [-ST_TERM, +ST_TERM], so the negative side is bounded exactly like
  // the positive side and the third law survives. mat_mix((-2500 + coh_j) >> 1)
  // means a blast pushes on ANY neighbour, not just on itself.
  //
  // -2500 IS WHERE IT STOPS BEING FREE. MEASURED, the E15 scene, peak outward
  // displacement of two free-standing sand walls and the peak number of
  // particles pinned at the VMAX velocity clamp:
  //     gammaCoh  +150 (control)  push L 16.9  R 13.0 wu   vclamp  0
  //     gammaCoh -1200            push L 22.1  R 23.1 wu   vclamp  0
  //     gammaCoh -2500            push L 26.8  R 29.3 wu   vclamp  0   <- SHIPPED
  //     gammaCoh -4000            push L 27.6  R 28.9 wu   vclamp 11
  // Below -2500 the extra strength buys ~1 wu of throw and starts pinning
  // particles at VMAX, which is saturation and is reported rather than shipped.
  // dbg was [0 x 8] at every setting, including -4000: nothing in the fixed-point
  // core saturates. It is the velocity clamp that gives first, and E15 gates it.
  //
  // burnTo IS FIRE, NOT SMOKE. The blast is over in 33 substeps; the fireball it
  // leaves lives another 151 and is what actually sets the room alight. That one
  // field is the difference between an explosion and a bang.
  mat({ xsph: 1200, gammaCoh: -2500, gammaCur: -625, adhesion: 0, grav: -250,
        cond: 24000, coolRate: 700,
        ignitePt: TEMP_MIN, fuel0: DEG(22), burnRate: 44000, burnHeat: 1400000,
        burnTo: MAT_FIRE,
        spawnTemp: DEG(1800), tint: 0xfff0a0 }),

  // ==========================================================================
  // v7 (SPEC §17). RETIRED MATTER.
  // ==========================================================================

  // 24 VOID — NOT A MATERIAL. The absence of one, and the only row in the table
  // whose job is to not be there.
  //
  // WHY IT NEEDS SOLVER SUPPORT AT ALL, when §12.4's rule is that a material
  // must be expressible as parameters plus a threshold: because a particle that
  // is PRESENT contributes to its neighbours' density no matter what its row
  // says. Density is geometry, not material. Zeroing every force here would
  // still leave an invisible obstacle in the middle of the fluid. So `phantom`
  // reaches into the solver, and it reaches in for EIGHT LINES: `predict`
  // freezes it, `buildNbr` gives it no neighbours and lets nobody see it, and
  // `thermal` sets FLAG_DEAD from this row. Nothing else in sim.wgsl knows a
  // retired particle exists, and no material id appears anywhere in the shader.
  //
  // EVERY FIELD HERE IS ZERO ON PURPOSE and createEngine asserts it. `eps` sits
  // at the §2.3 floor and `dpMax` is 1 only to satisfy the existing validators
  // (both must be positive); neither is ever read in anger, because a phantom
  // has no neighbours and `solveA` with cnt == 0 produces rho = W(0) = 214942
  // against RHO0 = 1048576, and the density constraint is COMPRESSION-ONLY, so
  // Cq = 0, lam = 0, dp = 0. A zero-neighbour particle produces exactly zero
  // position correction STRUCTURALLY, not by tuning.
  //
  // It is invisible (render.js look class 4, culled in the vertex shader), it is
  // not paintable, it has no keyboard letter and it is filtered out of every
  // list a player sees — all of that on `phantom`, never on the id 24.
  mat({ phantom: 1,
        eps: 32768, dpMax: 1, xsph: 0, gammaCoh: 0, gammaCur: 0, adhesion: 0, grav: 0,
        cond: 0, coolRate: 0, fric: 0,
        spawnTemp: DEG(20), tint: 0x000000 }),

  // ==========================================================================
  // v8 (SPEC §18). THE MATERIAL WHOSE WHOLE IDENTITY IS A BOND RAMP.
  // ==========================================================================

  // 25 RICE — raw it POURS, cooked it SETS. The only material in the roster
  // whose thermal behaviour is entirely in the bond block: no melt, no boil, no
  // burn, no corrosion. `rampSign: +1` over 45..95 deg means kEff is exactly 0
  // at room temperature and 1.0 once it is hot, so a raw grain and a cooked
  // grain of the SAME material behave completely differently.
  //
  // IT IS ALSO THE SCENE THAT FOUND A BLOCKER. Raw RICE has bondK != 0 and
  // kEff == 0, which is the exact combination that made bonds IMMORTAL in the
  // draft design: the gather was entered on the ramped stiffness, so the hard
  // break never ran, and the block held 1004 bonds stretched to 316 wu — past
  // the point where d2 overflows i32. test/bonds.html B5 is that scene, kept as
  // a permanent gate. `min` (not mat_mix) mixing is what makes a raw grain
  // bonded to a cooked one carry ZERO force rather than half.
  //
  // ε_break 0.50 is deliberately loose: 2-5 % shatters (ICE at 2.4 % and WOOD
  // at 4.6 % went 652 bonds -> 8 in 200 substeps on a 1 wu drop while still
  // reporting K = 1.0). Grains want to stay stuck to their neighbours through a
  // pour and only give up when genuinely pulled apart.
  mat({ fric: 114688,
        xsph: 10000, gammaCoh: 80, gammaCur: 20, adhesion: 0, grav: 1500,
        cond: 3000, coolRate: 90,
        spawnTemp: DEG(20), tint: 0xeee4c6,
        bondK: 16384, bondBreak: 147456, bondReform: 1,
        bondFormV: 2600, bondFormP: 2000, bondMu: 0, bondGroup: 6,
        rampLo: DEG(45), rampHi: DEG(95), rampSign: 1,
        // §20: BOND, not RIGID — Darien's call, and it is right. RICE IS
        // STICKY, NOT RIGID: a cooked pot should squash when you press it.
        solidMode: SOLID_BOND }),

  // ==========================================================================
  // v15 (SPEC §26.F). THE MATERIAL THAT NOTHING HAPPENS TO.
  // ==========================================================================

  // 26 BEDROCK — Darien asked for it by name: "Do we maybe need a completely
  // inert material to build indestructible walls/wheels/rigid bodies?"
  //
  // YES, AND THE REASON IS THAT EVERY OTHER SOLID IN THE ROSTER IS A REAGENT.
  // Measured on the shipped table before this row existed: of the four
  // SOLID_RIGID rows a machine could be built from, ICE melts at ambient, WOOD
  // burns, STONE melts at 950 deg and is eaten by ACID into SAND, and METAL
  // melts at 1200 and is eaten by ACID into GAS. There was no way to build a
  // waterwheel and then run a fire experiment next to it — the apparatus was
  // always part of the experiment. That is a real gap in a sandbox whose whole
  // point is machines, and it is exactly what OE-CAKE's own `Wall` was for.
  //
  // INERT IS A SET OF ABSENCES, AND EVERY ONE OF THEM IS A TABLE DEFAULT:
  //   meltPt / boilPt   NEVER_HOT   — no phase change upwards
  //   freezePt / condPt NEVER_COLD  — none downwards either
  //   ignitePt / fuel0  NEVER_HOT/0 — nothing sets it alight
  //   corrPt            NEVER_HOT   — ACID runs off it
  //   corrode           0           — and it attacks nothing itself
  //   cond / coolRate   0 / 0       — it does not even carry warmth along
  // so the row states them by NOT stating them. `cond: 0` with `coolRate: 0`
  // and no reachable transition is the exact combination the tooltip's own
  // derivation calls out as *"Heat and cold do nothing to it whatsoever"* —
  // which is how a player finds out, with nobody writing that sentence for it.
  //
  // WHAT IT IS NOT: it is not a WALL. §19.A's anchored matter is what pins
  // something in place, and it is a property of the gesture. BEDROCK is matter
  // — it falls, it has weight, you can build a wheel out of it and spin it.
  // Anchor it and you get an indestructible wall; do not, and you get an
  // indestructible cart. Two mechanisms, composable, which is the §26 insight
  // (property bits, not a list of materials) applied in the small.
  //
  // grav 2400 makes it the HEAVIEST row in the roster (METAL 2000, SAND 1800),
  // which is both the honest reading of "the rock under everything" and the
  // second axis E0's no-filler rule needs against STONE and METAL — the first
  // being that its reaction signature is empty where theirs are not.
  //
  // §18: bonds like STONE's and METAL's — `bondBreak: 0` means the SOFT break
  // never fires and only §18's hard break at the kernel radius can sever it —
  // in its own `bondGroup` (9), so a bedrock frame does not weld itself to the
  // stone it is holding up. `rampSign: 0` takes `bond_k`'s INERT branch: there
  // is no melt window to soften towards, so its stiffness is a constant.
  //
  // §20: SOLID_RIGID. A drawn bedrock wheel is one object with zero flexion,
  // and it is the row that finally makes "build a machine and then set fire to
  // the room" a thing you can do.
  mat({ fric: 131072,
        xsph: 40000, gammaCoh: 900, adhesion: 500, grav: 2400,
        cond: 0, coolRate: 0,
        spawnTemp: DEG(20), tint: 0x4a3a8c,
        bondK: 32768, bondBreak: 0, bondReform: 0, bondMu: 0, bondGroup: 9,
        // Absolute: it seals but can never be released by pressure.
        burstP: 0,
        rampSign: 0,
        solidMode: SOLID_RIGID }),

  // 27 ORBIES — independent, non-bonding beads. The playful "gumball machine"
  // identity comes from their per-particle diffraction renderer, not a hidden
  // cohesion trick: neighbouring Orbies can touch and jostle, but never fuse
  // into one soft body. Like Gel, they are thermally inert and non-reactive.
  mat({ fric: 53248,
        xsph: 10500, gammaCoh: 70, gammaCur: 110, adhesion: 0, grav: 1200,
        cond: 0, coolRate: 0,
        spawnTemp: DEG(20), tint: 0xff84e8 }),
];

// ============================================================================
// Deterministic integer PRNGs. These are the ONLY sources of "randomness" that
// may touch sim state. Both are pure integer, seeded, and reproducible.
// ============================================================================
// ============================================================================
// §20 — RIGID BODY HOST MATH
// Everything here is INTEGER and runs ONCE PER BODY at weld time, on the host,
// exactly as the kernel LUT bake and the `rampInv` bake do. Nothing in this
// block runs inside a substep, and the shader mirrors none of it — it consumes
// the numbers it produces.
// ============================================================================

/** The Q22 quarter-turn cosine table the shader interpolates. RG_QN+2 entries;
 *  the last is a zero pad that a frac of 0 can index but never weight. */
export function buildCosTable() {
  const T = new Int32Array(RG_COS_N);
  for (let k = 0; k <= RG_QN; k++) T[k] = Math.round(Math.cos((k * Math.PI) / (2 * RG_QN)) * COS_ONE);
  T[RG_QN] = 0;                       // cos(pi/2) is EXACTLY zero
  T[RG_QN + 1] = 0;
  return T;
}

/** Round-half-to-EVEN integer divide. Sign-symmetric: idivRne(-a,n) === -idivRne(a,n). */
export function idivRne(a, n) {
  if (n <= 0) return 0;
  const neg = a < 0, x = neg ? -a : a;
  let q = Math.floor(x / n);
  const r2 = 2 * (x - q * n);
  if (r2 > n || (r2 === n && (q & 1) === 1)) q += 1;
  return neg ? -q : q;
}

/** floor(sqrt(x)) for an exact integer x < 2^52. The float sqrt only SEEDS. */
export function isqrt64(x) {
  if (x <= 0) return 0;
  let r = Math.floor(Math.sqrt(x));
  while (r > 0 && r * r > x) r--;
  while ((r + 1) * (r + 1) <= x) r++;
  return r;
}

/**
 * THE THREE PER-BODY PRE-SHIFTS, chosen from the FULL member list so they stay
 * valid for every member set the body can ever have. NEITHER IS EVER
 * RECOMPUTED, so a membership event cannot change them — which is what makes
 * the accumulator bounds hold through a body burning down to nothing.
 *
 *   rsh — |o >> rsh| <= RS_TARGET, so one raw torque term is at most
 *         2 * 2048 * dpMax and TQ_TERM_R (2^27) is 2x that.
 *   ish — SUM over ALL members of (o >> ish)^2 < 2^30, so the inertia sum
 *         cannot overflow no matter which members are alive.
 *   osh — n * (rMax >> osh) < 2^30, so the re-base centroid sum cannot
 *         overflow. It is an EXACT floor shift on both sides, which is what
 *         makes the re-base idempotent: (o - (d<<osh)) >> osh === (o>>osh) - d.
 */
export function rigidShifts(ox, oy) {
  const N = ox.length;
  let rMax = 0;
  for (let k = 0; k < N; k++) {
    const a = Math.abs(ox[k]), b = Math.abs(oy[k]);
    const m = isqrt64(a * a + b * b);
    if (m > rMax) rMax = m;
  }
  let rsh = 0; while ((rMax >> rsh) > RS_TARGET) rsh++;
  let ish = 0;
  for (;;) {
    let sum = 0;
    for (let k = 0; k < N; k++) {
      const a = ox[k] >> ish, b = oy[k] >> ish;
      sum += a * a + b * b;
    }
    if (sum < SUMI_MAX) break;
    ish++;
    if (ish > 30) throw new Error('[aether] §20: body too large for an i32 inertia sum');
  }
  let osh = 0; while (N * (rMax >> osh) >= SUMI_MAX) osh++;
  return { rMax, rsh, ish, osh };
}

/**
 * invI and iq from the live inertia sum — THE HOST TWIN OF sim.wgsl's
 * `rg_inv_inertia`, and it must agree with it to the bit. It exists so a body
 * record is complete the instant it is written, before any substep runs.
 */
export function rigidInvInertia(sumIs2, rsh, ish) {
  if (sumIs2 <= 0) return { invI: 0, iq: 0 };
  const bl = 32 - Math.clz32(sumIs2);
  const iqNat = bl - 1 + 2 * ish - rsh;
  const iq = Math.min(iqNat, RG_IQ_MAX);
  const E = iq + rsh - 2 * ish;
  if (iqNat < RG_IQ_MIN || E < 1 || E > 62) return { invI: 0, iq: 0 };
  // RNE(BAM_PER_RAD * 2^E / sumIs2), computed exactly in BigInt so the host and
  // fp_div_s agree bit for bit at every E.
  const num = BigInt(BAM_PER_RAD) << BigInt(E);
  const den = BigInt(sumIs2);
  let q = num / den;
  const r = num - q * den;
  if (2n * r > den || (2n * r === den && (q & 1n) === 1n)) q += 1n;
  if (q > 2147483647n) q = 2147483647n;
  return { invI: Number(q), iq };
}

export function xorshift32(seed) {
  let s = (seed >>> 0) || 0x2545f491;
  return () => {
    s ^= (s << 13); s >>>= 0;
    s ^= (s >>> 17);
    s ^= (s << 5);  s >>>= 0;
    return s >>> 0;
  };
}

export function splitmix32(a) {
  a = (a + 0x9e3779b9) | 0;
  let t = a;
  t = Math.imul(t ^ (t >>> 16), 0x21f0aaad);
  t = Math.imul(t ^ (t >>> 15), 0x735a2d97);
  return (t ^ (t >>> 15)) >>> 0;
}

// ============================================================================
// Small utilities
// ============================================================================
const assert = (cond, msg) => { if (!cond) throw new Error('[aether] ' + msg); };
const ceilDiv = (a, b) => Math.ceil(a / b) | 0;

async function fetchText(url, who) {
  let res;
  try {
    res = await fetch(url, { cache: 'no-store' });
  } catch (e) {
    throw new Error(`[aether] could not fetch ${url}: ${e.message}`);
  }
  if (!res.ok) {
    throw new Error(
      `[aether] MISSING SHADER: ${url} (HTTP ${res.status}).` +
      (who ? ` This file is owned by ${who}; the engine cannot build pipelines without it.` : '')
    );
  }
  return await res.text();
}

/** Create a shader module and HARD FAIL on any warning or error. */
async function makeModule(device, code, label) {
  const module = device.createShaderModule({ code, label });
  const ci = await module.getCompilationInfo();
  const bad = ci.messages.filter((m) => m.type !== 'info');
  if (bad.length) {
    const lines = code.split('\n');
    const detail = bad.map((m) => {
      const src = lines[m.lineNum - 1] ?? '';
      return `  ${m.type} L${m.lineNum}:${m.linePos} ${m.message}\n    | ${src.trim()}`;
    }).join('\n');
    throw new Error(`[aether] shader "${label}" failed to compile:\n${detail}`);
  }
  return module;
}

// ============================================================================
// WGSL reflection
// ----------------------------------------------------------------------------
// Explicit bind group layouts are mandatory, but hand-maintaining a
// binding->pass table across three implementers is exactly the kind of drift
// that produces "[Invalid BindGroup] is invalid due to a previous error".
// So the layouts are DERIVED from the shader text: parse the global resource
// declarations (binding number, address space, access mode), build the
// function call graph, and take the transitive closure of globals each entry
// point actually reaches. The result is minimal, access-exact, and cannot
// disagree with whatever implementer F wrote.
// ============================================================================

export function stripComments(src) {
  let out = '';
  for (let i = 0; i < src.length; i++) {
    if (src[i] === '/' && src[i + 1] === '/') {
      while (i < src.length && src[i] !== '\n') i++;
      out += '\n';
    } else if (src[i] === '/' && src[i + 1] === '*') {
      let depth = 1; i += 2;
      while (i < src.length && depth > 0) {
        if (src[i] === '/' && src[i + 1] === '*') { depth++; i += 2; }
        else if (src[i] === '*' && src[i + 1] === '/') { depth--; i += 2; }
        else { if (src[i] === '\n') out += '\n'; i++; }
      }
      i--;
    } else {
      out += src[i];
    }
  }
  return out;
}

function matchBrace(src, openIdx) {
  let depth = 0;
  for (let i = openIdx; i < src.length; i++) {
    if (src[i] === '{') depth++;
    else if (src[i] === '}') { depth--; if (depth === 0) return i; }
  }
  return -1;
}

function matchParen(src, openIdx) {
  let depth = 0;
  for (let i = openIdx; i < src.length; i++) {
    if (src[i] === '(') depth++;
    else if (src[i] === ')') { depth--; if (depth === 0) return i; }
  }
  return -1;
}

const RE_GLOBAL =
  /@group\s*\(\s*(\d+)\s*\)\s*@binding\s*\(\s*(\d+)\s*\)\s*var\s*(?:<\s*([A-Za-z_]+)\s*(?:,\s*([A-Za-z_]+)\s*)?>)?\s*([A-Za-z_]\w*)\s*:/g;

/**
 * Reflect a (comment-free) WGSL source.
 * @returns {{globals:Map<string,object>, entries:Map<string,object>}}
 */
export function reflectWgsl(rawSrc) {
  const src = stripComments(rawSrc);

  // ---- global resources
  const globals = new Map();
  RE_GLOBAL.lastIndex = 0;
  let m;
  while ((m = RE_GLOBAL.exec(src)) !== null) {
    const [, group, binding, space, access, name] = m;
    globals.set(name, {
      name,
      group: Number(group),
      binding: Number(binding),
      space: space || 'handle',
      access: access || (space === 'storage' ? 'read' : 'read'),
    });
  }

  // ---- function bodies + attributes
  const fns = new Map();
  const RE_FN = /\bfn\s+([A-Za-z_]\w*)\s*\(/g;
  let f;
  while ((f = RE_FN.exec(src)) !== null) {
    const name = f[1];
    const parenOpen = f.index + f[0].length - 1;
    const parenClose = matchParen(src, parenOpen);
    if (parenClose < 0) continue;
    const braceOpen = src.indexOf('{', parenClose);
    if (braceOpen < 0) continue;
    const braceClose = matchBrace(src, braceOpen);
    if (braceClose < 0) continue;

    // Attributes are whatever sits between the previous statement terminator
    // and the `fn` keyword.
    let attrStart = f.index;
    for (let i = f.index - 1; i >= 0; i--) {
      const c = src[i];
      if (c === ';' || c === '}') { attrStart = i + 1; break; }
      if (i === 0) attrStart = 0;
    }
    const attrs = src.slice(attrStart, f.index);
    const params = src.slice(parenOpen, parenClose + 1);
    const body = src.slice(braceOpen, braceClose + 1);

    const isCompute = /@compute\b/.test(attrs);
    let wg = null;
    const wgm = /@workgroup_size\s*\(([^)]*)\)/.exec(attrs);
    if (wgm) wg = wgm[1].split(',').map((s) => s.trim());

    fns.set(name, {
      name, isCompute, wg, body,
      attrStart, fnStart: f.index, bodyStart: braceOpen, bodyEnd: braceClose,
      tokens: new Set((params + body).match(/[A-Za-z_]\w*/g) || []),
    });
    RE_FN.lastIndex = braceClose;
  }

  // ---- transitive closure of reachable globals
  const fnNames = new Set(fns.keys());
  const globalNames = new Set(globals.keys());
  const cache = new Map();
  function reach(name, stack) {
    if (cache.has(name)) return cache.get(name);
    if (stack.has(name)) return new Set();
    stack.add(name);
    const fn = fns.get(name);
    const used = new Set();
    if (fn) {
      for (const t of fn.tokens) {
        if (globalNames.has(t)) used.add(t);
        else if (fnNames.has(t) && t !== name) {
          for (const g of reach(t, stack)) used.add(g);
        }
      }
    }
    stack.delete(name);
    cache.set(name, used);
    return used;
  }

  const entries = new Map();
  for (const [name, fn] of fns) {
    if (!fn.isCompute) continue;
    const used = reach(name, new Set());
    const res = [...used].map((g) => globals.get(g))
      .sort((a, b) => a.binding - b.binding);
    entries.set(name, { name, wg: fn.wg, resources: res, fn, fluidOnly: guardsOnNFluid(fn.body) });
  }
  return { globals, entries, fns };
}

// ─────────────────────────────────────────────────────────────────────────────
// DISPATCH SIZING (§5, perf). A pass whose FIRST statement pair is
//     let <id> = gid.x;  if (<id> >= P.nFluid) { return; }
// provably does nothing for any invocation with gid.x >= nFluid: the guard is
// the first thing executed and it returns before touching a single buffer. Such
// a pass may therefore be dispatched over nFluid instead of n, and 50 of the 58
// dispatches in a substep qualify. Skipping threads that would have returned
// immediately cannot change any output, any debug counter or any hash — but
// because that is exactly the kind of claim that rots, it is DERIVED FROM THE
// SHADER TEXT rather than kept in a hand-maintained list here. Loosen the guard
// in sim.wgsl and this returns false on its own, and the pass silently goes
// back to dispatching over n instead of silently going wrong.
//
// The regex is deliberately strict — an `i` bound to anything but gid.x, an
// extra statement before the guard, or a guard on P.n all fail it closed.
// test/perf/dispatch.html asserts the classification matches the shipped
// shader entry point by entry point, and the determinism gate asserts the
// chain is unmoved.
// ─────────────────────────────────────────────────────────────────────────────
const RE_NFLUID_GUARD =
  /^\s*\{\s*let\s+([A-Za-z_]\w*)\s*(?::\s*u32\s*)?=\s*gid\s*\.\s*x\s*;\s*if\s*\(\s*\1\s*>=\s*P\s*\.\s*nFluid\s*\)\s*\{\s*return\s*;\s*\}/;
export function guardsOnNFluid(body) {
  return typeof body === 'string' && RE_NFLUID_GUARD.test(body);
}

function layoutEntryFor(res) {
  if (res.space === 'uniform') return { buffer: { type: 'uniform' } };
  if (res.space === 'storage') {
    return { buffer: { type: res.access === 'read_write' ? 'storage' : 'read-only-storage' } };
  }
  throw new Error(`[aether] unsupported address space "${res.space}" on binding ${res.binding}`);
}

// ============================================================================
// Scenes — integer-only construction, seeded, no float reaching sim state
// ============================================================================

function emptyScene(n, nFluid, domWwu, domHwu) {
  return {
    n, nFluid,
    pos: new Int32Array(n * 2),
    vel: new Int32Array(n * 2),
    matId: new Uint32Array(n),
    domW: (domWwu * ONE) | 0,
    domH: (domHwu * ONE) | 0,
    cellShift: CELL_SHIFT,
  };
}

// NOTE ON THE Y AXIS: `Mat.grav` is POSITIVE and `predict` ADDS it to v.y, so
// +y is the gravity direction. Everything below treats +y as DOWN: the floor
// sits at high y, and render.js flips y for display.

/**
 * Static container: 2 floor rows at the bottom, plus 2 columns up each side.
 * 128 x 72 wu with wallRows = 60 gives exactly 2*128 + 4*60 = 496 particles.
 */
function boundaryRing(domWwu, domHwu, wallRows) {
  const out = [];
  const half = ONE >> 1;
  for (let row = 0; row < 2; row++) {
    const y = domHwu * ONE - half - row * ONE;         // domH-0.5, domH-1.5
    for (let c = 0; c < domWwu; c++) out.push([half + c * ONE, y]);
  }
  for (const xBase of [half, domWwu * ONE - half]) {
    for (let col = 0; col < 2; col++) {
      const x = xBase === half ? half + col * ONE : xBase - col * ONE;
      for (let r = 0; r < wallRows; r++) {
        out.push([x, domHwu * ONE - half - (2 + r) * ONE]);
      }
    }
  }
  return out;
}

/**
 * Closed viewport collision ring whose particle centres all sit just outside
 * the domain. The renderer can therefore hide boundary matter entirely while
 * simulated matter reaches the very last visible pixel instead of resting
 * against a row of dark circles inset from the screen edge.
 */
function boundaryRingFullBleed(domWwu, domHwu) {
  const out = [];
  const half = ONE >> 1;
  for (let row = 0; row < 2; row++) {
    const top = -half - row * ONE;
    const bottom = domHwu * ONE + half + row * ONE;
    for (let c = 0; c < domWwu; c++) {
      const x = half + c * ONE;
      out.push([x, top], [x, bottom]);
    }
  }
  for (let col = 0; col < 2; col++) {
    const left = -half - col * ONE;
    const right = domWwu * ONE + half + col * ONE;
    for (let r = 0; r < domHwu; r++) {
      const y = half + r * ONE;
      out.push([left, y], [right, y]);
    }
  }
  return out;
}

/**
 * Build a canonical scene. Integer construction only — the only entropy is a
 * seeded xorshift32, and it is quantised to exact ticks.
 *
 * 'dambreak_v1' is the determinism-gate scene and its definition is FROZEN.
 * Changing any number below changes every golden hash.
 */
export function makeScene(name, opts = {}) {
  const DOM_W = 128, DOM_H = 72;

  if (name === 'dambreak_v1') {
    // 60x60 water column standing on the floor against the left wall, plus 496
    // static boundary particles => n = 4096, nFluid = 3600.
    // FROZEN: this is the determinism-gate scene. Every number below is part of
    // the golden hash.
    const CW = 60, CH = 60;
    const fluid = [];
    const rnd = xorshift32(0x1234567);
    for (let gy = 0; gy < CH; gy++) {
      for (let gx = 0; gx < CW; gx++) {
        // +-0.031 wu deterministic jitter: breaks the perfect lattice so the
        // liveness metrics measure fluid, not a frozen crystal.
        const jx = (rnd() % 4097) - 2048;
        const jy = (rnd() % 4097) - 2048;
        fluid.push([3 * ONE + gx * ONE + jx, (DOM_H - 3) * ONE - gy * ONE + jy, MAT_WATER]);
      }
    }
    const bnd = boundaryRing(DOM_W, DOM_H, 60);
    const s = emptyScene(fluid.length + bnd.length, fluid.length, DOM_W, DOM_H);
    for (let i = 0; i < fluid.length; i++) {
      s.pos[i * 2] = fluid[i][0]; s.pos[i * 2 + 1] = fluid[i][1];
      s.matId[i] = fluid[i][2];
    }
    for (let k = 0; k < bnd.length; k++) {
      const i = fluid.length + k;
      s.pos[i * 2] = bnd[k][0]; s.pos[i * 2 + 1] = bnd[k][1];
      s.matId[i] = MAT_WATER;
    }
    return s;
  }

  if (name === 'overdense_v1') {
    // ── THE ORDERING CONTROL SCENE. FROZEN. ────────────────────────────────────
    // A 32x32 lattice at 0.5 wu — 4x rest density — which is dense enough that
    // buildNbr's MAXNBR truncation fires (dbg[6] > 0) on the very first substep.
    //
    // WHY IT EXISTS. Truncation is the ONLY order-sensitive consumer in the whole
    // pipeline: it decides WHICH neighbours to drop. Every other pair loop is a
    // per-term-clamped plain i32 sum and is therefore commutative. dambreak_v1 —
    // the gate's only scene until 2026-07-25 — tops out at 22 real neighbours
    // against a cap of 48, so it can never enter that path, which made the gate
    // structurally blind to a `canonicalize` regression. Measured on this scene:
    // with canonicalize shipped, 5/5 runs bit-identical; with it defeated,
    // 5 unique chains out of 5 runs. See test/determinism.html stage C.
    //
    // This scene is NOT physically sane and is not meant to be — it is a
    // deliberately over-dense state used to exercise the truncation path. It is
    // exempt from the dbg[6] == 0 assertion for exactly that reason.
    const SIDE = 32, SP = ONE >> 1;      // 0.5 wu
    const cx = 64 * ONE, cy = 40 * ONE;
    const fluid = [];
    for (let gy = 0; gy < SIDE; gy++)
      for (let gx = 0; gx < SIDE; gx++)
        fluid.push([cx + (gx - SIDE / 2) * SP, cy + (gy - SIDE / 2) * SP]);
    const bnd = boundaryRing(DOM_W, DOM_H, 60);
    const s = emptyScene(fluid.length + bnd.length, fluid.length, DOM_W, DOM_H);
    for (let i = 0; i < fluid.length; i++) {
      s.pos[i * 2] = fluid[i][0]; s.pos[i * 2 + 1] = fluid[i][1]; s.matId[i] = MAT_WATER;
    }
    for (let k = 0; k < bnd.length; k++) {
      const i = fluid.length + k;
      s.pos[i * 2] = bnd[k][0]; s.pos[i * 2 + 1] = bnd[k][1]; s.matId[i] = MAT_WATER;
    }
    return s;
  }

  if (name === 'pool_v1') {
    const CW = opts.w ?? 100, CH = opts.h ?? 24;
    const mat = opts.mat ?? MAT_WATER;
    // Fail here with the actual cause rather than in loadScene with a bare
    // "pos outside [0, 2^27]" thirty frames of head-scratching later.
    assert(13 + CW <= DOM_W - 2 && CH <= DOM_H - 5,
      `pool_v1 w=${CW} h=${CH} does not fit the ${DOM_W}x${DOM_H} wu domain ` +
      `(max w=${DOM_W - 15}, max h=${DOM_H - 5})`);
    const fluid = [];
    const rnd = xorshift32(0x0be7a1);
    for (let gy = 0; gy < CH; gy++) {
      for (let gx = 0; gx < CW; gx++) {
        const jx = (rnd() % 4097) - 2048;
        const jy = (rnd() % 4097) - 2048;
        fluid.push([13 * ONE + gx * ONE + jx, (DOM_H - 3) * ONE - gy * ONE + jy, mat]);
      }
    }
    const bnd = boundaryRing(DOM_W, DOM_H, 60);
    const s = emptyScene(fluid.length + bnd.length, fluid.length, DOM_W, DOM_H);
    for (let i = 0; i < fluid.length; i++) {
      s.pos[i * 2] = fluid[i][0]; s.pos[i * 2 + 1] = fluid[i][1]; s.matId[i] = fluid[i][2];
    }
    for (let k = 0; k < bnd.length; k++) {
      const i = fluid.length + k;
      s.pos[i * 2] = bnd[k][0]; s.pos[i * 2 + 1] = bnd[k][1]; s.matId[i] = MAT_WATER;
    }
    return s;
  }

  if (name === 'droplet_v1') {
    // A flattened blob released in free space: beads up under surface tension.
    const mat = opts.mat ?? MAT_WATER;
    const RX = opts.rx ?? 13, RY = opts.ry ?? 2;
    const cx = 64 * ONE, cy = 22 * ONE;
    const fluid = [];
    for (let gy = -RY; gy <= RY; gy++) {
      for (let gx = -RX; gx <= RX; gx++) {
        const nx = gx / RX, ny = gy / RY;
        if (nx * nx + ny * ny > 1.0) continue;
        fluid.push([cx + gx * ONE, cy + gy * ONE, mat]);
      }
    }
    const bnd = boundaryRing(DOM_W, DOM_H, 60);
    const s = emptyScene(fluid.length + bnd.length, fluid.length, DOM_W, DOM_H);
    for (let i = 0; i < fluid.length; i++) {
      s.pos[i * 2] = fluid[i][0]; s.pos[i * 2 + 1] = fluid[i][1]; s.matId[i] = fluid[i][2];
    }
    for (let k = 0; k < bnd.length; k++) {
      const i = fluid.length + k;
      s.pos[i * 2] = bnd[k][0]; s.pos[i * 2 + 1] = bnd[k][1]; s.matId[i] = MAT_WATER;
    }
    return s;
  }

  if (name === 'stretch_v1') {
    // Two blobs with opposing velocities — necking / snapping test.
    const mat = opts.mat ?? MAT_GOO;
    const pull = opts.pull ?? 8000;
    const fluid = [];
    const R = 6;
    for (const side of [-1, 1]) {
      const cx = 64 * ONE + side * 8 * ONE, cy = 36 * ONE;
      for (let gy = -R; gy <= R; gy++) {
        for (let gx = -R; gx <= R; gx++) {
          if (gx * gx + gy * gy > R * R) continue;
          fluid.push([cx + gx * ONE, cy + gy * ONE, mat, side * pull, 0]);
        }
      }
    }
    const bnd = boundaryRing(DOM_W, DOM_H, 60);
    const s = emptyScene(fluid.length + bnd.length, fluid.length, DOM_W, DOM_H);
    for (let i = 0; i < fluid.length; i++) {
      s.pos[i * 2] = fluid[i][0]; s.pos[i * 2 + 1] = fluid[i][1];
      s.vel[i * 2] = fluid[i][3]; s.vel[i * 2 + 1] = fluid[i][4];
      s.matId[i] = fluid[i][2];
    }
    for (let k = 0; k < bnd.length; k++) {
      const i = fluid.length + k;
      s.pos[i * 2] = bnd[k][0]; s.pos[i * 2 + 1] = bnd[k][1]; s.matId[i] = MAT_WATER;
    }
    return s;
  }

  if (name === 'empty_v1') {
    // Container only — the interactive toy's starting state.
    //
    // §27. THE CONTAINER IS THE ONLY SCENE WHOSE SIZE IS AN ARGUMENT, and it is
    // one because an OE-CAKE save is a WORLD, not a sticker: `importOecInto`
    // sizes the tank to the file rather than shrinking the file to the tank
    // (SPEC §27). Every other scene here is frozen at 128 x 72 because a golden
    // hash is a function of the domain.
    //
    // THE DEFAULT MUST STAY BYTE-IDENTICAL. `makeScene('empty_v1')` with no
    // options is the toy's opening state and the `empty_v1` arm of several
    // gates; 72 - 12 == 60 is exactly the wallRows this line has always passed,
    // written as a rule rather than as a number so a bigger tank gets the same
    // open top (2 floor rows, side columns stopping 12 wu short of the rim).
    const W = Math.max(8, opts.domWwu ?? DOM_W) | 0;
    const H = Math.max(16, opts.domHwu ?? DOM_H) | 0;
    const bnd = opts.fullBleed
      ? boundaryRingFullBleed(W, H)
      : boundaryRing(W, H, H - 12);
    const s = emptyScene(bnd.length, 0, W, H);
    for (let k = 0; k < bnd.length; k++) {
      s.pos[k * 2] = bnd[k][0]; s.pos[k * 2 + 1] = bnd[k][1]; s.matId[k] = MAT_WATER;
    }
    return s;
  }

  throw new Error(`[aether] unknown scene "${name}"`);
}

/** Deterministic permutation of a scene's fluid block (negative control). */
export function permuteScene(scene, seed) {
  const out = {
    ...scene,
    pos: scene.pos.slice(), vel: scene.vel.slice(), matId: scene.matId.slice(),
    // §12: temp/fuel/flags are per-particle state, so a permutation that did not
    // carry them would silently be a DIFFERENT scene, not a reordering of the
    // same one — and this function's only job is to be the same scene.
    temp: scene.temp ? scene.temp.slice() : null,
    fuel: scene.fuel ? scene.fuel.slice() : null,
    flags: scene.flags ? scene.flags.slice() : null,
  };
  const rnd = xorshift32(seed);
  const perm = new Int32Array(scene.nFluid);
  for (let i = 0; i < scene.nFluid; i++) perm[i] = i;
  for (let i = scene.nFluid - 1; i > 0; i--) {
    const j = rnd() % (i + 1);
    const t = perm[i]; perm[i] = perm[j]; perm[j] = t;
  }
  for (let i = 0; i < scene.nFluid; i++) {
    const s = perm[i];
    out.pos[i * 2] = scene.pos[s * 2]; out.pos[i * 2 + 1] = scene.pos[s * 2 + 1];
    out.vel[i * 2] = scene.vel[s * 2]; out.vel[i * 2 + 1] = scene.vel[s * 2 + 1];
    out.matId[i] = scene.matId[s];
    if (out.temp)  out.temp[i]  = scene.temp[s];
    if (out.fuel)  out.fuel[i]  = scene.fuel[s];
    if (out.flags) out.flags[i] = scene.flags[s];
  }
  return out;
}

// ============================================================================
// Pass list (§5). Data, not code — validated against the shader's entry points.
// ============================================================================
const PASSES = [
  // §20 TRUE RIGID BODIES, and it runs FIRST. A body integrates its own four
  // degrees of freedom and PLACES its members into derived.pred; `predict` then
  // skips them, because integrating a member a second time is exactly the
  // independent per-member motion §20 exists to remove.
  { fn: 'rigidStep',      dim: 'bodies' },
  { fn: 'predict',        dim: 'n'  },
  { fn: 'gridCount',      dim: 'n'  },
  { fn: 'scanBlock',      dim: 'b'  },
  { fn: 'scanTop',        dim: '1'  },
  { fn: 'scanAdd',        dim: 'c'  },
  { fn: 'scatter',        dim: 'n'  },
  { fn: 'canonicalize',   dim: 'n'  },
  { fn: 'buildNbr',       dim: 'n'  },
  { fn: 'solveA',         dim: 'n'  },
  // §5.1b 4-colour block Gauss-Seidel. One shared solveA per iteration, then the
  // four (solveB_cK, applyDp_cK) pairs in colour order 0,1,2,3. The colour
  // ORDER is wire format for product 2, exactly like ITERS (§11 R7).
  { fn: 'solveB_c0',      dim: 'n'  },
  { fn: 'applyDp_c0',     dim: 'n'  },
  { fn: 'solveB_c1',      dim: 'n'  },
  { fn: 'applyDp_c1',     dim: 'n'  },
  { fn: 'solveB_c2',      dim: 'n'  },
  { fn: 'applyDp_c2',     dim: 'n'  },
  { fn: 'solveB_c3',      dim: 'n'  },
  { fn: 'applyDp_c3',     dim: 'n'  },
  // §16 granular friction. A per-term-clamped gather (order-independent like
  // every other pair loop) plus its elementwise applicator. It runs INSIDE the
  // iteration loop, once per density solve: `friction` reads pred[j] and pos[j]
  // and writes only derived[i].dp, `applyFric` reads derived[i].dp and writes
  // only derived[i].pred — two dispatches, so neither reads another
  // invocation's write, exactly as §5.1 requires of solveB/applyDp.
  { fn: 'friction',       dim: 'n'  },
  { fn: 'applyFric',      dim: 'n'  },
  // §20. ONE dispatch per solver iteration, at the END of the iteration, where
  // every external correction for this member has already landed in
  // derived[i].dp: solveB wrote it (applyDp early-returned for the member) and
  // `friction` ADDED to it. One workgroup per body reduces both into the body's
  // force and torque, folds them with a CARRIED REMAINDER, and re-places every
  // member. Reduce + per-body update + place in ONE dispatch is what makes §20
  // cost +6 dispatches instead of +14.
  { fn: 'rigidSolve',     dim: 'bodies' },
  // §29: final-motion crossing clamp, before finalize turns pred into state.
  { fn: 'contain',        dim: 'n'  },
  { fn: 'containOverflow', dim: 'n' },
  { fn: 'finalize',       dim: 'n'  },
  { fn: 'xsph',           dim: 'n'  },
  { fn: 'normals',        dim: 'n'  },
  { fn: 'surfaceTension', dim: 'n'  },
  // §12 thermal tail. `conduct` is a per-term-clamped gather (order-independent
  // like every other pair loop); `thermal` is purely elementwise. They run LAST
  // so a material change lands on the NEXT substep — the same one-substep lag
  // stAcc has, and for the same reason (§5.2).
  { fn: 'conduct',        dim: 'n'  },
  { fn: 'thermal',        dim: 'n'  },
  { fn: 'markCongealNbr', dim: 'n'  },
  // §29: finite held walls release under crush before membership recount.
  { fn: 'containStress',  dim: 'n'  },
  // §20 membership, AFTER thermal so a member that melted, burned, corroded or
  // was retired THIS substep leaves on the same substep it stopped being the
  // material it was. IDEMPOTENT — with nothing leaving, the re-base offset is
  // exactly 0 and the pass is a no-op — so it is dispatched unconditionally,
  // with no dirty flag, no data-dependent branch and no host readback.
  { fn: 'rigidMember',    dim: 'bodies' },
  // §18 bond formation, at the TAIL and only on a formation substep. Running it
  // here is load-bearing, not tidy: after `finalize`, state_out[i].pos ==
  // derived[i].pred and `nbr` was built from `pred`, so the neighbour list and
  // the positions bondForm1 reads are the SAME positions — and a particle that
  // melted or transmuted this substep already carries its new matId.
  { fn: 'bondForm1',      dim: 'n'  },
  { fn: 'bondForm2',      dim: 'n'  },
];

// Entry points whose correctness does not depend on the workgroup size (used
// by the workgroup-invariance negative control). scanBlock / scanTop are
// EXCLUDED: they are hard-wired to 256 threads x 4 elements per block.
const WG_VARIABLE = new Set([
  'predict', 'gridCount', 'scatter', 'canonicalize', 'buildNbr',
  'solveA', 'finalize', 'xsph', 'normals', 'surfaceTension',
  'friction', 'applyFric',
  'solveB_c0', 'solveB_c1', 'solveB_c2', 'solveB_c3',
  'applyDp_c0', 'applyDp_c1', 'applyDp_c2', 'applyDp_c3',
  'conduct', 'thermal', 'markCongealNbr',
  'contain', 'containOverflow', 'containStress',
  'bondForm1', 'bondForm2',
  // §20's three passes are DELIBERATELY ABSENT: each reduces over exactly 256
  // workgroup lanes with a hard-wired tree, so they are as size-locked as
  // scanBlock / scanTop.
]);

// Binding -> buffer name (§3.3). One number, everywhere.
const BINDING_NAMES = {
  0: 'params', 1: 'state_in', 2: 'state_out', 3: 'derived',
  4: 'cellCount', 5: 'cellStart', 6: 'blockSums', 7: 'cellOf',
  8: 'bucketIds', 9: 'sortedIds', 10: 'nbr', 11: 'nbrN',
  12: 'luts', 13: 'mats', 14: 'hashState', 15: 'dbg',
  16: 'bond', 17: 'bondCand', 18: 'rigid',
};

// ============================================================================
// createEngine
// ============================================================================
export async function createEngine(opts = {}) {
  const {
    canvas = null,
    maxParticles = 20000,
    maxCells = 65536,
    debug = true,
    device: injectedDevice = null,
    submitBatch = 25,
    chain = true,
    simSource = null,
    fixedSource = null,
    tables = null,
    mats = MATS,
    iters = ITERS,
    tableHash = TABLE_HASH,
    workgroupOverride = 0,   // negative control: 0 = leave shader as authored
    // Negative control for the dispatch-sizing optimisation, same shape as
    // workgroupOverride. false forces every `dim: 'n'` pass back to dispatching
    // over n, which is what the engine did before 2026-07-26. A page can then
    // build one engine each way and require the chains to be identical, which
    // is the only honest way to claim an optimisation is free.
    fluidOnlyDispatch = true,
    // ---- PROFILING ONLY. Off by default and it must stay that way. ----------
    // `profile: true` requests the `timestamp-query` feature and makes
    // _encodeSubstep put EVERY dispatch in its own compute pass with a
    // timestampWrites pair, so `Engine.profileSubstep()` can attribute GPU
    // nanoseconds per pass. It changes the ENCODING, never the arithmetic:
    // dispatch order inside one command encoder is fully ordered across pass
    // boundaries, which is the same argument the shipped substep already relies
    // on to split itself around `clearBuffer`. That is not taken on trust —
    // test/perf/profile.html asserts the chain is byte-identical with profile
    // on and off before it reports a single nanosecond.
    profile = false,
    baseUrl = new URL('./', import.meta.url),
  } = opts;

  // ---- device ---------------------------------------------------------------
  let device = injectedDevice, adapter = null, adapterInfo = {};
  if (!device) {
    assert(typeof navigator !== 'undefined' && navigator.gpu, 'navigator.gpu missing (WebGPU unavailable)');
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    assert(adapter, 'no WebGPU adapter');
    const wantFeatures = [];
    if (profile) {
      assert(adapter.features.has('timestamp-query'),
        'profile: true needs the timestamp-query feature and this adapter does not expose it');
      wantFeatures.push('timestamp-query');
    }
    device = await adapter.requestDevice({ label: 'aether', requiredFeatures: wantFeatures });
    const i = adapter.info || {};
    adapterInfo = { vendor: i.vendor, architecture: i.architecture, device: i.device, description: i.description };
  } else if (device.adapterInfo) {
    const i = device.adapterInfo;
    adapterInfo = { vendor: i.vendor, architecture: i.architecture, device: i.device, description: i.description };
  }

  const lim = device.limits;
  assert(lim.maxComputeInvocationsPerWorkgroup >= 256,
    `maxComputeInvocationsPerWorkgroup = ${lim.maxComputeInvocationsPerWorkgroup} < 256`);
  assert(lim.maxStorageBuffersPerShaderStage >= 8,
    `maxStorageBuffersPerShaderStage = ${lim.maxStorageBuffersPerShaderStage} < 8`);

  const uncaptured = [];
  if (device.addEventListener) {
    device.addEventListener('uncapturederror', (e) => {
      uncaptured.push(String(e.error?.message ?? e.error));
    });
  }

  // ---- shader assembly ------------------------------------------------------
  // fixed.wgsl is concatenated AHEAD of sim.wgsl (WGSL has no #include).
  let fixedSrc = fixedSource ?? await fetchText(new URL('fixed.wgsl', baseUrl), 'implementer E');
  let simSrc   = simSource   ?? await fetchText(new URL('sim.wgsl', baseUrl), 'implementer F');

  const warnings = [];

  // fixed.wgsl's rsqrt seed table is injected by the host so the shader and the
  // BigInt reference share one source of truth. Skipped when the marker is gone.
  if (fixedSrc.includes('//@RSQRT_SEED@')) {
    const ref = await import(new URL('fixed_ref.mjs', baseUrl).href);
    fixedSrc = fixedSrc.replace('//@RSQRT_SEED@',
      `var<private> RSQRT_SEED: array<i32, 64> = array<i32, 64>(${[...ref.RSQRT_SEED].join(', ')});`);
  }

  // Legacy-binding guard: §2.6 requires fixed.wgsl to drop the `lut` binding at
  // 8 (that number is `bucketIds` in the v1 map) and move fp_dbg to 15.
  if (/@binding\s*\(\s*8\s*\)\s*var\s*<\s*storage[^>]*>\s*lut\b/.test(fixedSrc)) {
    throw new Error(
      '[aether] src/fixed.wgsl still declares `lut` at @binding(8). Spec §2.6 requires that ' +
      'declaration (and lut_lerp / LUT_N / LUT_SHIFT) to be deleted — binding 8 is `bucketIds` ' +
      'in the v1 map. This is implementer E\'s migration.');
  }
  if (/@binding\s*\(\s*9\s*\)\s*var\s*<\s*storage[^>]*>\s*fp_dbg\b/.test(fixedSrc)) {
    throw new Error(
      '[aether] src/fixed.wgsl declares fp_dbg at @binding(9). Spec §2.6 moves it to @binding(15). ' +
      'This is implementer E\'s migration.');
  }

  // Debug strip (§2.5). The block also contains fn fp_flag, which the rest of
  // fixed.wgsl calls unconditionally, so the release build substitutes a no-op
  // rather than deleting the text outright.
  const DBG_RE = /\/\/@DBG_BEGIN[\s\S]*?\/\/@DBG_END/g;
  if (!debug) {
    const stub = 'const FP_DEBUG : bool = false;\nfn fp_flag(slot : u32) { let _unused = slot; }';
    let replaced = 0;
    fixedSrc = fixedSrc.replace(DBG_RE, () => { replaced++; return replaced === 1 ? stub : ''; });
    simSrc = simSrc.replace(DBG_RE, '');
    if (replaced === 0) warnings.push('debug:false requested but no //@DBG_BEGIN block found in fixed.wgsl');
  }

  // Workgroup-size override (negative control only; never used by a real run).
  if (workgroupOverride) {
    simSrc = simSrc.replace(
      /@workgroup_size\s*\(([^)]*)\)([\s\S]{0,200}?)\bfn\s+([A-Za-z_]\w*)/g,
      (whole, sz, mid, fnName) =>
        WG_VARIABLE.has(fnName)
          ? `@workgroup_size(${workgroupOverride})${mid}fn ${fnName}`
          : whole);
  }

  const source = fixedSrc + '\n// ---- sim.wgsl ----\n' + simSrc;

  // ---- reflect + validate ---------------------------------------------------
  const refl = reflectWgsl(source);
  const missing = PASSES.map((p) => p.fn).filter((fn) => !refl.entries.has(fn));
  assert(missing.length === 0,
    `sim.wgsl is missing required @compute entry points: ${missing.join(', ')}. ` +
    `Found: ${[...refl.entries.keys()].join(', ') || '(none)'}`);

  for (const [name, e] of refl.entries) {
    const nStorage = e.resources.filter((r) => r.space === 'storage').length;
    assert(nStorage <= 8,
      `pass "${name}" needs ${nStorage} storage buffers (device limit 8): ` +
      e.resources.map((r) => `${r.binding}:${r.name}`).join(', '));
    for (const r of e.resources) {
      assert(r.group === 0, `pass "${name}" uses @group(${r.group}); only group 0 is allowed`);
      assert(BINDING_NAMES[r.binding] !== undefined,
        `pass "${name}" uses unmapped @binding(${r.binding}) (${r.name})`);
    }
  }
  if (!debug) {
    for (const [name, e] of refl.entries) {
      assert(!e.resources.some((r) => r.binding === 15),
        `debug stripped but pass "${name}" still references binding 15`);
    }
  }

  const simModule = await makeModule(device, source, 'aether-sim');

  // ---- hash module ----------------------------------------------------------
  // The hash kernels live in their own pipeline layout so they never disturb
  // the sim's bind groups; `state` is aliased onto the reserved binding 14.
  let hashSrc = await fetchText(new URL('hash/statehash.wgsl', baseUrl));
  hashSrc = hashSrc.replace(/@group\(0\)\s*@binding\(0\)(\s*var<storage,\s*read>\s+state)/,
    '@group(0) @binding(14)$1');
  assert(/@binding\(14\)\s*var<storage,\s*read>\s+state/.test(hashSrc),
    'could not alias statehash `state` onto binding 14');
  const hashModule = await makeModule(device, hashSrc, 'aether-statehash');

  // ---- tables ---------------------------------------------------------------
  let lutData = tables;
  let loadedTableHash = null;
  let declaredTableHash = null;
  if (!lutData) {
    let res = null;
    try { res = await fetch(new URL('tables.json', baseUrl), { cache: 'no-store' }); } catch { /* absent */ }
    if (res && res.ok) {
      const json = await res.json();
      declaredTableHash = json.tableHash ?? null;
      const b64 = json.b64 ?? json.data ?? json.base64;
      assert(typeof b64 === 'string' && b64.length > 0,
        'src/tables.json has no base64 payload (expected key "b64")');
      const bin = atob(b64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      lutData = new Int32Array(bytes.buffer, 0, bytes.byteLength >> 2);
      if (json.specVersion !== undefined) {
        assert(json.specVersion === SPEC_VERSION,
          `tables.json specVersion ${json.specVersion} != engine SPEC_VERSION ${SPEC_VERSION}`);
      }
      if (json.tn !== undefined) assert(json.tn === TN, `tables.json tn ${json.tn} != ${TN}`);
    } else {
      throw new Error(
        '[aether] MISSING TABLES: src/tables.json (owned by implementer F). The kernel LUTs are ' +
        'frozen bytes — the engine will not generate them at run time (spec §3.4).');
    }
  }
  assert(lutData.length === LUT_WORDS,
    `tables.json has ${lutData.length} i32 entries, expected ${LUT_WORDS} (5 x ${TN})`);
  loadedTableHash = digestState(lutData, { particleCount: lutData.length, strideWords: 1, hashedWords: 1 }).hex;
  if (typeof tableHash === 'string' && /^[0-9a-f]{64}$/.test(tableHash)) {
    assert(loadedTableHash === tableHash,
      `TABLE_HASH mismatch: tables.json hashes to ${loadedTableHash}, expected ${tableHash}`);
  } else {
    warnings.push(`engine TABLE_HASH is not frozen yet; loaded tables hash to ${loadedTableHash}`);
  }
  if (declaredTableHash && declaredTableHash !== loadedTableHash) {
    // Not fatal, but it MUST be reconciled before a glyph is ever encoded: two
    // different definitions of "the table hash" is exactly the drift §3.4 is
    // meant to prevent. engine.js uses the 4-lane Murmur3 digest of the bytes.
    warnings.push(
      `tables.json declares tableHash ${declaredTableHash} but the 4-lane Murmur3 digest of its ` +
      `bytes is ${loadedTableHash}. Two hash definitions are in play — reconcile before v1 ships.`);
  }

  // ---- buffers --------------------------------------------------------------
  const U = GPUBufferUsage;
  const SCD = U.STORAGE | U.COPY_DST | U.COPY_SRC;
  const mk = (label, size, usage) => device.createBuffer({ label, size, usage });

  const NCAP = maxParticles;
  const CCAP = maxCells;
  const BCAP = Math.max(ceilDiv(CCAP, 1024), 1024);

  const buf = {
    // 15 u32 = 60 B of scalars, then §23.B's two align(16) jet arrays at 64 and
    // 192: 8 jets x (cx, cy, r, _) and 8 x (fx, fy, _, _) = 320 B, then §24's
    // drive vector + 2 pad words at 320 = 336 B. A struct in
    // the UNIFORM address space has its size rounded up to a multiple of 16 and
    // an array member must START on one, which is why there is an explicit
    // `_padJet` word rather than a silent hole. Getting this wrong is a
    // validation error that arrives ASYNCHRONOUSLY and reads back as stable
    // zeros (PLATFORM_NOTES trap #2), i.e. as a sim that quietly stops running.
    // It was 64 through v12 and 48 through v8. THIS IS A UNIFORM, NOT A STORAGE
    // BUFFER: the 8-storage-buffers-per-stage cap that `friction` and
    // `bondForm1` sit at is untouched by §23.B.
    params:    mk('params',    PARAMS_BYTES,               U.UNIFORM | U.COPY_DST),
    stateA:    mk('stateA',    NCAP * 32,                  SCD),
    stateB:    mk('stateB',    NCAP * 32,                  SCD),
    derived:   mk('derived',   NCAP * DERIVED_BYTES,        SCD),
    cellCount: mk('cellCount', CCAP * 4,                   SCD),
    cellStart: mk('cellStart', (CCAP + 1) * 4,             SCD),
    blockSums: mk('blockSums', (BCAP + 1) * 4,             SCD),
    cellOf:    mk('cellOf',    NCAP * 4,                   SCD),
    bucketIds: mk('bucketIds', NCAP * 4,                   SCD),
    sortedIds: mk('sortedIds', NCAP * 4,                   SCD),
    nbr:       mk('nbr',       NCAP * MAXNBR * 4,          SCD),
    nbrN:      mk('nbrN',      NCAP * 4,                   SCD),
    luts:      mk('luts',      LUT_WORDS * 4,              U.STORAGE | U.COPY_DST),
    // Always MAT_MAX rows, whatever the live material count. The tail is zeroed
    // by writeBuffer below, so an out-of-range matId can only read zeros — and
    // loadScene rejects one anyway. Sizing this to `mats.length` would make the
    // buffer's length part of the scene, which is exactly the kind of implicit
    // coupling §3.2 exists to prevent.
    mats:      mk('mats',      MAT_MAX * MAT_WORDS * 4,    U.STORAGE | U.COPY_DST),
    // §18. fp_dbg widened 8 -> 16 words. Slots 0..7 keep their exact meaning AND
    // their exact position (assertSpecValidRegime names them positionally);
    // §18 takes slot 8 and 9..15 are reserved.
    dbg:       mk('dbg',       64,                         SCD),
    hashAcc:   mk('hashAcc',   32,                         SCD),
    hashChain: mk('hashChain', 32,                         SCD),
    // §18. 16 -> 32 bytes: HashParams gained `n_fluid` (was `_pad`) and
    // `bond_cap`. n_fluid is the bond fold's guard against rows nothing writes.
    hashParams: mk('hashParams', 32,                       U.UNIFORM | U.COPY_DST),
    // §18 bonds. COPY_DST is MANDATORY on both — clearBuffer requires it, and a
    // missing usage flag invalidates the ENTIRE command buffer and produces fake
    // stable zeros (PLATFORM_NOTES trap #3). `w == 0 => EMPTY` makes an all-zero
    // buffer a valid empty table, so there is no initialisation pass.
    // +3.84 MB at n = 10 000; +7.68 MB at the engine default 20 000.
    bond:      mk('bond',      NCAP * BOND_CAP * 4,        SCD),
    bondCand:  mk('bondCand',  NCAP * BOND_CAP * 4,        SCD),
    // §20 / Rung 4. ONE buffer, four regions: the Q22 quarter-turn cosine table (baked
    // by the host here, exactly as the kernel LUTs are baked offline), 1024 body
    // records of 32 words, 1024 constraint-driver sidecars of 10 words, and two
    // EXACT i32 rest-offset words per particle slot. The sidecar leaves the
    // body record/native snapshot wire shape untouched.
    //
    // No atomics: every §20 pass is one workgroup per body and reduces in
    // workgroup memory, so no two invocations write the same word. COPY_DST is
    // MANDATORY (clearBuffer) — a missing usage flag invalidates the ENTIRE
    // command buffer and reads back as stable zeros (PLATFORM_NOTES trap #3).
    rigid:     mk('rigid',     (RG_REST_BASE + NCAP * 2) * 4, SCD),
    // Rung 4 upload ring. Each submitBatch slot holds the complete address
    // space for driver records, but only active 40-byte records are written and
    // copied. Distinct source offsets let several substeps share one command
    // encoder without later queue writes collapsing earlier scene times.
    driverUpload: mk('driverUpload',
      Math.max(1, submitBatch | 0) * RG_MAX_BODIES * RG_DRIVER_WORDS * 4,
      U.COPY_SRC | U.COPY_DST),
    // §21 PER-PARTICLE COLOUR. ONE u32 of RGBA8 per SLOT, and it is NOT
    // SIMULATION STATE.
    //
    // THE WHOLE SAFETY ARGUMENT IS THAT THIS BUFFER IS UNREACHABLE FROM THE SIM.
    // No `sim.wgsl` pipeline binds it, `HASHED_WORDS` cannot see it (the digest
    // folds words of `stateA/B`, and this is a different allocation), and no
    // solver term reads it. So colour can never change a trajectory, can never
    // change a chain, and never has to be re-proven when it changes. The
    // determinism gate is blind to it BY CONSTRUCTION rather than by discipline.
    //
    // The engine owns it anyway — not the renderer — because it is indexed by
    // SLOT, and `spawnFluid` and `eraseWhere` are the only two things in this
    // repo that renumber a slot. A renderer-owned array would go stale on the
    // first erase, silently, and paint the wrong particles.
    //
    // Byte order is R,G,B,A little-endian (`r | g<<8 | b<<16 | a<<24`), and
    // **a == 0 means "no colour, use the material look"**. An all-zero buffer is
    // therefore exactly the pre-§21 picture, which is what makes the zero
    // initialisation free and the feature strictly additive.
    // 4 B per slot: 40 KB at n = 10 000, against `nbr`'s 3.8 MB.
    tint:      mk('tint',      NCAP * 4,                   SCD),
  };

  // §20. THE COSINE TABLE IS BAKED ONCE, HERE, AND IS THE SAME BYTES FOREVER.
  // Q16 is NOT enough and that cost a measurement pass to see: at 1024 steps per
  // quarter turn the linear-interpolation error is h^2/8 = 2.94e-7 but a Q16
  // entry is quantised at 1.53e-5 — FIFTY TIMES LARGER, so the table rounding
  // was the entire error. Q22 puts the entry quantisation (2.4e-7) just under
  // the interpolation bound, which is where a table belongs.
  device.queue.writeBuffer(buf.rigid, RG_COS_BASE * 4, buildCosTable());

  device.queue.writeBuffer(buf.luts, 0, lutData);

  // ---- mats validation -------------------------------------------------------
  // §6.2 documents `mats` with NAMED FIELDS, but the GPU struct is packed by
  // POSITION, so `mats[m][w]` on an array of {eps, dpMax, ...} objects yields
  // undefined -> 0 for every field and the engine builds an all-zero material
  // table. That disables PBF entirely (dpMax = 0 kills every position
  // correction) and drives fp_divshift into divide-by-zero millions of times,
  // and it used to do all of that SILENTLY — the only trace was dbg[1]/dbg[2].
  // Verified failure mode: three runs with deliberately different `xsph` values
  // produced byte-identical trajectories and dbg = [0, 2761920, 7534452, 0…].
  assert(Array.isArray(mats) && mats.length >= 1 && mats.length <= MAT_MAX,
    `mats must be an array of 1..${MAT_MAX} materials; got ${
      Array.isArray(mats) ? mats.length : typeof mats}`);
  const MAT_FIELD_NAMES = MAT_FIELDS;
  const matCount = mats.length;
  const F = (name) => MAT_FIELD_NAMES.indexOf(name);
  for (let m = 0; m < matCount; m++) {
    const row = mats[m];
    assert(Array.isArray(row) && row.length === MAT_WORDS,
      `mats[${m}] must be an ARRAY of exactly ${MAT_WORDS} integers in the order ` +
      `[${MAT_FIELD_NAMES.join(', ')}] — a named-field object silently becomes all zeros. ` +
      `Build rows with mat({...}); got ${Array.isArray(row) ? `length ${row.length}` : typeof row}`);
    for (let w = 0; w < MAT_WORDS; w++) {
      assert(Number.isInteger(row[w]),
        `mats[${m}].${MAT_FIELD_NAMES[w]} = ${row[w]} is not an integer`);
      assert(row[w] >= -2147483648 && row[w] <= 2147483647,
        `mats[${m}].${MAT_FIELD_NAMES[w]} = ${row[w]} does not fit in i32`);
    }
    const nm = MAT_NAMES[m] ?? `mat${m}`;
    // eps is the ONLY thing keeping solveA's fp_divshift denominator above the
    // §2.3 caller contract of den >= 2^15 when S collapses to 0 (a particle with
    // 1-3 neighbours). Without it the divide is undefined, not merely inaccurate.
    assert(row[F('eps')] >= 32768,
      `mats[${m}] ${nm}.eps = ${row[F('eps')]} < 32768; §2.3 requires the fp_divshift denominator ` +
      `>= 2^15 and eps is the only term that guarantees it when S = 0`);
    assert(row[F('dpMax')] > 0, `mats[${m}] ${nm}.dpMax <= 0 disables every position correction`);
    // §17. `phantom` is a flag, not a magnitude: it is read as `!= 0` by the
    // shader, so anything other than 0 or 1 is somebody mistaking word 3 for the
    // `vort` coefficient it used to be reserved for.
    assert(row[F('phantom')] === 0 || row[F('phantom')] === 1,
      `mats[${m}] ${nm}.phantom = ${row[F('phantom')]} must be 0 or 1; §17 word 3 is a flag ` +
      `(it was the reserved 'vort' slot through v6, so an old row reads as phantom = 0)`);
    // A PHANTOM THAT CARRIES ANY PHYSICS IS A GHOST, and this is the throw that
    // stops it. A retired particle is skipped by every pair loop, so a force on
    // it would be a force nothing balances and a transition on it would be a
    // corpse climbing out of the grave. All of it must be zero, and stating
    // WHICH field is wrong matters more here than anywhere else in this loop:
    // the failure mode is invisible (an inert particle that quietly drifts).
    if (row[F('phantom')] !== 0) {
      for (const zf of ['grav', 'gammaCoh', 'gammaCur', 'adhesion', 'xsph',
                        'cond', 'coolRate', 'fric', 'corrode', 'fuel0', 'burnRate', 'burnHeat',
                        'bondK']) {
        assert(row[F(zf)] === 0,
          `mats[${m}] ${nm}.phantom = 1 but ${zf} = ${row[F(zf)]}; retired matter must carry no ` +
          `physics at all (§17). A phantom is skipped by every pair loop, so any force on it is ` +
          `one nothing balances.`);
      }
      for (const [tf, sentinel] of [['meltPt', NEVER_HOT], ['boilPt', NEVER_HOT], ['ignitePt', NEVER_HOT],
                                    ['freezePt', NEVER_COLD], ['condPt', NEVER_COLD], ['corrPt', NEVER_HOT]]) {
        assert(row[F(tf)] === sentinel,
          `mats[${m}] ${nm}.phantom = 1 but ${tf} = ${row[F(tf)]} is reachable; retired matter must ` +
          `have every transition disabled by its NEVER sentinel (§17) — a corpse cannot come back`);
      }
      // §18.7 R3. bondGroup's inert value is -1, so it cannot live in the
      // `=== 0` loop above and needs its own line. THIS ASSERTION IS THE ONLY
      // THING THAT MAKES THE RETIRED-PARTNER RULE TRUE: the solve-time veto and
      // the formation reap are BOTH consequences of `bondK == 0` on this row.
      assert(row[F('bondGroup')] < 0,
        `mats[${m}] ${nm}.phantom = 1 but bondGroup = ${row[F('bondGroup')]}; retired matter cannot ` +
        `join a bond group (§18.7 R3).`);
    }
    assert(row[F('heatCap')] === 0, `mats[${m}] ${nm}.heatCap != 0; §12 reserves it (v4 is uniform capacity)`);
    // §16. Above 1.0 the tangential correction over-shoots the relative
    // displacement it is cancelling and the pair oscillates instead of sticking.
    assert(row[F('fric')] >= 0 && row[F('fric')] <= FRIC_MAX,
      `mats[${m}] ${nm}.fric = ${row[F('fric')]} outside [0, FRIC_MAX = ${FRIC_MAX}]; above 1.0 the ` +
      `tangential correction exceeds the relative displacement it cancels and the contact rings`);
    // §12 range contracts. Each one is load-bearing, not decorative:
    //   xsph  — the XSPH blend weight can exceed 1 above this and over-damp.
    //   cond  — explicit-diffusion stability at 2x rest density (see MATS).
    //   coolRate — > 1.0 overshoots AMBIENT and oscillates every substep.
    assert(row[F('xsph')] >= 0 && row[F('xsph')] <= XSPH_MAX,
      `mats[${m}] ${nm}.xsph = ${row[F('xsph')]} outside [0, ${XSPH_MAX}]`);
    assert(row[F('cond')] >= 0 && row[F('cond')] <= COND_MAX,
      `mats[${m}] ${nm}.cond = ${row[F('cond')]} outside [0, COND_MAX = ${COND_MAX}]; above that the ` +
      `explicit conduction update can exceed 1.0 per substep under compression and ring`);
    assert(row[F('coolRate')] >= 0 && row[F('coolRate')] <= 65536,
      `mats[${m}] ${nm}.coolRate = ${row[F('coolRate')]} outside [0, 65536]`);
    assert(row[F('fuel0')] >= 0, `mats[${m}] ${nm}.fuel0 < 0`);
    assert(row[F('burnRate')] >= 0, `mats[${m}] ${nm}.burnRate < 0`);
    assert(row[F('splitPct')] >= 0 && row[F('splitPct')] <= 256,
      `mats[${m}] ${nm}.splitPct = ${row[F('splitPct')]} outside [0, 256]`);
    // Every product id must name a material that EXISTS. An out-of-range id
    // indexes past the live rows into the zeroed tail, which is a material with
    // eps = 0 and dpMax = 0 — i.e. a silent divide-by-zero factory.
    for (const pf of ['meltTo', 'freezeTo', 'boilTo', 'condTo', 'burnTo', 'splitTo', 'corrTo']) {
      const t = row[F(pf)];
      assert(Number.isInteger(t) && t >= 0 && t < matCount,
        `mats[${m}] ${nm}.${pf} = ${t} is not a live material id (0..${matCount - 1})`);
    }
    for (const tf of ['meltPt', 'freezePt', 'boilPt', 'condPt', 'ignitePt', 'spawnTemp']) {
      const t = row[F(tf)];
      const sentinel = (t === NEVER_HOT || t === NEVER_COLD);
      assert(sentinel || (t >= TEMP_MIN && t <= TEMP_MAX),
        `mats[${m}] ${nm}.${tf} = ${t} outside [TEMP_MIN, TEMP_MAX] and is not a NEVER sentinel`);
    }
    // A material with fuel but no burn rate burns forever; that is legal (an
    // eternal flame) but it must be deliberate, so it is only allowed when
    // burnHeat is also 0 — i.e. the material is not actually a fire.
    assert(row[F('fuel0')] === 0 || row[F('burnRate')] > 0 || row[F('burnHeat')] === 0,
      `mats[${m}] ${nm} has fuel0 > 0, burnRate = 0 and burnHeat > 0: it would burn and heat forever`);

    // ---- §13 corrosion contracts -------------------------------------------
    // corrode's ceiling is what proves the dose accumulator cannot overflow:
    // MAXNBR * CORR_TERM = 48 * 2^24 = 8.05e8, 2.7x under 2^31 (see sim.wgsl).
    assert(row[F('corrode')] >= 0 && row[F('corrode')] <= CORR_MAX,
      `mats[${m}] ${nm}.corrode = ${row[F('corrode')]} outside [0, CORR_MAX = ${CORR_MAX}]`);
    const cpt = row[F('corrPt')];
    assert(cpt === NEVER_HOT || cpt > 0,
      `mats[${m}] ${nm}.corrPt = ${cpt} must be > 0 or the NEVER_HOT sentinel; ` +
      `a corrPt <= 0 dissolves the material on substep 1 with no acid anywhere, ` +
      `because the dose accumulator's floor is exactly 0`);

    // ---- §18 bond contracts -------------------------------------------------
    const bk = row[F('bondK')], bg = row[F('bondGroup')];
    assert(bk >= 0 && bk <= BOND_K_MAX,
      `mats[${m}] ${nm}.bondK = ${bk} outside [0, BOND_K_MAX = ${BOND_K_MAX}]; at 1.0 the dpMax ` +
      `clamp binds on impact and at 2.0 the Jacobi bond solve over-relaxes (rg 5.70 -> 18.29)`);
    assert(Number.isInteger(bg) && bg >= -1 && bg <= 63,
      `mats[${m}] ${nm}.bondGroup = ${bg} outside [-1, 63]`);
    // bondK != 0 with bondGroup < 0 is a row that would enter the gather (the
    // gather tests the TABLE value) and then veto every partner it ever meets.
    assert(bk === 0 || bg >= 0,
      `mats[${m}] ${nm}.bondK = ${bk} but bondGroup = ${bg}: a bonding row must name a group`);
    assert(bk !== 0 || bg < 0,
      `mats[${m}] ${nm}.bondGroup = ${bg} but bondK = 0: a non-bonding row must not name a group`);
    assert(row[F('bondReform')] === 0 || row[F('bondReform')] === 1,
      `mats[${m}] ${nm}.bondReform = ${row[F('bondReform')]} must be 0 or 1`);
    assert(row[F('bondBreak')] >= 0, `mats[${m}] ${nm}.bondBreak < 0`);
    assert(row[F('bondMu')] >= 0, `mats[${m}] ${nm}.bondMu < 0`);
    assert(row[F('bondFormV')] >= 0, `mats[${m}] ${nm}.bondFormV < 0`);
    assert(row[F('bondFormP')] >= 0, `mats[${m}] ${nm}.bondFormP < 0`);
    const rs = row[F('rampSign')];
    assert(rs === -1 || rs === 0 || rs === 1,
      `mats[${m}] ${nm}.rampSign = ${rs} must be -1, 0 or +1 (0 makes the branch bit-for-bit inert)`);
    // >= 1 degree of ramp. A narrower one would bake a rampInv large enough to
    // saturate the Q16 multiply, and a zero-width one would divide by zero.
    assert(row[F('rampHi')] - row[F('rampLo')] >= 65536,
      `mats[${m}] ${nm} ramp is ${row[F('rampHi')] - row[F('rampLo')]} ticks wide; §18 requires ` +
      `>= 65536 (1 degree) so the baked rampInv cannot saturate`);
    // ---- §29 containment (v17; words 46/47, reserved through v16) ----------
    const sealClass = row[F('bondK')] !== 0 && row[F('bondReform')] === 0 &&
                      row[F('solidMode')] !== SOLID_NONE && row[F('phantom')] === 0;
    assert(row[F('burstP')] >= 0, `mats[${m}] ${nm}.burstP < 0`);
    assert(row[F('sealMode')] === 0 || row[F('sealMode')] === 1,
      `mats[${m}] ${nm}.sealMode = ${row[F('sealMode')]} must be 0 or 1`);
    if (!sealClass) {
      assert(row[F('burstP')] === 0 && row[F('sealMode')] === 0,
        `mats[${m}] ${nm} is not seal-class but carries burstP/sealMode`);
    }
    // ---- §25 the thermal drive (v16; words 44/45, ex-rsv5/rsv6) -------------
    // Same range contract as coolRate and for the same reason: above 1.0 the
    // relaxation overshoots its target and oscillates every substep.
    assert(row[F('tempDrive')] >= 0 && row[F('tempDrive')] <= 65536,
      `mats[${m}] ${nm}.tempDrive = ${row[F('tempDrive')]} outside [0, 65536]`);
    if (row[F('tempDrive')] !== 0) {
      const tt = row[F('tempTarget')];
      assert(tt >= TEMP_MIN && tt <= TEMP_MAX,
        `mats[${m}] ${nm}.tempTarget = ${tt} outside [TEMP_MIN, TEMP_MAX] with tempDrive != 0`);
      // A DRIVEN row must not sit past its OWN transition thresholds: a heater
      // whose target is above its own meltPt melts itself on the substep it is
      // painted and the drive row stops existing — a self-destroying device.
      // (The clamp-then-drive order in `thermal` also relies on the target
      // being inside [TEMP_MIN, TEMP_MAX], asserted above.)
      assert(tt < row[F('meltPt')] && tt < row[F('boilPt')],
        `mats[${m}] ${nm}.tempTarget = ${tt} >= its own meltPt/boilPt; a heater must not sit ` +
        `above its own transition thresholds (§25)`);
      assert(tt > row[F('freezePt')] && tt > row[F('condPt')],
        `mats[${m}] ${nm}.tempTarget = ${tt} <= its own freezePt/condPt; a cooler must not sit ` +
        `below its own transition thresholds (§25)`);
    } else {
      assert(row[F('tempTarget')] === 0,
        `mats[${m}] ${nm}.tempTarget = ${row[F('tempTarget')]} with tempDrive = 0; an undriven ` +
        `row must keep word 44 at zero so every v15 row reads back byte-identically`);
    }
    // ---- §20 the solid axis --------------------------------------------------
    const sm = row[F('solidMode')];
    assert(sm === SOLID_NONE || sm === SOLID_BOND || sm === SOLID_RIGID,
      `mats[${m}] ${nm}.solidMode = ${sm} must be 0 (NONE), 1 (BOND) or 2 (RIGID)`);
    // THERE IS DELIBERATELY NO "SOLID_BOND IMPLIES bondK != 0" ASSERTION HERE,
    // and the reason is worth recording because the assertion was written first
    // and had to come out. `test/bonds.html` B0 — the gate that proves the whole
    // §18 mechanism is bit-for-bit inert, and therefore that the v4 chain
    // survives — runs a table with EVERY bond coefficient zeroed. A validator
    // that refused that table would make the control unconstructible, and losing
    // a control to tighten a nicety is a bad trade: B0 is the assertion that
    // keeps the golden honest across four version bumps.
    //
    // The invariant that DOES matter is enforced at runtime rather than at
    // authoring time, and it is enforced by construction: a particle is under §18
    // or under §20, never both, and FLAG_RIGID is what decides which. A member
    // gathers no bonds and `bondForm1` empties its candidate row, which also
    // drops the other half of any cross-body bond through bondForm2's mutual
    // test. The two mechanisms coexist in the ENGINE on purpose — a STONE block
    // that was never promoted to a body still bonds exactly as it did in v9,
    // which is what keeps every §18 measurement valid.
  }
  // ---- §18, the invariant that is not per-row -------------------------------
  // `min` mixing is provably free for the five ROW-valued bond coefficients ONLY
  // while every bonded material has a UNIQUE bondGroup — that is what makes both
  // ends of every bond the same material. The moment two materials share a group
  // (the obvious way to make sand stick to stone) that proof evaporates and
  // every measured number in SPEC §18 is invalid. This assertion is the guard.
  {
    const seen = new Map();
    for (let m = 0; m < matCount; m++) {
      const g = mats[m][F('bondGroup')];
      if (g < 0) continue;
      assert(!seen.has(g),
        `mats[${m}] ${MAT_NAMES[m] ?? m} shares bondGroup ${g} with ${MAT_NAMES[seen.get(g)] ?? seen.get(g)}. ` +
        `§18.11's proof that min() mixing is free for the ROW coefficients rests on every bonded ` +
        `material having a UNIQUE group; sharing one invalidates every number in SPEC §18.`);
      seen.set(g, m);
    }
  }
  // §18.24. The shader's fuse_row() excludes PHOTO by the literal row id 15
  // (`MAT_PHOTO_ROW` in sim.wgsl) — the Mat words carry no identity, so the
  // constant is the only bridge, and this is the assert that keeps it true if
  // the roster ever reorders.
  assert(MAT_PHOTO === 15,
    `MAT_PHOTO moved to ${MAT_PHOTO}; sim.wgsl's MAT_PHOTO_ROW is hard-coded 15 and must move with it`);
  // §18.5. The partner field is 17 bits, so a slot index above this would be
  // SILENTLY TRUNCATED into a bond naming the wrong particle. The assert is what
  // turns that into a load-time error.
  assert(maxParticles <= MAX_PARTICLES_BONDED,
    `maxParticles = ${maxParticles} exceeds MAX_PARTICLES_BONDED = ${MAX_PARTICLES_BONDED}; §18's ` +
    `packed bond word carries the partner slot in 17 bits and a larger index would be truncated ` +
    `into a bond naming a different particle`);
  // ---- §13, the one corrosion invariant that is not per-row -----------------
  // An attacker with a REACHABLE corrPt is legal and is how acid gets used up:
  // the SPEND term doses it in proportion to how much dissolvable matter it
  // touches. What is NOT legal is TWO attackers where either is dissolvable —
  // then both the ATTACK and the SPEND term fire on the same pair from both
  // sides and the pair eats itself at double rate. Vacuous while the roster has
  // one acid; it is here so the second one cannot land silently.
  {
    const atk = [];
    for (let m = 0; m < matCount; m++) if (mats[m][F('corrode')] > 0) atk.push(m);
    if (atk.length > 1) {
      for (const m of atk) {
        assert(mats[m][F('corrPt')] === NEVER_HOT,
          `mats[${m}] ${MAT_NAMES[m] ?? `mat${m}`} is one of ${atk.length} corrosive materials ` +
          `(${atk.map((k) => MAT_NAMES[k] ?? k).join(', ')}) and has a reachable corrPt: with more ` +
          `than one attacker in the table every attacker must be immune (§13)`);
      }
    }
  }
  assert(Number.isInteger(iters) && iters >= 1 && iters <= 16,
    `iters = ${iters} must be an integer in 1..16 (shipped default ${ITERS})`);

  // MAT_MAX rows are written, not matCount: the tail must be explicitly zeroed
  // rather than left as whatever the allocator handed us (PLATFORM_NOTES trap #7).
  const matWords = new Int32Array(MAT_MAX * MAT_WORDS);
  for (let m = 0; m < matCount; m++) {
    for (let w = 0; w < MAT_WORDS; w++) matWords[m * MAT_WORDS + w] = mats[m][w];
  }
  // §18. `rampInv` is BAKED HERE and never authored, never computed at runtime.
  // bond_k(M, T) is one multiply and one shift precisely because this divide
  // happens once on the host: floor(2^32 / (rampHi - rampLo)) in Q16, so
  // dT * rampInv >> 16 is (T - lo)/(hi - lo) in Q16. Writing it into matWords
  // rather than into `mats` keeps the authored table a pure input — two engines
  // built from the same MATS array stay byte-identical.
  {
    const iLo = F('rampLo'), iHi = F('rampHi'), iInv = F('rampInv');
    for (let m = 0; m < matCount; m++) {
      const span = mats[m][iHi] - mats[m][iLo];
      matWords[m * MAT_WORDS + iInv] = span > 0 ? Math.floor(2 ** 32 / span) : 0;
    }
  }
  device.queue.writeBuffer(buf.mats, 0, matWords);

  const bufForBinding = (b, parity) => {
    switch (b) {
      case 0:  return buf.params;
      case 1:  return parity === 0 ? buf.stateA : buf.stateB;
      case 2:  return parity === 0 ? buf.stateB : buf.stateA;
      case 3:  return buf.derived;
      case 4:  return buf.cellCount;
      case 5:  return buf.cellStart;
      case 6:  return buf.blockSums;
      case 7:  return buf.cellOf;
      case 8:  return buf.bucketIds;
      case 9:  return buf.sortedIds;
      case 10: return buf.nbr;
      case 11: return buf.nbrN;
      case 12: return buf.luts;
      case 13: return buf.mats;
      case 15: return buf.dbg;
      case 16: return buf.bond;
      case 17: return buf.bondCand;
      case 18: return buf.rigid;
      default: throw new Error(`[aether] no buffer for binding ${b}`);
    }
  };

  // ---- pipelines + explicit layouts ----------------------------------------
  const pipelines = new Map();   // fn -> {pipeline, groups:[bg0,bg1], resources}
  device.pushErrorScope('validation');
  // §18.9 / §18.19 risk 1. `friction` (now friction + bond gather) and
  // `bondForm1` are the FIRST TWO PIPELINES IN THIS REPO EVER TO SIT AT 8
  // STORAGE BUFFERS — the docs used to claim solveB_c* already did, and it does
  // not; it binds 6 (see docs/SPEC.md §3.3, corrected 2026-07-26 with the
  // measurement). So this is untested ground on any device, and PLATFORM_NOTES
  // trap #2 says a validation failure here would be ASYNCHRONOUS and would read
  // back as stable zeros — a silently dead solver rather than an error.
  // Ordering them first and naming them in the throw is what turns that into a
  // load-time failure with a cause. The pre-approved fallback if a device ever
  // rejects 8 is in SPEC §18.9: split the gather back out as `bondSolve`
  // (+1 dispatch per solver iteration) and `bondForm1` into 1a/1b.
  const AT_THE_CAP = ['friction', 'bondForm1'];
  const orderedPasses = [
    ...PASSES.filter((p) => AT_THE_CAP.includes(p.fn)),
    ...PASSES.filter((p) => !AT_THE_CAP.includes(p.fn)),
  ];
  for (const p of orderedPasses) {
    const e = refl.entries.get(p.fn);
    const bgl = device.createBindGroupLayout({
      label: `bgl:${p.fn}`,
      entries: e.resources.map((r) => ({
        binding: r.binding,
        visibility: GPUShaderStage.COMPUTE,
        ...layoutEntryFor(r),
      })),
    });
    const pl = device.createPipelineLayout({ label: `pl:${p.fn}`, bindGroupLayouts: [bgl] });
    const pipeline = device.createComputePipeline({
      label: `cp:${p.fn}`, layout: pl, compute: { module: simModule, entryPoint: p.fn },
    });
    const groups = [0, 1].map((parity) => device.createBindGroup({
      label: `bg:${p.fn}:${parity}`,
      layout: bgl,
      entries: e.resources.map((r) => ({ binding: r.binding, resource: { buffer: bufForBinding(r.binding, parity) } })),
    }));
    // Dispatch geometry is derived from the shader's DECLARED workgroup size,
    // never assumed to be 256. Assuming it silently under-dispatches (leaving a
    // tail of particles unprocessed) the moment anyone changes @workgroup_size.
    const wgx = Number.parseInt(e.wg?.[0] ?? '256', 10);
    assert(Number.isInteger(wgx) && wgx >= 1 && wgx <= 256,
      `pass "${p.fn}" declares @workgroup_size(${e.wg?.join(',')}), which is not a usable literal`);
    pipelines.set(p.fn, {
      pipeline, groups, resources: e.resources, wg: e.wg, wgx, dim: p.dim,
      // `dim: 'n'` means "one thread per particle". fluidOnly narrows that to
      // "one thread per FLUID particle" for the passes whose own first line
      // throws the rest away. See guardsOnNFluid().
      fluidOnly: fluidOnlyDispatch && p.dim === 'n' && e.fluidOnly === true,
      fluidOnlyDeclared: p.dim === 'n' && e.fluidOnly === true,
    });
    if (AT_THE_CAP.includes(p.fn)) {
      const nStorage = e.resources.filter((r) => r.space === 'storage').length;
      assert(nStorage <= (device.limits?.maxStorageBuffersPerShaderStage ?? 8),
        `[aether] pass "${p.fn}" needs ${nStorage} storage buffers but the device allows ` +
        `${device.limits?.maxStorageBuffersPerShaderStage}. SPEC §18.9 pre-specifies the split ` +
        `fallback (friction + bondSolve, bondForm1a + bondForm1b) for exactly this.`);
    }
  }
  // Resolve the §18 pipelines' validation BEFORE anything else is built, so a
  // failure names the pass instead of surfacing 200 lines later as zeros.
  {
    const bondErr = await device.popErrorScope();
    if (bondErr) {
      throw new Error('[aether] §18: creating the 8-storage-buffer pipelines (friction + bond ' +
        'gather, bondForm1) FAILED VALIDATION: ' + bondErr.message +
        '\nNo pipeline in this repo had ever run at 8 storage buffers before v8 — SPEC §18.9 ' +
        'specifies the split fallback for this device.');
    }
    device.pushErrorScope('validation');
  }

  // hash pipelines
  const hashBgl = device.createBindGroupLayout({
    label: 'bgl:hash',
    entries: [
      { binding: 14, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      // §18. The bond table is persistent solver state that determines future
      // positions; if it diverged while positions had not, the chain would be
      // blind until it showed up. 4 storage buffers in the hash group — well
      // clear of the device cap of 8.
      { binding: 4,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    ],
  });
  const hashPl = device.createPipelineLayout({ label: 'pl:hash', bindGroupLayouts: [hashBgl] });
  const hashPipes = {};
  for (const ep of ['clear_acc', 'digest', 'fold_chain']) {
    hashPipes[ep] = device.createComputePipeline({
      label: `cp:hash:${ep}`, layout: hashPl, compute: { module: hashModule, entryPoint: ep },
    });
  }
  const hashGroups = [0, 1].map((parity) => device.createBindGroup({
    label: `bg:hash:${parity}`,
    layout: hashBgl,
    entries: [
      { binding: 14, resource: { buffer: parity === 0 ? buf.stateA : buf.stateB } },
      { binding: 1,  resource: { buffer: buf.hashAcc } },
      { binding: 2,  resource: { buffer: buf.hashChain } },
      { binding: 3,  resource: { buffer: buf.hashParams } },
      { binding: 4,  resource: { buffer: buf.bond } },
    ],
  }));
  const setupErr = await device.popErrorScope();
  if (setupErr) throw new Error('[aether] pipeline setup failed validation: ' + setupErr.message);

  const engine = new Engine({
    device, adapter, adapterInfo, canvas, buf, pipelines, hashPipes, hashGroups,
    maxParticles: NCAP, maxCells: CCAP, debug, submitBatch, chain,
    uncaptured, warnings, source, refl, tableHash: loadedTableHash, declaredTableHash,
    iters, mats, matCount, profile, matWords,
  });
  return engine;
}

// ============================================================================
// Engine
// ============================================================================
class Engine {
  constructor(o) {
    this.device = o.device;
    this.adapter = o.adapter;
    this.adapterInfo = o.adapterInfo;
    this.canvas = o.canvas;
    this.buf = o.buf;
    this._pipes = o.pipelines;
    this._hashPipes = o.hashPipes;
    this._hashGroups = o.hashGroups;
    this.maxParticles = o.maxParticles;
    this.maxCells = o.maxCells;
    this.debug = o.debug;
    this.submitBatch = Math.max(1, o.submitBatch | 0);
    this.chainEnabled = !!o.chain;
    this._uncaptured = o.uncaptured;
    this.warnings = o.warnings;
    this.source = o.source;
    this.reflection = o.refl;
    this.tableHash = o.tableHash;
    this.declaredTableHash = o.declaredTableHash;
    this.iters = o.iters;
    // §25 (v16). A SHALLOW COPY, deliberately: interned rows are appended to
    // THIS ENGINE's view of the table at mint time, and mutating the caller's
    // array (usually the module-level MATS) would leak one engine's scene
    // state into every other engine on the page. Rows [0, _authoredCount) are
    // the exact authored row arrays, shared; the tail is per-engine.
    this.mats = o.mats.slice();
    this.matCount = o.matCount;
    // §25. The intern cache: key -> live rowId, rowId -> entry, and the
    // ORDERED event log that is the scene's material record (§25.2 — replays
    // carry it; `sceneRowsDigest` covers it; `matsHash` never does).
    this._authoredCount = o.matCount;
    this._internMap = new Map();
    this._internByRow = new Map();
    this._internEvents = [];
    this._internTick = 0;
    this._internSpawned = new Map();   // rowId -> particles ever spawned wearing it
    // §26.C. The BAKED material image exactly as `createEngine` uploaded it —
    // MAT_MAX rows, `rampInv` already derived, tail explicitly zeroed. Kept so
    // that `setGravityScale` can re-upload a scaled copy without duplicating
    // the bake, which is the only way the two can never disagree.
    this._matWords = o.matWords;
    // Profiling state. `_profile` gates the per-dispatch pass split; everything
    // else stays null unless profileSubstep() is actually called.
    this._profile = !!o.profile;
    this._ts = null;              // { querySet, resolve, staging, labels }
    this._tsLabels = null;

    this.n = 0; this.nFluid = 0;
    this.gridW = 0; this.gridH = 0; this.cellTotal = 0; this.cellShift = CELL_SHIFT;
    this.domW = 0; this.domH = 0;
    this.substepCount = 0;
    this.parity = 0;
    this._errors = [];
    this._scopes = [];
    this._initialPos = null;
    this._scene = null;
    this._mirror = null;
    this._mirrorPending = false;
    // Generation token for fire-and-forget readbacks. A scene restore or an
    // explicit sync can overtake an older Promise callback on the JS queue;
    // only the newest generation may publish a host mirror.
    this._mirrorEpoch = 0;
    // Monotonic count of spawnFluid calls, and the value the current _mirror
    // snapshot is known to include. See refreshMirror.
    this._spawnSeq = 0;
    this._mirrorSeq = 0;
    // §17. The permutation the last eraseWhere applied (old slot per new slot),
    // and a counter that increments every time one happened. Anything storing a
    // raw slot index has to be able to tell that it was renumbered; before this
    // there was no way to know except by re-reading the whole world.
    this._lastPermutation = null;
    // §20. The COMPACT body list. Records [0, _nBodies) in the rigid buffer are
    // the ones the three §20 passes dispatch over, so a dead record must be
    // compacted away by the host rather than skipped by the GPU — that is what
    // keeps `nBodies == 0` mean 'launch nothing at all'.
    this._nBodies = 0;
    this._bodies = [];       // host mirror of the live records, newest last
    // Rung 4. Imported module expressions, keyed by compact body-record index.
    // Each entry carries a pure callback plus an optional JSON-safe authored
    // spec so Undo and native `.suna` can reconstruct imported machines.
    this._bodyDrivers = new Map();
    // §23.B. The jet list, and the key that decides whether the uniform needs
    // rewriting. Both live here rather than in `loadScene` alone so that an
    // engine which never loads a scene still answers `jetCount` truthfully.
    this._jets = [];
    this._jetKey = '';
    // §24. The player's drive vector as the GPU has it. Same reason as above:
    // an engine that never loads a scene still answers `userDrive` truthfully.
    this._userA = [0, 0];
    // §26.C. The world's gravity multiplier, Q16, and whether the scene's
    // boundary container is currently in the world. Both start at the value
    // that makes every pre-§26 call byte-identical: ONE, and true.
    this._gravQ = ONE;
    this._wallsOn = true;

    // §18. `_substep` is the FORMATION SCHEDULE's clock: it counts substeps
    // ENCODED, so `(_substep % BOND_PERIOD) == 0` is a pure function of the
    // substep index and the schedule is part of the keystream.
    // `_weldSeq` is the 12-bit spawn-event id (§18.2 W1), monotonic, starting at
    // 1; `_bondDirty` forces a formation pass on the very next substep after any
    // spawn, which is what makes the tag's lifetime exactly one pass.
    this._substep = 0;
    this._weldSeq = 1;
    // §18.22. The gesture that is currently open, as a 12-bit id and as the same
    // id pre-shifted into WELD_TAG_MASK position (which is the form the shader
    // compares against). 0 = nothing open, and that is the state of every scene
    // the determinism gate or a golden ever sees.
    this._weldHoldSeq = 0;
    this._weldHold = 0;
    // §18.24. The gesture whose FUSION is armed, pre-shifted like _weldHold.
    // 0 = nothing fuses, and that is the state of every scene the determinism
    // gate or a golden ever sees. Armed by `armFuse`, disarmed by the next
    // `beginWeldGroup`, and cleared by `step()` after the formation pass that
    // follows the gesture's release — the same pass that clears the particle
    // tags, so "armed" and "some particle wears the tag" end together and a
    // 12-bit wrap can never resurrect a stale arm.
    this._fuseTag = 0;
    // §26.B. The tag the last spawnFluid batch wore. See `lastWeldTag`.
    this._lastWeldTag = 0;
    this._bondDirty = true;
    this._eraseSeq = 0;
  }

  get specVersion() { return SPEC_VERSION; }

  // ---- error discipline ----------------------------------------------------
  _guardOpen() {
    this.device.pushErrorScope('validation');
    this.device.pushErrorScope('internal');
  }
  _guardClose(label) {
    const dev = this.device;
    // The pop promises are tracked so checkErrors() cannot race ahead of them
    // and report "clean" before a validation failure has been delivered.
    //
    // THE ENTRIES REMOVE THEMSELVES ONCE THEY HAVE SETTLED, and that matters
    // for a reason that only shows up in the toy: the interactive loop calls
    // step() and refreshMirror() every frame and drains MESSAGES with
    // drainErrors(), which never touches this array — so before 2026-07-26 it
    // grew by 4 entries every frame for as long as the page was open and
    // nothing ever shortened it (measured in the shipped toy: exactly 4.000
    // per frame, 5682 retained after ~24 s of play). By the time the removal
    // handler runs, the `.then` above it has ALREADY pushed any message into
    // this._errors, so dropping the settled promise loses nothing: what
    // checkErrors() must wait for is the scopes still IN FLIGHT, and those are
    // the only ones left in the array. The in-flight count is bounded by GPU
    // latency (about a frame), so indexOf/splice here is over ~10 entries.
    // A rejection is routed into this._errors rather than left to reject the
    // tracked promise: checkErrors() already throws on a non-empty _errors, so
    // the discipline is unchanged, and a toy that never calls checkErrors()
    // cannot produce an unhandled rejection out of a dropped entry.
    const track = (p) => {
      const entry = p
        .catch((e) => { this._errors.push(`[${label}] scope rejected: ${String((e && e.message) || e)}`); })
        .finally(() => {
          const k = this._scopes.indexOf(entry);
          if (k >= 0) this._scopes.splice(k, 1);
        });
      this._scopes.push(entry);
    };
    track(dev.popErrorScope().then((e) => { if (e) this._errors.push(`[${label}] internal: ${e.message}`); }));
    track(dev.popErrorScope().then((e) => { if (e) this._errors.push(`[${label}] validation: ${e.message}`); }));
  }
  /** Resolve every outstanding error scope and throw if anything was flagged. */
  async checkErrors() {
    await this.device.queue.onSubmittedWorkDone();
    const pending = this._scopes; this._scopes = [];
    await Promise.all(pending);
    const all = [...this._errors, ...this._uncaptured];
    if (all.length) throw new Error('[aether] GPU errors:\n  ' + all.join('\n  '));
    return true;
  }
  drainErrors() {
    const all = [...this._errors, ...this._uncaptured];
    this._errors.length = 0; this._uncaptured.length = 0;
    return all;
  }

  // ---- scene ---------------------------------------------------------------
  loadScene(scene) {
    const { n, nFluid, pos, vel, matId } = scene;
    assert(Number.isInteger(n) && n >= 0, 'scene.n must be a non-negative integer');
    assert(n <= this.maxParticles, `scene.n = ${n} exceeds maxParticles = ${this.maxParticles}`);
    assert(nFluid >= 0 && nFluid <= n, 'scene.nFluid out of range');
    assert(pos.length === n * 2 && vel.length === n * 2 && matId.length === n, 'scene array lengths must be n*2 / n*2 / n');
    for (let i = 0; i < n; i++) {
      assert(matId[i] < this.matCount,
        `matId[${i}] = ${matId[i]} >= matCount ${this.matCount} (it would index the zeroed ` +
        'tail of the mats buffer: eps = 0, dpMax = 0, i.e. a silent divide-by-zero factory)');
      // Fluid matter must start inside the representable positive world. The
      // boundary tail is allowed just outside it: full-bleed viewport scenes
      // deliberately place their invisible collision ring offscreen, and an
      // exact native `.suna` restore must be able to feed that ring back in.
      const lo = i < nFluid ? 0 : -MAX_POS;
      assert(pos[i * 2] >= lo && pos[i * 2] <= MAX_POS &&
        pos[i * 2 + 1] >= lo && pos[i * 2 + 1] <= MAX_POS,
        `pos[${i}] outside [${lo}, 2^27]`);
    }
    // §12. flags / temp / fuel are LIVE STATE from v4 on. All three are optional
    // in a scene: absent temp means AMBIENT (not 0 — a box of 0-degree water
    // would freeze on the first substep), absent fuel means the material's own
    // fuel0, absent flags means nothing is alight.
    if (scene.flags) {
      for (let i = 0; i < n; i++) {
        assert(Number.isInteger(scene.flags[i]) && (scene.flags[i] & ~FLAG_MASK) === 0,
          `flags[${i}] = ${scene.flags[i]} sets a reserved bit (legal mask 0x${FLAG_MASK.toString(16)})`);
      }
    }
    if (scene.temp) {
      for (let i = 0; i < n; i++) {
        assert(Number.isInteger(scene.temp[i]) && scene.temp[i] >= TEMP_MIN && scene.temp[i] <= TEMP_MAX,
          `temp[${i}] = ${scene.temp[i]} outside [${TEMP_MIN}, ${TEMP_MAX}]`);
      }
    }
    if (scene.fuel) {
      for (let i = 0; i < n; i++) {
        assert(Number.isInteger(scene.fuel[i]) && scene.fuel[i] >= 0, `fuel[${i}] = ${scene.fuel[i]} < 0`);
      }
    }

    const cellShift = scene.cellShift ?? CELL_SHIFT;
    // `cell_coord` in sim.wgsl evaluates `(p.x - originX) >> P.cellShift`. WGSL
    // leaves a shift by >= 32 INDETERMINATE, so an out-of-range cellShift is the
    // single construct in the whole sim path that a driver update or a different
    // backend could legitimately evaluate differently. Before 2026-07-25 this was
    // unchecked and 63 and -1 both sailed through: `1 << 63` and `1 << -1` collapse
    // in JS so the cellTotal guard below happened to compute 1 and accept them.
    // 27 is the ceiling because MAX_POS is 2^27, so a larger shift is meaningless.
    assert(Number.isInteger(cellShift) && cellShift >= 1 && cellShift <= 27,
      `scene.cellShift = ${cellShift} outside the 1..27 contract (sim.wgsl shifts by it; ` +
      'WGSL leaves a shift by >= 32 indeterminate)');
    // A CELL MUST BE AT LEAST ONE KERNEL RADIUS WIDE. buildNbr walks a FIXED 3x3
    // stencil (sim.wgsl §"Fixed 3x3 stencil traversal"), which only sees |dcx|<=1
    // and |dcy|<=1. That is a complete neighbour search if and only if
    // 2^cellShift >= H; otherwise pairs in the (cellSize, H] shell sit two cells
    // apart and are SILENTLY MISSED — no dbg counter fires, the run stays
    // perfectly deterministic, and the physics is quietly wrong.
    // Added 2026-07-26. It was unasserted, and cellShift=17 (2.0 wu cells vs
    // H=2.5 wu) was in live use by two test pages; a particle at 1.99 wu and one
    // at 4.40 wu are 2.41 wu apart — inside the kernel, outside the stencil.
    //
    // The 4-colour Gauss-Seidel sweep depends on the SAME inequality for a second
    // reason (§5.1b): cell_colour is (cx&1)|((cy&1)<<1), so two cells of the same
    // colour are >= 2 apart in index, and only a 3x3 stencil guarantees that a
    // same-colour pair is therefore always the SAME cell. Widen the stencil and
    // the colouring stops being a colouring. Race-free either way (dp is
    // deferred), but the sweep silently degrades toward Jacobi.
    assert(Math.pow(2, cellShift) >= H,
      `scene.cellShift = ${cellShift} gives ${Math.pow(2, cellShift) / ONE} wu cells, smaller than the ` +
      `kernel radius H = ${H / ONE} wu. buildNbr's 3x3 stencil would silently miss neighbours, and ` +
      `the 4-colour solver sweep (§5.1b) would stop being a valid colouring. Minimum is ` +
      `cellShift = ${Math.ceil(Math.log2(H))}.`);
    const domW = scene.domW, domH = scene.domH;
    // Math.pow, not `1 << cellShift`: the shift wraps and the guard below would be
    // relying on that wraparound rather than on arithmetic.
    const cellSize = Math.pow(2, cellShift);
    const gridW = Math.max(1, ceilDiv(domW, cellSize));
    const gridH = Math.max(1, ceilDiv(domH, cellSize));
    const cellTotal = gridW * gridH;
    assert(cellTotal <= MAX_CELL_TOTAL, `cellTotal ${cellTotal} exceeds the 2-level scan limit ${MAX_CELL_TOTAL}`);
    assert(cellTotal <= this.maxCells, `cellTotal ${cellTotal} exceeds allocated maxCells ${this.maxCells}`);

    this.n = n; this.nFluid = nFluid;
    this.gridW = gridW; this.gridH = gridH; this.cellTotal = cellTotal;
    this.cellShift = cellShift; this.domW = domW; this.domH = domH;
    this.substepCount = 0; this.parity = 0;
    this._substep = 0;
    this._scene = scene;
    this._mirror = null;
    this._mirrorPending = false;
    this._mirrorEpoch++;
    // §18.2 W1/W4. One loadScene is ONE bonding group, so the whole scene shares
    // tag 1 and the sequence restarts. _bondDirty makes the formation pass run
    // on the very next substep regardless of BOND_PERIOD, which is also what
    // makes the 12-bit tag wrap unreachable: the only tags alive at any
    // formation pass are those issued since the previous substep.
    this._weldSeq = 1;
    // §18.22. A scene load ENDS any open gesture. A hold that outlived the world
    // it was drawn in would let the next stroke weld to a scene it never touched.
    this._weldHoldSeq = 0; this._weldHold = 0;
    // §18.24. A scene load disarms fusion for the same reason it releases the
    // hold: an arm that outlived the gesture it was armed in would fuse a
    // stroke onto a scene it never touched. Params word 82 is zeroed with the
    // rest of the fresh ArrayBuffer below.
    this._fuseTag = 0;
    this._bondDirty = true;
    // §20. A SCENE LOAD DISSOLVES EVERY BODY. A record that outlived the world it
    // was welded in would place members into slots that now hold other matter.
    this._nBodies = 0; this._bodies = []; this._bodyDrivers.clear();
    // §26.C. A scene load PUTS THE CONTAINER BACK, because `n` below counts the
    // scene's boundary block. `_gravQ` deliberately does NOT reset: it lives in
    // the material image, which `loadScene` does not touch, so the world keeps
    // the weight the host asked for across a Reset.
    this._wallsOn = true;

    // Params, written ONCE per scene apart from the authoring fields (`n`,
    // `nFluid`, `weldHold`, `nJets`). No frame/substep field exists.
    // 320 B: a uniform-address-space struct is size-rounded to 16, and §23.B's
    // two jet arrays must each start on a 16-byte boundary.
    const p = new ArrayBuffer(PARAMS_BYTES);
    const pu = new Uint32Array(p), pi = new Int32Array(p);
    pu[0] = n; pu[1] = nFluid; pu[2] = gridW; pu[3] = gridH;
    pu[4] = cellTotal; pu[5] = cellShift;
    pi[6] = 0; pi[7] = 0;            // originX, originY
    pi[8] = domW; pi[9] = domH;
    pi[10] = VMAX; pu[11] = MAXNBR;
    pu[12] = 0;                      // §18.22 weldHold — released by a scene load
    pu[13] = 0;                      // §20 nBodies — a scene load dissolves every body
    // §23.B. A SCENE LOAD DISSOLVES EVERY JET, for the same reason it dissolves
    // every body: a device that outlived the world it was placed in would push
    // matter that was never anywhere near it. The whole jet array is zeroed by
    // the ArrayBuffer, so the uniform is provably clean and not merely unread.
    pu[14] = 0;                      // §23.B nJets
    pu[15] = 0;                      // _padJet
    this._jets = [];
    this._jetKey = '';
    // §24. A SCENE LOAD RELEASES THE PLAYER'S KEYS, for the same reason it
    // dissolves every body and every jet: a key held down across a scene load
    // would drive matter that was never anywhere near the hand that pressed it.
    // Words 80..83 are already zero in the ArrayBuffer, so the uniform is
    // provably clean rather than merely unread.
    pi[80] = 0; pi[81] = 0;          // §24 userAx/userAy — keys released
    pu[82] = 0;                      // §18.24 fuseTag — disarmed by a scene load
    pu[83] = 0;                      // tail pad
    this._userA = [0, 0];
    this.device.queue.writeBuffer(this.buf.params, 0, p);

    // §7.6. hashedWords was 6 through v3 because words 6 and 7 (`temp`, `pad0`)
    // were reserved and provably never written. v4 makes them `temp` and `fuel`
    // — live simulation state — so the digest MUST cover all 8 or the gate
    // silently stops seeing every phase change and every fire.
    // §18. `_pad` became `n_fluid` and `bond_cap` was appended: 4 -> 8 words.
    // n_fluid is the bond fold's guard — bondForm1/2 are fluidOnly, so rows in
    // [nFluid, n) are NEVER WRITTEN and may hold a previous scene's bytes.
    // Folding them makes the digest a function of memory nothing writes.
    const hp = new Uint32Array([n, PARTICLE_WORDS, HASHED_WORDS, nFluid, BOND_CAP, 0, 0, 0]);
    this.device.queue.writeBuffer(this.buf.hashParams, 0, hp);

    // Zero the FULL capacity of every buffer — not just [0,n). Stale tails from
    // a previous scene are a confirmed hazard (spec §7.3 / report 4 H1).
    this._guardOpen();
    const enc = this.device.createCommandEncoder({ label: 'loadScene:clear' });
    for (const k of ['stateA', 'stateB', 'derived', 'cellCount', 'cellStart', 'blockSums',
      'cellOf', 'bucketIds', 'sortedIds', 'nbr', 'nbrN', 'dbg', 'hashAcc',
      // §18. The FULL capacity of both, not just [0, n): a stale row from a
      // previous scene would be read as a live bond to a slot that now holds
      // different matter.
      'bond', 'bondCand',
      // §21. A new scene has no imported colour. Clearing the FULL capacity is
      // what makes "alpha 0 = material look" hold for every slot the scene does
      // not fill, so a later spawnFluid into a slot a previous scene coloured
      // cannot inherit a stranger's pigment.
      'tint']) {
      enc.clearBuffer(this.buf[k]);
    }
    // §20. Clear the body records and the rest offsets but NOT the cosine table
    // in front of them — that is baked once at createEngine and is the same
    // bytes forever. Clearing it would leave every body placing its members at
    // the origin, silently.
    enc.clearBuffer(this.buf.rigid, RG_BODY_BASE * 4);
    this.device.queue.submit([enc.finish()]);
    this._guardClose('loadScene:clear');

    // state[0]
    const st = new Int32Array(n * PARTICLE_WORDS);
    const stu = new Uint32Array(st.buffer);
    const fuelOf = (m) => this.mats[m][MAT_FIELDS.indexOf('fuel0')];
    // §17. FLAG_DEAD is DERIVED, never authored. `thermal` recomputes it from
    // mats[mat].phantom every substep, but `buildNbr` runs BEFORE `thermal` in
    // substep 1 — so a scene that hand-set the bit wrongly would get one substep
    // of a phantom that pushes, or one substep of a corpse that walks, and then
    // silently correct itself. Deriving it here removes that window entirely,
    // and a scene that disagrees is a hard throw rather than a one-substep
    // glitch nothing would ever see.
    const iPhantom = MAT_FIELDS.indexOf('phantom');
    const deadOf = (m) => (this.mats[m][iPhantom] !== 0 ? FLAG_DEAD : 0);
    const sealOf = (m) => (sealRow(this.mats[m]) ? FLAG_SEAL : 0);
    const weldsOnSpawn = (m) => spawnWelds(this.mats[m]);
    for (let i = 0; i < n; i++) {
      const o = i * PARTICLE_WORDS;
      const m = matId[i] >>> 0;
      const want = deadOf(m);
      const given = (scene.flags ? scene.flags[i] : 0) >>> 0;
      // ONE-SIDED ON PURPOSE. A scene that omits the bit on a phantom is just a
      // scene that did not know about §17, and deriving it is the whole point.
      // A scene that SETS it on live matter is the dangerous direction and the
      // only one worth throwing for: that particle would be a ghost for exactly
      // one substep (buildNbr runs before thermal), push nothing, be pushed by
      // nothing, and then quietly come back to life with no trace anywhere.
      assert(!(given & FLAG_DEAD) || want !== 0,
        `scene.flags[${i}] sets FLAG_DEAD but matId ${m} (${MAT_NAMES[m] ?? m}) has ` +
        `phantom = ${this.mats[m][iPhantom]}. FLAG_DEAD is a pure function of the material row ` +
        `(§17) — retire a particle by giving it a phantom material, not by setting the flag.`);
      st[o + 0] = pos[i * 2]; st[o + 1] = pos[i * 2 + 1];
      st[o + 2] = vel[i * 2]; st[o + 3] = vel[i * 2 + 1];
      stu[o + 4] = m;
      // §18.2 W1. The weld tag is ENGINE-ASSIGNED and never authored, exactly
      // like FLAG_DEAD — loadScene's `(flags[i] & ~FLAG_MASK) === 0` assertion
      // above still refuses a scene that tries. It is written ONLY when the row
      // can bond, so every flags word on an all-WATER scene is bit-for-bit v7's
      // and the golden chain survives (gate B0).
      const tag = (i < nFluid && weldsOnSpawn(m)) ? (1 << WELD_TAG_SHIFT) : 0;
      stu[o + 5] = ((given | want | sealOf(m) | tag) >>> 0);
      st[o + 6] = scene.temp ? scene.temp[i] : AMBIENT;
      st[o + 7] = scene.fuel ? scene.fuel[i] : fuelOf(m);
    }
    if (n > 0) {
      this.device.queue.writeBuffer(this.buf.stateA, 0, st);
      // Boundary particles never move, so BOTH ping-pong buffers must agree on
      // them from substep 0 (finalize copies them through, but the hash of the
      // pre-first-step state should be consistent either way).
      this.device.queue.writeBuffer(this.buf.stateB, 0, st);
    }

    // derived[i].pred = pos[i] for ALL i, so boundary pred is valid at substep 0.
    const dv = new Int32Array(n * DERIVED_WORDS);
    for (let i = 0; i < n; i++) {
      dv[i * DERIVED_WORDS + 0] = pos[i * 2];
      dv[i * DERIVED_WORDS + 1] = pos[i * 2 + 1];
    }
    if (n > 0) this.device.queue.writeBuffer(this.buf.derived, 0, dv);

    this._initialPos = new Int32Array(pos);
    this._initialState = st;
    this.resetChain();
    return this;
  }

  resetChain() {
    this.device.queue.writeBuffer(this.buf.hashChain, 0, Uint32Array.from(CHAIN_INIT));
  }

  // ---- stepping ------------------------------------------------------------
  /**
   * Advance exactly `count` SUBSTEPS. Synchronous encode + submit.
   * Reads no clock. Batching is free to change: identical results were verified
   * across submit batch sizes 1 / 25 / 64.
   */
  step(count = 1) {
    assert(this.n > 0, 'step() called before loadScene()');
    assert(Number.isInteger(count) && count >= 0, 'step(count) needs a non-negative integer');
    if (count === 0) return this.substepCount;
    this._guardOpen();
    let enc = null;
    let batched = 0;
    for (let s = 0; s < count; s++) {
      if (enc === null) enc = this.device.createCommandEncoder({ label: 'aether:step' });
      // Rung 4. Each batched substep receives a distinct upload-ring source.
      // The encoded copy sits immediately before the solver dispatches which
      // consume it, preserving scene time without one queue submit per substep.
      this._serviceBodyConstraints(enc, batched);
      this._encodeSubstep(enc);
      this.parity = 1 - this.parity;
      this.substepCount++;
      if (++batched >= this.submitBatch) {
        this.device.queue.submit([enc.finish()]);
        enc = null; batched = 0;
      }
    }
    if (enc !== null) this.device.queue.submit([enc.finish()]);
    this._guardClose('step');
    // §18.24. THE ARM DIES WITH THE TAGS. If the armed gesture has been
    // released (weldHold 0), the substeps just encoded include the prompt
    // formation pass (`endWeldGroup` set _bondDirty) — the pass that both
    // FUSES the finished stroke onto whatever it overlaps and clears its
    // particle tags. After it, no particle wears the armed tag, so the arm is
    // dead weight and is cleared HERE, not in endWeldGroup: clearing on the
    // release edge would kill fusion for exactly the frozen-stroke case the
    // slingshot is (a paused draw runs NO formation pass until this step).
    if (this._fuseTag !== 0 && this._weldHold === 0) {
      this._fuseTag = 0;
      this._writeFuseTag();
    }
    return this.substepCount;
  }

  frame() { return this.step(SUBSTEPS_PER_FRAME); }

  _encodeSubstep(enc) {
    const p = this.parity;
    const cellBytes = this.cellTotal * 4;

    // PROFILE MODE (opt-in, off in every shipped path). Each dispatch is given
    // its own compute pass carrying a timestampWrites pair. Dispatch order in
    // one encoder is total across pass boundaries — the same guarantee the
    // shipped substep already leans on to split itself around clearBuffer — so
    // the program is unchanged. Asserted, not assumed: test/perf/profile.html
    // compares the 10 000-substep chain both ways before reporting anything.
    const prof = this._profile && this._ts !== null;
    const ts = this._ts;
    // Slot numbering is per SUBSTEP, so a plain step() in profile mode reuses
    // the same 100 query slots every substep instead of running off the end.
    // profileSubstep() encodes exactly one substep per encoder, so reading
    // ts.labels straight after the encode is still correct.
    if (prof) ts.labels = [];
    let open = null;
    const beginPass = (label) => { if (!prof) open = enc.beginComputePass({ label }); };
    const endPass = () => { if (open) { open.end(); open = null; } };

    const run = (_pass, fn) => {
      const e = this._pipes.get(fn);
      // scanBlock is the one pass whose grid is NOT items/threads: each
      // workgroup consumes 1024 cells (256 threads x 4 elements).
      const groups = e.dim === '1' ? 1
        : e.dim === 'b' ? Math.max(1, ceilDiv(this.cellTotal, 1024))
        : e.dim === 'c' ? Math.max(1, ceilDiv(this.cellTotal, e.wgx))
        // §20: ONE WORKGROUP PER BODY, and ZERO when there are no bodies. That
        // zero is not an optimisation — it is the reason §20 cannot move the
        // golden chain: on every scene the determinism gate, the goldens or
        // `loadScene` produce without an explicit body, the three §20 passes
        // launch no invocations at all and touch nothing.
        : e.dim === 'bodies' ? this._nBodies
        // A fluidOnly pass returns on its first line for gid.x >= nFluid, so
        // launching those threads is pure cost. 50 of 58 dispatches qualify.
        : Math.max(1, ceilDiv(e.fluidOnly ? this.nFluid : this.n, e.wgx));
      if (groups === 0) return;
      let pass = open;
      if (prof) {
        const slot = ts.labels.length;
        if (slot * 2 + 1 >= ts.capacity) throw new Error('[aether] profile querySet too small');
        ts.labels.push(fn);
        pass = enc.beginComputePass({
          label: `aether:prof:${fn}`,
          timestampWrites: {
            querySet: ts.querySet,
            beginningOfPassWriteIndex: slot * 2,
            endOfPassWriteIndex: slot * 2 + 1,
          },
        });
      }
      pass.setPipeline(e.pipeline);
      pass.setBindGroup(0, e.groups[p]);
      pass.dispatchWorkgroups(groups);
      if (prof) pass.end();
    };

    // clearBuffer is an encoder-level command, so the substep is split into two
    // compute passes around each clear. Dispatch order across passes in one
    // encoder is still fully ordered, so this is bit-identical to a single pass.
    enc.clearBuffer(this.buf.cellCount, 0, cellBytes);
    beginPass('aether:sub:a');
    let pass = open;
    // §20 TRUE RIGID BODIES, and it runs FIRST. A body integrates its own four
    // degrees of freedom and PLACES its members into derived.pred; `predict`
    // then early-returns for them, because integrating a member a second time
    // would give it an independent trajectory, which is exactly the flexion §20
    // exists to remove. Zero workgroups when there are no bodies, which is why
    // §20 cannot move the golden chain.
    run(pass, 'rigidStep');
    run(pass, 'predict');
    run(pass, 'gridCount');
    run(pass, 'scanBlock');
    run(pass, 'scanTop');
    run(pass, 'scanAdd');
    endPass();

    enc.clearBuffer(this.buf.cellCount, 0, cellBytes);   // reused as scatter cursor
    beginPass('aether:sub:b');
    pass = open;
    run(pass, 'scatter');
    run(pass, 'canonicalize');
    run(pass, 'buildNbr');
    // §5.1b 4-colour block Gauss-Seidel. lambda is shared (Jacobi) across the
    // colours; the POSITION update is Gauss-Seidel between colour classes.
    // No pass in here both reads and writes `pred`, which is the same argument
    // that makes the old deferred-dp Jacobi loop race-free (§5.1).
    for (let it = 0; it < this.iters; it++) {
      run(pass, 'solveA');
      for (let c = 0; c < NCOLOUR; c++) {
        run(pass, `solveB_c${c}`);
        run(pass, `applyDp_c${c}`);
      }
      // §16 granular friction, INSIDE the iteration loop — one friction solve
      // per density solve. This is not a tuning choice, it is what makes the
      // term a CONSTRAINT rather than a damper, and it was measured both ways:
      // Δx_i = pred_i - pos_i is the displacement accumulated SO FAR this
      // substep, so re-solving it drives the pair's total tangential relative
      // displacement toward zero instead of shaving a fixed fraction off the
      // increment. Outside the loop, a 12x12 SAND block still spread to 5.0x its
      // width at mu = 4.0; the numbers for the shipped arrangement are in
      // test/granular.html G1.
      run(pass, 'friction');
      run(pass, 'applyFric');
      // §20. ONE dispatch per solver ITERATION, at the END of the iteration,
      // where every external correction for a member has already landed in
      // derived[i].dp: solveB wrote it (applyDp early-returned for the member)
      // and `friction` ADDED to it. One workgroup per body reduces both into
      // the body's force and torque, folds them with a CARRIED REMAINDER, and
      // re-places every member. Reduce + per-body update + place in ONE
      // dispatch is what makes §20 cost +6 dispatches instead of +14.
      run(pass, 'rigidSolve');
    }
    // §29 sees every correction that can move matter this substep.
    run(pass, 'contain');
    run(pass, 'containOverflow');
    run(pass, 'finalize');
    run(pass, 'xsph');
    run(pass, 'normals');
    run(pass, 'surfaceTension');
    // §12. conduct reads state_out[j].temp for every j and writes only
    // derived[i].heat; thermal reads derived[i].heat and writes only
    // state_out[i]. Two dispatches, so neither reads another invocation's write
    // — the same rule §5.1 applies to applyDp.
    run(pass, 'conduct');
    run(pass, 'thermal');
    // §29 releases a crushed held endpoint before rigidMember recounts.
    run(pass, 'containStress');
    // §20 membership, AFTER thermal so a member that melted, burned, corroded or
    // was retired THIS substep leaves on the same substep it stopped being the
    // material it was, and BEFORE bondForm1 so the formation pass sees the
    // membership the substep ended with. IDEMPOTENT — with nothing leaving the
    // re-base offset is exactly 0 and the pass is a no-op — so it is dispatched
    // unconditionally, with no dirty flag, no data-dependent branch and no host
    // readback. It also CLOSES the substep: V = C - c, W = Q - theta.
    run(pass, 'rigidMember');
    // §18 bond formation, at the TAIL. Two dispatches on one substep in four
    // (or immediately after any spawn), so 58 -> 58.5 mean, +0.86 %.
    // `_substep` counts SUBSTEPS ENCODED, not frames, so the schedule is a pure
    // function of the substep index and is therefore part of the keystream —
    // BOND_PERIOD is wire format exactly like ITERS and the colour order.
    if ((this._substep % BOND_PERIOD) === 0 || this._bondDirty) {
      run(pass, 'markCongealNbr');
      run(pass, 'bondForm1');
      run(pass, 'bondForm2');
      this._bondDirty = false;
    }
    this._substep++;
    endPass();

    if (this.chainEnabled) this._encodeHash(enc, 1 - p, true);
  }

  _encodeHash(enc, parity, fold) {
    const pass = enc.beginComputePass({ label: 'aether:hash' });
    pass.setBindGroup(0, this._hashGroups[parity]);
    pass.setPipeline(this._hashPipes.clear_acc);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(this._hashPipes.digest);
    pass.dispatchWorkgroups(Math.max(1, ceilDiv(this.n, 256)));
    if (fold) {
      pass.setPipeline(this._hashPipes.fold_chain);
      pass.dispatchWorkgroups(1);
    }
    pass.end();
  }

  /**
   * The workgroup count this engine will launch for every pass of one substep,
   * for the CURRENT scene. Pure host arithmetic, no GPU work, no side effects —
   * it exists so "this optimisation skips N threads" is a measurement instead
   * of a claim, and so a regression in dispatch sizing is visible without a
   * profiler.
   */
  dispatchGeometry() {
    const per = [];
    let total = 0;
    for (const p of PASSES) {
      const e = this._pipes.get(p.fn);
      const items = e.dim === '1' ? null
        : e.dim === 'b' ? this.cellTotal
        : e.dim === 'c' ? this.cellTotal
        : e.dim === 'bodies' ? this._nBodies
        : (e.fluidOnly ? this.nFluid : this.n);
      const groups = e.dim === '1' ? 1
        : e.dim === 'b' ? Math.max(1, ceilDiv(this.cellTotal, 1024))
        : e.dim === 'c' ? Math.max(1, ceilDiv(this.cellTotal, e.wgx))
        : e.dim === 'bodies' ? this._nBodies
        : Math.max(1, ceilDiv(items, e.wgx));
      // the repeats a substep actually issues (§5: iters x the coloured sweep)
      const reps = /^solveA$/.test(p.fn) ? this.iters
        : /^(solveB|applyDp)_c\d$/.test(p.fn) ? this.iters
        : /^rigidSolve$/.test(p.fn) ? this.iters : 1;
      per.push({ pass: p.fn, dim: p.dim, fluidOnly: !!e.fluidOnly, items, groups, reps });
      total += groups * reps;
    }
    return {
      n: this.n, nFluid: this.nFluid, cellTotal: this.cellTotal,
      dispatchesPerSubstep: per.reduce((s, x) => s + x.reps, 0),
      totalWorkgroups: total, per,
    };
  }

  // ---- per-dispatch GPU profiling (opt-in, never on a shipped path) ---------
  /**
   * Encode `reps` substeps with `timestamp-query` around EVERY dispatch and
   * return the mean GPU nanoseconds attributed to each pass.
   *
   * Reading this number correctly matters more than producing it:
   *   - Splitting one compute pass into 50 adds real per-pass encoder overhead
   *     on Metal, so the SUM of these numbers is LARGER than the unprofiled
   *     substep. Use them to rank passes, not to quote an absolute budget.
   *   - `share` is each pass's fraction of the profiled total, which is the
   *     figure that survives that inflation.
   *   - The state it produces is identical to an unprofiled run. That is the
   *     property that makes it safe, and test/perf/profile.html gates it.
   */
  async profileSubstep(reps = 60) {
    assert(this._profile, 'profileSubstep() needs createEngine({ profile: true })');
    assert(this.n > 0, 'profileSubstep() called before loadScene()');
    const CAP = 256;                                   // 128 dispatches, 50 used
    if (this._ts === null) {
      this._ts = {
        capacity: CAP,
        querySet: this.device.createQuerySet({ type: 'timestamp', count: CAP }),
        resolve: this.device.createBuffer({ size: CAP * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }),
        labels: [],
      };
    }
    const ts = this._ts;
    const acc = new Map();
    let counted = 0;

    for (let r = 0; r < reps; r++) {
      ts.labels = [];
      this._guardOpen();
      const enc = this.device.createCommandEncoder({ label: 'aether:profile' });
      this._encodeSubstep(enc);
      this.parity = 1 - this.parity;
      this.substepCount++;
      const used = ts.labels.length * 2;
      enc.resolveQuerySet(ts.querySet, 0, used, ts.resolve, 0);
      this.device.queue.submit([enc.finish()]);
      this._guardClose('profileSubstep');

      const raw = new BigUint64Array(await this._read(ts.resolve, used * 8));
      // The first rep is a warm-up: pipelines, caches and the query set itself
      // are cold, and it reliably reads several times high.
      if (r === 0) continue;
      counted++;
      for (let k = 0; k < ts.labels.length; k++) {
        const ns = Number(raw[k * 2 + 1] - raw[k * 2]);
        acc.set(ts.labels[k], (acc.get(ts.labels[k]) ?? 0) + ns);
      }
    }

    // Passes that run more than once per substep (solveA, solveB_c*, applyDp_c*)
    // are summed, which is what "what does this pass cost per substep" means.
    const rows = [...acc].map(([pass, ns]) => ({ pass, ns: ns / counted }));
    const total = rows.reduce((s, x) => s + x.ns, 0);
    for (const x of rows) { x.us = +(x.ns / 1000).toFixed(3); x.share = +(x.ns / total).toFixed(4); }
    rows.sort((a, b) => b.ns - a.ns);
    return { reps: counted, dispatches: ts.labels.length, totalUs: +(total / 1000).toFixed(3), rows };
  }

  // ---- readback ------------------------------------------------------------
  async _read(src, byteLength, offset = 0) {
    const size = Math.ceil(byteLength / 4) * 4;
    if (size === 0) return new ArrayBuffer(0);
    const staging = this.device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this._guardOpen();
    const enc = this.device.createCommandEncoder({ label: 'aether:readback' });
    enc.copyBufferToBuffer(src, offset, staging, 0, size);
    this.device.queue.submit([enc.finish()]);
    this._guardClose('readback');
    await staging.mapAsync(GPUMapMode.READ);
    const copy = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();
    return copy;
  }

  async readState(parity = this.parity) {
    const b = parity === 0 ? this.buf.stateA : this.buf.stateB;
    return new Int32Array(await this._read(b, this.n * 32));
  }
  async readDerived() { return new Int32Array(await this._read(this.buf.derived, this.n * DERIVED_BYTES)); }
  async readNbrN()    { return new Uint32Array(await this._read(this.buf.nbrN, this.n * 4)); }
  async readCellStart() { return new Uint32Array(await this._read(this.buf.cellStart, (this.cellTotal + 1) * 4)); }
  async debugCounters() { return new Uint32Array(await this._read(this.buf.dbg, 64)); }
  /** §18. The packed bond table, nFluid rows of BOND_CAP u32. */
  async readBond()     { return new Uint32Array(await this._read(this.buf.bond, this.n * BOND_CAP * 4)); }
  async readBondCand() { return new Uint32Array(await this._read(this.buf.bondCand, this.n * BOND_CAP * 4)); }

  /**
   * Install an explicit, symmetric distance graph from an imported authoring
   * format.  This is intentionally an authoring API, beside spawnFluid and
   * replaceWhere: it changes scene state but never changes the solver.
   *
   * Rows touched by the supplied graph replace their proximity-generated
   * bonds. Untouched rows keep theirs. This lets a file with explicit joints
   * mean what it authored without dissolving unrelated bonded matter.
   */
  async installAuthoredBonds(pairs) {
    assert(Array.isArray(pairs), 'installAuthoredBonds expects an array');
    if (!pairs.length || this.nFluid === 0) return { installed: 0, skipped: pairs.length };
    const st = await this.readState();
    const stu = new Uint32Array(st.buffer, st.byteOffset, st.length);
    const bond = await this.readBond();
    const touched = new Uint8Array(this.nFluid);
    const rows = Array.from({ length: this.nFluid }, () => []);
    const iBondK = MAT_FIELDS.indexOf('bondK');
    let skipped = 0;

    for (const p of pairs) {
      const a = p?.a | 0, b = p?.b | 0;
      if (a < 0 || b < 0 || a >= this.nFluid || b >= this.nFluid || a === b) {
        skipped++; continue;
      }
      const fa = stu[a * PARTICLE_WORDS + 5], fb = stu[b * PARTICLE_WORDS + 5];
      const ma = stu[a * PARTICLE_WORDS + 4], mb = stu[b * PARTICLE_WORDS + 4];
      if ((fa & FLAG_RIGID) || (fb & FLAG_RIGID) ||
          this.mats[ma][iBondK] === 0 || this.mats[mb][iBondK] === 0) {
        skipped++; continue;
      }
      const l0 = Math.max(512, Math.min(5120, p.l0 | 0));
      touched[a] = 1; touched[b] = 1;
      rows[a].push({ j: b, l0 });
      rows[b].push({ j: a, l0 });
    }

    let installed = 0;
    const installedByKind = Object.create(null);
    for (let i = 0; i < this.nFluid; i++) {
      if (!touched[i]) continue;
      const unique = new Map();
      for (const p of rows[i]) if (!unique.has(p.j)) unique.set(p.j, p.l0);
      const list = [...unique].map(([j, l0]) => ({ j, l0 }))
        .sort((a, b) => a.j - b.j);
      if (list.length > BOND_CAP) skipped += list.length - BOND_CAP;
      const base = i * BOND_CAP;
      bond.fill(0, base, base + BOND_CAP);
      for (let k = 0; k < Math.min(BOND_CAP, list.length); k++) {
        const p = list[k];
        bond[base + k] = (((p.l0 & 0x1fff) << 17) | (p.j & BOND_J_MASK)) >>> 0;
      }
    }
    // Count mutual pairs actually present after both row caps.
    for (const p of pairs) {
      const a = p?.a | 0, b = p?.b | 0;
      if (a < 0 || b < 0 || a >= this.nFluid || b >= this.nFluid || a >= b) continue;
      const has = (i, j) => {
        const base = i * BOND_CAP;
        for (let k = 0; k < BOND_CAP && bond[base + k]; k++) {
          if ((bond[base + k] & BOND_J_MASK) === j) return true;
        }
        return false;
      };
      if (has(a, b) && has(b, a)) {
        installed++;
        const kind = typeof p?.kind === 'string' ? p.kind : 'authored';
        installedByKind[kind] = (installedByKind[kind] || 0) + 1;
      }
    }
    // Input files need not order endpoints. Recount from rows if all pairs were
    // b<a so the report remains correct.
    if (!installed) {
      for (let i = 0; i < this.nFluid; i++) for (let k = 0; k < BOND_CAP; k++) {
        const w = bond[i * BOND_CAP + k];
        if (!w) break;
        if ((w & BOND_J_MASK) > i && touched[i]) installed++;
      }
    }
    this.device.queue.writeBuffer(this.buf.bond, 0, bond);
    this.device.queue.writeBuffer(this.buf.bondCand, 0, bond);
    this._bondDirty = false;
    return { installed, skipped, installedByKind };
  }
  /**
   * §21. The per-slot RGBA8 colour sidecar. NOT simulation state — nothing in
   * `sim.wgsl` binds this buffer, it is not in the digest, and reading it can
   * never change a run.
   */
  async readTints(count = this.n) {
    return new Uint32Array(await this._read(this.buf.tint, Math.max(1, count) * 4));
  }
  /**
   * §21. Paint a contiguous run of SLOTS. Authoring only, exactly like
   * spawnFluid / eraseWhere / applyImpulse — the determinism gate never calls
   * it, and it cannot change a trajectory even if it did.
   * @param {number} first first slot index
   * @param {Uint32Array|number[]} words one RGBA8 word per slot; a == 0 clears
   */
  setTints(first, words) {
    assert(Number.isInteger(first) && first >= 0 && first + words.length <= this.maxParticles,
      `setTints: [${first}, ${first + words.length}) is outside [0, ${this.maxParticles})`);
    if (!words.length) return 0;
    this.device.queue.writeBuffer(this.buf.tint, first * 4, Uint32Array.from(words, (w) => w >>> 0));
    return words.length;
  }
  /** §21. Pack straight-alpha bytes into the sidecar's word order. */
  static packTint(r, g, b, a = 255) { return packTint(r, g, b, a); }
  async chainHex()      { return hex8(new Uint32Array(await this._read(this.buf.hashChain, 32))); }

  /**
   * Throw unless the run so far stayed inside the regime the spec proves bounds
   * for. MANDATORY on any path that treats the simulation as a keystream.
   *
   * SPEC §5.4: a non-zero dbg[6] means buildNbr dropped neighbours, so the §1.2
   * accumulator bounds are no longer proven and the run must be treated as
   * INVALID. Until 2026-07-25 nothing outside CI checked this, while the initial
   * particle state — which is exactly what product 2 encodes — can reach the
   * truncation regime at roughly 2.5x rest density with no adversarial effort.
   * A decode that silently entered an unproven regime is worse than one that
   * refuses.
   *
   * This is deliberately NOT called from step(): it costs a GPU sync, and paying
   * that per substep would be absurd. Call it after loading a scene and again at
   * the end of any keystream run.
   *
   * Note dbg counters are u32 and WRAP (measured: 3366859735 at 300 substeps ->
   * 2432398538 at 10000 on a 20k-particle scene), so only ==0 is meaningful.
   */
  async assertSpecValidRegime(where = 'run') {
    const dbg = await this.debugCounters();
    // fixed.wgsl:59-62 owns these names; keep the two lists in step.
    // fixed.wgsl:59-64 owns these names; keep the two lists in step.
    // §18 APPENDED slot 8 and did not move any of the first eight. It could not:
    // dbg[7] is scanTop's cell/scan capacity detector and this list is
    // positional, so taking 7 would have corrupted the one function that decides
    // whether a run is a valid keystream.
    // A bond row saturation makes the bond graph a function of TRAVERSAL ORDER
    // rather than of the particle set, which is precisely the property §5.4
    // amendment 2 had to pin as wire format for MAXNBR — so it invalidates the
    // run exactly like every other slot, with no special case here.
    const names = ['mul/shift saturation', 'divide saturation or bad shift', 'divide by zero',
      'sqrt of negative', 'rsqrt of non-positive', 'add/sub/neg saturation',
      'neighbour-list truncation', 'cell/scan capacity overflow',
      'bond row saturation',
      'RESERVED dbg[9]', 'RESERVED dbg[10]', 'RESERVED dbg[11]',
      'RESERVED dbg[12]', 'RESERVED dbg[13]', 'RESERVED dbg[14]', 'RESERVED dbg[15]'];
    const hit = [...dbg].map((v, i) => (v ? `${names[i]} (dbg[${i}]=${v})` : null)).filter(Boolean);
    assert(hit.length === 0,
      `[aether] ${where}: simulation left the spec-valid regime — ${hit.join('; ')}. ` +
      'Per SPEC §5.4 this run is INVALID as a keystream.');
    return true;
  }

  /** One-shot digest of the current state. Does NOT advance the chain. */
  async digest() {
    this._guardOpen();
    const enc = this.device.createCommandEncoder({ label: 'aether:digest' });
    this._encodeHash(enc, this.parity, false);
    this.device.queue.submit([enc.finish()]);
    this._guardClose('digest');
    const acc = new Uint32Array(await this._read(this.buf.hashAcc, 32));
    return {
      acc,
      hex: hex8(acc),
      setHex: hex8(acc.subarray(0, 4)),
      slotHex: hex8(acc.subarray(4, 8)),
    };
  }

  /** CPU digest of a readback — used to prove GPU/JS hash parity. */
  async digestCPU() {
    const st = await this.readState();
    // §18. THE BOND READBACK IS NOT OPTIONAL. Without it the JS digest omits the
    // bond fold the GPU performs, and `cpuDigestOfFinalReadback` — which the
    // golden records — silently diverges from the GPU digest on every bonded
    // scene. A one-line omission that would look exactly like a real divergence.
    const bond = await this.readBond();
    return digestState(st, {
      particleCount: this.n, strideWords: PARTICLE_WORDS, hashedWords: HASHED_WORDS,
      nFluid: this.nFluid, bond, bondCap: BOND_CAP,
    });
  }

  // ---- liveness ------------------------------------------------------------
  /**
   * @param {{window?: number}} o  window > 0 measures movement over a RECENT
   *   interval instead of only against the initial positions, which is the only
   *   way to detect a freeze (movedFraction latches at 1.0 forever). It does
   *   this by snapshotting positions and ADVANCING THE SIMULATION by `window`
   *   substeps — call it after every digest you care about.
   */
  async liveness({ window = 0 } = {}) {
    let recentPos = null;
    if (window > 0) {
      const before = await this.readState();
      recentPos = new Int32Array(this.nFluid * 2);
      for (let i = 0; i < this.nFluid; i++) {
        recentPos[i * 2] = before[i * PARTICLE_WORDS];
        recentPos[i * 2 + 1] = before[i * PARTICLE_WORDS + 1];
      }
      this.step(window);
    }
    const st = await this.readState();
    const dv = await this.readDerived();
    const nb = await this.readNbrN();
    let cs = null;
    try { cs = await this.readCellStart(); } catch { /* optional */ }
    return computeLiveness({
      state: st, derived: dv, nbrN: nb, cellStart: cs,
      n: this.n, nFluid: this.nFluid, domW: this.domW, domH: this.domH,
      initialPos: this._initialPos, recentPos,
    });
  }

  // ---- interactive authoring (NOT part of the deterministic contract) ------

  // ════════════════════════════════════════════════════════════════════════
  // §18.22 — ONE GESTURE, ONE WELD GROUP.
  //
  // THE BUG THIS EXISTS FOR. §18.2 W1 made the SPAWN EVENT the bonding group,
  // and that is still the right rule — it is what stops a poured emitter welding
  // into a lump and what makes two strokes independent. But `index.html` calls
  // `paint()` every frame the pointer is held, so a one-second drag at 120 fps
  // is ~120 spawn events. Each frame's crescent of brush welded into its own
  // sliver, bonded to nothing else, and the stone Darien painted fell in drops.
  // MEASURED through the real UI before the fix: a 30 wu STONE drag laid 180
  // particles as 39 bodies + 20 loose grains, largest 13, settling at r_g 18.72
  // wu against 9.49 for the identical 180 particles in ONE call.
  //
  // WHY THE TAG IS LATCHED RATHER THAN MAKING FORMATION BOND TO AN ALREADY-
  // WELDED PARTICLE. The alternative — "let a new particle bond to one that is
  // already welded into the same group" — needs a group id that OUTLIVES the
  // formation pass, i.e. the 12-bit tag would have to become permanent state.
  // That is worse in three separate ways:
  //   1. The wrap stops being unreachable and becomes REACHABLE: every id would
  //      have to stay unique for the lifetime of the scene, and 4095 strokes is
  //      half an hour of play — after which two strokes share an id and weld to
  //      each other on contact, which is exactly the cross-event violation
  //      §18.2 forbids and `bonds.html` B2 gates.
  //   2. It would have to be read in `bondForm1`, which already sits at the
  //      device's 8-storage-buffer cap (PLATFORM_NOTES), so any new buffer there
  //      is blocked outright.
  //   3. "Already welded" is not the same question as "same gesture": a particle
  //      welded to the group and then hard-broken away from it would still carry
  //      the id.
  // Latching keeps the invariant that makes all three go away — A TAG IS ALIVE
  // ONLY WHILE ITS GESTURE IS OPEN — and costs one uniform word and one branch.
  //
  // THE WRAP, RE-ARGUED (§18.2 W1's argument does not survive a latch as-is).
  // Live tags at any formation pass are (a) those issued since the previous
  // substep, unreachable at 4095/substep exactly as before, plus (b) AT MOST ONE
  // held tag. `_nextWeldSeq` SKIPS the held id, so a new spawn event can never
  // be handed the id an open gesture is holding: (a) and (b) are disjoint by
  // construction, not by counting. `bonds.html` B15 asserts the skip directly.
  // ════════════════════════════════════════════════════════════════════════

  /**
   * The next 12-bit spawn-event id, never 0, never the held one, and — §20 —
   * never one a LIVE RIGID BODY is still wearing.
   *
   * For a §18 bond the tag is transient: it is cleared on the formation pass
   * after the gesture and a collision would at worst weld two spawn events.
   * FOR A §20 BODY THE TAG IS THE BODY ID and it lives as long as the body does,
   * so handing the same id to a new gesture would make the new stroke's
   * particles read as members of the old body — `rg_same_body` would suppress
   * their contacts and `rigidMember` would try to place them from a record whose
   * rest offsets are somebody else's. That is the exact recycle hazard
   * SPEC19_DRAFT §19.B.3.3 identified for pivot bodies, and this is the fix:
   * ISSUED AND LIVE IDS ARE DISJOINT BY CONSTRUCTION, not by counting. The scan
   * is over at most RG_MAX_BODIES entries and runs once per gesture.
   */
  _nextWeldSeq() {
    const taken = (q) => {
      if (q === this._weldHoldSeq) return true;
      for (const b of this._bodies) if ((b.tag >>> WELD_TAG_SHIFT) === q) return true;
      return false;
    };
    let s = (this._weldSeq % WELD_SEQ_MAX) + 1;
    for (let guard = 0; guard <= WELD_SEQ_MAX && taken(s); guard++) s = (s % WELD_SEQ_MAX) + 1;
    this._weldSeq = s;
    return s;
  }

  _writeWeldHold() {
    // Word 12 of Params. Offset 48 is 4-aligned, so this is a 4-byte patch and
    // never a rewrite of the geometry fields beside it.
    this.device.queue.writeBuffer(this.buf.params, 48, new Uint32Array([this._weldHold >>> 0]));
  }

  /** §18.24. Params word 82 (byte 328, after the jet arrays and §24's drive
   *  words) — the word was `_padUsr0`, always 0, so the layout did not move. */
  _writeFuseTag() {
    this.device.queue.writeBuffer(this.buf.params, 328, new Uint32Array([this._fuseTag >>> 0]));
  }

  /**
   * Hold an EXISTING weld tag open. `beginWeldGroup` is the gesture-level way in;
   * this is the primitive, and it is what lets a test hold the tag `loadScene`
   * already stamped on a whole scene (`SCENE_WELD_TAG`).
   * @param {number} tag already shifted into WELD_TAG_MASK position
   */
  holdWeldGroup(tag) {
    const t = (tag >>> 0) & WELD_TAG_MASK;
    assert(t !== 0, 'holdWeldGroup(0): use endWeldGroup() to release');
    this._weldHold = t;
    this._weldHoldSeq = t >>> WELD_TAG_SHIFT;
    this._writeWeldHold();
    return t;
  }

  /**
   * Open a new weld group and hold it. Every `spawnFluid` until `endWeldGroup`
   * joins it, so a whole brush stroke is ONE body. Idempotent-safe: opening a
   * group while one is open closes the old one first, so a lost `pointerup`
   * costs one stroke and never leaks a hold across gestures.
   * @returns {number} the tag, already shifted
   */
  beginWeldGroup() {
    if (this._weldHold !== 0) this.endWeldGroup();
    // §18.24. A new gesture DISARMS any previous arm before it allocates, so
    // the armed tag can only ever be the current or most recent gesture's —
    // which is what makes a 12-bit tag-id wrap unable to resurrect a stale arm
    // (every intervening gesture cleared it long before the id could recycle).
    if (this._fuseTag !== 0) { this._fuseTag = 0; this._writeFuseTag(); }
    return this.holdWeldGroup(((this._nextWeldSeq() << WELD_TAG_SHIFT) & WELD_TAG_MASK) >>> 0);
  }

  /**
   * §18.24 — ARM DRAW-TIME FUSION for the gesture currently held open. Until
   * the formation pass after the gesture's release, new particles of this
   * gesture that land within BOND_R of EXISTING bonded-family matter (fuse_row:
   * bondK != 0, bondReform == 0, never PHOTO, not a §20 member) form
   * cross-object — and for the first time cross-MATERIAL — bonds: a RUBBER
   * band drawn across two anchored STONE poles is a slingshot. The arm is host
   * state exactly like the hold: a pure function of the call sequence, part of
   * the input stream, never of a clock. Emitters never call this; §31 arms one
   * whole-scene event while importing OE-CAKE's touching structures.
   */
  armFuse() {
    assert(this._weldHold !== 0, 'armFuse(): no weld group is open');
    this._fuseTag = this._weldHold;
    this._writeFuseTag();
    return this._fuseTag;
  }

  /** The tag whose fusion is armed, 0 if none (§18.24). */
  get fuseTag() { return this._fuseTag; }

  /**
   * §18.24 — MAY MATTER OF ROW `m` FUSE? The host-side twin of the shader's
   * `fuse_row`, plus the one check the shader cannot make: an interned §25 row
   * whose RECIPE base is PHOTO is refused here (the shader only knows row 15).
   * The UI consults this before arming a stroke.
   */
  canFuse(m) {
    const row = this.mats[m];
    if (!row) return false;
    const F = (f) => MAT_FIELDS.indexOf(f);
    if (row[F('phantom')] !== 0) return false;
    if (row[F('bondK')] === 0 || row[F('bondReform')] !== 0 || row[F('bondGroup')] < 0) return false;
    const base = this._internByRow.has(m) ? this._internByRow.get(m).base : m;
    return base !== MAT_PHOTO && m !== MAT_PHOTO;
  }

  /**
   * Close the open weld group. The next formation pass then clears the tag from
   * every particle wearing it, exactly as v8 did on the pass after a spawn — so
   * the stroke stops accepting new material and becomes an ordinary finished
   * body. Safe to call when nothing is open.
   */
  endWeldGroup() {
    if (this._weldHold === 0) return 0;
    const was = this._weldHold;
    this._weldHold = 0; this._weldHoldSeq = 0;
    this._writeWeldHold();
    // The pass that clears it must run promptly, not at the next BOND_PERIOD
    // boundary — otherwise a stroke released on substep 1 keeps accepting
    // material until substep 4.
    this._bondDirty = true;
    return was;
  }

  /** The tag currently held open, 0 if none. */
  get weldHold() { return this._weldHold; }

  /** §26.B — the weld tag the LAST `spawnFluid` batch was written with, already
   *  shifted into WELD_TAG_MASK position. 0 before any spawn. */
  get lastWeldTag() { return this._lastWeldTag | 0; }

  // ════════════════════════════════════════════════════════════════════════
  // §20 — MAKING A RIGID BODY. THE HOST SIDE.
  //
  // Everything here is AUTHORING, on the same side of the line as spawnFluid,
  // eraseWhere and applyImpulse: the determinism gate never calls any of them,
  // and the solver contract is "the same initial state plus the same ORDERED
  // list of (substepIndex, event) produces the same bytes".
  //
  // A body is created from a FINISHED GESTURE, never from a spawn batch. That
  // distinction is §18.22's whole lesson — `index.html` calls `spawnFluid` once
  // per pointer sample, so a 30-sample drag is 30 batches and one gesture — and
  // it is why `promoteWeldGroup` selects members by the WELD TAG the gesture
  // held open rather than by a slot range handed over by the caller. The slot
  // range in the record is a conservative WINDOW; the tag is the truth.
  // ════════════════════════════════════════════════════════════════════════

  /** Params word 13. Offset 52 is 4-aligned, so this is a 4-byte patch. */
  _writeNBodies() {
    this.device.queue.writeBuffer(this.buf.params, 52, new Uint32Array([this._nBodies >>> 0]));
  }

  // ════════════════════════════════════════════════════════════════════════
  // §23.B — JETS. AUTHORING, on the same side of the line as spawnFluid and
  // eraseWhere: the determinism gate never calls this, and a jet is a pure
  // function of the call sequence.
  //
  // WHY A DEVICE AND NOT A MATERIAL. Three reasons, and the third is the one
  // that decided it:
  //   1. 3 918 of the corpus's 4 182 Jet particles (93.7 %) sit on a Rigid or a
  //      Wall particle. Jet is not a substance anyone pours; it is a fitting
  //      bolted to something that does not move, or to something it pushes.
  //   2. A jet needs a DIRECTION. Direction is a property of the gesture that
  //      placed it, not of matter — we have no per-particle orientation and
  //      §11's storage register says we are not buying one for this.
  //   3. It costs nothing to be a device. A `Mat` row would need two new fields
  //      and would move `matsHash`; a per-particle field would need a word we
  //      do not have. A uniform costs one buffer resize and no binding, and the
  //      8-storage-buffer wall `friction` sits at is not touched at all.
  // ════════════════════════════════════════════════════════════════════════

  /**
   * REPLACE THE WHOLE JET LIST. Idempotent and cheap: an unchanged list writes
   * nothing, so a host that calls this every frame (which `index.html` does)
   * costs one string compare per frame rather than a uniform upload.
   *
   * @param {{x:number,y:number,r:number,fx:number,fy:number}[]} list
   *        TICKS for x / y / r; TICKS PER SUBSTEP SQUARED for fx / fy — the
   *        same units as `Mat.grav`. Integers; anything else is truncated here
   *        rather than in the shader, because a fractional uniform is a
   *        determinism hazard and this is the only place it can be caught.
   * @returns {number} how many jets are live (clamped to JET_MAX)
   */
  setJets(list) {
    const use = (list || []).slice(0, JET_MAX);
    // The key is the AUTHORITY on "did anything change", so it is built from
    // exactly the integers that reach the GPU — not from the objects, which may
    // carry host-only fields that must not trigger an upload.
    const q = use.map((j) => [j.x | 0, j.y | 0, j.r | 0, j.fx | 0, j.fy | 0]);
    const key = JSON.stringify(q);
    if (key === this._jetKey) return use.length;
    this._jetKey = key;
    this._jets = q;
    const pos = new Int32Array(JET_MAX * 4);
    const frc = new Int32Array(JET_MAX * 4);
    for (let k = 0; k < q.length; k++) {
      pos[k * 4] = q[k][0]; pos[k * 4 + 1] = q[k][1]; pos[k * 4 + 2] = q[k][2];
      frc[k * 4] = q[k][3]; frc[k * 4 + 1] = q[k][4];
    }
    // Order matters for nothing here — the arrays and the count go up in the
    // same submit — but the count is written LAST so a torn read can only ever
    // see fewer jets than are described, never more than are described.
    this.device.queue.writeBuffer(this.buf.params, JET_POS_OFF, pos);
    this.device.queue.writeBuffer(this.buf.params, JET_F_OFF, frc);
    this.device.queue.writeBuffer(this.buf.params, JET_N_OFF, new Uint32Array([q.length >>> 0]));
    return q.length;
  }

  /** The live jet list as the GPU has it: [x, y, r, fx, fy] per jet, in ticks. */
  get jets() { return (this._jets || []).map((j) => j.slice()); }
  get jetCount() { return (this._jets || []).length; }

  // ════════════════════════════════════════════════════════════════════════
  // §24 — USER: ARROW-KEY MATTER. Authoring, on the same side of the line as
  // spawnFluid, setJets and eraseWhere: the determinism gate never calls these
  // and the result is a pure function of the call sequence.
  //
  // THE CONTRACT, and it is the whole design constraint. KEY STATE IS AN INPUT.
  // It belongs in the replay record exactly as pointer events do, which means:
  //   * the HOST samples it, once per frame, at a fixed point in its own
  //     schedule — `index.html` does it in `loop()` beside `applyPointer()` and
  //     `serviceJets()`, BEFORE `step()`;
  //   * nothing in the sim path may read a clock, and nothing here does: the
  //     shader reads a uniform word the host wrote, exactly like `weldHold`;
  //   * a replay of the same (frame -> key state) sequence is byte-identical,
  //     because the sequence of `setUserDrive` calls IS the input.
  // The one thing a host must never do is derive the drive from elapsed time.
  // A frame that runs long does not get a bigger push; it gets the same push,
  // and that is what makes a replay a replay.
  // ════════════════════════════════════════════════════════════════════════

  /**
   * SET THE PLAYER'S DRIVE VECTOR. Idempotent and cheap: an unchanged vector
   * writes nothing, so a host that calls this every frame (which `index.html`
   * does) costs two integer compares rather than a uniform upload.
   *
   * @param {number} ax ticks per substep squared, the same units as `Mat.grav`
   * @param {number} ay  (+y is DOWN, as everywhere else in this engine)
   * @returns {number[]} the clamped, truncated pair the GPU now holds
   */
  setUserDrive(ax, ay) {
    const cl = (v) => Math.max(-USER_A_MAX, Math.min(USER_A_MAX, (v | 0)));
    const q = [cl(ax), cl(ay)];
    if (q[0] === this._userA[0] && q[1] === this._userA[1]) return this._userA.slice();
    this._userA = q;
    this.device.queue.writeBuffer(this.buf.params, USER_A_OFF, new Int32Array(q));
    return q.slice();
  }

  /** The drive vector as the GPU has it, in ticks/substep^2. */
  get userDrive() { return this._userA.slice(); }

  // ==========================================================================
  // §26.C — THE SCENE BOX: WORLD GRAVITY AND THE CONTAINER
  // --------------------------------------------------------------------------
  // Both of these are INPUTS, in exactly the sense §24 established for the
  // arrow keys: an authoring call the host makes at a fixed point in its own
  // schedule, never a clock, never a frame-rate-dependent quantity. A replay
  // that records (substepIndex, event) already has to record `spawnFluid` and
  // `setJets`; these join that list and nothing else changes.
  //
  // NEITHER IS A SHADER CHANGE. That is the whole design and it is why §26
  // does not bump the solver at all:
  //
  //   · GRAVITY re-uploads the `grav` COLUMN of the material image. The shader
  //     reads `mats[m].grav` in the three places it always did (`predict`, the
  //     axis branch, and `rigidRecount`'s gSum) and cannot tell the difference.
  //     At the default multiplier the write is skipped entirely — and even if
  //     it were not, `idivRne(g * ONE, ONE) == g` exactly for every row, so the
  //     bytes would be identical. The determinism gate never calls it.
  //
  //   · THE CONTAINER is dropped by moving `n` back to `nFluid`. The boundary
  //     block lives at [nFluid, n) by the invariant `spawnFluid` maintains, so
  //     "no walls" is a smaller dispatch grid and nothing else — no flag, no
  //     branch, no material row. The bytes stay in the buffer past `n` and are
  //     rewritten from the scene when the walls come back.
  // ==========================================================================

  /**
   * Scale the world's gravity. `q` is Q16: ONE is the table as authored, 0 is
   * weightlessness, 2*ONE is twice as heavy. Buoyancy scales with it, which is
   * correct — a gas rises because it is lighter than what is around it, so with
   * no gravity nothing rises either.
   *
   * @returns {number} the multiplier the GPU now holds
   */
  setGravityScale(q) {
    const qq = Math.max(0, Math.min(4 * ONE, Math.round(q)));
    if (qq === this._gravQ) return this._gravQ;
    this._gravQ = qq;
    const iGrav = MAT_FIELDS.indexOf('grav');
    const w = this._matWords.slice();
    for (let m = 0; m < this.matCount; m++) {
      w[m * MAT_WORDS + iGrav] = idivRne(this.mats[m][iGrav] * qq, ONE);
    }
    this.device.queue.writeBuffer(this.buf.mats, 0, w);
    return this._gravQ;
  }

  /** The world gravity multiplier the GPU currently holds, Q16. */
  get gravityScale() { return this._gravQ; }

  /**
   * The `grav` material row `m` currently has ON THE GPU. Identical to the
   * authored value at the default multiplier, and it is what §20's host-side
   * `gSum` seed must use so that a body welded under half gravity is born with
   * the same weight the shader will recount for it on the next substep.
   */
  gravOf(m) {
    const g = this.mats[m][MAT_FIELDS.indexOf('grav')];
    return this._gravQ === ONE ? g : idivRne(g * this._gravQ, ONE);
  }

  // ==========================================================================
  // §25 (v16) — THE INTERN CACHE: mint / recycle / project, and the record
  // --------------------------------------------------------------------------
  // `synth()` (module level) is the pure half; this is the impure half that
  // owns rowIds, refcounts, the mats tail and the GPU upload. The policy is
  // §25.1d verbatim: hit -> mint -> recycle-by-zero-refcount -> PROJECT onto
  // the nearest live interned row, and a stroke is NEVER refused mid-gesture.
  // Every outcome is an EVENT in `_internEvents` — the ordered record a
  // replay embeds (mint order is load-bearing: rowId assignment is part of
  // the record, and the properties gate asserts swapping two mints moves the
  // chain).
  // ==========================================================================

  /** The number of AUTHORED rows; interned rowIds live in [this, matCount). */
  get authoredCount() { return this._authoredCount; }

  /** The live interned rows, ordered by rowId. */
  internedRows() {
    return [...this._internByRow.values()].sort((a, b) => a.rowId - b.rowId);
  }

  /** RUNG 2.6 — the live intern entry for ONE row id, or null. O(1); the
   *  renderer reads it per frame to derive a composed row's look (base look
   *  pulled toward the composeTint word), so a recycled id re-derives with no
   *  cache to invalidate. Read-only metadata: nothing render-side can write. */
  internEntry(rowId) { return this._internByRow.get(rowId) ?? null; }

  /** §30 — ingredient set carried by one material/particle. Explicit imported
   * recipe metadata wins; otherwise authored base + composed modifier bits are
   * the honest reconstruction. Structural W/R ingredients live on flags. */
  materialElementMask(rowId, flags = 0) {
    const e = this._internByRow.get(rowId);
    let mask = e?.recipeMask !== undefined
      ? e.recipeMask >>> 0
      : authoredElementMask(e ? e.base : rowId) |
        elementMask(e ? modLetters(e.bits).toUpperCase() : '');
    if ((flags & FLAG_ANCHOR) !== 0) mask |= ELEMENT_BITS.W;
    if ((flags & FLAG_RIGID) !== 0) mask |= ELEMENT_BITS.R;
    return mask >>> 0;
  }

  /** Whether a row is simulated Inflow/Outflow matter, and the exact recipe
   * its filtering rule sees. Null for ordinary matter. */
  matterIOInfo(rowId, flags = 0) {
    const e = this._internByRow.get(rowId);
    if (!e || (e.bits & (MOD_BITS.I | MOD_BITS.O)) === 0) return null;
    return {
      inflow: (e.bits & MOD_BITS.I) !== 0,
      outflow: (e.bits & MOD_BITS.O) !== 0,
      ingredients: this.materialElementMask(rowId, flags),
      base: e.base, bits: e.bits, recipeMask: e.recipeMask ?? null,
    };
  }

  /** Cheap UI cadence guard over the existing host mirror. A null/missing
   * mirror intentionally reports zero: the next refresh will arm the service,
   * while an ordinary scene never pays a synchronising read just because an
   * I/O row was minted earlier in the session. */
  matterIOCount() {
    const mir = this._mirror;
    if (!mir) return 0;
    const u = new Uint32Array(mir.buffer, mir.byteOffset, mir.length);
    const nf = Math.min(this.nFluid, (mir.length / PARTICLE_WORDS) | 0);
    let n = 0;
    for (let i = 0; i < nf; i++) {
      const e = this._internByRow.get(u[i * PARTICLE_WORDS + 4]);
      if (e && (e.bits & (MOD_BITS.I | MOD_BITS.O))) n++;
    }
    return n;
  }

  /** The ordered §25.2 event log (MINT / RECYCLE / PROJECT), oldest first. */
  internEvents() { return this._internEvents.slice(); }

  /**
   * FNV-1a over the ordered event log — the interned-row counterpart of
   * `matsHash`, covering exactly what matsHash must not: scene-state rows.
   * Format `<liveCount>:<hex8>`; an engine that never minted reads `0:811c9dc5`
   * (the FNV offset basis — nothing folded).
   */
  sceneRowsDigest() {
    let h = 0x811c9dc5 >>> 0;
    const fold = (v) => {
      const x = v | 0;
      for (let b = 0; b < 4; b++) {
        h = (h ^ ((x >>> (b * 8)) & 255)) >>> 0;
        h = Math.imul(h, 0x01000193) >>> 0;
      }
    };
    for (const ev of this._internEvents) {
      fold(ev.type === 'MINT' ? 1 : ev.type === 'RECYCLE' ? 2 : 3);
      fold(ev.substep ?? 0); fold(ev.rowId ?? -1); fold(ev.base ?? -1); fold(ev.bits ?? 0);
      fold(ev.synthVersion ?? 0);
      if (ev.recipeMask !== undefined) fold(ev.recipeMask);
      if (ev.words) for (const w of ev.words) fold(w);
    }
    return `${this._internByRow.size}:${h.toString(16).padStart(8, '0')}`;
  }

  /**
   * Intern a composed material: returns the rowId to paint with, minting /
   * recycling / projecting as the ladder requires. `opts.pinned` is an
   * iterable of rowIds the HOST knows are still referenced outside the sim
   * (a device emitting the row, a pending replay event) — the engine cannot
   * see those and must not recycle them.
   *
   * Throws only on a row that refuses to EXIST (synth's self-destruction
   * contract) — never because the cache is full.
   *
   * @returns {{ rowId: number, kind: 'hit'|'mint'|'recycle'|'project', events: object[] }}
   */
  internMaterial(baseId, bits, opts = {}) {
    const pinned = new Set(opts.pinned ?? []);
    const before = this._internEvents.length;
    const recipeMask = opts.recipeMask === undefined ? null : (opts.recipeMask >>> 0);
    const rowId = this._internResolve(baseId | 0, bits | 0, 1, pinned, recipeMask);
    const events = this._internEvents.slice(before);
    const kind = events.length === 0 ? 'hit'
      : events.some((e) => e.type === 'PROJECT') ? 'project'
      : events.some((e) => e.type === 'RECYCLE') ? 'recycle' : 'mint';
    return { rowId, kind, events };
  }

  _internKey(base, bits, recipeMask = null) {
    return base + ':' + bits + (recipeMask === null ? '' : ':' + (recipeMask >>> 0));
  }

  _internResolve(base, bits, depth, pinned, recipeMask = null) {
    const key = this._internKey(base, bits, recipeMask);
    const hit = this._internMap.get(key);
    if (hit !== undefined) {
      this._internByRow.get(hit).tick = ++this._internTick;
      return hit;
    }
    const { words, needs, dropped, identity } = synth(base, bits, this.mats.slice(0, this._authoredCount));
    // Exact cancellation (D+L and friends): the composed row IS the base row.
    // No mint, no event, no cache slot — the brush paints the base.
    if (identity) return base;
    // ---- products closure, DEPTH CAPPED AT 2 (§25.4) -----------------------
    for (const need of needs) {
      const f = MAT_FIELDS.indexOf(need.field);
      if (depth + 1 > 2) {
        // Beyond the cap the product PROJECTS to the base row's own product
        // (already in `words`) and the record says so — the honest projection
        // rule applied to closure instead of capacity.
        this._internEvents.push({ type: 'PROJECT', substep: this._substep, base: need.base,
          bits: need.bits, rowId: words[f], reason: `products-closure depth > 2; ` +
          `${need.field} projects to the base row's product` });
        continue;
      }
      words[f] = this._internResolve(need.base, need.bits, depth + 1, pinned);
    }
    // ---- slot: free tail? --------------------------------------------------
    if (this.matCount < MAT_MAX) {
      return this._internMint(this.matCount, base, bits, words, dropped, recipeMask);
    }
    // ---- recycle the LRU zero-refcount row ---------------------------------
    let victim = null;
    for (const e of this._internByRow.values()) {
      if (!this._internRecyclable(e.rowId, pinned)) continue;
      if (!victim || e.tick < victim.tick || (e.tick === victim.tick && e.rowId < victim.rowId)) victim = e;
    }
    if (victim) {
      const recycle = { type: 'RECYCLE', substep: this._substep, rowId: victim.rowId,
        base: victim.base, bits: victim.bits };
      if (victim.recipeMask !== undefined) recycle.recipeMask = victim.recipeMask;
      this._internEvents.push(recycle);
      this._internMap.delete(this._internKey(victim.base, victim.bits, victim.recipeMask ?? null));
      this._internByRow.delete(victim.rowId);
      this._internSpawned.delete(victim.rowId);
      return this._internMint(victim.rowId, base, bits, words, dropped, recipeMask);
    }
    // ---- PROJECT onto the nearest live interned row (never refuse) ---------
    const target = this._internProject(words, bits, recipeMask);
    if (target < 0 && (bits & (MOD_BITS.I | MOD_BITS.O))) {
      throw new Error('[aether] internMaterial: the composed-material cache is full and no ' +
        'row with the same Inflow/Outflow recipe can be projected without changing behaviour');
    }
    const project = { type: 'PROJECT', substep: this._substep, base, bits,
      rowId: target, reason: 'intern cache full and nothing recyclable' };
    if (recipeMask !== null) project.recipeMask = recipeMask >>> 0;
    this._internEvents.push(project);
    return target;
  }

  /** A row is recyclable when NOTHING can still reach it (§25.1d step 3). */
  _internRecyclable(rowId, pinned) {
    if (pinned.has(rowId)) return false;
    // Referenced as a PRODUCT by any other live row: a particle could still
    // transition INTO it (composed Snow's meltTo names snow-water), so a
    // product target is live for as long as its parent is.
    const PF = ['meltTo', 'freezeTo', 'boilTo', 'condTo', 'burnTo', 'splitTo', 'corrTo']
      .map((f) => MAT_FIELDS.indexOf(f));
    for (const e of this._internByRow.values()) {
      if (e.rowId === rowId) continue;
      for (const f of PF) if (e.words[f] === rowId) return false;
    }
    // Wearers. The host counts spawns; erases and transitions it cannot see,
    // so a row that was EVER worn needs the position mirror to prove it is
    // bare — and only a mirror that provably contains every spawn (`_spawnSeq`
    // gate, see refreshMirror) is evidence. Stale mirror => assume live.
    const everWorn = (this._internSpawned.get(rowId) ?? 0) > 0;
    if (!everWorn) return true;
    if (!this._mirror || this._mirrorSeq !== this._spawnSeq) return false;
    const mir = this._mirror, mu = new Uint32Array(mir.buffer, mir.byteOffset, mir.length);
    const nn = Math.min(this.n, (mir.length / PARTICLE_WORDS) | 0);
    for (let i = 0; i < nn; i++) if (mu[i * PARTICLE_WORDS + 4] === rowId) return false;
    this._internSpawned.set(rowId, 0);
    return true;
  }

  /** Deterministic nearest-live-row distance over the synth axes (§22.4 style). */
  _internProject(words, bits = 0, recipeMask = null) {
    const AXES = ['xsph', 'gammaCoh', 'gammaCur', 'adhesion', 'grav', 'fric',
                  'bondK', 'bondBreak', 'bondReform', 'solidMode', 'tempTarget', 'tempDrive',
                  'burstP', 'sealMode']
      .map((f) => MAT_FIELDS.indexOf(f));
    let best = -1, bestD = Infinity;
    for (const e of this._internByRow.values()) {
      // I/O is behaviour, not merely a physical approximation: projection may
      // change mechanics, but may never turn a source/sink into ordinary matter
      // or change the ingredient-sharing filter it carries.
      if ((bits & (MOD_BITS.I | MOD_BITS.O)) !== 0) {
        if ((e.bits & (MOD_BITS.I | MOD_BITS.O)) !== (bits & (MOD_BITS.I | MOD_BITS.O))) continue;
        if ((e.recipeMask ?? null) !== recipeMask) continue;
      }
      let d = 0;
      for (const f of AXES) d += Math.abs(words[f] - e.words[f]);
      if (d < bestD || (d === bestD && e.rowId < best)) { best = e.rowId; bestD = d; }
    }
    // The cache cannot be full AND empty; but if a caller reaches here with no
    // interned rows at all, the honest fallback is the authored base the words
    // were built from — nearest by the same metric over authored rows.
    if (best < 0 && (bits & (MOD_BITS.I | MOD_BITS.O)) !== 0) return -1;
    if (best < 0) {
      for (let m = 0; m < this._authoredCount; m++) {
        if (isPhantom(m, this.mats)) continue;
        let d = 0;
        for (const f of AXES) d += Math.abs(words[f] - this.mats[m][f]);
        if (d < bestD) { best = m; bestD = d; }
      }
    }
    return best;
  }

  _internMint(rowId, base, bits, words, dropped, recipeMask = null) {
    const F = (f) => MAT_FIELDS.indexOf(f);
    // §25.3: the fresh unique group. 64 + rowId collides with nothing authored
    // ([-1, 63]) and with no other interned row by construction.
    if (words[F('bondGroup')] === -2) words[F('bondGroup')] = 64 + rowId;
    this._internValidate(words, rowId);
    this.mats[rowId] = words;
    if (rowId === this.matCount) this.matCount++;
    const entry = { rowId, base, bits, synthVersion: SYNTH_VERSION, words,
      tick: ++this._internTick, name: `${MAT_NAMES[base] ?? base}+${modLetters(bits)}`,
      dropped };
    if (recipeMask !== null) entry.recipeMask = recipeMask >>> 0;
    this._internMap.set(this._internKey(base, bits, recipeMask), rowId);
    this._internByRow.set(rowId, entry);
    const event = { type: 'MINT', substep: this._substep, rowId, base, bits,
      synthVersion: SYNTH_VERSION, words: words.slice(), dropped: dropped.slice() };
    if (recipeMask !== null) event.recipeMask = recipeMask >>> 0;
    this._internEvents.push(event);
    this._internUpload(rowId);
    return rowId;
  }

  /** The subset of createEngine's row contract a minted row must satisfy. */
  _internValidate(words, rowId) {
    const F = (f) => MAT_FIELDS.indexOf(f);
    const bad = (msg) => { throw new Error(`[aether] intern row ${rowId}: ${msg}`); };
    for (let w = 0; w < MAT_WORDS; w++) {
      if (!Number.isInteger(words[w]) || words[w] < -2147483648 || words[w] > 2147483647) {
        bad(`${MAT_FIELDS[w]} = ${words[w]} is not an i32`);
      }
    }
    if (words[F('eps')] < 32768) bad(`eps ${words[F('eps')]} < 32768 (§2.3)`);
    if (words[F('dpMax')] <= 0) bad('dpMax <= 0');
    if (words[F('xsph')] < 0 || words[F('xsph')] > XSPH_MAX) bad(`xsph ${words[F('xsph')]}`);
    if (words[F('cond')] < 0 || words[F('cond')] > COND_MAX) bad(`cond ${words[F('cond')]}`);
    if (words[F('coolRate')] < 0 || words[F('coolRate')] > 65536) bad('coolRate range');
    if (words[F('tempDrive')] < 0 || words[F('tempDrive')] > 65536) bad('tempDrive range');
    if (words[F('burstP')] < 0) bad('burstP < 0');
    if (words[F('sealMode')] !== 0 && words[F('sealMode')] !== 1) bad('sealMode must be 0 or 1');
    if (words[F('bondK')] !== 0 && words[F('bondGroup')] !== 64 + rowId) {
      bad(`bondGroup ${words[F('bondGroup')]} != ${64 + rowId} (the F1 fresh-group rule)`);
    }
    if (words[F('bondK')] < 0 || words[F('bondK')] > BOND_K_MAX) bad('bondK range');
    if (words[F('rampHi')] - words[F('rampLo')] < 65536) bad('ramp narrower than 1 degree');
    for (const pf of ['meltTo', 'freezeTo', 'boilTo', 'condTo', 'burnTo', 'splitTo', 'corrTo']) {
      const t = words[F(pf)];
      if (!(Number.isInteger(t) && t >= 0 && (t < this.matCount || t === rowId))) {
        bad(`${pf} = ${t} is not a live material id`);
      }
    }
  }

  /** Write ONE interned row into the baked image and the GPU (192 B). */
  _internUpload(rowId) {
    const F = (f) => MAT_FIELDS.indexOf(f);
    const words = this.mats[rowId];
    const off = rowId * MAT_WORDS;
    for (let w = 0; w < MAT_WORDS; w++) this._matWords[off + w] = words[w];
    // rampInv is BAKED here exactly as createEngine bakes authored rows.
    const span = words[F('rampHi')] - words[F('rampLo')];
    this._matWords[off + F('rampInv')] = span > 0 ? Math.floor(2 ** 32 / span) : 0;
    const up = new Int32Array(MAT_WORDS);
    up.set(this._matWords.subarray(off, off + MAT_WORDS));
    // A mint under a scaled world uploads the SCALED grav, exactly as
    // setGravityScale would have; `_matWords` keeps the unscaled truth.
    if (this._gravQ !== ONE) up[F('grav')] = idivRne(words[F('grav')] * this._gravQ, ONE);
    this.device.queue.writeBuffer(this.buf.mats, off * 4, up);
  }

  /**
   * §25.2 — THE REPLAY / LOAD PATH. Apply a recorded mint list: re-derive
   * every MINT at its recorded SYNTH_VERSION and REFUSE the load on any
   * word disagreeing — belt and braces, never a silent divergence.
   */
  applyMintEvents(list) {
    for (const ev of list ?? []) {
      if (ev.type === 'RECYCLE') {
        const e = this._internByRow.get(ev.rowId);
        if (!e || e.base !== ev.base || e.bits !== ev.bits ||
            (e.recipeMask ?? null) !== (ev.recipeMask ?? null)) {
          throw new Error(`[aether] applyMintEvents: RECYCLE of rowId ${ev.rowId} does not match ` +
            `the live entry — the record and the cache disagree; refusing the load`);
        }
        this._internEvents.push({ ...ev });
        this._internMap.delete(this._internKey(ev.base, ev.bits, ev.recipeMask ?? null));
        this._internByRow.delete(ev.rowId);
        this._internSpawned.delete(ev.rowId);
        continue;
      }
      if (ev.type === 'PROJECT') { this._internEvents.push({ ...ev }); continue; }
      if (ev.type !== 'MINT') throw new Error(`[aether] applyMintEvents: unknown event ${ev.type}`);
      const { words, needs } = synth(ev.base, ev.bits,
        this.mats.slice(0, this._authoredCount), ev.synthVersion);
      const F = (f) => MAT_FIELDS.indexOf(f);
      for (const need of needs) {
        // The child must already be live from an earlier event in this list.
        const child = this._internMap.get(this._internKey(need.base, need.bits));
        words[F(need.field)] = child !== undefined ? child : ev.words[F(need.field)];
      }
      if (words[F('bondGroup')] === -2) words[F('bondGroup')] = 64 + ev.rowId;
      for (let w = 0; w < MAT_WORDS; w++) {
        if (words[w] !== ev.words[w]) {
          throw new Error(`[aether] applyMintEvents REFUSED: rowId ${ev.rowId} ` +
            `(${MAT_NAMES[ev.base]}+${modLetters(ev.bits)}) re-derives ${MAT_FIELDS[w]} = ` +
            `${words[w]} at SYNTH_VERSION ${ev.synthVersion} but the record embeds ${ev.words[w]} — ` +
            `the header and this build disagree about the algebra; refusing the load`);
        }
      }
      if (this._internByRow.has(ev.rowId)) {
        throw new Error(`[aether] applyMintEvents: rowId ${ev.rowId} minted twice without a RECYCLE`);
      }
      this._internMint(ev.rowId, ev.base, ev.bits, words, ev.dropped ?? [],
        ev.recipeMask ?? null);
      // _internMint logged a fresh event with THIS engine's substep; rewrite it
      // to the recorded one so the digest reproduces the source engine's.
      this._internEvents[this._internEvents.length - 1].substep = ev.substep ?? 0;
    }
  }

  /** Drop every interned row (a new scene). The tail is zeroed like boot. */
  clearInterned() {
    if (this._internByRow.size === 0 && this._internEvents.length === 0) return;
    const first = this._authoredCount * MAT_WORDS;
    this._matWords.fill(0, first);
    this.device.queue.writeBuffer(this.buf.mats, first * 4,
      this._matWords.subarray(first));
    this.mats.length = this._authoredCount;
    this.matCount = this._authoredCount;
    this._internMap.clear(); this._internByRow.clear(); this._internSpawned.clear();
    this._internEvents.length = 0;
  }

  /** Is the scene's boundary container in the world? */
  get wallsOn() { return this._wallsOn; }

  /**
   * §26.C — TAKE THE CONTAINER AWAY, or put it back. Darien: "remove the
   * boundary container so matter falls out of the world."
   *
   * Removing it does NOT by itself let matter leave: `wall_clamp` pins every
   * predicted position inside the domain, by design and forever (it is what
   * bounds every accumulator the spec proves). So the container going away is
   * half of the feature and the host's own erase sweep is the other half — see
   * index.html's drain margin. This method owns the half that is state.
   *
   * @returns {boolean} whether the walls are on afterwards
   */
  setWalls(on) {
    const want = !!on;
    if (want === this._wallsOn) return this._wallsOn;
    const sc = this._scene;
    if (!sc) return this._wallsOn;
    const nB = sc.n - sc.nFluid;
    if (nB === 0) { this._wallsOn = want; return this._wallsOn; }
    if (!want) {
      this.n = this.nFluid;
    } else {
      if (this.nFluid + nB > this.maxParticles) return this._wallsOn;   // no room
      // Rewrite the boundary block from the SCENE, exactly as `spawnFluid`
      // does when it shifts it: the slots may since have been overwritten by
      // fluid, so they cannot be assumed to still hold their old bytes.
      const stateBuf = this.parity === 0 ? this.buf.stateA : this.buf.stateB;
      const iFuel0 = MAT_FIELDS.indexOf('fuel0');
      const st = new Int32Array(nB * PARTICLE_WORDS);
      const stu = new Uint32Array(st.buffer);
      for (let b = 0; b < nB; b++) {
        const o = b * PARTICLE_WORDS, src = sc.nFluid + b;
        const bm = sc.matId[src] >>> 0;
        st[o + 0] = sc.pos[src * 2]; st[o + 1] = sc.pos[src * 2 + 1];
        stu[o + 4] = bm;
        st[o + 6] = sc.temp ? sc.temp[src] : AMBIENT;
        st[o + 7] = sc.fuel ? sc.fuel[src] : this.mats[bm][iFuel0];
      }
      this.device.queue.writeBuffer(stateBuf, this.nFluid * 32, st);
      const dv = new Int32Array(nB * DERIVED_WORDS);
      for (let b = 0; b < nB; b++) {
        dv[b * DERIVED_WORDS + 0] = st[b * PARTICLE_WORDS + 0];
        dv[b * DERIVED_WORDS + 1] = st[b * PARTICLE_WORDS + 1];
      }
      this.device.queue.writeBuffer(this.buf.derived, this.nFluid * DERIVED_BYTES, dv);
      // §21. A boundary particle has no colour, and the slot may have been a
      // coloured import a moment ago. Zero it rather than inherit a pigment.
      this.device.queue.writeBuffer(this.buf.tint, this.nFluid * 4, new Uint32Array(nB));
      this.n = this.nFluid + nB;
    }
    this._wallsOn = want;
    this.device.queue.writeBuffer(this.buf.params, 0, new Uint32Array([this.n, this.nFluid]));
    this.device.queue.writeBuffer(this.buf.hashParams, 0,
      new Uint32Array([this.n, PARTICLE_WORDS, HASHED_WORDS, this.nFluid]));
    return this._wallsOn;
  }
  /** How many live bodies the player is driving. Host mirror; no GPU read. */
  get userBodyCount() { return this._bodies.filter((b) => b.user).length; }

  /**
   * FLAG (or unflag) ONE LIVE BODY as the player's. `index.html` sets it at
   * promotion time through `promoteWeldGroup`'s third argument; this exists so
   * a gate can run the SAME body with the flag on and off — a negative control
   * whose staging is byte-identical, which is the only kind worth having.
   *
   * @param {number} index the body's record index (`body.index`)
   * @param {boolean} on
   * @returns {boolean} whether a live body was addressed
   */
  setBodyUser(index, on) {
    const b = this._bodies.find((z) => z.index === index);
    if (!b) return false;
    b.user = !!on;
    this.device.queue.writeBuffer(
      this.buf.rigid, (RG_BODY_BASE + index * RG_BODY_WORDS + RB.USER) * 4,
      new Int32Array([on ? 1 : 0]));
    return true;
  }

  /** The live bodies, host-side: `{ tag, first, last, n, mat }`, newest last. */
  get bodies() { return this._bodies.slice(); }
  get bodyCount() { return this._nBodies; }
  get bodyDriverCount() { return this._bodyDrivers.size; }

  /**
   * Attach one pure imported-module evaluator to a live body.
   *
   * The callback receives only deterministic scene inputs:
   * `{substep,time,timeStep,usersX,usersY}`. It returns
   * `{mask, values:[positionX,...,torque]}` with values already quantised to
   * the engine units named by BODY_DRIVER. The callback is run once before
   * every encoded substep; no wall clock reaches it.
   */
  setBodyConstraint(index, evaluate, spec = null) {
    assert(Number.isInteger(index) && index >= 0 && index < this._nBodies,
      `body constraint index ${index} is not live`);
    assert(typeof evaluate === 'function', 'body constraint evaluator must be a function');
    this._bodyDrivers.set(index, { evaluate, spec });
    return true;
  }

  clearBodyConstraint(index) {
    if (!this._bodyDrivers.delete(index)) return false;
    this.device.queue.writeBuffer(
      this.buf.rigid,
      (RG_DRIVER_BASE + index * RG_DRIVER_WORDS) * 4,
      new Int32Array(RG_DRIVER_WORDS));
    return true;
  }

  clearBodyConstraints() {
    if (this._bodyDrivers.size === 0) return 0;
    const n = this._bodyDrivers.size;
    this._bodyDrivers.clear();
    const enc = this.device.createCommandEncoder({ label: 'aether:clearBodyConstraints' });
    enc.clearBuffer(this.buf.rigid, RG_DRIVER_BASE * 4, RG_MAX_BODIES * RG_DRIVER_WORDS * 4);
    this.device.queue.submit([enc.finish()]);
    return n;
  }

  _serviceBodyConstraints(encoder, batchSlot) {
    if (this._bodyDrivers.size === 0) return;
    assert(encoder && Number.isInteger(batchSlot) &&
      batchSlot >= 0 && batchSlot < this.submitBatch,
      `constraint upload batch slot ${batchSlot} outside 0..${this.submitBatch - 1}`);
    const context = Object.freeze({
      substep: this._substep,
      timeStep: this._substep,
      time: this._substep / 120,
      usersX: Math.sign(this._userA[0]),
      usersY: Math.sign(this._userA[1]),
    });
    const dead = [];
    for (const [index, driver] of [...this._bodyDrivers.entries()].sort((a, b) => a[0] - b[0])) {
      if (index < 0 || index >= this._nBodies) { dead.push(index); continue; }
      const result = driver.evaluate(context);
      const mask = result?.mask;
      const values = result?.values;
      assert(Number.isInteger(mask) && mask >= 0 && mask < (1 << 9),
        `body ${index} constraint returned invalid mask ${mask}`);
      assert(Array.isArray(values) && values.length === 9,
        `body ${index} constraint returned ${values?.length ?? 'no'} values; expected 9`);
      const rec = new Int32Array(RG_DRIVER_WORDS);
      rec[BODY_DRIVER.MASK] = mask;
      for (let i = 0; i < 9; i++) {
        const value = Number(values[i] ?? 0);
        assert(Number.isFinite(value), `body ${index} constraint slot ${i} is not finite`);
        rec[i + 1] = Math.max(-2147483648, Math.min(2147483647, Math.round(value)));
      }
      const stageWord = batchSlot * RG_MAX_BODIES * RG_DRIVER_WORDS +
        index * RG_DRIVER_WORDS;
      this.device.queue.writeBuffer(
        this.buf.driverUpload,
        stageWord * 4,
        rec);
      encoder.copyBufferToBuffer(
        this.buf.driverUpload,
        stageWord * 4,
        this.buf.rigid,
        (RG_DRIVER_BASE + index * RG_DRIVER_WORDS) * 4,
        RG_DRIVER_WORDS * 4);
    }
    for (const index of dead) this._bodyDrivers.delete(index);
  }

  /** Rung 4 driver sidecars, straight out of the buffer. Diagnostics/gates. */
  async readBodyDrivers() {
    const w = await this._read(
      this.buf.rigid,
      RG_MAX_BODIES * RG_DRIVER_WORDS * 4,
      RG_DRIVER_BASE * 4);
    return new Int32Array(w);
  }

  /** JSON-safe authored driver descriptions for Undo / native `.suna`. */
  get bodyConstraintSpecs() {
    return [...this._bodyDrivers.entries()]
      .filter(([, driver]) => driver.spec)
      .sort((a, b) => a[0] - b[0])
      .map(([index, driver]) => ({
        index,
        spec: JSON.parse(JSON.stringify(driver.spec)),
      }));
  }

  /**
   * Build ONE body record from an explicit member slot list and a state array.
   * Pure host integer math — `rigidShifts` / `rigidInvInertia` — plus two
   * writeBuffers. Returns the record descriptor, or null if the body is too
   * small to exist (`RIGID_MIN_MEMBERS`; n = 1 has invI = 0 and cannot rotate).
   *
   * THE REST FRAME IS CAPTURED IN THE WORLD FRAME WITH theta = 0, which is exact
   * by construction: a promoted body's rest shape IS its current shape, so there
   * is no fit to perform and no capture error to carry. (`fitRotation` — the
   * closed-form polar decomposition — is for the future case of expressing a
   * current configuration against a PRE-EXISTING rest shape. §20 never needs it,
   * and a split child must set theta = 0 for the same reason: carrying the
   * parent's theta while capturing world-frame offsets rotates them twice.)
   *
   * The body inherits the members' LINEAR AND ANGULAR MOMENTUM, so promoting a
   * stroke that was already moving does not stop it dead.
   */
  _writeBodyRecord(members, st, tag, pivot = null, user = false) {
    const N = members.length;
    if (N < RIGID_MIN_MEMBERS) return null;
    if (this._nBodies >= RG_MAX_BODIES) return null;
    const PW = PARTICLE_WORDS;
    const stu = new Uint32Array(st.buffer, st.byteOffset, st.length);

    // centroid, round-half-to-even so a symmetric shape lands on its own centre
    let sx = 0, sy = 0, svx = 0, svy = 0;
    for (const i of members) { sx += st[i * PW]; sy += st[i * PW + 1]; svx += st[i * PW + 2]; svy += st[i * PW + 3]; }
    // §19.B. AN AXIS BODY'S FRAME ORIGIN IS THE POINT THE USER PRESSED, not the
    // centroid — which is the single decision the whole feature rests on. Rest
    // offsets measured from it make RB_SUMIS2 the moment of inertia ABOUT THE
    // PIVOT with no parallel-axis term, RB_RMAX the true rim radius (so the
    // rim-speed cap is right), and `rigidMember`'s recount correct for every
    // member set the wheel can burn down to. It also starts at rest: a wheel is
    // hung on its axle, it is not thrown.
    const axis = pivot !== null;
    const cx = axis ? pivot.x | 0 : idivRne(sx, N);
    const cy = axis ? pivot.y | 0 : idivRne(sy, N);
    const vx = axis ? 0 : idivRne(svx, N);
    const vy = axis ? 0 : idivRne(svy, N);

    const ox = new Int32Array(N), oy = new Int32Array(N);
    for (let k = 0; k < N; k++) {
      ox[k] = st[members[k] * PW] - cx;
      oy[k] = st[members[k] * PW + 1] - cy;
    }
    const { rMax, rsh, ish, osh } = rigidShifts(ox, oy);
    let sumIs2 = 0, gSum = 0;
    for (let k = 0; k < N; k++) {
      const a = ox[k] >> ish, b = oy[k] >> ish;
      sumIs2 += a * a + b * b;
      // §26.C. The GPU's `grav`, not the authored one, so a body welded while
      // the world gravity slider is off is born weightless instead of spending
      // one substep heavy before `rigidRecount` corrects it. Identical to
      // `this.mats[...][iGrav]` at the default multiplier.
      gSum += this.gravOf(stu[members[k] * PW + 4]);
    }
    const { invI, iq } = rigidInvInertia(sumIs2, rsh, ish);
    // Angular momentum of the members about the new centre, expressed in the
    // SAME scaled arithmetic the shader's torque fold uses, so `omega` means the
    // same thing on substep 0 as it does on substep 1.
    let acc = 0;
    for (let k = 0; k < N; k++) {
      const i = members[k];
      acc += (ox[k] >> rsh) * (st[i * PW + 3] - vy) - (oy[k] >> rsh) * (st[i * PW + 2] - vx);
    }
    // RNE in BigInt: acc*invI reaches 2^52, where a float64 divide would start
    // dropping bits — and a body's opening angular rate must be a pure function
    // of the state it was welded from, not of how close that product got to the
    // mantissa. Same round-half-to-even the shader's shift performs.
    let omega = 0;
    if (invI !== 0) {
      const num = BigInt(acc) * BigInt(invI), den = 1n << BigInt(iq);
      let q = num / den, r = num - q * den;
      if (r < 0n) { q -= 1n; r += den; }                       // BigInt / truncates
      if (2n * r > den || (2n * r === den && (q & 1n) === 1n)) q += 1n;
      omega = Number(q);
    }
    // Rim-speed cap: no member may be asked to travel further than VMAX in one
    // substep by rotation alone. rMax = 0 cannot happen (N >= 4 distinct slots).
    const omegaMax = Math.max(1, Math.min(BAM_QUARTER,
      Math.floor((VMAX * BAM_PER_RAD) / Math.max(1, rMax))));
    omega = Math.max(-omegaMax, Math.min(omegaMax, omega));

    const idx = this._nBodies;
    const rec = new Int32Array(RG_BODY_WORDS);
    rec[RB.LIVE] = 1;
    rec[RB.TAG] = tag | 0;
    rec[RB.FIRST] = members[0];
    rec[RB.LAST] = members[N - 1];
    rec[RB.CX] = cx; rec[RB.CY] = cy;
    rec[RB.VX] = vx; rec[RB.VY] = vy;
    rec[RB.THETA] = 0; rec[RB.OMEGA] = omega;
    rec[RB.N] = N;
    rec[RB.SUMIS2] = sumIs2; rec[RB.GSUM] = gSum;
    rec[RB.INVI] = invI; rec[RB.IQ] = iq;
    rec[RB.RSH] = rsh; rec[RB.ISH] = ish; rec[RB.OSH] = osh;
    rec[RB.OMEGAM] = omegaMax; rec[RB.RMAX] = rMax;
    rec[RB.CQX] = cx; rec[RB.CQY] = cy; rec[RB.THETAQ] = 0;
    rec[RB.AXIS] = axis ? 1 : 0;
    rec[RB.USER] = user ? 1 : 0;                  // §24
    this.device.queue.writeBuffer(this.buf.rigid, (RG_BODY_BASE + idx * RG_BODY_WORDS) * 4, rec);

    // Rest offsets, EXACT i32 ticks, written one contiguous window at a time.
    // Exactness is not a luxury: it is what makes `rigidMember`'s re-base a
    // lossless integer subtraction rather than a re-capture (§20.4.2).
    // WRITTEN PER CONTIGUOUS RUN OF MEMBERS, NOT AS ONE SPAN.
    //
    // This used to be a single `writeBuffer` over [members[0], members[N-1]],
    // with the non-member slots inside that span left as ZERO. For a brush
    // gesture — the only caller until §22 — members are contiguous, one run, and
    // the two are byte-identical. For a caller whose members are INTERLEAVED
    // with another body's, the span write ZEROES THE OTHER BODY'S REST OFFSETS,
    // and a member with a zero rest offset is placed at its body's centre.
    //
    // MEASURED, on `Nlynch77_Semisolid_Rigid.oec` imported as 175 bodies over
    // 2 902 interleaved slots: ONE substep moved all 2 771 members by up to
    // 12.976 wu (exactly rMax), produced 67 102 exact duplicate pairs, took the
    // worst neighbour count from 16 to 112 and tripped dbg[6] to 9 610 — which
    // per SPEC §5.4 makes the run INVALID. `promote: false` on the same scene
    // reads 0 at every one of those.
    //
    // Runs are maximal and `members` is ascending, so this is one write for
    // every existing caller and the bytes it puts down are the same ones.
    let runStart = 0;
    for (let k = 1; k <= N; k++) {
      if (k < N && members[k] === members[k - 1] + 1) continue;
      const lo0 = members[runStart], len = k - runStart;
      const win = new Int32Array(len * 2);
      for (let q = 0; q < len; q++) { win[q * 2] = ox[runStart + q]; win[q * 2 + 1] = oy[runStart + q]; }
      this.device.queue.writeBuffer(this.buf.rigid, (RG_REST_BASE + lo0 * 2) * 4, win);
      runStart = k;
    }
    const lo = members[0], hi = members[N - 1];

    this._nBodies = idx + 1;
    this._bodies.push({ tag: tag | 0, first: lo, last: hi, n: N, index: idx, axis: !!axis, user: !!user });
    return this._bodies[this._bodies.length - 1];
  }

  /**
   * Turn every particle wearing `tag` whose material declares SOLID_RIGID into
   * ONE rigid body, and set FLAG_RIGID on them. Costs one GPU sync (it must read
   * the positions the gesture actually laid, which only the GPU has), so it is
   * called ONCE, on `pointerup`.
   *
   * @param {number} [tag] already shifted into WELD_TAG_MASK; defaults to the
   *        gesture currently held open.
   * @returns {Promise<object|null>} the body, or null if nothing qualified.
   */
  async promoteWeldGroup(tag = this._weldHold, pivot = null, user = false) {
    const t = (tag >>> 0) & WELD_TAG_MASK;
    if (t === 0 || this.nFluid === 0) return null;
    const st = await this.readState();
    const stu = new Uint32Array(st.buffer, st.byteOffset, st.length);
    const PW = PARTICLE_WORDS;
    const iSolid = MAT_FIELDS.indexOf('solidMode');
    const members = [];
    for (let i = 0; i < this.nFluid; i++) {
      const f = stu[i * PW + 5];
      if ((f & WELD_TAG_MASK) !== t) continue;
      if ((f & FLAG_RIGID) !== 0) continue;                 // already in a body
      // §19.A. ANCHORED MATTER IS NEVER PROMOTED. The two mechanisms are
      // mutually exclusive by construction rather than by a shader test: an
      // anchored particle already never moves, and a body containing one would
      // have to choose between placing it and honouring it.
      if ((f & FLAG_ANCHOR) !== 0) continue;
      if (this.mats[stu[i * PW + 4]][iSolid] !== SOLID_RIGID) continue;
      members.push(i);
    }
    const body = this._writeBodyRecord(members, st, t, pivot, user);
    if (body === null) return null;
    // Set FLAG_RIGID on exactly those slots, in the CURRENT parity buffer (the
    // one the next substep reads as state_in), one contiguous window.
    const lo = body.first, hi = body.last;
    const stateBuf = this.parity === 0 ? this.buf.stateA : this.buf.stateB;
    const win = new Uint32Array((hi - lo + 1) * PW);
    for (let i = lo; i <= hi; i++) {
      for (let w = 0; w < PW; w++) win[(i - lo) * PW + w] = stu[i * PW + w];
    }
    for (const i of members) win[(i - lo) * PW + 5] = (win[(i - lo) * PW + 5] | FLAG_RIGID) >>> 0;
    this.device.queue.writeBuffer(stateBuf, lo * 32, win);
    this._writeNBodies();
    // A member gathers no bonds and `bondForm1` empties its candidate row, which
    // is how the OTHER half of any bond it made during the gesture is dropped.
    // Run that pass promptly rather than at the next BOND_PERIOD boundary.
    this._bondDirty = true;
    return body;
  }

  /**
   * Turn newly bonded, free rigid-material clusters into body records.
   *
   * Thermal solidification has no drawing gesture to promote on pointer-up:
   * Lava→Stone and Molten→Metal first form their normal distance bonds in the
   * solver, then this authoring-side bridge gives that settled cluster the same
   * rigid-body representation as a drawn Stone or Metal object. The host gives
   * it exact slot/material transition witnesses from its already-scheduled
   * render mirror; a component containing a live bond AND one of those freshly
   * congealed members qualifies. An ordinary Stone/Metal stroke is never swept
   * into this thermal bridge.
  */
  async promoteLooseRigidComponents(linkTicks = Math.round(1.65 * ONE), eligible = []) {
    if (this.nFluid === 0 || this._nBodies >= RG_MAX_BODIES) return 0;
    const witnesses = new Map();
    for (const e of eligible || []) {
      if (Number.isInteger(e?.slot) && Number.isInteger(e?.mat)) witnesses.set(e.slot, e.mat >>> 0);
    }
    if (!witnesses.size) return 0;
    const st = await this.readState();
    const su = new Uint32Array(st.buffer, st.byteOffset, st.length);
    const bond = await this.readBond();
    const iSolid = MAT_FIELDS.indexOf('solidMode');
    const grid = new Map(), candidates = [];
    const cell = Math.max(1, linkTicks | 0), key = (x, y, m) => `${x},${y},${m}`;
    const hasLiveBond = new Uint8Array(this.nFluid);
    for (let i = 0; i < this.nFluid; i++) {
      const bo = i * BOND_CAP;
      for (let q = 0; q < BOND_CAP; q++) {
        const w = bond[bo + q] >>> 0;
        if (!w) break;
        if ((w & BOND_TOMB) === 0) { hasLiveBond[i] = 1; break; }
      }
    }
    for (let i = 0; i < this.nFluid; i++) {
      const o = i * PARTICLE_WORDS, f = su[o + 5] >>> 0, m = su[o + 4] >>> 0;
      if ((f & (FLAG_ANCHOR | FLAG_RIGID)) !== 0 || this.mats[m]?.[iSolid] !== SOLID_RIGID) continue;
      const k = key(Math.floor(st[o] / cell), Math.floor(st[o + 1] / cell), m);
      let a = grid.get(k); if (!a) { a = []; grid.set(k, a); }
      a.push(i); candidates.push(i);
    }
    if (!candidates.length) return 0;
    const lim2 = cell * cell, seen = new Set(), groups = [];
    for (const seed of candidates) {
      if (seen.has(seed)) continue;
      const so = seed * PARTICLE_WORDS, mat = su[so + 4] >>> 0;
      const part = [], todo = [seed]; let bonded = false, congealed = false;
      seen.add(seed);
      while (todo.length) {
        const i = todo.pop(), o = i * PARTICLE_WORDS;
        part.push(i); bonded ||= hasLiveBond[i] !== 0;
        congealed ||= witnesses.get(i) === (su[o + 4] >>> 0);
        const cx = Math.floor(st[o] / cell), cy = Math.floor(st[o + 1] / cell);
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          for (const j of (grid.get(key(cx + dx, cy + dy, mat)) || [])) {
            if (seen.has(j)) continue;
            const p = j * PARTICLE_WORDS, ex = st[p] - st[o], ey = st[p + 1] - st[o + 1];
            if (ex * ex + ey * ey > lim2) continue;
            seen.add(j); todo.push(j);
          }
        }
      }
      part.sort((a, b) => a - b);
      if (bonded && congealed && part.length >= RIGID_MIN_MEMBERS) groups.push(part);
    }
    if (!groups.length) return 0;
    let made = 0;
    for (const members of groups) {
      if (this._nBodies >= RG_MAX_BODIES) break;
      const tag = (this._nextWeldSeq() << WELD_TAG_SHIFT) & WELD_TAG_MASK;
      for (const i of members) {
        const o = i * PARTICLE_WORDS + 5;
        su[o] = ((su[o] & ~WELD_TAG_MASK) | tag | FLAG_RIGID) >>> 0;
      }
      if (this._writeBodyRecord(members, st, tag)) made++;
    }
    if (!made) return 0;
    const stateBuf = this.parity === 0 ? this.buf.stateA : this.buf.stateB;
    this.device.queue.writeBuffer(stateBuf, 0, st, 0, this.n * PARTICLE_WORDS);
    this._writeNBodies();
    this._bondDirty = true;
    this._mirror = null; this._mirrorPending = false; this._mirrorEpoch++;
    return made;
  }

  /**
   * The API-path constructor: make ONE body out of an explicit slot list.
   *
   * `promoteWeldGroup` is the path a PERSON takes and it is the one the UI gates
   * drive. This is the path a TEST takes, and the pair is deliberate — §18's
   * whole failure was that every acceptance number came from a call no user can
   * make, so both paths exist and both are gated. This one allocates a fresh
   * body id that NO live particle is currently wearing, which is what lets one
   * scene hold two independent bodies.
   *
   * @param {number[]} slots fluid slot indices, ascending
   * @returns {Promise<object|null>}
   */
  async makeRigidBody(slots, pivot = null, user = false) {
    if (slots.length === 0 || this.nFluid === 0) return null;
    const st = await this.readState();
    const stu = new Uint32Array(st.buffer, st.byteOffset, st.length);
    const PW = PARTICLE_WORDS;
    const iSolid = MAT_FIELDS.indexOf('solidMode');
    const members = [];
    for (const i of slots) {
      if (i < 0 || i >= this.nFluid) continue;
      if (this.mats[stu[i * PW + 4]][iSolid] !== SOLID_RIGID) continue;
      if ((stu[i * PW + 5] & FLAG_RIGID) !== 0) continue;
      if ((stu[i * PW + 5] & FLAG_ANCHOR) !== 0) continue;   // §19.A, exclusive
      members.push(i);
    }
    members.sort((a, b) => a - b);
    if (members.length < RIGID_MIN_MEMBERS) return null;
    // A tag no live particle wears, so the new body cannot capture a stranger.
    const inUse = new Set();
    for (let i = 0; i < this.nFluid; i++) inUse.add((stu[i * PW + 5] & WELD_TAG_MASK) >>> WELD_TAG_SHIFT);
    let seq = 0;
    for (let g = 0; g <= WELD_SEQ_MAX; g++) {
      const q = this._nextWeldSeq();
      if (!inUse.has(q)) { seq = q; break; }
    }
    assert(seq !== 0, '[aether] §20: no free weld tag for a new body');
    const tag = (seq << WELD_TAG_SHIFT) & WELD_TAG_MASK;
    for (const i of members) {
      stu[i * PW + 5] = ((stu[i * PW + 5] & ~WELD_TAG_MASK) | tag | FLAG_RIGID) >>> 0;
    }
    const body = this._writeBodyRecord(members, st, tag, pivot, user);
    if (body === null) return null;
    const lo = body.first, hi = body.last;
    const stateBuf = this.parity === 0 ? this.buf.stateA : this.buf.stateB;
    this.device.queue.writeBuffer(stateBuf, lo * 32, st, lo * PW, (hi - lo + 1) * PW);
    this._writeNBodies();
    this._bondDirty = true;
    return body;
  }

  /**
   * §19.B — THE API PATH FOR AN AXIS BODY. Identical to `makeRigidBody` except
   * that the body's frame origin is the pivot you name, in TICKS, and it never
   * moves. The pivot is not a particle: nothing occupies it, erasing "it" is a
   * no-op, and it can sit inside the wheel, on its rim or in thin air.
   */
  async makeAxisBody(slots, px, py, user = false) {
    return this.makeRigidBody(slots, { x: px | 0, y: py | 0 }, user);
  }

  /**
   * Close the open gesture AND promote it. This is the pair `index.html` calls
   * on `pointerup`, and the order matters: `endWeldGroup` first, so the tag stops
   * accepting material, then `promoteWeldGroup` on the tag that was held.
   *
   * @param {?{x:number,y:number}} pivot §19.B — pin the finished gesture at this
   *        point (ticks) instead of letting it fall freely.
   * @param {boolean} [user] §24 — the player's arrow keys drive the finished
   *        body. Like `pivot`, it is a property of the GESTURE that made the
   *        thing, resolved once at promotion time and frozen onto the record.
   */
  async endWeldGroupAndPromote(pivot = null, user = false) {
    const was = this.endWeldGroup();
    if (was === 0) return null;
    return this.promoteWeldGroup(was, pivot, user);
  }

  /** Dissolve every body: members become ordinary free particles where they are. */
  async dissolveAllBodies() {
    if (this._nBodies === 0) return 0;
    const st = await this.readState();
    const stu = new Uint32Array(st.buffer, st.byteOffset, st.length);
    const PW = PARTICLE_WORDS;
    let k = 0;
    for (let i = 0; i < this.nFluid; i++) {
      if ((stu[i * PW + 5] & FLAG_RIGID) === 0) continue;
      stu[i * PW + 5] = (stu[i * PW + 5] & ~FLAG_RIGID) >>> 0;
      k++;
    }
    const stateBuf = this.parity === 0 ? this.buf.stateA : this.buf.stateB;
    this.device.queue.writeBuffer(stateBuf, 0, st, 0, this.n * PW);
    this._nBodies = 0; this._bodies = []; this._bodyDrivers.clear();
    this._writeNBodies();
    const enc = this.device.createCommandEncoder({ label: 'aether:dissolve' });
    enc.clearBuffer(this.buf.rigid, RG_BODY_BASE * 4);
    this.device.queue.submit([enc.finish()]);
    return k;
  }

  /** §20 body records, straight out of the buffer. Diagnostics and gates. */
  async readBodies() {
    const w = await this._read(this.buf.rigid, RG_MAX_BODIES * RG_BODY_WORDS * 4, RG_BODY_BASE * 4);
    return new Int32Array(w);
  }

  /** §20 rest offsets, straight out of the buffer. Diagnostics and gates. */
  async readRestOffsets() {
    const w = await this._read(this.buf.rigid, this.n * 2 * 4, RG_REST_BASE * 4);
    return new Int32Array(w);
  }

  /** The bond-formation cadence is scene state just as surely as velocity is.
   * Native `.suna` snapshots keep it so loading immediately before a scheduled
   * formation pass behaves exactly like continuing without the save. */
  get formationClock() { return this._substep; }

  /**
   * Restore the engine-owned parts of an exact authoring snapshot after
   * `loadScene` has rebuilt the allocation/domain shell.
   *
   * This is deliberately narrower than a second scene loader: `loadScene`
   * remains the only path that validates dimensions, material ids and authored
   * flags. This method restores the state that cannot be expressed through
   * that public scene grammar—derived flags/weld tags, severed bond rows and
   * live rigid-body records—so Undo and `.suna` do not quietly soften or re-weld
   * a construction.
   */
  restoreAuthoringSnapshot({
    state, bonds, bondCand = bonds, bodyRecords, restOffsets, bodies = [],
    substepCount = 0, formationClock = substepCount,
  }) {
    assert(state instanceof Int32Array && state.length === this.n * PARTICLE_WORDS,
      `snapshot state has ${state?.length ?? -1} words; expected ${this.n * PARTICLE_WORDS}`);
    assert(bonds instanceof Uint32Array && bonds.length === this.n * BOND_CAP,
      `snapshot bonds have ${bonds?.length ?? -1} words; expected ${this.n * BOND_CAP}`);
    assert(bondCand instanceof Uint32Array && bondCand.length === this.n * BOND_CAP,
      `snapshot bond candidates have ${bondCand?.length ?? -1} words; expected ${this.n * BOND_CAP}`);
    assert(bodyRecords instanceof Int32Array &&
      bodyRecords.length === RG_MAX_BODIES * RG_BODY_WORDS,
      `snapshot body records have ${bodyRecords?.length ?? -1} words; expected ${RG_MAX_BODIES * RG_BODY_WORDS}`);
    assert(restOffsets instanceof Int32Array && restOffsets.length === this.n * 2,
      `snapshot rest offsets have ${restOffsets?.length ?? -1} words; expected ${this.n * 2}`);
    assert(Array.isArray(bodies) && bodies.length <= RG_MAX_BODIES,
      `snapshot has ${bodies?.length ?? -1} body descriptors; max ${RG_MAX_BODIES}`);

    // Both ping-pong copies agree at the restore boundary. The next encoded
    // substep reads parity 0 and writes parity 1, exactly like a fresh scene.
    this.parity = 0;
    this.device.queue.writeBuffer(this.buf.stateA, 0, state);
    this.device.queue.writeBuffer(this.buf.stateB, 0, state);
    this.device.queue.writeBuffer(this.buf.bond, 0, bonds);
    this.device.queue.writeBuffer(this.buf.bondCand, 0, bondCand);
    this.device.queue.writeBuffer(this.buf.rigid, RG_BODY_BASE * 4, bodyRecords);
    this._bodyDrivers.clear();
    this.device.queue.writeBuffer(
      this.buf.rigid,
      RG_DRIVER_BASE * 4,
      new Int32Array(RG_MAX_BODIES * RG_DRIVER_WORDS));
    if (restOffsets.length) {
      this.device.queue.writeBuffer(this.buf.rigid, RG_REST_BASE * 4, restOffsets);
    }

    this._bodies = bodies.map((b, index) => ({
      tag: b.tag | 0, first: b.first | 0, last: b.last | 0, n: b.n | 0,
      index: Number.isInteger(b.index) ? b.index : index,
      axis: !!b.axis, user: !!b.user,
    }));
    this._nBodies = this._bodies.length;
    this._writeNBodies();
    this.substepCount = Math.max(0, substepCount | 0);
    this._substep = Math.max(0, formationClock | 0);
    this._bondDirty = false;
    this._mirror = null;
    this._mirrorPending = false;
    this._mirrorEpoch++;
    this._initialState = new Int32Array(state);
    return this;
  }

  /**
   * Append fluid particles. Boundary particles are relocated to the tail so the
   * [0,nFluid) / [nFluid,n) invariant holds. Rewrites Params.n / Params.nFluid,
   * which is why this is an AUTHORING call: the determinism gate never uses it.
   *
   * @param {object[]} list
   * @param {{ownGroup?: boolean,noWeld?: boolean}} [opts] `ownGroup: true` forces a fresh weld
   *   group even while a gesture is open. Every one-shot spawn that is NOT part
   *   of the stroke in progress passes it (PNG import, `ignite`), so a modal
   *   import can never be welded onto the brush stroke underneath it.
   */
  spawnFluid(list, opts = {}) {
    const k = list.length;
    if (k === 0) return this.n;
    const nB = this.n - this.nFluid;
    const newFluid = this.nFluid + k;
    const newN = newFluid + nB;
    if (newN > this.maxParticles) return this.n;

    const stateBuf = this.parity === 0 ? this.buf.stateA : this.buf.stateB;
    const total = k + nB;
    const st = new Int32Array(total * PARTICLE_WORDS);
    const stu = new Uint32Array(st.buffer);
    // §12. A painted particle starts at its material's `spawnTemp` carrying its
    // material's `fuel0` unless the caller says otherwise. That is what makes
    // "paint FIRE" produce fire rather than a cold orange blob, and it is why
    // the UI never has to know the thermal table exists.
    const iSpawnT = MAT_FIELDS.indexOf('spawnTemp');
    const iFuel0  = MAT_FIELDS.indexOf('fuel0');
    // §17, same rule as loadScene: FLAG_DEAD is derived from the material row,
    // never taken from the caller. Painting VOID is not a thing the UI offers
    // (the picker filters on `phantom`) but the API must not be able to produce
    // a live particle wearing a corpse's flag or the reverse.
    const iPhantom = MAT_FIELDS.indexOf('phantom');
    // §18.2 W1 + §18.22. ONE GESTURE IS ONE BONDING GROUP — Darien's decision,
    // and this line is where it lives. A spawn that is not part of an open
    // gesture is its own group (one PNG import, one emitter batch, one
    // `loadScene`), so material arriving later does not weld to what is already
    // there. MEASURED: two touching 6x6 STONE blocks from two calls give exactly
    // 0 cross-event bonds at 0, 8 and 400 substeps apart, and 22 from one call.
    //
    // WHILE A GESTURE IS OPEN, every spawn joins it and the counter does NOT
    // advance — that is the whole of the drops fix. See the block above
    // `_nextWeldSeq` for why the tag is latched and for the re-argued wrap.
    const seq = (this._weldHoldSeq !== 0 && opts.ownGroup !== true)
      ? this._weldHoldSeq : this._nextWeldSeq();
    const weldTag = (seq << WELD_TAG_SHIFT) & WELD_TAG_MASK;
    // §26.B. The tag THIS batch was written with, so a one-shot spawn that owns
    // its own group (`ownGroup: true` — a PNG import, an emitter tick) can be
    // handed to `promoteWeldGroup` afterwards. Without it the only way to reach
    // the promotion path is `makeRigidBody`, which allocates a FRESH tag and is
    // the API path a test takes; §18.22's whole lesson is that the two must not
    // drift, so the import goes through the same promotion a gesture does.
    this._lastWeldTag = weldTag;
    for (let i = 0; i < k; i++) {
      const o = i * PARTICLE_WORDS, p = list[i];
      const m = Math.min(this.matCount - 1, Math.max(0, p.mat | 0)) >>> 0;
      // §25. Count wearers of interned rows (the recycle policy's refcount
      // floor) and touch the LRU clock — a row being painted is a row in use.
      if (m >= this._authoredCount) {
        this._internSpawned.set(m, (this._internSpawned.get(m) ?? 0) + 1);
        const e = this._internByRow.get(m);
        if (e) e.tick = ++this._internTick;
      }
      st[o + 0] = p.x | 0; st[o + 1] = p.y | 0;
      st[o + 2] = (p.vx | 0); st[o + 3] = (p.vy | 0);
      stu[o + 4] = m;
      // The tag is written ONLY when the row can bond, so a WATER brush stroke
      // produces flags words that are bit-for-bit v7's.
      stu[o + 5] = (((p.flags ?? 0) & SPAWN_FLAG_MASK) |
                    (this.mats[m][iPhantom] !== 0 ? FLAG_DEAD : 0) |
                    (sealRow(this.mats[m]) ? FLAG_SEAL : 0) |
                    (!opts.noWeld && spawnWelds(this.mats[m]) ? weldTag : 0)) >>> 0;
      st[o + 6] = Math.max(TEMP_MIN, Math.min(TEMP_MAX, p.temp ?? this.mats[m][iSpawnT])) | 0;
      st[o + 7] = Math.max(0, p.fuel ?? this.mats[m][iFuel0]) | 0;
    }
    const sc = this._scene;
    for (let b = 0; b < nB; b++) {
      const o = (k + b) * PARTICLE_WORDS, src = sc.nFluid + b;
      const bm = sc.matId[src] >>> 0;
      st[o + 0] = sc.pos[src * 2]; st[o + 1] = sc.pos[src * 2 + 1];
      stu[o + 4] = bm;
      st[o + 6] = sc.temp ? sc.temp[src] : AMBIENT;
      st[o + 7] = sc.fuel ? sc.fuel[src] : this.mats[bm][iFuel0];
    }
    this.device.queue.writeBuffer(stateBuf, this.nFluid * 32, st);

    const dv = new Int32Array(total * DERIVED_WORDS);
    for (let i = 0; i < total; i++) {
      dv[i * DERIVED_WORDS + 0] = st[i * PARTICLE_WORDS + 0];
      dv[i * DERIVED_WORDS + 1] = st[i * PARTICLE_WORDS + 1];
    }
    this.device.queue.writeBuffer(this.buf.derived, this.nFluid * DERIVED_BYTES, dv);

    // §18.7 R5. A REUSED SLOT'S ROW MUST BE ZEROED BEFORE THE PARTICLE IS
    // WRITTEN, in BOTH buffers. Today `_freeSlots` is empty so spawnFluid only
    // ever appends, and these rows are already zero — but writing them is what
    // makes slot reuse safe when vent phase 2 lands, and it costs one
    // writeBuffer of 3 KB per 16 particles.
    if (k > 0) {
      const zeros = new Uint32Array(k * BOND_CAP);
      this.device.queue.writeBuffer(this.buf.bond, this.nFluid * BOND_CAP * 4, zeros);
      this.device.queue.writeBuffer(this.buf.bondCand, this.nFluid * BOND_CAP * 4, zeros);
    }
    // §21. THE COLOUR BLOCK MOVES WITH THE MATTER — and the boundary block that
    // this spawn just pushed up by k slots is rewritten too, not merely assumed
    // to be zero. `total` covers both, exactly like the state write above: the
    // boundary particles have no colour, so their k-slot shift is a shift of
    // zeros, and writing them is what stops a previous occupant's pigment
    // reappearing under a wall. `p.tint` is optional and is the ONLY way colour
    // enters the engine (there is no colour -> material mapping here, by the
    // same rule that governs src/import.js).
    {
      const tn = new Uint32Array(total);
      for (let i = 0; i < k; i++) tn[i] = (list[i].tint ?? 0) >>> 0;
      this.device.queue.writeBuffer(this.buf.tint, this.nFluid * 4, tn);
    }
    this.device.queue.writeBuffer(this.buf.params, 0, new Uint32Array([newN, newFluid]));
    // §18.8. hashParams must carry the NEW n_fluid. Through v7 only loadScene
    // wrote this buffer at all; if `n_fluid` goes stale the bond fold silently
    // covers the wrong slot range.
    this.device.queue.writeBuffer(this.buf.hashParams, 0, new Uint32Array([newN, PARTICLE_WORDS, HASHED_WORDS, newFluid]));
    this.n = newN; this.nFluid = newFluid;
    // §18.2 W4. Run the formation pass on the very NEXT substep, not on the next
    // BOND_PERIOD boundary — a weld tag must not outlive one pass.
    this._bondDirty = true;
    // Tag this batch so a host-side density guard can tell whether the position
    // mirror it is about to count already contains it. See refreshMirror.
    this._spawnSeq++;
    return this.n;
  }

  /**
   * Refresh the host-side position mirror used by pointer interaction.
   *
   * THE MIRROR IS ALWAYS BEHIND, AND CALLERS MUST BE ABLE TO TELL BY HOW MUCH.
   * This is a fire-and-forget readback: the promise resolves some frames after
   * it was issued, so `_mirror` is a snapshot of a world that no longer exists.
   * That is fine for the collector pre-check (a false negative just costs a
   * sweep) and it was NOT fine for index.html's density guard, which counted
   * occupancy from it while up to 20 brush stamps landed per frame — measured,
   * a held-down brush reached dbg[6] = 12405 and put the run outside the
   * §5.4 spec-valid regime with the guard reporting no problem at all.
   *
   * `_spawnSeq` counts spawnFluid calls; `_mirrorSeq` records the value this
   * snapshot was taken AFTER. writeBuffer and the readback share the queue, so
   * the snapshot provably contains every spawn with seq <= _mirrorSeq and
   * provably contains none with a higher seq. A caller can therefore replay
   * exactly the spawns the mirror is missing instead of guessing a frame count.
   */
  refreshMirror() {
    if (this._mirrorPending || this.n === 0) return;
    this._mirrorPending = true;
    const seq = this._spawnSeq;
    const epoch = ++this._mirrorEpoch;
    this.readState().then((s) => {
      if (epoch !== this._mirrorEpoch) return;
      this._mirror = s; this._mirrorSeq = seq; this._mirrorPending = false;
    }).catch(() => {
      if (epoch === this._mirrorEpoch) this._mirrorPending = false;
    });
  }

  /**
   * Deterministic authoring boundary. Unlike refreshMirror(), this waits for
   * every earlier queue submission and returns only after the host mirror names
   * the exact current world. Replay recording/restoration uses it before the
   * first input so density guarding and object picking cannot inherit a stale
   * frame from the scene that came before.
   */
  async syncMirror() {
    // Retire an older fire-and-forget callback before waiting for its already
    // submitted copy; it must not overwrite this exact boundary afterward.
    const epoch = ++this._mirrorEpoch;
    this._mirrorPending = true;
    await this.device.queue.onSubmittedWorkDone();
    const seq = this._spawnSeq;
    const state = await this.readState();
    if (epoch === this._mirrorEpoch) {
      this._mirror = state;
      this._mirrorSeq = seq;
      this._mirrorPending = false;
    }
    return state;
  }

  /** Push matter near (cx,cy) by (ix,iy) ticks/substep. Authoring only. */
  applyImpulse(cx, cy, radius, ix, iy, limit = 3000) {
    const mir = this._mirror;
    if (!mir) return 0;
    const stateBuf = this.parity === 0 ? this.buf.stateA : this.buf.stateB;
    const r2 = radius * radius;
    let hits = 0;
    const nf = Math.min(this.nFluid, (mir.length / PARTICLE_WORDS) | 0);
    const patch = new Int32Array(2);
    for (let i = 0; i < nf && hits < limit; i++) {
      const o = i * PARTICLE_WORDS;
      const dx = mir[o] - cx, dy = mir[o + 1] - cy;
      if (dx * dx + dy * dy > r2) continue;
      let vx = mir[o + 2] + ix, vy = mir[o + 3] + iy;
      vx = Math.max(-VMAX, Math.min(VMAX, vx));
      vy = Math.max(-VMAX, Math.min(VMAX, vy));
      patch[0] = vx; patch[1] = vy;
      this.device.queue.writeBuffer(stateBuf, o * 4 + 8, patch);
      mir[o + 2] = vx; mir[o + 3] = vy;
      hits++;
    }
    return hits;
  }

  /**
   * Move a player-selected clump directly under the pointer.
   *
   * This is deliberately authoring rather than a Jet: Arrow is a hand, not a
   * force field. A free clump is held in a compact cursor-centred pocket; a
   * free rigid body is driven through its body record so it retains its exact
   * rest frame.
   * `vx`/`vy` are cursor velocity in ticks per substep and survive the release,
   * which makes the same gesture a drop or a throw without a second operation.
   * The ordered pointer journal is therefore the complete deterministic input.
   */
  dragGrab(grab, targetX, targetY, vx = 0, vy = 0, release = false) {
    if (!grab || !Array.isArray(grab.slots) || !grab.slots.length || !this._mirror) return 0;
    const mir = this._mirror;
    const nf = Math.min(this.nFluid, (mir.length / PARTICLE_WORDS) | 0);
    const slots = grab.slots.filter((i) => Number.isInteger(i) && i >= 0 && i < nf)
      .sort((a, b) => a - b);
    if (!slots.length) return 0;
    const cvx = Math.max(-VMAX, Math.min(VMAX, vx | 0));
    const cvy = Math.max(-VMAX, Math.min(VMAX, vy | 0));

    if (grab.rigid) {
      const body = this._bodies.find((b) => (b.tag >>> 0) === (grab.tag >>> 0));
      // An axis is scenery, never a hand-held object.
      if (!body || body.axis) return 0;
      // rigidStep predicts C + V.  Put C one velocity behind the pointer so
      // the substep lands at the hand rather than overshooting it.
      const cx = Math.max(ONE, Math.min(this.domW - ONE, (targetX | 0) - cvx));
      const cy = Math.max(ONE, Math.min(this.domH - ONE, (targetY | 0) - cvy));
      this.device.queue.writeBuffer(
        this.buf.rigid, (RG_BODY_BASE + body.index * RG_BODY_WORDS + RB.CX) * 4,
        new Int32Array([cx, cy, cvx, cvy]));
      // A carried force remainder belongs to the old trajectory.  Keeping it
      // would make a grabbed body visibly kick away from an otherwise still
      // cursor, so the hand owns those two translational remainders.
      this.device.queue.writeBuffer(
        this.buf.rigid, (RG_BODY_BASE + body.index * RG_BODY_WORDS + RB.RESX) * 4,
        new Int32Array(4));
      return slots.length;
    }

    // Arrow's loose-matter grab is a temporary little gravity pocket, not a
    // jet. Capture each selected particle's local position once and rewrite
    // that compact handful at the cursor every frame while held. That keeps a
    // water/powder glob evenly suspended instead of letting solver substeps
    // shear it apart between pointer events. `offsets` is host-only gesture
    // state; it neither changes the simulation format nor survives release.
    if (!Array.isArray(grab.offsets) || grab.offsets.length !== slots.length) {
      let sx = 0, sy = 0, n = 0;
      for (const i of slots) {
        const o = i * PARTICLE_WORDS;
        if (mir[o + 5] & FLAG_ANCHOR) continue;
        sx += mir[o]; sy += mir[o + 1]; n++;
      }
      if (!n) return 0;
      const cx = Math.round(sx / n), cy = Math.round(sy / n);
      grab.offsets = slots.map((i) => {
        const o = i * PARTICLE_WORDS;
        return [mir[o] - cx, mir[o + 1] - cy];
      });
    }
    const stateBuf = this.parity === 0 ? this.buf.stateA : this.buf.stateB;
    const patch = new Int32Array(PARTICLE_WORDS);
    let moved = 0;
    for (let k = 0; k < slots.length; k++) {
      const i = slots[k];
      const o = i * PARTICLE_WORDS;
      if (mir[o + 5] & FLAG_ANCHOR) continue;
      for (let w = 0; w < PARTICLE_WORDS; w++) patch[w] = mir[o + w];
      const [ox, oy] = grab.offsets[k] || [0, 0];
      patch[0] = Math.max(ONE, Math.min(this.domW - ONE, (targetX | 0) + ox));
      patch[1] = Math.max(ONE, Math.min(this.domH - ONE, (targetY | 0) + oy));
      // Holding the button suspends the glob. The final held sample carries
      // cursor velocity into the solver, so releasing still drops or throws it.
      patch[2] = release ? cvx : 0;
      patch[3] = release ? cvy : 0;
      // `o` is already the word offset (`slot * PARTICLE_WORDS`). Multiplying
      // it by PARTICLE_WORDS again addressed slot*8 and wrote past stateA/B for
      // ordinary large scenes; Arrow's loose-grab hand must address this slot.
      this.device.queue.writeBuffer(stateBuf, o * 4, patch);
      // Keep the picker mirror coherent between its intentionally sparse
      // readbacks.  The next GPU mirror supersedes these exact hand edits.
      mir[o] = patch[0]; mir[o + 1] = patch[1]; mir[o + 2] = patch[2]; mir[o + 3] = patch[3];
      moved++;
    }
    return moved;
  }

  /** Replace the material of fluid particles selected by a host predicate.
   * Authoring-only, one GPU readback. Bonds touching a changed particle are
   * tombstoned so an old material relationship cannot survive the edit. */
  async replaceWhere(test, newMat) {
    const m = newMat | 0;
    assert(m >= 0 && m < this.matCount && !isPhantom(m, this.mats),
      `replaceWhere material ${m} is not live`);
    if (this.nFluid === 0) return 0;
    const st = await this.readState();
    const su = new Uint32Array(st.buffer, st.byteOffset, st.length);
    const changed = new Uint8Array(this.nFluid);
    let n = 0;
    for (let i = 0; i < this.nFluid; i++) {
      const o = i * PARTICLE_WORDS;
      if (!test({ i, x: st[o], y: st[o + 1], vx: st[o + 2], vy: st[o + 3],
                  mat: su[o + 4], flags: su[o + 5], temp: st[o + 6], fuel: st[o + 7] })) continue;
      if (su[o + 4] === m) continue;
      changed[i] = 1; n++;
      su[o + 4] = m;
      // A replacement is new matter at an old position. It keeps authored
      // Burning/Anchor bits, but never inherits an old weld group or rigid-body
      // membership; those belong to the material relationship we sever below.
      su[o + 5] = (su[o + 5] &
        ~(FLAG_DEAD | FLAG_SEAL | FLAG_CONGEAL | FLAG_BURST | FLAG_RIGID | WELD_TAG_MASK)) |
        (this.mats[m][MAT_FIELDS.indexOf('phantom')] !== 0 ? FLAG_DEAD : 0) |
        (sealRow(this.mats[m]) ? FLAG_SEAL : 0);
      st[o + 6] = this.mats[m][MAT_FIELDS.indexOf('spawnTemp')];
      st[o + 7] = this.mats[m][MAT_FIELDS.indexOf('fuel0')];
    }
    if (!n) return 0;
    const bonds = await this.readBond();
    for (let i = 0; i < this.nFluid; i++) for (let k = 0; k < BOND_CAP; k++) {
      const q = i * BOND_CAP + k, w = bonds[q] >>> 0;
      if (!w) break;
      const j = w & BOND_J_MASK;
      if (changed[i] || (j < this.nFluid && changed[j])) bonds[q] = (w | BOND_TOMB) >>> 0;
    }
    const stateBuf = this.parity === 0 ? this.buf.stateA : this.buf.stateB;
    this.device.queue.writeBuffer(stateBuf, 0, st);
    this.device.queue.writeBuffer(this.buf.bond, 0, bonds);
    this.device.queue.writeBuffer(this.buf.bondCand, 0, bonds);
    this._bondDirty = true; this._mirror = null;
    this._mirrorPending = false; this._mirrorEpoch++;
    return n;
  }

  /** Sever bonds whose particle-to-particle segment crosses any supplied cut.
   * This edits the persistent bond graph in place; particles are not erased. */
  async sliceBondsMany(segments, kerf = Math.round(ONE * 0.52)) {
    if (this.nFluid === 0) return 0;
    assert(Array.isArray(segments), 'sliceBondsMany expects an array');
    if (!segments.length) return 0;
    const st = await this.readState();
    const bond = await this.readBond();
    const orient = (ax, ay, bx, by, cx, cy) =>
      (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    const crosses = (ax, ay, bx, by, cx, cy, dx, dy) => {
      if (Math.max(ax, bx) < Math.min(cx, dx) || Math.max(cx, dx) < Math.min(ax, bx) ||
          Math.max(ay, by) < Math.min(cy, dy) || Math.max(cy, dy) < Math.min(ay, by)) {
        return false;
      }
      const a = orient(ax, ay, bx, by, cx, cy), b = orient(ax, ay, bx, by, dx, dy);
      const c = orient(cx, cy, dx, dy, ax, ay), d = orient(cx, cy, dx, dy, bx, by);
      return ((a === 0 || b === 0 || (a < 0) !== (b < 0)) &&
              (c === 0 || d === 0 || (c < 0) !== (d < 0)));
    };
    const pointSegD2 = (px, py, ax, ay, bx, by) => {
      const ex = bx - ax, ey = by - ay, den = ex * ex + ey * ey;
      const t = den > 0 ? Math.max(0, Math.min(1, ((px - ax) * ex + (py - ay) * ey) / den)) : 0;
      const dx = px - (ax + ex * t), dy = py - (ay + ey * t);
      return dx * dx + dy * dy;
    };
    const kerf2 = Math.max(0, kerf | 0) ** 2;
    const inKerf = (ax, ay, bx, by, s) =>
      crosses(ax, ay, bx, by, s.x0 | 0, s.y0 | 0, s.x1 | 0, s.y1 | 0) ||
      pointSegD2(ax, ay, s.x0 | 0, s.y0 | 0, s.x1 | 0, s.y1 | 0) <= kerf2 ||
      pointSegD2(bx, by, s.x0 | 0, s.y0 | 0, s.x1 | 0, s.y1 | 0) <= kerf2 ||
      pointSegD2(s.x0 | 0, s.y0 | 0, ax, ay, bx, by) <= kerf2 ||
      pointSegD2(s.x1 | 0, s.y1 | 0, ax, ay, bx, by) <= kerf2;
    let cut = 0;
    for (let i = 0; i < this.nFluid; i++) for (let k = 0; k < BOND_CAP; k++) {
      const q = i * BOND_CAP + k, w = bond[q] >>> 0;
      if (!w) break;
      if (w & BOND_TOMB) continue;
      const j = w & BOND_J_MASK;
      if (j <= i || j >= this.nFluid) continue;
      let hit = false;
      for (const s of segments) {
        if (inKerf(st[i * PARTICLE_WORDS], st[i * PARTICLE_WORDS + 1],
                   st[j * PARTICLE_WORDS], st[j * PARTICLE_WORDS + 1], s)) {
          hit = true; break;
        }
      }
      if (!hit) continue;
      cut++;
      bond[q] = (w | BOND_TOMB) >>> 0;
      const jo = j * BOND_CAP;
      for (let z = 0; z < BOND_CAP; z++) {
        const w2 = bond[jo + z] >>> 0;
        if (!w2) break;
        if ((w2 & BOND_J_MASK) === i) { bond[jo + z] = (w2 | BOND_TOMB) >>> 0; break; }
      }
    }
    if (cut) {
      this.device.queue.writeBuffer(this.buf.bond, 0, bond);
      this.device.queue.writeBuffer(this.buf.bondCand, 0, bond);
      this._bondDirty = true;
    }
    return cut;
  }

  async sliceBonds(x0, y0, x1, y1) {
    return this.sliceBondsMany([{ x0, y0, x1, y1 }]);
  }

  /**
   * Rebuild rigid-body records when erasing a cut line has divided one body
   * into disconnected particle components. The solver intentionally stores a
   * rigid object as one record rather than as bonds, so bond slicing alone can
   * never split it.
   *
   * Components use the authored particle lattice (1.65 wu link radius). Small
   * chips below RIGID_MIN_MEMBERS become ordinary loose rigid material.
   */
  async splitDisconnectedBodies(linkTicks = Math.round(1.65 * ONE), cuts = [],
    kerf = Math.round(ONE * 0.52)) {
    if (this._nBodies === 0 || this.nFluid === 0) return { split: 0, bodies: this._nBodies };
    const st = await this.readState();
    const stu = new Uint32Array(st.buffer, st.byteOffset, st.length);
    const recs = await this.readBodies();
    const specs = [];
    let split = 0;

    const pointSegD2 = (px, py, ax, ay, bx, by) => {
      const ex = bx - ax, ey = by - ay, den = ex * ex + ey * ey;
      const t = den > 0 ? Math.max(0, Math.min(1, ((px - ax) * ex + (py - ay) * ey) / den)) : 0;
      const dx = px - (ax + ex * t), dy = py - (ay + ey * t);
      return dx * dx + dy * dy;
    };
    const orient = (ax, ay, bx, by, cx, cy) =>
      (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    const crosses = (ax, ay, bx, by, cx, cy, dx, dy) => {
      if (Math.max(ax, bx) < Math.min(cx, dx) || Math.max(cx, dx) < Math.min(ax, bx) ||
          Math.max(ay, by) < Math.min(cy, dy) || Math.max(cy, dy) < Math.min(ay, by)) return false;
      const a = orient(ax, ay, bx, by, cx, cy), b = orient(ax, ay, bx, by, dx, dy);
      const c = orient(cx, cy, dx, dy, ax, ay), d = orient(cx, cy, dx, dy, bx, by);
      return ((a === 0 || b === 0 || (a < 0) !== (b < 0)) &&
              (c === 0 || d === 0 || (c < 0) !== (d < 0)));
    };
    const kerf2 = Math.max(0, kerf | 0) ** 2;
    const severedByCut = (ax, ay, bx, by) => cuts.some((s) => {
      const x0 = s.x0 | 0, y0 = s.y0 | 0, x1 = s.x1 | 0, y1 = s.y1 | 0;
      return crosses(ax, ay, bx, by, x0, y0, x1, y1) ||
        pointSegD2(ax, ay, x0, y0, x1, y1) <= kerf2 ||
        pointSegD2(bx, by, x0, y0, x1, y1) <= kerf2 ||
        pointSegD2(x0, y0, ax, ay, bx, by) <= kerf2 ||
        pointSegD2(x1, y1, ax, ay, bx, by) <= kerf2;
    });
    const components = (members) => {
      if (members.length < 2) return [members];
      const cell = Math.max(1, linkTicks), lim2 = linkTicks * linkTicks;
      const grid = new Map(), key = (x, y) => x + ',' + y;
      for (const i of members) {
        const o = i * PARTICLE_WORDS;
        const k = key(Math.floor(st[o] / cell), Math.floor(st[o + 1] / cell));
        let a = grid.get(k); if (!a) { a = []; grid.set(k, a); }
        a.push(i);
      }
      const seen = new Set(), out = [];
      for (const seed of members) {
        if (seen.has(seed)) continue;
        const comp = [], q = [seed]; seen.add(seed);
        while (q.length) {
          const i = q.pop(), o = i * PARTICLE_WORDS;
          comp.push(i);
          const cx = Math.floor(st[o] / cell), cy = Math.floor(st[o + 1] / cell);
          for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
            for (const j of (grid.get(key(cx + dx, cy + dy)) || [])) {
              if (seen.has(j)) continue;
              const p = j * PARTICLE_WORDS;
              const ex = st[p] - st[o], ey = st[p + 1] - st[o + 1];
              if (ex * ex + ey * ey > lim2 || severedByCut(st[o], st[o + 1], st[p], st[p + 1])) continue;
              seen.add(j); q.push(j);
            }
          }
        }
        comp.sort((a, b) => a - b);
        out.push(comp);
      }
      out.sort((a, b) => b.length - a.length || a[0] - b[0]);
      return out;
    };

    for (const b of this._bodies) {
      const members = [];
      for (let i = b.first; i <= b.last && i < this.nFluid; i++) {
        const f = stu[i * PARTICLE_WORDS + 5];
        if ((f & FLAG_RIGID) && (f & WELD_TAG_MASK) === (b.tag >>> 0)) members.push(i);
      }
      if (!members.length) continue;
      const parts = components(members);
      if (parts.length > 1) split += parts.length - 1;
      const ro = b.index * RG_BODY_WORDS;
      specs.push({
        parts,
        axis: !!b.axis,
        user: !!b.user,
        pivot: b.axis ? { x: recs[ro + RB.CX], y: recs[ro + RB.CY] } : null,
      });
    }
    if (!split) return { split: 0, bodies: this._nBodies };

    // Dissolve the old records in the host copy. The replacement records and
    // rest offsets are uploaded in deterministic body/component order below.
    for (let i = 0; i < this.nFluid; i++) {
      const o = i * PARTICLE_WORDS + 5;
      if (stu[o] & FLAG_RIGID) stu[o] = (stu[o] & ~(FLAG_RIGID | WELD_TAG_MASK)) >>> 0;
    }
    const enc = this.device.createCommandEncoder({ label: 'aether:slice-rigid-rebuild' });
    enc.clearBuffer(this.buf.rigid, RG_BODY_BASE * 4);
    this.device.queue.submit([enc.finish()]);
    this._nBodies = 0; this._bodies = []; this._bodyDrivers.clear();

    for (const spec of specs) {
      for (let p = 0; p < spec.parts.length; p++) {
        const members = spec.parts[p];
        if (members.length < RIGID_MIN_MEMBERS || this._nBodies >= RG_MAX_BODIES) continue;
        const seq = this._nextWeldSeq();
        const tag = (seq << WELD_TAG_SHIFT) & WELD_TAG_MASK;
        for (const i of members) {
          const o = i * PARTICLE_WORDS + 5;
          stu[o] = ((stu[o] & ~WELD_TAG_MASK) | tag | FLAG_RIGID) >>> 0;
        }
        // Only the largest surviving piece keeps an axle/user driver. A cut-off
        // fragment is free matter, not a second copy of the original machine.
        this._writeBodyRecord(members, st, tag,
          p === 0 && spec.axis ? spec.pivot : null,
          p === 0 && spec.user);
      }
    }
    const stateBuf = this.parity === 0 ? this.buf.stateA : this.buf.stateB;
    this.device.queue.writeBuffer(stateBuf, 0, st, 0, this.n * PARTICLE_WORDS);
    this._writeNBodies();
    this._bondDirty = true;
    this._mirror = null;
    this._mirrorPending = false;
    this._mirrorEpoch++;
    return { split, bodies: this._nBodies };
  }

  /**
   * Delete fluid particles. AUTHORING ONLY — this is the eraser, and the
   * collector, and the only way smoke ever leaves the box.
   *
   * WHY IT IS HERE AND NOT IN THE SOLVER. Removing a particle on the GPU needs
   * a compaction, which needs an atomic allocation, whose ARRIVAL ORDER is not
   * reproducible on this device (measured: 1760-3430 of 4096 bucketIds slots
   * differ run to run — PLATFORM_NOTES). That is precisely the nondeterminism
   * `canonicalize` was built to absorb, and re-introducing it inside the physics
   * would cost the whole bit-exactness claim to save a host sync. So deletion
   * lives out here, next to spawnFluid and applyImpulse, on the authoring side
   * of the line — the determinism gate never calls any of them.
   *
   * Costs one full GPU sync. Call it on a user action or at most a few times a
   * second, never per frame.
   *
   * @param {(p:{i:number,x:number,y:number,vx:number,vy:number,mat:number,
   *            flags:number,temp:number,fuel:number}) => boolean} keep
   *        return FALSE to delete. Called once per fluid particle, in index
   *        order, so it is itself deterministic if the predicate is.
   * @returns {Promise<number>} how many particles were removed
   */
  async eraseWhere(keep) {
    if (this.nFluid === 0) return 0;
    const st = await this.readState();
    const dv = await this.readDerived();
    const stu = new Uint32Array(st.buffer, st.byteOffset, st.length);
    const PW = PARTICLE_WORDS, DW = DERIVED_WORDS;

    const survivors = [];
    for (let i = 0; i < this.nFluid; i++) {
      const o = i * PW;
      const ok = keep({
        i, x: st[o], y: st[o + 1], vx: st[o + 2], vy: st[o + 3],
        mat: stu[o + 4], flags: stu[o + 5], temp: st[o + 6], fuel: st[o + 7],
      });
      if (ok) survivors.push(i);
    }
    const removed = this.nFluid - survivors.length;
    if (removed === 0) return 0;

    const nB = this.n - this.nFluid;
    const newFluid = survivors.length;
    const newN = newFluid + nB;

    // Permute BOTH state and derived. Dropping derived instead (and letting the
    // next substep rebuild it) loses stAcc, which is applied one substep late —
    // the visible result is a one-frame surface-tension flinch across the whole
    // fluid every time the user erases anything.
    const outSt = new Int32Array(newN * PW);
    const outStu = new Uint32Array(outSt.buffer);
    const outDv = new Int32Array(newN * DW);
    const copy = (src, dst) => {
      for (let w = 0; w < PW; w++) outStu[dst * PW + w] = stu[src * PW + w];
      for (let w = 0; w < DW; w++) outDv[dst * DW + w] = dv[src * DW + w];
    };
    for (let k = 0; k < newFluid; k++) copy(survivors[k], k);
    for (let b = 0; b < nB; b++) copy(this.nFluid + b, newFluid + b);

    // §18.7 R4. PERMUTE THE BOND CONTENTS, not just the rows. eraseWhere is the
    // only thing in the engine that renumbers a live particle, so a bond word
    // holding a raw slot index is stale the instant it returns. Without the
    // content remap: 307 mis-pointed bonds measured. With it: 0 dangling
    // partners, 0 one-sided bonds, graph isomorphic to the survivor-induced
    // subgraph. Tombstones are dropped here too — they are already broken, and
    // carrying them would waste row slots across the compaction.
    const oldNFluid = this.nFluid;
    const bondsIn = [await this.readBond(), await this.readBondCand()];
    const bondsOut = [new Uint32Array(newN * BOND_CAP), new Uint32Array(newN * BOND_CAP)];
    const oldToNew = new Int32Array(oldNFluid).fill(-1);
    for (let k = 0; k < newFluid; k++) oldToNew[survivors[k]] = k;
    for (let b = 0; b < 2; b++) {
      const src = bondsIn[b], dst = bondsOut[b];
      for (let k = 0; k < newFluid; k++) {
        const so = survivors[k] * BOND_CAP, doo = k * BOND_CAP;
        let c = 0;
        for (let q = 0; q < BOND_CAP; q++) {
          const w = src[so + q];
          if (w === 0) break;                                   // rows are dense
          if (w & BOND_TOMB) continue;                          // drop tombstones
          const nj = oldToNew[w & BOND_J_MASK];
          if (nj < 0) continue;                                 // partner erased: DROP
          dst[doo + c++] = ((w & BOND_L0_MASK) | nj) >>> 0;
        }
        // the tail of `dst` is already 0 (fresh Uint32Array), which is the
        // zero-fill the dense-row contract needs.
      }
    }
    // §20. PERMUTE THE REST-OFFSET SIDECAR, and repair the body records.
    //
    // The sidecar is indexed by SLOT, exactly like `bond`, so it is stale the
    // instant eraseWhere returns — a member would be placed from another
    // particle's rest offset, which is a silent, permanent deformation of a
    // shape that is supposed to be incapable of deforming. That is gate G8's
    // defeat case.
    //
    // NOTHING IS RE-CAPTURED HERE EITHER. The surviving members keep the exact
    // offsets they had; only the centroid moved, and `rigidMember`'s re-base
    // repairs that on the next substep by an exact integer subtraction. The host
    // recomputes n / sumIs2 / gSum / invI only because `rigidStep` runs BEFORE
    // `rigidMember` and would otherwise spend one substep with a stale inertia.
    // `rsh`, `ish` and `osh` are NEVER recomputed — they were sized from the
    // FULL member list at weld time precisely so they stay valid for every
    // member set the body can ever have.
    if (this._nBodies > 0) {
      const recs = await this.readBodies();
      const restIn = await this.readRestOffsets();
      const oldDrivers = this._bodyDrivers;
      const restOut = new Int32Array(Math.max(newN, oldNFluid) * 2);
      for (let k = 0; k < newFluid; k++) {
        restOut[k * 2] = restIn[survivors[k] * 2];
        restOut[k * 2 + 1] = restIn[survivors[k] * 2 + 1];
      }
      const kept = [];
      for (const b of this._bodies) {
        const mem = [];
        for (let k = 0; k < newFluid; k++) {
          const f = outStu[k * PW + 5];
          if ((f & FLAG_RIGID) !== 0 && (f & WELD_TAG_MASK) === (b.tag >>> 0)) mem.push(k);
        }
        if (mem.length < RIGID_MIN_MEMBERS) {
          // The body is gone. Its survivors become ordinary free particles where
          // they are, with the velocity they had, and the record must be cleared
          // or a ghost integrates `theta` forever.
          for (const k of mem) outStu[k * PW + 5] = (outStu[k * PW + 5] & ~FLAG_RIGID) >>> 0;
          continue;
        }
        const src = b.index * RG_BODY_WORDS;
        const rec = recs.slice(src, src + RG_BODY_WORDS);
        const ish = rec[RB.ISH];
        let sumIs2 = 0, gSum = 0;
        for (const k of mem) {
          const a = restOut[k * 2] >> ish, c = restOut[k * 2 + 1] >> ish;
          sumIs2 += a * a + c * c;
          gSum += this.gravOf(outStu[k * PW + 4]);   // §26.C, as at weld time
        }
        const { invI, iq } = rigidInvInertia(sumIs2, rec[RB.RSH], ish);
        rec[RB.FIRST] = mem[0]; rec[RB.LAST] = mem[mem.length - 1];
        rec[RB.N] = mem.length; rec[RB.SUMIS2] = sumIs2; rec[RB.GSUM] = gSum;
        rec[RB.INVI] = invI; rec[RB.IQ] = iq;
        // §24. `rec` is copied WHOLE, so RB_USER rides through an erase with
        // the rest of the record — a body you were driving is still yours after
        // you cut a piece off it. The host mirror has to be told the same thing
        // explicitly, because it is rebuilt rather than copied.
        kept.push({
          desc: { tag: b.tag, first: mem[0], last: mem[mem.length - 1], n: mem.length,
            index: kept.length, axis: !!b.axis, user: !!b.user },
          oldIndex: b.index,
          rec,
        });
      }
      const packed = new Int32Array(Math.max(1, kept.length) * RG_BODY_WORDS);
      for (let k = 0; k < kept.length; k++) packed.set(kept[k].rec, k * RG_BODY_WORDS);
      if (kept.length > 0) this.device.queue.writeBuffer(this.buf.rigid, RG_BODY_BASE * 4, packed, 0, kept.length * RG_BODY_WORDS);
      if (kept.length < this._nBodies) {
        const dead = new Int32Array((this._nBodies - kept.length) * RG_BODY_WORDS);
        this.device.queue.writeBuffer(this.buf.rigid, (RG_BODY_BASE + kept.length * RG_BODY_WORDS) * 4, dead);
      }
      this.device.queue.writeBuffer(this.buf.rigid, RG_REST_BASE * 4, restOut);
      this._bodies = kept.map((x) => x.desc);
      this._nBodies = kept.length;
      this._bodyDrivers = new Map();
      for (let k = 0; k < kept.length; k++) {
        const driver = oldDrivers.get(kept[k].oldIndex);
        if (driver) this._bodyDrivers.set(k, driver);
      }
      // Driver records are time-varying and are re-evaluated before the next
      // substep; clear the old compact-index image so a diagnostic read between
      // erase and step cannot see a dead body's command.
      this.device.queue.writeBuffer(
        this.buf.rigid,
        RG_DRIVER_BASE * 4,
        new Int32Array(RG_MAX_BODIES * RG_DRIVER_WORDS));
      this._writeNBodies();
    }

    // §21. PERMUTE THE COLOUR SIDECAR. It is indexed by SLOT, so it is stale the
    // instant eraseWhere returns — and the failure is the loudest one in this
    // feature: erase half an imported photograph and every surviving particle
    // wears a neighbour's pixel. Same class as the bond contents above and the
    // §20 rest offsets below, and it is written over the FULL old extent so the
    // rows this compaction just turned into boundary or unused slots go back to
    // "no colour" rather than keeping a dead particle's pigment.
    {
      const tIn = await this.readTints(Math.max(newN, oldNFluid));
      const tOut = new Uint32Array(Math.max(newN, oldNFluid));
      for (let k = 0; k < newFluid; k++) tOut[k] = tIn[survivors[k]];
      this.device.queue.writeBuffer(this.buf.tint, 0, tOut);
    }

    const stateBuf = this.parity === 0 ? this.buf.stateA : this.buf.stateB;
    this.device.queue.writeBuffer(stateBuf, 0, outSt);
    this.device.queue.writeBuffer(this.buf.derived, 0, outDv);
    // Write the FULL old extent, so rows in [newNFluid, oldNFluid) — which
    // eraseWhere has just turned into boundary or unused slots — are zeroed
    // rather than left holding a previous configuration's bonds. That is
    // PLATFORM_NOTES trap #7 and it is the same hazard §18.8's n_fluid guard
    // defends on the hash side.
    {
      const wipe = Math.max(newN, oldNFluid) * BOND_CAP;
      const padded0 = new Uint32Array(wipe); padded0.set(bondsOut[0]);
      const padded1 = new Uint32Array(wipe); padded1.set(bondsOut[1]);
      this.device.queue.writeBuffer(this.buf.bond, 0, padded0);
      this.device.queue.writeBuffer(this.buf.bondCand, 0, padded1);
    }
    this.device.queue.writeBuffer(this.buf.params, 0, new Uint32Array([newN, newFluid]));
    this.device.queue.writeBuffer(this.buf.hashParams, 0, new Uint32Array([newN, PARTICLE_WORDS, HASHED_WORDS, newFluid]));
    this.n = newN; this.nFluid = newFluid;
    this._bondDirty = true;
    // Every surviving particle was renumbered, so a caller replaying "the
    // spawns the mirror is missing" would be replaying nonsense. Retire the
    // whole seq window: the next mirror is the first trustworthy one.
    this._mirror = null;
    this._mirrorPending = false;
    this._mirrorEpoch++;
    this._spawnSeq++; this._mirrorSeq = this._spawnSeq;
    // §17 / bonds. THIS CALL IS THE ONLY THING IN THE ENGINE THAT RENUMBERS A
    // LIVE PARTICLE, and `survivors` IS the permutation: survivors[k] is the OLD
    // slot of the particle that now lives in slot k. It was being computed and
    // thrown away. Anything holding raw slot indices — the bonded-body design is
    // the immediate case, but the reaction split is keyed to the slot too
    // (README known gap 7) — can now remap in the same call instead of
    // reconstructing a permutation it cannot see. Published, not returned, so
    // the existing `Promise<number>` contract does not move.
    this._lastPermutation = survivors;
    this._eraseSeq = (this._eraseSeq | 0) + 1;
    return removed;
  }

  /**
   * §30 — one deterministic Inflow/Outflow authoring tick.
   *
   * I/O particles are genuine simulated matter; this service only performs the
   * two operations the solver cannot own reproducibly: slot allocation and
   * compaction. Callers schedule the tick at a recorded substep boundary, just
   * like spawnFluid/eraseWhere. One connected Inflow region emits at most one
   * joint-less particle per call, and only while touched by non-Inflow matter.
   * Outflow absorbs contacting particles iff their 26-bit ingredient sets have
   * an empty intersection.
   *
   * @returns {Promise<{generated:number,absorbed:number,inflows:number,outflows:number}>}
   */
  async serviceMatterIO() {
    if (this.nFluid === 0) return { generated: 0, absorbed: 0, inflows: 0, outflows: 0 };
    const st = await this.readState();
    const u = new Uint32Array(st.buffer, st.byteOffset, st.length);
    const PW = PARTICLE_WORDS, nf = this.nFluid;
    const CELL = 2 * ONE;
    const CONTACT2 = Math.round(1.65 * ONE) ** 2;
    const LINK2 = Math.round(1.55 * ONE) ** 2;
    const CLEAR2 = Math.round(0.72 * ONE) ** 2;
    const key = (x, y) => `${Math.floor(x / CELL)},${Math.floor(y / CELL)}`;
    const grid = new Map();
    const addGrid = (i, x, y) => {
      const k = key(x, y);
      let a = grid.get(k); if (!a) { a = []; grid.set(k, a); }
      a.push(i);
    };
    for (let i = 0; i < nf; i++) addGrid(i, st[i * PW], st[i * PW + 1]);
    const near = (x, y) => {
      const cx = Math.floor(x / CELL), cy = Math.floor(y / CELL), out = [];
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const a = grid.get(`${cx + dx},${cy + dy}`);
        if (a) out.push(...a);
      }
      return out;
    };
    const io = new Array(nf);
    const inflow = [], outflow = [];
    for (let i = 0; i < nf; i++) {
      const o = i * PW;
      const q = this.matterIOInfo(u[o + 4], u[o + 5]);
      io[i] = q;
      if (q?.inflow) inflow.push(i);
      if (q?.outflow) outflow.push(i);
    }
    if (!inflow.length && !outflow.length) {
      return { generated: 0, absorbed: 0, inflows: 0, outflows: 0 };
    }

    // OUTFLOW. Slot ids are selected from this exact snapshot, then compacted
    // by eraseWhere's canonical index-order permutation.
    const victims = new Set();
    for (const i of outflow) {
      const oi = i * PW, sx = st[oi], sy = st[oi + 1], sm = io[i].ingredients;
      for (const j of near(sx, sy)) {
        if (j === i || victims.has(j)) continue;
        const oj = j * PW, dx = st[oj] - sx, dy = st[oj + 1] - sy;
        if (dx * dx + dy * dy > CONTACT2) continue;
        const tm = this.materialElementMask(u[oj + 4], u[oj + 5]);
        if ((sm & tm) === 0) victims.add(j);
      }
    }

    // INFLOW. Components are per source row, so two touching recipes remain
    // two independently reproducible sources.
    const sourceSet = new Set(inflow);
    const seen = new Set(), spawns = [];
    const DIRS = [[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]];
    const occupied = (x, y) => {
      for (const j of near(x, y)) {
        if (victims.has(j)) continue;
        const o = j * PW, dx = st[o] - x, dy = st[o + 1] - y;
        if (dx * dx + dy * dy < CLEAR2) return true;
      }
      for (const p of spawns) {
        const dx = p.x - x, dy = p.y - y;
        if (dx * dx + dy * dy < CLEAR2) return true;
      }
      return false;
    };
    for (const root of inflow) {
      if (seen.has(root) || victims.has(root)) continue;
      const mat = u[root * PW + 4], comp = [], q = [root];
      seen.add(root);
      for (let qi = 0; qi < q.length; qi++) {
        const i = q[qi], oi = i * PW; comp.push(i);
        for (const j of near(st[oi], st[oi + 1])) {
          if (!sourceSet.has(j) || seen.has(j) || u[j * PW + 4] !== mat) continue;
          const oj = j * PW, dx = st[oj] - st[oi], dy = st[oj + 1] - st[oi + 1];
          if (dx * dx + dy * dy <= LINK2) { seen.add(j); q.push(j); }
        }
      }
      const entry = this._internByRow.get(mat);
      if (!entry || (entry.bits & MOD_BITS.IO_NULL)) continue; // pure I emits Null
      let touch = null, source = -1, best2 = Infinity;
      for (const i of comp) {
        const oi = i * PW;
        for (const j of near(st[oi], st[oi + 1])) {
          if (sourceSet.has(j) || victims.has(j)) continue;
          const oj = j * PW, dx = st[oj] - st[oi], dy = st[oj + 1] - st[oi + 1];
          const d2 = dx * dx + dy * dy;
          if (d2 <= CONTACT2 && (d2 < best2 || (d2 === best2 && j < touch))) {
            best2 = d2; touch = j; source = i;
          }
        }
      }
      if (touch === null) continue;
      const outBits = entry.bits & ~(MOD_BITS.I | MOD_BITS.IO_NULL);
      const outRecipe = (entry.recipeMask ?? this.materialElementMask(mat, u[source * PW + 5])) &
        ~elementMask('IWRUN');
      let outMat = entry.base;
      if (outBits !== 0) {
        const r = this.internMaterial(entry.base, outBits, {
          recipeMask: (outBits & MOD_BITS.O) ? outRecipe : undefined,
        });
        outMat = r.rowId;
      }
      const os = source * PW, ot = touch * PW;
      const tx = st[ot] - st[os], ty = st[ot + 1] - st[os + 1];
      const dirs = DIRS.slice().sort((a, b) => {
        const da = a[0] * tx + a[1] * ty, db = b[0] * tx + b[1] * ty;
        return db - da;
      });
      for (const [dx, dy] of dirs) {
        const scale = (dx && dy) ? 79000 : 106000; // ~1.6 wu in either case
        const x = st[os] + Math.round(dx * scale);
        const y = st[os + 1] + Math.round(dy * scale);
        if (x < ONE || y < ONE || x > this.domW - ONE || y > this.domH - ONE) continue;
        if (occupied(x, y)) continue;
        spawns.push({ x, y, vx: dx * 1200, vy: dy * 1200, mat: outMat });
        break;
      }
    }

    const absorbed = victims.size
      ? await this.eraseWhere(({ i }) => !victims.has(i))
      : 0;
    const room = Math.max(0, this.maxParticles - this.n);
    if (spawns.length > room) spawns.length = room;
    if (spawns.length) this.spawnFluid(spawns, { ownGroup: true, noWeld: true });
    return {
      generated: spawns.length, absorbed,
      inflows: inflow.length, outflows: outflow.length,
    };
  }

  /**
   * §17. How many slots currently hold RETIRED matter, from the host mirror.
   *
   * Free: `_mirror` is a whole-state readback index.html already refreshes every
   * frame and it already carries matId. It is a snapshot of a world one or two
   * frames old (see refreshMirror), which is exactly the right precision for a
   * reaper trigger — being late by two frames costs nothing, and the alternative
   * is a GPU sync per frame to count corpses.
   *
   * Returns null when there is no mirror yet, so a caller can tell "none" from
   * "do not know" instead of being handed a confident 0.
   */
  retiredCount() {
    const mir = this._mirror;
    if (!mir) return null;
    const iPhantom = MAT_FIELDS.indexOf('phantom');
    const nf = Math.min(this.nFluid, (mir.length / PARTICLE_WORDS) | 0);
    const mu = new Uint32Array(mir.buffer, mir.byteOffset, mir.length);
    let k = 0;
    for (let i = 0; i < nf; i++) {
      const row = this.mats[mu[i * PARTICLE_WORDS + 4]];
      if (row && row[iPhantom] !== 0) k++;
    }
    return k;
  }

  /**
   * §17. THE REAPER. Retirement is in place, so corpses keep their slots and
   * `maxParticles` fills up with them on a long smoky session — a worse bug than
   * the one venting fixes. This is the collector that already exists, pointed at
   * every phantom row rather than at one material id, so it does not have to
   * know that VOID is called VOID.
   *
   * It costs a full GPU sync and it RENUMBERS (eraseWhere always did; see
   * _lastPermutation). Call it on a count threshold, never per frame.
   */
  reapRetired() {
    const iPhantom = MAT_FIELDS.indexOf('phantom');
    return this.eraseWhere(({ mat }) => {
      const row = this.mats[mat];
      return !row || row[iPhantom] === 0;
    });
  }

  /** Eraser convenience: delete every fluid particle inside a circle (ticks). */
  eraseCircle(cx, cy, radius) {
    const r2 = radius * radius;
    return this.eraseWhere(({ x, y }) => {
      const dx = x - cx, dy = y - cy;
      return dx * dx + dy * dy > r2;
    });
  }

  /** Collector convenience: delete every fluid particle of a given material. */
  eraseMaterial(matId) { return this.eraseWhere(({ mat }) => mat !== matId); }

  currentStateBuffer() { return this.parity === 0 ? this.buf.stateA : this.buf.stateB; }

  destroy() {
    for (const k of Object.keys(this.buf)) { try { this.buf[k].destroy(); } catch { /* already gone */ } }
    this._pipes.clear();
  }
}

// ============================================================================
// Liveness (§7.7)
// ============================================================================
export function computeLiveness({ state, derived, nbrN, cellStart, n, nFluid, domW, domH,
                                  initialPos, recentPos = null }) {
  const I32_MAX = 2147483647, I32_MIN = -2147483648;
  let ke = 0n;
  let moved = 0, recentMoved = 0, outside = 0, saturated = 0;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  let vxSum = 0, vySum = 0;

  for (let i = 0; i < n; i++) {
    const o = i * PARTICLE_WORDS;
    const x = state[o], y = state[o + 1], vx = state[o + 2], vy = state[o + 3];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (i < nFluid) {
      ke += BigInt(vx) * BigInt(vx) + BigInt(vy) * BigInt(vy);
      vxSum += vx; vySum += vy;
      if (initialPos && (x !== initialPos[i * 2] || y !== initialPos[i * 2 + 1])) moved++;
      // Movement over a RECENT window. `moved` is measured against the INITIAL
      // positions, so once a particle has ever moved it latches at 1.0 forever
      // and a freeze becomes undetectable. This one can fall back to 0.
      if (recentPos && (x !== recentPos[i * 2] || y !== recentPos[i * 2 + 1])) recentMoved++;
    }
    if (i < nFluid && (x < 0 || y < 0 || x > domW || y > domH)) outside++;
    for (let w = 0; w < 4; w++) {
      const v = state[o + w];
      if (v === I32_MAX || v === I32_MIN) saturated++;
    }
  }

  // KE about the MEAN velocity. This is the rigid-body rejector: it is exactly
  // zero for a block drifting uniformly (which total KE reports as maximally
  // alive) and near zero for a frozen lattice with a uniform jitter. Total KE
  // alone accepted both of those as "live" before 2026-07-25.
  const vxBar = nFluid ? vxSum / nFluid : 0, vyBar = nFluid ? vySum / nFluid : 0;
  let keAboutMean = 0;
  for (let i = 0; i < nFluid; i++) {
    const o = i * PARTICLE_WORDS;
    const dx = state[o + 2] - vxBar, dy = state[o + 3] - vyBar;
    keAboutMean += dx * dx + dy * dy;
  }
  const internalKePerParticle = nFluid ? keAboutMean / nFluid : 0;
  // v_rms of the PECULIAR velocity, in wu/substep — directly comparable to the
  // §8.5 A2 band, which is the only threshold in the project with a physical
  // meaning attached to residual motion.
  const internalVrms = Math.sqrt(internalKePerParticle) / ONE;

  let rhoSum = 0, rhoMax = 0, rhoCount = 0;
  const rhoVals = [];
  if (derived && derived.length >= nFluid * DERIVED_WORDS) {
    for (let i = 0; i < nFluid; i++) {
      const r = derived[i * DERIVED_WORDS + 10] / RHO0;
      rhoVals.push(r);
      rhoSum += r; rhoCount++;
      if (r > rhoMax) rhoMax = r;
    }
  }
  const meanDensity = rhoCount ? rhoSum / rhoCount : 0;
  let densVar = 0;
  for (const r of rhoVals) densVar += (r - meanDensity) * (r - meanDensity);
  const densityStd = rhoCount ? Math.sqrt(densVar / rhoCount) : 0;

  let maxNbr = 0, nbrSum = 0;
  if (nbrN) for (let i = 0; i < nFluid; i++) { const c = nbrN[i]; nbrSum += c; if (c > maxNbr) maxNbr = c; }

  let maxCellOccupancy = 0;
  if (cellStart) {
    for (let c = 0; c + 1 < cellStart.length; c++) {
      const occ = cellStart[c + 1] - cellStart[c];
      if (occ > maxCellOccupancy && occ < 0x7fffffff) maxCellOccupancy = occ;
    }
  }

  // §12 thermal instrumentation. `tempSum` is a BigInt because the whole point
  // of the conduction proof is that the TOTAL is conserved exactly, and a
  // float sum of 20 000 values near 2^29 loses the low bits that carry the
  // claim. matHistogram is how a phase-change test counts particles without
  // re-reading the state buffer.
  let tempSum = 0n, tempMin = Infinity, tempMax = -Infinity, burning = 0;
  const matHistogram = {};
  for (let i = 0; i < nFluid; i++) {
    const o = i * PARTICLE_WORDS;
    const t = state[o + 6];
    tempSum += BigInt(t);
    if (t < tempMin) tempMin = t;
    if (t > tempMax) tempMax = t;
    const m = state[o + 4] >>> 0;
    matHistogram[m] = (matHistogram[m] ?? 0) + 1;
    if ((state[o + 5] >>> 0) & 1) burning++;
  }

  const keNum = Number(ke);
  return {
    particleCount: n,
    fluidCount: nFluid,
    tempSum: tempSum.toString(),
    tempMean: nFluid ? Number(tempSum) / nFluid : 0,
    tempMin: nFluid ? tempMin : 0,
    tempMax: nFluid ? tempMax : 0,
    burningCount: burning,
    matHistogram,
    kineticEnergy: ke.toString(),
    kineticEnergyPerParticle: nFluid ? keNum / nFluid : 0,
    internalKePerParticle,
    internalVrms,
    meanVelocity: [vxBar, vyBar],
    meanDensity, maxDensity: rhoMax, densityStd,
    movedFraction: nFluid ? moved / nFluid : 0,
    recentMovedFraction: recentPos ? (nFluid ? recentMoved / nFluid : 0) : null,
    outsideDomain: outside,
    maxNbr, meanNbr: nFluid ? nbrSum / nFluid : 0,
    bbox: [minX, minY, maxX, maxY],
    nonFinite: saturated,
    maxCellOccupancy,
    restDensity: 1.0,
  };
}

/**
 * Turn a Liveness record into a list of violations. MUST be non-vacuous: run it
 * against an all-zero state and it has to return >= 3 entries, otherwise a
 * "stable zeros" run would pass the determinism gate (PLATFORM_NOTES trap #5).
 *
 * REWRITTEN 2026-07-25. The previous version had no teeth against the two
 * canonical degenerate states and was measured accepting BOTH of them:
 *   ACCEPTED  a rigid block drifting at uniform velocity  (zero internal dynamics,
 *             KE/particle 1e6, movedFraction 1.000)
 *   ACCEPTED  a frozen crystal with one tick of jitter    (KE/particle 1.00)
 * Causes and fixes:
 *   1. movedFraction was measured against the INITIAL positions, so it latches
 *      at 1.0 the first time anything moves and can NEVER report a freeze.
 *      -> recentMovedFraction, measured over a window (needs `recentPos`).
 *   2. minKePerParticle = 1 corresponds to v_rms = 1.5e-5 wu/substep. The real
 *      dam break sits at ~1e6, so the floor had six orders of magnitude of
 *      slack. -> gate on internalVrms against the §8.5 A2 band instead.
 *   3. Total KE cannot distinguish a fluid from a thrown brick.
 *      -> internalKePerParticle: KE about the MEAN velocity, exactly zero for a
 *      uniformly drifting block.
 *   4. densityRange/[maxDensityStd] were ~6x looser than the project's OWN A1
 *      acceptance thresholds, so the gate's density check could not fail on
 *      anything A1 would reject. -> defaults are now A1's numbers. Transient or
 *      flowing scenes must pass an explicit relaxed `expect`; see DAMBREAK_EXPECT.
 */
export function checkLiveness(L, expect = {}) {
  const {
    minMovedFraction = 0.5,
    minRecentMovedFraction = 0.5,
    densityRange = [0.98, 1.03],           // §8.5 A1
    maxDensityStd = 0.035,                 // §8.5 A1
    minKePerParticle = 1,
    // A frozen lattice and a rigid drift both land at ~0 here. The shipped
    // solver's SETTLED floor is v_rms ~ 0.003 (goo) to ~0.028 (water)
    // wu/substep, i.e. internalKePerParticle ~ 4e4 to ~3.4e6.
    minInternalKePerParticle = 1000,       // v_rms >= 4.8e-4 wu/substep
    internalVrmsRange = [1e-4, 0.5],
    maxOutside = 0,
    nbrRange = [4, MAXNBR],
    minBboxSpan = ONE,
  } = expect;

  const v = [];
  if (L.particleCount <= 0) v.push(`particleCount ${L.particleCount} <= 0`);
  if (L.movedFraction < minMovedFraction)
    v.push(`movedFraction ${L.movedFraction.toFixed(4)} < ${minMovedFraction} (particles are not moving)`);
  if (L.recentMovedFraction !== null && L.recentMovedFraction !== undefined &&
      L.recentMovedFraction < minRecentMovedFraction)
    v.push(`recentMovedFraction ${L.recentMovedFraction.toFixed(4)} < ${minRecentMovedFraction} ` +
           `(state is FROZEN — it moved once and then stopped)`);
  if (!(L.meanDensity >= densityRange[0] && L.meanDensity <= densityRange[1]))
    v.push(`meanDensity ${L.meanDensity.toFixed(4)} outside [${densityRange[0]}, ${densityRange[1]}]`);
  if (L.densityStd > maxDensityStd)
    v.push(`densityStd ${L.densityStd.toFixed(4)} > ${maxDensityStd}`);
  if (L.kineticEnergyPerParticle < minKePerParticle)
    v.push(`kineticEnergyPerParticle ${L.kineticEnergyPerParticle} < ${minKePerParticle} (state is frozen)`);
  if (L.internalKePerParticle < minInternalKePerParticle)
    v.push(`internalKePerParticle ${Number(L.internalKePerParticle).toFixed(1)} < ` +
           `${minInternalKePerParticle} (no INTERNAL dynamics — a rigid body drifting, or a ` +
           `frozen lattice; total KE cannot tell those from a fluid)`);
  if (!(L.internalVrms >= internalVrmsRange[0] && L.internalVrms <= internalVrmsRange[1]))
    v.push(`internalVrms ${Number(L.internalVrms).toExponential(3)} wu/substep outside ` +
           `[${internalVrmsRange[0]}, ${internalVrmsRange[1]}]`);
  if (L.outsideDomain > maxOutside)
    v.push(`outsideDomain ${L.outsideDomain} > ${maxOutside}`);
  if (L.maxNbr < nbrRange[0])
    v.push(`maxNbr ${L.maxNbr} < ${nbrRange[0]} (no neighbours found — grid is dead)`);
  if (L.maxNbr > nbrRange[1])
    v.push(`maxNbr ${L.maxNbr} > ${nbrRange[1]} (neighbour list overflowed)`);
  if (L.nonFinite > 0)
    v.push(`${L.nonFinite} saturated i32 words in state (I32_MAX/I32_MIN)`);
  const spanX = L.bbox[2] - L.bbox[0], spanY = L.bbox[3] - L.bbox[1];
  if (!(spanX >= minBboxSpan && spanY >= minBboxSpan))
    v.push(`bbox span (${spanX}, ${spanY}) collapsed below ${minBboxSpan} ticks`);
  return v;
}

/**
 * The relaxation a TRANSIENT / FLOWING scene needs against the settled-pool
 * defaults. dambreak_v1 at 10000 substeps measures meanDensity 1.0104 and
 * densityStd 0.0477: the mean is inside A1's band, the sigma is 1.4x over it,
 * because A1 describes a SETTLED POOL and a dam break still has a spreading
 * free surface and airborne fragments. That is a scene difference, not a defect
 * — but it must be stated explicitly at the call site rather than hidden by
 * baking the loosest band in as the default for everything, which is what the
 * old [0.90, 1.15] / 0.20 defaults did.
 */
export const DAMBREAK_EXPECT = {
  densityRange: [0.98, 1.04],
  maxDensityStd: 0.055,
};

/** The zero-state control: proves checkLiveness is not vacuous. */
export function zeroStateLiveness(n = 4096, nFluid = 3600) {
  const L = computeLiveness({
    state: new Int32Array(n * PARTICLE_WORDS),
    derived: new Int32Array(n * DERIVED_WORDS),
    nbrN: new Uint32Array(n),
    cellStart: null,
    n, nFluid, domW: 128 * ONE, domH: 72 * ONE,
    initialPos: new Int32Array(n * 2),
    recentPos: new Int32Array(n * 2),
  });
  return { liveness: L, violations: checkLiveness(L) };
}

/**
 * The DEGENERATE-STATE controls. checkLiveness must reject every one of these.
 * They are exported so the gate can assert it, because all three of the first
 * four were ACCEPTED as live before 2026-07-25.
 */
export function degenerateStateControls(n = 1024) {
  const mk = (fill) => {
    const state = new Int32Array(n * PARTICLE_WORDS);
    const derived = new Int32Array(n * DERIVED_WORDS);
    const nbrN = new Uint32Array(n);
    const initialPos = new Int32Array(n * 2);
    const recentPos = new Int32Array(n * 2);
    fill({ state, derived, nbrN, initialPos, recentPos });
    return computeLiveness({ state, derived, nbrN, cellStart: null, n, nFluid: n,
                             domW: 128 * ONE, domH: 128 * ONE, initialPos, recentPos });
  };
  const lattice = (o, i, jitter, vx, vy) => {
    const gx = i % 32, gy = (i / 32) | 0;
    o.initialPos[i * 2] = gx * ONE; o.initialPos[i * 2 + 1] = gy * ONE;
    o.state[i * PARTICLE_WORDS] = (gx + 30) * ONE + jitter;
    o.state[i * PARTICLE_WORDS + 1] = (gy + 30) * ONE + jitter;
    o.recentPos[i * 2] = o.state[i * PARTICLE_WORDS] - jitter;
    o.recentPos[i * 2 + 1] = o.state[i * PARTICLE_WORDS + 1] - jitter;
    o.state[i * PARTICLE_WORDS + 2] = vx; o.state[i * PARTICLE_WORDS + 3] = vy;
    o.derived[i * DERIVED_WORDS + 10] = RHO0;
    o.nbrN[i] = 20;
  };
  const cases = {
    'rigid block drifting uniformly': mk((o) => { for (let i = 0; i < n; i++) lattice(o, i, 1, 1000, 0); }),
    'frozen crystal, 1-tick jitter':  mk((o) => { for (let i = 0; i < n; i++) lattice(o, i, 1, 1, 0); }),
    'frozen solid (no motion at all)': mk((o) => { for (let i = 0; i < n; i++) lattice(o, i, 0, 0, 0); }),
    'all stacked at one point':       mk((o) => { for (let i = 0; i < n; i++) {
      o.state[i * PARTICLE_WORDS] = 64 * ONE; o.state[i * PARTICLE_WORDS + 1] = 64 * ONE;
      o.state[i * PARTICLE_WORDS + 2] = 8; o.derived[i * DERIVED_WORDS + 10] = RHO0; o.nbrN[i] = 20; } }),
  };
  const out = {};
  for (const [k, L] of Object.entries(cases)) out[k] = { liveness: L, violations: checkLiveness(L) };
  return out;
}

export { digestState, foldChain, newChain, hex8 };
