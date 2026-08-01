<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# SunaEngine

**Bit-exact particle physics.** The real [SunaBox](https://github.com/DARIENBATHALTER/sunabox) engine — its deterministic, water-capable core — extracted unmodified and published under the AGPL.

Live demos: **[engine.sunabox.dev](http://engine.sunabox.dev)** — two side-by-side simulations proving hash equality live, and a record/replay pool with deterministic scrubbing.

## What makes it different?

Mainstream physics engines compute in floating point, and floating-point results drift between GPUs, drivers, and compiler optimization levels — the same simulation on two machines can settle differently.

SunaEngine's simulation state is **signed 32-bit fixed point (Q16.16), integer arithmetic only**. Same scene, same input schedule → the same 256-bit state hash, on any WebGPU device. Determinism is not a tolerance; it is byte equality.

## Quick start

```bash
git clone https://github.com/DARIENBATHALTER/sunaengine.git
cd sunaengine
python3 -m http.server 8080
# open http://localhost:8080 in a WebGPU browser (Chrome/Edge 113+)
```

No build step, no npm, no external dependencies — the page is self-contained.

## Architecture

```
index.html                 — demo page (twins + replay pool), thin harness UI
src/
  engine.js                — WebGPU host: pipelines, buffers, fixed-substep scheduler
  sim.wgsl                 — 2D deterministic PBF solver (25 compute entry points)
  fixed.wgsl               — Q16.16 fixed-point numeric core (RNE rounding, saturating)
  hash/statehash.js        — canonical state hash, JS reference implementation
  hash/statehash.wgsl      — the same hash on the GPU (bit-identical by construction)
  render.js                — read-only renderer over the sim state
  render_diag.wgsl.js      — diagnostic render shader (divergence tint)
  suna.js / sunar.js       — scene container and replay codec
  tables.js / tables.json  — frozen kernel lookup tables
  demo_boot.js             — demo harness (original to this repo)
test/
  site_gates.mjs           — the demo-page gates (see below)
  extraction_gate.html/.md — cross-repo extraction proof
```

Every `src/` engine file is extracted **unmodified** from SunaBox commit `e41e06f`; each file's header carries the exact `diff` command that verifies it byte-for-byte against the source repo. `demo_boot.js` and `index.html` are the only original code here.

## Extraction proof

The claim "this is the real engine" is gated, not asserted. The frozen determinism scene (`dambreak_v1`: 4096 particles, 3600 fluid) stepped 1000 substeps must produce the same hash chain in **both** repos:

```
node test/site_gates.mjs g5
chain 20918f97fbcbb409e7d058ebcae6f6725973b9a19042168742b07e8c17126d14
```

`test/site_gates.mjs` (requires a local Chrome) runs the full suite:

- **g1 twins-equal** — two sims, same seed, hashes equal live
- **g2 twins-under-input** — identical injection schedule under held input, still equal, 3000-particle cap enforced
- **g3 divergence negative control** — one extra particle in one twin → hashes visibly diverge within substeps
- **g4 replay byte-identity** — record → replay reproduces the exact recorded digest; scrubbing is deterministic in both directions
- **g5 extraction cross-repo** — the frozen chain above
- **g6 hygiene** — zero console errors, zero external network requests, license and credit present

## Solver

2D position-based fluids (PBF), entirely in Q16.16:

- density constraints with fixed iteration counts (no data-dependent convergence)
- colour-classed constraint solve: Gauss-Seidel between colour classes, Jacobi within
- XSPH viscosity, cohesion/adhesion via frozen LUTs
- spatial-grid neighbor search, hard boundary walls
- fixed integer timestep — `step(count)` advances exactly `count` substeps; nothing reads a clock

## Determinism guarantee

- all simulation arithmetic is signed 32-bit integer, Q16.16 fixed point
- rounding: round-half-to-even (RNE) everywhere — symmetric, so pair forces are exactly antisymmetric
- overflow: saturating, never wrapping
- no floats, no atomics in physics, no subgroup/wave ops
- state hash: MurmurHash3_x86_32, 4 salted lanes, combined across particles by u32 addition (dispatch-order independent); two digests — `H_set` (permutation-invariant) and `H_slot` (slot-bound, the gate)

## Requirements

- Chrome or Edge 113+ (WebGPU), or any browser with WebGPU enabled

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE). Copyright (C) 2026 Darien Bathalter. Every shipped source file carries its AGPL header.

## Credit

A love letter to **OE-CAKE! by Prometech Software**, the toy that proved a fluid sandbox could feel like play.

Built by [Darien Bathalter](https://github.com/DARIENBATHALTER) — extracted from [SunaBox](https://github.com/DARIENBATHALTER/sunabox), the full particle sandbox.

---

*Determinism is the product.*
