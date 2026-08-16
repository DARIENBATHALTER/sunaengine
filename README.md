<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# SunaEngine

Deterministic 2D particle physics for WebGPU. All simulation math is 32-bit integer fixed point. The same scene with the same inputs produces the same 256-bit state hash on any GPU, any driver, any platform.

**Live demos: [engine.sunabox.dev](https://engine.sunabox.dev)**

- **The Twin Simulations**: two independent sims, same scene, same input schedule, never communicating. Stir either one; your input is quantized to substep indices and fed to both. The hashes stay identical, interaction after interaction.
- **[The Pachinko Proof](https://engine.sunabox.dev/float.html)**: an ordinary float solver runs the same pachinko machine twice, summing contact forces in ascending order in one pane and descending in the other. That is the kind of difference two GPUs or two drivers introduce on their own. The balls never finish in the same bins. The integer engine runs the same machine to the same digest, loop after loop.
- **Perfect replay, smooth rewind**: stir a pool, get a byte-verified replay of your session, then drag your own splash backwards through time.
- **[3D](https://engine.sunabox.dev/3d/)**: the same fixed-point method, one dimension up.

## Why integers

Floating-point results differ between GPUs, drivers, and compiler optimization levels. Fused multiply-add contraction, instruction reordering, and parallel accumulation order each flip low bits, and a physics sim amplifies one flipped bit into a different world within seconds. SunaEngine's state is signed 32-bit Q16.16 fixed point, integer arithmetic only, and every order-sensitive sum is either folded in a fixed order or provably unable to overflow. Determinism here is byte equality, not a tolerance.

## What determinism buys

- **Recording and replay.** A `.sunar` file stores the initial state plus the input schedule, indexed by substep. No frames are stored. It plays back bit-identically on any device and scrubs smoothly in both directions.
- **Rewind.** A float sim cannot run backwards onto the world it came from; the errors compound in reverse too. This one can.
- **Lockstep multiplayer.** If every machine computes the same world, only inputs need to cross the wire.

Replay hashes verified matching across Apple M series, iPhone (A series), Steam Deck (AMD), and NVIDIA (RTX 2070 Super). If your device produces a different hash, please open an issue: that is a driver bug worth reporting.

## Quick start

```bash
git clone https://github.com/DARIENBATHALTER/sunaengine.git
cd sunaengine
python3 -m http.server 8080
# open http://localhost:8080 in a WebGPU browser (Chrome/Edge 113+)
```

No build step, no npm, no external dependencies. The page is self-contained.

## Solver

2D position-based fluids (PBF), entirely in Q16.16:

- density constraints with fixed iteration counts (no data-dependent convergence)
- colour-classed constraint solve: Gauss-Seidel between colour classes, Jacobi within
- XSPH viscosity, cohesion and adhesion via frozen lookup tables
- spatial-grid neighbor search, hard boundary walls
- fixed integer timestep: `step(count)` advances exactly `count` substeps; nothing reads a clock

## Determinism rules

- all simulation arithmetic is signed 32-bit integer, Q16.16 fixed point
- rounding is round-half-to-even everywhere, so pair forces are exactly antisymmetric
- overflow saturates, never wraps
- no floats, no atomics in physics, no subgroup/wave ops
- state hash: MurmurHash3_x86_32, 4 salted lanes, combined across particles by u32 addition (dispatch-order independent); two digests, `H_set` (permutation-invariant) and `H_slot` (slot-bound, the gate)

## Architecture

```
index.html                 demo page (twins + replay pool)
float.html                 the pachinko proof (float vs integer, side by side)
3d/                        3D demo (same numeric core, one dimension up)
src/
  engine.js                WebGPU host: pipelines, buffers, fixed-substep scheduler
  sim.wgsl                 2D deterministic PBF solver (25 compute entry points)
  fixed.wgsl               Q16.16 fixed-point numeric core (RNE rounding, saturating)
  hash/statehash.js        canonical state hash, JS reference implementation
  hash/statehash.wgsl      the same hash on the GPU (bit-identical by construction)
  render.js                read-only renderer over the sim state
  suna.js / sunar.js       scene container and replay codec
  tables.js / tables.json  frozen kernel lookup tables
test/
  site_gates.mjs           automated gate suite (see below)
```

## Tests

`node test/site_gates.mjs` (requires a local Chrome) runs the gate suite:

- **g1 twins-equal**: two sims, same seed, hashes equal live
- **g2 twins-under-input**: identical injection schedule under held input, still equal
- **g3 divergence negative control**: one extra particle in one twin makes the hashes visibly diverge within substeps
- **g4 replay byte-identity**: record, then replay reproduces the exact recorded digest; scrubbing is deterministic in both directions
- **g5 frozen determinism chain**: the frozen `dambreak_v1` scene (4096 particles, 1000 substeps) must reproduce chain `20918f97fbcbb409e7d058ebcae6f6725973b9a19042168742b07e8c17126d14`
- **g6 hygiene**: zero console errors, zero external network requests, license and credit present

## Requirements

- Chrome or Edge 113+ (WebGPU), or any browser with WebGPU enabled

## License

GNU Affero General Public License v3.0. See [LICENSE](LICENSE). Copyright (C) 2026 Darien Bathalter.

## Credit

A love letter to **OE-CAKE! by Prometech Software**, the toy that proved a fluid sandbox could feel like play.

Built by [Darien Bathalter](https://github.com/DARIENBATHALTER). SunaEngine powers [SunaBox](https://sunabox.dev), a free browser physics sandbox.
