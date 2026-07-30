# ☀️ SunaEngine

**The Bit-Exact Physics Engine** — open-source, WebGPU-native, integer-only particle physics.

## What makes it different?

Every mainstream physics engine uses floating-point math (f32/f64). Floating-point results vary between devices, compilers, and even optimization levels — the same simulation on two different GPUs can produce subtly different outcomes.

**SunaEngine uses integer-only fixed-point arithmetic.** Same input → same 256-bit state hash, guaranteed, on any WebGPU-capable device. This is what we call **bit-determinism**.

## Quick Start

```bash
git clone https://github.com/DARIENBATHALTER/sunaengine.git
cd sunaengine
python3 -m http.server 8080
# Open http://localhost:8080 in Chrome 113+
```

## Architecture

```
src/
  engine.js     — WebGPU host: buffer management, dispatch scheduling
  sim.wgsl      — PBF solver: grid, neighbor search, density constraints
  hash.js       — Deterministic 256-bit state hash (Murmur3, 4-lane)
  common.js     — Shared constants (Q16.16, kernel params)
  tables.js     — Frozen kernel LUT loader
  tables.json   — Precomputed kernel tables (poly6, spiky, cohesion)
```

## Solver

Position-Based Dynamics (PBF) with:
- Density constraints (CFM relaxation via eps)
- XSPH viscosity
- Cohesion / surface tension (Akinci-style)
- 4-Jacobi iteration solve per substep
- Spatial grid neighbor search (~4.0 wu cells)
- Hard boundary walls

## Determinism Guarantee

- All arithmetic is signed 32-bit integer (Q16.16 fixed-point)
- Rounding: round-half-to-even (RNE) everywhere
- Overflow: saturating arithmetic (never wraps)
- No floating-point types, no atomics in physics, no subgroup/wave ops
- State hash: MurmurHash3_x86_32, 4 salted lanes, set + slot digests

## Requirements

- **Chrome 113+** or **Edge 113+** (WebGPU enabled)
- Firefox Nightly with `dom.webgpu.enabled = true`

## License

GNU General Public License v3.0 (GPL-3.0) — see [LICENSE](LICENSE).

SunaEngine is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

## Built by

[Darien Bathalter](https://github.com/DARIENBATHALTER) — extracted from [SunaBox](https://github.com/DARIENBATHALTER/sunabox), the full-featured particle physics sandbox.

---

*"Determinism is the product."*
