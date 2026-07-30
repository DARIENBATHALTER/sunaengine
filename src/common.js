// sunaEngine — shared constants
// Stripped to the essentials for demonstrating bit-determinism.

export const SPEC_VERSION = 1;

// World
export const ONE          = 65536;       // 1 wu in ticks (Q16.16)
export const H            = 163840;      // kernel radius 2.5 wu
export const DX0          = 65536;       // rest spacing 1.0 wu
export const RHO0         = 1048576;     // 2^20
export const PRE          = 4;
export const HS           = H >> PRE;
export const H2S          = HS * HS;
export const KSH          = 15;
export const TN           = 3202;
export const CELL_SHIFT   = 18;          // 4.0 wu cells

// Solver
export const ITERS        = 4;
export const MAXNBR       = 48;
export const SUBSTEPS_PER_FRAME = 2;
export const VMAX         = 65536;       // max velocity per component
export const MAX_POS      = 134217728;   // 2^27

// Particle layout — 8 words, 32 bytes
// [pos_x, pos_y, vel_x, vel_y, matId, flags, pad0, pad1]
export const PARTICLE_WORDS = 8;
export const HASHED_WORDS   = 8;

// Domain (default)
export const DOM_W = 128;
export const DOM_H = 72;

// Water material row (the ONLY material in the stripped engine)
// Index: matches the Mat struct layout but only mechanics fields matter
export const WATER_ROW = [
  40000,    // eps        Q21 CFM
  16384,    // dpMax      ticks
  6000,     // xsph       Q16 viscosity
  0,        // phantom    0 = real matter
  240,      // gammaCoh   cohesion
  60,       // gammaCur   curvature
  0,        // adhesion   wall wetting
  1200,     // grav       ticks/substep^2
];
