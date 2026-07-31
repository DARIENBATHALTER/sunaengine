// SunaEngine — real SunaBox (aether) engine code, extracted UNMODIFIED below this header
// from aether commit e41e06f. Verify: diff <(tail -n +6 src/render_diag.wgsl.js) <aether>/src/render_diag.wgsl.js
// Copyright (C) 2026 Darien Bathalter
// SPDX-License-Identifier: AGPL-3.0-only — see LICENSE. NO WARRANTY.

// Renderer-only diagnostic field resolve.
//
// Powder Toy's pressure/velocity instruments read a separately simulated air
// grid. SunaEngine intentionally has no such grid yet. These views therefore
// visualise fields reconstructed from particle velocity, local kernel mass and
// particle temperature. They never feed the solver and are named accordingly
// in the UI.
export const DIAG_WGSL = /* wgsl */`
struct DU {
  q : vec4<f32>, // mode, strength, 1/fieldW, 1/fieldH
  p : vec4<f32>, // time phase (frame-derived), line gain, base dim, reserved
};

@group(0) @binding(0) var<uniform> D : DU;
@group(0) @binding(1) var fieldTex : texture_2d<f32>;
@group(0) @binding(2) var bgTex : texture_2d<f32>;
@group(0) @binding(3) var samp : sampler;

struct VOut { @builtin(position) clip : vec4<f32>, @location(0) uv : vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) vi : u32) -> VOut {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var o : VOut;
  let q = p[vi];
  o.clip = vec4<f32>(q, 0.0, 1.0);
  o.uv = vec2<f32>((q.x + 1.0) * 0.5, 1.0 - (q.y + 1.0) * 0.5);
  return o;
}

fn raw(uv : vec2<f32>) -> vec4<f32> {
  return textureSampleLevel(fieldTex, samp, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0);
}

fn meanField(uv : vec2<f32>) -> vec3<f32> {
  let s = raw(uv);
  return s.xyz / max(s.a, 0.035);
}

fn rampSigned(v : f32) -> vec3<f32> {
  let x = clamp(abs(v), 0.0, 1.0);
  let red = vec3<f32>(1.00, 0.075, 0.025);
  let blue = vec3<f32>(0.02, 0.18, 1.00);
  return select(blue, red, v >= 0.0) * (0.18 + 0.82 * sqrt(x)) * x;
}

@fragment
fn fs(in : VOut) -> @location(0) vec4<f32> {
  let mode = i32(D.q.x + 0.5);
  let ex = vec2<f32>(D.q.z * 2.0, 0.0);
  let ey = vec2<f32>(0.0, D.q.w * 2.0);
  let c = raw(in.uv);
  let m = c.xyz / max(c.a, 0.035);
  let l = raw(in.uv - ex);
  let r = raw(in.uv + ex);
  let u = raw(in.uv - ey);
  let d = raw(in.uv + ey);
  let ml = l.xyz / max(l.a, 0.035);
  let mr = r.xyz / max(r.a, 0.035);
  let mu = u.xyz / max(u.a, 0.035);
  let md = d.xyz / max(d.a, 0.035);

  // A local kernel-density contrast is a deliberately named compression
  // proxy, not atmospheric pressure. Divergence supplies the dynamic part.
  let densityContrast = c.a - (l.a + r.a + u.a + d.a) * 0.25;
  let divergence = (mr.x - ml.x) + (md.y - mu.y);
  let compression = clamp(densityContrast * 0.34 - divergence * 0.85, -1.0, 1.0);
  let curl = clamp((mr.y - ml.y) - (md.x - mu.x), -1.0, 1.0);
  let speed = clamp(length(m.xy), 0.0, 1.0);

  var field = vec3<f32>(0.0);
  if (mode == 1) {
    field = rampSigned(compression);
  } else if (mode == 2) {
    field = rampSigned(compression);
    // Fine moving filaments make direction and speed legible without arrows.
    let along = dot(in.uv * vec2<f32>(960.0, 620.0), normalize(m.xy + vec2<f32>(0.0001)));
    let streak = pow(0.5 + 0.5 * cos(along - D.p.x), 18.0) * speed * D.p.y;
    field = field + vec3<f32>(streak);
  } else if (mode == 3) {
    let cold = vec3<f32>(0.02, 0.20, 1.00);
    let warm = vec3<f32>(1.00, 0.05, 0.015);
    let hot = vec3<f32>(1.00, 0.82, 0.18);
    let t = clamp(m.z, -0.45, 1.25);
    field = select(cold * clamp(-t * 2.2, 0.0, 1.0),
                   mix(warm, hot, clamp(t, 0.0, 1.0)) * clamp(t * 1.9, 0.0, 1.0),
                   t >= 0.0);
  } else if (mode == 4) {
    field = rampSigned(curl);
  } else if (mode == 5) {
    // Powder Game-style vector composite: horizontal motion red, vertical
    // motion blue, positive compression green.
    field = vec3<f32>(abs(m.x), max(compression, 0.0), abs(m.y));
  }

  let activity = clamp(max(max(abs(compression), speed), max(abs(curl), abs(m.z))), 0.0, 1.0);
  let base = textureSampleLevel(bgTex, samp, in.uv, 0.0).rgb * mix(1.0, D.p.z, activity);
  return vec4<f32>(base + field * D.q.y, 1.0);
}
`;
