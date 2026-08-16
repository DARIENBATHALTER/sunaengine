// ============================================================================
// aether-3d — screen-space fluid rendering
// ----------------------------------------------------------------------------
// Float lives here and ONLY here. The simulation is integer; rendering is a
// view of it and carries no determinism obligation.
//
// Passes: background -> sphere-impostor depth -> thickness -> narrow-range
// depth blur (separable) -> composite (normal from depth, dispersive
// refraction, Beer-Lambert absorption, Fresnel, one directional light).
// ============================================================================

struct RU {
  view      : mat4x4<f32>,
  proj      : mat4x4<f32>,
  invProj   : mat4x4<f32>,
  invView   : mat4x4<f32>,
  screen    : vec2<f32>,
  radius    : f32,
  nParticles: u32,
  lightDir  : vec4<f32>,
  worldSize : vec4<f32>,
  camPos    : vec4<f32>,
  tuning    : vec4<f32>,   // x = refract, y = dispersion, z = absorb, w = blurRadius
  ball      : vec4<f32>,   // xyz centre, w radius (0 = off)
};

@group(0) @binding(0) var<uniform> R : RU;
@group(0) @binding(1) var<storage, read> partsR : array<vec4<i32>>;   // stride 4 vec4s

const Q : f32 = 65536.0;
const STRIDE : u32 = 4u;   // Particle is 4 x vec4<i32>

fn pos_of(i: u32) -> vec3<f32> {
  let p = partsR[i * STRIDE + 0u];
  return vec3<f32>(f32(p.x), f32(p.y), f32(p.z)) / Q;
}
fn vel_of(i: u32) -> vec3<f32> {
  let v = partsR[i * STRIDE + 1u];
  return vec3<f32>(f32(v.x), f32(v.y), f32(v.z)) / Q;
}

// ════════════════════════════════════════════════════════════════════════════
// BACKGROUND — the inside of a gridded box, so refraction has something to bend
// ════════════════════════════════════════════════════════════════════════════
struct BgOut { @builtin(position) pos: vec4<f32>, @location(0) wp: vec3<f32> };

@vertex
fn bgVS(@builtin(vertex_index) vi: u32) -> BgOut {
  // 36 verts, unit cube, inward facing
  var C = array<vec3<f32>, 8>(
    vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 1.0),
    vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 1.0));
  var I = array<u32, 36>(
    0u,2u,1u, 0u,3u,2u,    // back
    4u,5u,6u, 4u,6u,7u,    // front
    0u,1u,5u, 0u,5u,4u,    // bottom
    3u,7u,6u, 3u,6u,2u,    // top
    0u,4u,7u, 0u,7u,3u,    // left
    1u,2u,6u, 1u,6u,5u);   // right

  let w = C[I[vi]] * R.worldSize.xyz;
  var o: BgOut;
  o.wp  = w;
  o.pos = R.proj * R.view * vec4<f32>(w, 1.0);
  return o;
}

fn grid_line(c: vec2<f32>, period: f32, width: f32) -> f32 {
  let g = abs(fract(c / period - 0.5) - 0.5) * period;
  let d = min(g.x, g.y);
  return 1.0 - smoothstep(0.0, width, d);
}

@fragment
fn bgFS(i: BgOut) -> @location(0) vec4<f32> {
  let w = i.wp;
  let S = R.worldSize.xyz;

  // pick the plane we are on and grid in its two tangent axes
  var uv: vec2<f32>;
  let e = 0.02;
  if (w.x < e * S.x || w.x > S.x - e * S.x)      { uv = w.zy; }
  else if (w.y < e * S.y || w.y > S.y - e * S.y) { uv = w.xz; }
  else                                           { uv = w.xy; }

  let major = grid_line(uv, 8.0, 0.085);
  let minor = grid_line(uv, 2.0, 0.040);

  // Bright enough to be worth refracting: a nearly black room bends into
  // nearly black water and the dispersion has nothing to show.
  let base = vec3<f32>(0.048, 0.062, 0.115);
  var col  = base;
  col += vec3<f32>(0.085, 0.140, 0.225) * minor * 0.70;
  col += vec3<f32>(0.230, 0.420, 0.720) * major;

  // gentle depth falloff so the box reads as a room
  let d = length(w - R.camPos.xyz);
  col *= mix(1.25, 0.55, clamp(d / (length(S) * 1.35), 0.0, 1.0));

  // Ground ring under the ball. A cursor sphere floating in a box has no
  // readable height without one — this is the only cue for the axis the mouse
  // cannot express.
  if (R.ball.w > 0.0 && w.y < 0.02 * S.y) {
    let rr = length(w.xz - R.ball.xz);
    let ring = 1.0 - smoothstep(0.0, 0.5, abs(rr - R.ball.w));
    col += vec3<f32>(0.22, 0.50, 0.85) * ring * 0.55;
  }
  return vec4<f32>(col, 1.0);
}

// ════════════════════════════════════════════════════════════════════════════
// DEPTH — sphere impostors. Writes linear view depth and a real z for testing.
// ════════════════════════════════════════════════════════════════════════════
struct DOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv        : vec2<f32>,
  @location(1) vc        : vec3<f32>,   // sphere centre in view space
  @location(2) spd       : f32,
  @location(3) @interpolate(flat) mat : f32,
};

// ---- material appearance. Physics lives in the LUT rows; this is looks only.
fn mat_inscat(m: i32) -> vec3<f32> {
  if (m == 1) { return vec3<f32>(0.42, 0.26, 0.05); }   // oil
  if (m == 2) { return vec3<f32>(0.62, 0.65, 0.70); }   // mercury
  if (m == 3) { return vec3<f32>(0.50, 0.58, 0.68); }   // gas
  if (m == 4) { return vec3<f32>(0.62, 0.18, 0.65); }   // orbies
  if (m == 5) { return vec3<f32>(0.22, 0.62, 0.24); }   // goo
  return vec3<f32>(0.09, 0.42, 0.62);                   // water
}
fn mat_absorb(m: i32) -> vec3<f32> {
  if (m == 1) { return vec3<f32>(0.45, 0.85, 1.60); }
  if (m == 2) { return vec3<f32>(5.00, 5.00, 5.00); }
  if (m == 3) { return vec3<f32>(0.10, 0.10, 0.13); }
  if (m == 4) { return vec3<f32>(0.55, 1.10, 0.45); }
  if (m == 5) { return vec3<f32>(0.95, 0.40, 1.15); }
  return vec3<f32>(1.00, 0.36, 0.20);
}
fn mat_f0(m: i32) -> f32 {
  if (m == 1) { return 0.035; }
  if (m == 2) { return 0.700; }   // mercury reflects instead of transmitting
  if (m == 3) { return 0.010; }
  if (m == 4) { return 0.060; }
  if (m == 5) { return 0.045; }
  return 0.020;
}

// Shared impostor body. WGSL forbids calling an entry point, so the two vertex
// stages below both delegate here rather than one calling the other.
fn impostor(vi: u32, ii: u32) -> DOut {
  var Q4 = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0,  1.0), vec2<f32>(-1.0,  1.0));
  let q = Q4[vi];

  let wp = pos_of(ii);
  let vc = (R.view * vec4<f32>(wp, 1.0)).xyz;

  var o: DOut;
  o.uv  = q;
  o.vc  = vc;
  o.spd = length(vel_of(ii));
  o.mat = f32(partsR[ii * STRIDE + 0u].w);
  o.pos = R.proj * vec4<f32>(vc + vec3<f32>(q * R.radius, 0.0), 1.0);
  return o;
}

@vertex
fn depthVS(@builtin(vertex_index) vi: u32,
           @builtin(instance_index) ii: u32) -> DOut {
  return impostor(vi, ii);
}

struct DFrag {
  @location(0) vdepth : vec4<f32>,
  @builtin(frag_depth) z : f32,
};

@fragment
fn depthFS(i: DOut) -> DFrag {
  let r2 = dot(i.uv, i.uv);
  if (r2 > 1.0) { discard; }
  let nz = sqrt(1.0 - r2);

  // surface point of the sphere in view space (camera looks down -z)
  let vp = i.vc + vec3<f32>(i.uv * R.radius, nz * R.radius);
  let clip = R.proj * vec4<f32>(vp, 1.0);

  var o: DFrag;
  o.vdepth = vec4<f32>(-vp.z, i.spd, i.mat, 1.0);   // depth, speed, material
  o.z = clip.z / clip.w;
  return o;
}

// ════════════════════════════════════════════════════════════════════════════
// THICKNESS — additive, depth test off. Beer-Lambert path length.
// ════════════════════════════════════════════════════════════════════════════
@vertex
fn thickVS(@builtin(vertex_index) vi: u32,
           @builtin(instance_index) ii: u32) -> DOut {
  return impostor(vi, ii);
}

@fragment
fn thickFS(i: DOut) -> @location(0) vec4<f32> {
  let r2 = dot(i.uv, i.uv);
  if (r2 > 1.0) { discard; }
  let nz = sqrt(1.0 - r2);
  return vec4<f32>(nz * R.radius * 2.0, 0.0, 0.0, 1.0);
}

// ════════════════════════════════════════════════════════════════════════════
// THE CURSOR BALL — one sphere impostor, drawn after the composite and depth
// tested against the fluid's own depth buffer so water in front occludes it.
// ════════════════════════════════════════════════════════════════════════════
struct BOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv        : vec2<f32>,
  @location(1) vc        : vec3<f32>,
};

@vertex
fn ballVS(@builtin(vertex_index) vi: u32) -> BOut {
  var Q4 = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0,  1.0), vec2<f32>(-1.0,  1.0));
  let q = Q4[vi];
  let vc = (R.view * vec4<f32>(R.ball.xyz, 1.0)).xyz;
  var o: BOut;
  o.uv = q;
  o.vc = vc;
  o.pos = R.proj * vec4<f32>(vc + vec3<f32>(q * R.ball.w, 0.0), 1.0);
  return o;
}

struct BFrag {
  @location(0) col : vec4<f32>,
  @builtin(frag_depth) z : f32,
};

@fragment
fn ballFS(i: BOut) -> BFrag {
  let r2 = dot(i.uv, i.uv);
  if (r2 > 1.0) { discard; }
  let nz = sqrt(1.0 - r2);

  let N  = vec3<f32>(i.uv, nz);
  let vp = i.vc + N * R.ball.w;
  let clip = R.proj * vec4<f32>(vp, 1.0);

  let V = normalize(-vp);
  let L = normalize((R.view * vec4<f32>(normalize(R.lightDir.xyz), 0.0)).xyz);
  let Hv = normalize(L + V);

  let lam  = clamp(dot(N, L), 0.0, 1.0);
  let spec = pow(clamp(dot(N, Hv), 0.0, 1.0), 140.0);
  let rim  = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 2.5);

  var col = vec3<f32>(0.055, 0.075, 0.115) * (0.35 + 0.65 * lam);
  col += vec3<f32>(0.34, 0.62, 1.00) * rim * 0.85;
  col += vec3<f32>(1.0, 0.97, 0.92) * spec * 1.5;

  var o: BFrag;
  o.col = vec4<f32>(col, 1.0);
  o.z = clip.z / clip.w;
  return o;
}

// ════════════════════════════════════════════════════════════════════════════
// FULLSCREEN TRIANGLE
// ════════════════════════════════════════════════════════════════════════════
struct FS { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn fullVS(@builtin(vertex_index) vi: u32) -> FS {
  var P3 = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0), vec2<f32>(-1.0, 1.0), vec2<f32>( 3.0, 1.0));
  let p = P3[vi];
  var o: FS;
  o.pos = vec4<f32>(p, 0.0, 1.0);
  o.uv  = vec2<f32>((p.x + 1.0) * 0.5, 1.0 - (p.y + 1.0) * 0.5);
  return o;
}

// ════════════════════════════════════════════════════════════════════════════
// NARROW-RANGE DEPTH BLUR — separable. Samples far from the centre depth are
// rejected so the surface does not smear across silhouettes.
// ════════════════════════════════════════════════════════════════════════════
@group(1) @binding(0) var srcTex : texture_2d<f32>;

struct BlurDir { dir: vec2<f32>, pad: vec2<f32> };
@group(1) @binding(1) var<uniform> BD : BlurDir;

// The kernel width is derived from the bead's PROJECTED size, not fixed in
// pixels: a constant pixel radius over-blurs near water and leaves far water
// bumpy, which is what makes a particle fluid read as sparkling gravel.
const BLUR_MAX : i32 = 40;

@fragment
fn blurFS(i: FS) -> @location(0) vec4<f32> {
  let px = vec2<i32>(i.pos.xy);
  let c  = textureLoad(srcTex, px, 0);
  if (c.x <= 0.0) { return c; }

  let focal = R.proj[1][1] * R.screen.y * 0.5;
  let projR = R.radius * focal / max(c.x, 0.5);          // bead radius in pixels
  let radius = clamp(i32(projR * R.tuning.w), 1, BLUR_MAX);
  let sigma  = max(1.0, f32(radius) * 0.55);
  let thresh = R.radius * 2.0;

  var sum: f32  = c.x;
  var spd: f32  = c.y;
  var wsum: f32 = 1.0;

  // Walk outward and BREAK past the computed radius. A `continue` over the full
  // -BLUR_MAX..BLUR_MAX span costs all 81 taps no matter how narrow the kernel
  // actually is, twice, at full resolution — which is how this pass came to cost
  // more than the entire simulation.
  for (var k = 1; k <= BLUR_MAX; k = k + 1) {
    if (k > radius) { break; }
    let fk = f32(k) / sigma;
    let w  = exp(-0.5 * fk * fk);
    let step = vec2<i32>(BD.dir * f32(k));

    for (var sgn = 0; sgn < 2; sgn = sgn + 1) {
      let o = px + select(step, -step, sgn == 1);
      if (o.x < 0 || o.y < 0 || o.x >= i32(R.screen.x) || o.y >= i32(R.screen.y)) { continue; }
      let s = textureLoad(srcTex, o, 0);
      if (s.x <= 0.0) { continue; }
      if (abs(s.x - c.x) > thresh) { continue; }
      sum  = sum  + s.x * w;
      spd  = spd  + s.y * w;
      wsum = wsum + w;
    }
  }
  if (wsum <= 0.0) { return c; }
  // material is NOT blurred: averaging ids would invent materials that do not
  // exist. The centre pixel owns the surface it is on.
  return vec4<f32>(sum / wsum, spd / wsum, c.z, 1.0);
}

// ════════════════════════════════════════════════════════════════════════════
// COMPOSITE
// ════════════════════════════════════════════════════════════════════════════
// Distinct binding numbers from the blur pass above: both entry points live in
// one module, so two declarations at the same @group/@binding would collide at
// WGSL level even though their pipelines use different layouts.
@group(1) @binding(4) var depthTex : texture_2d<f32>;
@group(1) @binding(5) var thickTex : texture_2d<f32>;
@group(1) @binding(6) var bgTex    : texture_2d<f32>;
@group(1) @binding(7) var samp     : sampler;

fn vpos_at(px: vec2<i32>) -> vec3<f32> {
  let d = textureLoad(depthTex, px, 0).x;
  let uv = (vec2<f32>(px) + vec2<f32>(0.5, 0.5)) / R.screen;
  let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, 0.5, 1.0);
  let h = R.invProj * ndc;
  let dir = h.xyz / h.w;
  let s = d / -dir.z;
  return dir * s;
}

@fragment
fn compositeFS(i: FS) -> @location(0) vec4<f32> {
  let px  = vec2<i32>(i.pos.xy);
  // textureSampleLevel, not textureSample: everything below the early return is
  // non-uniform control flow. The background has no mips, so LOD 0 is exact.
  let bg  = textureSampleLevel(bgTex, samp, i.uv, 0.0).rgb;
  let d   = textureLoad(depthTex, px, 0).x;

  if (d <= 0.0) { return vec4<f32>(bg, 1.0); }

  let vp = vpos_at(px);

  // normal from screen-space derivatives of the reconstructed position, taking
  // the smaller-magnitude difference on each axis so silhouettes stay sharp
  let ddxR = vpos_at(px + vec2<i32>(1, 0)) - vp;
  let ddxL = vp - vpos_at(px - vec2<i32>(1, 0));
  let ddyU = vpos_at(px + vec2<i32>(0, 1)) - vp;
  let ddyD = vp - vpos_at(px - vec2<i32>(0, 1));
  let dx = select(ddxR, ddxL, abs(ddxL.z) < abs(ddxR.z));
  let dy = select(ddyU, ddyD, abs(ddyD.z) < abs(ddyU.z));
  // cross(dy, dx), NOT cross(dx, dy): framebuffer y runs DOWN, so the naive
  // order yields a normal pointing away from the camera. dot(N,V) then pins to
  // zero, every pixel looks like a grazing angle, and the grazing-angle terms
  // (Fresnel, thin film) take over the whole surface.
  let N  = normalize(cross(dy, dx));

  let V = normalize(-vp);
  let L = normalize((R.view * vec4<f32>(normalize(R.lightDir.xyz), 0.0)).xyz);
  let Hv = normalize(L + V);

  // Accumulated path length, normalised: the thickness pass sums one chord per
  // bead, so raw values run to ~60 through a deep pool. Scale before it reaches
  // an exponential or the water goes black.
  let thick = textureLoad(thickTex, px >> vec2<u32>(1u, 1u), 0).x * 0.25;   // half-res target
  let spd   = textureLoad(depthTex, px, 0).y;
  let mid   = i32(textureLoad(depthTex, px, 0).z + 0.5);

  // ---- dispersive refraction: each channel bends by a slightly different eta.
  // The offset is in UV, so it must stay a small fraction of the screen; the
  // slider's 0..40 maps to 0..0.04 of screen width.
  let k  = R.tuning.x * 0.001 * clamp(thick * 0.35, 0.15, 1.0);
  let disp = R.tuning.y;
  let off = N.xy * k;
  let uvR = i.uv + off * (1.0 - disp);
  let uvG = i.uv + off;
  let uvB = i.uv + off * (1.0 + disp);
  var refr = vec3<f32>(
    textureSampleLevel(bgTex, samp, clamp(uvR, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).r,
    textureSampleLevel(bgTex, samp, clamp(uvG, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).g,
    textureSampleLevel(bgTex, samp, clamp(uvB, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).b);

  // ---- Beer-Lambert: red is absorbed first, which is why deep water is blue
  let absorb = mat_absorb(mid) * R.tuning.z;
  let trans  = exp(-absorb * thick);
  refr = refr * trans + mat_inscat(mid) * (1.0 - trans);

  // ---- Fresnel + one directional light
  let F0 = mat_f0(mid);
  let ct = clamp(dot(N, V), 0.0, 1.0);
  let F  = F0 + (1.0 - F0) * pow(1.0 - ct, 5.0);

  // cheap environment for the reflected term: horizon gradient + sun
  let Nw = normalize((R.invView * vec4<f32>(N, 0.0)).xyz);
  let sky = mix(vec3<f32>(0.04, 0.06, 0.12),
                vec3<f32>(0.30, 0.44, 0.68), clamp(Nw.y * 0.5 + 0.5, 0.0, 1.0));
  let sun = pow(clamp(dot(reflect(-V, N), L), 0.0, 1.0), 220.0);

  let spec = pow(clamp(dot(N, Hv), 0.0, 1.0), 96.0);

  // debug taps — lightDir.w carries the view mode
  let vm = i32(R.lightDir.w);
  if (vm == 1) { return vec4<f32>(N * 0.5 + vec3<f32>(0.5), 1.0); }
  if (vm == 2) { return vec4<f32>(vec3<f32>(thick / 12.0), 1.0); }
  if (vm == 3) { return vec4<f32>(vec3<f32>(d / 180.0), 1.0); }
  if (vm == 4) { return vec4<f32>(refr, 1.0); }
  if (vm == 5) { return vec4<f32>(vec3<f32>(F * 8.0), 1.0); }
  if (vm == 6) { return vec4<f32>(vec3<f32>(spec), 1.0); }

  var col = mix(refr, sky, F) + vec3<f32>(1.0, 0.96, 0.90) * (spec * 0.40 + sun * 1.6);

  // Thin-film sheen at grazing angles — the signature of the 2D look. Driven by
  // a SHALLOW thickness term: feeding the raw path length into a cosine makes it
  // oscillate several times per pixel and reads as noise, not iridescence.
  let rim  = pow(1.0 - ct, 4.0);
  let ph   = clamp(thick, 0.0, 6.0) * 0.9;
  let film = vec3<f32>(0.5 + 0.5 * cos(ph + 0.0),
                       0.5 + 0.5 * cos(ph + 2.1),
                       0.5 + 0.5 * cos(ph + 4.2));
  col += film * rim * disp * 1.1;

  // a touch of motion brightening so fast sheets read
  col += vec3<f32>(0.10, 0.18, 0.26) * clamp(spd * 5.0, 0.0, 1.0) * 0.35;

  return vec4<f32>(col, 1.0);
}
