// ============================================================================
// aether-3d — host
// ----------------------------------------------------------------------------
// The kernel tables are built here in double precision and uploaded as i32.
// That is safe for determinism because the tables are DATA: every device reads
// identical bits. The build uses only +, - , *, / and Math.sqrt, all of which
// IEEE-754 specifies as correctly rounded. Math.pow is NOT specified that way
// and is deliberately not used.
// ============================================================================

const CFG = {
  GX: 28, GY: 30, GZ: 28,       // cells; world is GX*2 x GY*2 x GZ*2 world units
  MAXNBR: 48,
  NMAX: 60000,
  LUT_N: 2049,
  RHO0: 1048576,                 // 2^20
  SG: 4000,                      // gradient table scale
  H: 2.0,                        // smoothing radius in world units
  SPACING: 1.0,
  substeps: 2,
  iters: 3,
  gravStep: 96,
  damp: 65340,
  stirAmp: 0,
};

const ONE = 65536;

// ------------------------------------------------------------------ mat4
const m4 = {
  ident: () => new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
  mul(a, b) {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
    return o;
  },
  persp(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2);
    const o = new Float32Array(16);
    o[0] = f / aspect; o[5] = f;
    o[10] = far / (near - far); o[11] = -1;
    o[14] = (near * far) / (near - far);
    return o;
  },
  lookAt(eye, at, up) {
    const z = norm(sub(eye, at)), x = norm(cross(up, z)), y = cross(z, x);
    const o = new Float32Array(16);
    o[0]=x[0]; o[4]=x[1]; o[8]=x[2];
    o[1]=y[0]; o[5]=y[1]; o[9]=y[2];
    o[2]=z[0]; o[6]=z[1]; o[10]=z[2];
    o[12]=-dot(x,eye); o[13]=-dot(y,eye); o[14]=-dot(z,eye); o[15]=1;
    return o;
  },
  inv(m) {
    const a=m, o=new Float32Array(16);
    const b00=a[0]*a[5]-a[1]*a[4],  b01=a[0]*a[6]-a[2]*a[4],  b02=a[0]*a[7]-a[3]*a[4];
    const b03=a[1]*a[6]-a[2]*a[5],  b04=a[1]*a[7]-a[3]*a[5],  b05=a[2]*a[7]-a[3]*a[6];
    const b06=a[8]*a[13]-a[9]*a[12],b07=a[8]*a[14]-a[10]*a[12],b08=a[8]*a[15]-a[11]*a[12];
    const b09=a[9]*a[14]-a[10]*a[13],b10=a[9]*a[15]-a[11]*a[13],b11=a[10]*a[15]-a[11]*a[14];
    let d=b00*b11-b01*b10+b02*b09+b03*b08-b04*b07+b05*b06;
    if (!d) return m4.ident();
    d = 1/d;
    o[0]=(a[5]*b11-a[6]*b10+a[7]*b09)*d;  o[1]=(a[2]*b10-a[1]*b11-a[3]*b09)*d;
    o[2]=(a[13]*b05-a[14]*b04+a[15]*b03)*d;o[3]=(a[10]*b04-a[9]*b05-a[11]*b03)*d;
    o[4]=(a[6]*b08-a[4]*b11-a[7]*b07)*d;  o[5]=(a[0]*b11-a[2]*b08+a[3]*b07)*d;
    o[6]=(a[14]*b02-a[12]*b05-a[15]*b01)*d;o[7]=(a[8]*b05-a[10]*b02+a[11]*b01)*d;
    o[8]=(a[4]*b10-a[5]*b08+a[7]*b06)*d;  o[9]=(a[1]*b08-a[0]*b10-a[3]*b06)*d;
    o[10]=(a[12]*b04-a[13]*b02+a[15]*b00)*d;o[11]=(a[9]*b02-a[8]*b04-a[11]*b00)*d;
    o[12]=(a[5]*b07-a[4]*b09-a[6]*b06)*d; o[13]=(a[0]*b09-a[1]*b07+a[2]*b06)*d;
    o[14]=(a[13]*b01-a[12]*b03-a[14]*b00)*d;o[15]=(a[8]*b03-a[9]*b01+a[10]*b00)*d;
    return o;
  },
};
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const norm=(a)=>{const l=Math.sqrt(dot(a,a))||1; return [a[0]/l,a[1]/l,a[2]/l];};

// ------------------------------------------------------- kernel tables
// ui indexes r2q >> 7 where r2q = (r_world * 256)^2, so r^2 = ui / 512.
function buildLUTs() {
  const N = CFG.LUT_N, h = CFG.H, h2 = h * h;
  const rawW = new Float64Array(N), rawG = new Float64Array(N);

  for (let ui = 0; ui < N; ui++) {
    const r2 = ui / 512;
    if (r2 >= h2) { rawW[ui] = 0; rawG[ui] = 0; continue; }
    const t = h2 - r2;
    rawW[ui] = t * t * t;                        // poly6, unnormalised

    const r = Math.sqrt(r2);
    const hr = h - r;
    // spiky gradient coefficient G with grad = G * d, i.e. (h-r)^2 / r.
    // r -> 0 is a removable blow-up; hold the first real bucket's value.
    rawG[ui] = r > 1e-6 ? (hr * hr) / r : 0;
  }
  for (let ui = 0; ui < N; ui++) { if (rawG[ui] > 0) { rawG[0] = rawG[ui]; break; } }

  // rest density: sum poly6 over a perfect lattice at CFG.SPACING, incl. self
  let sumRaw = 0;
  const rad = Math.ceil(h / CFG.SPACING);
  for (let i = -rad; i <= rad; i++)
    for (let j = -rad; j <= rad; j++)
      for (let k = -rad; k <= rad; k++) {
        const r2 = (i*i + j*j + k*k) * CFG.SPACING * CFG.SPACING;
        if (r2 >= h2) continue;
        const t = h2 - r2;
        sumRaw += t * t * t;
      }

  const SW = CFG.RHO0 / sumRaw;

  // The gradient table's scale is NOT free. rho is expressed in units where the
  // rest lattice sums to RHO0, so the gradient must be d(rho)/dx in those same
  // units or lambda comes out wrong by whatever factor the two disagree by.
  //
  // poly6's own gradient is  grad = SW * 6 * (h^2 - r^2)^2 * d.  We use the
  // SPIKY shape instead (poly6's gradient vanishes as r -> 0 and lets particles
  // clump), but pin its scale to poly6's gradient at r = 1 so the magnitude is
  // the physically consistent one rather than a tuned guess.
  const SG = SW * 6 * (h2 - 1) * (h2 - 1);
  CFG.SG = SG;

  // ---- boundary tables --------------------------------------------------
  // A clamped position is not a boundary condition. A particle against a wall
  // has neighbours on one side only, reads as a free surface, and gets squirted
  // along the wall. Model each wall as a SOLID HALF-SPACE of lattice matter:
  // the density it would contribute if the fluid simply continued through.
  // Indexed by wall distance >> 8, so b spans 0..512 for 0..h world units.
  const PN = 513;
  const planeW = new Int32Array(PN), planeG = new Int32Array(PN);
  const sp = CFG.SPACING, lat = Math.ceil(h / sp);
  for (let b = 0; b < PN; b++) {
    const dist = b / 256;
    let sw = 0, sg = 0;
    for (let k = 0; ; k++) {
      const pd = dist + k * sp;
      if (pd >= h) break;
      for (let i = -lat; i <= lat; i++)
        for (let j = -lat; j <= lat; j++) {
          const r2 = pd * pd + (i * sp) * (i * sp) + (j * sp) * (j * sp);
          if (r2 >= h2) continue;
          const t = h2 - r2;
          sw += t * t * t;
          sg += pd * t * t;          // d/d(dist) of the above, sign folded below
        }
    }
    planeW[b] = Math.round(sw * SW);
    planeG[b] = Math.round(6 * sg * SW);   // magnitude, pointing INTO the wall
  }

  PLANE = { w: planeW, g: planeG };

  const lut = new Int32Array(N * 2 + PN * 2 + BALL_N * 2);
  for (let ui = 0; ui < N; ui++) {
    lut[ui]     = Math.round(rawW[ui] * SW);
    lut[N + ui] = Math.round(rawG[ui] * SG);
  }
  lut.set(planeW, N * 2);
  lut.set(planeG, N * 2 + PN);
  return lut;
}

// ---- materials -------------------------------------------------------------
// Four integer words per material, uploaded with the kernel tables so they cost
// no extra storage binding (WebGPU guarantees only 8 per stage and we are at 7).
//
//   grav  per-substep velocity step, Q16.16. Negative rises.
//   rhoW  VOLUME CLAIM, not density — the §37 trick from the 2D engine. Every
//         kernel term is weighted (W * (256 + rhoW)) >> 8, so a swollen bead
//         raises its neighbours' density, the constraint decompresses, and the
//         bead really does take more room. 256 == double volume.
//   damp  per-substep velocity multiplier, Q16.16. This is the viscosity knob.
//   f0    reflectance x1000, used by the renderer only.
const MAT_WORDS = 4;
const MATERIALS = [
  { key: 'water',   name: 'Water',   grav:  96, rhoW:   0, damp: 65340, f0:  20 },
  { key: 'oil',     name: 'Oil',     grav:  58, rhoW:  30, damp: 65180, f0:  35 },
  { key: 'mercury', name: 'Mercury', grav: 240, rhoW:   0, damp: 65420, f0: 700 },
  { key: 'gas',     name: 'Gas',     grav: -46, rhoW: 120, damp: 64700, f0:  10 },
  { key: 'orbies',  name: 'Orbies',  grav:  88, rhoW: 210, damp: 65330, f0:  60 },
  { key: 'goo',     name: 'Goo',     grav:  92, rhoW:  50, damp: 62900, f0:  45 },
];
const MAT_OFF = CFG.LUT_N * 2 + 513 * 2 + 1024 * 2;

function matWords() {
  const a = new Int32Array(MATERIALS.length * MAT_WORDS);
  MATERIALS.forEach((m, i) => {
    a[i * MAT_WORDS + 0] = m.grav;
    a[i * MAT_WORDS + 1] = m.rhoW;
    a[i * MAT_WORDS + 2] = m.damp;
    a[i * MAT_WORDS + 3] = m.f0;
  });
  return a;
}

// ---- the cursor ball ------------------------------------------------------
// Indexed by r2q >> 12 where r2q is the CENTRE distance squared in reduced
// units, so the shader never needs a square root: b spans 0..1023 for r = 0..8
// world units. The sphere is treated locally as a half-space at depth (r - R),
// which is what makes the ball reuse the wall tables rather than inventing a
// second boundary model. Rebuilt on the host whenever the radius changes.
let PLANE = null;
const BALL_N = 1024;
const BALL_OFF_W = CFG.LUT_N * 2 + 513 * 2;
const BALL_OFF_G = BALL_OFF_W + BALL_N;

function buildBallLUT(R) {
  const w = new Int32Array(BALL_N), g = new Int32Array(BALL_N);
  if (!PLANE) return { w, g };
  const at = (tab, distWorld) => tab[Math.max(0, Math.min(512, Math.round(distWorld * 256)))];

  for (let b = 0; b < BALL_N; b++) {
    const r = Math.sqrt(b * 4096) / 256;          // world units from the centre
    const depth = r - R;                          // >0 outside the surface
    if (depth >= CFG.H) { w[b] = 0; g[b] = 0; continue; }
    const d = Math.max(depth, 0);
    w[b] = at(PLANE.w, d);
    // grad = G * dvec, and |dvec| = r, so G = |grad| / r. r -> 0 is the centre,
    // where the direction is undefined and the net push is zero by symmetry.
    g[b] = r > 1e-6 ? Math.min(Math.round(at(PLANE.g, d) / r), 1 << 24) : 0;
  }
  return { w, g };
}

// splitmix32 — deterministic jitter so the initial lattice is not perfectly
// symmetric (perfect symmetry produces standing artefacts on a free surface).
function smix(x) {
  x = (x + 0x9e3779b9) | 0;
  let z = x;
  z = Math.imul(z ^ (z >>> 16), 0x21f0aaad);
  z = Math.imul(z ^ (z >>> 15), 0x735a2d97);
  return (z ^ (z >>> 15)) >>> 0;
}

// ============================================================================
class App {
  constructor(canvas, ui) { this.canvas = canvas; this.ui = ui; this.running = true; }

  async boot() {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser.');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter.');
    this.adapter = adapter;
    this.device = await adapter.requestDevice({ label: 'aether-3d' });
    this.device.addEventListener?.('uncapturederror', (e) => {
      console.error('[aether-3d] uncaptured', e.error?.message || e.error);
    });

    const info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
    this.adapterInfo = {
      vendor: info.vendor || 'unknown',
      architecture: info.architecture || 'unknown',
      description: info.description || '',
    };

    // Release the device on unload. Without this, iterating on a page that
    // creates a GPUDevice per reload eventually wedges the GPU process:
    // requestDevice stops resolving and boot() hangs instead of throwing.
    addEventListener('pagehide', () => { try { this.device.destroy(); } catch {} });

    this.ctx = this.canvas.getContext('webgpu');
    this.format = navigator.gpu.getPreferredCanvasFormat();
    // COPY_SRC so frames can be read back off the GPU directly. drawImage() on a
    // WebGPU canvas races the compositor's present and returns black once the
    // swapchain texture has been handed over; copying the texture does not.
    this.ctx.configure({
      device: this.device, format: this.format, alphaMode: 'opaque',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });

    const bust = `?v=${Date.now()}`;
    const [fixedSrc, simSrc, renSrc] = await Promise.all([
      fetch('fixed.wgsl' + bust, { cache: 'no-store' }).then(r => r.text()),
      fetch('sim3d.wgsl' + bust, { cache: 'no-store' }).then(r => r.text()),
      fetch('render3d.wgsl' + bust, { cache: 'no-store' }).then(r => r.text()),
    ]);
    this.simModule = this.device.createShaderModule({
      label: 'sim', code: fixedSrc + '\n' + simSrc,
    });
    this.renModule = this.device.createShaderModule({ label: 'render', code: renSrc });

    for (const [name, mod] of [['sim', this.simModule], ['render', this.renModule]]) {
      const ci = await mod.getCompilationInfo();
      const errs = ci.messages.filter(m => m.type === 'error');
      if (errs.length) {
        console.error(`[aether-3d] ${name} shader errors:`);
        for (const e of errs) console.error(`  ${e.lineNum}:${e.linePos} ${e.message}`);
        throw new Error(`${name} shader failed to compile: ${errs[0].message} (line ${errs[0].lineNum})`);
      }
    }

    this.layout();
    this.makeBuffers();
    this.makePipelines();
    this.resize();
    this.reset();
    this.initCamera();
    return this;
  }

  layout() {
    const g = CFG;
    this.cellTotal = g.GX * g.GY * g.GZ;
    this.nBlocks = Math.ceil(this.cellTotal / 1024);
    this.world = [g.GX * g.H, g.GY * g.H, g.GZ * g.H];
  }

  // ------------------------------------------------------------ particles
  // scene 'pool'      — rest-spaced column, the one that is shown
  // scene 'overdense' — the SAME solver at 0.4 spacing, ~125 particles per cell.
  //   The ordering control is meaningless on a sparse scene: with one particle
  //   per cell there is no atomic contention, scatter lands in invocation order
  //   anyway, and defeating canonicalize provably changes nothing. Contention
  //   and neighbour-list truncation are what give scatter order something to be
  //   wrong ABOUT, so the control runs here.
  initialState(n, scene = 'pool') {
    const a = new Int32Array(n * 16);   // 4 x vec4<i32>
    const dense = scene === 'overdense';
    const sp = dense ? 0.4 : CFG.SPACING;
    const w = this.world;
    // A column in one corner: it collapses, sloshes, and settles. The footprint
    // is a fraction of the world and the HEIGHT follows from the count, so the
    // column can never be asked to stand taller than the box.
    const frac = dense ? 0.28 : 0.55;
    const bx = Math.floor((w[0] * frac) / sp);
    const bz = Math.floor((w[2] * frac) / sp);
    const byMax = Math.floor((w[1] * 0.92) / sp);
    const by = Math.min(byMax, Math.ceil(n / (bx * bz)));
    let c = 0;
    outer:
    for (let j = 0; j < by; j++)
      for (let k = 0; k < bz; k++)
        for (let i = 0; i < bx; i++) {
          if (c >= n) break outer;
          const jx = (smix(c * 3 + 0) % 8192) - 4096;
          const jy = (smix(c * 3 + 1) % 8192) - 4096;
          const jz = (smix(c * 3 + 2) % 8192) - 4096;
          const x = Math.round((1.5 + i * sp) * ONE) + jx;
          const y = Math.round((1.5 + j * sp) * ONE) + jy;
          const z = Math.round((1.5 + k * sp) * ONE) + jz;
          const o = c * 16;
          a[o + 0] = x; a[o + 1] = y; a[o + 2] = z;
          // 'layers' stacks the three liquids OUT of order on purpose: mercury
          // on top, oil at the bottom. Watching them change places is the proof
          // that separation is emergent and not painted in.
          if (scene === 'layers') {
            const f = j / by;
            a[o + 3] = f < 0.34 ? 1 : (f < 0.67 ? 0 : 2);
          } else {
            a[o + 3] = 0;
          }
          c++;
        }
    this.count = c;
    return a.subarray(0, c * 16);
  }

  makeBuffers() {
    const d = this.device;
    const n = this.nMax = CFG.NMAX;
    const B = GPUBufferUsage;

    this.offCellStart = 0;
    this.offCellOf = this.cellTotal + 1;
    this.offBucket  = this.offCellOf + n;
    this.offSorted  = this.offBucket + n;
    this.offBlock   = this.offSorted + n;
    const idxLen    = this.offBlock + this.nBlocks + 4;

    this.buf = {
      params: d.createBuffer({ size: 96, usage: B.UNIFORM | B.COPY_DST }),
      parts:  d.createBuffer({ size: n * 64, usage: B.STORAGE | B.COPY_DST | B.COPY_SRC }),
      cells:  d.createBuffer({ size: this.cellTotal * 4, usage: B.STORAGE }),
      idx:    d.createBuffer({ size: idxLen * 4, usage: B.STORAGE }),
      nbr:    d.createBuffer({ size: (n * CFG.MAXNBR + n) * 4, usage: B.STORAGE }),
      luts:   d.createBuffer({ size: (MAT_OFF + MATERIALS.length * MAT_WORDS) * 4, usage: B.STORAGE | B.COPY_DST }),
      digest: d.createBuffer({ size: 256 * 4, usage: B.STORAGE | B.COPY_SRC }),
      dbg:    d.createBuffer({ size: 16 * 4, usage: B.STORAGE | B.COPY_SRC | B.COPY_DST }),
      ru:     d.createBuffer({ size: 384, usage: B.UNIFORM | B.COPY_DST }),
      blurU:  d.createBuffer({ size: 16, usage: B.UNIFORM | B.COPY_DST }),
      blurV:  d.createBuffer({ size: 16, usage: B.UNIFORM | B.COPY_DST }),
      read:   d.createBuffer({ size: 256 * 4, usage: B.COPY_DST | B.MAP_READ }),
      readDbg:d.createBuffer({ size: 16 * 4, usage: B.COPY_DST | B.MAP_READ }),
    };

    d.queue.writeBuffer(this.buf.luts, 0, buildLUTs());
    d.queue.writeBuffer(this.buf.luts, MAT_OFF * 4, matWords());
    this.setBallRadius(3.4);
    d.queue.writeBuffer(this.buf.blurU, 0, new Float32Array([1, 0, 0, 0]));
    d.queue.writeBuffer(this.buf.blurV, 0, new Float32Array([0, 1, 0, 0]));
  }

  makePipelines() {
    const d = this.device;
    const S = (t) => ({ buffer: { type: t } });

    this.cbgl = d.createBindGroupLayout({
      entries: [
        { binding: 0,  visibility: GPUShaderStage.COMPUTE, ...S('uniform') },
        { binding: 1,  visibility: GPUShaderStage.COMPUTE, ...S('storage') },
        { binding: 2,  visibility: GPUShaderStage.COMPUTE, ...S('storage') },
        { binding: 3,  visibility: GPUShaderStage.COMPUTE, ...S('storage') },
        { binding: 4,  visibility: GPUShaderStage.COMPUTE, ...S('storage') },
        { binding: 5,  visibility: GPUShaderStage.COMPUTE, ...S('read-only-storage') },
        { binding: 6,  visibility: GPUShaderStage.COMPUTE, ...S('storage') },
        { binding: 15, visibility: GPUShaderStage.COMPUTE, ...S('storage') },
      ],
    });
    this.cbg = d.createBindGroup({
      layout: this.cbgl,
      entries: [
        { binding: 0,  resource: { buffer: this.buf.params } },
        { binding: 1,  resource: { buffer: this.buf.parts } },
        { binding: 2,  resource: { buffer: this.buf.cells } },
        { binding: 3,  resource: { buffer: this.buf.idx } },
        { binding: 4,  resource: { buffer: this.buf.nbr } },
        { binding: 5,  resource: { buffer: this.buf.luts } },
        { binding: 6,  resource: { buffer: this.buf.digest } },
        { binding: 15, resource: { buffer: this.buf.dbg } },
      ],
    });
    const cpl = d.createPipelineLayout({ bindGroupLayouts: [this.cbgl] });

    this.cp = {};
    for (const e of ['predict','clearCells','gridCount','scanBlock','scanTop','scanAdd',
                     'scatter','canonicalize','buildNbr','density','solveDp','applyDp',
                     'integrate','digestStripe']) {
      this.cp[e] = d.createComputePipeline({
        label: e, layout: cpl, compute: { module: this.simModule, entryPoint: e },
      });
    }

    // ---- render layouts
    this.rbgl0 = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, ...S('uniform') },
        { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, ...S('read-only-storage') },
      ],
    });
    this.rbg0 = d.createBindGroup({
      layout: this.rbgl0,
      entries: [
        { binding: 0, resource: { buffer: this.buf.ru } },
        { binding: 1, resource: { buffer: this.buf.parts } },
      ],
    });

    this.blurBgl = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'unfilterable-float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, ...S('uniform') },
      ],
    });
    this.compBgl = d.createBindGroupLayout({
      entries: [
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });

    const plB = d.createPipelineLayout({ bindGroupLayouts: [this.rbgl0, this.blurBgl] });
    const plC = d.createPipelineLayout({ bindGroupLayouts: [this.rbgl0, this.compBgl] });
    const pl0 = d.createPipelineLayout({ bindGroupLayouts: [this.rbgl0] });

    const HDR = 'rgba16float', LIN = 'rgba32float';

    this.rp = {
      bg: d.createRenderPipeline({
        layout: pl0,
        vertex:   { module: this.renModule, entryPoint: 'bgVS' },
        fragment: { module: this.renModule, entryPoint: 'bgFS', targets: [{ format: HDR }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
      }),
      depth: d.createRenderPipeline({
        layout: pl0,
        vertex:   { module: this.renModule, entryPoint: 'depthVS' },
        fragment: { module: this.renModule, entryPoint: 'depthFS', targets: [{ format: LIN }] },
        primitive: { topology: 'triangle-list' },
        depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      }),
      thick: d.createRenderPipeline({
        layout: pl0,
        vertex:   { module: this.renModule, entryPoint: 'thickVS' },
        fragment: { module: this.renModule, entryPoint: 'thickFS', targets: [{
          format: HDR,
          blend: { color: { srcFactor: 'one', dstFactor: 'one' },
                   alpha: { srcFactor: 'one', dstFactor: 'one' } },
        }] },
        primitive: { topology: 'triangle-list' },
      }),
      blur: d.createRenderPipeline({
        layout: plB,
        vertex:   { module: this.renModule, entryPoint: 'fullVS' },
        fragment: { module: this.renModule, entryPoint: 'blurFS', targets: [{ format: LIN }] },
        primitive: { topology: 'triangle-list' },
      }),
      comp: d.createRenderPipeline({
        layout: plC,
        vertex:   { module: this.renModule, entryPoint: 'fullVS' },
        fragment: { module: this.renModule, entryPoint: 'compositeFS', targets: [{ format: this.format }] },
        primitive: { topology: 'triangle-list' },
      }),
      // Depth-tests against the fluid prepass, so water in front of the ball
      // hides it. depthWrite off: nothing is drawn after this.
      ball: d.createRenderPipeline({
        layout: pl0,
        vertex:   { module: this.renModule, entryPoint: 'ballVS' },
        fragment: { module: this.renModule, entryPoint: 'ballFS', targets: [{ format: this.format }] },
        primitive: { topology: 'triangle-list' },
        depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less' },
      }),
    };

    this.samp = d.createSampler({ magFilter: 'linear', minFilter: 'linear',
                                  addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
  }

  resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width === w && this.canvas.height === h && this.tex) return;
    this.canvas.width = w; this.canvas.height = h;
    const d = this.device, U = GPUTextureUsage;
    for (const t of Object.values(this.tex || {})) t.destroy?.();
    const mk = (fmt, extra = 0) => d.createTexture({
      size: [w, h], format: fmt, usage: U.RENDER_ATTACHMENT | U.TEXTURE_BINDING | extra });
    this.tex = {
      bg:    mk('rgba16float'),
      depth: mk('rgba32float'),
      blurA: mk('rgba32float'),
      // Half resolution: thickness is a low-frequency quantity and this pass is
      // pure overdraw (N additive spheres, no depth test). Quartering its pixels
      // is invisible and is most of the render budget at high particle counts.
      thick: d.createTexture({ size: [Math.max(1, w >> 1), Math.max(1, h >> 1)],
        format: 'rgba16float', usage: U.RENDER_ATTACHMENT | U.TEXTURE_BINDING }),
      ds:    d.createTexture({ size: [w, h], format: 'depth24plus', usage: U.RENDER_ATTACHMENT }),
    };
    this.bgBlur = d.createBindGroup({ layout: this.blurBgl, entries: [
      { binding: 0, resource: this.tex.depth.createView() },
      { binding: 1, resource: { buffer: this.buf.blurU } }] });
    this.bgBlur2 = d.createBindGroup({ layout: this.blurBgl, entries: [
      { binding: 0, resource: this.tex.blurA.createView() },
      { binding: 1, resource: { buffer: this.buf.blurV } }] });
    this.bgComp = d.createBindGroup({ layout: this.compBgl, entries: [
      { binding: 4, resource: this.tex.depth.createView() },
      { binding: 5, resource: this.tex.thick.createView() },
      { binding: 6, resource: this.tex.bg.createView() },
      { binding: 7, resource: this.samp }] });
  }

  // ------------------------------------------------------------ state
  reset(n, scene = 'pool') {
    const want = n || this.count || 16000;
    const a = this.initialState(Math.min(want, this.nMax), scene);
    this.device.queue.writeBuffer(this.buf.parts, 0, a);
    this.device.queue.writeBuffer(this.buf.dbg, 0, new Uint32Array(16));
    this.frame = 0;
    this.substepCount = 0;
    this.seed = 0x9e37;
  }

  // Pour material into the world at the ball. Particles are appended past
  // `count`, so nothing already simulating is disturbed. The jitter is
  // splitmix over a running seed, so an identical call sequence produces an
  // identical world — spawning stays an input, not a source of noise.
  spawn(mat, cx, cy, cz, k) {
    k = Math.min(k, this.nMax - this.count);
    if (k <= 0) return 0;
    const a = new Int32Array(k * 16);
    const rad = this.ballR * 0.8;
    for (let i = 0; i < k; i++) {
      let dx, dy, dz, d2;
      do {
        dx = ((smix(this.seed++) % 2001) - 1000) / 1000;
        dy = ((smix(this.seed++) % 2001) - 1000) / 1000;
        dz = ((smix(this.seed++) % 2001) - 1000) / 1000;
        d2 = dx*dx + dy*dy + dz*dz;
      } while (d2 > 1);
      const o = i * 16;
      a[o + 0] = Math.round((cx + dx * rad) * ONE);
      a[o + 1] = Math.round((cy + dy * rad) * ONE);
      a[o + 2] = Math.round((cz + dz * rad) * ONE);
      a[o + 3] = mat;
    }
    this.device.queue.writeBuffer(this.buf.parts, this.count * 64, a);
    this.count += k;
    return k;
  }

  setBallRadius(R) {
    this.ballR = R;
    const { w, g } = buildBallLUT(R);
    this.device.queue.writeBuffer(this.buf.luts, BALL_OFF_W * 4, w);
    this.device.queue.writeBuffer(this.buf.luts, BALL_OFF_G * 4, g);
  }

  writeParams(defeat) {
    const p = new ArrayBuffer(96);
    const u = new Uint32Array(p), i = new Int32Array(p);
    u[0] = this.count; u[1] = this.cellTotal;
    u[2] = CFG.GX; u[3] = CFG.GY; u[4] = CFG.GZ;
    u[5] = this.offCellOf; u[6] = this.offBucket; u[7] = this.offSorted;
    u[8] = this.offBlock;  u[9] = this.nBlocks;
    u[10] = defeat ? 1 : 0; u[11] = this.frame;
    i[12] = CFG.gravStep; i[13] = CFG.damp;
    i[14] = (this.substepCount * 384) & 131071; i[15] = CFG.stirAmp;

    // The ball is an INPUT, not a source of nondeterminism: its position enters
    // the solver already quantised to Q16.16, so a recorded cursor track replays
    // bit-exactly. Nothing inside the sim ever sees a float.
    const b = this.ball;
    i[16] = Math.round(b.x * ONE);
    i[17] = Math.round(b.y * ONE);
    i[18] = Math.round(b.z * ONE);
    i[19] = Math.round((this.ballR + CFG.H) * ONE);   // reach: radius + kernel
    // While pouring, the ball is a NOZZLE, not a solid. spawn() seeds inside
    // ballR * 0.8, so leaving the collision on ejects every new particle at ~30x
    // the settled speed — a pour that detonates. The renderer still draws it.
    u[20] = (b.on && !this.brush.painting) ? 1 : 0;
    this.device.queue.writeBuffer(this.buf.params, 0, p);
  }

  // Camera eye, shared by the view matrix and the cursor ray so they can never
  // disagree about where the viewer is.
  eye() {
    const c = this.cam;
    return [
      c.target[0] + c.dist * Math.cos(c.pitch) * Math.sin(c.yaw),
      c.target[1] + c.dist * Math.sin(c.pitch),
      c.target[2] + c.dist * Math.cos(c.pitch) * Math.cos(c.yaw)];
  }

  // Cast the cursor onto the horizontal plane at the ball's current height.
  // Returns null when the ray is near-parallel to that plane or points away,
  // which is the case the naive division silently turns into a wild position.
  pickXZ(cx, cy) {
    const e = this.eye();
    const view = m4.lookAt(e, this.cam.target, [0, 1, 0]);
    const proj = m4.persp(50 * Math.PI / 180, this.canvas.width / this.canvas.height, 0.5, 900);
    const inv = m4.inv(m4.mul(proj, view));

    const nx = (cx / this.canvas.clientWidth) * 2 - 1;
    const ny = 1 - (cy / this.canvas.clientHeight) * 2;
    const un = (z) => {
      const x = inv[0]*nx + inv[4]*ny + inv[8]*z + inv[12];
      const y = inv[1]*nx + inv[5]*ny + inv[9]*z + inv[13];
      const w = inv[2]*nx + inv[6]*ny + inv[10]*z + inv[14];
      const q = inv[3]*nx + inv[7]*ny + inv[11]*z + inv[15];
      return [x/q, y/q, w/q];
    };
    const a = un(0), c = un(1);
    const dir = norm(sub(c, a));
    if (Math.abs(dir[1]) < 1e-4) return null;
    const t = (this.ball.y - e[1]) / dir[1];
    if (t <= 0) return null;

    const pad = this.ballR * 0.6;
    return [
      Math.max(pad, Math.min(this.world[0] - pad, e[0] + dir[0] * t)),
      Math.max(pad, Math.min(this.world[2] - pad, e[2] + dir[2] * t)),
    ];
  }

  substep(enc) {
    const n = this.count, wg = (x) => Math.ceil(x / 256);
    const pass = enc.beginComputePass();
    pass.setBindGroup(0, this.cbg);
    const go = (name, groups) => { pass.setPipeline(this.cp[name]); pass.dispatchWorkgroups(groups); };

    go('predict',    wg(n));
    go('clearCells', wg(this.cellTotal));
    go('gridCount',  wg(n));
    go('scanBlock',  this.nBlocks);
    go('scanTop',    1);
    go('scanAdd',    wg(this.cellTotal + 1));
    go('clearCells', wg(this.cellTotal));
    go('scatter',    wg(n));
    go('canonicalize', wg(n));
    go('buildNbr',   wg(n));
    for (let k = 0; k < CFG.iters; k++) {
      go('density', wg(n));
      go('solveDp', wg(n));
      go('applyDp', wg(n));
    }
    go('integrate', wg(n));
    pass.end();
    this.substepCount++;
  }

  // writeBuffer is a QUEUE operation: every write in this batch lands before the
  // single submit below, so a per-substep write would give all k substeps the
  // LAST value, not their own. Params are therefore written once per batch.
  // Called once per displayed frame, before stepping.
  tick() {
    if (this.brush.painting && this.ball.on) {
      this.spawn(this.brush.mat, this.ball.x, this.ball.y, this.ball.z, this.brush.rate);
    }
  }

  runSubsteps(k, defeat) {
    this.writeParams(defeat);
    const enc = this.device.createCommandEncoder();
    for (let s = 0; s < k; s++) this.substep(enc);
    this.device.queue.submit([enc.finish()]);
  }

  // ------------------------------------------------------------ digest
  async chain() {
    const enc = this.device.createCommandEncoder();
    this.writeParams(false);
    const pass = enc.beginComputePass();
    pass.setBindGroup(0, this.cbg);
    pass.setPipeline(this.cp.digestStripe);
    pass.dispatchWorkgroups(1);
    pass.end();
    enc.copyBufferToBuffer(this.buf.digest, 0, this.buf.read, 0, 256 * 4);
    this.device.queue.submit([enc.finish()]);
    await this.buf.read.mapAsync(GPUMapMode.READ);
    const w = new Uint32Array(this.buf.read.getMappedRange().slice(0));
    this.buf.read.unmap();

    // 8 independent FNV lanes over the 256 stripe words, folded in index order
    const lanes = new Uint32Array(8);
    for (let L = 0; L < 8; L++) lanes[L] = (2166136261 ^ (L * 0x9e3779b9)) >>> 0;
    for (let k = 0; k < 256; k++) {
      for (let L = 0; L < 8; L++) {
        let h = lanes[L] ^ w[k];
        h = Math.imul(h, 16777619) >>> 0;
        h = (h ^ (h >>> 13)) >>> 0;
        lanes[L] = h;
      }
    }
    return Array.from(lanes, x => x.toString(16).padStart(8, '0')).join('');
  }

  // Diagnostic: pull the first k particles back and report the solver's own
  // numbers. rho against RHO0 is the whole story when a fluid pancakes.
  async probe(k = 512) {
    const bytes = k * 64;
    if (!this._probeBuf || this._probeBuf.size < bytes) {
      this._probeBuf?.destroy?.();
      this._probeBuf = this.device.createBuffer({
        size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    }
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.buf.parts, 0, this._probeBuf, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await this._probeBuf.mapAsync(GPUMapMode.READ, 0, bytes);
    const a = new Int32Array(this._probeBuf.getMappedRange(0, bytes).slice(0));
    this._probeBuf.unmap();

    const rho = [], lam = [], py = [], sp = [];
    for (let i = 0; i < k; i++) {
      const o = i * 16;
      rho.push(a[o + 15]);            // dp.w
      lam.push(a[o + 11]);            // pr.w
      py.push(a[o + 1] / ONE);
      sp.push(Math.hypot(a[o+4], a[o+5], a[o+6]) / ONE);
    }
    const stat = (v) => {
      const s = [...v].sort((x, y) => x - y);
      return { min: s[0], p50: s[s.length >> 1], p95: s[(s.length * 95 / 100) | 0], max: s[s.length - 1] };
    };
    return { RHO0: CFG.RHO0, rho: stat(rho), lam: stat(lam), y: stat(py), speed: stat(sp) };
  }

  async dbgCounters() {
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.buf.dbg, 0, this.buf.readDbg, 0, 16 * 4);
    this.device.queue.submit([enc.finish()]);
    await this.buf.readDbg.mapAsync(GPUMapMode.READ);
    const v = Array.from(new Uint32Array(this.buf.readDbg.getMappedRange().slice(0)));
    this.buf.readDbg.unmap();
    return v;
  }

  // ------------------------------------------------------------ camera
  initCamera() {
    this.cam = { yaw: 0.72, pitch: 0.24, dist: 96, target: [this.world[0]/2, this.world[1]*0.38, this.world[2]/2] };
    this.ball = { x: this.world[0] / 2, y: this.world[1] * 0.30, z: this.world[2] / 2, on: false };
    this.brush = { mode: 'push', mat: 0, rate: 90, painting: false };

    // LEFT drag orbits. RIGHT drag (or shift) raises and lowers the ball, which
    // is the one axis a cursor cannot express on its own.
    let orbit = null, lift = null;
    const cv = this.canvas;
    cv.addEventListener('contextmenu', e => e.preventDefault());

    cv.addEventListener('pointerdown', e => {
      cv.setPointerCapture(e.pointerId);
      if (e.button === 2 || e.shiftKey) { lift = e.clientY; }
      else if (this.brush.mode === 'paint' && !e.altKey) { this.brush.painting = true; }
      else { orbit = { x: e.clientX, y: e.clientY }; }
    });
    cv.addEventListener('pointerup', () => { orbit = null; lift = null; this.brush.painting = false; });
    cv.addEventListener('pointerleave', () => { this.ball.on = false; });

    cv.addEventListener('pointermove', e => {
      if (orbit) {
        this.cam.yaw  -= (e.clientX - orbit.x) * 0.006;
        this.cam.pitch = Math.max(-1.4, Math.min(1.4, this.cam.pitch + (e.clientY - orbit.y) * 0.006));
        orbit = { x: e.clientX, y: e.clientY };
        return;
      }
      if (lift !== null) {
        const pad = this.ballR * 0.6;
        this.ball.y = Math.max(pad, Math.min(this.world[1] - pad,
                        this.ball.y - (e.clientY - lift) * 0.12));
        lift = e.clientY;
      }
      const r = cv.getBoundingClientRect();
      const hit = this.pickXZ(e.clientX - r.left, e.clientY - r.top);
      if (hit) { this.ball.x = hit[0]; this.ball.z = hit[1]; this.ball.on = this.ballEnabled; }
    });

    cv.addEventListener('wheel', e => {
      e.preventDefault();
      this.cam.dist = Math.max(25, Math.min(320, this.cam.dist * (1 + Math.sign(e.deltaY) * 0.08)));
    }, { passive: false });
  }

  writeRU() {
    const c = this.cam, w = this.canvas.width, h = this.canvas.height;
    const eye = [
      c.target[0] + c.dist * Math.cos(c.pitch) * Math.sin(c.yaw),
      c.target[1] + c.dist * Math.sin(c.pitch),
      c.target[2] + c.dist * Math.cos(c.pitch) * Math.cos(c.yaw)];
    const view = m4.lookAt(eye, c.target, [0, 1, 0]);
    const proj = m4.persp(50 * Math.PI / 180, w / h, 0.5, 900);

    const b = new ArrayBuffer(384);
    const f = new Float32Array(b), u = new Uint32Array(b);
    f.set(view, 0); f.set(proj, 16); f.set(m4.inv(proj), 32); f.set(m4.inv(view), 48);
    f[64] = w; f[65] = h;
    f[66] = this.ui.radius;
    u[67] = this.count;
    f.set([0.45, 0.86, 0.32, this.ui.view || 0], 68);   // light dir; w = debug view mode
    f.set([this.world[0], this.world[1], this.world[2], 0], 72);
    f.set([eye[0], eye[1], eye[2], 0], 76);
    f.set([this.ui.refract, this.ui.dispersion, this.ui.absorb, this.ui.blur], 80);
    const bl = this.ball;
    f.set([bl.x, bl.y, bl.z, bl.on ? this.ballR : 0], 84);
    this.device.queue.writeBuffer(this.buf.ru, 0, b);
  }

  // Render one frame and read it back as a data URL. The copy is recorded into
  // the SAME encoder as the draw, so it cannot miss the frame.
  async capture(scale = 1) {
    this.writeRU();
    const w = this.canvas.width, h = this.canvas.height;
    const bpr = Math.ceil(w * 4 / 256) * 256;
    const need = bpr * h;
    if (!this._capBuf || this._capBuf.size < need) {
      this._capBuf?.destroy?.();
      this._capBuf = this.device.createBuffer({
        size: need, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    }
    const tex = this.ctx.getCurrentTexture();
    const enc = this.device.createCommandEncoder();
    this.recordFrame(enc, tex.createView());
    enc.copyTextureToBuffer({ texture: tex }, { buffer: this._capBuf, bytesPerRow: bpr }, [w, h]);
    this.device.queue.submit([enc.finish()]);

    await this._capBuf.mapAsync(GPUMapMode.READ, 0, need);
    const src = new Uint8Array(this._capBuf.getMappedRange(0, need));
    const img = new ImageData(w, h);
    const bgra = this.format.startsWith('bgra');
    for (let y = 0; y < h; y++) {
      const so = y * bpr, dobase = y * w * 4;
      for (let x = 0; x < w; x++) {
        const s = so + x * 4, d = dobase + x * 4;
        img.data[d + 0] = bgra ? src[s + 2] : src[s + 0];
        img.data[d + 1] = src[s + 1];
        img.data[d + 2] = bgra ? src[s + 0] : src[s + 2];
        img.data[d + 3] = 255;
      }
    }
    this._capBuf.unmap();

    const full = new OffscreenCanvas(w, h);
    full.getContext('2d').putImageData(img, 0, 0);
    const ow = Math.round(w * scale), oh = Math.round(h * scale);
    const out = new OffscreenCanvas(ow, oh);
    out.getContext('2d').drawImage(full, 0, 0, ow, oh);
    const blob = await out.convertToBlob({ type: 'image/jpeg', quality: 0.92 });
    return await new Promise(r => { const fr = new FileReader(); fr.onload = () => r(fr.result); fr.readAsDataURL(blob); });
  }

  render() {
    this.writeRU();
    const enc = this.device.createCommandEncoder();
    this.recordFrame(enc, this.ctx.getCurrentTexture().createView());
    this.device.queue.submit([enc.finish()]);
  }

  recordFrame(enc, swap) {
    const clear = (view, r=0,g=0,b=0,a=1) => ({ view, loadOp: 'clear', storeOp: 'store', clearValue: {r,g,b,a} });

    let p = enc.beginRenderPass({ colorAttachments: [clear(this.tex.bg.createView())] });
    p.setPipeline(this.rp.bg); p.setBindGroup(0, this.rbg0); p.draw(36); p.end();

    p = enc.beginRenderPass({
      colorAttachments: [clear(this.tex.depth.createView())],
      depthStencilAttachment: { view: this.tex.ds.createView(), depthLoadOp: 'clear',
                                depthStoreOp: 'store', depthClearValue: 1.0 } });
    p.setPipeline(this.rp.depth); p.setBindGroup(0, this.rbg0); p.draw(6, this.count); p.end();

    p = enc.beginRenderPass({ colorAttachments: [clear(this.tex.thick.createView())] });
    p.setPipeline(this.rp.thick); p.setBindGroup(0, this.rbg0); p.draw(6, this.count); p.end();

    p = enc.beginRenderPass({ colorAttachments: [clear(this.tex.blurA.createView())] });
    p.setPipeline(this.rp.blur); p.setBindGroup(0, this.rbg0); p.setBindGroup(1, this.bgBlur); p.draw(3); p.end();

    p = enc.beginRenderPass({ colorAttachments: [clear(this.tex.depth.createView())] });
    p.setPipeline(this.rp.blur); p.setBindGroup(0, this.rbg0); p.setBindGroup(1, this.bgBlur2); p.draw(3); p.end();

    p = enc.beginRenderPass({ colorAttachments: [clear(swap)] });
    p.setPipeline(this.rp.comp); p.setBindGroup(0, this.rbg0); p.setBindGroup(1, this.bgComp); p.draw(3); p.end();

    if (this.ball.on) {
      p = enc.beginRenderPass({
        colorAttachments: [{ view: swap, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: this.tex.ds.createView(),
                                  depthLoadOp: 'load', depthStoreOp: 'store' } });
      p.setPipeline(this.rp.ball); p.setBindGroup(0, this.rbg0); p.draw(6); p.end();
    }
  }
}

export { App, CFG, ONE };
