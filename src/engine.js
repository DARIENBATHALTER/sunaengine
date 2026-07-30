// sunaEngine — WebGPU host engine
// Stripped PBF solver demonstrating bit-determinism.
import { digestState } from './hash.js';
import { DOM_W, DOM_H, ONE, CELL_SHIFT, MAXNBR, SUBSTEPS_PER_FRAME,
         VMAX, PARTICLE_WORDS } from './common.js';
import { loadFrozen } from './tables.js';

let tablesJson = null;

export class SunaEngine {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.w = opts.w ?? DOM_W;
    this.h = opts.h ?? DOM_H;
    this.maxParticles = opts.maxParticles ?? 8192;
    this._ready = false;
    this._device = null;
    this._n = 0;
    this._nFluid = 0;
    this._frame = 0;
  }

  async init() {
    if (!navigator.gpu) throw new Error('WebGPU not available');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No GPU adapter');
    const device = await adapter.requestDevice();
    this._device = device;

    // Load tables
    if (!tablesJson) {
      tablesJson = await fetch(new URL('./tables.json', import.meta.url)).then(r => r.json());
    }
    const lutImg = loadFrozen(tablesJson);
    this._lutData = lutImg;

    // Load WGSL
    const wgslCode = await fetch(new URL('./sim.wgsl', import.meta.url)).then(r => r.text());
    const module = device.createShaderModule({ code: wgslCode, label: 'sim' });
    const info = await module.getCompilationInfo();
    const bad = info.messages.filter(m => m.type !== 'info');
    if (bad.length) throw new Error(`Shader compile:\n${bad.map(m => `${m.type} L${m.lineNum}:${m.linePos} ${m.message}`).join('\n')}`);

    // Domain
    const cellShift = CELL_SHIFT;
    const cellSize = 1 << cellShift;
    const gridW = Math.ceil((this.w * ONE) / cellSize);
    const gridH = Math.ceil((this.h * ONE) / cellSize);
    const cellTotal = gridW * gridH;
    this._gridW = gridW;
    this._gridH = gridH;
    this._cellTotal = cellTotal;
    this._domW = this.w * ONE;
    this._domH = this.h * ONE;

    if (cellTotal > 1024) {
      throw new Error(`demo grid too large for the single-workgroup scan: ${cellTotal} cells (max 1024)`);
    }

    // Allocate buffers
    const n = this.maxParticles;
    const ST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const UNI = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
    const mkBuf = (size, usage, label) =>
      device.createBuffer({ size: Math.max(4, size), usage, label });

    // WebGPU guarantees only maxStorageBuffersPerShaderStage = 8. The first
    // public draft bound 12 storage buffers and EVERY pipeline silently failed
    // validation — the simulation never ran and the twin hashes still matched,
    // because two frozen worlds agree with each other. Scratch is therefore
    // packed into THREE buffers with explicit offsets in the shader:
    //   scratchA: cellCount[cellTotal] | cellStart[cellTotal+1]
    //   scratchB: cellOf[n] | bucketIds[n]
    //   nbrBlk:   nbr[n*MAXNBR] | nbrN[n]
    this.buf = {
      stateA:    mkBuf(n * PARTICLE_WORDS * 4, ST | GPUBufferUsage.COPY_SRC, 'stateA'),
      stateB:    mkBuf(n * PARTICLE_WORDS * 4, ST, 'stateB'),
      derived:   mkBuf(n * 64, ST, 'derived'),
      scratchA:  mkBuf((2 * cellTotal + 1) * 4, ST, 'scratchA'),
      scratchB:  mkBuf(2 * n * 4, ST, 'scratchB'),
      nbrBlk:    mkBuf((n * MAXNBR + n) * 4, ST, 'nbrBlk'),
      luts:      mkBuf(lutImg.length * 4, ST, 'luts'),
      params:    mkBuf(48, UNI, 'params'),
      readback:  mkBuf(n * PARTICLE_WORDS * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, 'readback'),
      pushReadback: mkBuf(n * PARTICLE_WORDS * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, 'pushReadback'),
    };

    // Write LUTs
    device.queue.writeBuffer(this.buf.luts, 0, new Int32Array(lutImg));

    // Explicit bind-group layout shared by all passes. 1 uniform + 7 storage,
    // inside the default limit of 8 storage buffers per compute stage.
    this._bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      ],
    });
    this._pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this._bindGroupLayout],
    });

    // Create pipelines
    const mkPipe = (entry) =>
      device.createComputePipeline({
        layout: this._pipelineLayout,
        label: entry,
        compute: { module, entryPoint: entry },
      });

    this.pipe = {
      gridCount:    mkPipe('gridCount', 256),
      gridScan:     mkPipe('gridScan', 256),
      gridSort:     mkPipe('gridSort', 256),
      buildNbr:     mkPipe('buildNbr', 256),
      predict:      mkPipe('predict', 256),
      solveA:       mkPipe('solveA', 256),
      solveB:       mkPipe('solveB', 256),
      applyDp:      mkPipe('applyDp', 256),
      finalize:     mkPipe('finalize', 256),
      copyBoundary: mkPipe('copyBoundary', 256),
    };

    this._ready = true;

    // Initialize empty state
    this._writeFullState(new Int32Array(n * PARTICLE_WORDS));
    this._setupWalls();
    return this;
  }

  _setupWalls() {
    const n = this.maxParticles;
    const domW = this._domW;
    const domH = this._domH;
    const state = new Int32Array(n * PARTICLE_WORDS);
    let bi = 0;
    const add = (x, y) => {
      if (bi >= n) return;
      const b = bi * PARTICLE_WORDS;
      state[b] = x; state[b + 1] = y;
      bi++;
    };
    for (let x = ONE; x < domW; x += ONE) {
      add(x, ONE);
      add(x, domH - ONE);
    }
    for (let y = ONE * 2; y < domH - ONE; y += ONE) {
      add(ONE, y);
      add(domW - ONE, y);
    }
    this._n = bi;
    this._nFluid = 0;
    this._writeFullState(state);
  }

  _writeFullState(data) {
    this._device.queue.writeBuffer(this.buf.stateA, 0, data);
    this._device.queue.writeBuffer(this.buf.stateB, 0, data);
    this._writeParams();
  }

  _writeParams() {
    const p = new Uint32Array([
      this._n, this._nFluid, this._gridW, this._gridH,
      this._cellTotal, CELL_SHIFT, 0, 0,
      this._domW, this._domH, VMAX, MAXNBR,
    ]);
    this._device.queue.writeBuffer(this.buf.params, 0, p);
  }

  /**
   * Spawn fluid particles at a point. Writes AFTER all existing particles.
   */
  spawn(cx, cy, radius = 5) {
    const maxN = this.maxParticles;
    const cxT = Math.round(cx * ONE);
    const cyT = Math.round(cy * ONE);
    const rT = Math.round(radius * ONE);
    // Rest spacing is 1.00 wu (the solver's RHO0 calibration). The old hex
    // packing (row pitch 0.866 wu) overpacked spawns by 1/0.866^2 = 1.33x rest
    // density, so every spawn detonated against RHO0 before it could fall.
    const spacing = Math.round(ONE * 1.02);
    const rLim = Math.ceil(radius) + 1;

    // Build list of new particle positions
    const newParticles = [];
    for (let row = -rLim; row <= rLim; row++) {
      for (let col = -rLim; col <= rLim; col++) {
        if (this._n + newParticles.length >= maxN) break;
        const x = cxT + col * spacing;
        const y = cyT + row * spacing;
        const dx = x - cxT, dy = y - cyT;
        if (dx * dx + dy * dy > rT * rT) continue;
        if (x <= ONE || x >= this._domW - ONE || y <= ONE || y >= this._domH - ONE) continue;
        newParticles.push({ x, y });
      }
    }

    if (newParticles.length === 0) return 0;

    // Write AFTER all existing particles (boundary + fluid)
    const offset = this._n * PARTICLE_WORDS;
    const spawnData = new Int32Array(newParticles.length * PARTICLE_WORDS);
    for (let i = 0; i < newParticles.length; i++) {
      const b = i * PARTICLE_WORDS;
      spawnData[b] = newParticles[i].x;
      spawnData[b + 1] = newParticles[i].y;
      // vel=(0,0), matId=0, flags=0, pad=0 — all zero by default
    }

    // Write into BOTH ping-pong slots; whichever is current, the next step's
    // input holds the spawn. `step()` keeps buf.stateA pointing at the input
    // of the next substep.
    this._device.queue.writeBuffer(this.buf.stateA, offset * 4, spawnData);
    this._device.queue.writeBuffer(this.buf.stateB, offset * 4, spawnData);

    this._n += newParticles.length;
    this._nFluid += newParticles.length;
    this._writeParams();
    return newParticles.length;
  }

  /**
   * Advance simulation by `substeps` substeps.
   * Grid: count -> deterministic prefix scan -> scatter -> neighbour lists.
   * Everything lives in one command encoder per substep; queue order between
   * passes inside an encoder is submission order, which is all we need.
   */
  step(substeps = SUBSTEPS_PER_FRAME) {
    if (!this._ready) return;
    const device = this._device;
    const pipe = this.pipe;
    const buf = this.buf;
    const n = this._n;
    const cellTotal = this._cellTotal;
    const wg = (count) => Math.ceil(count / 256);

    let stateIn = buf.stateA;
    let stateOut = buf.stateB;

    const bgA = device.createBindGroup({
      layout: this._bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: buf.params } },
        { binding: 1, resource: { buffer: buf.stateA } },
        { binding: 2, resource: { buffer: buf.stateB } },
        { binding: 3, resource: { buffer: buf.derived } },
        { binding: 4, resource: { buffer: buf.scratchA } },
        { binding: 5, resource: { buffer: buf.scratchB } },
        { binding: 6, resource: { buffer: buf.nbrBlk } },
        { binding: 7, resource: { buffer: buf.luts } },
      ],
    });
    const bgB = device.createBindGroup({
      layout: this._bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: buf.params } },
        { binding: 1, resource: { buffer: buf.stateB } },
        { binding: 2, resource: { buffer: buf.stateA } },
        { binding: 3, resource: { buffer: buf.derived } },
        { binding: 4, resource: { buffer: buf.scratchA } },
        { binding: 5, resource: { buffer: buf.scratchB } },
        { binding: 6, resource: { buffer: buf.nbrBlk } },
        { binding: 7, resource: { buffer: buf.luts } },
      ],
    });

    for (let s = 0; s < substeps; s++) {
      const enc = device.createCommandEncoder();
      const bg = (stateIn === buf.stateA) ? bgA : bgB;

      // 1. Clear the cell counters (only that segment of scratchA).
      enc.clearBuffer(buf.scratchA, 0, cellTotal * 4);

      const run = (pipeline, groups) => {
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(groups);
        pass.end();
      };

      run(pipe.gridCount, wg(n));
      run(pipe.gridScan, 1);
      run(pipe.gridSort, wg(n));
      run(pipe.buildNbr, wg(n));
      run(pipe.predict, wg(n));
      run(pipe.solveA, wg(n));
      for (let iter = 0; iter < 4; iter++) {
        run(pipe.solveB, wg(n));
        run(pipe.applyDp, wg(n));
      }
      run(pipe.finalize, wg(n));
      run(pipe.copyBoundary, wg(n));

      device.queue.submit([enc.finish()]);

      // Swap
      [stateIn, stateOut] = [stateOut, stateIn];
    }

    this.buf.stateA = stateIn;
    this.buf.stateB = stateOut;
    this._frame++;
  }

  /**
   * Read back particle positions and draw as circles on canvas.
   */
  async render() {
    if (!this._ready) return null;
    const device = this._device;
    const buf = this.buf;
    const n = this._n;
    const stateSize = n * PARTICLE_WORDS * 4;
    if (stateSize === 0) return null;

    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buf.stateA, 0, buf.readback, 0, stateSize);
    device.queue.submit([enc.finish()]);

    await buf.readback.mapAsync(GPUMapMode.READ);
    const raw = new Int32Array(buf.readback.getMappedRange().slice(0));
    const state = new Int32Array(raw);
    buf.readback.unmap();

    // Canvas2D rendering
    const ctx = this.ctx;
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    const scaleX = cw / this.w;
    const scaleY = ch / this.h;

    ctx.clearRect(0, 0, cw, ch);
    ctx.fillStyle = '#080812';
    ctx.fillRect(0, 0, cw, ch);

    for (let i = 0; i < n; i++) {
      const b = i * PARTICLE_WORDS;
      const px = state[b];
      const py = state[b + 1];
      const cx = (px / ONE) * scaleX;
      const cy = ch - (py / ONE) * scaleY;
      const r = Math.max(2, scaleX * 0.5);

      // Buffer layout is [walls, fluid]: fluid particles are the LAST nFluid
      // slots. (The stripped renderer mistook walls for the fluid band, so the
      // box edge glowed bright and the water rendered dim.)
      const wallCount = n - this._nFluid;
      ctx.fillStyle = i >= wallCount
        ? 'rgba(64, 170, 255, 0.9)'
        : 'rgba(96, 96, 112, 0.45)';

      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();
    }

    this._lastState = state;
    return state;
  }

  /**
   * Apply an impulse to particles near (cx, cy).
   * Overlapping pushes are dropped: one mapped readback at a time.
   */
  async pushParticles(cx, cy, radius, fx, fy) {
    if (!this._ready || this._pushPending) return;
    this._pushPending = true;
    try {
      const n = this._n;
    const stateSize = n * PARTICLE_WORDS * 4;
    if (stateSize === 0) return;

    const enc = this._device.createCommandEncoder();
    enc.copyBufferToBuffer(this.buf.stateA, 0, this.buf.pushReadback, 0, stateSize);
    this._device.queue.submit([enc.finish()]);

    await this.buf.pushReadback.mapAsync(GPUMapMode.READ);
    const raw = new Int32Array(this.buf.pushReadback.getMappedRange().slice(0));
    const state = new Int32Array(raw);
    this.buf.pushReadback.unmap();

    const cxT = Math.round(cx * ONE);
    const cyT = Math.round(cy * ONE);
    const rT = Math.round(radius * ONE);
    const r2T = rT * rT;

    for (let i = 0; i < n; i++) {
      const b = i * PARTICLE_WORDS;
      const px = state[b];
      const py = state[b + 1];
      const dx = px - cxT;
      const dy = py - cyT;
      if (dx * dx + dy * dy > r2T) continue;
      state[b + 2] = Math.max(-VMAX, Math.min(VMAX, state[b + 2] + fx));
      state[b + 3] = Math.max(-VMAX, Math.min(VMAX, state[b + 3] + fy));
    }

    this._device.queue.writeBuffer(this.buf.stateA, 0, state, 0, stateSize);
    this._device.queue.writeBuffer(this.buf.stateB, 0, state, 0, stateSize);
    } finally {
      this._pushPending = false;
    }
  }

  getHash() {
    if (!this._lastState) return '';
    const u32 = new Uint32Array(this._lastState.buffer, this._lastState.byteOffset, this._lastState.length);
    return digestState(u32, this._n, PARTICLE_WORDS).hex;
  }

  snapshot() {
    if (!this._lastState) return null;
    return {
      state: new Int32Array(this._lastState),
      n: this._n,
      nFluid: this._nFluid,
      frame: this._frame,
    };
  }

  restore(snap) {
    if (!snap) return;
    // Destroy old buffers before creating new ones
    if (this.buf.stateA) { this.buf.stateA.destroy(); }
    if (this.buf.stateB) { this.buf.stateB.destroy(); }
    const ST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.buf.stateA = this._device.createBuffer({
      size: Math.max(4, snap.state.length * 4), usage: ST, label: 'stateA',
    });
    this._device.queue.writeBuffer(this.buf.stateA, 0, snap.state);
    this.buf.stateB = this._device.createBuffer({
      size: Math.max(4, snap.state.length * 4), usage: ST, label: 'stateB',
    });
    this._device.queue.writeBuffer(this.buf.stateB, 0, snap.state);
    this._n = snap.n;
    this._nFluid = snap.nFluid;
    this._frame = snap.frame;
    this._writeParams();
  }

  async reset() {
    this._n = 0;
    this._nFluid = 0;
    this._frame = 0;
    this._lastState = null;
    // Recreate full-sized buffers in case restore() shrunk them
    if (this.buf.stateA) this.buf.stateA.destroy();
    if (this.buf.stateB) this.buf.stateB.destroy();
    const n = this.maxParticles;
    const ST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.buf.stateA = this._device.createBuffer({
      size: Math.max(4, n * PARTICLE_WORDS * 4), usage: ST | GPUBufferUsage.COPY_SRC, label: 'stateA',
    });
    this.buf.stateB = this._device.createBuffer({
      size: Math.max(4, n * PARTICLE_WORDS * 4), usage: ST, label: 'stateB',
    });
    this._writeFullState(new Int32Array(n * PARTICLE_WORDS));
    this._setupWalls();
  }

  get n() { return this._n; }
  get nFluid() { return this._nFluid; }
  get frame() { return this._frame; }
}
