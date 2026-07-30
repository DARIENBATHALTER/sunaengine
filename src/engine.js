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

    // Allocate buffers
    const n = this.maxParticles;
    const ST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const UNI = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
    const mkBuf = (size, usage, label) =>
      device.createBuffer({ size: Math.max(4, size), usage, label });

    this.buf = {
      stateA:    mkBuf(n * PARTICLE_WORDS * 4, ST | GPUBufferUsage.COPY_SRC, 'stateA'),
      stateB:    mkBuf(n * PARTICLE_WORDS * 4, ST, 'stateB'),
      derived:   mkBuf(n * 64, ST, 'derived'),
      cellCount: mkBuf(cellTotal * 4, ST, 'cellCount'),
      cellStart: mkBuf((cellTotal + 1) * 4, ST, 'cellStart'),
      blockSums: mkBuf(Math.ceil(cellTotal / 256) * 4, ST, 'blockSums'),
      cellOf:    mkBuf(n * 4, ST, 'cellOf'),
      bucketIds: mkBuf(n * 4, ST, 'bucketIds'),
      sortedIds: mkBuf(n * 4, ST, 'sortedIds'),
      nbr:       mkBuf(n * MAXNBR * 4, ST, 'nbr'),
      nbrN:      mkBuf(n * 4, ST, 'nbrN'),
      luts:      mkBuf(lutImg.length * 4, ST, 'luts'),
      params:    mkBuf(48, UNI, 'params'),
      readback:  mkBuf(n * PARTICLE_WORDS * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, 'readback'),
    };

    // Write LUTs
    device.queue.writeBuffer(this.buf.luts, 0, new Int32Array(lutImg));

    // Create pipelines
    const mkPipe = (entry, wgs) =>
      device.createComputePipeline({
        layout: 'auto',
        label: entry,
        compute: { module, entryPoint: entry, ...(wgs != null ? { constants: { wgs } } : {}) },
      });

    this.pipe = {
      scanBlock:    mkPipe('scanBlock', 256),
      scanAdd:      mkPipe('scanAdd', 256),
      gridCount:    mkPipe('gridCount', 256),
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
   * Spawn fluid particles at a point. Builds state on CPU and writes to GPU.
   */
  spawn(cx, cy, radius = 5) {
    const n = this.maxParticles;
    const state = new Int32Array(n * PARTICLE_WORDS);

    // Read back to preserve existing state (simplified: just rebuild)
    // For now, we track state on CPU side
    let nextIdx = this._nFluid;
    const cxT = Math.round(cx * ONE);
    const cyT = Math.round(cy * ONE);
    const rT = Math.round(radius * ONE);
    const spacing = ONE;

    const rows = Math.ceil(rT / spacing) * 2 + 1;

    // Copy existing fluid + boundary into state
    for (let i = 0; i < this._n && i < n; i++) {
      // We'll need to read back. For simplicity, zero out and rebuild.
      // Actually, let's just build from scratch using our tracking.
    }

    // Simpler approach: just spawn new particles by writing directly
    // Build a list of new particle positions
    const newParticles = [];
    for (let row = -rows; row <= rows && nextIdx + newParticles.length < n; row++) {
      for (let col = -rows; col <= rows && nextIdx + newParticles.length < n; col++) {
        const x = cxT + Math.round(col * spacing + (row & 1) * spacing / 2);
        const y = cyT + Math.round(row * spacing * 0.866);
        const dx = x - cxT, dy = y - cyT;
        if (dx * dx + dy * dy > rT * rT) continue;
        if (x <= ONE || x >= this._domW - ONE || y <= ONE || y >= this._domH - ONE) continue;
        newParticles.push({ x, y });
      }
    }

    if (newParticles.length === 0) return 0;

    // Read back current state to preserve existing particles
    // Instead, we'll just write the new particles at the correct offset
    const offset = this._nFluid * PARTICLE_WORDS;
    const spawnData = new Int32Array(newParticles.length * PARTICLE_WORDS);
    for (let i = 0; i < newParticles.length; i++) {
      const b = i * PARTICLE_WORDS;
      spawnData[b] = newParticles[i].x;
      spawnData[b + 1] = newParticles[i].y;
      // vel = (0,0), matId = 0, flags = 0, pad = 0
    }

    this._device.queue.writeBuffer(this.buf.stateA, offset * 4, spawnData);
    this._device.queue.writeBuffer(this.buf.stateB, offset * 4, spawnData);

    this._n += newParticles.length;
    this._nFluid += newParticles.length;
    this._writeParams();
    return newParticles.length;
  }

  /**
   * Advance simulation by `substeps` substeps.
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

    for (let s = 0; s < substeps; s++) {
      const enc = device.createCommandEncoder();

      // Helper: create bind group with current state buffers
      const makeBG = (sin, sout) => device.createBindGroup({
        layout: pipe.gridCount.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: buf.params } },
          { binding: 1, resource: { buffer: sin } },
          { binding: 2, resource: { buffer: sout } },
          { binding: 3, resource: { buffer: buf.derived } },
          { binding: 4, resource: { buffer: buf.cellCount } },
          { binding: 5, resource: { buffer: buf.cellStart } },
          { binding: 6, resource: { buffer: buf.blockSums } },
          { binding: 7, resource: { buffer: buf.cellOf } },
          { binding: 8, resource: { buffer: buf.bucketIds } },
          { binding: 9, resource: { buffer: buf.sortedIds } },
          { binding: 10, resource: { buffer: buf.nbr } },
          { binding: 11, resource: { buffer: buf.nbrN } },
          { binding: 12, resource: { buffer: buf.luts } },
        ],
      });

      // 1. Clear cellCount
      enc.clearBuffer(buf.cellCount, 0, cellTotal * 4);

      // 2. gridCount
      {
        const pass = enc.beginComputePass();
        pass.setPipeline(pipe.gridCount);
        pass.setBindGroup(0, makeBG(stateIn, stateOut));
        pass.dispatchWorkgroups(wg(n));
        pass.end();
      }

      // 3. Prefix sum scan (two passes)
      enc.clearBuffer(buf.blockSums, 0, buf.blockSums.size);
      {
        const pass = enc.beginComputePass();
        pass.setPipeline(pipe.scanBlock);
        pass.setBindGroup(0, makeBG(stateIn, stateOut));
        pass.dispatchWorkgroups(wg(cellTotal));
        pass.end();
      }
      {
        const pass = enc.beginComputePass();
        pass.setPipeline(pipe.scanAdd);
        pass.setBindGroup(0, makeBG(stateIn, stateOut));
        pass.dispatchWorkgroups(wg(cellTotal));
        pass.end();
      }

      // 4. gridSort
      {
        const pass = enc.beginComputePass();
        pass.setPipeline(pipe.gridSort);
        pass.setBindGroup(0, makeBG(stateIn, stateOut));
        pass.dispatchWorkgroups(wg(n));
        pass.end();
      }

      // 5. buildNbr
      {
        const pass = enc.beginComputePass();
        pass.setPipeline(pipe.buildNbr);
        pass.setBindGroup(0, makeBG(stateIn, stateOut));
        pass.dispatchWorkgroups(wg(n));
        pass.end();
      }

      // 6. predict
      {
        const pass = enc.beginComputePass();
        pass.setPipeline(pipe.predict);
        pass.setBindGroup(0, makeBG(stateIn, stateOut));
        pass.dispatchWorkgroups(wg(n));
        pass.end();
      }

      // 7. solveA
      {
        const pass = enc.beginComputePass();
        pass.setPipeline(pipe.solveA);
        pass.setBindGroup(0, makeBG(stateIn, stateOut));
        pass.dispatchWorkgroups(wg(n));
        pass.end();
      }

      // 8. PBF iterations (solveB + applyDp) × 4
      for (let iter = 0; iter < 4; iter++) {
        {
          const pass = enc.beginComputePass();
          pass.setPipeline(pipe.solveB);
          pass.setBindGroup(0, makeBG(stateIn, stateOut));
          pass.dispatchWorkgroups(wg(n));
          pass.end();
        }
        {
          const pass = enc.beginComputePass();
          pass.setPipeline(pipe.applyDp);
          pass.setBindGroup(0, makeBG(stateIn, stateOut));
          pass.dispatchWorkgroups(wg(n));
          pass.end();
        }
      }

      // 9. finalize
      {
        const pass = enc.beginComputePass();
        pass.setPipeline(pipe.finalize);
        pass.setBindGroup(0, makeBG(stateIn, stateOut));
        pass.dispatchWorkgroups(wg(n));
        pass.end();
      }

      // 10. copy boundary
      {
        const pass = enc.beginComputePass();
        pass.setPipeline(pipe.copyBoundary);
        pass.setBindGroup(0, makeBG(stateIn, stateOut));
        pass.dispatchWorkgroups(wg(n));
        pass.end();
      }

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

      ctx.fillStyle = i < this._nFluid
        ? 'rgba(64, 160, 255, 0.85)'
        : 'rgba(96, 96, 112, 0.5)';

      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();
    }

    this._lastState = state;
    return state;
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
    this.buf.stateA = this._device.createBuffer({
      size: Math.max(4, snap.state.length * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label: 'stateA',
    });
    this._device.queue.writeBuffer(this.buf.stateA, 0, snap.state);
    this.buf.stateB = this._device.createBuffer({
      size: Math.max(4, snap.state.length * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label: 'stateB',
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
    const n = this.maxParticles;
    this._writeFullState(new Int32Array(n * PARTICLE_WORDS));
    this._setupWalls();
  }

  get n() { return this._n; }
  get nFluid() { return this._nFluid; }
  get frame() { return this._frame; }
}
