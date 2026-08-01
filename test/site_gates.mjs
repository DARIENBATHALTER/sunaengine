#!/usr/bin/env node
// SunaEngine — SITE GATES: UI-path checks of index.html through ONE headless
// Chrome (serial, foreground), plus the cross-repo extraction proof re-run.
// Copyright (C) 2026 Darien Bathalter
// SPDX-License-Identifier: AGPL-3.0-only — see LICENSE. NO WARRANTY.
//
// Usage:
//   node test/site_gates.mjs                # all gates, g1..g6
//   node test/site_gates.mjs g1 g3 g6      # a subset
//   node test/site_gates.mjs g5.aether     # g5 split into resumable thirds
//   node test/site_gates.mjs g5.suna       #   (long dam-break runs; JSON
//   node test/site_gates.mjs g5.compare    #    lands in $TMPDIR/suna-site-gates)
//
// House rules:
//   * every gate asserts MECHANISMS, STATE and COUNTS — never prose;
//   * expectations are never adjusted; the code is;
//   * at most one headless Chrome exists at any moment: g1–g4+g6 share a
//     single instance, and g5's two cdp.mjs runs start only after it is dead,
//     serially. Only OUR spawned child is ever killed — never a pattern.
//
// The gates (from the demo brief):
//   g1 TWINS-EQUAL          hashes equal AND changing over ~600 substeps
//   g2 TWINS-UNDER-INPUT    real Drop-More press; equal after settle; n grew;
//                           n NEVER exceeds the cap (driven to refusal)
//   g3 DIVERGENCE CONTROL   one extra particle in sim B only -> DOM must show
//                           divergence within 60 substeps (display is live)
//   g4 REPLAY-BYTE-IDENTITY pointer interaction -> synthetic mouseleave ->
//                           auto-replay final digest == recorded digest;
//                           3 scrub positions (incl. backward), each lands on
//                           the same digest twice
//   g5 EXTRACTION           cross-repo dam-break proof still byte-identical
//                           to the frozen values in extraction_gate.md
//   g6 HYGIENE              zero console errors; only-localhost network with
//                           DNS hard-blocked; no quarantine bytes in the tree;
//                           AGPL LICENSE; OE-CAKE! credit line present

import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync, existsSync } from 'node:fs';
import http from 'node:http';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..'); // sunaengine
const AETHER = '/Users/darien/aether';
const SCRATCH = join(tmpdir(), 'suna-site-gates');
mkdirSync(SCRATCH, { recursive: true });

// Frozen cross-repo expectation (test/extraction_gate.md, 2026-07-31,
// aether @ e41e06f). Never adjusted.
const FROZEN = {
  chain: '20918f97fbcbb409e7d058ebcae6f6725973b9a19042168742b07e8c17126d14',
  checkpoints: {
    1: 'c31aa57fb218829e3b78250e5909ddbbbb872c7d60e5b674f78d1a7164d8e0be',
    10: '95984e71128a222a27f285a340d178384a588ca74d27abb2448e08d3c4239c04',
    100: '320e900be986aa4e63ee1d110a4fb798d7e07e39ca7828782c475c1f206204eb',
    1000: 'b1844e32b4d2bd4b3dbd91cd4ffb7ead0bf568d6fea0dc53c8d555134f946d9b',
  },
};

// Must equal src/demo_boot.js's DEMO_PARTICLE_CAP (raised to 6000 2026-08-01;
// the gates run in node and the demo module needs a browser, so the number is
// pinned here and the mismatch shows up as exactly the red this comment sits on).
const CAP = 6000;
const HEX64 = /^[0-9a-f]{64}$/;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------------------
// reporting
// ---------------------------------------------------------------------------
const lines = [];
const say = (s) => { lines.push(s); console.log(s); };
class GateFail extends Error {}
const A = (cond, msg) => { if (!cond) throw new GateFail(msg); say('  ok  ' + msg); };
const results = new Map(); // gate -> 'GREEN' | 'RED: reason'

// ---------------------------------------------------------------------------
// static file server (self-contained; MIME matters for ES modules)
// ---------------------------------------------------------------------------
const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.json': 'application/json', '.wgsl': 'text/plain', '.png': 'image/png',
  '.md': 'text/plain', '.css': 'text/css',
};
function serveDir(dir) {
  const server = http.createServer((req, res) => {
    try {
      const p = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname));
      if (p.includes('..')) { res.writeHead(400); res.end(); return; }
      let file = join(dir, p);
      if (existsSync(file) && statSync(file).isDirectory()) file = join(file, 'index.html');
      const body = readFileSync(file);
      res.writeHead(200, { 'content-type': MIME[extname(file)] || 'application/octet-stream' });
      res.end(body);
    } catch { res.writeHead(404); res.end('not found'); }
  });
  return new Promise((res) => server.listen(0, '127.0.0.1', () => res({ server, port: server.address().port })));
}

// ---------------------------------------------------------------------------
// one headless Chrome, driven over raw CDP (node >= 22: global WebSocket)
// ---------------------------------------------------------------------------
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
let chromeChild = null;
const PROFILE = join(SCRATCH, 'profile-' + process.pid);
const killChrome = () => {
  if (chromeChild) { try { chromeChild.kill(); } catch {} chromeChild = null; }
  try { rmSync(PROFILE, { recursive: true, force: true }); } catch {}
};
process.on('exit', killChrome);
for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP']) process.on(sig, () => { killChrome(); process.exit(1); });

async function launchChrome() {
  const port = 9533 + (process.pid % 199); // disjoint from cdp.mjs's 9333+pid%200
  chromeChild = spawn(CHROME, [
    '--headless=new',
    `--remote-debugging-port=${port}`,
    '--enable-unsafe-webgpu',
    '--enable-features=WebGPU',
    '--use-angle=metal',
    '--no-sandbox',
    '--no-first-run',
    '--disable-dev-shm-usage',
    '--window-size=1280,3200',
    // DNS is dead except 127.0.0.1 (MAP * catches IP literals too — measured:
    // without the EXCLUDE, even http://127.0.0.1 dies with ERR_NAME_NOT_RESOLVED).
    // Any request that tries to leave localhost hard-fails; g6 then asserts the
    // page still booted fully and every observed request was local.
    '--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1',
    `--user-data-dir=${PROFILE}`,
    'about:blank',
  ], { stdio: ['ignore', 'ignore', 'pipe'] });
  let stderr = '';
  chromeChild.stderr.on('data', (d) => { stderr += d; });
  for (let i = 0; i < 100; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
      const page = list.find((t) => t.type === 'page');
      if (page?.webSocketDebuggerUrl) return connectCdp(page.webSocketDebuggerUrl);
    } catch {}
    await sleep(100);
  }
  throw new Error('Chrome CDP never came up. stderr: ' + stderr.slice(0, 1500));
}

function connectCdp(wsUrl) {
  return new Promise((resolveP, rejectP) => {
    const ws = new WebSocket(wsUrl);
    const pending = new Map();
    const consoleEvents = [];   // { type, text }
    const exceptions = [];      // strings
    const requests = [];        // urls
    let id = 0;
    ws.addEventListener('open', () => resolveP({ send, ev, until, consoleEvents, exceptions, requests, close: () => ws.close() }));
    ws.addEventListener('error', (e) => rejectP(new Error('ws error: ' + e.message)));
    ws.addEventListener('message', (m) => {
      const msg = JSON.parse(m.data);
      if (msg.id !== undefined && pending.has(msg.id)) {
        const { res, rej } = pending.get(msg.id);
        pending.delete(msg.id);
        msg.error ? rej(new Error(JSON.stringify(msg.error))) : res(msg.result);
      } else if (msg.method === 'Runtime.consoleAPICalled') {
        consoleEvents.push({
          type: msg.params.type,
          text: msg.params.args.map((a) => a.value ?? a.description ?? '').join(' '),
        });
      } else if (msg.method === 'Runtime.exceptionThrown') {
        exceptions.push(msg.params.exceptionDetails?.exception?.description ||
                        msg.params.exceptionDetails?.text || 'exception');
      } else if (msg.method === 'Network.requestWillBeSent') {
        requests.push(msg.params.request.url);
      }
    });
    function send(method, params = {}) {
      const myId = ++id;
      return new Promise((res, rej) => {
        pending.set(myId, { res, rej });
        ws.send(JSON.stringify({ id: myId, method, params }));
      });
    }
    async function ev(expression, { awaitP = false } = {}) {
      const r = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: awaitP });
      if (r.exceptionDetails) {
        throw new GateFail('page eval threw: ' +
          (r.exceptionDetails.exception?.description || r.exceptionDetails.text));
      }
      return r.result?.value;
    }
    async function until(desc, expression, timeout = 60000, interval = 150) {
      const deadline = Date.now() + timeout;
      let last;
      while (Date.now() < deadline) {
        last = await ev(expression);
        if (last) return last;
        await sleep(interval);
      }
      throw new GateFail(`timeout (${timeout}ms) waiting for ${desc}; last=${JSON.stringify(last)}`);
    }
  });
}

// shared page-state readers ---------------------------------------------------
const READY = `!!(window.__GATE__ && __GATE__.T.ready && __GATE__.P.ready)`;
const TWINS_STATE = `(() => {
  const g = window.__GATE__, $ = (i) => document.getElementById(i);
  const strip = (i) => $(i).textContent.replace(/\\s+/g, '');
  return {
    sub: g.T.a.substep, subB: g.T.b.substep,
    na: g.T.a.n, nb: g.T.b.n,
    hashA: strip('hashA'), hashB: strip('hashB'),
    clsA: $('hashA').className, clsB: $('hashB').className,
    verdictCls: $('twinsVerdict').className,
    verdictSub: Number((($('twinsVerdict').textContent.match(/substep ([\\d\\s]+)/) || [,'-1'])[1]).replace(/\\D/g, '') || '-1'),
    lastEqual: g.T.lastEqual,
  };
})()`;

async function navTo(cdp, url) {
  await cdp.send('Page.navigate', { url });
  await cdp.until('page boot (__GATE__ + both figures ready)', READY, 90000);
}
async function centerOf(cdp, id) {
  const r = await cdp.ev(`(() => { const r = document.getElementById('${id}').getBoundingClientRect();
    return { x: Math.round(r.x + r.width / 2), y: Math.round(r.y + r.height / 2), w: r.width, h: r.height }; })()`);
  if (!(r.w > 0 && r.h > 0 && r.y < 3200)) throw new GateFail(`#${id} not laid out in viewport: ${JSON.stringify(r)}`);
  return r;
}
const mouse = (cdp, type, x, y, buttons) => cdp.send('Input.dispatchMouseEvent', {
  type, x, y, button: 'left', buttons, clickCount: type === 'mouseMoved' ? 0 : 1, pointerType: 'mouse',
});

// ---------------------------------------------------------------------------
// g1 — TWINS-EQUAL (fresh load; leaves twins CLEAN for g3 on the same load)
// ---------------------------------------------------------------------------
async function g1(cdp, baseUrl) {
  await navTo(cdp, baseUrl + '/index.html?gate=1');
  await cdp.until('first twin digest rendered', `document.getElementById('hashA').textContent.replace(/\\s+/g,'').length === 64`, 60000);
  const t0 = await cdp.ev(TWINS_STATE);
  A(HEX64.test(t0.hashA) && HEX64.test(t0.hashB), `t0 digests are 64-hex (substep ${t0.sub})`);
  A(t0.hashA === t0.hashB, `t0 digests equal: ${t0.hashA.slice(0, 16)}…`);
  await cdp.until('twins reach substep 600', `__GATE__.T.a.substep >= 600`, 180000, 300);
  await cdp.until('digest refresh at substep >= 600', `(${TWINS_STATE}).verdictSub >= 600`, 60000, 300);
  const t1 = await cdp.ev(TWINS_STATE);
  A(t1.sub === t1.subB, `sims in lockstep: substep A ${t1.sub} == B ${t1.subB}`);
  A(HEX64.test(t1.hashA), `digest at substep ${t1.verdictSub} is 64-hex`);
  A(t1.hashA === t1.hashB, `digests equal @ ${t1.verdictSub}: ${t1.hashA.slice(0, 16)}…`);
  A(t1.hashA !== t0.hashA, `digest CHANGED over time (t0 ${t0.hashA.slice(0, 8)}… != t600 ${t1.hashA.slice(0, 8)}…) — sim is not frozen`);
  A(t1.clsA.includes('agree') && !t1.clsA.includes('disagree') &&
    t1.clsB.includes('agree') && !t1.clsB.includes('disagree'), `both hash elements carry .agree`);
  A(!t1.verdictCls.includes('bad') && t1.lastEqual === true, `verdict state: lastEqual === true, class '${t1.verdictCls}'`);
  A(t1.na === t1.nb && t1.na > 0 && t1.na <= CAP, `particle counts: A ${t1.na} == B ${t1.nb}, 0 < n <= ${CAP}`);
  return t1;
}

// ---------------------------------------------------------------------------
// g3 — DIVERGENCE NEGATIVE CONTROL (same load as g1; poisons the twins)
// ---------------------------------------------------------------------------
async function g3(cdp) {
  const pre = await cdp.ev(TWINS_STATE);
  A(pre.lastEqual === true && pre.hashA === pre.hashB, `precondition: twins equal @ substep ${pre.sub}`);
  A(pre.nb + 1 <= CAP, `cap headroom for the control particle: nb ${pre.nb} + 1 <= ${CAP}`);
  const sInj = await cdp.ev(`(() => { const g = __GATE__; const s = g.T.b.substep;
    g.T.b.injectWater({ substep: s, particles: [{ x: g.wu(64), y: g.wu(30) }] }); return s; })()`);
  say(`  --  injected 1 extra particle into sim B ONLY at substep ${sInj}`);
  // (a) the STATE claim: digests differ within 60 substeps of the perturbation.
  // Both digests are submitted in the same task, so they sample one substep.
  await cdp.until('sims advance past the injection substep', `__GATE__.T.a.substep >= ${sInj} + 2`, 15000, 30);
  const probe = await cdp.ev(`(() => { const g = __GATE__; const s = g.T.a.substep;
    return Promise.all([g.T.a.digest(), g.T.b.digest()]).then(([da, db]) => ({ s, da, db })); })()`, { awaitP: true });
  A(probe.s - sInj <= 60, `probe taken within 60 substeps: @ ${probe.s}, injected @ ${sInj} (Δ ${probe.s - sInj})`);
  A(probe.da !== probe.db, `digests DIFFER ${probe.s - sInj} substeps after perturbation: ${probe.da.slice(0, 8)}… != ${probe.db.slice(0, 8)}…`);
  // (b) the DISPLAY claim: the page's own readout flips — it is a live
  // computation over engine state, not cosmetics.
  let det = null;
  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    const s = await cdp.ev(TWINS_STATE);
    if (s.clsA.includes('disagree')) { det = s; break; }
    if (s.sub > sInj + 240) throw new GateFail(`display never flipped by substep ${s.sub} (injection @ ${sInj}, refresh cadence is 36 substeps)`);
    await sleep(60);
  }
  if (!det) throw new GateFail('timed out polling for the divergence display');
  A(det.hashA !== det.hashB, `DOM shows the divergent digests: ${det.hashA.slice(0, 8)}… != ${det.hashB.slice(0, 8)}…`);
  A(det.sub - sInj <= 240, `display flipped by substep ${det.sub} (injected @ ${sInj}, Δ ${det.sub - sInj})`);
  A(det.clsA.includes('disagree') && det.clsB.includes('disagree'), `both hash elements flipped to .disagree`);
  A(det.verdictCls.includes('bad') && det.lastEqual === false, `verdict state: lastEqual === false, class '${det.verdictCls}' — the display is a live computation`);
  A(det.nb === det.na + 1, `count mechanism: nb ${det.nb} == na ${det.na} + 1`);
}

// ---------------------------------------------------------------------------
// g2 — TWINS-UNDER-INPUT (fresh load: cap must have headroom to be driven)
// ---------------------------------------------------------------------------
async function g2(cdp, baseUrl) {
  await navTo(cdp, baseUrl + '/index.html?gate=1');
  await cdp.until('opening pour complete (substep >= 410)', `__GATE__.T.a.substep >= 410`, 180000, 300);
  await cdp.until('digest refresh post-pour', `(${TWINS_STATE}).verdictSub >= 410`, 60000, 300);
  const pre = await cdp.ev(TWINS_STATE);
  A(pre.hashA === pre.hashB && pre.na === pre.nb, `precondition: equal @ ${pre.verdictSub}, n ${pre.na}`);
  const btn = await centerOf(cdp, 'dropTwins');
  await mouse(cdp, 'mouseMoved', btn.x, btn.y, 0);
  await mouse(cdp, 'mousePressed', btn.x, btn.y, 1);
  await cdp.until(`#dropTwins reacts to the real pointerdown (.held)`, `document.getElementById('dropTwins').classList.contains('held')`, 5000, 50);
  say('  --  holding Drop More via CDP Input (real button, real listeners)');
  let maxSeen = pre.na, capReachedAt = -1;
  const holdDeadline = Date.now() + 45000;
  while (Date.now() < holdDeadline) {
    const s = await cdp.ev(`(() => ({ na: __GATE__.T.a.n, nb: __GATE__.T.b.n, sub: __GATE__.T.a.substep }))()`);
    if (s.na !== s.nb) throw new GateFail(`counts split under held input: na ${s.na} != nb ${s.nb}`);
    if (s.na > CAP) throw new GateFail(`cap breached: n ${s.na} > ${CAP}`);
    maxSeen = Math.max(maxSeen, s.na);
    if (s.na + 9 > CAP) { capReachedAt = s.sub; break; } // next 3x3 droplet must be refused
    await sleep(150);
  }
  if (capReachedAt < 0) throw new GateFail(`hold never drove n to the cap (max seen ${maxSeen})`);
  // keep holding through ~1.2s of refused droplets, sampling the cap the whole time
  const refuseUntil = Date.now() + 1200;
  let refusedFrom = maxSeen;
  while (Date.now() < refuseUntil) {
    const s = await cdp.ev(`(() => ({ na: __GATE__.T.a.n, nb: __GATE__.T.b.n }))()`);
    if (s.na > CAP || s.nb > CAP) throw new GateFail(`cap breached during refusal window: ${s.na}/${s.nb}`);
    if (s.na !== s.nb) throw new GateFail(`refusal not identical across twins: ${s.na} != ${s.nb}`);
    refusedFrom = s.na;
    await sleep(100);
  }
  await mouse(cdp, 'mouseReleased', btn.x, btn.y, 0);
  await cdp.until('#dropTwins released', `!document.getElementById('dropTwins').classList.contains('held')`, 5000, 50);
  A(refusedFrom + 9 > CAP && refusedFrom <= CAP, `cap held under ~1.2s of refused droplets: n pinned at ${refusedFrom} (cap ${CAP}, droplet 9)`);
  const settleTo = capReachedAt + 240;
  await cdp.until(`settle to substep ${settleTo}`, `__GATE__.T.a.substep >= ${settleTo}`, 120000, 300);
  await cdp.until('digest refresh after settle', `(${TWINS_STATE}).verdictSub >= ${settleTo}`, 60000, 300);
  const post = await cdp.ev(TWINS_STATE);
  A(post.na > pre.na, `particle count GREW under input: ${pre.na} -> ${post.na}`);
  A(post.na <= CAP && post.nb <= CAP && post.na === post.nb, `counts equal and capped: ${post.na}/${post.nb} <= ${CAP} (max ever seen ${maxSeen})`);
  A(HEX64.test(post.hashA) && post.hashA === post.hashB, `digests equal after settle @ ${post.verdictSub}: ${post.hashA.slice(0, 16)}…`);
  A(post.hashA !== pre.hashA, `digest changed vs pre-input (${pre.hashA.slice(0, 8)}… -> ${post.hashA.slice(0, 8)}…)`);
  A(post.clsA.includes('agree') && !post.verdictCls.includes('bad') && post.lastEqual === true, `DOM state: .agree + lastEqual === true`);

  // g2b — HOVER STIR: real mouse MOTION over sim A's canvas, no button ever
  // pressed, must move particles in BOTH sims identically and keep the
  // digests equal. pushHits counts particles actually displaced, so a stir
  // that silently no-ops cannot pass.
  const cvA = await centerOf(cdp, 'cvA');
  const hitsPre = await cdp.ev(`(() => ({ a: __GATE__.T.a.pushHits, b: __GATE__.T.b.pushHits }))()`);
  say('  --  stirring sim A by cursor motion alone (16 real mouseMoved, buttons: 0)');
  // Stir where the water IS: post-pour it fills the bottom of the world, so
  // sweep the cursor through the lower third of the canvas, not mid-air.
  const yStir = Math.round(cvA.y + cvA.h * 0.30);
  for (let i = 0; i < 16; i++) {
    await mouse(cdp, 'mouseMoved', Math.round(cvA.x - 40 + i * 5), yStir + ((i & 1) ? 5 : -5), 0);
    await sleep(45);
  }
  await cdp.until('stir displaced particles', `__GATE__.T.a.pushHits > ${hitsPre.a}`, 15000, 100);
  await mouse(cdp, 'mouseMoved', 5, 5, 0);   // off-canvas: the settle is un-stirred
  const hits = await cdp.ev(`(() => ({ a: __GATE__.T.a.pushHits, b: __GATE__.T.b.pushHits }))()`);
  A(hits.a > hitsPre.a, `hover stir moved particles with no click: pushHits ${hitsPre.a} -> ${hits.a}`);
  A(hits.a === hits.b, `stir applied IDENTICALLY to both sims: hits A ${hits.a} == B ${hits.b}`);
  const subStir = await cdp.ev(`__GATE__.T.a.substep`);
  const stirSettle = subStir + 180;
  await cdp.until(`post-stir settle to ${stirSettle}`, `__GATE__.T.a.substep >= ${stirSettle}`, 90000, 300);
  await cdp.until('digest refresh post-stir', `(${TWINS_STATE}).verdictSub >= ${stirSettle}`, 60000, 300);
  const post2 = await cdp.ev(TWINS_STATE);
  A(HEX64.test(post2.hashA) && post2.hashA === post2.hashB, `digests equal after the stir @ ${post2.verdictSub}: ${post2.hashA.slice(0, 16)}…`);
  A(post2.hashA !== post.hashA, `the stir left a real mark: digest changed vs pre-stir settle`);
}

// ---------------------------------------------------------------------------
// g4 — REPLAY-BYTE-IDENTITY + deterministic scrub (fresh load)
// ---------------------------------------------------------------------------
async function g4(cdp, baseUrl) {
  await navTo(cdp, baseUrl + '/index.html?gate=1');
  await cdp.until('pool warmed (substep >= 60)', `__GATE__.P.demo.substep >= 60`, 90000, 200);
  const cv = await centerOf(cdp, 'cvP');
  const p1 = { x: Math.round(cv.x - cv.w * 0.2), y: Math.round(cv.y - cv.h * 0.1) };
  const p2 = { x: Math.round(cv.x + cv.w * 0.15), y: p1.y };
  await mouse(cdp, 'mouseMoved', p1.x, p1.y, 0);
  await mouse(cdp, 'mousePressed', p1.x, p1.y, 1);
  for (let i = 1; i <= 3; i++) {
    await mouse(cdp, 'mouseMoved', Math.round(p1.x + (p2.x - p1.x) * i / 3), p1.y, 1);
    await sleep(60);
  }
  await mouse(cdp, 'mouseReleased', p2.x, p2.y, 0);
  await cdp.until(`first interaction armed recording (mode 'recording')`, `__GATE__.P.mode === 'recording'`, 15000, 50);
  const startSub = await cdp.ev(`__GATE__.P.demo._rec.startSubstep`);
  say(`  --  recording armed by canvas pointer; journal starts at substep ${startSub}`);
  const btn = await centerOf(cdp, 'dropPool');
  await mouse(cdp, 'mouseMoved', btn.x, btn.y, 0);
  await mouse(cdp, 'mousePressed', btn.x, btn.y, 1);
  await sleep(800);
  await mouse(cdp, 'mouseReleased', btn.x, btn.y, 0);
  await mouse(cdp, 'mouseMoved', p2.x, p2.y + 20, 0);
  await mouse(cdp, 'mousePressed', p2.x, p2.y + 20, 1);
  await mouse(cdp, 'mouseReleased', p2.x, p2.y + 20, 0);
  await cdp.until('record >= 260 substeps of interaction', `__GATE__.P.demo.substep >= ${startSub} + 260`, 90000, 200);
  const rec = await cdp.ev(`(() => ({ mode: __GATE__.P.mode, events: __GATE__.P.demo._rec.events.length, n: __GATE__.P.demo.n }))()`);
  A(rec.mode === 'recording' && rec.events >= 3, `journal live: mode 'recording', ${rec.events} injection events, n ${rec.n}`);
  // "the moment their mouse leaves the window" — the page listens for a
  // document mouseout with no relatedTarget; dispatch exactly that.
  await cdp.ev(`document.dispatchEvent(new MouseEvent('mouseout', { relatedTarget: null }))`);
  await cdp.until(`auto-replay ran to completion (mode 'scrub')`, `__GATE__.P.mode === 'scrub'`, 150000, 200);
  const rep = await cdp.ev(`(() => { const P = __GATE__.P, $ = (i) => document.getElementById(i);
    return { dur: P.doc.durationSubsteps, endSub: P.demo.substep,
      recorded: P.doc.final.digest,
      replayedDom: $('hashP').textContent.replace(/\\s+/g, ''),
      verdictCls: $('poolVerdict').className,
      kf: P.keyframes.map((k) => k.substep),
      events: P.doc.events.length,
      scrubOn: $('scrubRow').classList.contains('on'),
      scrubMax: Number($('scrub').max), n: P.demo.n };
  })()`);
  A(rep.dur >= 260, `recorded duration ${rep.dur} substeps, ${rep.events} events`);
  A(rep.endSub === rep.dur, `replay ended AT the recorded duration: substep ${rep.endSub} == ${rep.dur}`);
  A(HEX64.test(rep.recorded), `recorded digest is 64-hex: ${rep.recorded.slice(0, 16)}…`);
  A(rep.replayedDom === rep.recorded, `REPLAY == LIVE, byte-identical at substep ${rep.dur}: ${rep.replayedDom.slice(0, 16)}…`);
  const engineNow = await cdp.ev(`__GATE__.P.demo.digest()`, { awaitP: true });
  A(engineNow === rep.recorded, `engine state itself re-digests to the recorded value (not just the DOM)`);
  A(!rep.verdictCls.includes('bad'), `verdict class '${rep.verdictCls}' (no mismatch state)`);
  const expectKf = Math.floor(rep.dur / 24) + 1;
  A(rep.kf.length === expectKf && rep.kf.every((s, i) => s === i * 24),
    `${rep.kf.length} scrub keyframes at exact 24-substep cadence (expected ${expectKf})`);
  A(rep.scrubOn && rep.scrubMax === rep.dur, `scrubber shown, max ${rep.scrubMax} == duration`);
  A(rep.n <= CAP, `pool n ${rep.n} <= ${CAP}`);
  // -- deterministic scrub: 3 positions, backward included, each landed twice --
  const scrub = async (t) => {
    // scrubTo is async now (pushes sync the mirror), busy-guarded latest-wins:
    // dispatch the input, then WAIT for the landing instead of reading the
    // in-flight state — the same contract a human finger gets.
    await cdp.ev(`(() => { const $ = (i) => document.getElementById(i);
      const sc = $('scrub'); sc.value = String(${t});
      sc.dispatchEvent(new Event('input', { bubbles: true })); })()`);
    await cdp.until(`scrub(${t}) lands`,
      `__GATE__.P.mode === 'scrub' && !__GATE__.P.scrubBusy && __GATE__.P.scrubNext == null && __GATE__.P.demo.substep === ${t}`,
      30000, 100);
    const r = await cdp.ev(`(() => { const $ = (i) => document.getElementById(i);
      return { sub: __GATE__.P.demo.substep, mode: __GATE__.P.mode,
               pos: Number($('scrubPos').textContent.replace(/\\D/g, '')) }; })()`);
    if (r.mode !== 'scrub' || r.sub !== t || r.pos !== t) {
      throw new GateFail(`scrub(${t}) landed wrong: ${JSON.stringify(r)}`);
    }
    return cdp.ev(`__GATE__.P.demo.digest()`, { awaitP: true });
  };
  const tB = rep.dur - 5, tA = Math.min(37, rep.dur - 9), tC = 10;
  const hB1 = await scrub(tB);            // backward from dur
  const hA1 = await scrub(tA);            // backward
  const hC1 = await scrub(tC);            // BACKWARD again, below first keyframe gap
  const hB2 = await scrub(tB);            // forward re-visit
  const hA2 = await scrub(tA);            // backward re-visit
  const hC2 = await scrub(tC);            // backward re-visit
  A([hA1, hB1, hC1].every((h) => HEX64.test(h)), `scrub digests are 64-hex`);
  A(hB1 === hB2, `scrub(${tB}) twice (fwd+bwd approach) -> same state ${hB1.slice(0, 16)}…`);
  A(hA1 === hA2, `scrub(${tA}) twice -> same state ${hA1.slice(0, 16)}…`);
  A(hC1 === hC2, `scrub(${tC}) twice -> same state ${hC1.slice(0, 16)}…`);
  A(hA1 !== hB1 && hB1 !== hC1 && hA1 !== hC1, `the three scrub targets are three DISTINCT states`);
}

// ---------------------------------------------------------------------------
// g5 — extraction gate re-run (cross-repo, both sides via their own cdp.mjs)
// ---------------------------------------------------------------------------
function runCdpRunner(runner, url, timeoutMs) {
  return new Promise((resolveP, rejectP) => {
    const p = spawn('node', [runner, url, String(timeoutMs)], { stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '', err = '';
    p.stdout.on('data', (d) => { out += d; });
    p.stderr.on('data', (d) => { err += d; });
    p.on('close', (code) => {
      try { resolveP({ code, json: JSON.parse(out) }); }
      catch { rejectP(new GateFail(`unparseable runner output (exit ${code}): ${out.slice(0, 400)} ${err.slice(0, 400)}`)); }
    });
  });
}
async function g5aether() {
  const { server, port } = await serveDir(AETHER);
  try {
    say(`  --  aether served on :${port}; running aether's own harness (its own headless Chrome, serial)`);
    const r = await runCdpRunner(join(AETHER, 'test/harness/cdp.mjs'),
      `http://127.0.0.1:${port}/test/determinism.html?substeps=1000&runs=1`, 570000);
    if (!r.json.ok) throw new GateFail('aether runner not ok: ' + JSON.stringify(r.json).slice(0, 400));
    writeFileSync(join(SCRATCH, 'g5_aether.json'), JSON.stringify(r.json));
    say(`  ok  aether dambreak_v1 x 1000: chain ${r.json.result.chain}`);
  } finally { server.close(); }
}
async function g5suna() {
  const { server, port } = await serveDir(ROOT);
  try {
    say(`  --  sunaengine served on :${port}; running the extracted harness (one headless Chrome, serial)`);
    const r = await runCdpRunner(join(ROOT, 'test/harness/cdp.mjs'),
      `http://127.0.0.1:${port}/test/extraction_gate.html?substeps=1000`, 570000);
    if (!r.json.ok) throw new GateFail('suna runner not ok: ' + JSON.stringify(r.json).slice(0, 400));
    writeFileSync(join(SCRATCH, 'g5_suna.json'), JSON.stringify(r.json));
    say(`  ok  suna dambreak_v1 x 1000: chain ${r.json.result.crossRepo.chain}`);
  } finally { server.close(); }
}
function g5compare() {
  const a = JSON.parse(readFileSync(join(SCRATCH, 'g5_aether.json'), 'utf8')).result;
  const s = JSON.parse(readFileSync(join(SCRATCH, 'g5_suna.json'), 'utf8')).result;
  A(s.pass === true, `suna in-page checks pass (twins/replay/keyframe/dbg): pass === true`);
  A(s.dbgAllZero === true, `suna dbg counters all zero`);
  A(a.chain === s.crossRepo.chain, `chains byte-identical across repos: ${a.chain.slice(0, 16)}…`);
  A(a.chain === FROZEN.chain, `chain matches the FROZEN expectation: ${FROZEN.chain}`);
  for (const k of Object.keys(FROZEN.checkpoints)) {
    A(a.checkpoints[k] === s.crossRepo.checkpoints[k] && a.checkpoints[k] === FROZEN.checkpoints[k],
      `checkpoint ${k}: both repos == frozen ${FROZEN.checkpoints[k].slice(0, 16)}…`);
  }
  A(s.crossRepo.n === 4096 && s.crossRepo.nFluid === 3600, `frozen scene shape: n 4096, fluid 3600`);
}
async function g5() { await g5aether(); await g5suna(); g5compare(); }

// ---------------------------------------------------------------------------
// g6 — hygiene (browser half is fed by the g1–g4 session; static half is pure fs)
// ---------------------------------------------------------------------------
function sha256(file) { return createHash('sha256').update(readFileSync(file)).digest('hex'); }
function* walk(dir) {
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) yield* walk(p); else if (e.isFile()) yield p;
  }
}
function g6static() {
  const tracked = execFileSync('git', ['-C', ROOT, 'ls-files'], { encoding: 'utf8' }).trim().split('\n');
  A(tracked.length >= 15, `${tracked.length} tracked files`);
  A(tracked.every((f) => !/^private\//.test(f) && !/^research\//.test(f)), `no tracked path under private/ or research/`);
  A(!tracked.includes('.DS_Store') && tracked.every((f) => !f.endsWith('/.DS_Store')), `.DS_Store not tracked`);
  const shipped = new Map(tracked.map((f) => [sha256(join(ROOT, f)), f]));
  const forbidden = [];
  for (const dir of [join(AETHER, 'private'), join(AETHER, 'research/oecake')]) {
    if (existsSync(dir)) for (const f of walk(dir)) forbidden.push(f);
  }
  if (existsSync(join(AETHER, 'index.html'))) forbidden.push(join(AETHER, 'index.html'));
  let scanned = 0;
  for (const f of forbidden) {
    const h = sha256(f);
    scanned++;
    if (shipped.has(h)) throw new GateFail(`quarantined bytes shipped: ${f} === ${shipped.get(h)}`);
  }
  A(scanned >= 100, `no quarantine file in the tree: ${scanned} forbidden files (private/, research/oecake, aether index.html) vs ${shipped.size} shipped hashes — zero intersection`);
  const lic = readFileSync(join(ROOT, 'LICENSE'), 'utf8');
  A(lic.includes('GNU AFFERO GENERAL PUBLIC LICENSE') && lic.includes('Version 3'), `LICENSE is GNU AGPL Version 3`);
  const page = readFileSync(join(ROOT, 'index.html'), 'utf8');
  A(page.includes('A love letter to OE-CAKE! by Prometech Software'), `OE-CAKE! / Prometech credit line present in index.html`);
  A(!/https?:\/\/(?!127\.0\.0\.1|localhost)[a-z0-9.-]+\/[^"'\s]*\.(js|css|woff2?|mjs)/i.test(page), `no external script/style/font URL in index.html`);
}
function g6browser(cdp, port) {
  const errs = cdp.consoleEvents.filter((c) => c.type === 'error');
  const warns = cdp.consoleEvents.filter((c) => c.type === 'warning' && /\[sunaengine\]/.test(c.text));
  A(errs.length === 0, `0 console.error across all ${cdp.consoleEvents.length} console events (3 page loads)`);
  A(cdp.exceptions.length === 0, `0 uncaught exceptions`);
  A(warns.length === 0, `0 [sunaengine] boot-fallback warnings`);
  const external = cdp.requests.filter((u) => !u.startsWith(`http://127.0.0.1:${port}/`) && !u.startsWith('data:'));
  A(external.length === 0, `all ${cdp.requests.length} network requests hit 127.0.0.1:${port} (DNS was hard-blocked: MAP * ~NOTFOUND)`);
  A(cdp.requests.length >= 10, `request log actually saw the module graph (${cdp.requests.length} requests)`);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
const argv = process.argv.slice(2);
const selected = argv.length ? argv : ['g1', 'g3', 'g2', 'g4', 'g6', 'g5'];

async function runGate(name, fn) {
  say(`\n== ${name} ==`);
  try {
    await fn();
    results.set(name, 'GREEN');
    say(`${name} GREEN`);
  } catch (e) {
    if (!(e instanceof GateFail)) say('  !! non-gate error: ' + (e.stack || e.message));
    results.set(name, 'RED: ' + e.message);
    say(`${name} RED — ${e.message}`);
  }
}

const uiGates = selected.filter((g) => ['g1', 'g2', 'g3', 'g4', 'g6'].includes(g));
const wantsWindowErrors = [];

if (uiGates.length) {
  const { server, port } = await serveDir(ROOT);
  const base = `http://127.0.0.1:${port}`;
  const cdp = await launchChrome();
  await cdp.send('Runtime.enable');
  await cdp.send('Page.enable');
  await cdp.send('Network.enable');
  await cdp.send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 3200, deviceScaleFactor: 1, mobile: false });
  const grabWindowErrors = async () => wantsWindowErrors.push(...(await cdp.ev('window.__ERRORS__ || []')));

  // load 1: g1 then g3 (g3 deliberately poisons the twins — it goes last on this load)
  if (selected.includes('g1') || selected.includes('g3')) {
    if (selected.includes('g1')) await runGate('g1 TWINS-EQUAL', () => g1(cdp, base));
    else await navTo(cdp, base + '/index.html?gate=1');
    if (selected.includes('g3')) await runGate('g3 DIVERGENCE-NEGATIVE-CONTROL', () => g3(cdp));
    await grabWindowErrors();
  }
  // load 2: g2 (needs cap headroom and clean twins)
  if (selected.includes('g2')) { await runGate('g2 TWINS-UNDER-INPUT', () => g2(cdp, base)); await grabWindowErrors(); }
  // load 3: g4
  if (selected.includes('g4')) { await runGate('g4 REPLAY-BYTE-IDENTITY', () => g4(cdp, base)); await grabWindowErrors(); }
  if (selected.includes('g6')) {
    await runGate('g6 HYGIENE', async () => {
      A(wantsWindowErrors.length === 0, `window.__ERRORS__ empty on every load (${wantsWindowErrors.length ? wantsWindowErrors.join('; ') : '0 entries'})`);
      g6browser(cdp, port);
      g6static();
    });
  }
  cdp.close();
  killChrome();          // g5's runners each launch their own Chrome — ours must be dead first
  server.close();
}

for (const g of selected) {
  if (g === 'g5') await runGate('g5 EXTRACTION-CROSS-REPO', g5);
  else if (g === 'g5.aether') await runGate('g5.aether', g5aether);
  else if (g === 'g5.suna') await runGate('g5.suna', g5suna);
  else if (g === 'g5.compare') await runGate('g5.compare', async () => g5compare());
}

say('\n== SUMMARY ==');
let red = 0;
for (const [g, r] of results) { say(`${g}: ${r}`); if (r !== 'GREEN') red++; }
say(red === 0 ? `SITE GATES: ${results.size}/${results.size} GREEN` : `SITE GATES: ${red} RED`);
process.exit(red === 0 ? 0 : 1);
