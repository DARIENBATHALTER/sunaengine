#!/usr/bin/env node
// SunaEngine — real SunaBox (aether) engine code, extracted UNMODIFIED below this header
// from aether commit e41e06f. Verify: diff <(tail -n +7 test/harness/cdp.mjs) <(tail -n +2 <aether>/test/harness/cdp.mjs)
// Copyright (C) 2026 Darien Bathalter
// SPDX-License-Identifier: AGPL-3.0-only — see LICENSE. NO WARRANTY.

// Dependency-free headless-Chrome runner via CDP (Node 22+ global WebSocket/fetch).
// Usage: node cdp.mjs <url> [timeoutMs]
// Evaluates window.__RESULT__ (awaiting it if it's a promise) and prints JSON.

import { spawn } from 'node:child_process';
import { rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const url = process.argv[2];
const timeoutMs = Number(process.argv[3] || 120000);
const port = 9333 + (process.pid % 200);
const PROFILE = join(tmpdir(), 'aether-cdp', String(process.pid));
const cleanup = () => { try { rmSync(PROFILE, { recursive: true, force: true }); } catch {} };
process.on('exit', cleanup);
for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP']) process.on(sig, () => { cleanup(); process.exit(1); });

// Pre-flight. Without this, a dead static server or a mistyped path makes the
// page never define window.__RESULT__, and the runner burns the ENTIRE timeout
// (up to 15 minutes) before saying anything useful. Ask node first: it costs
// one request and turns a silent hang into a one-line error.
try {
  const probe = await fetch(url, { redirect: 'follow' });
  if (!probe.ok) {
    console.log(JSON.stringify({ ok: false, error: `page returned HTTP ${probe.status} — ${url}` }));
    process.exit(1);
  }
} catch (e) {
  console.log(JSON.stringify({
    ok: false,
    error: `cannot reach ${url} (${e.message}). Is the static server running?\n` +
           `  python3 -m http.server <port> --bind 127.0.0.1 --directory /Users/darien/aether &`,
  }));
  process.exit(1);
}

const chrome = spawn(CHROME, [
  '--headless=new',
  `--remote-debugging-port=${port}`,
  '--enable-unsafe-webgpu',
  '--enable-features=WebGPU',
  '--use-angle=metal',
  '--no-sandbox',
  '--no-first-run',
  '--disable-dev-shm-usage',
  `--user-data-dir=${PROFILE}`,
  'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe'] });

let stderr = '';
chrome.stderr.on('data', (d) => { stderr += d.toString(); });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function targetUrl() {
  for (let i = 0; i < 100; i++) {
    try {
      const r = await fetch(`http://127.0.0.1:${port}/json/list`);
      const list = await r.json();
      const page = list.find((t) => t.type === 'page');
      if (page?.webSocketDebuggerUrl) return page.webSocketDebuggerUrl;
    } catch { /* not up yet */ }
    await sleep(100);
  }
  throw new Error('Chrome CDP never came up. stderr:\n' + stderr.slice(0, 2000));
}

function connect(wsUrl) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    const pending = new Map();
    const logs = [];
    let id = 0;
    ws.addEventListener('open', () => resolve({ ws, send, logs }));
    ws.addEventListener('error', (e) => reject(new Error('ws error ' + e.message)));
    ws.addEventListener('message', (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id !== undefined && pending.has(msg.id)) {
        const { resolve: res, reject: rej } = pending.get(msg.id);
        pending.delete(msg.id);
        msg.error ? rej(new Error(JSON.stringify(msg.error))) : res(msg.result);
      } else if (msg.method === 'Runtime.consoleAPICalled') {
        logs.push(msg.params.args.map((a) => a.value ?? a.description ?? '').join(' '));
      } else if (msg.method === 'Runtime.exceptionThrown') {
        logs.push('EXCEPTION: ' + (msg.params.exceptionDetails?.exception?.description || ''));
      }
    });
    function send(method, params = {}) {
      const myId = ++id;
      return new Promise((res, rej) => {
        pending.set(myId, { resolve: res, reject: rej });
        ws.send(JSON.stringify({ id: myId, method, params }));
      });
    }
  });
}

// EMIT AND EXIT, IN THAT ORDER, AND THE ORDER IS NOT FREE.
// `process.exit()` does NOT wait for a pending stdout write, and when stdout is
// a PIPE (which is exactly how test/run.mjs invokes this file) node's write is
// asynchronous above the 64 KiB pipe buffer. So a gate whose JSON crossed 64 KiB
// had its output silently CUT IN HALF and the suite reported it as
// "unparseable runner output" — a REAL gate failing for a reason that has
// nothing to do with the gate. Found 2026-07-27: `test/elements.html` grew past
// 64 KiB (67 892 bytes) when §18.23's E19 was added, and it reported red in the
// suite while passing 143/143 when the same command was redirected to a FILE,
// because a file write is synchronous. Any gate could have hit this at any time.
// `write`'s callback fires after the data is flushed, so waiting on it is the
// fix; the timeout is belt-and-braces so a wedged pipe cannot hang the suite.
const emitAndExit = (obj, code) => {
  const text = JSON.stringify(obj, null, 2) + '\n';
  let done = false;
  const bail = () => { if (!done) { done = true; try { chrome.kill(); } catch { /* gone */ } process.exit(code); } };
  const t = setTimeout(bail, 10000);
  process.stdout.write(text, () => { clearTimeout(t); bail(); });
};

const fail = (msg, extra = {}) => emitAndExit({ ok: false, error: msg, ...extra }, 1);

try {
  const wsUrl = await targetUrl();
  const { send, logs } = await connect(wsUrl);
  await send('Runtime.enable');
  await send('Page.enable');
  await send('Page.navigate', { url });

  // Poll for window.__RESULT__ to exist, then await it.
  const deadline = Date.now() + timeoutMs;
  let result = null;
  while (Date.now() < deadline) {
    const r = await send('Runtime.evaluate', {
      expression: 'typeof window.__RESULT__ !== "undefined"',
      returnByValue: true,
    });
    if (r.result?.value === true) {
      const out = await send('Runtime.evaluate', {
        expression: 'Promise.resolve(window.__RESULT__)',
        awaitPromise: true,
        returnByValue: true,
      });
      if (out.exceptionDetails) {
        const detail = out.exceptionDetails;
        const description = detail.exception?.description || detail.text || 'window.__RESULT__ rejected';
        fail(description, { logs, exceptionDetails: detail });
      }
      result = out.result?.value;
      if (result === undefined) {
        fail('window.__RESULT__ resolved without a serializable value', { logs });
      }
      break;
    }
    await sleep(200);
  }
  if (result === null) fail('timeout waiting for window.__RESULT__', { logs });
  emitAndExit({ ok: true, result, logs }, 0);
} catch (e) {
  fail(e.message, { stderr: stderr.slice(0, 2000) });
}
