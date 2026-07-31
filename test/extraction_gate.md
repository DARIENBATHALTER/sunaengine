# The extraction gate — cross-repo determinism proof

Copyright (C) 2026 Darien Bathalter — AGPL-3.0-only, see LICENSE.

## The claim

The engine in this repo (`src/engine.js` + `src/sim.wgsl` + `src/fixed.wgsl` +
`src/hash/` + `src/tables.json`) is the REAL aether/SunaBox engine, extracted
**unmodified** from aether commit `e41e06f` — not a reimplementation. Every
extracted file is byte-identical to its aether source below a 5-line license
header: `diff <(tail -n +6 src/engine.js) <aether>/src/engine.js` is empty
(for `test/harness/cdp.mjs`, which keeps its shebang on line 1:
`diff <(tail -n +7 test/harness/cdp.mjs) <(tail -n +2 <aether>/test/harness/cdp.mjs)`).
`src/tables.json` carries no header and is byte-identical outright.

The proof that the *behavior* came along with the *bytes*: the frozen
determinism-gate scene (`dambreak_v1`, 4096 particles, 3600 fluid), stepped
1000 substeps, produces a per-substep hash chain and checkpoint digests that
are **byte-identical** in both repos, each measured by its own harness in
headless Chrome (WebGPU, Metal).

## The commands (re-runnable)

Serve both repos (any static server; ports are arbitrary):

```sh
python3 -m http.server 8951 --bind 127.0.0.1 --directory /Users/darien/aether &
python3 -m http.server 8952 --bind 127.0.0.1 --directory /Users/darien/sunaengine &
```

Run the two sides SERIALLY (one headless Chrome at a time):

```sh
# aether's own harness, aether's own gate page:
node /Users/darien/aether/test/harness/cdp.mjs \
  "http://127.0.0.1:8951/test/determinism.html?substeps=1000&runs=1" 570000 \
  > /tmp/aether_gate.json

# this repo's gate page, through the same (extracted) runner:
node /Users/darien/sunaengine/test/harness/cdp.mjs \
  "http://127.0.0.1:8952/test/extraction_gate.html?substeps=1000" 570000 \
  > /tmp/suna_gate.json

# the comparison — byte equality or the extraction is wrong:
python3 - <<'EOF'
import json
a = json.load(open('/tmp/aether_gate.json'))['result']
s = json.load(open('/tmp/suna_gate.json'))['result']
assert s['pass'] is True, 'in-page checks failed'
assert a['chain'] == s['crossRepo']['chain'], (a['chain'], s['crossRepo']['chain'])
assert a['checkpoints'] == s['crossRepo']['checkpoints']
print('EXTRACTION GATE GREEN')
print('chain', a['chain'])
EOF
```

## The recorded result (2026-07-31, aether @ e41e06f, Apple Silicon / Metal)

Both repos, byte-identical:

```
chain            20918f97fbcbb409e7d058ebcae6f6725973b9a19042168742b07e8c17126d14
checkpoint 1     c31aa57fb218829e3b78250e5909ddbbbb872c7d60e5b674f78d1a7164d8e0be
checkpoint 10    95984e71128a222a27f285a340d178384a588ca74d27abb2448e08d3c4239c04
checkpoint 100   320e900be986aa4e63ee1d110a4fb798d7e07e39ca7828782c475c1f206204eb
checkpoint 1000  b1844e32b4d2bd4b3dbd91cd4ffb7ead0bf568d6fea0dc53c8d555134f946d9b
dbg              all 16 words zero
```

If this gate ever reads red, the extraction is wrong. The expectation is never
adjusted; the code is.

## What else the gate page proves (the demos' machinery)

`test/extraction_gate.html` also runs, in-page:

* **twins** — two sims, one injection schedule → one chain, bit-exact
  (Demo 1's mechanism);
* **record → replay** — a `.sunar` recorded through `src/demo_boot.js` replays
  to a byte-identical final digest, and a ONE-TICK perturbation of one
  scheduled particle does not (the negative control that proves the gate can
  fail);
* **scrub keyframe** — a mid-run `snapshot()` (which carries `buf.derived`,
  the engine's cross-substep scratch) rejoins its own timeline bit-exactly
  after `restore()`. Measured while building this: a state-only restore
  diverges on the very first substep — pos+vel on every fluid particle —
  because `derived` is warm cross-substep input; that is why `snapshot()`
  captures it and why `startRecording()` normalizes the live engine onto the
  load boundary its replay will reconstruct.
