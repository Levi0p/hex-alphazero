// The trained AlphaZero agent in plain JavaScript: Hex rules, the network's
// forward pass, and MCTS. Mirrors hexaz/hex.py, hexaz/net.py and hexaz/mcts.py.
// Works in the browser (window.AZ) and in Node (require) for the parity test.

(function (root) {
  "use strict";

  // ---------- Hex rules (canonical form, same as hexaz/hex.py) ----------
  // board[i] is +1 for the player to move, -1 for the opponent, and the player
  // to move always connects top to bottom. After a move: negate and transpose.

  const NB = [[-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0]];

  function connectsTopBottom(board, n) {
    const seen = new Uint8Array(n * n);
    const stack = [];
    for (let c = 0; c < n; c++) if (board[c] === 1) { stack.push(c); seen[c] = 1; }
    while (stack.length) {
      const i = stack.pop();
      const r = (i / n) | 0, c = i % n;
      if (r === n - 1) return true;
      for (const [dr, dc] of NB) {
        const rr = r + dr, cc = c + dc;
        if (rr < 0 || rr >= n || cc < 0 || cc >= n) continue;
        const j = rr * n + cc;
        if (!seen[j] && board[j] === 1) { seen[j] = 1; stack.push(j); }
      }
    }
    return false;
  }

  // Returns {board: next canonical board, won: did the mover just win}.
  function step(board, n, a) {
    const b = Int8Array.from(board);
    b[a] = 1;
    const won = connectsTopBottom(b, n);
    const next = new Int8Array(n * n);
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) next[c * n + r] = -b[r * n + c];
    return { board: next, won };
  }

  // ---------- Network forward pass (same layers as hexaz/net.py) ----------

  function f32(s) {
    const bin = typeof atob === "function" ? atob(s) : Buffer.from(s, "base64").toString("binary");
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }

  function loadModel(json) {
    const conv = (c) => ({ k: f32(c.k), b: f32(c.b), kh: c.shape[0], cin: c.shape[2], cout: c.shape[3] });
    const ln = (l) => ({ scale: f32(l.scale), bias: f32(l.bias) });
    return {
      n: json.size, ch: json.channels,
      stem: conv(json.stem),
      res: json.res.map((r) => ({ ln0: ln(r.ln0), conv0: conv(r.conv0), ln1: ln(r.ln1), conv1: conv(r.conv1) })),
      finalLn: ln(json.final_ln),
      policy: conv(json.policy),
      v1: { w: f32(json.v1.w), b: f32(json.v1.b) },
      v2: { w: f32(json.v2.w), b: f32(json.v2.b) },
    };
  }

  // x: Float32Array [n*n*cin] (row-major H, W, C). "SAME" zero padding, kernel HWIO.
  function conv2d(x, n, c) {
    const { k, b, kh, cin, cout } = c;
    const pad = (kh - 1) >> 1;
    const out = new Float32Array(n * n * cout);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const o = (i * n + j) * cout;
      for (let co = 0; co < cout; co++) out[o + co] = b[co];
      for (let di = 0; di < kh; di++) {
        const ii = i + di - pad;
        if (ii < 0 || ii >= n) continue;
        for (let dj = 0; dj < kh; dj++) {
          const jj = j + dj - pad;
          if (jj < 0 || jj >= n) continue;
          const xo = (ii * n + jj) * cin;
          const ko = (di * kh + dj) * cin * cout;
          for (let ci = 0; ci < cin; ci++) {
            const xv = x[xo + ci];
            if (xv === 0) continue;
            const kr = ko + ci * cout;
            for (let co = 0; co < cout; co++) out[o + co] += xv * k[kr + co];
          }
        }
      }
    }
    return out;
  }

  // LayerNorm over channels at every cell, then ReLU (always used together here).
  function lnRelu(x, n, ch, l) {
    const out = new Float32Array(x.length);
    for (let p = 0; p < n * n; p++) {
      let mean = 0, v = 0;
      for (let c = 0; c < ch; c++) mean += x[p * ch + c];
      mean /= ch;
      for (let c = 0; c < ch; c++) { const d = x[p * ch + c] - mean; v += d * d; }
      const inv = 1 / Math.sqrt(v / ch + 1e-6);
      for (let c = 0; c < ch; c++) {
        const y = (x[p * ch + c] - mean) * inv * l.scale[c] + l.bias[c];
        out[p * ch + c] = y > 0 ? y : 0;
      }
    }
    return out;
  }

  // Returns {logits: Float32Array[n*n], value: number} for a canonical board.
  function evaluate(m, board) {
    const n = m.n, ch = m.ch;
    const x = new Float32Array(n * n * 3);
    for (let i = 0; i < n * n; i++) {
      x[i * 3] = board[i] === 1 ? 1 : 0;
      x[i * 3 + 1] = board[i] === -1 ? 1 : 0;
      x[i * 3 + 2] = 1;
    }
    let h = conv2d(x, n, m.stem);
    for (const r of m.res) {
      const t = conv2d(lnRelu(conv2d(lnRelu(h, n, ch, r.ln0), n, r.conv0), n, ch, r.ln1), n, r.conv1);
      for (let i = 0; i < h.length; i++) h[i] += t[i];
    }
    h = lnRelu(h, n, ch, m.finalLn);
    const logits = conv2d(h, n, m.policy);

    const pooled = new Float32Array(ch);
    for (let p = 0; p < n * n; p++) for (let c = 0; c < ch; c++) pooled[c] += h[p * ch + c] / (n * n);
    let v = m.v2.b[0];
    for (let o = 0; o < ch; o++) {
      let z = m.v1.b[o];
      for (let c = 0; c < ch; c++) z += pooled[c] * m.v1.w[c * ch + o];
      if (z > 0) v += z * m.v2.w[o];
    }
    return { logits, value: Math.tanh(v) };
  }

  // ---------- MCTS (same PUCT rule as hexaz/mcts.py) ----------

  function makeNode(m, board) {
    const { logits, value } = evaluate(m, board);
    const n2 = board.length;
    let max = -Infinity;
    for (let a = 0; a < n2; a++) if (board[a] === 0 && logits[a] > max) max = logits[a];
    const prior = new Float32Array(n2);
    let sum = 0;
    for (let a = 0; a < n2; a++) if (board[a] === 0) { prior[a] = Math.exp(logits[a] - max); sum += prior[a]; }
    for (let a = 0; a < n2; a++) prior[a] /= sum;
    return { board, prior, children: new Map(), visits: 1, valueSum: value, terminal: false, netValue: value };
  }

  function search(m, board, sims, cPuct = 1.5) {
    const n = m.n;
    const rootNode = makeNode(m, board);
    for (let s = 0; s < sims; s++) {
      const path = [rootNode];
      let node = rootNode, leafValue;
      for (;;) {
        let best = -1, bestScore = -Infinity;
        const sq = Math.sqrt(node.visits);
        for (let a = 0; a < node.board.length; a++) {
          if (node.board[a] !== 0) continue;
          const child = node.children.get(a);
          const cv = child ? child.visits : 0;
          const q = child ? -child.valueSum / cv : 0;
          const score = q + cPuct * node.prior[a] * sq / (1 + cv);
          if (score > bestScore) { bestScore = score; best = a; }
        }
        let child = node.children.get(best);
        if (child && child.terminal) { path.push(child); leafValue = -1; break; }
        if (child) { node = child; path.push(node); continue; }
        const { board: nb, won } = step(node.board, n, best);
        if (won) {
          child = { board: nb, terminal: true, visits: 0, valueSum: 0, children: new Map() };
          leafValue = -1;
        } else {
          child = makeNode(m, nb);
          child.visits = 0; child.valueSum = 0;
          leafValue = child.netValue;
        }
        node.children.set(best, child);
        path.push(child);
        break;
      }
      // Back up, flipping sign each level (players alternate).
      let v = leafValue;
      for (let i = path.length - 1; i >= 0; i--) {
        path[i].visits += 1;
        path[i].valueSum += v;
        v = -v;
      }
    }
    const visits = new Float32Array(board.length);
    for (const [a, c] of rootNode.children) visits[a] = c.visits;
    let best = 0;
    for (let a = 0; a < visits.length; a++) if (visits[a] > visits[best]) best = a;
    return { move: best, visits, rootValue: rootNode.valueSum / rootNode.visits, prior: rootNode.prior };
  }

  const AZ = { step, connectsTopBottom, loadModel, evaluate, search };
  if (typeof module !== "undefined" && module.exports) module.exports = AZ;
  else root.AZ = AZ;
})(typeof window !== "undefined" ? window : this);
