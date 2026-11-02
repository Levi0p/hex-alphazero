"""Export a trained network for the browser page (web/).

Writes web/models/hex{N}.json: the layer sizes plus every weight array as
base64 float32, in the same order web/az.js reads them. Also writes a few test
positions with the Python outputs so tests/test_web.py can check that the
JavaScript network gives the same answers.

    python -m hexaz.export_web --run runs/hex7_seed0
"""

import argparse
import base64
import json
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from hexaz import hex as H
from hexaz.net import PolicyValueNet


def b64(x) -> str:
    return base64.b64encode(np.asarray(x, np.float32).tobytes()).decode()


def export(run_dir: str, out_dir: str = "web/models", label: str | None = None) -> Path:
    cfg = json.loads((Path(run_dir) / "config.json").read_text())
    n, ch, blocks = cfg["size"], cfg["channels"], cfg["blocks"]
    net = PolicyValueNet(ch, blocks)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, n, n, 3)))
    params = flax.serialization.from_bytes(params, (Path(run_dir) / "params.msgpack").read_bytes())
    p = params["params"]

    def conv(c):
        return {"k": b64(c["kernel"]), "b": b64(c["bias"]), "shape": list(c["kernel"].shape)}

    def ln(l):
        return {"scale": b64(l["scale"]), "bias": b64(l["bias"])}

    model = {
        "size": n, "channels": ch, "blocks": blocks,
        "stem": conv(p["Conv_0"]),
        "res": [
            {"ln0": ln(p[f"ResBlock_{i}"]["LayerNorm_0"]), "conv0": conv(p[f"ResBlock_{i}"]["Conv_0"]),
             "ln1": ln(p[f"ResBlock_{i}"]["LayerNorm_1"]), "conv1": conv(p[f"ResBlock_{i}"]["Conv_1"])}
            for i in range(blocks)
        ],
        "final_ln": ln(p["LayerNorm_0"]),
        "policy": conv(p["Conv_1"]),
        "v1": {"w": b64(p["Dense_0"]["kernel"]), "b": b64(p["Dense_0"]["bias"])},
        "v2": {"w": b64(p["Dense_1"]["kernel"]), "b": b64(p["Dense_1"]["bias"])},
    }

    # Reference outputs on a few random positions, for the JS parity test.
    rng = np.random.default_rng(0)
    boards = []
    for k in range(4):
        b = np.zeros(n * n, np.int8)
        cells = rng.permutation(n * n)[: 2 * k + 1]
        b[cells[::2]] = 1
        b[cells[1::2]] = -1
        boards.append(b)
    obs = jax.vmap(H.observation)(H.State(jnp.array(np.stack(boards)), None, None, None))
    logits, value = net.apply(params, obs)
    model["check"] = [
        {"board": b.tolist(), "logits": np.asarray(l).tolist(), "value": float(v)}
        for b, l, v in zip(boards, logits, value)
    ]

    out = Path(out_dir) / f"hex{label or n}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model))
    # Same model as a script tag, so web/index.html also works opened as a local file.
    out.with_suffix(".js").write_text(
        f"(window.AZ_MODELS = window.AZ_MODELS || {{}})[{json.dumps(label or n)}] = {json.dumps(model)};\n"
    )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out-dir", default="web/models")
    print(export(ap.parse_args().run, ap.parse_args().out_dir))
