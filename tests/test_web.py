"""The browser agent (web/az.js) must give the same network outputs as JAX."""

import shutil
import subprocess
from pathlib import Path

import pytest

from hexaz.export_web import export

ROOT = Path(__file__).resolve().parent.parent

CHECK_JS = """
const AZ = require(process.argv[2]);
const json = require(process.argv[3]);
const m = AZ.loadModel(json);
let worst = 0;
for (const c of json.check) {
  const { logits, value } = AZ.evaluate(m, Int8Array.from(c.board));
  worst = Math.max(worst, Math.abs(value - c.value));
  for (let i = 0; i < logits.length; i++) worst = Math.max(worst, Math.abs(logits[i] - c.logits[i]));
}
console.log(worst);
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_js_matches_python(tmp_path):
    runs = sorted(p for p in (ROOT / "runs").glob("*/params.msgpack") if (p.parent / "config.json").exists())
    if not runs:
        pytest.skip("no trained run to export")
    model = export(str(runs[0].parent), str(tmp_path), label="test")
    script = tmp_path / "check.js"
    script.write_text(CHECK_JS)
    out = subprocess.run(
        ["node", str(script), str(ROOT / "web" / "az.js"), str(model)],
        capture_output=True, text=True, check=True,
    )
    assert float(out.stdout) < 1e-3, out.stdout
