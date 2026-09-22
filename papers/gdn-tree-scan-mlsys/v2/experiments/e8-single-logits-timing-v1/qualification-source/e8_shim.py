#!/usr/bin/env python3
"""Apply the E8-only post-patcher edit to one exact emitted eagle.py revision."""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED = "aa022e2a1fc7993d0f7d4223f8366471e02e3c0e2740934c7a9490f7b45b01d7"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def render(raw, arm, qualify):
    if digest(raw) != EXPECTED:
        raise ValueError("REFUSED: emitted eagle.py differs from the frozen E1 source")
    if arm not in ("on", "off") or type(qualify) is not bool:
        raise ValueError("invalid arm/qualification")
    text = raw.decode()
    edits = []

    def replace(old, new, count=1):
        nonlocal text
        if text.count(old) != count:
            raise ValueError(f"REFUSED: anchor count {text.count(old)} != {count}: {old[:80]!r}")
        text = text.replace(old, new)
        edits.append({"anchor_sha256": digest(old.encode()), "count": count})

    replace("import torch\n", "import torch\nfrom e8_head_gate import primary_logits as _e8_primary, legacy_logits as _e8_legacy, finish_proposal as _e8_finish\n_E8_ARM = " + repr(arm) + "\n_E8_QUALIFY = " + repr(qualify) + "\n")
    replace("True  # FR13_DRAFTER_SINGLE_LOGITS baked ON", repr(arm == "on") + "  # E8 source-controlled arm; no worker-env dependency")
    replace('"1",  # FR13_DRAFTER_SINGLE_LOGITS baked ON', repr("1" if arm == "on" else "0") + ",  # E8 actual source arm")
    replace('and os.environ.get("FR13_FIX1_SELFCHECK", "0") == "1"', "and _E8_QUALIFY  # existing FIX1 check; baked from qualification manifest")
    replace("return self.model.compute_logits(hidden_states).argmax(dim=-1)", "return _e8_legacy(self, hidden_states, _E8_ARM, _E8_QUALIFY).argmax(dim=-1)")
    replace("_fr10_logits = self.model.compute_logits(\n                        sample_hidden_states\n                    )", '_fr10_logits = _e8_primary(self, sample_hidden_states, "root", _E8_ARM, _E8_QUALIFY)')
    replace("_fr10_logits = self.model.compute_logits(sample_hidden_states)", '_fr10_logits = _e8_primary(self, sample_hidden_states, "root", _E8_ARM, _E8_QUALIFY)')
    replace("_fr10_step_logits = self.model.compute_logits(\n                            last_hidden_states[:batch_size]\n                        )", '_fr10_step_logits = _e8_primary(self, last_hidden_states[:batch_size], "loop", _E8_ARM, _E8_QUALIFY)', 2)
    replace("            return _fr10_packed\n", "            _e8_finish(self, _fr10_packed, _E8_ARM, _E8_QUALIFY)\n            return _fr10_packed\n")
    compile(text, "eagle.e8.py", "exec")
    data = text.encode()
    return data, {"schema": "e8.shim.v1", "arm": arm, "qualify": qualify, "input_sha256": EXPECTED, "output_sha256": digest(data), "edits": edits}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eagle", default="/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/eagle.py")
    ap.add_argument("--arm", choices=["on", "off"], required=True)
    ap.add_argument("--qualify", action="store_true")
    ap.add_argument("--output", help="write a separate dry-run file instead of modifying the container module")
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    p = Path(a.eagle)
    data, report = render(p.read_bytes(), a.arm, a.qualify)
    out = Path(a.output) if a.output else p
    if a.output and out.exists():
        raise ValueError("REFUSED: output already exists")
    out.write_bytes(data)
    Path(a.report).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
