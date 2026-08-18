#!/usr/bin/env python3
"""FR14 lane 5A: container-side patcher that captures the REAL pre-``lm_head``
hidden states -- and the engine's own argmax -- from a live serve.

RUNS INSIDE THE CONTAINER against the pinned image's site-packages, after
``scripts/fr14_patch_nvfp4_lmhead.py`` and before ``vllm serve``.  Edits land in
the container's ephemeral writable layer and die with ``docker rm -f``; the
image digest never moves.  DEFAULT OFF: with ``FR14_LANE5A_CAPTURE`` unset the
patcher is a no-op, and the production launcher does not mount this file at all.

WHAT IT TAKES, AND WHY THERE
----------------------------
``LogitsProcessor._get_logits`` (``logits_processor.py:96``) is the exact call
site whose GEMM the NVFP4 port replaces:
``lm_head.quant_method.apply(lm_head, hidden_states, ...)``.  Capturing its
INPUT gives the offline characterisation genuinely real hidden states -- the
ones the served model produced on the served prompts -- instead of synthetic
vectors that would flatter or slander the quantisation by accident.

Capturing its OUTPUT's argmax alongside is the control that makes the offline
work checkable: if an offline NVFP4 dequant of the on-disk head, applied to
these hidden states, does NOT reproduce the device kernel's argmax, then the
offline model of the head is wrong and every number built on it is void.

Three files, all raw little-endian:
    <path>              float32 [rows, hidden]  -- hidden states
    <path>.top.bin      int32 argmax, float32 top-1 logit, float32 top-2 logit
    <path>.meta.json    row/call counts, logit widths, live quant method

WHY BAKED, NOT READ AT RUNTIME.  vLLM v1 builds the model in a separate
EngineCore/worker process whose environment is curated -- this repo's launcher
notes only 14 of 66 ``FR13_*`` vars survive into it.  The path and the row cap
are therefore read HERE, in the container's own shell, and baked into the
emitted source as literals.  Same rule ``fr14_patch_nvfp4_lmhead.py`` follows,
same reason.

The capture body is wrapped in ``try/except``: a probe must never be able to
take down the serve it is observing.
"""

from __future__ import annotations

import json
import os
import py_compile
import sys
from pathlib import Path

SP = Path("/usr/local/lib/python3.12/dist-packages/vllm")
LP_PATH = SP / "model_executor/layers/logits_processor.py"

CAPTURE = os.environ.get("FR14_LANE5A_CAPTURE", "")
MAX_ROWS = int(os.environ.get("FR14_LANE5A_CAPTURE_MAX_ROWS", "8192"))

applied: dict[str, str] = {}


def _replace_once(path: Path, needle: str, replacement: str, tag: str) -> str:
    text = path.read_text()
    n = text.count(needle)
    if n != 1:
        raise RuntimeError(
            f"FR14 lane5A patch: anchor {tag!r} occurs {n} times in {path} "
            "(expected exactly 1) -- refusing to patch"
        )
    path.write_text(text.replace(needle, replacement, 1))
    return "patched"


LP_NEEDLE = """        # Get the logits for the next tokens.
        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
"""

LP_REPLACEMENT = '''        # Get the logits for the next tokens.
        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)

        # FR14_LANE5A_CAPTURE (probe only, baked path -- never read from the
        # curated worker's os.environ). Banks this call's INPUT and the
        # device kernel's own argmax, so the offline head model can be checked
        # against the kernel rather than merely asserted to match it.
        try:
            _fr14 = globals().get("_FR14_CAP")
            if _fr14 is None:
                import json as _fr14_json
                _fr14 = {{
                    "path": {capture!r},
                    "max_rows": {max_rows!r},
                    "rows": 0,
                    "calls": 0,
                    "widths": {{}},
                    "json": _fr14_json,
                    "fh": open({capture!r}, "wb"),
                    "th": open({capture!r} + ".top.bin", "wb"),
                }}
                globals()["_FR14_CAP"] = _fr14
            _fr14["calls"] += 1
            _w = int(logits.shape[-1]) if logits is not None else -1
            _fr14["widths"][str(_w)] = _fr14["widths"].get(str(_w), 0) + 1
            if logits is not None and _fr14["rows"] < _fr14["max_rows"]:
                _h = hidden_states.reshape(-1, hidden_states.shape[-1])
                _lg = logits.reshape(-1, logits.shape[-1])
                _n = min(_h.shape[0], _fr14["max_rows"] - _fr14["rows"])
                _fr14["fh"].write(
                    _h[:_n].detach().to("cpu", torch.float32).contiguous().numpy().tobytes()
                )
                _t2 = _lg[:_n].detach().float().topk(2, dim=-1)
                # One 12-byte record PER ROW (int32 argmax, float32 top1,
                # float32 top2) so the file parses without knowing where the
                # engine happened to put its call boundaries.
                _rec = torch.empty((_n, 3), dtype=torch.int32)
                _rec[:, 0] = _t2.indices[:, 0].to("cpu", torch.int32)
                _rec[:, 1:] = _t2.values.to("cpu", torch.float32).contiguous().view(torch.int32)
                _fr14["th"].write(_rec.contiguous().numpy().tobytes())
                _fr14["rows"] += _n
                _fr14["fh"].flush()
                _fr14["th"].flush()
                with open(_fr14["path"] + ".meta.json", "w") as _mf:
                    _fr14["json"].dump(
                        {{
                            "rows": _fr14["rows"],
                            "hidden": int(hidden_states.shape[-1]),
                            "hidden_dtype": "float32",
                            "top_record": "int32 argmax then float32 top1, float32 top2",
                            "calls": _fr14["calls"],
                            "logit_widths": _fr14["widths"],
                            "quant_method": type(lm_head.quant_method).__name__,
                        }},
                        _mf,
                    )
        except Exception as _fr14_e:  # a probe must never break its subject
            print("FR14_LANE5A_CAPTURE failed: %r" % (_fr14_e,), file=sys.stderr)
'''.format(capture=CAPTURE, max_rows=MAX_ROWS)


_t = LP_PATH.read_text()
if "FR14_LANE5A_CAPTURE" in _t:
    applied["logits_processor"] = "already-patched"
elif not CAPTURE:
    applied["logits_processor"] = "skipped (FR14_LANE5A_CAPTURE unset)"
else:
    if "\nimport sys\n" not in _t:
        LP_PATH.write_text(_t.replace("import torch\n", "import sys\n\nimport torch\n", 1))
    applied["logits_processor"] = _replace_once(
        LP_PATH, LP_NEEDLE, LP_REPLACEMENT, "logits_processor:_get_logits"
    )

py_compile.compile(str(LP_PATH), doraise=True)

print(
    json.dumps(
        {
            "fr14_lane5a_patch": applied,
            "capture_path": CAPTURE,
            "max_rows": MAX_ROWS,
        },
        indent=1,
    )
)
sys.exit(0)
