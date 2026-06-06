#!/usr/bin/env python3
"""Apply the SWE agentic vLLM source fixes inside the FR12 direct server.

The FR12 WY-tree launcher uses the direct FR10/FR12 vLLM patch path so the
serving process imports ``lumo_flywheel_serving.fr10_gdn_tree_kernel`` from the
workspace.  The real SWE workload still needs the FR9 Qwen parser/protocol
guards from the maintained Round-F relaunch path.  This script reuses those
prelaunch blocks inside the direct launcher instead of duplicating them.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

WORKSPACE = Path("/workspace")
if WORKSPACE.is_dir():
    sys.path.insert(0, str(WORKSPACE))

from scripts.run_track_b_loop import _track_b_runtime_prelaunch_shell
from scripts.swe_x86_helpers.relaunch_qwen36_round import (
    _CAUSAL_CONV_CUDAGRAPH_ASSERT_FIX_BLOCK,
    _KEEP_MARKER,
    _QWEN36_FP8_CONFIG_FIX_BLOCK,
    _QWEN_REASONING_STREAM_BOUNDARY_BLOCK,
    _SPEC_TRACE_BLOCK,
)


def main() -> int:
    full = _track_b_runtime_prelaunch_shell()
    idx = full.find(_KEEP_MARKER)
    if idx < 0:
        raise RuntimeError("forced tool_choice marker not found in Track-B prelaunch")
    qwen_tool_parser_base = full[: idx + len(_KEEP_MARKER)]
    shell = (
        _QWEN36_FP8_CONFIG_FIX_BLOCK
        + _CAUSAL_CONV_CUDAGRAPH_ASSERT_FIX_BLOCK
        + qwen_tool_parser_base
        + _QWEN_REASONING_STREAM_BOUNDARY_BLOCK
        + _SPEC_TRACE_BLOCK
    )
    subprocess.run(["bash", "-lc", shell], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
