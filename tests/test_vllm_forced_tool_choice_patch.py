"""Regression test for the vLLM Responses API forced tool_choice patch.

Background: vllm/parser/abstract_parser.py:_parse_tool_calls bypasses the
configured tool parser when tool_choice is forced (ToolChoiceFunction or
ChatCompletionNamedToolChoiceParam) and stuffs the raw model output text
into FunctionCall.arguments. Codex 0.128.0 uses auto tool_choice so the
production path is unaffected, but Track B's Step 0d correctness gate
uses forced tool_choice and saw raw qwen3_xml payloads land in
arguments verbatim. Upstream Issue #23227 closed as not-planned.

The patch (codified in scripts/run_track_b_loop.py:_track_b_runtime_
prelaunch_shell) runs the configured tool parser on `content` and uses
its parsed arguments. The forced name still overrides whatever the
parser thinks.

This test runs the patch logic end-to-end against the same vLLM image
the running container uses, with the actual qwen3_xml parser, to assert
forced tool_choice + Qwen3 newline-delimited XML produces parsed JSON
arguments. Skipped if Docker or the image is unavailable.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

VLLM_IMAGE = "lumo-flywheel-vllm:26.01-py3-v0.19.0"

INNER_SCRIPT = r'''
import json
from pathlib import Path

ap = Path("/usr/local/lib/python3.12/dist-packages/vllm/parser/abstract_parser.py")
src = ap.read_text(encoding="utf-8")
if "Local patch (Lumo Track B 2026-05-08)" not in src:
    old = (
        "        if request.tool_choice and isinstance(request.tool_choice, ToolChoiceFunction):\n"
        "            # Forced Function Call (Responses API style)\n"
        "            assert content is not None\n"
        "            function_calls.append(\n"
        "                FunctionCall(name=request.tool_choice.name, arguments=content)\n"
        "            )\n"
        "            return function_calls, None  # Clear content since tool is called.\n"
    )
    new = (
        "        if request.tool_choice and isinstance(request.tool_choice, ToolChoiceFunction):\n"
        "            # Local patch (Lumo Track B 2026-05-08)\n"
        "            assert content is not None\n"
        "            arguments = content\n"
        "            if self._tool_parser is not None:\n"
        "                tool_call_info = self._tool_parser.extract_tool_calls(content, request=request)\n"
        "                if tool_call_info is not None and tool_call_info.tools_called and tool_call_info.tool_calls:\n"
        "                    arguments = tool_call_info.tool_calls[0].function.arguments\n"
        "            function_calls.append(\n"
        "                FunctionCall(name=request.tool_choice.name, arguments=arguments)\n"
        "            )\n"
        "            return function_calls, None\n"
    )
    if old not in src:
        print("PATCH_TARGET_NOT_FOUND")
        raise SystemExit(2)
    ap.write_text(src.replace(old, new), encoding="utf-8")

from vllm.parser.parser_manager import ParserManager
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

class _Tok:
    eos_token_id = 0
    vocab_size = 1
    def encode(self, *a, **kw): return []
    def decode(self, *a, **kw): return ""
    def get_vocab(self): return {"<think>": 100, "</think>": 101}

pm = ParserManager()
ParserCls = pm.get_parser(
    tool_parser_name="qwen3_xml",
    reasoning_parser_name="qwen3",
    enable_auto_tools=True,
    model_name="qwen3.5-27b",
)
parser = ParserCls(_Tok())

req = ResponsesRequest(
    model="qwen3.5-27b",
    input="ignored",
    tools=[{
        "type": "function",
        "name": "read_file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }],
    tool_choice={"type": "function", "name": "read_file"},
)
content = "\n\n<tool_call>\n<function=read_file>\n<parameter=path>\nAGENTS.md\n</parameter>\n</function>\n</tool_call>"

fcs, _ = parser._parse_tool_calls(request=req, content=content, enable_auto_tools=True)
assert len(fcs) == 1, f"expected 1 function_call got {len(fcs)}"
print(f"NAME={fcs[0].name}")
print(f"ARGS={fcs[0].arguments}")
parsed = json.loads(fcs[0].arguments)
assert parsed == {"path": "AGENTS.md"}, f"unexpected parsed: {parsed}"
print("VERIFICATION_PASSED")
'''


def _docker_available() -> bool:
    return bool(shutil.which("docker"))


def _image_available(image: str) -> bool:
    try:
        subprocess.run(
            ["docker", "image", "inspect", image],
            check=True,
            capture_output=True,
            timeout=10,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


@pytest.mark.skipif(not _docker_available(), reason="docker not available")
@pytest.mark.skipif(not _image_available(VLLM_IMAGE), reason=f"{VLLM_IMAGE} not built locally")
def test_forced_tool_choice_runs_parser_on_qwen3_xml_payload() -> None:
    """Forced tool_choice + qwen3_xml \\n-delimited XML must parse to JSON.

    Without the patch, FunctionCall.arguments contains the raw XML payload
    and json.loads fails. With the patch, arguments contains parsed JSON
    matching the model's intent (in this case `{"path": "AGENTS.md"}`).
    """
    result = subprocess.run(
        ["docker", "run", "--rm", VLLM_IMAGE, "python3", "-c", INNER_SCRIPT],
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert result.returncode == 0, (
        f"docker invocation failed (rc={result.returncode}). "
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr[-1500:]}"
    )
    assert "VERIFICATION_PASSED" in result.stdout, (
        f"verification did not pass. stdout:\n{result.stdout}"
    )
    assert "ARGS=" in result.stdout
    args_line = next(line for line in result.stdout.splitlines() if line.startswith("ARGS="))
    args_value = args_line.removeprefix("ARGS=")
    assert args_value.startswith("{") and args_value.endswith("}"), (
        f"arguments expected to be JSON object, got: {args_value!r}"
    )
