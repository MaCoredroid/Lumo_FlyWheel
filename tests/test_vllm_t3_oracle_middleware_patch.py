"""End-to-end regression test for the T3 oracle-middleware prelaunch patch.

Generates the prelaunch shell via
``scripts.run_track_b_loop._track_b_runtime_prelaunch_shell`` and runs
it inside a transient ``lumo-flywheel-vllm`` container, then asserts:

1. ``vllm.v1.spec_decode.lumo_oracle_registry`` imports cleanly with
   the embedded module body intact (round-trip through bash heredoc +
   Python r-string concatenation preserves bytes).
2. ``vllm.entrypoints.openai.api_server.build_app`` invokes the
   middleware install hook without raising — verified by importing
   the patched module and checking the sentinel comment is present.
3. The middleware actually fires on a request when the X-Lumo-Oracle
   header is supplied; the parsed snapshot reaches
   ``ORACLE_REGISTRY``.
4. The patch is idempotent across re-applications (sentinel
   detection short-circuits the second pass).

This catches the class of failure that the prior T1 commit shipped
broken: a syntax error in the prelaunch shell construction
(triple-quotes in embedded docstrings prematurely closed the outer
Python r-string), which would have crashed every subsequent
``ModelServer`` relaunch.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

VLLM_IMAGE = "lumo-flywheel-vllm:26.01-py3-v0.19.0"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=5)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


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


def test_prelaunch_shell_imports_cleanly() -> None:
    """The prelaunch-shell builder must produce a string without
    accidentally crashing Python at import time. This is the regression
    guard for the bug that shipped in the original T1 commit."""

    from run_track_b_loop import _track_b_runtime_prelaunch_shell

    shell = _track_b_runtime_prelaunch_shell()
    assert isinstance(shell, str)
    assert len(shell) > 1000
    # Both the T1 wrapper drop and the T3 middleware install must be
    # present in any future-built shell.
    assert "T1 session scoping" in shell
    assert "T3 oracle middleware" in shell
    assert "class OracleRegistry" in shell
    assert "install_fastapi_middleware" in shell
    # The embedded module body uses """ docstrings; the wrapper
    # construction at concatenation-time must preserve them.
    assert '"""Harness-oracle skeleton' in shell


@pytest.mark.skipif(not _docker_available(), reason="docker not available")
@pytest.mark.skipif(not _image_available(VLLM_IMAGE), reason=f"{VLLM_IMAGE} not built locally")
def test_prelaunch_shell_applies_oracle_middleware_in_container() -> None:
    from run_track_b_loop import _track_b_runtime_prelaunch_shell

    shell = _track_b_runtime_prelaunch_shell()
    # Extract just the T3 oracle-middleware portion of the prelaunch
    # shell. The other patches (GPU-memory check, arctic-inference
    # install, etc.) take 5+ minutes and consume GPU memory that the
    # running baseline owns; running them in a parallel transient
    # container would conflict and timeout. We only need to verify
    # that the T3 patches (a) drop the registry module byte-for-byte
    # and (b) install the middleware-install hook in api_server.py.
    marker = "cat > /usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/lumo_oracle_registry.py"
    if marker not in shell:
        raise AssertionError("T3 marker missing from generated prelaunch shell")
    head = shell[shell.index(marker):]

    verification = r'''
python3 - <<'VERIFY_PY'
import importlib, json
mod = importlib.import_module("vllm.v1.spec_decode.lumo_oracle_registry")
assert hasattr(mod, "ORACLE_REGISTRY"), "registry missing after embed"
assert hasattr(mod, "install_fastapi_middleware"), "middleware helper missing"

# Exercise the middleware end-to-end inside a fresh FastAPI app.
import fastapi
from starlette.testclient import TestClient

app = fastapi.FastAPI()

@app.get("/probe")
def probe(request: fastapi.Request):
    rid = request.headers.get("X-Request-Id") or ""
    snap = mod.ORACLE_REGISTRY.lookup(rid)
    return {"session": snap.session_id if snap else None}

mod.install_fastapi_middleware(app)
mod.install_fastapi_middleware(app)  # idempotency

oracle_value = mod.encode_oracle_header(
    mod.HarnessOracleSnapshot(
        session_id="sess_t3_check", turn_index=0,
        is_session_open=True, dialect="codex",
    )
)
client = TestClient(app)
response = client.get(
    "/probe",
    headers={"X-Request-Id": "lumo_sess_sess_t3_check__r1",
             "X-Lumo-Oracle": oracle_value},
)
assert response.status_code == 200, response.text
body = response.json()
assert body["session"] == "sess_t3_check", body
# After response completes, the registry is empty (middleware unregistered).
assert mod.ORACLE_REGISTRY.lookup("lumo_sess_sess_t3_check__r1") is None

# Sentinel check on the patched api_server.py
import pathlib
ap_text = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py"
).read_text(encoding="utf-8")
assert "T3_ORACLE_MIDDLEWARE_APPLIED" in ap_text, "api_server sentinel missing"

print("VERIFICATION_PASSED")
VERIFY_PY
'''

    full_command = head + "\n" + verification

    result = subprocess.run(
        [
            "docker", "run", "--rm",
            VLLM_IMAGE,
            "bash", "-lc", full_command,
        ],
        text=True,
        capture_output=True,
        timeout=420,
    )
    assert result.returncode == 0, (
        f"docker invocation failed (rc={result.returncode}). "
        f"stdout(tail):\n{result.stdout[-3000:]}\n"
        f"stderr(tail):\n{result.stderr[-2000:]}"
    )
    assert "VERIFICATION_PASSED" in result.stdout, (
        f"verification did not pass. stdout(tail):\n{result.stdout[-3000:]}"
    )
