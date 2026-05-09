"""Integration test for the X-Lumo-Oracle FastAPI middleware.

Exercises ``install_fastapi_middleware`` inside the
``lumo-flywheel-vllm`` image (where FastAPI + Starlette are
available) and asserts:

1. The middleware registers a snapshot in ``ORACLE_REGISTRY`` keyed
   by ``X-Request-Id`` for every inbound request that carries an
   ``X-Lumo-Oracle`` header.
2. The snapshot is visible to handlers that look it up by request_id
   (the ``SuffixDecodingProposer.propose()`` lookup pattern).
3. The middleware unregisters on response completion so the registry
   stays bounded.
4. Re-installation on the same app is a no-op.
5. Requests without the oracle header cause no registry mutation.

Skipped when Docker or the image is unavailable.
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

VLLM_IMAGE = "lumo-flywheel-vllm:26.01-py3-v0.19.0"

_HARNESS_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "lumo_flywheel_serving"
    / "vllm_harness_oracle.py"
)


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "info"], check=True, capture_output=True, timeout=5
        )
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


INNER_SCRIPT = textwrap.dedent(
    """
    import sys, json
    sys.path.insert(0, "/lumo_src")
    from vllm_harness_oracle import (
        ORACLE_HEADER, ORACLE_REGISTRY, HarnessOracleSnapshot,
        encode_oracle_header, install_fastapi_middleware, parse_oracle_header,
    )
    import fastapi
    from starlette.testclient import TestClient

    app = fastapi.FastAPI()
    captured = {}

    @app.get("/probe")
    def _probe(request: fastapi.Request):
        rid = request.headers.get("X-Request-Id") or ""
        snap = ORACLE_REGISTRY.lookup(rid)
        captured["session_id"] = snap.session_id if snap else None
        captured["dialect"] = snap.dialect if snap else None
        return {"ok": "yes"}

    # Idempotent: install twice.
    install_fastapi_middleware(app)
    install_fastapi_middleware(app)

    snap = HarnessOracleSnapshot(session_id="sess_abc123", turn_index=0,
                                 is_session_open=True, dialect="codex")
    header_value = encode_oracle_header(snap)
    rid = "lumo_sess_sess_abc123__test1"

    ORACLE_REGISTRY.clear()
    client = TestClient(app)
    response = client.get(
        "/probe",
        headers={"X-Request-Id": rid, ORACLE_HEADER: header_value},
    )
    assert response.status_code == 200, response.text

    # The handler saw the snapshot via the registry.
    assert captured["session_id"] == "sess_abc123", captured
    assert captured["dialect"] == "codex", captured

    # Middleware unregistered on response completion.
    assert ORACLE_REGISTRY.lookup(rid) is None
    assert len(ORACLE_REGISTRY) == 0

    # Request without oracle header: no mutation.
    response2 = client.get("/probe", headers={"X-Request-Id": "no-oracle"})
    assert response2.status_code == 200
    assert ORACLE_REGISTRY.lookup("no-oracle") is None
    assert len(ORACLE_REGISTRY) == 0

    print("VERIFICATION_PASSED")
    """
).strip()


@pytest.mark.skipif(not _docker_available(), reason="docker not available")
@pytest.mark.skipif(not _image_available(VLLM_IMAGE), reason=f"{VLLM_IMAGE} not built locally")
def test_oracle_middleware_registers_and_unregisters_inside_vllm_image() -> None:
    result = subprocess.run(
        [
            "docker", "run", "--rm",
            "-v", f"{_HARNESS_PATH.parent}:/lumo_src:ro",
            VLLM_IMAGE,
            "python3", "-c", INNER_SCRIPT,
        ],
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"docker invocation failed (rc={result.returncode}). "
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr[-2000:]}"
    )
    assert "VERIFICATION_PASSED" in result.stdout, (
        f"verification did not pass. stdout:\n{result.stdout}"
    )
