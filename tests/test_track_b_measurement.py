from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import requests


def _load_measure_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "measure_track_b_real_workload.py"
    spec = importlib.util.spec_from_file_location("measure_track_b_real_workload", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_http_error_preserves_response_body() -> None:
    measure = _load_measure_module()
    response = requests.Response()
    response.status_code = 500
    response.reason = "Internal Server Error"
    response.url = "http://127.0.0.1:9950/v1/responses"
    response._content = b'{"error":"spec decode failed"}'

    with pytest.raises(requests.HTTPError) as exc_info:
        measure._raise_for_status_with_body(response)

    message = str(exc_info.value)
    assert "500 Server Error" in message
    assert "response_body=" in message
    assert "spec decode failed" in message
