from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import sample_dcgm_during_task as sampler  # noqa: E402


def test_dcgm_sampler_stamps_runtime_config_hash(monkeypatch, tmp_path: Path) -> None:
    class FakeNvmlSampler:
        def __init__(self, gpu: int) -> None:
            self.gpu = gpu
            self.closed = False

        def sample(self) -> dict[str, object]:
            return {
                "ts": "2026-05-07T21:30:00.000Z",
                "gpu": self.gpu,
                "telemetry_source": "nvml",
                "profile_fields_available": False,
                "profile_fields_unavailable_reason": "nvml_fallback_only",
                "dram_active_pct": 0.4,
                "sm_active_pct": 0.5,
            }

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(sampler, "NvmlSampler", FakeNvmlSampler)
    out = tmp_path / "dcgm_samples.jsonl"

    rc = sampler.run(
        Namespace(
            out=str(out),
            gpu=0,
            interval_s=0.001,
            duration_s=0.0,
            flush_every=1,
            runtime_config_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            allow_unstamped_smoke=False,
        )
    )

    assert rc == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["runtime_config_hash"] == "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert rows[0]["profile_fields_unavailable_reason"] == "nvml_fallback_only"


def test_dcgm_sampler_rejects_unstamped_measurement(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="runtime-config-hash"):
        sampler.run(
            Namespace(
                out=str(tmp_path / "dcgm_samples.jsonl"),
                gpu=0,
                interval_s=0.001,
                duration_s=0.0,
                flush_every=1,
                runtime_config_hash="",
                allow_unstamped_smoke=False,
            )
        )

    assert not (tmp_path / "dcgm_samples.jsonl").exists()
