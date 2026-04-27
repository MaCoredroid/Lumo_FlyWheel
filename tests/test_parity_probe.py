from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from lumo_flywheel_serving import parity_probe
from lumo_flywheel_serving.parity_fixture import (
    DELTANET_STATE_DEBUG_KIND,
    LOGITS_DEBUG_KIND,
)


def _write_probes_input(fixture_dir: Path, probe_count: int, *, output_token_count: int = 16) -> None:
    rows = [
        {
            "probe_index": i,
            "prompt": f"probe-{i}",
            "output_token_count": output_token_count,
            "family_id": "test-family",
        }
        for i in range(probe_count)
    ]
    (fixture_dir / "probes_input.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _save_logits_pt(path: Path, *, request_id: str, token: int, values: list[float], torch_module: Any) -> None:
    torch_module.save(
        {
            "kind": LOGITS_DEBUG_KIND,
            "request_id": request_id,
            "generated_token_index": token,
            "source_shape": (len(values),),
            "saved_shape": (len(values),),
            "logits_is_truncated": False,
            "logits": torch_module.tensor(values, dtype=torch_module.float32),
        },
        path,
    )


def _save_state_pt(path: Path, *, request_id: str, token: int, values: list[float], torch_module: Any) -> None:
    torch_module.save(
        {
            "kind": DELTANET_STATE_DEBUG_KIND,
            "request_id": request_id,
            "generated_token_index": token,
            "layers": {
                "model.layers.0.linear_attn": [
                    {
                        "state_index": 1,
                        "state_role": "recurrent_ssm_state",
                        "saved_shape": (len(values),),
                        "saved_dtype": "torch.float32",
                        "tensor": torch_module.tensor(values, dtype=torch_module.float32),
                    }
                ]
            },
        },
        path,
    )


def _array_sha256(np_module: Any, array: Any) -> str:
    import hashlib

    contiguous = np_module.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(tuple(int(dim) for dim in contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _build_test_fixture(
    tmp_path: Path,
    *,
    kernel_target: str,
    torch_module: Any,
    np_module: Any,
    probe_count: int = 2,
    target_token: int = 1,
    state_tokens: tuple[int, ...] = (1, 1024),
    reference_logit_values: list[float] | None = None,
    reference_state_values_by_token: dict[int, list[float]] | None = None,
) -> Path:
    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    _write_probes_input(fixture_dir, probe_count, output_token_count=max(state_tokens) if state_tokens else 16)

    logit_values = reference_logit_values or [1.0, 2.0, 3.0, 4.0]
    logit_source_paths: list[Path] = []
    logit_source_sha: list[str] = []
    logit_source_request_ids: list[str] = []
    logit_source_tokens: list[int] = []
    logit_first_16: list[Any] = []
    for i in range(probe_count):
        request_id = f"src-{i}"
        path = sources_dir / f"logits_req_{request_id}_tok_{target_token:06d}.pt"
        _save_logits_pt(path, request_id=request_id, token=target_token, values=logit_values, torch_module=torch_module)
        logit_source_paths.append(path)
        array = np_module.asarray(logit_values, dtype=np_module.float32)
        logit_source_sha.append(_array_sha256(np_module, array))
        logit_first_16.append(array[:16].astype(np_module.float32, copy=False))
        logit_source_request_ids.append(request_id)
        logit_source_tokens.append(target_token)

    np_module.savez(
        fixture_dir / f"{kernel_target}_reference_logits.npz",
        probe_index=np_module.arange(probe_count, dtype=np_module.int32),
        reference_storage=np_module.array(["sha256_float32_with_first_16_sample_from_real_vllm_debug_export"]),
        source_artifact_path_by_probe=np_module.array([str(p) for p in logit_source_paths]),
        source_artifact_sha256_by_probe=np_module.array([_file_sha256(p) for p in logit_source_paths]),
        source_request_id_by_probe=np_module.array(logit_source_request_ids),
        source_generated_token_index_by_probe=np_module.array(logit_source_tokens, dtype=np_module.int32),
        source_logits_sha256_by_probe=np_module.array(logit_source_sha),
        source_logits_sample_first_16_float32=np_module.stack(
            [a[:16] for a in logit_first_16], axis=0
        ).astype(np_module.float32),
        source_saved_shape_by_probe=np_module.array([json.dumps([len(logit_values)]) for _ in range(probe_count)]),
        source_shape_by_probe=np_module.array([json.dumps([len(logit_values)]) for _ in range(probe_count)]),
        source_logits_is_truncated_by_probe=np_module.array([False] * probe_count, dtype=np_module.bool_),
    )

    if kernel_target == "deltanet":
        state_payload: dict[str, Any] = {
            "probe_index": np_module.arange(probe_count, dtype=np_module.int32),
            "state_storage": np_module.array(["debug_export_file_sha256_from_real_vllm_state_snapshot"]),
        }
        for token in state_tokens:
            values = (reference_state_values_by_token or {}).get(token, [float(token), 1.0])
            file_shas: list[str] = []
            paths: list[str] = []
            request_ids: list[str] = []
            for i in range(probe_count):
                request_id = f"src-{i}"
                state_path = sources_dir / f"state_req_{request_id}_tok_{token:06d}.pt"
                _save_state_pt(
                    state_path,
                    request_id=request_id,
                    token=token,
                    values=values,
                    torch_module=torch_module,
                )
                file_shas.append(_file_sha256(state_path))
                paths.append(str(state_path))
                request_ids.append(request_id)
            state_payload[f"state_token_{token}"] = np_module.array(file_shas)
            state_payload[f"state_token_{token}_source_path_by_probe"] = np_module.array(paths)
            state_payload[f"state_token_{token}_request_id_by_probe"] = np_module.array(request_ids)
        np_module.savez(fixture_dir / "deltanet_reference_state.npz", **state_payload)

    fixture_yaml = {
        "fixture_id": f"test-family-{kernel_target}-v1",
        "generated_at": "2026-04-26T00:00:00Z",
        "probe_count": probe_count,
        "probe_token_lengths": [16],
        "probe_input_ref": "probes_input.jsonl",
        "reference_logits_ref": f"{kernel_target}_reference_logits.npz",
        "tolerances": {
            "rtol_logit": 0.001,
            "atol_logit": 0.001,
            "rtol_state": 0.005,
            "atol_state": 0.005,
        },
    }
    if kernel_target == "deltanet":
        fixture_yaml["reference_state_snapshots_ref"] = "deltanet_reference_state.npz"
        fixture_yaml["state_checkpoints_at_token"] = list(state_tokens)
        fixture_yaml["parity_check_method"] = "logit_plus_state_compare"
    else:
        fixture_yaml["parity_check_method"] = "per_token_logit_compare"

    import yaml as _yaml

    (fixture_dir / f"{kernel_target}_v1.yaml").write_text(
        _yaml.safe_dump(fixture_yaml, sort_keys=False), encoding="utf-8"
    )
    return fixture_dir


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_fake_post(
    *,
    staging_dir: Path,
    kernel_target: str,
    state_tokens: tuple[int, ...],
    target_token: int,
    candidate_logits_for_probe: Callable[[int], list[float]],
    candidate_state_for_probe: Callable[[int, int], list[float]] | None,
    torch_module: Any,
) -> Callable[..., Any]:
    state: dict[str, int] = {"probe": 0}

    class _Resp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"id": "cmpl-test"}

    def _post(url: str, headers: dict[str, str], json: dict[str, Any], timeout: float) -> Any:
        probe_index = state["probe"]
        request_id = f"cand-{probe_index}"
        staging_dir.mkdir(parents=True, exist_ok=True)
        _save_logits_pt(
            staging_dir / f"logits_req_{request_id}_tok_{target_token:06d}.pt",
            request_id=request_id,
            token=target_token,
            values=candidate_logits_for_probe(probe_index),
            torch_module=torch_module,
        )
        if kernel_target == "deltanet":
            assert candidate_state_for_probe is not None
            for token in state_tokens:
                _save_state_pt(
                    staging_dir / f"state_req_{request_id}_tok_{token:06d}.pt",
                    request_id=request_id,
                    token=token,
                    values=candidate_state_for_probe(probe_index, token),
                    torch_module=torch_module,
                )
        state["probe"] += 1
        return _Resp()

    return _post


def _patch_quiet_window(monkeypatch) -> None:
    monkeypatch.setattr(parity_probe.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(parity_probe.time, "time", _make_clock())


def _make_clock() -> Callable[[], float]:
    state = {"now": 0.0}

    def _now() -> float:
        state["now"] += 1.0
        return state["now"]

    return _now


def _common_dirs(tmp_path: Path) -> tuple[Path, Path]:
    debug = tmp_path / "debug_export"
    debug.mkdir()
    staging = debug / "staging"
    return debug, staging


def test_parity_probe_passes_when_candidate_matches_reference_exactly(tmp_path: Path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    fixture_dir = _build_test_fixture(
        tmp_path,
        kernel_target="deltanet",
        torch_module=torch,
        np_module=np,
        probe_count=2,
        target_token=1,
        state_tokens=(1, 1024),
        reference_logit_values=[1.0, 2.0, 3.0, 4.0],
        reference_state_values_by_token={1: [1.0, 0.5], 1024: [1024.0, 0.5]},
    )
    debug, staging = _common_dirs(tmp_path)

    monkeypatch.setattr(
        parity_probe.requests,
        "post",
        _make_fake_post(
            staging_dir=staging,
            kernel_target="deltanet",
            state_tokens=(1, 1024),
            target_token=1,
            candidate_logits_for_probe=lambda i: [1.0, 2.0, 3.0, 4.0],
            candidate_state_for_probe=lambda i, t: [1.0, 0.5] if t == 1 else [1024.0, 0.5],
            torch_module=torch,
        ),
    )
    _patch_quiet_window(monkeypatch)

    result = parity_probe.run_parity_probe(
        repo_root=tmp_path,
        fixture_dir=fixture_dir,
        kernel_target="deltanet",
        endpoint="http://127.0.0.1:8100/v1",
        model="test-model",
        debug_export_dir=debug,
    )
    assert result.pass_, result.as_dict()
    assert result.reason == "ran_passed"
    assert result.probes_passed == 2
    assert result.first_diverging_probe is None
    assert result.tolerance_overshoot == 0.0


def test_parity_probe_passes_within_tolerance(tmp_path: Path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    fixture_dir = _build_test_fixture(
        tmp_path,
        kernel_target="gatedattn",
        torch_module=torch,
        np_module=np,
        probe_count=1,
        target_token=1,
        state_tokens=(),
        reference_logit_values=[1.0, 2.0, 3.0, 4.0],
    )
    debug, staging = _common_dirs(tmp_path)

    monkeypatch.setattr(
        parity_probe.requests,
        "post",
        _make_fake_post(
            staging_dir=staging,
            kernel_target="gatedattn",
            state_tokens=(),
            target_token=1,
            candidate_logits_for_probe=lambda i: [1.0005, 2.0005, 3.0005, 4.0005],
            candidate_state_for_probe=None,
            torch_module=torch,
        ),
    )
    _patch_quiet_window(monkeypatch)

    result = parity_probe.run_parity_probe(
        repo_root=tmp_path,
        fixture_dir=fixture_dir,
        kernel_target="gatedattn",
        endpoint="http://127.0.0.1:8100/v1",
        model="test-model",
        debug_export_dir=debug,
    )
    assert result.pass_, result.as_dict()
    assert result.reason == "ran_passed"
    assert result.tolerance_overshoot == 0.0


def test_parity_probe_fails_on_logit_divergence(tmp_path: Path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    fixture_dir = _build_test_fixture(
        tmp_path,
        kernel_target="gatedattn",
        torch_module=torch,
        np_module=np,
        probe_count=2,
        target_token=1,
        state_tokens=(),
        reference_logit_values=[1.0, 2.0, 3.0, 4.0],
    )
    debug, staging = _common_dirs(tmp_path)

    monkeypatch.setattr(
        parity_probe.requests,
        "post",
        _make_fake_post(
            staging_dir=staging,
            kernel_target="gatedattn",
            state_tokens=(),
            target_token=1,
            candidate_logits_for_probe=lambda i: [1.0, 2.0, 3.0, 5.5],
            candidate_state_for_probe=None,
            torch_module=torch,
        ),
    )
    _patch_quiet_window(monkeypatch)

    result = parity_probe.run_parity_probe(
        repo_root=tmp_path,
        fixture_dir=fixture_dir,
        kernel_target="gatedattn",
        endpoint="http://127.0.0.1:8100/v1",
        model="test-model",
        debug_export_dir=debug,
    )
    assert not result.pass_, result.as_dict()
    assert result.reason == "parity_logit_diverged"
    assert result.first_diverging_probe == 0
    assert result.tolerance_overshoot > 0.0


def test_parity_probe_fails_on_state_divergence_deltanet(tmp_path: Path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    fixture_dir = _build_test_fixture(
        tmp_path,
        kernel_target="deltanet",
        torch_module=torch,
        np_module=np,
        probe_count=1,
        target_token=1,
        state_tokens=(1, 1024),
        reference_logit_values=[1.0, 2.0, 3.0, 4.0],
        reference_state_values_by_token={1: [1.0, 0.5], 1024: [1024.0, 0.5]},
    )
    debug, staging = _common_dirs(tmp_path)

    monkeypatch.setattr(
        parity_probe.requests,
        "post",
        _make_fake_post(
            staging_dir=staging,
            kernel_target="deltanet",
            state_tokens=(1, 1024),
            target_token=1,
            candidate_logits_for_probe=lambda i: [1.0, 2.0, 3.0, 4.0],
            candidate_state_for_probe=lambda i, t: ([1.0, 0.5] if t == 1 else [9999.0, 0.5]),
            torch_module=torch,
        ),
    )
    _patch_quiet_window(monkeypatch)

    result = parity_probe.run_parity_probe(
        repo_root=tmp_path,
        fixture_dir=fixture_dir,
        kernel_target="deltanet",
        endpoint="http://127.0.0.1:8100/v1",
        model="test-model",
        debug_export_dir=debug,
    )
    assert not result.pass_, result.as_dict()
    assert result.reason == "parity_state_diverged"
    assert result.first_diverging_probe == 0


def test_parity_probe_returns_capture_failed_when_no_files_appear(tmp_path: Path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    fixture_dir = _build_test_fixture(
        tmp_path,
        kernel_target="gatedattn",
        torch_module=torch,
        np_module=np,
        probe_count=1,
        target_token=1,
        state_tokens=(),
        reference_logit_values=[1.0, 2.0, 3.0, 4.0],
    )
    debug, _staging = _common_dirs(tmp_path)

    class _Resp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"id": "cmpl-test"}

    def _post_no_export(*_a: Any, **_k: Any) -> Any:
        return _Resp()

    monkeypatch.setattr(parity_probe.requests, "post", _post_no_export)
    _patch_quiet_window(monkeypatch)

    result = parity_probe.run_parity_probe(
        repo_root=tmp_path,
        fixture_dir=fixture_dir,
        kernel_target="gatedattn",
        endpoint="http://127.0.0.1:8100/v1",
        model="test-model",
        debug_export_dir=debug,
        quiet_timeout_s=0.5,
        quiet_window_s=0.1,
    )
    assert not result.pass_
    assert result.reason == "capture_failed"
    assert result.first_diverging_probe == 0


def test_parity_probe_returns_endpoint_unreachable_on_request_exception(tmp_path: Path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    fixture_dir = _build_test_fixture(
        tmp_path,
        kernel_target="gatedattn",
        torch_module=torch,
        np_module=np,
        probe_count=1,
        target_token=1,
        state_tokens=(),
        reference_logit_values=[1.0, 2.0, 3.0, 4.0],
    )
    debug, _staging = _common_dirs(tmp_path)

    def _post_explodes(*_a: Any, **_k: Any) -> Any:
        raise parity_probe.requests.ConnectionError("connection refused")

    monkeypatch.setattr(parity_probe.requests, "post", _post_explodes)
    _patch_quiet_window(monkeypatch)

    result = parity_probe.run_parity_probe(
        repo_root=tmp_path,
        fixture_dir=fixture_dir,
        kernel_target="gatedattn",
        endpoint="http://127.0.0.1:8100/v1",
        model="test-model",
        debug_export_dir=debug,
    )
    assert not result.pass_
    assert result.reason == "endpoint_unreachable"
    assert "ConnectionError" in (result.error_detail or "")
