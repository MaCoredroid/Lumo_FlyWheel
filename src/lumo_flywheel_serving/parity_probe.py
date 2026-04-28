"""L0c parity probe — verifies a candidate kernel matches a parity fixture within tolerance.

Reuses vLLM's debug-export hooks (the same primitive `scripts/build_parity_fixture.py`
uses to capture reference fixtures) to capture candidate logits/state tensors per probe,
then compares against fixture references via two tiers:

1. Exact: candidate logits/state sha256 == fixture-stored sha256 → instant pass for that
   probe, no array load needed.
2. Tolerance: load both candidate and reference arrays (reference from the source .pt
   path the fixture provenance points to), compare via element-wise rtol/atol.

The fixture format consumed here is the post-`_convert_debug_artifacts_to_p2b_fixture_set`
schema (the schema currently on disk for the heavy family). It stores per-probe sha256 +
a 16-float sample plus a path back to the original source .pt.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
import yaml

from lumo_flywheel_serving.parity_fixture import (
    DELTANET_STATE_DEBUG_KIND,
    LOGITS_DEBUG_KIND,
    _flatten_state_payload,
    load_debug_export_pt,
)

DEBUG_EXPORT_RE = re.compile(
    r"^(?P<kind>logits|state)_req_(?P<request_id>.+)_tok_(?P<token>[0-9]{6})\.pt$"
)


@dataclass(frozen=True)
class ParityProbeResult:
    pass_: bool
    fixture_id: str
    kernel_target: str
    probes_total: int
    probes_passed: int
    first_diverging_probe: int | None
    tolerance_overshoot: float
    reason: str
    error_detail: str | None = None
    per_probe: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    checkpoints_checked: tuple[int, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "pass": self.pass_,
            "reason": self.reason,
            "fixture_id": self.fixture_id,
            "kernel_target": self.kernel_target,
            "probes_total": self.probes_total,
            "probes_passed": self.probes_passed,
            "tolerance_overshoot": self.tolerance_overshoot,
            "checkpoints_checked": list(self.checkpoints_checked),
        }
        if self.first_diverging_probe is not None:
            payload["first_diverging_probe"] = self.first_diverging_probe
        if self.error_detail is not None:
            payload["error_detail"] = self.error_detail
        return payload


def run_parity_probe(
    *,
    repo_root: Path,
    fixture_dir: Path,
    kernel_target: str,
    endpoint: str,
    model: str,
    api_key: str = "EMPTY",
    debug_export_dir: Path,
    request_timeout_s: float = 1800.0,
    quiet_timeout_s: float = 30.0,
    quiet_window_s: float = 0.5,
) -> ParityProbeResult:
    if kernel_target not in {"deltanet", "gatedattn"}:
        raise ValueError(f"unsupported kernel_target: {kernel_target!r}")

    fixture_yaml_path = fixture_dir / f"{kernel_target}_v1.yaml"
    if not fixture_yaml_path.is_file():
        raise FileNotFoundError(f"parity fixture YAML missing: {fixture_yaml_path}")
    fixture = yaml.safe_load(fixture_yaml_path.read_text(encoding="utf-8"))
    if not isinstance(fixture, dict):
        raise ValueError(f"parity fixture YAML is not a mapping: {fixture_yaml_path}")
    fixture_id = str(fixture.get("fixture_id", ""))
    tolerances = fixture.get("tolerances") or {}
    rtol_logit = float(tolerances.get("rtol_logit", 0.001))
    atol_logit = float(tolerances.get("atol_logit", 0.001))
    rtol_state = float(tolerances.get("rtol_state", 0.005))
    atol_state = float(tolerances.get("atol_state", 0.005))
    state_checkpoints: tuple[int, ...] = (
        tuple(int(t) for t in (fixture.get("state_checkpoints_at_token") or []))
        if kernel_target == "deltanet"
        else ()
    )

    probes_path = fixture_dir / "probes_input.jsonl"
    if not probes_path.is_file():
        raise FileNotFoundError(f"probes_input.jsonl missing: {probes_path}")
    probes = [json.loads(line) for line in probes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not probes:
        raise ValueError(f"probes_input.jsonl is empty: {probes_path}")

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("numpy is required for parity probe") from exc

    logits_npz_path = fixture_dir / f"{kernel_target}_reference_logits.npz"
    if not logits_npz_path.is_file():
        raise FileNotFoundError(f"reference logits NPZ missing: {logits_npz_path}")
    logits_ref = dict(np.load(logits_npz_path, allow_pickle=False))
    if "source_artifact_path_by_probe" not in logits_ref:
        raise ValueError(
            f"unsupported logits fixture schema (missing source_artifact_path_by_probe): {logits_npz_path}"
        )

    state_ref: dict[str, Any] | None = None
    if kernel_target == "deltanet":
        state_npz_path = fixture_dir / "deltanet_reference_state.npz"
        if not state_npz_path.is_file():
            raise FileNotFoundError(f"reference state NPZ missing: {state_npz_path}")
        state_ref = dict(np.load(state_npz_path, allow_pickle=False))
        for token in state_checkpoints:
            key = f"state_token_{token}_source_path_by_probe"
            if key not in state_ref:
                raise ValueError(f"unsupported state fixture schema (missing {key}): {state_npz_path}")

    debug_export_dir = Path(debug_export_dir).resolve()
    staging_dir = debug_export_dir / "staging"
    archive_root = debug_export_dir / "candidate"
    archive_root.mkdir(parents=True, exist_ok=True)

    per_probe: list[dict[str, Any]] = []
    overshoot = 0.0
    first_diverging: int | None = None
    fail_reason: str | None = None
    fail_detail: str | None = None

    for probe_index, probe in enumerate(probes):
        probe_index_int = int(probe.get("probe_index", probe_index))
        target_logit_token = int(logits_ref["source_generated_token_index_by_probe"][probe_index])
        require_state = bool(state_checkpoints)
        minimum_completion_tokens = (
            max(state_checkpoints) + 1 if require_state else None
        )

        _clear_staging(staging_dir)
        try:
            _post_completion(
                endpoint=endpoint,
                api_key=api_key,
                model=model,
                probe=probe,
                timeout_s=request_timeout_s,
                minimum_completion_tokens=minimum_completion_tokens,
            )
        except requests.RequestException as exc:
            return ParityProbeResult(
                pass_=False,
                fixture_id=fixture_id,
                kernel_target=kernel_target,
                probes_total=len(probes),
                probes_passed=probe_index,
                first_diverging_probe=probe_index_int,
                tolerance_overshoot=overshoot,
                reason="endpoint_unreachable",
                error_detail=f"{type(exc).__name__}: {exc}",
                per_probe=tuple(per_probe),
                checkpoints_checked=state_checkpoints,
            )

        try:
            archive_dir = archive_root / f"probe_{probe_index_int:06d}"
            artifact = _archive_probe_files(
                staging_dir=staging_dir,
                archive_dir=archive_dir,
                expected_state_tokens=state_checkpoints,
                require_state=require_state,
                quiet_timeout_s=quiet_timeout_s,
                quiet_window_s=quiet_window_s,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            return ParityProbeResult(
                pass_=False,
                fixture_id=fixture_id,
                kernel_target=kernel_target,
                probes_total=len(probes),
                probes_passed=probe_index,
                first_diverging_probe=probe_index_int,
                tolerance_overshoot=overshoot,
                reason="capture_failed",
                error_detail=str(exc),
                per_probe=tuple(per_probe),
                checkpoints_checked=state_checkpoints,
            )

        probe_record: dict[str, Any] = {"probe_index": probe_index_int}
        try:
            logit_outcome = _compare_logits(
                np_module=np,
                candidate_paths=artifact["logits"],
                target_token=target_logit_token,
                fixture_logits=logits_ref,
                fixture_index=probe_index,
                rtol=rtol_logit,
                atol=atol_logit,
            )
            probe_record["logits"] = logit_outcome
            overshoot = max(overshoot, float(logit_outcome["overshoot"]))
            if not logit_outcome["pass"]:
                fail_reason = "parity_logit_diverged"
                fail_detail = logit_outcome.get("detail")
                first_diverging = probe_index_int
                per_probe.append(probe_record)
                break

            if state_checkpoints:
                state_outcome = _compare_state(
                    np_module=np,
                    candidate_paths=artifact["state"],
                    state_tokens=state_checkpoints,
                    fixture_state=state_ref or {},
                    fixture_index=probe_index,
                    rtol=rtol_state,
                    atol=atol_state,
                )
                probe_record["state"] = state_outcome
                overshoot = max(overshoot, float(state_outcome["overshoot"]))
                if not state_outcome["pass"]:
                    fail_reason = "parity_state_diverged"
                    fail_detail = state_outcome.get("detail")
                    first_diverging = probe_index_int
                    per_probe.append(probe_record)
                    break
        except (FileNotFoundError, ValueError, KeyError) as exc:
            return ParityProbeResult(
                pass_=False,
                fixture_id=fixture_id,
                kernel_target=kernel_target,
                probes_total=len(probes),
                probes_passed=probe_index,
                first_diverging_probe=probe_index_int,
                tolerance_overshoot=overshoot,
                reason="comparison_failed",
                error_detail=f"{type(exc).__name__}: {exc}",
                per_probe=tuple(per_probe),
                checkpoints_checked=state_checkpoints,
            )

        per_probe.append(probe_record)

    if fail_reason is not None:
        return ParityProbeResult(
            pass_=False,
            fixture_id=fixture_id,
            kernel_target=kernel_target,
            probes_total=len(probes),
            probes_passed=len(per_probe) - 1,
            first_diverging_probe=first_diverging,
            tolerance_overshoot=overshoot,
            reason=fail_reason,
            error_detail=fail_detail,
            per_probe=tuple(per_probe),
            checkpoints_checked=state_checkpoints,
        )

    return ParityProbeResult(
        pass_=True,
        fixture_id=fixture_id,
        kernel_target=kernel_target,
        probes_total=len(probes),
        probes_passed=len(probes),
        first_diverging_probe=None,
        tolerance_overshoot=overshoot,
        reason="ran_passed",
        error_detail=None,
        per_probe=tuple(per_probe),
        checkpoints_checked=state_checkpoints,
    )


def _post_completion(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    probe: dict[str, Any],
    timeout_s: float,
    minimum_completion_tokens: int | None,
) -> dict[str, Any]:
    output_token_count = max(int(probe["output_token_count"]), int(minimum_completion_tokens or 0))
    payload: dict[str, Any] = {
        "model": model,
        "prompt": probe["prompt"],
        "max_tokens": output_token_count,
        "temperature": 0,
        "seed": 0,
    }
    if output_token_count >= 1024:
        payload["min_tokens"] = output_token_count
        payload["ignore_eos"] = True
    response = requests.post(
        f"{endpoint.rstrip('/')}/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=timeout_s,
    )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("completion response was not a JSON object")
    return body


def _clear_staging(staging_dir: Path) -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)
    for path in staging_dir.iterdir():
        if path.is_file():
            path.unlink()


def _wait_quiet(staging_dir: Path, *, timeout_s: float, quiet_window_s: float) -> None:
    deadline = time.time() + timeout_s
    previous: tuple[str, ...] | None = None
    quiet_started: float | None = None
    while time.time() < deadline:
        current = tuple(sorted(path.name for path in staging_dir.glob("*") if path.is_file()))
        if current == previous and current:
            quiet_started = quiet_started or time.time()
            if time.time() - quiet_started >= quiet_window_s:
                return
        else:
            previous = current
            quiet_started = None
        time.sleep(0.05)


def _archive_probe_files(
    *,
    staging_dir: Path,
    archive_dir: Path,
    expected_state_tokens: tuple[int, ...],
    require_state: bool,
    quiet_timeout_s: float,
    quiet_window_s: float,
) -> dict[str, list[Path]]:
    _wait_quiet(staging_dir, timeout_s=quiet_timeout_s, quiet_window_s=quiet_window_s)
    files = [path for path in staging_dir.glob("*.pt") if path.is_file()]
    if not files:
        raise FileNotFoundError(f"no debug .pt exports produced in {staging_dir}")
    archive_dir.mkdir(parents=True, exist_ok=True)
    logits: list[Path] = []
    state: list[Path] = []
    for path in sorted(files):
        match = DEBUG_EXPORT_RE.fullmatch(path.name)
        if match is None:
            continue
        target = archive_dir / path.name
        shutil.move(str(path), target)
        if match.group("kind") == "logits":
            logits.append(target)
        else:
            state.append(target)
    if not logits:
        raise RuntimeError(f"no logits debug exports produced in {staging_dir}")
    if require_state:
        observed_tokens = {
            int(load_debug_export_pt(path)["generated_token_index"]) for path in state
        }
        missing = [token for token in expected_state_tokens if token not in observed_tokens]
        if missing:
            raise RuntimeError(f"missing required DeltaNet state tokens: {missing}")
    return {"logits": logits, "state": state}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(np_module: Any, array: Any) -> str:
    contiguous = np_module.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(tuple(int(dim) for dim in contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _candidate_logits_at_token(np_module: Any, candidate_paths: list[Path], token: int) -> Any:
    for path in candidate_paths:
        payload = load_debug_export_pt(path)
        if payload.get("kind") != LOGITS_DEBUG_KIND:
            continue
        if int(payload.get("generated_token_index", -1)) != int(token):
            continue
        tensor = payload["logits"]
        if hasattr(tensor, "detach"):
            tensor = tensor.detach()
        if hasattr(tensor, "cpu"):
            tensor = tensor.cpu()
        if hasattr(tensor, "numpy"):
            # bf16/fp16/fp8 torch tensors must upcast to float32 before .numpy().
            if hasattr(tensor, "float"):
                tensor = tensor.float()
            array = tensor.numpy()
        else:
            array = np_module.asarray(tensor)
        return np_module.ascontiguousarray(array.astype(np_module.float32, copy=False)).reshape(-1)
    raise FileNotFoundError(f"no candidate logits export for token {token} in {candidate_paths}")


def _candidate_state_at_token(candidate_paths: list[Path], token: int) -> Any:
    for path in candidate_paths:
        payload = load_debug_export_pt(path)
        if payload.get("kind") != DELTANET_STATE_DEBUG_KIND:
            continue
        if int(payload.get("generated_token_index", -1)) != int(token):
            continue
        flat, _ = _flatten_state_payload(payload)
        return flat
    raise FileNotFoundError(f"no candidate state export for token {token} in {candidate_paths}")


def _compute_overshoot(np_module: Any, candidate: Any, reference: Any, *, rtol: float, atol: float) -> float:
    if candidate.shape != reference.shape:
        return float("inf")
    diff = np_module.abs(candidate.astype(np_module.float64) - reference.astype(np_module.float64))
    allowed = atol + rtol * np_module.abs(reference.astype(np_module.float64))
    excess = diff - allowed
    excess_max = float(np_module.max(excess)) if excess.size else 0.0
    return max(0.0, excess_max)


def _compare_logits(
    *,
    np_module: Any,
    candidate_paths: list[Path],
    target_token: int,
    fixture_logits: dict[str, Any],
    fixture_index: int,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    candidate_array = _candidate_logits_at_token(np_module, candidate_paths, target_token)
    candidate_sha = _array_sha256(np_module, candidate_array)
    fixture_sha = str(fixture_logits["source_logits_sha256_by_probe"][fixture_index])
    if candidate_sha == fixture_sha:
        return {
            "pass": True,
            "exact_match": True,
            "overshoot": 0.0,
            "candidate_sha256": candidate_sha,
            "reference_sha256": fixture_sha,
            "target_token": int(target_token),
        }

    source_path = Path(str(fixture_logits["source_artifact_path_by_probe"][fixture_index]))
    if not source_path.is_absolute():
        source_path = Path.cwd() / source_path
    if not source_path.is_file():
        return {
            "pass": False,
            "exact_match": False,
            "overshoot": float("inf"),
            "candidate_sha256": candidate_sha,
            "reference_sha256": fixture_sha,
            "target_token": int(target_token),
            "detail": f"reference_source_missing:{source_path}",
        }
    payload = load_debug_export_pt(source_path)
    tensor = payload["logits"]
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        # bf16/fp16/fp8 torch tensors must upcast to float32 before .numpy().
        if hasattr(tensor, "float"):
            tensor = tensor.float()
        ref_array = tensor.numpy()
    else:
        ref_array = np_module.asarray(tensor)
    ref_array = np_module.ascontiguousarray(ref_array.astype(np_module.float32, copy=False)).reshape(-1)
    overshoot = _compute_overshoot(np_module, candidate_array, ref_array, rtol=rtol, atol=atol)
    passed = overshoot == 0.0 and candidate_array.shape == ref_array.shape
    detail: str | None = None
    if not passed:
        if candidate_array.shape != ref_array.shape:
            detail = f"shape_mismatch:candidate={candidate_array.shape},reference={ref_array.shape}"
        else:
            detail = f"overshoot={overshoot:.6e}"
    return {
        "pass": passed,
        "exact_match": False,
        "overshoot": overshoot,
        "candidate_sha256": candidate_sha,
        "reference_sha256": fixture_sha,
        "target_token": int(target_token),
        "detail": detail,
    }


def _compare_state(
    *,
    np_module: Any,
    candidate_paths: list[Path],
    state_tokens: tuple[int, ...],
    fixture_state: dict[str, Any],
    fixture_index: int,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    per_token: list[dict[str, Any]] = []
    aggregate_overshoot = 0.0
    for token in state_tokens:
        candidate_array = _candidate_state_at_token(candidate_paths, token)
        candidate_sha = _array_sha256(np_module, candidate_array)
        ref_file_sha = str(fixture_state[f"state_token_{token}"][fixture_index])
        candidate_file_sha: str | None = None
        for path in candidate_paths:
            payload = load_debug_export_pt(path)
            if (
                payload.get("kind") == DELTANET_STATE_DEBUG_KIND
                and int(payload.get("generated_token_index", -1)) == int(token)
            ):
                candidate_file_sha = _file_sha256(path)
                break
        record: dict[str, Any] = {
            "token": int(token),
            "candidate_array_sha256": candidate_sha,
            "candidate_file_sha256": candidate_file_sha,
            "reference_file_sha256": ref_file_sha,
        }
        if candidate_file_sha == ref_file_sha:
            record.update({"pass": True, "exact_match": True, "overshoot": 0.0})
            per_token.append(record)
            continue

        source_path = Path(str(fixture_state[f"state_token_{token}_source_path_by_probe"][fixture_index]))
        if not source_path.is_absolute():
            source_path = Path.cwd() / source_path
        if not source_path.is_file():
            record.update(
                {
                    "pass": False,
                    "exact_match": False,
                    "overshoot": float("inf"),
                    "detail": f"reference_source_missing:{source_path}",
                }
            )
            per_token.append(record)
            return {
                "pass": False,
                "overshoot": float("inf"),
                "tokens": per_token,
                "detail": record["detail"],
            }
        ref_payload = load_debug_export_pt(source_path)
        ref_array, _ = _flatten_state_payload(ref_payload)
        overshoot = _compute_overshoot(
            np_module, candidate_array, ref_array, rtol=rtol, atol=atol
        )
        token_passed = overshoot == 0.0 and candidate_array.shape == ref_array.shape
        record.update(
            {
                "pass": token_passed,
                "exact_match": False,
                "overshoot": overshoot,
                "detail": None
                if token_passed
                else (
                    f"shape_mismatch:candidate={candidate_array.shape},reference={ref_array.shape}"
                    if candidate_array.shape != ref_array.shape
                    else f"overshoot={overshoot:.6e}"
                ),
            }
        )
        aggregate_overshoot = max(aggregate_overshoot, float(overshoot))
        per_token.append(record)
        if not token_passed:
            return {
                "pass": False,
                "overshoot": aggregate_overshoot,
                "tokens": per_token,
                "detail": record["detail"],
            }
    return {
        "pass": True,
        "overshoot": aggregate_overshoot,
        "tokens": per_token,
        "detail": None,
    }
