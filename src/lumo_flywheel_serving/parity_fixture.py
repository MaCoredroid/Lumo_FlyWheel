from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
import yaml

REFERENCE_BASELINE = {
    "attention_backend": "flash-attn-4",
    "deltanet_kernel": "triton-chunked-delta-v2",
    "fp8_gemm_kernel": "cublas",
    "torch_compile_mode": "default",
    "cuda_graph_capture": "off",
}
REFERENCE_REPRODUCIBILITY_RUNS = 3
HEAVY_FAMILY_ID = "responses-sdk-adapter-cutover"
SIBLING_FAMILY_IDS = (
    "codex-provider-rollover",
    "codex-skill-runtime-v2-split",
    "esm-plugin-loader-modernization",
    "nightly-regression-watch",
    "objective-driven-repo-improvement",
    "policy-aware-request-resolution",
    "release-manifest-v2-modernization",
    "sqlalchemy-2-session-modernization",
)
P2B_FAMILY_PROBE_COUNTS = {HEAVY_FAMILY_ID: 64, **{family_id: 16 for family_id in SIBLING_FAMILY_IDS}}
KERNEL_TARGETS = ("deltanet", "gatedattn")
REFERENCED_KEYS = ("probe_input_ref", "reference_logits_ref", "reference_state_snapshots_ref")
DEFAULT_WEIGHT_VERSION_ID = "2e1b21350ce589fcaafbb3c7d7eac526a7aed582"
SYNTHETIC_TEST_ARTIFACT_PURPOSE = "test_only_synthetic_placeholder"
LOGITS_DEBUG_KIND = "full_vocab_logits_before_sampling"
DELTANET_STATE_DEBUG_KIND = "qwen35_mamba_deltanet_recurrent_state"


@dataclass(frozen=True)
class FixtureValidation:
    path: str
    exists: bool
    errors: tuple[str, ...] = ()
    content_hash: str | None = None

    @property
    def pass_(self) -> bool:
        return self.exists and not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "exists": self.exists,
            "pass": self.pass_,
            "errors": list(self.errors),
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class DebugProbeArtifacts:
    probe_index: int
    request_id: str
    logits_paths: tuple[Path, ...]
    state_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class DebugAggregationResult:
    logits_path: str
    state_path: str | None
    shapes: dict[str, Any]


def _repo_relative(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def family_fixture_dir(repo_root: Path, family_id: str) -> Path:
    return repo_root / "benchmark_blueprints" / "families" / family_id / "parity_fixture"


def fixture_yaml_path(repo_root: Path, family_id: str, kernel_target: str) -> Path:
    return family_fixture_dir(repo_root, family_id) / f"{kernel_target}_v1.yaml"


def fixture_content_hash(fixture_yaml_path: str | Path) -> str:
    """Canonical manifest hash over a parity fixture yaml and every referenced blob."""
    fixture_path = Path(fixture_yaml_path)
    fixture = yaml.safe_load(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(fixture, dict):
        raise ValueError(f"fixture must be a mapping: {fixture_path}")
    base_dir = fixture_path.parent
    digest = hashlib.sha256()
    digest.update(fixture_path.read_bytes())
    for key in sorted(REFERENCED_KEYS):
        if key in fixture and fixture[key]:
            ref_path = base_dir / str(fixture[key])
            if not ref_path.exists():
                raise FileNotFoundError(
                    f"fixture {fixture_path} references {key}={fixture[key]} which does not exist"
                )
            digest.update(b"\x00" + key.encode("ascii") + b"\x00")
            digest.update(ref_path.read_bytes())
    return digest.hexdigest()


def deterministic_probe_rows(repo_root: Path, family_id: str, probe_count: int) -> list[dict[str, Any]]:
    trace_path = repo_root / "benchmark_blueprints" / "families" / family_id / "seed_trace_v5.jsonl"
    if not trace_path.is_file():
        raise FileNotFoundError(f"seed trace missing for {family_id}: {trace_path}")
    seed_rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(trace_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{trace_path}:{line_number}: expected JSON object")
        seed_rows.append(row)
    if not seed_rows:
        raise ValueError(f"seed trace is empty for {family_id}: {trace_path}")

    probes: list[dict[str, Any]] = []
    for index in range(probe_count):
        source = seed_rows[index % len(seed_rows)]
        label = str(source.get("capture_prompt_label", f"turn_{source.get('turn_index', index)}"))
        output_tokens = int(source.get("request_max_output_tokens") or source.get("output_tokens") or 1)
        prompt_tokens = int(source.get("prompt_tokens") or 0)
        prompt = (
            f"Parity fixture probe {index:03d} for family {family_id}. "
            f"Source label: {label}. Approximate prompt tokens: {prompt_tokens}."
        )
        probes.append(
            {
                "probe_index": index,
                "family_id": family_id,
                "source_turn_index": source.get("turn_index"),
                "capture_prompt_label": label,
                "prompt_token_count_hint": prompt_tokens,
                "prompt": prompt,
                "output_token_count": output_tokens,
            }
        )
    return probes


def probe_token_lengths(kernel_target: str) -> list[int]:
    if kernel_target == "deltanet":
        return [16, 64, 256, 1024, 4096]
    if kernel_target == "gatedattn":
        return [16, 64, 256, 1024]
    raise ValueError(f"unsupported kernel_target: {kernel_target}")


def fixture_payload(
    *,
    family_id: str,
    kernel_target: str,
    probe_count: int,
    weight_version_id: str,
    vllm_version: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "fixture_id": f"{family_id}-{kernel_target}-v1",
        "generated_at": generated_at or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "generated_against": {
            "vllm_version": vllm_version,
            "weight_version_id": weight_version_id,
            "reference_baseline": dict(REFERENCE_BASELINE),
            "reference_reproducibility_runs": REFERENCE_REPRODUCIBILITY_RUNS,
        },
        "probe_count": probe_count,
        "probe_token_lengths": probe_token_lengths(kernel_target),
        "probe_input_ref": "probes_input.jsonl",
        "reference_logits_ref": f"{kernel_target}_reference_logits.npz",
        "tolerances": {
            "rtol_logit": 1.0e-3,
            "atol_logit": 1.0e-3,
        },
        "parity_check_method": "per_token_logit_compare",
    }
    if kernel_target == "deltanet":
        payload["reference_state_snapshots_ref"] = "deltanet_reference_state.npz"
        payload["state_checkpoints_at_token"] = [1, 1024]
        payload["tolerances"]["rtol_state"] = 5.0e-3
        payload["tolerances"]["atol_state"] = 5.0e-3
        payload["parity_check_method"] = "logit_plus_state_compare"
    return payload


def load_debug_export_pt(path: str | Path) -> dict[str, Any]:
    """Load one repo-owned vLLM debug export tensor file."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("torch is required to load vLLM debug export .pt artifacts") from exc

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"debug export is not a mapping: {path}")
    if payload.get("kind") not in {LOGITS_DEBUG_KIND, DELTANET_STATE_DEBUG_KIND}:
        raise ValueError(f"unsupported debug export kind in {path}: {payload.get('kind')!r}")
    return payload


def summarize_debug_export_pt(path: str | Path) -> dict[str, Any]:
    payload = load_debug_export_pt(path)
    return {
        "path": str(path),
        "kind": payload.get("kind"),
        "request_id": payload.get("request_id"),
        "generated_token_index": payload.get("generated_token_index"),
        "saved_shape": payload.get("saved_shape"),
        "source_shape": payload.get("source_shape"),
        "logits_is_truncated": payload.get("logits_is_truncated"),
    }


def write_debug_export_npz_companions(
    *,
    fixture_dir: str | Path,
    kernel_target: str,
    probe_artifacts: list[DebugProbeArtifacts],
    expected_probe_count: int,
    require_state_tokens: tuple[int, ...] = (1, 1024),
) -> DebugAggregationResult:
    """Aggregate real vLLM debug-export .pt files into fixture companion .npz files.

    This intentionally has no synthetic fallback. Missing or malformed debug exports
    raise before any production companion can be written.
    """
    if kernel_target not in KERNEL_TARGETS:
        raise ValueError(f"unsupported kernel_target: {kernel_target}")
    if len(probe_artifacts) != expected_probe_count:
        raise ValueError(
            f"expected {expected_probe_count} probe debug captures, got {len(probe_artifacts)}"
        )
    if sorted(artifact.probe_index for artifact in probe_artifacts) != list(range(expected_probe_count)):
        raise ValueError("probe debug captures must cover contiguous probe_index values")

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("numpy is required to aggregate parity fixture debug exports") from exc

    output_dir = Path(fixture_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_artifacts = sorted(probe_artifacts, key=lambda artifact: artifact.probe_index)

    logits_payload: dict[str, Any] = {
        "probe_index": np.array([artifact.probe_index for artifact in sorted_artifacts], dtype=np.int32),
        "request_id": np.array([artifact.request_id for artifact in sorted_artifacts]),
    }
    shapes: dict[str, Any] = {"logits": {}, "state": {}}
    for artifact in sorted_artifacts:
        logits_by_token = _load_logits_by_token(artifact)
        token_indices = sorted(logits_by_token)
        logits_arrays = [logits_by_token[token] for token in token_indices]
        member_prefix = f"probe_{artifact.probe_index:06d}"
        if logits_arrays:
            logits_payload[f"{member_prefix}_logits"] = np.stack(logits_arrays, axis=0)
            logits_payload[f"{member_prefix}_logit_tokens"] = np.array(token_indices, dtype=np.int32)
            shapes["logits"][member_prefix] = tuple(int(dim) for dim in logits_payload[f"{member_prefix}_logits"].shape)
        else:
            raise ValueError(f"probe {artifact.probe_index} has no logits debug exports")
    logits_payload["logits_member_names"] = np.array(
        [f"probe_{artifact.probe_index:06d}_logits" for artifact in sorted_artifacts]
    )
    logits_path = output_dir / f"{kernel_target}_reference_logits.npz"
    np.savez(logits_path, **logits_payload)

    state_path: Path | None = None
    if kernel_target == "deltanet":
        state_payload: dict[str, Any] = {
            "probe_index": logits_payload["probe_index"],
            "request_id": logits_payload["request_id"],
            "state_storage": np.array(["sha256_float32_flattened_with_sample"]),
        }
        for token in require_state_tokens:
            digests: list[str] = []
            lengths: list[int] = []
            samples: list[Any] = []
            metadata: list[str] = []
            for artifact in sorted_artifacts:
                state_by_token = _load_state_by_token(artifact)
                if token not in state_by_token:
                    raise ValueError(f"probe {artifact.probe_index} missing DeltaNet state token {token}")
                vector, meta = state_by_token[token]
                digests.append(_array_sha256(vector))
                lengths.append(int(vector.size))
                samples.append(vector[:16].astype(np.float32, copy=False))
                metadata.append(json.dumps(meta, sort_keys=True))
            state_payload[f"state_token_{token}"] = np.array(digests)
            state_payload[f"state_token_{token}_lengths"] = np.array(lengths, dtype=np.int64)
            state_payload[f"state_token_{token}_sample_first_16_float32"] = np.stack(samples, axis=0)
            state_payload[f"state_token_{token}_metadata_json"] = np.array(metadata)
            shapes["state"][f"state_token_{token}"] = (len(digests),)
        state_path = output_dir / "deltanet_reference_state.npz"
        np.savez(state_path, **state_payload)

    return DebugAggregationResult(
        logits_path=str(logits_path),
        state_path=str(state_path) if state_path is not None else None,
        shapes=shapes,
    )


def assert_debug_capture_runs_reproduce(
    *,
    runs: list[list[DebugProbeArtifacts]],
    kernel_target: str,
    require_state_tokens: tuple[int, ...] = (1, 1024),
) -> None:
    if len(runs) != REFERENCE_REPRODUCIBILITY_RUNS:
        raise ValueError(f"expected {REFERENCE_REPRODUCIBILITY_RUNS} reproducibility runs, got {len(runs)}")
    if not runs:
        raise ValueError("no debug capture runs supplied")
    baseline = _capture_signature(runs[0], kernel_target=kernel_target, require_state_tokens=require_state_tokens)
    for run_index, run in enumerate(runs[1:], start=2):
        current = _capture_signature(run, kernel_target=kernel_target, require_state_tokens=require_state_tokens)
        if current != baseline:
            raise ValueError(f"fixture_reference_nondeterministic: run {run_index} differs from run 1")


def _capture_signature(
    artifacts: list[DebugProbeArtifacts],
    *,
    kernel_target: str,
    require_state_tokens: tuple[int, ...],
) -> list[tuple[Any, ...]]:
    signature: list[tuple[Any, ...]] = []
    for artifact in sorted(artifacts, key=lambda item: item.probe_index):
        logits_by_token = _load_logits_by_token(artifact)
        logits_sig = tuple(
            (token, _array_sha256(array))
            for token, array in sorted(logits_by_token.items())
        )
        state_sig: tuple[Any, ...] = ()
        if kernel_target == "deltanet":
            state_by_token = _load_state_by_token(artifact)
            state_sig = tuple(
                (token, _array_sha256(state_by_token[token][0]))
                for token in require_state_tokens
            )
        signature.append((artifact.probe_index, logits_sig, state_sig))
    return signature


def _load_logits_by_token(artifact: DebugProbeArtifacts) -> dict[int, Any]:
    logits_by_token: dict[int, Any] = {}
    for path in sorted(artifact.logits_paths):
        payload = load_debug_export_pt(path)
        if payload.get("kind") != LOGITS_DEBUG_KIND:
            raise ValueError(f"expected logits debug export, got {payload.get('kind')!r}: {path}")
        if payload.get("request_id") != artifact.request_id:
            raise ValueError(f"request_id mismatch for {path}")
        token = int(payload["generated_token_index"])
        if token in logits_by_token:
            raise ValueError(f"duplicate logits token {token} for probe {artifact.probe_index}")
        logits_by_token[token] = _tensor_payload_to_numpy(payload["logits"])
    return logits_by_token


def _load_state_by_token(artifact: DebugProbeArtifacts) -> dict[int, tuple[Any, dict[str, Any]]]:
    state_by_token: dict[int, tuple[Any, dict[str, Any]]] = {}
    for path in sorted(artifact.state_paths):
        payload = load_debug_export_pt(path)
        if payload.get("kind") != DELTANET_STATE_DEBUG_KIND:
            raise ValueError(f"expected DeltaNet state debug export, got {payload.get('kind')!r}: {path}")
        if payload.get("request_id") != artifact.request_id:
            raise ValueError(f"request_id mismatch for {path}")
        token = int(payload["generated_token_index"])
        if token in state_by_token:
            raise ValueError(f"duplicate state token {token} for probe {artifact.probe_index}")
        state_by_token[token] = _flatten_state_payload(payload)
    return state_by_token


def _tensor_payload_to_numpy(value: Any) -> Any:
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def _flatten_state_payload(payload: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    pieces: list[Any] = []
    entries: list[dict[str, Any]] = []
    layers = payload.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError(f"state payload has no layers for request {payload.get('request_id')}")
    for layer_name in sorted(layers):
        layer_entries = layers[layer_name]
        if not isinstance(layer_entries, list):
            raise ValueError(f"state layer payload is not a list: {layer_name}")
        for entry in sorted(layer_entries, key=lambda item: int(item.get("state_index", 0))):
            array = _tensor_payload_to_numpy(entry["tensor"]).reshape(-1)
            pieces.append(array)
            entries.append(
                {
                    "layer_name": layer_name,
                    "state_index": int(entry.get("state_index", -1)),
                    "state_role": entry.get("state_role"),
                    "shape": list(array.shape),
                    "saved_shape": list(entry.get("saved_shape", ())),
                    "saved_dtype": entry.get("saved_dtype"),
                }
            )
    if not pieces:
        raise ValueError(f"state payload has no tensors for request {payload.get('request_id')}")
    return np.concatenate(pieces).astype(np.float32, copy=False), {"entries": entries}


def _pad_float_vectors(vectors: list[Any]) -> tuple[Any, Any]:
    import numpy as np

    lengths = np.array([int(vector.size) for vector in vectors], dtype=np.int32)
    width = int(lengths.max()) if len(lengths) else 0
    padded = np.full((len(vectors), width), np.nan, dtype=np.float32)
    for row, vector in enumerate(vectors):
        padded[row, : vector.size] = vector.astype(np.float32, copy=False)
    return padded, lengths


def _array_sha256(array: Any) -> str:
    import numpy as np

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(tuple(int(dim) for dim in contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def validate_fixture(
    fixture_path: str | Path,
    *,
    repo_root: str | Path,
    expected_family_id: str,
    expected_kernel_target: str,
    expected_probe_count: int,
    expected_weight_version_id: str | None = None,
) -> FixtureValidation:
    path = Path(fixture_path)
    root = Path(repo_root)
    relative = _repo_relative(root, path)
    if not path.is_file():
        return FixtureValidation(path=relative, exists=False, errors=("fixture_yaml_missing",))
    errors: list[str] = []
    try:
        fixture = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return FixtureValidation(path=relative, exists=True, errors=(f"fixture_yaml_invalid:{exc}",))
    if not isinstance(fixture, dict):
        return FixtureValidation(path=relative, exists=True, errors=("fixture_yaml_not_mapping",))

    if fixture.get("artifact_purpose") == SYNTHETIC_TEST_ARTIFACT_PURPOSE:
        errors.append("production_fixture_declares_synthetic_test_placeholder")

    expected_fixture_id = f"{expected_family_id}-{expected_kernel_target}-v1"
    if fixture.get("fixture_id") != expected_fixture_id:
        errors.append("fixture_id_mismatch")
    if fixture.get("probe_count") != expected_probe_count:
        errors.append("probe_count_mismatch")
    generated_against = fixture.get("generated_against")
    if not isinstance(generated_against, dict):
        errors.append("generated_against_missing")
    else:
        if generated_against.get("reference_baseline") != REFERENCE_BASELINE:
            errors.append("reference_baseline_mismatch")
        if generated_against.get("reference_reproducibility_runs") != REFERENCE_REPRODUCIBILITY_RUNS:
            errors.append("reference_reproducibility_runs_mismatch")
        if expected_weight_version_id is not None and generated_against.get("weight_version_id") != expected_weight_version_id:
            errors.append("weight_version_id_mismatch")

    if fixture.get("probe_input_ref") != "probes_input.jsonl":
        errors.append("probe_input_ref_mismatch")
    if fixture.get("reference_logits_ref") != f"{expected_kernel_target}_reference_logits.npz":
        errors.append("reference_logits_ref_mismatch")
    if expected_kernel_target == "deltanet":
        if fixture.get("reference_state_snapshots_ref") != "deltanet_reference_state.npz":
            errors.append("reference_state_snapshots_ref_mismatch")
        if fixture.get("state_checkpoints_at_token") != [1, 1024]:
            errors.append("state_checkpoints_mismatch")
        if fixture.get("parity_check_method") != "logit_plus_state_compare":
            errors.append("parity_check_method_mismatch")
    else:
        if "reference_state_snapshots_ref" in fixture:
            errors.append("unexpected_state_snapshots_ref")
        if fixture.get("parity_check_method") != "per_token_logit_compare":
            errors.append("parity_check_method_mismatch")

    probes_path = path.parent / str(fixture.get("probe_input_ref", ""))
    if not probes_path.is_file():
        errors.append("probe_input_blob_missing")
    else:
        probe_rows = [line for line in probes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(probe_rows) != expected_probe_count:
            errors.append("probe_input_count_mismatch")

    logits_ref = fixture.get("reference_logits_ref")
    if isinstance(logits_ref, str):
        errors.extend(_validate_npz_reference_blob(path.parent / logits_ref, required_members=("probe_index",)))
    if expected_kernel_target == "deltanet":
        state_ref = fixture.get("reference_state_snapshots_ref")
        if isinstance(state_ref, str):
            errors.extend(
                _validate_npz_reference_blob(
                    path.parent / state_ref,
                    required_members=("state_token_1", "state_token_1024", "probe_index"),
                )
            )

    content_hash: str | None = None
    try:
        content_hash = fixture_content_hash(path)
    except FileNotFoundError as exc:
        errors.append(f"referenced_blob_missing:{exc}")
    except ValueError as exc:
        errors.append(f"fixture_content_hash_invalid:{exc}")
    return FixtureValidation(path=relative, exists=True, errors=tuple(errors), content_hash=content_hash)


def _validate_npz_reference_blob(path: Path, *, required_members: tuple[str, ...]) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return errors
    if path.suffix != ".npz":
        errors.append(f"reference_blob_not_npz:{path.name}")
        return errors
    if path.read_bytes()[:4] != b"PK\x03\x04":
        errors.append(f"reference_blob_not_zip_npz:{path.name}")
        return errors
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
    except zipfile.BadZipFile:
        errors.append(f"reference_blob_bad_npz:{path.name}")
        return errors

    stems = {Path(name).stem for name in names}
    missing = [member for member in required_members if member not in stems]
    if missing:
        errors.append(f"reference_blob_missing_members:{path.name}:{','.join(missing)}")
    if "synthetic_test_placeholder" in stems:
        errors.append(f"reference_blob_contains_synthetic_test_placeholder:{path.name}")
    return errors


def validate_p2b_fixture_set(
    repo_root: str | Path,
    *,
    expected_weight_version_id: str,
) -> dict[str, Any]:
    root = Path(repo_root)
    validations: dict[str, dict[str, Any]] = {}
    all_errors: list[str] = []
    for family_id, probe_count in P2B_FAMILY_PROBE_COUNTS.items():
        for kernel_target in KERNEL_TARGETS:
            path = fixture_yaml_path(root, family_id, kernel_target)
            validation = validate_fixture(
                path,
                repo_root=root,
                expected_family_id=family_id,
                expected_kernel_target=kernel_target,
                expected_probe_count=probe_count,
                expected_weight_version_id=expected_weight_version_id,
            )
            key = f"{family_id}/{kernel_target}"
            validations[key] = validation.as_dict()
            all_errors.extend(f"{key}:{error}" for error in validation.errors)
    return {"pass": not all_errors, "fixtures": validations, "errors": all_errors}


def fetch_endpoint_capabilities(endpoint: str, *, api_key: str, model: str, timeout: float = 30.0) -> dict[str, Any]:
    base = endpoint.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"}
    capabilities: dict[str, Any] = {
        "endpoint": base,
        "model": model,
        "health_ok": False,
        "models_ok": False,
        "vllm_version": "unknown",
        "full_logits_available": False,
        "deltanet_state_snapshots_available": False,
        "openai_logprobs_full_vocab_available": False,
        "dev_collective_rpc_available": False,
    }
    server_base = base.removesuffix("/v1")
    health = requests.get(f"{server_base}/health", headers=headers, timeout=timeout)
    capabilities["health_status_code"] = health.status_code
    capabilities["health_ok"] = health.status_code == 200

    models = requests.get(f"{base}/models", headers=headers, timeout=timeout)
    capabilities["models_status_code"] = models.status_code
    if models.status_code == 200:
        payload = models.json()
        served = [item.get("id") for item in payload.get("data", []) if isinstance(item, dict)]
        capabilities["served_models"] = served
        capabilities["models_ok"] = model in served

    version = requests.get(f"{server_base}/version", headers=headers, timeout=timeout)
    if version.status_code == 200:
        version_payload = version.json()
        if isinstance(version_payload, dict):
            capabilities["vllm_version"] = str(version_payload.get("version", "unknown"))

    server_info = requests.get(f"{server_base}/server_info?config_format=json", headers=headers, timeout=timeout)
    capabilities["server_info_status_code"] = server_info.status_code
    if server_info.status_code == 200:
        server_payload = server_info.json()
        if isinstance(server_payload, dict):
            vllm_config = server_payload.get("vllm_config")
            if isinstance(vllm_config, dict):
                model_config = vllm_config.get("model_config")
                if isinstance(model_config, dict):
                    capabilities["max_logprobs"] = model_config.get("max_logprobs")
                    capabilities["logprobs_mode"] = model_config.get("logprobs_mode")

    openapi = requests.get(f"{server_base}/openapi.json", headers=headers, timeout=timeout)
    capabilities["openapi_status_code"] = openapi.status_code
    if openapi.status_code == 200:
        openapi_payload = openapi.json()
        paths = openapi_payload.get("paths", {}) if isinstance(openapi_payload, dict) else {}
        if isinstance(paths, dict):
            capabilities["dev_collective_rpc_available"] = "/collective_rpc" in paths

    probe_payload = {
        "model": model,
        "prompt": "P2b logit introspection capability probe.",
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": 100000,
    }
    logprobs_probe = requests.post(f"{base}/completions", headers=headers, json=probe_payload, timeout=timeout)
    capabilities["full_vocab_logprobs_probe_status_code"] = logprobs_probe.status_code
    if logprobs_probe.status_code == 200:
        capabilities["openai_logprobs_full_vocab_available"] = True
    else:
        try:
            capabilities["full_vocab_logprobs_probe_error"] = logprobs_probe.json().get("error", logprobs_probe.text)
        except ValueError:
            capabilities["full_vocab_logprobs_probe_error"] = logprobs_probe.text

    capabilities["missing_repo_supported_hooks"] = (
        "no OpenAI-compatible route returns full-vocabulary logits or raw logits",
        "no route returns DeltaNet recurrent state snapshots at generated-token checkpoints",
        "the vLLM dev collective_rpc endpoint is control-plane only here and does not provide a repo-owned "
        "data-plane export for GPUModelRunner logits plus GatedDeltaNetAttention ssm_state",
    )
    capabilities["blocking_reason"] = (
        "HLD section 2.2 requires full per-token reference logits and DeltaNet recurrent state snapshots; "
        "the live server exposes only capped OpenAI logprobs and no state-snapshot export hook"
    )
    return capabilities


def p2b_blocked_payload(
    *,
    family_id: str,
    probe_count: int,
    weight_version_id: str,
    capabilities: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "BLOCKED_NEEDS_USER_HELP",
        "halt_reason": "missing_real_kernel_logit_state_introspection",
        "family_id": family_id,
        "probe_count": probe_count,
        "weight_version_id": weight_version_id,
        "capabilities": capabilities,
        "required_by_hld": {
            "gatedattn": "full per-token reference logits for every probe across 3 bit-identical runs",
            "deltanet": "full per-token reference logits plus recurrent state snapshots at tokens [1, 1024]",
        },
        "required_vllm_or_model_change": (
            "Add a repo-supported debug export hook in the vLLM model runner that writes raw logits before sampling "
            "and the Qwen3.5 GatedDeltaNetAttention recurrent ssm_state after generated tokens 1 and 1024, keyed by "
            "request/probe id, without synthetic data or OpenAI logprob truncation."
        ),
        "files_written": [],
    }
