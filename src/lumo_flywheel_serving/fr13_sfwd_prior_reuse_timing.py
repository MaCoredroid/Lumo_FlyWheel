"""Source-bound candidate-serving control for SFWD prior-reuse timing."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import torch

from lumo_flywheel_serving import fr13_sfwd_prior_reuse as candidate
from lumo_flywheel_serving import (
    fr13_sfwd_prior_reuse_descriptorless as candidate_kernel,
)
from lumo_flywheel_serving.fr10_gdn_tree_kernel import (
    _FR13_FIXED32_MODE,
    _FR13_FIXED32_MODES,
)


CANDIDATE = candidate.CANDIDATE
DRAFT_VOCAB_K = 65536
DRAFT_VOCAB_ROOT = 1
DRAFT_VOCAB_BLOCKS_SHA256 = (
    "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
)
QUALIFIED_CANDIDATE_SOURCE_SHA256 = (
    "42fc6ae355a268cb33b454d02914862b2af7fb6b665d808d8899533992750623"
)
QUALIFIED_CANDIDATE_KERNEL_SOURCE_SHA256 = (
    "ff36101628cc15ead6fef6a7d17c2eb6decbc910c110c635b96d059fea1c1203"
)
QUALIFIED_REDUCED_GATE_SHA256 = (
    "46c7556b26356b0d53d83b5d6143816f0c04de46d142d2225ce0c497bc4dcfa4"
)
QUALIFIED_SOURCE_COMMIT = "7c9fda4bc643176f43404ddd4d633789fc46ef23"
TASK_MARKER_SHA256 = (
    "04fe7f61a0e0bbd48bf28127385c481b85550b291535f3705511494ba24c8463"
)
TIMING_ARM = "/logs/fr13_fixed32_sfwd_prior_reuse.timing.arm"
TIMING_GATE = "/logs/fr13_fixed32_sfwd_prior_reuse.timing_gate.json"
TIMING_ENGAGEMENT = "/logs/fr13_fixed32_sfwd_prior_reuse.timing_engagement.json"
REAL_EVENT_PATH = "/logs/fr13_fixed32_sfwd_state_fusion.real_event.arm"
_CREDENTIAL_IDS: set[int] = set()
_STATE = {
    "gate_sha256": None,
    "candidate_source_sha256": None,
    "candidate_kernel_source_sha256": None,
    "task_marker_sha256": None,
    "layers": set(),
    "launches": 0,
    "emitted": False,
}


def _regular_ascii(path: str, label: str, *, limit: int) -> bytes:
    try:
        info = os.lstat(path)
    except OSError as error:
        raise RuntimeError(f"{label} cannot be inspected: {path}: {error}") from error
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
    ):
        raise RuntimeError(f"{label} must be one regular non-symlink file: {path}")
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise RuntimeError(f"{label} cannot be read: {path}: {error}") from error
    if not raw or len(raw) > limit:
        raise RuntimeError(f"{label} is empty or exceeds {limit} bytes: {path}")
    try:
        raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"{label} must be ASCII: {path}") from error
    return raw


def _qualified_module_sha256(module, expected: str, label: str) -> str:
    try:
        path = Path(module.__file__)
        info = path.lstat()
        raw = path.read_bytes()
    except (OSError, TypeError) as error:
        raise RuntimeError(f"{label} cannot be hashed: {error}") from error
    if (
        path.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
    ):
        raise RuntimeError(f"{label} is not regular")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected:
        raise RuntimeError(f"{label} drifted")
    return digest


def _candidate_source_sha256() -> str:
    return _qualified_module_sha256(
        candidate,
        QUALIFIED_CANDIDATE_SOURCE_SHA256,
        "FR13 SFWD packed x-gather launcher source",
    )


def _candidate_kernel_source_sha256() -> str:
    return _qualified_module_sha256(
        candidate_kernel,
        QUALIFIED_CANDIDATE_KERNEL_SOURCE_SHA256,
        "FR13 SFWD packed x-gather kernel source",
    )


def _strict_json(raw: bytes, label: str) -> dict[str, object]:
    def reject_duplicates(pairs):
        payload = {}
        for key, value in pairs:
            if key in payload:
                raise RuntimeError(f"{label} has duplicate JSON key: {key}")
            payload[key] = value
        return payload

    def reject_nonfinite(value: str):
        raise RuntimeError(f"{label} has non-finite JSON value: {value}")

    try:
        payload = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not strict ASCII JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return payload


def _validate_gate(payload: dict[str, object]) -> None:
    required = {
        "schema": "fr13.fixed32.sfwd_prior_reuse.reduced_b1_byte_pass.v1",
        "status": "pass_source_only",
        "run_classification": "one_real_swe_verified_k64_root_b1_byte_diagnostic",
        "candidate": CANDIDATE,
        "source_commit": QUALIFIED_SOURCE_COMMIT,
        "candidate_source_sha256": QUALIFIED_CANDIDATE_SOURCE_SHA256,
        "candidate_kernel_source_sha256": (
            QUALIFIED_CANDIDATE_KERNEL_SOURCE_SHA256
        ),
        "task_count": 1,
        "task_verdict_counts": {"resolved": 1},
        "task_failure_mode_counts": {"tests_passed": 1},
        "swe_orchestrator_exit_code": 0,
        "real_task_authenticated": True,
        "batch_size": 1,
        "physical_rows_per_request": 32,
        "candidate_conv_launches_per_layer": 1,
        "layer_count": 48,
        "comparison_records": 22080,
        "compared_byte_surfaces": ["conv_out", "commit_source_stage"],
        "reference_always_served": True,
        "no_fallback": True,
        "production_enabled": False,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "draft_vocab_k": DRAFT_VOCAB_K,
        "draft_vocab_root": DRAFT_VOCAB_ROOT,
        "source_descriptor_device_validation": False,
        "source_descriptor_launcher_argument": False,
        "topology_host_validation": "exact_parent_each_launch",
        "contains_raw_logs": False,
        "contains_task_or_model_content": False,
    }
    mismatches = [
        key for key, expected in required.items() if payload.get(key) != expected
    ]
    if mismatches:
        raise RuntimeError(
            "FR13 SFWD packed x-gather reduced gate contract mismatch: "
            + ",".join(sorted(set(mismatches)))
        )


def fixed32_sfwd_prior_reuse_timing_control(
    *,
    environ=None,
    arm_path: str | None = None,
    gate_path: str | None = None,
) -> dict[str, object] | None:
    """Validate the qualified gate before candidate bytes may be served."""
    env = os.environ if environ is None else environ
    selector = str(env.get("FR13_FIXED32_SFWD_PRIOR_REUSE_PRODUCTION", ""))
    if selector not in ("", "0", "1"):
        raise RuntimeError(
            "FR13_FIXED32_SFWD_PRIOR_REUSE_PRODUCTION must be exactly 0 or 1"
        )
    arm = arm_path or str(
        env.get("FR13_FIXED32_SFWD_PRIOR_REUSE_TIMING_ARM_PATH", TIMING_ARM)
    )
    if selector != "1" and not os.path.exists(arm):
        return None
    if _regular_ascii(arm, "FR13 SFWD prior-reuse timing arm", limit=8).strip() != b"1":
        raise RuntimeError("FR13 SFWD prior-reuse timing arm must contain exactly 1")
    if _FR13_FIXED32_MODE not in _FR13_FIXED32_MODES:
        raise RuntimeError("FR13 SFWD prior-reuse timing requires fixed32")
    if str(env.get("FR13_FIXED32_SFWD_PRIOR_REUSE_TIMING_AB", "")) != "1":
        raise RuntimeError("FR13 SFWD prior-reuse production requires its timing route")
    if str(env.get("FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB", "")) == "1":
        raise RuntimeError("FR13 SFWD prior-reuse byte and timing routes are exclusive")
    expected_digest = str(
        env.get("FR13_FIXED32_SFWD_PRIOR_REUSE_GATE_SHA256", "")
    ).strip()
    if expected_digest != QUALIFIED_REDUCED_GATE_SHA256:
        raise RuntimeError(
            "FR13 SFWD prior-reuse timing gate identity is not qualified"
        )
    path = gate_path or str(
        env.get("FR13_FIXED32_SFWD_PRIOR_REUSE_TIMING_GATE_PATH", TIMING_GATE)
    )
    raw = _regular_ascii(path, "FR13 SFWD prior-reuse reduced gate", limit=131072)
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise RuntimeError("FR13 SFWD prior-reuse reduced gate SHA-256 mismatch")
    payload = _strict_json(raw, "FR13 SFWD prior-reuse reduced gate")
    _validate_gate(payload)
    source_digest = _candidate_source_sha256()
    kernel_source_digest = _candidate_kernel_source_sha256()
    credential = dict(payload)
    credential["reduced_gate_sha256"] = expected_digest
    credential["runtime_candidate_source_sha256"] = source_digest
    credential["runtime_candidate_kernel_source_sha256"] = kernel_source_digest
    credential["task_marker_sha256"] = TASK_MARKER_SHA256
    _CREDENTIAL_IDS.add(id(credential))
    return credential


def _authenticated_real_event(credential: dict[str, object]) -> str | None:
    path = os.environ.get(
        "FR13_FIXED32_SFWD_STATE_FUSION_REAL_EVENT_PATH", REAL_EVENT_PATH
    )
    if (
        not path
        or not os.path.isabs(path)
        or Path(path).name != Path(REAL_EVENT_PATH).name
    ):
        raise RuntimeError("FR13 SFWD prior-reuse timing real-event path is invalid")
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise RuntimeError(
            f"FR13 SFWD prior-reuse timing marker cannot be inspected: {error}"
        ) from error
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o444
    ):
        raise RuntimeError(
            "FR13 SFWD prior-reuse timing marker must be mode-0444 regular"
        )
    raw = _regular_ascii(path, "FR13 SFWD prior-reuse timing marker", limit=256)
    digest = hashlib.sha256(raw).hexdigest()
    expected_digest = str(credential.get("task_marker_sha256", ""))
    if digest != expected_digest or expected_digest != TASK_MARKER_SHA256:
        raise RuntimeError("FR13 SFWD prior-reuse timing marker is not task-bound")
    return digest


def fixed32_sfwd_prior_reuse_timing_engagement(
    *, credential: dict[str, object], layer_key: int, batch_size: int
) -> dict[str, object]:
    """Publish one attestation after 48 unique candidate-served layers."""
    if id(credential) not in _CREDENTIAL_IDS:
        raise RuntimeError("FR13 SFWD prior-reuse timing credential was not validated")
    required_credential = {
        "reduced_gate_sha256": QUALIFIED_REDUCED_GATE_SHA256,
        "runtime_candidate_source_sha256": QUALIFIED_CANDIDATE_SOURCE_SHA256,
        "runtime_candidate_kernel_source_sha256": (
            QUALIFIED_CANDIDATE_KERNEL_SOURCE_SHA256
        ),
        "task_marker_sha256": TASK_MARKER_SHA256,
    }
    if any(
        credential.get(key) != expected
        for key, expected in required_credential.items()
    ):
        raise RuntimeError("FR13 SFWD prior-reuse timing credential drifted")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("FR13 SFWD prior-reuse timing is eager-only")
    batch = int(batch_size)
    candidate.fixed32_sfwd_prior_reuse_contract(
        batch,
        tree_rows=32,
        conv_width=4,
        conv_state_len=candidate.CONV_STATE_LEN,
    )
    if batch != 1:
        raise RuntimeError("FR13 SFWD prior-reuse timing is B1-only")
    task_marker_sha256 = _authenticated_real_event(credential)
    record = {
        "candidate": CANDIDATE,
        "candidate_kernel": (
            "_fr13_fixed32_sfwd_prior_reuse_packed_xgather_kernel"
        ),
        "batch_size": batch,
        "layer_count": len(_STATE["layers"]),
        "launches_observed": int(_STATE["launches"]),
        "reduced_gate_sha256": str(credential.get("reduced_gate_sha256", "")),
        "candidate_source_sha256": str(
            credential.get("runtime_candidate_source_sha256", "")
        ),
        "candidate_kernel_source_sha256": str(
            credential.get("runtime_candidate_kernel_source_sha256", "")
        ),
        "candidate_served": True,
        "sole_conv_source_producer": True,
        "real_task_bound": task_marker_sha256 is not None,
    }
    if task_marker_sha256 is None:
        return record
    gate_digest = str(credential.get("reduced_gate_sha256", ""))
    source_digest = str(credential.get("runtime_candidate_source_sha256", ""))
    kernel_source_digest = str(
        credential.get("runtime_candidate_kernel_source_sha256", "")
    )
    if _STATE["gate_sha256"] is None:
        _STATE["gate_sha256"] = gate_digest
        _STATE["candidate_source_sha256"] = source_digest
        _STATE["candidate_kernel_source_sha256"] = kernel_source_digest
        _STATE["task_marker_sha256"] = task_marker_sha256
    elif (
        _STATE["gate_sha256"] != gate_digest
        or _STATE["candidate_source_sha256"] != source_digest
        or _STATE["candidate_kernel_source_sha256"] != kernel_source_digest
        or _STATE["task_marker_sha256"] != task_marker_sha256
    ):
        raise RuntimeError("FR13 SFWD prior-reuse timing identity changed")
    key = int(layer_key)
    if key not in _STATE["layers"] and len(_STATE["layers"]) >= 48:
        raise RuntimeError("FR13 SFWD prior-reuse timing observed more than 48 layers")
    _STATE["launches"] = int(_STATE["launches"]) + 1
    _STATE["layers"].add(key)
    record.update(
        layer_count=len(_STATE["layers"]),
        launches_observed=int(_STATE["launches"]),
    )
    if len(_STATE["layers"]) != 48 or bool(_STATE["emitted"]):
        return record
    if int(_STATE["launches"]) != 48:
        raise RuntimeError(
            "FR13 SFWD prior-reuse timing requires exactly one launch per layer"
        )
    layer_key_digest = hashlib.sha256(
        json.dumps(
            sorted(_STATE["layers"]),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    path = os.environ.get(
        "FR13_FIXED32_SFWD_PRIOR_REUSE_TIMING_ENGAGEMENT_PATH",
        TIMING_ENGAGEMENT,
    )
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        "schema": "fr13.fixed32.sfwd_xgather.timing_engagement.v1",
        "status": "engaged",
        "run_classification": (
            "one_real_swe_verified_k64_root_b1_packed_xgather_timing_diagnostic"
        ),
        **record,
        "task_marker_sha256": task_marker_sha256,
        "layer_count": 48,
        "layer_key_digest": layer_key_digest,
        "physical_rows_per_request": 32,
        "source_rows_per_request": 36,
        "conv_rows_per_program": 32,
        "conv_block_c": 64,
        "conv_num_warps": candidate.NUM_WARPS,
        "draft_vocab_k": DRAFT_VOCAB_K,
        "draft_vocab_root": DRAFT_VOCAB_ROOT,
        "draft_vocab_blocks_sha256": DRAFT_VOCAB_BLOCKS_SHA256,
        "candidate_conv_launches_per_layer": 1,
        "incumbent_conv_launches_per_layer": 0,
        "source_descriptor_device_validation": False,
        "source_descriptor_launcher_argument": False,
        "topology_host_validation": "exact_parent_each_launch",
        "gdn_level_path_programs": [1, 11],
        "gdn_physical_launches_per_layer": 2,
        "gdn_ring_export_unchanged": True,
        "gdn_flags_export_unchanged": True,
        "real_task_gate_bound": True,
        "fallback_permitted": False,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_eligible": False,
    }
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="ascii") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
    _STATE["emitted"] = True
    return record
