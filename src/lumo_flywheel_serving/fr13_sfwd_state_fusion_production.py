"""Source-bound candidate serving control for the fixed32 SFWD fusion gate."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

import torch

from lumo_flywheel_serving import fr10_gdn_tree_kernel as kernel


CANDIDATE = "fixed32_sfwd_state_fusion_v1"
QUALIFIED_KERNEL_SOURCE_SHA256 = (
    "a9decdbe60db4227e4128ea05bfa405e09e284bfbc7c3ed216cd73d3f72d35ec"
)
QUALIFIED_CLOSURE_SHA256 = (
    "792c14941d933e9978811cd7dba33e9c9877638ed7f8fca4aa2044643cfbf038"
)
_CLOSURE_NAMES = (
    "_FR13_FIXED32_MODES",
    "_FR13_FIXED32_PARENT",
    "_FR13_FIXED32_SFWD_STATE_FUSION_CANDIDATE_ID",
    "_FR13_FIXED32_SFWD_CONV_STATE_LEN",
    "fixed32_sfwd_state_fusion_contract",
    "_fr13_fixed32_sfwd_state_fusion_kernel",
    "_fr13_fixed32_conv_source_flat_expected",
    "launch_fixed32_sfwd_state_fusion",
)
PRODUCTION_ARM = "/logs/fr13_fixed32_sfwd_state_fusion.production.arm"
PRODUCTION_PASS = "/logs/fr13_fixed32_sfwd_state_fusion.production_pass.json"
PRODUCTION_PASS_SHA256 = (
    "/logs/fr13_fixed32_sfwd_state_fusion.production_pass.sha256"
)
PRODUCTION_ENGAGEMENT = (
    "/logs/fr13_fixed32_sfwd_state_fusion.production_engagement.json"
)
BYTE_ENABLED = "/logs/fr13_fixed32_sfwd_state_fusion_byte_ab.enabled"
_CREDENTIAL_IDS: set[int] = set()
_STATE = {
    "live_pass_sha256": None,
    "source_sha256": None,
    "closure_sha256": None,
    "layers": set(),
    "launches": 0,
    "emitted": False,
}


def _regular_ascii(path: str, label: str, *, limit: int) -> bytes:
    if os.path.islink(path) or not os.path.isfile(path):
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")
    try:
        with open(path, "rb") as handle:
            raw = handle.read(limit + 1)
    except OSError as error:
        raise RuntimeError(f"{label} cannot be read: {path}: {error}") from error
    if not raw or len(raw) > limit:
        raise RuntimeError(f"{label} is empty or exceeds {limit} bytes: {path}")
    try:
        raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"{label} must be ASCII: {path}") from error
    return raw


def _kernel_source_sha256() -> str:
    try:
        raw = Path(kernel.__file__).resolve().read_bytes()
    except OSError as error:
        raise RuntimeError(f"FR13 SFWD kernel source cannot be hashed: {error}") from error
    return hashlib.sha256(raw).hexdigest()


def fixed32_sfwd_state_fusion_candidate_closure() -> dict[str, object]:
    """Hash the executable candidate closure qualified by the live byte pass."""
    path = Path(kernel.__file__).resolve()
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=os.fspath(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise RuntimeError(
            f"FR13 SFWD candidate closure cannot be parsed: {error}"
        ) from error
    members: dict[str, str] = {}
    for node in tree.body:
        names: list[str] = []
        if isinstance(node, ast.Assign):
            names = [
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            ]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = [node.name]
        for name in names:
            if name in _CLOSURE_NAMES:
                if name in members:
                    raise RuntimeError(
                        f"FR13 SFWD candidate closure duplicates {name}"
                    )
                members[name] = ast.dump(
                    node, annotate_fields=True, include_attributes=False
                )
    if tuple(name for name in _CLOSURE_NAMES if name not in members):
        missing = tuple(name for name in _CLOSURE_NAMES if name not in members)
        raise RuntimeError(f"FR13 SFWD candidate closure is missing {missing!r}")
    canonical = "".join(
        f"{name}\0{members[name]}\0" for name in _CLOSURE_NAMES
    ).encode("ascii")
    digest = hashlib.sha256(canonical).hexdigest()
    return {
        "schema": "fr13.fixed32.sfwd_state_fusion.source_closure.v1",
        "sha256": digest,
        "members": {
            name: hashlib.sha256(members[name].encode("ascii")).hexdigest()
            for name in _CLOSURE_NAMES
        },
    }


def fixed32_sfwd_state_fusion_production_control(
    *,
    environ=None,
    arm_path: str | None = None,
    pass_path: str | None = None,
    pass_sha256_path: str | None = None,
) -> dict[str, object] | None:
    """Validate the default-off live PASS before candidate bytes are served."""
    env = os.environ if environ is None else environ
    selector = str(env.get("FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION", ""))
    if selector not in ("", "0", "1"):
        raise RuntimeError(
            "FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION must be exactly 0 or 1"
        )
    arm = arm_path or str(
        env.get("FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION_ARM_PATH", PRODUCTION_ARM)
    )
    if selector != "1" and not os.path.exists(arm):
        return None
    if _regular_ascii(arm, "FR13 SFWD production arm", limit=8).strip() != b"1":
        raise RuntimeError("FR13 SFWD production arm must contain exactly 1")
    if kernel._FR13_FIXED32_MODE not in kernel._FR13_FIXED32_MODES:
        raise RuntimeError("FR13 SFWD production timing requires fixed32")
    if (
        str(env.get("FR13_DRAFT_VOCAB_ROOT", "")) != "1"
        or str(env.get("FR13_DRAFT_VOCAB_K", "")) != "65536"
    ):
        raise RuntimeError("FR13 SFWD production timing requires ROOT=1 K=65536")
    enabled = str(
        env.get("FR13_FIXED32_SFWD_STATE_FUSION_ENABLED_PATH", BYTE_ENABLED)
    )
    if (
        str(env.get("FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB", "")) == "1"
        or os.path.exists(enabled)
    ):
        raise RuntimeError(
            "FR13 SFWD byte gate and production timing are mutually exclusive"
        )

    live_path = pass_path or str(
        env.get(
            "FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION_PASS_PATH",
            PRODUCTION_PASS,
        )
    )
    digest_path = pass_sha256_path or str(
        env.get(
            "FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION_PASS_SHA256_PATH",
            PRODUCTION_PASS_SHA256,
        )
    )
    live_raw = _regular_ascii(live_path, "FR13 SFWD live PASS", limit=131072)
    expected_digest = _regular_ascii(
        digest_path, "FR13 SFWD PASS SHA-256", limit=80
    ).decode("ascii").strip()
    if (
        len(expected_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_digest)
    ):
        raise RuntimeError("FR13 SFWD PASS SHA-256 is malformed")
    env_digest = str(
        env.get("FR13_FIXED32_SFWD_STATE_FUSION_LIVE_PASS_SHA256", "")
    ).strip()
    if env_digest and env_digest != expected_digest:
        raise RuntimeError("FR13 SFWD PASS SHA-256 sources disagree")
    actual_digest = hashlib.sha256(live_raw).hexdigest()
    if actual_digest != expected_digest:
        raise RuntimeError("FR13 SFWD live PASS SHA-256 mismatch")
    try:
        payload = json.loads(live_raw.decode("ascii"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"FR13 SFWD live PASS is invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("FR13 SFWD live PASS must be an object")
    runtime_source_digest = _kernel_source_sha256()
    closure = fixed32_sfwd_state_fusion_candidate_closure()
    closure_digest = str(closure["sha256"])
    if closure_digest != QUALIFIED_CLOSURE_SHA256:
        raise RuntimeError(
            "FR13 SFWD candidate closure drifted from the byte-qualified source"
        )
    required = {
        "schema": "fr13.fixed32.sfwd_state_fusion.live_pass.v1",
        "status": "byte_pass_source_only",
        "run_classification": (
            "one_real_swe_verified_full_vocab_b1_byte_timing_diagnostic"
        ),
        "candidate": CANDIDATE,
        "source_sha256": QUALIFIED_KERNEL_SOURCE_SHA256,
        "task_marker": "swe_verified:astropy__astropy-12907",
        "batch": 1,
        "layer_count": 48,
        "physical_rows_per_request": 32,
        "candidate_conv_launches_per_layer": 1,
        "gdn_level_path_programs": [1, 11],
        "gdn_physical_launches_per_layer": 2,
        "gdn_ring_export_unchanged": True,
        "gdn_flags_export_unchanged": True,
        "compared_byte_surfaces": ["conv_out", "commit_source_stage"],
        "real_task_authenticated": True,
        "reference_always_served": True,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_eligible": False,
    }
    mismatches = [
        key for key, expected in required.items() if payload.get(key) != expected
    ]
    layer_keys = payload.get("layer_keys")
    if (
        not isinstance(layer_keys, list)
        or len(layer_keys) != 48
        or len(set(layer_keys)) != 48
        or any(
            not isinstance(key, str)
            or not key.startswith("0x")
            or len(key) <= 2
            or any(character not in "0123456789abcdef" for character in key[2:])
            for key in layer_keys
        )
    ):
        mismatches.append("layer_keys")
    if mismatches:
        raise RuntimeError(
            "FR13 SFWD live PASS contract mismatch: "
            + ",".join(sorted(set(mismatches)))
        )
    credential = dict(payload)
    credential["live_pass_sha256"] = actual_digest
    credential["qualified_source_sha256"] = QUALIFIED_KERNEL_SOURCE_SHA256
    credential["runtime_source_sha256"] = runtime_source_digest
    credential["candidate_closure_sha256"] = closure_digest
    _CREDENTIAL_IDS.add(id(credential))
    return credential


def fixed32_sfwd_state_fusion_production_engagement(
    *, credential: dict[str, object], layer_key: int, batch_size: int
) -> dict[str, object]:
    """Publish one attestation after 48 unique candidate-served layers."""
    if id(credential) not in _CREDENTIAL_IDS:
        raise RuntimeError("FR13 SFWD production credential was not validated")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("FR13 SFWD production timing is eager-only")
    batch = int(batch_size)
    kernel.fixed32_sfwd_state_fusion_contract(
        batch,
        tree_rows=32,
        conv_width=4,
        conv_state_len=kernel._FR13_FIXED32_SFWD_CONV_STATE_LEN,
    )
    if batch != 1:
        raise RuntimeError("FR13 SFWD production timing is B1-only")
    live_digest = str(credential.get("live_pass_sha256", ""))
    source_digest = str(credential.get("runtime_source_sha256", ""))
    closure_digest = str(credential.get("candidate_closure_sha256", ""))
    if _STATE["live_pass_sha256"] is None:
        _STATE["live_pass_sha256"] = live_digest
        _STATE["source_sha256"] = source_digest
        _STATE["closure_sha256"] = closure_digest
    elif (
        _STATE["live_pass_sha256"] != live_digest
        or _STATE["source_sha256"] != source_digest
        or _STATE["closure_sha256"] != closure_digest
    ):
        raise RuntimeError("FR13 SFWD production identity changed")
    _STATE["launches"] = int(_STATE["launches"]) + 1
    _STATE["layers"].add(int(layer_key))
    record = {
        "candidate": CANDIDATE,
        "batch_size": batch,
        "layer_count": len(_STATE["layers"]),
        "launches_observed": int(_STATE["launches"]),
        "live_pass_sha256": live_digest,
        "source_sha256": source_digest,
        "qualified_source_sha256": QUALIFIED_KERNEL_SOURCE_SHA256,
        "candidate_closure_sha256": closure_digest,
        "candidate_served": True,
    }
    if len(_STATE["layers"]) != 48 or bool(_STATE["emitted"]):
        return record
    path = os.environ.get(
        "FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION_ENGAGEMENT_PATH",
        PRODUCTION_ENGAGEMENT,
    )
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        "schema": "fr13.fixed32.sfwd_state_fusion.production_engagement.v1",
        "status": "engaged",
        "run_classification": "real_swe_verified_exact4_k64_b1_kernel_stack",
        **record,
        "layer_count": 48,
        "layer_keys": [f"0x{key:x}" for key in sorted(_STATE["layers"])],
        "physical_rows_per_request": 32,
        "source_rows_per_request": 36,
        "candidate_conv_launches_per_layer": 1,
        "incumbent_conv_launches_per_layer": 0,
        "gdn_level_path_programs": [1, 11],
        "gdn_physical_launches_per_layer": 2,
        "gdn_ring_export_unchanged": True,
        "gdn_flags_export_unchanged": True,
        "real_task_pass_bound": True,
        "draft_vocab_root": 1,
        "draft_vocab_k": 65536,
        "timing_eligible": True,
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
