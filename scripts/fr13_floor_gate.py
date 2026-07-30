#!/usr/bin/env python3
"""Read-only floor-SLO reducer for canonical real SWE-Verified campaigns.

The reducer deliberately uses different uncertainty models for B=1 and B=4.
At B=1, a whole SWE task is the sampling cluster. At B=4, task brackets
overlap global counters, so the reducer selects their counter-index union once
and reports only time-series-conditional moving-block sensitivity. Fixed-work
census files bind the complete SFWD stream before that task selection is made.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import re
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fr13_fixed32_contract as fixed32_contract  # noqa: E402
from fr13_fixed32_contract import (  # noqa: E402
    CONTAINER_FA2_DESTINATION,
    ContractError as Fixed32ContractError,
    expected_pid1_argv,
    fixed32_tree_text,
    validate_external_manifest,
    validate_runtime_attestation,
)
from fr13_fixed32_topology import (  # noqa: E402
    HYDRA27_ACTIVE_DRAFTS,
    HYDRA27_VALID_MASK,
    PHYSICAL_DRAFTS,
    TAIL6_ACTIVE_DRAFTS,
    TAIL6_VALID_MASK,
)
from fr13_fixed32_flush_protocol import (  # noqa: E402
    ACK_KEYS as FLUSH_ACK_KEYS,
    ACK_SCHEMA as FLUSH_ACK_SCHEMA,
    READY_NONCE as FLUSH_READY_NONCE,
    REQUEST_KEYS as FLUSH_REQUEST_KEYS,
    REQUEST_SCHEMA as FLUSH_REQUEST_SCHEMA,
    RESULT_SCHEMA as FLUSH_RESULT_SCHEMA,
)
from fr13_fixed32_work_census import (  # noqa: E402
    CensusError as WorkCensusError,
)
from fr13_fixed32_work_census import SCHEMA as WORK_CENSUS_EVENT_SCHEMA  # noqa: E402
from fr13_fixed32_work_census import (  # noqa: E402
    TERMINAL_SCHEMA as WORK_CENSUS_TERMINAL_SCHEMA,
)
from fr13_fixed32_work_census import CONV_PREGATHER_BLOCK  # noqa: E402
from fr13_fixed32_work_census import CONV_PREGATHER_LAYERS  # noqa: E402
from fr13_fixed32_work_census import CONV_PREGATHER_ROW_ELEMS  # noqa: E402
from fr13_fixed32_work_census import FIXED_WORK_SCOPE  # noqa: E402
from fr13_fixed32_work_census import MODE_SEMANTICS as WORK_CENSUS_MODE_SEMANTICS  # noqa: E402
from fr13_fixed32_work_census import REPORT_SCHEMA as WORK_CENSUS_REPORT_SCHEMA  # noqa: E402
from fr13_fixed32_work_census import SUPPORTED_BATCH_SIZES  # noqa: E402
from fr13_fixed32_work_census import load_jsonl as load_work_census_jsonl  # noqa: E402
from fr13_fixed32_work_census import reference_event as work_census_fixture  # noqa: E402
from fr13_fixed32_work_census import (  # noqa: E402
    reference_terminal_summary as work_census_terminal_fixture,
)
from fr13_fixed32_work_census import (  # noqa: E402
    validate_campaign as validate_work_census_campaign,
)
from fr13_fixed32_work_census import validate_event as validate_work_census_event  # noqa: E402
from fr13_fixed32_work_census import (  # noqa: E402
    forward_graph_structural_signature,
)
from fr13_runtime_manifest import (  # noqa: E402
    ManifestError as RuntimeManifestError,
)
from fr13_runtime_manifest import build_manifest as build_runtime_manifest  # noqa: E402


class GateError(RuntimeError):
    """An input artifact failed a fail-closed gate."""


CANONICAL_TASK_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
    "astropy__astropy-13453",
    "astropy__astropy-13579",
    "astropy__astropy-13977",
    "astropy__astropy-14096",
    "astropy__astropy-14182",
    "astropy__astropy-14309",
    "astropy__astropy-14365",
    "astropy__astropy-14369",
    "astropy__astropy-14508",
    "astropy__astropy-14539",
    "astropy__astropy-14598",
    "astropy__astropy-14995",
)
PINNED_SWE_VERIFIED_PARQUET_SHA256 = (
    "a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd"
)
PINNED_SWE_VERIFIED_PARQUET_RELATIVE = (
    ".cache/huggingface/hub/"
    "datasets--princeton-nlp--SWE-bench_Verified/blobs/"
    + PINNED_SWE_VERIFIED_PARQUET_SHA256
)
EVIDENCE_SETS = {
    4: {
        "relative_path": "config/fr13_fixed32/subset_b4_four.json",
        "sha256": ("0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"),
        "task_ids": CANONICAL_TASK_IDS[:4],
    },
    16: {
        "relative_path": "config/fr13_fixed32/subset_b4_sixteen.json",
        "sha256": ("47b0a3c9be49e2cb5f7e7217ae03c267a05359f269f3e3b038942f57d7dc0b5c"),
        "task_ids": CANONICAL_TASK_IDS,
    },
}

METRICS = {
    "fwd_s": "vllm:fr13_decode_forward_gpu_seconds_total",
    "fwd_steps": "vllm:fr13_decode_forward_gpu_steps_total",
    "fwd_drafts": "vllm:fr13_decode_forward_gpu_drafts_total",
    "wall_s": "vllm:fr13_decode_step_wall_seconds_total",
    "wall_drafts": "vllm:fr13_decode_step_wall_drafts_total",
    "wall_steps": "vllm:fr13_decode_step_wall_steps_total",
    "wall_attempts": "vllm:fr13_decode_step_wall_attempts_total",
    "wall_rejected": "vllm:fr13_decode_step_wall_rejected_total",
    "spec_drafts": "vllm:spec_decode_num_drafts_total",
    "spec_tokens": "vllm:spec_decode_num_draft_tokens_total",
}
INTEGRAL_METRICS = {
    "fwd_steps",
    "fwd_drafts",
    "wall_drafts",
    "wall_steps",
    "wall_attempts",
    "wall_rejected",
    "spec_drafts",
    "spec_tokens",
}
EXPECTED_METRIC_LABELS = {
    key: (
        'engine="0",model_name="qwen3.6-27b"'
        if key in {"spec_drafts", "spec_tokens"}
        else ""
    )
    for key in METRICS
}
PRETASK_REQUIRED_METRICS = frozenset({"spec_drafts", "spec_tokens"})
SAMPLE_RE = re.compile(
    r"^(?P<name>[^\s{]+)(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+0-9.eE]+)$"
)
ORCHESTRATOR_HEADER_RE = re.compile(
    r"^=== \[[^\]]+\] dataset=(?P<dataset>\S+) tag=\S+ "
    r"n=(?P<tasks>\d+) concurrency=(?P<concurrency>\d+) ===$"
)
ORCHESTRATOR_DONE_RE = re.compile(r"^=== \[[^\]]+\] DONE n=(?P<tasks>\d+) .+ ===$")
TASK_START_RE = re.compile(r"^\[[^\]]+\] -> (?P<task>\S+)$")
TASK_END_RE = re.compile(r"^\[[^\]]+\] <- (?P<task>\S+) .+$")
ARM_HEADER_RE = re.compile(
    r"^=== BIGDENOM-VARIANT SWEServe ARM (?P<arm>\S+) "
    r"kind=(?P<kind>\S+) .* expect=(?P<tokens>\d+) .* "
    r"subset=(?P<subset>\S+) ===$"
)
ENGINE_CORE_PID_RE = re.compile(r"^PID (?P<pid>\d+) cmd=\[VLLM::EngineCore(?:\s|\])")
FIXED32_TREE = fixed32_tree_text()
FIXED32_PRESEED = (
    "[FR13_SUBTREE_PARALLEL] preseeded: n=32 schedule=fixed32 "
    "levels=[1, 11] lens=[5, 7] critical=12 (monolith 32) "
    "route_armed=1 selfcheck_armed=0"
)
FIXED32_ENGAGED = (
    "[FR13_SUBTREE_PARALLEL ENGAGED] n_actual=32 schedule=fixed32 critical=12"
)
FIXED32_WORK_ENGAGED = (
    "[FR13_FIXED32_WORK] engaged: physical_drafts=31 rows=32 "
    "gdn_launches=2 gdn_programs=12 gdn_slots=82 taw_walk=12 "
    "taw_buffer=32 output_slots=32 path_slots=16 reqkey_slots=16 "
    "kv_slots=16 conv_layers=48 committer_slots=16"
)
TAIL6_TOPOLOGY = (
    "[FR13_FIXED32] topology engaged: mode=tail6_fixed32 active_drafts=21 "
    "valid_mask=0x7a9ce73f"
)
HYDRA27_TOPOLOGY = (
    "[FR13_FIXED32] topology engaged: mode=hydra27_fixed32 active_drafts=27 "
    "valid_mask=0x7abdffff"
)
FIXED32_MODE_SPECS = {
    "tail6_fixed32": {
        "active_drafts": TAIL6_ACTIVE_DRAFTS,
        "valid_mask": TAIL6_VALID_MASK,
        "topology_needle": TAIL6_TOPOLOGY,
    },
    "hydra27_fixed32": {
        "active_drafts": HYDRA27_ACTIVE_DRAFTS,
        "valid_mask": HYDRA27_VALID_MASK,
        "topology_needle": HYDRA27_TOPOLOGY,
    },
}

WEIGHT_STREAM_LOWER_BOUND_MS = 98.6
COMPUTE_MS_PER_ROW = 0.54
SLO_MULTIPLIER = 1.15
REQUIRED_COVERAGE = 1.0
MIN_RETAINED_WALL_FRACTION = 0.99
MIN_FULL_GRAPH_FRACTION = 0.99
MIN_TASK_COUNTER_STEPS = 64
MIN_B4_EXACT_EVENTS = 512
MIN_B4_GE3_FRACTION = 0.65
MIN_B4_MEAN_OCCUPANCY = 2.9
MAX_B4_MEAN_OCCUPANCY_GAP = 0.25
BLOCK_SENSITIVITY = (64, 128, 256, 512)
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260729
RUNTIME_MANIFEST_PROFILE = "fixed32"
RUNTIME_MANIFEST_SEQUENCE = "scripts/fr13_fixed32_floor_timers_seq.sh"
FIXED32_BOUNDARY_SCHEMA = "fr13-fixed32-task-boundary-v1"
FIXED32_RUNTIME_SNAPSHOT_SCHEMA = "fr13-fixed32-boundary-snapshot-v3"
SFWD_MAIN_SIDECAR_SCHEMA = "fr13.sfwd_gpu_timer.v2"
SFWD_SAMPLE_SIDECAR_SCHEMA = "fr13.sfwd_per_step_samples.v2"
FIXED32_CHAT_TRAFFIC_AUDIT_SCHEMA = (
    "fr13-fixed32-chat-task-provenance-audit-v2"
)
FIXED32_REAL_TASK_PROVENANCE_SCHEMA = "fr13-fixed32-real-task-provenance-v3"
FIXED32_QWEN_RUNTIME_ATTESTATION_SCHEMA = (
    "fr13-fixed32-qwen-runtime-attestation-v1"
)
FIXED32_MOUNTED_RUNTIME_PROOF_SCHEMA = (
    "fr13-fixed32-qwen-mounted-runtime-proof-v1"
)
FIXED32_MOUNTED_RUNTIME_PROOF_FILENAME = (
    "qwen_mounted_runtime_proof.json"
)
FIXED32_AGENT_PLACEMENT_SCHEMA = "fr13-fixed32-agent-placement-v1"
FIXED32_QWEN_CODE_VERSION = "0.19.4"
FIXED32_QWEN_SYSTEM_SETTINGS_SHA256 = (
    "8a872a4f6f257f6d7a45f24f42500964f56e1500c5342218b71d02afe4d31fb6"
)
FIXED32_QWEN_BUNDLE_TREE = {
    "schema": "fr13-qwen-agent-bundle-manifest-v1",
    "roots": ["**"],
    "summary": {
        "entry_count": 10_499,
        "directory_count": 1_514,
        "regular_file_count": 8_970,
        "symlink_count": 15,
        "regular_file_bytes": 327_941_291,
        "executable_regular_file_count": 93,
        "manifest_bytes": 2_057_964,
    },
    "entrypoints": {
        "bin/qwen": {
            "path": "bin/qwen",
            "type": "file",
            "mode": "0755",
            "bytes": 217,
            "sha256": (
                "286a61bd49fd103d0ea29a8d971030b60ac0a6e7f19b292bdf9b39858e1161e2"
            ),
        },
        "node/bin/node": {
            "path": "node/bin/node",
            "type": "file",
            "mode": "0755",
            "bytes": 124_835_376,
            "sha256": (
                "93956de2e59480474a7b46571da1651180b1a050cdf32641ebec4ce6e478e068"
            ),
        },
        "npm/bin/qwen": {
            "path": "npm/bin/qwen",
            "type": "symlink",
            "mode": "0777",
            "target": (
                "../lib/node_modules/@qwen-code/qwen-code/cli-entry.js"
            ),
        },
        "npm/lib/node_modules/@qwen-code/qwen-code/cli-entry.js": {
            "path": (
                "npm/lib/node_modules/@qwen-code/qwen-code/cli-entry.js"
            ),
            "type": "file",
            "mode": "0755",
            "bytes": 777,
            "sha256": (
                "98335eda2e0eaa737640cb5d43da032dee457ff7931c429f972ba3ff8a695d3a"
            ),
        },
    },
    "manifest_sha256": (
        "2643d1d64c03887654794d9bd00a88fbf9ced7362e034557cf196b8a37e744bc"
    ),
}
FIXED32_CLEARED_AGENT_ENVIRONMENT = [
    "BASH_ENV",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "NODE_OPTIONS",
    "NODE_PATH",
]
FIXED32_QWEN_REMOTE_SETTINGS = {
    "bytes": 37,
    "sha256": FIXED32_QWEN_SYSTEM_SETTINGS_SHA256,
    "mode": "0444",
    "uid": 1000,
    "gid": 1000,
    "nlink": 1,
    "xattrs": [],
}
FIXED32_MEASURED_HOST_IDENTITY = {
    "hostname_sha256": (
        "4a30015a37d4584ee4bb29d3144df3ad061b688baaff100d4b0dc675e4988a87"
    ),
    "system": "Linux",
    "machine": "aarch64",
    "kernel": "6.14.0-1015-nvidia",
    "machine_id_sha256": (
        "2bbfaa2351d92eefaec4bdf90f25f31ebab0973aaf88d7f9c01650b01cdd9113"
    ),
    "docker_daemon_id_sha256": (
        "53fb603eb13afb8d6eb165a2aefa510dd6f9c06faf7c4c80a7e966ce4d5c4fce"
    ),
}
FIXED32_AGENT_HOST_IDENTITY = {
    "hostname_sha256": (
        "84674d76bbc389eb6479d5f3a617bf6b25e05f486d455353e945a700df66afa4"
    ),
    "system": "Linux",
    "machine": "x86_64",
    "kernel": "6.14.0-37-generic",
    "machine_id_sha256": (
        "e49c50112bb466cb283df00e3f3e7344abc702f887f78550ba44c8611ef657f8"
    ),
    "docker_daemon_id_sha256": (
        "2606e93ef9583c1aabb076ed439f9fcf1d60068ddd9ecdc464bbe3a29d9f371f"
    ),
}
FIXED32_AGENT_PLACEMENT = {
    "schema": FIXED32_AGENT_PLACEMENT_SCHEMA,
    "agent_host_identity": FIXED32_AGENT_HOST_IDENTITY,
    "measured_host_identity": FIXED32_MEASURED_HOST_IDENTITY,
    "identities_distinct": True,
}
FIXED32_AGENT_IMAGE_IDENTITIES = {
    task_id: {
        "instance_id": task_id,
        "image": (
            "swebench/sweb.eval.x86_64."
            f"{task_id.replace('__', '_1776_')}:latest"
        ),
        "id": f"sha256:{image_id}",
        "repo_digest": (
            "swebench/sweb.eval.x86_64."
            f"{task_id.replace('__', '_1776_')}@sha256:{repo_digest}"
        ),
        "architecture": "amd64",
        "os": "linux",
    }
    for task_id, (image_id, repo_digest) in zip(
        CANONICAL_TASK_IDS,
        (
            (
                "cce639c4d4c4f8a47893a81a1a1894d3f2c77e603694da4000581450783ef532",
                "f3f63bb87d581c0e7b47f900dd82165b71040e1758d3c29e915e2b18da9baf63",
            ),
            (
                "bdbc30d363cfbd9902e0d338fc959782d1a92a9e85236c0dca92c95372088783",
                "42797a2c686ed35b43e28dd2f58149381c3bf79abea2346fb7062c009e9fa528",
            ),
            (
                "236d427ece8e0d3d282598d981ec9a3baff668041cab16f4545500956cb807db",
                "a43e166eb5ae9e477349b87d800ece7648f8c746f88d94f6f6cff0df1e2caf82",
            ),
            (
                "1047cc1d43b1a33fd5c3c5ca53b85d7380cdda26bf4c18ada15dd12a8d9b076d",
                "423240067cb26131788348337e59a4d51225fd491eccab99e919c1bc4d4b02e0",
            ),
            (
                "67e4955c1ea7f9013a51299a236574f87bc1ca2da02c9aa6ac640f8ce61d2f8b",
                "6b3593a3d1fa032b8b81f9a1e33738aae14c1cdc8fd887e0a089592c7f9abf9b",
            ),
            (
                "38c16ead9bc5b6a8cfb991a85511471ca91a1f334ca5d9001830c069beb3c94e",
                "c645e9435a1499dab327b76fae926057a2f407a291b5d173bf1971b7a6fe911b",
            ),
            (
                "fc877eb6e9cada009834e33eeab24e7d6690e189932d3f3c4e5f7a911e7e57b6",
                "bcc442a117def63c8011091500b4e2dd49980a08b74e3c2fbdad4e42745278bb",
            ),
            (
                "3ac2306a3e172713ad0f00dd5daa6d3cfd6990fe3b52afbf585fe1d6161ae8b9",
                "f0277869f5874118b6395d945a278d88e8912dbb2970ee2d0289f5591adab8a3",
            ),
            (
                "770f48b2842d8115639b73b022994b2e1b20af676367509244b9dcbad3f903e9",
                "374a7f3206d23fd41aaf6ac3361a34dad941bdbe92ec81ff1b4c3e0163e38453",
            ),
            (
                "f896444b1e4977454bd415d711926f7ce0f25c8b4c6c417b26d38df59e8a3ca6",
                "89fc1b9379b349c8d1773cc3e6982cb32d2732f90a240047c688735db35c5212",
            ),
            (
                "3925b434247f8fc7aabe79a902d03d4566d697c4f35b7a39fdfa209263efb358",
                "bb1ec8d27d478ab7469805680c824a4030fa3f32a592558f7a02485b76f3226b",
            ),
            (
                "1a8a749d6b8edb837761e2d09bdf9459e3bdb2436d4070412d886de135ff0a3a",
                "113488bfd3a6fb9678705496c18791bf46d98ad0303a682a3d405223eb2e85c4",
            ),
            (
                "f1f44383c21a57c5047ac6cdb4375407350dd14d54e455ec3b5dfec79ee26c7c",
                "a6aea03ce1c6a2e897e78a4339e44f02643deb9007e0628a058034779181ce71",
            ),
            (
                "290a743498af81faf833324ccb3dfaf877e1d4fdd60594efc1a5f4835601316e",
                "a8d0f9829ec24dfb23a2f0097a245ee60faf1b396b33b3af5c22d7ac5f3c00ab",
            ),
            (
                "3f3a0c5f4cd49b03ef4fde7f8e8bdf18833ee685d223395893b61426a6e62b8d",
                "f1ff70694c403ee7018fef2a00638caa1555948d1c9b821175a7f4bdf2933a52",
            ),
            (
                "99afb65d48b892e0d2e015eeb0794175d26e6a092c81598aa5a32fb0978b30cc",
                "b29a3bf3daebe6055a2bba46bc98db070043acd53801bd783c2f620813a87eae",
            ),
        ),
        strict=True,
    )
}
FIXED32_INGRESS_LEDGER_SCHEMA = "fr13.fixed32.ingress-ledger-record.v1"
FIXED32_INGRESS_PREFLIGHT_SCHEMA = "fr13-fixed32-ingress-auth-preflight-v1"
FIXED32_PROXY_INGRESS_BEGIN_SCHEMA = "fr13-fixed32-proxy-ingress-begin-v1"
FIXED32_ENGINE_INGRESS_BEGIN_SCHEMA = "fr13-fixed32-engine-ingress-begin-v1"
FIXED32_PROXY_INGRESS_FINALIZE_SCHEMA = (
    "fr13-fixed32-proxy-ingress-finalize-v1"
)
FIXED32_ENGINE_INGRESS_FINALIZE_SCHEMA = (
    "fr13-fixed32-engine-ingress-finalize-v1"
)
FIXED32_INGRESS_LEDGER_KEYS = frozenset(
    {
        "schema",
        "seq",
        "role",
        "phase",
        "event",
        "route",
        "task_key_id",
        "logical_id_sha256",
        "wire_id_sha256",
        "engine_request_id_sha256",
        "status_code",
        "outcome",
        "reason",
        "evidence_sha256",
        "prev_sha256",
        "record_sha256",
    }
)
SWE_VERIFIED_DATASET = "princeton-nlp/SWE-bench_Verified"
FIXED32_COUNTER_KEYS = frozenset(
    {
        "pure_decode_forward_steps",
        "complete_work_census_events",
        "work_census_first_forward_step",
        "work_census_last_forward_step",
        "sfwd_pending",
        "dfwd_pending",
        "cfwd_pending",
    }
)
FIXED32_STEP_METRIC = "vllm:fr13_fixed32_pure_decode_forward_steps_total"
FIXED32_CENSUS_METRIC = "vllm:fr13_fixed32_complete_work_census_events_total"
T95_ONE_SIDED = {
    3: 2.3533634348018264,
    15: 1.7530503556925547,
}


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise GateError(f"missing required artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pinned_swe_verified_parquet(path: Path) -> None:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != PINNED_SWE_VERIFIED_PARQUET_SHA256:
        raise GateError(
            f"{path}: pinned SWE-Verified Parquet SHA-256 mismatch: "
            f"expected {PINNED_SWE_VERIFIED_PARQUET_SHA256}, "
            f"got {actual_sha256}"
        )


@functools.lru_cache(maxsize=2)
def pinned_dataset_record_digests(repo_text: str) -> dict[str, str]:
    path = Path(repo_text) / PINNED_SWE_VERIFIED_PARQUET_RELATIVE
    if not path.is_file():
        raise GateError(f"pinned SWE-Verified Parquet is missing: {path}")
    validate_pinned_swe_verified_parquet(path)
    try:
        import pyarrow.parquet as pq

        rows = pq.read_table(path).to_pylist()
    except Exception as error:
        raise GateError(
            f"cannot read pinned SWE-Verified Parquet {path}: {error}"
        ) from error
    digests: dict[str, str] = {}
    for row in rows:
        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str) or instance_id in digests:
            raise GateError(f"{path}: invalid or duplicate instance_id")
        digests[instance_id] = hashlib.sha256(
            json.dumps(
                row,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    missing = sorted(set(CANONICAL_TASK_IDS) - set(digests))
    if missing:
        raise GateError(f"{path}: canonical task records are missing: {missing}")
    return digests


def read_text(path: Path) -> str:
    if not path.is_file():
        raise GateError(f"missing required artifact: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def strict_utf8_artifact(path: Path, *, label: str) -> tuple[bytes, str]:
    if not path.is_file():
        raise GateError(f"missing required artifact: {path}")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise GateError(f"{label}: artifact is not strict UTF-8: {error}") from error
    return raw, text


def exact_json_text(text: str, *, label: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise GateError(f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise GateError(f"{label}: non-finite JSON constant {value!r}")

    try:
        payload = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise GateError(f"{label}: invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise GateError(f"{label}: JSON root must be an object")
    return payload


def exact_json(path: Path, *, label: str) -> dict[str, Any]:
    _raw, text = strict_utf8_artifact(path, label=label)
    return exact_json_text(text, label=label)


def exact_keys(payload: dict[str, Any], keys: set[str] | frozenset[str], label: str) -> None:
    if set(payload) != set(keys):
        raise GateError(
            f"{label}: keys mismatch missing={sorted(set(keys) - set(payload))} "
            f"extra={sorted(set(payload) - set(keys))}"
        )


def first_json_difference(actual: object, expected: object, path: str = "$") -> str:
    if type(actual) is not type(expected):
        return (
            f"{path}: type {type(actual).__name__} != "
            f"{type(expected).__name__}"
        )
    if isinstance(actual, dict):
        if set(actual) != set(expected):
            return (
                f"{path}: keys {sorted(actual)} != "
                f"{sorted(expected)}"
            )
        for key in expected:
            if actual[key] != expected[key]:
                return first_json_difference(
                    actual[key],
                    expected[key],
                    f"{path}.{key}",
                )
        return f"{path}: dictionaries differ"
    if isinstance(actual, list):
        if len(actual) != len(expected):
            return f"{path}: length {len(actual)} != {len(expected)}"
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            if actual_item != expected_item:
                return first_json_difference(
                    actual_item,
                    expected_item,
                    f"{path}[{index}]",
                )
        return f"{path}: lists differ"
    return f"{path}: {actual!r} != {expected!r}"


def fixed32_metric_text(text: str, *, label: str, metric: str) -> int:
    values = []
    for line in text.splitlines():
        match = SAMPLE_RE.match(line)
        if match is not None and match.group("name") == metric:
            if match.group("labels"):
                raise GateError(f"{label}: fixed32 flush metric must be unlabelled")
            values.append(float(match.group("value")))
    if len(values) != 1:
        raise GateError(f"{label}: expected exactly one {metric} series")
    return integral(values[0], f"{label}:{metric}")


def validate_fixed32_counters(payload: object, *, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise GateError(f"{label}: counters must be an object")
    exact_keys(payload, FIXED32_COUNTER_KEYS, f"{label}.counters")
    for key in (
        "pure_decode_forward_steps",
        "complete_work_census_events",
        "sfwd_pending",
        "dfwd_pending",
        "cfwd_pending",
    ):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GateError(f"{label}.{key}: expected nonnegative integer")
    for key in ("work_census_first_forward_step", "work_census_last_forward_step"):
        value = payload[key]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise GateError(f"{label}.{key}: expected null or nonnegative integer")
    if any(payload[key] != 0 for key in ("sfwd_pending", "dfwd_pending", "cfwd_pending")):
        raise GateError(f"{label}: flush acknowledged pending timer samples")
    steps = payload["pure_decode_forward_steps"]
    events = payload["complete_work_census_events"]
    first = payload["work_census_first_forward_step"]
    last = payload["work_census_last_forward_step"]
    if events > steps:
        raise GateError(f"{label}: complete census events exceed pure-decode steps")
    if events == 0:
        if first is not None or last is not None:
            raise GateError(f"{label}: empty census requires null first/last")
    elif first is None or last is None or not 0 <= first <= last < steps:
        raise GateError(f"{label}: census first/last range is invalid")
    return payload


def strict_fixed32_ack_payload(
    payload: object,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise GateError(f"{label}: ack must be an object")
    exact_keys(payload, FLUSH_ACK_KEYS, label)
    if payload["schema"] != FLUSH_ACK_SCHEMA or payload["status"] != "ok":
        raise GateError(f"{label}: ack schema/status mismatch")
    if not isinstance(payload["mode"], str):
        raise GateError(f"{label}.mode: expected string")
    ack_producer_pid = strict_positive_int(
        payload["producer_pid"],
        label=f"{label}.producer_pid",
    )
    generation = strict_nonnegative_int(
        payload["generation"],
        label=f"{label}.generation",
    )
    nonce = payload["nonce"]
    if (
        not isinstance(nonce, str)
        or re.fullmatch(r"[0-9a-f]{64}", nonce) is None
    ):
        raise GateError(f"{label}: invalid nonce")
    if not isinstance(payload["action"], str):
        raise GateError(f"{label}: action must be a string")
    counters = validate_fixed32_counters(payload["counters"], label=label)
    return {
        **payload,
        "producer_pid": ack_producer_pid,
        "generation": generation,
        "counters": counters,
    }


def validate_fixed32_ack(
    payload: object,
    *,
    label: str,
    mode: str,
    producer_pid: int,
) -> dict[str, Any]:
    ack = strict_fixed32_ack_payload(payload, label=label)
    if ack["mode"] != mode or ack["producer_pid"] != producer_pid:
        raise GateError(f"{label}: ack identity mismatch")
    return ack


def integral(value: float, label: str) -> int:
    rounded = round(value)
    if not math.isfinite(value) or abs(value - rounded) > 1e-6:
        raise GateError(f"{label} is not an integral counter: {value}")
    return int(rounded)


def strict_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise GateError(f"{label}: expected nonnegative integer")
    return value


def strict_positive_int(value: object, *, label: str) -> int:
    result = strict_nonnegative_int(value, label=label)
    if result == 0:
        raise GateError(f"{label}: expected positive integer")
    return result


def strict_optional_nonnegative_int(
    value: object,
    *,
    label: str,
) -> int | None:
    if value is None:
        return None
    return strict_nonnegative_int(value, label=label)


def strict_nonnegative_number(value: object, *, label: str) -> float:
    if (
        type(value) not in {int, float}
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise GateError(f"{label}: expected finite nonnegative number")
    return float(value)


def strict_nonnegative_int_map(
    value: object,
    *,
    expected_keys: set[str] | frozenset[str],
    label: str,
) -> dict[str, int]:
    if not isinstance(value, dict):
        raise GateError(f"{label}: expected object")
    exact_keys(value, expected_keys, label)
    return {
        key: strict_nonnegative_int(item, label=f"{label}.{key}")
        for key, item in value.items()
    }


def strict_nonnegative_int_list(
    value: object,
    *,
    label: str,
) -> list[int]:
    if not isinstance(value, list):
        raise GateError(f"{label}: expected list")
    return [
        strict_nonnegative_int(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    ]


def metric_snapshot_text(
    text: str,
    *,
    label: str,
) -> tuple[dict[str, float], dict[str, str]]:
    wanted = {metric: key for key, metric in METRICS.items()}
    found: dict[str, float] = {}
    labels: dict[str, str] = {}
    for line in text.splitlines():
        match = SAMPLE_RE.match(line)
        if match is None:
            if line.startswith(tuple(wanted)):
                raise GateError(f"{label}: malformed required metric line {line!r}")
            continue
        key = wanted.get(match.group("name"))
        if key is None:
            continue
        if key in found:
            raise GateError(f"{label}: duplicate metric series for {METRICS[key]}")
        value = float(match.group("value"))
        if not math.isfinite(value) or value < 0:
            raise GateError(f"{label}: invalid metric {METRICS[key]}={value}")
        if key in INTEGRAL_METRICS:
            integral(value, f"{label}:{key}")
        found[key] = value
        labels[key] = match.group("labels") or ""
    missing = sorted(set(METRICS) - set(found))
    if missing:
        raise GateError(f"{label}: missing required metrics {missing}")
    return found, labels


def pretask_metric_snapshot_text(
    text: str,
    *,
    label: str,
) -> tuple[dict[str, float], dict[str, str]]:
    """Parse a raw generation-zero API scrape without requiring lazy worker series."""

    wanted = {metric: key for key, metric in METRICS.items()}
    found: dict[str, float] = {}
    labels: dict[str, str] = {}
    for line in text.splitlines():
        match = SAMPLE_RE.match(line)
        if match is None:
            if line.startswith(tuple(wanted)):
                raise GateError(f"{label}: malformed pretask metric line {line!r}")
            continue
        key = wanted.get(match.group("name"))
        if key is None:
            continue
        if key in found:
            raise GateError(f"{label}: duplicate metric series for {METRICS[key]}")
        value = float(match.group("value"))
        if not math.isfinite(value) or value < 0:
            raise GateError(f"{label}: invalid metric {METRICS[key]}={value}")
        if key in INTEGRAL_METRICS:
            integral(value, f"{label}:{key}")
        found[key] = value
        labels[key] = match.group("labels") or ""
    missing = sorted(PRETASK_REQUIRED_METRICS - set(found))
    if missing:
        raise GateError(f"{label}: missing required pretask metrics {missing}")
    if any(value != 0.0 for value in found.values()):
        raise GateError(f"{label}: pretask decode metrics are not exact zero")
    if any(labels[key] != EXPECTED_METRIC_LABELS[key] for key in found):
        raise GateError(f"{label}: pretask metric labels differ from the contract")
    return found, labels


def metric_snapshot(path: Path) -> tuple[dict[str, float], dict[str, str]]:
    _raw, text = strict_utf8_artifact(path, label=str(path))
    return metric_snapshot_text(text, label=str(path))


def load_metric_artifact(path: Path) -> dict[str, Any]:
    raw, text = strict_utf8_artifact(path, label=str(path))
    values, labels = metric_snapshot_text(text, label=str(path))
    return {
        "values": values,
        "labels": labels,
        "identity": {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        },
        "fixed32": {
            "pure_decode_forward_steps": fixed32_metric_text(
                text,
                label=str(path),
                metric=FIXED32_STEP_METRIC,
            ),
            "complete_work_census_events": fixed32_metric_text(
                text,
                label=str(path),
                metric=FIXED32_CENSUS_METRIC,
            ),
        },
    }


def validate_runtime_boundary_snapshot(
    path: Path,
    *,
    ack: dict[str, Any],
    server_capacity: int,
    metrics_path: Path | None,
    metric_values: dict[str, float] | None,
    reference: object,
    census_path: Path,
) -> dict[str, Any]:
    ack = strict_fixed32_ack_payload(
        ack,
        label=f"{path}:ack",
    )
    payload = exact_json(path, label=str(path))
    exact_keys(
        payload,
        {
            "schema",
            "mode",
            "producer_pid",
            "generation",
            "nonce",
            "action",
            "counters",
            "metrics",
        },
        str(path),
    )
    snapshot_producer_pid = strict_positive_int(
        payload["producer_pid"],
        label=f"{path}:producer_pid",
    )
    snapshot_generation = strict_nonnegative_int(
        payload["generation"],
        label=f"{path}:generation",
    )
    snapshot_counters = validate_fixed32_counters(
        payload["counters"],
        label=f"{path}:snapshot",
    )
    if (
        payload["schema"] != FIXED32_RUNTIME_SNAPSHOT_SCHEMA
        or payload["mode"] != ack["mode"]
        or snapshot_producer_pid != ack["producer_pid"]
        or snapshot_generation != ack["generation"]
        or payload["nonce"] != ack["nonce"]
        or payload["action"] != ack["action"]
        or snapshot_counters != ack["counters"]
    ):
        raise GateError(f"{path}: runtime snapshot does not bind to flush ack")
    expected_reference = {
        "schema": FIXED32_RUNTIME_SNAPSHOT_SCHEMA,
        "generation": ack["generation"],
        "path": str(path),
        "sha256": sha256_file(path),
    }
    if reference is not None:
        if not isinstance(reference, dict):
            raise GateError(
                f"{path}: task boundary runtime snapshot reference is malformed"
            )
        exact_keys(
            reference,
            {"schema", "generation", "path", "sha256"},
            f"{path}:reference",
        )
        reference_generation = strict_nonnegative_int(
            reference["generation"],
            label=f"{path}:reference.generation",
        )
        if (
            reference["schema"] != FIXED32_RUNTIME_SNAPSHOT_SCHEMA
            or reference_generation != expected_reference["generation"]
            or not isinstance(reference["path"], str)
            or reference["path"] != expected_reference["path"]
            or not isinstance(reference["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]) is None
            or reference["sha256"] != expected_reference["sha256"]
        ):
            raise GateError(
                f"{path}: task boundary runtime snapshot reference mismatch"
            )

    metrics = payload["metrics"]
    exact_keys(
        metrics,
        {"fixed32", "sfwd", "dfwd", "cfwd", "committer", "conv_pregather"},
        f"{path}:metrics",
    )
    fixed = metrics["fixed32"]
    exact_keys(
        fixed,
        {
            "pure_decode_forward_steps",
            "complete_work_census_events",
            "complete_spec_rows",
            "spec_drafts",
            "spec_tokens",
            "batch_histogram",
            "first_forward_step",
            "last_forward_step",
            "events_sha256",
        },
        f"{path}:fixed32",
    )
    steps = strict_nonnegative_int(
        fixed["pure_decode_forward_steps"],
        label=f"{path}:fixed32.pure_decode_forward_steps",
    )
    events = strict_nonnegative_int(
        fixed["complete_work_census_events"],
        label=f"{path}:fixed32.complete_work_census_events",
    )
    complete_spec_rows = strict_nonnegative_int(
        fixed["complete_spec_rows"],
        label=f"{path}:fixed32.complete_spec_rows",
    )
    spec_drafts = strict_nonnegative_int(
        fixed["spec_drafts"],
        label=f"{path}:fixed32.spec_drafts",
    )
    spec_tokens = strict_nonnegative_int(
        fixed["spec_tokens"],
        label=f"{path}:fixed32.spec_tokens",
    )
    first_forward_step = strict_optional_nonnegative_int(
        fixed["first_forward_step"],
        label=f"{path}:fixed32.first_forward_step",
    )
    last_forward_step = strict_optional_nonnegative_int(
        fixed["last_forward_step"],
        label=f"{path}:fixed32.last_forward_step",
    )
    if (
        steps != ack["counters"]["pure_decode_forward_steps"]
        or events != ack["counters"]["complete_work_census_events"]
        or first_forward_step
        != ack["counters"]["work_census_first_forward_step"]
        or last_forward_step
        != ack["counters"]["work_census_last_forward_step"]
        or complete_spec_rows != spec_drafts
        or spec_tokens != 31 * spec_drafts
    ):
        raise GateError(f"{path}: fixed counters do not reconcile")
    histogram = fixed["batch_histogram"]
    batch_counts_raw = strict_nonnegative_int_map(
        histogram,
        expected_keys={"1", "2", "3", "4"},
        label=f"{path}:fixed32.batch_histogram",
    )
    batch_counts = {
        int(batch): count for batch, count in batch_counts_raw.items()
    }
    if (
        sum(batch_counts.values()) != events
        or sum(batch * count for batch, count in batch_counts.items()) != spec_drafts
        or any(
            batch > server_capacity and count
            for batch, count in batch_counts.items()
        )
    ):
        raise GateError(f"{path}: batch histogram does not reconcile")
    if (
        not isinstance(fixed["events_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", fixed["events_sha256"]) is None
    ):
        raise GateError(f"{path}: fixed32 events digest is malformed")

    sfwd = metrics["sfwd"]
    exact_keys(
        sfwd,
        {
            "gpu_seconds",
            "steps",
            "drafts",
            "wall_seconds",
            "wall_drafts",
            "wall_steps",
            "wall_rejected",
        },
        f"{path}:sfwd",
    )
    sfwd_values: dict[str, float | int] = {
        "gpu_seconds": strict_nonnegative_number(
            sfwd["gpu_seconds"],
            label=f"{path}:sfwd.gpu_seconds",
        ),
        "wall_seconds": strict_nonnegative_number(
            sfwd["wall_seconds"],
            label=f"{path}:sfwd.wall_seconds",
        ),
    }
    for key in ("steps", "drafts", "wall_drafts", "wall_steps", "wall_rejected"):
        sfwd_values[key] = strict_nonnegative_int(
            sfwd[key],
            label=f"{path}:sfwd.{key}",
        )
    if (
        sfwd_values["steps"] != steps
        or sfwd_values["drafts"] != spec_drafts
    ):
        raise GateError(f"{path}: SFWD counters do not reconcile")
    for label in ("dfwd", "cfwd"):
        span = metrics[label]
        exact_keys(span, {"gpu_seconds", "spans"}, f"{path}:{label}")
        strict_nonnegative_number(
            span["gpu_seconds"],
            label=f"{path}:{label}.gpu_seconds",
        )
        if strict_nonnegative_int(
            span["spans"],
            label=f"{path}:{label}.spans",
        ) != events:
            raise GateError(f"{path}: {label} spans do not reconcile")

    committer = metrics["committer"]
    pregather = metrics["conv_pregather"]
    expected_by_batch = {
        str(batch): batch_counts[batch] for batch in range(1, 5)
    }
    expected_capture_by_batch = {
        str(batch): int(batch <= server_capacity) for batch in range(1, 5)
    }
    zero_by_batch = {str(batch): 0 for batch in range(1, 5)}
    expected_ready_capacities = {
        str(batch): server_capacity
        for batch in range(1, server_capacity + 1)
    }
    expected_preseeded_batches = list(range(1, server_capacity + 1))
    batch_keys = {"1", "2", "3", "4"}
    if not isinstance(committer, dict) or not isinstance(pregather, dict):
        raise GateError(f"{path}: fixed32 counter groups must be objects")
    exact_keys(
        committer,
        {
            "actual_replays_by_batch",
            "actual_replays_enqueued",
            "all_batches_ready",
            "captures",
            "fast_route_ready",
            "maximum_ready_capacity",
            "nonpure_committer_replays_by_batch",
            "nonpure_committer_replays_enqueued",
            "nonpure_dispatch",
            "preseeded_batches",
            "preseeded_graphs",
            "ready_capacities",
            "required_capacity",
        },
        f"{path}:committer",
    )
    exact_keys(
        pregather,
        {
            "preseeded",
            "pointer_entries",
            "preseeded_batches",
            "max_batch_size",
            "graph_capture_stages",
            "graph_capture_stages_by_batch",
            "profile_capture_stages",
            "aux_capture_stages",
            "actual_stages",
            "actual_stages_by_batch",
            "graph_replay_stages",
            "graph_replay_stages_by_batch",
        },
        f"{path}:conv_pregather",
    )
    for key in (
        "actual_replays_enqueued",
        "captures",
        "maximum_ready_capacity",
        "nonpure_committer_replays_enqueued",
        "preseeded_graphs",
        "required_capacity",
    ):
        strict_nonnegative_int(
            committer[key],
            label=f"{path}:committer.{key}",
        )
    for key in (
        "pointer_entries",
        "max_batch_size",
        "graph_capture_stages",
        "profile_capture_stages",
        "aux_capture_stages",
        "actual_stages",
        "graph_replay_stages",
    ):
        strict_nonnegative_int(
            pregather[key],
            label=f"{path}:conv_pregather.{key}",
        )
    committer_by_batch = strict_nonnegative_int_map(
        committer["actual_replays_by_batch"],
        expected_keys=batch_keys,
        label=f"{path}:committer.actual_replays_by_batch",
    )
    nonpure_by_batch = strict_nonnegative_int_map(
        committer["nonpure_committer_replays_by_batch"],
        expected_keys=batch_keys,
        label=(
            f"{path}:committer.nonpure_committer_replays_by_batch"
        ),
    )
    nonpure_dispatch = strict_nonnegative_int_map(
        committer["nonpure_dispatch"],
        expected_keys={
            "guarded_steps",
            "piecewise_steps",
            "none_steps",
            "forbidden_full_steps",
        },
        label=f"{path}:committer.nonpure_dispatch",
    )
    committer_ready_capacities = strict_nonnegative_int_map(
        committer["ready_capacities"],
        expected_keys=set(expected_ready_capacities),
        label=f"{path}:committer.ready_capacities",
    )
    committer_preseeded_batches = strict_nonnegative_int_list(
        committer["preseeded_batches"],
        label=f"{path}:committer.preseeded_batches",
    )
    capture_by_batch = strict_nonnegative_int_map(
        pregather["graph_capture_stages_by_batch"],
        expected_keys=batch_keys,
        label=f"{path}:conv_pregather.graph_capture_stages_by_batch",
    )
    host_by_batch = strict_nonnegative_int_map(
        pregather["actual_stages_by_batch"],
        expected_keys=batch_keys,
        label=f"{path}:conv_pregather.actual_stages_by_batch",
    )
    replay_by_batch = strict_nonnegative_int_map(
        pregather["graph_replay_stages_by_batch"],
        expected_keys=batch_keys,
        label=f"{path}:conv_pregather.graph_replay_stages_by_batch",
    )
    pregather_preseeded_batches = strict_nonnegative_int_list(
        pregather["preseeded_batches"],
        label=f"{path}:conv_pregather.preseeded_batches",
    )
    nonpure_replays = committer[
        "nonpure_committer_replays_enqueued"
    ]
    expected_raw_by_batch = {
        str(batch): expected_by_batch[str(batch)]
        + nonpure_by_batch[str(batch)]
        for batch in range(1, 5)
    }
    if (
        committer["all_batches_ready"] is not True
        or committer["fast_route_ready"] is not True
        or committer["required_capacity"] != server_capacity
        or committer["maximum_ready_capacity"] != server_capacity
        or committer["captures"] != server_capacity
        or committer["preseeded_graphs"] != server_capacity
        or committer_preseeded_batches != expected_preseeded_batches
        or committer_ready_capacities != expected_ready_capacities
        or nonpure_dispatch["guarded_steps"]
        != (
            nonpure_dispatch["piecewise_steps"]
            + nonpure_dispatch["none_steps"]
            + nonpure_dispatch["forbidden_full_steps"]
        )
        or nonpure_dispatch["forbidden_full_steps"] != 0
        or nonpure_replays != sum(nonpure_by_batch.values())
        or nonpure_replays > nonpure_dispatch["guarded_steps"]
        or any(
            nonpure_by_batch[str(batch)]
            for batch in range(server_capacity, 5)
        )
        or committer["actual_replays_enqueued"]
        != events + nonpure_replays
        or committer_by_batch != expected_raw_by_batch
        or pregather["preseeded"] is not True
        or pregather["pointer_entries"] != 48
        or pregather_preseeded_batches != expected_preseeded_batches
        or pregather["max_batch_size"] != server_capacity
        or pregather["graph_capture_stages"] != server_capacity
        or capture_by_batch != expected_capture_by_batch
        or pregather["profile_capture_stages"] != 0
        or pregather["aux_capture_stages"] != 0
        or pregather["actual_stages"] != 0
        or host_by_batch != zero_by_batch
        or pregather["graph_replay_stages"] != events
        or replay_by_batch != expected_by_batch
    ):
        raise GateError(
            f"{path}: committer/nonpure/in-graph pregather counters do not reconcile"
        )

    census_lines = read_text(census_path).splitlines()
    if len(census_lines) < events + 1:
        raise GateError(f"{path}: census stream is shorter than snapshot prefix")
    try:
        census_prefix = [json.loads(line) for line in census_lines[:events]]
    except json.JSONDecodeError as error:
        raise GateError(f"{census_path}: invalid JSONL: {error}") from error
    expected_events_hash = hashlib.sha256(
        json.dumps(
            census_prefix,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if fixed["events_sha256"] != expected_events_hash:
        raise GateError(f"{path}: census prefix digest mismatch")

    if metrics_path is not None:
        if metric_values is None:
            raise GateError(f"{metrics_path}: cached parsed metrics are missing")
        observed = metric_values
        expected_metrics = {
            "fwd_s": sfwd_values["gpu_seconds"],
            "fwd_steps": float(steps),
            "fwd_drafts": float(spec_drafts),
            "wall_s": sfwd_values["wall_seconds"],
            "wall_drafts": sfwd_values["wall_drafts"],
            "wall_steps": sfwd_values["wall_steps"],
            "wall_attempts": (
                sfwd_values["wall_steps"] + sfwd_values["wall_rejected"]
            ),
            "wall_rejected": sfwd_values["wall_rejected"],
            "spec_drafts": float(spec_drafts),
            "spec_tokens": float(31 * spec_drafts),
        }
        for key, expected in expected_metrics.items():
            if not math.isclose(
                observed[key], expected, rel_tol=0.0, abs_tol=1e-9
            ):
                raise GateError(
                    f"{metrics_path}: metric {key} differs from frozen "
                    "runtime snapshot"
                )
    return {
        "path": str(path),
        "sha256": expected_reference["sha256"],
        "generation": ack["generation"],
        "events_sha256": expected_events_hash,
        "committer": {
            "actual_replays_enqueued": committer[
                "actual_replays_enqueued"
            ],
            "actual_replays_by_batch": committer_by_batch,
            "nonpure_committer_replays_enqueued": nonpure_replays,
            "nonpure_committer_replays_by_batch": nonpure_by_batch,
            "nonpure_dispatch": nonpure_dispatch,
        },
    }


def validate_subset(path: Path, task_count: int) -> dict[str, Any]:
    if task_count not in EVIDENCE_SETS:
        raise GateError(f"{path}: unsupported canonical subset size {task_count}")
    expected = EVIDENCE_SETS[task_count]
    actual_hash = sha256_file(path)
    if actual_hash != expected["sha256"]:
        raise GateError(
            f"{path}: canonical subset sha256 mismatch; "
            f"expected {expected['sha256']}, got {actual_hash}"
        )
    payload = exact_json(path, label=f"{path}: canonical subset")
    if payload.get("dataset_name") != "princeton-nlp/SWE-bench_Verified":
        raise GateError(f"{path}: subset is not SWE-bench_Verified")
    if payload.get("split") != "test":
        raise GateError(f"{path}: canonical subset split is not test")
    actual_ids = payload.get("instance_ids")
    if actual_ids != list(expected["task_ids"]):
        raise GateError(
            f"{path}: subset IDs do not exactly match canonical {task_count}-task set"
        )
    return {
        "path": str(path),
        "sha256": actual_hash,
        "task_ids": list(expected["task_ids"]),
    }


def validate_canonical_subset(path: Path) -> dict[str, Any]:
    """Bind an exact4/exact16 subset by pinned bytes and parsed task identity."""
    actual_hash = sha256_file(path)
    matching_counts = [
        task_count
        for task_count, expected in EVIDENCE_SETS.items()
        if actual_hash == expected["sha256"]
    ]
    if len(matching_counts) != 1:
        raise GateError(
            f"{path}: fixed32 subset SHA-256 is not canonical exact4/exact16; "
            f"got {actual_hash}"
        )
    task_count = matching_counts[0]
    binding = validate_subset(path, task_count)
    return {"task_count": task_count, **binding}


def parse_orchestrator(arm_dir: Path, task_count: int) -> dict[str, Any]:
    path = arm_dir / "swe_orchestrator.log"
    lines = read_text(path).splitlines()
    headers = [match for line in lines if (match := ORCHESTRATOR_HEADER_RE.match(line))]
    done = [match for line in lines if (match := ORCHESTRATOR_DONE_RE.match(line))]
    if len(headers) != 1 or len(done) != 1:
        raise GateError(
            f"{path}: expected exactly one campaign header and one DONE footer"
        )
    header = headers[0]
    recorded_tasks = int(header.group("tasks"))
    concurrency = int(header.group("concurrency"))
    if header.group("dataset") != "princeton-nlp/SWE-bench_Verified":
        raise GateError(f"{path}: campaign is not SWE-bench_Verified")
    if recorded_tasks != task_count or int(done[0].group("tasks")) != task_count:
        raise GateError(
            f"{path}: requested {task_count} tasks but header/footer do not match"
        )
    if concurrency not in (1, 4):
        raise GateError(f"{path}: unsupported inferred concurrency {concurrency}")
    expected_ids = sorted(EVIDENCE_SETS[task_count]["task_ids"])
    starts = sorted(
        match.group("task") for line in lines if (match := TASK_START_RE.match(line))
    )
    ends = sorted(
        match.group("task") for line in lines if (match := TASK_END_RE.match(line))
    )
    if starts != expected_ids or ends != expected_ids:
        raise GateError(
            f"{path}: start/completion records are not the canonical completed set"
        )
    return {
        "path": str(path),
        "inferred_concurrency": concurrency,
        "completed_task_ids": expected_ids,
    }


def task_directories(arm_dir: Path, task_count: int) -> list[Path]:
    root = arm_dir / "swe_out" / "verified" / "per_task"
    if not root.is_dir():
        raise GateError(f"missing task artifact directory: {root}")
    directories = sorted(path for path in root.iterdir() if path.is_dir())
    expected_ids = sorted(EVIDENCE_SETS[task_count]["task_ids"])
    if [path.name for path in directories] != expected_ids:
        raise GateError(
            f"{root}: task directories are not the exact canonical completed set"
        )
    for task_dir in directories:
        metadata_path = task_dir / "runner_metadata.json"
        metadata = exact_json(metadata_path, label=str(metadata_path))
        if metadata.get("instance_id") != task_dir.name or not metadata.get("ended_at"):
            raise GateError(f"{metadata_path}: task is not recorded as completed")
        for name in ("vllm_metrics_pre.txt", "vllm_metrics_post.txt"):
            if not (task_dir / name).is_file():
                raise GateError(f"{task_dir}: incomplete metrics bracket")
    return directories


def fixed32_task_key_id(task_id: str) -> str:
    return hashlib.sha256(
        b"fr13-fixed32-task-key-id-v1\0" + task_id.encode("utf-8")
    ).hexdigest()


def fixed32_canonical_task_set_sha256(task_ids: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(
            sorted(task_ids),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _fixed32_digest(value: object, *, label: str, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise GateError(f"{label}: expected a lowercase SHA-256 digest")
    return value


def _fixed32_ingress_record_contract(
    row: dict[str, Any],
    *,
    role: str,
    canonical_task_keys: set[str],
    canonical_task_set_sha256: str,
    label: str,
) -> None:
    event = row["event"]
    route = row["route"]
    task_key_id = row["task_key_id"]
    logical_id = row["logical_id_sha256"]
    wire_id = row["wire_id_sha256"]
    engine_id = row["engine_request_id_sha256"]
    status_code = row["status_code"]
    outcome = row["outcome"]
    reason = row["reason"]
    evidence = row["evidence_sha256"]

    if route is not None and route not in {"chat", "responses"}:
        raise GateError(f"{label}: invalid ingress route")
    if task_key_id is not None and task_key_id not in canonical_task_keys:
        raise GateError(f"{label}: noncanonical task key")
    if status_code is not None and (
        isinstance(status_code, bool)
        or not isinstance(status_code, int)
        or not 100 <= status_code <= 599
    ):
        raise GateError(f"{label}: invalid HTTP status")

    if event == "request_rejected":
        allowed_reasons = (
            {
                "missing_bearer",
                "malformed_bearer",
                "unknown_task_key",
                "invalid_task_bearer",
                "campaign_not_active",
                "campaign_finalized",
            }
            if role == "proxy"
            else {
                "missing_bearer",
                "invalid_engine_bearer",
                "campaign_not_active",
                "campaign_finalized",
                "invalid_task_key",
                "invalid_wire_id",
                "request_id_mismatch",
                "invalid_json",
                "body_too_large",
            }
        )
        if (
            route not in {"chat", "responses"}
            or logical_id is not None
            or wire_id is not None
            or engine_id is not None
            or status_code is not None
            or outcome != "rejected"
            or reason not in allowed_reasons
            or evidence is not None
        ):
            raise GateError(f"{label}: rejected-request record contract mismatch")
        return

    if event in {"campaign_begin", "campaign_finalize"}:
        expected_outcome = "begun" if event == "campaign_begin" else "finalized"
        if (
            route is not None
            or task_key_id is not None
            or logical_id is not None
            or wire_id is not None
            or engine_id is not None
            or status_code is not None
            or outcome != expected_outcome
            or reason is not None
            or evidence != canonical_task_set_sha256
        ):
            raise GateError(f"{label}: campaign lifecycle record mismatch")
        return

    if (
        route not in {"chat", "responses"}
        or task_key_id not in canonical_task_keys
    ):
        raise GateError(f"{label}: campaign request identity mismatch")

    if role == "proxy" and event == "logical_begin":
        if (
            logical_id is None
            or wire_id is not None
            or engine_id is not None
            or status_code is not None
            or outcome != "accepted"
            or reason is not None
            or evidence is not None
        ):
            raise GateError(f"{label}: logical-begin record mismatch")
        return
    if role == "proxy" and event == "logical_complete":
        if (
            logical_id is None
            or wire_id is not None
            or engine_id is not None
            or status_code is not None
            or (
                (outcome, reason)
                not in {
                    ("completed", None),
                    ("aborted", "handler_error"),
                    ("aborted", "no_completed_attempt"),
                }
            )
            or evidence is not None
        ):
            raise GateError(f"{label}: logical-complete record mismatch")
        return
    if role == "proxy" and event in {"attempt_begin", "attempt_result"}:
        if (
            logical_id is None
            or wire_id is None
            or engine_id is None
            or evidence is None
        ):
            raise GateError(f"{label}: upstream-attempt identity is incomplete")
        if event == "attempt_begin":
            if (
                status_code is not None
                or outcome != "dispatched"
                or reason is not None
            ):
                raise GateError(f"{label}: attempt-begin record mismatch")
        elif (
            (outcome == "response" and status_code is not None and reason is None)
            or (
                outcome == "exception"
                and status_code is None
                and reason == "network_error"
            )
        ):
            pass
        else:
            raise GateError(f"{label}: attempt-result record mismatch")
        return
    if role == "engine" and event in {"request_accepted", "request_complete"}:
        if (
            logical_id is not None
            or wire_id is None
            or engine_id is None
            or status_code is not None
            or evidence is None
        ):
            raise GateError(f"{label}: engine-request identity is incomplete")
        if event == "request_accepted":
            if outcome != "accepted" or reason is not None:
                raise GateError(f"{label}: engine acceptance record mismatch")
        elif (outcome, reason) not in {
            ("completed", None),
            ("exception", "app_error"),
        }:
            raise GateError(f"{label}: engine completion record mismatch")
        return
    raise GateError(f"{label}: event is invalid for role {role}")


def load_fixed32_ingress_ledger(
    path: Path,
    *,
    role: str,
    canonical_task_keys: set[str],
    canonical_task_set_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise GateError(f"{path}: ingress ledger must be a regular file")
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise GateError(f"{path}: ingress ledger is empty or truncated")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise GateError(f"{path}: ingress ledger is not strict UTF-8") from error
    lines = text.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise GateError(f"{path}: ingress ledger contains a blank record")

    allowed_events = (
        {
            "request_rejected",
            "campaign_begin",
            "logical_begin",
            "attempt_begin",
            "attempt_result",
            "logical_complete",
            "campaign_finalize",
        }
        if role == "proxy"
        else {
            "request_rejected",
            "campaign_begin",
            "request_accepted",
            "request_complete",
            "campaign_finalize",
        }
    )
    rows: list[dict[str, Any]] = []
    previous = "0" * 64
    phase = "preflight"
    active_logical: dict[str, tuple[str, str]] = {}
    active_attempts: dict[str, tuple[str, str, str, str, str]] = {}
    active_engine: dict[str, tuple[str, str, str, str]] = {}
    began = False
    finalized = False
    for sequence, line in enumerate(lines):
        label = f"{path}:{sequence + 1}"
        row = exact_json_text(line, label=label)
        exact_keys(row, FIXED32_INGRESS_LEDGER_KEYS, label)
        if (
            row["schema"] != FIXED32_INGRESS_LEDGER_SCHEMA
            or type(row["seq"]) is not int
            or row["seq"] != sequence
            or row["role"] != role
            or row["phase"] != phase
            or row["event"] not in allowed_events
            or row["prev_sha256"] != previous
        ):
            raise GateError(f"{label}: ingress ledger chain metadata mismatch")
        for key in (
            "task_key_id",
            "logical_id_sha256",
            "wire_id_sha256",
            "engine_request_id_sha256",
            "evidence_sha256",
        ):
            _fixed32_digest(row[key], label=f"{label}.{key}", optional=True)
        _fixed32_digest(row["prev_sha256"], label=f"{label}.prev_sha256")
        claimed = _fixed32_digest(
            row["record_sha256"], label=f"{label}.record_sha256"
        )
        unsigned = dict(row)
        del unsigned["record_sha256"]
        actual = hashlib.sha256(
            json.dumps(
                unsigned,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if claimed != actual:
            raise GateError(f"{label}: ingress ledger record digest mismatch")
        _fixed32_ingress_record_contract(
            row,
            role=role,
            canonical_task_keys=canonical_task_keys,
            canonical_task_set_sha256=canonical_task_set_sha256,
            label=label,
        )

        event = row["event"]
        if event == "campaign_begin":
            if began or phase != "preflight":
                raise GateError(f"{label}: duplicate campaign begin")
            began = True
            phase = "campaign"
        elif event == "campaign_finalize":
            if (
                not began
                or finalized
                or active_logical
                or active_attempts
                or active_engine
            ):
                raise GateError(f"{label}: campaign finalized with active work")
            finalized = True
            phase = "finalized"
        elif event == "logical_begin":
            logical = row["logical_id_sha256"]
            if phase != "campaign" or logical in active_logical:
                raise GateError(f"{label}: duplicate/inactive logical request")
            active_logical[str(logical)] = (row["route"], row["task_key_id"])
        elif event == "attempt_begin":
            logical = str(row["logical_id_sha256"])
            wire = str(row["wire_id_sha256"])
            if logical not in active_logical or wire in active_attempts:
                raise GateError(f"{label}: attempt has no unique active logical")
            if active_logical[logical] != (row["route"], row["task_key_id"]):
                raise GateError(f"{label}: attempt changed logical ownership")
            active_attempts[wire] = (
                logical,
                row["route"],
                row["task_key_id"],
                row["engine_request_id_sha256"],
                row["evidence_sha256"],
            )
        elif event == "attempt_result":
            wire = str(row["wire_id_sha256"])
            expected = active_attempts.pop(wire, None)
            observed = (
                str(row["logical_id_sha256"]),
                row["route"],
                row["task_key_id"],
                row["engine_request_id_sha256"],
                row["evidence_sha256"],
            )
            if expected != observed:
                raise GateError(f"{label}: attempt result changed identity")
        elif event == "logical_complete":
            logical = str(row["logical_id_sha256"])
            if (
                active_logical.get(logical)
                != (row["route"], row["task_key_id"])
                or any(item[0] == logical for item in active_attempts.values())
            ):
                raise GateError(f"{label}: logical completion changed identity")
            del active_logical[logical]
        elif event == "request_accepted":
            engine = str(row["engine_request_id_sha256"])
            if phase != "campaign" or engine in active_engine:
                raise GateError(f"{label}: duplicate/inactive engine request")
            active_engine[engine] = (
                row["route"],
                row["task_key_id"],
                row["wire_id_sha256"],
                row["evidence_sha256"],
            )
        elif event == "request_complete":
            engine = str(row["engine_request_id_sha256"])
            expected = active_engine.pop(engine, None)
            observed = (
                row["route"],
                row["task_key_id"],
                row["wire_id_sha256"],
                row["evidence_sha256"],
            )
            if expected != observed:
                raise GateError(f"{label}: engine completion changed identity")
        previous = str(claimed)
        rows.append(row)
    if (
        not began
        or not finalized
        or phase != "finalized"
        or active_logical
        or active_attempts
        or active_engine
    ):
        raise GateError(f"{path}: ingress ledger is not exactly finalized")
    return rows, {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "records": len(rows),
        "chain_head_sha256": previous,
    }


def _fixed32_artifact_identity(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise GateError(f"{path}: expected a regular artifact")
    raw = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _validate_fixed32_ingress_preflight(
    arm_dir: Path,
    *,
    role: str,
) -> dict[str, Any]:
    path = arm_dir / f"fixed32_{role}_ingress_preflight.json"
    payload = exact_json(path, label=str(path))
    expected_requests = [
        {
            "route": route,
            "auth_case": auth_case,
            "status_code": 401,
        }
        for route in ("/v1/chat/completions", "/v1/responses")
        for auth_case in ("missing_bearer", "wrong_bearer")
    ]
    expected: dict[str, Any] = {
        "schema": FIXED32_INGRESS_PREFLIGHT_SCHEMA,
        "role": role,
        "rejected_requests": 4,
        "accepted_requests": 0,
        "requests": expected_requests,
        "denied_alternate_routes": [
            {"method": "POST", "route": route, "status_code": 403}
            for route in (
                (
                    "/admin/invalidate",
                    "/admin/load_tuned_config",
                )
                if role == "proxy"
                else ("/v1/completions", "/reset_prefix_cache")
            )
        ],
    }
    if role == "engine":
        expected["non_inference_bypass"] = [
            {"route": route, "status_code": 200}
            for route in ("/health", "/metrics", "/v1/models")
        ]
    if payload != expected:
        raise GateError(
            f"{path}: fixed32 ingress preflight differs: "
            f"{first_json_difference(payload, expected)}"
        )
    return _fixed32_artifact_identity(path) | {
        "schema": FIXED32_INGRESS_PREFLIGHT_SCHEMA,
        "role": role,
        "rejected_requests": 4,
        "accepted_requests": 0,
    }


def _fixed32_ingress_task_counts(
    rows: list[dict[str, Any]],
    *,
    role: str,
    canonical_task_keys: set[str],
) -> dict[str, dict[str, int]]:
    if role == "proxy":
        counts = {
            key_id: {
                "accepted_logical_requests": 0,
                "completed_logical_model_requests": 0,
                "aborted_logical_requests": 0,
                "accepted_attempts": 0,
                "completed_attempts": 0,
                "failed_attempts": 0,
            }
            for key_id in sorted(canonical_task_keys)
        }
        for row in rows:
            task_key_id = row["task_key_id"]
            if row["event"] == "logical_begin":
                counts[task_key_id]["accepted_logical_requests"] += 1
            elif row["event"] == "logical_complete" and row["outcome"] == "completed":
                counts[task_key_id]["completed_logical_model_requests"] += 1
            elif row["event"] == "logical_complete" and row["outcome"] == "aborted":
                counts[task_key_id]["aborted_logical_requests"] += 1
            elif row["event"] == "attempt_begin":
                counts[task_key_id]["accepted_attempts"] += 1
            elif row["event"] == "attempt_result":
                key = (
                    "failed_attempts"
                    if row["outcome"] == "exception"
                    else "completed_attempts"
                )
                counts[task_key_id][key] += 1
        return counts
    counts = {
        key_id: {
            "accepted_engine_requests": 0,
            "completed_engine_requests": 0,
        }
        for key_id in sorted(canonical_task_keys)
    }
    for row in rows:
        task_key_id = row["task_key_id"]
        if row["event"] == "request_accepted":
            counts[task_key_id]["accepted_engine_requests"] += 1
        elif row["event"] == "request_complete" and row["outcome"] == "completed":
            counts[task_key_id]["completed_engine_requests"] += 1
    return counts


def _validate_fixed32_ingress_reports(
    arm_dir: Path,
    *,
    role: str,
    task_ids: list[str],
    task_counts: dict[str, dict[str, int]],
    rows: list[dict[str, Any]],
    ledger_identity: dict[str, Any],
    canonical_task_set_sha256: str,
) -> dict[str, Any]:
    begin_path = arm_dir / f"fixed32_{role}_ingress_begin.json"
    begin = exact_json(begin_path, label=str(begin_path))
    begin_index = next(
        (
            index
            for index, row in enumerate(rows)
            if row["event"] == "campaign_begin"
        ),
        -1,
    )
    begin_schema = (
        FIXED32_PROXY_INGRESS_BEGIN_SCHEMA
        if role == "proxy"
        else FIXED32_ENGINE_INGRESS_BEGIN_SCHEMA
    )
    expected_begin = {
        "schema": begin_schema,
        "role": role,
        "phase": "campaign",
        "canonical_task_count": len(task_ids),
        "canonical_task_set_sha256": canonical_task_set_sha256,
        "preflight_rejected_requests": 4,
        "ledger_records": begin_index + 1,
        "ledger_chain_head_sha256": rows[begin_index]["record_sha256"],
    }
    if begin != expected_begin:
        raise GateError(
            f"{begin_path}: fixed32 ingress begin report differs: "
            f"{first_json_difference(begin, expected_begin)}"
        )

    preflight_rows = rows[:begin_index]
    expected_preflight_reasons = [
        (route, reason)
        for route in ("chat", "responses")
        for reason in (
            ("missing_bearer", "malformed_bearer")
            if role == "proxy"
            else ("missing_bearer", "invalid_engine_bearer")
        )
    ]
    observed_preflight_reasons = [
        (row["route"], row["reason"]) for row in preflight_rows
    ]
    if (
        len(preflight_rows) != 4
        or any(
            row["event"] != "request_rejected"
            or row["phase"] != "preflight"
            or row["task_key_id"] is not None
            for row in preflight_rows
        )
        or observed_preflight_reasons != expected_preflight_reasons
    ):
        raise GateError(
            f"{arm_dir}: {role} ingress preflight ledger is not the exact "
            "four rejected auth checks"
        )
    campaign_rejected = sum(
        row["event"] == "request_rejected" and row["phase"] == "campaign"
        for row in rows
    )
    if campaign_rejected != 0:
        raise GateError(
            f"{arm_dir}: {role} ingress rejected campaign inference traffic"
        )

    finalize_path = arm_dir / f"fixed32_{role}_ingress_finalize.json"
    finalize = exact_json(finalize_path, label=str(finalize_path))
    task_evidence = [
        {"task_key_id": key_id, **counts}
        for key_id, counts in sorted(task_counts.items())
    ]
    totals = {
        key: sum(counts[key] for counts in task_counts.values())
        for key in next(iter(task_counts.values()))
    }
    common = {
        "role": role,
        "phase": "finalized",
        "canonical_task_count": len(task_ids),
        "canonical_task_set_sha256": canonical_task_set_sha256,
        "active_requests": 0,
        "preflight_rejected_requests": 4,
        "campaign_rejected_requests": 0,
        "task_evidence": task_evidence,
        "ledger_records": ledger_identity["records"],
        "ledger_chain_head_sha256": ledger_identity["chain_head_sha256"],
    }
    if role == "proxy":
        expected_finalize = {
            "schema": FIXED32_PROXY_INGRESS_FINALIZE_SCHEMA,
            **common,
            "active_attempts": 0,
            "accepted_logical_requests": totals["accepted_logical_requests"],
            "completed_logical_requests": totals[
                "completed_logical_model_requests"
            ],
            "aborted_logical_requests": totals["aborted_logical_requests"],
            "accepted_attempts": totals["accepted_attempts"],
            "completed_attempts": totals["completed_attempts"],
            "failed_attempts": totals["failed_attempts"],
        }
    else:
        expected_finalize = {
            "schema": FIXED32_ENGINE_INGRESS_FINALIZE_SCHEMA,
            **common,
            "accepted_engine_requests": totals["accepted_engine_requests"],
            "completed_engine_requests": totals["completed_engine_requests"],
        }
    if finalize != expected_finalize:
        raise GateError(
            f"{finalize_path}: fixed32 ingress finalize report differs: "
            f"{first_json_difference(finalize, expected_finalize)}"
        )
    return {
        "begin": _fixed32_artifact_identity(begin_path)
        | {
            "schema": begin_schema,
            "ledger_records": begin["ledger_records"],
            "ledger_chain_head_sha256": begin[
                "ledger_chain_head_sha256"
            ],
        },
        "finalize": _fixed32_artifact_identity(finalize_path)
        | {
            "schema": expected_finalize["schema"],
            "ledger_records": finalize["ledger_records"],
            "ledger_chain_head_sha256": finalize[
                "ledger_chain_head_sha256"
            ],
        },
        "task_counts": task_counts,
        "totals": totals,
    }


def validate_fixed32_ingress_and_census(
    arm_dir: Path,
    *,
    mode: str,
    task_ids: list[str],
    task_bindings: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    canonical_task_keys = {
        fixed32_task_key_id(task_id) for task_id in task_ids
    }
    canonical_task_set_sha256 = fixed32_canonical_task_set_sha256(task_ids)
    task_by_key = {
        fixed32_task_key_id(task_id): task_id for task_id in task_ids
    }
    if set(task_bindings) != set(task_ids) or any(
        binding.get("task_key_id") != fixed32_task_key_id(task_id)
        for task_id, binding in task_bindings.items()
    ):
        raise GateError(f"{arm_dir}: task provenance key binding is not canonical")

    preflights = {
        role: _validate_fixed32_ingress_preflight(arm_dir, role=role)
        for role in ("proxy", "engine")
    }
    ledgers: dict[str, list[dict[str, Any]]] = {}
    ledger_identities: dict[str, dict[str, Any]] = {}
    reports: dict[str, dict[str, Any]] = {}
    task_counts: dict[str, dict[str, dict[str, int]]] = {}
    for role, path in (
        (
            "proxy",
            arm_dir / "logs" / "fr13_fixed32_proxy_ingress.jsonl",
        ),
        (
            "engine",
            arm_dir / "logs" / "fr13_fixed32_engine_ingress.jsonl",
        ),
    ):
        rows, identity = load_fixed32_ingress_ledger(
            path,
            role=role,
            canonical_task_keys=canonical_task_keys,
            canonical_task_set_sha256=canonical_task_set_sha256,
        )
        counts = _fixed32_ingress_task_counts(
            rows,
            role=role,
            canonical_task_keys=canonical_task_keys,
        )
        ledgers[role] = rows
        ledger_identities[role] = identity
        task_counts[role] = counts
        reports[role] = _validate_fixed32_ingress_reports(
            arm_dir,
            role=role,
            task_ids=task_ids,
            task_counts=counts,
            rows=rows,
            ledger_identity=identity,
            canonical_task_set_sha256=canonical_task_set_sha256,
        )

    proxy_attempts = {
        row["wire_id_sha256"]: row
        for row in ledgers["proxy"]
        if row["event"] == "attempt_begin"
    }
    proxy_results = {
        row["wire_id_sha256"]: row
        for row in ledgers["proxy"]
        if row["event"] == "attempt_result"
    }
    engine_accepts = {
        row["wire_id_sha256"]: row
        for row in ledgers["engine"]
        if row["event"] == "request_accepted"
    }
    engine_completes = {
        row["wire_id_sha256"]: row
        for row in ledgers["engine"]
        if row["event"] == "request_complete"
    }
    attempt_keys = set(proxy_attempts)
    if (
        len(proxy_attempts)
        != sum(row["event"] == "attempt_begin" for row in ledgers["proxy"])
        or len(proxy_results)
        != sum(row["event"] == "attempt_result" for row in ledgers["proxy"])
        or len(engine_accepts)
        != sum(row["event"] == "request_accepted" for row in ledgers["engine"])
        or len(engine_completes)
        != sum(row["event"] == "request_complete" for row in ledgers["engine"])
        or attempt_keys != set(proxy_results)
        or attempt_keys != set(engine_accepts)
        or attempt_keys != set(engine_completes)
    ):
        raise GateError(
            f"{arm_dir}: proxy and engine attempt census is not one-to-one"
        )

    logical_successes: dict[str, str] = {}
    successful_engine_ids: dict[str, str] = {}
    attempts_by_logical: dict[str, list[dict[str, Any]]] = {}
    for wire_id in sorted(attempt_keys):
        attempt = proxy_attempts[wire_id]
        result = proxy_results[wire_id]
        accepted = engine_accepts[wire_id]
        completed = engine_completes[wire_id]
        proxy_identity = (
            attempt["route"],
            attempt["task_key_id"],
            attempt["wire_id_sha256"],
            attempt["engine_request_id_sha256"],
            attempt["evidence_sha256"],
        )
        if proxy_identity != (
            result["route"],
            result["task_key_id"],
            result["wire_id_sha256"],
            result["engine_request_id_sha256"],
            result["evidence_sha256"],
        ) or proxy_identity != (
            accepted["route"],
            accepted["task_key_id"],
            accepted["wire_id_sha256"],
            accepted["engine_request_id_sha256"],
            accepted["evidence_sha256"],
        ) or proxy_identity != (
            completed["route"],
            completed["task_key_id"],
            completed["wire_id_sha256"],
            completed["engine_request_id_sha256"],
            completed["evidence_sha256"],
        ):
            raise GateError(
                f"{arm_dir}: proxy/engine attempt identity parity failed"
            )
        if (
            result["outcome"] != "response"
            or completed["outcome"] != "completed"
            or result["status_code"] not in {200, 400}
        ):
            raise GateError(
                f"{arm_dir}: fixed32 attempt did not close with a complete "
                "engine response"
            )
        logical = str(attempt["logical_id_sha256"])
        attempts_by_logical.setdefault(logical, []).append(result)
        if result["status_code"] == 200:
            if logical in logical_successes:
                raise GateError(
                    f"{arm_dir}: logical request has multiple successful attempts"
                )
            logical_successes[logical] = str(
                attempt["engine_request_id_sha256"]
            )
            successful_engine_ids[
                str(attempt["engine_request_id_sha256"])
            ] = str(attempt["task_key_id"])

    completed_logicals = {
        str(row["logical_id_sha256"]): row
        for row in ledgers["proxy"]
        if row["event"] == "logical_complete"
    }
    if (
        any(row["outcome"] != "completed" for row in completed_logicals.values())
        or set(completed_logicals) != set(attempts_by_logical)
        or set(completed_logicals) != set(logical_successes)
        or any(
            sum(result["status_code"] == 200 for result in results) != 1
            for results in attempts_by_logical.values()
        )
    ):
        raise GateError(
            f"{arm_dir}: each completed logical request must have exactly one "
            "successful physical attempt"
        )

    for task_key_id, proxy_counts in task_counts["proxy"].items():
        engine_counts = task_counts["engine"][task_key_id]
        if (
            proxy_counts["accepted_logical_requests"]
            != (
                proxy_counts["completed_logical_model_requests"]
                + proxy_counts["aborted_logical_requests"]
            )
            or proxy_counts["aborted_logical_requests"] != 0
            or proxy_counts["accepted_attempts"]
            != proxy_counts["completed_attempts"]
            or proxy_counts["failed_attempts"] != 0
            or engine_counts["accepted_engine_requests"]
            != proxy_counts["accepted_attempts"]
            or engine_counts["completed_engine_requests"]
            != proxy_counts["completed_attempts"]
        ):
            raise GateError(
                f"{arm_dir}: per-task proxy/engine counts do not reconcile"
            )
        task_id = task_by_key[task_key_id]
        binding = task_bindings[task_id]
        expected_evidence = binding.get("task_evidence")
        if expected_evidence != {
            "completed_logical_model_requests": proxy_counts[
                "completed_logical_model_requests"
            ],
            "aborted_logical_requests": proxy_counts[
                "aborted_logical_requests"
            ],
            "accepted_attempts": proxy_counts["accepted_attempts"],
            "completed_attempts": proxy_counts["completed_attempts"],
            "failed_attempts": proxy_counts["failed_attempts"],
        }:
            raise GateError(
                f"{arm_dir}: runner task evidence differs from final proxy ledger"
            )
        evidence_after_records = binding.get(
            "task_auth_evidence_after_ledger_records"
        )
        evidence_after_head = binding.get(
            "task_auth_evidence_after_ledger_chain_head_sha256"
        )
        if (
            type(evidence_after_records) is not int
            or not 0 < evidence_after_records < len(ledgers["proxy"])
            or ledgers["proxy"][evidence_after_records - 1][
                "record_sha256"
            ]
            != evidence_after_head
        ):
            raise GateError(
                f"{arm_dir}: task-auth after snapshot is not an exact "
                "proxy-ledger prefix"
            )
        evidence_after_payload = {
            "schema": "fr13-fixed32-task-auth-evidence-v1",
            "task_key_id": task_key_id,
            **expected_evidence,
            "phase": "campaign",
            "ledger_records": evidence_after_records,
            "ledger_chain_head_sha256": evidence_after_head,
        }
        if canonical_json_sha256(evidence_after_payload) != binding.get(
            "task_auth_evidence_after_sha256"
        ):
            raise GateError(
                f"{arm_dir}: task-auth after snapshot digest does not bind "
                "its exact counter/ledger payload"
            )
        if (
            binding.get("trace_completed_logical_model_requests")
            != proxy_counts["completed_logical_model_requests"]
        ):
            raise GateError(
                f"{arm_dir}: trace request count differs from authenticated "
                "proxy task count"
            )

    census_path = arm_dir / "logs" / "fr13_fixed32_work_census.jsonl"
    try:
        located_records = load_work_census_jsonl(census_path)
    except WorkCensusError as error:
        raise GateError(f"{census_path}: {error}") from error
    if len(located_records) < 2:
        raise GateError(f"{census_path}: census has no complete event stream")
    event_records = located_records[:-1]
    terminal, terminal_source = located_records[-1]
    if (
        not isinstance(terminal, dict)
        or terminal.get("schema") != WORK_CENSUS_TERMINAL_SCHEMA
        or terminal.get("mode") != mode
        or terminal.get("final") is not True
        or terminal.get("event_count") != len(event_records)
        or terminal.get("first_event_index") != 0
        or terminal.get("last_event_index") != len(event_records) - 1
        or terminal.get("first_forward_step_index") != 0
        or terminal.get("last_forward_step_index") != len(event_records) - 1
        or terminal.get("events_sha256")
        != hashlib.sha256(
            json.dumps(
                [record for record, _source in event_records],
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()
    ):
        raise GateError(f"{terminal_source}: terminal work census mismatch")

    successful_memberships = {
        engine_id: 0 for engine_id in successful_engine_ids
    }
    per_task_memberships = {task_id: 0 for task_id in task_ids}
    for expected_index, (raw_event, source) in enumerate(event_records):
        if (
            not isinstance(raw_event, dict)
            or raw_event.get("schema") != WORK_CENSUS_EVENT_SCHEMA
        ):
            raise GateError(f"{source}: work-census event schema mismatch")
        try:
            event = validate_work_census_event(raw_event, source=source)
        except WorkCensusError as error:
            raise GateError(f"{source}: {error}") from error
        if (
            event.mode != mode
            or event.event_index != expected_index
            or event.forward_step_index != expected_index
        ):
            raise GateError(f"{source}: work-census sequence is not exact")
        runtime = raw_event["drafter_runtime"]
        request_digests = runtime["request_id_sha256s"]
        if len(request_digests) != event.batch_size:
            raise GateError(f"{source}: per-request census width mismatch")
        for engine_id in request_digests:
            task_key_id = successful_engine_ids.get(engine_id)
            if task_key_id is None:
                raise GateError(
                    f"{source}: census request has no successful authenticated "
                    "proxy/engine attempt"
                )
            task_id = task_by_key[task_key_id]
            interval = task_bindings[task_id]["forward_step_interval"]
            if not interval[0] <= expected_index < interval[1]:
                raise GateError(
                    f"{source}: request is outside its canonical task bracket"
                )
            successful_memberships[engine_id] += 1
            per_task_memberships[task_id] += 1
    if any(count <= 0 for count in successful_memberships.values()):
        raise GateError(
            f"{census_path}: successful engine request absent from decode census"
        )
    successful_by_task = {
        task_id: sum(
            task_key_id == fixed32_task_key_id(task_id)
            for task_key_id in successful_engine_ids.values()
        )
        for task_id in task_ids
    }
    for task_id in task_ids:
        successful_task_ids = sorted(
            engine_id
            for engine_id, task_key_id in successful_engine_ids.items()
            if task_key_id == fixed32_task_key_id(task_id)
        )
        if (
            successful_by_task[task_id]
            != task_bindings[task_id][
                "trace_completed_logical_model_requests"
            ]
            or (
                task_bindings[task_id]["trace_engine_id_joinable"]
                and successful_task_ids
                != task_bindings[task_id]["trace_model_request_id_sha256s"]
            )
        ):
            raise GateError(
                f"{census_path}: task successful request evidence differs "
                "from terminal trace"
            )

    return {
        "canonical_task_set_sha256": canonical_task_set_sha256,
        "preflight": preflights,
        "proxy": {
            "ledger": ledger_identities["proxy"],
            **reports["proxy"],
        },
        "engine": {
            "ledger": ledger_identities["engine"],
            **reports["engine"],
        },
        "exact_proxy_engine_attempt_parity": True,
        "zero_campaign_rejections": True,
        "zero_failed_or_aborted_requests": True,
        "census": _fixed32_artifact_identity(census_path)
        | {
            "event_schema": WORK_CENSUS_EVENT_SCHEMA,
            "terminal_schema": WORK_CENSUS_TERMINAL_SCHEMA,
            "event_count": len(event_records),
            "successful_engine_requests": len(successful_engine_ids),
            "request_step_memberships": sum(
                successful_memberships.values()
            ),
            "per_task_request_step_memberships": per_task_memberships,
            "all_successful_requests_present": True,
            "all_census_requests_authenticated": True,
            "all_census_requests_inside_task_brackets": True,
        },
    }


def _fixed32_qwen_runtime_attestation(
    payload: dict[str, Any],
    *,
    label: str,
) -> str:
    exact_keys(
        payload,
        {
            "schema",
            "launcher",
            "agent_env",
            "host_mode",
            "qwen_code_version",
            "bundle_tree",
            "bundle_manifest_sha256",
            "bundle_snapshot",
            "cleared_agent_environment",
            "system_settings",
        },
        label,
    )
    if (
        payload["schema"] != FIXED32_QWEN_RUNTIME_ATTESTATION_SCHEMA
        or payload["launcher"] != "qwen-code-instance-image"
        or payload["agent_env"] != "instance_image"
        or payload["host_mode"] != "remote"
        or payload["qwen_code_version"] != FIXED32_QWEN_CODE_VERSION
    ):
        raise GateError(f"{label}: Qwen runtime identity is not canonical")
    if payload["bundle_tree"] != FIXED32_QWEN_BUNDLE_TREE:
        raise GateError(f"{label}: Qwen executable bundle tree differs")
    if (
        _fixed32_digest(
            payload["bundle_manifest_sha256"],
            label=f"{label}:bundle_manifest_sha256",
        )
        != FIXED32_QWEN_BUNDLE_TREE["manifest_sha256"]
    ):
        raise GateError(f"{label}: Qwen bundle manifest digest differs")
    if payload["bundle_snapshot"] != {
        "kind": "per-task-content-addressed-snapshot",
        "basename": (
            "qwen_bundle-" + FIXED32_QWEN_BUNDLE_TREE["manifest_sha256"]
        ),
        "container_path": "/opt/qwen",
        "mount_mode": "ro",
    }:
        raise GateError(f"{label}: Qwen bundle snapshot identity differs")
    if (
        payload["cleared_agent_environment"]
        != FIXED32_CLEARED_AGENT_ENVIRONMENT
    ):
        raise GateError(f"{label}: Qwen injection environment is not cleared")
    expected_settings = {
        "source": "config/fr13_fixed32/qwen_system_settings.json",
        "bytes": 37,
        "sha256": FIXED32_QWEN_SYSTEM_SETTINGS_SHA256,
        "container_path": "/run/fr13/qwen-system-settings.json",
        "mount_mode": "ro",
        "environment": {
            "name": "QWEN_CODE_SYSTEM_SETTINGS_PATH",
            "value": "/run/fr13/qwen-system-settings.json",
        },
        "remote_file": {
            "mode": "0444",
            "uid": 1000,
            "gid": 1000,
            "nlink": 1,
            "xattrs": [],
        },
        "enable_auto_skill": False,
    }
    if payload["system_settings"] != expected_settings:
        raise GateError(f"{label}: Qwen system-settings evidence differs")
    return canonical_json_sha256(payload)


def _fixed32_agent_image_identity(
    payload: Any,
    *,
    task_id: str,
    label: str,
) -> str:
    expected = FIXED32_AGENT_IMAGE_IDENTITIES.get(task_id)
    if expected is None:
        raise GateError(f"{label}: task has no pinned agent image identity")
    if payload != expected:
        raise GateError(f"{label}: agent image identity differs")
    return canonical_json_sha256(payload)


def _fixed32_agent_placement(payload: Any, *, label: str) -> str:
    if payload != FIXED32_AGENT_PLACEMENT:
        raise GateError(f"{label}: agent placement identity differs")
    if (
        payload["agent_host_identity"] == payload["measured_host_identity"]
        or payload["identities_distinct"] is not True
    ):
        raise GateError(f"{label}: agent and measured hosts are not distinct")
    return canonical_json_sha256(payload)


def _fixed32_remote_settings_observation(
    payload: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise GateError(f"{label}: remote settings observation is not an object")
    identity_digest = payload.get("file_identity_sha256")
    static_payload = {
        key: value
        for key, value in payload.items()
        if key != "file_identity_sha256"
    }
    if (
        static_payload != FIXED32_QWEN_REMOTE_SETTINGS
        or not isinstance(identity_digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", identity_digest)
    ):
        raise GateError(f"{label}: remote settings identity differs")
    return {
        **FIXED32_QWEN_REMOTE_SETTINGS,
        "file_identity_sha256": identity_digest,
    }


def _fixed32_mounted_runtime_proof(
    payload: Any,
    *,
    label: str,
) -> str:
    identity_digest = (
        payload.get("system_settings", {}).get("file_identity_sha256")
        if isinstance(payload, dict)
        and isinstance(payload.get("system_settings"), dict)
        else None
    )
    if (
        not isinstance(identity_digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", identity_digest)
    ):
        raise GateError(f"{label}: mounted settings identity is malformed")
    expected = {
        "schema": FIXED32_MOUNTED_RUNTIME_PROOF_SCHEMA,
        "bundle_tree": {
            "container_path": "/opt/qwen",
            "mount_mode": "ro",
            "write_probe_errno": 30,
            "observation": {
                "qwen_code_version": FIXED32_QWEN_CODE_VERSION,
                "bundle_tree": FIXED32_QWEN_BUNDLE_TREE,
            },
        },
        "system_settings": {
            "container_path": "/run/fr13/qwen-system-settings.json",
            "mount_mode": "ro",
            "write_probe_errno": 30,
            **FIXED32_QWEN_REMOTE_SETTINGS,
            "file_identity_sha256": identity_digest,
        },
    }
    if payload != expected:
        raise GateError(f"{label}: mounted runtime proof differs")
    return canonical_json_sha256(payload)


def _fixed32_trace_model_requests(
    trace_path: Path,
    *,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    raw, text = strict_utf8_artifact(trace_path, label=str(trace_path))
    if not raw:
        raise GateError(f"{trace_path}: fixed32 trace is empty")
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise GateError(f"{trace_path}:{line_number}: blank JSONL record")
        event = exact_json_text(line, label=f"{trace_path}:{line_number}")
        events.append(event)
    init_event = events[0]
    if (
        init_event.get("type") != "system"
        or init_event.get("subtype") != "init"
        or init_event.get("qwen_code_version")
        != FIXED32_QWEN_CODE_VERSION
    ):
        raise GateError(
            f"{trace_path}: trace does not start with the pinned Qwen "
            f"{FIXED32_QWEN_CODE_VERSION} init record"
        )
    try:
        trace_requests = (
            fixed32_contract.validate_fixed32_trace_model_requests(
                events,
                expected_session_id=fixed32_contract.fixed32_trace_session_id(
                    provenance.get("instance_id")
                ),
            )
        )
    except Fixed32ContractError as error:
        raise GateError(f"{trace_path}: {error}") from error
    response_ids = trace_requests["model_request_ids"]
    completed_requests = trace_requests["completed_logical_model_requests"]
    response_id_digests = sorted(
        hashlib.sha256(response_id.encode("utf-8")).hexdigest()
        for response_id in response_ids
    )
    response_ids_sha256 = hashlib.sha256(
        json.dumps(
            response_id_digests,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if (
        provenance.get("trace_path") != str(trace_path.resolve())
        or provenance.get("trace_sha256") != hashlib.sha256(raw).hexdigest()
        or provenance.get("trace_bytes") != len(raw)
        or provenance.get("event_count") != len(events)
        or provenance.get("trace_completed_logical_model_requests")
        != completed_requests
        or provenance.get("trace_model_request_ids_sha256")
        != response_ids_sha256
    ):
        raise GateError(
            f"{trace_path}: v3 provenance does not match strict trace request "
            "evidence"
        )
    return {
        **_fixed32_artifact_identity(trace_path),
        "event_count": len(events),
        "completed_logical_model_requests": completed_requests,
        "model_request_id_sha256s": response_id_digests,
        "model_request_ids_sha256": response_ids_sha256,
        "engine_id_joinable": trace_requests["engine_id_joinable"],
    }


def _fixed32_proxy_runtime_identity(
    arm_dir: Path,
    *,
    task_ids: list[str],
) -> dict[str, Any]:
    env_path = arm_dir / "offload_proxy_env.txt"
    raw, text = strict_utf8_artifact(env_path, label=str(env_path))
    entries: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line or "=" not in line:
            raise GateError(
                f"{env_path}:{line_number}: malformed proxy environment entry"
            )
        key, value = line.split("=", 1)
        if key in entries:
            raise GateError(
                f"{env_path}:{line_number}: duplicate proxy environment key "
                f"{key!r}"
            )
        entries[key] = value
    required = {
        "LUMO_PROXY_FIXED32_DISABLE_RAW_DUMPS": "1",
        "LUMO_PROXY_FIXED32_TASK_IDS": ",".join(task_ids),
    }
    for key, expected in required.items():
        if entries.get(key) != expected:
            raise GateError(
                f"{env_path}: expected exactly {key}={expected!r}"
            )
    for key in (
        "LUMO_PROXY_FIXED32_SECRET_FILE",
        "LUMO_PROXY_FIXED32_LEDGER_PATH",
    ):
        if not entries.get(key):
            raise GateError(f"{env_path}: required proxy path {key} is absent")
    forbidden_env = {
        "LUMO_PROXY_PAIR_DUMP_DIR",
        "LUMO_PROXY_REQUEST_DUMP_DIR",
    }
    present_env = sorted(forbidden_env.intersection(entries))
    if present_env:
        raise GateError(
            f"{env_path}: fixed32 raw-dump environment is present: "
            f"{present_env}"
        )
    forbidden_paths = (
        arm_dir / "proxy_pair_dumps",
        arm_dir / "proxy_request_dumps",
    )
    present_paths = [
        str(path)
        for path in forbidden_paths
        if path.exists() or path.is_symlink()
    ]
    if present_paths:
        raise GateError(
            f"{arm_dir}: fixed32 raw proxy dump artifacts exist: "
            f"{present_paths}"
        )
    return {
        "path": str(env_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "canonical_task_set_sha256": fixed32_canonical_task_set_sha256(
            task_ids
        ),
        "raw_dump_environment_absent": True,
        "raw_dump_artifacts_absent": True,
    }


def build_fixed32_chat_traffic_audit(
    arm_dir: Path,
    *,
    mode: str,
    subset: dict[str, Any],
    dataset_record_digests: dict[str, str],
) -> dict[str, Any]:
    task_ids = list(subset["task_ids"])
    task_dirs = task_directories(arm_dir, len(task_ids))
    task_bindings: dict[str, dict[str, Any]] = {}
    audit_tasks: dict[str, dict[str, Any]] = {}
    for task_dir in task_dirs:
        task_id = task_dir.name
        metadata_path = task_dir / "runner_metadata.json"
        metadata = exact_json(metadata_path, label=str(metadata_path))
        if (
            metadata.get("instance_id") != task_id
            or metadata.get("dataset_name") != SWE_VERIFIED_DATASET
            or not isinstance(metadata.get("ended_at"), str)
            or not metadata["ended_at"]
            or metadata.get("fixed32_dataset_record_sha256")
            != dataset_record_digests[task_id]
        ):
            raise GateError(
                f"{metadata_path}: fixed32 task/dataset identity is not exact"
            )
        agent = metadata.get("agent")
        codex = metadata.get("codex")
        if (
            not isinstance(agent, dict)
            or not isinstance(codex, dict)
            or canonical_json_sha256(codex) != canonical_json_sha256(agent)
        ):
            raise GateError(
                f"{metadata_path}: fixed32 agent/codex terminal metadata differs"
            )
        exit_code = agent.get("exit_code")
        if (
            isinstance(exit_code, bool)
            or not isinstance(exit_code, int)
            or exit_code != 0
            or agent.get("timed_out") is not False
            or agent.get("offloaded") is not True
            or agent.get("network_drop") is not False
            or agent.get("stall_killed") not in {None, False}
            or agent.get("agent_env") != "instance_image"
        ):
            raise GateError(
                f"{metadata_path}: fixed32 task agent did not complete cleanly"
            )
        eval_report = metadata.get("eval_report")
        verdict = (
            eval_report.get("verdict")
            if isinstance(eval_report, dict)
            else None
        )
        passed = (
            eval_report.get("passed")
            if isinstance(eval_report, dict)
            else None
        )
        harness_exit_code = (
            eval_report.get("harness_exit_code")
            if isinstance(eval_report, dict)
            else None
        )
        if (
            not isinstance(eval_report, dict)
            or eval_report.get("instance_id") != task_id
            or eval_report.get("dataset_name") != SWE_VERIFIED_DATASET
            or verdict not in {"resolved", "failed"}
            or not isinstance(passed, bool)
            or passed is not (verdict == "resolved")
            or isinstance(harness_exit_code, bool)
            or not isinstance(harness_exit_code, int)
        ):
            raise GateError(
                f"{metadata_path}: fixed32 task has no terminal SWE verdict"
            )
        eval_path = task_dir / "eval" / "eval_report.json"
        if exact_json(eval_path, label=str(eval_path)) != eval_report:
            raise GateError(
                f"{eval_path}: terminal eval differs from runner metadata"
            )

        provenance = metadata.get("fixed32_real_task_provenance")
        task_key_id = fixed32_task_key_id(task_id)
        if (
            not isinstance(provenance, dict)
            or provenance.get("schema")
            != FIXED32_REAL_TASK_PROVENANCE_SCHEMA
            or provenance.get("instance_id") != task_id
            or provenance.get("task_key_id") != task_key_id
        ):
            raise GateError(
                f"{metadata_path}: fixed32 task provenance v3 is not exact"
            )
        expected_image = FIXED32_AGENT_IMAGE_IDENTITIES[task_id]
        image_identity = agent.get("instance_image_identity")
        image_identity_sha256 = _fixed32_agent_image_identity(
            image_identity,
            task_id=task_id,
            label=f"{metadata_path}:instance_image_identity",
        )
        if (
            agent.get("instance_image") != expected_image["image"]
            or _fixed32_digest(
                agent.get("instance_image_identity_sha256"),
                label=(
                    f"{metadata_path}:"
                    "agent.instance_image_identity_sha256"
                ),
            )
            != image_identity_sha256
            or _fixed32_digest(
                agent.get("instance_image_postrun_identity_sha256"),
                label=(
                    f"{metadata_path}:"
                    "agent.instance_image_postrun_identity_sha256"
                ),
            )
            != image_identity_sha256
            or agent.get("instance_image_run_reference")
            != expected_image["repo_digest"]
            or provenance.get("instance_image_identity_sha256")
            != image_identity_sha256
            or provenance.get("instance_image_id") != expected_image["id"]
            or provenance.get("instance_image_repo_digest")
            != expected_image["repo_digest"]
        ):
            raise GateError(
                f"{metadata_path}: agent image pre/run/post evidence differs"
            )
        placement = agent.get("agent_placement")
        placement_sha256 = _fixed32_agent_placement(
            placement,
            label=f"{metadata_path}:agent_placement",
        )
        if (
            _fixed32_digest(
                agent.get("agent_placement_sha256"),
                label=f"{metadata_path}:agent.agent_placement_sha256",
            )
            != placement_sha256
            or _fixed32_digest(
                agent.get("agent_postrun_placement_sha256"),
                label=(
                    f"{metadata_path}:"
                    "agent.agent_postrun_placement_sha256"
                ),
            )
            != placement_sha256
            or provenance.get("agent_placement_sha256")
            != placement_sha256
            or provenance.get("agent_host_identity")
            != FIXED32_AGENT_HOST_IDENTITY
            or provenance.get("measured_host_identity")
            != FIXED32_MEASURED_HOST_IDENTITY
        ):
            raise GateError(
                f"{metadata_path}: agent placement pre/post evidence differs"
            )
        nested_attestation = agent.get("qwen_runtime_attestation")
        if not isinstance(nested_attestation, dict):
            raise GateError(
                f"{metadata_path}: Qwen runtime attestation is missing"
            )
        attestation_sha256 = _fixed32_qwen_runtime_attestation(
            nested_attestation,
            label=f"{metadata_path}:qwen_runtime_attestation",
        )
        if (
            _fixed32_digest(
                agent.get("qwen_runtime_attestation_sha256"),
                label=(
                    f"{metadata_path}:"
                    "agent.qwen_runtime_attestation_sha256"
                ),
            )
            != attestation_sha256
            or _fixed32_digest(
                agent.get("qwen_runtime_postrun_attestation_sha256"),
                label=(
                    f"{metadata_path}:"
                    "agent.qwen_runtime_postrun_attestation_sha256"
                ),
            )
            != attestation_sha256
        ):
            raise GateError(
                f"{metadata_path}: Qwen pre/post attestation digests differ"
            )
        attestation_path = task_dir / "qwen_runtime_attestation.json"
        postrun_attestation_path = (
            task_dir / "qwen_runtime_attestation_post.json"
        )
        attestation_raw, attestation_text = strict_utf8_artifact(
            attestation_path,
            label=str(attestation_path),
        )
        postrun_raw, postrun_text = strict_utf8_artifact(
            postrun_attestation_path,
            label=str(postrun_attestation_path),
        )
        persisted_attestation = exact_json_text(
            attestation_text,
            label=str(attestation_path),
        )
        persisted_postrun_attestation = exact_json_text(
            postrun_text,
            label=str(postrun_attestation_path),
        )
        mounted_proof_path = (
            task_dir / FIXED32_MOUNTED_RUNTIME_PROOF_FILENAME
        )
        mounted_proof_raw, mounted_proof_text = strict_utf8_artifact(
            mounted_proof_path,
            label=str(mounted_proof_path),
        )
        persisted_mounted_proof = exact_json_text(
            mounted_proof_text,
            label=str(mounted_proof_path),
        )
        mounted_proof_sha256 = _fixed32_mounted_runtime_proof(
            persisted_mounted_proof,
            label=str(mounted_proof_path),
        )
        remote_settings_observation = (
            _fixed32_remote_settings_observation(
                agent.get("qwen_remote_settings_observation"),
                label=f"{metadata_path}:qwen_remote_settings_observation",
            )
        )
        remote_settings_observation_sha256 = canonical_json_sha256(
            remote_settings_observation
        )
        if mounted_proof_raw != (
            json.dumps(
                persisted_mounted_proof,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii"):
            raise GateError(
                f"{mounted_proof_path}: proof is not canonical JSON"
            )
        if (
            persisted_attestation != nested_attestation
            or persisted_postrun_attestation != nested_attestation
            or _fixed32_qwen_runtime_attestation(
                persisted_attestation,
                label=str(attestation_path),
            )
            != attestation_sha256
            or _fixed32_qwen_runtime_attestation(
                persisted_postrun_attestation,
                label=str(postrun_attestation_path),
            )
            != attestation_sha256
            or provenance.get("qwen_code_version")
            != FIXED32_QWEN_CODE_VERSION
            or provenance.get("qwen_system_settings_sha256")
            != FIXED32_QWEN_SYSTEM_SETTINGS_SHA256
            or provenance.get("qwen_runtime_attestation_sha256")
            != attestation_sha256
            or provenance.get("qwen_runtime_attestation_file_sha256")
            != hashlib.sha256(attestation_raw).hexdigest()
            or provenance.get(
                "qwen_runtime_postrun_attestation_file_sha256"
            )
            != hashlib.sha256(postrun_raw).hexdigest()
            or agent.get("qwen_bundle_snapshot")
            != nested_attestation["bundle_snapshot"]
            or provenance.get("qwen_bundle_snapshot")
            != nested_attestation["bundle_snapshot"]
            or agent.get("qwen_mounted_runtime_proof")
            != persisted_mounted_proof
            or agent.get("qwen_mounted_runtime_proof_sha256")
            != mounted_proof_sha256
            or agent.get("qwen_mounted_runtime_proof_file_sha256")
            != hashlib.sha256(mounted_proof_raw).hexdigest()
            or provenance.get("qwen_mounted_runtime_proof_sha256")
            != mounted_proof_sha256
            or provenance.get(
                "qwen_mounted_runtime_proof_file_sha256"
            )
            != hashlib.sha256(mounted_proof_raw).hexdigest()
            or persisted_mounted_proof["system_settings"][
                "file_identity_sha256"
            ]
            != remote_settings_observation["file_identity_sha256"]
            or provenance.get(
                "qwen_remote_settings_file_identity_sha256"
            )
            != remote_settings_observation["file_identity_sha256"]
            or agent.get("qwen_remote_settings_observation_sha256")
            != remote_settings_observation_sha256
            or agent.get(
                "qwen_remote_settings_postrun_observation_sha256"
            )
            != remote_settings_observation_sha256
            or provenance.get(
                "qwen_remote_settings_observation_sha256"
            )
            != remote_settings_observation_sha256
        ):
            raise GateError(
                f"{metadata_path}: Qwen runtime attestation evidence differs"
            )
        trace = _fixed32_trace_model_requests(
            task_dir / "qwen_trace.jsonl",
            provenance=provenance,
        )
        evidence_keys = (
            "completed_logical_model_requests",
            "aborted_logical_requests",
            "accepted_attempts",
            "completed_attempts",
            "failed_attempts",
        )
        evidence = {
            key: strict_nonnegative_int(
                provenance.get(key),
                label=f"{metadata_path}:{key}",
            )
            for key in evidence_keys
        }
        if (
            evidence["completed_logical_model_requests"]
            != trace["completed_logical_model_requests"]
            or evidence["aborted_logical_requests"] != 0
            or evidence["failed_attempts"] != 0
            or evidence["accepted_attempts"]
            != evidence["completed_attempts"] + evidence["failed_attempts"]
            or evidence["completed_attempts"]
            < evidence["completed_logical_model_requests"]
        ):
            raise GateError(
                f"{metadata_path}: trace/task-auth counts do not reconcile"
            )
        before_sha256 = _fixed32_digest(
            provenance.get("task_auth_evidence_before_sha256"),
            label=f"{metadata_path}:task_auth_evidence_before_sha256",
        )
        after_sha256 = _fixed32_digest(
            provenance.get("task_auth_evidence_after_sha256"),
            label=f"{metadata_path}:task_auth_evidence_after_sha256",
        )
        after_records = strict_positive_int(
            provenance.get("task_auth_evidence_after_ledger_records"),
            label=f"{metadata_path}:task_auth_evidence_after_ledger_records",
        )
        after_head = _fixed32_digest(
            provenance.get(
                "task_auth_evidence_after_ledger_chain_head_sha256"
            ),
            label=(
                f"{metadata_path}:"
                "task_auth_evidence_after_ledger_chain_head_sha256"
            ),
        )

        boundary_path = task_dir / "fixed32_task_boundary.json"
        boundary = exact_json(boundary_path, label=str(boundary_path))
        if (
            metadata.get("fixed32_task_boundary") != boundary
            or boundary.get("schema") != FIXED32_BOUNDARY_SCHEMA
            or boundary.get("instance_id") != task_id
            or boundary.get("mode") != mode
        ):
            raise GateError(
                f"{boundary_path}: task boundary identity is not exact"
            )
        interval_payload = boundary.get("forward_step_interval")
        if not isinstance(interval_payload, dict):
            raise GateError(f"{boundary_path}: forward interval is missing")
        exact_keys(
            interval_payload,
            {
                "start_forward_step",
                "end_forward_step",
                "expected_complete_events",
            },
            f"{boundary_path}:forward_step_interval",
        )
        start = strict_nonnegative_int(
            interval_payload["start_forward_step"],
            label=f"{boundary_path}:start_forward_step",
        )
        end = strict_positive_int(
            interval_payload["end_forward_step"],
            label=f"{boundary_path}:end_forward_step",
        )
        if (
            end <= start
            or interval_payload["expected_complete_events"] != end - start
        ):
            raise GateError(f"{boundary_path}: task interval is invalid")
        terminal = provenance.get("agent_terminal")
        if not isinstance(terminal, dict) or any(
            terminal.get(key) != agent.get(key)
            or type(terminal.get(key)) is not type(agent.get(key))
            for key in ("exit_code", "timed_out", "offloaded", "network_drop")
        ):
            raise GateError(
                f"{metadata_path}: provenance terminal differs from agent"
            )
        task_bindings[task_id] = {
            "task_key_id": task_key_id,
            "trace_completed_logical_model_requests": trace[
                "completed_logical_model_requests"
            ],
            "trace_engine_id_joinable": trace["engine_id_joinable"],
            "trace_model_request_id_sha256s": trace[
                "model_request_id_sha256s"
            ],
            "task_evidence": evidence,
            "task_auth_evidence_before_sha256": before_sha256,
            "task_auth_evidence_after_sha256": after_sha256,
            "task_auth_evidence_after_ledger_records": after_records,
            "task_auth_evidence_after_ledger_chain_head_sha256": after_head,
            "forward_step_interval": [start, end],
        }
        audit_tasks[task_id] = {
            "task_key_id": task_key_id,
            "dataset_record_sha256": dataset_record_digests[task_id],
            "trace": {
                key: value
                for key, value in trace.items()
                if key != "engine_id_joinable"
            },
            "task_auth": {
                **evidence,
                "evidence_before_sha256": before_sha256,
                "evidence_after_sha256": after_sha256,
                "evidence_after_ledger_records": after_records,
                "evidence_after_ledger_chain_head_sha256": after_head,
            },
            "terminal": {
                "agent": {
                    key: agent[key]
                    for key in (
                        "exit_code",
                        "timed_out",
                        "offloaded",
                        "network_drop",
                    )
                },
                "eval": {
                    "verdict": verdict,
                    "passed": passed,
                    "harness_exit_code": harness_exit_code,
                },
                "eval_artifact": _fixed32_artifact_identity(eval_path),
            },
            "boundary": _fixed32_artifact_identity(boundary_path)
            | {"forward_step_interval": [start, end]},
        }

    ingress = validate_fixed32_ingress_and_census(
        arm_dir,
        mode=mode,
        task_ids=task_ids,
        task_bindings=task_bindings,
    )
    intervals = sorted(
        (
            binding["forward_step_interval"]
            for binding in task_bindings.values()
        ),
        key=lambda interval: (interval[0], interval[1]),
    )
    merged_intervals: list[list[int]] = []
    for start, end in intervals:
        if not merged_intervals or start > merged_intervals[-1][1]:
            merged_intervals.append([start, end])
        else:
            merged_intervals[-1][1] = max(merged_intervals[-1][1], end)
    complete_events = ingress["census"]["event_count"]
    if merged_intervals != [[0, complete_events]]:
        raise GateError(
            f"{arm_dir}: canonical task brackets do not exactly cover the "
            "authenticated census stream"
        )
    fetch_status = arm_dir / "offload_fetch_status.txt"
    if read_text(fetch_status) != "ok\n":
        raise GateError(f"{fetch_status}: offload artifact fetch was not successful")
    proxy_runtime = _fixed32_proxy_runtime_identity(
        arm_dir,
        task_ids=task_ids,
    )
    return {
        "schema": FIXED32_CHAT_TRAFFIC_AUDIT_SCHEMA,
        "mode": mode,
        "dataset_name": SWE_VERIFIED_DATASET,
        "subset": {
            "sha256": subset["sha256"],
            "task_count": len(task_ids),
            "task_ids": task_ids,
        },
        "checks": {
            "all_canonical_tasks_validated": True,
            "all_task_identity_and_dataset_hashes_exact": True,
            "all_task_agent_and_eval_terminal": True,
            "all_trace_request_counts_match_authenticated_proxy": True,
            "all_proxy_attempts_match_engine_requests": True,
            "all_successful_engine_requests_match_census": True,
            "all_census_requests_inside_task_brackets": True,
            "no_campaign_rejections_or_aborted_requests": True,
            "no_fixed32_traffic_outside_task_brackets": True,
            "raw_proxy_request_and_response_dumps_disabled": True,
        },
        "offload_fetch_status": _fixed32_artifact_identity(fetch_status),
        "proxy_runtime": proxy_runtime,
        "complete_stream": {
            "pure_decode_forward_steps": complete_events,
            "complete_work_census_events": complete_events,
            "merged_forward_step_intervals": merged_intervals,
        },
        "ingress": ingress,
        "tasks": audit_tasks,
    }


def validate_real_task_provenance(
    arm_dir: Path,
    task_dirs: list[Path],
    *,
    mode: str,
    subset: dict[str, Any],
    windows: list[dict[str, Any]],
    flush_chain: dict[str, Any],
    dataset_record_digests: dict[str, str],
) -> dict[str, Any]:
    task_ids = list(subset["task_ids"])
    if [task_dir.name for task_dir in task_dirs] != task_ids:
        raise GateError(f"{arm_dir}: task directories are not canonical order")
    windows_by_task = {window["task_id"]: window for window in windows}
    if list(windows_by_task) != task_ids:
        raise GateError(f"{arm_dir}: metric windows are not canonical order")
    if list(flush_chain["tasks"]) != task_ids:
        raise GateError(f"{arm_dir}: flush tasks are not canonical order")

    expected_audit = build_fixed32_chat_traffic_audit(
        arm_dir,
        mode=mode,
        subset=subset,
        dataset_record_digests=dataset_record_digests,
    )
    audit_path = arm_dir / "fixed32_chat_traffic_audit.json"
    audit = exact_json(audit_path, label=str(audit_path))
    if audit != expected_audit:
        raise GateError(
            f"{audit_path}: v2 authenticated chat-task audit differs: "
            f"{first_json_difference(audit, expected_audit)}"
        )

    for task_id in task_ids:
        audit_boundary = expected_audit["tasks"][task_id]["boundary"]
        flush_boundary = flush_chain["tasks"][task_id]
        interval = list(windows_by_task[task_id]["fwd_span"])
        boundary_path = Path(flush_boundary["path"])
        expected_boundary = {
            "path": str(boundary_path),
            "sha256": flush_boundary["sha256"],
            "bytes": boundary_path.stat().st_size,
            "forward_step_interval": interval,
        }
        if (
            audit_boundary != expected_boundary
            or flush_boundary["forward_step_interval"] != interval
        ):
            raise GateError(
                f"{task_id}: authenticated ingress/census bracket differs "
                "from the exact metric/flush bracket"
            )

    final_counters = flush_chain["final"]["counters"]
    complete_steps = final_counters["pure_decode_forward_steps"]
    complete_events = final_counters["complete_work_census_events"]
    complete_stream = expected_audit["complete_stream"]
    if (
        complete_steps != complete_events
        or complete_stream["pure_decode_forward_steps"] != complete_steps
        or complete_stream["complete_work_census_events"] != complete_events
        or expected_audit["ingress"]["census"]["event_count"]
        != complete_events
    ):
        raise GateError(
            f"{arm_dir}: authenticated ingress/census stream does not close "
            "the terminal flush stream"
        )

    fetch = expected_audit["offload_fetch_status"]
    return {
        "all_canonical_tasks_have_real_model_traffic": True,
        "all_validated_chat_task_traffic_bound": True,
        "fixed32_ingress_proxy_engine_exact": expected_audit["ingress"][
            "exact_proxy_engine_attempt_parity"
        ],
        "fixed32_zero_campaign_rejections": expected_audit["ingress"][
            "zero_campaign_rejections"
        ],
        "fixed32_raw_proxy_dumps_disabled": (
            expected_audit["proxy_runtime"]["raw_dump_environment_absent"]
            and expected_audit["proxy_runtime"]["raw_dump_artifacts_absent"]
        ),
        "all_agents_completed_cleanly": True,
        "all_tasks_have_terminal_eval_verdicts": True,
        "offload_fetch_status": {
            "path": fetch["path"],
            "sha256": fetch["sha256"],
        },
        "chat_traffic_audit": {
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
            "bytes": audit_path.stat().st_size,
            "schema": FIXED32_CHAT_TRAFFIC_AUDIT_SCHEMA,
        },
        "tasks": expected_audit["tasks"],
    }


def resolve_subset_from_runlog(
    repo: Path,
    runroot: Path,
    arm: str,
    expected_kind: str,
    expected_tokens: int,
    task_count: int,
    concurrency: int,
) -> dict[str, Any]:
    path = runroot / f"{arm}.runlog"
    text = read_text(path)
    matches = [
        match for line in text.splitlines() if (match := ARM_HEADER_RE.match(line))
    ]
    if len(matches) != 1:
        raise GateError(f"{path}: expected exactly one BIGDENOM arm header")
    match = matches[0]
    if (
        match.group("arm") != arm
        or match.group("kind") != expected_kind
        or int(match.group("tokens")) != expected_tokens
    ):
        raise GateError(f"{path}: arm header does not match the requested arm")
    subset_path = Path(match.group("subset"))
    if not subset_path.is_absolute():
        subset_path = repo / subset_path
    subset = validate_subset(subset_path.resolve(), task_count)
    if expected_kind not in FIXED32_MODE_SPECS:
        raise GateError(f"{path}: unsupported fixed-32 mode {expected_kind!r}")
    process_path = runroot / arm / "fixed32_process_identity.json"
    process_identity = exact_json(process_path, label=str(process_path))
    if process_identity.get("schema") != "fr13-fixed32-process-identity-v1":
        raise GateError(f"{process_path}: wrong process identity schema")
    pid1 = process_identity.get("pid1")
    engine_core = process_identity.get("engine_core")
    if (
        not isinstance(pid1, dict)
        or isinstance(pid1.get("pid"), bool)
        or pid1.get("pid") != 1
        or not isinstance(pid1.get("environ"), list)
        or not isinstance(engine_core, dict)
        or isinstance(engine_core.get("pid"), bool)
        or not isinstance(engine_core.get("pid"), int)
        or not isinstance(engine_core.get("environ"), list)
    ):
        raise GateError(f"{process_path}: incomplete PID1/EngineCore identity")
    argv = pid1.get("argv")
    expected_argv = expected_pid1_argv(concurrency)
    if argv != expected_argv:
        raise GateError(
            f"{process_path}: PID1 argv differs from the exact fixed32 contract"
        )
    fa2_path = str(CONTAINER_FA2_DESTINATION)
    if not any(
        fa2_path in line for line in engine_core.get("forked_fa2_maps", [])
    ):
        raise GateError(
            f"{process_path}: EngineCore did not map the pinned forked FA2 binary"
        )
    ratio_needle = f"draft_tokens/drafts={float(expected_tokens):.1f}"
    if text.count(ratio_needle) != 1:
        raise GateError(f"{path}: exact warmup draft-token ratio needle is absent")
    engine_core_pids = [
        int(pid_match.group("pid"))
        for line in text.splitlines()
        if (pid_match := ENGINE_CORE_PID_RE.match(line))
    ]
    if len(engine_core_pids) != 1:
        raise GateError(
            f"{path}: expected exactly one recorded VLLM::EngineCore PID, "
            f"got {engine_core_pids}"
        )
    if engine_core.get("pid") != engine_core_pids[0]:
        raise GateError(
            f"{process_path}: EngineCore PID differs from the runlog producer PID"
        )
    return {
        "runlog": str(path),
        "subset": subset,
        "pid1_argv": argv,
        "pid1_exact_contract": True,
        "process_identity": {
            "path": str(process_path),
            "sha256": sha256_file(process_path),
        },
        "engine_core_pid": engine_core_pids[0],
    }


def fixed32_required_env(
    arm_dir: Path,
    *,
    mode: str,
    task_ids: list[str],
) -> dict[str, str]:
    try:
        mode_spec = FIXED32_MODE_SPECS[mode]
    except KeyError as error:
        raise GateError(f"{arm_dir}: unsupported fixed-32 mode {mode!r}") from error
    if tuple(task_ids) != EVIDENCE_SETS.get(len(task_ids), {}).get("task_ids"):
        raise GateError(f"{arm_dir}: ingress task IDs are not canonical exact4/16")
    required_env = {
        "FR13_HYDRA23": "0",
        "FR13_TAIL_MODE": "1",
        "FR13_DRAFT_SOURCE": "merged",
        "FR13_TREE_GDN_GEOM_OVERRIDE": "BV=8",
        "FR13_FIXED32_MODE": mode,
        "FR13_FIXED32_VALID_MASK": f"{mode_spec['valid_mask']:#010x}",
        "FR13_FIXED32_ACTIVE_NODES": str(mode_spec["active_drafts"]),
        "FR13_FIXED32_PHYSICAL_DRAFTS": "31",
        "FR13_FIXED32_ENGINE_PID_FILE": "/logs/fr13_fixed32_engine_pid",
        "FR13_FIXED32_FLUSH_REQUEST_PATH": (
            "/logs/fr13_fixed32_flush_request.json"
        ),
        "FR13_FIXED32_FLUSH_ACK_PATH": "/logs/fr13_fixed32_flush_ack.json",
        "FR13_FIXED32_WORK_CENSUS_PATH": (
            "/logs/fr13_fixed32_work_census.jsonl"
        ),
        "FR13_FIXED32_BOUNDARY_SNAPSHOT_PATH": (
            "/logs/fr13_fixed32_boundary_snapshot"
        ),
        "FR13_FIXED32_ENGINE_INGRESS_LEDGER_PATH": (
            "/logs/fr13_fixed32_engine_ingress.jsonl"
        ),
        "FR13_FIXED32_INGRESS_SECRET_FILE": (
            "/run/fr13_fixed32_ingress_secret"
        ),
        "FR13_FIXED32_INGRESS_TASK_IDS": ",".join(task_ids),
        "FR13_FIXED32_MIDDLEWARE_FLAGS": (
            "--middleware "
            "lumo_flywheel_serving.inference_proxy."
            "Fixed32EngineIngressMiddleware"
        ),
        "FR13_FIXED32_WORK_CENSUS": "1",
        "FR13_FIXED32_DEVICE_PUBLISH": "1",
        "FR13_FIXED32_ACCEPT_PACK": "1",
        "FR13_FIXED32_REQKEY_DEVICE": "1",
        "FR13_FIXED32_KV_REMAP16": "1",
        "FR13_FIXED32_COMMIT_DEVICE_FILL": "1",
        "FR13_FIXED32_TAW_WALK_CAP": "12",
        "FR13_DEVICE_MULTIDRAFT": "1",
        "FR13_DRAFTER_GRAPH": "1",
        "FR13_DRAFTER_SINGLE_LOGITS": "1",
        "FR13_DM_DEPTHSYNC": "1",
        "FR13_TAW": "1",
        "FR13_PARENT_GATHER": "1",
        "FR13_EAGER_PACK": "1",
        "FR13_COMMIT_BATCH_OUTPUT": "1",
        "FR13_COMMITTER_NATIVE": "1",
        "FR13_COMMITTER_BATCHED": "1",
        "FR13_COMMITTER_GRAPH": "1",
        "FR13_COMMIT_OVERLAP": "0",
        "FR13_RING_EXPORT": "1",
        "FR13_REPLAY_ROUTE": "1",
        "FR13_ATTN_KV_REMAP": "1",
        "FR13_SLOT_REORDER": "1",
        "FR13_KV_REMAP_SYNCFREE": "1",
        "FR13_FA2_TREE_BIAS": "1",
        "FR13_TREE_CONV_FUSED": "1",
        "FR13_CONV_WB_FUSED": "1",
        "FR13_CONV_WB_BATCHED": "1",
        "FR13_CONV_PREGATHER": "1",
        "FR13_CONV_COMMITTED_PATH": "1",
        "FR13_APC_COMMIT_TO_RUNNING_ROW": "1",
        "FR13_TREE_RUNROW_INIT": "1",
        "FR13_FLAGS_INKERNEL": "1",
        "FR13_SFWD_GPU_TIMER": "1",
        "FR13_SFWD_GPU_TIMER_JSON": (
            f"/workspace/output/fr13_sfwd_sidecar/{arm_dir.name}.json"
        ),
        "FR13_TIMER_EXPLICIT_FLUSH": "1",
        "FR13_STEP_WALL_CAP_S": "1.5",
        "FR13_STEP_GRAPH": "0",
        "FR13_SUBTREE_PARALLEL": "1",
        "FR13_SUBTREE_PARALLEL_SELFCHECK": "0",
    }
    required_env.update(
        {
            "ATTENTION_BACKEND": "TREE_ATTN",
            "FR10_DECODE_MODE_DEFAULT": "tree_mtp",
            "FR10_METRICS": "0",
            "VLLM_BATCH_INVARIANT": "0",
            "LUMO_BATCH_INVARIANT_VLLM": "0",
            "LUMO_FB_KERNEL_ROWS": "1",
            "LUMO_FB_PROJ_PAD_ROWS": "16",
            "FR13_ENABLE_APC": "1",
            "FR13_APC_CONFIG_ONLY": "0",
            "FR13_INPUTPREP_GUARD": "1",
            "FR13_DRAFT_VOCAB_K": "65536",
            "FR13_DRAFT_VOCAB_BLOCKS": (
                "/workspace/scripts/fr13_dvk_subset_blocks.json"
            ),
            "FR13_DEVICE_MULTIDRAFT": "1",
            "FR13_DFWD_GPU_TIMER": "1",
            "FR13_DFWD_GPU_TIMER_JSON": (
                f"/workspace/output/fr13_sfwd_sidecar/{arm_dir.name}_dfwd.json"
            ),
            "FR13_CFWD_GPU_TIMER": "1",
            "FR13_CFWD_GPU_TIMER_JSON": (
                f"/workspace/output/fr13_sfwd_sidecar/{arm_dir.name}_cfwd.json"
            ),
            "FR13_SFWD_GPU_TIMER_MAXPENDING": "256",
            "FR13_SFWD_GPU_TIMER_SAMPLES_MAX": "200000",
            "FR13_SFWD_GPU_TIMER_DUMP_S": "1",
            "FR13_SFWD_SAMPLES_DUMP_S": "30",
            "FR13_SPAN_GPU_TIMER_DUMP_S": "1",
            "FR13_WEIGHT_FLOOR_MS": "98.6",
            "FR13_COMPUTE_MS_PER_ROW": "0.54",
            "FR13_APC_CONV_FIX": "1",
            "FR13_APC_CONV_SNAPSHOT": "1",
            "FR13_APC_ZERO_MAMBA_ON_ALLOC": "1",
            "FR13_APC_COPY_SRC_FIX": "1",
            "FR13_FREE_TREE_POSGLOBALS": "0",
            "FR13_APC_BLOCK_ALIGN_45477": "1",
            "FR13_FULL_ATTN_KV_FP8": "0",
            "FR13_SERVE_BATCH_FLAGS": "",
            "FR13_MULTIDRAFT_GPU_TIMER": "0",
            "FR13_REPLAY_GPU_TIMER": "0",
            "FR13_COMMIT_FULL_GPU_TIMER": "0",
            "FR13_COMMITTER_SG_TIMER": "0",
            "FR13_REPLAY_ONLY_GPU_TIMER": "0",
            "FR13_GRAPH_TIMER": "0",
            "FR13_KVREMAP_TIMER": "0",
            "FR13_STATEREMAP_TIMER": "0",
            "FR13_DFWD_SPLIT_NEEDLE": "0",
            "FR13_REPLAY_MULTISTREAM": "0",
            "FR13_BRANCH_ACCEPT_DIAG": "0",
            "FR13_FORCE_SPINE_COMMIT": "0",
            "FR13_FIX1_SELFCHECK": "0",
            "FR13_COMMIT_ARGMAX_GATE": "0",
            "FR13_FORK_MARGIN_DUMP": "0",
            "FR13_CHASE_DIAG": "0",
            "FR13_REPLAY_BOUNDARY_LOG": "0",
            "FR13_GDN_SUBOP_MAB": "0",
            "FR13_CONV_SUBOP_MAB": "0",
            "FR13_FA2_MAB": "0",
            "FR13_REPLAY_DURABLE_AB": "0",
            "FR13_TREE_POSREAD_PROBE": "0",
            "FR13_LEAK_PROBE": "0",
            "FR13_SERVE_LOG": "0",
            "FR13_TORCH_DET_WARN": "0",
            "FR13_TCF_DIAG_OVERRIDE": "0",
            "FR13_TCF_SELFCHECK": "0",
            "FR13_PARENT_GATHER_SELFCHECK": "0",
            "FR13_TORCHPROF": "0",
            "FR13_TORCH_PROF": "",
            "FR13_DVK_DRAFTID_DUMP": "",
            "LUMO_NSYS_WRAP_VLLM": "0",
            "FR13_FIXED32_NVTX_PROFILE": "0",
            "LUMO_FA_ACTIVATION_REPLAY_BATCH4_DIAG": "0",
            "LUMO_FA_REPLAY_COMMIT_BATCH4_RUNNER_DIAG": "0",
            "LUMO_IR_DIAGNOSTIC_UNISOLATED": "0",
            "LUMO_IR_ALLOW_UNVERIFIED_SPINES2_MEASUREMENT": "0",
            "FR10_TREE_GDN_CAPTURE_PAYLOAD": "",
            "FR10_TREE_GDN_COMMIT_HANDOFF_LOG": "",
            "FR10_TREE_GDN_SRC_NATIVE_PAYLOAD": "",
            "FR10_ROOT_HIDDEN_CAPTURE": "",
            "FR10_ROOT_LOGIT_CAPTURE": "",
            "FR10_LAYER_HIDDEN_CAPTURE": "",
            "FR12_FULL_ATTN_CAPTURE": "",
            "FR12_SUBKERNEL_CAPTURE": "",
            "FR13_TREE_ATTN_OP_CAPTURE": "",
            "FR13_FLASH_ATTN_OP_CAPTURE": "",
            "FR13_PREPROCESS_INPUT_CAPTURE": "",
            "FR13_PREFILL_GDN_CAPTURE": "",
            "FR13_DECODE_GDN_CAPTURE": "",
            "FR10_SPINE_LOGIT_CAPTURE": "",
            "FR13_FINAL_LOGIT_CAPTURE": "",
            "FR13_HIDDEN_SUBSTITUTE": "",
            "LUMO_MTP_DRAFT_TRACE_FILE": "",
            "LUMO_TREE_SAMPLER_DEBUG_LOG": "",
            "LUMO_TREE_PATH_LCP_LOG": "",
        }
    )
    return required_env


def validate_fixed32_pretask_zero_traffic(
    arm_dir: Path,
    *,
    mode: str,
) -> dict[str, Any]:
    marker_path = arm_dir / "fixed32_pretask_zero_traffic.json"
    marker = exact_json(marker_path, label=str(marker_path))
    exact_keys(
        marker,
        {
            "schema",
            "mode",
            "no_positive_probe",
            "generation_probe_commands_executed",
            "metrics",
            "work_census",
            "ready_ack",
        },
        str(marker_path),
    )
    if (
        marker["schema"] != "fr13-fixed32-pretask-zero-traffic-v1"
        or marker["mode"] != mode
        or marker["no_positive_probe"] is not True
        or marker["generation_probe_commands_executed"] != 0
        or isinstance(marker["generation_probe_commands_executed"], bool)
    ):
        raise GateError(f"{marker_path}: fixed32 pretask traffic claim is invalid")

    metrics_path = arm_dir / "metrics_before_swe.txt"
    metrics = marker["metrics"]
    if not isinstance(metrics, dict):
        raise GateError(f"{marker_path}: metrics identity is missing")
    exact_keys(
        metrics,
        {"path", "sha256", "spec_drafts", "spec_tokens"},
        f"{marker_path}:metrics",
    )
    _metrics_raw, metrics_text = strict_utf8_artifact(
        metrics_path,
        label=str(metrics_path),
    )
    metric_values, _metric_labels = pretask_metric_snapshot_text(
        metrics_text,
        label=str(metrics_path),
    )
    if (
        metrics["path"] != str(metrics_path.resolve())
        or metrics["sha256"] != sha256_file(metrics_path)
        or metrics["spec_drafts"] != 0
        or metrics["spec_tokens"] != 0
        or isinstance(metrics["spec_drafts"], bool)
        or isinstance(metrics["spec_tokens"], bool)
        or integral(metric_values["spec_drafts"], f"{metrics_path}:spec drafts") != 0
        or integral(metric_values["spec_tokens"], f"{metrics_path}:spec tokens") != 0
    ):
        raise GateError(f"{marker_path}: pretask decode metrics are not exact zero")

    census_path = arm_dir / "logs" / "fr13_fixed32_work_census.jsonl"
    census = marker["work_census"]
    if not isinstance(census, dict):
        raise GateError(f"{marker_path}: work-census baseline is missing")
    exact_keys(
        census,
        {"path", "exists", "bytes", "sha256"},
        f"{marker_path}:work_census",
    )
    empty_sha256 = hashlib.sha256(b"").hexdigest()
    if (
        census["path"] != str(census_path.resolve())
        or not isinstance(census["exists"], bool)
        or census["bytes"] != 0
        or isinstance(census["bytes"], bool)
        or census["sha256"] != empty_sha256
    ):
        raise GateError(f"{marker_path}: pretask work census was not empty")

    ready_path = arm_dir / "fixed32_ready_ack.json"
    ready = marker["ready_ack"]
    if not isinstance(ready, dict):
        raise GateError(f"{marker_path}: ready-ack identity is missing")
    exact_keys(
        ready,
        {"path", "sha256", "generation"},
        f"{marker_path}:ready_ack",
    )
    if (
        ready["path"] != str(ready_path.resolve())
        or ready["sha256"] != sha256_file(ready_path)
        or ready["generation"] != 0
        or isinstance(ready["generation"], bool)
    ):
        raise GateError(f"{marker_path}: zero-generation ready ack is not bound")

    forbidden = (
        "warmup_probe.json",
        "warmup_request_metrics.jsonl",
        "warmup_probe_stdout.log",
        "metrics_before_warmup.txt",
        "metrics_after_warmup.txt",
        "docker_after_warmup.log",
        "reset_prefix_cache.txt",
    )
    present = [name for name in forbidden if (arm_dir / name).exists()]
    if present:
        raise GateError(
            f"{arm_dir}: fixed32 forbidden pretask probe artifacts exist: {present}"
        )
    return {
        "path": str(marker_path),
        "sha256": sha256_file(marker_path),
        "metrics": dict(metrics),
        "work_census": dict(census),
        "ready_ack": dict(ready),
        "forbidden_probe_artifacts_absent": True,
    }


def validate_runtime_needles(
    arm_dir: Path,
    *,
    mode: str,
    expected_tokens: int,
    task_ids: list[str],
) -> dict[str, Any]:
    env_path = arm_dir / "container_env.txt"
    env_lines = read_text(env_path).splitlines()
    try:
        mode_spec = FIXED32_MODE_SPECS[mode]
    except KeyError as error:
        raise GateError(f"{arm_dir}: unsupported fixed-32 mode {mode!r}") from error
    required_env = fixed32_required_env(
        arm_dir,
        mode=mode,
        task_ids=task_ids,
    )
    for key, expected in required_env.items():
        values = [
            line.removeprefix(f"{key}=")
            for line in env_lines
            if line.startswith(f"{key}=")
        ]
        if values != [expected]:
            raise GateError(
                f"{env_path}: expected exactly {key}={expected}, got {values}"
            )
    process_path = arm_dir / "fixed32_process_identity.json"
    process_identity = exact_json(process_path, label=str(process_path))
    pid1_entries = (process_identity.get("pid1") or {}).get("environ")
    if not isinstance(pid1_entries, list):
        raise GateError(f"{process_path}: PID1 environment is missing")
    pid1_env: dict[str, str] = {}
    for entry in pid1_entries:
        if not isinstance(entry, str) or "=" not in entry:
            raise GateError(f"{process_path}: malformed PID1 environment entry")
        key, value = entry.split("=", 1)
        if key in pid1_env:
            raise GateError(f"{process_path}: duplicate PID1 environment key {key}")
        pid1_env[key] = value
    for key, expected in required_env.items():
        if pid1_env.get(key) != expected:
            raise GateError(
                f"{process_path}: PID1 expected {key}={expected}, "
                f"got {pid1_env.get(key)!r}"
            )
    runtime_attestation_path = (
        arm_dir / "logs" / "fr13_fixed32_runtime_attestation.json"
    )
    try:
        runtime_attestation = validate_runtime_attestation(
            exact_json(
                runtime_attestation_path,
                label=str(runtime_attestation_path),
            )
        )
    except Fixed32ContractError as error:
        raise GateError(
            f"{runtime_attestation_path}: invalid runtime attestation: {error}"
        ) from error
    runtime_fa2 = runtime_attestation["forked_fa2"]
    if (
        runtime_fa2["source"].get("path")
        != str(fixed32_contract.CONTAINER_FA2_SOURCE)
        or runtime_fa2["destination"].get("path")
        != str(CONTAINER_FA2_DESTINATION)
    ):
        raise GateError(
            f"{runtime_attestation_path}: runtime FA2 paths differ from the "
            "fixed32 contract"
        )
    secret_identity_path = (
        arm_dir / "logs" / "fr13_fixed32_ingress_secret_identity.json"
    )
    secret_identity = exact_json(
        secret_identity_path,
        label=str(secret_identity_path),
    )
    exact_keys(
        secret_identity,
        {
            "schema",
            "path",
            "regular",
            "symlink",
            "uid",
            "gid",
            "mode",
            "bytes",
        },
        str(secret_identity_path),
    )
    if (
        secret_identity["schema"]
        != "fr13-fixed32-ingress-secret-identity-v1"
        or secret_identity["path"] != "/run/fr13_fixed32_ingress_secret"
        or secret_identity["regular"] is not True
        or secret_identity["symlink"] is not False
        or secret_identity["uid"] != 0
        or isinstance(secret_identity["uid"], bool)
        or secret_identity["gid"] != 0
        or isinstance(secret_identity["gid"], bool)
        or secret_identity["mode"] != "0600"
        or isinstance(secret_identity["bytes"], bool)
        or not isinstance(secret_identity["bytes"], int)
        or not 0 < secret_identity["bytes"] <= 16 * 1024
    ):
        raise GateError(
            f"{secret_identity_path}: in-container ingress secret identity "
            "does not prove a root-owned mode-0600 regular copy"
        )
    container_identity_path = arm_dir / "fixed32_container_identity.json"
    container_identity = exact_json(
        container_identity_path,
        label=str(container_identity_path),
    )
    expected_container_identity = {
        "schema": "fr13-fixed32-container-identity-v1",
        "name": f"/fr13-bigdenom-{arm_dir.name}",
        "image_id": fixed32_contract.IMAGE_ID,
        "configured_image": fixed32_contract.IMAGE_REFERENCE,
        "platform": fixed32_contract.IMAGE_OS,
        "running": True,
    }
    if container_identity != expected_container_identity:
        raise GateError(
            f"{container_identity_path}: running container identity differs "
            "from the fixed32 external contract"
        )
    eval_preflight = arm_dir / "eval_offload_preflight.txt"
    if (
        read_text(eval_preflight)
        != "eval offload: configured evaluator reachable\n"
    ):
        raise GateError(f"{eval_preflight}: fixed32 evaluator was not offloaded")
    pretask_zero_traffic = validate_fixed32_pretask_zero_traffic(
        arm_dir,
        mode=mode,
    )
    log_path = arm_dir / "docker_full.log"
    log = read_text(log_path)
    needles = (
        FIXED32_PRESEED,
        FIXED32_ENGAGED,
        FIXED32_WORK_ENGAGED,
        mode_spec["topology_needle"],
    )
    for needle in needles:
        if log.count(needle) != 1:
            raise GateError(
                f"{log_path}: expected exactly one current runtime needle {needle!r}"
            )
    other_mode = "hydra27_fixed32" if mode == "tail6_fixed32" else "tail6_fixed32"
    other_needle = FIXED32_MODE_SPECS[other_mode]["topology_needle"]
    if other_needle in log:
        raise GateError(f"{log_path}: emitted both fixed-32 mode topology needles")
    return {
        "container_env": str(env_path),
        "docker_after_canonical_tasks": str(log_path),
        "fixed32_mode": mode,
        "active_drafts": mode_spec["active_drafts"],
        "valid_mask": f"{mode_spec['valid_mask']:#010x}",
        "draft_tokens_per_event": expected_tokens,
        "required_container_env": required_env,
        "pid1_required_env_exact": True,
        "pretask_zero_traffic": pretask_zero_traffic,
        "runtime_attestation": {
            "path": str(runtime_attestation_path),
            "sha256": sha256_file(runtime_attestation_path),
            "canonical_sha256": runtime_attestation[
                "overall_canonical_sha256"
            ],
            "vllm": runtime_attestation["vllm"],
            "forked_fa2": runtime_attestation["forked_fa2"],
            "arctic": {
                "name": runtime_attestation["arctic"].get("name"),
                "version": runtime_attestation["arctic"]["version"],
                "canonical_sha256": runtime_attestation["arctic"][
                    "canonical_sha256"
                ],
                "pinned_source_url": runtime_attestation["arctic"][
                    "pinned_source_url"
                ],
                "pinned_source_sha256": runtime_attestation["arctic"][
                    "pinned_source_sha256"
                ],
                "cache_class_module": runtime_attestation["arctic"][
                    "cache_class_module"
                ],
                "cache_class_qualname": runtime_attestation["arctic"][
                    "cache_class_qualname"
                ],
            },
        },
        "container_identity": {
            "path": str(container_identity_path),
            "sha256": sha256_file(container_identity_path),
            **container_identity,
        },
        "ingress_secret_identity": {
            "path": str(secret_identity_path),
            "sha256": sha256_file(secret_identity_path),
            **secret_identity,
        },
        "runtime_needles": list(needles),
        "eval_offload_preflight": {
            "path": str(eval_preflight),
            "sha256": sha256_file(eval_preflight),
        },
    }


def canonical_json_sha256(payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_runtime_manifest(path: Path) -> dict[str, Any]:
    payload = exact_json(path, label=str(path))
    digest = payload.get("overall_canonical_sha256")
    unsigned = {
        key: value
        for key, value in payload.items()
        if key != "overall_canonical_sha256"
    }
    if digest != canonical_json_sha256(unsigned):
        raise GateError(f"{path}: runtime manifest canonical digest mismatch")
    if payload.get("schema") != "fr13-runtime-manifest-v1":
        raise GateError(f"{path}: wrong runtime manifest schema")
    if payload.get("profile") != RUNTIME_MANIFEST_PROFILE:
        raise GateError(f"{path}: wrong runtime manifest profile")
    if payload.get("sequence") != RUNTIME_MANIFEST_SEQUENCE:
        raise GateError(f"{path}: wrong runtime manifest sequence")
    return payload


def load_external_manifest(path: Path) -> dict[str, Any]:
    try:
        return validate_external_manifest(exact_json(path, label=str(path)))
    except Fixed32ContractError as error:
        raise GateError(f"{path}: invalid external manifest: {error}") from error


def validate_external_fingerprint(runroot: Path) -> dict[str, Any]:
    at_launch = runroot / "external_manifest.at_launch.json"
    at_end = runroot / "external_manifest.at_end.json"
    launch = load_external_manifest(at_launch)
    end = load_external_manifest(at_end)
    if launch != end or at_launch.read_bytes() != at_end.read_bytes():
        raise GateError(f"{runroot}: external manifest changed during the campaign")
    return {
        "at_launch": str(at_launch),
        "at_end": str(at_end),
        "byte_equal": True,
        "schema": launch["schema"],
        "image": launch["image"],
        "forked_fa2": launch["forked_fa2"],
        "model": {
            "root": launch["model"]["root"],
            "file_count": launch["model"]["file_count"],
            "canonical_sha256": launch["model"]["canonical_sha256"],
        },
        "arctic_source": launch["arctic_source"],
        "overall_canonical_sha256": launch["overall_canonical_sha256"],
        "file_sha256": sha256_file(at_launch),
    }


def validate_source_fingerprint(repo: Path, runroot: Path) -> dict[str, Any]:
    at_launch = runroot / "runtime_manifest.at_launch.json"
    at_end = runroot / "runtime_manifest.at_end.json"
    launch = load_runtime_manifest(at_launch)
    end = load_runtime_manifest(at_end)
    if launch != end:
        raise GateError(f"{runroot}: runtime manifest changed during the campaign")
    try:
        current = build_runtime_manifest(
            repo,
            profile=RUNTIME_MANIFEST_PROFILE,
            sequence=RUNTIME_MANIFEST_SEQUENCE,
        )
    except RuntimeManifestError as error:
        raise GateError(f"cannot rebuild current runtime manifest: {error}") from error
    if current != end:
        raise GateError(
            f"{runroot}: current runtime closure does not match the campaign manifest"
        )
    summary = launch.get("summary")
    if (
        not isinstance(summary, dict)
        or summary.get("file_count") != 62
        or summary.get("python_package_file_count") != 25
    ):
        raise GateError(f"{at_launch}: runtime closure cardinality is not pinned")
    return {
        "at_launch": str(at_launch),
        "at_end": str(at_end),
        "byte_equal": True,
        "matches_current_runtime_closure": True,
        "profile": RUNTIME_MANIFEST_PROFILE,
        "sequence": RUNTIME_MANIFEST_SEQUENCE,
        "file_count": summary["file_count"],
        "python_package_file_count": summary["python_package_file_count"],
        "overall_canonical_sha256": launch["overall_canonical_sha256"],
    }


def load_windows(
    arm_dir: Path,
    task_dirs: list[Path],
    expected_tokens: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    windows = []
    for task_dir in task_dirs:
        pre_artifact = load_metric_artifact(task_dir / "vllm_metrics_pre.txt")
        post_artifact = load_metric_artifact(task_dir / "vllm_metrics_post.txt")
        pre = pre_artifact["values"]
        post = post_artifact["values"]
        pre_labels = pre_artifact["labels"]
        post_labels = post_artifact["labels"]
        if pre_labels != post_labels:
            raise GateError(f"{task_dir.name}: pre/post required metric labels differ")
        if pre_labels != EXPECTED_METRIC_LABELS:
            raise GateError(
                f"{task_dir.name}: required metric labels do not match the "
                "pinned qwen3.6-27b series"
            )
        for snapshot_name, snapshot in (("pre", pre), ("post", post)):
            snapshot_steps = integral(
                snapshot["wall_steps"],
                f"{task_dir.name}:{snapshot_name} wall steps",
            )
            snapshot_attempts = integral(
                snapshot["wall_attempts"],
                f"{task_dir.name}:{snapshot_name} wall attempts",
            )
            snapshot_rejected = integral(
                snapshot["wall_rejected"],
                f"{task_dir.name}:{snapshot_name} wall rejected",
            )
            if snapshot_attempts != snapshot_steps + snapshot_rejected:
                raise GateError(
                    f"{task_dir.name}: {snapshot_name} wall attempts != retained "
                    f"+ rejected: {snapshot_attempts} != {snapshot_steps} + "
                    f"{snapshot_rejected}"
                )
        delta: dict[str, float] = {}
        for key in METRICS:
            value = post[key] - pre[key]
            if value < -1e-9:
                raise GateError(f"{task_dir.name}: counter {key} regressed")
            delta[key] = max(0.0, value)
        for key in set(METRICS) - {"wall_rejected"}:
            if delta[key] <= 0:
                raise GateError(f"{task_dir.name}: non-positive {key} delta")
        wall_steps = integral(delta["wall_steps"], f"{task_dir.name}:wall steps")
        wall_attempts = integral(
            delta["wall_attempts"], f"{task_dir.name}:wall attempts"
        )
        wall_rejected = integral(
            delta["wall_rejected"], f"{task_dir.name}:wall rejected"
        )
        if wall_attempts != wall_steps + wall_rejected:
            raise GateError(
                f"{task_dir.name}: wall attempts != retained + rejected: "
                f"{wall_attempts} != {wall_steps} + {wall_rejected}"
            )
        if wall_rejected != 0:
            raise GateError(
                f"{task_dir.name}: censored wall intervals in task window: "
                f"{wall_rejected}"
            )
        for family in ("fwd", "wall"):
            steps = integral(delta[f"{family}_steps"], f"{task_dir.name}:{family}")
            drafts = integral(
                delta[f"{family}_drafts"], f"{task_dir.name}:{family} drafts"
            )
            if steps < MIN_TASK_COUNTER_STEPS:
                raise GateError(
                    f"{task_dir.name}: {family} exposure below "
                    f"{MIN_TASK_COUNTER_STEPS} retained steps: {steps}"
                )
            if not steps <= drafts <= concurrency * steps:
                raise GateError(
                    f"{task_dir.name}: {family} drafts/step is outside "
                    f"[1, {concurrency}]: drafts={drafts}, steps={steps}"
                )
        expected_draft_tokens = delta["spec_drafts"] * expected_tokens
        if not math.isclose(
            delta["spec_tokens"], expected_draft_tokens, rel_tol=0, abs_tol=1e-6
        ):
            raise GateError(
                f"{task_dir.name}: draft-token ratio is not exactly {expected_tokens}"
            )
        windows.append(
            {
                "task_id": task_dir.name,
                "pre": pre,
                "post": post,
                "delta": delta,
                "metric_labels": pre_labels,
                "metric_artifacts": {
                    "pre": pre_artifact["identity"],
                    "post": post_artifact["identity"],
                },
                "fixed32_metrics": {
                    "pre": pre_artifact["fixed32"],
                    "post": post_artifact["fixed32"],
                },
                "fwd_span": (
                    integral(pre["fwd_steps"], f"{task_dir.name}:fwd pre"),
                    integral(post["fwd_steps"], f"{task_dir.name}:fwd post"),
                ),
                "wall_span": (
                    integral(pre["wall_steps"], f"{task_dir.name}:wall pre"),
                    integral(post["wall_steps"], f"{task_dir.name}:wall post"),
                ),
            }
        )
    windows = sorted(windows, key=lambda item: item["task_id"])
    first_labels = windows[0]["metric_labels"]
    if any(window["metric_labels"] != first_labels for window in windows[1:]):
        raise GateError(f"{arm_dir}: required metric labels differ across tasks")
    return windows


def validate_flush_chain(
    arm_dir: Path,
    task_dirs: list[Path],
    windows: list[dict[str, Any]],
    *,
    mode: str,
    producer_pid: int,
    complete_steps: int,
    server_capacity: int,
    dataset_record_digests: dict[str, str],
) -> dict[str, Any]:
    pid_path = arm_dir / "logs" / "fr13_fixed32_engine_pid"
    if read_text(pid_path) != f"{producer_pid}\n":
        raise GateError(f"{pid_path}: EngineCore PID file is not exact")
    mode_path = arm_dir / "logs" / "fr13_fixed32_mode.flag"
    if read_text(mode_path) != f"{mode}\n":
        raise GateError(f"{mode_path}: fixed32 mode sidecar is not exact")

    ready_path = arm_dir / "fixed32_ready_ack.json"
    ready = validate_fixed32_ack(
        exact_json(ready_path, label=str(ready_path)),
        label=str(ready_path),
        mode=mode,
        producer_pid=producer_pid,
    )
    if (
        ready["generation"] != 0
        or ready["nonce"] != FLUSH_READY_NONCE
        or ready["action"] != "ready"
        or ready["counters"]
        != {
            "pure_decode_forward_steps": 0,
            "complete_work_census_events": 0,
            "work_census_first_forward_step": None,
            "work_census_last_forward_step": None,
            "sfwd_pending": 0,
            "dfwd_pending": 0,
            "cfwd_pending": 0,
        }
    ):
        raise GateError(f"{ready_path}: generation-zero ready ack is not pristine")

    windows_by_task = {window["task_id"]: window for window in windows}
    task_acks: list[dict[str, Any]] = []
    task_reports: dict[str, Any] = {}
    runtime_by_generation: dict[int, dict[str, Any]] = {}
    for task_dir in task_dirs:
        boundary_path = task_dir / "fixed32_task_boundary.json"
        boundary = exact_json(boundary_path, label=str(boundary_path))
        exact_keys(
            boundary,
            {
                "schema",
                "instance_id",
                "mode",
                "producer_pid",
                "pre",
                "post",
                "pre_runtime_snapshot",
                "post_runtime_snapshot",
                "forward_step_interval",
            },
            str(boundary_path),
        )
        if (
            boundary["schema"] != FIXED32_BOUNDARY_SCHEMA
            or boundary["instance_id"] != task_dir.name
            or boundary["mode"] != mode
            or boundary["producer_pid"] != producer_pid
        ):
            raise GateError(f"{boundary_path}: task boundary identity mismatch")
        pre = validate_fixed32_ack(
            boundary["pre"],
            label=f"{boundary_path}:pre",
            mode=mode,
            producer_pid=producer_pid,
        )
        post = validate_fixed32_ack(
            boundary["post"],
            label=f"{boundary_path}:post",
            mode=mode,
            producer_pid=producer_pid,
        )
        if pre["action"] != "snapshot" or post["action"] != "snapshot":
            raise GateError(f"{boundary_path}: task boundaries must be snapshot acks")
        if pre["generation"] >= post["generation"]:
            raise GateError(f"{boundary_path}: pre generation is not before post")
        start = pre["counters"]["pure_decode_forward_steps"]
        end = post["counters"]["pure_decode_forward_steps"]
        event_delta = (
            post["counters"]["complete_work_census_events"]
            - pre["counters"]["complete_work_census_events"]
        )
        expected_interval = {
            "start_forward_step": start,
            "end_forward_step": end,
            "expected_complete_events": event_delta,
        }
        if boundary["forward_step_interval"] != expected_interval:
            raise GateError(f"{boundary_path}: derived forward-step interval mismatch")
        if end <= start or event_delta != end - start:
            raise GateError(f"{boundary_path}: incomplete task census interval")

        window = windows_by_task[task_dir.name]
        if tuple(window["fwd_span"]) != (start, end):
            raise GateError(
                f"{boundary_path}: flush interval does not match metrics fwd span"
            )
        runtime_reports: dict[str, Any] = {}
        for snapshot, ack in (("pre", pre), ("post", post)):
            metrics_path = task_dir / f"vllm_metrics_{snapshot}.txt"
            fixed32_values = window["fixed32_metrics"][snapshot]
            if (
                fixed32_values["pure_decode_forward_steps"]
                != ack["counters"]["pure_decode_forward_steps"]
            ):
                raise GateError(f"{metrics_path}: fixed32 step metric/ack mismatch")
            if (
                fixed32_values["complete_work_census_events"]
                != ack["counters"]["complete_work_census_events"]
            ):
                raise GateError(f"{metrics_path}: fixed32 census metric/ack mismatch")
            runtime_path = (
                arm_dir
                / "logs"
                / f"fr13_fixed32_boundary_snapshot.{ack['generation']}.json"
            )
            runtime_reports[snapshot] = validate_runtime_boundary_snapshot(
                runtime_path,
                ack=ack,
                server_capacity=server_capacity,
                metrics_path=metrics_path,
                metric_values=window[snapshot],
                reference=boundary[f"{snapshot}_runtime_snapshot"],
                census_path=(
                    arm_dir / "logs" / "fr13_fixed32_work_census.jsonl"
                ),
            )
            runtime_by_generation[ack["generation"]] = runtime_reports[
                snapshot
            ]

        metadata_path = task_dir / "runner_metadata.json"
        metadata = exact_json(metadata_path, label=str(metadata_path))
        if metadata.get("fixed32_task_boundary") != boundary:
            raise GateError(f"{metadata_path}: embedded task boundary differs from artifact")
        if (
            metadata.get("fixed32_dataset_record_sha256")
            != dataset_record_digests[task_dir.name]
        ):
            raise GateError(
                f"{metadata_path}: fixed32 dataset record digest mismatch"
            )
        task_acks.extend((pre, post))
        task_reports[task_dir.name] = {
            "path": str(boundary_path),
            "sha256": sha256_file(boundary_path),
            "pre_generation": pre["generation"],
            "post_generation": post["generation"],
            "forward_step_interval": [start, end],
            "runtime_snapshots": runtime_reports,
        }

    result_path = arm_dir / "fixed32_final_flush.json"
    result = exact_json(result_path, label=str(result_path))
    exact_keys(result, {"schema", "ack"}, str(result_path))
    if result["schema"] != FLUSH_RESULT_SCHEMA:
        raise GateError(f"{result_path}: wrong flush client result schema")
    final_ack = validate_fixed32_ack(
        result["ack"],
        label=f"{result_path}:ack",
        mode=mode,
        producer_pid=producer_pid,
    )
    if final_ack["action"] != "final":
        raise GateError(f"{result_path}: terminal ack action is not final")
    final_runtime_path = (
        arm_dir
        / "logs"
        / f"fr13_fixed32_boundary_snapshot.{final_ack['generation']}.json"
    )
    final_runtime = validate_runtime_boundary_snapshot(
        final_runtime_path,
        ack=final_ack,
        server_capacity=server_capacity,
        metrics_path=None,
        metric_values=None,
        reference=None,
        census_path=arm_dir / "logs" / "fr13_fixed32_work_census.jsonl",
    )
    runtime_by_generation[final_ack["generation"]] = final_runtime
    census_path = arm_dir / "logs" / "fr13_fixed32_work_census.jsonl"
    census_lines = read_text(census_path).splitlines()
    if not census_lines:
        raise GateError(f"{census_path}: terminal census is empty")
    terminal = exact_json_text(
        census_lines[-1],
        label=f"{census_path}:terminal",
    )
    terminal_nonpure_by_batch = strict_nonnegative_int_map(
        terminal.get("nonpure_committer_replays_by_batch"),
        expected_keys={"1", "2", "3", "4"},
        label=f"{census_path}:terminal.nonpure_committer_replays_by_batch",
    )
    terminal_nonpure_dispatch = strict_nonnegative_int_map(
        terminal.get("nonpure_dispatch"),
        expected_keys={
            "guarded_steps",
            "piecewise_steps",
            "none_steps",
            "forbidden_full_steps",
        },
        label=f"{census_path}:terminal.nonpure_dispatch",
    )
    if (
        final_runtime["committer"][
            "nonpure_committer_replays_by_batch"
        ]
        != terminal_nonpure_by_batch
        or final_runtime["committer"]["nonpure_dispatch"]
        != terminal_nonpure_dispatch
    ):
        raise GateError(
            f"{census_path}: final runtime nonpure ledgers differ from terminal"
        )
    stderr_path = arm_dir / "fixed32_final_flush.stderr"
    if read_text(stderr_path) != "":
        raise GateError(f"{stderr_path}: terminal flush wrote stderr")

    ack_path = arm_dir / "logs" / "fr13_fixed32_flush_ack.json"
    current_ack = validate_fixed32_ack(
        exact_json(ack_path, label=str(ack_path)),
        label=str(ack_path),
        mode=mode,
        producer_pid=producer_pid,
    )
    if current_ack != final_ack:
        raise GateError(f"{ack_path}: current ack differs from terminal result")

    request_path = arm_dir / "logs" / "fr13_fixed32_flush_request.json"
    request = exact_json(request_path, label=str(request_path))
    exact_keys(request, FLUSH_REQUEST_KEYS, str(request_path))
    if (
        request["schema"] != FLUSH_REQUEST_SCHEMA
        or request["mode"] != mode
        or request["producer_pid"] != producer_pid
        or request["action"] != "final"
        or request["generation"] != final_ack["generation"]
        or request["prev_generation"] != final_ack["generation"] - 1
        or request["nonce"] != final_ack["nonce"]
    ):
        raise GateError(f"{request_path}: terminal request/ack binding mismatch")
    logs_dir = arm_dir / "logs"
    temp_residue = []
    for pattern in (
        ".fr13_fixed32_flush_request.json.*.tmp",
        "fr13_fixed32_flush_request.json.tmp.*",
        "fr13_fixed32_flush_ack.json.tmp.*",
        "fr13_fixed32_work_census.jsonl.tmp.*",
        "fr13_fixed32_boundary_snapshot.*.json.tmp.*",
    ):
        temp_residue.extend(logs_dir.glob(pattern))
    temp_residue.extend(
        task_dir / "fixed32_task_boundary.json.tmp"
        for task_dir in task_dirs
        if (task_dir / "fixed32_task_boundary.json.tmp").exists()
    )
    if temp_residue:
        raise GateError(
            f"{arm_dir}: stale atomic-write temporary files: "
            f"{sorted(str(path) for path in temp_residue)}"
        )

    generations = [ready, *task_acks, final_ack]
    ordered = sorted(generations, key=lambda ack: ack["generation"])
    expected_generations = list(range(len(ordered)))
    actual_generations = [ack["generation"] for ack in ordered]
    if actual_generations != expected_generations:
        raise GateError(
            f"{arm_dir}: flush generation chain is not exact: {actual_generations}"
        )
    if [ack["action"] for ack in ordered] != [
        "ready",
        *(["snapshot"] * (2 * len(task_dirs))),
        "final",
    ]:
        raise GateError(f"{arm_dir}: flush action chain is not exact")
    nonces = [ack["nonce"] for ack in ordered]
    if len(nonces) != len(set(nonces)) or nonces[0] != FLUSH_READY_NONCE:
        raise GateError(f"{arm_dir}: flush nonce chain is not unique")
    if any(
        current["counters"]["pure_decode_forward_steps"]
        < previous["counters"]["pure_decode_forward_steps"]
        or current["counters"]["complete_work_census_events"]
        < previous["counters"]["complete_work_census_events"]
        for previous, current in zip(ordered, ordered[1:], strict=False)
    ):
        raise GateError(f"{arm_dir}: flush counters regress across generations")
    if set(runtime_by_generation) != set(expected_generations[1:]):
        raise GateError(
            f"{arm_dir}: runtime snapshot generations do not match flush chain"
        )
    ordered_runtime = [
        runtime_by_generation[generation]
        for generation in expected_generations[1:]
    ]
    for previous, current in zip(
        ordered_runtime,
        ordered_runtime[1:],
        strict=False,
    ):
        previous_committer = previous["committer"]
        current_committer = current["committer"]
        if (
            current_committer["actual_replays_enqueued"]
            < previous_committer["actual_replays_enqueued"]
            or current_committer["nonpure_committer_replays_enqueued"]
            < previous_committer["nonpure_committer_replays_enqueued"]
            or any(
                current_committer[map_name][batch]
                < previous_committer[map_name][batch]
                for map_name in (
                    "actual_replays_by_batch",
                    "nonpure_committer_replays_by_batch",
                )
                for batch in ("1", "2", "3", "4")
            )
            or any(
                current_committer["nonpure_dispatch"][key]
                < previous_committer["nonpure_dispatch"][key]
                for key in (
                    "guarded_steps",
                    "piecewise_steps",
                    "none_steps",
                    "forbidden_full_steps",
                )
            )
        ):
            raise GateError(
                f"{arm_dir}: runtime committer/nonpure ledgers regress "
                "across generations"
            )

    final_counters = final_ack["counters"]
    if (
        final_counters["pure_decode_forward_steps"] != complete_steps
        or final_counters["complete_work_census_events"] != complete_steps
        or final_counters["work_census_first_forward_step"] != 0
        or final_counters["work_census_last_forward_step"] != complete_steps - 1
    ):
        raise GateError(f"{arm_dir}: final flush counters do not close complete stream")
    expected_runtime_paths = {
        arm_dir
        / "logs"
        / f"fr13_fixed32_boundary_snapshot.{ack['generation']}.json"
        for ack in [*task_acks, final_ack]
    }
    actual_runtime_paths = set(
        (arm_dir / "logs").glob("fr13_fixed32_boundary_snapshot.*.json")
    )
    if actual_runtime_paths != expected_runtime_paths:
        raise GateError(
            f"{arm_dir}: runtime boundary snapshot generation set is not exact"
        )
    return {
        "ready": {"path": str(ready_path), "sha256": sha256_file(ready_path)},
        "tasks": task_reports,
        "final": {
            "result_path": str(result_path),
            "result_sha256": sha256_file(result_path),
            "request_path": str(request_path),
            "request_sha256": sha256_file(request_path),
            "ack_path": str(ack_path),
            "ack_sha256": sha256_file(ack_path),
            "generation": final_ack["generation"],
            "counters": final_counters,
            "runtime_snapshot": final_runtime,
        },
        "generation_chain": actual_generations,
        "all_pending_counts_zero": True,
        "task_intervals_bound_to_metrics": True,
    }


def unique_sidecar(
    sidecar_dir: Path,
    arm: str,
    concurrency: int,
    expected_pid: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    if not sidecar_dir.is_dir():
        raise GateError(f"missing per-step sidecar directory: {sidecar_dir}")
    prefix = f"{arm}.json.samples."
    paths = sorted(
        path
        for path in sidecar_dir.iterdir()
        if path.is_file() and path.name.startswith(prefix)
    )
    if len(paths) != 1:
        raise GateError(
            f"{sidecar_dir}: expected one per-step sidecar for {arm}, found {paths}"
        )
    path = paths[0]
    payload = exact_json(path, label=str(path))
    exact_keys(
        payload,
        {
            "schema",
            "pid",
            "final",
            "fwd_indices",
            "fwd_drafts",
            "fwd_ms",
            "fwd_cg",
            "fwd_host_ms",
            "fwd_exec_ms",
            "fwd_cpu_tail_ms",
            "wall_drafts",
            "wall_ms",
            "wall_fwd_indices",
            "samples_capped",
        },
        str(path),
    )
    if payload["schema"] != SFWD_SAMPLE_SIDECAR_SCHEMA:
        raise GateError(f"{path}: wrong sidecar schema")
    if payload["final"] is not True:
        raise GateError(f"{path}: per-step sidecar lacks an explicit final flush")
    try:
        suffix_pid = int(path.name.removeprefix(prefix))
    except ValueError as error:
        raise GateError(f"{path}: sidecar filename has no integral PID") from error
    if payload["pid"] != suffix_pid:
        raise GateError(f"{path}: sidecar payload PID does not match filename")
    if suffix_pid != expected_pid:
        raise GateError(
            f"{path}: sidecar PID {suffix_pid} does not match recorded "
            f"VLLM::EngineCore PID {expected_pid}"
        )
    if payload["samples_capped"] is not False:
        raise GateError(f"{path}: sidecar samples are capped or cap state is absent")
    fwd_fields = (
        "fwd_indices",
        "fwd_drafts",
        "fwd_ms",
        "fwd_cg",
        "fwd_host_ms",
        "fwd_exec_ms",
        "fwd_cpu_tail_ms",
    )
    wall_fields = ("wall_drafts", "wall_ms", "wall_fwd_indices")
    for key in (*fwd_fields, *wall_fields):
        if not isinstance(payload[key], list):
            raise GateError(f"{path}: {key} must be an array")
    lengths = {key: len(payload[key]) for key in (*fwd_fields, *wall_fields)}
    if len({lengths[key] for key in fwd_fields}) != 1:
        raise GateError(f"{path}: forward sidecar array lengths differ: {lengths}")
    if len({lengths[key] for key in wall_fields}) != 1:
        raise GateError(f"{path}: wall sidecar array lengths differ: {lengths}")
    fwd_indices = strict_nonnegative_int_list(
        payload["fwd_indices"], label=f"{path}:fwd_indices"
    )
    wall_fwd_indices = strict_nonnegative_int_list(
        payload["wall_fwd_indices"], label=f"{path}:wall_fwd_indices"
    )
    if any(
        right <= left
        for left, right in zip(fwd_indices, fwd_indices[1:], strict=False)
    ):
        raise GateError(f"{path}: fwd_indices are not strictly increasing and unique")
    if any(
        right <= left
        for left, right in zip(
            wall_fwd_indices, wall_fwd_indices[1:], strict=False
        )
    ):
        raise GateError(
            f"{path}: wall_fwd_indices are not strictly increasing and unique"
        )
    fwd_index_positions = {
        fwd_index: position for position, fwd_index in enumerate(fwd_indices)
    }
    missing_wall_fwd_indices = [
        fwd_index
        for fwd_index in wall_fwd_indices
        if fwd_index not in fwd_index_positions
    ]
    if missing_wall_fwd_indices:
        raise GateError(
            f"{path}: wall_fwd_indices do not bind to retained forward samples: "
            f"{missing_wall_fwd_indices[:8]}"
        )
    numeric_fields = (
        "fwd_drafts",
        "fwd_ms",
        "fwd_host_ms",
        "fwd_exec_ms",
        "fwd_cpu_tail_ms",
        "wall_drafts",
        "wall_ms",
    )
    for key in numeric_fields:
        if any(type(value) not in {int, float} for value in payload[key]):
            raise GateError(f"{path}: {key} contains a non-numeric value")
    if any(
        not isinstance(value, str) or not value
        for value in payload["fwd_cg"]
    ):
        raise GateError(f"{path}: fwd_cg contains an invalid dispatch tag")
    arrays = {
        "fwd_indices": np.asarray(fwd_indices, dtype=np.int64),
        "fwd_drafts": np.asarray(payload["fwd_drafts"], dtype=np.float64),
        "fwd_ms": np.asarray(payload["fwd_ms"], dtype=np.float64),
        "fwd_full": np.asarray(
            [value == "FULL" for value in payload["fwd_cg"]], dtype=np.bool_
        ),
        "wall_drafts": np.asarray(payload["wall_drafts"], dtype=np.float64),
        "wall_ms": np.asarray(payload["wall_ms"], dtype=np.float64),
        "wall_fwd_indices": np.asarray(wall_fwd_indices, dtype=np.int64),
    }
    for key in ("fwd_drafts", "fwd_ms", "wall_drafts", "wall_ms"):
        values = arrays[key]
        if not np.all(np.isfinite(values)):
            raise GateError(f"{path}: {key} contains non-finite values")
    if np.any(arrays["fwd_ms"] <= 0) or np.any(arrays["wall_ms"] <= 0):
        raise GateError(f"{path}: timing arrays contain non-positive values")
    for key in ("fwd_drafts", "wall_drafts"):
        values = arrays[key]
        if (
            np.any(values < 1)
            or np.any(values > concurrency)
            or np.any(values != np.rint(values))
        ):
            raise GateError(
                f"{path}: {key} is inconsistent with inferred concurrency {concurrency}"
            )
    paired_positions = np.asarray(
        [fwd_index_positions[index] for index in wall_fwd_indices],
        dtype=np.int64,
    )
    if not np.array_equal(
        arrays["wall_drafts"],
        arrays["fwd_drafts"][paired_positions],
    ):
        raise GateError(
            f"{path}: wall occupancy differs from its paired forward occupancy"
        )
    return (
        payload,
        arrays,
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "pid": suffix_pid,
            "pid_bound_to_runlog": True,
            "final": True,
            "samples_capped": False,
            "array_lengths": lengths,
            "forward_indices_strictly_increasing_unique": True,
            "wall_predecessor_indices_strictly_increasing_unique": True,
            "wall_predecessors_bound_to_forward_samples": True,
            "wall_forward_occupancy_equal": True,
        },
    )


def unique_main_sidecar(
    sidecar_dir: Path,
    arm: str,
    expected_pid: int,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    prefix = f"{arm}.json."
    paths = sorted(
        path
        for path in sidecar_dir.iterdir()
        if path.is_file()
        and path.name.startswith(prefix)
        and not path.name.startswith(f"{arm}.json.samples.")
    )
    if len(paths) != 1:
        raise GateError(
            f"{sidecar_dir}: expected one main timer sidecar for {arm}, found {paths}"
        )
    path = paths[0]
    payload = exact_json(path, label=str(path))
    exact_keys(
        payload,
        {
            "schema",
            "pid",
            "final",
            "decode_forward_gpu_seconds",
            "n_pure_decode_steps_timed",
            "n_forward_starts",
            "n_forward_dropped",
            "n_forward_pending",
            "n_drafts_in_timed_steps",
            "decode_step_wall_seconds",
            "n_drafts_in_wall_steps",
            "n_wall_steps",
            "n_wall_rejected",
            "n_wall_chain_resets",
            "n_wall_request_set_resets",
            "n_wall_invalid_request_ids",
            "n_wall_bookkeeping_errors",
            "wall_chain_open",
            "wall_open_fwd_index",
            "wall_open_drafts",
            "wall_cap_s",
            "metric_name",
            "note",
        },
        str(path),
    )
    if payload["schema"] != SFWD_MAIN_SIDECAR_SCHEMA:
        raise GateError(f"{path}: wrong main sidecar schema")
    try:
        suffix_pid = int(path.name.removeprefix(prefix))
    except ValueError as error:
        raise GateError(f"{path}: main sidecar filename has no integral PID") from error
    if payload["pid"] != suffix_pid or suffix_pid != expected_pid:
        raise GateError(
            f"{path}: main sidecar PID is not bound to the recorded EngineCore"
        )
    if payload["final"] is not True:
        raise GateError(f"{path}: main sidecar lacks an explicit final flush")
    wall_cap = strict_nonnegative_number(
        payload["wall_cap_s"], label=f"{path}:wall_cap_s"
    )
    if not math.isclose(wall_cap, 1.5, rel_tol=0, abs_tol=0):
        raise GateError(f"{path}: main sidecar wall cap is not exactly 1.5 seconds")
    if (
        payload["metric_name"]
        != "vllm:fr13_decode_forward_gpu_seconds_total"
        or not isinstance(payload["note"], str)
        or not payload["note"]
    ):
        raise GateError(f"{path}: main sidecar metric identity is not exact")
    counters = {
        "fwd_steps": strict_nonnegative_int(
            payload["n_pure_decode_steps_timed"],
            label=f"{path}:fwd steps",
        ),
        "fwd_started": strict_nonnegative_int(
            payload["n_forward_starts"], label=f"{path}:fwd starts"
        ),
        "fwd_dropped": strict_nonnegative_int(
            payload["n_forward_dropped"], label=f"{path}:fwd dropped"
        ),
        "fwd_pending": strict_nonnegative_int(
            payload["n_forward_pending"], label=f"{path}:fwd pending"
        ),
        "fwd_drafts": strict_nonnegative_int(
            payload["n_drafts_in_timed_steps"], label=f"{path}:fwd drafts"
        ),
        "wall_steps": strict_nonnegative_int(
            payload["n_wall_steps"], label=f"{path}:wall steps"
        ),
        "wall_drafts": strict_nonnegative_int(
            payload["n_drafts_in_wall_steps"], label=f"{path}:wall drafts"
        ),
        "wall_rejected": strict_nonnegative_int(
            payload["n_wall_rejected"], label=f"{path}:wall rejected"
        ),
        "wall_chain_resets": strict_nonnegative_int(
            payload["n_wall_chain_resets"], label=f"{path}:wall chain resets"
        ),
        "wall_request_set_resets": strict_nonnegative_int(
            payload["n_wall_request_set_resets"],
            label=f"{path}:wall request-set resets",
        ),
        "wall_invalid_request_ids": strict_nonnegative_int(
            payload["n_wall_invalid_request_ids"],
            label=f"{path}:wall invalid request IDs",
        ),
        "wall_bookkeeping_errors": strict_nonnegative_int(
            payload["n_wall_bookkeeping_errors"],
            label=f"{path}:wall bookkeeping errors",
        ),
    }
    if payload["wall_chain_open"] is not True and payload["wall_chain_open"] is not False:
        raise GateError(f"{path}: wall_chain_open must be boolean")
    wall_chain_open = payload["wall_chain_open"]
    wall_open_fwd_index = strict_optional_nonnegative_int(
        payload["wall_open_fwd_index"], label=f"{path}:wall open fwd index"
    )
    wall_open_drafts = strict_nonnegative_int(
        payload["wall_open_drafts"], label=f"{path}:wall open drafts"
    )
    if (
        counters["wall_rejected"] != 0
        or counters["wall_invalid_request_ids"] != 0
        or counters["wall_bookkeeping_errors"] != 0
        or counters["fwd_dropped"] != 0
        or counters["fwd_pending"] != 0
    ):
        raise GateError(
            f"{path}: formal timer integrity counters are nonzero: "
            f"rejected={counters['wall_rejected']} "
            f"invalid_ids={counters['wall_invalid_request_ids']} "
            f"bookkeeping_errors={counters['wall_bookkeeping_errors']} "
            f"fwd_dropped={counters['fwd_dropped']} "
            f"fwd_pending={counters['fwd_pending']}"
        )
    if counters["wall_request_set_resets"] > counters["wall_chain_resets"]:
        raise GateError(f"{path}: request-set resets exceed all chain resets")
    if counters["fwd_started"] != (
        counters["fwd_steps"]
        + counters["fwd_dropped"]
        + counters["fwd_pending"]
    ):
        raise GateError(f"{path}: forward start/drain accounting is not exact")
    open_count = int(wall_chain_open)
    if counters["fwd_started"] != (
        counters["wall_steps"]
        + counters["wall_rejected"]
        + counters["wall_chain_resets"]
        + open_count
    ):
        raise GateError(f"{path}: wall predecessor disposition is not exact")
    if wall_chain_open:
        if (
            counters["fwd_started"] == 0
            or wall_open_fwd_index != counters["fwd_started"] - 1
            or not 1 <= wall_open_drafts
        ):
            raise GateError(f"{path}: open wall predecessor state is inconsistent")
    elif wall_open_fwd_index is not None or wall_open_drafts != 0:
        raise GateError(f"{path}: closed wall chain retains open state")
    expected_lengths = {
        "fwd": counters["fwd_steps"],
        "wall": counters["wall_steps"],
    }
    actual_lengths = {
        "fwd": len(arrays["fwd_ms"]),
        "wall": len(arrays["wall_ms"]),
    }
    if actual_lengths != expected_lengths:
        raise GateError(
            f"{path}: final sample lengths do not match main counters: "
            f"{actual_lengths} != {expected_lengths}"
        )
    if not np.array_equal(
        arrays["fwd_indices"],
        np.arange(counters["fwd_steps"], dtype=np.int64),
    ):
        raise GateError(f"{path}: final forward indices are not exact and contiguous")
    if wall_chain_open and (
        integral(
            arrays["fwd_drafts"][wall_open_fwd_index],
            f"{path}:wall open forward occupancy",
        )
        != wall_open_drafts
    ):
        raise GateError(f"{path}: open wall occupancy differs from its forward")
    if counters["fwd_steps"] == 0:
        raise GateError(f"{path}: final timer sidecar has no forward samples")
    retained_wall_fraction = counters["wall_steps"] / counters["fwd_steps"]
    if retained_wall_fraction < MIN_RETAINED_WALL_FRACTION:
        raise GateError(
            f"{path}: retained wall fraction {retained_wall_fraction:.6f} "
            f"is below {MIN_RETAINED_WALL_FRACTION:.6f}"
        )
    reconciliation = {
        "forward": reconcile_counter_interval(
            arrays,
            "fwd",
            (0, counters["fwd_steps"]),
            strict_nonnegative_number(
                payload["decode_forward_gpu_seconds"],
                label=f"{path}:decode forward seconds",
            ),
            counters["fwd_drafts"],
        ),
        "wall": reconcile_counter_interval(
            arrays,
            "wall",
            (0, counters["wall_steps"]),
            strict_nonnegative_number(
                payload["decode_step_wall_seconds"],
                label=f"{path}:decode wall seconds",
            ),
            counters["wall_drafts"],
        ),
    }
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "pid": suffix_pid,
        "pid_bound_to_runlog": True,
        "final": True,
        "wall_cap_s": wall_cap,
        "counters": counters,
        "wall_attempts": counters["wall_steps"] + counters["wall_rejected"],
        "wall_chain_open": wall_chain_open,
        "wall_open_fwd_index": wall_open_fwd_index,
        "wall_open_drafts": wall_open_drafts,
        "retained_wall_fraction": retained_wall_fraction,
        "minimum_retained_wall_fraction": MIN_RETAINED_WALL_FRACTION,
        "exact_forward_and_wall_disposition_accounting": True,
        "full_array_reconciliation": reconciliation,
    }


def merged_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if any(end <= start for start, end in spans):
        raise GateError(f"empty or reversed counter interval: {spans}")
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def selected_counter_indices(
    spans: list[tuple[int, int]],
    *,
    available_steps: int,
    label: str,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Return the ordered, de-duplicated counter-index union for task spans."""
    union = merged_spans(spans)
    for start, end in union:
        if start < 0 or end > available_steps:
            raise GateError(
                f"{label}: task counter union {(start, end)} is outside "
                f"available sidecar steps [0, {available_steps})"
            )
    indices = [index for start, end in union for index in range(start, end)]
    if not indices:
        raise GateError(f"{label}: task counter union selected no steps")
    return union, indices


def assert_nonoverlap(spans: list[tuple[int, int]], label: str) -> None:
    ordered = sorted(spans)
    for left, right in zip(ordered, ordered[1:], strict=False):
        if right[0] < left[1]:
            raise GateError(
                f"{label}: B=1 task counter intervals overlap: {left}, {right}"
            )


def select_span(values: np.ndarray, span: tuple[int, int]) -> np.ndarray:
    start, end = span
    available_end = min(end, len(values))
    if available_end <= start:
        return values[:0]
    return values[start:available_end]


def coverage_for_span(length: int, span: tuple[int, int]) -> dict[str, Any]:
    start, end = span
    expected = end - start
    selected = max(0, min(end, length) - start)
    return {
        "counter_interval": [start, end],
        "expected_steps": expected,
        "selected_steps": selected,
        "fraction": selected / expected,
    }


def paired_wall_selection(
    arrays: dict[str, np.ndarray],
    *,
    fwd_span: tuple[int, int],
    wall_span: tuple[int, int],
    label: str,
) -> dict[str, Any]:
    """Bind retained wall samples to their exact predecessor forward samples."""
    fwd_start, fwd_end = fwd_span
    expected_fwd_steps = fwd_end - fwd_start
    if expected_fwd_steps <= 0:
        raise GateError(f"{label}: empty forward span")
    wall_fwd_indices = select_span(arrays["wall_fwd_indices"], wall_span)
    wall_ms = select_span(arrays["wall_ms"], wall_span)
    wall_drafts = select_span(arrays["wall_drafts"], wall_span)
    expected_wall_steps = wall_span[1] - wall_span[0]
    if (
        len(wall_fwd_indices) != expected_wall_steps
        or len(wall_ms) != expected_wall_steps
        or len(wall_drafts) != expected_wall_steps
    ):
        raise GateError(f"{label}: wall pairing arrays do not cover the wall span")
    if np.any(wall_fwd_indices < fwd_start) or np.any(
        wall_fwd_indices >= fwd_end
    ):
        raise GateError(
            f"{label}: retained wall predecessor is outside its forward span"
        )
    if len(np.unique(wall_fwd_indices)) != len(wall_fwd_indices):
        raise GateError(f"{label}: retained wall predecessor indices are not unique")
    paired_fwd_ms = arrays["fwd_ms"][wall_fwd_indices]
    paired_fwd_drafts = arrays["fwd_drafts"][wall_fwd_indices]
    paired_fwd_full = arrays["fwd_full"][wall_fwd_indices]
    if not np.array_equal(wall_drafts, paired_fwd_drafts):
        raise GateError(
            f"{label}: retained wall occupancy differs from paired forward occupancy"
        )
    retained_fraction = len(wall_fwd_indices) / expected_fwd_steps
    if retained_fraction < MIN_RETAINED_WALL_FRACTION:
        raise GateError(
            f"{label}: retained wall fraction {retained_fraction:.6f} "
            f"is below {MIN_RETAINED_WALL_FRACTION:.6f}"
        )
    index_values = [int(value) for value in wall_fwd_indices]
    index_sha256 = hashlib.sha256(
        json.dumps(index_values, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return {
        "fwd_ms": paired_fwd_ms,
        "fwd_drafts": paired_fwd_drafts,
        "fwd_full": paired_fwd_full,
        "wall_ms": wall_ms,
        "wall_drafts": wall_drafts,
        "report": {
            "forward_counter_interval": list(fwd_span),
            "wall_counter_interval": list(wall_span),
            "forward_steps": expected_fwd_steps,
            "paired_wall_steps": len(wall_fwd_indices),
            "retained_wall_fraction": retained_fraction,
            "minimum_retained_wall_fraction": MIN_RETAINED_WALL_FRACTION,
            "paired_forward_index_min": index_values[0],
            "paired_forward_index_max": index_values[-1],
            "paired_forward_indices_sha256": index_sha256,
            "all_wall_predecessors_inside_forward_span": True,
            "wall_forward_occupancy_equal": True,
        },
    }


def reconcile_counter_interval(
    arrays: dict[str, np.ndarray],
    family: str,
    span: tuple[int, int],
    counter_seconds: float,
    counter_drafts: float,
) -> dict[str, Any]:
    start, end = span
    expected_steps = end - start
    ms_values = select_span(arrays[f"{family}_ms"], span)
    draft_values = select_span(arrays[f"{family}_drafts"], span)
    if len(ms_values) != expected_steps or len(draft_values) != expected_steps:
        raise GateError(
            f"{family}: sidecar does not completely cover counter interval "
            f"{span}; selected ms/drafts={len(ms_values)}/{len(draft_values)}"
        )
    sidecar_drafts = integral(
        float(draft_values.sum()), f"{family}: sidecar interval drafts"
    )
    expected_drafts = integral(counter_drafts, f"{family}: counter interval drafts")
    if sidecar_drafts != expected_drafts:
        raise GateError(
            f"{family}: sidecar/counter draft mismatch over {span}: "
            f"{sidecar_drafts} != {expected_drafts}"
        )
    sidecar_ms = math.fsum(float(value) for value in ms_values)
    counter_ms = 1000.0 * counter_seconds
    # Samples are serialized after rounding each observation to 4 decimal ms.
    rounding_bound_ms = expected_steps * 0.000051 + 1e-6
    error_ms = sidecar_ms - counter_ms
    if abs(error_ms) > rounding_bound_ms:
        raise GateError(
            f"{family}: sidecar/counter timing mismatch over {span}: "
            f"sidecar={sidecar_ms} ms counter={counter_ms} ms "
            f"error={error_ms} ms bound={rounding_bound_ms} ms"
        )
    return {
        "counter_interval": [start, end],
        "steps": expected_steps,
        "counter_drafts": expected_drafts,
        "sidecar_drafts": sidecar_drafts,
        "counter_ms": counter_ms,
        "sidecar_ms": sidecar_ms,
        "timing_error_ms": error_ms,
        "rounding_bound_ms": rounding_bound_ms,
        "exact_drafts_and_steps": True,
        "timing_within_per_sample_rounding_bound": True,
    }


def legacy_slo(rows_per_step: float) -> tuple[float, float]:
    reference = max(
        WEIGHT_STREAM_LOWER_BOUND_MS,
        COMPUTE_MS_PER_ROW * rows_per_step,
    )
    return reference, SLO_MULTIPLIER * reference


def cluster_summary(values: list[float]) -> dict[str, Any]:
    count = len(values)
    df = count - 1
    critical = T95_ONE_SIDED.get(df)
    if critical is None:
        raise GateError(f"no pinned one-sided t critical for df={df}")
    point = statistics.fmean(values)
    sample_sd = statistics.stdev(values)
    standard_error = sample_sd / math.sqrt(count)
    return {
        "cluster_count": count,
        "df": df,
        "point_estimate": point,
        "sample_sd_across_task_means": sample_sd,
        "standard_error": standard_error,
        "t_0_95_one_sided": critical,
        "u95": point + critical * standard_error,
    }


def b1_arm_statistics(
    windows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    expected_tokens: int,
) -> dict[str, Any]:
    assert_nonoverlap([window["fwd_span"] for window in windows], "forward")
    assert_nonoverlap([window["wall_span"] for window in windows], "wall")
    task_rows = []
    coverage_rows = []
    selected_full_graph = []
    for window in windows:
        delta = window["delta"]
        fwd_steps = delta["fwd_steps"]
        wall_steps = delta["wall_steps"]
        verify_ms = 1000.0 * delta["fwd_s"] / fwd_steps
        wall_ms = 1000.0 * delta["wall_s"] / wall_steps
        rows = (delta["wall_drafts"] / wall_steps) * (expected_tokens + 1)
        slo_reference, slo_limit = legacy_slo(rows)
        fwd_coverage = coverage_for_span(len(arrays["fwd_ms"]), window["fwd_span"])
        wall_coverage = coverage_for_span(len(arrays["wall_ms"]), window["wall_span"])
        if (
            fwd_coverage["fraction"] < REQUIRED_COVERAGE
            or wall_coverage["fraction"] < REQUIRED_COVERAGE
        ):
            raise GateError(
                f"{window['task_id']}: per-step sidecar coverage is not 100%"
            )
        reconciliation = {
            "forward": reconcile_counter_interval(
                arrays,
                "fwd",
                window["fwd_span"],
                delta["fwd_s"],
                delta["fwd_drafts"],
            ),
            "wall": reconcile_counter_interval(
                arrays,
                "wall",
                window["wall_span"],
                delta["wall_s"],
                delta["wall_drafts"],
            ),
        }
        pairing = paired_wall_selection(
            arrays,
            fwd_span=window["fwd_span"],
            wall_span=window["wall_span"],
            label=f"{window['task_id']}: wall predecessor binding",
        )
        fwd_sample = select_span(arrays["fwd_ms"], window["fwd_span"])
        full_sample = select_span(arrays["fwd_full"], window["fwd_span"])
        wall_sample = pairing["wall_ms"]
        selected_full_graph.append(full_sample)
        task_rows.append(
            {
                "task_id": window["task_id"],
                "verify_ms_per_step": verify_ms,
                "wall_ms_per_step": wall_ms,
                "rows_per_step": rows,
                "legacy_slo_reference_ms": slo_reference,
                "legacy_slo_limit_ms": slo_limit,
                "legacy_slo_excess_ms": wall_ms - slo_limit,
                "selected_sample_verify_ms_per_step": float(fwd_sample.mean()),
                "selected_sample_wall_ms_per_step": float(wall_sample.mean()),
                "wall_forward_pairing": pairing["report"],
            }
        )
        coverage_rows.append(
            {
                "task_id": window["task_id"],
                "forward": fwd_coverage,
                "wall": wall_coverage,
                "counter_reconciliation": reconciliation,
                "wall_forward_pairing": pairing["report"],
            }
        )
    full_graph_fraction = float(np.concatenate(selected_full_graph).mean())
    if full_graph_fraction < MIN_FULL_GRAPH_FRACTION:
        raise GateError("B=1 selected FULL graph fraction is below 99%")
    equal_task = {
        "wall_ms_per_step": cluster_summary(
            [row["wall_ms_per_step"] for row in task_rows]
        ),
        "verify_ms_per_step": cluster_summary(
            [row["verify_ms_per_step"] for row in task_rows]
        ),
        "rows_per_step": cluster_summary([row["rows_per_step"] for row in task_rows]),
        "legacy_slo_excess_ms": cluster_summary(
            [row["legacy_slo_excess_ms"] for row in task_rows]
        ),
    }
    total_fwd_steps = sum(row["delta"]["fwd_steps"] for row in windows)
    total_wall_steps = sum(row["delta"]["wall_steps"] for row in windows)
    weighted_verify = (
        1000.0 * sum(row["delta"]["fwd_s"] for row in windows) / total_fwd_steps
    )
    weighted_wall = (
        1000.0 * sum(row["delta"]["wall_s"] for row in windows) / total_wall_steps
    )
    weighted_rows = (
        sum(row["delta"]["wall_drafts"] for row in windows)
        / total_wall_steps
        * (expected_tokens + 1)
    )
    weighted_reference, weighted_limit = legacy_slo(weighted_rows)
    return {
        "inference_scope": (
            "equal-weight SWE task clusters; the t interval treats each whole "
            "task as one cluster and makes no within-task independence assumption"
        ),
        "bracket_mode": "nonoverlapping_task_clusters",
        "task_cluster_equal_weight": equal_task,
        "step_weighted_counter_point": {
            "verify_ms_per_step": weighted_verify,
            "wall_ms_per_step": weighted_wall,
            "rows_per_step": weighted_rows,
            "legacy_slo_reference_ms": weighted_reference,
            "legacy_slo_limit_ms": weighted_limit,
            "legacy_slo_excess_ms": weighted_wall - weighted_limit,
        },
        "per_task": task_rows,
        "sidecar_coverage_by_task": coverage_rows,
        "selected_full_graph_fraction": full_graph_fraction,
        "gate": {
            "statistic": "equal_task_legacy_slo_excess_ms_u95_le_0",
            "pass": equal_task["legacy_slo_excess_ms"]["u95"] <= 0,
        },
    }


def outer_counter_point(
    windows: list[dict[str, Any]],
    family: str,
    expected_tokens: int,
    concurrency: int,
) -> dict[str, Any]:
    step_key = f"{family}_steps"
    seconds_key = f"{family}_s"
    drafts_key = f"{family}_drafts"
    span_key = f"{family}_span"
    spans = [window[span_key] for window in windows]
    union = merged_spans(spans)
    if len(union) != 1:
        raise GateError(
            f"B=4 {family} task brackets do not form one contiguous union: {union}"
        )
    start, end = union[0]
    start_window = min(windows, key=lambda row: (row["pre"][step_key], row["task_id"]))
    end_window = max(windows, key=lambda row: (row["post"][step_key], row["task_id"]))
    if (
        integral(start_window["pre"][step_key], f"{family}:union pre") != start
        or integral(end_window["post"][step_key], f"{family}:union post") != end
    ):
        raise GateError(f"{family}: union endpoint snapshot mismatch")
    steps = end - start
    seconds = end_window["post"][seconds_key] - start_window["pre"][seconds_key]
    drafts = end_window["post"][drafts_key] - start_window["pre"][drafts_key]
    if seconds <= 0 or drafts <= 0:
        raise GateError(f"{family}: non-positive outer counter delta")
    if not steps <= drafts <= concurrency * steps:
        raise GateError(
            f"{family}: outer drafts/step is outside [1, {concurrency}]: "
            f"drafts={drafts}, steps={steps}"
        )
    point = {
        "counter_interval": [start, end],
        "steps": steps,
        "seconds": seconds,
        "drafts": drafts,
        "ms_per_step": 1000.0 * seconds / steps,
        "drafts_per_step": drafts / steps,
        "rows_per_step": drafts / steps * (expected_tokens + 1),
    }
    return point


def moving_block_means(
    arrays: tuple[np.ndarray, ...],
    reps: int,
    block: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    lengths = {len(values) for values in arrays}
    if len(lengths) != 1:
        raise GateError("moving-block arrays have different lengths")
    sample_count = lengths.pop()
    if sample_count < block:
        raise GateError(
            f"need at least {block} samples for requested block sensitivity; "
            f"got {sample_count}"
        )
    blocks_per_rep = math.ceil(sample_count / block)
    offsets = np.arange(block, dtype=np.int64)
    rng = np.random.default_rng(seed)
    outputs = tuple(np.empty(reps, dtype=np.float64) for _ in arrays)
    for lower in range(0, reps, 32):
        upper = min(reps, lower + 32)
        starts = rng.integers(
            0,
            sample_count - block + 1,
            size=(upper - lower, blocks_per_rep),
        )
        indices = (starts[:, :, None] + offsets).reshape(upper - lower, -1)
        indices = indices[:, :sample_count]
        for values, output in zip(arrays, outputs, strict=True):
            output[lower:upper] = values[indices].mean(axis=1)
    return outputs


def b4_arm_statistics(
    windows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    expected_tokens: int,
    concurrency: int,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    fwd_point = outer_counter_point(windows, "fwd", expected_tokens, concurrency)
    wall_point = outer_counter_point(windows, "wall", expected_tokens, concurrency)
    task_pairings = [
        {
            "task_id": window["task_id"],
            **paired_wall_selection(
                arrays,
                fwd_span=window["fwd_span"],
                wall_span=window["wall_span"],
                label=f"{window['task_id']}: B=4 wall predecessor binding",
            )["report"],
        }
        for window in windows
    ]
    fwd_span = tuple(fwd_point["counter_interval"])
    wall_span = tuple(wall_point["counter_interval"])
    fwd_coverage = coverage_for_span(len(arrays["fwd_ms"]), fwd_span)
    wall_coverage = coverage_for_span(len(arrays["wall_ms"]), wall_span)
    if (
        fwd_coverage["fraction"] < REQUIRED_COVERAGE
        or wall_coverage["fraction"] < REQUIRED_COVERAGE
    ):
        raise GateError("B=4 union sidecar coverage is not 100%")
    reconciliation = {
        "forward": reconcile_counter_interval(
            arrays,
            "fwd",
            fwd_span,
            fwd_point["seconds"],
            fwd_point["drafts"],
        ),
        "wall": reconcile_counter_interval(
            arrays,
            "wall",
            wall_span,
            wall_point["seconds"],
            wall_point["drafts"],
        ),
    }
    all_fwd_full = select_span(arrays["fwd_full"], fwd_span)
    pairing = paired_wall_selection(
        arrays,
        fwd_span=fwd_span,
        wall_span=wall_span,
        label="B=4 union wall predecessor binding",
    )
    fwd_ms = pairing["fwd_ms"]
    fwd_full = pairing["fwd_full"]
    wall_ms = pairing["wall_ms"]
    wall_drafts = pairing["wall_drafts"]
    wall_rows = wall_drafts * (expected_tokens + 1)
    full_graph_fraction = float(fwd_full.mean())
    all_forward_full_graph_fraction = float(all_fwd_full.mean())
    if (
        full_graph_fraction < MIN_FULL_GRAPH_FRACTION
        or all_forward_full_graph_fraction < MIN_FULL_GRAPH_FRACTION
    ):
        raise GateError("B=4 selected FULL graph fraction is below 99%")
    paired_fwd_point = float(fwd_ms.mean())
    slo_reference, slo_limit = legacy_slo(wall_point["rows_per_step"])
    max_possible_compute_reference = COMPUTE_MS_PER_ROW * 4 * (expected_tokens + 1)
    if max_possible_compute_reference >= WEIGHT_STREAM_LOWER_BOUND_MS:
        raise GateError("B=4 campaign violates the pinned weight-bound dominance")
    point_excess = wall_point["ms_per_step"] - slo_limit
    sensitivity = []
    for block in BLOCK_SENSITIVITY:
        fwd_boot, wall_boot, rows_boot = moving_block_means(
            (fwd_ms, wall_ms, wall_rows), reps, block, seed + 2000 + block
        )
        bootstrap_reference = np.maximum(
            WEIGHT_STREAM_LOWER_BOUND_MS,
            COMPUTE_MS_PER_ROW * rows_boot,
        )
        bootstrap_excess = wall_boot - SLO_MULTIPLIER * bootstrap_reference
        sample_excess = float(wall_ms.mean()) - SLO_MULTIPLIER * max(
            WEIGHT_STREAM_LOWER_BOUND_MS,
            COMPUTE_MS_PER_ROW * float(wall_rows.mean()),
        )
        sensitivity.append(
            {
                "block_steps": block,
                "verify_ms_per_step_u95": (
                    paired_fwd_point
                    + float(np.quantile(fwd_boot - fwd_ms.mean(), 0.95))
                ),
                "wall_ms_per_step_u95": (
                    wall_point["ms_per_step"]
                    + float(np.quantile(wall_boot - wall_ms.mean(), 0.95))
                ),
                "legacy_slo_excess_ms_u95": (
                    point_excess
                    + float(np.quantile(bootstrap_excess - sample_excess, 0.95))
                ),
            }
        )
    worst = {
        "verify_ms_per_step_u95": max(
            row["verify_ms_per_step_u95"] for row in sensitivity
        ),
        "wall_ms_per_step_u95": max(row["wall_ms_per_step_u95"] for row in sensitivity),
        "legacy_slo_excess_ms_u95": max(
            row["legacy_slo_excess_ms_u95"] for row in sensitivity
        ),
    }
    exact_b4_selected = wall_drafts == concurrency
    exact_b4_count = int(np.count_nonzero(exact_b4_selected))
    if exact_b4_count < MIN_B4_EXACT_EVENTS:
        raise GateError(
            "B=4 exact-occupancy wall stratum has insufficient evidence: "
            f"{exact_b4_count} < {MIN_B4_EXACT_EVENTS}"
        )
    exact_b4_wall_ms = wall_ms[exact_b4_selected]
    exact_b4_fwd_ms = fwd_ms[exact_b4_selected]
    exact_b4_rows = concurrency * (expected_tokens + 1)
    exact_b4_reference, exact_b4_limit = legacy_slo(exact_b4_rows)
    exact_b4_wall_point = float(exact_b4_wall_ms.mean())
    exact_b4_fwd_point = float(exact_b4_fwd_ms.mean())
    exact_b4_sensitivity = []
    for block in BLOCK_SENSITIVITY:
        exact_fwd_boot, exact_wall_boot = moving_block_means(
            (exact_b4_fwd_ms, exact_b4_wall_ms),
            reps,
            block,
            seed + 3000 + block,
        )
        exact_b4_sensitivity.append(
            {
                "block_steps": block,
                "verify_ms_per_step_u95": (
                    exact_b4_fwd_point
                    + float(
                        np.quantile(
                            exact_fwd_boot - exact_b4_fwd_point,
                            0.95,
                        )
                    )
                ),
                "wall_ms_per_step_u95": (
                    exact_b4_wall_point
                    + float(
                        np.quantile(
                            exact_wall_boot - exact_b4_wall_point,
                            0.95,
                        )
                    )
                ),
            }
        )
    exact_b4_worst_wall_u95 = max(
        row["wall_ms_per_step_u95"] for row in exact_b4_sensitivity
    )
    exact_b4 = {
        "wall_drafts": concurrency,
        "selected_steps": exact_b4_count,
        "rows_per_step": exact_b4_rows,
        "verify_ms_per_step": exact_b4_fwd_point,
        "wall_ms_per_step": exact_b4_wall_point,
        "legacy_slo_reference_ms": exact_b4_reference,
        "legacy_slo_limit_ms": exact_b4_limit,
        "moving_block_u95_sensitivity": {
            "reps": reps,
            "blocks": exact_b4_sensitivity,
            "worst_wall_ms_per_step_u95": exact_b4_worst_wall_u95,
        },
        "gate": {
            "statistic": "worst_block_exact_b4_wall_ms_per_step_u95_le_slo",
            "pass": exact_b4_worst_wall_u95 <= exact_b4_limit,
        },
    }
    occupancy = []
    for drafts in range(1, 5):
        selected = wall_drafts == drafts
        if not np.any(selected):
            continue
        rows = drafts * (expected_tokens + 1)
        reference, limit = legacy_slo(rows)
        occupancy.append(
            {
                "wall_drafts": drafts,
                "selected_steps": int(np.count_nonzero(selected)),
                "wall_ms_per_step": float(wall_ms[selected].mean()),
                "rows_per_step": rows,
                "legacy_slo_reference_ms": reference,
                "legacy_slo_limit_ms": limit,
                "legacy_slo_excess_ms": (float(wall_ms[selected].mean()) - limit),
            }
        )
    return {
        "inference_scope": (
            "conditional on this one campaign time series; moving-block "
            "sensitivity is explicitly NOT a task-general uncertainty claim"
        ),
        "bracket_mode": (
            "overlap-safe_single_counter_index_union; task-sum is forbidden"
        ),
        "union_counter_point": {
            "verify_ms_per_step": paired_fwd_point,
            "all_forward_verify_ms_per_step": fwd_point["ms_per_step"],
            "wall_ms_per_step": wall_point["ms_per_step"],
            "rows_per_step": wall_point["rows_per_step"],
            "legacy_slo_reference_ms": slo_reference,
            "legacy_slo_limit_ms": slo_limit,
            "legacy_slo_excess_ms": point_excess,
        },
        "union_intervals": {
            "forward": fwd_point["counter_interval"],
            "wall": wall_point["counter_interval"],
        },
        "sidecar_coverage": {
            "forward": fwd_coverage,
            "wall": wall_coverage,
        },
        "sidecar_counter_reconciliation": reconciliation,
        "wall_forward_pairing": pairing["report"],
        "wall_forward_pairing_by_task": task_pairings,
        "selected_full_graph_fraction": full_graph_fraction,
        "all_forward_full_graph_fraction": all_forward_full_graph_fraction,
        "forward_wall_occupancy_sequence_equal": True,
        "wall_occupancy_strata": occupancy,
        "exact_b4_stratum": exact_b4,
        "moving_block_u95_sensitivity": {
            "reps": reps,
            "wall_centered_on_complete_union_counter_point": True,
            "verify_centered_on_paired_forward_sample": True,
            "blocks": sensitivity,
            "worst_across_requested_blocks": worst,
        },
        "gate": {
            "statistic": (
                "union_worst_block_legacy_slo_excess_ms_u95_le_0_and_"
                "exact_b4_worst_block_wall_u95_le_slo"
            ),
            "union_pass": worst["legacy_slo_excess_ms_u95"] <= 0,
            "exact_b4_pass": exact_b4["gate"]["pass"],
            "pass": (
                worst["legacy_slo_excess_ms_u95"] <= 0
                and exact_b4["gate"]["pass"]
            ),
        },
    }


def reduce_arm(
    repo: Path,
    runroot: Path,
    sidecar_dir: Path,
    arm: str,
    *,
    mode: str,
    task_count: int,
    expected_concurrency: int | None,
    reps: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    arm_dir = runroot / arm
    try:
        mode_spec = FIXED32_MODE_SPECS[mode]
    except KeyError as error:
        raise GateError(f"{arm}: unsupported fixed-32 mode {mode!r}") from error
    expected_tokens = PHYSICAL_DRAFTS
    expected_kind = mode
    orchestrator = parse_orchestrator(arm_dir, task_count)
    concurrency = orchestrator["inferred_concurrency"]
    if expected_concurrency is not None and concurrency != expected_concurrency:
        raise GateError(
            f"{arm}: inferred concurrency {concurrency} != "
            f"expected {expected_concurrency}"
        )
    task_dirs = task_directories(arm_dir, task_count)
    launch = resolve_subset_from_runlog(
        repo,
        runroot,
        arm,
        expected_kind,
        expected_tokens,
        task_count,
        concurrency,
    )
    runtime = validate_runtime_needles(
        arm_dir,
        mode=mode,
        expected_tokens=expected_tokens,
        task_ids=launch["subset"]["task_ids"],
    )
    windows = load_windows(arm_dir, task_dirs, expected_tokens, concurrency)
    _, arrays, sidecar = unique_sidecar(
        sidecar_dir,
        arm,
        concurrency,
        launch["engine_core_pid"],
    )
    main_sidecar = unique_main_sidecar(
        sidecar_dir,
        arm,
        launch["engine_core_pid"],
        arrays,
    )
    flush_chain = validate_flush_chain(
        arm_dir,
        task_dirs,
        windows,
        mode=mode,
        producer_pid=launch["engine_core_pid"],
        complete_steps=len(arrays["fwd_drafts"]),
        server_capacity=concurrency,
        dataset_record_digests=pinned_dataset_record_digests(str(repo)),
    )
    real_task_provenance = validate_real_task_provenance(
        arm_dir,
        task_dirs,
        mode=mode,
        subset=launch["subset"],
        windows=windows,
        flush_chain=flush_chain,
        dataset_record_digests=pinned_dataset_record_digests(str(repo)),
    )
    census_intervals, census_indices = selected_counter_indices(
        [tuple(window["fwd_span"]) for window in windows],
        available_steps=len(arrays["fwd_drafts"]),
        label=f"{arm}: work census",
    )
    complete_census_indices = list(range(len(arrays["fwd_drafts"])))
    complete_census_batch_sequence = [
        integral(value, f"{arm}: complete fwd_drafts") for value in arrays["fwd_drafts"]
    ]
    selected_census_batch_sequence = [
        integral(arrays["fwd_drafts"][index], f"{arm}: selected fwd_drafts")
        for index in census_indices
    ]
    complete_occupied_batch_histogram = {
        str(batch_size): complete_census_batch_sequence.count(batch_size)
        for batch_size in range(1, concurrency + 1)
        if batch_size in complete_census_batch_sequence
    }
    selected_occupied_batch_histogram = {
        str(batch_size): selected_census_batch_sequence.count(batch_size)
        for batch_size in range(1, concurrency + 1)
        if batch_size in selected_census_batch_sequence
    }
    work_census_expected = {
        "path": str(arm_dir / "logs" / "fr13_fixed32_work_census.jsonl"),
        "producer_pid": launch["engine_core_pid"],
        "binding": (
            "complete_pure_decode_sfwd_stream_then_posthoc_"
            "canonical_task_forward_counter_union"
        ),
        "complete_stream": {
            "forward_step_indices": complete_census_indices,
            "event_count": len(complete_census_batch_sequence),
            "batch_size_sequence": complete_census_batch_sequence,
            "occupied_batch_histogram": complete_occupied_batch_histogram,
        },
        "canonical_task_selection": {
            "counter_intervals": [list(span) for span in census_intervals],
            "forward_step_indices": census_indices,
            "event_count": len(selected_census_batch_sequence),
            "batch_size_sequence": selected_census_batch_sequence,
            "occupied_batch_histogram": selected_occupied_batch_histogram,
        },
    }
    endpoint_counters = {
        "fwd_steps": max(
            integral(window["post"]["fwd_steps"], f"{arm}:post fwd steps")
            for window in windows
        ),
        "wall_steps": max(
            integral(window["post"]["wall_steps"], f"{arm}:post wall steps")
            for window in windows
        ),
        "wall_rejected": max(
            integral(window["post"]["wall_rejected"], f"{arm}:post wall rejected")
            for window in windows
        ),
        "wall_attempts": max(
            integral(window["post"]["wall_attempts"], f"{arm}:post wall attempts")
            for window in windows
        ),
    }
    main_endpoints = {
        "fwd_steps": main_sidecar["counters"]["fwd_steps"],
        "wall_steps": main_sidecar["counters"]["wall_steps"],
        "wall_rejected": main_sidecar["counters"]["wall_rejected"],
        "wall_attempts": main_sidecar["wall_attempts"],
    }
    if endpoint_counters != main_endpoints:
        raise GateError(
            f"{arm}: final task bracket counters do not match the explicitly "
            f"flushed main sidecar: {endpoint_counters} != {main_endpoints}"
        )
    main_sidecar["final_task_bracket_endpoints"] = endpoint_counters
    statistics_out = (
        b1_arm_statistics(windows, arrays, expected_tokens)
        if concurrency == 1
        else b4_arm_statistics(
            windows,
            arrays,
            expected_tokens,
            concurrency,
            reps,
            seed,
        )
    )
    task_points = {
        window["task_id"]: {
            "wall_ms": 1000.0
            * window["delta"]["wall_s"]
            / window["delta"]["wall_steps"],
            "verify_ms": 1000.0
            * window["delta"]["fwd_s"]
            / window["delta"]["fwd_steps"],
            "rows": (
                window["delta"]["wall_drafts"]
                / window["delta"]["wall_steps"]
                * (expected_tokens + 1)
            ),
        }
        for window in windows
    }
    return (
        {
            "arm": expected_kind,
            "artifact_dir": str(arm_dir),
            "inferred_concurrency": concurrency,
            "expected_draft_tokens_per_event": expected_tokens,
            "active_logical_drafts_per_event": mode_spec["active_drafts"],
            "valid_mask": f"{mode_spec['valid_mask']:#010x}",
            "canonical_task_ids": [window["task_id"] for window in windows],
            "provenance": {
                "orchestrator": orchestrator,
                "launch": launch,
                "runtime": runtime,
                "real_tasks": real_task_provenance,
                "metric_labels": windows[0]["metric_labels"],
                "task_metric_brackets": {
                    window["task_id"]: window["metric_artifacts"]
                    for window in windows
                },
                "metric_hashes_derived_from_parsed_bytes": True,
                "all_required_provenance_valid": True,
            },
            "sidecar": {
                "per_step": sidecar,
                "main": main_sidecar,
            },
            "flush_chain": flush_chain,
            "work_census_expected": work_census_expected,
            "statistics": statistics_out,
        },
        task_points,
    )


def b1_comparison(
    tail_points: dict[str, dict[str, float]],
    hydra_points: dict[str, dict[str, float]],
) -> dict[str, Any]:
    if list(tail_points) != list(hydra_points):
        raise GateError("Tail6-fixed32/Hydra27-fixed32 task order differs")
    wall_deltas = []
    verify_deltas = []
    excess_deltas = []
    per_task = []
    for task_id in tail_points:
        tail = tail_points[task_id]
        hydra = hydra_points[task_id]
        tail_limit = legacy_slo(tail["rows"])[1]
        hydra_limit = legacy_slo(hydra["rows"])[1]
        wall_delta = hydra["wall_ms"] - tail["wall_ms"]
        verify_delta = hydra["verify_ms"] - tail["verify_ms"]
        excess_delta = (hydra["wall_ms"] - hydra_limit) - (tail["wall_ms"] - tail_limit)
        wall_deltas.append(wall_delta)
        verify_deltas.append(verify_delta)
        excess_deltas.append(excess_delta)
        per_task.append(
            {
                "task_id": task_id,
                "hydra_minus_tail_wall_ms_per_step": wall_delta,
                "hydra_minus_tail_verify_ms_per_step": verify_delta,
                "hydra_minus_tail_legacy_slo_excess_ms": excess_delta,
            }
        )
    return {
        "scope": (
            "paired equal-task cluster diagnostic; Hydra27-fixed32 minus Tail6-fixed32"
        ),
        "wall_ms_per_step_delta": cluster_summary(wall_deltas),
        "verify_ms_per_step_delta": cluster_summary(verify_deltas),
        "legacy_slo_excess_ms_delta": cluster_summary(excess_deltas),
        "per_task": per_task,
    }


def validate_work_census_v5_report(
    report: dict[str, Any],
    *,
    required_batch: int,
) -> dict[str, Any]:
    expected_modes = ("tail6_fixed32", "hydra27_fixed32")
    expected_batches = tuple(SUPPORTED_BATCH_SIZES)
    expected_batch_keys = {str(batch) for batch in expected_batches}
    exact_keys(
        report,
        {
            "schema",
            "status",
            "required_batch_sizes",
            "event_counts",
            "batch_size_sequences",
            "forward_step_indices",
            "event_ids",
            "producer_pids",
            "terminal_summaries",
            "drafter_graph_registries",
            "forward_graph_registries",
            "conv_pregather_auxiliary",
            "physical_work_histograms",
            "scope",
            "semantic_modes",
            "normalized_work_signature",
            "normalized_work_signature_sha256",
        },
        "fixed32 work-census v5 report",
    )
    if (
        report["schema"] != WORK_CENSUS_REPORT_SCHEMA
        or report["status"] != "PASS"
        or report["required_batch_sizes"] != [required_batch]
        or report["scope"] != FIXED_WORK_SCOPE
        or report["semantic_modes"] != WORK_CENSUS_MODE_SEMANTICS
        or not isinstance(report["normalized_work_signature"], dict)
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(report["normalized_work_signature_sha256"]),
        )
        is None
        or canonical_json_sha256(report["normalized_work_signature"])
        != report["normalized_work_signature_sha256"]
    ):
        raise GateError("fixed32 work-census v5 report contract mismatch")

    histograms = report["physical_work_histograms"]
    event_counts = report["event_counts"]
    if not isinstance(histograms, dict) or not isinstance(event_counts, dict):
        raise GateError("fixed32 physical-work histograms are malformed")
    exact_keys(histograms, set(expected_modes), "physical_work_histograms")
    exact_keys(event_counts, set(expected_modes), "event_counts")
    observed_by_mode: dict[str, set[int]] = {}
    signatures_by_mode: dict[str, dict[int, str]] = {}
    for mode in expected_modes:
        mode_histogram = histograms[mode]
        mode_event_counts = event_counts[mode]
        if not isinstance(mode_histogram, dict) or not isinstance(
            mode_event_counts, dict
        ):
            raise GateError(f"{mode}: physical-work histogram is malformed")
        exact_keys(
            mode_histogram,
            expected_batch_keys,
            f"physical_work_histograms.{mode}",
        )
        observed_by_mode[mode] = set()
        signatures_by_mode[mode] = {}
        for batch in expected_batches:
            batch_key = str(batch)
            entry = mode_histogram[batch_key]
            if not isinstance(entry, dict):
                raise GateError(f"{mode}: B{batch} histogram entry is malformed")
            exact_keys(
                entry,
                {"event_count", "normalized_event_signatures"},
                f"physical_work_histograms.{mode}.{batch_key}",
            )
            event_count = entry["event_count"]
            signatures = entry["normalized_event_signatures"]
            expected_count = mode_event_counts.get(batch_key, 0)
            if (
                isinstance(event_count, bool)
                or not isinstance(event_count, int)
                or event_count < 0
                or event_count != expected_count
                or not isinstance(signatures, dict)
            ):
                raise GateError(
                    f"{mode}: B{batch} physical-work event count is inconsistent"
                )
            for signature, count in signatures.items():
                if (
                    re.fullmatch(r"[0-9a-f]{64}", str(signature)) is None
                    or isinstance(count, bool)
                    or not isinstance(count, int)
                    or count <= 0
                ):
                    raise GateError(
                        f"{mode}: B{batch} physical-work signature is malformed"
                    )
            if sum(signatures.values()) != event_count:
                raise GateError(
                    f"{mode}: B{batch} signature counts do not reconcile"
                )
            if event_count == 0:
                if signatures:
                    raise GateError(
                        f"{mode}: B{batch} unobserved histogram is nonempty"
                    )
                continue
            if len(signatures) != 1:
                raise GateError(
                    f"{mode}: B{batch} does not have one physical-work signature"
                )
            observed_by_mode[mode].add(batch)
            signatures_by_mode[mode][batch] = next(iter(signatures))
        expected_count_keys = {
            str(batch) for batch in observed_by_mode[mode]
        }
        if set(mode_event_counts) != expected_count_keys:
            raise GateError(f"{mode}: event-count histogram keys are inconsistent")

    tail_mode, hydra_mode = expected_modes
    if observed_by_mode[tail_mode] != observed_by_mode[hydra_mode]:
        raise GateError(
            "Tail/Hydra occupied batch sets differ, so per-B physical work "
            "cannot be compared"
        )
    observed_batches = sorted(observed_by_mode[tail_mode])
    if required_batch not in observed_batches:
        raise GateError(f"fixed32 work census lacks required B{required_batch}")
    physical_per_batch: dict[str, dict[str, Any]] = {}
    for batch in observed_batches:
        tail_signature = signatures_by_mode[tail_mode][batch]
        hydra_signature = signatures_by_mode[hydra_mode][batch]
        if tail_signature != hydra_signature:
            raise GateError(
                f"B{batch}: Tail/Hydra normalized physical-work SHA differs"
            )
        physical_per_batch[str(batch)] = {
            "normalized_event_signature_sha256": tail_signature,
            "tail_event_count": histograms[tail_mode][str(batch)]["event_count"],
            "hydra_event_count": histograms[hydra_mode][str(batch)]["event_count"],
        }

    registries = report["drafter_graph_registries"]
    terminals = report["terminal_summaries"]
    if not isinstance(registries, dict) or not isinstance(terminals, dict):
        raise GateError("fixed32 drafter graph registries are malformed")
    exact_keys(registries, set(expected_modes), "drafter_graph_registries")
    exact_keys(terminals, set(expected_modes), "terminal_summaries")
    registry_by_mode: dict[str, dict[int, dict[str, Any]]] = {}
    registry_keys = {
        "batch_size",
        "graph_signature",
        "captures",
        "capture_origin",
        "measured_replays",
        "unmeasured_replays",
    }
    for mode in expected_modes:
        rows = registries[mode]
        terminal = terminals[mode]
        if (
            not isinstance(rows, list)
            or not rows
            or not isinstance(terminal, dict)
            or terminal.get("drafter_graph_registry") != rows
            or terminal.get("scope") != FIXED_WORK_SCOPE
        ):
            raise GateError(f"{mode}: terminal drafter registry/scope mismatch")
        registry_by_mode[mode] = {}
        ordered_batches = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise GateError(f"{mode}: drafter registry row {index} is malformed")
            exact_keys(
                row,
                registry_keys,
                f"drafter_graph_registries.{mode}[{index}]",
            )
            batch = row["batch_size"]
            if (
                isinstance(batch, bool)
                or not isinstance(batch, int)
                or batch not in expected_batches
                or batch in registry_by_mode[mode]
                or not isinstance(row["graph_signature"], str)
                or re.fullmatch(r"[0-9a-f]{64}", row["graph_signature"])
                is None
                or row["captures"] != 1
                or isinstance(row["captures"], bool)
                or not isinstance(row["capture_origin"], str)
                or row["capture_origin"] not in {"measured", "unmeasured"}
                or any(
                    isinstance(row[key], bool)
                    or not isinstance(row[key], int)
                    or row[key] < 0
                    for key in ("measured_replays", "unmeasured_replays")
                )
                or row["measured_replays"]
                != histograms[mode][str(batch)]["event_count"]
            ):
                raise GateError(f"{mode}: drafter registry row {index} is invalid")
            ordered_batches.append(batch)
            registry_by_mode[mode][batch] = row
        if ordered_batches != sorted(ordered_batches):
            raise GateError(f"{mode}: drafter registry rows are not sorted")
        if not observed_by_mode[mode].issubset(registry_by_mode[mode]):
            raise GateError(
                f"{mode}: drafter registry does not cover occupied batches"
            )

    if set(registry_by_mode[tail_mode]) != set(registry_by_mode[hydra_mode]):
        raise GateError("Tail/Hydra drafter graph registry batch sets differ")
    lifecycle_per_batch: dict[str, dict[str, Any]] = {}
    for batch in sorted(registry_by_mode[tail_mode]):
        tail_row = registry_by_mode[tail_mode][batch]
        hydra_row = registry_by_mode[hydra_mode][batch]
        if (
            tail_row["graph_signature"] != hydra_row["graph_signature"]
            or tail_row["capture_origin"] != hydra_row["capture_origin"]
        ):
            raise GateError(
                f"B{batch}: Tail/Hydra drafter graph lifecycle differs"
            )
        lifecycle_per_batch[str(batch)] = {
            "graph_signature": tail_row["graph_signature"],
            "captures_per_arm": 1,
            "capture_origin": tail_row["capture_origin"],
            "tail_measured_replays": tail_row["measured_replays"],
            "hydra_measured_replays": hydra_row["measured_replays"],
            "tail_unmeasured_replays": tail_row["unmeasured_replays"],
            "hydra_unmeasured_replays": hydra_row["unmeasured_replays"],
        }

    forward_registries = report["forward_graph_registries"]
    auxiliary_by_mode = report["conv_pregather_auxiliary"]
    if not isinstance(forward_registries, dict) or not isinstance(
        auxiliary_by_mode, dict
    ):
        raise GateError("fixed32 forward graph pregather proof is malformed")
    exact_keys(
        forward_registries,
        set(expected_modes),
        "forward_graph_registries",
    )
    exact_keys(
        auxiliary_by_mode,
        set(expected_modes),
        "conv_pregather_auxiliary",
    )
    forward_registry_keys = {
        "batch_size",
        "graph_signature",
        "conv_layout_sha256",
        "captures",
        "capture_origin",
        "stage_calls",
        "stage_before_all_consumes",
        "layers",
        "requests",
        "row_elems",
        "programs",
        "ssi_pointer_entries",
        "ssi_groups",
        "source_validations",
        "staged_rows",
        "consume_calls",
        "consume_hits",
        "consume_fallbacks",
        "freshness_matches",
        "measured_replays",
    }
    auxiliary_keys = {
        "profile_capture_stages",
        "aux_capture_stages",
        "host_actual_stages",
        "host_actual_stages_by_batch",
    }
    expected_zero_by_batch = {
        str(batch): 0 for batch in expected_batches
    }
    forward_by_mode: dict[str, dict[int, dict[str, Any]]] = {}
    nonpure_dispatch_by_mode: dict[str, dict[str, int]] = {}
    nonpure_committer_replays_by_mode: dict[str, dict[str, int]] = {}
    nonpure_dispatch_keys = {
        "guarded_steps",
        "piecewise_steps",
        "none_steps",
        "forbidden_full_steps",
    }
    for mode in expected_modes:
        rows = forward_registries[mode]
        terminal = terminals[mode]
        auxiliary = auxiliary_by_mode[mode]
        if (
            not isinstance(rows, list)
            or not rows
            or not isinstance(terminal, dict)
            or terminal.get("forward_graph_registry") != rows
            or terminal.get("conv_pregather_auxiliary") != auxiliary
        ):
            raise GateError(
                f"{mode}: terminal forward graph pregather proof mismatch"
            )
        nonpure_dispatch = terminal.get("nonpure_dispatch")
        if not isinstance(nonpure_dispatch, dict):
            raise GateError(
                f"{mode}: terminal nonpure dispatch proof is malformed"
            )
        exact_keys(
            nonpure_dispatch,
            nonpure_dispatch_keys,
            f"terminal_summaries.{mode}.nonpure_dispatch",
        )
        if (
            any(
                type(nonpure_dispatch[key]) is not int
                or nonpure_dispatch[key] < 0
                for key in nonpure_dispatch_keys
            )
            or nonpure_dispatch["guarded_steps"]
            != (
                nonpure_dispatch["piecewise_steps"]
                + nonpure_dispatch["none_steps"]
                + nonpure_dispatch["forbidden_full_steps"]
            )
            or nonpure_dispatch["forbidden_full_steps"] != 0
        ):
            raise GateError(
                f"{mode}: terminal nonpure dispatch counts do not reconcile"
            )
        nonpure_dispatch_by_mode[mode] = dict(nonpure_dispatch)
        nonpure_committer = terminal.get(
            "nonpure_committer_replays_by_batch"
        )
        if not isinstance(nonpure_committer, dict):
            raise GateError(
                f"{mode}: terminal nonpure committer proof is malformed"
            )
        exact_keys(
            nonpure_committer,
            expected_batch_keys,
            (
                f"terminal_summaries.{mode}."
                "nonpure_committer_replays_by_batch"
            ),
        )
        if (
            any(
                type(nonpure_committer[key]) is not int
                or nonpure_committer[key] < 0
                for key in expected_batch_keys
            )
            or sum(nonpure_committer.values())
            > nonpure_dispatch["guarded_steps"]
        ):
            raise GateError(
                f"{mode}: terminal nonpure committer counts are invalid"
            )
        nonpure_committer_replays_by_mode[mode] = dict(nonpure_committer)
        if not isinstance(auxiliary, dict):
            raise GateError(f"{mode}: pregather auxiliary proof is malformed")
        exact_keys(
            auxiliary,
            auxiliary_keys,
            f"conv_pregather_auxiliary.{mode}",
        )
        if (
            any(
                type(auxiliary[key]) is not int or auxiliary[key] != 0
                for key in (
                    "profile_capture_stages",
                    "aux_capture_stages",
                    "host_actual_stages",
                )
            )
            or auxiliary["host_actual_stages_by_batch"]
            != expected_zero_by_batch
            or any(
                type(value) is not int
                for value in auxiliary["host_actual_stages_by_batch"].values()
            )
        ):
            raise GateError(
                f"{mode}: pregather auxiliary/host stage counts are not zero"
            )
        forward_by_mode[mode] = {}
        ordered_batches: list[int] = []
        signatures: set[str] = set()
        layout_signatures: set[str] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise GateError(
                    f"{mode}: forward graph registry row {index} is malformed"
                )
            exact_keys(
                row,
                forward_registry_keys,
                f"forward_graph_registries.{mode}[{index}]",
            )
            batch = row["batch_size"]
            signature = row["graph_signature"]
            layout_signature = row["conv_layout_sha256"]
            if (
                type(batch) is not int
                or batch not in expected_batches
                or batch in forward_by_mode[mode]
                or not isinstance(signature, str)
                or re.fullmatch(r"[0-9a-f]{64}", signature) is None
                or signature != forward_graph_structural_signature(batch)
                or signature in signatures
                or not isinstance(layout_signature, str)
                or re.fullmatch(r"[0-9a-f]{64}", layout_signature) is None
                or layout_signature in layout_signatures
            ):
                raise GateError(
                    f"{mode}: forward graph registry row {index} identity is invalid"
                )
            expected_programs = (
                CONV_PREGATHER_LAYERS
                * batch
                * (
                    (
                        CONV_PREGATHER_ROW_ELEMS
                        + CONV_PREGATHER_BLOCK
                        - 1
                    )
                    // CONV_PREGATHER_BLOCK
                )
            )
            expected_row = {
                "batch_size": batch,
                "captures": 1,
                "capture_origin": "final_full",
                "stage_calls": 1,
                "stage_before_all_consumes": True,
                "layers": CONV_PREGATHER_LAYERS,
                "requests": batch,
                "row_elems": CONV_PREGATHER_ROW_ELEMS,
                "programs": expected_programs,
                "ssi_pointer_entries": CONV_PREGATHER_LAYERS,
                "ssi_groups": 3,
                "source_validations": CONV_PREGATHER_LAYERS,
                "staged_rows": CONV_PREGATHER_LAYERS * batch,
                "consume_calls": CONV_PREGATHER_LAYERS,
                "consume_hits": CONV_PREGATHER_LAYERS,
                "consume_fallbacks": 0,
                "freshness_matches": CONV_PREGATHER_LAYERS,
                "measured_replays": histograms[mode][str(batch)][
                    "event_count"
                ],
            }
            if any(
                row[key] != expected
                or (
                    isinstance(expected, int)
                    and not isinstance(expected, bool)
                    and type(row[key]) is not int
                )
                or (
                    isinstance(expected, bool)
                    and type(row[key]) is not bool
                )
                for key, expected in expected_row.items()
            ):
                raise GateError(
                    f"{mode}: forward graph registry row {index} "
                    "does not prove one ordered final-FULL pregather capture"
                )
            ordered_batches.append(batch)
            signatures.add(signature)
            layout_signatures.add(layout_signature)
            forward_by_mode[mode][batch] = row
        expected_registry_batches = list(range(1, required_batch + 1))
        if ordered_batches != expected_registry_batches:
            raise GateError(
                f"{mode}: forward graph registry must be exact B1.."
                f"B{required_batch}"
            )
        if not observed_by_mode[mode].issubset(forward_by_mode[mode]):
            raise GateError(
                f"{mode}: forward graph registry does not cover occupied batches"
            )

    forward_per_batch: dict[str, dict[str, Any]] = {}
    for batch in range(1, required_batch + 1):
        tail_row = forward_by_mode[tail_mode][batch]
        hydra_row = forward_by_mode[hydra_mode][batch]
        if (
            tail_row["graph_signature"] != hydra_row["graph_signature"]
            or tail_row["conv_layout_sha256"]
            != hydra_row["conv_layout_sha256"]
        ):
            raise GateError(
                f"B{batch}: Tail/Hydra final-FULL forward graph/layout "
                "signatures differ"
            )
        forward_per_batch[str(batch)] = {
            "graph_signature": tail_row["graph_signature"],
            "conv_layout_sha256": tail_row["conv_layout_sha256"],
            "captures_per_arm": 1,
            "capture_origin": "final_full",
            "stage_calls_per_capture": 1,
            "stage_before_all_consumes": True,
            "layers": CONV_PREGATHER_LAYERS,
            "requests": batch,
            "row_elems": CONV_PREGATHER_ROW_ELEMS,
            "programs": tail_row["programs"],
            "ssi_pointer_entries": CONV_PREGATHER_LAYERS,
            "ssi_groups": 3,
            "source_validations": CONV_PREGATHER_LAYERS,
            "staged_rows": CONV_PREGATHER_LAYERS * batch,
            "consume_calls": CONV_PREGATHER_LAYERS,
            "consume_hits": CONV_PREGATHER_LAYERS,
            "consume_fallbacks": 0,
            "freshness_matches": CONV_PREGATHER_LAYERS,
            "tail_measured_replays": tail_row["measured_replays"],
            "hydra_measured_replays": hydra_row["measured_replays"],
        }

    return {
        "physical_work_comparison": {
            "observed_batch_sizes": observed_batches,
            "per_batch": physical_per_batch,
            "event_counts_compared": False,
            "one_normalized_signature_per_occupied_batch": True,
            "signature_keys_equal_across_arms": True,
        },
        "drafter_graph_lifecycle": {
            "registry_batch_sizes": sorted(registry_by_mode[tail_mode]),
            "per_batch": lifecycle_per_batch,
            "graph_signature_and_capture_origin_equal_across_arms": True,
            "replay_counts_may_differ": True,
        },
        "forward_graph_pregather_lifecycle": {
            "registry_batch_sizes": list(range(1, required_batch + 1)),
            "per_batch": forward_per_batch,
            "one_final_full_capture_per_batch_per_arm": True,
            "graph_signatures_unique_within_each_arm": True,
            "conv_layout_signatures_unique_within_each_arm": True,
            "graph_signatures_equal_across_arms_per_batch": True,
            "conv_layout_signatures_equal_across_arms_per_batch": True,
            "measured_replays_match_event_histograms": True,
            "stage_precedes_all_layer_consumes": True,
            "profile_auxiliary_and_host_stage_counts_zero": True,
            "nonpure_dispatch_by_mode": nonpure_dispatch_by_mode,
            "nonpure_committer_replays_by_mode": (
                nonpure_committer_replays_by_mode
            ),
            "forbidden_mixed_full_dispatches_zero": True,
        },
        "scope": json.loads(json.dumps(FIXED_WORK_SCOPE)),
    }


def reduce_campaign(
    repo: Path,
    runroot: Path,
    tag: str,
    task_count: int,
    expected_concurrency: int | None,
    sidecar_dir: Path,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    if task_count not in EVIDENCE_SETS:
        raise GateError("task count must be exactly 4 or 16")
    if reps != BOOTSTRAP_REPS or seed != BOOTSTRAP_SEED:
        raise GateError(
            "formal moving-block parameters are pinned to "
            f"reps={BOOTSTRAP_REPS}, seed={BOOTSTRAP_SEED}"
        )
    source_fingerprint = validate_source_fingerprint(repo, runroot)
    external_fingerprint = validate_external_fingerprint(runroot)
    tail_name = f"tail6_fixed32_{tag}"
    hydra_name = f"hydra27_fixed32_{tag}"
    tail, tail_points = reduce_arm(
        repo,
        runroot,
        sidecar_dir,
        tail_name,
        mode="tail6_fixed32",
        task_count=task_count,
        expected_concurrency=expected_concurrency,
        reps=reps,
        seed=seed + 100_000,
    )
    hydra, hydra_points = reduce_arm(
        repo,
        runroot,
        sidecar_dir,
        hydra_name,
        mode="hydra27_fixed32",
        task_count=task_count,
        expected_concurrency=expected_concurrency,
        reps=reps,
        seed=seed + 200_000,
    )
    if tail["inferred_concurrency"] != hydra["inferred_concurrency"]:
        raise GateError("Tail6-fixed32/Hydra27-fixed32 inferred concurrency differs")
    tail_subset = tail["provenance"]["launch"]["subset"]
    hydra_subset = hydra["provenance"]["launch"]["subset"]
    if tail_subset != hydra_subset:
        raise GateError(
            "Tail6-fixed32/Hydra27-fixed32 canonical subset provenance differs"
        )
    tail_attestation = tail["provenance"]["runtime"]["runtime_attestation"]
    hydra_attestation = hydra["provenance"]["runtime"]["runtime_attestation"]
    tail_attested_identity = {
        key: value
        for key, value in tail_attestation.items()
        if key not in {"path"}
    }
    hydra_attested_identity = {
        key: value
        for key, value in hydra_attestation.items()
        if key not in {"path"}
    }
    if tail_attested_identity != hydra_attested_identity:
        raise GateError(
            "Tail6-fixed32/Hydra27-fixed32 runtime attestations differ"
        )
    runtime_attestation_match = {
        "byte_equal": tail_attestation["sha256"] == hydra_attestation["sha256"],
        "canonical_sha256": tail_attestation["canonical_sha256"],
        "file_sha256": tail_attestation["sha256"],
        "vllm": tail_attestation["vllm"],
        "forked_fa2": tail_attestation["forked_fa2"],
        "arctic": tail_attestation["arctic"],
    }
    concurrency = tail["inferred_concurrency"]
    census_paths = {
        "tail6_fixed32": Path(tail["work_census_expected"]["path"]),
        "hydra27_fixed32": Path(hydra["work_census_expected"]["path"]),
    }
    for path in census_paths.values():
        if Path(f"{path}.tmp").exists():
            raise GateError(f"{path}: stale work-census temporary file is present")
    try:
        work_census_report = validate_work_census_campaign(
            load_work_census_jsonl(census_paths["tail6_fixed32"]),
            load_work_census_jsonl(census_paths["hydra27_fixed32"]),
            required_batches=(concurrency,),
        )
    except WorkCensusError as error:
        raise GateError(f"fixed32 work census failed: {error}") from error
    work_census_v5 = validate_work_census_v5_report(
        work_census_report,
        required_batch=concurrency,
    )
    b4_occupancy: dict[str, dict[str, Any]] = {}
    for mode, arm_result in (
        ("tail6_fixed32", tail),
        ("hydra27_fixed32", hydra),
    ):
        expected_census = arm_result["work_census_expected"]
        complete_stream = expected_census["complete_stream"]
        if (
            work_census_report["forward_step_indices"][mode]
            != complete_stream["forward_step_indices"]
        ):
            raise GateError(
                f"{mode}: work-census global forward-step indices do not "
                "exactly match the complete SFWD stream"
            )
        actual_histogram = work_census_report["event_counts"][mode]
        if actual_histogram != complete_stream["occupied_batch_histogram"]:
            raise GateError(
                f"{mode}: work-census occupancy does not match the complete "
                f"SFWD stream: {actual_histogram} != "
                f"{complete_stream['occupied_batch_histogram']}"
            )
        if sum(actual_histogram.values()) != complete_stream["event_count"]:
            raise GateError(
                f"{mode}: work-census event count does not match the complete "
                "SFWD stream"
            )
        actual_batch_sequence = work_census_report["batch_size_sequences"][mode]
        if actual_batch_sequence != complete_stream["batch_size_sequence"]:
            raise GateError(
                f"{mode}: work-census batch sequence does not match the "
                "complete SFWD stream"
            )
        if work_census_report["producer_pids"][mode] != expected_census["producer_pid"]:
            raise GateError(
                f"{mode}: work-census PID does not match recorded EngineCore PID"
            )
        terminal = work_census_report["terminal_summaries"][mode]
        final_counters = arm_result["flush_chain"]["final"]["counters"]
        if (
            terminal["producer_pid"] != expected_census["producer_pid"]
            or terminal["event_count"]
            != final_counters["complete_work_census_events"]
            or terminal["last_forward_step_index"]
            != final_counters["work_census_last_forward_step"]
        ):
            raise GateError(
                f"{mode}: final flush counters do not match terminal census summary"
            )
        selection = expected_census["canonical_task_selection"]
        if (
            selection["forward_step_indices"]
            != complete_stream["forward_step_indices"]
        ):
            raise GateError(
                f"{mode}: canonical-task forward union does not cover the "
                "complete post-ready decode stream"
            )
        selected_batches = [
            actual_batch_sequence[index] for index in selection["forward_step_indices"]
        ]
        if selected_batches != selection["batch_size_sequence"]:
            raise GateError(
                f"{mode}: post-hoc work-census task selection does not match "
                "the canonical task forward union"
            )
        selected_histogram = {
            str(batch_size): selected_batches.count(batch_size)
            for batch_size in range(1, concurrency + 1)
            if batch_size in selected_batches
        }
        if (
            len(selected_batches) != selection["event_count"]
            or selected_histogram != selection["occupied_batch_histogram"]
        ):
            raise GateError(
                f"{mode}: post-hoc work-census task selection cardinality "
                "or occupancy differs from the canonical task forward union"
            )
        if concurrency == 4:
            selected_count = len(selected_batches)
            exact_b4_events = selected_batches.count(4)
            ge3_events = sum(batch >= 3 for batch in selected_batches)
            mean_occupancy = sum(selected_batches) / selected_count
            ge3_fraction = ge3_events / selected_count
            if (
                exact_b4_events < MIN_B4_EXACT_EVENTS
                or ge3_fraction < MIN_B4_GE3_FRACTION
                or mean_occupancy < MIN_B4_MEAN_OCCUPANCY
            ):
                raise GateError(
                    f"{mode}: B4 canonical-task exposure is under-occupied: "
                    f"exact_b4={exact_b4_events} ge3_fraction={ge3_fraction:.6f} "
                    f"mean={mean_occupancy:.6f}"
                )
            b4_occupancy[mode] = {
                "selected_events": selected_count,
                "exact_b4_events": exact_b4_events,
                "at_least_b3_events": ge3_events,
                "at_least_b3_fraction": ge3_fraction,
                "mean_occupancy": mean_occupancy,
            }
    if concurrency == 4:
        mean_gap = abs(
            b4_occupancy["tail6_fixed32"]["mean_occupancy"]
            - b4_occupancy["hydra27_fixed32"]["mean_occupancy"]
        )
        if mean_gap > MAX_B4_MEAN_OCCUPANCY_GAP:
            raise GateError(
                "fixed32 B4 arm mean occupancies are not matched: "
                f"gap={mean_gap:.6f}"
            )
        b4_occupancy["matched_arm_mean_gap"] = mean_gap
    work_census = {
        "report": work_census_report,
        "physical_work_comparison": work_census_v5[
            "physical_work_comparison"
        ],
        "drafter_graph_lifecycle": work_census_v5[
            "drafter_graph_lifecycle"
        ],
        "forward_graph_pregather_lifecycle": work_census_v5[
            "forward_graph_pregather_lifecycle"
        ],
        "scope": work_census_v5["scope"],
        "scope_interpretation": (
            "Exact equality is limited to direct observations and "
            "contract-derived work listed in scope. The explicit "
            "data_dependent_unproven entries prevent a total memory-traffic "
            "or hardware-cycle claim."
        ),
        "files": {
            mode: {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for mode, path in census_paths.items()
        },
        "complete_terminal_stream_reconciled_to_sfwd_sidecar": True,
        "canonical_task_forward_counter_union_selected_posthoc": True,
        "canonical_task_forward_union_covers_complete_stream": True,
        "b4_occupancy_gate": (
            b4_occupancy if concurrency == 4 else "not_applicable_b1"
        ),
    }
    comparison = (
        b1_comparison(tail_points, hydra_points)
        if concurrency == 1
        else {
            "scope": (
                "descriptive union-counter point differences only; no paired "
                "task-general B=4 inference"
            ),
            "hydra_minus_tail_wall_ms_per_step": (
                hydra["statistics"]["union_counter_point"]["wall_ms_per_step"]
                - tail["statistics"]["union_counter_point"]["wall_ms_per_step"]
            ),
            "hydra_minus_tail_verify_ms_per_step": (
                hydra["statistics"]["union_counter_point"]["verify_ms_per_step"]
                - tail["statistics"]["union_counter_point"]["verify_ms_per_step"]
            ),
        }
    )
    gates = {
        "source_runtime_fingerprint_equal": True,
        "external_artifact_fingerprint_equal": True,
        "arm_runtime_attestations_equal": True,
        "running_container_image_identity_exact": True,
        "task_metric_bracket_bytes_bound": True,
        "fixed32_pretask_zero_positive_traffic": all(
            arm["provenance"]["runtime"]["pretask_zero_traffic"][
                "forbidden_probe_artifacts_absent"
            ]
            for arm in (tail, hydra)
        ),
        "all_canonical_tasks_have_real_model_traffic": all(
            arm["provenance"]["real_tasks"][
                "all_canonical_tasks_have_real_model_traffic"
            ]
            for arm in (tail, hydra)
        ),
        "all_validated_chat_task_traffic_bound": all(
            arm["provenance"]["real_tasks"][
                "all_validated_chat_task_traffic_bound"
            ]
            for arm in (tail, hydra)
        ),
        "fixed32_ingress_proxy_engine_exact": all(
            arm["provenance"]["real_tasks"][
                "fixed32_ingress_proxy_engine_exact"
            ]
            for arm in (tail, hydra)
        ),
        "fixed32_zero_campaign_rejections": all(
            arm["provenance"]["real_tasks"][
                "fixed32_zero_campaign_rejections"
            ]
            for arm in (tail, hydra)
        ),
        "fixed32_raw_proxy_dumps_disabled": all(
            arm["provenance"]["real_tasks"][
                "fixed32_raw_proxy_dumps_disabled"
            ]
            for arm in (tail, hydra)
        ),
        "all_task_agents_completed_cleanly": all(
            arm["provenance"]["real_tasks"]["all_agents_completed_cleanly"]
            for arm in (tail, hydra)
        ),
        "all_tasks_have_terminal_swe_verdicts": all(
            arm["provenance"]["real_tasks"][
                "all_tasks_have_terminal_eval_verdicts"
            ]
            for arm in (tail, hydra)
        ),
        "canonical_exact_4_or_16_task_binding": task_count in EVIDENCE_SETS,
        "canonical_completed_task_set": True,
        "canonical_subset_hash": True,
        "uncapped_sidecars": True,
        "sidecar_coverage_eq_1_0": True,
        "sidecar_counter_reconciliation": True,
        "sidecar_wall_predecessor_binding_exact": True,
        "sidecar_wall_forward_occupancy_equal": True,
        "sidecar_retained_wall_fraction_ge_pinned_minimum": True,
        "sidecar_timer_integrity_counters_zero": True,
        "fixed32_work_census_exact": True,
        "fixed32_per_batch_physical_work_equal": True,
        "fixed32_drafter_graph_lifecycle_exact_and_matched": True,
        "fixed32_forward_graph_pregather_exact": True,
        "fixed32_scope_limitations_explicit": bool(
            work_census_v5["scope"]["data_dependent_unproven"]
        ),
        "canonical_task_forward_union_covers_complete_stream": True,
        "fixed32_flush_generation_chain_exact": True,
        "fixed32_task_boundaries_exact": True,
        "b4_occupancy_exposure": True,
        "tail6_fixed32_legacy_slo": tail["statistics"]["gate"]["pass"],
        "hydra27_fixed32_legacy_slo": hydra["statistics"]["gate"]["pass"],
    }
    return {
        "schema": "fr13.canonical_swe_verified_fixed32_floor_gate.v11",
        "analysis_valid": True,
        "gate_verdict": "PASS" if all(gates.values()) else "FAIL",
        "repo": str(repo),
        "runroot": str(runroot),
        "tag": tag,
        "task_count": task_count,
        "inferred_concurrency": concurrency,
        "source_runtime_fingerprint": source_fingerprint,
        "external_artifact_fingerprint": external_fingerprint,
        "matched_runtime_attestation": runtime_attestation_match,
        "fixed32_work_census": work_census,
        "slo_definition": {
            "name": "legacy_aggressive_weight_stream_slo",
            "formula": ("wall_ms_per_step <= 1.15 * max(98.6, 0.54 * rows_per_step)"),
            "weight_stream_lower_bound_ms": WEIGHT_STREAM_LOWER_BOUND_MS,
            "compute_ms_per_row": COMPUTE_MS_PER_ROW,
            "multiplier": SLO_MULTIPLIER,
            "interpretation": (
                "98.6 ms is a weight-stream lower bound used by an aggressive "
                "legacy SLO; it is not a measured full physical hardware floor"
            ),
        },
        "uncertainty_model": (
            "B=1 uses equal-weight whole-task clusters and one-sided t U95; "
            "B=4 uses one overlap-safe physical-step union plus a separately "
            "gated exact-B4 wall stratum, both with conditional moving-block "
            "U95 sensitivity at blocks 64/128/256/512"
        ),
        "evidence_requirements": {
            "sidecar_counter_interval_coverage": REQUIRED_COVERAGE,
            "sidecar_main_schema": SFWD_MAIN_SIDECAR_SCHEMA,
            "sidecar_sample_schema": SFWD_SAMPLE_SIDECAR_SCHEMA,
            "sidecar_drafts_and_steps": "exactly_reconciled",
            "sidecar_timing": "within_4_decimal_ms_per_sample_rounding_bound",
            "wall_samples": "exact_predecessor_forward_index_bound",
            "minimum_retained_wall_fraction": MIN_RETAINED_WALL_FRACTION,
            "wall_invalid_request_ids": 0,
            "wall_bookkeeping_errors": 0,
            "runtime_flush": "ready_snapshot_per_task_final_generation_chain_exact",
            "minimum_retained_steps_per_task_and_family": MIN_TASK_COUNTER_STEPS,
            "wall_rejected_delta_per_task": 0,
            "wall_cap_seconds": 1.5,
            "b4_selected_exact_events_minimum": MIN_B4_EXACT_EVENTS,
            "b4_selected_at_least_b3_fraction_minimum": MIN_B4_GE3_FRACTION,
            "b4_selected_mean_occupancy_minimum": MIN_B4_MEAN_OCCUPANCY,
            "b4_arm_mean_occupancy_gap_maximum": MAX_B4_MEAN_OCCUPANCY_GAP,
            "moving_block_bootstrap_reps": BOOTSTRAP_REPS,
            "moving_block_bootstrap_seed": BOOTSTRAP_SEED,
        },
        "arms": {"tail6_fixed32": tail, "hydra27_fixed32": hydra},
        "comparison": comparison,
        "gates": gates,
    }


def fixture_metrics(
    fwd_ms: list[float],
    wall_ms: list[float],
    drafts: list[int],
    index: int,
    tokens_per_draft: int,
) -> str:
    fwd_seconds = sum(fwd_ms[:index]) / 1000.0
    wall_seconds = sum(wall_ms[:index]) / 1000.0
    draft_count = sum(drafts[:index])
    values = {
        "fwd_s": fwd_seconds,
        "fwd_steps": index,
        "fwd_drafts": draft_count,
        "wall_s": wall_seconds,
        "wall_drafts": draft_count,
        "wall_steps": index,
        "wall_attempts": index,
        "wall_rejected": 0,
        "spec_drafts": draft_count,
        "spec_tokens": draft_count * tokens_per_draft,
    }
    lines = []
    for key, metric in METRICS.items():
        labels = (
            '{engine="0",model_name="qwen3.6-27b"}'
            if key in {"spec_drafts", "spec_tokens"}
            else ""
        )
        lines.append(f"{metric}{labels} {values[key]}")
    lines.append(f"{FIXED32_STEP_METRIC} {index}")
    lines.append(f"{FIXED32_CENSUS_METRIC} {index}")
    return "\n".join(lines) + "\n"


def replace_metric_values(text: str, replacements: dict[str, float]) -> str:
    names = {METRICS[key]: value for key, value in replacements.items()}
    seen: set[str] = set()
    output = []
    for line in text.splitlines():
        match = SAMPLE_RE.match(line)
        if match is not None and match.group("name") in names:
            name = match.group("name")
            line = line[: match.start("value")] + f"{names[name]:.12g}"
            seen.add(name)
        output.append(line)
    if seen != set(names):
        raise AssertionError(f"fixture metric replacements missing {set(names) - seen}")
    return "\n".join(output) + "\n"


def fixture_external_manifest() -> dict[str, Any]:
    model_files = fixed32_contract.expected_model_file_records()
    payload: dict[str, Any] = {
        "schema": fixed32_contract.EXTERNAL_SCHEMA,
        "canonical_format": fixed32_contract.CANONICAL_FORMAT,
        "image": {
            "reference": fixed32_contract.IMAGE_REFERENCE,
            "id": fixed32_contract.IMAGE_ID,
            "repo_digests": [fixed32_contract.IMAGE_REFERENCE],
            "os": fixed32_contract.IMAGE_OS,
            "architecture": fixed32_contract.IMAGE_ARCHITECTURE,
        },
        "forked_fa2": {
            "path": fixed32_contract.FA2_REPO_RELATIVE,
            "size": fixed32_contract.FA2_SIZE,
            "sha256": fixed32_contract.FA2_SHA256,
        },
        "model": {
            "root": str(fixed32_contract.MODEL_ROOT),
            "file_count": len(model_files),
            "files": model_files,
            "canonical_sha256": fixed32_contract.MODEL_CANONICAL_SHA256,
        },
        "arctic_source": {
            "version": fixed32_contract.ARCTIC_VERSION,
            "url": fixed32_contract.ARCTIC_SDIST_URL,
            "sha256": fixed32_contract.ARCTIC_SDIST_SHA256,
        },
    }
    payload["overall_canonical_sha256"] = canonical_json_sha256(payload)
    validate_external_manifest(payload)
    return payload


def fixture_runtime_attestation() -> dict[str, Any]:
    arctic_files = [
        {
            "path": "arctic_inference/suffix_decoding/cache.py",
            "size": 32,
            "sha256": hashlib.sha256(b"fixed32-fixture-arctic").hexdigest(),
        }
    ]
    fa2_source = {
        "path": str(fixed32_contract.CONTAINER_FA2_SOURCE),
        "size": fixed32_contract.FA2_SIZE,
        "sha256": fixed32_contract.FA2_SHA256,
    }
    fa2_destination = {
        "path": str(CONTAINER_FA2_DESTINATION),
        "size": fixed32_contract.FA2_SIZE,
        "sha256": fixed32_contract.FA2_SHA256,
    }
    payload: dict[str, Any] = {
        "schema": fixed32_contract.RUNTIME_SCHEMA,
        "canonical_format": fixed32_contract.CANONICAL_FORMAT,
        "python": {
            "version": "3.12.3",
            "implementation": "CPython",
        },
        "vllm": {
            "version": fixed32_contract.VLLM_VERSION,
            "module_path": "/usr/local/lib/python3.12/dist-packages/vllm/__init__.py",
        },
        "forked_fa2": {
            "source": fa2_source,
            "destination": fa2_destination,
            "byte_identical": True,
        },
        "arctic": {
            "name": "arctic-inference",
            "version": fixed32_contract.ARCTIC_VERSION,
            "files": arctic_files,
            "canonical_sha256": canonical_json_sha256(arctic_files),
            "cache_class_module": "arctic_inference.suffix_decoding.cache",
            "cache_class_qualname": "SuffixDecodingCache",
            "pinned_source_url": fixed32_contract.ARCTIC_SDIST_URL,
            "pinned_source_sha256": fixed32_contract.ARCTIC_SDIST_SHA256,
        },
    }
    payload["overall_canonical_sha256"] = canonical_json_sha256(payload)
    validate_runtime_attestation(payload)
    return payload


def fixture_qwen_runtime_attestation() -> dict[str, Any]:
    bundle_tree = json.loads(
        json.dumps(FIXED32_QWEN_BUNDLE_TREE, sort_keys=True)
    )
    payload = {
        "schema": FIXED32_QWEN_RUNTIME_ATTESTATION_SCHEMA,
        "launcher": "qwen-code-instance-image",
        "agent_env": "instance_image",
        "host_mode": "remote",
        "qwen_code_version": FIXED32_QWEN_CODE_VERSION,
        "bundle_tree": bundle_tree,
        "bundle_manifest_sha256": bundle_tree["manifest_sha256"],
        "bundle_snapshot": {
            "kind": "per-task-content-addressed-snapshot",
            "basename": (
                "qwen_bundle-" + bundle_tree["manifest_sha256"]
            ),
            "container_path": "/opt/qwen",
            "mount_mode": "ro",
        },
        "cleared_agent_environment": list(
            FIXED32_CLEARED_AGENT_ENVIRONMENT
        ),
        "system_settings": {
            "source": "config/fr13_fixed32/qwen_system_settings.json",
            "bytes": 37,
            "sha256": FIXED32_QWEN_SYSTEM_SETTINGS_SHA256,
            "container_path": "/run/fr13/qwen-system-settings.json",
            "mount_mode": "ro",
            "environment": {
                "name": "QWEN_CODE_SYSTEM_SETTINGS_PATH",
                "value": "/run/fr13/qwen-system-settings.json",
            },
            "remote_file": {
                "mode": "0444",
                "uid": 1000,
                "gid": 1000,
                "nlink": 1,
                "xattrs": [],
            },
            "enable_auto_skill": False,
        },
    }
    _fixed32_qwen_runtime_attestation(
        payload,
        label="fixture Qwen runtime attestation",
    )
    return payload


def fixture_mounted_runtime_proof() -> dict[str, Any]:
    payload = {
        "schema": FIXED32_MOUNTED_RUNTIME_PROOF_SCHEMA,
        "bundle_tree": {
            "container_path": "/opt/qwen",
            "mount_mode": "ro",
            "write_probe_errno": 30,
            "observation": {
                "qwen_code_version": FIXED32_QWEN_CODE_VERSION,
                "bundle_tree": json.loads(
                    json.dumps(FIXED32_QWEN_BUNDLE_TREE)
                ),
            },
        },
        "system_settings": {
            "container_path": "/run/fr13/qwen-system-settings.json",
            "mount_mode": "ro",
            "write_probe_errno": 30,
            **FIXED32_QWEN_REMOTE_SETTINGS,
            "file_identity_sha256": "1" * 64,
        },
    }
    _fixed32_mounted_runtime_proof(
        payload,
        label="fixture mounted-runtime proof",
    )
    return payload


def write_fixture_arm(
    repo: Path,
    runroot: Path,
    sidecar_dir: Path,
    tag: str,
    *,
    hydra: bool,
    concurrency: int,
) -> None:
    kind = "hydra27_fixed32" if hydra else "tail6_fixed32"
    arm = f"{kind}_{tag}"
    arm_dir = runroot / arm
    arm_dir.mkdir(parents=True)
    logs_dir = arm_dir / "logs"
    logs_dir.mkdir()
    tokens = PHYSICAL_DRAFTS
    subset = (repo / EVIDENCE_SETS[4]["relative_path"]).resolve()
    producer_pid = 100 + int(hydra)
    pid1_argv = expected_pid1_argv(concurrency)
    (runroot / f"{arm}.runlog").write_text(
        f"=== BIGDENOM-VARIANT SWEServe ARM {arm} kind={kind} "
        f"launcher=forked expect={tokens} xflags=[] "
        f"subset={subset} ===\n"
        f"PID 1 cmd=[{' '.join(pid1_argv)} ]\n"
        f"PID {producer_pid} cmd=[VLLM::EngineCore ]\n"
        f"spec engagement OK: drafts delta=8.0 "
        f"draft_tokens/drafts={float(tokens):.1f}\n",
        encoding="utf-8",
    )
    mode_spec = FIXED32_MODE_SPECS[kind]
    required_env = fixed32_required_env(
        arm_dir,
        mode=kind,
        task_ids=list(EVIDENCE_SETS[4]["task_ids"]),
    )
    (arm_dir / "container_env.txt").write_text(
        "".join(f"{key}={value}\n" for key, value in required_env.items()),
        encoding="utf-8",
    )
    process_identity = {
        "schema": "fr13-fixed32-process-identity-v1",
        "pid1": {
            "pid": 1,
            "argv": pid1_argv,
            "environ": sorted(f"{key}={value}" for key, value in required_env.items()),
            "forked_fa2_maps": [],
        },
        "engine_core": {
            "pid": producer_pid,
            "argv": ["VLLM::EngineCore"],
            "environ": sorted(f"{key}={value}" for key, value in required_env.items()),
            "forked_fa2_maps": [
                "7f000000-7f100000 r-xp 00000000 00:00 0 "
                f"{CONTAINER_FA2_DESTINATION}"
            ],
        },
    }
    (arm_dir / "fixed32_process_identity.json").write_text(
        json.dumps(
            process_identity,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (arm_dir / "fixed32_engine_cmdline.txt").write_text(
        "VLLM::EngineCore\n", encoding="utf-8"
    )
    (arm_dir / "fixed32_container_identity.json").write_text(
        json.dumps(
            {
                "schema": "fr13-fixed32-container-identity-v1",
                "name": f"/fr13-bigdenom-{arm}",
                "image_id": fixed32_contract.IMAGE_ID,
                "configured_image": fixed32_contract.IMAGE_REFERENCE,
                "platform": fixed32_contract.IMAGE_OS,
                "running": True,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (logs_dir / "fr13_fixed32_runtime_attestation.json").write_text(
        json.dumps(
            fixture_runtime_attestation(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (logs_dir / "fr13_fixed32_ingress_secret_identity.json").write_text(
        json.dumps(
            {
                "schema": "fr13-fixed32-ingress-secret-identity-v1",
                "path": "/run/fr13_fixed32_ingress_secret",
                "regular": True,
                "symlink": False,
                "uid": 0,
                "gid": 0,
                "mode": "0600",
                "bytes": 192,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    needles = (
        FIXED32_PRESEED,
        FIXED32_ENGAGED,
        FIXED32_WORK_ENGAGED,
        mode_spec["topology_needle"],
    )
    (arm_dir / "docker_full.log").write_text(
        "\n".join(needles) + "\n", encoding="utf-8"
    )
    (arm_dir / "eval_offload_preflight.txt").write_text(
        "eval offload: configured evaluator reachable\n",
        encoding="utf-8",
    )
    task_ids = EVIDENCE_SETS[4]["task_ids"]
    (arm_dir / "offload_proxy_env.txt").write_text(
        "\n".join(
            (
                "LUMO_PROXY_FIXED32_DISABLE_RAW_DUMPS=1",
                (
                    "LUMO_PROXY_FIXED32_LEDGER_PATH="
                    "/home/fixture/fr13_fixed32_proxy_ingress.jsonl"
                ),
                (
                    "LUMO_PROXY_FIXED32_SECRET_FILE="
                    "/home/fixture/fr13_fixed32_ingress_secret"
                ),
                "LUMO_PROXY_FIXED32_TASK_IDS=" + ",".join(task_ids),
            )
        )
        + "\n",
        encoding="ascii",
    )
    orchestrator_lines = [
        "=== [2026-01-01T00:00:00Z] "
        "dataset=princeton-nlp/SWE-bench_Verified tag=verified "
        f"n=4 concurrency={concurrency} ==="
    ]
    for index, task_id in enumerate(task_ids):
        orchestrator_lines.extend(
            [
                f"[2026-01-01T00:00:{index * 2 + 1:02d}Z] -> {task_id}",
                f"[2026-01-01T00:00:{index * 2 + 2:02d}Z] <- {task_id} "
                "verdict=resolved elapsed_total=1.0s",
            ]
        )
    orchestrator_lines.append(
        "=== [2026-01-01T00:01:00Z] DONE n=4 "
        "resolved_rate=1.0 verdicts={'resolved': 4} ==="
    )
    (arm_dir / "swe_orchestrator.log").write_text(
        "\n".join(orchestrator_lines) + "\n", encoding="utf-8"
    )

    if concurrency == 1:
        lengths = (64, 80, 96, 112)
        fwd_ms = []
        wall_ms = []
        drafts = []
        intervals = []
        cursor = 0
        for task_index, length in enumerate(lengths):
            start = cursor
            fwd_value = 70.0 + task_index * 2 + (3.0 if hydra else 0.0)
            wall_value = 100.0 + task_index * 4 + (5.0 if hydra else 0.0)
            fwd_ms.extend([fwd_value] * length)
            wall_ms.extend([wall_value] * length)
            drafts.extend([1] * length)
            cursor += length
            intervals.append((start, cursor))
    else:
        sample_count = 645
        fwd_ms = [
            70.0 + (index % 13) * 0.1 + (3.0 if hydra else 0.0)
            for index in range(sample_count)
        ]
        wall_ms = [
            100.0 + (index % 17) * 0.2 + (5.0 if hydra else 0.0)
            for index in range(sample_count)
        ]
        occupancy_pattern = (3, 4, 4, 4, 4)
        drafts = [
            occupancy_pattern[index % len(occupancy_pattern)]
            for index in range(sample_count)
        ]
        intervals = [(0, 500), (20, 620), (40, 640), (60, 645)]

    requests_per_task = 1 if concurrency == 1 else 4
    engine_request_ids = {
        task_id: [
            f"chatcmpl-fixture-{kind}-{task_index}-{request_index}"
            for request_index in range(requests_per_task)
        ]
        for task_index, task_id in enumerate(task_ids)
    }
    census_records = []
    for event_index, batch_size in enumerate(drafts):
        eligible_requests = [
            request_id
            for task_id, (start, end) in zip(
                task_ids,
                intervals,
                strict=True,
            )
            if start <= event_index < end
            for request_id in engine_request_ids[task_id]
        ]
        if len(eligible_requests) < int(batch_size):
            raise AssertionError(
                "fixture task brackets do not provide enough active requests"
            )
        offset = event_index % len(eligible_requests)
        rotated = eligible_requests[offset:] + eligible_requests[:offset]
        record = work_census_fixture(
            kind,
            int(batch_size),
            f"{kind}-{event_index}",
            event_index=event_index,
            forward_step_index=event_index,
            request_ids=rotated[: int(batch_size)],
        )
        record["producer_pid"] = producer_pid
        census_records.append(record)

    def flush_counters(step: int) -> dict[str, Any]:
        return {
            "pure_decode_forward_steps": step,
            "complete_work_census_events": step,
            "work_census_first_forward_step": 0 if step else None,
            "work_census_last_forward_step": step - 1 if step else None,
            "sfwd_pending": 0,
            "dfwd_pending": 0,
            "cfwd_pending": 0,
        }

    def flush_ack(generation: int, step: int, action: str) -> dict[str, Any]:
        return {
            "schema": FLUSH_ACK_SCHEMA,
            "mode": kind,
            "producer_pid": producer_pid,
            "generation": generation,
            "nonce": (
                FLUSH_READY_NONCE
                if generation == 0
                else f"{generation:064x}"
            ),
            "action": action,
            "status": "ok",
            "counters": flush_counters(step),
        }

    def write_runtime_snapshot(
        ack: dict[str, Any],
        step: int,
    ) -> dict[str, Any]:
        prefix = census_records[:step]
        histogram = {
            str(batch): drafts[:step].count(batch) for batch in range(1, 5)
        }
        draft_count = sum(drafts[:step])
        by_batch = {
            str(batch): histogram[str(batch)] for batch in range(1, 5)
        }
        capture_by_batch = {
            str(batch): int(batch <= concurrency) for batch in range(1, 5)
        }
        zero_by_batch = {str(batch): 0 for batch in range(1, 5)}
        snapshot = {
            "schema": FIXED32_RUNTIME_SNAPSHOT_SCHEMA,
            "mode": kind,
            "producer_pid": producer_pid,
            "generation": ack["generation"],
            "nonce": ack["nonce"],
            "action": ack["action"],
            "counters": ack["counters"],
            "metrics": {
                "fixed32": {
                    "pure_decode_forward_steps": step,
                    "complete_work_census_events": step,
                    "complete_spec_rows": draft_count,
                    "spec_drafts": draft_count,
                    "spec_tokens": tokens * draft_count,
                    "batch_histogram": histogram,
                    "first_forward_step": 0 if step else None,
                    "last_forward_step": step - 1 if step else None,
                    "events_sha256": hashlib.sha256(
                        json.dumps(
                            prefix,
                            ensure_ascii=True,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                },
                "sfwd": {
                    "gpu_seconds": sum(fwd_ms[:step]) / 1000.0,
                    "steps": step,
                    "drafts": draft_count,
                    "wall_seconds": sum(wall_ms[:step]) / 1000.0,
                    "wall_drafts": draft_count,
                    "wall_steps": step,
                    "wall_rejected": 0,
                },
                "dfwd": {
                    "gpu_seconds": step * 0.001,
                    "spans": step,
                },
                "cfwd": {
                    "gpu_seconds": step * 0.002,
                    "spans": step,
                },
                "committer": {
                    "captures": concurrency,
                    "actual_replays_enqueued": step,
                    "actual_replays_by_batch": by_batch,
                    "nonpure_committer_replays_enqueued": 0,
                    "nonpure_committer_replays_by_batch": zero_by_batch,
                    "nonpure_dispatch": {
                        "guarded_steps": 0,
                        "piecewise_steps": 0,
                        "none_steps": 0,
                        "forbidden_full_steps": 0,
                    },
                    "preseeded_graphs": concurrency,
                    "preseeded_batches": list(range(1, concurrency + 1)),
                    "ready_capacities": {
                        str(batch): concurrency
                        for batch in range(1, concurrency + 1)
                    },
                    "maximum_ready_capacity": concurrency,
                    "required_capacity": concurrency,
                    "fast_route_ready": True,
                    "all_batches_ready": True,
                },
                "conv_pregather": {
                    "preseeded": True,
                    "pointer_entries": 48,
                    "preseeded_batches": list(range(1, concurrency + 1)),
                    "max_batch_size": concurrency,
                    "graph_capture_stages": concurrency,
                    "graph_capture_stages_by_batch": capture_by_batch,
                    "profile_capture_stages": 0,
                    "aux_capture_stages": 0,
                    "actual_stages": 0,
                    "actual_stages_by_batch": zero_by_batch,
                    "graph_replay_stages": step,
                    "graph_replay_stages_by_batch": by_batch,
                },
            },
        }
        snapshot_path = (
            logs_dir
            / f"fr13_fixed32_boundary_snapshot.{ack['generation']}.json"
        )
        snapshot_path.write_text(
            json.dumps(snapshot, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "schema": FIXED32_RUNTIME_SNAPSHOT_SCHEMA,
            "generation": ack["generation"],
            "path": str(snapshot_path),
            "sha256": sha256_file(snapshot_path),
        }

    boundary_points = []
    for task_id, (start, end) in zip(task_ids, intervals, strict=True):
        boundary_points.extend(((start, "pre", task_id), (end, "post", task_id)))
    generation_by_boundary = {
        (task_id, boundary): generation
        for generation, (_step, boundary, task_id) in enumerate(
            sorted(boundary_points), start=1
        )
    }

    ready_ack = flush_ack(0, 0, "ready")
    ready_path = arm_dir / "fixed32_ready_ack.json"
    ready_path.write_text(
        json.dumps(ready_ack, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics_before_path = arm_dir / "metrics_before_swe.txt"
    metrics_before_path.write_text(
        fixture_metrics(fwd_ms, wall_ms, drafts, 0, tokens),
        encoding="utf-8",
    )
    census_path = logs_dir / "fr13_fixed32_work_census.jsonl"
    pretask_marker = {
        "schema": "fr13-fixed32-pretask-zero-traffic-v1",
        "mode": kind,
        "no_positive_probe": True,
        "generation_probe_commands_executed": 0,
        "metrics": {
            "path": str(metrics_before_path.resolve()),
            "sha256": sha256_file(metrics_before_path),
            "spec_drafts": 0,
            "spec_tokens": 0,
        },
        "work_census": {
            "path": str(census_path.resolve()),
            "exists": False,
            "bytes": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
        },
        "ready_ack": {
            "path": str(ready_path.resolve()),
            "sha256": sha256_file(ready_path),
            "generation": 0,
        },
    }
    (arm_dir / "fixed32_pretask_zero_traffic.json").write_text(
        json.dumps(
            pretask_marker,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (logs_dir / "fr13_fixed32_engine_pid").write_text(
        f"{producer_pid}\n", encoding="utf-8"
    )
    (logs_dir / "fr13_fixed32_mode.flag").write_text(
        f"{kind}\n", encoding="utf-8"
    )

    task_root = arm_dir / "swe_out" / "verified" / "per_task"
    (arm_dir / "offload_fetch_status.txt").write_text("ok\n", encoding="utf-8")
    dataset_record_digests = pinned_dataset_record_digests(str(repo))
    canonical_task_set_sha256 = fixed32_canonical_task_set_sha256(
        list(task_ids)
    )
    task_key_ids = {
        task_id: fixed32_task_key_id(task_id) for task_id in task_ids
    }

    def append_ingress_record(
        rows: list[dict[str, Any]],
        *,
        role: str,
        phase: str,
        event: str,
        route: str | None = None,
        task_key_id: str | None = None,
        logical_id_sha256: str | None = None,
        wire_id_sha256: str | None = None,
        engine_request_id_sha256: str | None = None,
        status_code: int | None = None,
        outcome: str,
        reason: str | None = None,
        evidence_sha256: str | None = None,
    ) -> dict[str, Any]:
        row = {
            "schema": FIXED32_INGRESS_LEDGER_SCHEMA,
            "seq": len(rows),
            "role": role,
            "phase": phase,
            "event": event,
            "route": route,
            "task_key_id": task_key_id,
            "logical_id_sha256": logical_id_sha256,
            "wire_id_sha256": wire_id_sha256,
            "engine_request_id_sha256": engine_request_id_sha256,
            "status_code": status_code,
            "outcome": outcome,
            "reason": reason,
            "evidence_sha256": evidence_sha256,
            "prev_sha256": (
                rows[-1]["record_sha256"] if rows else "0" * 64
            ),
        }
        row["record_sha256"] = canonical_json_sha256(row)
        rows.append(row)
        return row

    proxy_rows: list[dict[str, Any]] = []
    engine_rows: list[dict[str, Any]] = []
    for role, rows, second_reason in (
        ("proxy", proxy_rows, "malformed_bearer"),
        ("engine", engine_rows, "invalid_engine_bearer"),
    ):
        for route in ("chat", "responses"):
            for reason in ("missing_bearer", second_reason):
                append_ingress_record(
                    rows,
                    role=role,
                    phase="preflight",
                    event="request_rejected",
                    route=route,
                    outcome="rejected",
                    reason=reason,
                )
        append_ingress_record(
            rows,
            role=role,
            phase="preflight",
            event="campaign_begin",
            outcome="begun",
            evidence_sha256=canonical_task_set_sha256,
        )
    proxy_begin_row = proxy_rows[-1]
    engine_begin_row = engine_rows[-1]
    proxy_task_snapshots: dict[str, dict[str, Any]] = {}
    for task_id in task_ids:
        task_key_id = task_key_ids[task_id]
        zero_evidence = {
            "schema": "fr13-fixed32-task-auth-evidence-v1",
            "task_key_id": task_key_id,
            "completed_logical_model_requests": 0,
            "aborted_logical_requests": 0,
            "accepted_attempts": 0,
            "completed_attempts": 0,
            "failed_attempts": 0,
            "phase": "campaign",
            "ledger_records": len(proxy_rows),
            "ledger_chain_head_sha256": proxy_rows[-1][
                "record_sha256"
            ],
        }
        for request_index, engine_request_id in enumerate(
            engine_request_ids[task_id]
        ):
            logical_id = hashlib.sha256(
                f"fixture-logical:{task_id}:{request_index}".encode("utf-8")
            ).hexdigest()
            wire_id = engine_request_id.removeprefix("chatcmpl-")
            wire_digest = hashlib.sha256(
                wire_id.encode("utf-8")
            ).hexdigest()
            engine_digest = hashlib.sha256(
                engine_request_id.encode("utf-8")
            ).hexdigest()
            evidence_digest = hashlib.sha256(
                f"fixture-evidence:{task_id}:{request_index}".encode(
                    "utf-8"
                )
            ).hexdigest()
            append_ingress_record(
                proxy_rows,
                role="proxy",
                phase="campaign",
                event="logical_begin",
                route="chat",
                task_key_id=task_key_id,
                logical_id_sha256=logical_id,
                outcome="accepted",
            )
            append_ingress_record(
                proxy_rows,
                role="proxy",
                phase="campaign",
                event="attempt_begin",
                route="chat",
                task_key_id=task_key_id,
                logical_id_sha256=logical_id,
                wire_id_sha256=wire_digest,
                engine_request_id_sha256=engine_digest,
                outcome="dispatched",
                evidence_sha256=evidence_digest,
            )
            append_ingress_record(
                engine_rows,
                role="engine",
                phase="campaign",
                event="request_accepted",
                route="chat",
                task_key_id=task_key_id,
                wire_id_sha256=wire_digest,
                engine_request_id_sha256=engine_digest,
                outcome="accepted",
                evidence_sha256=evidence_digest,
            )
            append_ingress_record(
                engine_rows,
                role="engine",
                phase="campaign",
                event="request_complete",
                route="chat",
                task_key_id=task_key_id,
                wire_id_sha256=wire_digest,
                engine_request_id_sha256=engine_digest,
                outcome="completed",
                evidence_sha256=evidence_digest,
            )
            append_ingress_record(
                proxy_rows,
                role="proxy",
                phase="campaign",
                event="attempt_result",
                route="chat",
                task_key_id=task_key_id,
                logical_id_sha256=logical_id,
                wire_id_sha256=wire_digest,
                engine_request_id_sha256=engine_digest,
                status_code=200,
                outcome="response",
                evidence_sha256=evidence_digest,
            )
            append_ingress_record(
                proxy_rows,
                role="proxy",
                phase="campaign",
                event="logical_complete",
                route="chat",
                task_key_id=task_key_id,
                logical_id_sha256=logical_id,
                outcome="completed",
            )
        request_count = len(engine_request_ids[task_id])
        task_evidence = {
            "completed_logical_model_requests": request_count,
            "aborted_logical_requests": 0,
            "accepted_attempts": request_count,
            "completed_attempts": request_count,
            "failed_attempts": 0,
        }
        after_evidence = {
            "schema": "fr13-fixed32-task-auth-evidence-v1",
            "task_key_id": task_key_id,
            **task_evidence,
            "phase": "campaign",
            "ledger_records": len(proxy_rows),
            "ledger_chain_head_sha256": proxy_rows[-1][
                "record_sha256"
            ],
        }
        proxy_task_snapshots[task_id] = {
            **task_evidence,
            "task_auth_evidence_before_sha256": canonical_json_sha256(
                zero_evidence
            ),
            "task_auth_evidence_after_sha256": canonical_json_sha256(
                after_evidence
            ),
            "task_auth_evidence_after_ledger_records": len(proxy_rows),
            "task_auth_evidence_after_ledger_chain_head_sha256": (
                proxy_rows[-1]["record_sha256"]
            ),
        }
    append_ingress_record(
        proxy_rows,
        role="proxy",
        phase="campaign",
        event="campaign_finalize",
        outcome="finalized",
        evidence_sha256=canonical_task_set_sha256,
    )
    append_ingress_record(
        engine_rows,
        role="engine",
        phase="campaign",
        event="campaign_finalize",
        outcome="finalized",
        evidence_sha256=canonical_task_set_sha256,
    )

    def write_ingress_ledger(
        role: str,
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        path = logs_dir / f"fr13_fixed32_{role}_ingress.jsonl"
        text = (
            "\n".join(
                json.dumps(
                    row,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                for row in rows
            )
            + "\n"
        )
        path.write_text(text, encoding="ascii")
        raw = text.encode("ascii")
        return {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "records": len(rows),
            "chain_head_sha256": rows[-1]["record_sha256"],
        }

    proxy_ledger_identity = write_ingress_ledger("proxy", proxy_rows)
    engine_ledger_identity = write_ingress_ledger("engine", engine_rows)
    preflight_requests = [
        {
            "route": route,
            "auth_case": auth_case,
            "status_code": 401,
        }
        for route in ("/v1/chat/completions", "/v1/responses")
        for auth_case in ("missing_bearer", "wrong_bearer")
    ]
    for role in ("proxy", "engine"):
        preflight: dict[str, Any] = {
            "schema": FIXED32_INGRESS_PREFLIGHT_SCHEMA,
            "role": role,
            "rejected_requests": 4,
            "accepted_requests": 0,
            "requests": preflight_requests,
            "denied_alternate_routes": [
                {"method": "POST", "route": route, "status_code": 403}
                for route in (
                    (
                        "/admin/invalidate",
                        "/admin/load_tuned_config",
                    )
                    if role == "proxy"
                    else ("/v1/completions", "/reset_prefix_cache")
                )
            ],
        }
        if role == "engine":
            preflight["non_inference_bypass"] = [
                {"route": route, "status_code": 200}
                for route in ("/health", "/metrics", "/v1/models")
            ]
        (arm_dir / f"fixed32_{role}_ingress_preflight.json").write_text(
            json.dumps(
                preflight,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="ascii",
        )

    for role, begin_row, begin_schema in (
        ("proxy", proxy_begin_row, FIXED32_PROXY_INGRESS_BEGIN_SCHEMA),
        ("engine", engine_begin_row, FIXED32_ENGINE_INGRESS_BEGIN_SCHEMA),
    ):
        begin = {
            "schema": begin_schema,
            "role": role,
            "phase": "campaign",
            "canonical_task_count": len(task_ids),
            "canonical_task_set_sha256": canonical_task_set_sha256,
            "preflight_rejected_requests": 4,
            "ledger_records": begin_row["seq"] + 1,
            "ledger_chain_head_sha256": begin_row["record_sha256"],
        }
        (arm_dir / f"fixed32_{role}_ingress_begin.json").write_text(
            json.dumps(
                begin,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="ascii",
        )

    canonical_task_keys = set(task_key_ids.values())
    proxy_counts = _fixed32_ingress_task_counts(
        proxy_rows,
        role="proxy",
        canonical_task_keys=canonical_task_keys,
    )
    engine_counts = _fixed32_ingress_task_counts(
        engine_rows,
        role="engine",
        canonical_task_keys=canonical_task_keys,
    )
    for role, counts, identity in (
        ("proxy", proxy_counts, proxy_ledger_identity),
        ("engine", engine_counts, engine_ledger_identity),
    ):
        totals = {
            key: sum(task_count[key] for task_count in counts.values())
            for key in next(iter(counts.values()))
        }
        final_common = {
            "role": role,
            "phase": "finalized",
            "canonical_task_count": len(task_ids),
            "canonical_task_set_sha256": canonical_task_set_sha256,
            "active_requests": 0,
            "preflight_rejected_requests": 4,
            "campaign_rejected_requests": 0,
            "task_evidence": [
                {"task_key_id": key_id, **task_count}
                for key_id, task_count in sorted(counts.items())
            ],
            "ledger_records": identity["records"],
            "ledger_chain_head_sha256": identity[
                "chain_head_sha256"
            ],
        }
        if role == "proxy":
            final = {
                "schema": FIXED32_PROXY_INGRESS_FINALIZE_SCHEMA,
                **final_common,
                "active_attempts": 0,
                "accepted_logical_requests": totals[
                    "accepted_logical_requests"
                ],
                "completed_logical_requests": totals[
                    "completed_logical_model_requests"
                ],
                "aborted_logical_requests": totals[
                    "aborted_logical_requests"
                ],
                "accepted_attempts": totals["accepted_attempts"],
                "completed_attempts": totals["completed_attempts"],
                "failed_attempts": totals["failed_attempts"],
            }
        else:
            final = {
                "schema": FIXED32_ENGINE_INGRESS_FINALIZE_SCHEMA,
                **final_common,
                "accepted_engine_requests": totals[
                    "accepted_engine_requests"
                ],
                "completed_engine_requests": totals[
                    "completed_engine_requests"
                ],
            }
        (arm_dir / f"fixed32_{role}_ingress_finalize.json").write_text(
            json.dumps(
                final,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="ascii",
        )

    for task_id, (start, end) in zip(task_ids, intervals, strict=True):
        task_dir = task_root / task_id
        task_dir.mkdir(parents=True)
        pre_ack = flush_ack(
            generation_by_boundary[(task_id, "pre")], start, "snapshot"
        )
        post_ack = flush_ack(
            generation_by_boundary[(task_id, "post")], end, "snapshot"
        )
        pre_runtime_snapshot = write_runtime_snapshot(pre_ack, start)
        post_runtime_snapshot = write_runtime_snapshot(post_ack, end)
        boundary = {
            "schema": FIXED32_BOUNDARY_SCHEMA,
            "instance_id": task_id,
            "mode": kind,
            "producer_pid": producer_pid,
            "pre": pre_ack,
            "post": post_ack,
            "pre_runtime_snapshot": pre_runtime_snapshot,
            "post_runtime_snapshot": post_runtime_snapshot,
            "forward_step_interval": {
                "start_forward_step": start,
                "end_forward_step": end,
                "expected_complete_events": end - start,
            },
        }
        boundary_path = task_dir / "fixed32_task_boundary.json"
        boundary_path.write_text(
            json.dumps(boundary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        model_trace_events = [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "id": engine_request_id,
                    "stop_reason": "end_turn",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Implemented and verified {task_id}.",
                        }
                    ],
                    "usage": {
                        "input_tokens": 128,
                        "output_tokens": 32,
                        "total_tokens": 160,
                    },
                },
            }
            for engine_request_id in engine_request_ids[task_id]
        ]
        trace_events = [
            {
                "type": "system",
                "subtype": "init",
                "qwen_code_version": FIXED32_QWEN_CODE_VERSION,
                "uuid": "system",
                "session_id": fixed32_contract.fixed32_trace_session_id(
                    task_id
                ),
                "parent_tool_use_id": None,
            },
            *model_trace_events,
        ]
        trace_path = task_dir / "qwen_trace.jsonl"
        trace_path.write_text(
            "\n".join(
                json.dumps(
                    trace_event,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                for trace_event in trace_events
            )
            + "\n",
            encoding="utf-8",
        )
        raw_trace = trace_path.read_bytes()
        qwen_runtime_attestation = fixture_qwen_runtime_attestation()
        qwen_runtime_attestation_sha256 = canonical_json_sha256(
            qwen_runtime_attestation
        )
        qwen_attestation_path = (
            task_dir / "qwen_runtime_attestation.json"
        )
        qwen_postrun_attestation_path = (
            task_dir / "qwen_runtime_attestation_post.json"
        )
        qwen_attestation_text = (
            json.dumps(
                qwen_runtime_attestation,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        qwen_attestation_path.write_text(
            qwen_attestation_text,
            encoding="utf-8",
        )
        qwen_postrun_attestation_path.write_text(
            qwen_attestation_text,
            encoding="utf-8",
        )
        image_identity = json.loads(
            json.dumps(FIXED32_AGENT_IMAGE_IDENTITIES[task_id])
        )
        image_identity_sha256 = canonical_json_sha256(image_identity)
        placement = json.loads(json.dumps(FIXED32_AGENT_PLACEMENT))
        placement_sha256 = canonical_json_sha256(placement)
        mounted_runtime_proof = fixture_mounted_runtime_proof()
        mounted_runtime_proof_sha256 = canonical_json_sha256(
            mounted_runtime_proof
        )
        remote_settings_observation = {
            **FIXED32_QWEN_REMOTE_SETTINGS,
            "file_identity_sha256": "1" * 64,
        }
        remote_settings_observation_sha256 = canonical_json_sha256(
            remote_settings_observation
        )
        mounted_runtime_proof_path = (
            task_dir / FIXED32_MOUNTED_RUNTIME_PROOF_FILENAME
        )
        mounted_runtime_proof_path.write_text(
            json.dumps(
                mounted_runtime_proof,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="ascii",
        )
        agent = {
            "elapsed_s": 1.0,
            "exit_code": 0,
            "timed_out": False,
            "offloaded": True,
            "codex_host": "alienware",
            "network_drop": False,
            "stall_killed": False,
            "patch_down_rc": 0,
            "agent_env": "instance_image",
            "instance_image": image_identity["image"],
            "instance_image_identity": image_identity,
            "instance_image_identity_sha256": image_identity_sha256,
            "instance_image_postrun_identity_sha256": image_identity_sha256,
            "instance_image_run_reference": image_identity["repo_digest"],
            "agent_placement": placement,
            "agent_placement_sha256": placement_sha256,
            "agent_postrun_placement_sha256": placement_sha256,
            "qwen_bundle_snapshot": qwen_runtime_attestation[
                "bundle_snapshot"
            ],
            "qwen_remote_settings_observation": (
                remote_settings_observation
            ),
            "qwen_remote_settings_observation_sha256": (
                remote_settings_observation_sha256
            ),
            "qwen_remote_settings_postrun_observation_sha256": (
                remote_settings_observation_sha256
            ),
            "qwen_mounted_runtime_proof": mounted_runtime_proof,
            "qwen_mounted_runtime_proof_sha256": (
                mounted_runtime_proof_sha256
            ),
            "qwen_mounted_runtime_proof_file_sha256": sha256_file(
                mounted_runtime_proof_path
            ),
            "qwen_runtime_attestation": qwen_runtime_attestation,
            "qwen_runtime_attestation_sha256": (
                qwen_runtime_attestation_sha256
            ),
            "qwen_runtime_postrun_attestation_sha256": (
                qwen_runtime_attestation_sha256
            ),
        }
        agent_terminal = {
            "exit_code": 0,
            "timed_out": False,
            "offloaded": True,
            "network_drop": False,
            "stall_killed": False,
            "patch_down_rc": 0,
            "agent_env": "instance_image",
        }
        eval_report = {
            "track": "swe_bench",
            "instance_id": task_id,
            "model_id": "qwen3.6-27b-fp8::fixture",
            "dataset_name": SWE_VERIFIED_DATASET,
            "verdict": "resolved",
            "passed": True,
            "failure_mode": "tests_passed",
            "harness_exit_code": 0,
        }
        eval_dir = task_dir / "eval"
        eval_dir.mkdir()
        eval_path = eval_dir / "eval_report.json"
        eval_path.write_text(
            json.dumps(eval_report, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        trace_request_digests = sorted(
            hashlib.sha256(request_id.encode("utf-8")).hexdigest()
            for request_id in engine_request_ids[task_id]
        )
        task_auth = proxy_task_snapshots[task_id]
        real_task_provenance = {
            "schema": FIXED32_REAL_TASK_PROVENANCE_SCHEMA,
            "instance_id": task_id,
            "task_key_id": task_key_ids[task_id],
            "qwen_code_version": FIXED32_QWEN_CODE_VERSION,
            "qwen_system_settings_sha256": (
                FIXED32_QWEN_SYSTEM_SETTINGS_SHA256
            ),
            "instance_image_identity_sha256": image_identity_sha256,
            "instance_image_id": image_identity["id"],
            "instance_image_repo_digest": image_identity["repo_digest"],
            "agent_placement_sha256": placement_sha256,
            "agent_host_identity": placement["agent_host_identity"],
            "measured_host_identity": placement[
                "measured_host_identity"
            ],
            "qwen_bundle_snapshot": qwen_runtime_attestation[
                "bundle_snapshot"
            ],
            "qwen_remote_settings_file_identity_sha256": "1" * 64,
            "qwen_remote_settings_observation_sha256": (
                remote_settings_observation_sha256
            ),
            "qwen_mounted_runtime_proof_sha256": (
                mounted_runtime_proof_sha256
            ),
            "qwen_mounted_runtime_proof_file_sha256": sha256_file(
                mounted_runtime_proof_path
            ),
            "qwen_runtime_attestation_sha256": (
                qwen_runtime_attestation_sha256
            ),
            "qwen_runtime_attestation_file_sha256": sha256_file(
                qwen_attestation_path
            ),
            "qwen_runtime_postrun_attestation_file_sha256": sha256_file(
                qwen_postrun_attestation_path
            ),
            "trace_path": str(trace_path.resolve()),
            "trace_sha256": hashlib.sha256(raw_trace).hexdigest(),
            "trace_bytes": len(raw_trace),
            "event_count": len(trace_events),
            "assistant_event_count": len(model_trace_events),
            "assistant_output_event_count": len(model_trace_events),
            "qwen_assistant_event_count": len(model_trace_events),
            "codex_agent_message_event_count": 0,
            "positive_token_usage": True,
            "usage_record_count": len(model_trace_events),
            "positive_usage_record_count": len(model_trace_events),
            "usage_max_by_field": {
                "input_tokens": 128,
                "output_tokens": 32,
                "total_tokens": 160,
            },
            "trace_completed_logical_model_requests": len(model_trace_events),
            "trace_model_request_ids_sha256": hashlib.sha256(
                json.dumps(
                    trace_request_digests,
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            **task_auth,
            "agent_terminal": agent_terminal,
        }
        (task_dir / "runner_metadata.json").write_text(
            json.dumps(
                {
                    "instance_id": task_id,
                    "dataset_name": SWE_VERIFIED_DATASET,
                    "started_at": "2026-01-01T00:00:00Z",
                    "ended_at": "2026-01-01T00:00:01Z",
                    "agent": agent,
                    "codex": agent,
                    "eval_report": eval_report,
                    "fixed32_real_task_provenance": real_task_provenance,
                    "fixed32_dataset_record_sha256": (
                        dataset_record_digests[task_id]
                    ),
                    "fixed32_task_boundary": boundary,
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        metrics_pre_path = task_dir / "vllm_metrics_pre.txt"
        metrics_post_path = task_dir / "vllm_metrics_post.txt"
        metrics_pre_path.write_text(
            fixture_metrics(fwd_ms, wall_ms, drafts, start, tokens),
            encoding="utf-8",
        )
        metrics_post_path.write_text(
            fixture_metrics(fwd_ms, wall_ms, drafts, end, tokens),
            encoding="utf-8",
        )

    (arm_dir / "metrics_after_swe.txt").write_text(
        fixture_metrics(fwd_ms, wall_ms, drafts, len(fwd_ms), tokens),
        encoding="utf-8",
    )
    final_generation = len(boundary_points) + 1
    final_ack = flush_ack(final_generation, len(fwd_ms), "final")
    write_runtime_snapshot(final_ack, len(fwd_ms))
    final_request = {
        "schema": FLUSH_REQUEST_SCHEMA,
        "mode": kind,
        "producer_pid": producer_pid,
        "prev_generation": final_generation - 1,
        "generation": final_generation,
        "nonce": final_ack["nonce"],
        "action": "final",
    }
    (logs_dir / "fr13_fixed32_flush_request.json").write_text(
        json.dumps(final_request, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (logs_dir / "fr13_fixed32_flush_ack.json").write_text(
        json.dumps(final_ack, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (arm_dir / "fixed32_final_flush.json").write_text(
        json.dumps(
            {"schema": FLUSH_RESULT_SCHEMA, "ack": final_ack},
            ensure_ascii=True,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (arm_dir / "fixed32_final_flush.stderr").write_text("", encoding="utf-8")
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    sidecar = {
        "schema": SFWD_SAMPLE_SIDECAR_SCHEMA,
        "pid": 100 + int(hydra),
        "final": True,
        "fwd_indices": list(range(len(fwd_ms))),
        "fwd_drafts": drafts,
        "fwd_ms": fwd_ms,
        "fwd_cg": ["FULL"] * len(fwd_ms),
        "fwd_host_ms": [1.0] * len(fwd_ms),
        "fwd_exec_ms": [1.0] * len(fwd_ms),
        "fwd_cpu_tail_ms": [1.0] * len(fwd_ms),
        "wall_drafts": drafts,
        "wall_ms": wall_ms,
        "wall_fwd_indices": list(range(len(wall_ms))),
        "samples_capped": False,
    }
    (sidecar_dir / f"{arm}.json.samples.{100 + int(hydra)}").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    main_sidecar = {
        "schema": SFWD_MAIN_SIDECAR_SCHEMA,
        "pid": 100 + int(hydra),
        "final": True,
        "decode_forward_gpu_seconds": sum(fwd_ms) / 1000.0,
        "n_pure_decode_steps_timed": len(fwd_ms),
        "n_forward_starts": len(fwd_ms),
        "n_forward_dropped": 0,
        "n_forward_pending": 0,
        "n_drafts_in_timed_steps": sum(drafts),
        "decode_step_wall_seconds": sum(wall_ms) / 1000.0,
        "n_drafts_in_wall_steps": sum(drafts),
        "n_wall_steps": len(wall_ms),
        "n_wall_rejected": 0,
        "n_wall_chain_resets": 0,
        "n_wall_request_set_resets": 0,
        "n_wall_invalid_request_ids": 0,
        "n_wall_bookkeeping_errors": 0,
        "wall_chain_open": False,
        "wall_open_fwd_index": None,
        "wall_open_drafts": 0,
        "wall_cap_s": 1.5,
        "metric_name": "vllm:fr13_decode_forward_gpu_seconds_total",
        "note": "fixed32 self-test fixture",
    }
    (sidecar_dir / f"{arm}.json.{100 + int(hydra)}").write_text(
        json.dumps(main_sidecar), encoding="utf-8"
    )
    census_records.append(
        work_census_terminal_fixture(
            census_records,
            fixture_synthetic_runtime_proof=True,
        )
    )
    census_path.write_text(
        "\n".join(
            json.dumps(
                record,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            for record in census_records
        )
        + "\n",
        encoding="utf-8",
    )
    chat_traffic_audit = build_fixed32_chat_traffic_audit(
        arm_dir.resolve(),
        mode=kind,
        subset=validate_subset(subset, len(task_ids)),
        dataset_record_digests=dataset_record_digests,
    )
    (arm_dir / "fixed32_chat_traffic_audit.json").write_text(
        json.dumps(
            chat_traffic_audit,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )


def write_fixture_campaign(
    repo: Path, base: Path, *, concurrency: int
) -> tuple[Path, Path, str]:
    tag = f"fixture_b{concurrency}"
    runroot = base / f"campaign_b{concurrency}"
    sidecar_dir = base / f"sidecars_b{concurrency}"
    runroot.mkdir()
    runtime_manifest = build_runtime_manifest(
        repo,
        profile=RUNTIME_MANIFEST_PROFILE,
        sequence=RUNTIME_MANIFEST_SEQUENCE,
    )
    rendered_manifest = json.dumps(
        runtime_manifest,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    for name in ("runtime_manifest.at_launch.json", "runtime_manifest.at_end.json"):
        (runroot / name).write_text(rendered_manifest + "\n", encoding="utf-8")
    external_manifest = json.dumps(
        fixture_external_manifest(),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    for name in ("external_manifest.at_launch.json", "external_manifest.at_end.json"):
        (runroot / name).write_text(external_manifest + "\n", encoding="utf-8")
    write_fixture_arm(
        repo,
        runroot,
        sidecar_dir,
        tag,
        hydra=False,
        concurrency=concurrency,
    )
    write_fixture_arm(
        repo,
        runroot,
        sidecar_dir,
        tag,
        hydra=True,
        concurrency=concurrency,
    )
    return runroot, sidecar_dir, tag


def expect_gate_error(callable_obj: Any, needle: str) -> None:
    try:
        callable_obj()
    except GateError as error:
        if needle not in str(error):
            raise AssertionError(
                f"expected error containing {needle!r}, got {error!r}"
            ) from error
    else:
        raise AssertionError(f"expected GateError containing {needle!r}")


def self_test(repo: Path) -> None:
    for task_count in (4, 16):
        validate_subset(
            (repo / EVIDENCE_SETS[task_count]["relative_path"]).resolve(),
            task_count,
        )
    assert cluster_summary([float(value) for value in range(16)])["df"] == 15
    pairing_arrays = {
        "fwd_ms": np.ones(100, dtype=np.float64),
        "fwd_full": np.ones(100, dtype=np.bool_),
        "fwd_drafts": np.ones(100, dtype=np.float64),
        "wall_ms": np.ones(100, dtype=np.float64),
        "wall_drafts": np.ones(100, dtype=np.float64),
        "wall_fwd_indices": np.arange(100, dtype=np.int64),
    }
    expect_gate_error(
        lambda: paired_wall_selection(
            {
                **pairing_arrays,
                "wall_ms": pairing_arrays["wall_ms"][:98],
                "wall_drafts": pairing_arrays["wall_drafts"][:98],
                "wall_fwd_indices": pairing_arrays["wall_fwd_indices"][:98],
            },
            fwd_span=(0, 100),
            wall_span=(0, 98),
            label="retention adversary",
        ),
        "retained wall fraction",
    )
    crossing_indices = np.arange(100, dtype=np.int64)
    crossing_indices[20] = 19
    expect_gate_error(
        lambda: paired_wall_selection(
            {**pairing_arrays, "wall_fwd_indices": crossing_indices},
            fwd_span=(20, 100),
            wall_span=(20, 100),
            label="B=4 boundary-crossing adversary",
        ),
        "outside its forward span",
    )
    mismatched_occupancy = pairing_arrays["wall_drafts"].copy()
    mismatched_occupancy[0] = 2.0
    expect_gate_error(
        lambda: paired_wall_selection(
            {**pairing_arrays, "wall_drafts": mismatched_occupancy},
            fwd_span=(0, 100),
            wall_span=(0, 100),
            label="occupancy adversary",
        ),
        "occupancy differs",
    )
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        pinned_parquet = repo / PINNED_SWE_VERIFIED_PARQUET_RELATIVE
        corrupted_parquet = base / "corrupted-swe-verified.parquet"
        corrupted_parquet.write_bytes(pinned_parquet.read_bytes() + b"\x00")
        expect_gate_error(
            lambda: validate_pinned_swe_verified_parquet(corrupted_parquet),
            "pinned SWE-Verified Parquet SHA-256 mismatch",
        )
        b1_root, b1_sidecars, b1_tag = write_fixture_campaign(repo, base, concurrency=1)
        b1 = reduce_campaign(
            repo,
            b1_root,
            b1_tag,
            4,
            1,
            b1_sidecars,
            BOOTSTRAP_REPS,
            BOOTSTRAP_SEED,
        )
        assert b1["analysis_valid"]
        assert b1["schema"] == "fr13.canonical_swe_verified_fixed32_floor_gate.v11"
        assert b1["inferred_concurrency"] == 1
        assert b1["gates"]["fixed32_flush_generation_chain_exact"]
        assert b1["gates"]["fixed32_task_boundaries_exact"]
        assert b1["gates"]["external_artifact_fingerprint_equal"]
        assert b1["gates"]["arm_runtime_attestations_equal"]
        assert b1["gates"]["running_container_image_identity_exact"]
        assert b1["gates"]["all_canonical_tasks_have_real_model_traffic"]
        assert b1["gates"]["all_task_agents_completed_cleanly"]
        assert b1["gates"]["all_tasks_have_terminal_swe_verdicts"]
        assert b1["gates"]["task_metric_bracket_bytes_bound"]
        assert b1["gates"]["fixed32_pretask_zero_positive_traffic"]
        assert b1["gates"]["all_validated_chat_task_traffic_bound"]
        assert b1["gates"]["fixed32_ingress_proxy_engine_exact"]
        assert b1["gates"]["fixed32_zero_campaign_rejections"]
        assert b1["gates"]["fixed32_raw_proxy_dumps_disabled"]
        assert b1["gates"]["canonical_exact_4_or_16_task_binding"]
        assert b1["gates"]["canonical_task_forward_union_covers_complete_stream"]
        assert b1["gates"]["fixed32_per_batch_physical_work_equal"]
        assert b1["gates"]["fixed32_drafter_graph_lifecycle_exact_and_matched"]
        assert b1["gates"]["fixed32_forward_graph_pregather_exact"]
        assert b1["gates"]["fixed32_scope_limitations_explicit"]
        assert b1["external_artifact_fingerprint"]["byte_equal"]
        assert b1["matched_runtime_attestation"]["byte_equal"]
        assert b1["source_runtime_fingerprint"]["file_count"] == 62
        assert b1["source_runtime_fingerprint"]["python_package_file_count"] == 25
        tail_metric_brackets = b1["arms"]["tail6_fixed32"]["provenance"][
            "task_metric_brackets"
        ]
        assert set(tail_metric_brackets) == set(EVIDENCE_SETS[4]["task_ids"])
        for task_bracket in tail_metric_brackets.values():
            for artifact in task_bracket.values():
                artifact_bytes = Path(artifact["path"]).read_bytes()
                assert artifact["bytes"] == len(artifact_bytes)
                assert artifact["sha256"] == hashlib.sha256(
                    artifact_bytes
                ).hexdigest()
        tail_real_tasks = b1["arms"]["tail6_fixed32"]["provenance"][
            "real_tasks"
        ]["tasks"]
        for task_record in tail_real_tasks.values():
            trace_record = task_record["trace"]
            trace_bytes = Path(trace_record["path"]).read_bytes()
            assert trace_record["bytes"] == len(trace_bytes)
            assert trace_record["sha256"] == hashlib.sha256(
                trace_bytes
            ).hexdigest()
            eval_record = task_record["terminal"]["eval_artifact"]
            eval_bytes = Path(eval_record["path"]).read_bytes()
            assert eval_record["bytes"] == len(eval_bytes)
            assert eval_record["sha256"] == hashlib.sha256(
                eval_bytes
            ).hexdigest()
        chat_audit_record = b1["arms"]["tail6_fixed32"]["provenance"][
            "real_tasks"
        ]["chat_traffic_audit"]
        chat_audit_bytes = Path(chat_audit_record["path"]).read_bytes()
        assert chat_audit_record["bytes"] == len(chat_audit_bytes)
        assert chat_audit_record["sha256"] == hashlib.sha256(
            chat_audit_bytes
        ).hexdigest()
        b1_census_expected = b1["arms"]["tail6_fixed32"]["work_census_expected"]
        assert b1_census_expected["canonical_task_selection"]["counter_intervals"] == [
            [0, 352]
        ]
        assert b1_census_expected["canonical_task_selection"]["event_count"] == 352
        assert b1_census_expected["complete_stream"]["event_count"] == 352
        b1_work_census = b1["fixed32_work_census"]
        assert b1_work_census["report"]["schema"] == WORK_CENSUS_REPORT_SCHEMA
        assert b1_work_census["physical_work_comparison"][
            "observed_batch_sizes"
        ] == [1]
        assert (
            b1_work_census["physical_work_comparison"]["event_counts_compared"]
            is False
        )
        assert b1_work_census["drafter_graph_lifecycle"][
            "registry_batch_sizes"
        ] == [1]
        assert b1_work_census["forward_graph_pregather_lifecycle"][
            "registry_batch_sizes"
        ] == [1]
        assert b1_work_census["forward_graph_pregather_lifecycle"][
            "graph_signatures_equal_across_arms_per_batch"
        ]
        assert len(b1_work_census["scope"]["data_dependent_unproven"]) == 6
        for census_artifact in b1_work_census["files"].values():
            census_bytes = Path(census_artifact["path"]).read_bytes()
            assert census_artifact["bytes"] == len(census_bytes)
            assert census_artifact["sha256"] == hashlib.sha256(
                census_bytes
            ).hexdigest()

        bad_physical_report = json.loads(
            json.dumps(b1_work_census["report"])
        )
        bad_physical_entry = bad_physical_report[
            "physical_work_histograms"
        ]["tail6_fixed32"]["1"]
        bad_physical_entry["normalized_event_signatures"] = {
            "0" * 64: bad_physical_entry["event_count"]
        }
        expect_gate_error(
            lambda: validate_work_census_v5_report(
                bad_physical_report,
                required_batch=1,
            ),
            "normalized physical-work SHA differs",
        )

        bad_lifecycle_report = json.loads(
            json.dumps(b1_work_census["report"])
        )
        bad_registry_row = bad_lifecycle_report[
            "drafter_graph_registries"
        ]["tail6_fixed32"][0]
        bad_terminal_row = bad_lifecycle_report["terminal_summaries"][
            "tail6_fixed32"
        ]["drafter_graph_registry"][0]
        new_origin = (
            "unmeasured"
            if bad_registry_row["capture_origin"] == "measured"
            else "measured"
        )
        bad_registry_row["capture_origin"] = new_origin
        bad_terminal_row["capture_origin"] = new_origin
        expect_gate_error(
            lambda: validate_work_census_v5_report(
                bad_lifecycle_report,
                required_batch=1,
            ),
            "drafter graph lifecycle differs",
        )

        bad_scope_report = json.loads(json.dumps(b1_work_census["report"]))
        bad_scope_report["scope"]["data_dependent_unproven"].pop()
        expect_gate_error(
            lambda: validate_work_census_v5_report(
                bad_scope_report,
                required_batch=1,
            ),
            "v5 report contract mismatch",
        )

        bad_forward_report = json.loads(json.dumps(b1_work_census["report"]))
        for parent in (
            bad_forward_report["forward_graph_registries"]["tail6_fixed32"],
            bad_forward_report["terminal_summaries"]["tail6_fixed32"][
                "forward_graph_registry"
            ],
        ):
            parent[0]["measured_replays"] += 1
        expect_gate_error(
            lambda: validate_work_census_v5_report(
                bad_forward_report,
                required_batch=1,
            ),
            "does not prove one ordered final-FULL pregather capture",
        )

        bad_auxiliary_report = json.loads(
            json.dumps(b1_work_census["report"])
        )
        for parent in (
            bad_auxiliary_report["conv_pregather_auxiliary"][
                "tail6_fixed32"
            ],
            bad_auxiliary_report["terminal_summaries"]["tail6_fixed32"][
                "conv_pregather_auxiliary"
            ],
        ):
            parent["profile_capture_stages"] = 1
        expect_gate_error(
            lambda: validate_work_census_v5_report(
                bad_auxiliary_report,
                required_batch=1,
            ),
            "pregather auxiliary/host stage counts are not zero",
        )
        tail_b1 = b1["arms"]["tail6_fixed32"]["statistics"]
        equal = tail_b1["task_cluster_equal_weight"]["wall_ms_per_step"]
        assert equal["cluster_count"] == 4 and equal["df"] == 3
        assert math.isclose(equal["point_estimate"], 106.0)
        assert not math.isclose(
            equal["point_estimate"],
            tail_b1["step_weighted_counter_point"]["wall_ms_per_step"],
        )
        assert all(
            row["counter_reconciliation"]["wall"]["exact_drafts_and_steps"]
            for row in tail_b1["sidecar_coverage_by_task"]
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS - 1,
                BOOTSTRAP_SEED,
            ),
            "formal moving-block parameters are pinned",
        )

        tail_arm = b1_root / f"tail6_fixed32_{b1_tag}"
        first_task_dir = (
            tail_arm / "swe_out" / "verified" / "per_task" / CANONICAL_TASK_IDS[0]
        )
        boundary_path = first_task_dir / "fixed32_task_boundary.json"
        metadata_path = first_task_dir / "runner_metadata.json"
        good_boundary_bytes = boundary_path.read_bytes()
        good_metadata_bytes = metadata_path.read_bytes()
        bad_boundary = json.loads(good_boundary_bytes)
        bad_boundary["post"]["generation"] += 100
        bad_metadata = json.loads(good_metadata_bytes)
        bad_metadata["fixed32_task_boundary"] = bad_boundary
        boundary_path.write_text(
            json.dumps(bad_boundary, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metadata_path.write_text(
            json.dumps(bad_metadata, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "missing required artifact",
        )
        boundary_path.write_bytes(good_boundary_bytes)
        metadata_path.write_bytes(good_metadata_bytes)

        final_result_path = tail_arm / "fixed32_final_flush.json"
        final_ack_path = tail_arm / "logs" / "fr13_fixed32_flush_ack.json"
        good_final_result_bytes = final_result_path.read_bytes()
        good_final_ack_bytes = final_ack_path.read_bytes()
        bad_final_result = json.loads(good_final_result_bytes)
        bad_final_ack = bad_final_result["ack"]
        bad_final_ack["counters"]["pure_decode_forward_steps"] += 1
        bad_final_ack["counters"]["complete_work_census_events"] += 1
        final_result_path.write_text(
            json.dumps(bad_final_result, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        final_ack_path.write_text(
            json.dumps(bad_final_ack, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "runtime snapshot does not bind to flush ack",
        )
        final_result_path.write_bytes(good_final_result_bytes)
        final_ack_path.write_bytes(good_final_ack_bytes)

        final_result = json.loads(good_final_result_bytes)
        final_ack = final_result["ack"]
        final_runtime_snapshot_path = (
            tail_arm
            / "logs"
            / (
                "fr13_fixed32_boundary_snapshot."
                f"{final_ack['generation']}.json"
            )
        )
        good_runtime_snapshot = json.loads(
            final_runtime_snapshot_path.read_bytes()
        )
        pregather_tamper_path = base / "pregather_boundary_tamper.json"

        def expect_pregather_boundary_failure(
            label: str,
            mutate: Any,
            needle: str = (
                "committer/nonpure/in-graph pregather counters do not reconcile"
            ),
        ) -> None:
            tampered = json.loads(json.dumps(good_runtime_snapshot))
            mutate(tampered["metrics"]["conv_pregather"])
            pregather_tamper_path.write_text(
                json.dumps(tampered, ensure_ascii=True, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            expect_gate_error(
                lambda: validate_runtime_boundary_snapshot(
                    pregather_tamper_path,
                    ack=final_ack,
                    server_capacity=1,
                    metrics_path=None,
                    metric_values=None,
                    reference=None,
                    census_path=(
                        tail_arm
                        / "logs"
                        / "fr13_fixed32_work_census.jsonl"
                    ),
                ),
                needle,
            )

        pregather_boundary_tampers = (
            (
                "capture-scalar",
                lambda counters: counters.__setitem__(
                    "graph_capture_stages",
                    counters["graph_capture_stages"] + 1,
                ),
            ),
            (
                "capture-histogram",
                lambda counters: counters[
                    "graph_capture_stages_by_batch"
                ].__setitem__("1", 0),
            ),
            (
                "profile-stage",
                lambda counters: counters.__setitem__(
                    "profile_capture_stages", 1
                ),
            ),
            (
                "aux-stage",
                lambda counters: counters.__setitem__(
                    "aux_capture_stages", 1
                ),
            ),
            (
                "host-stage-scalar",
                lambda counters: counters.__setitem__("actual_stages", 1),
            ),
            (
                "host-stage-histogram",
                lambda counters: counters[
                    "actual_stages_by_batch"
                ].__setitem__("1", 1),
            ),
            (
                "replay-scalar",
                lambda counters: counters.__setitem__(
                    "graph_replay_stages",
                    counters["graph_replay_stages"] + 1,
                ),
            ),
            (
                "replay-histogram",
                lambda counters: counters[
                    "graph_replay_stages_by_batch"
                ].__setitem__(
                    "1",
                    counters["graph_replay_stages_by_batch"]["1"] + 1,
                ),
            ),
        )
        for label, mutate in pregather_boundary_tampers:
            expect_pregather_boundary_failure(label, mutate)
        expect_pregather_boundary_failure(
            "legacy-stage-key",
            lambda counters: counters.__setitem__("stage_launches", 0),
            "conv_pregather: keys mismatch",
        )

        tail_metric_paths = sorted(
            (tail_arm / "swe_out" / "verified" / "per_task").glob(
                "*/vllm_metrics_*.txt"
            )
        )
        original_metric_bytes = {path: path.read_bytes() for path in tail_metric_paths}
        runtime_snapshot_paths = sorted(
            (tail_arm / "logs").glob(
                "fr13_fixed32_boundary_snapshot.*.json"
            )
        )
        original_runtime_snapshot_bytes = {
            path: path.read_bytes() for path in runtime_snapshot_paths
        }
        for snapshot_path in runtime_snapshot_paths:
            runtime_snapshot = json.loads(snapshot_path.read_bytes())
            runtime_snapshot["metrics"]["sfwd"]["wall_rejected"] += 5
            snapshot_path.write_text(
                json.dumps(
                    runtime_snapshot,
                    ensure_ascii=True,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        boundary_paths = sorted(
            (tail_arm / "swe_out" / "verified" / "per_task").glob(
                "*/fixed32_task_boundary.json"
            )
        )
        original_boundary_bytes = {
            path: path.read_bytes() for path in boundary_paths
        }
        metadata_paths = [path.with_name("runner_metadata.json") for path in boundary_paths]
        original_metadata_bytes = {
            path: path.read_bytes() for path in metadata_paths
        }
        chat_audit_path = tail_arm / "fixed32_chat_traffic_audit.json"
        original_chat_audit_bytes = chat_audit_path.read_bytes()
        for boundary_path in boundary_paths:
            boundary = json.loads(boundary_path.read_bytes())
            for snapshot in ("pre", "post"):
                generation = boundary[snapshot]["generation"]
                snapshot_path = (
                    tail_arm
                    / "logs"
                    / f"fr13_fixed32_boundary_snapshot.{generation}.json"
                )
                boundary[f"{snapshot}_runtime_snapshot"]["sha256"] = (
                    sha256_file(snapshot_path)
                )
            boundary_path.write_text(
                json.dumps(boundary, ensure_ascii=True, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            metadata_path = boundary_path.with_name("runner_metadata.json")
            metadata = json.loads(metadata_path.read_bytes())
            metadata["fixed32_task_boundary"] = boundary
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=True, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        for metric_path in tail_metric_paths:
            values, _ = metric_snapshot(metric_path)
            metric_path.write_text(
                replace_metric_values(
                    metric_path.read_text(encoding="utf-8"),
                    {
                        "wall_attempts": values["wall_attempts"] + 5,
                        "wall_rejected": values["wall_rejected"] + 5,
                    },
                ),
                encoding="utf-8",
            )
        tail_main_path = next(
            path
            for path in b1_sidecars.iterdir()
            if path.name.startswith(f"tail6_fixed32_{b1_tag}.json.")
            and ".json.samples." not in path.name
        )
        original_main_bytes = tail_main_path.read_bytes()
        warmup_main = json.loads(original_main_bytes)
        warmup_main["n_wall_rejected"] += 5
        tail_main_path.write_text(json.dumps(warmup_main), encoding="utf-8")
        warmup_audit = json.loads(original_chat_audit_bytes)
        for task_id, task_audit in warmup_audit["tasks"].items():
            task_dir = (
                tail_arm / "swe_out" / "verified" / "per_task" / task_id
            )
            boundary_path = task_dir / "fixed32_task_boundary.json"
            task_audit["boundary"]["sha256"] = sha256_file(boundary_path)
            task_audit["boundary"]["bytes"] = boundary_path.stat().st_size
        chat_audit_path.write_text(
            json.dumps(
                warmup_audit,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "formal timer integrity counters are nonzero",
        )
        for metric_path, payload in original_metric_bytes.items():
            metric_path.write_bytes(payload)
        for snapshot_path, payload in original_runtime_snapshot_bytes.items():
            snapshot_path.write_bytes(payload)
        for boundary_path, payload in original_boundary_bytes.items():
            boundary_path.write_bytes(payload)
        for metadata_path, payload in original_metadata_bytes.items():
            metadata_path.write_bytes(payload)
        tail_main_path.write_bytes(original_main_bytes)
        chat_audit_path.write_bytes(original_chat_audit_bytes)

        first_task = (
            tail_arm / "swe_out" / "verified" / "per_task" / CANONICAL_TASK_IDS[0]
        )
        pre_path = first_task / "vllm_metrics_pre.txt"
        pre_values, _ = metric_snapshot(pre_path)
        good_pre = pre_path.read_text(encoding="utf-8")
        post_path = first_task / "vllm_metrics_post.txt"
        good_post = post_path.read_text(encoding="utf-8")

        pre_path.write_text(
            replace_metric_values(
                good_pre,
                {"wall_attempts": pre_values["wall_attempts"] + 1},
            ),
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "pre wall attempts != retained + rejected",
        )
        pre_path.write_text(good_pre, encoding="utf-8")
        wall_line = next(
            line
            for line in good_post.splitlines()
            if line.startswith(f"{METRICS['wall_s']} ")
        )
        zero_wall_line = f"{METRICS['wall_s']} {pre_values['wall_s']}"
        post_path.write_text(
            good_post.replace(wall_line, zero_wall_line, 1),
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "non-positive wall_s delta",
        )
        post_path.write_text(good_post, encoding="utf-8")

        post_values, _ = metric_snapshot(post_path)
        post_path.write_text(
            replace_metric_values(
                good_post,
                {
                    "wall_attempts": post_values["wall_attempts"] + 1,
                    "wall_rejected": post_values["wall_rejected"] + 1,
                },
            ),
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "censored wall intervals in task window",
        )
        post_path.write_text(good_post, encoding="utf-8")

        post_path.write_text(
            replace_metric_values(
                good_post,
                {"wall_attempts": post_values["wall_attempts"] + 1},
            ),
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "wall attempts != retained + rejected",
        )
        post_path.write_text(good_post, encoding="utf-8")

        one_step_post = {
            "fwd_s": pre_values["fwd_s"] + 0.070,
            "fwd_steps": pre_values["fwd_steps"] + 1,
            "fwd_drafts": pre_values["fwd_drafts"] + 1,
            "wall_s": pre_values["wall_s"] + 0.100,
            "wall_drafts": pre_values["wall_drafts"] + 1,
            "wall_steps": pre_values["wall_steps"] + 1,
            "wall_attempts": pre_values["wall_attempts"] + 1,
            "wall_rejected": pre_values["wall_rejected"],
            "spec_drafts": pre_values["spec_drafts"] + 1,
            "spec_tokens": pre_values["spec_tokens"] + PHYSICAL_DRAFTS,
        }
        post_path.write_text(
            replace_metric_values(good_post, one_step_post),
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "exposure below 64 retained steps",
        )
        post_path.write_text(good_post, encoding="utf-8")

        post_values, _ = metric_snapshot(post_path)
        fwd_drafts_line = next(
            line
            for line in good_post.splitlines()
            if line.startswith(f"{METRICS['fwd_drafts']} ")
        )
        post_path.write_text(
            good_post.replace(
                fwd_drafts_line,
                f"{METRICS['fwd_drafts']} {post_values['fwd_drafts'] + 1}",
                1,
            ),
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "fwd drafts/step is outside [1, 1]",
        )
        post_path.write_text(good_post, encoding="utf-8")

        post_path.write_text(
            good_post.replace(
                'model_name="qwen3.6-27b"',
                'model_name="wrong-series"',
                1,
            ),
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "pre/post required metric labels differ",
        )
        post_path.write_text(good_post, encoding="utf-8")

        wrong_series_pre = good_pre.replace(
            'model_name="qwen3.6-27b"',
            'model_name="wrong-series"',
        )
        wrong_series_post = good_post.replace(
            'model_name="qwen3.6-27b"',
            'model_name="wrong-series"',
        )
        pre_path.write_text(wrong_series_pre, encoding="utf-8")
        post_path.write_text(wrong_series_post, encoding="utf-8")
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "pinned qwen3.6-27b series",
        )
        pre_path.write_text(good_pre, encoding="utf-8")
        post_path.write_text(good_post, encoding="utf-8")

        tail_sidecar = next(
            path
            for path in b1_sidecars.iterdir()
            if path.name.startswith(f"tail6_fixed32_{b1_tag}.json.samples.")
        )
        good_sidecar = tail_sidecar.read_bytes()
        sidecar_payload = json.loads(good_sidecar)
        sidecar_payload["final"] = False
        tail_sidecar.write_text(json.dumps(sidecar_payload), encoding="utf-8")
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "per-step sidecar lacks an explicit final flush",
        )
        tail_sidecar.write_bytes(good_sidecar)

        sidecar_payload = json.loads(good_sidecar)
        sidecar_payload["wall_ms"][5] += 1.0
        tail_sidecar.write_text(json.dumps(sidecar_payload), encoding="utf-8")
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "sidecar/counter timing mismatch",
        )
        tail_sidecar.write_bytes(good_sidecar)

        sidecar_payload = json.loads(good_sidecar)
        sidecar_payload["fwd_indices"][5] = sidecar_payload["fwd_indices"][4]
        tail_sidecar.write_text(json.dumps(sidecar_payload), encoding="utf-8")
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "fwd_indices are not strictly increasing and unique",
        )
        tail_sidecar.write_bytes(good_sidecar)

        sidecar_payload = json.loads(good_sidecar)
        sidecar_payload["wall_fwd_indices"][5] = (
            sidecar_payload["wall_fwd_indices"][4]
        )
        tail_sidecar.write_text(json.dumps(sidecar_payload), encoding="utf-8")
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "wall_fwd_indices are not strictly increasing and unique",
        )
        tail_sidecar.write_bytes(good_sidecar)

        main_sidecar = next(
            path
            for path in b1_sidecars.iterdir()
            if path.name.startswith(f"tail6_fixed32_{b1_tag}.json.")
            and ".json.samples." not in path.name
        )
        good_main_sidecar = main_sidecar.read_bytes()
        main_payload = json.loads(good_main_sidecar)
        main_payload["n_wall_rejected"] = 1_000_000
        main_sidecar.write_text(json.dumps(main_payload), encoding="utf-8")
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "formal timer integrity counters are nonzero",
        )
        main_sidecar.write_bytes(good_main_sidecar)

        main_payload = json.loads(good_main_sidecar)
        main_payload["n_wall_invalid_request_ids"] = 1
        main_sidecar.write_text(json.dumps(main_payload), encoding="utf-8")
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "formal timer integrity counters are nonzero",
        )
        main_sidecar.write_bytes(good_main_sidecar)

        main_payload = json.loads(good_main_sidecar)
        main_payload["n_wall_request_set_resets"] = 1
        main_sidecar.write_text(json.dumps(main_payload), encoding="utf-8")
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "request-set resets exceed all chain resets",
        )
        main_sidecar.write_bytes(good_main_sidecar)

        sidecar_payload = json.loads(good_sidecar)
        for key in (
            "fwd_indices",
            "fwd_drafts",
            "fwd_ms",
            "fwd_cg",
            "fwd_host_ms",
            "fwd_exec_ms",
            "fwd_cpu_tail_ms",
            "wall_drafts",
            "wall_ms",
            "wall_fwd_indices",
        ):
            sidecar_payload[key].pop()
        tail_sidecar.write_text(json.dumps(sidecar_payload), encoding="utf-8")
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "final sample lengths do not match main counters",
        )
        tail_sidecar.write_bytes(good_sidecar)

        census_path = tail_arm / "logs" / "fr13_fixed32_work_census.jsonl"
        good_census = census_path.read_bytes()
        census_lines = good_census.decode("utf-8").splitlines()
        census_path.write_text(
            "\n".join(census_lines[:-1]) + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "census stream is shorter than snapshot prefix",
        )
        census_path.write_bytes(good_census)

        census_records = [
            json.loads(line) for line in good_census.decode("utf-8").splitlines()
        ]
        census_records[0]["gdn"]["padded_slots"] -= 1
        census_path.write_text(
            "\n".join(
                json.dumps(record, ensure_ascii=True, sort_keys=True)
                for record in census_records
            )
            + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "census prefix digest mismatch",
        )
        census_path.write_bytes(good_census)

        launch_manifest = b1_root / "runtime_manifest.at_launch.json"
        good_manifest = launch_manifest.read_bytes()
        tampered_manifest = json.loads(good_manifest)
        tampered_manifest["closures"]["host_script_source"][0]["sha256"] = "0" * 64
        launch_manifest.write_text(
            json.dumps(tampered_manifest, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "runtime manifest canonical digest mismatch",
        )
        launch_manifest.write_bytes(good_manifest)

        def rerun_b1_fixture() -> dict[str, Any]:
            return reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            )

        task_root = tail_arm / "swe_out" / "verified" / "per_task"
        swap_task_dirs = [
            task_root / task_id for task_id in CANONICAL_TASK_IDS[:2]
        ]
        swap_trace_paths = [
            task_dir / "qwen_trace.jsonl" for task_dir in swap_task_dirs
        ]
        swap_metadata_paths = [
            task_dir / "runner_metadata.json" for task_dir in swap_task_dirs
        ]
        original_swap_artifacts = {
            path: path.read_bytes()
            for path in (*swap_trace_paths, *swap_metadata_paths)
        }

        def write_rebound_trace(
            trace_path: Path,
            metadata_path: Path,
            events: list[dict[str, Any]],
        ) -> None:
            trace_text = (
                "\n".join(
                    json.dumps(
                        event,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    for event in events
                )
                + "\n"
            )
            trace_path.write_text(trace_text, encoding="utf-8")
            response_ids = [
                event["message"]["id"]
                for event in events
                if event.get("type") == "assistant"
                and isinstance(event.get("message"), dict)
                and event["message"].get("stop_reason") is not None
            ]
            response_digests = sorted(
                hashlib.sha256(response_id.encode("utf-8")).hexdigest()
                for response_id in response_ids
            )
            metadata = json.loads(metadata_path.read_bytes())
            provenance = metadata["fixed32_real_task_provenance"]
            raw_trace = trace_text.encode("utf-8")
            provenance["trace_sha256"] = hashlib.sha256(raw_trace).hexdigest()
            provenance["trace_bytes"] = len(raw_trace)
            provenance["event_count"] = len(events)
            provenance["trace_completed_logical_model_requests"] = len(
                response_ids
            )
            provenance["trace_model_request_ids_sha256"] = hashlib.sha256(
                json.dumps(
                    response_digests,
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=True, sort_keys=True),
                encoding="utf-8",
            )

        swap_events = [
            [json.loads(line) for line in path.read_text().splitlines()]
            for path in swap_trace_paths
        ]
        swap_assistant_events = [
            next(
                event
                for event in events
                if event.get("type") == "assistant"
            )
            for events in swap_events
        ]
        (
            swap_assistant_events[0]["message"]["id"],
            swap_assistant_events[1]["message"]["id"],
        ) = (
            swap_assistant_events[1]["message"]["id"],
            swap_assistant_events[0]["message"]["id"],
        )
        try:
            for trace_path, metadata_path, events in zip(
                swap_trace_paths,
                swap_metadata_paths,
                swap_events,
                strict=True,
            ):
                write_rebound_trace(trace_path, metadata_path, events)
            expect_gate_error(
                rerun_b1_fixture,
                "task successful request evidence differs from terminal trace",
            )
        finally:
            for path, raw in original_swap_artifacts.items():
                path.write_bytes(raw)

        replay_events = [
            json.loads(line)
            for line in swap_trace_paths[0].read_text().splitlines()
        ]
        replay_events.append(
            json.loads(
                json.dumps(
                    next(
                        event
                        for event in replay_events
                        if event.get("type") == "assistant"
                    )
                )
            )
        )
        try:
            write_rebound_trace(
                swap_trace_paths[0],
                swap_metadata_paths[0],
                replay_events,
            )
            expect_gate_error(
                rerun_b1_fixture,
                "legacy terminal response IDs are empty or duplicated",
            )
        finally:
            for path, raw in original_swap_artifacts.items():
                path.write_bytes(raw)

        def rewrite_ingress_rows(
            path: Path,
            rows: list[dict[str, Any]],
        ) -> None:
            previous = "0" * 64
            for sequence, row in enumerate(rows):
                row["seq"] = sequence
                row["prev_sha256"] = previous
                unsigned = dict(row)
                unsigned.pop("record_sha256", None)
                row["record_sha256"] = canonical_json_sha256(unsigned)
                previous = row["record_sha256"]
            path.write_text(
                "\n".join(
                    json.dumps(
                        row,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    for row in rows
                )
                + "\n",
                encoding="ascii",
            )

        engine_ledger_path = (
            tail_arm / "logs" / "fr13_fixed32_engine_ingress.jsonl"
        )
        engine_finalize_path = tail_arm / "fixed32_engine_ingress_finalize.json"
        original_direct_artifacts = {
            path: path.read_bytes()
            for path in (engine_ledger_path, engine_finalize_path)
        }
        direct_rows = [
            json.loads(line)
            for line in engine_ledger_path.read_text().splitlines()
        ]
        direct_rows.insert(
            -1,
            {
                "schema": FIXED32_INGRESS_LEDGER_SCHEMA,
                "seq": 0,
                "role": "engine",
                "phase": "campaign",
                "event": "request_rejected",
                "route": "chat",
                "task_key_id": None,
                "logical_id_sha256": None,
                "wire_id_sha256": None,
                "engine_request_id_sha256": None,
                "status_code": None,
                "outcome": "rejected",
                "reason": "invalid_engine_bearer",
                "evidence_sha256": None,
                "prev_sha256": "",
                "record_sha256": "",
            },
        )
        try:
            rewrite_ingress_rows(engine_ledger_path, direct_rows)
            direct_finalize = json.loads(engine_finalize_path.read_bytes())
            direct_finalize["campaign_rejected_requests"] = 1
            direct_finalize["ledger_records"] = len(direct_rows)
            direct_finalize["ledger_chain_head_sha256"] = direct_rows[-1][
                "record_sha256"
            ]
            engine_finalize_path.write_text(
                json.dumps(
                    direct_finalize,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n",
                encoding="ascii",
            )
            expect_gate_error(
                rerun_b1_fixture,
                "engine ingress rejected campaign inference traffic",
            )
        finally:
            for path, raw in original_direct_artifacts.items():
                path.write_bytes(raw)

        proxy_ledger_path = (
            tail_arm / "logs" / "fr13_fixed32_proxy_ingress.jsonl"
        )
        proxy_finalize_path = tail_arm / "fixed32_proxy_ingress_finalize.json"
        accepted_metadata_path = (
            task_root / CANONICAL_TASK_IDS[3] / "runner_metadata.json"
        )
        original_accepted_artifacts = {
            path: path.read_bytes()
            for path in (
                proxy_ledger_path,
                engine_ledger_path,
                proxy_finalize_path,
                engine_finalize_path,
                accepted_metadata_path,
            )
        }
        proxy_rows = [
            json.loads(line)
            for line in proxy_ledger_path.read_text().splitlines()
        ]
        engine_rows = [
            json.loads(line)
            for line in engine_ledger_path.read_text().splitlines()
        ]
        extra_task_key = fixed32_task_key_id(CANONICAL_TASK_IDS[3])
        extra_logical = hashlib.sha256(b"fixture-extra-logical").hexdigest()
        extra_wire = hashlib.sha256(b"fixture-extra-wire").hexdigest()
        extra_engine = hashlib.sha256(b"fixture-extra-engine").hexdigest()
        extra_evidence = hashlib.sha256(b"fixture-extra-evidence").hexdigest()

        def ingress_row(
            *,
            role: str,
            event: str,
            logical: str | None,
            wire: str | None,
            engine: str | None,
            status_code: int | None,
            outcome: str,
            evidence: str | None,
        ) -> dict[str, Any]:
            return {
                "schema": FIXED32_INGRESS_LEDGER_SCHEMA,
                "seq": 0,
                "role": role,
                "phase": "campaign",
                "event": event,
                "route": "chat",
                "task_key_id": extra_task_key,
                "logical_id_sha256": logical,
                "wire_id_sha256": wire,
                "engine_request_id_sha256": engine,
                "status_code": status_code,
                "outcome": outcome,
                "reason": None,
                "evidence_sha256": evidence,
                "prev_sha256": "",
                "record_sha256": "",
            }

        proxy_rows[-1:-1] = [
            ingress_row(
                role="proxy",
                event="logical_begin",
                logical=extra_logical,
                wire=None,
                engine=None,
                status_code=None,
                outcome="accepted",
                evidence=None,
            ),
            ingress_row(
                role="proxy",
                event="attempt_begin",
                logical=extra_logical,
                wire=extra_wire,
                engine=extra_engine,
                status_code=None,
                outcome="dispatched",
                evidence=extra_evidence,
            ),
            ingress_row(
                role="proxy",
                event="attempt_result",
                logical=extra_logical,
                wire=extra_wire,
                engine=extra_engine,
                status_code=200,
                outcome="response",
                evidence=extra_evidence,
            ),
            ingress_row(
                role="proxy",
                event="logical_complete",
                logical=extra_logical,
                wire=None,
                engine=None,
                status_code=None,
                outcome="completed",
                evidence=None,
            ),
        ]
        engine_rows[-1:-1] = [
            ingress_row(
                role="engine",
                event="request_accepted",
                logical=None,
                wire=extra_wire,
                engine=extra_engine,
                status_code=None,
                outcome="accepted",
                evidence=extra_evidence,
            ),
            ingress_row(
                role="engine",
                event="request_complete",
                logical=None,
                wire=extra_wire,
                engine=extra_engine,
                status_code=None,
                outcome="completed",
                evidence=extra_evidence,
            ),
        ]
        try:
            rewrite_ingress_rows(proxy_ledger_path, proxy_rows)
            rewrite_ingress_rows(engine_ledger_path, engine_rows)
            for finalize_path, rows, scalar_fields, evidence_fields in (
                (
                    proxy_finalize_path,
                    proxy_rows,
                    (
                        "accepted_logical_requests",
                        "completed_logical_requests",
                        "accepted_attempts",
                        "completed_attempts",
                    ),
                    (
                        "accepted_logical_requests",
                        "completed_logical_model_requests",
                        "accepted_attempts",
                        "completed_attempts",
                    ),
                ),
                (
                    engine_finalize_path,
                    engine_rows,
                    (
                        "accepted_engine_requests",
                        "completed_engine_requests",
                    ),
                    (
                        "accepted_engine_requests",
                        "completed_engine_requests",
                    ),
                ),
            ):
                finalize = json.loads(finalize_path.read_bytes())
                finalize["ledger_records"] = len(rows)
                finalize["ledger_chain_head_sha256"] = rows[-1][
                    "record_sha256"
                ]
                for field in scalar_fields:
                    finalize[field] += 1
                task_evidence = next(
                    evidence
                    for evidence in finalize["task_evidence"]
                    if evidence["task_key_id"] == extra_task_key
                )
                for field in evidence_fields:
                    task_evidence[field] += 1
                finalize_path.write_text(
                    json.dumps(
                        finalize,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="ascii",
                )
            accepted_metadata = json.loads(accepted_metadata_path.read_bytes())
            accepted_provenance = accepted_metadata[
                "fixed32_real_task_provenance"
            ]
            for field in (
                "completed_logical_model_requests",
                "accepted_attempts",
                "completed_attempts",
            ):
                accepted_provenance[field] += 1
            accepted_provenance[
                "task_auth_evidence_after_ledger_records"
            ] = len(proxy_rows) - 1
            accepted_provenance[
                "task_auth_evidence_after_ledger_chain_head_sha256"
            ] = proxy_rows[-2]["record_sha256"]
            accepted_after = {
                "schema": "fr13-fixed32-task-auth-evidence-v1",
                "task_key_id": extra_task_key,
                **{
                    field: accepted_provenance[field]
                    for field in (
                        "completed_logical_model_requests",
                        "aborted_logical_requests",
                        "accepted_attempts",
                        "completed_attempts",
                        "failed_attempts",
                    )
                },
                "phase": "campaign",
                "ledger_records": len(proxy_rows) - 1,
                "ledger_chain_head_sha256": proxy_rows[-2]["record_sha256"],
            }
            accepted_provenance["task_auth_evidence_after_sha256"] = (
                canonical_json_sha256(accepted_after)
            )
            accepted_metadata_path.write_text(
                json.dumps(
                    accepted_metadata,
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            expect_gate_error(
                rerun_b1_fixture,
                "trace/task-auth counts do not reconcile",
            )
        finally:
            for path, raw in original_accepted_artifacts.items():
                path.write_bytes(raw)

        external_launch_path = b1_root / "external_manifest.at_launch.json"
        good_external_launch = external_launch_path.read_bytes()
        bad_external = json.loads(good_external_launch)
        bad_external["forked_fa2"]["sha256"] = "0" * 64
        external_launch_path.write_text(
            json.dumps(bad_external, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "external manifest canonical digest mismatch",
        )
        external_launch_path.write_bytes(good_external_launch)

        external_end_path = b1_root / "external_manifest.at_end.json"
        good_external_end = external_end_path.read_bytes()
        different_external = json.loads(good_external_end)
        external_end_path.write_text(
            json.dumps(
                different_external,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "external manifest changed during the campaign",
        )
        external_end_path.write_bytes(good_external_end)

        hydra_arm = b1_root / f"hydra27_fixed32_{b1_tag}"
        hydra_attestation_path = (
            hydra_arm / "logs" / "fr13_fixed32_runtime_attestation.json"
        )
        good_hydra_attestation = hydra_attestation_path.read_bytes()
        different_attestation = json.loads(good_hydra_attestation)
        different_attestation["python"]["version"] = "3.12.4"
        different_attestation.pop("overall_canonical_sha256")
        different_attestation["overall_canonical_sha256"] = canonical_json_sha256(
            different_attestation
        )
        hydra_attestation_path.write_text(
            json.dumps(
                different_attestation,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "runtime attestations differ",
        )
        hydra_attestation_path.write_bytes(good_hydra_attestation)

        tail_attestation_path = (
            tail_arm / "logs" / "fr13_fixed32_runtime_attestation.json"
        )
        good_tail_attestation = tail_attestation_path.read_bytes()
        wrong_fa2_path_attestation = json.loads(good_tail_attestation)
        wrong_fa2_path_attestation["forked_fa2"]["source"]["path"] = "/tmp/wrong.so"
        wrong_fa2_path_attestation.pop("overall_canonical_sha256")
        wrong_fa2_path_attestation["overall_canonical_sha256"] = (
            canonical_json_sha256(wrong_fa2_path_attestation)
        )
        tail_attestation_path.write_text(
            json.dumps(
                wrong_fa2_path_attestation,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "runtime FA2 paths differ",
        )
        tail_attestation_path.write_bytes(good_tail_attestation)

        container_identity_path = tail_arm / "fixed32_container_identity.json"
        good_container_identity = container_identity_path.read_bytes()
        wrong_container_identity = json.loads(good_container_identity)
        wrong_container_identity["image_id"] = "sha256:" + "0" * 64
        container_identity_path.write_text(
            json.dumps(
                wrong_container_identity,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "running container identity differs",
        )
        container_identity_path.write_bytes(good_container_identity)

        process_path = tail_arm / "fixed32_process_identity.json"
        good_process_identity = process_path.read_bytes()
        wrong_pid1 = json.loads(good_process_identity)
        wrong_pid1["pid1"]["argv"].append("--unexpected")
        process_path.write_text(
            json.dumps(wrong_pid1, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "PID1 argv differs from the exact fixed32 contract",
        )
        process_path.write_bytes(good_process_identity)

        wrong_process_map = json.loads(good_process_identity)
        wrong_process_map["engine_core"]["forked_fa2_maps"] = []
        process_path.write_text(
            json.dumps(wrong_process_map, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "EngineCore did not map the pinned forked FA2 binary",
        )
        process_path.write_bytes(good_process_identity)

        wrong_pid1_env = json.loads(good_process_identity)
        env_index = wrong_pid1_env["pid1"]["environ"].index(
            "FR13_COMMITTER_GRAPH=1"
        )
        wrong_pid1_env["pid1"]["environ"][env_index] = "FR13_COMMITTER_GRAPH=0"
        process_path.write_text(
            json.dumps(wrong_pid1_env, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "PID1 expected FR13_COMMITTER_GRAPH=1",
        )
        process_path.write_bytes(good_process_identity)

        trace_path = first_task_dir / "qwen_trace.jsonl"
        good_trace = trace_path.read_bytes()
        trace_path.write_text("{}\n", encoding="utf-8")
        expect_gate_error(
            rerun_b1_fixture,
            "trace does not start with the pinned Qwen 0.19.4 init record",
        )
        trace_path.write_bytes(good_trace)

        task_metadata_path = first_task_dir / "runner_metadata.json"
        good_task_metadata = task_metadata_path.read_bytes()
        empty_model_trace = (
            "\n".join(
                json.dumps(
                    event,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                for event in (
                    {
                        "type": "system",
                        "subtype": "init",
                        "qwen_code_version": FIXED32_QWEN_CODE_VERSION,
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": "No model output.",
                        "usage": {"input_tokens": 1},
                    },
                )
            )
            + "\n"
        ).encode("utf-8")
        trace_path.write_bytes(empty_model_trace)
        empty_trace_metadata = json.loads(good_task_metadata)
        empty_trace_provenance = empty_trace_metadata[
            "fixed32_real_task_provenance"
        ]
        empty_trace_provenance["trace_sha256"] = hashlib.sha256(
            empty_model_trace
        ).hexdigest()
        empty_trace_provenance["trace_bytes"] = len(empty_model_trace)
        task_metadata_path.write_text(
            json.dumps(empty_trace_metadata, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "legacy terminal response IDs are empty",
        )
        trace_path.write_bytes(good_trace)
        task_metadata_path.write_bytes(good_task_metadata)

        failed_agent_metadata = json.loads(good_task_metadata)
        failed_agent_metadata["agent"]["network_drop"] = True
        failed_agent_metadata["codex"]["network_drop"] = True
        task_metadata_path.write_text(
            json.dumps(failed_agent_metadata, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "task agent did not complete cleanly",
        )
        task_metadata_path.write_bytes(good_task_metadata)

        eval_path = first_task_dir / "eval" / "eval_report.json"
        good_eval_report = eval_path.read_bytes()
        nonterminal_eval_metadata = json.loads(good_task_metadata)
        nonterminal_eval_metadata["eval_report"]["passed"] = False
        task_metadata_path.write_text(
            json.dumps(
                nonterminal_eval_metadata,
                ensure_ascii=True,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        eval_path.write_text(
            json.dumps(
                nonterminal_eval_metadata["eval_report"],
                ensure_ascii=True,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "task has no terminal SWE verdict",
        )
        task_metadata_path.write_bytes(good_task_metadata)
        eval_path.write_bytes(good_eval_report)

        traffic_audit_path = tail_arm / "fixed32_chat_traffic_audit.json"
        good_traffic_audit = traffic_audit_path.read_bytes()
        bad_traffic_audit = json.loads(good_traffic_audit)
        bad_traffic_audit["checks"][
            "no_fixed32_traffic_outside_task_brackets"
        ] = False
        traffic_audit_path.write_text(
            json.dumps(
                bad_traffic_audit,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "v2 authenticated chat-task audit differs",
        )
        traffic_audit_path.write_bytes(good_traffic_audit)

        traffic_audit_path.unlink()
        expect_gate_error(
            rerun_b1_fixture,
            "missing required artifact",
        )
        traffic_audit_path.write_bytes(good_traffic_audit)

        bad_task_metadata = json.loads(good_task_metadata)
        bad_task_metadata["fixed32_dataset_record_sha256"] = "0" * 64
        task_metadata_path.write_text(
            json.dumps(
                bad_task_metadata,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "fixed32 dataset record digest mismatch",
        )
        task_metadata_path.write_bytes(good_task_metadata)

        pretask_metrics_path = tail_arm / "metrics_before_swe.txt"
        pretask_marker_path = tail_arm / "fixed32_pretask_zero_traffic.json"
        good_pretask_metrics = pretask_metrics_path.read_bytes()
        good_pretask_marker = pretask_marker_path.read_bytes()
        bad_pretask_metrics = replace_metric_values(
            good_pretask_metrics.decode("utf-8"),
            {"spec_drafts": 1.0, "spec_tokens": float(PHYSICAL_DRAFTS)},
        )
        pretask_metrics_path.write_text(bad_pretask_metrics, encoding="utf-8")
        rebound_pretask_marker = json.loads(good_pretask_marker)
        rebound_pretask_marker["metrics"]["sha256"] = sha256_file(
            pretask_metrics_path
        )
        pretask_marker_path.write_text(
            json.dumps(
                rebound_pretask_marker,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "pretask decode metrics are not exact zero",
        )
        pretask_metrics_path.write_bytes(good_pretask_metrics)
        pretask_marker_path.write_bytes(good_pretask_marker)

        bad_non_spec_metrics = replace_metric_values(
            good_pretask_metrics.decode("utf-8"),
            {"fwd_s": 0.001, "fwd_steps": 1.0},
        )
        pretask_metrics_path.write_text(
            bad_non_spec_metrics,
            encoding="utf-8",
        )
        rebound_pretask_marker = json.loads(good_pretask_marker)
        rebound_pretask_marker["metrics"]["sha256"] = sha256_file(
            pretask_metrics_path
        )
        pretask_marker_path.write_text(
            json.dumps(
                rebound_pretask_marker,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "pretask decode metrics are not exact zero",
        )
        pretask_metrics_path.write_bytes(good_pretask_metrics)
        pretask_marker_path.write_bytes(good_pretask_marker)

        forbidden_probe_path = tail_arm / "warmup_probe.json"
        forbidden_probe_path.write_text("{}\n", encoding="utf-8")
        expect_gate_error(
            rerun_b1_fixture,
            "fixed32 forbidden pretask probe artifacts exist",
        )
        forbidden_probe_path.unlink()

        b4_root, b4_sidecars, b4_tag = write_fixture_campaign(repo, base, concurrency=4)
        b4 = reduce_campaign(
            repo,
            b4_root,
            b4_tag,
            4,
            4,
            b4_sidecars,
            BOOTSTRAP_REPS,
            BOOTSTRAP_SEED,
        )
        b4_repeat = reduce_campaign(
            repo,
            b4_root,
            b4_tag,
            4,
            4,
            b4_sidecars,
            BOOTSTRAP_REPS,
            BOOTSTRAP_SEED,
        )
        assert json.dumps(b4, sort_keys=True, allow_nan=False) == json.dumps(
            b4_repeat, sort_keys=True, allow_nan=False
        )
        assert b4["analysis_valid"]
        tail_b4 = b4["arms"]["tail6_fixed32"]["statistics"]
        assert tail_b4["bracket_mode"].startswith("overlap-safe")
        assert tail_b4["union_intervals"]["wall"] == [0, 645]
        assert tail_b4["sidecar_coverage"]["wall"]["selected_steps"] == 645
        assert tail_b4["sidecar_coverage"]["wall"]["fraction"] == 1.0
        assert tail_b4["sidecar_counter_reconciliation"]["wall"][
            "exact_drafts_and_steps"
        ]
        b4_census_expected = b4["arms"]["tail6_fixed32"]["work_census_expected"]
        assert b4_census_expected["canonical_task_selection"]["counter_intervals"] == [
            [0, 645]
        ]
        assert b4_census_expected["canonical_task_selection"]["event_count"] == 645
        assert b4_census_expected["complete_stream"]["event_count"] == 645
        assert b4["fixed32_work_census"]["physical_work_comparison"][
            "observed_batch_sizes"
        ] == [3, 4]
        assert b4["fixed32_work_census"]["drafter_graph_lifecycle"][
            "registry_batch_sizes"
        ] == [3, 4]
        assert b4["fixed32_work_census"][
            "forward_graph_pregather_lifecycle"
        ]["registry_batch_sizes"] == [1, 2, 3, 4]
        naive_task_sum = sum(
            end - start for start, end in ((0, 500), (20, 620), (40, 640), (60, 645))
        )
        assert naive_task_sum > 645
        expect_gate_error(
            lambda: assert_nonoverlap(
                [(0, 500), (20, 620), (40, 640), (60, 645)],
                "synthetic B=4 task-sum",
            ),
            "counter intervals overlap",
        )
        blocks = tail_b4["moving_block_u95_sensitivity"]["blocks"]
        assert [row["block_steps"] for row in blocks] == list(BLOCK_SENSITIVITY)
        worst = tail_b4["moving_block_u95_sensitivity"][
            "worst_across_requested_blocks"
        ]["legacy_slo_excess_ms_u95"]
        assert worst == max(row["legacy_slo_excess_ms_u95"] for row in blocks)
        assert tail_b4["forward_wall_occupancy_sequence_equal"]
        assert (
            tail_b4["exact_b4_stratum"]["selected_steps"] >= MIN_B4_EXACT_EVENTS
        )
        assert tail_b4["exact_b4_stratum"]["gate"]["pass"]

        paired_steps = 700
        paired_indices = np.arange(7, paired_steps, dtype=np.int64)
        paired_fwd_ms = np.full(paired_steps, 70.0)
        paired_fwd_ms[:7] = 1_000.0
        paired_wall_ms = np.full(len(paired_indices), 100.0)
        paired_drafts = np.asarray(
            [
                (3, 4, 4, 4, 4)[index % 5]
                for index in range(paired_steps)
            ],
            dtype=np.float64,
        )
        paired_wall_drafts = paired_drafts[paired_indices]
        paired_windows = [
            {
                "task_id": "synthetic-paired-b4",
                "fwd_span": (0, paired_steps),
                "wall_span": (0, len(paired_indices)),
                "pre": {
                    "fwd_steps": 0.0,
                    "fwd_s": 0.0,
                    "fwd_drafts": 0.0,
                    "wall_steps": 0.0,
                    "wall_s": 0.0,
                    "wall_drafts": 0.0,
                },
                "post": {
                    "fwd_steps": float(paired_steps),
                    "fwd_s": float(paired_fwd_ms.sum() / 1000.0),
                    "fwd_drafts": float(paired_drafts.sum()),
                    "wall_steps": float(len(paired_indices)),
                    "wall_s": float(paired_wall_ms.sum() / 1000.0),
                    "wall_drafts": float(paired_wall_drafts.sum()),
                },
            }
        ]
        paired_b4 = b4_arm_statistics(
            paired_windows,
            {
                "fwd_indices": np.arange(paired_steps, dtype=np.int64),
                "fwd_ms": paired_fwd_ms,
                "fwd_full": np.ones(paired_steps, dtype=np.bool_),
                "fwd_drafts": paired_drafts,
                "wall_ms": paired_wall_ms,
                "wall_drafts": paired_wall_drafts,
                "wall_fwd_indices": paired_indices,
            },
            PHYSICAL_DRAFTS,
            4,
            64,
            BOOTSTRAP_SEED,
        )
        assert paired_b4["wall_forward_pairing"]["retained_wall_fraction"] == 0.99
        assert paired_b4["union_counter_point"]["verify_ms_per_step"] == 70.0
        assert (
            paired_b4["union_counter_point"]["all_forward_verify_ms_per_step"]
            > 70.0
        )
        assert all(
            row["verify_ms_per_step_u95"] == 70.0
            for row in paired_b4["moving_block_u95_sensitivity"]["blocks"]
        )

        adversarial_steps = 10_240
        adversarial_drafts = np.asarray(
            [4.0 if index % 20 == 0 else 3.0 for index in range(adversarial_steps)]
        )
        adversarial_wall_ms = np.asarray(
            [
                1_000.0 if index % 20 == 0 else 60.0
                for index in range(adversarial_steps)
            ]
        )
        adversarial_fwd_ms = np.full(adversarial_steps, 70.0)
        adversarial_draft_total = float(adversarial_drafts.sum())
        adversarial_windows = [
            {
                "task_id": "synthetic-exact-b4-regression",
                "fwd_span": (0, adversarial_steps),
                "wall_span": (0, adversarial_steps),
                "pre": {
                    "fwd_steps": 0.0,
                    "fwd_s": 0.0,
                    "fwd_drafts": 0.0,
                    "wall_steps": 0.0,
                    "wall_s": 0.0,
                    "wall_drafts": 0.0,
                },
                "post": {
                    "fwd_steps": float(adversarial_steps),
                    "fwd_s": float(adversarial_fwd_ms.sum() / 1000.0),
                    "fwd_drafts": adversarial_draft_total,
                    "wall_steps": float(adversarial_steps),
                    "wall_s": float(adversarial_wall_ms.sum() / 1000.0),
                    "wall_drafts": adversarial_draft_total,
                },
            }
        ]
        adversarial_b4 = b4_arm_statistics(
            adversarial_windows,
            {
                "fwd_indices": np.arange(adversarial_steps, dtype=np.int64),
                "fwd_ms": adversarial_fwd_ms,
                "fwd_full": np.ones(adversarial_steps),
                "fwd_drafts": adversarial_drafts,
                "wall_ms": adversarial_wall_ms,
                "wall_drafts": adversarial_drafts.copy(),
                "wall_fwd_indices": np.arange(
                    adversarial_steps, dtype=np.int64
                ),
            },
            PHYSICAL_DRAFTS,
            4,
            256,
            BOOTSTRAP_SEED,
        )
        assert adversarial_b4["gate"]["union_pass"]
        assert not adversarial_b4["gate"]["exact_b4_pass"]
        assert not adversarial_b4["gate"]["pass"]

        b4_tail_arm = b4_root / f"tail6_fixed32_{b4_tag}"
        b4_census_path = b4_tail_arm / "logs" / "fr13_fixed32_work_census.jsonl"
        good_b4_census = b4_census_path.read_bytes()
        b4_census_records = [
            json.loads(line) for line in good_b4_census.decode("utf-8").splitlines()
        ]
        b4_census_events = b4_census_records[:-1]
        for record in b4_census_events:
            record["forward_step_index"] += 1
        b4_census_records = [
            *b4_census_events,
            work_census_terminal_fixture(
                b4_census_events,
                fixture_synthetic_runtime_proof=True,
            ),
        ]
        b4_census_path.write_text(
            "\n".join(
                json.dumps(record, ensure_ascii=True, sort_keys=True)
                for record in b4_census_records
            )
            + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b4_root,
                b4_tag,
                4,
                4,
                b4_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "census prefix digest mismatch",
        )
        b4_census_path.write_bytes(good_b4_census)

        hydra_arm = b1_root / f"hydra27_fixed32_{b1_tag}"
        env_path = hydra_arm / "container_env.txt"
        good_env = env_path.read_text(encoding="utf-8")
        env_path.write_text(
            good_env.replace(
                "FR13_FIXED32_MODE=hydra27_fixed32",
                "FR13_FIXED32_MODE=tail6_fixed32",
            ),
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "expected exactly FR13_FIXED32_MODE=hydra27_fixed32",
        )
        env_path.write_text(good_env, encoding="utf-8")

        env_path.write_text(
            good_env.replace("FR13_COMMITTER_GRAPH=1", "FR13_COMMITTER_GRAPH=0"),
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "expected exactly FR13_COMMITTER_GRAPH=1",
        )
        env_path.write_text(good_env, encoding="utf-8")

        env_path.write_text(
            good_env.replace("FR13_STEP_WALL_CAP_S=1.5", "FR13_STEP_WALL_CAP_S=0.1"),
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "expected exactly FR13_STEP_WALL_CAP_S=1.5",
        )
        env_path.write_text(good_env, encoding="utf-8")

        runtime_path = hydra_arm / "docker_full.log"
        good_runtime = runtime_path.read_text(encoding="utf-8")
        runtime_path.write_text(
            good_runtime.replace(
                "levels=[1, 11] lens=[5, 7] critical=12",
                "levels=[1, 10] lens=[12, 7] critical=19",
            ),
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "current runtime needle",
        )
        runtime_path.write_text(good_runtime, encoding="utf-8")

        corrupt_subset = base / "corrupt_subset.json"
        corrupt_subset.write_text('{"instance_ids":[]}\n', encoding="utf-8")
        runlog_path = b1_root / f"hydra27_fixed32_{b1_tag}.runlog"
        good_runlog = runlog_path.read_text(encoding="utf-8")
        canonical_path = str((repo / EVIDENCE_SETS[4]["relative_path"]).resolve())
        runlog_path.write_text(
            good_runlog.replace(canonical_path, str(corrupt_subset)),
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "canonical subset sha256 mismatch",
        )
        runlog_path.write_text(good_runlog, encoding="utf-8")

        original_sidecar = next(
            path
            for path in b1_sidecars.iterdir()
            if path.name.startswith(f"tail6_fixed32_{b1_tag}.json.samples.")
        )
        duplicate = b1_sidecars / f"tail6_fixed32_{b1_tag}.json.samples.999"
        duplicate.write_bytes(original_sidecar.read_bytes())
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "expected one per-step sidecar",
        )
    print("self-test OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reduce a completed matched Tail6-fixed32/Hydra27-fixed32 canonical "
            "SWE-Verified campaign without mutating campaign artifacts."
        )
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--runroot", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--task-count", type=int, choices=(4, 16))
    parser.add_argument(
        "--expect-concurrency",
        type=int,
        choices=(1, 4),
        help="optional assertion; concurrency is always inferred from each arm log",
    )
    parser.add_argument(
        "--sidecar-dir",
        type=Path,
        help="defaults to REPO/output/fr13_sfwd_sidecar",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    if args.self_test:
        self_test(repo)
        return 0
    missing = [
        name
        for name, value in (
            ("--runroot", args.runroot),
            ("--tag", args.tag),
            ("--task-count", args.task_count),
        )
        if value is None
    ]
    if missing:
        raise SystemExit("required arguments: " + ", ".join(missing))
    runroot = (
        args.runroot.resolve()
        if args.runroot.is_absolute()
        else (repo / args.runroot).resolve()
    )
    sidecar_dir = (
        args.sidecar_dir.resolve()
        if args.sidecar_dir is not None and args.sidecar_dir.is_absolute()
        else (repo / (args.sidecar_dir or Path("output/fr13_sfwd_sidecar"))).resolve()
    )
    try:
        report = reduce_campaign(
            repo,
            runroot,
            args.tag,
            args.task_count,
            args.expect_concurrency,
            sidecar_dir,
            args.bootstrap_reps,
            args.seed,
        )
    except GateError as error:
        report = {
            "schema": "fr13.canonical_swe_verified_floor_gate.v1",
            "analysis_valid": False,
            "gate_verdict": "NOT_EVALUATED_INVALID_INPUT",
            "repo": str(repo),
            "runroot": str(runroot),
            "tag": args.tag,
            "task_count": args.task_count,
            "error": str(error),
        }
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["gate_verdict"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
