#!/usr/bin/env python3
"""SWE-Bench per-instance orchestrator for Q36-A + Codex CLI 0.128.0.

Spec ref: docs/reports/auto_research/swe-bench-bounded-time-spec-20260520.md
          §9 (artifact layout), §11 (per-task protocol), §7 (concurrency).

For each instance from a pre-registered subset (built by
scripts/build_swe_bench_subset.py):
  1. Hydrate the workspace at the SWE-Bench base_commit, drop AGENTS.md
     with the problem_statement.
  2. Launch codex-runner:v1 Docker against the codex-bench proxy at
     :8022 with a wall budget.
  3. Diff workspace vs base_commit -> patch.diff (per-attempt artifact).
  4. Invoke codex-bench-eval-swe on the patch.
  5. Write per-task artifacts under
     output/swe_bench_q36_a_temp06/<dataset>/per_task/<instance_id>/.
  6. Aggregate predictions.jsonl + campaign_summary.json.

Defaults follow the spec:
  - Concurrency: 1 (LLD-05 §4.6 default; bump after Sprint-1 validation).
  - Codex wall budget: NO harness cap by default (0 = unlimited; codex self-limits on
    stream_idle_timeout_ms + turn limit). Eval buffer: 30 min. Override: SWE_AGENT_WALL_S / --agent-wall-s.
  - Proxy: http://127.0.0.1:8022/v1
  - Reasoning effort: high (carried over from launch_qwen36_ablation_point.py).
  - Temperature is governed by the vLLM relaunch bundle (Q36-A: temp=0.6).
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as _cf
import contextlib
import datetime as _dt
import hashlib
import json
import math
import os
import secrets
import shlex
import shutil
import stat
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fr13_fixed32_contract as fixed32_contract  # noqa: E402

DEFAULT_OUT_ROOT = REPO_ROOT / "output" / "swe_bench_q36_a_temp06"
DEFAULT_REPO_CACHE = REPO_ROOT / ".cache" / "swe_bench_repos"
DEFAULT_HF_HOME = REPO_ROOT / ".cache" / "huggingface"
PINNED_SWE_VERIFIED_PARQUET = (
    DEFAULT_HF_HOME
    / "hub"
    / "datasets--princeton-nlp--SWE-bench_Verified"
    / "blobs"
    / "a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd"
)
DEFAULT_ENDPOINT = "http://127.0.0.1:8022/v1"
DEFAULT_METRICS_URL = "http://127.0.0.1:9950/metrics"
DEFAULT_MODEL = "qwen3.6-27b"
DEFAULT_AGENT_WALL_S = 0  # per-attempt codex wall. 0 = NO upper limit (user 2026-07-01):
# hard tasks that genuinely need a long time were TIMED OUT mid-fix -> partial/wrong patch ->
# tests_failed (e.g. chain5 astropy-13453: timed_out=True at 27min under the old 25-min cap). With
# nudge-only (SWE_EMPTY_PATCH_RETRIES=0) a task gets ONE attempt, so the old fresh-session retry no
# longer silently grants extra wall-time -> we remove the harness cap entirely and let codex run to
# its own natural completion. This does NOT mean "hang forever": codex self-terminates on its own
# stream_idle_timeout_ms=600000 (10-min no-output idle) and its internal turn limit, and per-turn
# generation is bounded by MAX_OUTPUT (32768). Set SWE_AGENT_WALL_S>0 (or --agent-wall-s N) to
# reinstate a hard per-attempt wall of N seconds. NOTE: serial (concurrency=1) so a single stuck
# task can stall the whole sweep; the codex idle timeout is the real backstop.
DEFAULT_EVAL_TIMEOUT_S = 30 * 60
DEFAULT_MODEL_NAME_TAG = "qwen3.6-27b-fp8::qwen-code-0.19.4::q36-a"
# Same capture path used by launch_qwen36_ablation_point.py / Track B benches.
DEFAULT_PROXY_CAPTURE = Path("/tmp/track_b_e2e_proxy_capture/request_metrics.jsonl")
DEFAULT_DCGM_SAMPLER = REPO_ROOT / "scripts" / "sample_dcgm_during_task.py"
DEFAULT_DCGM_INTERVAL_S = 0.1  # 10 Hz — 1/10th of Track B's 100 Hz default to keep
# per-task dcgm_samples.jsonl under GitHub's 100 MB per-file hard limit. At 10 Hz a
# 30-min Codex task lands ~6 MB; 100 Hz lands ~60 MB and risks rejection at the
# campaign scale (500 + 731 instances).
FIXED32_INGRESS_SECRET_FILE_ENV = "FR13_FIXED32_INGRESS_SECRET_FILE"
_FIXED32_QWEN_CODE_VERSION = "0.19.4"
_FIXED32_QWEN_CAP_CHUNK_RELATIVE_PATH = (
    "npm/lib/node_modules/@qwen-code/qwen-code/chunks/chunk-BFG6OZN7.js"
)
_FIXED32_QWEN_SETTINGS_PATH = (
    REPO_ROOT / "config" / "fr13_fixed32" / "qwen_system_settings.json"
)
_FIXED32_QWEN_SETTINGS_BYTES = b'{"memory":{"enableAutoSkill":false}}\n'
_FIXED32_QWEN_SETTINGS_SHA256 = (
    "8a872a4f6f257f6d7a45f24f42500964f56e1500c5342218b71d02afe4d31fb6"
)
_FIXED32_QWEN_SETTINGS_CONTAINER_PATH = "/run/fr13/qwen-system-settings.json"
_FIXED32_QWEN_SETTINGS_ENV = "QWEN_CODE_SYSTEM_SETTINGS_PATH"
_FIXED32_QWEN_SETTINGS_MODE = "0444"
_FIXED32_QWEN_SETTINGS_UID = 1000
_FIXED32_QWEN_SETTINGS_GID = 1000
_FIXED32_QWEN_ATTESTATION_SCHEMA = (
    "fr13-fixed32-qwen-runtime-attestation-v1"
)
_FIXED32_MOUNTED_RUNTIME_PROOF_SCHEMA = (
    "fr13-fixed32-qwen-mounted-runtime-proof-v1"
)
_FIXED32_MOUNTED_RUNTIME_PROOF_FILENAME = (
    "qwen_mounted_runtime_proof.json"
)
_FIXED32_QWEN_CAMPAIGN_PROOF_SCHEMA = (
    "fr13-fixed32-qwen-campaign-provenance-v1"
)
_FIXED32_QWEN_CAMPAIGN_PROOF_FILENAME = (
    "fixed32_qwen_campaign_provenance.json"
)
_FIXED32_QWEN_CAMPAIGN_METRICS_PRE_FILENAME = (
    "fixed32_qwen_campaign_metrics_pre.txt"
)
_FIXED32_QWEN_CAMPAIGN_METRICS_POST_FILENAME = (
    "fixed32_qwen_campaign_metrics_post.txt"
)
_FIXED32_PENDING_RUNNER_METADATA_FILENAME = "runner_metadata.pending.json"
_FIXED32_CAMPAIGN_RUNTIME_ARGS_KEY = "_fixed32_campaign_runtime_args"
_FIXED32_COMMITTER_ACCEPTED_LENGTH_FULL_MASK = 0x0FFF
_FIXED32_CFWD_QUALIFICATION_CLASSIFICATION = (
    "cfwd_layer_batch_real_swe_qualification"
)
_FIXED32_CFWD_B4_QUALIFICATION_CLASSIFICATION = (
    "cfwd_layer_batch_real_swe_b4_qualification"
)
_FIXED32_AGENT_PLACEMENT_SCHEMA = "fr13-fixed32-agent-placement-v1"
_FIXED32_AGENT_HOST_ALIAS = "alienware"
_FIXED32_MEASURED_HOST_IDENTITY = {
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
_FIXED32_AGENT_HOST_IDENTITY = {
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
_FIXED32_QWEN_BUNDLE_TREE_SCHEMA = (
    "fr13-qwen-agent-bundle-manifest-v1"
)
_FIXED32_QWEN_BUNDLE_TREE_ROOTS = ["**"]
_FIXED32_QWEN_BUNDLE_TREE_REQUIRED_ENTRYPOINTS = [
    "bin/qwen",
    "node/bin/node",
    "npm/bin/qwen",
    "npm/lib/node_modules/@qwen-code/qwen-code/cli-entry.js",
    _FIXED32_QWEN_CAP_CHUNK_RELATIVE_PATH,
]
_FIXED32_QWEN_BUNDLE_TREE_SUMMARY = {
    "entry_count": 10_499,
    "directory_count": 1_514,
    "regular_file_count": 8_970,
    "symlink_count": 15,
    "regular_file_bytes": 327_941_291,
    "executable_regular_file_count": 93,
    "manifest_bytes": 2_057_964,
}
_FIXED32_QWEN_BUNDLE_TREE_ENTRYPOINTS = {
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
        "target": "../lib/node_modules/@qwen-code/qwen-code/cli-entry.js",
    },
    "npm/lib/node_modules/@qwen-code/qwen-code/cli-entry.js": {
        "path": "npm/lib/node_modules/@qwen-code/qwen-code/cli-entry.js",
        "type": "file",
        "mode": "0755",
        "bytes": 777,
        "sha256": (
            "98335eda2e0eaa737640cb5d43da032dee457ff7931c429f972ba3ff8a695d3a"
        ),
    },
    _FIXED32_QWEN_CAP_CHUNK_RELATIVE_PATH: {
        "path": _FIXED32_QWEN_CAP_CHUNK_RELATIVE_PATH,
        "type": "file",
        "mode": "0644",
        "bytes": 5_451_144,
        "sha256": (
            "d61b71c03180822e875976a721a856144b70ae8b7ff687910021a5cb91a7db89"
        ),
    },
}
_FIXED32_QWEN_BUNDLE_TREE_SHA256 = (
    "594cac41e2d5ed505e0646f318b263ff70e200bcffe97326fe1c042fdc220516"
)
_FIXED32_QWEN_BUNDLE_REMOTE_BASENAME = (
    "qwen_agent_bundle-" + _FIXED32_QWEN_BUNDLE_TREE_SHA256
)
_FIXED32_QWEN_BUNDLE_REMOTE_PATH = (
    "~/" + _FIXED32_QWEN_BUNDLE_REMOTE_BASENAME
)
_FIXED32_QWEN_BUNDLE_TREE_EXPECTED = {
    "schema": _FIXED32_QWEN_BUNDLE_TREE_SCHEMA,
    "roots": _FIXED32_QWEN_BUNDLE_TREE_ROOTS,
    "summary": _FIXED32_QWEN_BUNDLE_TREE_SUMMARY,
    "entrypoints": _FIXED32_QWEN_BUNDLE_TREE_ENTRYPOINTS,
    "manifest_sha256": _FIXED32_QWEN_BUNDLE_TREE_SHA256,
}
_FIXED32_AGENT_IMAGE_IDENTITIES = {
    "astropy__astropy-12907": {
        "id": "sha256:cce639c4d4c4f8a47893a81a1a1894d3f2c77e603694da4000581450783ef532",
        "repo_digest": (
            "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907@"
            "sha256:f3f63bb87d581c0e7b47f900dd82165b71040e1758d3c29e915e2b18da9baf63"
        ),
    },
    "astropy__astropy-13033": {
        "id": "sha256:bdbc30d363cfbd9902e0d338fc959782d1a92a9e85236c0dca92c95372088783",
        "repo_digest": (
            "swebench/sweb.eval.x86_64.astropy_1776_astropy-13033@"
            "sha256:42797a2c686ed35b43e28dd2f58149381c3bf79abea2346fb7062c009e9fa528"
        ),
    },
    "astropy__astropy-13236": {
        "id": "sha256:236d427ece8e0d3d282598d981ec9a3baff668041cab16f4545500956cb807db",
        "repo_digest": (
            "swebench/sweb.eval.x86_64.astropy_1776_astropy-13236@"
            "sha256:a43e166eb5ae9e477349b87d800ece7648f8c746f88d94f6f6cff0df1e2caf82"
        ),
    },
    "astropy__astropy-13398": {
        "id": "sha256:1047cc1d43b1a33fd5c3c5ca53b85d7380cdda26bf4c18ada15dd12a8d9b076d",
        "repo_digest": (
            "swebench/sweb.eval.x86_64.astropy_1776_astropy-13398@"
            "sha256:423240067cb26131788348337e59a4d51225fd491eccab99e919c1bc4d4b02e0"
        ),
    },
    "astropy__astropy-13453": {
        "id": "sha256:67e4955c1ea7f9013a51299a236574f87bc1ca2da02c9aa6ac640f8ce61d2f8b",
        "repo_digest": (
            "swebench/sweb.eval.x86_64.astropy_1776_astropy-13453@"
            "sha256:6b3593a3d1fa032b8b81f9a1e33738aae14c1cdc8fd887e0a089592c7f9abf9b"
        ),
    },
    "astropy__astropy-13579": {
        "id": "sha256:38c16ead9bc5b6a8cfb991a85511471ca91a1f334ca5d9001830c069beb3c94e",
        "repo_digest": (
            "swebench/sweb.eval.x86_64.astropy_1776_astropy-13579@"
            "sha256:c645e9435a1499dab327b76fae926057a2f407a291b5d173bf1971b7a6fe911b"
        ),
    },
    "astropy__astropy-13977": {
        "id": "sha256:fc877eb6e9cada009834e33eeab24e7d6690e189932d3f3c4e5f7a911e7e57b6",
        "repo_digest": (
            "swebench/sweb.eval.x86_64.astropy_1776_astropy-13977@"
            "sha256:bcc442a117def63c8011091500b4e2dd49980a08b74e3c2fbdad4e42745278bb"
        ),
    },
    "astropy__astropy-14096": {
        "id": "sha256:3ac2306a3e172713ad0f00dd5daa6d3cfd6990fe3b52afbf585fe1d6161ae8b9",
        "repo_digest": (
            "swebench/sweb.eval.x86_64.astropy_1776_astropy-14096@"
            "sha256:f0277869f5874118b6395d945a278d88e8912dbb2970ee2d0289f5591adab8a3"
        ),
    },
    "astropy__astropy-14182": {
        "id": "sha256:770f48b2842d8115639b73b022994b2e1b20af676367509244b9dcbad3f903e9",
        "repo_digest": (
            "swebench/sweb.eval.x86_64.astropy_1776_astropy-14182@"
            "sha256:374a7f3206d23fd41aaf6ac3361a34dad941bdbe92ec81ff1b4c3e0163e38453"
        ),
    },
    "astropy__astropy-14309": {
        "id": "sha256:f896444b1e4977454bd415d711926f7ce0f25c8b4c6c417b26d38df59e8a3ca6",
        "repo_digest": (
            "swebench/sweb.eval.x86_64.astropy_1776_astropy-14309@"
            "sha256:89fc1b9379b349c8d1773cc3e6982cb32d2732f90a240047c688735db35c5212"
        ),
    },
    "astropy__astropy-14365": {
        "id": "sha256:3925b434247f8fc7aabe79a902d03d4566d697c4f35b7a39fdfa209263efb358",
        "repo_digest": (
            "swebench/sweb.eval.x86_64.astropy_1776_astropy-14365@"
            "sha256:bb1ec8d27d478ab7469805680c824a4030fa3f32a592558f7a02485b76f3226b"
        ),
    },
    "astropy__astropy-14369": {
        "id": "sha256:1a8a749d6b8edb837761e2d09bdf9459e3bdb2436d4070412d886de135ff0a3a",
        "repo_digest": (
            "swebench/sweb.eval.x86_64.astropy_1776_astropy-14369@"
            "sha256:113488bfd3a6fb9678705496c18791bf46d98ad0303a682a3d405223eb2e85c4"
        ),
    },
    "astropy__astropy-14508": {
        "id": "sha256:f1f44383c21a57c5047ac6cdb4375407350dd14d54e455ec3b5dfec79ee26c7c",
        "repo_digest": (
            "swebench/sweb.eval.x86_64.astropy_1776_astropy-14508@"
            "sha256:a6aea03ce1c6a2e897e78a4339e44f02643deb9007e0628a058034779181ce71"
        ),
    },
    "astropy__astropy-14539": {
        "id": "sha256:290a743498af81faf833324ccb3dfaf877e1d4fdd60594efc1a5f4835601316e",
        "repo_digest": (
            "swebench/sweb.eval.x86_64.astropy_1776_astropy-14539@"
            "sha256:a8d0f9829ec24dfb23a2f0097a245ee60faf1b396b33b3af5c22d7ac5f3c00ab"
        ),
    },
    "astropy__astropy-14598": {
        "id": "sha256:3f3a0c5f4cd49b03ef4fde7f8e8bdf18833ee685d223395893b61426a6e62b8d",
        "repo_digest": (
            "swebench/sweb.eval.x86_64.astropy_1776_astropy-14598@"
            "sha256:f1ff70694c403ee7018fef2a00638caa1555948d1c9b821175a7f4bdf2933a52"
        ),
    },
    "astropy__astropy-14995": {
        "id": "sha256:99afb65d48b892e0d2e015eeb0794175d26e6a092c81598aa5a32fb0978b30cc",
        "repo_digest": (
            "swebench/sweb.eval.x86_64.astropy_1776_astropy-14995@"
            "sha256:b29a3bf3daebe6055a2bba46bc98db070043acd53801bd783c2f620813a87eae"
        ),
    },
}
_FIXED32_CLEARED_AGENT_ENV = (
    "BASH_ENV",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "NODE_OPTIONS",
    "NODE_PATH",
)


def _fixed32_task_auth(instance_id: str) -> tuple[str, str]:
    secret_file = os.environ.get(FIXED32_INGRESS_SECRET_FILE_ENV, "").strip()
    if not secret_file:
        raise Fixed32BoundaryError(
            f"fixed32 requires {FIXED32_INGRESS_SECRET_FILE_ENV}"
        )
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from lumo_flywheel_serving.inference_proxy import (
        derive_fixed32_task_bearer,
    )

    try:
        return derive_fixed32_task_bearer(secret_file, instance_id)
    except Exception as exc:
        raise Fixed32BoundaryError(
            f"fixed32 task credential derivation failed for {instance_id}: "
            f"{type(exc).__name__}"
        ) from exc


def _fixed32_task_auth_evidence(
    *,
    endpoint: str,
    task_bearer: str,
    task_key_id: str,
) -> dict[str, Any]:
    normalized_endpoint = endpoint.rstrip("/")
    evidence_url = (
        normalized_endpoint[:-3]
        if normalized_endpoint.endswith("/v1")
        else normalized_endpoint
    ) + "/admin/fixed32/ingress/task-evidence"
    if AGENT_HOST:
        remote_query = (
            "import os,sys,urllib.request;"
            "req=urllib.request.Request(sys.argv[1],data=b'{}',method='POST',"
            "headers={'Authorization':'Bearer '+os.environ['FR13_TASK_BEARER'],"
            "'Content-Type':'application/json'});"
            "print(urllib.request.urlopen(req,timeout=30).read().decode('utf-8'))"
        )
        command = (
            "IFS= read -r FR13_TASK_BEARER && export FR13_TASK_BEARER && "
            "python3 -c "
            + shlex.quote(remote_query)
            + " "
            + shlex.quote(evidence_url)
        )
        completed = subprocess.run(
            ["ssh", *_EVAL_SSH_OPTS, AGENT_HOST, command],
            input=task_bearer + "\n",
            text=True,
            capture_output=True,
            timeout=45,
            check=False,
        )
        if completed.returncode != 0:
            raise Fixed32BoundaryError(
                "fixed32 task-auth evidence query failed on the agent host"
            )
        raw = completed.stdout
    else:
        import urllib.request

        request = urllib.request.Request(
            evidence_url,
            data=b"{}",
            headers={
                "Authorization": f"Bearer {task_bearer}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read().decode("utf-8")
        except Exception as exc:
            raise Fixed32BoundaryError(
                "fixed32 task-auth evidence query failed"
            ) from exc
    try:
        def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            parsed: dict[str, Any] = {}
            for key, value in pairs:
                if key in parsed:
                    raise ValueError("duplicate task-auth evidence key")
                parsed[key] = value
            return parsed

        def _reject_nonfinite(value: str) -> Any:
            raise ValueError(f"nonfinite task-auth evidence value: {value}")

        payload = json.loads(
            raw,
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise Fixed32BoundaryError(
            "fixed32 task-auth evidence response is invalid JSON"
        ) from exc
    expected_keys = {
        "schema",
        "task_key_id",
        "completed_logical_model_requests",
        "aborted_logical_requests",
        "accepted_attempts",
        "completed_attempts",
        "failed_attempts",
        "phase",
        "ledger_records",
        "ledger_chain_head_sha256",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_keys
        or payload["schema"] != "fr13-fixed32-task-auth-evidence-v1"
        or payload["task_key_id"] != task_key_id
        or payload["phase"] != "campaign"
    ):
        raise Fixed32BoundaryError("fixed32 task-auth evidence contract mismatch")
    for key in (
        "completed_logical_model_requests",
        "aborted_logical_requests",
        "accepted_attempts",
        "completed_attempts",
        "failed_attempts",
        "ledger_records",
    ):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise Fixed32BoundaryError(
                f"fixed32 task-auth evidence {key} is invalid"
            )
    head = payload["ledger_chain_head_sha256"]
    if (
        not isinstance(head, str)
        or len(head) != 64
        or any(char not in "0123456789abcdef" for char in head)
    ):
        raise Fixed32BoundaryError(
            "fixed32 task-auth evidence ledger head is invalid"
        )
    return payload


def _agent_subprocess_env(task_bearer: str | None) -> dict[str, str]:
    env = dict(os.environ)
    for name in (
        FIXED32_INGRESS_SECRET_FILE_ENV,
        "LUMO_PROXY_FIXED32_SECRET_FILE",
        "LUMO_PROXY_FIXED32_LEDGER_PATH",
        "FR13_FIXED32_ENGINE_INGRESS_LEDGER_PATH",
    ):
        env.pop(name, None)
    env["OPENAI_API_KEY"] = task_bearer if task_bearer is not None else "EMPTY"
    return env


def _remote_agent_command(command: str) -> str:
    return (
        "IFS= read -r OPENAI_API_KEY && export OPENAI_API_KEY && "
        f"{command}"
    )

CODEX_TEMPLATE = (
    "docker run --rm --name {container_name} --network=host -u 1000:1000 "
    "-v {workspace}:/workspace:rw "
    "-e OPENAI_API_KEY -e OPENAI_BASE_URL={endpoint} -e HOME=/tmp "
    "-w /workspace codex-runner:v1 "
    "codex exec --json --skip-git-repo-check "
    "--dangerously-bypass-approvals-and-sandbox -C /workspace "
    "-c 'model_provider=\"local-proxy\"' "
    "-c 'model_providers.local-proxy={{name=\"local-proxy\","
    "base_url=\"{endpoint}\",env_key=\"OPENAI_API_KEY\","
    "wire_api=\"responses\",stream_idle_timeout_ms=600000}}' "
    "-c 'model_reasoning_effort=\"high\"' "
    "-c 'model_supports_reasoning_summaries=true' "
    "-c 'model_reasoning_summary=\"auto\"' "
    "--model {model}"
)

# qwen-code agent harness (SWE_AGENT=qwen_code). Qwen's own CLI: native XML tools +
# search/replace edits over /v1/chat/completions — aligned with Qwen's training, unlike
# codex's V4A apply_patch. ENTRYPOINT of qwen-code-runner:v1 is `qwen`, so we append its
# flags; the prompt is appended as the LAST argv element = the value of the trailing `-p`
# (works for both the local argv path and the remote shlex.quote path). --yolo auto-approves
# all edits/shell; --max-session-turns bounds a task (no harness wall) so a stuck task can't
# hang the serial queue. Sampling (temp 0.6) is forced proxy-side on /v1/chat/completions.
QWEN_CODE_TEMPLATE = (
    "docker run --rm --name {container_name} --network=host -u 1000:1000 "
    "-v {workspace}:/workspace:rw "
    "-e OPENAI_API_KEY -e OPENAI_BASE_URL={endpoint} "
    "-e OPENAI_MODEL={model} -e QWEN_MODEL={model} -e HOME=/tmp "
    # R1 context-budget fix (FR13_CONTEXT_COMPRESSION_DESIGN, verified in-image v0.19.4):
    # qwen-code reserves max(ESCALATED_MAX_TOKENS=64000, outputTokenLimit=65536) output tokens
    # unless overridden -> contextLimit = 131072-65536 = 65536 -> hard limit 48875.2. The proxy
    # caps real output at LUMO_PROXY_MAX_OUTPUT_TOKENS=32768, so reserve exactly that:
    # contextLimit=98304 -> hard limit 75304 (+54%). Uniform across all arms (eval-fair).
    "-e QWEN_CODE_MAX_OUTPUT_TOKENS=32768 "
    # FR13 §59 belt: raise qwen-code's stream-idle abort (built-in default 120000ms)
    # so a transient mid-stream upstream flake (GB10 emit/transport wedge) is not
    # converted into a fatal patch-less give-up. qwen-code 0.19.4 reads
    # QWEN_STREAM_IDLE_TIMEOUT_MS (env precedence: config field > env > default).
    # Modest raise (240000ms) paired with the runner stall-watchdog; on the live
    # ssh/offload path ${VAR:-240000} shell-expands remotely. Overridable via env.
    # NOTE: double braces — this string goes through str.format(); single braces
    # parse as a replacement field and crash dispatch (KeyError, tcfix_i5).
    "-e QWEN_STREAM_IDLE_TIMEOUT_MS=${{QWEN_STREAM_IDLE_TIMEOUT_MS:-240000}} "
    "-w /workspace qwen-code-runner:v1 "
    # user 2026-07-07: NO turn limit (100000 = effectively unlimited); tasks run to natural
    # submit/give-up. Backstop = 600s stall-watchdog + QWEN_STREAM_IDLE_TIMEOUT_MS, NOT a turn cap.
    "--yolo --output-format stream-json --max-session-turns 100000 "
    "--session-id {session_id} -p"
)


def _selected_swe_agent() -> str:
    return os.environ.get("SWE_AGENT", "qwen_code").strip().lower()


def _agent_template() -> str:
    """Pick the agent command template from SWE_AGENT (default 'qwen_code' — user 2026-07-07).
    Codex has a built-in nudge/auto-continue that suppresses give-ups and CONFOUNDS the give-up
    gate; qwen-code is the honest agent (native XML tools, aligned with Qwen training)."""
    agent = _selected_swe_agent()
    if agent in ("qwen_code", "qwen-code", "qwen"):
        return QWEN_CODE_TEMPLATE
    return CODEX_TEMPLATE


# --- FR13 §58: SWE_AGENT_ENV=instance_image (default unset='legacy') -----------
# When unset/'legacy' EVERY code path below is byte-identical to the current
# behavior. When ='instance_image', the qwen-code agent runs INSIDE the official
# SWE-bench per-instance eval image editing /testbed (a real git checkout at
# base_commit with the conda 'testbed' env), so it can `import astropy` + run a
# repro pytest before finishing — the env the grader already uses. The node+qwen
# runtime is injected read-only from a relocatable host bundle (built by
# scripts/prepare_qwen_agent_bundle.sh) so NO per-instance image is rebuilt.
def _swe_agent_env() -> str:
    """Agent environment. DEFAULT = 'instance_image' (user 2026-07-06): the agent
    runs INSIDE the official SWE-bench per-instance eval image with the conda
    'testbed' env — the benchmark-faithful setup (smoke-proven §67: import astropy
    editable + qwen 0.19.4 in-image). Requires the per-instance image on the codex
    host + the Qwen agent bundle (both provisioned). Fixed32 uses the
    full-tree-pinned cap-256 derivation produced by
    fr13_derive_qwen_agent_bundle_cap256.py. Set SWE_AGENT_ENV=legacy (or
    worktree) to fall back to the old
    qwen-code-runner:v1-over-bare-worktree behavior (which cannot self-verify, §58).
    A missing image FAILS LOUD per-instance (never silently falls back)."""
    val = os.environ.get("SWE_AGENT_ENV", "instance_image").strip().lower()
    if val in ("legacy", "worktree", "0", "off", "none"):
        return "legacy"
    return "instance_image"


def _validate_fixed32_retry_policy() -> None:
    retry_count = os.environ.get("SWE_EMPTY_PATCH_RETRIES", "0")
    if retry_count != "0":
        raise Fixed32BoundaryError(
            "fixed32 requires SWE_EMPTY_PATCH_RETRIES=0 exactly; "
            f"observed {retry_count!r}"
        )


def _validate_fixed32_agent_runtime_mode(
    *,
    remote_host: str | None,
) -> None:
    """Reject fixed32 launchers that cannot mount the attested Qwen settings."""
    if _swe_agent_env() != "instance_image":
        raise Fixed32BoundaryError(
            "fixed32 requires SWE_AGENT_ENV=instance_image; legacy/worktree "
            "launchers cannot produce acceptance artifacts"
        )
    if _selected_swe_agent() not in ("qwen_code", "qwen-code", "qwen"):
        raise Fixed32BoundaryError(
            "fixed32 requires SWE_AGENT=qwen_code so the pinned Qwen runtime "
            "and system settings can be attested"
        )
    if remote_host != _FIXED32_AGENT_HOST_ALIAS:
        raise Fixed32BoundaryError(
            "fixed32 requires --agent-host alienware exactly; local agents "
            "and alternate host aliases cannot produce canonical runtime "
            "attestations"
        )
    _validate_fixed32_retry_policy()


def _instance_image_name(instance_id: str, *, arch: str = "x86_64",
                         namespace: str = "swebench", tag: str = "latest") -> str:
    """Per-instance SWE-bench eval image name, derived via swebench's OWN naming
    util (TestSpec.instance_image_key) rather than hand-rolled — so it is
    byte-identical to what the eval targets. The eval (swe_eval_x86_worker.py)
    calls make_test_spec(namespace='swebench', instance_image_tag='latest') whose
    arch defaults to 'x86_64' (the codex host, alienware, is x86_64).
    instance_image_key reads ONLY instance_id/arch/namespace/tag, so a minimal
    TestSpec reproduces the exact name (incl. the '__'->'_1776_' remap + .lower()).
    Fails loud (ImportError) if swebench is not installed on the codex host."""
    from swebench.harness.test_spec.test_spec import TestSpec
    spec = TestSpec(
        instance_id=instance_id, repo="", version="", repo_script_list=[],
        eval_script_list=[], env_script_list=[], arch=arch,
        FAIL_TO_PASS=[], PASS_TO_PASS=[], language="py", docker_specs={},
        namespace=namespace, instance_image_tag=tag,
    )
    return spec.instance_image_key


def _host_arch(remote_host: str | None) -> str:
    """uname -m of the AGENT host (local when remote_host is None). Returns the
    raw machine string ('x86_64' | 'aarch64' | ...)."""
    if remote_host:
        r = _net_retry(["ssh", *_EVAL_SSH_OPTS, remote_host, "uname -m"],
                       what=f"host_arch:{remote_host}", timeout=30, max_attempts=2)
        return (r.stdout or "").strip() or "unknown"
    return os.uname().machine


# uname -m -> the arch token swebench uses in image names.
_SWEB_ARCH = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "arm64", "arm64": "arm64"}


def _record_image_arch_tag(workspace: Path, instance_id: str, entry: dict) -> None:
    """USER 2026-07-05: some official SWE-bench per-instance images are x86_64-only
    (fine for the alienware offload, NOT runnable on GB10/aarch64) — persist a
    per-instance arch tag so FUTURE runs know placement constraints without
    rediscovering them. Sidecar: <workspace>/../images_arch.json (merge-update);
    the same fields also ride the returned codex_meta into runner_metadata.json."""
    try:
        sidecar = Path(workspace).parent / "images_arch.json"
        data: dict = {}
        if sidecar.is_file():
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        data[instance_id] = entry
        sidecar.write_text(json.dumps(data, indent=1, sort_keys=True),
                           encoding="utf-8")
    except Exception:
        pass  # tag is observability; never fail the run for it


# Fully-explicit container PATH (NO host '$PATH' dependency — a literal '$PATH'
# would survive shlex.split on the local argv path and land in the container
# verbatim, and would expand to the WRONG host on the ssh path). Puts the qwen
# shim + the conda 'testbed' env (import astropy / pytest) ahead of conda base +
# the standard ubuntu PATH the eval image ships.
_INSTANCE_CONTAINER_PATH = (
    "/opt/qwen/bin:/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
# In-container wrapper. Contains NO single quotes so it survives `bash -c
# '<wrapper>'` under BOTH shlex.split (local argv) AND the remote ssh login
# shell. It: (1) materializes AGENTS.md into /testbed from a base64 env
# (brace/quote-safe for any problem statement), (2) runs qwen whose
# --output-format stream-json emits NDJSON events live (one JSON object per line,
# flushed as each event occurs) as the ONLY stdout -> the trace. This replaces the
# old buffered `json` single-array framing, which did one giant process.stdout.write
# at process.exit() that drained only ONE 64KiB pipe buffer -> every >64KB trace was
# clamped to exactly 65536 bytes, cut mid-string, losing the terminal result record.
# NDJSON also matches the .jsonl filename and lets the byte-growth watchdog see real
# streaming growth, (3) extracts the patch with the SAME git flags as legacy
# _extract_patch, INSIDE /testbed (a real git checkout at base_commit — the
# secondary broken-.git finding does not apply here), writing it to the /out
# bind-mount (a fresh dir, NOT /testbed, so it never pollutes the diff; AGENTS.md
# is untracked so `git diff <base_commit>` excludes it exactly as legacy does),
# (4) preserves qwen's exit code.
_REMOTE_AGENT_TRACE_FILENAME = "qwen_trace.jsonl"
_INSTANCE_TRACE_OUTPUT_PATH = f"/out/{_REMOTE_AGENT_TRACE_FILENAME}"
_INSTANCE_WRAPPER = (
    "printf %s \"$SWE_AGENTS_B64\" | base64 -d > /testbed/AGENTS.md; "
    "PROMPT=$(printf %s \"$SWE_PROMPT_B64\" | base64 -d); "
    "/opt/qwen/bin/qwen --yolo --output-format stream-json "
    "--max-session-turns 100000 --session-id \"$SWE_SESSION_ID\" "
    "-p \"$PROMPT\"; "
    "rc=$?; "
    # Diff against HEAD, NOT base_commit. In the SWE-bench per-instance image /testbed
    # HEAD is `<base_commit> + the committed env-setup commit` (efa06c664 "SWE-bench";
    # e.g. it pins setuptools==68.0.0 in pyproject.toml). `git diff <base_commit>` would
    # capture that env-setup delta (~305B) as part of the patch (polluting/masking the
    # agent's real edits, breaking eval) — the §73 matrix-abort bug. The env-setup is
    # COMMITTED at HEAD, so `git diff HEAD` yields ONLY the agent's uncommitted source
    # edits (matches the gold-patch convention; eval applies on top of the same
    # env-setup state). AGENTS.md is untracked so still excluded.
    "git -C /testbed diff --no-color --binary HEAD > /out/patch.diff 2>/dev/null; "
    "exit $rc"
)


def _instance_wrapper(*, trace_output_path: str | None) -> str:
    if trace_output_path is None:
        return _INSTANCE_WRAPPER
    if trace_output_path != _INSTANCE_TRACE_OUTPUT_PATH:
        raise Fixed32BoundaryError("instance trace output path is not canonical")
    qwen_start = "/opt/qwen/bin/qwen --yolo --output-format stream-json "
    qwen_end = '-p "$PROMPT"; '
    if (
        _INSTANCE_WRAPPER.count(qwen_start) != 1
        or _INSTANCE_WRAPPER.count(qwen_end) != 1
    ):
        raise Fixed32BoundaryError("instance Qwen wrapper shape is unexpected")
    wrapper = _INSTANCE_WRAPPER.replace(
        qwen_start,
        (
            f'test -f {trace_output_path} && test ! -L {trace_output_path} '
            f"|| exit 89; {qwen_start}"
        ),
        1,
    )
    return wrapper.replace(
        qwen_end,
        f'-p "$PROMPT" > {trace_output_path}; ',
        1,
    )


def _fixed32_canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _fixed32_load_json_object(raw: str, *, label: str) -> dict[str, Any]:
    def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for key, value in pairs:
            if key in parsed:
                raise ValueError(f"duplicate key {key!r}")
            parsed[key] = value
        return parsed

    def _nonfinite(value: str) -> Any:
        raise ValueError(f"nonfinite value {value}")

    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_object,
            parse_constant=_nonfinite,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise Fixed32BoundaryError(
            f"fixed32 {label} is not strict JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise Fixed32BoundaryError(f"fixed32 {label} is not a JSON object")
    return payload


def _validate_fixed32_agent_placement_observation(
    observation: Any,
    *,
    measured_observation: Any,
    remote_host: str | None,
) -> dict[str, Any]:
    if remote_host != _FIXED32_AGENT_HOST_ALIAS:
        raise Fixed32BoundaryError(
            "fixed32 requires the exact canonical agent host alias "
            f"{_FIXED32_AGENT_HOST_ALIAS!r}"
        )
    if observation != _FIXED32_AGENT_HOST_IDENTITY:
        raise Fixed32BoundaryError(
            "fixed32 agent host or Docker daemon identity differs from "
            "the canonical offload placement"
        )
    if measured_observation != _FIXED32_MEASURED_HOST_IDENTITY:
        raise Fixed32BoundaryError(
            "fixed32 measured host or Docker daemon identity differs from "
            "the canonical GB10 placement"
        )
    if observation == measured_observation:
        raise Fixed32BoundaryError(
            "fixed32 agent host must be distinct from the measured GB10"
        )
    return {
        "schema": _FIXED32_AGENT_PLACEMENT_SCHEMA,
        "agent_host_identity": dict(observation),
        "measured_host_identity": dict(measured_observation),
        "identities_distinct": True,
    }


_FIXED32_HOST_IDENTITY_SCRIPT = (
    "import hashlib,json,os,pathlib,socket,subprocess\n"
    "daemon=subprocess.run(['docker','info','--format','{{.ID}}'],"
    "capture_output=True,text=True,check=False)\n"
    "if daemon.returncode != 0 or not daemon.stdout.strip():\n"
    " raise RuntimeError('Docker daemon identity is unavailable')\n"
    "uname=os.uname()\n"
    "digest=lambda value: hashlib.sha256("
    "value.strip().encode('utf-8')).hexdigest()\n"
    "print(json.dumps({"
    "'hostname_sha256':digest(socket.gethostname()),"
    "'system':uname.sysname,'machine':uname.machine,"
    "'kernel':uname.release,"
    "'machine_id_sha256':digest("
    "pathlib.Path('/etc/machine-id').read_text(encoding='utf-8')),"
    "'docker_daemon_id_sha256':digest(daemon.stdout)},"
    "sort_keys=True,separators=(',',':')))\n"
)


def _inspect_fixed32_measured_host_local() -> dict[str, Any]:
    inspected = subprocess.run(
        [sys.executable, "-c", _FIXED32_HOST_IDENTITY_SCRIPT],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if inspected.returncode != 0:
        raise Fixed32BoundaryError(
            "fixed32 cannot attest the measured host and Docker daemon"
        )
    observation = _fixed32_load_json_object(
        inspected.stdout or "",
        label="measured-host placement observation",
    )
    if observation != _FIXED32_MEASURED_HOST_IDENTITY:
        raise Fixed32BoundaryError(
            "fixed32 measured host or Docker daemon identity differs from "
            "the canonical GB10 placement"
        )
    return dict(observation)


def _inspect_fixed32_agent_placement_remote(host: str) -> dict[str, Any]:
    measured_observation = _inspect_fixed32_measured_host_local()
    if host != _FIXED32_AGENT_HOST_ALIAS:
        return _validate_fixed32_agent_placement_observation(
            None,
            measured_observation=measured_observation,
            remote_host=host,
        )
    inspected = _net_retry(
        [
            "ssh",
            *_EVAL_SSH_OPTS,
            host,
            "python3 -c " + shlex.quote(_FIXED32_HOST_IDENTITY_SCRIPT),
        ],
        what="fixed32_agent_placement_attestation",
        timeout=60,
        max_attempts=3,
    )
    if inspected.returncode != 0:
        raise Fixed32BoundaryError(
            "fixed32 cannot attest the canonical agent host and Docker daemon"
        )
    return _validate_fixed32_agent_placement_observation(
        _fixed32_load_json_object(
            inspected.stdout or "",
            label="agent placement observation",
        ),
        measured_observation=measured_observation,
        remote_host=host,
    )


def _fixed32_qwen_settings_metadata(
    path: Path = _FIXED32_QWEN_SETTINGS_PATH,
) -> dict[str, Any]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise Fixed32BoundaryError(
            f"fixed32 Qwen system settings cannot be read: {path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    digest = hashlib.sha256(data).hexdigest()
    if (
        data != _FIXED32_QWEN_SETTINGS_BYTES
        or len(data) != 37
        or digest != _FIXED32_QWEN_SETTINGS_SHA256
    ):
        raise Fixed32BoundaryError(
            "fixed32 Qwen system settings differ from the canonical "
            f"37-byte payload: path={path} bytes={len(data)} sha256={digest}"
        )
    return {
        "source": str(_FIXED32_QWEN_SETTINGS_PATH.relative_to(REPO_ROOT)),
        "bytes": len(data),
        "sha256": digest,
        "container_path": _FIXED32_QWEN_SETTINGS_CONTAINER_PATH,
        "mount_mode": "ro",
        "environment": {
            "name": _FIXED32_QWEN_SETTINGS_ENV,
            "value": _FIXED32_QWEN_SETTINGS_CONTAINER_PATH,
        },
        "remote_file": {
            "mode": _FIXED32_QWEN_SETTINGS_MODE,
            "uid": _FIXED32_QWEN_SETTINGS_UID,
            "gid": _FIXED32_QWEN_SETTINGS_GID,
            "nlink": 1,
            "xattrs": [],
        },
        "enable_auto_skill": False,
    }


_FIXED32_QWEN_BUNDLE_TREE_SCRIPT = r"""
import hashlib
import json
import os
import pathlib
import stat
import sys

schema = sys.argv[2]
roots = json.loads(sys.argv[3])
required_entrypoints = json.loads(sys.argv[4])
root = pathlib.Path(sys.argv[1]).expanduser()
root_resolved = root.resolve(strict=True)
entries = []
stamps = {}
package_chunks = []
package_relative = (
    "npm/lib/node_modules/@qwen-code/qwen-code/package.json"
)


def require_ascii(value, *, label):
    try:
        value.encode("ascii")
    except UnicodeEncodeError as error:
        raise RuntimeError(label + " must be ASCII: " + repr(value)) from error
    return value


def relative_path(path):
    relative = "." if path == root else path.relative_to(root).as_posix()
    return require_ascii(relative, label="Qwen bundle relative path")


def stamp(metadata):
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def reject_xattrs(path):
    attributes = os.listxattr(path, follow_symlinks=False)
    if attributes:
        raise RuntimeError(
            "Qwen bundle entry has extended attributes: "
            + path.as_posix()
        )


def record(path):
    relative = relative_path(path)
    before = path.lstat()
    reject_xattrs(path)
    mode = f"{stat.S_IMODE(before.st_mode):04o}"
    if stat.S_ISDIR(before.st_mode):
        entries.append(
            {"path": relative, "type": "directory", "mode": mode}
        )
    elif stat.S_ISREG(before.st_mode):
        if before.st_nlink != 1:
            raise RuntimeError(
                "Qwen bundle hardlink is forbidden: " + relative
            )
        digest = hashlib.sha256()
        byte_count = 0
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if stamp(opened) != stamp(before):
                raise RuntimeError(
                    "Qwen bundle file changed before hashing: " + relative
                )
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                byte_count += len(chunk)
                if relative == package_relative:
                    package_chunks.append(chunk)
            after_read = os.fstat(descriptor)
            if stamp(after_read) != stamp(opened):
                raise RuntimeError(
                    "Qwen bundle file changed while hashing: " + relative
                )
        finally:
            os.close(descriptor)
        entries.append(
            {
                "path": relative,
                "type": "file",
                "mode": mode,
                "bytes": byte_count,
                "sha256": digest.hexdigest(),
            }
        )
    elif stat.S_ISLNK(before.st_mode):
        if before.st_nlink != 1:
            raise RuntimeError(
                "Qwen bundle symlink has multiple links: " + relative
            )
        target = require_ascii(
            os.readlink(path),
            label="Qwen bundle symlink target",
        )
        if pathlib.PurePath(target).is_absolute():
            raise RuntimeError(
                "Qwen bundle symlink target must be relative: " + relative
            )
        resolved_target = (path.parent / target).resolve(strict=True)
        try:
            resolved_target.relative_to(root_resolved)
        except ValueError as error:
            raise RuntimeError(
                "Qwen bundle symlink escapes its root: " + relative
            ) from error
        entries.append(
            {
                "path": relative,
                "type": "symlink",
                "mode": mode,
                "target": target,
            }
        )
    else:
        raise RuntimeError("unsupported bundle entry type: " + relative)
    after = path.lstat()
    reject_xattrs(path)
    if stamp(after) != stamp(before):
        raise RuntimeError(
            "Qwen bundle entry changed during inspection: " + relative
        )
    stamps[path] = stamp(after)


def walk_error(error):
    raise error


def enumerate_tree_paths():
    observed = ["."]
    for current, dirnames, filenames in os.walk(
        root,
        topdown=True,
        onerror=walk_error,
        followlinks=False,
    ):
        for name in dirnames + filenames:
            require_ascii(name, label="Qwen bundle path component")
        dirnames.sort(key=lambda name: name.encode("ascii"))
        filenames.sort(key=lambda name: name.encode("ascii"))
        current_path = pathlib.Path(current)
        if current_path != root:
            observed.append(relative_path(current_path))
        for name in list(dirnames):
            child = current_path / name
            if stat.S_ISLNK(child.lstat().st_mode):
                observed.append(relative_path(child))
                dirnames.remove(name)
        for name in filenames:
            observed.append(relative_path(current_path / name))
    return sorted(observed, key=lambda relative: relative.encode("ascii"))


record(root)
for current, dirnames, filenames in os.walk(
    root,
    topdown=True,
    onerror=walk_error,
    followlinks=False,
):
    for name in dirnames + filenames:
        require_ascii(name, label="Qwen bundle path component")
    dirnames.sort(key=lambda name: name.encode("ascii"))
    filenames.sort(key=lambda name: name.encode("ascii"))
    current_path = pathlib.Path(current)
    if current_path != root:
        record(current_path)
    for name in list(dirnames):
        child = current_path / name
        if stat.S_ISLNK(child.lstat().st_mode):
            record(child)
            dirnames.remove(name)
    for name in filenames:
        record(current_path / name)

entries.sort(key=lambda entry: entry["path"].encode("ascii"))
paths = [entry["path"] for entry in entries]
if len(paths) != len(set(paths)):
    raise RuntimeError("duplicate path in Qwen bundle manifest")
entry_by_path = {entry["path"]: entry for entry in entries}
entrypoints = {}
for relative in required_entrypoints:
    if relative not in entry_by_path:
        raise RuntimeError("missing Qwen bundle entrypoint: " + relative)
    entrypoints[relative] = entry_by_path[relative]

for path, expected_stamp in stamps.items():
    reject_xattrs(path)
    if stamp(path.lstat()) != expected_stamp:
        raise RuntimeError(
            "Qwen bundle tree changed during inspection: "
            + path.as_posix()
        )

if enumerate_tree_paths() != paths:
    raise RuntimeError("Qwen bundle path set changed during inspection")

content_bytes = sum(entry.get("bytes", 0) for entry in entries)
manifest = {
    "schema": schema,
    "roots": roots,
    "entry_count": len(entries),
    "content_bytes": content_bytes,
    "entries": entries,
}
manifest_raw = json.dumps(
    manifest,
    ensure_ascii=True,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
summary = {
    "entry_count": len(entries),
    "directory_count": sum(
        entry["type"] == "directory" for entry in entries
    ),
    "regular_file_count": sum(
        entry["type"] == "file" for entry in entries
    ),
    "symlink_count": sum(entry["type"] == "symlink" for entry in entries),
    "regular_file_bytes": content_bytes,
    "executable_regular_file_count": sum(
        entry["type"] == "file"
        and bool(int(entry["mode"], 8) & 0o111)
        for entry in entries
    ),
    "manifest_bytes": len(manifest_raw),
}
if not package_chunks:
    raise RuntimeError("Qwen package metadata was not captured")
package = json.loads(b"".join(package_chunks).decode("utf-8"))
print(
    json.dumps(
        {
            "qwen_code_version": package.get("version"),
            "bundle_tree": {
                "schema": schema,
                "roots": roots,
                "summary": summary,
                "entrypoints": entrypoints,
                "manifest_sha256": hashlib.sha256(
                    manifest_raw
                ).hexdigest(),
            },
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
)
"""


def _validate_fixed32_qwen_bundle_observation(
    observation: Any,
) -> dict[str, Any]:
    if not isinstance(observation, dict) or set(observation) != {
        "qwen_code_version",
        "bundle_tree",
    }:
        raise Fixed32BoundaryError(
            "fixed32 Qwen bundle observation has the wrong top-level fields"
        )
    if observation["qwen_code_version"] != _FIXED32_QWEN_CODE_VERSION:
        raise Fixed32BoundaryError(
            "fixed32 Qwen bundle version mismatch: "
            f"expected={_FIXED32_QWEN_CODE_VERSION} "
            f"observed={observation['qwen_code_version']!r}"
        )
    tree = observation["bundle_tree"]
    if tree != _FIXED32_QWEN_BUNDLE_TREE_EXPECTED:
        observed_digest = (
            tree.get("manifest_sha256") if isinstance(tree, dict) else None
        )
        raise Fixed32BoundaryError(
            "fixed32 Qwen executable bundle-tree manifest mismatch: "
            f"expected={_FIXED32_QWEN_BUNDLE_TREE_SHA256} "
            f"observed={observed_digest!r}"
        )
    return {
        "qwen_code_version": _FIXED32_QWEN_CODE_VERSION,
        "bundle_tree": json.loads(
            json.dumps(
                _FIXED32_QWEN_BUNDLE_TREE_EXPECTED,
                ensure_ascii=True,
                sort_keys=True,
            )
        ),
    }


def _observe_fixed32_qwen_bundle_local(bundle_root: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _FIXED32_QWEN_BUNDLE_TREE_SCRIPT,
            str(bundle_root),
            _FIXED32_QWEN_BUNDLE_TREE_SCHEMA,
            json.dumps(
                _FIXED32_QWEN_BUNDLE_TREE_ROOTS,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
            json.dumps(
                _FIXED32_QWEN_BUNDLE_TREE_REQUIRED_ENTRYPOINTS,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise Fixed32BoundaryError(
            "fixed32 cannot inspect the local Qwen executable bundle tree at "
            f"{bundle_root}: rc={completed.returncode} "
            f"stderr={(completed.stderr or '').strip()!r}"
        )
    return _fixed32_load_json_object(
        completed.stdout or "",
        label="local Qwen executable bundle-tree observation",
    )


def _inspect_fixed32_qwen_bundle_local(bundle_root: Path) -> dict[str, Any]:
    return _validate_fixed32_qwen_bundle_observation(
        _observe_fixed32_qwen_bundle_local(bundle_root)
    )


def _inspect_fixed32_qwen_bundle_remote_path(
    host: str,
    bundle_path: str,
) -> dict[str, Any]:
    command = (
        "python3 -c "
        + shlex.quote(_FIXED32_QWEN_BUNDLE_TREE_SCRIPT)
        + " "
        + shlex.quote(bundle_path)
        + " "
        + shlex.quote(_FIXED32_QWEN_BUNDLE_TREE_SCHEMA)
        + " "
        + shlex.quote(
            json.dumps(
                _FIXED32_QWEN_BUNDLE_TREE_ROOTS,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        + " "
        + shlex.quote(
            json.dumps(
                _FIXED32_QWEN_BUNDLE_TREE_REQUIRED_ENTRYPOINTS,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    )
    completed = _net_retry(
        ["ssh", *_EVAL_SSH_OPTS, host, command],
        what="fixed32_qwen_snapshot_tree_attestation",
        timeout=300,
        max_attempts=3,
    )
    if completed.returncode != 0:
        raise Fixed32BoundaryError(
            "fixed32 cannot inspect the remote Qwen executable bundle tree: "
            f"host={host} rc={completed.returncode}"
        )
    observation = _fixed32_load_json_object(
        completed.stdout or "",
        label="remote Qwen executable bundle-tree observation",
    )
    return _validate_fixed32_qwen_bundle_observation(observation)


def _inspect_fixed32_qwen_bundle_remote(host: str) -> dict[str, Any]:
    return _inspect_fixed32_qwen_bundle_remote_path(
        host,
        _FIXED32_QWEN_BUNDLE_REMOTE_PATH,
    )


def _create_fixed32_qwen_snapshot_remote(
    *,
    host: str,
    instance_id: str,
    task_root: str,
) -> tuple[str, dict[str, Any]]:
    staging = f"{task_root}/.qwen_bundle_snapshot"
    copied = _net_retry(
        [
            "ssh",
            *_EVAL_SSH_OPTS,
            host,
            (
                "set -eu; umask 077; source=$HOME/"
                + shlex.quote(_FIXED32_QWEN_BUNDLE_REMOTE_BASENAME)
                + "; "
                'test -d "$source"; test ! -L "$source"; '
                f"test ! -e {staging}; test ! -L {staging}; "
                f'cp -a --reflink=auto -- "$source" {staging}; '
                f"test -d {staging}; test ! -L {staging}"
            ),
        ],
        what=f"fixed32_qwen_snapshot_copy:{instance_id}",
        timeout=300,
        max_attempts=1,
    )
    if copied.returncode != 0:
        raise Fixed32BoundaryError(
            "fixed32 Qwen bundle snapshot creation failed"
        )
    staging_observation = _inspect_fixed32_qwen_bundle_remote_path(
        host,
        staging,
    )
    manifest_sha256 = staging_observation["bundle_tree"]["manifest_sha256"]
    snapshot = f"{task_root}/qwen_bundle-{manifest_sha256}"
    promoted = _net_retry(
        [
            "ssh",
            *_EVAL_SSH_OPTS,
            host,
            (
                "set -eu; "
                f"test -d {staging}; test ! -L {staging}; "
                f"test ! -e {snapshot}; test ! -L {snapshot}; "
                f"mv -- {staging} {snapshot}; "
                f"test -d {snapshot}; test ! -L {snapshot}"
            ),
        ],
        what=f"fixed32_qwen_snapshot_promote:{instance_id}",
        timeout=60,
        max_attempts=1,
    )
    if promoted.returncode != 0:
        raise Fixed32BoundaryError(
            "fixed32 Qwen bundle snapshot promotion failed"
        )
    snapshot_observation = _inspect_fixed32_qwen_bundle_remote_path(
        host,
        snapshot,
    )
    if snapshot_observation != staging_observation:
        raise Fixed32BoundaryError(
            "fixed32 Qwen bundle snapshot changed during promotion"
        )
    return snapshot, snapshot_observation


def _validate_fixed32_agent_image_observation(
    observation: Any,
    *,
    instance_id: str,
    expected_image: str,
) -> dict[str, Any]:
    expected = _FIXED32_AGENT_IMAGE_IDENTITIES.get(instance_id)
    if expected is None:
        raise Fixed32BoundaryError(
            f"fixed32 has no pinned agent image for {instance_id}"
        )
    canonical_image = expected["repo_digest"].split("@", 1)[0] + ":latest"
    if expected_image != canonical_image:
        raise Fixed32BoundaryError(
            "fixed32 agent image tag differs from the canonical "
            f"SWE-Verified image: instance={instance_id} "
            f"expected={canonical_image!r} observed={expected_image!r}"
        )
    canonical = {
        "instance_id": instance_id,
        "image": canonical_image,
        "id": expected["id"],
        "repo_digest": expected["repo_digest"],
        "architecture": "amd64",
        "os": "linux",
    }
    if observation != canonical:
        raise Fixed32BoundaryError(
            "fixed32 agent image identity differs from the canonical "
            f"SWE-Verified image: instance={instance_id} "
            f"observed={observation!r}"
        )
    return canonical


def _inspect_fixed32_agent_image_remote(
    *,
    host: str,
    instance_id: str,
    image: str,
) -> dict[str, Any]:
    script = (
        "import json,subprocess,sys\n"
        "run=subprocess.run(['docker','image','inspect',sys.argv[1]],"
        "capture_output=True,text=True,check=False)\n"
        "if run.returncode != 0:\n"
        " raise SystemExit(run.returncode)\n"
        "payload=json.loads(run.stdout)\n"
        "if not isinstance(payload,list) or len(payload)!=1:\n"
        " raise RuntimeError('docker image inspect cardinality differs')\n"
        "item=payload[0]\n"
        "digests=item.get('RepoDigests')\n"
        "if not isinstance(digests,list) or len(digests)!=1:\n"
        " raise RuntimeError('agent image must have one RepoDigest')\n"
        "print(json.dumps({'instance_id':sys.argv[2],'image':sys.argv[1],"
        "'id':item.get('Id'),'repo_digest':digests[0],"
        "'architecture':item.get('Architecture'),'os':item.get('Os')},"
        "sort_keys=True,separators=(',',':')))\n"
    )
    inspected = _net_retry(
        [
            "ssh",
            *_EVAL_SSH_OPTS,
            host,
            (
                "python3 -c "
                + shlex.quote(script)
                + " "
                + shlex.quote(image)
                + " "
                + shlex.quote(instance_id)
            ),
        ],
        what=f"fixed32_agent_image_attestation:{instance_id}",
        timeout=60,
        max_attempts=3,
    )
    if inspected.returncode != 0:
        raise Fixed32BoundaryError(
            "fixed32 cannot inspect the remote agent image: "
            f"host={host} instance={instance_id} rc={inspected.returncode}"
        )
    return _validate_fixed32_agent_image_observation(
        _fixed32_load_json_object(
            inspected.stdout or "",
            label=f"remote agent image observation {instance_id}",
        ),
        instance_id=instance_id,
        expected_image=image,
    )


_FIXED32_QWEN_SETTINGS_REMOTE_SCRIPT = r"""
import base64
import hashlib
import json
import os
import pathlib
import stat
import sys

action = sys.argv[1]
path = pathlib.Path(sys.argv[2]).expanduser()
expected_data = base64.b64decode(sys.argv[3], validate=True)
expected_mode = int(sys.argv[4], 8)
expected_uid = int(sys.argv[5])
expected_gid = int(sys.argv[6])
parent = path.parent
parent_metadata = parent.lstat()
if not stat.S_ISDIR(parent_metadata.st_mode):
    raise RuntimeError("settings parent is not a directory")
if parent.is_symlink():
    raise RuntimeError("settings parent is a symlink")
parent_fd = os.open(
    parent,
    os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
)


def identity(metadata):
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


try:
    if action == "create":
        try:
            os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError("settings destination is not fresh")
        descriptor = os.open(
            path.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | os.O_NOFOLLOW,
            expected_mode,
            dir_fd=parent_fd,
        )
        try:
            view = memoryview(expected_data)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise RuntimeError("settings write made no progress")
                view = view[written:]
            os.fchmod(descriptor, expected_mode)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    elif action != "verify":
        raise RuntimeError("unsupported settings operation")

    before = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise RuntimeError("settings path is not a single-link regular file")
    if (
        stat.S_IMODE(before.st_mode) != expected_mode
        or before.st_uid != expected_uid
        or before.st_gid != expected_gid
    ):
        raise RuntimeError("settings mode or owner is noncanonical")
    if os.listxattr(path, follow_symlinks=False):
        raise RuntimeError("settings path has extended attributes")
    descriptor = os.open(
        path.name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=parent_fd,
    )
    try:
        opened = os.fstat(descriptor)
        if identity(opened) != identity(before):
            raise RuntimeError("settings path changed before exact read")
        chunks = []
        while True:
            chunk = os.read(descriptor, 4096)
            if not chunk:
                break
            chunks.append(chunk)
        after_read = os.fstat(descriptor)
        if identity(after_read) != identity(opened):
            raise RuntimeError("settings path changed during exact read")
    finally:
        os.close(descriptor)
    after = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
    if identity(after) != identity(before):
        raise RuntimeError("settings path changed after exact read")
    if os.listxattr(path, follow_symlinks=False):
        raise RuntimeError("settings path gained extended attributes")
    data = b"".join(chunks)
    if data != expected_data:
        raise RuntimeError("settings bytes differ from canonical payload")
    print(
        json.dumps(
            {
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "mode": f"{stat.S_IMODE(after.st_mode):04o}",
                "uid": after.st_uid,
                "gid": after.st_gid,
                "nlink": after.st_nlink,
                "xattrs": [],
                "file_identity_sha256": hashlib.sha256(
                    f"{after.st_dev}:{after.st_ino}".encode("ascii")
                ).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
finally:
    os.close(parent_fd)
"""


def _fixed32_expected_remote_settings_observation() -> dict[str, Any]:
    return {
        "bytes": len(_FIXED32_QWEN_SETTINGS_BYTES),
        "sha256": _FIXED32_QWEN_SETTINGS_SHA256,
        "mode": _FIXED32_QWEN_SETTINGS_MODE,
        "uid": _FIXED32_QWEN_SETTINGS_UID,
        "gid": _FIXED32_QWEN_SETTINGS_GID,
        "nlink": 1,
        "xattrs": [],
    }


def _fixed32_qwen_settings_remote_command(
    *, action: str, remote_path: str
) -> str:
    return " ".join(
        (
            "python3",
            "-c",
            shlex.quote(_FIXED32_QWEN_SETTINGS_REMOTE_SCRIPT),
            shlex.quote(action),
            shlex.quote(remote_path),
            shlex.quote(
                base64.b64encode(_FIXED32_QWEN_SETTINGS_BYTES).decode("ascii")
            ),
            shlex.quote(_FIXED32_QWEN_SETTINGS_MODE),
            str(_FIXED32_QWEN_SETTINGS_UID),
            str(_FIXED32_QWEN_SETTINGS_GID),
        )
    )


def _validate_fixed32_remote_settings_observation(
    payload: Any,
) -> dict[str, Any]:
    expected = _fixed32_expected_remote_settings_observation()
    if not isinstance(payload, dict):
        raise Fixed32BoundaryError(
            "fixed32 remote Qwen system-settings identity differs"
        )
    identity_digest = payload.get("file_identity_sha256")
    static_payload = {
        key: value
        for key, value in payload.items()
        if key != "file_identity_sha256"
    }
    if (
        static_payload != expected
        or not isinstance(identity_digest, str)
        or len(identity_digest) != 64
        or any(character not in "0123456789abcdef" for character in identity_digest)
    ):
        raise Fixed32BoundaryError(
            "fixed32 remote Qwen system-settings identity differs"
        )
    return {**expected, "file_identity_sha256": identity_digest}


def _require_fixed32_remote_settings_stable(
    before: Any,
    after: Any,
) -> dict[str, Any]:
    canonical_before = _validate_fixed32_remote_settings_observation(before)
    canonical_after = _validate_fixed32_remote_settings_observation(after)
    if canonical_after != canonical_before:
        raise Fixed32BoundaryError(
            "fixed32 Qwen settings file identity changed"
        )
    return canonical_before


def _install_fixed32_qwen_settings_remote(
    *,
    host: str,
    remote_path: str,
) -> dict[str, Any]:
    installed = _net_retry(
        [
            "ssh",
            *_EVAL_SSH_OPTS,
            host,
            _fixed32_qwen_settings_remote_command(
                action="create",
                remote_path=remote_path,
            ),
        ],
        what="fixed32_qwen_settings_install",
        timeout=60,
        max_attempts=1,
    )
    if installed.returncode != 0:
        raise Fixed32BoundaryError(
            "fixed32 Qwen system-settings exclusive install failed"
        )
    return _validate_fixed32_remote_settings_observation(
        _fixed32_load_json_object(
            installed.stdout or "",
            label="remote Qwen system-settings install",
        )
    )


def _verify_fixed32_qwen_settings_remote(
    *,
    host: str,
    remote_path: str,
) -> dict[str, Any]:
    verified = _net_retry(
        [
            "ssh",
            *_EVAL_SSH_OPTS,
            host,
            _fixed32_qwen_settings_remote_command(
                action="verify",
                remote_path=remote_path,
            ),
        ],
        what="fixed32_qwen_settings_rehash",
        timeout=60,
        max_attempts=3,
    )
    if verified.returncode != 0:
        raise Fixed32BoundaryError(
            "fixed32 Qwen system-settings remote verification failed"
        )
    return _validate_fixed32_remote_settings_observation(
        _fixed32_load_json_object(
            verified.stdout or "",
            label="remote Qwen system-settings verification",
        )
    )


def _build_fixed32_qwen_runtime_attestation(
    *,
    bundle_observation: Any,
    host_mode: str,
) -> dict[str, Any]:
    normalized_bundle = _validate_fixed32_qwen_bundle_observation(
        bundle_observation
    )
    if host_mode != "remote":
        raise Fixed32BoundaryError(
            "fixed32 Qwen attestation requires the remote agent host, "
            f"got: {host_mode!r}"
        )
    return {
        "schema": _FIXED32_QWEN_ATTESTATION_SCHEMA,
        "launcher": "qwen-code-instance-image",
        "agent_env": "instance_image",
        "host_mode": host_mode,
        "qwen_code_version": _FIXED32_QWEN_CODE_VERSION,
        "bundle_tree": normalized_bundle["bundle_tree"],
        "bundle_manifest_sha256": normalized_bundle["bundle_tree"][
            "manifest_sha256"
        ],
        "bundle_snapshot": {
            "kind": "per-task-content-addressed-snapshot",
            "basename": (
                "qwen_bundle-"
                + normalized_bundle["bundle_tree"]["manifest_sha256"]
            ),
            "container_path": "/opt/qwen",
            "mount_mode": "ro",
        },
        "cleared_agent_environment": list(_FIXED32_CLEARED_AGENT_ENV),
        "system_settings": _fixed32_qwen_settings_metadata(),
    }


def _validate_fixed32_qwen_runtime_attestation(
    attestation: Any,
) -> str:
    if not isinstance(attestation, dict) or set(attestation) != {
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
    }:
        raise Fixed32BoundaryError(
            "fixed32 Qwen runtime attestation has the wrong fields"
        )
    if (
        attestation["schema"] != _FIXED32_QWEN_ATTESTATION_SCHEMA
        or attestation["launcher"] != "qwen-code-instance-image"
        or attestation["agent_env"] != "instance_image"
        or attestation["host_mode"] != "remote"
        or attestation["qwen_code_version"] != _FIXED32_QWEN_CODE_VERSION
    ):
        raise Fixed32BoundaryError(
            "fixed32 Qwen runtime attestation identity is invalid"
        )
    normalized_bundle = _validate_fixed32_qwen_bundle_observation(
        {
            "qwen_code_version": attestation["qwen_code_version"],
            "bundle_tree": attestation["bundle_tree"],
        }
    )
    expected_manifest = normalized_bundle["bundle_tree"]["manifest_sha256"]
    if attestation["bundle_manifest_sha256"] != expected_manifest:
        raise Fixed32BoundaryError(
            "fixed32 Qwen runtime bundle-manifest digest mismatch"
        )
    if attestation["bundle_snapshot"] != {
        "kind": "per-task-content-addressed-snapshot",
        "basename": "qwen_bundle-" + expected_manifest,
        "container_path": "/opt/qwen",
        "mount_mode": "ro",
    }:
        raise Fixed32BoundaryError(
            "fixed32 Qwen runtime snapshot identity differs"
        )
    if attestation["cleared_agent_environment"] != list(
        _FIXED32_CLEARED_AGENT_ENV
    ):
        raise Fixed32BoundaryError(
            "fixed32 Qwen runtime injection environment is not cleared"
        )
    if attestation["system_settings"] != _fixed32_qwen_settings_metadata():
        raise Fixed32BoundaryError(
            "fixed32 Qwen runtime attestation has noncanonical system settings"
        )
    return _fixed32_canonical_json_sha256(attestation)


def _persist_fixed32_qwen_runtime_attestation(
    *,
    workspace: Path,
    attestation: dict[str, Any],
    filename: str = "qwen_runtime_attestation.json",
) -> str:
    digest = _validate_fixed32_qwen_runtime_attestation(attestation)
    path = workspace.parent / filename
    path.write_text(
        json.dumps(attestation, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return digest


_FIXED32_MOUNTED_RUNTIME_PROOF_SCRIPT = r"""
import base64
import errno
import hashlib
import json
import os
import pathlib
import stat
import sys


def strict_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def reject_nonfinite(value):
    raise ValueError("nonfinite JSON value")


def load_json(path):
    return json.loads(
        pathlib.Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=strict_object,
        parse_constant=reject_nonfinite,
    )


def identity(metadata):
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def require_read_only_create(path):
    descriptor = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
    except OSError as error:
        if error.errno != errno.EROFS:
            raise RuntimeError("bundle mount write probe was not EROFS") from error
        return error.errno
    else:
        os.close(descriptor)
        pathlib.Path(path).unlink(missing_ok=True)
        raise RuntimeError("bundle mount accepted a write probe")


def require_read_only_file(path):
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_NOFOLLOW)
    except OSError as error:
        if error.errno != errno.EROFS:
            raise RuntimeError("settings mount write probe was not EROFS") from error
        return error.errno
    else:
        os.close(descriptor)
        raise RuntimeError("settings mount accepted a write probe")


observation = load_json(sys.argv[1])
expected_observation = load_json(sys.argv[2])
if observation != expected_observation:
    raise RuntimeError("mounted Qwen tree differs from the task snapshot")

settings_path = pathlib.Path(sys.argv[3])
expected_data = base64.b64decode(sys.argv[4], validate=True)
expected_mode = int(sys.argv[5], 8)
expected_uid = int(sys.argv[6])
expected_gid = int(sys.argv[7])
before = settings_path.lstat()
if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
    raise RuntimeError("mounted settings is not a single-link regular file")
if (
    stat.S_IMODE(before.st_mode) != expected_mode
    or before.st_uid != expected_uid
    or before.st_gid != expected_gid
):
    raise RuntimeError("mounted settings mode or owner differs")
if os.listxattr(settings_path, follow_symlinks=False):
    raise RuntimeError("mounted settings has extended attributes")
descriptor = os.open(
    settings_path,
    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
)
try:
    opened = os.fstat(descriptor)
    if identity(opened) != identity(before):
        raise RuntimeError("mounted settings changed before exact read")
    chunks = []
    while True:
        chunk = os.read(descriptor, 4096)
        if not chunk:
            break
        chunks.append(chunk)
    after_read = os.fstat(descriptor)
    if identity(after_read) != identity(opened):
        raise RuntimeError("mounted settings changed during exact read")
finally:
    os.close(descriptor)
after = settings_path.lstat()
if identity(after) != identity(before):
    raise RuntimeError("mounted settings changed after exact read")
if os.listxattr(settings_path, follow_symlinks=False):
    raise RuntimeError("mounted settings gained extended attributes")
data = b"".join(chunks)
if data != expected_data:
    raise RuntimeError("mounted settings bytes differ")

proof = {
    "schema": "fr13-fixed32-qwen-mounted-runtime-proof-v1",
    "bundle_tree": {
        "container_path": "/opt/qwen",
        "mount_mode": "ro",
        "write_probe_errno": require_read_only_create(
            "/opt/qwen/.fr13-fixed32-write-probe"
        ),
        "observation": observation,
    },
    "system_settings": {
        "container_path": str(settings_path),
        "mount_mode": "ro",
        "write_probe_errno": require_read_only_file(settings_path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "mode": f"{stat.S_IMODE(after.st_mode):04o}",
        "uid": after.st_uid,
        "gid": after.st_gid,
        "nlink": after.st_nlink,
        "xattrs": [],
        "file_identity_sha256": hashlib.sha256(
            f"{after.st_dev}:{after.st_ino}".encode("ascii")
        ).hexdigest(),
    },
}
print(
    json.dumps(
        proof,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
)
"""


def _validate_fixed32_mounted_runtime_proof(
    proof: Any,
    *,
    expected_bundle_observation: dict[str, Any],
) -> str:
    identity_digest = (
        proof.get("system_settings", {}).get("file_identity_sha256")
        if isinstance(proof, dict)
        and isinstance(proof.get("system_settings"), dict)
        else None
    )
    if (
        not isinstance(identity_digest, str)
        or len(identity_digest) != 64
        or any(character not in "0123456789abcdef" for character in identity_digest)
    ):
        raise Fixed32BoundaryError(
            "fixed32 mounted settings file identity is malformed"
        )
    expected = {
        "schema": _FIXED32_MOUNTED_RUNTIME_PROOF_SCHEMA,
        "bundle_tree": {
            "container_path": "/opt/qwen",
            "mount_mode": "ro",
            "write_probe_errno": 30,
            "observation": expected_bundle_observation,
        },
        "system_settings": {
            "container_path": _FIXED32_QWEN_SETTINGS_CONTAINER_PATH,
            "mount_mode": "ro",
            "write_probe_errno": 30,
            **_fixed32_expected_remote_settings_observation(),
            "file_identity_sha256": identity_digest,
        },
    }
    if proof != expected:
        raise Fixed32BoundaryError(
            "fixed32 mounted Qwen tree or system-settings proof differs"
        )
    return _fixed32_canonical_json_sha256(proof)


def _load_fixed32_mounted_runtime_proof(
    path: Path,
    *,
    expected_bundle_observation: dict[str, Any],
) -> tuple[dict[str, Any], str, str]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise Fixed32BoundaryError(
            "fixed32 mounted-runtime proof cannot be read"
        ) from exc
    proof = _fixed32_load_json_object(
        text,
        label="mounted-runtime proof",
    )
    digest = _validate_fixed32_mounted_runtime_proof(
        proof,
        expected_bundle_observation=expected_bundle_observation,
    )
    canonical_raw = (
        json.dumps(
            proof,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    if raw != canonical_raw:
        raise Fixed32BoundaryError(
            "fixed32 mounted-runtime proof is not canonical JSON"
        )
    return proof, digest, hashlib.sha256(raw).hexdigest()


def _pull_fixed32_mounted_runtime_proof(
    *,
    host: str,
    remote_out: str,
    task_dir: Path,
    expected_bundle_observation: dict[str, Any],
) -> tuple[dict[str, Any], str, str]:
    local_path = task_dir / _FIXED32_MOUNTED_RUNTIME_PROOF_FILENAME
    pulled = _net_retry(
        [
            "scp",
            *_EVAL_SSH_OPTS,
            (
                f"{host}:{remote_out}/"
                f"{_FIXED32_MOUNTED_RUNTIME_PROOF_FILENAME}"
            ),
            str(local_path),
        ],
        what="fixed32_mounted_runtime_proof_download",
        timeout=120,
        max_attempts=3,
    )
    if pulled.returncode != 0:
        raise Fixed32BoundaryError(
            "fixed32 mounted-runtime proof download failed"
        )
    return _load_fixed32_mounted_runtime_proof(
        local_path,
        expected_bundle_observation=expected_bundle_observation,
    )


def _fixed32_runtime_proof_wrapper(
    expected_bundle_observation: dict[str, Any],
) -> tuple[str, str]:
    encoded = {
        "scanner": base64.b64encode(
            _FIXED32_QWEN_BUNDLE_TREE_SCRIPT.encode("utf-8")
        ).decode("ascii"),
        "proof": base64.b64encode(
            _FIXED32_MOUNTED_RUNTIME_PROOF_SCRIPT.encode("utf-8")
        ).decode("ascii"),
        "roots": base64.b64encode(
            json.dumps(
                _FIXED32_QWEN_BUNDLE_TREE_ROOTS,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).decode("ascii"),
        "entrypoints": base64.b64encode(
            json.dumps(
                _FIXED32_QWEN_BUNDLE_TREE_REQUIRED_ENTRYPOINTS,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).decode("ascii"),
        "expected": base64.b64encode(
            (
                json.dumps(
                    expected_bundle_observation,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("ascii")
        ).decode("ascii"),
        "settings": base64.b64encode(
            _FIXED32_QWEN_SETTINGS_BYTES
        ).decode("ascii"),
    }
    environment = " ".join(
        f"-e {name}='{value}'"
        for name, value in (
            ("SWE_QWEN_TREE_SCANNER_B64", encoded["scanner"]),
            ("SWE_QWEN_RUNTIME_PROOF_B64", encoded["proof"]),
            ("SWE_QWEN_TREE_ROOTS_B64", encoded["roots"]),
            ("SWE_QWEN_TREE_ENTRYPOINTS_B64", encoded["entrypoints"]),
            ("SWE_QWEN_EXPECTED_TREE_B64", encoded["expected"]),
            ("SWE_QWEN_SETTINGS_B64", encoded["settings"]),
        )
    )
    wrapper = (
        "umask 077; "
        "printf %s \"$SWE_QWEN_TREE_SCANNER_B64\" | base64 -d "
        "> /tmp/fr13_qwen_tree_scanner.py || exit 81; "
        "printf %s \"$SWE_QWEN_RUNTIME_PROOF_B64\" | base64 -d "
        "> /tmp/fr13_qwen_runtime_proof.py || exit 82; "
        "printf %s \"$SWE_QWEN_TREE_ROOTS_B64\" | base64 -d "
        "> /tmp/fr13_qwen_tree_roots.json || exit 83; "
        "printf %s \"$SWE_QWEN_TREE_ENTRYPOINTS_B64\" | base64 -d "
        "> /tmp/fr13_qwen_tree_entrypoints.json || exit 84; "
        "printf %s \"$SWE_QWEN_EXPECTED_TREE_B64\" | base64 -d "
        "> /tmp/fr13_qwen_expected_tree.json || exit 85; "
        "ROOTS=$(cat /tmp/fr13_qwen_tree_roots.json); "
        "ENTRYPOINTS=$(cat /tmp/fr13_qwen_tree_entrypoints.json); "
        "python3 /tmp/fr13_qwen_tree_scanner.py /opt/qwen "
        f"{_FIXED32_QWEN_BUNDLE_TREE_SCHEMA} \"$ROOTS\" \"$ENTRYPOINTS\" "
        "> /tmp/fr13_qwen_mounted_tree.json || exit 86; "
        "python3 /tmp/fr13_qwen_runtime_proof.py "
        "/tmp/fr13_qwen_mounted_tree.json "
        "/tmp/fr13_qwen_expected_tree.json "
        f"{_FIXED32_QWEN_SETTINGS_CONTAINER_PATH} "
        "\"$SWE_QWEN_SETTINGS_B64\" "
        f"{_FIXED32_QWEN_SETTINGS_MODE} "
        f"{_FIXED32_QWEN_SETTINGS_UID} "
        f"{_FIXED32_QWEN_SETTINGS_GID} "
        f"> /out/{_FIXED32_MOUNTED_RUNTIME_PROOF_FILENAME} || exit 87; "
        f"chmod 0444 /out/{_FIXED32_MOUNTED_RUNTIME_PROOF_FILENAME} "
        "|| exit 88; "
        "umask 022; "
    )
    return environment, wrapper


def _instance_agent_command(*, container_name: str, image: str, endpoint: str,
                            model: str, host_out_dir: str, bundle_src: str,
                            agents_md_b64: str, prompt_b64: str,
                            base_commit: str, session_id: str,
                            system_settings_src: str | None = None,
                            bundle_observation: dict[str, Any] | None = None,
                            trace_output_path: str | None = None) -> str:
    """Render the instance_image docker command. Same qwen flags/env as the
    legacy qwen-code template, but the image is the per-instance eval image, the
    node+qwen runtime is bind-mounted read-only from the host bundle at
    /opt/qwen, PATH is prepended, workdir is /testbed, and AGENTS.md/prompt are
    passed base64 (brace/quote-safe). Pure string builder — self-testable with no
    docker/GPU. Runs as -u 0:0 because /testbed + the conda env are root-owned in
    the eval image."""
    system_settings_args = ""
    runtime_proof_environment = ""
    runtime_proof_wrapper = ""
    if system_settings_src is not None:
        if bundle_observation is None:
            raise Fixed32BoundaryError(
                "fixed32 mounted-runtime proof requires a bundle observation"
            )
        system_settings_args = (
            f"-e {_FIXED32_QWEN_SETTINGS_ENV}="
            f"{_FIXED32_QWEN_SETTINGS_CONTAINER_PATH} "
            f"-v {system_settings_src}:"
            f"{_FIXED32_QWEN_SETTINGS_CONTAINER_PATH}:ro "
        )
        runtime_proof_environment, runtime_proof_wrapper = (
            _fixed32_runtime_proof_wrapper(bundle_observation)
        )
    cleared_environment_args = " ".join(
        f"-e {name}=" for name in _FIXED32_CLEARED_AGENT_ENV
    )
    instance_wrapper = _instance_wrapper(trace_output_path=trace_output_path)
    return (
        f"docker run --rm --name {container_name} --network=host -u 0:0 "
        f"-e OPENAI_API_KEY -e OPENAI_BASE_URL={endpoint} "
        f"-e OPENAI_MODEL={model} -e QWEN_MODEL={model} -e HOME=/tmp "
        f"-e QWEN_CODE_MAX_OUTPUT_TOKENS=32768 "
        f"-e QWEN_STREAM_IDLE_TIMEOUT_MS=600000 "
        f"-e PATH={_INSTANCE_CONTAINER_PATH} "
        f"-e SWE_AGENTS_B64='{agents_md_b64}' -e SWE_PROMPT_B64='{prompt_b64}' "
        f"-e SWE_BASE_COMMIT='{base_commit}' "
        f"-e SWE_SESSION_ID='{session_id}' "
        f"{cleared_environment_args} "
        f"{runtime_proof_environment} "
        f"{system_settings_args}"
        f"-v {host_out_dir}:/out -v {bundle_src}:/opt/qwen:ro "
        f"-w /testbed {image} "
        f"bash -c '{runtime_proof_wrapper}{instance_wrapper}'"
    )

# Default operator prompt (first attempt).
DEFAULT_AGENT_PROMPT = (
    "Read the task prompt at /workspace/AGENTS.md and complete it in this workspace. "
    "Edit the source files directly to implement the fix. Do not write a diff file -- "
    "modify the files in place so that running pytest passes the tests described in the prompt."
)
# Bundle B #2/#8: retry prompt when the first attempt left no patch (agent
# returned without editing source).
RETRY_PROMPT_EMPTY = (
    "Your previous attempt finished WITHOUT leaving any code change in the working tree. "
    "Re-read /workspace/AGENTS.md, inspect the relevant source files, and EDIT them now to "
    "implement the fix. Do not stop until you have made a concrete source edit. Do not waste "
    "time on environment setup or pip/conda installs -- the grader uses its own environment."
)
# Bundle B #3/#8: retry prompt when the first attempt looped on the same failing
# command (e.g. repeated pip/build errors) and left no patch.
RETRY_PROMPT_SETUP_LOOP = (
    "Your previous attempt repeatedly hit the same failing command (likely an environment/"
    "install/build step) and never edited the source. STOP trying that approach entirely. "
    "The grader builds its own environment, so you do NOT need the project to install or import. "
    "Read /workspace/AGENTS.md and the relevant source files, and directly EDIT the source to "
    "implement the fix."
)


def _prior_attempt_brief(trace_path: Path) -> str:
    """Summarize the prior codex attempt (files it inspected + its last message) so an
    agent_gave_up retry can be re-driven IN-CONTEXT rather than re-exploring from scratch."""
    files: list[str] = []
    last_msg = ""
    try:
        for line in trace_path.read_text(errors="replace").splitlines():
            try:
                item = json.loads(line).get("item", {})
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(item, dict):
                continue
            if item.get("type") == "command_execution":
                cmd = str(item.get("command", ""))
                for tok in cmd.replace("'", " ").replace('"', " ").split():
                    t = tok.strip(",;:()[]")
                    if t.endswith(".py") and "/" in t and t not in files:
                        files.append(t)
            elif item.get("type") == "agent_message":
                txt = item.get("text", "")
                if isinstance(txt, str) and txt.strip():
                    last_msg = txt.strip()
    except Exception:  # noqa: BLE001
        pass
    parts: list[str] = []
    if files:
        parts.append("You already inspected: " + ", ".join(files[:8]) + ".")
    if last_msg:
        parts.append('Your last message was: "' + last_msg[:400] + '".')
    return " ".join(parts)


def _retry_prompt_continue(brief: str) -> str:
    """State-conditional continuation for agent_gave_up: the agent EXPLORED then emitted a
    tool-call-free terminal reply (general temp-0.6 codex flake; reasoning is inert on Qwen3.6
    in this stack). Re-drive it with its own accumulated context + a hard must-act directive."""
    pre = (brief + " ") if brief else ""
    return (
        "Your previous attempt EXPLORED the code but STOPPED with NO edit in the working tree -- "
        "that is a FAILED attempt, not a completed one. " + pre +
        "Do NOT read or grep files again; you have enough context. Your VERY NEXT action MUST be "
        "an apply_patch that edits the source to implement the fix described in /workspace/AGENTS.md. "
        "Every response you produce must call a tool -- never end your turn with an analysis-only or "
        "summary message. Do not stop until the source files are edited."
    )


def _iso_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_GIT_LOCK = threading.Lock()  # serialize per-task git ops when --concurrency>1

_AUTOCOMMIT_TASK_ARTIFACT_RELS = (
    "runner_metadata.json",
    "patch.diff",
    "qwen_trace.jsonl",
    "eval/predictions.jsonl",
    "eval/eval_report.json",
    "orchestrator_crash.json",
)
_AUTOCOMMIT_FIXED32_CAMPAIGN_RELS = (
    "runner_metadata.json",
    "patch.diff",
    "qwen_trace.jsonl",
    "eval/predictions.jsonl",
    "eval/eval_report.json",
    "qwen_runtime_attestation.json",
    "qwen_runtime_attestation_post.json",
    _FIXED32_MOUNTED_RUNTIME_PROOF_FILENAME,
    "fixed32_task_boundary.json",
    "vllm_metrics_pre.txt",
    "vllm_metrics_post.txt",
)


def _autocommit_paths(
    paths: list[str],
    message: str,
    *,
    strict_push: bool = False,
) -> None:
    """Explicit-path artifact commit and optional fail-closed push."""
    if not paths:
        if strict_push:
            raise RuntimeError("artifact commit path set is empty")
        return
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}

    def _git(argv, timeout=120):
        return subprocess.run(
            ["git", *argv],
            cwd=str(REPO_ROOT),
            env=env,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        ).returncode

    with _GIT_LOCK:
        add_rc = _git(["add", "-f", "--", *paths])
        if add_rc != 0:
            if strict_push:
                raise RuntimeError(f"git add failed with rc={add_rc}")
            return
        staged_rc = _git(["diff", "--cached", "--quiet", "--", *paths])
        if staged_rc not in (0, 1):
            if strict_push:
                raise RuntimeError(
                    f"git staged-diff check failed with rc={staged_rc}"
                )
            return
        committed = False
        if staged_rc == 1:
            commit_rc = _git(["commit", "-m", message, "--", *paths])
            if commit_rc != 0:
                if strict_push:
                    raise RuntimeError(f"git commit failed with rc={commit_rc}")
                return
            committed = True
        if strict_push:
            for attempt in range(3):
                push_rc = _git(["push", "origin", "HEAD"], timeout=300)
                if push_rc == 0:
                    return
                if attempt < 2:
                    time.sleep(2**attempt)
            raise RuntimeError("git push failed after 3 attempts")
        if committed:
            _git(["push", "origin", "HEAD"], timeout=300)


def _autocommit_task_artifacts(task_dir: Path, instance_id: str) -> None:
    """Best-effort auto-commit+push of ONE task's curated trace artifacts to the
    shared branch. NEVER raises (whole-body try/except = the `|| true` contract).
    output/ is .gitignored so traces are force-added (git add -f), matching the
    440k+ artifacts already tracked under output/. Commit uses an EXPLICIT pathspec
    (git commit -- <paths>, NOT add-all) so it never sweeps another session's staged
    work on the shared branch, and carries the repo Co-Authored-By trailer. _GIT_LOCK
    serializes git so --concurrency>1 tasks don't race on index.lock.
    LUMO_SWE_AUTOCOMMIT=0 disables it (e.g. a big campaign that doesn't want a push
    per task)."""
    try:
        if os.environ.get("LUMO_SWE_AUTOCOMMIT", "1").strip().lower() in ("0", "off", "false", "no"):
            return
        paths = [
            str(task_dir / rel)
            for rel in _AUTOCOMMIT_TASK_ARTIFACT_RELS
            if (task_dir / rel).is_file()
        ]
        msg = (f"FR13 SWE auto-commit: {instance_id} trace artifacts\n\n"
               "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>")
        _autocommit_paths(paths, msg)
    except Exception:
        pass  # auto-commit is observability; a git error must never fail the run


def _autocommit_fixed32_campaign_artifacts(
    *,
    dataset_out: Path,
    per_task_root: Path,
    instance_ids: list[str],
    taw_campaign_arm_artifact_path: Path | None = None,
) -> None:
    """Commit and push one replay-complete B4 artifact unit."""
    if os.environ.get("LUMO_SWE_AUTOCOMMIT", "1").strip().lower() in (
        "0",
        "off",
        "false",
        "no",
    ):
        return
    required = [
        dataset_out / _FIXED32_QWEN_CAMPAIGN_PROOF_FILENAME,
        dataset_out / _FIXED32_QWEN_CAMPAIGN_METRICS_PRE_FILENAME,
        dataset_out / _FIXED32_QWEN_CAMPAIGN_METRICS_POST_FILENAME,
        *(
            [taw_campaign_arm_artifact_path]
            if taw_campaign_arm_artifact_path is not None
            else []
        ),
        *(
            per_task_root / instance_id / rel
            for instance_id in instance_ids
            for rel in _AUTOCOMMIT_FIXED32_CAMPAIGN_RELS
        ),
    ]
    missing = [
        str(path)
        for path in required
        if not path.is_file() or path.is_symlink()
    ]
    if missing:
        raise Fixed32BoundaryError(
            "fixed32 B4 campaign artifact publication set is incomplete: "
            f"{missing}"
        )
    msg = (
        "FR13 SWE auto-commit: finalized B4 campaign artifacts\n\n"
        "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
    )
    _autocommit_paths(
        [str(path) for path in required],
        msg,
        strict_push=True,
    )


def _metrics_text(metrics_url: str, *, strict: bool = False) -> str:
    """Fetch a Prometheus metrics snapshot from the vLLM container."""
    import urllib.request
    try:
        req = urllib.request.Request(metrics_url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise RuntimeError(
                f"metrics fetch failed: {type(exc).__name__}: {exc}"
            ) from exc
        return f"# metrics fetch failed: {type(exc).__name__}: {exc}\n"


def _sidecar_fwd_gpu_totals() -> dict | None:
    """FR13_SFWD_GPU_TIMER: cumulative pure-decode-forward GPU stats from the worker
    timer's JSON sidecar(s): {seconds, steps, drafts}. ALL THREE are restricted to
    the SAME pure-decode steps the timer measured (prefill/mixed/idle EXCLUDED), so
    seconds/steps (per-forward) and seconds/drafts (per-spec-event) are prefill-
    INDEPENDENT. CRITICAL: do NOT divide seconds by the GLOBAL
    spec_decode_num_drafts_total -- that counts drafts on mixed prefill+decode steps
    the timer excludes, reintroducing a prefill-load-dependent confound (measured:
    ~49% of global drafts land on non-timed steps at B=4 deployment). The worker
    writes the sidecar per-pid to FR13_SFWD_GPU_TIMER_JSON; in single-API-server mode
    the worker Counter is NOT aggregated into the API-server /metrics, so this sidecar
    is the robust channel. Returns None when the timer is off / no sidecar. Sums
    across per-pid files (one per worker)."""
    base = os.environ.get("FR13_SFWD_GPU_TIMER_JSON")
    if not base:
        return None
    # the boot env sets a /workspace path (docker -v "$REPO:/workspace"); on the
    # host that is REPO_ROOT. Translate so this host-side reader finds the file.
    if base.startswith("/workspace/"):
        base = str(REPO_ROOT / base[len("/workspace/"):])
    import glob as _glob
    pat = base.replace("{pid}", "*") if "{pid}" in base else base + ".*"
    files = set(_glob.glob(pat))
    if os.path.exists(base):
        files.add(base)
    secs = steps = drafts = 0.0
    wall_s = wall_drafts = wall_steps = wall_rejected = 0.0
    found = False
    for f in files:
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
            secs += float(d.get("decode_forward_gpu_seconds", 0.0))
            steps += float(d.get("n_pure_decode_steps_timed", 0.0))
            drafts += float(d.get("n_drafts_in_timed_steps", 0.0))
            # FR13_STEP_WALL: measured full-step wall basis (absent on old
            # sidecars -> 0.0 -> reducer emits null, never a crash).
            wall_s += float(d.get("decode_step_wall_seconds", 0.0))
            wall_drafts += float(d.get("n_drafts_in_wall_steps", 0.0))
            wall_steps += float(d.get("n_wall_steps", 0.0))
            wall_rejected += float(d.get("n_wall_rejected", 0.0))
            found = True
        except Exception:  # noqa: BLE001
            continue
    return {
        "seconds": secs, "steps": steps, "drafts": drafts,
        "wall_seconds": wall_s, "wall_drafts": wall_drafts,
        "wall_steps": wall_steps,
        "wall_attempts": wall_steps + wall_rejected,
        "wall_rejected": wall_rejected,
    } if found else None


def _sidecar_span_totals(env_name: str) -> dict | None:
    """FR13_DFWD/CFWD_GPU_TIMER: cumulative component GPU-span stats from a
    _Fr13SpanTimer JSON sidecar (schema fr13.span_gpu_timer.v1): {seconds,
    spans}. drafter = propose_draft_token_ids (all D spine forwards);
    committer = the spec-decode rejection-sampler dispatch in _sample. Same
    per-pid file layout + /workspace->host translation as the sfwd sidecar
    (_sidecar_fwd_gpu_totals); sums across per-pid files. Returns None when
    the timer is off / no sidecar."""
    base = os.environ.get(env_name)
    if not base:
        return None
    if base.startswith("/workspace/"):
        base = str(REPO_ROOT / base[len("/workspace/"):])
    import glob as _glob
    pat = base.replace("{pid}", "*") if "{pid}" in base else base + ".*"
    files = set(_glob.glob(pat))
    if os.path.exists(base):
        files.add(base)
    secs = spans = 0.0
    found = False
    for f in files:
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
            secs += float(d.get("gpu_seconds", 0.0))
            spans += float(d.get("n_spans", 0.0))
            found = True
        except Exception:  # noqa: BLE001
            continue
    return {"seconds": secs, "spans": spans} if found else None


def _metrics_snapshot(
    metrics_url: str,
    *,
    strict: bool = False,
    fixed32_counters: dict[str, Any] | None = None,
) -> str:
    """/metrics text, plus (FR13_SFWD_GPU_TIMER) synthetic counter lines carrying the
    worker timer's cumulative pure-decode-forward GPU seconds AND its MATCHED
    denominators (pure-decode forward count + drafts-on-those-steps). The matched
    denominators are essential for prefill-independence: dividing the pure-decode-only
    GPU seconds by the GLOBAL spec_decode_num_drafts_total (which counts mixed-step
    drafts) reintroduces the prefill confound; dividing by the matched count (both
    pure-decode-only) does not. In single-API-server mode /metrics does not expose
    these worker counters, so we synthesize them here for the reducer's per-task
    bracket path.

    Double-count guard: if /metrics ALREADY exposes the seconds counter (multiprocess
    aggregation active), use that and do NOT append. No-op / byte-identical when the
    timer is off / no sidecar."""
    text = _metrics_text(metrics_url, strict=strict)
    if "fr13_decode_forward_gpu_seconds_total" not in text:
        t = _sidecar_fwd_gpu_totals()
        if t is not None:
            if not text.endswith("\n"):
                text += "\n"
            text += f"vllm:fr13_decode_forward_gpu_seconds_total {t['seconds']:.9f}\n"
            text += f"vllm:fr13_decode_forward_gpu_steps_total {t['steps']:.1f}\n"
            text += f"vllm:fr13_decode_forward_gpu_drafts_total {t['drafts']:.1f}\n"
            # FR13_STEP_WALL: measured full-step wall (start-to-start between
            # consecutive pure-decode steps, idle-capped) -- the MEASURED twin
            # the derived fullstep TPS must align with.
            text += f"vllm:fr13_decode_step_wall_seconds_total {t['wall_seconds']:.9f}\n"
            text += f"vllm:fr13_decode_step_wall_drafts_total {t['wall_drafts']:.1f}\n"
            text += f"vllm:fr13_decode_step_wall_steps_total {t['wall_steps']:.1f}\n"
            text += (
                "vllm:fr13_decode_step_wall_attempts_total "
                f"{t['wall_attempts']:.1f}\n"
            )
            text += (
                "vllm:fr13_decode_step_wall_rejected_total "
                f"{t['wall_rejected']:.1f}\n"
            )
    # FR13_DFWD/CFWD_GPU_TIMER: the SAME synthetic-line route for the drafter /
    # committer span timers (their prometheus Counters are also worker-process-
    # local in single-API-server mode; the sidecar is the robust channel). The
    # spans counter exists ONLY as a synthetic line (the worker Counter carries
    # seconds alone), so a multiprocess-aggregated /metrics run reports seconds
    # without spans -> the reducer's ms-per-step is null there. No-op /
    # byte-identical when a timer is off / no sidecar.
    for _mbase, _env in (
        ("fr13_drafter_gpu", "FR13_DFWD_GPU_TIMER_JSON"),
        ("fr13_committer_gpu", "FR13_CFWD_GPU_TIMER_JSON"),
    ):
        if f"{_mbase}_seconds_total" in text:
            continue  # multiprocess /metrics already exposes the counter
        st = _sidecar_span_totals(_env)
        if st is None:
            continue
        if not text.endswith("\n"):
            text += "\n"
        text += f"vllm:{_mbase}_seconds_total {st['seconds']:.9f}\n"
        text += f"vllm:{_mbase}_spans_total {st['spans']:.1f}\n"
    if fixed32_counters is not None:
        if not text.endswith("\n"):
            text += "\n"
        text += (
            "vllm:fr13_fixed32_pure_decode_forward_steps_total "
            f"{fixed32_counters['pure_decode_forward_steps']}\n"
        )
        text += (
            "vllm:fr13_fixed32_complete_work_census_events_total "
            f"{fixed32_counters['complete_work_census_events']}\n"
        )
        sidecar = _sidecar_fwd_gpu_totals()
        if sidecar is None:
            raise RuntimeError("fixed32 snapshot has no SFWD timer sidecar")
        if int(sidecar["steps"]) != fixed32_counters["pure_decode_forward_steps"]:
            raise RuntimeError(
                "fixed32 flush/SFWD sidecar mismatch: "
                f"ack={fixed32_counters['pure_decode_forward_steps']} "
                f"sidecar={sidecar['steps']}"
            )
    return text


class Fixed32BoundaryError(RuntimeError):
    """A fixed32 task lacks a valid runtime interval or real-task provenance."""


_FIXED32_TAW_REAL_TASK_ARM_NAME = (
    "fr13_fixed32_taw_native_precompute.real_event.arm"
)
_FIXED32_BM8_REAL_TASK_ARM_NAME = "fr13_dfwd_unified_bm8.real_event.arm"
_FIXED32_CUTLASS_REAL_TASK_ARM_NAME = (
    "fr13_fixed32_cutlass_streamk.real_event.arm"
)
_FIXED32_COMMITTER_LAYER_BATCH_REAL_TASK_ARM_NAME = (
    "fr13_fixed32_committer_layer_batch.real_event.arm"
)
_FIXED32_TAW_REAL_TASK_MARKER_PREFIX = "swe_verified:"
_FIXED32_TAW_REAL_TASK_ID_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/"
)


def _fixed32_taw_file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _fixed32_taw_directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
    )


class _Fixed32TawRealTaskArm:
    """Atomically bind one diagnostic TAW run to one pinned SWE task."""

    arm_name = _FIXED32_TAW_REAL_TASK_ARM_NAME
    artifact_name = "fixed32_taw_real_task_arm.json"
    label = "TAW"
    schema = "fr13-fixed32-taw-real-task-arm-v1"

    def __init__(self, *, path: Path, instance_id: str) -> None:
        if (
            not instance_id
            or any(
                character not in _FIXED32_TAW_REAL_TASK_ID_CHARACTERS
                for character in instance_id
            )
        ):
            raise Fixed32BoundaryError(
                f"fixed32 {self.label} real-task arm instance ID is not kernel-canonical"
            )
        marker = f"{_FIXED32_TAW_REAL_TASK_MARKER_PREFIX}{instance_id}"
        marker_bytes = (marker + "\n").encode("ascii")
        if len(marker) > 256:
            raise Fixed32BoundaryError(
                f"fixed32 {self.label} real-task arm marker exceeds 256 bytes"
            )
        if not path.is_absolute() or path.name != self.arm_name:
            raise Fixed32BoundaryError(
                f"fixed32 {self.label} real-task arm requires the absolute canonical filename"
            )
        try:
            canonical_parent = path.parent.resolve(strict=True)
        except OSError as error:
            raise Fixed32BoundaryError(
                f"fixed32 {self.label} real-task arm parent is unavailable: {error}"
            ) from error
        if canonical_parent != path.parent:
            raise Fixed32BoundaryError(
                f"fixed32 {self.label} real-task arm parent must not traverse symlinks"
            )

        self.path = path
        self.instance_id = instance_id
        self.marker = marker
        self.marker_bytes = marker_bytes
        self.marker_sha256 = hashlib.sha256(marker_bytes).hexdigest()
        self.rotated_path = path.with_name(
            path.name.removesuffix(".arm")
            + f".ended.{self.marker_sha256}.arm"
        )
        self.state = "planned"
        self.started_at: str | None = None
        self.ended_at: str | None = None
        self._live_identity: tuple[int, ...] | None = None
        self._rotated_identity: tuple[int, ...] | None = None

    @property
    def active(self) -> bool:
        return self.state in {"published", "active", "rotation_linked"}

    def _open_parent(self) -> int:
        try:
            before = self.path.parent.lstat()
        except OSError as error:
            raise Fixed32BoundaryError(
                f"fixed32 {self.label} real-task arm cannot inspect its parent: {error}"
            ) from error
        if (
            not stat.S_ISDIR(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o700
            or before.st_uid != os.geteuid()
        ):
            raise Fixed32BoundaryError(
                f"fixed32 {self.label} real-task arm parent must be an owned mode-0700 directory"
            )
        try:
            descriptor = os.open(
                self.path.parent,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
        except OSError as error:
            raise Fixed32BoundaryError(
                f"fixed32 {self.label} real-task arm cannot open its parent: {error}"
            ) from error
        opened = os.fstat(descriptor)
        if (
            _fixed32_taw_directory_identity(opened)
            != _fixed32_taw_directory_identity(before)
        ):
            os.close(descriptor)
            raise Fixed32BoundaryError(
                f"fixed32 {self.label} real-task arm parent changed while opening"
            )
        return descriptor

    @staticmethod
    def _name_exists(parent_descriptor: int, name: str) -> bool:
        try:
            os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True

    def _read_exact(self, parent_descriptor: int, name: str) -> os.stat_result:
        try:
            before = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise Fixed32BoundaryError(
                f"fixed32 {self.label} real-task marker is unavailable: {error}"
            ) from error
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o400
            or before.st_uid != os.geteuid()
            or before.st_size != len(self.marker_bytes)
        ):
            raise Fixed32BoundaryError(
                f"fixed32 {self.label} real-task marker metadata is noncanonical"
            )
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_descriptor,
            )
        except OSError as error:
            raise Fixed32BoundaryError(
                f"fixed32 {self.label} real-task marker cannot be opened exactly: {error}"
            ) from error
        try:
            opened = os.fstat(descriptor)
            if _fixed32_taw_file_identity(opened) != _fixed32_taw_file_identity(before):
                raise Fixed32BoundaryError(
                    f"fixed32 {self.label} real-task marker changed before exact read"
                )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 4096)
                if not chunk:
                    break
                chunks.append(chunk)
            after_read = os.fstat(descriptor)
            if (
                _fixed32_taw_file_identity(after_read)
                != _fixed32_taw_file_identity(opened)
                ):
                    raise Fixed32BoundaryError(
                        f"fixed32 {self.label} real-task marker changed during exact read"
                    )
        finally:
            os.close(descriptor)
        after = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if _fixed32_taw_file_identity(after) != _fixed32_taw_file_identity(before):
            raise Fixed32BoundaryError(
                f"fixed32 {self.label} real-task marker changed after exact read"
            )
        if b"".join(chunks) != self.marker_bytes:
            raise Fixed32BoundaryError(
                f"fixed32 {self.label} real-task marker bytes are noncanonical"
            )
        return before

    def start(self) -> None:
        if self.state != "planned":
            raise Fixed32BoundaryError(
                f"fixed32 {self.label} real-task arm start was invoked twice"
            )
        parent_descriptor = self._open_parent()
        temporary_name: str | None = None
        try:
            if self._name_exists(parent_descriptor, self.path.name):
                raise Fixed32BoundaryError(
                    f"fixed32 {self.label} real-task arm destination is not fresh"
                )
            if self._name_exists(parent_descriptor, self.rotated_path.name):
                raise Fixed32BoundaryError(
                    f"fixed32 {self.label} real-task arm rotation destination is not fresh"
                )
            temporary_name = (
                f".{self.path.name}.tmp.{os.getpid()}.{secrets.token_hex(16)}"
            )
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_CLOEXEC
                | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent_descriptor,
            )
            try:
                view = memoryview(self.marker_bytes)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise Fixed32BoundaryError(
                            f"fixed32 {self.label} real-task marker write made no progress"
                        )
                    view = view[written:]
                os.fchmod(descriptor, 0o400)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.link(
                temporary_name,
                self.path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            self.state = "published"
            os.unlink(temporary_name, dir_fd=parent_descriptor)
            temporary_name = None
            os.fsync(parent_descriptor)
            metadata = self._read_exact(parent_descriptor, self.path.name)
            self._live_identity = _fixed32_taw_file_identity(metadata)
            self.started_at = _iso_now()
            self.state = "active"
        except Fixed32BoundaryError:
            raise
        except Exception as error:
            raise Fixed32BoundaryError(
                f"fixed32 {self.label} real-task arm creation failed: "
                f"{type(error).__name__}: {error}"
            ) from error
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name, dir_fd=parent_descriptor)
                except FileNotFoundError:
                    pass
            os.close(parent_descriptor)

    def finish(self) -> None:
        if not self.active:
            if self.state == "ended":
                return
            raise Fixed32BoundaryError(
                f"fixed32 {self.label} real-task arm end has no active task"
            )
        parent_descriptor = self._open_parent()
        try:
            metadata = self._read_exact(parent_descriptor, self.path.name)
            identity = _fixed32_taw_file_identity(metadata)
            if self._live_identity is not None and identity != self._live_identity:
                raise Fixed32BoundaryError(
                    f"fixed32 {self.label} real-task marker identity changed during the task"
                )
            if self._name_exists(parent_descriptor, self.rotated_path.name):
                raise Fixed32BoundaryError(
                    f"fixed32 {self.label} real-task arm rotation destination is not fresh"
                )
            os.link(
                self.path.name,
                self.rotated_path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            self.state = "rotation_linked"
            os.unlink(self.path.name, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
            rotated = self._read_exact(
                parent_descriptor,
                self.rotated_path.name,
            )
            self._rotated_identity = _fixed32_taw_file_identity(rotated)
            self.ended_at = _iso_now()
            self.state = "ended"
        except Fixed32BoundaryError:
            raise
        except Exception as error:
            raise Fixed32BoundaryError(
                f"fixed32 {self.label} real-task arm rotation failed: "
                f"{type(error).__name__}: {error}"
            ) from error
        finally:
            os.close(parent_descriptor)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_classification": "b1_diagnostic",
            "gate_eligible": False,
            "floor_acceptance_eligible": False,
            "instance_id": self.instance_id,
            "marker": self.marker,
            "marker_bytes": len(self.marker_bytes),
            "marker_sha256": self.marker_sha256,
            "live_path": str(self.path),
            "rotated_path": str(self.rotated_path),
            "state": self.state,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


class _Fixed32TawCampaignArm(_Fixed32TawRealTaskArm):
    """Publish one TAW marker for an exact-B4 canonical SWE campaign."""

    artifact_name = "fixed32_taw_campaign_arm.json"
    schema = "fr13-fixed32-taw-campaign-arm-v1"

    def __init__(
        self,
        *,
        path: Path,
        subset_binding: dict[str, Any],
        concurrency: int,
    ) -> None:
        if concurrency != 4:
            raise Fixed32BoundaryError(
                "fixed32 TAW campaign arm requires exact B4 concurrency"
            )
        if not isinstance(subset_binding, dict):
            raise Fixed32BoundaryError(
                "fixed32 TAW campaign arm subset binding is not an object"
            )
        try:
            from fr13_floor_gate import (
                GateError as FloorGateError,
                validate_canonical_subset,
            )

            binding_path = Path(subset_binding["path"])
            validated = validate_canonical_subset(binding_path)
        except (KeyError, TypeError, ValueError, OSError, FloorGateError) as error:
            raise Fixed32BoundaryError(
                f"fixed32 TAW campaign arm subset binding is invalid: {error}"
            ) from error
        if subset_binding != validated:
            raise Fixed32BoundaryError(
                "fixed32 TAW campaign arm differs from the canonical subset binding"
            )
        task_count = validated["task_count"]
        if task_count not in (4, 16):
            raise Fixed32BoundaryError(
                "fixed32 TAW campaign arm requires a canonical 4/16-task set"
            )
        subset_sha256 = validated["sha256"]
        task_ids = validated["task_ids"]
        campaign_identity = f"campaign{task_count}_{subset_sha256}"
        super().__init__(path=path, instance_id=campaign_identity)
        self.task_count = task_count
        self.concurrency = concurrency
        self.subset_path = validated["path"]
        self.subset_sha256 = subset_sha256
        self.task_ids = list(task_ids)

    def as_dict(self) -> dict[str, Any]:
        payload = super().as_dict()
        payload.pop("instance_id")
        payload.update(
            {
                "run_classification": "b4_taw_diagnostic",
                "batch_size": 4,
                "concurrency": self.concurrency,
                "task_count": self.task_count,
                "subset_path": self.subset_path,
                "subset_sha256": self.subset_sha256,
                "task_ids": self.task_ids,
            }
        )
        return payload


def _run_with_fixed32_taw_campaign_arm(
    *,
    arm: _Fixed32TawCampaignArm | None,
    artifact_path: Path | None,
    action: Any,
) -> Any:
    """Keep one authenticated marker live across all concurrent B4 workers."""
    if arm is None:
        if artifact_path is not None:
            raise Fixed32BoundaryError(
                "fixed32 TAW campaign artifact path has no campaign arm"
            )
        return action()
    if artifact_path is None:
        raise Fixed32BoundaryError(
            "fixed32 TAW campaign arm has no artifact path"
        )

    def _write_artifact() -> None:
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                arm.as_dict(),
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="ascii",
        )
        os.replace(temporary, artifact_path)

    try:
        arm.start()
        _write_artifact()
        return action()
    finally:
        if arm.active:
            arm.finish()
            _write_artifact()


class _Fixed32Bm8RealTaskArm(_Fixed32TawRealTaskArm):
    """Atomically bind the BM8 byte gate to one pinned SWE task."""

    arm_name = _FIXED32_BM8_REAL_TASK_ARM_NAME
    artifact_name = "fixed32_bm8_real_task_arm.json"
    label = "BM8"
    schema = "fr13-fixed32-bm8-real-task-arm-v1"


class _Fixed32CutlassRealTaskArm(_Fixed32TawRealTaskArm):
    """Atomically bind the CUTLASS byte gate to one pinned SWE task."""

    arm_name = _FIXED32_CUTLASS_REAL_TASK_ARM_NAME
    artifact_name = "fixed32_cutlass_streamk_real_task_arm.json"
    label = "CUTLASS Stream-K"
    schema = "fr13-fixed32-cutlass-streamk-real-task-arm-v1"


class _Fixed32CommitterLayerBatchRealTaskArm(_Fixed32TawRealTaskArm):
    """Bind one CFWD layer-batch qualification to one pinned SWE task."""

    arm_name = _FIXED32_COMMITTER_LAYER_BATCH_REAL_TASK_ARM_NAME
    artifact_name = "fixed32_committer_layer_batch_real_task_arm.json"
    label = "CFWD layer-batch"
    schema = "fr13-fixed32-committer-layer-batch-real-task-arm-v1"

    def as_dict(self) -> dict[str, Any]:
        payload = super().as_dict()
        payload.update(
            {
                "run_classification": _FIXED32_CFWD_QUALIFICATION_CLASSIFICATION,
                "performance_measurement": False,
                "timing_eligible": False,
                "process_local_qualification_only": True,
                "durable_production_pass": False,
                "timing_requires_same_server_process": True,
                "same_process_timing_handoff_implemented": False,
            }
        )
        return payload


class _Fixed32CommitterLayerBatchCampaignArm(_Fixed32TawCampaignArm):
    """Bind CFWD qualification to one canonical exact4/16 B4 campaign."""

    arm_name = _FIXED32_COMMITTER_LAYER_BATCH_REAL_TASK_ARM_NAME
    artifact_name = "fixed32_committer_layer_batch_campaign_arm.json"
    label = "CFWD layer-batch campaign"
    schema = "fr13-fixed32-committer-layer-batch-campaign-arm-v1"

    def as_dict(self) -> dict[str, Any]:
        payload = super().as_dict()
        payload.update(
            {
                "run_classification": (
                    _FIXED32_CFWD_B4_QUALIFICATION_CLASSIFICATION
                ),
                "performance_measurement": False,
                "timing_eligible": False,
                "process_local_qualification_only": True,
                "durable_production_pass": False,
                "timing_requires_same_server_process": True,
                "same_process_timing_handoff_contract_implemented": True,
                "same_process_timing_execution_implemented": False,
            }
        )
        return payload


_FIXED32_TOKEN_USAGE_FIELDS = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "cached_input_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    }
)
_FIXED32_AGENT_TERMINAL_FIELDS = (
    "exit_code",
    "timed_out",
    "offloaded",
    "network_drop",
    "stall_killed",
    "cause",
    "ws_down_rc",
    "patch_down_rc",
    "setup_error",
    "transport_error",
    "dispatch_error",
    "error",
    "elapsed_s",
    "container_name",
    "codex_host",
    "agent_env",
    "instance_image",
)


def _fixed32_has_model_output(value: Any) -> bool:
    """Return whether an assistant content value contains model-emitted output."""
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_fixed32_has_model_output(item) for item in value)
    if not isinstance(value, dict):
        return False
    if value.get("type") in {"tool_use", "function_call"}:
        return bool(str(value.get("name", "")).strip())
    return any(
        _fixed32_has_model_output(value.get(key))
        for key in ("text", "thinking", "content", "output_text")
        if key in value
    )


def _fixed32_usage_records(value: Any) -> list[dict[str, Any]]:
    """Collect every explicit ``usage`` object without interpreting other numbers."""
    records: list[dict[str, Any]] = []
    if isinstance(value, list):
        for item in value:
            records.extend(_fixed32_usage_records(item))
    elif isinstance(value, dict):
        for key, item in value.items():
            if key == "usage" and isinstance(item, dict):
                records.append(item)
            else:
                records.extend(_fixed32_usage_records(item))
    return records


def _fixed32_load_trace_events(
    trace_path: Path,
    *,
    instance_id: str,
) -> tuple[bytes, list[dict[str, Any]]]:
    try:
        trace_metadata = trace_path.lstat()
        if (
            not stat.S_ISREG(trace_metadata.st_mode)
            or trace_metadata.st_nlink != 1
        ):
            raise Fixed32BoundaryError(
                f"fixed32 real-task provenance {instance_id}: "
                f"trace is not a single-link regular file: {trace_path}"
            )
        raw_trace = trace_path.read_bytes()
    except OSError as exc:
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: "
            f"cannot read trace {trace_path}: {type(exc).__name__}: {exc}"
        ) from exc
    if not raw_trace:
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: trace is empty: {trace_path}"
        )
    if not raw_trace.endswith(b"\n"):
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: "
            f"trace is not newline-framed: {trace_path}"
        )
    try:
        trace_text = raw_trace.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: trace is not UTF-8: {trace_path}"
        ) from exc

    def _trace_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for key, value in pairs:
            if key in parsed:
                raise ValueError("duplicate trace key")
            parsed[key] = value
        return parsed

    def _trace_nonfinite(value: str) -> Any:
        raise ValueError(f"nonfinite trace value: {value}")

    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(trace_text.splitlines(), start=1):
        if not line.strip():
            raise Fixed32BoundaryError(
                f"fixed32 real-task provenance {instance_id}: "
                f"blank JSONL record at {trace_path}:{line_number}"
            )
        try:
            event = json.loads(
                line,
                object_pairs_hook=_trace_object,
                parse_constant=_trace_nonfinite,
            )
        except (ValueError, json.JSONDecodeError) as exc:
            raise Fixed32BoundaryError(
                f"fixed32 real-task provenance {instance_id}: "
                f"invalid JSON at {trace_path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(event, dict):
            raise Fixed32BoundaryError(
                f"fixed32 real-task provenance {instance_id}: "
                f"JSONL record is not an object at {trace_path}:{line_number}"
            )
        events.append(event)
    if not events:
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: trace has no JSONL events"
        )
    return raw_trace, events


def _fixed32_artifact_identity(path: Path, raw: bytes) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _write_fixed32_campaign_metrics(path: Path) -> bytes:
    raw = _metrics_snapshot(
        DEFAULT_METRICS_URL,
        strict=True,
    ).encode("utf-8")
    if not raw:
        raise Fixed32BoundaryError("fixed32 B4 campaign metric snapshot is empty")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)
    return raw


def _fixed32_real_task_provenance(
    *,
    instance_id: str,
    trace_path: Path,
    agent_meta: dict[str, Any],
    task_key_id: str,
    task_auth_before: dict[str, Any],
    task_auth_after: dict[str, Any],
    metrics_pre_path: Path | None = None,
    metrics_post_path: Path | None = None,
    campaign_trace_requests: dict[str, Any] | None = None,
    campaign_metric_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and describe the real offloaded model run for one fixed32 task.

    This deliberately accepts both qwen-code instance-image JSONL
    (``type=assistant`` with a nested assistant ``message``) and Codex JSONL
    (completed ``agent_message`` items). Every physical line must be a JSON
    object; a partial final line therefore fails closed.
    """
    if not isinstance(agent_meta, dict):
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: agent metadata is not an object"
        )
    if (
        not isinstance(task_key_id, str)
        or len(task_key_id) != 64
        or any(char not in "0123456789abcdef" for char in task_key_id)
    ):
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: task key ID is malformed"
        )
    evidence_counter_keys = (
        "completed_logical_model_requests",
        "aborted_logical_requests",
        "accepted_attempts",
        "completed_attempts",
        "failed_attempts",
    )
    if (
        not isinstance(task_auth_before, dict)
        or not isinstance(task_auth_after, dict)
        or task_auth_before.get("task_key_id") != task_key_id
        or task_auth_after.get("task_key_id") != task_key_id
    ):
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: "
            "task-auth evidence does not bind to the task key"
        )
    task_auth_deltas: dict[str, int] = {}
    for key in evidence_counter_keys:
        before = task_auth_before.get(key)
        after = task_auth_after.get(key)
        if (
            isinstance(before, bool)
            or not isinstance(before, int)
            or isinstance(after, bool)
            or not isinstance(after, int)
            or before < 0
            or after < before
        ):
            raise Fixed32BoundaryError(
                f"fixed32 real-task provenance {instance_id}: "
                f"task-auth counter {key} is invalid"
            )
        task_auth_deltas[key] = after - before

    exit_code = agent_meta.get("exit_code")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: "
            f"agent exit_code must be an integer, got {exit_code!r}"
        )
    if not isinstance(agent_meta.get("timed_out"), bool):
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: "
            f"agent timed_out must be boolean, got {agent_meta.get('timed_out')!r}"
        )
    if exit_code != 0 or agent_meta["timed_out"]:
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: "
            f"agent terminal state is incomplete: exit_code={exit_code} "
            f"timed_out={agent_meta['timed_out']}"
        )
    if agent_meta.get("offloaded") is not True:
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: "
            f"agent offloaded must be true, got {agent_meta.get('offloaded')!r}"
        )
    if agent_meta.get("network_drop") is not False:
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: "
            f"agent network_drop must be false, got {agent_meta.get('network_drop')!r}"
        )
    stall_killed = agent_meta.get("stall_killed")
    if stall_killed is not None and stall_killed is not False:
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: "
            f"agent stall_killed is set: {stall_killed!r}"
        )
    ws_down_rc = agent_meta.get("ws_down_rc")
    if ws_down_rc is not None and (
        isinstance(ws_down_rc, bool)
        or not isinstance(ws_down_rc, int)
        or ws_down_rc != 0
    ):
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: "
            f"workspace transport failed with rc={ws_down_rc!r}"
        )
    patch_down_rc = agent_meta.get("patch_down_rc")
    if patch_down_rc is not None and (
        isinstance(patch_down_rc, bool)
        or not isinstance(patch_down_rc, int)
        or patch_down_rc not in (0, 1)
    ):
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: "
            f"patch transport failed with rc={patch_down_rc!r}"
        )
    for key in ("setup_error", "transport_error", "dispatch_error"):
        if agent_meta.get(key):
            raise Fixed32BoundaryError(
                f"fixed32 real-task provenance {instance_id}: "
                f"agent terminal field {key}={agent_meta[key]!r}"
            )
    terminal_cause = " ".join(
        str(agent_meta.get(key, ""))
        for key in ("cause", "error")
        if agent_meta.get(key)
    ).lower()
    if any(
        marker in terminal_cause
        for marker in ("setup", "transport", "network_drop", "network-drop", "stall")
    ):
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: "
            f"infrastructure terminal cause {terminal_cause!r}"
        )

    if agent_meta.get("agent_env") != "instance_image":
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: "
            "agent did not use the canonical instance-image launcher"
        )
    instance_image = agent_meta.get("instance_image")
    if not isinstance(instance_image, str):
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: "
            "canonical instance image name is missing"
        )
    image_identity = _validate_fixed32_agent_image_observation(
        agent_meta.get("instance_image_identity"),
        instance_id=instance_id,
        expected_image=instance_image,
    )
    image_identity_sha256 = _fixed32_canonical_json_sha256(image_identity)
    if (
        agent_meta.get("instance_image_identity_sha256")
        != image_identity_sha256
        or agent_meta.get("instance_image_postrun_identity_sha256")
        != image_identity_sha256
        or agent_meta.get("instance_image_run_reference")
        != image_identity["repo_digest"]
    ):
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: "
            "agent image pre/run/post identities differ"
        )
    placement = _validate_fixed32_agent_placement_observation(
        (
            agent_meta.get("agent_placement", {}).get("agent_host_identity")
            if isinstance(agent_meta.get("agent_placement"), dict)
            else None
        ),
        measured_observation=(
            agent_meta.get("agent_placement", {}).get(
                "measured_host_identity"
            )
            if isinstance(agent_meta.get("agent_placement"), dict)
            else None
        ),
        remote_host=_FIXED32_AGENT_HOST_ALIAS,
    )
    placement_sha256 = _fixed32_canonical_json_sha256(placement)
    if (
        agent_meta.get("agent_placement") != placement
        or agent_meta.get("agent_placement_sha256") != placement_sha256
        or agent_meta.get("agent_postrun_placement_sha256")
        != placement_sha256
    ):
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: "
            "agent placement pre/post identities differ"
        )
    runtime_attestation = agent_meta.get("qwen_runtime_attestation")
    runtime_attestation_sha256 = (
        _validate_fixed32_qwen_runtime_attestation(runtime_attestation)
    )
    if (
        agent_meta.get("qwen_runtime_attestation_sha256")
        != runtime_attestation_sha256
        or agent_meta.get("qwen_runtime_postrun_attestation_sha256")
        != runtime_attestation_sha256
    ):
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: "
            "Qwen pre/post runtime attestation digests differ"
        )
    remote_settings_observation = (
        _validate_fixed32_remote_settings_observation(
            agent_meta.get("qwen_remote_settings_observation")
        )
    )
    remote_settings_observation_sha256 = (
        _fixed32_canonical_json_sha256(remote_settings_observation)
    )
    if (
        agent_meta.get("qwen_remote_settings_observation_sha256")
        != remote_settings_observation_sha256
        or agent_meta.get(
            "qwen_remote_settings_postrun_observation_sha256"
        )
        != remote_settings_observation_sha256
    ):
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: "
            "remote settings pre/post identities differ"
        )
    if (
        not isinstance(runtime_attestation, dict)
        or agent_meta.get("qwen_bundle_snapshot")
        != runtime_attestation.get("bundle_snapshot")
    ):
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: "
            "snapshot or remote settings identity differs"
        )
    expected_bundle_observation = {
        "qwen_code_version": runtime_attestation["qwen_code_version"],
        "bundle_tree": runtime_attestation["bundle_tree"],
    }
    mounted_runtime_proof = agent_meta.get("qwen_mounted_runtime_proof")
    mounted_runtime_proof_sha256 = (
        _validate_fixed32_mounted_runtime_proof(
            mounted_runtime_proof,
            expected_bundle_observation=expected_bundle_observation,
        )
    )
    proof_path = (
        trace_path.parent / _FIXED32_MOUNTED_RUNTIME_PROOF_FILENAME
    )
    (
        persisted_mounted_runtime_proof,
        persisted_mounted_runtime_proof_sha256,
        mounted_runtime_proof_file_sha256,
    ) = _load_fixed32_mounted_runtime_proof(
        proof_path,
        expected_bundle_observation=expected_bundle_observation,
    )
    if (
        persisted_mounted_runtime_proof != mounted_runtime_proof
        or persisted_mounted_runtime_proof_sha256
        != mounted_runtime_proof_sha256
        or agent_meta.get("qwen_mounted_runtime_proof_sha256")
        != mounted_runtime_proof_sha256
        or agent_meta.get("qwen_mounted_runtime_proof_file_sha256")
        != mounted_runtime_proof_file_sha256
        or mounted_runtime_proof["system_settings"][
            "file_identity_sha256"
        ]
        != remote_settings_observation["file_identity_sha256"]
    ):
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: "
            "mounted runtime proof artifact differs"
        )

    attestation_artifacts: dict[str, bytes] = {}
    for filename in (
        "qwen_runtime_attestation.json",
        "qwen_runtime_attestation_post.json",
    ):
        path = trace_path.parent / filename
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise Fixed32BoundaryError(
                f"fixed32 real-task provenance {instance_id}: "
                f"cannot read Qwen runtime attestation {path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        artifact = _fixed32_load_json_object(
            text,
            label=f"Qwen runtime attestation {path}",
        )
        if artifact != runtime_attestation:
            raise Fixed32BoundaryError(
                f"fixed32 real-task provenance {instance_id}: "
                f"Qwen runtime attestation artifact differs: {path}"
            )
        if (
            _validate_fixed32_qwen_runtime_attestation(artifact)
            != runtime_attestation_sha256
        ):
            raise Fixed32BoundaryError(
                f"fixed32 real-task provenance {instance_id}: "
                f"Qwen runtime attestation digest differs: {path}"
            )
        attestation_artifacts[filename] = raw

    raw_trace, events = _fixed32_load_trace_events(
        trace_path,
        instance_id=instance_id,
    )
    init_event = events[0]
    if (
        init_event.get("type") != "system"
        or init_event.get("subtype") != "init"
        or init_event.get("qwen_code_version")
        != _FIXED32_QWEN_CODE_VERSION
    ):
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: "
            "trace does not start with the pinned Qwen 0.19.4 init record"
        )

    assistant_event_count = 0
    assistant_output_event_count = 0
    qwen_assistant_event_count = 0
    codex_agent_message_event_count = 0
    usage_records: list[dict[str, Any]] = []
    for event in events:
        usage_records.extend(_fixed32_usage_records(event))
        event_type = event.get("type")
        assistant_content: Any = None
        is_assistant_event = False
        if event_type == "assistant":
            message = event.get("message")
            if isinstance(message, dict) and message.get("role") == "assistant":
                is_assistant_event = True
                qwen_assistant_event_count += 1
                assistant_content = message.get("content")
        elif event_type == "message" and event.get("role") == "assistant":
            is_assistant_event = True
            qwen_assistant_event_count += 1
            assistant_content = event.get("content")
        elif event_type == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                is_assistant_event = True
                codex_agent_message_event_count += 1
                assistant_content = item.get("text", item.get("content"))
        if is_assistant_event:
            assistant_event_count += 1
            if _fixed32_has_model_output(assistant_content):
                assistant_output_event_count += 1

    if assistant_output_event_count <= 0:
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: "
            "trace contains no nonempty assistant/model output"
        )
    metric_paths = (metrics_pre_path, metrics_post_path)
    if any(path is not None for path in metric_paths) and any(
        path is None for path in metric_paths
    ):
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: "
            "pre/post vLLM metrics must be supplied together"
        )
    campaign_arguments = (
        campaign_trace_requests,
        campaign_metric_binding,
    )
    if any(value is not None for value in campaign_arguments) and any(
        value is None for value in campaign_arguments
    ):
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: "
            "campaign trace requests and metric binding must be supplied together"
        )
    if metrics_pre_path is not None and campaign_trace_requests is not None:
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: "
            "task and campaign metric scopes are mutually exclusive"
        )
    metrics_pre_raw: bytes | None = None
    metrics_post_raw: bytes | None = None
    if metrics_pre_path is not None and metrics_post_path is not None:
        for label, path in (
            ("pre", metrics_pre_path),
            ("post", metrics_post_path),
        ):
            if not path.is_file() or path.is_symlink():
                raise Fixed32BoundaryError(
                    f"fixed32 real-task provenance {instance_id}: "
                    f"{label} vLLM metrics are missing or symlinked"
                )
        try:
            metrics_pre_raw = metrics_pre_path.read_bytes()
            metrics_post_raw = metrics_post_path.read_bytes()
        except OSError as exc:
            raise Fixed32BoundaryError(
                f"fixed32 real-task provenance {instance_id}: "
                f"cannot read vLLM metrics: {type(exc).__name__}: {exc}"
            ) from exc
        if not metrics_pre_raw or not metrics_post_raw:
            raise Fixed32BoundaryError(
                f"fixed32 real-task provenance {instance_id}: "
                "pre/post vLLM metrics are empty"
            )
    try:
        base_trace_requests = (
            fixed32_contract.validate_fixed32_trace_model_requests(
                events,
                expected_session_id=fixed32_contract.fixed32_trace_session_id(
                    instance_id
                ),
            )
            if campaign_trace_requests is not None
            else None
        )
        trace_requests = (
            campaign_trace_requests
            if campaign_trace_requests is not None
            else fixed32_contract.validate_fixed32_trace_model_requests(
                events,
                expected_session_id=fixed32_contract.fixed32_trace_session_id(
                    instance_id
                ),
                expected_completed_logical_model_requests=(
                    task_auth_deltas["completed_logical_model_requests"]
                    if metrics_pre_raw is not None
                    else None
                ),
                metrics_pre=metrics_pre_raw,
                metrics_post=metrics_post_raw,
            )
        )
    except fixed32_contract.ContractError as exc:
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: "
            f"trace cannot independently count completed model requests: {exc}"
        ) from exc
    campaign_artifact: dict[str, Any] | None = None
    campaign_metric_evidence_sha256: str | None = None
    if campaign_trace_requests is not None:
        if (
            not isinstance(campaign_trace_requests, dict)
            or base_trace_requests is None
            or not isinstance(base_trace_requests, dict)
            or not isinstance(campaign_metric_binding, dict)
            or set(campaign_metric_binding)
            != {"artifact", "metric_evidence_sha256"}
        ):
            raise Fixed32BoundaryError(
                f"fixed32 real-task provenance {instance_id}: "
                "campaign metric binding is not exact"
            )
        campaign_artifact = campaign_metric_binding["artifact"]
        campaign_metric_evidence_sha256 = campaign_metric_binding[
            "metric_evidence_sha256"
        ]
        task_metric_evidence = trace_requests.get(
            "qwen_compaction_metric_evidence"
        )
        artifact_digest = (
            campaign_artifact.get("sha256")
            if isinstance(campaign_artifact, dict)
            else None
        )
        base_request_ids = base_trace_requests["model_request_ids"]
        base_request_ids_sha256 = hashlib.sha256(
            json.dumps(
                base_request_ids,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        full_request_ids = trace_requests.get("model_request_ids")
        if (
            not isinstance(campaign_artifact, dict)
            or set(campaign_artifact) != {"path", "sha256", "bytes"}
            or not isinstance(campaign_artifact.get("path"), str)
            or not campaign_artifact["path"]
            or not isinstance(artifact_digest, str)
            or len(artifact_digest) != 64
            or any(char not in "0123456789abcdef" for char in artifact_digest)
            or type(campaign_artifact.get("bytes")) is not int
            or campaign_artifact["bytes"] <= 0
            or not isinstance(campaign_metric_evidence_sha256, str)
            or len(campaign_metric_evidence_sha256) != 64
            or any(
                char not in "0123456789abcdef"
                for char in campaign_metric_evidence_sha256
            )
            or not isinstance(task_metric_evidence, dict)
            or task_metric_evidence.get("schema")
            != fixed32_contract.QWEN_CAMPAIGN_TASK_METRIC_SCHEMA
            or task_metric_evidence.get("campaign_metric_evidence_sha256")
            != campaign_metric_evidence_sha256
            or trace_requests.get("qwen_campaign_metric_evidence_sha256")
            != campaign_metric_evidence_sha256
            or task_metric_evidence.get("base_model_request_ids_sha256")
            != base_request_ids_sha256
            or task_metric_evidence.get(
                "trace_completed_requests_before_failed_compactions"
            )
            != len(base_request_ids)
            or trace_requests.get(
                "hidden_successful_compaction_model_requests"
            )
            != base_trace_requests.get(
                "hidden_successful_compaction_model_requests",
                base_trace_requests.get("hidden_compaction_model_requests", 0),
            )
            or trace_requests.get("synthetic_compaction_failure_terminal")
            != base_trace_requests.get(
                "synthetic_compaction_failure_terminal",
                False,
            )
            or not isinstance(full_request_ids, list)
            or full_request_ids[: len(base_request_ids)] != base_request_ids
            or any(
                not isinstance(request_id, str)
                or not request_id.startswith(
                    "qwen-hidden-failed-compaction-sha256:"
                )
                for request_id in full_request_ids[len(base_request_ids) :]
            )
        ):
            raise Fixed32BoundaryError(
                f"fixed32 real-task provenance {instance_id}: "
                "campaign trace request evidence does not bind to the trace"
            )
    qwen_model_request_ids = trace_requests["model_request_ids"]
    request_id_digests = sorted(
        hashlib.sha256(response_id.encode("utf-8")).hexdigest()
        for response_id in qwen_model_request_ids
    )
    if (
        task_auth_deltas["completed_logical_model_requests"]
        != len(qwen_model_request_ids)
        or task_auth_deltas["aborted_logical_requests"] != 0
        or task_auth_deltas["failed_attempts"] != 0
        or task_auth_deltas["accepted_attempts"]
        != (
            task_auth_deltas["completed_attempts"]
            + task_auth_deltas["failed_attempts"]
        )
        or task_auth_deltas["completed_attempts"]
        < task_auth_deltas["completed_logical_model_requests"]
    ):
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: "
            "trace and task-auth request counts do not reconcile"
        )

    usage_max_by_field: dict[str, int] = {}
    positive_usage_record_count = 0
    for usage in usage_records:
        recognized: dict[str, int] = {}
        for key in _FIXED32_TOKEN_USAGE_FIELDS:
            value = usage.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                continue
            recognized[key] = value
            usage_max_by_field[key] = max(value, usage_max_by_field.get(key, 0))
        if any(value > 0 for value in recognized.values()):
            positive_usage_record_count += 1
    positive_token_usage = any(value > 0 for value in usage_max_by_field.values())
    if not positive_token_usage:
        raise Fixed32BoundaryError(
            f"fixed32 real-task provenance {instance_id}: "
            "trace contains no recognized positive token usage"
        )

    return {
        "schema": "fr13-fixed32-real-task-provenance-v3",
        "instance_id": instance_id,
        "task_key_id": task_key_id,
        "qwen_code_version": _FIXED32_QWEN_CODE_VERSION,
        "qwen_system_settings_sha256": _FIXED32_QWEN_SETTINGS_SHA256,
        "qwen_runtime_attestation_sha256": runtime_attestation_sha256,
        "instance_image_identity_sha256": image_identity_sha256,
        "instance_image_id": image_identity["id"],
        "instance_image_repo_digest": image_identity["repo_digest"],
        "agent_placement_sha256": placement_sha256,
        "agent_host_identity": placement["agent_host_identity"],
        "measured_host_identity": placement["measured_host_identity"],
        "qwen_bundle_snapshot": runtime_attestation["bundle_snapshot"],
        "qwen_remote_settings_file_identity_sha256": (
            remote_settings_observation["file_identity_sha256"]
        ),
        "qwen_remote_settings_observation_sha256": (
            remote_settings_observation_sha256
        ),
        "qwen_mounted_runtime_proof_sha256": (
            mounted_runtime_proof_sha256
        ),
        "qwen_mounted_runtime_proof_file_sha256": (
            mounted_runtime_proof_file_sha256
        ),
        "qwen_runtime_attestation_file_sha256": hashlib.sha256(
            attestation_artifacts["qwen_runtime_attestation.json"]
        ).hexdigest(),
        "qwen_runtime_postrun_attestation_file_sha256": hashlib.sha256(
            attestation_artifacts["qwen_runtime_attestation_post.json"]
        ).hexdigest(),
        "trace_completed_logical_model_requests": len(qwen_model_request_ids),
        "hidden_successful_compaction_model_requests": trace_requests.get(
            "hidden_successful_compaction_model_requests",
            trace_requests.get("hidden_compaction_model_requests", 0),
        ),
        "hidden_failed_compaction_model_requests": trace_requests.get(
            "hidden_failed_compaction_model_requests",
            0,
        ),
        "synthetic_compaction_failure_terminal": trace_requests.get(
            "synthetic_compaction_failure_terminal",
            False,
        ),
        "qwen_metric_scope": (
            "campaign" if campaign_artifact is not None else "task"
        ),
        "qwen_campaign_metric_proof": campaign_artifact,
        "qwen_campaign_metric_evidence_sha256": (
            campaign_metric_evidence_sha256
        ),
        "qwen_compaction_metric_evidence": trace_requests.get(
            "qwen_compaction_metric_evidence"
        ),
        "trace_model_request_ids_sha256": hashlib.sha256(
            json.dumps(
                request_id_digests,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "completed_logical_model_requests": task_auth_deltas[
            "completed_logical_model_requests"
        ],
        "aborted_logical_requests": task_auth_deltas[
            "aborted_logical_requests"
        ],
        "accepted_attempts": task_auth_deltas["accepted_attempts"],
        "completed_attempts": task_auth_deltas["completed_attempts"],
        "failed_attempts": task_auth_deltas["failed_attempts"],
        "task_auth_evidence_before_sha256": hashlib.sha256(
            json.dumps(
                task_auth_before,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "task_auth_evidence_after_sha256": hashlib.sha256(
            json.dumps(
                task_auth_after,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "task_auth_evidence_after_ledger_records": task_auth_after[
            "ledger_records"
        ],
        "task_auth_evidence_after_ledger_chain_head_sha256": task_auth_after[
            "ledger_chain_head_sha256"
        ],
        "trace_path": str(trace_path.resolve()),
        "trace_sha256": hashlib.sha256(raw_trace).hexdigest(),
        "trace_bytes": len(raw_trace),
        "event_count": len(events),
        "assistant_event_count": assistant_event_count,
        "assistant_output_event_count": assistant_output_event_count,
        "qwen_assistant_event_count": qwen_assistant_event_count,
        "codex_agent_message_event_count": codex_agent_message_event_count,
        "positive_token_usage": positive_token_usage,
        "usage_record_count": len(usage_records),
        "positive_usage_record_count": positive_usage_record_count,
        "usage_max_by_field": dict(sorted(usage_max_by_field.items())),
        "agent_terminal": {
            key: agent_meta.get(key) for key in _FIXED32_AGENT_TERMINAL_FIELDS
        },
    }


_FIXED32_COUNTER_KEYS = frozenset(
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
_FIXED32_BOUNDARY_SNAPSHOT_SCHEMA = "fr13-fixed32-boundary-snapshot-v4"
_FIXED32_BOUNDARY_TOP_KEYS = frozenset(
    {
        "schema",
        "mode",
        "producer_pid",
        "generation",
        "nonce",
        "action",
        "counters",
        "metrics",
    }
)
_FIXED32_BOUNDARY_METRIC_KEYS = frozenset(
    {
        "fixed32",
        "sfwd",
        "dfwd",
        "cfwd",
        "boot_warm",
        "committer",
        "conv_pregather",
    }
)
_FIXED32_REQUIRED_METRICS = {
    "vllm:fr13_decode_forward_gpu_seconds_total",
    "vllm:fr13_decode_forward_gpu_steps_total",
    "vllm:fr13_decode_forward_gpu_drafts_total",
    "vllm:fr13_decode_step_wall_seconds_total",
    "vllm:fr13_decode_step_wall_drafts_total",
    "vllm:fr13_decode_step_wall_steps_total",
    "vllm:fr13_decode_step_wall_attempts_total",
    "vllm:fr13_decode_step_wall_rejected_total",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:fr13_fixed32_pure_decode_forward_steps_total",
    "vllm:fr13_fixed32_complete_work_census_events_total",
    "vllm:fr13_drafter_gpu_seconds_total",
    "vllm:fr13_drafter_gpu_spans_total",
    "vllm:fr13_committer_gpu_seconds_total",
    "vllm:fr13_committer_gpu_spans_total",
}


def _validate_fixed32_counter_payload(
    counters: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(counters, dict) or set(counters) != _FIXED32_COUNTER_KEYS:
        raise Fixed32BoundaryError(
            f"{label} counters keys mismatch: {sorted(counters) if isinstance(counters, dict) else counters!r}"
        )
    for key in (
        "pure_decode_forward_steps",
        "complete_work_census_events",
        "sfwd_pending",
        "dfwd_pending",
        "cfwd_pending",
    ):
        value = counters[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise Fixed32BoundaryError(f"{label}.{key} must be a nonnegative integer")
    for key in ("work_census_first_forward_step", "work_census_last_forward_step"):
        value = counters[key]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise Fixed32BoundaryError(f"{label}.{key} must be null or nonnegative int")
    if any(counters[key] != 0 for key in ("sfwd_pending", "dfwd_pending", "cfwd_pending")):
        raise Fixed32BoundaryError(f"{label} acknowledged pending timer samples")

    steps = counters["pure_decode_forward_steps"]
    events = counters["complete_work_census_events"]
    first = counters["work_census_first_forward_step"]
    last = counters["work_census_last_forward_step"]
    if events > steps:
        raise Fixed32BoundaryError(f"{label} census events exceed pure-decode steps")
    if events == 0:
        if first is not None or last is not None:
            raise Fixed32BoundaryError(f"{label} empty census must have null first/last")
    elif first is None or last is None or not 0 <= first <= last < steps:
        raise Fixed32BoundaryError(f"{label} census first/last range is invalid")
    return dict(counters)


def _validate_fixed32_ack(ack: Any, *, label: str) -> dict[str, Any]:
    mode = getattr(ack, "mode", None)
    if not isinstance(mode, str):
        raise Fixed32BoundaryError(f"{label}.mode must be a string")
    producer_pid = getattr(ack, "producer_pid", None)
    if type(producer_pid) is not int or producer_pid <= 0:
        raise Fixed32BoundaryError(
            f"{label}.producer_pid must be a positive integer"
        )
    generation = getattr(ack, "generation", None)
    if type(generation) is not int or generation < 0:
        raise Fixed32BoundaryError(
            f"{label}.generation must be a nonnegative integer"
        )
    nonce = getattr(ack, "nonce", None)
    if (
        not isinstance(nonce, str)
        or len(nonce) != 64
        or any(character not in "0123456789abcdef" for character in nonce)
    ):
        raise Fixed32BoundaryError(
            f"{label}.nonce must be 64 lowercase hex characters"
        )
    action = getattr(ack, "action", None)
    if not isinstance(action, str):
        raise Fixed32BoundaryError(f"{label}.action must be a string")
    return _validate_fixed32_counter_payload(
        getattr(ack, "counters", None),
        label=label,
    )


def _fixed32_duplicate_checked(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise Fixed32BoundaryError(f"duplicate boundary snapshot key {key!r}")
        payload[key] = value
    return payload


def _fixed32_reject_constant(value: str) -> Any:
    raise Fixed32BoundaryError(
        f"non-finite boundary snapshot constant {value!r}"
    )


def _fixed32_exact_keys(payload: Any, expected: frozenset[str], label: str) -> None:
    if not isinstance(payload, dict) or frozenset(payload) != expected:
        actual = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise Fixed32BoundaryError(
            f"{label} keys mismatch: expected={sorted(expected)} actual={actual}"
        )


def _fixed32_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Fixed32BoundaryError(f"{label} must be a nonnegative integer")
    return value


def _fixed32_optional_nonnegative_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    return _fixed32_nonnegative_int(value, label)


def _fixed32_nonnegative_int_map(
    value: Any,
    *,
    expected_keys: frozenset[str],
    label: str,
) -> dict[str, int]:
    if not isinstance(value, dict) or frozenset(value) != expected_keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise Fixed32BoundaryError(
            f"{label} keys mismatch: expected={sorted(expected_keys)} actual={actual}"
        )
    return {
        key: _fixed32_nonnegative_int(item, f"{label}.{key}")
        for key, item in value.items()
    }


def _fixed32_nonnegative_int_list(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, list):
        raise Fixed32BoundaryError(f"{label} must be a list")
    return [
        _fixed32_nonnegative_int(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    ]


def _fixed32_nonnegative_float(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise Fixed32BoundaryError(f"{label} must be finite and nonnegative")
    return float(value)


def _load_fixed32_boundary_snapshot(
    *,
    base_path: Path,
    ack: Any,
    server_capacity: int,
    allow_incomplete_layer_batch_coverage: bool = False,
) -> tuple[dict[str, Any], Path, str]:
    if type(allow_incomplete_layer_batch_coverage) is not bool:
        raise Fixed32BoundaryError(
            "generation boundary snapshot coverage policy must be boolean"
        )
    ack_counters = _validate_fixed32_ack(
        ack,
        label="generation boundary snapshot ack",
    )
    path = Path(f"{base_path}.{ack.generation}.json")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise Fixed32BoundaryError(
            f"cannot read generation boundary snapshot {path}: {error}"
        ) from error
    if not raw or len(raw) > 1024 * 1024:
        raise Fixed32BoundaryError(
            f"generation boundary snapshot has invalid size: {path}"
        )
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_fixed32_duplicate_checked,
            parse_constant=_fixed32_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Fixed32BoundaryError(
            f"invalid generation boundary snapshot {path}: {error}"
        ) from error
    _fixed32_exact_keys(payload, _FIXED32_BOUNDARY_TOP_KEYS, str(path))
    producer_pid = _fixed32_nonnegative_int(
        payload["producer_pid"], f"{path}:producer_pid"
    )
    _fixed32_nonnegative_int(
        payload["generation"], f"{path}:generation"
    )
    if producer_pid == 0:
        raise Fixed32BoundaryError(f"{path}:producer_pid must be positive")
    snapshot_counters = _validate_fixed32_counter_payload(
        payload["counters"],
        label=f"{path}:snapshot",
    )
    if (
        payload["schema"] != _FIXED32_BOUNDARY_SNAPSHOT_SCHEMA
        or payload["mode"] != ack.mode
        or payload["producer_pid"] != ack.producer_pid
        or payload["generation"] != ack.generation
        or payload["nonce"] != ack.nonce
        or payload["action"] != ack.action
        or snapshot_counters != ack_counters
    ):
        raise Fixed32BoundaryError(
            f"generation boundary snapshot does not bind to ack: {path}"
        )
    metrics = payload["metrics"]
    _fixed32_exact_keys(
        metrics,
        _FIXED32_BOUNDARY_METRIC_KEYS,
        f"{path}:metrics",
    )
    fixed = metrics["fixed32"]
    _fixed32_exact_keys(
        fixed,
        frozenset(
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
            }
        ),
        f"{path}:fixed32",
    )
    steps = _fixed32_nonnegative_int(
        fixed["pure_decode_forward_steps"], f"{path}:fixed32.steps"
    )
    events = _fixed32_nonnegative_int(
        fixed["complete_work_census_events"], f"{path}:fixed32.events"
    )
    spec_drafts = _fixed32_nonnegative_int(
        fixed["spec_drafts"], f"{path}:fixed32.spec_drafts"
    )
    complete_spec_rows = _fixed32_nonnegative_int(
        fixed["complete_spec_rows"],
        f"{path}:fixed32.complete_spec_rows",
    )
    spec_tokens = _fixed32_nonnegative_int(
        fixed["spec_tokens"], f"{path}:fixed32.spec_tokens"
    )
    first_forward_step = _fixed32_optional_nonnegative_int(
        fixed["first_forward_step"],
        f"{path}:fixed32.first_forward_step",
    )
    last_forward_step = _fixed32_optional_nonnegative_int(
        fixed["last_forward_step"],
        f"{path}:fixed32.last_forward_step",
    )
    if (
        steps != ack_counters["pure_decode_forward_steps"]
        or events != ack_counters["complete_work_census_events"]
        or first_forward_step
        != ack_counters["work_census_first_forward_step"]
        or last_forward_step
        != ack_counters["work_census_last_forward_step"]
        or complete_spec_rows != spec_drafts
        or spec_tokens != 31 * spec_drafts
    ):
        raise Fixed32BoundaryError(f"{path}: fixed counters do not reconcile")
    histogram = fixed["batch_histogram"]
    if not isinstance(histogram, dict) or set(histogram) != {"1", "2", "3", "4"}:
        raise Fixed32BoundaryError(f"{path}: batch histogram keys mismatch")
    histogram = {
        int(batch): _fixed32_nonnegative_int(count, f"{path}:batch{batch}")
        for batch, count in histogram.items()
    }
    if (
        sum(histogram.values()) != events
        or sum(batch * count for batch, count in histogram.items()) != spec_drafts
        or any(batch > server_capacity and count for batch, count in histogram.items())
    ):
        raise Fixed32BoundaryError(f"{path}: batch histogram does not reconcile")
    digest = fixed["events_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise Fixed32BoundaryError(f"{path}: events digest is malformed")

    sfwd = metrics["sfwd"]
    _fixed32_exact_keys(
        sfwd,
        frozenset(
            {
                "gpu_seconds",
                "steps",
                "drafts",
                "wall_seconds",
                "wall_drafts",
                "wall_steps",
                "wall_rejected",
            }
        ),
        f"{path}:sfwd",
    )
    _fixed32_nonnegative_float(sfwd["gpu_seconds"], f"{path}:sfwd.gpu_seconds")
    _fixed32_nonnegative_float(sfwd["wall_seconds"], f"{path}:sfwd.wall_seconds")
    for key in ("steps", "drafts", "wall_drafts", "wall_steps", "wall_rejected"):
        _fixed32_nonnegative_int(sfwd[key], f"{path}:sfwd.{key}")
    if sfwd["steps"] != steps or sfwd["drafts"] != spec_drafts:
        raise Fixed32BoundaryError(f"{path}: SFWD counters do not reconcile")
    for label in ("dfwd", "cfwd"):
        span = metrics[label]
        _fixed32_exact_keys(
            span,
            frozenset({"gpu_seconds", "spans"}),
            f"{path}:{label}",
        )
        _fixed32_nonnegative_float(
            span["gpu_seconds"], f"{path}:{label}.gpu_seconds"
        )
        if _fixed32_nonnegative_int(span["spans"], f"{path}:{label}.spans") != events:
            raise Fixed32BoundaryError(f"{path}: {label} spans do not reconcile")

    boot_warm = metrics["boot_warm"]
    boot_warm_keys = frozenset(
        {
            "schema",
            "classification",
            "hardware_scope",
            "wrapper_bookkeeping_warmed",
            "copy_source_dtype",
            "copy_destination_dtype",
            "mode",
            "capacity",
            "vocab_size",
            "batches",
            "taw_executions",
            "output_copy_pairs",
            "slot_copy_pairs",
            "spec_copy_pairs",
            "flags_zero_fills",
            "persistent_copy_state_restored",
            "flags_state_restored",
            "conv_commit_direct_launches",
            "conv_commit_gather_launches",
            "conv_commit_scatter_launches",
            "committer_replays",
            "observed_event_absent",
            "pending_event_absent",
            "taw_cache_lease_current",
            "taw_rng_state_restored",
            "taw_staging_state_restored",
            "taw_measured_state_restored",
            "committer_route_lease_current",
            "committer_bank_state_restored",
            "committer_conv_bank_state_restored",
            "committer_conv_staging_state_restored",
            "committer_alias_destination_contract",
            "committer_input_state_restored",
            "committer_measured_state_restored",
            "committer_scratch_overwrite_proven",
        }
    )
    _fixed32_exact_keys(
        boot_warm,
        boot_warm_keys,
        f"{path}:boot_warm",
    )
    boot_capacity = _fixed32_nonnegative_int(
        boot_warm["capacity"],
        f"{path}:boot_warm.capacity",
    )
    boot_vocab = _fixed32_nonnegative_int(
        boot_warm["vocab_size"],
        f"{path}:boot_warm.vocab_size",
    )
    boot_batches = _fixed32_nonnegative_int_list(
        boot_warm["batches"],
        label=f"{path}:boot_warm.batches",
    )
    taw_executions = _fixed32_nonnegative_int(
        boot_warm["taw_executions"],
        f"{path}:boot_warm.taw_executions",
    )
    output_copy_pairs = _fixed32_nonnegative_int(
        boot_warm["output_copy_pairs"],
        f"{path}:boot_warm.output_copy_pairs",
    )
    slot_copy_pairs = _fixed32_nonnegative_int(
        boot_warm["slot_copy_pairs"],
        f"{path}:boot_warm.slot_copy_pairs",
    )
    spec_copy_pairs = _fixed32_nonnegative_int(
        boot_warm["spec_copy_pairs"],
        f"{path}:boot_warm.spec_copy_pairs",
    )
    flags_zero_fills = _fixed32_nonnegative_int(
        boot_warm["flags_zero_fills"],
        f"{path}:boot_warm.flags_zero_fills",
    )
    conv_direct = _fixed32_nonnegative_int(
        boot_warm["conv_commit_direct_launches"],
        f"{path}:boot_warm.conv_commit_direct_launches",
    )
    conv_gathers = _fixed32_nonnegative_int(
        boot_warm["conv_commit_gather_launches"],
        f"{path}:boot_warm.conv_commit_gather_launches",
    )
    conv_scatters = _fixed32_nonnegative_int(
        boot_warm["conv_commit_scatter_launches"],
        f"{path}:boot_warm.conv_commit_scatter_launches",
    )
    committer_replays = _fixed32_nonnegative_int(
        boot_warm["committer_replays"],
        f"{path}:boot_warm.committer_replays",
    )
    boot_true_fields = (
        "observed_event_absent",
        "pending_event_absent",
        "persistent_copy_state_restored",
        "flags_state_restored",
        "taw_cache_lease_current",
        "taw_rng_state_restored",
        "taw_staging_state_restored",
        "taw_measured_state_restored",
        "committer_route_lease_current",
        "committer_bank_state_restored",
        "committer_conv_bank_state_restored",
        "committer_conv_staging_state_restored",
        "committer_input_state_restored",
        "committer_measured_state_restored",
        "committer_scratch_overwrite_proven",
    )
    if (
        boot_warm["schema"] != "fr13-fixed32-boot-warm-v3"
        or boot_warm["classification"] != "unmeasured_boot"
        or boot_warm["hardware_scope"] != "device_postprocess_kernels"
        or boot_warm["wrapper_bookkeeping_warmed"] is not False
        or boot_warm["copy_source_dtype"] != "torch.int64"
        or boot_warm["copy_destination_dtype"] != "torch.int32"
        or boot_warm["mode"] != ack.mode
        or boot_capacity != server_capacity
        or boot_vocab <= 0
        or boot_batches != list(range(1, server_capacity + 1))
        or taw_executions != server_capacity
        or output_copy_pairs != server_capacity
        or slot_copy_pairs != server_capacity * (server_capacity + 1) // 2
        or spec_copy_pairs != server_capacity
        or flags_zero_fills != 1
        or conv_direct != server_capacity
        or conv_gathers != 0
        or conv_scatters != 0
        or committer_replays != server_capacity
        or boot_warm["committer_alias_destination_contract"]
        != "exact_alias_only_16x3"
        or any(boot_warm[key] is not True for key in boot_true_fields)
    ):
        raise Fixed32BoundaryError(
            f"{path}: boot-warm evidence does not prove unmeasured readiness"
        )

    committer = metrics["committer"]
    conv = metrics["conv_pregather"]
    expected_by_batch = {str(batch): histogram[batch] for batch in range(1, 5)}
    expected_capture_by_batch = {
        str(batch): int(batch <= server_capacity) for batch in range(1, 5)
    }
    zero_by_batch = {str(batch): 0 for batch in range(1, 5)}
    expected_ready_capacities = {
        str(batch): server_capacity
        for batch in range(1, server_capacity + 1)
    }
    expected_preseeded_batches = list(range(1, server_capacity + 1))
    batch_keys = frozenset(str(batch) for batch in range(1, 5))
    ready_capacity_keys = frozenset(expected_ready_capacities)
    _fixed32_exact_keys(
        committer,
        frozenset(
            {
                "actual_replays_by_batch",
                "actual_replays_enqueued",
                "all_batches_ready",
                "captures",
                "fast_route_ready",
                "gate_precompute_launches",
                "maximum_ready_capacity",
                "layer_batch_gate_attempts_by_batch",
                "layer_batch_gate_coverage_mask_by_batch",
                "layer_batch_gate_passed_by_batch",
                "metadata_fusion_consumed_by_batch",
                "metadata_fusion_fallbacks_by_batch",
                "metadata_fusion_published_by_batch",
                "nonpure_dispatch",
                "nonpure_committer_replays_by_batch",
                "nonpure_committer_replays_enqueued",
                "preseeded_batches",
                "preseeded_graphs",
                "ready_capacities",
                "required_capacity",
            }
        ),
        f"{path}:committer",
    )
    _fixed32_exact_keys(
        conv,
        frozenset(
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
            }
        ),
        f"{path}:conv_pregather",
    )
    for key in (
        "actual_replays_enqueued",
        "captures",
        "gate_precompute_launches",
        "maximum_ready_capacity",
        "nonpure_committer_replays_enqueued",
        "preseeded_graphs",
        "required_capacity",
    ):
        _fixed32_nonnegative_int(committer[key], f"{path}:committer.{key}")
    for key in (
        "pointer_entries",
        "max_batch_size",
        "graph_capture_stages",
        "profile_capture_stages",
        "aux_capture_stages",
        "actual_stages",
        "graph_replay_stages",
    ):
        _fixed32_nonnegative_int(conv[key], f"{path}:conv_pregather.{key}")
    committer_by_batch = _fixed32_nonnegative_int_map(
        committer["actual_replays_by_batch"],
        expected_keys=batch_keys,
        label=f"{path}:committer.actual_replays_by_batch",
    )
    metadata_fusion_by_kind = {
        key: _fixed32_nonnegative_int_map(
            committer[key],
            expected_keys=batch_keys,
            label=f"{path}:committer.{key}",
        )
        for key in (
            "metadata_fusion_published_by_batch",
            "metadata_fusion_consumed_by_batch",
            "metadata_fusion_fallbacks_by_batch",
        )
    }
    metadata_fusion_active = any(
        value
        for counters in metadata_fusion_by_kind.values()
        for value in counters.values()
    )
    if metadata_fusion_active and (
        metadata_fusion_by_kind["metadata_fusion_published_by_batch"]
        != committer_by_batch
        or metadata_fusion_by_kind["metadata_fusion_consumed_by_batch"]
        != committer_by_batch
        or any(
            metadata_fusion_by_kind["metadata_fusion_fallbacks_by_batch"].values()
        )
    ):
        raise Fixed32BoundaryError(
            f"{path}: committer metadata fusion counters do not reconcile"
        )
    layer_batch_gate_passed_by_batch = _fixed32_nonnegative_int_map(
        committer["layer_batch_gate_passed_by_batch"],
        expected_keys=ready_capacity_keys,
        label=f"{path}:committer.layer_batch_gate_passed_by_batch",
    )
    _fixed32_nonnegative_int_map(
        committer["layer_batch_gate_attempts_by_batch"],
        expected_keys=ready_capacity_keys,
        label=f"{path}:committer.layer_batch_gate_attempts_by_batch",
    )
    layer_batch_gate_coverage_mask_by_batch = _fixed32_nonnegative_int_map(
        committer["layer_batch_gate_coverage_mask_by_batch"],
        expected_keys=ready_capacity_keys,
        label=f"{path}:committer.layer_batch_gate_coverage_mask_by_batch",
    )
    if any(value not in (0, 1) for value in layer_batch_gate_passed_by_batch.values()):
        raise Fixed32BoundaryError(
            f"{path}: committer layer-batch gate pass state is not boolean"
        )
    expected_full_coverage = {
        key: _FIXED32_COMMITTER_ACCEPTED_LENGTH_FULL_MASK
        for key in ready_capacity_keys
    }
    if any(
        value > _FIXED32_COMMITTER_ACCEPTED_LENGTH_FULL_MASK
        for value in layer_batch_gate_coverage_mask_by_batch.values()
    ):
        raise Fixed32BoundaryError(
            f"{path}: committer layer-batch accepted-length coverage contains "
            "unreachable depths"
        )
    if (
        not allow_incomplete_layer_batch_coverage
        and layer_batch_gate_coverage_mask_by_batch != expected_full_coverage
    ):
        raise Fixed32BoundaryError(
            f"{path}: committer layer-batch accepted-length coverage is "
            "incomplete before measurement"
        )
    if layer_batch_gate_passed_by_batch != {
        key: int(
            layer_batch_gate_coverage_mask_by_batch[key]
            == _FIXED32_COMMITTER_ACCEPTED_LENGTH_FULL_MASK
        )
        for key in ready_capacity_keys
    }:
        raise Fixed32BoundaryError(
            f"{path}: committer layer-batch gate pass/coverage state diverged"
        )
    nonpure_replays_by_batch = _fixed32_nonnegative_int_map(
        committer["nonpure_committer_replays_by_batch"],
        expected_keys=batch_keys,
        label=(
            f"{path}:committer.nonpure_committer_replays_by_batch"
        ),
    )
    nonpure_dispatch = _fixed32_nonnegative_int_map(
        committer["nonpure_dispatch"],
        expected_keys=frozenset(
            {
                "guarded_steps",
                "piecewise_steps",
                "none_steps",
                "forbidden_full_steps",
            }
        ),
        label=f"{path}:committer.nonpure_dispatch",
    )
    committer_ready_capacities = _fixed32_nonnegative_int_map(
        committer["ready_capacities"],
        expected_keys=ready_capacity_keys,
        label=f"{path}:committer.ready_capacities",
    )
    committer_preseeded_batches = _fixed32_nonnegative_int_list(
        committer["preseeded_batches"],
        label=f"{path}:committer.preseeded_batches",
    )
    conv_capture_by_batch = _fixed32_nonnegative_int_map(
        conv["graph_capture_stages_by_batch"],
        expected_keys=batch_keys,
        label=f"{path}:conv_pregather.graph_capture_stages_by_batch",
    )
    conv_host_by_batch = _fixed32_nonnegative_int_map(
        conv["actual_stages_by_batch"],
        expected_keys=batch_keys,
        label=f"{path}:conv_pregather.actual_stages_by_batch",
    )
    conv_replays_by_batch = _fixed32_nonnegative_int_map(
        conv["graph_replay_stages_by_batch"],
        expected_keys=batch_keys,
        label=f"{path}:conv_pregather.graph_replay_stages_by_batch",
    )
    conv_preseeded_batches = _fixed32_nonnegative_int_list(
        conv["preseeded_batches"],
        label=f"{path}:conv_pregather.preseeded_batches",
    )
    nonpure_replays = committer[
        "nonpure_committer_replays_enqueued"
    ]
    expected_raw_by_batch = {
        str(batch): expected_by_batch[str(batch)]
        + nonpure_replays_by_batch[str(batch)]
        for batch in range(1, 5)
    }
    if (
        committer["all_batches_ready"] is not True
        or committer["fast_route_ready"] is not True
        or committer["required_capacity"] != server_capacity
        or committer["maximum_ready_capacity"] != server_capacity
        or committer["preseeded_graphs"] != server_capacity
        or committer["captures"] != server_capacity
        or committer_preseeded_batches != expected_preseeded_batches
        or committer_ready_capacities != expected_ready_capacities
        or nonpure_dispatch["guarded_steps"]
        != (
            nonpure_dispatch["piecewise_steps"]
            + nonpure_dispatch["none_steps"]
            + nonpure_dispatch["forbidden_full_steps"]
        )
        or nonpure_dispatch["forbidden_full_steps"] != 0
        or nonpure_replays != sum(nonpure_replays_by_batch.values())
        or nonpure_replays > nonpure_dispatch["guarded_steps"]
        or any(
            nonpure_replays_by_batch[str(batch)]
            for batch in range(server_capacity, 5)
        )
        or committer["actual_replays_enqueued"]
        != events + nonpure_replays
        or committer_by_batch != expected_raw_by_batch
        or conv["preseeded"] is not True
        or conv["pointer_entries"] != 48
        or conv_preseeded_batches != expected_preseeded_batches
        or conv["max_batch_size"] != server_capacity
        or conv["graph_capture_stages"] != server_capacity
        or conv_capture_by_batch != expected_capture_by_batch
        or conv["profile_capture_stages"] != 0
        or conv["aux_capture_stages"] != 0
        or conv["actual_stages"] != 0
        or conv_host_by_batch != zero_by_batch
        or conv["graph_replay_stages"] != events
        or conv_replays_by_batch != expected_by_batch
    ):
        raise Fixed32BoundaryError(
            f"{path}: committer/nonpure/in-graph pregather counters do not reconcile"
        )
    return payload, path, hashlib.sha256(raw).hexdigest()


def _fixed32_metrics_snapshot(
    *,
    metrics_url: str,
    snapshot: dict[str, Any],
) -> str:
    """Render required counters from one immutable runtime generation."""
    raw = _metrics_text(metrics_url, strict=True)
    retained = []
    for line in raw.splitlines():
        stripped = line.lstrip()
        series = (
            stripped.split("{", 1)[0].split(None, 1)[0]
            if stripped
            else ""
        )
        if series not in _FIXED32_REQUIRED_METRICS:
            retained.append(line)
    metrics = snapshot["metrics"]
    fixed = metrics["fixed32"]
    sfwd = metrics["sfwd"]
    dfwd = metrics["dfwd"]
    cfwd = metrics["cfwd"]
    exact = (
        (
            "vllm:fr13_decode_forward_gpu_seconds_total",
            sfwd["gpu_seconds"],
            "",
        ),
        ("vllm:fr13_decode_forward_gpu_steps_total", sfwd["steps"], ""),
        ("vllm:fr13_decode_forward_gpu_drafts_total", sfwd["drafts"], ""),
        (
            "vllm:fr13_decode_step_wall_seconds_total",
            sfwd["wall_seconds"],
            "",
        ),
        (
            "vllm:fr13_decode_step_wall_drafts_total",
            sfwd["wall_drafts"],
            "",
        ),
        ("vllm:fr13_decode_step_wall_steps_total", sfwd["wall_steps"], ""),
        (
            "vllm:fr13_decode_step_wall_attempts_total",
            sfwd["wall_steps"] + sfwd["wall_rejected"],
            "",
        ),
        (
            "vllm:fr13_decode_step_wall_rejected_total",
            sfwd["wall_rejected"],
            "",
        ),
        (
            "vllm:spec_decode_num_drafts_total",
            fixed["spec_drafts"],
            '{engine="0",model_name="qwen3.6-27b"}',
        ),
        (
            "vllm:spec_decode_num_draft_tokens_total",
            fixed["spec_tokens"],
            '{engine="0",model_name="qwen3.6-27b"}',
        ),
        (
            "vllm:fr13_fixed32_pure_decode_forward_steps_total",
            fixed["pure_decode_forward_steps"],
            "",
        ),
        (
            "vllm:fr13_fixed32_complete_work_census_events_total",
            fixed["complete_work_census_events"],
            "",
        ),
        ("vllm:fr13_drafter_gpu_seconds_total", dfwd["gpu_seconds"], ""),
        ("vllm:fr13_drafter_gpu_spans_total", dfwd["spans"], ""),
        ("vllm:fr13_committer_gpu_seconds_total", cfwd["gpu_seconds"], ""),
        ("vllm:fr13_committer_gpu_spans_total", cfwd["spans"], ""),
    )
    retained.extend(
        f"{name}{labels} {value:.17g}"
        if isinstance(value, float)
        else f"{name}{labels} {value}"
        for name, value, labels in exact
    )
    return "\n".join(retained) + "\n"


class _Fixed32TaskBracket:
    """One task's strict flush-before-metrics evidence bracket."""

    allow_incomplete_layer_batch_coverage = False
    finish_arm_before_post_snapshot = False

    def __init__(
        self,
        *,
        client: Any,
        task_dir: Path,
        instance_id: str,
        boundary_snapshot_base: Path,
        server_capacity: int,
        taw_real_task_arm: _Fixed32TawRealTaskArm | None = None,
    ) -> None:
        self.client = client
        self.task_dir = task_dir
        self.instance_id = instance_id
        self.boundary_snapshot_base = boundary_snapshot_base
        self.server_capacity = server_capacity
        self.taw_real_task_arm = taw_real_task_arm
        self.pre_ack = None
        self.post_ack = None
        self.pre_snapshot_ref = None
        self.post_snapshot_ref = None
        self.pre_layer_batch_gate_attempts_by_batch = None
        self.pre_layer_batch_gate_coverage_mask_by_batch = None
        self.post_layer_batch_gate_attempts_by_batch = None
        self.post_layer_batch_gate_coverage_mask_by_batch = None
        self.post_attempted = False
        self.artifact_path = task_dir / "fixed32_task_boundary.json"
        self.taw_arm_artifact_path = task_dir / (
            taw_real_task_arm.artifact_name
            if taw_real_task_arm is not None
            else "fixed32_taw_real_task_arm.json"
        )

    @property
    def started(self) -> bool:
        return self.pre_ack is not None

    @property
    def complete(self) -> bool:
        return self.post_ack is not None

    def _load_boundary_snapshot(self, ack: Any) -> tuple[dict[str, Any], Path, str]:
        return _load_fixed32_boundary_snapshot(
            base_path=self.boundary_snapshot_base,
            ack=ack,
            server_capacity=self.server_capacity,
            allow_incomplete_layer_batch_coverage=(
                self.allow_incomplete_layer_batch_coverage
            ),
        )

    def _validate_layer_batch_gate_transition(
        self,
        *,
        post_attempts: dict[str, int],
        post_coverage: dict[str, int],
    ) -> None:
        if post_attempts != self.pre_layer_batch_gate_attempts_by_batch:
            raise Fixed32BoundaryError(
                "fixed32 task interval attempted a committer layer-batch byte gate"
            )
        if post_coverage != self.pre_layer_batch_gate_coverage_mask_by_batch:
            raise Fixed32BoundaryError(
                "fixed32 task interval changed committer layer-batch "
                "accepted-length coverage"
            )

    def _artifact_classification(self) -> dict[str, Any]:
        return {}

    def _finish_real_task_arm(self) -> None:
        if self.taw_real_task_arm is None:
            return
        self.taw_real_task_arm.finish()
        self._write_taw_arm_artifact()

    def _write_artifact(self) -> None:
        pre = self.pre_ack.as_dict() if self.pre_ack is not None else None
        post = self.post_ack.as_dict() if self.post_ack is not None else None
        interval = None
        if pre is not None and post is not None:
            interval = {
                "start_forward_step": pre["counters"]["pure_decode_forward_steps"],
                "end_forward_step": post["counters"]["pure_decode_forward_steps"],
                "expected_complete_events": (
                    post["counters"]["complete_work_census_events"]
                    - pre["counters"]["complete_work_census_events"]
                ),
            }
        payload = {
            "schema": "fr13-fixed32-task-boundary-v1",
            "instance_id": self.instance_id,
            "mode": self.client.mode,
            "producer_pid": self.client.producer_pid,
            "pre": pre,
            "post": post,
            "pre_runtime_snapshot": self.pre_snapshot_ref,
            "post_runtime_snapshot": self.post_snapshot_ref,
            "forward_step_interval": interval,
        }
        payload.update(self._artifact_classification())
        temporary = self.artifact_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.artifact_path)

    def _write_taw_arm_artifact(self) -> None:
        if self.taw_real_task_arm is None:
            return
        temporary = self.taw_arm_artifact_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                self.taw_real_task_arm.as_dict(),
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.taw_arm_artifact_path)

    def pre(self, metrics_path: Path) -> None:
        if self.started or self.post_attempted:
            raise Fixed32BoundaryError("fixed32 task pre bracket was invoked twice")
        try:
            ack = self.client.snapshot()
            _validate_fixed32_ack(ack, label="pre")
            snapshot, snapshot_path, snapshot_sha = self._load_boundary_snapshot(
                ack
            )
            gate_attempts = dict(
                snapshot["metrics"]["committer"][
                    "layer_batch_gate_attempts_by_batch"
                ]
            )
            gate_coverage = dict(
                snapshot["metrics"]["committer"][
                    "layer_batch_gate_coverage_mask_by_batch"
                ]
            )
            metrics = _fixed32_metrics_snapshot(
                metrics_url=DEFAULT_METRICS_URL,
                snapshot=snapshot,
            )
        except Exception as exc:
            raise Fixed32BoundaryError(
                f"fixed32 pre bracket failed: {type(exc).__name__}: {exc}"
            ) from exc
        self.pre_ack = ack
        self.pre_snapshot_ref = {
            "schema": snapshot["schema"],
            "generation": ack.generation,
            "path": str(snapshot_path),
            "sha256": snapshot_sha,
        }
        self.pre_layer_batch_gate_attempts_by_batch = gate_attempts
        self.pre_layer_batch_gate_coverage_mask_by_batch = gate_coverage
        try:
            metrics_path.write_text(metrics, encoding="utf-8")
            if self.taw_real_task_arm is not None:
                self.taw_real_task_arm.start()
                self._write_taw_arm_artifact()
            self._write_artifact()
        except Exception as exc:
            cleanup_error = None
            if (
                self.taw_real_task_arm is not None
                and self.taw_real_task_arm.active
            ):
                try:
                    self._finish_real_task_arm()
                except Exception as error:  # noqa: BLE001
                    cleanup_error = error
            self.pre_ack = None
            self.pre_snapshot_ref = None
            self.pre_layer_batch_gate_attempts_by_batch = None
            self.pre_layer_batch_gate_coverage_mask_by_batch = None
            detail = f"{type(exc).__name__}: {exc}"
            if cleanup_error is not None:
                detail += (
                    "; arm cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise Fixed32BoundaryError(
                f"fixed32 pre bracket publication failed: {detail}"
            ) from exc

    def post(self, metrics_path: Path) -> dict[str, Any]:
        if not self.started:
            raise Fixed32BoundaryError("fixed32 post bracket has no pre bracket")
        if self.post_attempted:
            if self.complete:
                return json.loads(self.artifact_path.read_text(encoding="utf-8"))
            raise Fixed32BoundaryError("fixed32 post bracket already failed")
        self.post_attempted = True
        arm_error = None
        if (
            self.finish_arm_before_post_snapshot
            and self.taw_real_task_arm is not None
        ):
            try:
                self._finish_real_task_arm()
            except Exception as exc:  # noqa: BLE001
                arm_error = exc
        snapshot_error = None
        if arm_error is None:
            try:
                ack = self.client.snapshot()
                counters = _validate_fixed32_ack(ack, label="post")
                pre_counters = self.pre_ack.counters
                start = pre_counters["pure_decode_forward_steps"]
                end = counters["pure_decode_forward_steps"]
                event_delta = (
                    counters["complete_work_census_events"]
                    - pre_counters["complete_work_census_events"]
                )
                if end <= start:
                    raise Fixed32BoundaryError(
                        "fixed32 real task produced no pure-decode step"
                    )
                if event_delta != end - start:
                    raise Fixed32BoundaryError(
                        "fixed32 task interval has incomplete census events: "
                        f"steps={end - start} events={event_delta}"
                    )
                snapshot, snapshot_path, snapshot_sha = (
                    self._load_boundary_snapshot(ack)
                )
                post_gate_attempts = dict(
                    snapshot["metrics"]["committer"][
                        "layer_batch_gate_attempts_by_batch"
                    ]
                )
                post_gate_coverage = dict(
                    snapshot["metrics"]["committer"][
                        "layer_batch_gate_coverage_mask_by_batch"
                    ]
                )
                self._validate_layer_batch_gate_transition(
                    post_attempts=post_gate_attempts,
                    post_coverage=post_gate_coverage,
                )
                self.post_layer_batch_gate_attempts_by_batch = post_gate_attempts
                self.post_layer_batch_gate_coverage_mask_by_batch = post_gate_coverage
                metrics = _fixed32_metrics_snapshot(
                    metrics_url=DEFAULT_METRICS_URL,
                    snapshot=snapshot,
                )
            except Exception as exc:
                snapshot_error = exc
        if (
            not self.finish_arm_before_post_snapshot
            and self.taw_real_task_arm is not None
        ):
            try:
                self._finish_real_task_arm()
            except Exception as exc:  # noqa: BLE001
                arm_error = exc
        if snapshot_error is not None or arm_error is not None:
            try:
                self._write_artifact()
            except Exception:
                pass
            details = []
            if snapshot_error is not None:
                details.append(
                    f"snapshot={type(snapshot_error).__name__}: {snapshot_error}"
                )
            if arm_error is not None:
                details.append(f"arm={type(arm_error).__name__}: {arm_error}")
            cause = snapshot_error if snapshot_error is not None else arm_error
            raise Fixed32BoundaryError(
                "fixed32 post bracket failed: " + "; ".join(details)
            ) from cause
        self.post_ack = ack
        self.post_snapshot_ref = {
            "schema": snapshot["schema"],
            "generation": ack.generation,
            "path": str(snapshot_path),
            "sha256": snapshot_sha,
        }
        try:
            metrics_path.write_text(metrics, encoding="utf-8")
            self._write_artifact()
        except Exception as exc:
            self.post_ack = None
            self.post_snapshot_ref = None
            try:
                self._write_artifact()
            except Exception:
                pass
            raise Fixed32BoundaryError(
                "fixed32 post bracket artifact failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return json.loads(self.artifact_path.read_text(encoding="utf-8"))


class _Fixed32CfwdQualificationTaskBracket(_Fixed32TaskBracket):
    """Collect non-timing CFWD byte coverage from one authenticated B1 task."""

    allow_incomplete_layer_batch_coverage = True
    finish_arm_before_post_snapshot = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.server_capacity != 1:
            raise Fixed32BoundaryError(
                "fixed32 CFWD qualification is B1/sequential only"
            )
        if not isinstance(
            self.taw_real_task_arm,
            _Fixed32CommitterLayerBatchRealTaskArm,
        ):
            raise Fixed32BoundaryError(
                "fixed32 CFWD qualification requires its dedicated real-task arm"
            )

    def _validate_layer_batch_gate_transition(
        self,
        *,
        post_attempts: dict[str, int],
        post_coverage: dict[str, int],
    ) -> None:
        pre_attempts = self.pre_layer_batch_gate_attempts_by_batch
        pre_coverage = self.pre_layer_batch_gate_coverage_mask_by_batch
        if (
            not isinstance(pre_attempts, dict)
            or not isinstance(pre_coverage, dict)
            or set(post_attempts) != set(pre_attempts)
            or set(post_coverage) != set(pre_coverage)
        ):
            raise Fixed32BoundaryError(
                "fixed32 CFWD qualification gate maps changed shape"
            )
        for batch in sorted(pre_coverage):
            attempt_delta = post_attempts[batch] - pre_attempts[batch]
            new_mask = post_coverage[batch] & ~pre_coverage[batch]
            if (
                attempt_delta < 0
                or post_coverage[batch] & pre_coverage[batch]
                != pre_coverage[batch]
                or attempt_delta != new_mask.bit_count()
            ):
                raise Fixed32BoundaryError(
                    "fixed32 CFWD qualification coverage did not advance "
                    "monotonically"
                )

    @staticmethod
    def _covered_lengths(mask: int) -> list[int]:
        return [
            length
            for length in range(
                _FIXED32_COMMITTER_ACCEPTED_LENGTH_FULL_MASK.bit_length()
            )
            if mask & (1 << length)
        ]

    def _artifact_classification(self) -> dict[str, Any]:
        pre_attempts = self.pre_layer_batch_gate_attempts_by_batch
        pre_coverage = self.pre_layer_batch_gate_coverage_mask_by_batch
        post_attempts = self.post_layer_batch_gate_attempts_by_batch
        post_coverage = self.post_layer_batch_gate_coverage_mask_by_batch
        coverage = None
        if isinstance(pre_attempts, dict) and isinstance(pre_coverage, dict):
            coverage = {
                "accepted_length_full_mask": (
                    _FIXED32_COMMITTER_ACCEPTED_LENGTH_FULL_MASK
                ),
                "pre_attempts_by_batch": dict(pre_attempts),
                "pre_coverage_mask_by_batch": dict(pre_coverage),
                "post_attempts_by_batch": None,
                "post_coverage_mask_by_batch": None,
                "attempt_delta_by_batch": None,
                "new_coverage_mask_by_batch": None,
                "newly_covered_lengths_by_batch": None,
                "remaining_coverage_mask_by_batch": None,
                "coverage_complete": False,
                "shadow_reference_replays": None,
                "shadow_candidate_replays": None,
                "new_depth_reference_served_replays": None,
                "new_depth_served_route": "native_reference",
                "formal_work_census_eligible": False,
            }
            if isinstance(post_attempts, dict) and isinstance(post_coverage, dict):
                attempt_delta = {
                    batch: post_attempts[batch] - pre_attempts[batch]
                    for batch in pre_attempts
                }
                new_coverage = {
                    batch: post_coverage[batch] & ~pre_coverage[batch]
                    for batch in pre_coverage
                }
                remaining = {
                    batch: (
                        _FIXED32_COMMITTER_ACCEPTED_LENGTH_FULL_MASK
                        & ~post_coverage[batch]
                    )
                    for batch in post_coverage
                }
                shadow_replays = sum(attempt_delta.values())
                coverage.update(
                    {
                        "post_attempts_by_batch": dict(post_attempts),
                        "post_coverage_mask_by_batch": dict(post_coverage),
                        "attempt_delta_by_batch": attempt_delta,
                        "new_coverage_mask_by_batch": new_coverage,
                        "newly_covered_lengths_by_batch": {
                            batch: self._covered_lengths(mask)
                            for batch, mask in new_coverage.items()
                        },
                        "remaining_coverage_mask_by_batch": remaining,
                        "coverage_complete": not any(remaining.values()),
                        "shadow_reference_replays": shadow_replays,
                        "shadow_candidate_replays": shadow_replays,
                        "new_depth_reference_served_replays": shadow_replays,
                    }
                )
        return {
            "run_classification": _FIXED32_CFWD_QUALIFICATION_CLASSIFICATION,
            "acceptance_valid": False,
            "performance_measurement": False,
            "timing_eligible": False,
            "gate_eligible": False,
            "floor_acceptance_eligible": False,
            "process_local_qualification_only": True,
            "durable_production_pass": False,
            "timing_requires_same_server_process": True,
            "same_process_timing_handoff_implemented": False,
            "qualification_coverage": coverage,
        }


def _fixed32_cfwd_b4_qualification_transition(
    *,
    pre_attempts: dict[str, int],
    pre_coverage: dict[str, int],
    post_attempts: dict[str, int],
    post_coverage: dict[str, int],
) -> dict[str, dict[str, int]]:
    """Validate global B1..B4 coverage without assigning it to one task."""
    expected = {str(batch) for batch in range(1, 5)}
    maps = (pre_attempts, pre_coverage, post_attempts, post_coverage)
    if any(
        not isinstance(mapping, dict)
        or set(mapping) != expected
        or any(type(value) is not int or value < 0 for value in mapping.values())
        for mapping in maps
    ):
        raise Fixed32BoundaryError(
            "fixed32 CFWD B4 qualification gate maps changed shape"
        )
    attempt_delta: dict[str, int] = {}
    new_coverage: dict[str, int] = {}
    for batch in sorted(expected, key=int):
        delta = post_attempts[batch] - pre_attempts[batch]
        new_mask = post_coverage[batch] & ~pre_coverage[batch]
        new_bits = new_mask.bit_count()
        batch_size = int(batch)
        if (
            delta < 0
            or post_coverage[batch] > _FIXED32_COMMITTER_ACCEPTED_LENGTH_FULL_MASK
            or post_coverage[batch] & pre_coverage[batch]
            != pre_coverage[batch]
            or (delta == 0) != (new_bits == 0)
            or delta > new_bits
            or new_bits > batch_size * delta
        ):
            raise Fixed32BoundaryError(
                "fixed32 CFWD B4 qualification coverage did not advance "
                "monotonically"
            )
        attempt_delta[batch] = delta
        new_coverage[batch] = new_mask
    return {
        "attempt_delta_by_batch": attempt_delta,
        "new_coverage_mask_by_batch": new_coverage,
    }


class _Fixed32CfwdB4QualificationMemberTaskBracket(_Fixed32TaskBracket):
    """Bracket one member while qualification ownership remains campaign-wide."""

    allow_incomplete_layer_batch_coverage = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.server_capacity != 4:
            raise Fixed32BoundaryError(
                "fixed32 CFWD B4 qualification member requires capacity 4"
            )
        if self.taw_real_task_arm is not None:
            raise Fixed32BoundaryError(
                "fixed32 CFWD B4 qualification forbids per-task arms"
            )

    def _validate_layer_batch_gate_transition(
        self,
        *,
        post_attempts: dict[str, int],
        post_coverage: dict[str, int],
    ) -> None:
        pre_attempts = self.pre_layer_batch_gate_attempts_by_batch
        pre_coverage = self.pre_layer_batch_gate_coverage_mask_by_batch
        if not isinstance(pre_attempts, dict) or not isinstance(pre_coverage, dict):
            raise Fixed32BoundaryError(
                "fixed32 CFWD B4 member lacks pre-campaign gate state"
            )
        _fixed32_cfwd_b4_qualification_transition(
            pre_attempts=pre_attempts,
            pre_coverage=pre_coverage,
            post_attempts=post_attempts,
            post_coverage=post_coverage,
        )

    def _artifact_classification(self) -> dict[str, Any]:
        return {
            "run_classification": (
                _FIXED32_CFWD_B4_QUALIFICATION_CLASSIFICATION + "_member"
            ),
            "acceptance_valid": False,
            "performance_measurement": False,
            "timing_eligible": False,
            "gate_eligible": False,
            "floor_acceptance_eligible": False,
            "qualification_scope": "canonical_exact4_or_16_campaign",
            "per_task_qualification_claim": False,
            "process_local_qualification_only": True,
            "durable_production_pass": False,
        }


class _Fixed32CfwdSameServerTimingHandoff:
    """An in-memory-only proof required immediately before CFWD timing."""

    schema = "fr13-fixed32-cfwd-same-server-timing-handoff-v1"

    def __init__(
        self,
        *,
        client: Any,
        campaign_arm: _Fixed32CommitterLayerBatchCampaignArm,
        qualification_post_ack: Any,
        post_attempts: dict[str, int],
        post_coverage: dict[str, int],
        boundary_snapshot_base: Path,
        server_capacity: int,
    ) -> None:
        full = {
            str(batch): _FIXED32_COMMITTER_ACCEPTED_LENGTH_FULL_MASK
            for batch in range(1, 5)
        }
        if campaign_arm.state != "ended" or campaign_arm.path.exists():
            raise Fixed32BoundaryError(
                "fixed32 CFWD timing handoff requires a removed campaign arm"
            )
        if post_coverage != full:
            raise Fixed32BoundaryError(
                "fixed32 CFWD timing handoff requires complete B1..B4 coverage"
            )
        if (
            qualification_post_ack.mode != client.mode
            or qualification_post_ack.producer_pid != client.producer_pid
        ):
            raise Fixed32BoundaryError(
                "fixed32 CFWD timing handoff server binding is inconsistent"
            )
        self._mode = client.mode
        self._producer_pid = client.producer_pid
        self._container = client.container
        self._campaign_arm = campaign_arm
        self._post_attempts = dict(post_attempts)
        self._post_coverage = dict(post_coverage)
        self._boundary_snapshot_base = boundary_snapshot_base
        self._server_capacity = server_capacity
        self.qualification_post_generation = qualification_post_ack.generation
        self.server_process_identity_sha256 = hashlib.sha256(
            json.dumps(
                [self._container, self._mode, self._producer_pid],
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
        self.state = "qualification_complete_process_local"
        self.timing_pre_generation: int | None = None

    def validate_timing_pre(
        self,
        *,
        client: Any,
    ) -> dict[str, Any]:
        if self.state != "qualification_complete_process_local":
            raise Fixed32BoundaryError(
                "fixed32 CFWD timing handoff can be consumed only once"
            )
        if (
            client.container != self._container
            or client.mode != self._mode
            or client.producer_pid != self._producer_pid
        ):
            raise Fixed32BoundaryError(
                "fixed32 CFWD timing handoff did not retain the same server process"
            )
        if self._campaign_arm.state != "ended" or self._campaign_arm.path.exists():
            raise Fixed32BoundaryError(
                "fixed32 CFWD timing handoff campaign arm became live"
            )
        ack = client.snapshot()
        snapshot, _, _ = _load_fixed32_boundary_snapshot(
            base_path=self._boundary_snapshot_base,
            ack=ack,
            server_capacity=self._server_capacity,
        )
        if (
            ack.mode != self._mode
            or ack.producer_pid != self._producer_pid
            or ack.generation <= self.qualification_post_generation
        ):
            raise Fixed32BoundaryError(
                "fixed32 CFWD timing handoff did not retain the same server process"
            )
        committer = snapshot.get("metrics", {}).get("committer", {})
        attempts = committer.get("layer_batch_gate_attempts_by_batch")
        coverage = committer.get("layer_batch_gate_coverage_mask_by_batch")
        if attempts != self._post_attempts or coverage != self._post_coverage:
            raise Fixed32BoundaryError(
                "fixed32 CFWD timing handoff gate state changed after qualification"
            )
        self.state = "timing_prevalidated_process_local"
        self.timing_pre_generation = ack.generation
        return self.as_dict()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_classification": (
                _FIXED32_CFWD_B4_QUALIFICATION_CLASSIFICATION
            ),
            "state": self.state,
            "server_process_identity_sha256": (
                self.server_process_identity_sha256
            ),
            "qualification_post_generation": self.qualification_post_generation,
            "timing_pre_generation": self.timing_pre_generation,
            "campaign_marker_sha256": self._campaign_arm.marker_sha256,
            "subset_sha256": self._campaign_arm.subset_sha256,
            "task_count": self._campaign_arm.task_count,
            "batch_size": 4,
            "concurrency": 4,
            "accepted_length_full_mask": (
                _FIXED32_COMMITTER_ACCEPTED_LENGTH_FULL_MASK
            ),
            "coverage_mask_by_batch": dict(self._post_coverage),
            "performance_measurement": False,
            "timing_eligible": False,
            "timing_window_authorized": (
                self.state == "timing_prevalidated_process_local"
            ),
            "process_local_qualification_only": True,
            "durable_production_pass": False,
            "same_process_timing_handoff_contract_implemented": True,
            "same_process_timing_execution_implemented": False,
            "artifact_is_replayable_credential": False,
            "next_required_lifecycle": (
                "paired_authenticated_ingress_phases_before_timing"
            ),
        }


class _Fixed32CfwdB4QualificationCampaignBracket:
    """Own the one arm and the global flush bracket for exact4/16 B4."""

    schema = "fr13-fixed32-cfwd-b4-qualification-campaign-v1"

    def __init__(
        self,
        *,
        client: Any,
        boundary_snapshot_base: Path,
        server_capacity: int,
        campaign_arm: _Fixed32CommitterLayerBatchCampaignArm,
        artifact_path: Path,
        arm_artifact_path: Path,
        metrics_pre_path: Path,
        metrics_post_path: Path,
    ) -> None:
        if server_capacity != 4 or campaign_arm.concurrency != 4:
            raise Fixed32BoundaryError(
                "fixed32 CFWD B4 campaign qualification requires exact B4"
            )
        if campaign_arm.task_count not in (4, 16):
            raise Fixed32BoundaryError(
                "fixed32 CFWD B4 campaign qualification requires exact4/16"
            )
        self.client = client
        self.boundary_snapshot_base = boundary_snapshot_base
        self.server_capacity = server_capacity
        self.campaign_arm = campaign_arm
        self.artifact_path = artifact_path
        self.arm_artifact_path = arm_artifact_path
        self.metrics_pre_path = metrics_pre_path
        self.metrics_post_path = metrics_post_path
        self.state = "planned"
        self.pre_ack = None
        self.post_ack = None
        self.pre_snapshot_ref: dict[str, Any] | None = None
        self.post_snapshot_ref: dict[str, Any] | None = None
        self.pre_attempts: dict[str, int] | None = None
        self.pre_coverage: dict[str, int] | None = None
        self.post_attempts: dict[str, int] | None = None
        self.post_coverage: dict[str, int] | None = None
        self.transition: dict[str, dict[str, int]] | None = None
        self.action_succeeded = False
        self.handoff: _Fixed32CfwdSameServerTimingHandoff | None = None

    @staticmethod
    def _snapshot_gate_maps(
        snapshot: dict[str, Any],
    ) -> tuple[dict[str, int], dict[str, int]]:
        committer = snapshot["metrics"]["committer"]
        return (
            dict(committer["layer_batch_gate_attempts_by_batch"]),
            dict(committer["layer_batch_gate_coverage_mask_by_batch"]),
        )

    @staticmethod
    def _snapshot_ref(ack: Any, path: Path, sha256: str) -> dict[str, Any]:
        return {
            "schema": _FIXED32_BOUNDARY_SNAPSHOT_SCHEMA,
            "generation": ack.generation,
            "path": str(path),
            "sha256": sha256,
        }

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        os.replace(temporary, path)

    def _write_arm_artifact(self) -> None:
        self._write_json(self.arm_artifact_path, self.campaign_arm.as_dict())

    def _coverage_artifact(self) -> dict[str, Any] | None:
        if self.pre_attempts is None or self.pre_coverage is None:
            return None
        payload: dict[str, Any] = {
            "accepted_length_full_mask": (
                _FIXED32_COMMITTER_ACCEPTED_LENGTH_FULL_MASK
            ),
            "pre_attempts_by_batch": dict(self.pre_attempts),
            "pre_coverage_mask_by_batch": dict(self.pre_coverage),
            "post_attempts_by_batch": None,
            "post_coverage_mask_by_batch": None,
            "attempt_delta_by_batch": None,
            "new_coverage_mask_by_batch": None,
            "remaining_coverage_mask_by_batch": None,
            "newly_covered_lengths_by_batch": None,
            "coverage_complete": False,
        }
        if (
            self.post_attempts is not None
            and self.post_coverage is not None
            and self.transition is not None
        ):
            remaining = {
                batch: (
                    _FIXED32_COMMITTER_ACCEPTED_LENGTH_FULL_MASK & ~mask
                )
                for batch, mask in self.post_coverage.items()
            }
            new_coverage = self.transition["new_coverage_mask_by_batch"]
            payload.update(
                {
                    "post_attempts_by_batch": dict(self.post_attempts),
                    "post_coverage_mask_by_batch": dict(self.post_coverage),
                    "attempt_delta_by_batch": dict(
                        self.transition["attempt_delta_by_batch"]
                    ),
                    "new_coverage_mask_by_batch": dict(new_coverage),
                    "remaining_coverage_mask_by_batch": remaining,
                    "newly_covered_lengths_by_batch": {
                        batch: _Fixed32CfwdQualificationTaskBracket._covered_lengths(
                            mask
                        )
                        for batch, mask in new_coverage.items()
                    },
                    "coverage_complete": not any(remaining.values()),
                }
            )
        return payload

    def as_dict(self) -> dict[str, Any]:
        interval = None
        if self.pre_ack is not None and self.post_ack is not None:
            interval = {
                "start_forward_step": self.pre_ack.counters[
                    "pure_decode_forward_steps"
                ],
                "end_forward_step": self.post_ack.counters[
                    "pure_decode_forward_steps"
                ],
                "expected_complete_events": (
                    self.post_ack.counters["complete_work_census_events"]
                    - self.pre_ack.counters["complete_work_census_events"]
                ),
            }
        return {
            "schema": self.schema,
            "run_classification": (
                _FIXED32_CFWD_B4_QUALIFICATION_CLASSIFICATION
            ),
            "state": self.state,
            "mode": self.client.mode,
            "task_count": self.campaign_arm.task_count,
            "task_ids": list(self.campaign_arm.task_ids),
            "subset_sha256": self.campaign_arm.subset_sha256,
            "batch_size": 4,
            "concurrency": 4,
            "pre": self.pre_ack.as_dict() if self.pre_ack is not None else None,
            "post": self.post_ack.as_dict() if self.post_ack is not None else None,
            "pre_runtime_snapshot": self.pre_snapshot_ref,
            "post_runtime_snapshot": self.post_snapshot_ref,
            "forward_step_interval": interval,
            "campaign_arm": self.campaign_arm.as_dict(),
            "qualification_coverage": self._coverage_artifact(),
            "action_succeeded": self.action_succeeded,
            "acceptance_valid": False,
            "performance_measurement": False,
            "timing_eligible": False,
            "gate_eligible": False,
            "floor_acceptance_eligible": False,
            "process_local_qualification_only": True,
            "durable_production_pass": False,
            "same_process_timing_handoff_contract_implemented": True,
            "same_process_timing_execution_implemented": False,
            "timing_handoff": (
                self.handoff.as_dict() if self.handoff is not None else None
            ),
        }

    def _write_artifact(self) -> None:
        self._write_json(self.artifact_path, self.as_dict())

    def pre(self) -> None:
        if self.state != "planned":
            raise Fixed32BoundaryError(
                "fixed32 CFWD B4 campaign pre bracket was invoked twice"
            )
        try:
            ack = self.client.snapshot()
            snapshot, snapshot_path, snapshot_sha = _load_fixed32_boundary_snapshot(
                base_path=self.boundary_snapshot_base,
                ack=ack,
                server_capacity=self.server_capacity,
                allow_incomplete_layer_batch_coverage=True,
            )
            attempts, coverage = self._snapshot_gate_maps(snapshot)
            metrics = _fixed32_metrics_snapshot(
                metrics_url=DEFAULT_METRICS_URL,
                snapshot=snapshot,
            )
            self.metrics_pre_path.write_text(metrics, encoding="utf-8")
            self.pre_ack = ack
            self.pre_snapshot_ref = self._snapshot_ref(
                ack, snapshot_path, snapshot_sha
            )
            self.pre_attempts = attempts
            self.pre_coverage = coverage
            self.campaign_arm.start()
            self.state = "active"
            self._write_arm_artifact()
            self._write_artifact()
        except Exception as error:
            cleanup_error = None
            if self.campaign_arm.active:
                try:
                    self.campaign_arm.finish()
                    self._write_arm_artifact()
                except Exception as cleanup:  # noqa: BLE001
                    cleanup_error = cleanup
            self.state = "pre_failed"
            try:
                self._write_artifact()
            except Exception:
                pass
            detail = f"{type(error).__name__}: {error}"
            if cleanup_error is not None:
                detail += (
                    "; arm cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise Fixed32BoundaryError(
                "fixed32 CFWD B4 campaign pre bracket failed: " + detail
            ) from error

    def post(self, *, action_succeeded: bool) -> dict[str, Any]:
        if self.state != "active" or self.pre_ack is None:
            raise Fixed32BoundaryError(
                "fixed32 CFWD B4 campaign post bracket has no active pre bracket"
            )
        self.action_succeeded = action_succeeded
        try:
            self.campaign_arm.finish()
            self._write_arm_artifact()
        except Exception as error:
            self.state = "teardown_failed"
            try:
                self._write_artifact()
            except Exception:
                pass
            raise Fixed32BoundaryError(
                "fixed32 CFWD B4 campaign arm teardown failed: "
                f"{type(error).__name__}: {error}"
            ) from error
        try:
            ack = self.client.snapshot()
            counters = _validate_fixed32_ack(ack, label="CFWD B4 campaign post")
            start = self.pre_ack.counters["pure_decode_forward_steps"]
            end = counters["pure_decode_forward_steps"]
            event_delta = (
                counters["complete_work_census_events"]
                - self.pre_ack.counters["complete_work_census_events"]
            )
            if end <= start or event_delta != end - start:
                raise Fixed32BoundaryError(
                    "fixed32 CFWD B4 campaign interval lacks complete decode events"
                )
            snapshot, snapshot_path, snapshot_sha = _load_fixed32_boundary_snapshot(
                base_path=self.boundary_snapshot_base,
                ack=ack,
                server_capacity=self.server_capacity,
                allow_incomplete_layer_batch_coverage=True,
            )
            attempts, coverage = self._snapshot_gate_maps(snapshot)
            if self.pre_attempts is None or self.pre_coverage is None:
                raise Fixed32BoundaryError(
                    "fixed32 CFWD B4 campaign lost its pre gate state"
                )
            transition = _fixed32_cfwd_b4_qualification_transition(
                pre_attempts=self.pre_attempts,
                pre_coverage=self.pre_coverage,
                post_attempts=attempts,
                post_coverage=coverage,
            )
            metrics = _fixed32_metrics_snapshot(
                metrics_url=DEFAULT_METRICS_URL,
                snapshot=snapshot,
            )
            self.metrics_post_path.write_text(metrics, encoding="utf-8")
            self.post_ack = ack
            self.post_snapshot_ref = self._snapshot_ref(
                ack, snapshot_path, snapshot_sha
            )
            self.post_attempts = attempts
            self.post_coverage = coverage
            self.transition = transition
            complete = all(
                mask == _FIXED32_COMMITTER_ACCEPTED_LENGTH_FULL_MASK
                for mask in coverage.values()
            )
            if action_succeeded and complete:
                self.handoff = _Fixed32CfwdSameServerTimingHandoff(
                    client=self.client,
                    campaign_arm=self.campaign_arm,
                    qualification_post_ack=ack,
                    post_attempts=attempts,
                    post_coverage=coverage,
                    boundary_snapshot_base=self.boundary_snapshot_base,
                    server_capacity=self.server_capacity,
                )
                self.state = "qualified_process_local"
            elif action_succeeded:
                self.state = "coverage_incomplete"
            else:
                self.state = "action_failed"
            self._write_artifact()
        except Exception as error:
            self.state = "post_failed"
            try:
                self._write_artifact()
            except Exception:
                pass
            raise Fixed32BoundaryError(
                "fixed32 CFWD B4 campaign post bracket failed: "
                f"{type(error).__name__}: {error}"
            ) from error
        return self.as_dict()

    def run(self, action: Any) -> Any:
        self.pre()
        try:
            result = action()
        except BaseException as action_error:
            try:
                self.post(action_succeeded=False)
            except BaseException as cleanup_error:
                raise Fixed32BoundaryError(
                    "fixed32 CFWD B4 campaign action and teardown both failed: "
                    f"action={type(action_error).__name__}: {action_error}; "
                    f"teardown={type(cleanup_error).__name__}: {cleanup_error}"
                ) from cleanup_error
            raise
        self.post(action_succeeded=True)
        return result


class _Fixed32EagerKernelDiagnosticTaskBracket(_Fixed32TaskBracket):
    """Authenticate a real task without claiming graph-census evidence."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.pre_metrics_ref: dict[str, Any] | None = None
        self.post_metrics_ref: dict[str, Any] | None = None

    @property
    def started(self) -> bool:
        return self.pre_metrics_ref is not None

    @property
    def complete(self) -> bool:
        return self.post_metrics_ref is not None

    def _write_artifact(self) -> None:
        payload = {
            "schema": "fr13-fixed32-eager-kernel-diagnostic-task-bracket-v1",
            "instance_id": self.instance_id,
            "mode": self.client.mode,
            "producer_pid": self.client.producer_pid,
            "run_classification": "eager_kernel_byte_diagnostic",
            "acceptance_valid": False,
            "flush_protocol_used": False,
            "pre_metrics": self.pre_metrics_ref,
            "post_metrics": self.post_metrics_ref,
        }
        temporary = self.artifact_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.artifact_path)

    @staticmethod
    def _metrics_ref(path: Path, raw: bytes) -> dict[str, Any]:
        return {
            "path": str(path),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    def pre(self, metrics_path: Path) -> None:
        if self.started or self.post_attempted:
            raise Fixed32BoundaryError(
                "fixed32 eager diagnostic task pre bracket was invoked twice"
            )
        try:
            metrics = _metrics_snapshot(DEFAULT_METRICS_URL).encode("utf-8")
            metrics_path.write_bytes(metrics)
            self.pre_metrics_ref = self._metrics_ref(metrics_path, metrics)
            if self.taw_real_task_arm is not None:
                self.taw_real_task_arm.start()
                self._write_taw_arm_artifact()
            self._write_artifact()
        except Exception as exc:
            cleanup_error = None
            if (
                self.taw_real_task_arm is not None
                and self.taw_real_task_arm.active
            ):
                try:
                    self.taw_real_task_arm.finish()
                    self._write_taw_arm_artifact()
                except Exception as error:  # noqa: BLE001
                    cleanup_error = error
            self.pre_metrics_ref = None
            detail = f"{type(exc).__name__}: {exc}"
            if cleanup_error is not None:
                detail += (
                    "; arm cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise Fixed32BoundaryError(
                f"fixed32 eager diagnostic pre bracket failed: {detail}"
            ) from exc

    def post(self, metrics_path: Path) -> dict[str, Any]:
        if not self.started:
            raise Fixed32BoundaryError(
                "fixed32 eager diagnostic post bracket has no pre bracket"
            )
        if self.post_attempted:
            if self.complete:
                return json.loads(self.artifact_path.read_text(encoding="utf-8"))
            raise Fixed32BoundaryError(
                "fixed32 eager diagnostic post bracket already failed"
            )
        self.post_attempted = True
        metrics_error = None
        try:
            metrics = _metrics_snapshot(DEFAULT_METRICS_URL).encode("utf-8")
            metrics_path.write_bytes(metrics)
            self.post_metrics_ref = self._metrics_ref(metrics_path, metrics)
        except Exception as exc:
            metrics_error = exc
        arm_error = None
        if self.taw_real_task_arm is not None:
            try:
                self.taw_real_task_arm.finish()
                self._write_taw_arm_artifact()
            except Exception as exc:  # noqa: BLE001
                arm_error = exc
        if metrics_error is not None or arm_error is not None:
            self.post_metrics_ref = None
            try:
                self._write_artifact()
            except Exception:
                pass
            details = []
            if metrics_error is not None:
                details.append(
                    f"metrics={type(metrics_error).__name__}: {metrics_error}"
                )
            if arm_error is not None:
                details.append(f"arm={type(arm_error).__name__}: {arm_error}")
            cause = metrics_error if metrics_error is not None else arm_error
            raise Fixed32BoundaryError(
                "fixed32 eager diagnostic post bracket failed: "
                + "; ".join(details)
            ) from cause
        try:
            self._write_artifact()
        except Exception as exc:
            self.post_metrics_ref = None
            try:
                self._write_artifact()
            except Exception:
                pass
            raise Fixed32BoundaryError(
                "fixed32 eager diagnostic post bracket artifact failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return json.loads(self.artifact_path.read_text(encoding="utf-8"))


def _spawn_dcgm_sampler(out_path: Path, interval_s: float = DEFAULT_DCGM_INTERVAL_S
                       ) -> subprocess.Popen[bytes] | None:
    """Spawn the same DCGM/NVML sampler Track B uses, writing JSONL to out_path."""
    if not DEFAULT_DCGM_SAMPLER.is_file():
        return None
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    interp = str(venv_python) if venv_python.is_file() else sys.executable
    cmd = [
        interp, str(DEFAULT_DCGM_SAMPLER),
        "--out", str(out_path),
        "--interval-s", str(interval_s),
        "--allow-unstamped-smoke",
    ]
    try:
        return subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:  # noqa: BLE001
        return None


def _stop_dcgm_sampler(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None:
        return
    try:
        proc.send_signal(2)  # SIGINT — sampler writes a clean tail on signal
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def _load_subset(subset_json: Path) -> tuple[str, list[str]]:
    payload = json.loads(subset_json.read_text())
    dataset_name = payload["dataset_name"]
    instance_ids = list(payload["instance_ids"])
    return dataset_name, instance_ids


def _load_dataset(
    dataset_name: str,
    *,
    pinned_verified: bool = False,
) -> dict[str, dict]:
    os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
    if pinned_verified:
        if dataset_name != "princeton-nlp/SWE-bench_Verified":
            raise RuntimeError(
                f"fixed32 dataset is not the pinned SWE-Verified source: "
                f"{dataset_name!r}"
            )
        if not PINNED_SWE_VERIFIED_PARQUET.is_file():
            raise RuntimeError(
                f"pinned SWE-Verified Parquet is missing: "
                f"{PINNED_SWE_VERIFIED_PARQUET}"
            )
        import pyarrow.parquet as pq

        rows = pq.read_table(PINNED_SWE_VERIFIED_PARQUET).to_pylist()
        return {row["instance_id"]: dict(row) for row in rows}
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="test")
    out: dict[str, dict] = {}
    for ex in ds:
        out[ex["instance_id"]] = dict(ex)
    return out


def _repo_clone_url(repo: str) -> str:
    return f"https://github.com/{repo}.git"


def _ensure_repo_cache(repo: str, cache_root: Path) -> Path:
    safe = repo.replace("/", "__")
    cache_path = cache_root / safe
    if not cache_path.is_dir():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", _repo_clone_url(repo), str(cache_path)],
            check=True,
        )
    return cache_path


def _fetch_commit(cache_path: Path, base_commit: str) -> None:
    rc = subprocess.run(
        ["git", "-C", str(cache_path), "cat-file", "-e", base_commit],
    ).returncode
    if rc != 0:
        subprocess.run(
            ["git", "-C", str(cache_path), "fetch", "origin", base_commit],
            check=False,
        )


def _hydrate_workspace(
    *,
    cache_path: Path,
    base_commit: str,
    workspace_path: Path,
) -> None:
    if _swe_agent_env() == "instance_image":
        # No source worktree: /testbed lives inside the per-instance eval image.
        # The workspace dir is only a host-side staging area (AGENTS.md + the
        # /out mount that receives the extracted patch).
        if workspace_path.exists():
            shutil.rmtree(workspace_path)
        workspace_path.mkdir(parents=True, exist_ok=True)
        return
    if workspace_path.exists():
        shutil.rmtree(workspace_path)
    workspace_path.parent.mkdir(parents=True, exist_ok=True)
    # Absolutize: `git -C <cache> worktree add <relpath>` resolves the
    # destination against <cache>, not against the script CWD.
    abs_workspace = workspace_path.resolve()
    subprocess.run(
        ["git", "-C", str(cache_path), "worktree", "add", "--detach",
         str(abs_workspace), base_commit],
        check=True,
    )


def _remove_workspace(cache_path: Path, workspace_path: Path) -> None:
    if _swe_agent_env() == "instance_image":
        # instance_image mode never registered a git worktree; just drop the
        # host staging dir (cache_path is None in this mode).
        if workspace_path.exists():
            shutil.rmtree(workspace_path, ignore_errors=True)
        return
    abs_workspace = workspace_path.resolve() if workspace_path.exists() else workspace_path
    if not abs_workspace.exists():
        return
    subprocess.run(
        ["git", "-C", str(cache_path), "worktree", "remove", "--force",
         str(abs_workspace)],
        check=False,
    )
    if abs_workspace.exists():
        shutil.rmtree(abs_workspace, ignore_errors=True)


def _write_agents_md(workspace: Path, instance: dict) -> None:
    body = []
    body.append(f"# SWE-Bench task: {instance['instance_id']}")
    body.append("")
    body.append(f"**Repo:** `{instance['repo']}`  ")
    body.append(f"**Base commit:** `{instance['base_commit']}`  ")
    if instance.get("version"):
        body.append(f"**Version:** `{instance['version']}`  ")
    body.append("")
    body.append("## Problem statement")
    body.append("")
    body.append(instance.get("problem_statement") or "(empty problem statement)")
    body.append("")
    body.append("## Required behavior")
    body.append("")
    body.append(
        "Implement the fix described in the problem statement by editing the "
        "source files in this workspace. Do NOT modify any test files. The "
        "hidden grader will apply its own test patch and run the test suite; "
        "your code must make those tests pass without breaking existing ones."
    )
    body.append("")
    # Bundle B #7: model_reasoning_effort="high" is inert on Qwen 3.6 in this
    # stack (Harmony-path only), so steer reasoning depth via the operator
    # prompt instead. Also pre-empts the common failure modes: don't burn the
    # budget on env setup (the grader builds its own env), and always emit a
    # real source edit before finishing.
    body.append("## How to work (important)")
    body.append("")
    if _swe_agent_env() == "instance_image":
        # instance_image mode: the agent runs INSIDE the prepared testbed env
        # (conda 'testbed' with the project editable-installed, on PATH), so the
        # "do NOT install/build" band-aid is replaced with a "reproduce + verify"
        # instruction — the whole point of §58 is that self-verification is now
        # possible.
        body.append(
            "- Reason carefully and thoroughly before each tool call. First inspect "
            "the relevant source files to confirm your understanding of the bug, "
            "then make the minimal correct edit.\n"
            "- You have a WORKING prepared environment: the project is already "
            "installed and importable (run `python -c \"import <project>\"` to "
            "confirm). REPRODUCE the bug with a short `python`/`pytest` command, "
            "make your fix, then RE-RUN it to verify the fix before finishing. You "
            "do NOT need to (re)install or build the project.\n"
            "- You MUST finish by leaving an actual code change in the working tree. "
            "Do not stop until you have edited the source files to implement the fix."
        )
    else:
        body.append(
            "- Reason carefully and thoroughly before each tool call. First inspect "
            "the relevant source files to confirm your understanding of the bug, "
            "then make the minimal correct edit.\n"
            "- Do NOT spend your time trying to `pip install` or build/conda the "
            "project — the grader runs in its own prepared environment. If an "
            "install/build command fails, do not retry it; just edit the source.\n"
            "- You MUST finish by leaving an actual code change in the working tree. "
            "Do not stop until you have edited the source files to implement the fix."
        )
    body.append("")
    (workspace / "AGENTS.md").write_text("\n".join(body) + "\n", encoding="utf-8")


def _classify_empty_patch_cause(trace_path: Path) -> str:
    """Inspect a codex trace to decide why an empty-patch run produced no edit.

    Returns 'setup_loop' if the agent repeatedly ran the same failing command
    (>=3 identical failing command_execution items — the pip/conda/build loop),
    else 'agent_gave_up'. Drives the Bundle B #2/#3/#8 state-conditional retry.
    """
    try:
        from collections import Counter
        failing: Counter = Counter()
        for line in trace_path.read_text(errors="replace").splitlines():
            if '"type":"command_execution"' not in line:
                continue
            try:
                item = json.loads(line).get("item", {})
            except Exception:  # noqa: BLE001
                continue
            if item.get("type") != "command_execution":
                continue
            ec = item.get("exit_code")
            cmd = item.get("command")
            if isinstance(cmd, str) and ec not in (0, None):
                failing[cmd.strip()] += 1
        if failing and max(failing.values()) >= 3:
            return "setup_loop"
    except Exception:  # noqa: BLE001
        pass
    return "agent_gave_up"


def _extract_patch(cache_path: Path, workspace: Path, base_commit: str) -> str:
    if _swe_agent_env() == "instance_image":
        # The patch was produced by `git -C /testbed diff --no-color --binary
        # <base_commit>` INSIDE the eval image (same flags as below) and copied to
        # <workspace>/patch.diff by the agent run. Read it back here.
        pf = Path(workspace) / "patch.diff"
        return pf.read_text(encoding="utf-8", errors="replace") if pf.is_file() else ""
    # Stage tracked-file diffs and untracked file additions against base_commit.
    proc = subprocess.run(
        ["git", "-C", str(workspace), "diff", "--no-color", "--binary", base_commit],
        capture_output=True, text=True, check=False,
    )
    return proc.stdout


def _run_agent_local(
    *,
    workspace: Path,
    endpoint: str,
    model: str,
    timeout_s: int,
    instance_id: str,
    stdout_path: Path,
    stderr_path: Path,
    trace_path: Path,
    base_commit: str | None = None,
    prompt: str = DEFAULT_AGENT_PROMPT,
    task_bearer: str | None = None,
) -> dict[str, Any]:
    """Run codex-runner:v1 and capture trace/stdout/stderr to separate files.

    Track B layout (matching launch_qwen36_ablation_point.py):
      - trace_path : the stream-json NDJSON event stream from the qwen-code agent
      - stdout_path: qwen_stdout.log (kept for parity; usually empty when
        stream-json is the agent output mode)
      - stderr_path: qwen_stderr.log (qwen-code CLI stderr noise)
    """
    if _swe_agent_env() == "instance_image":
        return _run_agent_instance(
            remote_host=None, workspace=workspace, endpoint=endpoint, model=model,
            timeout_s=timeout_s, instance_id=instance_id, base_commit=base_commit,
            stdout_path=stdout_path, stderr_path=stderr_path, trace_path=trace_path,
            prompt=prompt, task_bearer=task_bearer,
        )
    container_name = f"swe-agent-{instance_id.replace('/', '_')[:48]}-{int(time.time())}"
    cmd = _agent_template().format(
        container_name=container_name,
        workspace=str(workspace),
        endpoint=endpoint,
        model=model,
        session_id=fixed32_contract.fixed32_trace_session_id(instance_id),
    )
    # Pass the prompt as a separate argv element instead of shell-embedding it.
    # The dynamic agent_gave_up retry prompt is built from the prior-attempt trace
    # (arbitrary code/quotes/braces); inlining it as "\"{prompt}\"" + shlex.split
    # raised ValueError("No closing quotation") on an unbalanced quote and crashed
    # the orchestrator (verdict=orchestrator_crash -> NORESULT). A list argv passes
    # the prompt verbatim with no shell parsing, robust to any content.
    cmd_argv = shlex.split(cmd)
    cmd_argv.append(prompt)
    started = time.monotonic()
    rc: int | None = None
    timed_out = False
    # Codex emits its --json event stream to stdout and any human-readable
    # messages / errors to stderr. Track B keeps them in separate files.
    with trace_path.open("w", encoding="utf-8") as trace_f, \
         stderr_path.open("w", encoding="utf-8") as stderr_f:
        try:
            completed = subprocess.run(
                cmd_argv,
                stdout=trace_f,
                stderr=stderr_f,
                # timeout_s<=0 => NO harness wall (codex runs to its own idle/turn limit).
                timeout=(None if timeout_s <= 0 else max(timeout_s, 30)),
                check=False,
                env=_agent_subprocess_env(task_bearer),
            )
            rc = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            subprocess.run(
                ["docker", "kill", container_name], check=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            try:
                subprocess.run(
                    ["docker", "wait", container_name], check=False,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=30,
                )
            except subprocess.TimeoutExpired:
                pass
            rc = -1
    elapsed = time.monotonic() - started
    # stdout_path kept for layout parity with Track B; the agent doesn't
    # write to it under `codex exec --json`.
    if not stdout_path.is_file():
        stdout_path.write_text("", encoding="utf-8")
    return {
        "elapsed_s": round(elapsed, 3),
        "exit_code": rc if rc is not None else -1,
        "timed_out": timed_out,
        "container_name": container_name,
    }


# Eval-offload config. When EVAL_HOST is set (via --eval-host), the eval
# step runs on a native x86_64 box over SSH instead of locally on aarch64
# (recovers old-python instances that can't build the conda env on arm64,
# and keeps the whole dataset single-arch). See scripts/swe_eval_offload.py.
EVAL_HOST: str | None = None
_EVAL_SSH_OPTS = [
    "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
    "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
    "-o", "StrictHostKeyChecking=accept-new",
]

# --- FR13 CODEX-OFFLOAD config (the unified-memory contamination fix) ---------
# When AGENT_HOST is set (via --codex-host), the codex-runner Docker runs on a
# native x86 box (alienware) instead of locally on the GB10. The workspace is
# rsynced there before the run and the (modified) workspace is rsynced back
# after, so patch extraction + eval stay on the GB10 unchanged. This leaves the
# GB10 running ONLY vLLM during the codex agent loop, so the unified-memory
# bandwidth (273 GB/s, shared Grace CPU + Blackwell GPU) is uncontended and the
# timing-sensitive deploy-speed numbers (s/fwd) are clean. The lossless numbers
# are timing-independent so unaffected; only WHERE the codex compute runs moves.
# The codex docker on alienware hits the alienware-LOCAL proxy (AGENT_ENDPOINT,
# default 127.0.0.1:8023 — 8022 is occupied on alienware by an unrelated host
# service). Reuses the eval-offload SSH/rsync plumbing (_net_retry below).
AGENT_HOST: str | None = None
# alienware-local proxy the offloaded codex docker hits (8022 is taken on
# alienware by an unrelated host service; the offload proxy listens on 8023).
AGENT_ENDPOINT: str | None = None
_AGENT_REMOTE_ROOT = "~/lumo_proxy_offload/codex_work"
_AGENT_NET_LOG = DEFAULT_OUT_ROOT / "swe_codex_offload_network.log"
_REMOTE_BASE = "~/swe_eval_offload"
_REMOTE_WORKER = "~/swe_eval_offload/swe_eval_x86_worker.py"
_REMOTE_VENV_PY = "~/swe_eval_offload/venv/bin/python"
_REMOTE_HF_HOME = "~/.cache/huggingface"
_EVAL_NET_LOG = DEFAULT_OUT_ROOT / "swe_eval_offload_network.log"


def _eval_net_log(msg: str) -> None:
    try:
        _EVAL_NET_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _EVAL_NET_LOG.open("a", encoding="utf-8") as f:
            f.write(f"[{_iso_now()}] {msg}\n")
    except Exception:  # noqa: BLE001
        pass


def _net_retry(argv: list[str], *, what: str, timeout: int,
               max_attempts: int = 5, ok_rcs: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess:
    backoff = 5
    last: subprocess.CompletedProcess | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            cp = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
            if cp.returncode in ok_rcs:
                if attempt > 1:
                    _eval_net_log(f"RECOVERED {what} on attempt {attempt}")
                return cp
            last = cp
            if cp.returncode == 255:
                _eval_net_log(f"FAIL {what} attempt={attempt}/{max_attempts} rc=255(transport)")
            else:
                _eval_net_log(f"NONRETRY {what} rc={cp.returncode} (remote ran)")
                return cp
        except subprocess.TimeoutExpired:
            _eval_net_log(f"TIMEOUT {what} attempt={attempt}/{max_attempts} after {timeout}s")
            last = subprocess.CompletedProcess(argv, 124, "", "timeout")
        if attempt < max_attempts:
            time.sleep(backoff)
            backoff = min(backoff * 3, 300)
    _eval_net_log(f"GIVEUP {what} after {max_attempts} attempts")
    return last if last is not None else subprocess.CompletedProcess(argv, 1, "", "unknown")


def _run_eval_remote(
    *, host: str, instance_id: str, patch_path: Path, output_dir: Path,
    dataset_name: str, model_name: str, timeout_s: int, eval_log_path: Path,
) -> dict[str, Any]:
    """Offload one eval to a native x86_64 host over SSH (network-tolerant)."""
    started = time.monotonic()
    remote_dir = f"{_REMOTE_BASE}/work/{instance_id}"
    mk = _net_retry(["ssh", *_EVAL_SSH_OPTS, host, f"mkdir -p {remote_dir} && echo ok"],
                    what=f"mkdir:{instance_id}", timeout=30)
    if mk.returncode != 0:
        eval_log_path.write_text(f"remote mkdir failed rc={mk.returncode}\n{mk.stderr}", encoding="utf-8")
        return {"exit_code": -1, "elapsed_s": round(time.monotonic() - started, 3), "offloaded": True}
    up = _net_retry(["scp", *_EVAL_SSH_OPTS, str(patch_path), f"{host}:{remote_dir}/patch.diff"],
                    what=f"scp_up:{instance_id}", timeout=120)
    if up.returncode != 0:
        eval_log_path.write_text(f"scp patch up failed rc={up.returncode}\n{up.stderr}", encoding="utf-8")
        return {"exit_code": -1, "elapsed_s": round(time.monotonic() - started, 3), "offloaded": True}
    remote_cmd = (
        f"cd {_REMOTE_BASE} && HF_HOME={_REMOTE_HF_HOME} {_REMOTE_VENV_PY} {_REMOTE_WORKER} "
        f"--instance-id {instance_id} --patch-path {remote_dir}/patch.diff "
        f"--output-dir {remote_dir}/out --dataset-name '{dataset_name}' "
        f"--model-name '{model_name}' --timeout-s {timeout_s} --cache-level env"
    )
    ev = _net_retry(["ssh", *_EVAL_SSH_OPTS, host, remote_cmd],
                    what=f"eval:{instance_id}", timeout=timeout_s + 900,
                    max_attempts=3, ok_rcs=(0, 1, 2))
    output_dir.mkdir(parents=True, exist_ok=True)
    with eval_log_path.open("w", encoding="utf-8") as f:
        f.write(f"[offload host={host} worker_rc={ev.returncode}]\n")
        f.write(ev.stdout or "")
        f.write("\n-- stderr --\n")
        f.write(ev.stderr or "")
    # fetch artifacts
    for fname, attempts in (("eval_report.json", 4), ("normalized_eval.json", 3),
                            ("eval.log", 3), ("predictions.jsonl", 2)):
        _net_retry(["scp", *_EVAL_SSH_OPTS, f"{host}:{remote_dir}/out/{fname}", str(output_dir / fname)],
                   what=f"scp_{fname}:{instance_id}", timeout=120, max_attempts=attempts)
    _net_retry(["ssh", *_EVAL_SSH_OPTS, host, f"rm -rf {remote_dir}"],
               what=f"cleanup:{instance_id}", timeout=30, max_attempts=2)
    return {"exit_code": ev.returncode, "elapsed_s": round(time.monotonic() - started, 3),
            "offloaded": True, "eval_host": host}


# ---- FR13 CODEX-OFFLOAD: run the codex-runner Docker on alienware -----------
# rsync robustness (req #3): a dropped workspace sync must NOT lose the run.
# --partial keeps a half-transferred file for the next attempt to resume;
# --append-verify resumes + re-checksums; the _net_retry loop reconnects.
_RSYNC_RESILIENT = [
    "rsync", "-az", "--partial", "--append-verify", "--timeout=120",
    "-e", "ssh " + " ".join(_EVAL_SSH_OPTS),
]


def _agent_net_log(msg: str) -> None:
    try:
        _AGENT_NET_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _AGENT_NET_LOG.open("a", encoding="utf-8") as f:
            f.write(f"[{_iso_now()}] {msg}\n")
    except Exception:  # noqa: BLE001
        pass


# --- FR13 §59 runner stall-watchdog (default OFF) ----------------------------
# Diagnosis: a per-request GB10 emit/transport wedge froze the offloaded qwen-code
# agent mid-stream. With SWE_AGENT_WALL_S=0 the harness imposes NO wall, so a
# pre-first-byte wedge (Mode B, the client idle guard never arms) hung one banked
# run for 37 min until an external watchdog. This bounds BOTH modes by KILLING the
# ssh (and best-effort the remote container) when the agent's trace stops growing,
# keying on TRACE GROWTH (not wallclock) so a legitimately slow-but-progressing
# task survives. It KILLS + CLASSIFIES (cause=infra_stall_suspect); it NEVER
# retries or discards — that would risk the firm nudge ban.
def _stall_kill_s() -> float:
    """LUMO_SWE_STALL_KILL_S seconds; 0/unset/invalid => 0.0 (watchdog OFF)."""
    raw = os.environ.get("LUMO_SWE_STALL_KILL_S", "").strip()
    if not raw:
        return 0.0
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return val if val > 0 else 0.0


def _kill_proc(proc: Any) -> None:
    try:
        proc.kill()
    except Exception:  # noqa: BLE001 - already-dead proc, etc.
        pass


def _monitor_proc_with_stall_watchdog(
    proc: Any,
    *,
    trace_path: Path,
    wall_timeout_s: float | None,
    stall_kill_s: float,
    poll_s: float = 5.0,
    on_stall_kill: Any = None,
) -> dict[str, Any]:
    """Block until ``proc`` exits, enforcing an optional wall timeout AND (when
    stall_kill_s>0) a trace-growth stall-watchdog.

    Returns metadata: {returncode, timed_out, stall_killed, last_trace_growth_ts}.
    On a stall (no ``trace_path`` byte-growth for stall_kill_s) the local proc is
    killed AFTER invoking on_stall_kill() (best-effort remote cleanup). On a wall
    timeout the local proc is killed (caller handles remote cleanup, mirroring the
    legacy path). NEVER retries. stall_kill_s<=0 disables the stall-watchdog, so
    the OFF behavior matches the legacy blocking subprocess.run(timeout=wall)."""
    start = time.monotonic()
    last_size = -1
    last_growth_mono = start
    last_growth_ts = time.time()
    timed_out = False
    stall_killed = False
    rc: int | None = None
    while True:
        try:
            rc = proc.wait(timeout=poll_s)
            break
        except subprocess.TimeoutExpired:
            pass
        now = time.monotonic()
        try:
            size = trace_path.stat().st_size
        except OSError:
            size = last_size
        if size > last_size:
            last_size = size
            last_growth_mono = now
            last_growth_ts = time.time()
        if wall_timeout_s and wall_timeout_s > 0 and (now - start) >= wall_timeout_s:
            timed_out = True
            _kill_proc(proc)
            try:
                rc = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                rc = -1
            break
        if stall_kill_s and stall_kill_s > 0 and (now - last_growth_mono) >= stall_kill_s:
            stall_killed = True
            if on_stall_kill is not None:
                try:
                    on_stall_kill()
                except Exception:  # noqa: BLE001 - best-effort remote cleanup
                    pass
            _kill_proc(proc)
            try:
                rc = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                rc = -1
            break
    return {
        "returncode": rc if rc is not None else -1,
        "timed_out": timed_out,
        "stall_killed": stall_killed,
        "last_trace_growth_ts": last_growth_ts,
    }


def _run_agent_remote(
    *,
    host: str,
    workspace: Path,
    endpoint: str,
    model: str,
    timeout_s: int,
    instance_id: str,
    stdout_path: Path,
    stderr_path: Path,
    trace_path: Path,
    base_commit: str | None = None,
    prompt: str = DEFAULT_AGENT_PROMPT,
    task_bearer: str | None = None,
) -> dict[str, Any]:
    """Run codex-runner:v1 ON alienware (x86) over SSH, keeping the GB10 vLLM-only.

    Workspace lifecycle: rsync the GB10 worktree -> alienware, run the codex
    docker there (against the alienware-LOCAL proxy at `endpoint`), rsync the
    modified workspace back. Patch extraction + eval stay on the GB10 unchanged.
    Network-robust: resilient rsync (req #3) + a generous SSH timeout for the
    codex run; an SSH transport failure (rc 255) is CLASSIFIED network-drop and
    flagged (req #4) so the harness never reads a wire-stalled trajectory as a
    real run.
    """
    if _swe_agent_env() == "instance_image":
        return _run_agent_instance(
            remote_host=host, workspace=workspace, endpoint=endpoint, model=model,
            timeout_s=timeout_s, instance_id=instance_id, base_commit=base_commit,
            stdout_path=stdout_path, stderr_path=stderr_path, trace_path=trace_path,
            prompt=prompt, task_bearer=task_bearer,
        )
    started = time.monotonic()
    safe_id = instance_id.replace("/", "_")[:48]
    remote_ws = f"{_AGENT_REMOTE_ROOT}/{safe_id}/workspace"
    container_name = f"swe-agent-{safe_id}-{int(time.time())}"
    net_drop = False

    mk = _net_retry(["ssh", *_EVAL_SSH_OPTS, host, f"mkdir -p {remote_ws} && echo ok"],
                    what=f"codex_mkdir:{instance_id}", timeout=30)
    if mk.returncode != 0:
        net_drop = mk.returncode == 255
        stderr_path.write_text(f"remote mkdir failed rc={mk.returncode}\n{mk.stderr}", encoding="utf-8")
        trace_path.write_text("", encoding="utf-8")
        return {"elapsed_s": round(time.monotonic() - started, 3), "exit_code": -1,
                "timed_out": False, "container_name": container_name,
                "offloaded": True, "codex_host": host, "network_drop": net_drop}

    # push the hydrated worktree (trailing slash = contents into remote_ws)
    up = _net_retry([*_RSYNC_RESILIENT, f"{str(workspace).rstrip('/')}/", f"{host}:{remote_ws}/"],
                    what=f"codex_ws_up:{instance_id}", timeout=600, max_attempts=5)
    if up.returncode != 0:
        net_drop = True
        _agent_net_log(f"WS_UP_FAIL {instance_id} rc={up.returncode} (network-drop)")
        stderr_path.write_text(f"workspace rsync up failed rc={up.returncode}\n{up.stderr}", encoding="utf-8")
        trace_path.write_text("", encoding="utf-8")
        return {"elapsed_s": round(time.monotonic() - started, 3), "exit_code": -1,
                "timed_out": False, "container_name": container_name,
                "offloaded": True, "codex_host": host, "network_drop": net_drop}

    # the codex docker on alienware mounts the remote workspace + hits the
    # alienware-local proxy endpoint. ~ expands on the remote login shell.
    remote_cmd = _agent_template().format(
        container_name=container_name,
        workspace=remote_ws,
        endpoint=endpoint,
        model=model,
        session_id=fixed32_contract.fixed32_trace_session_id(instance_id),
    )
    # BUGFIX: CODEX_TEMPLATE no longer carries a {prompt} placeholder — the local path
    # appends the prompt as a separate argv element (see L459-466). The remote path used
    # to pass prompt=prompt into .format(), which silently DROPPED it (str.format ignores
    # unused kwargs) -> the remote codex launched with NO prompt -> "No prompt provided via
    # stdin" -> empty patch / agent_gave_up in ~3s (every offload SWE episode). Append the
    # prompt as a shell-quoted positional arg so it survives the ssh remote login shell and
    # reaches `codex exec` as its positional prompt, mirroring the local argv append.
    remote_cmd = _remote_agent_command(
        remote_cmd + " " + shlex.quote(prompt)
    )
    timed_out = False
    rc: int | None = None
    # run codex over SSH; trace (codex --json) -> trace_path, ssh stderr -> stderr_path.
    # SSH timeout = codex wall + a teardown buffer; a true codex timeout is
    # handled by killing the remote container (mirrors the local path).
    ssh_codex = ["ssh", *_EVAL_SSH_OPTS, "-o", "ConnectTimeout=20", host, remote_cmd]
    # FR13 §59: Popen + trace-growth stall-watchdog (env LUMO_SWE_STALL_KILL_S,
    # default 0=OFF). Wall timeout is enforced inside the monitor so OFF behavior
    # matches the legacy subprocess.run(timeout=wall). timeout_s<=0 => no wall.
    stall_kill_s = _stall_kill_s()
    wall_timeout_s = None if timeout_s <= 0 else max(timeout_s, 30) + 120
    stall_killed = False
    last_trace_growth_ts: float | None = None

    def _remote_stall_kill() -> None:
        # best-effort kill the remote container so a stalled run doesn't linger
        _net_retry(["ssh", *_EVAL_SSH_OPTS, host, f"docker kill {container_name} 2>/dev/null; "
                    f"docker wait {container_name} 2>/dev/null; echo killed"],
                   what=f"codex_stall_kill:{instance_id}", timeout=60, max_attempts=2)

    with trace_path.open("w", encoding="utf-8") as tf, \
         stderr_path.open("w", encoding="utf-8") as ef:
        proc = subprocess.Popen(
            ssh_codex,
            stdin=subprocess.PIPE,
            stdout=tf,
            stderr=ef,
        )
        if proc.stdin is None:
            raise Fixed32BoundaryError("agent credential pipe is unavailable")
        proc.stdin.write(
            ((task_bearer if task_bearer is not None else "EMPTY") + "\n").encode(
                "utf-8"
            )
        )
        proc.stdin.close()
        mon = _monitor_proc_with_stall_watchdog(
            proc, trace_path=trace_path, wall_timeout_s=wall_timeout_s,
            stall_kill_s=stall_kill_s, poll_s=5.0, on_stall_kill=_remote_stall_kill,
        )
    rc = mon["returncode"]
    timed_out = mon["timed_out"]
    stall_killed = mon["stall_killed"]
    last_trace_growth_ts = mon["last_trace_growth_ts"]
    if stall_killed:
        # infra_stall_suspect: the agent trace stopped growing (a suspected GB10
        # emit/transport wedge). Killed + CLASSIFIED; NEVER retried (nudge ban).
        _agent_net_log(f"CODEX_STALL_KILL {instance_id} no trace growth for {stall_kill_s}s "
                       "(infra_stall_suspect; NOT agent_gave_up, NOT retried)")
        rc = -1
    elif timed_out:
        # best-effort kill the remote container so it doesn't linger on alienware
        _net_retry(["ssh", *_EVAL_SSH_OPTS, host, f"docker kill {container_name} 2>/dev/null; "
                    f"docker wait {container_name} 2>/dev/null; echo killed"],
                   what=f"codex_kill:{instance_id}", timeout=60, max_attempts=2)
        rc = -1
    elif rc == 255:
        # SSH transport died — CLASSIFY network-drop (req #4), not a model run.
        net_drop = True
        _agent_net_log(f"CODEX_SSH_TRANSPORT_DROP {instance_id} rc=255 (network-drop, not a fork)")
    elapsed = time.monotonic() - started

    # rsync the (modified) workspace back to the GB10 for patch extraction.
    # delete-after keeps the GB10 worktree authoritative for git diff.
    down = _net_retry([*_RSYNC_RESILIENT, f"{host}:{remote_ws}/", f"{str(workspace).rstrip('/')}/"],
                      what=f"codex_ws_down:{instance_id}", timeout=600, max_attempts=5)
    if down.returncode != 0:
        net_drop = True
        _agent_net_log(f"WS_DOWN_FAIL {instance_id} rc={down.returncode} (network-drop) — "
                       "patch may be incomplete; FLAGGED")
    # cleanup the remote workspace (non-fatal)
    _net_retry(["ssh", *_EVAL_SSH_OPTS, host,
                f"docker rm -f {container_name} 2>/dev/null; rm -rf {_AGENT_REMOTE_ROOT}/{safe_id}; echo ok"],
               what=f"codex_cleanup:{instance_id}", timeout=60, max_attempts=2)

    if not stdout_path.is_file():
        stdout_path.write_text("", encoding="utf-8")
    result: dict[str, Any] = {
        "elapsed_s": round(elapsed, 3),
        "exit_code": rc if rc is not None else -1,
        "timed_out": timed_out,
        "container_name": container_name,
        "offloaded": True,
        "codex_host": host,
        "network_drop": net_drop,
        "ws_down_rc": down.returncode,
        "stall_killed": stall_killed,
    }
    if stall_killed:
        result["cause"] = "infra_stall_suspect"
        result["last_trace_growth_ts"] = last_trace_growth_ts
    return result


_REMOTE_AGENT_TRACE_OBSERVATION_SCHEMA = (
    "fr13-remote-qwen-trace-observation-v1"
)
_REMOTE_AGENT_TRACE_OBSERVATION_SCRIPT = r"""
import hashlib
import json
import os
import pathlib
import stat
import sys

path = pathlib.Path(sys.argv[1]).expanduser()
schema = sys.argv[2]
parent = path.parent
parent_metadata = parent.lstat()
if not stat.S_ISDIR(parent_metadata.st_mode) or parent.is_symlink():
    raise RuntimeError("trace parent is not a real directory")
if (
    stat.S_IMODE(parent_metadata.st_mode) != 0o700
    or parent_metadata.st_uid != os.geteuid()
):
    raise RuntimeError("trace parent is not private or shell-owned")
parent_fd = os.open(
    parent,
    os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
)


def identity(metadata):
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


try:
    before = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise RuntimeError("trace is not a single-link regular file")
    if (
        stat.S_IMODE(before.st_mode) != 0o600
        or before.st_uid != os.geteuid()
    ):
        raise RuntimeError("trace is not private or shell-owned")
    if os.listxattr(path, follow_symlinks=False):
        raise RuntimeError("trace has extended attributes")
    descriptor = os.open(
        path.name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=parent_fd,
    )
    try:
        opened = os.fstat(descriptor)
        if identity(opened) != identity(before):
            raise RuntimeError("trace changed before exact read")
        digest = hashlib.sha256()
        byte_count = 0
        final_byte = b""
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            final_byte = chunk[-1:]
        after_read = os.fstat(descriptor)
        if identity(after_read) != identity(opened):
            raise RuntimeError("trace changed during exact read")
    finally:
        os.close(descriptor)
    after = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
    if identity(after) != identity(before):
        raise RuntimeError("trace changed after exact read")
    if os.listxattr(path, follow_symlinks=False):
        raise RuntimeError("trace gained extended attributes")
    print(
        json.dumps(
            {
                "schema": schema,
                "bytes": byte_count,
                "sha256": digest.hexdigest(),
                "newline_framed": byte_count > 0 and final_byte == b"\n",
                "mode": f"{stat.S_IMODE(after.st_mode):04o}",
                "uid": after.st_uid,
                "gid": after.st_gid,
                "nlink": after.st_nlink,
                "xattrs": [],
                "file_identity_sha256": hashlib.sha256(
                    f"{after.st_dev}:{after.st_ino}".encode("ascii")
                ).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
finally:
    os.close(parent_fd)
"""


def _remote_agent_trace_path(remote_out: str) -> str:
    return f"{remote_out}/{_REMOTE_AGENT_TRACE_FILENAME}"


def _remote_agent_trace_capture_command(
    command: str,
    *,
    remote_trace_path: str,
) -> str:
    if remote_trace_path.startswith("~/"):
        rendered_path = "$HOME/" + shlex.quote(remote_trace_path[2:])
    elif remote_trace_path.startswith("/"):
        rendered_path = shlex.quote(remote_trace_path)
    else:
        raise Fixed32BoundaryError("remote trace path is not absolute")
    return (
        "IFS= read -r OPENAI_API_KEY && export OPENAI_API_KEY && "
        "umask 077 && "
        f"trace_path={rendered_path} && "
        'test -d "$(dirname -- "$trace_path")" && '
        'test ! -L "$trace_path" && test ! -e "$trace_path" && '
        '(set -C; : > "$trace_path") && '
        'chmod 0600 -- "$trace_path" && '
        "{ "
        f"{command} > /dev/null; "
        "qwen_rc=$?; "
        'if [ ! -f "$trace_path" ] || [ -L "$trace_path" ]; then '
        "exit 90; fi; "
        'exit "$qwen_rc"; '
        "}"
    )


def _validate_remote_agent_trace_observation(payload: Any) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "bytes",
        "sha256",
        "newline_framed",
        "mode",
        "uid",
        "gid",
        "nlink",
        "xattrs",
        "file_identity_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise Fixed32BoundaryError("remote Qwen trace observation is malformed")
    for key in ("uid", "gid"):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise Fixed32BoundaryError(
                "remote Qwen trace observation is malformed"
            )
    byte_count = payload["bytes"]
    if (
        payload["schema"] != _REMOTE_AGENT_TRACE_OBSERVATION_SCHEMA
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
        or not isinstance(payload["newline_framed"], bool)
        or payload["mode"] != "0600"
        or payload["nlink"] != 1
        or payload["xattrs"] != []
    ):
        raise Fixed32BoundaryError("remote Qwen trace observation is malformed")
    for key in ("sha256", "file_identity_sha256"):
        digest = payload[key]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise Fixed32BoundaryError(
                "remote Qwen trace observation is malformed"
            )
    return payload


def _observe_remote_agent_trace(
    *,
    host: str,
    instance_id: str,
    remote_trace_path: str,
) -> dict[str, Any]:
    command = " ".join(
        (
            "python3",
            "-c",
            shlex.quote(_REMOTE_AGENT_TRACE_OBSERVATION_SCRIPT),
            shlex.quote(remote_trace_path),
            shlex.quote(_REMOTE_AGENT_TRACE_OBSERVATION_SCHEMA),
        )
    )
    observed = _net_retry(
        ["ssh", *_EVAL_SSH_OPTS, host, command],
        what=f"qwen_trace_observe:{instance_id}",
        timeout=120,
        max_attempts=3,
    )
    if observed.returncode != 0:
        raise Fixed32BoundaryError(
            "remote Qwen trace observation failed: "
            f"host={host} instance={instance_id} rc={observed.returncode}"
        )
    return _validate_remote_agent_trace_observation(
        _fixed32_load_json_object(
            observed.stdout or "",
            label=f"remote Qwen trace observation {instance_id}",
        )
    )


def _pull_remote_agent_trace(
    *,
    host: str,
    instance_id: str,
    remote_trace_path: str,
    trace_path: Path,
) -> dict[str, Any]:
    before = _observe_remote_agent_trace(
        host=host,
        instance_id=instance_id,
        remote_trace_path=remote_trace_path,
    )
    pull_path = trace_path.with_name(
        f".{trace_path.name}.{secrets.token_hex(8)}.pull"
    )
    if pull_path.exists() or pull_path.is_symlink():
        raise Fixed32BoundaryError("local Qwen trace pull path is not fresh")
    try:
        pulled = _net_retry(
            [
                "scp",
                *_EVAL_SSH_OPTS,
                f"{host}:{remote_trace_path}",
                str(pull_path),
            ],
            what=f"qwen_trace_down:{instance_id}",
            timeout=120,
            max_attempts=4,
        )
        if pulled.returncode != 0:
            raise Fixed32BoundaryError(
                "remote Qwen trace download failed: "
                f"host={host} instance={instance_id} rc={pulled.returncode}"
            )
        after = _observe_remote_agent_trace(
            host=host,
            instance_id=instance_id,
            remote_trace_path=remote_trace_path,
        )
        if after != before:
            raise Fixed32BoundaryError(
                "remote Qwen trace identity changed during download"
            )
        try:
            local_metadata = pull_path.lstat()
            if (
                not stat.S_ISREG(local_metadata.st_mode)
                or local_metadata.st_nlink != 1
            ):
                raise Fixed32BoundaryError(
                    "downloaded Qwen trace is not a single-link regular file"
                )
            raw_trace = pull_path.read_bytes()
        except OSError as exc:
            raise Fixed32BoundaryError(
                "downloaded Qwen trace cannot be read"
            ) from exc
        if (
            len(raw_trace) != before["bytes"]
            or hashlib.sha256(raw_trace).hexdigest() != before["sha256"]
            or raw_trace.endswith(b"\n") != before["newline_framed"]
        ):
            raise Fixed32BoundaryError(
                "downloaded Qwen trace identity differs from the remote capture"
            )

        # Install the exact pulled bytes before semantic validation. If the
        # trace is malformed, the evidence remains intact at its canonical path.
        os.replace(pull_path, trace_path)
        _raw_trace, events = _fixed32_load_trace_events(
            trace_path,
            instance_id=instance_id,
        )
        return {
            "schema": _REMOTE_AGENT_TRACE_OBSERVATION_SCHEMA,
            "bytes": before["bytes"],
            "sha256": before["sha256"],
            "event_count": len(events),
        }
    finally:
        try:
            pull_path.unlink(missing_ok=True)
        except OSError:
            pass


def _fixed32_remote_agent_paths(
    instance_id: str,
    *,
    nonce: str | None = None,
) -> tuple[str, str, str, str]:
    allowed = frozenset(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    )
    safe_id = "".join(
        character if character in allowed else "_" for character in instance_id
    )[:48]
    safe_id = safe_id or "task"
    launch_nonce = nonce or secrets.token_hex(8)
    if (
        not launch_nonce
        or any(character not in allowed for character in launch_nonce)
    ):
        raise Fixed32BoundaryError("remote agent launch nonce is unsafe")
    container_name = f"swe-qwen-{safe_id}-{launch_nonce}"
    task_root = f"{_AGENT_REMOTE_ROOT}/{safe_id}-{launch_nonce}"
    return (
        container_name,
        task_root,
        f"{task_root}/out",
        f"{task_root}/qwen_system_settings.json",
    )


def _prepare_remote_agent_task(
    *,
    host: str,
    instance_id: str,
    task_root: str,
    out_dir: str,
) -> None:
    prepared = _net_retry(
        [
            "ssh",
            *_EVAL_SSH_OPTS,
            host,
            (
                f"set -eu; rm -rf -- {task_root}; "
                f"test ! -e {task_root}; test ! -L {task_root}; "
                f"umask 077; mkdir -m 0700 -- {task_root}; "
                f"mkdir -m 0700 -- {out_dir}; "
                f"test -d {out_dir}; test ! -L {out_dir}"
            ),
        ],
        what=f"qwen_task_prepare:{instance_id}",
        timeout=60,
        max_attempts=3,
    )
    if prepared.returncode != 0:
        raise Fixed32BoundaryError(
            "remote Qwen task-root preparation failed: "
            f"host={host} instance={instance_id} rc={prepared.returncode}"
        )


def _cleanup_remote_agent_task(
    *,
    host: str,
    instance_id: str,
    container_name: str,
    task_root: str,
) -> None:
    cleanup = _net_retry(
        [
            "ssh",
            *_EVAL_SSH_OPTS,
            host,
            (
                "set -eu; "
                f"ids=$(docker ps -aq --filter name=^/{container_name}$); "
                'if [ -n "$ids" ]; then docker rm -f $ids >/dev/null; fi; '
                f"rm -rf -- {task_root}; "
                f"test ! -e {task_root}; test ! -L {task_root}"
            ),
        ],
        what=f"qwen_task_cleanup:{instance_id}",
        timeout=60,
        max_attempts=3,
    )
    if cleanup.returncode != 0:
        raise Fixed32BoundaryError(
            "remote Qwen task cleanup failed: "
            f"host={host} instance={instance_id} rc={cleanup.returncode}"
        )


def _stop_remote_agent_container(
    *,
    host: str,
    instance_id: str,
    container_name: str,
) -> None:
    stopped = _net_retry(
        [
            "ssh",
            *_EVAL_SSH_OPTS,
            host,
            (
                "set -eu; "
                f"ids=$(docker ps -aq --filter name=^/{container_name}$); "
                'if [ -n "$ids" ]; then docker rm -f $ids >/dev/null; fi; '
                f"test -z \"$(docker ps -aq --filter "
                f"name=^/{container_name}$)\""
            ),
        ],
        what=f"qwen_stop:{instance_id}",
        timeout=60,
        max_attempts=2,
    )
    if stopped.returncode != 0:
        raise Fixed32BoundaryError(
            "remote Qwen container did not stop before trace download: "
            f"host={host} instance={instance_id} rc={stopped.returncode}"
        )


def _run_agent_instance(
    *,
    remote_host: str | None,
    workspace: Path,
    endpoint: str,
    model: str,
    timeout_s: int,
    instance_id: str,
    base_commit: str | None,
    stdout_path: Path,
    stderr_path: Path,
    trace_path: Path,
    prompt: str = DEFAULT_AGENT_PROMPT,
    task_bearer: str | None = None,
) -> dict[str, Any]:
    """SWE_AGENT_ENV=instance_image (§58): run qwen-code INSIDE the SWE-bench
    per-instance eval image editing /testbed, with the node+qwen runtime injected
    read-only from the host bundle. remote_host=None -> local docker (GB10);
    else -> docker on the codex host (alienware) over SSH, mirroring the legacy
    offload plumbing (fail-loud image precondition, network-drop classification,
    generous SSH timeout). The wrapper extracts the patch to a /out bind-mount;
    we copy it back to <workspace>/patch.diff where _extract_patch reads it."""
    import base64 as _b64
    started = time.monotonic()
    fixed32_launch = task_bearer is not None
    if fixed32_launch:
        _validate_fixed32_agent_runtime_mode(remote_host=remote_host)
        _fixed32_qwen_settings_metadata()
    container_name, remote_task_root, remote_out, remote_settings_root = (
        _fixed32_remote_agent_paths(instance_id)
    )
    trace_session_id = fixed32_contract.fixed32_trace_session_id(instance_id)
    # ARCH PLACEMENT (user 2026-07-05): derive the image variant from the AGENT
    # host's architecture. Some instances only publish x86_64 images — those must
    # run on the x86 offload host; on an aarch64 host with no arm64 variant we
    # fail LOUD (never silently fall back to the bare-worktree env, class 9).
    host_machine = _host_arch(remote_host)
    sweb_arch = _SWEB_ARCH.get(host_machine)
    if sweb_arch is None:
        raise RuntimeError(
            f"SWE_AGENT_ENV=instance_image: unsupported agent-host arch "
            f"{host_machine!r} (host={remote_host or 'local'})")
    image = _instance_image_name(instance_id, arch=sweb_arch)
    _record_image_arch_tag(workspace, instance_id, {
        "image": image, "image_arch": sweb_arch, "agent_host": remote_host or "local",
        "host_machine": host_machine})
    if not base_commit:
        raise RuntimeError(
            "SWE_AGENT_ENV=instance_image requires base_commit (the agent's "
            "patch is `git diff <base_commit>` inside /testbed)")
    agents_md_path = Path(workspace) / "AGENTS.md"
    agents_md_text = (agents_md_path.read_text(encoding="utf-8")
                      if agents_md_path.is_file() else "")
    agents_b64 = _b64.b64encode(agents_md_text.encode("utf-8")).decode("ascii")
    # The shared operator/retry prompts reference `/workspace/AGENTS.md` (the
    # legacy qwen-code-runner mount). In instance_image mode AGENTS.md is written
    # to /testbed/AGENTS.md, the workdir is /testbed, and there is NO /workspace
    # mount — so remap the path here (confined to this mode; legacy never calls
    # this function) or the agent is told to read a file that does not exist.
    prompt = prompt.replace("/workspace/", "/testbed/")
    prompt_b64 = _b64.b64encode(prompt.encode("utf-8")).decode("ascii")
    net_drop = False
    timed_out = False
    rc: int | None = None
    patch_local = Path(workspace) / "patch.diff"
    runtime_attestation: dict[str, Any] | None = None
    runtime_attestation_sha256: str | None = None
    runtime_postrun_attestation_sha256: str | None = None
    placement_observation: dict[str, Any] | None = None
    mounted_runtime_proof: dict[str, Any] | None = None
    mounted_runtime_proof_sha256: str | None = None
    mounted_runtime_proof_file_sha256: str | None = None

    if remote_host:
        image_observation: dict[str, Any] | None = None
        if fixed32_launch:
            placement_observation = _inspect_fixed32_agent_placement_remote(
                remote_host
            )
            image_observation = _inspect_fixed32_agent_image_remote(
                host=remote_host,
                instance_id=instance_id,
                image=image,
            )
        else:
            # Fail loud on missing infrastructure; never fall back to a
            # worktree when the canonical instance image is absent.
            inspected = _net_retry(
                [
                    "ssh",
                    *_EVAL_SSH_OPTS,
                    remote_host,
                    f"docker image inspect {shlex.quote(image)} "
                    ">/dev/null 2>&1 && echo present || echo absent",
                ],
                what=f"img_inspect:{instance_id}",
                timeout=60,
                max_attempts=3,
            )
            if "present" not in (inspected.stdout or ""):
                raise RuntimeError(
                    f"SWE_AGENT_ENV=instance_image: image {image} ABSENT on "
                    f"codex host {remote_host} (host arch={host_machine}). "
                    "Refusing to fall back."
                )
        bundle_observation: dict[str, Any] | None = None
        body_error: BaseException | None = None
        try:
            _prepare_remote_agent_task(
                host=remote_host,
                instance_id=instance_id,
                task_root=remote_task_root,
                out_dir=remote_out,
            )
            remote_system_settings: str | None = None
            remote_bundle_snapshot: str | None = None
            remote_settings_observation: dict[str, Any] | None = None
            if fixed32_launch:
                remote_system_settings = remote_settings_root
                remote_settings_observation = (
                    _install_fixed32_qwen_settings_remote(
                        host=remote_host,
                        remote_path=remote_system_settings,
                    )
                )
                (
                    remote_bundle_snapshot,
                    bundle_observation,
                ) = _create_fixed32_qwen_snapshot_remote(
                    host=remote_host,
                    instance_id=instance_id,
                    task_root=remote_task_root,
                )
                remote_settings_observation = (
                    _require_fixed32_remote_settings_stable(
                        remote_settings_observation,
                        _verify_fixed32_qwen_settings_remote(
                            host=remote_host,
                            remote_path=remote_system_settings,
                        ),
                    )
                )
                runtime_attestation = (
                    _build_fixed32_qwen_runtime_attestation(
                        bundle_observation=bundle_observation,
                        host_mode="remote",
                    )
                )
                runtime_attestation_sha256 = (
                    _persist_fixed32_qwen_runtime_attestation(
                        workspace=workspace,
                        attestation=runtime_attestation,
                    )
                )
            run_image = (
                image_observation["repo_digest"]
                if image_observation is not None
                else image
            )
            # The fixed32 command runs the immutable preinspected RepoDigest,
            # while the canonical tag remains in the persisted identity.
            cmd = _instance_agent_command(
                container_name=container_name,
                image=run_image,
                endpoint=endpoint,
                model=model,
                host_out_dir=remote_out,
                bundle_src=(
                    remote_bundle_snapshot
                    if remote_bundle_snapshot is not None
                    else "~/qwen_agent_bundle"
                ),
                agents_md_b64=agents_b64,
                prompt_b64=prompt_b64,
                base_commit=base_commit,
                session_id=trace_session_id,
                system_settings_src=remote_system_settings,
                bundle_observation=bundle_observation,
                trace_output_path=(
                    _INSTANCE_TRACE_OUTPUT_PATH if fixed32_launch else None
                ),
            )
            remote_trace_path = (
                _remote_agent_trace_path(remote_out)
                if fixed32_launch
                else None
            )
            remote_command = (
                _remote_agent_trace_capture_command(
                    cmd,
                    remote_trace_path=remote_trace_path,
                )
                if remote_trace_path is not None
                else _remote_agent_command(cmd)
            )
            ssh_cmd = [
                "ssh",
                *_EVAL_SSH_OPTS,
                "-o",
                "ConnectTimeout=20",
                remote_host,
                remote_command,
            ]
            try:
                trace_output = (
                    contextlib.nullcontext(subprocess.DEVNULL)
                    if fixed32_launch
                    else trace_path.open("w", encoding="utf-8")
                )
                with trace_output as tf, stderr_path.open(
                    "w", encoding="utf-8"
                ) as ef:
                    completed = subprocess.run(
                        ssh_cmd,
                        input=(
                            task_bearer
                            if task_bearer is not None
                            else "EMPTY"
                        )
                        + "\n",
                        text=True,
                        stdout=tf,
                        stderr=ef,
                        timeout=(
                            None
                            if timeout_s <= 0
                            else max(timeout_s, 30) + 120
                        ),
                        check=False,
                    )
                rc = completed.returncode
                if rc == 255:
                    net_drop = True
                    _agent_net_log(
                        f"QWEN_SSH_TRANSPORT_DROP {instance_id} rc=255 "
                        "(network-drop, not a fork)"
                    )
                    if fixed32_launch:
                        _stop_remote_agent_container(
                            host=remote_host,
                            instance_id=instance_id,
                            container_name=container_name,
                        )
            except subprocess.TimeoutExpired:
                timed_out = True
                if fixed32_launch:
                    _stop_remote_agent_container(
                        host=remote_host,
                        instance_id=instance_id,
                        container_name=container_name,
                    )
                else:
                    _net_retry(
                        [
                            "ssh",
                            *_EVAL_SSH_OPTS,
                            remote_host,
                            f"docker kill {container_name} 2>/dev/null; "
                            f"docker wait {container_name} 2>/dev/null; "
                            "echo killed",
                        ],
                        what=f"qwen_kill:{instance_id}",
                        timeout=60,
                        max_attempts=2,
                    )
                rc = -1
            elapsed = time.monotonic() - started
            trace_capture: dict[str, Any] | None = None
            if remote_trace_path is not None:
                trace_capture = _pull_remote_agent_trace(
                    host=remote_host,
                    instance_id=instance_id,
                    remote_trace_path=remote_trace_path,
                    trace_path=trace_path,
                )
            if fixed32_launch:
                if bundle_observation is None:
                    raise Fixed32BoundaryError(
                        "fixed32 snapshot observation is unavailable"
                    )
                (
                    mounted_runtime_proof,
                    mounted_runtime_proof_sha256,
                    mounted_runtime_proof_file_sha256,
                ) = _pull_fixed32_mounted_runtime_proof(
                    host=remote_host,
                    remote_out=remote_out,
                    task_dir=workspace.parent,
                    expected_bundle_observation=bundle_observation,
                )
            # scp rc=1 means the agent produced no patch.
            pd = _net_retry(
                [
                    "scp",
                    *_EVAL_SSH_OPTS,
                    f"{remote_host}:{remote_out}/patch.diff",
                    str(patch_local),
                ],
                what=f"qwen_patch_down:{instance_id}",
                timeout=120,
                max_attempts=4,
                ok_rcs=(0, 1),
            )
            if pd.returncode not in (0, 1):
                net_drop = True
            if pd.returncode != 0 and not patch_local.is_file():
                patch_local.write_text("", encoding="utf-8")
            if fixed32_launch:
                try:
                    if remote_system_settings is None:
                        raise Fixed32BoundaryError(
                            "fixed32 remote settings path is unavailable"
                        )
                    if (
                        remote_bundle_snapshot is None
                        or bundle_observation is None
                        or remote_settings_observation is None
                        or placement_observation is None
                    ):
                        raise Fixed32BoundaryError(
                            "fixed32 pre-run runtime identity is unavailable"
                        )
                    postrun_settings_observation = (
                        _require_fixed32_remote_settings_stable(
                            remote_settings_observation,
                            _verify_fixed32_qwen_settings_remote(
                                host=remote_host,
                                remote_path=remote_system_settings,
                            ),
                        )
                    )
                    postrun_bundle_observation = (
                        _inspect_fixed32_qwen_bundle_remote_path(
                            remote_host,
                            remote_bundle_snapshot,
                        )
                    )
                    postrun_attestation = (
                        _build_fixed32_qwen_runtime_attestation(
                            bundle_observation=postrun_bundle_observation,
                            host_mode="remote",
                        )
                    )
                    postrun_image_observation = (
                        _inspect_fixed32_agent_image_remote(
                            host=remote_host,
                            instance_id=instance_id,
                            image=image,
                        )
                    )
                    postrun_placement_observation = (
                        _inspect_fixed32_agent_placement_remote(remote_host)
                    )
                    if (
                        postrun_attestation != runtime_attestation
                        or postrun_image_observation != image_observation
                        or postrun_bundle_observation != bundle_observation
                        or postrun_settings_observation
                        != remote_settings_observation
                        or postrun_placement_observation
                        != placement_observation
                        or mounted_runtime_proof is None
                        or mounted_runtime_proof["system_settings"][
                            "file_identity_sha256"
                        ]
                        != remote_settings_observation[
                            "file_identity_sha256"
                        ]
                    ):
                        raise Fixed32BoundaryError(
                            "fixed32 pre-mounted-post runtime identity differs"
                        )
                    runtime_postrun_attestation_sha256 = (
                        _persist_fixed32_qwen_runtime_attestation(
                            workspace=workspace,
                            attestation=postrun_attestation,
                            filename=(
                                "qwen_runtime_attestation_post.json"
                            ),
                        )
                    )
                except Exception as exc:
                    if isinstance(exc, Fixed32BoundaryError):
                        raise
                    raise Fixed32BoundaryError(
                        "fixed32 Qwen post-run attestation failed: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
            if not stdout_path.is_file():
                stdout_path.write_text("", encoding="utf-8")
            result = {
                "elapsed_s": round(elapsed, 3),
                "exit_code": rc if rc is not None else -1,
                "timed_out": timed_out,
                "container_name": container_name,
                "offloaded": True,
                "codex_host": remote_host,
                "network_drop": net_drop,
                "patch_down_rc": pd.returncode,
                "agent_env": "instance_image",
                "instance_image": image,
                "host_arch": host_machine,
                "image_arch": sweb_arch,
            }
            if fixed32_launch:
                if trace_capture is None:
                    raise Fixed32BoundaryError(
                        "fixed32 remote trace capture evidence is unavailable"
                    )
                result["qwen_trace_capture"] = trace_capture
                if (
                    image_observation is None
                    or placement_observation is None
                    or bundle_observation is None
                    or remote_settings_observation is None
                    or mounted_runtime_proof is None
                    or mounted_runtime_proof_sha256 is None
                    or mounted_runtime_proof_file_sha256 is None
                ):
                    raise Fixed32BoundaryError(
                        "fixed32 mounted runtime evidence is incomplete"
                    )
                image_identity_sha256 = (
                    _fixed32_canonical_json_sha256(image_observation)
                )
                placement_identity_sha256 = (
                    _fixed32_canonical_json_sha256(placement_observation)
                )
                remote_settings_identity_sha256 = (
                    _fixed32_canonical_json_sha256(
                        remote_settings_observation
                    )
                )
                result["qwen_runtime_attestation"] = runtime_attestation
                result["qwen_runtime_attestation_sha256"] = (
                    runtime_attestation_sha256
                )
                result["qwen_runtime_postrun_attestation_sha256"] = (
                    runtime_postrun_attestation_sha256
                )
                result["instance_image_identity"] = image_observation
                result["instance_image_identity_sha256"] = (
                    image_identity_sha256
                )
                result["instance_image_postrun_identity_sha256"] = (
                    image_identity_sha256
                )
                result["instance_image_run_reference"] = (
                    image_observation["repo_digest"]
                )
                result["qwen_bundle_snapshot"] = (
                    runtime_attestation["bundle_snapshot"]
                )
                result["qwen_mounted_runtime_proof"] = mounted_runtime_proof
                result["qwen_mounted_runtime_proof_sha256"] = (
                    mounted_runtime_proof_sha256
                )
                result["qwen_mounted_runtime_proof_file_sha256"] = (
                    mounted_runtime_proof_file_sha256
                )
                result["qwen_remote_settings_observation"] = (
                    remote_settings_observation
                )
                result["qwen_remote_settings_observation_sha256"] = (
                    remote_settings_identity_sha256
                )
                result[
                    "qwen_remote_settings_postrun_observation_sha256"
                ] = remote_settings_identity_sha256
                result["agent_placement"] = placement_observation
                result["agent_placement_sha256"] = (
                    placement_identity_sha256
                )
                result["agent_postrun_placement_sha256"] = (
                    placement_identity_sha256
                )
            return result
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            try:
                _cleanup_remote_agent_task(
                    host=remote_host,
                    instance_id=instance_id,
                    container_name=container_name,
                    task_root=remote_task_root,
                )
            except Fixed32BoundaryError as cleanup_error:
                if body_error is not None:
                    raise Fixed32BoundaryError(
                        "remote Qwen task failed and cleanup also failed: "
                        f"task_error={type(body_error).__name__}: "
                        f"{body_error}; cleanup_error={cleanup_error}"
                    ) from body_error
                raise

    # ---- local docker path (secondary; production uses --codex-host) ----
    local_bundle_root = Path(
        os.path.expanduser("~/qwen_agent_bundle")
    )
    insp = subprocess.run(["docker", "image", "inspect", image],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if insp.returncode != 0:
        raise RuntimeError(
            f"SWE_AGENT_ENV=instance_image: image {image} ABSENT locally "
            f"(host arch={host_machine}; x86_64-only instances must run offloaded "
            f"on alienware via --codex-host) "
            f"(pull it first; refusing to fall back)")
    cmd = _instance_agent_command(
        container_name=container_name, image=image, endpoint=endpoint, model=model,
        host_out_dir=str(Path(workspace).resolve()),
        bundle_src=str(local_bundle_root),
        agents_md_b64=agents_b64, prompt_b64=prompt_b64,
        base_commit=base_commit, session_id=trace_session_id,
        system_settings_src=None)
    cmd_argv = shlex.split(cmd)
    try:
        with trace_path.open("w", encoding="utf-8") as tf, \
             stderr_path.open("w", encoding="utf-8") as ef:
            completed = subprocess.run(
                cmd_argv, stdout=tf, stderr=ef,
                timeout=(None if timeout_s <= 0 else max(timeout_s, 30)),
                check=False,
                env=_agent_subprocess_env(task_bearer))
        rc = completed.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        subprocess.run(["docker", "kill", container_name], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            subprocess.run(["docker", "wait", container_name], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        except subprocess.TimeoutExpired:
            pass
        rc = -1
    elapsed = time.monotonic() - started
    if not patch_local.is_file():
        patch_local.write_text("", encoding="utf-8")
    if not stdout_path.is_file():
        stdout_path.write_text("", encoding="utf-8")
    result = {
        "elapsed_s": round(elapsed, 3),
        "exit_code": rc if rc is not None else -1,
        "timed_out": timed_out,
        "container_name": container_name,
        "agent_env": "instance_image",
        "instance_image": image,
        "host_arch": host_machine,
        "image_arch": sweb_arch,
    }
    return result


def _run_agent_dispatch(**kwargs: Any) -> dict[str, Any]:
    """Route the codex run to alienware (AGENT_HOST set) or local (GB10).

    On the offload path the codex docker on alienware must hit the alienware-
    LOCAL proxy, so the GB10-side `--endpoint` is overridden with AGENT_ENDPOINT.
    """
    if kwargs.get("task_bearer") is not None:
        _validate_fixed32_agent_runtime_mode(remote_host=AGENT_HOST)
    if AGENT_HOST:
        kwargs = dict(kwargs)
        if AGENT_ENDPOINT:
            kwargs["endpoint"] = AGENT_ENDPOINT
        return _run_agent_remote(host=AGENT_HOST, **kwargs)
    return _run_agent_local(**kwargs)


def _run_eval(
    *,
    instance_id: str,
    patch_path: Path,
    output_dir: Path,
    dataset_name: str,
    model_name: str,
    timeout_s: int,
    eval_log_path: Path,
) -> dict[str, Any]:
    if EVAL_HOST:
        return _run_eval_remote(
            host=EVAL_HOST, instance_id=instance_id, patch_path=patch_path,
            output_dir=output_dir, dataset_name=dataset_name, model_name=model_name,
            timeout_s=timeout_s, eval_log_path=eval_log_path,
        )
    cbe_exe = shutil.which("codex-bench-eval-swe")
    if cbe_exe is None:
        cbe_exe = str(REPO_ROOT / ".venv" / "bin" / "codex-bench-eval-swe")
    cmd = [
        cbe_exe,
        "--instance-id", instance_id,
        "--patch-path", str(patch_path),
        "--output-dir", str(output_dir),
        "--dataset-name", dataset_name,
        "--model-name", model_name,
        "--timeout-s", str(timeout_s),
        "--cache-level", "env",
    ]
    started = time.monotonic()
    with eval_log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - started
    return {
        "exit_code": proc.returncode,
        "elapsed_s": round(elapsed, 3),
    }


def _process_one(
    *,
    instance_id: str,
    instance: dict,
    dataset_name: str,
    per_task_root: Path,
    repo_cache_root: Path,
    endpoint: str,
    model: str,
    model_name: str,
    agent_wall_s: int,
    eval_timeout_s: int,
    skip_existing: bool,
    fixed32_bracket: _Fixed32TaskBracket | None = None,
    fixed32_b1_diagnostic: bool = False,
    fixed32_cfwd_qualification: bool = False,
) -> dict[str, Any]:
    # Use absolute paths everywhere so docker volume mounts and git
    # worktree add (which resolves relative to -C cache) both work.
    task_dir = (per_task_root / instance_id).resolve()
    task_dir.mkdir(parents=True, exist_ok=True)
    runner_meta_path = task_dir / "runner_metadata.json"
    fixed32_campaign_scope = (
        fixed32_bracket is not None and fixed32_bracket.server_capacity == 4
    )
    pending_runner_meta_path = (
        task_dir / _FIXED32_PENDING_RUNNER_METADATA_FILENAME
    )
    if skip_existing and runner_meta_path.is_file():
        return {"instance_id": instance_id, "status": "skipped_existing"}

    workspace_path = task_dir / "workspace"
    patch_path = task_dir / "patch.diff"
    qwen_stdout = task_dir / "qwen_stdout.log"
    qwen_stderr = task_dir / "qwen_stderr.log"
    qwen_trace = task_dir / "qwen_trace.jsonl"
    prompt_md = task_dir / "prompt.md"
    metrics_pre = task_dir / "vllm_metrics_pre.txt"
    metrics_post = task_dir / "vllm_metrics_post.txt"
    per_turn_json = task_dir / "vllm_per_turn.json"
    dcgm_samples = task_dir / "dcgm_samples.jsonl"
    eval_log = task_dir / "eval_invocation.log"
    eval_output = task_dir / "eval"
    eval_output.mkdir(parents=True, exist_ok=True)

    started_iso = _iso_now()
    summary: dict[str, Any] = {
        "instance_id": instance_id,
        "dataset_name": dataset_name,
        "started_at": started_iso,
        "repo": instance.get("repo"),
        "base_commit": instance.get("base_commit"),
    }
    task_bearer: str | None = None
    task_key_id: str | None = None
    fixed32_initial_provenance_args: dict[str, Any] | None = None
    if fixed32_bracket is not None:
        task_bearer, task_key_id = _fixed32_task_auth(instance_id)
        summary["fixed32_dataset_record_sha256"] = hashlib.sha256(
            json.dumps(
                instance,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if fixed32_cfwd_qualification:
            cfwd_classification = (
                _FIXED32_CFWD_B4_QUALIFICATION_CLASSIFICATION + "_member"
                if fixed32_campaign_scope
                else _FIXED32_CFWD_QUALIFICATION_CLASSIFICATION
            )
            summary["fixed32_run_classification"] = {
                "run_classification": cfwd_classification,
                "performance_measurement": False,
                "timing_eligible": False,
                "gate_eligible": False,
                "floor_acceptance_eligible": False,
                "process_local_qualification_only": True,
                "durable_production_pass": False,
                "timing_requires_same_server_process": True,
                "same_process_timing_handoff_contract_implemented": (
                    fixed32_campaign_scope
                ),
                "same_process_timing_execution_implemented": False,
            }
        elif fixed32_b1_diagnostic:
            summary["fixed32_run_classification"] = {
                "run_classification": "b1_diagnostic",
                "gate_eligible": False,
                "floor_acceptance_eligible": False,
            }

    cache_path = None
    try:
        if _swe_agent_env() == "instance_image":
            # instance_image mode: /testbed lives inside the per-instance eval
            # image, so skip the repo clone/fetch/worktree entirely; the
            # workspace is only a host staging dir (AGENTS.md + /out patch sink).
            _hydrate_workspace(
                cache_path=None,
                base_commit=instance["base_commit"],
                workspace_path=workspace_path,
            )
        else:
            cache_path = _ensure_repo_cache(instance["repo"], repo_cache_root)
            _fetch_commit(cache_path, instance["base_commit"])
            _hydrate_workspace(
                cache_path=cache_path,
                base_commit=instance["base_commit"],
                workspace_path=workspace_path,
            )
        _write_agents_md(workspace_path, instance)
    except Exception as exc:  # noqa: BLE001
        if fixed32_bracket is not None:
            raise Fixed32BoundaryError(
                f"fixed32 task hydration failed for {instance_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        summary["status"] = "hydration_failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        runner_meta_path.write_text(json.dumps(summary, indent=2))
        if cache_path is not None:
            _remove_workspace(cache_path, workspace_path)
        return summary

    # Write prompt.md mirroring the Track B layout. The codex agent receives
    # both the docker-CLI prompt (the "Read the task prompt..." string) and
    # the AGENTS.md content inside /workspace.
    prompt_md.write_text(
        "## Codex CLI invocation prompt\n\n"
        '"Read the task prompt at /workspace/AGENTS.md and complete it in this '
        "workspace. Edit the source files directly to implement the fix. Do not "
        "write a diff file -- modify the files in place so that running pytest "
        'passes the tests described in the prompt."\n\n'
        f"## AGENTS.md (workspace/{instance_id})\n\n"
        + (workspace_path / "AGENTS.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    # Snapshot proxy capture byte offset and Prometheus metrics before Codex.
    proxy_capture = DEFAULT_PROXY_CAPTURE
    proxy_offset_before = (
        proxy_capture.stat().st_size if proxy_capture.is_file() else 0
    )
    if fixed32_bracket is not None:
        fixed32_bracket.pre(metrics_pre)
    else:
        metrics_pre.write_text(_metrics_snapshot(DEFAULT_METRICS_URL), encoding="utf-8")

    # Start the DCGM/NVML sampler in parallel with the Codex agent.
    dcgm_proc = _spawn_dcgm_sampler(dcgm_samples)

    try:
        task_auth_before = None
        if fixed32_bracket is not None:
            if task_bearer is None or task_key_id is None:
                raise Fixed32BoundaryError("fixed32 task credential is unavailable")
            task_auth_before = _fixed32_task_auth_evidence(
                endpoint=(
                    AGENT_ENDPOINT
                    if AGENT_HOST and AGENT_ENDPOINT
                    else endpoint
                ),
                task_bearer=task_bearer,
                task_key_id=task_key_id,
            )
        codex_meta = _run_agent_dispatch(
            workspace=workspace_path,
            endpoint=endpoint,
            model=model,
            timeout_s=agent_wall_s,
            instance_id=instance_id,
            base_commit=instance["base_commit"],
            stdout_path=qwen_stdout,
            stderr_path=qwen_stderr,
            trace_path=qwen_trace,
            task_bearer=task_bearer,
        )
        if fixed32_bracket is not None:
            if task_bearer is None or task_key_id is None or task_auth_before is None:
                raise Fixed32BoundaryError("fixed32 task-auth state is unavailable")
            task_auth_after = _fixed32_task_auth_evidence(
                endpoint=(
                    AGENT_ENDPOINT
                    if AGENT_HOST and AGENT_ENDPOINT
                    else endpoint
                ),
                task_bearer=task_bearer,
                task_key_id=task_key_id,
            )
            fixed32_initial_provenance_args = {
                "instance_id": instance_id,
                "trace_path": qwen_trace,
                "agent_meta": codex_meta,
                "task_key_id": task_key_id,
                "task_auth_before": task_auth_before,
                "task_auth_after": task_auth_after,
            }
        summary["agent"] = codex_meta
        # Backward-compat: keep the legacy "codex" key so existing reducers
        # (meta.get("codex") in fr13_bigdenom_swe_serve.sh, fr13_standard_metrics.py)
        # still resolve. New reducers read meta.get("agent") or meta.get("codex").
        summary["codex"] = codex_meta
    except Fixed32BoundaryError:
        raise
    except Exception as exc:
        if fixed32_bracket is not None:
            raise Fixed32BoundaryError(
                f"fixed32 agent dispatch failed for {instance_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        raise
    finally:
        _stop_dcgm_sampler(dcgm_proc)

    patch_text = ""
    try:
        patch_text = _extract_patch(cache_path, workspace_path, instance["base_commit"])
    except Exception as exc:  # noqa: BLE001
        summary["patch_extract_error"] = f"{type(exc).__name__}: {exc}"

    # Bundle B #2/#3/#8: if the first attempt left no patch, classify why and
    # re-launch codex ONCE with a state-conditional directive prompt. Bounded
    # to a single retry to cap the wall-time premium.
    if not patch_text.strip():
        cause = _classify_empty_patch_cause(qwen_trace)
        # Bundle B #2/#3/#8 -> in-context continuation: an agent_gave_up attempt EXPLORED
        # then emitted a tool-call-free terminal reply (general temp-0.6 codex flake). Re-drive
        # it up to SWE_EMPTY_PATCH_RETRIES times with its OWN accumulated context + a hard
        # must-act directive (a plain fresh re-roll re-confirms the stop ~88% of the time).
        # DEFAULT 0 = NUDGE-ONLY (project decision, user 2026-07-01): the new-context retry (a
        # FRESH codex session with only a brief of the prior attempt) is REMOVED by default. It
        # predates and is SUPERSEDED by the in-session nudge (LUMO_PROXY_AUTO_CONTINUE_MESSAGE),
        # which recovers empty-patch give-ups IN-SESSION with FULL context preserved — strictly
        # better, because the clean-context restart "empirically breaks the explain-instead-of-act
        # stall" (relaunch_proxy_remote.sh). Running both was redundant + the retry is ~3x slower on
        # hard tasks + loses context. Set SWE_EMPTY_PATCH_RETRIES>=1 ONLY to re-enable the legacy
        # fresh-session retry (small tail-resolve gain at a big wall-time cost).
        if fixed32_bracket is not None:
            _validate_fixed32_retry_policy()
            max_retries = 0
        else:
            max_retries = max(
                0,
                int(os.environ.get("SWE_EMPTY_PATCH_RETRIES", "0")),
            )
        summary["empty_patch_retry"] = {"cause": cause, "attempted": True,
                                        "max_retries": max_retries, "recovered_patch_bytes": 0}
        prev_trace = qwen_trace
        for ridx in range(1, max_retries + 1):
            if cause == "setup_loop":
                retry_prompt = RETRY_PROMPT_SETUP_LOOP
            else:
                retry_prompt = _retry_prompt_continue(_prior_attempt_brief(prev_trace))
            suffix = "" if ridx == 1 else str(ridx)
            print(f"[{_iso_now()}]    {instance_id}: empty patch ({cause}); retry {ridx}/{max_retries}",
                  flush=True)
            retry_trace = task_dir / f"qwen_trace_retry{suffix}.jsonl"
            retry_stderr = task_dir / f"qwen_stderr_retry{suffix}.log"
            retry_stdout = task_dir / f"qwen_stdout_retry{suffix}.log"
            retry_dcgm = _spawn_dcgm_sampler(task_dir / f"dcgm_samples_retry{suffix}.jsonl")
            try:
                retry_auth_before = None
                if fixed32_bracket is not None:
                    if task_bearer is None or task_key_id is None:
                        raise Fixed32BoundaryError(
                            "fixed32 retry task credential is unavailable"
                        )
                    retry_auth_before = _fixed32_task_auth_evidence(
                        endpoint=(
                            AGENT_ENDPOINT
                            if AGENT_HOST and AGENT_ENDPOINT
                            else endpoint
                        ),
                        task_bearer=task_bearer,
                        task_key_id=task_key_id,
                    )
                codex_meta_retry = _run_agent_dispatch(
                    workspace=workspace_path, endpoint=endpoint, model=model,
                    timeout_s=agent_wall_s, instance_id=instance_id,
                    base_commit=instance["base_commit"],
                    stdout_path=retry_stdout, stderr_path=retry_stderr, trace_path=retry_trace,
                    prompt=retry_prompt,
                    task_bearer=task_bearer,
                )
                if fixed32_bracket is not None:
                    if (
                        task_bearer is None
                        or task_key_id is None
                        or retry_auth_before is None
                    ):
                        raise Fixed32BoundaryError(
                            "fixed32 retry task-auth state is unavailable"
                        )
                    retry_auth_after = _fixed32_task_auth_evidence(
                        endpoint=(
                            AGENT_ENDPOINT
                            if AGENT_HOST and AGENT_ENDPOINT
                            else endpoint
                        ),
                        task_bearer=task_bearer,
                        task_key_id=task_key_id,
                    )
                    retry_provenance = _fixed32_real_task_provenance(
                        instance_id=instance_id,
                        trace_path=retry_trace,
                        agent_meta=codex_meta_retry,
                        task_key_id=task_key_id,
                        task_auth_before=retry_auth_before,
                        task_auth_after=retry_auth_after,
                    )
            except Fixed32BoundaryError:
                raise
            except Exception as exc:  # noqa: BLE001 — non-fixed32 retries remain best-effort
                if fixed32_bracket is not None:
                    raise Fixed32BoundaryError(
                        f"fixed32 retry agent dispatch failed for {instance_id}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                summary[f"codex_retry{suffix}_dispatch_error"] = f"{type(exc).__name__}: {exc}"
                print(f"[{_iso_now()}]    {instance_id}: retry {ridx} dispatch failed "
                      f"({type(exc).__name__}: {exc}); keeping empty patch as failed", flush=True)
                break
            finally:
                _stop_dcgm_sampler(retry_dcgm)
            summary[f"codex_retry{suffix}"] = codex_meta_retry
            if fixed32_bracket is not None:
                summary.setdefault("fixed32_real_task_retry_provenance", []).append(
                    retry_provenance
                )
            try:
                retry_patch = _extract_patch(cache_path, workspace_path, instance["base_commit"])
            except Exception as exc:  # noqa: BLE001
                retry_patch = ""
                summary[f"patch_extract_error_retry{suffix}"] = f"{type(exc).__name__}: {exc}"
            prev_trace = retry_trace
            if retry_patch.strip():
                patch_text = retry_patch
                summary["empty_patch_retry"]["recovered_patch_bytes"] = len(retry_patch)
                summary["empty_patch_retry"]["attempts_used"] = ridx
                print(f"[{_iso_now()}]    {instance_id}: retry {ridx} produced {len(retry_patch)}B patch",
                      flush=True)
                break
        else:
            summary["empty_patch_retry"]["attempts_used"] = max_retries

    if fixed32_bracket is not None:
        summary["fixed32_task_boundary"] = fixed32_bracket.post(metrics_post)
        if fixed32_initial_provenance_args is None:
            raise Fixed32BoundaryError(
                "fixed32 initial task provenance inputs are unavailable"
            )
        if fixed32_campaign_scope:
            summary[_FIXED32_CAMPAIGN_RUNTIME_ARGS_KEY] = {
                **fixed32_initial_provenance_args,
                "metrics_pre_path": metrics_pre,
                "metrics_post_path": metrics_post,
            }
        else:
            summary["fixed32_real_task_provenance"] = (
                _fixed32_real_task_provenance(
                    **fixed32_initial_provenance_args,
                    metrics_pre_path=metrics_pre,
                    metrics_post_path=metrics_post,
                )
            )
    else:
        metrics_post.write_text(_metrics_snapshot(DEFAULT_METRICS_URL), encoding="utf-8")

    # Slice the new proxy rows into a per-task file (matches Track B layout).
    # This must happen after the optional empty-patch retry so retry traffic is
    # included in the task's request-metrics evidence.
    per_task_metrics = task_dir / "vllm_request_metrics.jsonl"
    proxy_row_count = 0
    try:
        if proxy_capture.is_file():
            with proxy_capture.open("rb") as src:
                src.seek(proxy_offset_before)
                payload = src.read()
            per_task_metrics.write_bytes(payload)
            summary["vllm_request_metrics_bytes"] = len(payload)
            proxy_row_count = payload.count(b"\n")
        else:
            per_task_metrics.write_bytes(b"")
            summary["vllm_request_metrics_bytes"] = 0
            summary["vllm_request_metrics_warning"] = (
                "proxy capture file not present; verbose request metrics unavailable"
            )
    except Exception as exc:  # noqa: BLE001
        summary["vllm_request_metrics_error"] = f"{type(exc).__name__}: {exc}"

    # Write a simple vllm_per_turn.json (delta of pre/post snapshot sizes
    # plus the per-task proxy row count). This is a smaller artifact than
    # Track B's full Prometheus per-request normalization but preserves the
    # filename + role in the artifact set.
    try:
        per_turn_json.write_text(
            json.dumps(
                {
                    "schema": "lumo.swe_bench_q36_a.vllm_per_turn.v1",
                    "instance_id": instance_id,
                    "metrics_pre_bytes": metrics_pre.stat().st_size,
                    "metrics_post_bytes": metrics_post.stat().st_size,
                    "proxy_request_rows": proxy_row_count,
                    "model_id": model,
                    "endpoint": endpoint,
                    "metrics_url": DEFAULT_METRICS_URL,
                    "deferred_full_normalization": True,
                    "deferred_reason": (
                        "SWE-Bench campaign captures raw pre/post snapshots and the "
                        "proxy capture slice; full per-request Prometheus normalization "
                        "deferred to LLD-12 / closeout aggregation."
                    ),
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        summary["vllm_per_turn_error"] = f"{type(exc).__name__}: {exc}"

    patch_path.write_text(patch_text, encoding="utf-8")
    summary["patch_bytes"] = len(patch_text)

    # Always run the evaluator -- the CLI handles empty-patch -> exit 1.
    eval_meta = _run_eval(
        instance_id=instance_id,
        patch_path=patch_path,
        output_dir=eval_output,
        dataset_name=dataset_name,
        model_name=model_name,
        timeout_s=eval_timeout_s,
        eval_log_path=eval_log,
    )
    summary["eval"] = eval_meta

    eval_report_path = eval_output / "eval_report.json"
    if eval_report_path.is_file():
        try:
            summary["eval_report"] = json.loads(eval_report_path.read_text())
        except Exception:  # noqa: BLE001
            pass

    # Tear down the worktree to free disk; preserve patch and artifacts.
    _remove_workspace(cache_path, workspace_path)

    summary["ended_at"] = _iso_now()
    if fixed32_campaign_scope:
        pending_summary = {
            key: value
            for key, value in summary.items()
            if key != _FIXED32_CAMPAIGN_RUNTIME_ARGS_KEY
        }
        pending_runner_meta_path.write_text(
            json.dumps(pending_summary, indent=2),
            encoding="utf-8",
        )
    else:
        runner_meta_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
    return summary


def _finalize_fixed32_qwen_campaign_provenance(
    *,
    summaries: list[dict[str, Any]],
    instance_ids: list[str],
    dataset_out: Path,
    per_task_root: Path,
    campaign_metrics_pre_path: Path,
    campaign_metrics_post_path: Path,
) -> dict[str, Any]:
    """Prove one B4 union window before publishing any task provenance."""
    if len(instance_ids) < 2 or len(summaries) != len(instance_ids):
        raise Fixed32BoundaryError(
            "fixed32 B4 campaign finalization has an incomplete task set"
        )
    summaries_by_id = {
        summary.get("instance_id"): summary for summary in summaries
    }
    if (
        len(summaries_by_id) != len(instance_ids)
        or set(summaries_by_id) != set(instance_ids)
    ):
        raise Fixed32BoundaryError(
            "fixed32 B4 campaign finalization task identities differ"
        )

    records: list[dict[str, Any]] = []
    for instance_id in instance_ids:
        summary = summaries_by_id[instance_id]
        runtime_args = summary.get(_FIXED32_CAMPAIGN_RUNTIME_ARGS_KEY)
        boundary = summary.get("fixed32_task_boundary")
        task_dir = (per_task_root / instance_id).resolve()
        pending_path = task_dir / _FIXED32_PENDING_RUNNER_METADATA_FILENAME
        runner_path = task_dir / "runner_metadata.json"
        if (
            not isinstance(runtime_args, dict)
            or not isinstance(boundary, dict)
            or boundary.get("instance_id") != instance_id
            or runner_path.exists()
            or not pending_path.is_file()
            or pending_path.is_symlink()
        ):
            raise Fixed32BoundaryError(
                f"fixed32 B4 campaign task {instance_id} is not unpublished and complete"
            )
        try:
            pending_summary = json.loads(
                pending_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Fixed32BoundaryError(
                f"fixed32 B4 pending metadata {pending_path} is invalid: {exc}"
            ) from exc
        expected_pending = {
            key: value
            for key, value in summary.items()
            if key != _FIXED32_CAMPAIGN_RUNTIME_ARGS_KEY
        }
        if pending_summary != expected_pending:
            raise Fixed32BoundaryError(
                f"fixed32 B4 pending metadata differs for {instance_id}"
            )

        task_key_id = runtime_args.get("task_key_id")
        task_auth_before = runtime_args.get("task_auth_before")
        task_auth_after = runtime_args.get("task_auth_after")
        if (
            not isinstance(task_key_id, str)
            or not isinstance(task_auth_before, dict)
            or not isinstance(task_auth_after, dict)
            or task_auth_before.get("task_key_id") != task_key_id
            or task_auth_after.get("task_key_id") != task_key_id
        ):
            raise Fixed32BoundaryError(
                f"fixed32 B4 task-auth binding is invalid for {instance_id}"
            )
        completed_before = task_auth_before.get(
            "completed_logical_model_requests"
        )
        completed_after = task_auth_after.get(
            "completed_logical_model_requests"
        )
        if (
            type(completed_before) is not int
            or type(completed_after) is not int
            or completed_before < 0
            or completed_after <= completed_before
        ):
            raise Fixed32BoundaryError(
                f"fixed32 B4 task-auth request count is invalid for {instance_id}"
            )
        trace_path = runtime_args.get("trace_path")
        metrics_pre_path = runtime_args.get("metrics_pre_path")
        metrics_post_path = runtime_args.get("metrics_post_path")
        if (
            trace_path != task_dir / "qwen_trace.jsonl"
            or metrics_pre_path != task_dir / "vllm_metrics_pre.txt"
            or metrics_post_path != task_dir / "vllm_metrics_post.txt"
        ):
            raise Fixed32BoundaryError(
                f"fixed32 B4 task artifact paths differ for {instance_id}"
            )
        boundary_schema = boundary.get("schema")
        start: int | None = None
        end: int | None = None
        pre_generation: int | None = None
        post_generation: int | None = None
        if boundary_schema == "fr13-fixed32-task-boundary-v1":
            pre = boundary.get("pre")
            post = boundary.get("post")
            interval = boundary.get("forward_step_interval")
            if (
                not isinstance(pre, dict)
                or not isinstance(post, dict)
                or not isinstance(interval, dict)
                or type(pre.get("generation")) is not int
                or type(post.get("generation")) is not int
                or pre["generation"] < 1
                or post["generation"] <= pre["generation"]
                or not isinstance(pre.get("counters"), dict)
                or not isinstance(post.get("counters"), dict)
            ):
                raise Fixed32BoundaryError(
                    f"fixed32 B4 boundary endpoints are invalid for {instance_id}"
                )
            start = pre["counters"].get("pure_decode_forward_steps")
            end = post["counters"].get("pure_decode_forward_steps")
            if (
                type(start) is not int
                or type(end) is not int
                or start < 0
                or end <= start
                or interval
                != {
                    "start_forward_step": start,
                    "end_forward_step": end,
                    "expected_complete_events": end - start,
                }
            ):
                raise Fixed32BoundaryError(
                    f"fixed32 B4 forward interval is invalid for {instance_id}"
                )
            pre_generation = pre["generation"]
            post_generation = post["generation"]
        elif (
            boundary_schema
            == "fr13-fixed32-eager-kernel-diagnostic-task-bracket-v1"
        ):
            pre_metrics = boundary.get("pre_metrics")
            post_metrics = boundary.get("post_metrics")
            if (
                boundary.get("run_classification")
                != "eager_kernel_byte_diagnostic"
                or boundary.get("acceptance_valid") is not False
                or boundary.get("flush_protocol_used") is not False
                or not isinstance(pre_metrics, dict)
                or not isinstance(post_metrics, dict)
            ):
                raise Fixed32BoundaryError(
                    f"fixed32 B4 eager boundary is invalid for {instance_id}"
                )
            for label, path, reference in (
                ("pre", metrics_pre_path, pre_metrics),
                ("post", metrics_post_path, post_metrics),
            ):
                if not path.is_file() or path.is_symlink():
                    raise Fixed32BoundaryError(
                        f"fixed32 B4 eager {label} metrics are missing for "
                        f"{instance_id}"
                    )
                raw = path.read_bytes()
                if reference != _fixed32_artifact_identity(path, raw):
                    raise Fixed32BoundaryError(
                        f"fixed32 B4 eager {label} metric identity differs for "
                        f"{instance_id}"
                    )
        else:
            raise Fixed32BoundaryError(
                f"fixed32 B4 boundary schema is unsupported for {instance_id}"
            )
        raw_trace, events = _fixed32_load_trace_events(
            trace_path,
            instance_id=instance_id,
        )
        records.append(
            {
                "instance_id": instance_id,
                "summary": summary,
                "runtime_args": runtime_args,
                "task_dir": task_dir,
                "pending_path": pending_path,
                "runner_path": runner_path,
                "boundary": boundary,
                "boundary_schema": boundary_schema,
                "start": start,
                "end": end,
                "pre_generation": pre_generation,
                "post_generation": post_generation,
                "task_key_id": task_key_id,
                "completed": completed_after - completed_before,
                "trace_path": trace_path,
                "trace_raw": raw_trace,
                "events": events,
                "metrics_pre_path": metrics_pre_path,
                "metrics_post_path": metrics_post_path,
            }
        )

    boundary_schemas = {record["boundary_schema"] for record in records}
    if len(boundary_schemas) != 1:
        raise Fixed32BoundaryError(
            "fixed32 B4 campaign mixes task-boundary schemas"
        )
    boundary_schema = next(iter(boundary_schemas))
    stream_coverage: dict[str, Any] | None = None
    if boundary_schema == "fr13-fixed32-task-boundary-v1":
        merged: list[list[int]] = []
        graph_records = sorted(
            records,
            key=lambda item: (item["start"], item["end"]),
        )
        for record in graph_records:
            start = record["start"]
            end = record["end"]
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        complete_stream_steps = max(record["end"] for record in records)
        if merged != [[0, complete_stream_steps]]:
            raise Fixed32BoundaryError(
                "fixed32 B4 task intervals do not cover the complete campaign "
                "stream"
            )
        stream_coverage = {
            "start_forward_step": 0,
            "end_forward_step": complete_stream_steps,
            "complete_stream_forward_steps": complete_stream_steps,
        }

    for label, path in (
        ("pre", campaign_metrics_pre_path),
        ("post", campaign_metrics_post_path),
    ):
        if not path.is_file() or path.is_symlink():
            raise Fixed32BoundaryError(
                f"fixed32 B4 campaign {label} metrics are missing or symlinked"
            )
    metrics_pre_raw = campaign_metrics_pre_path.read_bytes()
    metrics_post_raw = campaign_metrics_post_path.read_bytes()
    if not metrics_pre_raw or not metrics_post_raw:
        raise Fixed32BoundaryError(
            "fixed32 B4 campaign endpoint metrics are empty"
        )
    campaign_tasks = [
        {
            "instance_id": record["instance_id"],
            "expected_session_id": fixed32_contract.fixed32_trace_session_id(
                record["instance_id"]
            ),
            "expected_completed_logical_model_requests": record["completed"],
            "events": record["events"],
        }
        for record in records
    ]
    try:
        reconciliation = (
            fixed32_contract.validate_fixed32_qwen_campaign_metrics(
                campaign_tasks,
                metrics_pre=metrics_pre_raw,
                metrics_post=metrics_post_raw,
            )
        )
    except fixed32_contract.ContractError as exc:
        raise Fixed32BoundaryError(
            f"fixed32 B4 campaign metrics do not reconcile: {exc}"
        ) from exc

    proof_path = (
        dataset_out / _FIXED32_QWEN_CAMPAIGN_PROOF_FILENAME
    ).resolve()
    proof = {
        "schema": _FIXED32_QWEN_CAMPAIGN_PROOF_SCHEMA,
        "metric_scope": "concurrent_campaign_union",
        "concurrency": 4,
        "task_ids": list(instance_ids),
        "selection": {
            "basis": "runner_owned_campaign_endpoint_metrics",
            "task_boundary_schema": boundary_schema,
            "task_stream_coverage": stream_coverage,
        },
        "metrics_pre": _fixed32_artifact_identity(
            campaign_metrics_pre_path,
            metrics_pre_raw,
        ),
        "metrics_post": _fixed32_artifact_identity(
            campaign_metrics_post_path,
            metrics_post_raw,
        ),
        "tasks": [
            {
                "instance_id": record["instance_id"],
                "task_key_id": record["task_key_id"],
                "expected_completed_logical_model_requests": record[
                    "completed"
                ],
                "trace": _fixed32_artifact_identity(
                    record["trace_path"],
                    record["trace_raw"],
                ),
            }
            for record in records
        ],
        "metric_evidence_sha256": reconciliation[
            "metric_evidence_sha256"
        ],
        "metric_evidence": reconciliation["metric_evidence"],
    }
    proof_raw = (
        json.dumps(
            proof,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    proof_identity = _fixed32_artifact_identity(proof_path, proof_raw)
    campaign_binding = {
        "artifact": proof_identity,
        "metric_evidence_sha256": reconciliation[
            "metric_evidence_sha256"
        ],
    }

    finalized: dict[str, dict[str, Any]] = {}
    for record in records:
        runtime_args = {
            key: value
            for key, value in record["runtime_args"].items()
            if key not in {"metrics_pre_path", "metrics_post_path"}
        }
        provenance = _fixed32_real_task_provenance(
            **runtime_args,
            campaign_trace_requests=reconciliation["tasks"][
                record["instance_id"]
            ],
            campaign_metric_binding=campaign_binding,
        )
        finalized[record["instance_id"]] = {
            key: value
            for key, value in record["summary"].items()
            if key != _FIXED32_CAMPAIGN_RUNTIME_ARGS_KEY
        }
        finalized[record["instance_id"]][
            "fixed32_real_task_provenance"
        ] = provenance
        finalized[record["instance_id"]][
            "fixed32_qwen_campaign_proof"
        ] = proof_identity

    proof_tmp = proof_path.with_suffix(".json.tmp")
    proof_tmp.write_bytes(proof_raw)
    metadata_temps: dict[str, Path] = {}
    for record in records:
        runner_path = record["runner_path"]
        temporary = runner_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                finalized[record["instance_id"]],
                indent=2,
            ),
            encoding="utf-8",
        )
        metadata_temps[record["instance_id"]] = temporary
    os.replace(proof_tmp, proof_path)
    for record in records:
        instance_id = record["instance_id"]
        os.replace(metadata_temps[instance_id], record["runner_path"])
        record["pending_path"].unlink()
        record["summary"].pop(_FIXED32_CAMPAIGN_RUNTIME_ARGS_KEY, None)
        record["summary"].update(finalized[instance_id])
    return proof


def _aggregate(per_task_root: Path, summary_path: Path, predictions_path: Path,
               started_at: str, ended_at: str, model_name: str) -> dict[str, Any]:
    instance_summaries: list[dict[str, Any]] = []
    verdict_counter: Counter = Counter()
    failure_mode_counter: Counter = Counter()
    repo_counter: Counter = Counter()
    repo_pass_counter: Counter = Counter()
    eval_wall: list[float] = []
    codex_wall: list[float] = []
    predictions_lines: list[str] = []
    for task_dir in sorted(p for p in per_task_root.iterdir() if p.is_dir()):
        meta_path = task_dir / "runner_metadata.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text())
        instance_summaries.append(meta)
        verdict = (meta.get("eval_report") or {}).get("verdict", "missing")
        failure_mode = (meta.get("eval_report") or {}).get("failure_mode", "missing")
        verdict_counter[verdict] += 1
        failure_mode_counter[failure_mode] += 1
        repo = meta.get("repo") or "unknown"
        repo_counter[repo] += 1
        if verdict == "resolved":
            repo_pass_counter[repo] += 1
        if (meta.get("eval_report") or {}).get("eval_wall_clock_seconds") is not None:
            eval_wall.append(float(meta["eval_report"]["eval_wall_clock_seconds"]))
        _agent_block = meta.get("agent") or meta.get("codex") or {}
        if _agent_block.get("elapsed_s") is not None:
            codex_wall.append(float(_agent_block["elapsed_s"]))
        pred_file = task_dir / "eval" / "predictions.jsonl"
        if pred_file.is_file():
            predictions_lines.extend(
                line for line in pred_file.read_text().splitlines() if line.strip()
            )

    def _percentiles(xs: list[float]) -> dict[str, float]:
        if not xs:
            return {}
        xs = sorted(xs)
        def _pct(p: float) -> float:
            i = max(0, min(len(xs) - 1, int(round(p * (len(xs) - 1)))))
            return round(xs[i], 3)
        return {"p50": _pct(0.5), "p90": _pct(0.9), "p99": _pct(0.99),
                "min": round(min(xs), 3), "max": round(max(xs), 3)}

    summary = {
        "model_name_or_path": model_name,
        "started_at": started_at,
        "ended_at": ended_at,
        "instances_total": len(instance_summaries),
        "verdict_counts": dict(verdict_counter),
        "failure_mode_counts": dict(failure_mode_counter),
        "per_repo_total": dict(repo_counter),
        "per_repo_resolved": dict(repo_pass_counter),
        "eval_wall_seconds": _percentiles(eval_wall),
        "codex_wall_seconds": _percentiles(codex_wall),
        "resolved_rate": (
            round(verdict_counter["resolved"] / len(instance_summaries), 4)
            if instance_summaries else None
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    predictions_path.write_text("\n".join(predictions_lines) + ("\n" if predictions_lines else ""))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True,
                        help="JSON subset emitted by build_swe_bench_subset.py")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--dataset-tag", default=None,
                        help="Override the per-dataset subdirectory name "
                             "(default: 'verified' or 'pro' inferred from subset).")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME_TAG)
    parser.add_argument("--agent-wall-s", type=int,
                        default=int(os.environ.get("SWE_AGENT_WALL_S", str(DEFAULT_AGENT_WALL_S))))
    parser.add_argument("--eval-timeout-s", type=int, default=DEFAULT_EVAL_TIMEOUT_S)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Agent concurrency. LLD-05 §4.6 default is 1; raise after Sprint-1 validation.")
    parser.add_argument("--repo-cache", type=Path, default=DEFAULT_REPO_CACHE)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N instances (for smoke runs).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip instances whose runner_metadata.json is already on disk.")
    parser.add_argument("--eval-host", default=None,
                        help="SSH host to offload the eval step to (native x86_64). When set, "
                             "the agent runs locally on arm64 but eval runs on the x86 box — "
                             "recovers old-python instances + keeps the dataset single-arch.")
    parser.add_argument("--agent-host", "--codex-host", dest="agent_host", default=None,
                        help="SSH host to offload the SWE agent Docker to (native x86_64, "
                             "e.g. alienware). When set, the agent loop runs on the x86 box "
                             "(workspace rsynced there + back) so the GB10 runs ONLY vLLM and the "
                             "unified-memory bandwidth is uncontended = clean deploy-speed numbers. "
                             "--codex-host is a deprecated alias kept so queued serve scripts parse.")
    parser.add_argument("--agent-endpoint", "--codex-endpoint", dest="agent_endpoint", default=None,
                        help="The agent docker on --agent-host hits this (alienware-local) proxy "
                             "endpoint (default http://127.0.0.1:8023/v1). Distinct from --endpoint "
                             "(the on-GB10 legacy proxy) because 8022 is taken on alienware. "
                             "--codex-endpoint is a deprecated alias.")
    parser.add_argument("--fixed32-container")
    parser.add_argument("--fixed32-producer-pid", type=int)
    parser.add_argument(
        "--fixed32-mode",
        choices=("tail6_fixed32", "hydra27_fixed32"),
    )
    parser.add_argument("--fixed32-flush-request", type=Path)
    parser.add_argument("--fixed32-flush-ack", type=Path)
    parser.add_argument("--fixed32-boundary-snapshot", type=Path)
    parser.add_argument(
        "--fixed32-taw-real-event-arm",
        type=Path,
        help=(
            "Host path mounted at the diagnostic kernel's exact /logs "
            "real-event marker path. B1 or exact-B4 TAW diagnostic only."
        ),
    )
    parser.add_argument(
        "--fixed32-bm8-real-event-arm",
        type=Path,
        help=(
            "Host path mounted at the BM8 diagnostic kernel's exact /logs "
            "real-event marker path. B1 diagnostic only."
        ),
    )
    parser.add_argument(
        "--fixed32-cutlass-real-event-arm",
        type=Path,
        help=(
            "Host path mounted at the CUTLASS Stream-K diagnostic's exact "
            "/logs real-event marker path. B1 diagnostic only."
        ),
    )
    parser.add_argument(
        "--fixed32-committer-layer-batch-real-event-arm",
        type=Path,
        help=(
            "Host path mounted at the CFWD layer-batch qualification kernel's "
            "exact /logs real-event marker path. B1 task or canonical exact4/16 "
            "B4 campaign qualification only."
        ),
    )
    args = parser.parse_args(argv)

    fixed32_values = (
        args.fixed32_container,
        args.fixed32_producer_pid,
        args.fixed32_mode,
        args.fixed32_flush_request,
        args.fixed32_flush_ack,
        args.fixed32_boundary_snapshot,
    )
    fixed32_enabled = any(value is not None for value in fixed32_values)
    if fixed32_enabled and any(value is None for value in fixed32_values):
        parser.error("all six --fixed32-* runtime binding options are required together")
    diagnostic_text = os.environ.get("FR13_FIXED32_B1_DIAGNOSTIC", "0")
    if diagnostic_text not in {"0", "1"}:
        parser.error("FR13_FIXED32_B1_DIAGNOSTIC must be exactly 0 or 1")
    fixed32_b1_diagnostic = diagnostic_text == "1"
    if fixed32_b1_diagnostic and not fixed32_enabled:
        parser.error("FR13_FIXED32_B1_DIAGNOSTIC=1 requires fixed32 runtime binding")
    cfwd_qualification_text = os.environ.get(
        "FR13_FIXED32_COMMITTER_LAYER_BATCH_QUALIFICATION",
        "0",
    )
    if cfwd_qualification_text not in {"0", "1"}:
        parser.error(
            "FR13_FIXED32_COMMITTER_LAYER_BATCH_QUALIFICATION must be exactly 0 or 1"
        )
    fixed32_cfwd_qualification = cfwd_qualification_text == "1"
    taw_diagnostic_text = os.environ.get(
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE",
        "0",
    )
    if taw_diagnostic_text not in {"0", "1"}:
        parser.error(
            "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE must be exactly 0 or 1"
        )
    fixed32_taw_diagnostic = taw_diagnostic_text == "1"
    if fixed32_taw_diagnostic:
        if not fixed32_enabled:
            parser.error(
                "fixed32 TAW native real-task arm requires fixed32 runtime binding"
            )
        if args.fixed32_taw_real_event_arm is None:
            parser.error(
                "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=1 requires "
                "--fixed32-taw-real-event-arm"
            )
        if (
            args.fixed32_flush_request is not None
            and args.fixed32_taw_real_event_arm.parent
            != args.fixed32_flush_request.parent
        ):
            parser.error(
                "--fixed32-taw-real-event-arm must share the mounted fixed32 "
                "logs directory"
            )
    elif args.fixed32_taw_real_event_arm is not None:
        parser.error(
            "--fixed32-taw-real-event-arm requires "
            "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=1"
        )
    bm8_diagnostic_text = os.environ.get(
        "FR13_DFWD_UNIFIED_BM8_LIVE_AB",
        "0",
    )
    if bm8_diagnostic_text not in {"0", "1"}:
        parser.error("FR13_DFWD_UNIFIED_BM8_LIVE_AB must be exactly 0 or 1")
    fixed32_bm8_diagnostic = bm8_diagnostic_text == "1"
    if fixed32_bm8_diagnostic:
        if fixed32_taw_diagnostic:
            parser.error("fixed32 TAW and BM8 real-task diagnostics are exclusive")
        if not fixed32_enabled or not fixed32_b1_diagnostic:
            parser.error(
                "fixed32 BM8 real-task arm requires fixed32 B1 diagnostic mode"
            )
        if args.fixed32_bm8_real_event_arm is None:
            parser.error(
                "FR13_DFWD_UNIFIED_BM8_LIVE_AB=1 requires "
                "--fixed32-bm8-real-event-arm"
            )
        if (
            args.fixed32_flush_request is not None
            and args.fixed32_bm8_real_event_arm.parent
            != args.fixed32_flush_request.parent
        ):
            parser.error(
                "--fixed32-bm8-real-event-arm must share the mounted fixed32 "
                "logs directory"
            )
    elif args.fixed32_bm8_real_event_arm is not None:
        parser.error(
            "--fixed32-bm8-real-event-arm requires "
            "FR13_DFWD_UNIFIED_BM8_LIVE_AB=1"
        )
    cutlass_wave = os.environ.get("FR13_FIXED32_CUTLASS_WAVE", "stock")
    if cutlass_wave not in {
        "stock",
        "streamk_coop128",
        "streamk_coop128_byte_ab",
        "streamk_force_wide256",
        "streamk_force_wide256_byte_ab",
        "static_persistent_stocktile",
        "static_persistent_stocktile_byte_ab",
        "divisor_static_stocktile",
        "divisor_static_stocktile_byte_ab",
        "identity_stage2_static",
        "identity_stage2_static_byte_ab",
        "identity_stage2_pingpong_b1",
        "identity_stage2_pingpong_b1_byte_ab",
        "identity_onen_b1",
        "identity_onen_b1_byte_ab",
        "identity_stockshape_b4",
        "identity_stockshape_b4_byte_ab",
        "identity_stockshape_stage2_b4",
        "identity_stockshape_stage2_b4_byte_ab",
        "identity_twom_b4",
        "identity_twom_b4_byte_ab",
        "identity_hybrid_n5120_b4",
        "identity_hybrid_n5120_b4_byte_ab",
        "identity_divisor_b4",
        "identity_divisor_b4_byte_ab",
        "persistent_b4_m128",
        "persistent_b4_m128_byte_ab",
        "persistent_b4_m128_static",
        "persistent_b4_m128_static_byte_ab",
    }:
        parser.error("FR13_FIXED32_CUTLASS_WAVE has an unsupported value")
    fixed32_cutlass_diagnostic = cutlass_wave in {
        "streamk_coop128_byte_ab",
        "streamk_force_wide256_byte_ab",
        "static_persistent_stocktile_byte_ab",
        "divisor_static_stocktile_byte_ab",
        "identity_stage2_static_byte_ab",
        "identity_stage2_pingpong_b1_byte_ab",
        "identity_onen_b1_byte_ab",
    }
    fixed32_cutlass_b4_diagnostic = cutlass_wave in {
        "identity_divisor_b4_byte_ab",
        "identity_stockshape_b4_byte_ab",
        "identity_stockshape_stage2_b4_byte_ab",
        "identity_twom_b4_byte_ab",
        "identity_hybrid_n5120_b4_byte_ab",
        "persistent_b4_m128_byte_ab",
        "persistent_b4_m128_static_byte_ab",
    }
    batch_gdn_eager_diagnostic = os.environ.get(
        "FR13_FIXED32_BATCH_GDN_BYTE_AB", "0"
    )
    if batch_gdn_eager_diagnostic not in {"0", "1"}:
        parser.error("FR13_FIXED32_BATCH_GDN_BYTE_AB must be exactly 0 or 1")
    treeconv_zero_tail_graph_diagnostic = os.environ.get(
        "FR13_FIXED32_CONV_COMMIT_ZERO_TAIL_BYTE_AB", "0"
    )
    if treeconv_zero_tail_graph_diagnostic not in {"0", "1"}:
        parser.error(
            "FR13_FIXED32_CONV_COMMIT_ZERO_TAIL_BYTE_AB must be exactly 0 or 1"
        )
    sfwd_state_fusion_eager_diagnostic = os.environ.get(
        "FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB", "0"
    )
    if sfwd_state_fusion_eager_diagnostic not in {"0", "1"}:
        parser.error(
            "FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB must be exactly 0 or 1"
        )
    sfwd_state_fusion_timing_text = os.environ.get(
        "FR13_FIXED32_SFWD_STATE_FUSION_TIMING_AB",
        "0",
    )
    if sfwd_state_fusion_timing_text not in {"0", "1"}:
        parser.error(
            "FR13_FIXED32_SFWD_STATE_FUSION_TIMING_AB must be exactly 0 or 1"
        )
    sfwd_prior_reuse_text = os.environ.get(
        "FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB",
        "0",
    )
    if sfwd_prior_reuse_text not in {"0", "1"}:
        parser.error(
            "FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB must be exactly 0 or 1"
        )
    if (
        sum(
            value == "1"
            for value in (
                sfwd_state_fusion_eager_diagnostic,
                sfwd_state_fusion_timing_text,
                sfwd_prior_reuse_text,
            )
        )
        > 1
    ):
        parser.error("fixed32 SFWD diagnostics are exclusive")
    fixed32_eager_kernel_diagnostic = (
        fixed32_cutlass_diagnostic
        or fixed32_cutlass_b4_diagnostic
        or batch_gdn_eager_diagnostic == "1"
        or sfwd_state_fusion_eager_diagnostic == "1"
        or sfwd_state_fusion_timing_text == "1"
        or sfwd_prior_reuse_text == "1"
    )
    if (
        fixed32_eager_kernel_diagnostic
        and os.environ.get("ENFORCE_EAGER", "0") != "1"
    ):
        parser.error("fixed32 eager kernel diagnostic requires ENFORCE_EAGER=1")
    if treeconv_zero_tail_graph_diagnostic == "1":
        if not fixed32_enabled:
            parser.error("tree-conv zero-tail byte diagnostic requires fixed32")
        if os.environ.get("ENFORCE_EAGER", "0") != "0":
            parser.error(
                "tree-conv zero-tail byte diagnostic requires FULL graph mode"
            )
        if os.environ.get("FR13_FIXED32_CONV_COMMIT_ZERO_TAIL", "0") != "0":
            parser.error(
                "tree-conv zero-tail production and byte diagnostic are exclusive"
            )
        if any(
            (
                fixed32_cutlass_diagnostic,
                fixed32_cutlass_b4_diagnostic,
                batch_gdn_eager_diagnostic == "1",
                sfwd_state_fusion_eager_diagnostic == "1",
                sfwd_state_fusion_timing_text == "1",
                sfwd_prior_reuse_text == "1",
                fixed32_taw_diagnostic,
                fixed32_bm8_diagnostic,
                fixed32_cfwd_qualification,
            )
        ):
            parser.error(
                "tree-conv zero-tail byte diagnostic must be the only kernel diagnostic"
            )
    if fixed32_cfwd_qualification:
        if not fixed32_enabled:
            parser.error(
                "fixed32 CFWD layer-batch qualification requires fixed32 runtime"
            )
        if os.environ.get("FR13_FIXED32_COMMITTER_LAYER_BATCH", "0") != "1":
            parser.error(
                "fixed32 CFWD layer-batch qualification requires "
                "FR13_FIXED32_COMMITTER_LAYER_BATCH=1"
            )
        if args.fixed32_committer_layer_batch_real_event_arm is None:
            parser.error(
                "FR13_FIXED32_COMMITTER_LAYER_BATCH_QUALIFICATION=1 requires "
                "--fixed32-committer-layer-batch-real-event-arm"
            )
        if (
            args.fixed32_flush_request is not None
            and args.fixed32_committer_layer_batch_real_event_arm.parent
            != args.fixed32_flush_request.parent
        ):
            parser.error(
                "--fixed32-committer-layer-batch-real-event-arm must share "
                "the mounted fixed32 logs directory"
            )
        if (
            fixed32_taw_diagnostic
            or fixed32_bm8_diagnostic
            or fixed32_eager_kernel_diagnostic
        ):
            parser.error(
                "fixed32 CFWD layer-batch qualification must be the only "
                "real-task kernel diagnostic"
            )
    elif args.fixed32_committer_layer_batch_real_event_arm is not None:
        parser.error(
            "--fixed32-committer-layer-batch-real-event-arm requires "
            "FR13_FIXED32_COMMITTER_LAYER_BATCH_QUALIFICATION=1"
        )
    if fixed32_cutlass_diagnostic:
        if fixed32_taw_diagnostic or fixed32_bm8_diagnostic:
            parser.error(
                "fixed32 CUTLASS, TAW, and BM8 real-task diagnostics are exclusive"
            )
        if not fixed32_enabled or not fixed32_b1_diagnostic:
            parser.error(
                "fixed32 CUTLASS real-task arm requires fixed32 B1 diagnostic mode"
            )
        if args.fixed32_cutlass_real_event_arm is None:
            parser.error(
                "a CUTLASS Stream-K byte diagnostic requires "
                "--fixed32-cutlass-real-event-arm"
            )
        if (
            args.fixed32_flush_request is not None
            and args.fixed32_cutlass_real_event_arm.parent
            != args.fixed32_flush_request.parent
        ):
            parser.error(
                "--fixed32-cutlass-real-event-arm must share the mounted "
                "fixed32 logs directory"
            )
    elif args.fixed32_cutlass_real_event_arm is not None:
        parser.error(
            "--fixed32-cutlass-real-event-arm requires the "
            "CUTLASS Stream-K byte-diagnostic selector"
        )
    if fixed32_cutlass_b4_diagnostic:
        if fixed32_taw_diagnostic or fixed32_bm8_diagnostic:
            parser.error(
                "fixed32 CUTLASS B4, TAW, and BM8 diagnostics are exclusive"
            )
        if not fixed32_enabled or fixed32_b1_diagnostic:
            parser.error(
                "fixed32 CUTLASS B4 diagnostic requires non-B1 fixed32 mode"
            )
    fixed32_client = None
    fixed32_subset = None
    if fixed32_enabled:
        if args.limit is not None or args.skip_existing:
            parser.error("fixed32 campaigns forbid --limit and --skip-existing")
        try:
            _validate_fixed32_agent_runtime_mode(
                remote_host=args.agent_host
            )
            _fixed32_qwen_settings_metadata()
            _inspect_fixed32_agent_placement_remote(args.agent_host)
        except Fixed32BoundaryError as error:
            parser.error(str(error))
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from fr13_floor_gate import GateError as FloorGateError
        from fr13_floor_gate import validate_fixed32_run_subset

        try:
            fixed32_subset = validate_fixed32_run_subset(
                args.subset,
                b1_diagnostic=fixed32_b1_diagnostic,
            )
        except FloorGateError as error:
            parser.error(f"fixed32 run subset validation failed: {error}")
        try:
            serving_batch = int(os.environ["MAX_NUM_SEQS_OVR"])
        except (KeyError, ValueError):
            parser.error("fixed32 requires integer MAX_NUM_SEQS_OVR in the runner environment")
        if fixed32_b1_diagnostic:
            if serving_batch != 1 or args.concurrency != 1:
                parser.error(
                    "fixed32 B1 diagnostic requires concurrency and serving batch exactly 1"
                )
        elif serving_batch not in (1, 4) or args.concurrency != serving_batch:
            parser.error(
                "fixed32 requires concurrency to equal the serving batch (exactly B1 or B4)"
            )
        if fixed32_cfwd_qualification:
            if fixed32_b1_diagnostic:
                if serving_batch != 1 or args.concurrency != 1:
                    parser.error(
                        "fixed32 CFWD B1 qualification requires exact B1"
                    )
            elif serving_batch != 4 or args.concurrency != 4:
                parser.error(
                    "fixed32 CFWD campaign qualification requires exact B4"
                )
        if (
            fixed32_taw_diagnostic
            and not fixed32_b1_diagnostic
            and (serving_batch != 4 or args.concurrency != 4)
        ):
            parser.error(
                "fixed32 TAW native campaign arm requires exact B4 concurrency"
            )
        from fr13_fixed32_flush_protocol import Fixed32FlushClient

        fixed32_client = Fixed32FlushClient(
            container=args.fixed32_container,
            producer_pid=args.fixed32_producer_pid,
            mode=args.fixed32_mode,
            request_path=args.fixed32_flush_request,
            ack_path=args.fixed32_flush_ack,
        )
        try:
            ready = fixed32_client.connect()
            _validate_fixed32_ack(ready, label="ready")
        except Exception as error:
            parser.error(
                f"fixed32 runtime generation-zero binding failed: "
                f"{type(error).__name__}: {error}"
            )

    global EVAL_HOST, AGENT_HOST, AGENT_ENDPOINT
    EVAL_HOST = args.eval_host
    if EVAL_HOST:
        print(f"[eval-offload] eval step -> {EVAL_HOST} (native x86_64)", flush=True)
    AGENT_HOST = args.agent_host
    AGENT_ENDPOINT = args.agent_endpoint or (
        "http://127.0.0.1:8023/v1" if AGENT_HOST else None)
    if AGENT_HOST:
        print(f"[codex-offload] codex docker -> {AGENT_HOST} (x86); endpoint={AGENT_ENDPOINT} "
              f"(GB10 stays vLLM-only — uncontended deploy-speed)", flush=True)

    dataset_name, instance_ids = _load_subset(args.subset)
    if args.limit is not None:
        instance_ids = instance_ids[: args.limit]
    if fixed32_enabled:
        if "SWE-bench_Verified" not in dataset_name:
            parser.error(f"fixed32 dataset is not SWE-bench Verified: {dataset_name!r}")
        if fixed32_subset is None or instance_ids != fixed32_subset["task_ids"]:
            parser.error(
                "fixed32 loaded task set differs from the validated canonical subset"
            )
    dataset_tag = args.dataset_tag or ("pro" if "Pro" in dataset_name else "verified")
    dataset_out = args.out_root / dataset_tag
    per_task_root = dataset_out / "per_task"
    per_task_root.mkdir(parents=True, exist_ok=True)

    print(f"=== [{_iso_now()}] dataset={dataset_name} tag={dataset_tag} n={len(instance_ids)} "
          f"concurrency={args.concurrency} ===", flush=True)
    dataset_records = _load_dataset(
        dataset_name,
        pinned_verified=fixed32_enabled,
    )
    missing = [i for i in instance_ids if i not in dataset_records]
    if missing:
        print(f"WARNING: {len(missing)} subset instances missing from dataset: {missing[:5]}",
              flush=True)
        instance_ids = [i for i in instance_ids if i in dataset_records]

    args.repo_cache.mkdir(parents=True, exist_ok=True)

    # Startup-time worktree GC: prune any stale worktrees registered in
    # cached repos whose checkout directory either no longer exists OR
    # whose parent per_task/<id>/ directory lacks a runner_metadata.json
    # (signature of an aborted run). Without this, orchestrator restarts
    # accumulate stale worktrees that hold 50-300 MB of checkout files
    # each and never get cleaned, eating into the tight ~7-8 GiB host
    # MemAvailable budget on this unified-memory DGX Spark host.
    if args.repo_cache.is_dir():
        for repo_cache in sorted(args.repo_cache.iterdir()):
            if not repo_cache.is_dir():
                continue
            try:
                out = subprocess.run(
                    ["git", "-C", str(repo_cache), "worktree", "list", "--porcelain"],
                    capture_output=True, text=True, check=False, timeout=10,
                )
                if out.returncode != 0:
                    continue
            except Exception:  # noqa: BLE001
                continue
            for line in out.stdout.splitlines():
                if not line.startswith("worktree "):
                    continue
                wt_path = Path(line[len("worktree "):])
                if wt_path == repo_cache:
                    continue  # main checkout
                parent = wt_path.parent
                runner_meta = parent / "runner_metadata.json"
                if not wt_path.exists() or not runner_meta.is_file():
                    print(f"[startup-gc] removing stale worktree: {wt_path}", flush=True)
                    subprocess.run(
                        ["git", "-C", str(repo_cache), "worktree", "remove",
                         "--force", str(wt_path)],
                        capture_output=True, check=False, timeout=15,
                    )
                    if wt_path.exists():
                        shutil.rmtree(wt_path, ignore_errors=True)
            subprocess.run(
                ["git", "-C", str(repo_cache), "worktree", "prune"],
                capture_output=True, check=False, timeout=10,
            )

    started_at = _iso_now()
    summaries: list[dict[str, Any]] = []
    campaign_metrics_pre_path: Path | None = None
    campaign_metrics_post_path: Path | None = None
    if fixed32_enabled and serving_batch == 4:
        campaign_metrics_pre_path = (
            dataset_out / _FIXED32_QWEN_CAMPAIGN_METRICS_PRE_FILENAME
        ).resolve()
        campaign_metrics_post_path = (
            dataset_out / _FIXED32_QWEN_CAMPAIGN_METRICS_POST_FILENAME
        ).resolve()
        _write_fixed32_campaign_metrics(campaign_metrics_pre_path)
    taw_campaign_arm: _Fixed32TawCampaignArm | None = None
    taw_campaign_arm_artifact_path: Path | None = None
    if fixed32_taw_diagnostic and not fixed32_b1_diagnostic:
        if args.fixed32_taw_real_event_arm is None or fixed32_subset is None:
            raise Fixed32BoundaryError(
                "fixed32 TAW B4 campaign arm lacks its validated binding"
            )
        taw_campaign_arm = _Fixed32TawCampaignArm(
            path=args.fixed32_taw_real_event_arm,
            subset_binding=fixed32_subset,
            concurrency=args.concurrency,
        )
        taw_campaign_arm_artifact_path = (
            dataset_out / taw_campaign_arm.artifact_name
        )
    cfwd_campaign_bracket: (
        _Fixed32CfwdB4QualificationCampaignBracket | None
    ) = None
    cfwd_campaign_arm_artifact_path: Path | None = None
    if fixed32_cfwd_qualification and not fixed32_b1_diagnostic:
        if (
            args.fixed32_committer_layer_batch_real_event_arm is None
            or fixed32_subset is None
            or fixed32_client is None
            or args.fixed32_boundary_snapshot is None
        ):
            raise Fixed32BoundaryError(
                "fixed32 CFWD B4 campaign arm lacks its validated binding"
            )
        cfwd_campaign_arm = _Fixed32CommitterLayerBatchCampaignArm(
            path=args.fixed32_committer_layer_batch_real_event_arm,
            subset_binding=fixed32_subset,
            concurrency=args.concurrency,
        )
        cfwd_campaign_arm_artifact_path = (
            dataset_out / cfwd_campaign_arm.artifact_name
        )
        cfwd_campaign_bracket = _Fixed32CfwdB4QualificationCampaignBracket(
            client=fixed32_client,
            boundary_snapshot_base=args.fixed32_boundary_snapshot,
            server_capacity=serving_batch,
            campaign_arm=cfwd_campaign_arm,
            artifact_path=(
                dataset_out / "fixed32_cfwd_b4_qualification_campaign.json"
            ),
            arm_artifact_path=cfwd_campaign_arm_artifact_path,
            metrics_pre_path=(
                dataset_out / "fixed32_cfwd_b4_qualification_metrics_pre.txt"
            ),
            metrics_post_path=(
                dataset_out / "fixed32_cfwd_b4_qualification_metrics_post.txt"
            ),
        )

    def _job(iid: str) -> dict[str, Any]:
        t0 = time.time()
        print(f"[{_iso_now()}] -> {iid}", flush=True)
        taw_real_task_arm = None
        arm_path = (
            args.fixed32_committer_layer_batch_real_event_arm
            if (
                args.fixed32_committer_layer_batch_real_event_arm is not None
                and fixed32_b1_diagnostic
            )
            else (
                (
                    args.fixed32_taw_real_event_arm
                    if taw_campaign_arm is None
                    else None
                )
                if args.fixed32_taw_real_event_arm is not None
                else (
                    args.fixed32_bm8_real_event_arm
                    if args.fixed32_bm8_real_event_arm is not None
                    else args.fixed32_cutlass_real_event_arm
                )
            )
        )
        if arm_path is not None:
            pinned_task_ids = (
                fixed32_subset.get("task_ids")
                if isinstance(fixed32_subset, dict)
                else None
            )
            if pinned_task_ids != [iid]:
                raise Fixed32BoundaryError(
                    "fixed32 kernel real-task arm differs from the pinned "
                    "single-task SWE-Verified binding"
                )
            if args.fixed32_committer_layer_batch_real_event_arm is not None:
                arm_type = _Fixed32CommitterLayerBatchRealTaskArm
            elif args.fixed32_taw_real_event_arm is not None:
                arm_type = _Fixed32TawRealTaskArm
            elif args.fixed32_bm8_real_event_arm is not None:
                arm_type = _Fixed32Bm8RealTaskArm
            else:
                arm_type = _Fixed32CutlassRealTaskArm
            taw_real_task_arm = arm_type(
                path=arm_path,
                instance_id=iid,
            )
        if fixed32_cfwd_qualification:
            fixed32_bracket_type = (
                _Fixed32CfwdQualificationTaskBracket
                if fixed32_b1_diagnostic
                else _Fixed32CfwdB4QualificationMemberTaskBracket
            )
        elif fixed32_eager_kernel_diagnostic:
            fixed32_bracket_type = _Fixed32EagerKernelDiagnosticTaskBracket
        else:
            fixed32_bracket_type = _Fixed32TaskBracket
        fixed32_bracket = (
            fixed32_bracket_type(
                client=fixed32_client,
                task_dir=(per_task_root / iid).resolve(),
                instance_id=iid,
                boundary_snapshot_base=args.fixed32_boundary_snapshot,
                server_capacity=serving_batch,
                taw_real_task_arm=taw_real_task_arm,
            )
            if fixed32_client is not None
            else None
        )
        try:
            try:
                res = _process_one(
                    instance_id=iid,
                    instance=dataset_records[iid],
                    dataset_name=dataset_name,
                    per_task_root=per_task_root,
                    repo_cache_root=args.repo_cache,
                    endpoint=args.endpoint,
                    model=args.model,
                    model_name=args.model_name,
                    agent_wall_s=args.agent_wall_s,
                    eval_timeout_s=args.eval_timeout_s,
                    skip_existing=args.skip_existing,
                    fixed32_bracket=fixed32_bracket,
                    fixed32_b1_diagnostic=fixed32_b1_diagnostic,
                    fixed32_cfwd_qualification=fixed32_cfwd_qualification,
                )
            except Fixed32BoundaryError:
                raise
            except Exception as exc:  # noqa: BLE001
                res = {"instance_id": iid, "status": "orchestrator_crash",
                       "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()}
                # PERSIST the crash (2026-07-06): previously the traceback lived only
                # in this in-memory dict — an orchestrator_crash left NOTHING on disk
                # to diagnose (cost a wasted GPU boot to rediscover via reproduction).
                # Print to the run log AND write a per-task crash file.
                print(f"[{_iso_now()}] !! {iid} orchestrator_crash: {res['error']}\n"
                      f"{res['traceback']}", flush=True)
                try:
                    crash_dir = per_task_root / iid
                    crash_dir.mkdir(parents=True, exist_ok=True)
                    (crash_dir / "orchestrator_crash.json").write_text(
                        json.dumps(res, indent=1), encoding="utf-8")
                except Exception:
                    pass
        finally:
            if (
                fixed32_bracket is not None
                and fixed32_bracket.started
                and not fixed32_bracket.complete
                and not fixed32_bracket.post_attempted
            ):
                fixed32_bracket.post(
                    fixed32_bracket.task_dir / "vllm_metrics_post.txt"
                )
        verdict = (res.get("eval_report") or {}).get("verdict", res.get("status", "?"))
        elapsed = time.time() - t0
        print(f"[{_iso_now()}] <- {iid} verdict={verdict} elapsed_total={elapsed:.1f}s",
              flush=True)
        # FR13 ENDPOINT CIRCUIT BREAKER (2026-07-25, bar17 post-mortem): when
        # the agent's model endpoint dies (offload proxy death), every task
        # "completes" in seconds with an [API Error: fetch failed] single
        # message and a failed verdict — bar17 burned 9 subset tasks in 80s
        # this way. Three consecutive instant-fails => the endpoint is gone:
        # ABORT the campaign loudly instead of torching the rest of the
        # subset. Threshold 30s is far below any honest session (minutes).
        global _FR13_CB_INSTANT_FAILS
        try:
            _FR13_CB_INSTANT_FAILS
        except NameError:
            _FR13_CB_INSTANT_FAILS = 0
        if verdict != "resolved" and elapsed < 30.0:
            _FR13_CB_INSTANT_FAILS += 1
        else:
            _FR13_CB_INSTANT_FAILS = 0
        if _FR13_CB_INSTANT_FAILS >= 3:
            raise SystemExit(
                f"FR13 CIRCUIT BREAKER: {_FR13_CB_INSTANT_FAILS} consecutive "
                f"instant-fail tasks (<30s, last={iid}) — the model endpoint "
                "is dead (offload proxy?). Aborting the campaign; completed "
                "task artifacts are preserved."
            )
        if fixed32_bracket is None or fixed32_bracket.server_capacity == 1:
            _autocommit_task_artifacts(per_task_root / iid, iid)
        return res

    def _run_jobs() -> None:
        if args.concurrency <= 1:
            for iid in instance_ids:
                summaries.append(_job(iid))
        else:
            with _cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                for res in ex.map(_job, instance_ids):
                    summaries.append(res)

    if cfwd_campaign_bracket is not None:
        cfwd_campaign_bracket.run(_run_jobs)
    else:
        _run_with_fixed32_taw_campaign_arm(
            arm=taw_campaign_arm,
            artifact_path=taw_campaign_arm_artifact_path,
            action=_run_jobs,
        )

    if fixed32_enabled and serving_batch == 4:
        assert campaign_metrics_pre_path is not None
        assert campaign_metrics_post_path is not None
        _write_fixed32_campaign_metrics(campaign_metrics_post_path)
        _finalize_fixed32_qwen_campaign_provenance(
            summaries=summaries,
            instance_ids=instance_ids,
            dataset_out=dataset_out,
            per_task_root=per_task_root,
            campaign_metrics_pre_path=campaign_metrics_pre_path,
            campaign_metrics_post_path=campaign_metrics_post_path,
        )
        # Qualification contains task traces and is not itself a publishable
        # timing unit. Its reduced artifact must be produced explicitly after
        # review; never let the generic campaign autocommitter sweep raw output.
        if cfwd_campaign_bracket is None:
            _autocommit_fixed32_campaign_artifacts(
                dataset_out=dataset_out,
                per_task_root=per_task_root,
                instance_ids=instance_ids,
                taw_campaign_arm_artifact_path=taw_campaign_arm_artifact_path,
            )

    ended_at = _iso_now()
    summary = _aggregate(
        per_task_root=per_task_root,
        summary_path=dataset_out / "campaign_summary.json",
        predictions_path=dataset_out / "predictions.jsonl",
        started_at=started_at,
        ended_at=ended_at,
        model_name=args.model_name,
    )
    if fixed32_cfwd_qualification:
        if cfwd_campaign_bracket is not None:
            campaign_payload = cfwd_campaign_bracket.as_dict()
            qualification = campaign_payload.get("qualification_coverage")
            coverage_complete = bool(
                isinstance(qualification, dict)
                and qualification.get("coverage_complete")
            )
            summary["fixed32_run_classification"] = {
                "run_classification": (
                    _FIXED32_CFWD_B4_QUALIFICATION_CLASSIFICATION
                ),
                "performance_measurement": False,
                "timing_eligible": False,
                "gate_eligible": False,
                "floor_acceptance_eligible": False,
                "process_local_qualification_only": True,
                "durable_production_pass": False,
                "timing_requires_same_server_process": True,
                "same_process_timing_handoff_contract_implemented": True,
                "same_process_timing_execution_implemented": False,
                "coverage_complete": coverage_complete,
                "campaign_qualification": campaign_payload,
            }
        else:
            completed = []
            for task in summaries:
                boundary = task.get("fixed32_task_boundary")
                qualification = (
                    boundary.get("qualification_coverage")
                    if isinstance(boundary, dict)
                    else None
                )
                if isinstance(qualification, dict):
                    completed.append(
                        bool(qualification.get("coverage_complete"))
                    )
            summary["fixed32_run_classification"] = {
                "run_classification": _FIXED32_CFWD_QUALIFICATION_CLASSIFICATION,
                "performance_measurement": False,
                "timing_eligible": False,
                "gate_eligible": False,
                "floor_acceptance_eligible": False,
                "process_local_qualification_only": True,
                "durable_production_pass": False,
                "timing_requires_same_server_process": True,
                "same_process_timing_handoff_contract_implemented": False,
                "same_process_timing_execution_implemented": False,
                "coverage_complete": bool(completed) and all(completed),
            }
        (dataset_out / "campaign_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
    elif fixed32_b1_diagnostic:
        summary["fixed32_run_classification"] = {
            "run_classification": "b1_diagnostic",
            "gate_eligible": False,
            "floor_acceptance_eligible": False,
        }
        (dataset_out / "campaign_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
    elif taw_campaign_arm is not None:
        summary["fixed32_run_classification"] = {
            "run_classification": "b4_taw_diagnostic",
            "gate_eligible": False,
            "floor_acceptance_eligible": False,
            "campaign_arm": taw_campaign_arm.as_dict(),
        }
        (dataset_out / "campaign_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
    print(f"=== [{ended_at}] DONE n={summary['instances_total']} "
          f"resolved_rate={summary.get('resolved_rate')} "
          f"verdicts={summary['verdict_counts']} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
