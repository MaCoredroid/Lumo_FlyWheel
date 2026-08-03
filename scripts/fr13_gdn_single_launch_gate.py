#!/usr/bin/env python3
"""Reduce one real-task fixed32 ordered-GDN diagnostic into a scoped credential."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, REPO / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fr13_fixed32_contract as fixed32_contract  # noqa: E402
import fr13_fixed32_work_census as work_census  # noqa: E402
import fr13_floor_gate as floor_gate  # noqa: E402
import fr13_runtime_manifest as runtime_manifest  # noqa: E402


SCHEMA = "fr13.fixed32.gdn_single_launch.real_task_credential.v3"
LIVE_SCHEMA = "fr13.fixed32.gdn_single_launch.live_observation.v2"
CANDIDATE = "fixed32_gdn_single_launch_tree_v2"
REFERENCE = "fixed32_gdn_two_launch_reference_v1"
DATASET = "princeton-nlp/SWE-bench_Verified"
B1_TASK_IDS = ("astropy__astropy-12907",)
EXACT4_TASK_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
)
MODE = {
    "tail6_fixed32": {
        "topology": "Tail23",
        "slug": "tail23",
        "logical_drafts": 23,
        "valid_mask": 0x7A9CE7FF,
    },
    "hydra27_fixed32": {
        "topology": "Hydra27",
        "slug": "hydra27",
        "logical_drafts": 27,
        "valid_mask": 0x7ABDFFFF,
    },
}
ENTRYPOINT = {
    ("hydra27_fixed32", 1): "scripts/fr13_run_b1_gdn_single_launch_live_gate.sh",
    ("tail6_fixed32", 4): (
        "scripts/fr13_run_b4_tail23_gdn_single_launch_live_gate.sh"
    ),
    ("hydra27_fixed32", 4): (
        "scripts/fr13_run_b4_hydra27_gdn_single_launch_live_gate.sh"
    ),
}
COMMON_RUNNER = "scripts/fr13_run_gdn_single_launch_live_gate.sh"
KERNEL_SOURCE = "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
PATCHER_SOURCE = "scripts/fr10_phase4_patch_vllm_tree_gdn.py"
SERVER_LAUNCHER = "scripts/fr13_launch_forked_fa2_tree_server.sh"
BLOCK_MAP = "scripts/fr13_dvk_subset_blocks.json"
INGRESS_SOURCE = "src/lumo_flywheel_serving/inference_proxy.py"
VALIDATOR_SOURCES = (
    "scripts/fr13_fixed32_contract.py",
    "scripts/fr13_fixed32_work_census.py",
    "scripts/fr13_floor_gate.py",
    "scripts/fr13_runtime_manifest.py",
)
TEST_SOURCES = (
    "tests/test_fr13_fixed32_gdn_path_bv_live_gate.py",
    "tests/test_fr13_fixed32_gdn_single_launch_real_gate.py",
    "tests/test_fr13_gdn_single_launch_campaign_gate.py",
)
RUNTIME_SOURCE_SECTIONS = frozenset(
    {"host_script_source", "python_package_source", "verdict_tools"}
)
RUNTIME_CLOSURE_SECTIONS = RUNTIME_SOURCE_SECTIONS | {
    "runtime_data_and_config"
}
RUNTIME_SOURCE_IDENTITY_SCHEMA = "fr13-runtime-source-identity-v1"
SUBSET_PATH = {
    1: "config/fr13_fixed32/subset_b1_diagnostic_one.json",
    4: "config/fr13_fixed32/subset_b4_four.json",
}
BLOCK_MAP_SHA256 = (
    "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
)
TAW_ROUTE = work_census.TAW_ROUTE
HEX = frozenset("0123456789abcdef")
OPEN_BASE_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
OPEN_DIRECTORY_FLAGS = OPEN_BASE_FLAGS | getattr(os, "O_DIRECTORY", 0)
OPEN_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
READ_CHUNK_BYTES = 1024 * 1024


class GateError(RuntimeError):
    pass


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _regular(path: Path, label: str) -> bytes:
    try:
        info = path.lstat()
    except OSError as error:
        raise GateError(f"{label} is missing: {path}") from error
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
    ):
        raise GateError(f"{label} must be a singly-linked regular file: {path}")
    return path.read_bytes()


def _open_no_symlinks(base: Path, relative_path: str, label: str) -> int:
    canonical = _canonical_repo_path(relative_path, label)
    parts = PurePosixPath(canonical).parts
    try:
        directory_fd = os.open(base, OPEN_DIRECTORY_FLAGS)
    except OSError as error:
        raise GateError(f"cannot open {label} parent: {error}") from error
    try:
        for part in parts[:-1]:
            next_fd = os.open(
                part,
                OPEN_DIRECTORY_FLAGS | OPEN_NOFOLLOW,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        return os.open(
            parts[-1],
            OPEN_BASE_FLAGS | OPEN_NOFOLLOW,
            dir_fd=directory_fd,
        )
    except OSError as error:
        raise GateError(f"cannot securely open {label}: {error}") from error
    finally:
        os.close(directory_fd)


def _file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _secure_regular_relative(
    base: Path, relative_path: str, label: str
) -> bytes:
    file_fd = _open_no_symlinks(base, relative_path, label)
    captured = bytearray()
    try:
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise GateError(f"{label} must be a singly-linked regular file")
        while True:
            chunk = os.read(file_fd, READ_CHUNK_BYTES)
            if not chunk:
                break
            captured.extend(chunk)
        after = os.fstat(file_fd)
    finally:
        os.close(file_fd)
    if (
        _file_identity(before) != _file_identity(after)
        or len(captured) != before.st_size
    ):
        raise GateError(f"{label} changed while being read")
    check_fd = _open_no_symlinks(base, relative_path, label)
    try:
        current = os.fstat(check_fd)
        if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
            raise GateError(f"{label} must remain a singly-linked regular file")
    finally:
        os.close(check_fd)
    if _file_identity(before) != _file_identity(current):
        raise GateError(f"{label} changed while being read")
    return bytes(captured)


def _validate_arm_git_head(arm: Path, source_commit: str) -> dict[str, Any]:
    raw = _secure_regular_relative(arm, "git_head.txt", "arm git_head.txt")
    expected = f"{source_commit}\n".encode("ascii")
    if raw != expected:
        raise GateError(
            "arm git_head.txt does not contain the exact source commit"
        )
    return {
        "path": "git_head.txt",
        "sha256": _sha256(raw),
        "bytes": len(raw),
        "source_commit": source_commit,
    }


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _regular(path, label)

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise GateError(f"{label} has duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        payload = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                GateError(f"{label} has non-finite value {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateError(f"{label} is not ASCII JSON: {error}") from error
    if not isinstance(payload, dict):
        raise GateError(f"{label} JSON root is not an object")
    return payload, raw


def _canonical(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(raw)
    os.chmod(temporary, 0o400)
    os.replace(temporary, path)


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX for character in value)
    ):
        raise GateError(f"{label} is not a lowercase SHA-256")
    return value


def _canonical_repo_path(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
    ):
        raise GateError(f"{label} is not a canonical repo-relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise GateError(f"{label} is not a canonical repo-relative path")
    return value


def _runtime_manifest_records(
    payload: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], frozenset[str]]:
    closures = payload.get("closures")
    if not isinstance(closures, dict) or set(closures) != RUNTIME_CLOSURE_SECTIONS:
        raise GateError("runtime manifest closure sections drifted")
    records_by_path: dict[str, dict[str, Any]] = {}
    source_paths: set[str] = set()
    for section in sorted(RUNTIME_CLOSURE_SECTIONS):
        records = closures.get(section)
        if not isinstance(records, list):
            raise GateError(f"runtime manifest {section} closure is not an array")
        for index, record in enumerate(records):
            label = f"runtime manifest {section}[{index}]"
            if not isinstance(record, dict) or set(record) != {
                "path",
                "sha256",
                "size",
            }:
                raise GateError(f"{label} record shape drifted")
            path = _canonical_repo_path(record.get("path"), f"{label}.path")
            _require_sha256(record.get("sha256"), f"{label}.sha256")
            size = record.get("size")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise GateError(f"{label}.size is invalid")
            if path in records_by_path:
                raise GateError(f"runtime manifest repeats source path {path}")
            records_by_path[path] = record
            if section in RUNTIME_SOURCE_SECTIONS:
                source_paths.add(path)
    return records_by_path, frozenset(source_paths)


def _git_object_bytes(repo: Path, source_commit: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{source_commit}:{path}"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise GateError(
            f"claimed source commit does not contain {path}: {detail}"
        )
    return result.stdout


def _git_bound_closure(
    repo: Path, source_commit: str, paths: set[str] | frozenset[str]
) -> dict[str, str]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise GateError("source commit is not a full lowercase Git object ID")
    normalized = {
        _canonical_repo_path(path, "source closure path") for path in paths
    }
    closure: dict[str, str] = {}
    for path in sorted(normalized):
        current = _regular(repo / path, f"current source closure {path}")
        committed = _git_object_bytes(repo, source_commit, path)
        if current != committed:
            raise GateError(
                f"current source closure differs from {source_commit}:{path}"
            )
        closure[path] = _sha256(committed)
    return closure


def _validate_runtime_manifest(
    payload: dict[str, Any],
    *,
    repo: Path,
    source_commit: str,
    git_closure: dict[str, str],
    required_paths: tuple[str, ...],
) -> str:
    unsigned = {
        key: value
        for key, value in payload.items()
        if key != "overall_canonical_sha256"
    }
    digest = _sha256(
        json.dumps(
            unsigned,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    if (
        payload.get("schema") != "fr13-runtime-manifest-v1"
        or payload.get("profile") != "fixed32"
        or payload.get("sequence")
        != "scripts/fr13_fixed32_floor_timers_seq.sh"
        or payload.get("overall_canonical_sha256") != digest
    ):
        raise GateError("runtime manifest identity or canonical digest drifted")
    by_path, source_paths = _runtime_manifest_records(payload)
    source_records = [by_path[path] for path in sorted(source_paths)]
    if payload.get("source_identity") != {
        "schema": RUNTIME_SOURCE_IDENTITY_SCHEMA,
        "git_commit": source_commit,
        "source_file_count": len(source_records),
        "source_closure_sha256": _sha256(
            json.dumps(
                source_records,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ),
    }:
        raise GateError("runtime manifest Git/source identity drifted")
    for path in required_paths:
        expected = git_closure.get(path)
        record = by_path.get(path)
        if expected is None or not isinstance(record, dict):
            raise GateError(f"runtime manifest does not bind {path}")
        if (
            record.get("sha256") != expected
            or record.get("size") != len(_regular(repo / path, path))
        ):
            raise GateError(f"runtime manifest does not bind {path}")
    for path in source_paths:
        expected = git_closure.get(path)
        record = by_path[path]
        if (
            expected is None
            or record.get("sha256") != expected
            or record.get("size") != len(_regular(repo / path, path))
        ):
            raise GateError(
                f"runtime source closure is not bound to {source_commit}:{path}"
            )
    summary = payload.get("summary")
    if (
        not isinstance(summary, dict)
        or summary.get("file_count") != len(by_path)
        or summary.get("python_package_file_count")
        != len(payload["closures"]["python_package_source"])
        or summary.get("total_size")
        != sum(int(record["size"]) for record in by_path.values())
    ):
        raise GateError("runtime manifest summary does not reconcile")
    return digest


def _validate_health(payload: dict[str, Any], task_ids: list[str]) -> None:
    tasks = payload.get("tasks")
    if (
        payload.get("swe_orchestrator_rc") != 0
        or not isinstance(tasks, list)
        or [task.get("instance_id") for task in tasks] != task_ids
        or any(task.get("codex_timed_out") is not False for task in tasks)
        or any(task.get("verdict") == "missing" for task in tasks)
    ):
        raise GateError("health record does not prove all canonical tasks completed")


def _rebuild_traffic_audit(
    arm: Path,
    *,
    mode: str,
    subset: dict[str, Any],
    concurrency: int,
) -> tuple[dict[str, Any], bytes]:
    task_ids = list(subset["task_ids"])
    try:
        pinned = floor_gate.pinned_dataset_record_digests(str(REPO))
    except floor_gate.GateError as error:
        raise GateError(f"cannot derive pinned dataset records: {error}") from error
    missing = [task_id for task_id in task_ids if task_id not in pinned]
    if missing:
        raise GateError(f"pinned dataset records omit canonical tasks: {missing}")
    dataset_hashes = {task_id: pinned[task_id] for task_id in task_ids}
    if set(dataset_hashes) != set(task_ids) or len(dataset_hashes) != len(task_ids):
        raise GateError("pinned dataset record binding is not the exact task set")
    expected = floor_gate.build_fixed32_chat_traffic_audit(
        arm,
        mode=mode,
        subset=subset,
        dataset_record_digests=dataset_hashes,
        concurrency=concurrency,
    )
    audit_path = arm / "fixed32_chat_traffic_audit.json"
    actual, raw = _load_json(audit_path, "authenticated traffic audit")
    if actual != expected:
        raise GateError(
            "authenticated traffic audit differs from raw task, Qwen, ingress, "
            "and census evidence: "
            + floor_gate.first_json_difference(actual, expected)
        )
    return actual, raw


def _authenticated_request_task_map(
    arm: Path, expected_tasks: list[str]
) -> dict[str, str]:
    canonical_keys = {
        floor_gate.fixed32_task_key_id(task_id) for task_id in expected_tasks
    }
    task_by_key = {
        floor_gate.fixed32_task_key_id(task_id): task_id
        for task_id in expected_tasks
    }
    ledgers: dict[str, list[dict[str, Any]]] = {}
    for role in ("proxy", "engine"):
        path = arm / "logs" / f"fr13_fixed32_{role}_ingress.jsonl"
        try:
            rows, _identity = floor_gate.load_fixed32_ingress_ledger(
                path,
                role=role,
                canonical_task_keys=canonical_keys,
                canonical_task_set_sha256=(
                    floor_gate.fixed32_canonical_task_set_sha256(expected_tasks)
                ),
            )
        except floor_gate.GateError as error:
            raise GateError(
                f"cannot reload authenticated {role} request ledger: {error}"
            ) from error
        ledgers[role] = rows
    proxy_successes = {
        row["wire_id_sha256"]: row
        for row in ledgers["proxy"]
        if row["event"] == "attempt_result" and row["status_code"] == 200
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
    if (
        len(proxy_successes)
        != sum(
            row["event"] == "attempt_result" and row["status_code"] == 200
            for row in ledgers["proxy"]
        )
        or set(proxy_successes) - set(engine_accepts)
        or set(proxy_successes) - set(engine_completes)
    ):
        raise GateError("authenticated successful request ledger is not one-to-one")
    request_task_map: dict[str, str] = {}
    for wire_id, proxy in proxy_successes.items():
        accepted = engine_accepts[wire_id]
        completed = engine_completes[wire_id]
        identity = (
            proxy["task_key_id"],
            proxy["engine_request_id_sha256"],
            proxy["evidence_sha256"],
        )
        if (
            proxy["outcome"] != "response"
            or completed["outcome"] != "completed"
            or identity
            != (
                accepted["task_key_id"],
                accepted["engine_request_id_sha256"],
                accepted["evidence_sha256"],
            )
            or identity
            != (
                completed["task_key_id"],
                completed["engine_request_id_sha256"],
                completed["evidence_sha256"],
            )
        ):
            raise GateError(
                "authenticated proxy/engine request identity parity failed"
            )
        task_id = task_by_key.get(str(identity[0]))
        request_digest = _require_sha256(
            identity[1], "authenticated engine request digest"
        )
        if task_id is None or request_digest in request_task_map:
            raise GateError(
                "authenticated request-to-task mapping is not one-to-one"
            )
        request_task_map[request_digest] = task_id
    if not request_task_map:
        raise GateError("authenticated request-to-task mapping is empty")
    return request_task_map


def _validate_live_pass(
    payload: dict[str, Any],
    *,
    mode: str,
    batch: int,
    expected_tasks: list[str],
    kernel_sha256: str,
    graph_signature: str,
    census_events: list[dict[str, Any]],
    request_task_map: dict[str, str],
) -> dict[str, Any]:
    contract = MODE[mode]
    diagnostic_identity = (
        f"fixed32_gdn_single_launch_tree_v2:{contract['slug']}:b{batch}"
    )
    comparator_events = [
        event["gdn_comparator"]
        for event in census_events
        if "gdn_comparator" in event
    ]
    if not comparator_events:
        raise GateError("work census contains no GDN comparator event")
    event_ids: set[str] = set()
    request_tuples: set[tuple[str, ...]] = set()
    covered_tasks: set[str] = set()
    per_task_request_digests = {task_id: set() for task_id in expected_tasks}
    for index, comparator in enumerate(comparator_events):
        if not isinstance(comparator, dict):
            raise GateError(f"GDN comparator event {index} is malformed")
        if comparator.get("structural_graph_signature") != graph_signature:
            raise GateError("GDN comparator structural scope differs from census")
        event_id = str(comparator.get("census_event_id", ""))
        request_tuple = tuple(comparator.get("request_id_sha256s", ()))
        if (
            not event_id
            or event_id in event_ids
            or len(request_tuple) != batch
            or request_tuple in request_tuples
        ):
            raise GateError("GDN comparator event/request tuple is duplicated")
        event_ids.add(event_id)
        request_tuples.add(request_tuple)
        event_tasks: set[str] = set()
        for request_digest in request_tuple:
            task_id = request_task_map.get(request_digest)
            if task_id is None:
                raise GateError(
                    "GDN comparator request has no authenticated ledger task"
                )
            event_tasks.add(task_id)
            covered_tasks.add(task_id)
            per_task_request_digests[task_id].add(request_digest)
        marker_task = str(comparator.get("observed_task_marker", "")).removeprefix(
            "swe_verified:"
        )
        if marker_task not in event_tasks:
            raise GateError(
                "GDN comparator event marker was relabeled from its requests"
            )
    if covered_tasks != set(expected_tasks):
        raise GateError(
            "GDN comparator authenticated task union is not the exact scope"
        )
    capture_manifest_sha256s = {
        comparator["runtime_capture_manifest_sha256"]
        for comparator in comparator_events
    }
    if len(capture_manifest_sha256s) != 1:
        raise GateError("GDN comparator events do not share one capture manifest")
    canonical_events = json.dumps(
        comparator_events,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    expected = {
        "schema": LIVE_SCHEMA,
        "status": "observed_pending_authenticated_coverage_join",
        "candidate": CANDIDATE,
        "source_sha256": kernel_sha256,
        "mode": mode,
        "batch_size": batch,
        "expected_batch": batch,
        "covered_batches": [batch],
        "records_per_comparator_event": 48,
        "comparator_event_count": len(comparator_events),
        "comparator_events_sha256": _sha256(canonical_events),
        "comparator_events": comparator_events,
        "physical_rows": 32,
        "reference_bv": 8,
        "candidate_bv": 8,
        "reference_physical_launches_per_request_layer": 2,
        "candidate_physical_launches_per_request_layer": 1,
        "compared_byte_surfaces": [
            "output",
            "ring_k",
            "ring_v",
            "ring_a",
            "ring_b",
            "flags",
            "counter",
        ],
        "reference_served": True,
        "candidate_served": False,
        "production_eligible": False,
        "performance_measurement": False,
        "acceptance_valid": False,
        "logical_topology": contract["topology"],
        "logical_drafts": contract["logical_drafts"],
        "valid_mask": contract["valid_mask"],
        "draft_vocab_k": 65536,
        "draft_vocab_root": 1,
        "gate_mode": "post_measured_replay_distinct_request_tuple",
        "coverage_authority": "authenticated_proxy_engine_request_join",
        "diagnostic_identity": diagnostic_identity,
    }
    if set(payload) != set(expected):
        raise GateError("batch-specific GDN live PASS record shape drifted")
    for key, value in expected.items():
        if payload.get(key) != value:
            raise GateError(f"batch-specific GDN live PASS field drifted: {key}")
    return {
        "comparator_event_count": len(comparator_events),
        "comparator_events_sha256": expected["comparator_events_sha256"],
        "runtime_capture_manifest_sha256s": sorted(
            capture_manifest_sha256s
        ),
        "authenticated_task_ids": list(expected_tasks),
        "per_task_request_id_sha256s": {
            task_id: sorted(per_task_request_digests[task_id])
            for task_id in expected_tasks
        },
    }


def reduce(args: argparse.Namespace) -> dict[str, Any]:
    mode = args.mode
    batch = args.batch
    if (mode, batch) not in ENTRYPOINT:
        raise GateError("unsupported mode/batch credential scope")
    if re.fullmatch(r"[0-9a-f]{40}", args.source_commit) is None:
        raise GateError("source commit is not a full lowercase Git object ID")
    expected_tasks = list(B1_TASK_IDS if batch == 1 else EXACT4_TASK_IDS)
    subset_path = (REPO / SUBSET_PATH[batch]).resolve(strict=True)
    if args.subset.resolve(strict=True) != subset_path:
        raise GateError("subset path is not the canonical batch-specific source")
    subset = floor_gate.validate_fixed32_run_subset(
        subset_path,
        b1_diagnostic=batch == 1,
    )
    if subset["task_ids"] != expected_tasks:
        raise GateError("subset is not the exact expected task set")
    concurrency = batch
    arm = args.arm.resolve(strict=True)
    arm_git_head = _validate_arm_git_head(arm, args.source_commit)

    runtime_launch, runtime_launch_raw = _load_json(
        args.runtime_launch, "runtime launch manifest"
    )
    runtime_end, runtime_end_raw = _load_json(
        args.runtime_end, "runtime end manifest"
    )
    if runtime_launch_raw != runtime_end_raw or runtime_launch != runtime_end:
        raise GateError("runtime/source manifest changed from launch to end")
    external_launch, external_launch_raw = _load_json(
        args.external_launch, "external launch manifest"
    )
    external_end, external_end_raw = _load_json(
        args.external_end, "external end manifest"
    )
    if external_launch_raw != external_end_raw or external_launch != external_end:
        raise GateError("external image/model/FA2 manifest changed from launch to end")
    try:
        fixed32_contract.validate_external_manifest(external_end)
    except fixed32_contract.ContractError as error:
        raise GateError(f"external manifest is invalid: {error}") from error

    entrypoint = ENTRYPOINT[(mode, batch)]
    try:
        regenerated_runtime = runtime_manifest.build_manifest(
            REPO,
            profile="fixed32",
            sequence="scripts/fr13_fixed32_floor_timers_seq.sh",
            source_commit=args.source_commit,
        )
    except runtime_manifest.ManifestError as error:
        raise GateError(f"cannot regenerate runtime manifest: {error}") from error
    if runtime_end != regenerated_runtime:
        raise GateError(
            "runtime manifest is not the canonical current Git-bound profile"
        )
    manifest_closure_paths = (
        entrypoint,
        COMMON_RUNNER,
        "scripts/fr13_gdn_single_launch_gate.py",
        *VALIDATOR_SOURCES,
        KERNEL_SOURCE,
        PATCHER_SOURCE,
        SERVER_LAUNCHER,
        BLOCK_MAP,
        INGRESS_SOURCE,
        SUBSET_PATH[batch],
    )
    _records_by_path, runtime_source_paths = _runtime_manifest_records(runtime_end)
    git_paths = set(manifest_closure_paths) | set(TEST_SOURCES) | set(
        runtime_source_paths
    )
    required_closure = _git_bound_closure(
        REPO,
        args.source_commit,
        git_paths,
    )
    if required_closure[BLOCK_MAP] != BLOCK_MAP_SHA256:
        raise GateError("K64 block map identity drifted")
    runtime_digest = _validate_runtime_manifest(
        runtime_end,
        repo=REPO,
        source_commit=args.source_commit,
        git_closure=required_closure,
        required_paths=manifest_closure_paths,
    )
    source_closure_sha256 = _sha256(
        _canonical(
            {
                "source_commit": args.source_commit,
                "files": required_closure,
            }
        )
    )

    health, health_raw = _load_json(arm / "health.json", "health record")
    _validate_health(health, expected_tasks)
    audit, audit_raw = _rebuild_traffic_audit(
        arm,
        mode=mode,
        subset=subset,
        concurrency=concurrency,
    )
    if audit.get("complete_stream", {}).get("complete_work_census_events", 0) <= 0:
        raise GateError("authenticated campaign has no complete work-census events")
    request_task_map = _authenticated_request_task_map(arm, expected_tasks)

    census_path = arm / "logs" / "fr13_fixed32_work_census.jsonl"
    census_raw = _regular(census_path, "fixed32 work census")
    try:
        located_census = work_census.load_jsonl_bytes(
            census_raw, source=str(census_path)
        )
        work_report = work_census.validate_arm(
            located_census,
            expected_mode=mode,
            expected_route=TAW_ROUTE,
            required_batches=(batch,),
        )
    except work_census.CensusError as error:
        raise GateError(f"graph/work census is invalid: {error}") from error
    forward_rows = work_report["terminal_summary"]["forward_graph_registry"]
    forward_row = next(
        (row for row in forward_rows if row.get("batch_size") == batch),
        None,
    )
    if not isinstance(forward_row, dict):
        raise GateError(f"graph/work census has no B{batch} FULL graph")
    graph_signature = _require_sha256(
        forward_row.get("graph_signature"), "FULL graph signature"
    )

    live, live_raw = _load_json(
        args.live_pass, "GDN single-launch live observation"
    )
    kernel_sha256 = required_closure[KERNEL_SOURCE]
    comparator_coverage = _validate_live_pass(
        live,
        mode=mode,
        batch=batch,
        expected_tasks=expected_tasks,
        kernel_sha256=kernel_sha256,
        graph_signature=graph_signature,
        census_events=[
            dict(record)
            for record, _source in located_census[:-1]
            if isinstance(record, dict)
        ],
        request_task_map=request_task_map,
    )

    contract = MODE[mode]
    credential = {
        "schema": SCHEMA,
        "status": "PASS",
        "credential_scope": f"{contract['slug']}:b{batch}",
        "diagnostic_identity": live["diagnostic_identity"],
        "run_classification": (
            "one_real_swe_verified_b1_graph_byte_diagnostic"
            if batch == 1
            else "real_swe_verified_exact4_b4_graph_byte_diagnostic"
        ),
        "mode": mode,
        "logical_topology": contract["topology"],
        "logical_drafts": contract["logical_drafts"],
        "valid_mask": contract["valid_mask"],
        "expected_batch": batch,
        "batch_size": batch,
        "concurrency": concurrency,
        "task_ids": expected_tasks,
        "comparator_coverage": comparator_coverage,
        "subset_sha256": subset["sha256"],
        "candidate": CANDIDATE,
        "reference": REFERENCE,
        "physical_rows": 32,
        "draft_vocab_k": 65536,
        "draft_vocab_root": 1,
        "draft_vocab_blocks_sha256": BLOCK_MAP_SHA256,
        "reference_physical_launches_per_request_layer": 2,
        "candidate_physical_launches_per_request_layer": 1,
        "reference_served": True,
        "state_restored": True,
        "raw_byte_equal": True,
        "graph_signature": graph_signature,
        "work_census_event_count": work_report["event_count"],
        "work_census_sha256": _sha256(census_raw),
        "normalized_work_signature_sha256": work_report[
            "normalized_work_signature_sha256"
        ],
        "task_completion_verified": True,
        "finalized_ingress_verified": True,
        "qwen_compaction_algebra_replayed": True,
        "qwen_per_task_binding_verified": True,
        "runtime_launch_end_stable": True,
        "external_launch_end_stable": True,
        "graph_work_census_verified": True,
        "batch_specific_pass_verified": True,
        "production_enabled": False,
        "performance_measurement": False,
        "acceptance_valid": False,
        "floor_acceptance_eligible": False,
        "source_commit": args.source_commit,
        "arm_git_head": arm_git_head,
        "arm_git_head_source_commit_match": True,
        "source_commit_git_show_verified": True,
        "source_closure_file_count": len(required_closure),
        "source_closure_sha256": source_closure_sha256,
        "test_source_sha256": {
            path: required_closure[path] for path in TEST_SOURCES
        },
        "kernel_source_sha256": kernel_sha256,
        "runtime_manifest_source_commit": args.source_commit,
        "runtime_manifest_canonical_sha256": runtime_digest,
        "runtime_manifest_sha256": _sha256(runtime_end_raw),
        "external_manifest_sha256": _sha256(external_end_raw),
        "entrypoint_sha256": required_closure[entrypoint],
        "common_runner_sha256": required_closure[COMMON_RUNNER],
        "reducer_sha256": required_closure[
            "scripts/fr13_gdn_single_launch_gate.py"
        ],
        "live_pass_sha256": _sha256(live_raw),
        "health_sha256": _sha256(health_raw),
        "authenticated_traffic_audit_sha256": _sha256(audit_raw),
    }
    _atomic_write(args.output, _canonical(credential))
    return credential


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--mode", choices=tuple(MODE), required=True)
    parser.add_argument("--batch", type=int, choices=(1, 4), required=True)
    parser.add_argument("--live-pass", type=Path, required=True)
    parser.add_argument("--runtime-launch", type=Path, required=True)
    parser.add_argument("--runtime-end", type=Path, required=True)
    parser.add_argument("--external-launch", type=Path, required=True)
    parser.add_argument("--external-end", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    credential = reduce(args)
    print(json.dumps(credential, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
