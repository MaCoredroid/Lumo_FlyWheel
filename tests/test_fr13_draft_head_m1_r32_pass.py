from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts import fr13_draft_head_m1_r32_pass as gate


SOURCE_COMMIT = "b" * 40
RUNTIME_SHA256 = "a" * 64
EVENTS_SHA256 = "c" * 64
BOUNDARY_SHA256 = "d" * 64
FLUSH_NONCE = "f" * 64


def _live_payload(events: int = 3) -> dict[str, object]:
    return {
        "schema": gate.LIVE_SCHEMA,
        "status": "PASS",
        "suite": "SWE-Verified",
        "instance_id": gate.EXPECTED_INSTANCE,
        "task_marker": f"swe_verified:{gate.EXPECTED_INSTANCE}",
        "concurrency": 1,
        "batch_size": 1,
        "source_commit": SOURCE_COMMIT,
        "runtime_source_sha256": RUNTIME_SHA256,
        "candidate_source_sha256": gate.EXPECTED_SOURCE_SHA256,
        "candidate_binary_sha256": gate.EXPECTED_BINARY_SHA256,
        "candidate_binary_bytes": gate.EXPECTED_BINARY_BYTES,
        "geometry": copy.deepcopy(gate.EXPECTED_GEOMETRY),
        "candidate": copy.deepcopy(gate.EXPECTED_CANDIDATE),
        "graph_lifecycle": {
            "captured_loop_calls": 4,
            "selected_root_calls": 1,
            "fallback_calls": 0,
            "observed_measured_replays": events,
            "drafter_graph_id": 17,
            "drafter_graph_signature": gate.EXPECTED_GRAPH_SIGNATURE,
            "capture_origin": "unmeasured",
            "last_measured_forward_step_index": events - 1,
        },
        "completed_events": events,
        "complete_work_census_events": events,
        "work_census_last_event_index": events - 1,
        "events_sha256": EVENTS_SHA256,
        "flush_generation": 2,
        "flush_nonce": FLUSH_NONCE,
        "producer_pid": 123,
        "boundary_snapshot_sha256": BOUNDARY_SHA256,
        "site_labels": list(gate.SITE_LABELS),
        "per_site_full_logit_comparisons": {
            site: events for site in gate.SITE_LABELS
        },
        "per_site_compared_bf16_values": {
            site: events * 65536 for site in gate.SITE_LABELS
        },
        "per_site_raw_bf16_mismatches": {
            site: 0 for site in gate.SITE_LABELS
        },
        "full_logit_comparisons": events * len(gate.SITE_LABELS),
        "compared_bf16_values": events * len(gate.SITE_LABELS) * 65536,
        "raw_bf16_mismatches": 0,
        "served_return": "incumbent BF16 K64 reference logits unchanged",
        "performance_measurement": False,
        "device_counted_without_measured_host_sync": True,
        "finalized_by_fixed32_flush": True,
        "flush_action": "final",
    }


def _write_json(path: Path, payload: dict[str, object]) -> str:
    raw = gate.common.canonical_bytes(payload) + b"\n"
    path.write_bytes(raw)
    return gate._sha256_bytes(raw)


def _set_path(payload: dict[str, object], path: tuple[str, ...], value: object) -> None:
    target: dict[str, object] = payload
    for key in path[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = value


def _patch_candidate_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "_bind_repo", lambda repo, _commit: repo)
    monkeypatch.setattr(
        gate,
        "_bind_candidate_files",
        lambda **_kwargs: (
            RUNTIME_SHA256,
            gate.EXPECTED_SOURCE_SHA256,
            gate.EXPECTED_BINARY_SHA256,
        ),
    )


def test_repository_binding_requires_exact_clean_head(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "r32@example.test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "R32 Test"],
        check=True,
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("bound\n", encoding="ascii")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "bind"], check=True
    )
    head = (
        subprocess.run(
            ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
        )
        .stdout.decode("ascii")
        .strip()
    )

    assert gate._bind_repo(tmp_path, head) == tmp_path.resolve()
    with pytest.raises(ValueError, match="does not equal"):
        gate._bind_repo(tmp_path, "0" * 40)
    tracked.write_text("dirty\n", encoding="ascii")
    with pytest.raises(ValueError, match="tracked source changes"):
        gate._bind_repo(tmp_path, head)


def test_validate_live_result_accepts_exact_five_site_reference_record() -> None:
    summary = gate._validate_live_result(
        _live_payload(),
        expected_source_commit=SOURCE_COMMIT,
        expected_runtime_source_sha256=RUNTIME_SHA256,
    )

    assert summary["completed_events"] == 3
    assert summary["per_site_full_logit_comparisons"] == {
        site: 3 for site in gate.SITE_LABELS
    }
    assert summary["per_site_raw_bf16_mismatches"] == {
        site: 0 for site in gate.SITE_LABELS
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "FAIL"),
        (("instance_id",), "astropy__astropy-13236"),
        (("source_commit",), "0" * 40),
        (("candidate_binary_sha256",), "0" * 64),
        (("served_return",), "candidate logits"),
        (("performance_measurement",), True),
        (("per_site_raw_bf16_mismatches", "root"), 1),
        (("per_site_full_logit_comparisons", "root"), 0),
        (("graph_lifecycle", "drafter_graph_signature"), "0" * 64),
        (("graph_lifecycle", "fallback_calls"), 1),
        (("graph_lifecycle", "selected_root_calls"), True),
    ],
)
def test_validate_live_result_rejects_semantic_tampering(
    path: tuple[str, ...], value: object
) -> None:
    payload = _live_payload(events=1)
    _set_path(payload, path, value)

    with pytest.raises(ValueError):
        gate._validate_live_result(
            payload,
            expected_source_commit=SOURCE_COMMIT,
            expected_runtime_source_sha256=RUNTIME_SHA256,
        )


def test_validate_live_result_rejects_schema_extension() -> None:
    payload = _live_payload()
    payload["diagnostic_override"] = True

    with pytest.raises(ValueError, match="key set drifted"):
        gate._validate_live_result(
            payload,
            expected_source_commit=SOURCE_COMMIT,
            expected_runtime_source_sha256=RUNTIME_SHA256,
        )


def test_issue_refuses_credential_when_authenticated_traffic_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_candidate_binding(monkeypatch)
    live = tmp_path / "live.json"
    final_flush = tmp_path / "final.json"
    boundary = tmp_path / "boundary.json"
    audit = tmp_path / "audit.json"
    for path in (final_flush, boundary, audit):
        path.write_text("{}\n", encoding="ascii")
    live_payload = _live_payload()
    live_payload["boundary_snapshot_sha256"] = gate.common.sha256_file(boundary)
    expected_live_sha256 = _write_json(live, live_payload)
    monkeypatch.setattr(
        gate.common,
        "validate_live_evidence",
        lambda **_kwargs: {
            "completed_events": 3,
            "events_sha256": EVENTS_SHA256,
            "boundary_snapshot_sha256": BOUNDARY_SHA256,
        },
    )

    def reject_traffic(**_kwargs: object) -> dict[str, object]:
        raise ValueError("synthetic or unauthenticated traffic")

    monkeypatch.setattr(gate.common, "validate_chat_traffic_audit", reject_traffic)
    out = tmp_path / "qualification.json"

    with pytest.raises(ValueError, match="unauthenticated"):
        gate.issue_credential(
            live_result=live,
            expected_live_sha256=expected_live_sha256,
            final_flush=final_flush,
            boundary_snapshot=boundary,
            chat_traffic_audit=audit,
            runtime_source=tmp_path / "runtime.py",
            candidate_source=tmp_path / "candidate.cu",
            candidate_binary=tmp_path / "candidate.so",
            expected_source_commit=SOURCE_COMMIT,
            out=out,
            repo=tmp_path,
        )

    assert not out.exists()


def test_issue_and_verify_bind_authenticated_evidence_and_reject_tampered_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_candidate_binding(monkeypatch)
    live = tmp_path / "live.json"
    final_flush = tmp_path / "final.json"
    boundary = tmp_path / "boundary.json"
    audit = tmp_path / "audit.json"
    for path in (final_flush, boundary, audit):
        path.write_text("{}\n", encoding="ascii")
    live_payload = _live_payload()
    live_payload["boundary_snapshot_sha256"] = gate.common.sha256_file(boundary)
    expected_live_sha256 = _write_json(live, live_payload)
    calls: list[str] = []

    def validate_terminal(**_kwargs: object) -> dict[str, object]:
        calls.append("terminal")
        return {
            "completed_events": 3,
            "events_sha256": EVENTS_SHA256,
            "boundary_snapshot_sha256": gate.common.sha256_file(boundary),
        }

    def validate_traffic(**_kwargs: object) -> dict[str, object]:
        calls.append("traffic")
        return {
            "completed_events": 3,
            "chat_traffic_audit_sha256": gate.common.sha256_file(audit),
            "trace_completed_logical_model_requests": 2,
        }

    def validate_rebuilt(**_kwargs: object) -> None:
        calls.append("rebuilt")

    monkeypatch.setattr(gate.common, "validate_live_evidence", validate_terminal)
    monkeypatch.setattr(
        gate.common, "validate_chat_traffic_audit", validate_traffic
    )
    monkeypatch.setattr(
        gate.common, "validate_rebuilt_chat_traffic_audit", validate_rebuilt
    )
    credential_path = tmp_path / "qualification.json"
    credential = gate.issue_credential(
        live_result=live,
        expected_live_sha256=expected_live_sha256,
        final_flush=final_flush,
        boundary_snapshot=boundary,
        chat_traffic_audit=audit,
        runtime_source=tmp_path / "runtime.py",
        candidate_source=tmp_path / "candidate.cu",
        candidate_binary=tmp_path / "candidate.so",
        expected_source_commit=SOURCE_COMMIT,
        out=credential_path,
        repo=tmp_path,
    )

    assert calls == ["terminal", "traffic", "rebuilt"]
    assert credential["authenticated_one_task_completion"] is True
    assert credential["performance_measurement"] is False
    assert credential["production_admitted"] is False
    assert credential_path.stat().st_mode & 0o777 == 0o600
    credential_sha256 = gate.common.sha256_file(credential_path)
    assert gate.verify_credential(
        credential_path=credential_path,
        expected_credential_sha256=credential_sha256,
        runtime_source=tmp_path / "runtime.py",
        candidate_source=tmp_path / "candidate.cu",
        candidate_binary=tmp_path / "candidate.so",
        expected_source_commit=SOURCE_COMMIT,
        repo=tmp_path,
    )["status"] == "PASS"

    tampered = copy.deepcopy(credential)
    tampered["per_site_raw_bf16_mismatches"]["root"] = 1
    body = dict(tampered)
    body.pop("canonical_sha256")
    tampered["canonical_sha256"] = gate._sha256_bytes(
        gate.common.canonical_bytes(body)
    )
    tampered_path = tmp_path / "tampered.json"
    tampered_sha256 = _write_json(tampered_path, tampered)
    with pytest.raises(ValueError, match="contract drifted"):
        gate.verify_credential(
            credential_path=tampered_path,
            expected_credential_sha256=tampered_sha256,
            runtime_source=tmp_path / "runtime.py",
            candidate_source=tmp_path / "candidate.cu",
            candidate_binary=tmp_path / "candidate.so",
            expected_source_commit=SOURCE_COMMIT,
            repo=tmp_path,
        )


def test_verify_rejects_extra_key_even_with_recomputed_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_candidate_binding(monkeypatch)
    payload = {
        key: None for key in gate.CREDENTIAL_KEYS if key != "canonical_sha256"
    }
    payload["canonical_sha256"] = gate._sha256_bytes(
        gate.common.canonical_bytes(
            {key: value for key, value in payload.items() if key != "canonical_sha256"}
        )
    )
    payload["diagnostic_override"] = True
    body = dict(payload)
    body.pop("canonical_sha256")
    payload["canonical_sha256"] = gate._sha256_bytes(
        gate.common.canonical_bytes(body)
    )
    path = tmp_path / "extra-key.json"
    expected_sha256 = _write_json(path, payload)

    with pytest.raises(ValueError, match="key set drifted"):
        gate.verify_credential(
            credential_path=path,
            expected_credential_sha256=expected_sha256,
            runtime_source=tmp_path / "runtime.py",
            candidate_source=tmp_path / "candidate.cu",
            candidate_binary=tmp_path / "candidate.so",
            expected_source_commit=SOURCE_COMMIT,
            repo=tmp_path,
        )
