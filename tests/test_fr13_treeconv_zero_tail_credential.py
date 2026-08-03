from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import fr13_fixed32_work_census as work_census  # noqa: E402
import fr13_treeconv_zero_tail_credential as credential  # noqa: E402
from lumo_flywheel_serving import inference_proxy  # noqa: E402


def _write_json(path: Path, value: object, *, compact: bool = False) -> None:
    if compact:
        encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    else:
        encoded = json.dumps(value, ensure_ascii=True, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded + "\n", encoding="ascii")


def _identity(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _refresh_qwen_metadata(qwen: Path) -> None:
    proof = json.loads(qwen.read_bytes())
    proof_identity = _identity(qwen)
    for task_id in proof["task_ids"]:
        metadata_path = qwen.parent / "per_task" / task_id / "runner_metadata.json"
        _write_json(
            metadata_path,
            {
                "instance_id": task_id,
                "fixed32_qwen_campaign_proof": proof_identity,
                "fixed32_real_task_provenance": {
                    "schema": "fr13-fixed32-real-task-provenance-v3",
                    "instance_id": task_id,
                    "qwen_metric_scope": "campaign",
                    "qwen_campaign_metric_proof": proof_identity,
                    "qwen_campaign_metric_evidence_sha256": proof[
                        "metric_evidence_sha256"
                    ],
                },
            },
        )


def _make_qwen_proof(tmp_path: Path, tasks: list[str]) -> Path:
    dataset = tmp_path / "verified"
    pre = dataset / "fixed32_qwen_campaign_metrics_pre.txt"
    post = dataset / "fixed32_qwen_campaign_metrics_post.txt"
    pre.parent.mkdir(parents=True)
    pre.write_text("pre metrics\n", encoding="ascii")
    post.write_text("post metrics\n", encoding="ascii")
    task_proofs = []
    for task_id in tasks:
        trace = dataset / "per_task" / task_id / "qwen_trace.jsonl"
        _write_json(trace, {"event": "request_complete", "instance_id": task_id})
        task_proofs.append(
            {
                "instance_id": task_id,
                "task_key_id": inference_proxy.fixed32_task_key_id(task_id),
                "expected_completed_logical_model_requests": 1,
                "trace": _identity(trace),
            }
        )
    qwen = dataset / "fixed32_qwen_campaign_provenance.json"
    _write_json(
        qwen,
        {
            "schema": credential.QWEN_SCHEMA,
            "metric_scope": "concurrent_campaign_union",
            "concurrency": 4,
            "task_ids": tasks,
            "selection": {
                "basis": "runner_owned_campaign_endpoint_metrics",
                "task_boundary_schema": "fr13-fixed32-eager-kernel-diagnostic-task-bracket-v1",
                "task_stream_coverage": None,
            },
            "metrics_pre": _identity(pre),
            "metrics_post": _identity(post),
            "tasks": task_proofs,
            "metric_evidence_sha256": hashlib.sha256(b"metric evidence").hexdigest(),
            "metric_evidence": {"request_count": 4},
        },
        compact=True,
    )
    _refresh_qwen_metadata(qwen)
    return qwen


def _inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    batch: int,
) -> dict[str, object]:
    tasks = list(credential.CANONICAL_SUBSETS[batch]["task_ids"])
    subset = ROOT / (
        "config/fr13_fixed32/subset_b1_diagnostic_one.json"
        if batch == 1
        else "config/fr13_fixed32/subset_b4_four.json"
    )
    health = tmp_path / "health.json"
    _write_json(
        health,
        {"swe_orchestrator_rc": 0, "tasks": [{"instance_id": task} for task in tasks]},
    )
    state_src_sha256 = hashlib.sha256(b"physical32-state-src").hexdigest()
    comparator = tmp_path / "comparator.jsonl"
    _write_json(
        comparator,
        {
            "schema": credential.RECORD_SCHEMA,
            "invocation": 0,
            "mode": "hydra27_fixed32",
            "batch": batch,
            "physical_drafts": 31,
            "physical_rows_root_inclusive": 32,
            "conv_layers": 48,
            "conv_channels": 10240,
            "conv_state_length": 34,
            "source_rows_per_request": 36,
            "live_state_columns": 3,
            "state_src_sha256": state_src_sha256,
            "compared_bytes": batch * 48 * 10240 * 34 * 2,
            "differing_bytes": 0,
            "first_mismatch_layer": None,
            "byte_equal": True,
            "candidate_zero_tail": True,
            "reference_zero_tail": False,
            "reference_restored_and_served": True,
            "timing_eligible": False,
        },
    )
    work = tmp_path / "work.jsonl"
    _write_json(
        work,
        work_census.reference_event(
            "hydra27_fixed32", batch, "event-0", event_index=0, forward_step_index=1
        ),
    )
    eager_terminal = tmp_path / "eager-terminal.json"
    _write_json(
        eager_terminal,
        {
            "acceptance_valid": False,
            "flush_protocol_used": False,
            "run_classification": "eager_kernel_byte_diagnostic",
            "schema": "fr13-fixed32-eager-kernel-terminal-v1",
        },
    )
    ledger = tmp_path / "ledger.jsonl"
    _write_json(
        ledger,
        {
            "event": "campaign_begin",
            "evidence_sha256": inference_proxy.fixed32_canonical_task_set_sha256(
                tuple(tasks)
            ),
        },
    )
    monkeypatch.setattr(
        inference_proxy,
        "verify_fixed32_ingress_ledger",
        lambda *args, **kwargs: {"chain_head_sha256": "a" * 64},
    )

    repo = tmp_path / "repo"
    source = repo / credential.SOURCE_RELATIVE
    source.parent.mkdir(parents=True)
    source.write_text("# zero-tail source\n", encoding="ascii")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", credential.SOURCE_RELATIVE], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "source"], check=True)
    source_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    source_raw = source.read_bytes()
    runtime = tmp_path / "runtime.json"
    _write_json(
        runtime,
        {
            "schema": "fr13-runtime-manifest-v1",
            "closures": {
                "python_package_source": [
                    {
                        "path": credential.SOURCE_RELATIVE,
                        "sha256": hashlib.sha256(source_raw).hexdigest(),
                        "size": len(source_raw),
                    }
                ]
            },
        },
    )
    container_env = tmp_path / "container_env.txt"
    container_env.write_text(
        "\n".join(
            f"{key}={value}"
            for key, value in {
                **credential.REQUIRED_CONTAINER_ENV,
                "FR13_FIXED32_MODE": "hydra27_fixed32",
            }.items()
        )
        + "\n",
        encoding="ascii",
    )
    qwen = _make_qwen_proof(tmp_path, tasks) if batch == 4 else None
    return {
        "comparator_path": comparator,
        "subset_path": subset,
        "health_path": health,
        "ledger_path": ledger,
        "work_census_path": work,
        "eager_terminal_path": eager_terminal,
        "runtime_manifest_path": runtime,
        "source_path": source,
        "repo_path": repo,
        "container_env_path": container_env,
        "source_commit": source_commit,
        "mode": "hydra27_fixed32",
        "batch_size": batch,
        "qwen_campaign_path": qwen,
    }


@pytest.mark.parametrize("batch", (1, 4))
def test_credential_binds_real_task_source_and_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batch: int
) -> None:
    payload = credential.issue_credential(**_inputs(tmp_path, monkeypatch, batch=batch))
    assert payload["status"] == "pass"
    assert payload["task_ids"] == list(credential.CANONICAL_SUBSETS[batch]["task_ids"])
    assert payload["draft_vocab_k"] == 65536
    assert payload["physical_rows_root_inclusive"] == 32
    assert payload["reference_always_served"] is True
    assert payload["timing_eligible"] is False
    assert len(payload["container_env_sha256"]) == 64
    assert (payload["qwen_campaign_proof"] is not None) == (batch == 4)


def test_credential_rejects_comparator_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path, monkeypatch, batch=1)
    path = inputs["comparator_path"]
    assert isinstance(path, Path)
    row = json.loads(path.read_bytes())
    row.update({"differing_bytes": 1, "byte_equal": False})
    _write_json(path, row)
    with pytest.raises(credential.CredentialError, match="comparison contract or bytes differ"):
        credential.issue_credential(**inputs)


def test_credential_rejects_exact_work_count_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path, monkeypatch, batch=1)
    path = inputs["work_census_path"]
    assert isinstance(path, Path)
    row = json.loads(path.read_bytes())
    row["tree_attn"]["q_rows"] += 1
    _write_json(path, row)
    with pytest.raises(credential.CredentialError, match="exact-count mismatch"):
        credential.issue_credential(**inputs)


@pytest.mark.parametrize("batch", (1, 4))
def test_credential_rejects_noncanonical_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batch: int
) -> None:
    inputs = _inputs(tmp_path, monkeypatch, batch=batch)
    original = Path(inputs["subset_path"])
    subset = tmp_path / "subset-tampered.json"
    value = json.loads(original.read_bytes())
    if batch == 4:
        value["instance_ids"] = list(reversed(value["instance_ids"]))
    _write_json(subset, value)
    inputs["subset_path"] = subset
    with pytest.raises(credential.CredentialError, match="pinned SWE-Verified"):
        credential.issue_credential(**inputs)


@pytest.mark.parametrize(
    ("key", "value"),
    (("metric_scope", "task"), ("concurrency", 3), ("task_ids", list(reversed(credential.CANONICAL_SUBSETS[4]["task_ids"])))),
)
def test_b4_credential_rejects_campaign_contract_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, value: object
) -> None:
    inputs = _inputs(tmp_path, monkeypatch, batch=4)
    qwen = Path(inputs["qwen_campaign_path"])
    proof = json.loads(qwen.read_bytes())
    proof[key] = value
    _write_json(qwen, proof, compact=True)
    _refresh_qwen_metadata(qwen)
    with pytest.raises(credential.CredentialError, match="campaign union contract mismatch"):
        credential.issue_credential(**inputs)


def test_b4_credential_rejects_per_task_provenance_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path, monkeypatch, batch=4)
    qwen = Path(inputs["qwen_campaign_path"])
    task_id = credential.CANONICAL_SUBSETS[4]["task_ids"][0]
    metadata_path = qwen.parent / "per_task" / task_id / "runner_metadata.json"
    metadata = json.loads(metadata_path.read_bytes())
    metadata["fixed32_real_task_provenance"]["qwen_campaign_metric_evidence_sha256"] = "0" * 64
    _write_json(metadata_path, metadata)
    with pytest.raises(credential.CredentialError, match="per-task provenance binding"):
        credential.issue_credential(**inputs)


def test_credential_rejects_source_commit_byte_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path, monkeypatch, batch=1)
    source = Path(inputs["source_path"])
    source.write_text("# modified after commit\n", encoding="ascii")
    raw = source.read_bytes()
    runtime = Path(inputs["runtime_manifest_path"])
    manifest = json.loads(runtime.read_bytes())
    manifest["closures"]["python_package_source"][0].update(
        {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
    )
    _write_json(runtime, manifest)
    with pytest.raises(credential.CredentialError, match="source commit does not bind"):
        credential.issue_credential(**inputs)


def test_credential_rejects_k64_environment_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path, monkeypatch, batch=1)
    env_path = Path(inputs["container_env_path"])
    env_path.write_text(
        env_path.read_text(encoding="ascii").replace(
            "FR13_DRAFT_VOCAB_K=65536", "FR13_DRAFT_VOCAB_K=32768"
        ),
        encoding="ascii",
    )
    with pytest.raises(credential.CredentialError, match="physical32 K64/root1"):
        credential.issue_credential(**inputs)
