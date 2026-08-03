from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import fr13_treeconv_zero_tail_credential as credential  # noqa: E402
from lumo_flywheel_serving import inference_proxy  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="ascii")


def _inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    batch: int,
) -> dict[str, object]:
    tasks = [f"task-{index}" for index in range(batch)]
    subset = tmp_path / "subset.json"
    _write_json(
        subset,
        {
            "dataset_name": "princeton-nlp/SWE-bench_Verified",
            "split": "test",
            "instance_ids": tasks,
        },
    )
    health = tmp_path / "health.json"
    _write_json(
        health,
        {
            "swe_orchestrator_rc": 0,
            "tasks": [{"instance_id": task} for task in tasks],
        },
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
    work.write_text(
        json.dumps(
            {
                "schema": credential.WORK_SCHEMA,
                "mode": "hydra27_fixed32",
                "batch_size": batch,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
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
    task_set_sha256 = inference_proxy.fixed32_canonical_task_set_sha256(
        tuple(tasks)
    )
    _write_json(
        ledger,
        {"event": "campaign_begin", "evidence_sha256": task_set_sha256},
    )
    monkeypatch.setattr(
        inference_proxy,
        "verify_fixed32_ingress_ledger",
        lambda *args, **kwargs: {"chain_head_sha256": "a" * 64},
    )
    runtime = tmp_path / "runtime.json"
    source = tmp_path / "kernel.py"
    source.write_text("# zero-tail source\n", encoding="ascii")
    source_raw = source.read_bytes()
    _write_json(
        runtime,
        {
            "schema": "fr13-runtime-manifest-v1",
            "closures": {
                "python_package_source": [
                    {
                        "path": "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py",
                        "sha256": hashlib.sha256(source_raw).hexdigest(),
                        "size": len(source_raw),
                    }
                ]
            },
        },
    )
    qwen = None
    if batch == 4:
        qwen = tmp_path / "qwen.json"
        _write_json(qwen, {"schema": credential.QWEN_SCHEMA})
    return {
        "comparator_path": comparator,
        "subset_path": subset,
        "health_path": health,
        "ledger_path": ledger,
        "work_census_path": work,
        "eager_terminal_path": eager_terminal,
        "runtime_manifest_path": runtime,
        "source_path": source,
        "source_commit": "b" * 40,
        "mode": "hydra27_fixed32",
        "batch_size": batch,
        "qwen_campaign_path": qwen,
    }


@pytest.mark.parametrize("batch", (1, 4))
def test_credential_binds_real_task_source_and_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batch: int
) -> None:
    inputs = _inputs(tmp_path, monkeypatch, batch=batch)
    payload = credential.issue_credential(**inputs)
    assert payload["status"] == "pass"
    assert payload["batch_size"] == batch
    assert payload["task_count"] == batch
    assert payload["draft_vocab_k"] == 65536
    assert payload["physical_rows_root_inclusive"] == 32
    assert payload["reference_always_served"] is True
    assert payload["timing_eligible"] is False
    assert len(payload["source_file_sha256"]) == 64
    assert len(payload["work_census_sha256"]) == 64
    assert (payload["qwen_campaign_proof"] is not None) == (batch == 4)


@pytest.mark.parametrize(
    ("target", "mutation", "match"),
    (
        (
            "comparator_path",
            lambda row: {**row, "differing_bytes": 1, "byte_equal": False},
            "comparison contract or bytes differ",
        ),
        (
            "work_census_path",
            lambda row: {**row, "mode": "tail6_fixed32"},
            "work census is incomplete or mismatched",
        ),
    ),
)
def test_credential_rejects_comparator_and_work_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    mutation,
    match: str,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch, batch=1)
    path = inputs[target]
    assert isinstance(path, Path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0] = mutation(rows[0])
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="ascii",
    )
    with pytest.raises(credential.CredentialError, match=match):
        credential.issue_credential(**inputs)


def test_b4_credential_rejects_compaction_proof_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path, monkeypatch, batch=4)
    qwen = inputs["qwen_campaign_path"]
    assert isinstance(qwen, Path)
    _write_json(qwen, {"schema": "tampered"})
    with pytest.raises(credential.CredentialError, match="proof schema mismatch"):
        credential.issue_credential(**inputs)


def test_credential_rejects_source_manifest_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path, monkeypatch, batch=1)
    source = inputs["source_path"]
    assert isinstance(source, Path)
    source.write_text("# modified after manifest\n", encoding="ascii")
    with pytest.raises(
        credential.CredentialError,
        match="runtime manifest does not bind tree-conv source",
    ):
        credential.issue_credential(**inputs)
