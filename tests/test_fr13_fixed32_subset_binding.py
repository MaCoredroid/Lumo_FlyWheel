from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import fr13_floor_gate as floor_gate  # noqa: E402


@pytest.mark.parametrize("task_count", (4, 16))
def test_canonical_subset_binding_accepts_clean_worktree_copy(
    tmp_path: Path,
    task_count: int,
) -> None:
    expected = floor_gate.EVIDENCE_SETS[task_count]
    source = REPO / expected["relative_path"]
    copied = (
        tmp_path
        / "clean-worktree"
        / "config"
        / "fr13_fixed32"
        / source.name
    )
    copied.parent.mkdir(parents=True)
    shutil.copyfile(source, copied)

    binding = floor_gate.validate_canonical_subset(copied)

    assert binding["task_count"] == task_count
    assert binding["sha256"] == expected["sha256"]
    assert binding["task_ids"] == list(expected["task_ids"])
    assert binding["path"] == str(copied)


def test_canonical_subset_binding_rejects_byte_tamper_with_same_tasks(
    tmp_path: Path,
) -> None:
    expected = floor_gate.EVIDENCE_SETS[4]
    source = REPO / expected["relative_path"]
    tampered = tmp_path / "subset-copy.json"
    tampered.write_bytes(source.read_bytes() + b"\n")

    with pytest.raises(
        floor_gate.GateError,
        match="subset SHA-256 is not canonical exact4/exact16",
    ):
        floor_gate.validate_canonical_subset(tampered)


def test_canonical_subset_binding_rejects_task_tamper_even_with_matching_hash_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = floor_gate.EVIDENCE_SETS[4]
    payload = json.loads((REPO / expected["relative_path"]).read_text())
    payload["instance_ids"][0] = "astropy__astropy-99999"
    tampered = tmp_path / "task-tampered.json"
    tampered.write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="ascii",
    )
    monkeypatch.setitem(
        expected,
        "sha256",
        hashlib.sha256(tampered.read_bytes()).hexdigest(),
    )

    with pytest.raises(
        floor_gate.GateError,
        match="subset IDs do not exactly match canonical 4-task set",
    ):
        floor_gate.validate_canonical_subset(tampered)


def test_fixed32_runtime_consumers_use_content_bound_subset_validation() -> None:
    driver = (REPO / "scripts/fr13_b4_campaign_driver.sh").read_text()
    serve = (REPO / "scripts/fr13_bigdenom_swe_serve_variant.sh").read_text()
    runner = (REPO / "scripts/run_swe_bench_q36_a.py").read_text()

    for source in (driver, serve, runner):
        assert "validate_canonical_subset" in source
    assert 'case "$(realpath "$SUBSET")"' not in driver
    assert "canonical_subsets = {" not in runner
    assert 'raw_subset.get("task_ids")' not in serve
