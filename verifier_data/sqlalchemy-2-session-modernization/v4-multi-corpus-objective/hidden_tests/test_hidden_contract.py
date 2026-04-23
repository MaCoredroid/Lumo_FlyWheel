import os
import pathlib
import sys

import pytest

WORKSPACE = pathlib.Path(os.environ["AGENT_WS"]).resolve()
VARIANT_ID = os.environ["VARIANT_ID"]

sys.path.insert(0, str(WORKSPACE))

from app.admin_cli import plan_batch, reconcile_batch
from app.db import make_session_factory
from app.repository import create_entry, entry_exists, get_entry, pending_entry_count
from app.worker import settle_batch, settle_entry


def _make_factory(tmp_path):
    return make_session_factory(f"sqlite:///{tmp_path / 'ledger.sqlite'}")


def _seed_entries(session_factory, *external_ids):
    with session_factory.begin() as session:
        for index, external_id in enumerate(external_ids, start=1):
            create_entry(session, external_id, 10 * index)


def test_repository_uses_select_and_no_helper_commit():
    repository_py = (WORKSPACE / "app/repository.py").read_text()
    assert "session.query(" not in repository_py
    assert ".commit(" not in repository_py
    assert "select(" in repository_py


def test_entry_exists_and_pending_count_are_consistent(tmp_path):
    session_factory = _make_factory(tmp_path)
    _seed_entries(session_factory, "ext-1", "ext-2")

    with session_factory() as session:
        assert entry_exists(session, "ext-1") is True
        assert entry_exists(session, "missing") is False
        assert pending_entry_count(session) == 2


def test_worker_failure_rolls_back_to_pending(tmp_path):
    session_factory = _make_factory(tmp_path)
    _seed_entries(session_factory, "ext-1")

    with pytest.raises(RuntimeError):
        settle_entry(session_factory, "ext-1", fail_after_mark=True)

    with session_factory() as session:
        assert get_entry(session, "ext-1").status == "pending"


def test_admin_dry_run_stays_read_only(tmp_path):
    session_factory = _make_factory(tmp_path)
    _seed_entries(session_factory, "ext-1")

    result = plan_batch(session_factory, "ext-1", dry_run=True)

    assert result["dry_run"] is True
    with session_factory() as session:
        assert get_entry(session, "ext-1").status == "pending"


def test_cutover_note_names_transaction_boundaries_and_rollback():
    deploy_note = (WORKSPACE / "docs/deploy/sqlalchemy2-cutover.md").read_text().lower()
    assert "transaction" in deploy_note
    assert "rollback" in deploy_note
    assert "dry-run" in deploy_note or "dry run" in deploy_note


def test_no_global_session_singleton():
    db_py = (WORKSPACE / "app/db.py").read_text().lower()
    assert "global_session" not in db_py


def test_v2_rejects_compatibility_shim_language():
    if VARIANT_ID != "v2-noisy-distractor":
        pytest.skip("variant-specific")
    deploy_note = (WORKSPACE / "docs/deploy/sqlalchemy2-cutover.md").read_text().lower()
    assert "compatibility shim" not in deploy_note


def test_v3_preserves_dirty_readme_context():
    if VARIANT_ID != "v3-dirty-state":
        pytest.skip("variant-specific")
    readme = (WORKSPACE / "README.md").read_text()
    assert "Local notes:" in readme
    assert "Do not rewrite or clean up this file" in readme


def test_v4_batch_helpers_are_atomic(tmp_path):
    if VARIANT_ID not in {"v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        pytest.skip("variant-specific")
    session_factory = _make_factory(tmp_path)
    _seed_entries(session_factory, "ext-1", "ext-2")

    with pytest.raises(RuntimeError):
        settle_batch(session_factory, ["ext-1", "ext-2"], fail_after_first=True)

    with session_factory() as session:
        assert get_entry(session, "ext-1").status == "pending"
        assert get_entry(session, "ext-2").status == "pending"

    with pytest.raises(RuntimeError):
        reconcile_batch(
            session_factory,
            ["ext-1", "ext-2"],
            dry_run=False,
            fail_after_first=True,
        )

    with session_factory() as session:
        assert get_entry(session, "ext-1").status == "pending"
        assert get_entry(session, "ext-2").status == "pending"


def test_v5_incident_recovery_is_acknowledged_and_retry_is_idempotent(tmp_path):
    if VARIANT_ID != "v5-recovery-in-thread":
        pytest.skip("variant-specific")
    session_factory = _make_factory(tmp_path)
    _seed_entries(session_factory, "ext-1")

    with pytest.raises(RuntimeError):
        settle_entry(session_factory, "ext-1", fail_after_mark=True)
    settle_entry(session_factory, "ext-1", fail_after_mark=False)

    with session_factory() as session:
        assert get_entry(session, "ext-1").status == "settled"

    deploy_note = (WORKSPACE / "docs/deploy/sqlalchemy2-cutover.md").read_text().lower()
    incident_note = (WORKSPACE / "incident_context/retry-state-rollback.md").read_text().lower()
    assert "incident" in deploy_note
    assert "rollback" in deploy_note
    assert "rolled back" in incident_note
