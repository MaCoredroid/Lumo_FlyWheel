import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.admin_cli import plan_batch
from app.db import make_session_factory
from app.repository import create_entry, get_entry


def test_dry_run_does_not_persist_status_change(tmp_path):
    session_factory = make_session_factory(f"sqlite:///{tmp_path / 'ledger.sqlite'}")
    with session_factory() as session:
        create_entry(session, "ext-1", 25)
    plan_batch(session_factory, "ext-1", dry_run=True)
    with session_factory() as session:
        assert get_entry(session, "ext-1").status == "pending"
