import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.db import make_session_factory
from app.repository import create_entry, get_entry
from app.worker import settle_entry


def test_worker_failure_does_not_persist_processing_status(tmp_path):
    session_factory = make_session_factory(f"sqlite:///{tmp_path / 'ledger.sqlite'}")
    with session_factory.begin() as session:
        create_entry(session, "ext-1", 25)
    try:
        settle_entry(session_factory, "ext-1", fail_after_mark=True)
    except RuntimeError:
        pass
    with session_factory() as session:
        assert get_entry(session, "ext-1").status == "pending"
