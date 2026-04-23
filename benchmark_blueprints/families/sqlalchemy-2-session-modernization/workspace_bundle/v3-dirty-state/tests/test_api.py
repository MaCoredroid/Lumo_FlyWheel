import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.api import create_ledger_entry, read_ledger_status
from app.db import make_session_factory


def test_create_and_read_entry(tmp_path):
    session_factory = make_session_factory(f"sqlite:///{tmp_path / 'ledger.sqlite'}")
    with session_factory.begin() as session:
        create_ledger_entry(session, "ext-1", 25)
        assert read_ledger_status(session, "ext-1") == "pending"
    with session_factory() as session:
        assert read_ledger_status(session, "ext-1") == "pending"
