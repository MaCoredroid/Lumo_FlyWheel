from contextlib import contextmanager

from sqlalchemy.orm import Session

from app.repository import create_entry, get_entry


@contextmanager
def _session_scope(session_or_factory):
    if isinstance(session_or_factory, Session):
        if session_or_factory.in_transaction():
            yield session_or_factory
            return
        with session_or_factory.begin():
            yield session_or_factory
            return

    with session_or_factory.begin() as session:
        yield session


def create_ledger_entry(session_or_factory, external_id, amount):
    with _session_scope(session_or_factory) as session:
        return create_entry(session, external_id, amount)


def read_ledger_status(session_or_factory, external_id):
    with _session_scope(session_or_factory) as session:
        return get_entry(session, external_id).status
