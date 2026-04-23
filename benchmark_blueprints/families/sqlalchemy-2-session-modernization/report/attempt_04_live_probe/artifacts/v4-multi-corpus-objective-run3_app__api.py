from sqlalchemy.orm import Session

from app.repository import create_entry, get_entry


def create_ledger_entry(session_or_factory, external_id, amount):
    if isinstance(session_or_factory, Session):
        return create_entry(session_or_factory, external_id, amount)

    with session_or_factory.begin() as session:
        return create_entry(session, external_id, amount)


def read_ledger_status(session_or_factory, external_id):
    if isinstance(session_or_factory, Session):
        return get_entry(session_or_factory, external_id).status

    with session_or_factory() as session:
        return get_entry(session, external_id).status
