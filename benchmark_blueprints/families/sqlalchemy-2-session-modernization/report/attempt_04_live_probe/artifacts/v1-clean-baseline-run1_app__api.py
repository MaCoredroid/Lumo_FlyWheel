from app.repository import create_entry, get_entry


def _write_boundary(session):
    if session.in_transaction():
        return session.begin_nested()
    return session.begin()


def create_ledger_entry(session, external_id, amount):
    with _write_boundary(session):
        return create_entry(session, external_id, amount)


def read_ledger_status(session, external_id):
    return get_entry(session, external_id).status
