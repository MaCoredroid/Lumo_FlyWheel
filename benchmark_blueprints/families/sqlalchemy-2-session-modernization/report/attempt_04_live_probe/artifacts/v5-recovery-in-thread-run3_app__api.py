from app.repository import create_entry, get_entry


def create_ledger_entry(session, external_id, amount):
    entry = create_entry(session, external_id, amount)
    session.flush()
    return entry


def read_ledger_status(session, external_id):
    return get_entry(session, external_id).status
