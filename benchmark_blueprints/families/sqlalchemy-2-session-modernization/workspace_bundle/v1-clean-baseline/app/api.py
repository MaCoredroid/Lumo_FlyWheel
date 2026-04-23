from app.repository import create_entry, get_entry


def create_ledger_entry(session, external_id, amount):
    return create_entry(session, external_id, amount)


def read_ledger_status(session, external_id):
    return get_entry(session, external_id).status
