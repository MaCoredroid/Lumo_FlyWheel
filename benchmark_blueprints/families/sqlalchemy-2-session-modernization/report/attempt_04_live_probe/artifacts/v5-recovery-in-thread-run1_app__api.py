from app.repository import create_entry, get_entry


def create_ledger_entry(session_or_factory, external_id, amount):
    if hasattr(session_or_factory, "add"):
        return create_entry(session_or_factory, external_id, amount)
    with session_or_factory.begin() as session:
        return create_entry(session, external_id, amount)


def read_ledger_status(session_or_factory, external_id):
    if hasattr(session_or_factory, "execute"):
        return get_entry(session_or_factory, external_id).status
    with session_or_factory() as session:
        return get_entry(session, external_id).status
