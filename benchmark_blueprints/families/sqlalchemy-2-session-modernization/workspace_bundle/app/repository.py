from app.models import LedgerEntry


def create_entry(session, external_id, amount):
    entry = LedgerEntry(external_id=external_id, amount=amount, status="pending")
    session.add(entry)
    session.commit()
    return entry


def get_entry(session, external_id):
    return session.query(LedgerEntry).filter_by(external_id=external_id).one()
