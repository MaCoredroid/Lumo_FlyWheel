from sqlalchemy import select

from app.models import LedgerEntry


def create_ledger_entry(session, external_id, amount):
    entry = LedgerEntry(external_id=external_id, amount=amount, status="pending")
    session.add(entry)
    return entry


def read_ledger_status(session, external_id):
    stmt = select(LedgerEntry).where(LedgerEntry.external_id == external_id)
    return session.execute(stmt).scalar_one().status
