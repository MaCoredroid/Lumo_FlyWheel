from sqlalchemy import select

from app.models import LedgerEntry
from app.repository import create_entry


def create_ledger_entry(session, external_id, amount):
    return create_entry(session, external_id, amount)


def read_ledger_status(session, external_id):
    stmt = select(LedgerEntry).where(LedgerEntry.external_id == external_id)
    return session.execute(stmt).scalar_one().status
