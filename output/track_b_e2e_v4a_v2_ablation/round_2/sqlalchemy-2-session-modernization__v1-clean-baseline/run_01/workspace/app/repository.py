from sqlalchemy import select

from app.models import LedgerEntry


def create_entry(session, external_id, amount):
    entry = LedgerEntry(external_id=external_id, amount=amount, status="pending")
    session.add(entry)
    return entry


def get_entry(session, external_id):
    stmt = select(LedgerEntry).where(LedgerEntry.external_id == external_id)
    return session.execute(stmt).scalars().one()


def entry_exists(session, external_id):
    return (
        session.execute(select(LedgerEntry.id).where(LedgerEntry.external_id == external_id)).scalars().first()
        is not None
    )


def pending_entry_count(session):
    stmt = select(LedgerEntry).where(LedgerEntry.status == "pending")
    return session.execute(stmt).scalars().count()
