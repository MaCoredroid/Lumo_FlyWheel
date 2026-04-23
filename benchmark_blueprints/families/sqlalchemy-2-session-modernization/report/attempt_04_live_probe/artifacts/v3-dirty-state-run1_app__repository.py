from sqlalchemy import func, select

from app.models import LedgerEntry


def create_entry(session, external_id, amount):
    entry = LedgerEntry(external_id=external_id, amount=amount, status="pending")
    session.add(entry)
    session.flush()
    return entry


def get_entry(session, external_id):
    return session.execute(
        select(LedgerEntry).where(LedgerEntry.external_id == external_id)
    ).scalar_one()


def entry_exists(session, external_id):
    return (
        session.execute(
            select(LedgerEntry.id).where(LedgerEntry.external_id == external_id)
        ).scalar_one_or_none()
        is not None
    )


def pending_entry_count(session):
    return session.execute(
        select(func.count(LedgerEntry.id)).where(LedgerEntry.status == "pending")
    ).scalar_one()
