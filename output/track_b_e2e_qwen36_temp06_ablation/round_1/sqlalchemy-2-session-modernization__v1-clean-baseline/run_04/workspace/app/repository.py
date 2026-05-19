from sqlalchemy import select, func

from app.models import LedgerEntry


def create_entry(session, external_id, amount):
    entry = LedgerEntry(external_id=external_id, amount=amount, status="pending")
    session.add(entry)
    return entry


def get_entry(session, external_id):
    return session.scalars(
        select(LedgerEntry).where(LedgerEntry.external_id == external_id)
    ).one()


def entry_exists(session, external_id):
    return session.scalar(
        select(LedgerEntry.id).where(LedgerEntry.external_id == external_id)
    ) is not None


def pending_entry_count(session):
    return session.scalar(
        select(func.count()).select_from(LedgerEntry).where(LedgerEntry.status == "pending")
    )
