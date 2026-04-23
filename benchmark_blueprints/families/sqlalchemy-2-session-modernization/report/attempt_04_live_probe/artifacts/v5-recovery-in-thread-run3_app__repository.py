from sqlalchemy import func, select

from app.models import LedgerEntry


def create_entry(session, external_id, amount):
    entry = LedgerEntry(external_id=external_id, amount=amount, status="pending")
    session.add(entry)
    session.flush()
    return entry


def get_entry(session, external_id):
    return session.execute(
        select(LedgerEntry).filter_by(external_id=external_id)
    ).scalar_one()


def entry_exists(session, external_id):
    return session.execute(
        select(LedgerEntry.id).filter_by(external_id=external_id).limit(1)
    ).first() is not None


def pending_entry_count(session):
    return session.execute(
        select(func.count()).select_from(LedgerEntry).filter_by(status="pending")
    ).scalar_one()
