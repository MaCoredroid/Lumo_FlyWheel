from sqlalchemy import select

from app.models import LedgerEntry


def settle_entry(session_factory, external_id, fail_after_mark=False):
    with session_factory() as session:
        stmt = select(LedgerEntry).where(LedgerEntry.external_id == external_id)
        entry = session.execute(stmt).scalar_one()
        entry.status = "processing"
        if fail_after_mark:
            raise RuntimeError("boom")
        entry.status = "settled"
        session.commit()


def settle_batch(session_factory, external_ids, fail_after_first=False):
    with session_factory() as session:
        for index, external_id in enumerate(external_ids):
            stmt = select(LedgerEntry).where(LedgerEntry.external_id == external_id)
            entry = session.execute(stmt).scalar_one()
            entry.status = "processing"
            if fail_after_first and index == 0:
                raise RuntimeError("batch-boom")
            entry.status = "settled"
            session.commit()
