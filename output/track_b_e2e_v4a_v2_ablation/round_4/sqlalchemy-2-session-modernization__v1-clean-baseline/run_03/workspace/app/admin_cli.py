from sqlalchemy import select

from app.models import LedgerEntry


def plan_batch(session_factory, external_id, dry_run=True):
    with session_factory() as session:
        stmt = select(LedgerEntry).where(LedgerEntry.external_id == external_id)
        entry = session.execute(stmt).scalar_one()
        original_status = entry.status
        entry.status = "queued"
        if dry_run:
            entry.status = original_status
            session.rollback()
            return {"external_id": external_id, "dry_run": True}
        session.commit()
        return {"external_id": external_id, "dry_run": False}


def reconcile_batch(session_factory, external_ids, dry_run=True, fail_after_first=False):
    with session_factory() as session:
        results = []
        changes = []
        for index, external_id in enumerate(external_ids):
            stmt = select(LedgerEntry).where(LedgerEntry.external_id == external_id)
            entry = session.execute(stmt).scalar_one()
            original_status = entry.status
            entry.status = "queued"
            changes.append((entry, original_status))
            results.append({"external_id": external_id, "dry_run": dry_run})
            if fail_after_first and index == 0:
                raise RuntimeError("cli-boom")
        if dry_run:
            for entry, original_status in changes:
                entry.status = original_status
            session.rollback()
        else:
            session.commit()
        return results
