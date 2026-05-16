from sqlalchemy import select

from app.models import LedgerEntry


def plan_batch(session_factory, external_id, dry_run=True):
    with session_factory.begin() as session:
        stmt = select(LedgerEntry).where(LedgerEntry.external_id == external_id)
        entry = session.execute(stmt).scalar_one()
        if not dry_run:
            entry.status = "queued"
        return {"external_id": external_id, "dry_run": dry_run}


def reconcile_batch(session_factory, external_ids, dry_run=True, fail_after_first=False):
    with session_factory.begin() as session:
        results = []
        for index, external_id in enumerate(external_ids):
            stmt = select(LedgerEntry).where(LedgerEntry.external_id == external_id)
            entry = session.execute(stmt).scalar_one()
            if not dry_run:
                entry.status = "queued"
            results.append({"external_id": external_id, "dry_run": dry_run})
            if fail_after_first and index == 0:
                raise RuntimeError("cli-boom")
        return results
