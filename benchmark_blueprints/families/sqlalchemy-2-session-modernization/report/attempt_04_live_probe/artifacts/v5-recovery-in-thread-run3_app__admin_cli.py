from app.repository import get_entry


def plan_batch(session_factory, external_id, dry_run=True):
    with session_factory() as session:
        transaction = session.begin()
        try:
            entry = get_entry(session, external_id)
            entry.status = "queued"
            if dry_run:
                transaction.rollback()
                return {"external_id": external_id, "dry_run": True}
            transaction.commit()
            return {"external_id": external_id, "dry_run": False}
        except Exception:
            transaction.rollback()
            raise


def reconcile_batch(session_factory, external_ids, dry_run=True, fail_after_first=False):
    with session_factory() as session:
        transaction = session.begin()
        try:
            results = []
            for index, external_id in enumerate(external_ids):
                entry = get_entry(session, external_id)
                entry.status = "queued"
                results.append({"external_id": external_id, "dry_run": dry_run})
                if fail_after_first and index == 0:
                    raise RuntimeError("cli-boom")
            if dry_run:
                transaction.rollback()
                return results
            transaction.commit()
            return results
        except Exception:
            transaction.rollback()
            raise
