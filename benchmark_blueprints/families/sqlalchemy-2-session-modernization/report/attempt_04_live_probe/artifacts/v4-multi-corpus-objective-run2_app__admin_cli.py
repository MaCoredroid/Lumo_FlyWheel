from app.repository import get_entry


def plan_batch(session_factory, external_id, dry_run=True):
    with session_factory() as session:
        entry = get_entry(session, external_id)
        entry.status = "queued"
        result = {"external_id": external_id, "dry_run": dry_run}
        if dry_run:
            session.rollback()
            return result
        session.commit()
        return result


def reconcile_batch(session_factory, external_ids, dry_run=True, fail_after_first=False):
    with session_factory() as session:
        results = []
        try:
            for index, external_id in enumerate(external_ids):
                entry = get_entry(session, external_id)
                entry.status = "queued"
                results.append({"external_id": external_id, "dry_run": dry_run})
                if fail_after_first and index == 0:
                    raise RuntimeError("cli-boom")
            if dry_run:
                session.rollback()
            else:
                session.commit()
            return results
        except Exception:
            session.rollback()
            raise
