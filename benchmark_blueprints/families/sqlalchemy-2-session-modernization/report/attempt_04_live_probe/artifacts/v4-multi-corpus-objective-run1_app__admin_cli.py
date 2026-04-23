from app.repository import get_entry


def plan_batch(session_factory, external_id, dry_run=True):
    session = session_factory()
    try:
        entry = get_entry(session, external_id)
        entry.status = "queued"
        if dry_run:
            session.rollback()
            return {"external_id": external_id, "dry_run": True}
        session.commit()
        return {"external_id": external_id, "dry_run": False}
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reconcile_batch(session_factory, external_ids, dry_run=True, fail_after_first=False):
    session = session_factory()
    try:
        results = []
        for index, external_id in enumerate(external_ids):
            entry = get_entry(session, external_id)
            entry.status = "queued"
            results.append({"external_id": external_id, "dry_run": dry_run})
            if fail_after_first and index == 0:
                raise RuntimeError("cli-boom")
        if dry_run:
            session.rollback()
            return results
        session.commit()
        return results
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
