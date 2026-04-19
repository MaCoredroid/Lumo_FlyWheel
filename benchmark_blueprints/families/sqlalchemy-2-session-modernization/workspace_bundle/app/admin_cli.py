from app.repository import get_entry


def plan_batch(session_factory, external_id, dry_run=True):
    with session_factory() as session:
        entry = get_entry(session, external_id)
        entry.status = "queued"
        if dry_run:
            session.commit()
            return {"external_id": external_id, "dry_run": True}
        session.commit()
        return {"external_id": external_id, "dry_run": False}
