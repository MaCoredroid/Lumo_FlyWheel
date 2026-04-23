from app.repository import get_entry


def plan_batch(session_factory, external_id, dry_run=True):
    if dry_run:
        with session_factory() as session:
            get_entry(session, external_id)
            return {
                "external_id": external_id,
                "dry_run": True,
                "planned_status": "queued",
            }

    with session_factory.begin() as session:
        entry = get_entry(session, external_id)
        entry.status = "queued"
    return {
        "external_id": external_id,
        "dry_run": False,
        "planned_status": "queued",
    }


def reconcile_batch(session_factory, external_ids, dry_run=True, fail_after_first=False):
    if dry_run:
        with session_factory() as session:
            for external_id in external_ids:
                get_entry(session, external_id)
        return [
            {
                "external_id": external_id,
                "dry_run": True,
                "planned_status": "queued",
            }
            for external_id in external_ids
        ]

    with session_factory.begin() as session:
        results = []
        for index, external_id in enumerate(external_ids):
            entry = get_entry(session, external_id)
            entry.status = "queued"
            results.append(
                {
                    "external_id": external_id,
                    "dry_run": False,
                    "planned_status": "queued",
                }
            )
            if fail_after_first and index == 0:
                raise RuntimeError("cli-boom")
        return results
