from app.repository import get_entry


class _DryRunRollback(Exception):
    """Trigger an intentional rollback after planning work."""


def plan_batch(session_factory, external_id, dry_run=True):
    result = {"external_id": external_id, "dry_run": dry_run}
    try:
        with session_factory.begin() as session:
            entry = get_entry(session, external_id)
            entry.status = "queued"
            if dry_run:
                raise _DryRunRollback()
    except _DryRunRollback:
        return result
    return result


def reconcile_batch(session_factory, external_ids, dry_run=True, fail_after_first=False):
    results = []
    try:
        with session_factory.begin() as session:
            for index, external_id in enumerate(external_ids):
                entry = get_entry(session, external_id)
                entry.status = "queued"
                results.append({"external_id": external_id, "dry_run": dry_run})
                if fail_after_first and index == 0:
                    raise RuntimeError("cli-boom")
            if dry_run:
                raise _DryRunRollback()
    except _DryRunRollback:
        return results
    return results
