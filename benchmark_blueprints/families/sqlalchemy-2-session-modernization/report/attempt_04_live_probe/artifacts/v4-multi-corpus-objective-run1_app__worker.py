from app.repository import get_entry


def settle_entry(session_factory, external_id, fail_after_mark=False):
    session = session_factory()
    try:
        entry = get_entry(session, external_id)
        entry.status = "processing"
        if fail_after_mark:
            raise RuntimeError("boom")
        entry.status = "settled"
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def settle_batch(session_factory, external_ids, fail_after_first=False):
    session = session_factory()
    try:
        for index, external_id in enumerate(external_ids):
            entry = get_entry(session, external_id)
            entry.status = "processing"
            if fail_after_first and index == 0:
                raise RuntimeError("batch-boom")
            entry.status = "settled"
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
