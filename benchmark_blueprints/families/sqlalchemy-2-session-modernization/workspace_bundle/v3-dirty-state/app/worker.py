from app.repository import get_entry


def settle_entry(session_factory, external_id, fail_after_mark=False):
    with session_factory() as session:
        entry = get_entry(session, external_id)
        entry.status = "processing"
        session.commit()
        if fail_after_mark:
            raise RuntimeError("boom")
        entry.status = "settled"
        session.commit()


def settle_batch(session_factory, external_ids, fail_after_first=False):
    with session_factory() as session:
        for index, external_id in enumerate(external_ids):
            entry = get_entry(session, external_id)
            entry.status = "processing"
            session.commit()
            if fail_after_first and index == 0:
                raise RuntimeError("batch-boom")
            entry.status = "settled"
            session.commit()
