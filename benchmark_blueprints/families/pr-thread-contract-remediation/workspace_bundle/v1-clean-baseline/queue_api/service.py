from __future__ import annotations

MISSING = object()


def build_summary(rows, owner_filter=MISSING, include_unowned=False):
    buckets = []
    by_key = {}
    for row in rows:
        owner = row.get("owner")
        if owner_filter is not MISSING:
            if owner_filter is None:
                if owner is not None:
                    continue
            elif owner != owner_filter:
                continue
        key = owner if owner is not None else "__unowned__"
        bucket = by_key.get(key)
        if bucket is None:
            bucket = {"owner": owner, "count": 0, "queue_ids": []}
            by_key[key] = bucket
            buckets.append(bucket)
        bucket["count"] += 1
        bucket["queue_ids"].append(row["queue_id"])
    if not include_unowned:
        return [bucket for bucket in buckets if bucket["owner"] is not None]
    return sorted(
        buckets,
        key=lambda bucket: (bucket["owner"] is None, -bucket["count"], bucket["owner"] or ""),
    )
