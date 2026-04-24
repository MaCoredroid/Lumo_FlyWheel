from __future__ import annotations


def _base(bucket):
    return {"count": bucket["count"], "queue_ids": list(bucket["queue_ids"])}


def serialize_bucket(bucket):
    payload = _base(bucket)
    payload["owner"] = bucket["owner"]
    return payload


def serialize_export_bucket(bucket):
    payload = _base(bucket)
    payload["owner"] = bucket["owner"]
    payload["owner_label"] = bucket["owner"] or "unowned"
    return payload
