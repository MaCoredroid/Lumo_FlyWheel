from __future__ import annotations


def _base(bucket):
    return {"count": bucket["count"], "queue_ids": list(bucket["queue_ids"])}


def _with_owner(payload, owner):
    if owner is not None:
        payload["owner"] = owner
    return payload


def serialize_bucket(bucket):
    return _with_owner(_base(bucket), bucket["owner"])


def serialize_export_bucket(bucket):
    payload = _with_owner(_base(bucket), bucket["owner"])
    payload["owner_label"] = bucket["owner"] or "unowned"
    return payload
