from __future__ import annotations

from queue_api import serializers, service


class QueueRepository:
    def __init__(self, rows):
        self._rows = list(rows)

    def list_rows(self):
        return list(self._rows)


def parse_owner_filter(params):
    if "owner" not in params:
        return service.MISSING
    return params.get("owner")


def get_queue_summary(params, repository):
    buckets = service.build_summary(
        repository.list_rows(),
        owner_filter=parse_owner_filter(params),
        include_unowned=bool(params.get("include_unowned")),
    )
    serializer = serializers.serialize_export_bucket if params.get("mode") == "export" else serializers.serialize_bucket
    return {"summary": [serializer(bucket) for bucket in buckets]}
