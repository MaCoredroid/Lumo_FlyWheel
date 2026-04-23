from __future__ import annotations


def merge_blocked_owner_rows(
    fs_rows: list[dict[str, object]], env_rows: list[dict[str, object]]
) -> dict[str, object]:
    grouped: dict[str, dict[str, object]] = {}
    for row in [*fs_rows, *env_rows]:
        owner_key = str(row["owner_key"])
        bucket = grouped.setdefault(
            owner_key,
            {
                "owner_key": owner_key,
                "display_owner": row["display_owner"],
                "blocked_count": 0,
                "sources": [],
            },
        )
        bucket["blocked_count"] += int(row["blocked_count"])
        bucket["sources"].append(str(row["source"]))
    return {
        "blocked_owner_total": len(grouped),
        "blocked_owners": sorted(grouped.values(), key=lambda item: item["owner_key"]),
    }
