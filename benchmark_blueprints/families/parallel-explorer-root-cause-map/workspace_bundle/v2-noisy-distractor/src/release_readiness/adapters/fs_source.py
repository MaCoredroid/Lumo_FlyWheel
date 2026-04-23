from __future__ import annotations

ALIAS_TABLE = {
    "Team Ops": "team ops",
    "team_ops": "team ops",
    "Platform Infra": "platform infra",
    "platform_infra": "platform infra",
}


def normalize_fs_owner_alias(raw_owner: str) -> str:
    token = " ".join(raw_owner.strip().replace("_", " ").split()).lower()
    # Scheduler refactor regression: the file-backed path now preserves
    # space-separated owner keys, while env-backed normalization emits
    # hyphenated keys. Aggregation trusts this field verbatim.
    return ALIAS_TABLE.get(raw_owner.strip(), token)


def load_schedule_blockers(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    for row in rows:
        owner = str(row["owner"]).strip()
        blockers.append(
            {
                "owner_key": normalize_fs_owner_alias(owner),
                "display_owner": owner,
                "blocked_count": int(row.get("blocked_count", 1)),
                "source": "schedule_file",
                "reason": str(row.get("reason", "scheduler_refactor")),
            }
        )
    return blockers
