from __future__ import annotations


def normalize_env_owner_token(token: str) -> str:
    return "-".join(token.strip().lower().replace("_", " ").replace("/", " ").split())


def load_env_watchlist(csv_value: str) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    for raw in csv_value.split(","):
        token = raw.strip()
        if not token:
            continue
        blockers.append(
            {
                "owner_key": normalize_env_owner_token(token),
                "display_owner": token,
                "blocked_count": 1,
                "source": "env_watchlist",
                "reason": "watchlist_owner",
            }
        )
    return blockers
