
from __future__ import annotations

from pathlib import Path

from codex.policy import normalize_policy


def _parse_policy_table(text: str) -> dict[str, str]:
    in_policy = False
    out: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_policy = line == "[policy]"
            continue
        if not in_policy or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"')
    return out


def load_config(path: str | Path) -> dict[str, str]:
    target = Path(path)
    policy = _parse_policy_table(target.read_text(encoding="utf-8"))
    if "sandbox" not in policy or "approval_policy" not in policy:
        raise ValueError("config file must define sandbox and approval_policy")
    return normalize_policy(
        {
            "sandbox": str(policy["sandbox"]),
            "approval_policy": str(policy["approval_policy"]),
        }
    )
