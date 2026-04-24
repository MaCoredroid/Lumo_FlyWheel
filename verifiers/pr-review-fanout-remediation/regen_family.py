#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
FAMILY_ID = "pr-review-fanout-remediation"
FAMILY_ROOT = REPO / "benchmark_blueprints" / "families" / FAMILY_ID
VERIFIER_ROOT = REPO / "verifiers" / FAMILY_ID
VERIFIER_DATA_ROOT = REPO / "verifier_data" / FAMILY_ID
SCORER = VERIFIER_ROOT / "score_review_fanout.py"

WRITE_ALLOWED = [
    "src/policy/approval_router.py",
    "src/policy/preview.py",
    "tests/test_preview.py",
    "docs/approval_policy.md",
    "artifacts/review/thread_triage.json",
    "artifacts/review/reply_draft.md",
    "artifacts/review/remediation_report.md",
]

READONLY_TREES = [
    ".scenario_variant",
    "AGENTS.md",
    "Dockerfile",
    ".github/review_export",
    "artifacts/review/thread_digest.md",
    "tests/test_approval_router.py",
    "release_context",
    "incident_context",
]


@dataclass(frozen=True)
class Variant:
    variant_id: str
    label: str
    summary_suffix: str
    required_report_phrases: tuple[str, ...]
    report_scope_phrases: tuple[str, ...]
    release_context: str | None = None
    incident_context: str | None = None
    previous_attempt_patch: str | None = None


VARIANTS = [
    Variant(
        variant_id="v1-clean-baseline",
        label="V1 clean baseline",
        summary_suffix="Straight triage plus preview fallback repair.",
        required_report_phrases=("stale digest", "did not apply", "T-214-03"),
        report_scope_phrases=("T-214-01", "T-214-04", "T-214-05"),
    ),
    Variant(
        variant_id="v2-noisy-distractor",
        label="V2 noisy distractor",
        summary_suffix="Summary and digest disagree on whether the duplicate thread still needs code.",
        required_report_phrases=("stale digest", "summary", "duplicate"),
        report_scope_phrases=("T-214-01", "T-214-02", "T-214-03"),
    ),
    Variant(
        variant_id="v3-dirty-state",
        label="V3 dirty state",
        summary_suffix="A half-finished alias patch is parked in artifacts and should not be revived.",
        required_report_phrases=("legacy_preview_hint", "did not revive", "previous attempt"),
        report_scope_phrases=("T-214-01", "T-214-04", "T-214-05"),
        previous_attempt_patch="""diff --git a/src/policy/preview.py b/src/policy/preview.py
+++ b/src/policy/preview.py
@@
-    return {
-        "request_id": request_id,
-        "status": "preview_unavailable",
-        "actor": actor,
-    }
+    return {
+        "request_id": request_id,
+        "status": "preview_unavailable",
+        "actor": actor,
+        "legacy_preview_hint": approval_state,
+    }
""",
    ),
    Variant(
        variant_id="v4-multi-corpus-objective",
        label="V4 multi-corpus objective",
        summary_suffix="Downstream fallback consumers read approval_state before any secondary booleans.",
        required_report_phrases=("consumer contract", "approval_state", "downstream"),
        report_scope_phrases=("T-214-01", "T-214-04", "T-214-05"),
        release_context="""# Preview Consumer Contract

The downstream preview reviewer bot reads `approval_state` from the fallback
payload on `preview_unavailable` responses before it looks at any boolean helper
fields. `requires_human_review` is still expected, but it is secondary.

Do not restore `legacy_preview_hint`. The consumer was explicitly updated to
read `approval_state` instead, and aliases now create divergent behavior between
the live preview path and the fallback path.
""",
    ),
    Variant(
        variant_id="v5-recovery-in-thread",
        label="V5 recovery in thread",
        summary_suffix="A prior alias-based rollback reopened the main fallback thread.",
        required_report_phrases=("INC-214", "legacy_preview_hint", "no alias"),
        report_scope_phrases=("T-214-01", "T-214-04", "T-214-05"),
        release_context="""# Preview Consumer Contract

The downstream preview reviewer bot reads `approval_state` from fallback
payloads and treats `requires_human_review` as a secondary field. Do not add
legacy aliases to the payload.
""",
        incident_context="""# INC-214 Preview Alias Rollback

The first remediation attempt for PR-214 reintroduced `legacy_preview_hint`
inside the fallback payload so the reopened comment would "see something".
That alias briefly satisfied one reviewer screenshot, but it broke the preview
reviewer bot because the bot now consumes `approval_state`.

Recovery rule for this variant:

- do not revive `legacy_preview_hint`
- mention `INC-214` when replying to the reopened fallback thread
- keep `approval_state` and `requires_human_review` aligned on both the live
  preview path and the fallback path
""",
    ),
]


def sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def tree_hash(root: Path, rel: str) -> str:
    target = root / rel
    if not target.exists():
        return "MISSING"
    if target.is_file():
        return sha256_file(target)
    h = sha256()
    for path in sorted(target.rglob("*")):
        relpath = path.relative_to(target).as_posix()
        if path.is_dir():
            h.update(f"D:{relpath}\n".encode())
        else:
            h.update(f"F:{relpath}\n".encode())
            h.update(sha256_file(path).encode())
            h.update(b"\n")
    return h.hexdigest()


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def write_json(path: Path, obj: object) -> None:
    write(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def overlay_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, target)


def agents_md() -> str:
    return """# Agent Instructions — `pr-review-fanout-remediation`

## Task

You are on a draft PR branch with exported review threads. Classify every visible
thread before editing code. Only three threads are actionable. One thread is a
duplicate of the reopened preview-fallback regression, and one suggestion points
at an outdated diff hunk that should not be applied on the current branch.

## Inputs

- `src/policy/approval_router.py`
- `src/policy/preview.py`
- `tests/test_preview.py` — writable regression test surface
- `tests/test_approval_router.py` — visible read-only test
- `docs/approval_policy.md`
- `.github/review_export/pr_214_threads.json`
- `.github/review_export/pr_214_summary.md`
- `artifacts/review/thread_digest.md` — stale, non-authoritative
- `.scenario_variant`
- optional `release_context/` and `incident_context/`

## Required Outputs

- update `src/policy/approval_router.py`
- update `src/policy/preview.py`
- update `tests/test_preview.py`
- update `docs/approval_policy.md`
- write `artifacts/review/thread_triage.json`
- write `artifacts/review/reply_draft.md`
- write `artifacts/review/remediation_report.md`

## Rules

- Classify threads before editing code.
- Use exact thread ids from the review export.
- Treat `thread_digest.md` as a hint, not authority.
- Do not modify `.github/review_export/`, `artifacts/review/thread_digest.md`,
  `tests/test_approval_router.py`, `AGENTS.md`, `Dockerfile`, `.scenario_variant`,
  `release_context/`, or `incident_context/`.
- Do not add `legacy_preview_hint`.
- Do not use the network.

## What Good Looks Like

- `thread_triage.json` marks the fallback issue actionable, the duplicate thread
  as duplicate, and the outdated hunk as obsolete.
- `build_policy_preview(..., preview_enabled=False, approval_state="manual_review", ...)`
  keeps `approval_state="human_review_required"` and `requires_human_review=true`.
- the writable preview test covers the fallback case.
- docs show the corrected fallback payload.
- replies map each thread id to the change that was made, duplicated, or
  intentionally not applied.
"""


def dockerfile() -> str:
    return "FROM python:3.12-bookworm\nWORKDIR /workspace\n"


def init_py() -> str:
    return ""


def approval_router_broken() -> str:
    return """from src.policy.preview import build_preview_payload

LEGACY_APPROVAL_STATES = {"manual_review": "human_review_required"}


def normalize_approval_state(approval_state) -> str:
    state = (approval_state or "manual_review").strip()
    return LEGACY_APPROVAL_STATES.get(state, state)


def build_policy_preview(
    request_id: str,
    approval_state,
    preview_enabled: bool,
    actor: str,
) -> dict:
    return build_preview_payload(
        request_id=request_id,
        approval_state=approval_state,
        preview_enabled=preview_enabled,
        actor=actor,
    )
"""


def approval_router_fixed() -> str:
    return """from src.policy.preview import build_preview_payload

LEGACY_APPROVAL_STATES = {"manual_review": "human_review_required"}


def normalize_approval_state(approval_state) -> str:
    state = (approval_state or "manual_review").strip()
    return LEGACY_APPROVAL_STATES.get(state, state)


def build_policy_preview(
    request_id: str,
    approval_state,
    preview_enabled: bool,
    actor: str,
) -> dict:
    normalized_state = normalize_approval_state(approval_state)
    return build_preview_payload(
        request_id=request_id,
        approval_state=normalized_state,
        preview_enabled=preview_enabled,
        actor=actor,
    )
"""


def preview_broken() -> str:
    return """def build_preview_payload(
    request_id: str,
    approval_state: str,
    preview_enabled: bool,
    actor: str,
) -> dict:
    if preview_enabled:
        return {
            "request_id": request_id,
            "actor": actor,
            "preview_path": f"/preview/{request_id}",
            "approval_state": approval_state,
            "requires_human_review": approval_state == "human_review_required",
        }
    return {
        "request_id": request_id,
        "status": "preview_unavailable",
        "actor": actor,
    }
"""


def preview_fixed() -> str:
    return """def build_preview_payload(
    request_id: str,
    approval_state: str,
    preview_enabled: bool,
    actor: str,
) -> dict:
    payload = {
        "request_id": request_id,
        "actor": actor,
        "approval_state": approval_state,
        "requires_human_review": approval_state == "human_review_required",
    }
    if preview_enabled:
        payload["preview_path"] = f"/preview/{request_id}"
        return payload
    payload["status"] = "preview_unavailable"
    return payload
"""


def preview_test_broken() -> str:
    return """import unittest

from src.policy.preview import build_preview_payload


class PreviewPayloadTests(unittest.TestCase):
    def test_live_preview_includes_human_review_fields(self) -> None:
        payload = build_preview_payload(
            request_id="req-214",
            approval_state="human_review_required",
            preview_enabled=True,
            actor="review-bot",
        )
        self.assertEqual(payload["approval_state"], "human_review_required")
        self.assertTrue(payload["requires_human_review"])
        self.assertEqual(payload["preview_path"], "/preview/req-214")

    def test_preview_unavailable_keeps_status(self) -> None:
        payload = build_preview_payload(
            request_id="req-214",
            approval_state="human_review_required",
            preview_enabled=False,
            actor="review-bot",
        )
        self.assertEqual(payload["status"], "preview_unavailable")
        self.assertEqual(payload["actor"], "review-bot")


if __name__ == "__main__":
    unittest.main()
"""


def preview_test_fixed() -> str:
    return """import unittest

from src.policy.preview import build_preview_payload


class PreviewPayloadTests(unittest.TestCase):
    def test_live_preview_includes_human_review_fields(self) -> None:
        payload = build_preview_payload(
            request_id="req-214",
            approval_state="human_review_required",
            preview_enabled=True,
            actor="review-bot",
        )
        self.assertEqual(payload["approval_state"], "human_review_required")
        self.assertTrue(payload["requires_human_review"])
        self.assertEqual(payload["preview_path"], "/preview/req-214")

    def test_preview_unavailable_keeps_fallback_contract(self) -> None:
        payload = build_preview_payload(
            request_id="req-214",
            approval_state="human_review_required",
            preview_enabled=False,
            actor="review-bot",
        )
        self.assertEqual(payload["status"], "preview_unavailable")
        self.assertEqual(payload["approval_state"], "human_review_required")
        self.assertTrue(payload["requires_human_review"])
        self.assertEqual(payload["actor"], "review-bot")


if __name__ == "__main__":
    unittest.main()
"""


def router_test() -> str:
    return """import unittest

from src.policy.approval_router import build_policy_preview, normalize_approval_state


class ApprovalRouterTests(unittest.TestCase):
    def test_manual_review_normalizes(self) -> None:
        self.assertEqual(normalize_approval_state("manual_review"), "human_review_required")

    def test_live_preview_uses_normalized_state(self) -> None:
        payload = build_policy_preview(
            request_id="req-214",
            approval_state="manual_review",
            preview_enabled=True,
            actor="review-bot",
        )
        self.assertEqual(payload["approval_state"], "human_review_required")


if __name__ == "__main__":
    unittest.main()
"""


def docs_broken() -> str:
    return """# Approval Policy

Preview routing keeps the live preview payload aligned with policy routing.

## Fallback Response

When preview rendering is unavailable, return:

```json
{
  "request_id": "req-214",
  "status": "preview_unavailable",
  "actor": "review-bot"
}
```

## Notes

- `manual_review` is retired in favor of `human_review_required`.
- The fallback example above still mirrors the old reviewer write-up and needs
  to be updated once the reopened thread lands.
"""


def docs_fixed(variant: Variant) -> str:
    extra = ""
    if variant.variant_id == "v4-multi-corpus-objective":
        extra = (
            "\n- Downstream consumers read `approval_state` on fallback payloads first, "
            "so the fallback contract must keep that field populated.\n"
        )
    if variant.variant_id == "v5-recovery-in-thread":
        extra = (
            "\n- `INC-214` rolled back the alias-based fix. Keep the fallback contract "
            "on `approval_state` and do not revive `legacy_preview_hint`.\n"
        )
    return f"""# Approval Policy

Preview routing keeps the live preview payload aligned with policy routing.

## Fallback Response

When preview rendering is unavailable, return:

```json
{{
  "request_id": "req-214",
  "status": "preview_unavailable",
  "actor": "review-bot",
  "approval_state": "human_review_required",
  "requires_human_review": true
}}
```

## Notes

- `manual_review` is retired in favor of `human_review_required`.
- The fallback response keeps both `approval_state` and
  `requires_human_review` aligned with the live preview path.{extra}"""


def review_export_json() -> dict:
    return {
        "pr_id": 214,
        "threads": [
            {
                "thread_id": "T-214-01",
                "status": "open",
                "path": "src/policy/preview.py",
                "summary": "Fallback preview payload still drops approval_state and requires_human_review after the reopen.",
            },
            {
                "thread_id": "T-214-02",
                "status": "open",
                "path": "src/policy/preview.py",
                "summary": "Still seeing the same fallback preview issue in the review capture. This looks duplicated from the reopened payload thread.",
            },
            {
                "thread_id": "T-214-03",
                "status": "outdated",
                "path": "src/policy/approval_router.py",
                "summary": "Could we revive legacy_preview_branch here?",
            },
            {
                "thread_id": "T-214-04",
                "status": "open",
                "path": "tests/test_preview.py",
                "summary": "Please add a regression test for preview_enabled=False plus manual_review normalization.",
            },
            {
                "thread_id": "T-214-05",
                "status": "open",
                "path": "docs/approval_policy.md",
                "summary": "Docs example still shows the old fallback payload and misses human_review_required.",
            },
        ],
    }


def review_export_summary(variant: Variant) -> str:
    return f"""# PR-214 Review Summary

- Review export contains five visible threads.
- Use the JSON export as source of truth for thread state.
- `thread_digest.md` is stale and may list outdated work as still actionable.
- {variant.summary_suffix}
"""


def thread_digest(variant: Variant) -> str:
    body = [
        "# Thread Digest (stale scratch pad)",
        "",
        "This file was written before the reopened review export landed. It is not authoritative.",
        "",
        "- T-214-01: actionable fallback payload fix",
        "- T-214-02: actionable fallback payload fix",
        "- T-214-03: actionable router cleanup",
        "- T-214-04: test follow-up",
        "- T-214-05: docs follow-up",
    ]
    if variant.variant_id == "v2-noisy-distractor":
        body.extend(
            [
                "",
                "Scratch note: summary mail said \"all open comments still need code\". This was never reconciled with the export.",
            ]
        )
    if variant.variant_id == "v3-dirty-state":
        body.extend(
            [
                "",
                "Scratch note: partial alias patch parked under artifacts/review/previous_attempt.patch.",
            ]
        )
    if variant.variant_id == "v4-multi-corpus-objective":
        body.extend(
            [
                "",
                "Scratch note: consumer contract follow-up was added after this digest.",
            ]
        )
    if variant.variant_id == "v5-recovery-in-thread":
        body.extend(
            [
                "",
                "Scratch note: this digest predates INC-214 and should not be used as a recovery plan.",
            ]
        )
    return "\n".join(body) + "\n"


def triage_json() -> list[dict[str, object]]:
    return [
        {
            "thread_id": "T-214-01",
            "disposition": "actionable",
            "rationale": "Reopened fallback payload bug still needs the code fix.",
        },
        {
            "thread_id": "T-214-02",
            "disposition": "duplicate",
            "duplicate_of": "T-214-01",
            "rationale": "Covers the same fallback payload bug as T-214-01.",
        },
        {
            "thread_id": "T-214-03",
            "disposition": "obsolete",
            "rationale": "Suggestion targeted an outdated diff hunk and should not be applied on the current branch.",
        },
        {
            "thread_id": "T-214-04",
            "disposition": "actionable",
            "rationale": "Fallback regression test is still missing from the writable preview test file.",
        },
        {
            "thread_id": "T-214-05",
            "disposition": "actionable",
            "rationale": "Docs still show the stale fallback payload example.",
        },
    ]


def reply_draft(variant: Variant) -> str:
    extra = ""
    if variant.variant_id == "v4-multi-corpus-objective":
        extra = "\nAlso noted the downstream consumer contract: fallback consumers read `approval_state` first, so I kept that field populated instead of introducing an alias.\n"
    if variant.variant_id == "v5-recovery-in-thread":
        extra = "\nAlso called out `INC-214`: I kept the fix on `approval_state` / `requires_human_review` and did not reintroduce `legacy_preview_hint`.\n"
    return f"""# Review Replies

## T-214-01

Fixed the reopened fallback path in `src/policy/approval_router.py` and
`src/policy/preview.py` so the `preview_unavailable` response now keeps
`approval_state=\"human_review_required\"` and `requires_human_review=true`.

## T-214-02

Closing as duplicate of `T-214-01`. The same fallback payload fix covers this
thread, so I did not apply a second code path.

## T-214-03

Did not apply this suggestion. The comment points at an outdated diff hunk and
reviving `legacy_preview_branch` would move the branch away from the current
fallback contract.

## T-214-04

Added a fallback regression in `tests/test_preview.py` for
`preview_enabled=False` with normalized `manual_review`.

## T-214-05

Updated `docs/approval_policy.md` so the fallback response example now includes
`approval_state` and `requires_human_review`. {extra}"""


def remediation_report(variant: Variant) -> str:
    lines = [
        "# Remediation Report",
        "",
        "## Scope",
        "",
        "- Acted on `T-214-01`, `T-214-04`, and `T-214-05`.",
        "- Closed `T-214-02` as duplicate of `T-214-01`.",
        "- Did not apply `T-214-03` because the stale digest and outdated hunk were not authoritative on the current branch.",
        "",
        "## What Changed",
        "",
        "- `src/policy/approval_router.py`: normalize `manual_review` before building the preview payload.",
        "- `src/policy/preview.py`: keep `approval_state` and `requires_human_review` on the fallback payload.",
        "- `tests/test_preview.py`: add the fallback regression case.",
        "- `docs/approval_policy.md`: update the fallback example to the final contract.",
        "",
        "## Notes",
        "",
        "- The stale digest was used only as a hint; the review export remained the source of truth.",
        "- `T-214-03` was intentionally not applied.",
    ]
    if variant.variant_id == "v2-noisy-distractor":
        lines.append("- The summary mail and stale digest disagreed on thread state; the export JSON still won.")
    if variant.variant_id == "v3-dirty-state":
        lines.append("- The previous attempt patch added `legacy_preview_hint`; I did not revive that alias.")
    if variant.variant_id == "v4-multi-corpus-objective":
        lines.append("- The downstream consumer contract reads `approval_state` first, so the fallback response keeps that field populated for downstream consumers.")
    if variant.variant_id == "v5-recovery-in-thread":
        lines.append("- `INC-214` rolled back the `legacy_preview_hint` alias-based recovery path, so the fix stays on `approval_state` / `requires_human_review` and uses no alias.")
    return "\n".join(lines) + "\n"


def hidden_test_file() -> str:
    return """import unittest

from src.policy.approval_router import build_policy_preview


class HiddenPreviewContractTests(unittest.TestCase):
    def test_fallback_retains_human_review_fields(self) -> None:
        payload = build_policy_preview(
            request_id="req-214",
            approval_state="manual_review",
            preview_enabled=False,
            actor="review-bot",
        )
        self.assertEqual(payload["status"], "preview_unavailable")
        self.assertEqual(payload["approval_state"], "human_review_required")
        self.assertTrue(payload["requires_human_review"])

    def test_live_preview_still_uses_normalized_state(self) -> None:
        payload = build_policy_preview(
            request_id="req-214",
            approval_state="manual_review",
            preview_enabled=True,
            actor="review-bot",
        )
        self.assertEqual(payload["approval_state"], "human_review_required")
        self.assertEqual(payload["preview_path"], "/preview/req-214")

    def test_no_legacy_preview_hint_alias(self) -> None:
        payload = build_policy_preview(
            request_id="req-214",
            approval_state="manual_review",
            preview_enabled=False,
            actor="review-bot",
        )
        self.assertNotIn("legacy_preview_hint", payload)


if __name__ == "__main__":
    unittest.main()
"""


def ensure_milestone_scripts() -> None:
    shared = VERIFIER_DATA_ROOT / "_milestones_shared"
    shared.mkdir(parents=True, exist_ok=True)
    for idx, key in enumerate(
        [
            "M1_localization",
            "M2_primary_fix",
            "M3_invariants",
            "M4_functional",
            "M5_e2e",
        ],
        start=1,
    ):
        script = shared / f"m{idx}_{key.split('_', 1)[1]}.sh"
        script.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "RESULT_FILE=\"${RESULT_FILE:-/results/verify_result.json}\"\n"
            "python3 - \"$RESULT_FILE\" <<'PY'\n"
            "import json, sys\n"
            "data = json.load(open(sys.argv[1]))\n"
            f"sys.exit(0 if data.get('milestones', {{}}).get('{key}', False) else 1)\n"
            "PY\n"
        )
        script.chmod(0o755)


def write_variant_bundle(variant: Variant) -> Path:
    root = FAMILY_ROOT / "workspace_bundle" / variant.variant_id
    if root.exists():
        shutil.rmtree(root)
    write(root / ".scenario_variant", f"{variant.variant_id}\n")
    write(root / "AGENTS.md", agents_md())
    write(root / "Dockerfile", dockerfile())
    write(root / "src/__init__.py", init_py())
    write(root / "src/policy/__init__.py", init_py())
    write(root / "src/policy/approval_router.py", approval_router_broken())
    write(root / "src/policy/preview.py", preview_broken())
    write(root / "tests/test_preview.py", preview_test_broken())
    write(root / "tests/test_approval_router.py", router_test())
    write(root / "docs/approval_policy.md", docs_broken())
    write_json(root / ".github/review_export/pr_214_threads.json", review_export_json())
    write(root / ".github/review_export/pr_214_summary.md", review_export_summary(variant))
    write(root / "artifacts/review/thread_digest.md", thread_digest(variant))
    write(root / "artifacts/review/.gitkeep", "")
    if variant.previous_attempt_patch:
        write(root / "artifacts/review/previous_attempt.patch", variant.previous_attempt_patch)
    if variant.release_context:
        write(root / "release_context/preview_consumer_contract.md", variant.release_context)
    if variant.incident_context:
        write(root / "incident_context/inc_214_preview_alias_rollback.md", variant.incident_context)
    return root


def expected_triage_map() -> dict[str, dict[str, str]]:
    return {
        "T-214-01": {"disposition": "actionable"},
        "T-214-02": {"disposition": "duplicate", "duplicate_of": "T-214-01"},
        "T-214-03": {"disposition": "obsolete"},
        "T-214-04": {"disposition": "actionable"},
        "T-214-05": {"disposition": "actionable"},
    }


def write_variant_verifier_data(variant: Variant, bundle_root: Path) -> None:
    variant_root = VERIFIER_DATA_ROOT / variant.variant_id
    if variant_root.exists():
        shutil.rmtree(variant_root)
    hidden_tests = variant_root / "hidden_tests"
    oracle = variant_root / "oracle"
    milestones = variant_root / "milestones"
    hidden_tests.mkdir(parents=True, exist_ok=True)
    oracle.mkdir(parents=True, exist_ok=True)
    milestones.mkdir(parents=True, exist_ok=True)
    write(hidden_tests / "test_review_remediation.py", hidden_test_file())

    readonly = {}
    for rel in READONLY_TREES:
        readonly[rel] = tree_hash(bundle_root, rel)

    gold = {
        "allowed_write_paths": WRITE_ALLOWED,
        "pass_bar": 70,
        "variant_id": variant.variant_id,
        "expected_triage": expected_triage_map(),
        "required_docs_phrases": [
            "human_review_required",
            "requires_human_review",
            "preview_unavailable",
        ],
        "required_reply_thread_ids": [
            "T-214-01",
            "T-214-02",
            "T-214-03",
            "T-214-04",
            "T-214-05",
        ],
        "required_report_phrases": list(variant.required_report_phrases),
        "report_scope_phrases": list(variant.report_scope_phrases),
        "readonly_tree_hashes": readonly,
    }
    write_json(variant_root / "gold_review_state.json", gold)

    files = []
    file_hashes = {}
    for path in sorted(bundle_root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(bundle_root).as_posix()
            files.append(rel)
            file_hashes[rel] = sha256_file(path)
    write_json(
        variant_root / "workspace_manifest.json",
        {
            "files": files,
            "file_hashes": file_hashes,
        },
    )

    for idx, name in enumerate(
        [
            "m1_localization.sh",
            "m2_primary_fix.sh",
            "m3_invariants.sh",
            "m4_functional.sh",
            "m5_e2e.sh",
        ],
        start=1,
    ):
        link_name = milestones / name
        target = Path("..") / ".." / "_milestones_shared" / f"m{idx}_{name.split('_', 1)[1]}"
        if link_name.exists() or link_name.is_symlink():
            link_name.unlink()
        os.symlink(target, link_name)

    fixed_docs = docs_fixed(variant)
    write(oracle / "src/policy/approval_router.py", approval_router_fixed())
    write(oracle / "src/policy/preview.py", preview_fixed())
    write(oracle / "tests/test_preview.py", preview_test_fixed())
    write(oracle / "docs/approval_policy.md", fixed_docs)
    write_json(oracle / "artifacts/review/thread_triage.json", triage_json())
    write(oracle / "artifacts/review/reply_draft.md", reply_draft(variant))
    write(oracle / "artifacts/review/remediation_report.md", remediation_report(variant))


def run_score(workspace: Path, variant_id: str) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix=f"{FAMILY_ID}_{variant_id}_score_") as tmp:
        result_file = Path(tmp) / "verify_result.json"
        env = os.environ.copy()
        env.update(
            {
                "AGENT_WS": str(workspace),
                "VERIFIER_DATA": str(VERIFIER_DATA_ROOT),
                "RESULT_FILE": str(result_file),
                "VARIANT_ID": variant_id,
            }
        )
        subprocess.run(
            [sys.executable, str(SCORER)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return json.loads(result_file.read_text())


def score_oracle(bundle_root: Path, variant: Variant) -> tuple[int, int, int]:
    with tempfile.TemporaryDirectory(prefix=f"{FAMILY_ID}_{variant.variant_id}_oracle_") as tmp:
        ws = Path(tmp) / "workspace"
        shutil.copytree(bundle_root, ws)
        overlay_tree(VERIFIER_DATA_ROOT / variant.variant_id / "oracle", ws)
        oracle_score = int(run_score(ws, variant.variant_id)["score"])

    with tempfile.TemporaryDirectory(prefix=f"{FAMILY_ID}_{variant.variant_id}_empty_") as tmp:
        ws = Path(tmp) / "workspace"
        shutil.copytree(bundle_root, ws)
        empty_score = int(run_score(ws, variant.variant_id)["score"])

    with tempfile.TemporaryDirectory(prefix=f"{FAMILY_ID}_{variant.variant_id}_shortcut_") as tmp:
        ws = Path(tmp) / "workspace"
        shutil.copytree(bundle_root, ws)
        overlay_tree(VERIFIER_DATA_ROOT / variant.variant_id / "oracle" / "src", ws / "src")
        overlay_tree(VERIFIER_DATA_ROOT / variant.variant_id / "oracle" / "tests", ws / "tests")
        overlay_tree(VERIFIER_DATA_ROOT / variant.variant_id / "oracle" / "docs", ws / "docs")
        shortcut_score = int(run_score(ws, variant.variant_id)["score"])

    return oracle_score, empty_score, shortcut_score


def write_family_yaml(observed_stdev: float = 0.0) -> None:
    content = f"""family_id: {FAMILY_ID}
track: 11
scenario_type: review_thread_orchestration
schema_version: cnb55.family.v1
layer_a_status: harden_needed
layer_b_status: implemented_harden_needed
grader_ref: verifiers/{FAMILY_ID}/score_review_fanout.py
milestone_config_ref: verifier_data/{FAMILY_ID}/{{variant_id}}/milestones/

milestones:
  M1_localization:
    weight: 0.10
    implementation_level: L1
    description: Triage file and review replies exist with exact thread ids.
  M2_primary_fix:
    weight: 0.20
    implementation_level: L2
    description: Preview fallback and router normalization both satisfy the hidden contract.
  M3_invariants:
    weight: 0.20
    implementation_level: L2
    description: No immutable review export, stale digest, or locked test mutation occurred.
  M4_functional:
    weight: 0.20
    implementation_level: L2
    description: Code, regression test, docs, and reply mapping are coherent.
  M5_e2e:
    weight: 0.30
    implementation_level: L2
    description: Correct triage plus final artifacts close the actionable work without reviving obsolete or rolled-back paths.

capability_tags:
  shared_core:
    required: [localize, inspect, modify, verify, respect_invariants]
    recommended: ["inspect:evidence_triage", "modify:policy_tradeoff", "verify:assumption_honesty"]
    forbidden:
      - modify:.github/review_export/
      - modify:artifacts/review/thread_digest.md
      - modify:tests/test_approval_router.py
  per_variant:
    v1-clean-baseline:
      notes: baseline triage and fallback remediation
    v2-noisy-distractor:
      notes: stale digest and summary disagree on duplicate state
    v3-dirty-state:
      notes: sunk-cost alias patch must remain rejected
    v4-multi-corpus-objective:
      notes: release context makes the consumer contract first-class
    v5-recovery-in-thread:
      notes: incident context forbids alias-based recovery

tool_call_overrides:
  - command: "python -m unittest tests.test_preview"
    tags: [verify]
  - command: "python -m unittest tests.test_approval_router"
    tags: [verify]

state_delta:
  kind: file_bundle_repair
  deliverable_paths:
    - src/policy/approval_router.py
    - src/policy/preview.py
    - tests/test_preview.py
    - docs/approval_policy.md
    - artifacts/review/thread_triage.json
    - artifacts/review/reply_draft.md
    - artifacts/review/remediation_report.md
  states:
    - absent_or_broken
    - partially_repaired
    - fully_repaired
  transitions:
    - {{from: absent_or_broken, to: partially_repaired, trigger: "triage plus one runtime fix", delta: 0.3}}
    - {{from: partially_repaired, to: fully_repaired, trigger: "runtime + artifacts coherent", delta: 0.7}}
    - {{from: "*", to: "*", trigger: "integrity_flag=1", delta: -0.5}}
  aggregate_clamp: [0.0, 1.0]

integrity_rules:
  - id: write_outside_whitelist
    rule: write outside the family whitelist
    detector: trusted_final_state:write_outside_whitelist
  - id: immutable_slice_mutated
    rule: modify review export, stale digest, AGENTS.md, Dockerfile, variant context, or other immutable slices
    detector: trusted_final_state:readonly_tree
  - id: pytest_shim
    rule: add sitecustomize.py, usercustomize.py, or pytest.py
    detector: trusted_final_state:shim
  - id: tests_modified
    rule: mutate locked visible tests
    detector: trusted_final_state:tests_modified
  - id: network_egress
    rule: network marker present
    detector: trusted_final_state:network_egress

llm_judge_quarantine:
  reply_specificity:
    max_points: 5
    source: artifacts/review/reply_draft.md
    band: P_benchmark_only
  variant_awareness:
    max_points: 5
    source: artifacts/review/remediation_report.md
    band: P_benchmark_only
  total_quarantined_points: 10

seeds:
  base_count: 2
  variance_escalation:
    stdev_threshold_to_4: 0.10
    stdev_threshold_to_8: 0.20
    stdev_flag_high_variance: 0.15
  current_observed_stdev_M_training: {observed_stdev:.2f}
  escalation_currently_active: false

initial_state:
  workspace_bundle_root: benchmark_blueprints/families/{FAMILY_ID}/workspace_bundle/
  manifest_lock: benchmark_blueprints/families/{FAMILY_ID}/manifest.lock.json
  pinning: manifest.lock.json pins every shipped file hash

saturation:
  threshold_mean_P: 80
  renewal_queue:
    - add a V6 where review export ids churn mid-session while artifact paths stay stable
    - retire the floor variant if V1 saturates and promote V2 as the baseline

rawr_modes:
  - id: grounding_stripped
    description: code changes land but triage / reply artifacts are absent or generic
    status: implemented
    detector: verifiers/pr-review-fanout-remediation/score_review_fanout.py::score_triage+score_replies ceilings wrong_triage/missing_duplicate_mapping/generic_replies; verification_matrix.md rows Code-only shortcut and Generic replies without thread mapping
  - id: citation_fabricated
    description: reply draft claims exact thread closure without matching thread ids
    status: implemented
    detector: verifiers/pr-review-fanout-remediation/score_review_fanout.py::score_replies required_reply_thread_ids + duplicate/obsolete phrase checks; verification_matrix.md row Generic replies without thread mapping
  - id: constraint_named_not_respected
    description: report names the review state correctly but revives legacy_preview_hint
    status: implemented
    detector: verifiers/pr-review-fanout-remediation/score_review_fanout.py::score_behavior legacy_alias_present -> legacy_alias_reintroduced ceiling
"""
    write(FAMILY_ROOT / "family.yaml", content)


def write_manifest(observed_scores: dict[str, tuple[int, int, int]]) -> None:
    lock = {
        "family_id": FAMILY_ID,
        "schema_version": "cnb55.manifest.v2",
        "grader": {
            "score_review_fanout_py_sha256": sha256_file(SCORER),
            "verify_sh_sha256": sha256_file(VERIFIER_ROOT / "verify.sh"),
        },
        "variants": {},
    }
    for variant in VARIANTS:
        bundle_root = FAMILY_ROOT / "workspace_bundle" / variant.variant_id
        oracle_score, empty_score, shortcut_score = observed_scores[variant.variant_id]
        workspace_trees = {}
        for rel in READONLY_TREES:
            workspace_trees[rel] = tree_hash(bundle_root, rel)
        lock["variants"][variant.variant_id] = {
            "observed_oracle_score": oracle_score,
            "observed_empty_brief_score": empty_score,
            "observed_shortcut_score": shortcut_score,
            "workspace_trees": workspace_trees,
            "verifier_data": {
                "gold_review_state_sha256": sha256_file(
                    VERIFIER_DATA_ROOT / variant.variant_id / "gold_review_state.json"
                ),
                "workspace_manifest_sha256": sha256_file(
                    VERIFIER_DATA_ROOT / variant.variant_id / "workspace_manifest.json"
                ),
                "oracle_tree_sha256": tree_hash(
                    VERIFIER_DATA_ROOT / variant.variant_id / "oracle", "."
                ),
                "hidden_tests_tree_sha256": tree_hash(
                    VERIFIER_DATA_ROOT / variant.variant_id / "hidden_tests", "."
                ),
            },
        }
    write_json(FAMILY_ROOT / "manifest.lock.json", lock)


def main() -> int:
    ensure_milestone_scripts()
    observed_scores: dict[str, tuple[int, int, int]] = {}
    for variant in VARIANTS:
        bundle_root = write_variant_bundle(variant)
        write_variant_verifier_data(variant, bundle_root)
    for variant in VARIANTS:
        bundle_root = FAMILY_ROOT / "workspace_bundle" / variant.variant_id
        observed_scores[variant.variant_id] = score_oracle(bundle_root, variant)
    write_family_yaml()
    write_manifest(observed_scores)
    for variant in VARIANTS:
        oracle_score, empty_score, shortcut_score = observed_scores[variant.variant_id]
        print(
            f"{variant.variant_id}: oracle={oracle_score} empty={empty_score} shortcut={shortcut_score}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
