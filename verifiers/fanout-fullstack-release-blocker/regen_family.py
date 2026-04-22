#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
FAMILY_ROOT = REPO / "benchmark_blueprints/families/fanout-fullstack-release-blocker"
VERIFIER_ROOT = REPO / "verifiers/fanout-fullstack-release-blocker"
VERIFIER_DATA_ROOT = REPO / "verifier_data/fanout-fullstack-release-blocker"

WRITE_ALLOWED = [
    "services/api/src/review_state.py",
    "services/api/src/routes/releases.py",
    "apps/admin/src/components/ReleaseGateForm.tsx",
    "apps/admin/src/lib/api.ts",
    "docs/runbooks/release_gating.md",
    "artifacts/dom",
    "artifacts/screenshots",
    "artifacts/report",
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
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, target)


@dataclass(frozen=True)
class Variant:
    variant_id: str
    label: str
    requires_release_order: bool
    incident_ref: str | None
    extra_report_phrases: tuple[str, ...]
    extra_docs_phrases: tuple[str, ...]


VARIANTS = [
    Variant("v1-clean-baseline", "V1 clean baseline", False, None, (), ()),
    Variant("v2-noisy-distractor", "V2 noisy distractor", False, None, ("legacy fixture",), ()),
    Variant("v3-dirty-state", "V3 dirty state", False, None, ("compatibility shim",), ("compatibility shim",)),
    Variant("v4-multi-corpus-objective", "V4 multi-corpus objective", True, None, ("request before echo",), ("request before echo",)),
    Variant("v5-recovery-in-thread", "V5 recovery in thread", True, "INC-204", ("inc-204", "no dual-write"), ("inc-204", "no dual-write")),
]


def agents_md(variant: Variant) -> str:
    extra = []
    if variant.variant_id == "v2-noisy-distractor":
        extra.append("- `fixtures/release_payloads/legacy_manual_review_snapshot.json` is a read-only migration record. Do not rewrite fixtures to make the task look finished.")
    if variant.variant_id == "v3-dirty-state":
        extra.append("- The repo contains a half-finished compatibility path. Finishing that shim is not enough; the live request path must be renamed end to end.")
    if variant.requires_release_order:
        extra.append("- `release_context/` is present. Read it before editing the runbook or report; the verification order is part of the task.")
    if variant.incident_ref:
        extra.append(f"- `incident_context/` is present. Read it before writing the report; {variant.incident_ref} explains why dual-write aliases are not acceptable.")
    extra_block = "\n".join(extra)
    return f"""# Agent Instructions — `fanout-fullstack-release-blocker`

## Task

Repair the release-gating rename across backend parsing, frontend submission, and
operator documentation. The only acceptable finish state is coherent end-to-end:
the admin form submits `human_review_required`, the backend stores and echoes
`human_review_required`, and the runbook tells operators to verify that exact token.

## Inputs

- `services/api/src/review_state.py`
- `services/api/src/routes/releases.py`
- `services/api/tests/test_release_payloads.py` — visible test, read-only
- `apps/admin/src/components/ReleaseGateForm.tsx`
- `apps/admin/src/lib/api.ts`
- `apps/admin/tests/release-gate.spec.ts` — visible test, read-only
- `docs/runbooks/release_gating.md`
- `fixtures/release_payloads/` — read-only seeded payloads
- `artifacts/preview/expected_screenshot_contract.md` — read-only proof contract
- `.scenario_variant` — current variant id
{extra_block}

## Required Outputs

- backend and frontend code changes in place
- runbook update in `docs/runbooks/release_gating.md`
- integration report at `artifacts/report/integration_report.md`
- DOM/request proof at `artifacts/dom/release_gate_capture.json`

## Rules

- Do not modify `services/api/tests/`, `apps/admin/tests/`, `fixtures/`,
  `artifacts/preview/`, `AGENTS.md`, `Dockerfile`, `.scenario_variant`,
  `release_context/`, or `incident_context/`.
- Do not write outside the fixed code/doc paths above plus `artifacts/dom/`
  and `artifacts/report/`.
- Do not add `sitecustomize.py`, `usercustomize.py`, or `pytest.py`.
- Do not use the network.

## What Good Looks Like

- backend active state is `human_review_required`
- legacy `manual_review` survives only as a narrow compatibility mapping if a
  visible test requires it
- frontend request body sends `approval_state: "human_review_required"`
- runbook names the retired token, the new token, and the verification order
- proof artifact shows the seeded request payload and the server echo agreeing
"""


def dockerfile() -> str:
    return "FROM python:3.12-bookworm\nWORKDIR /workspace\n"


def review_state_broken() -> str:
    return """DEFAULT_APPROVAL_STATE = "manual_review"
ACTIVE_APPROVAL_STATES = {"manual_review", "auto_approve", "blocked"}


def normalize_review_state(raw_state: str | None) -> str:
    state = (raw_state or DEFAULT_APPROVAL_STATE).strip()
    if state in ACTIVE_APPROVAL_STATES:
        return state
    raise ValueError(f"unsupported approval_state: {raw_state!r}")
"""


def review_state_fixed() -> str:
    return """DEFAULT_APPROVAL_STATE = "human_review_required"
ACTIVE_APPROVAL_STATES = {"human_review_required", "auto_approve", "blocked"}
RETIRED_APPROVAL_STATES = {"manual_review": "human_review_required"}


def normalize_review_state(raw_state: str | None) -> str:
    state = (raw_state or DEFAULT_APPROVAL_STATE).strip()
    if state in RETIRED_APPROVAL_STATES:
        return RETIRED_APPROVAL_STATES[state]
    if state in ACTIVE_APPROVAL_STATES:
        return state
    raise ValueError(f"unsupported approval_state: {raw_state!r}")
"""


def releases_broken() -> str:
    return """from services.api.src.review_state import normalize_review_state


def build_release_gate_update(payload: dict) -> dict:
    approval_state = normalize_review_state(payload.get("approval_state", "manual_review"))
    return {
        "release_id": payload["release_id"],
        "approval_state": approval_state,
    }


def serialize_release_gate(record: dict) -> dict:
    return {
        "release_id": record["release_id"],
        "approval_state": record.get("approval_state", "manual_review"),
        "operator_hint": "watch manual_review in the admin echo",
    }
"""


def releases_fixed() -> str:
    return """from services.api.src.review_state import normalize_review_state


def build_release_gate_update(payload: dict) -> dict:
    approval_state = normalize_review_state(payload.get("approval_state"))
    return {
        "release_id": payload["release_id"],
        "approval_state": approval_state,
    }


def serialize_release_gate(record: dict) -> dict:
    return {
        "release_id": record["release_id"],
        "approval_state": normalize_review_state(record.get("approval_state")),
        "operator_hint": "watch human_review_required in the admin echo",
    }
"""


def backend_test() -> str:
    return """from services.api.src.review_state import normalize_review_state
from services.api.src.routes.releases import build_release_gate_update, serialize_release_gate


def test_legacy_manual_review_normalizes() -> None:
    assert normalize_review_state("manual_review") == "human_review_required"


def test_release_gate_update_emits_new_token() -> None:
    update = build_release_gate_update({"release_id": "rel-ship-0422", "approval_state": "human_review_required"})
    assert update["approval_state"] == "human_review_required"


def test_release_gate_echo_uses_new_token() -> None:
    emitted = serialize_release_gate({"release_id": "rel-ship-0422", "approval_state": "human_review_required"})
    assert emitted["approval_state"] == "human_review_required"
"""


def form_broken() -> str:
    return """export const defaultGateState = {
  releaseId: "rel-ship-0422",
  approvalState: "manual_review",
  operatorLabel: "Manual review",
};

export function buildSeededGateForm() {
  return { ...defaultGateState };
}
"""


def form_fixed() -> str:
    return """export const defaultGateState = {
  releaseId: "rel-ship-0422",
  approvalState: "human_review_required",
  operatorLabel: "Human review required",
};

export function buildSeededGateForm() {
  return { ...defaultGateState };
}
"""


def api_broken() -> str:
    return """export function buildReleaseGateRequest(formState: { releaseId: string; approvalState: string }) {
  return {
    path: `/api/releases/${formState.releaseId}/gate`,
    method: "POST",
    body: {
      release_id: formState.releaseId,
      approval_state: formState.approvalState || "manual_review",
    },
  };
}

export function summarizeGateEcho(response: { approval_state: string }) {
  return `Server echoed ${response.approval_state || "manual_review"}`;
}
"""


def api_fixed() -> str:
    return """export function buildReleaseGateRequest(formState: { releaseId: string; approvalState: string }) {
  return {
    path: `/api/releases/${formState.releaseId}/gate`,
    method: "POST",
    body: {
      release_id: formState.releaseId,
      approval_state: formState.approvalState || "human_review_required",
    },
  };
}

export function summarizeGateEcho(response: { approval_state: string }) {
  return `Server echoed ${response.approval_state || "human_review_required"}`;
}
"""


def frontend_test() -> str:
    return """import { buildReleaseGateRequest } from "../src/lib/api";

test("release gate request posts approval_state", () => {
  const request = buildReleaseGateRequest({
    releaseId: "rel-ship-0422",
    approvalState: "human_review_required",
  });
  expect(request.body.approval_state).toBe("human_review_required");
});
"""


def runbook_broken(variant: Variant) -> str:
    extra = ""
    if variant.requires_release_order:
        extra += "\nKeep the old operator order: read the server echo first and only then inspect the UI payload."
    if variant.incident_ref:
        extra += "\nPrevious rollback note: dual-write can be revisited later."
    return f"""# Release gating runbook

Current rollout note still refers to `manual_review`.

1. Submit the release gate from the admin form.
2. Watch the server echo for `manual_review`.
3. Confirm the operator checklist matches the same token.
{extra}
"""


def runbook_fixed(variant: Variant) -> str:
    lines = [
        "# Release gating runbook",
        "",
        "The retired token is `manual_review`. The live rollout token is `human_review_required`.",
        "",
        "Verification order:",
        "1. intercept the outgoing request and confirm `approval_state` is `human_review_required`",
        "2. verify the server echo or persisted record also reports `human_review_required`",
        "3. complete the operator checklist only after the request and echo agree",
    ]
    if variant.variant_id == "v3-dirty-state":
        lines.append("4. remove the compatibility shim from the live path instead of treating it as the final fix")
    if variant.requires_release_order:
        lines.append("5. request before echo matters because the release fanout reads the request path first")
    if variant.incident_ref:
        lines.append(f"6. {variant.incident_ref} means no dual-write alias: do not emit both tokens in the live path")
    return "\n".join(lines) + "\n"


def preview_contract() -> str:
    return """# Proof contract

Write `artifacts/dom/release_gate_capture.json` with this shape:

- `schema_version`: `cnb55.release_gate_capture.v1`
- `variant_id`: contents of `.scenario_variant`
- `seed_release_id`: `rel-ship-0422`
- `proof_type`: `dom_intercept` or `request_echo_pair`
- `captured_request.request_path`: `/api/releases/rel-ship-0422/gate`
- `captured_request.approval_state`: `human_review_required`
- `server_echo.echo_path`: `/api/releases/rel-ship-0422`
- `server_echo.approval_state`: `human_review_required`
"""


def report_fixed(variant: Variant) -> str:
    lines = [
        "# Integration report",
        "",
        "## Backend",
        "- Updated the active approval state to `human_review_required`.",
        "- Left `manual_review` only as a narrow legacy compatibility mapping.",
        "",
        "## Frontend",
        "- Updated the seeded form state and request client to submit `human_review_required`.",
        "",
        "## Docs",
        "- Rewrote the runbook so operators verify the new token in request and echo order.",
        "",
        "## Proof",
        "- Captured the seeded request payload and server echo for `rel-ship-0422`.",
    ]
    if variant.variant_id == "v2-noisy-distractor":
        lines.append("- Left the legacy fixture unchanged so the fix stays on the live request path, not the fixture.")
    if variant.variant_id == "v3-dirty-state":
        lines.append("- Removed the compatibility shim dependency from the live path instead of stopping at the half-finished alias.")
    if variant.requires_release_order:
        lines.append("- The report keeps request before echo because release fanout evaluates the request path first.")
    if variant.incident_ref:
        lines.append(f"- {variant.incident_ref} is called out explicitly; no dual-write fallback remains in the live path.")
    return "\n".join(lines) + "\n"


def proof_fixed(variant: Variant) -> dict:
    notes = "Request payload and server echo agree on human_review_required."
    if variant.incident_ref:
        notes += f" {variant.incident_ref} kept the fix single-token with no dual-write."
    return {
        "schema_version": "cnb55.release_gate_capture.v1",
        "variant_id": variant.variant_id,
        "seed_release_id": "rel-ship-0422",
        "proof_type": "dom_intercept",
        "captured_request": {
            "request_path": "/api/releases/rel-ship-0422/gate",
            "approval_state": "human_review_required",
        },
        "server_echo": {
            "echo_path": "/api/releases/rel-ship-0422",
            "approval_state": "human_review_required",
        },
        "notes": notes,
    }


def hidden_test(variant: Variant) -> str:
    incident_check = ""
    if variant.incident_ref:
        incident_check = f"""

def test_incident_ceiling_off(result):
    assert "incident_blind_recovery" not in result.get("ceilings_applied", [])
"""
    return f"""from __future__ import annotations

import json
import os
from pathlib import Path


RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))


def load():
    return json.loads(RESULT_FILE.read_text())


def test_schema_version():
    result = load()
    assert result["schema_version"] == "cnb55.verify_result.v3"


def test_primary_fix_milestones():
    result = load()
    assert result["milestones"]["M2_primary_fix"] is True


def test_live_path_milestones():
    result = load()
    assert result["milestones"]["live_request_correct"] is True
    assert result["milestones"]["live_echo_correct"] is True
{incident_check}
"""


def milestone_script(slot: str) -> str:
    key = {
        "m1_localize.sh": "M1_localization",
        "m2_primary_fix.sh": "M2_primary_fix",
        "m3_invariants.sh": "M3_invariants",
        "m4_functional.sh": "M4_functional",
        "m5_e2e.sh": "M5_e2e",
    }[slot]
    return f"""#!/usr/bin/env bash
set -euo pipefail
RESULT_FILE="${{RESULT_FILE:-/results/verify_result.json}}"
python3 - "$RESULT_FILE" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
sys.exit(0 if data.get("milestones", {{}}).get("{key}", False) else 1)
PY
"""


def release_context(variant: Variant) -> str:
    return """# Release context

The release fanout reads the intercepted request payload before the server echo
appears in the operator console. If the runbook verifies echo first, operators
can falsely approve a stale UI submission.
"""


def incident_context(variant: Variant) -> str:
    return f"""# {variant.incident_ref} dual-write rollback

The previous rollout tried to emit both `manual_review` and
`human_review_required` during the same session. The downstream release-gating
fanout treated that as conflicting state and rolled the deployment back.
The recovery rule is strict: no dual-write alias in the live path.
"""


def build_workspace_bundle(variant: Variant) -> Path:
    root = FAMILY_ROOT / "workspace_bundle" / variant.variant_id
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    write(root / ".scenario_variant", variant.variant_id + "\n")
    write(root / "AGENTS.md", agents_md(variant))
    write(root / "Dockerfile", dockerfile())
    write(root / "services/api/src/review_state.py", review_state_broken())
    write(root / "services/api/src/routes/releases.py", releases_broken())
    write(root / "services/api/tests/test_release_payloads.py", backend_test())
    write(root / "apps/admin/src/components/ReleaseGateForm.tsx", form_broken())
    write(root / "apps/admin/src/lib/api.ts", api_broken())
    write(root / "apps/admin/tests/release-gate.spec.ts", frontend_test())
    write(root / "docs/runbooks/release_gating.md", runbook_broken(variant))
    write_json(root / "fixtures/release_payloads/current_release_seed.json", {
        "release_id": "rel-ship-0422",
        "approval_state": "manual_review",
        "display_name": "Seeded release gate",
    })
    write_json(root / "fixtures/release_payloads/legacy_manual_review_snapshot.json", {
        "snapshot_date": "2026-04-18",
        "approval_state": "manual_review",
        "note": "read-only legacy fixture",
    })
    write(root / "artifacts/preview/expected_screenshot_contract.md", preview_contract())
    write(root / "artifacts/dom/.gitkeep", "")
    write(root / "artifacts/report/.gitkeep", "")
    if variant.requires_release_order:
        write(root / "release_context/migration_order.md", release_context(variant))
    if variant.incident_ref:
        write(root / "incident_context/inc_204_dual_write.md", incident_context(variant))
    return root


def build_oracle(variant: Variant) -> Path:
    root = VERIFIER_DATA_ROOT / variant.variant_id / "oracle"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    write(root / "services/api/src/review_state.py", review_state_fixed())
    write(root / "services/api/src/routes/releases.py", releases_fixed())
    write(root / "apps/admin/src/components/ReleaseGateForm.tsx", form_fixed())
    write(root / "apps/admin/src/lib/api.ts", api_fixed())
    write(root / "docs/runbooks/release_gating.md", runbook_fixed(variant))
    write(root / "artifacts/report/integration_report.md", report_fixed(variant))
    write_json(root / "artifacts/dom/release_gate_capture.json", proof_fixed(variant))
    return root


def baseline_manifest(bundle: Path) -> dict:
    file_hashes = {}
    files = []
    for path in sorted(bundle.rglob("*")):
        if path.is_file():
            rel = path.relative_to(bundle).as_posix()
            files.append(rel)
            file_hashes[rel] = sha256_file(path)
    return {"files": files, "file_hashes": file_hashes}


def score_workspace(variant: Variant, workspace: Path) -> dict:
    result_path = workspace.parent / f"{variant.variant_id}_verify_result.json"
    env = {
        "AGENT_WS": str(workspace),
        "VERIFIER_DATA": str(VERIFIER_DATA_ROOT),
        "VARIANT_ID": variant.variant_id,
        "RESULT_FILE": str(result_path),
    }
    subprocess.run(
        [sys.executable, str(VERIFIER_ROOT / "score_release_blocker.py")],
        check=True,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return json.loads(result_path.read_text())


def synthesized_score(variant: Variant, builder: str) -> dict:
    scratch = FAMILY_ROOT / "_tmp_regen" / variant.variant_id / builder
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(FAMILY_ROOT / "workspace_bundle" / variant.variant_id, scratch)
    if builder == "oracle":
        overlay_tree(VERIFIER_DATA_ROOT / variant.variant_id / "oracle", scratch)
    elif builder == "backend_only":
        for rel in [
            "services/api/src/review_state.py",
            "services/api/src/routes/releases.py",
        ]:
            target = scratch / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(VERIFIER_DATA_ROOT / variant.variant_id / "oracle" / rel, target)
    elif builder == "empty":
        pass
    else:
        raise ValueError(builder)
    result = score_workspace(variant, scratch)
    shutil.rmtree(scratch)
    return result


def write_variant_data(variant: Variant, bundle: Path, oracle: Path) -> dict:
    variant_root = VERIFIER_DATA_ROOT / variant.variant_id
    variant_root.mkdir(parents=True, exist_ok=True)
    manifest = baseline_manifest(bundle)
    write_json(variant_root / "workspace_manifest.json", manifest)
    write(variant_root / "hidden_tests/test_release_blocker_properties.py", hidden_test(variant))
    milestone_root = variant_root / "milestones"
    milestone_root.mkdir(parents=True, exist_ok=True)
    for slot in [
        "m1_localize.sh",
        "m2_primary_fix.sh",
        "m3_invariants.sh",
        "m4_functional.sh",
        "m5_e2e.sh",
    ]:
        target = milestone_root / slot
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(Path("..") / ".." / "_milestones_shared" / slot)

    gold = {
        "variant_id": variant.variant_id,
        "pass_bar": 70,
        "seed_release_id": "rel-ship-0422",
        "request_path": "/api/releases/rel-ship-0422/gate",
        "echo_path": "/api/releases/rel-ship-0422",
        "proof_artifact_path": "artifacts/dom/release_gate_capture.json",
        "integration_report_path": "artifacts/report/integration_report.md",
        "allowed_write_paths": WRITE_ALLOWED,
        "runtime_paths": [
            "services/api/src/review_state.py",
            "services/api/src/routes/releases.py",
            "apps/admin/src/components/ReleaseGateForm.tsx",
            "apps/admin/src/lib/api.ts",
            "docs/runbooks/release_gating.md",
        ],
        "runtime_paths_without_legacy_token": [
            "services/api/src/routes/releases.py",
            "apps/admin/src/components/ReleaseGateForm.tsx",
            "apps/admin/src/lib/api.ts",
        ],
        "required_docs_phrases": [
            "manual_review",
            "human_review_required",
            "intercept the outgoing request",
            "verify the server echo",
        ] + list(variant.extra_docs_phrases),
        "required_report_phrases": [
            "backend",
            "frontend",
            "docs",
            "proof",
        ] + list(variant.extra_report_phrases),
        "requires_release_order": variant.requires_release_order,
        "incident_ref": variant.incident_ref,
        "readonly_tree_hashes": {
            ".scenario_variant": tree_hash(bundle, ".scenario_variant"),
            "AGENTS.md": tree_hash(bundle, "AGENTS.md"),
            "Dockerfile": tree_hash(bundle, "Dockerfile"),
            "services/api/tests": tree_hash(bundle, "services/api/tests"),
            "apps/admin/tests": tree_hash(bundle, "apps/admin/tests"),
            "fixtures": tree_hash(bundle, "fixtures"),
            "artifacts/preview": tree_hash(bundle, "artifacts/preview"),
            "release_context": tree_hash(bundle, "release_context"),
            "incident_context": tree_hash(bundle, "incident_context"),
        },
    }
    write_json(variant_root / "gold_release_state.json", gold)
    oracle_score = synthesized_score(variant, "oracle")
    empty_score = synthesized_score(variant, "empty")
    shortcut_score = synthesized_score(variant, "backend_only")
    return {
        "oracle_score": int(oracle_score["P_benchmark"]),
        "empty_score": int(empty_score["P_benchmark"]),
        "shortcut_score": int(shortcut_score["P_benchmark"]),
        "manifest": manifest,
    }


def write_family_yaml(summary: dict[str, dict[str, int]]) -> None:
    content = f"""family_id: fanout-fullstack-release-blocker
track: 11
scenario_type: parallel_cross_surface_repair
schema_version: cnb55.family.v1
layer_a_status: in_progress
layer_b_status: implemented_pending_probe
grader_ref: verifiers/fanout-fullstack-release-blocker/score_release_blocker.py
milestone_config_ref: verifier_data/fanout-fullstack-release-blocker/{{variant_id}}/milestones/

milestones:
  M1_localization:
    weight: 0.10
    implementation_level: L1
    description: Report names backend, frontend, docs, and proof.
  M2_primary_fix:
    weight: 0.20
    implementation_level: L2
    description: Backend and frontend both emit `human_review_required`.
  M3_invariants:
    weight: 0.20
    implementation_level: L2
    description: No integrity rule fired and the submission stayed within whitelist.
  M4_functional:
    weight: 0.20
    implementation_level: L2
    description: Backend, frontend, docs, and proof contract all align.
  M5_e2e:
    weight: 0.30
    implementation_level: L2
    description: Captured request and server echo both show `human_review_required`.

capability_tags:
  shared_core:
    required: [localize, inspect, modify, verify, respect_invariants]
    recommended: [inspect:evidence_triage, verify:assumption_honesty]
    forbidden:
      - modify:tests/
      - modify:fixtures/
      - modify:artifacts/preview/
  per_variant:
    v1-clean-baseline:
      notes: clean rename baseline
    v2-noisy-distractor:
      notes: ignore read-only legacy fixture noise
    v3-dirty-state:
      notes: do not stop at compatibility shim
    v4-multi-corpus-objective:
      notes: release_context changes verification order
    v5-recovery-in-thread:
      notes: incident_context forbids dual-write recovery

tool_call_overrides:
  - command: "pytest services/api/tests/test_release_payloads.py"
    tags: [verify]
  - command: "npm test -- release-gate"
    tags: [verify]

state_delta:
  kind: file_bundle_repair
  deliverable_paths:
    - services/api/src/review_state.py
    - services/api/src/routes/releases.py
    - apps/admin/src/components/ReleaseGateForm.tsx
    - apps/admin/src/lib/api.ts
    - docs/runbooks/release_gating.md
    - artifacts/report/integration_report.md
    - artifacts/dom/release_gate_capture.json
  states:
    - absent_or_broken
    - partially_repaired
    - fully_repaired
  transitions:
    - {{from: absent_or_broken, to: partially_repaired, trigger: "one surface aligned", delta: 0.3}}
    - {{from: partially_repaired, to: fully_repaired, trigger: "proof + request/echo alignment", delta: 0.7}}
    - {{from: "*", to: "*", trigger: "integrity_flag=1", delta: -0.5}}
  aggregate_clamp: [0.0, 1.0]

integrity_rules:
  - id: write_outside_whitelist
    rule: write outside the family whitelist
    detector: trusted_final_state:write_outside_whitelist
  - id: immutable_slice_mutated
    rule: modify tests, fixtures, preview artifacts, or variant evidence
    detector: trusted_final_state:readonly_tree
  - id: pytest_shim
    rule: add sitecustomize.py, usercustomize.py, or pytest.py
    detector: trusted_final_state:shim
  - id: tests_modified
    rule: mutate visible tests
    detector: trusted_final_state:tests_modified
  - id: network_egress
    rule: network marker present
    detector: trusted_final_state:network_egress

llm_judge_quarantine:
  report_cross_surface:
    max_points: 8
    source: artifacts/report/integration_report.md
    band: P_benchmark_only
  report_variant_awareness:
    max_points: 5
    source: artifacts/report/integration_report.md
    band: P_benchmark_only
  total_quarantined_points: 13

seeds:
  base_count: 2
  variance_escalation:
    stdev_threshold_to_4: 0.10
    stdev_threshold_to_8: 0.20
    stdev_flag_high_variance: 0.15
  current_observed_stdev_M_training: 0.0
  escalation_currently_active: false

initial_state:
  workspace_bundle_root: benchmark_blueprints/families/fanout-fullstack-release-blocker/workspace_bundle/
  manifest_lock: benchmark_blueprints/families/fanout-fullstack-release-blocker/manifest.lock.json
  pinning: manifest.lock.json pins every shipped file hash

saturation:
  threshold_mean_P: 80
  renewal_queue:
    - add a V6 where approval state changes mid-session
    - retire the easy floor variant if V1 saturates

rawr_modes:
  - id: grounding_stripped
    description: report exists but no proof artifact
  - id: constraint_named_not_respected
    description: docs mention the rename but the runtime path still emits the old token
"""
    write(FAMILY_ROOT / "family.yaml", content)


def write_manifest_lock(summary: dict[str, dict[str, int]]) -> None:
    manifest = {
        "family_id": "fanout-fullstack-release-blocker",
        "schema_version": "cnb55.manifest.v2",
        "track": 11,
        "track_name": "Subagents Orchestration",
        "cli": {},
        "grader": {
            "score_release_blocker_py_sha256": sha256_file(VERIFIER_ROOT / "score_release_blocker.py"),
            "verify_sh_sha256": sha256_file(VERIFIER_ROOT / "verify.sh"),
        },
        "variants": {},
    }
    for variant in VARIANTS:
        variant_root = VERIFIER_DATA_ROOT / variant.variant_id
        manifest["variants"][variant.variant_id] = {
            "observed_oracle_score": summary[variant.variant_id]["oracle_score"],
            "observed_empty_score": summary[variant.variant_id]["empty_score"],
            "observed_shortcut_score": summary[variant.variant_id]["shortcut_score"],
            "workspace_trees": {
                ".scenario_variant": tree_hash(FAMILY_ROOT / "workspace_bundle" / variant.variant_id, ".scenario_variant"),
                "AGENTS.md": tree_hash(FAMILY_ROOT / "workspace_bundle" / variant.variant_id, "AGENTS.md"),
                "Dockerfile": tree_hash(FAMILY_ROOT / "workspace_bundle" / variant.variant_id, "Dockerfile"),
                "services/api/tests": tree_hash(FAMILY_ROOT / "workspace_bundle" / variant.variant_id, "services/api/tests"),
                "apps/admin/tests": tree_hash(FAMILY_ROOT / "workspace_bundle" / variant.variant_id, "apps/admin/tests"),
                "fixtures": tree_hash(FAMILY_ROOT / "workspace_bundle" / variant.variant_id, "fixtures"),
                "artifacts/preview": tree_hash(FAMILY_ROOT / "workspace_bundle" / variant.variant_id, "artifacts/preview"),
                "release_context": tree_hash(FAMILY_ROOT / "workspace_bundle" / variant.variant_id, "release_context"),
                "incident_context": tree_hash(FAMILY_ROOT / "workspace_bundle" / variant.variant_id, "incident_context"),
            },
            "verifier_data": {
                "gold_release_state_sha256": sha256_file(variant_root / "gold_release_state.json"),
                "workspace_manifest_sha256": sha256_file(variant_root / "workspace_manifest.json"),
                "hidden_tests_tree_sha256": tree_hash(variant_root, "hidden_tests"),
            },
        }
    write_json(FAMILY_ROOT / "manifest.lock.json", manifest)


def write_shared_files() -> None:
    shared = VERIFIER_DATA_ROOT / "_milestones_shared"
    shared.mkdir(parents=True, exist_ok=True)
    for slot in [
        "m1_localize.sh",
        "m2_primary_fix.sh",
        "m3_invariants.sh",
        "m4_functional.sh",
        "m5_e2e.sh",
    ]:
        path = shared / slot
        write(path, milestone_script(slot))
        path.chmod(0o755)
    write(
        VERIFIER_DATA_ROOT / "_rubrics/partial_progress.md",
        "# Partial-progress rubric\n\nReport coverage and variant-awareness stay in the P-only band.\n",
    )


def write_matrices() -> None:
    subprocess.run(
        [
            sys.executable,
            str(VERIFIER_ROOT / "run_verification_matrix.py"),
            "--variant",
            "v1-clean-baseline",
            "--out",
            str(FAMILY_ROOT / "verification_matrix.md"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(VERIFIER_ROOT / "run_verification_matrix.py"),
            "--variant",
            "v5-recovery-in-thread",
            "--out",
            str(FAMILY_ROOT / "verification_matrix_v5-recovery-in-thread.md"),
        ],
        check=True,
    )


def main() -> int:
    if (FAMILY_ROOT / "_tmp_regen").exists():
        shutil.rmtree(FAMILY_ROOT / "_tmp_regen")
    write_shared_files()

    summary: dict[str, dict[str, int]] = {}
    for variant in VARIANTS:
        bundle = build_workspace_bundle(variant)
        oracle = build_oracle(variant)
        summary[variant.variant_id] = write_variant_data(variant, bundle, oracle)

    write_family_yaml(summary)
    write_manifest_lock(summary)
    write_matrices()

    tmp_dir = FAMILY_ROOT / "_tmp_regen"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
