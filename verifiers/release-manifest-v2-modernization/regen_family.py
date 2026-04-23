#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FAMILY_ID = "release-manifest-v2-modernization"
TRACK = 3
FAMILY_ROOT = REPO / "benchmark_blueprints" / "families" / FAMILY_ID
VERIFIER_ROOT = REPO / "verifiers" / FAMILY_ID
VERIFIER_DATA_ROOT = REPO / "verifier_data" / FAMILY_ID
WORKSPACE_BUNDLE = FAMILY_ROOT / "workspace_bundle"

WRITE_ALLOWED = [
    ".github/workflows/release.yml",
    ".github/workflows/reusable_release.yml",
    "release/manifest.v2.toml",
    "codex/config.toml",
    "docs/releases/staging_rollout.md",
    "artifacts/release_smoke_report.json",
]
READONLY_TREE_PATHS = [
    "tests",
    "logs",
    "config/env",
    "release/manifest.toml",
    "deploy",
    "scripts",
    "AGENTS.md",
    "Dockerfile",
    ".scenario_variant",
]
VISIBLE_TESTS = [
    "tests/test_manifest_contract.py",
    "tests/test_release_driver.py",
]


@dataclass(frozen=True)
class Variant:
    variant_id: str
    label: str
    extra_agents_note: str
    required_docs_phrases: tuple[str, ...]
    docs_order_pairs: tuple[tuple[str, str], ...]
    release_context_files: dict[str, str]
    incident_context_files: dict[str, str]


VARIANTS = [
    Variant(
        variant_id="v1-clean-baseline",
        label="clean baseline",
        extra_agents_note="No extra corpus. Finish the reusable workflow cutover end to end.",
        required_docs_phrases=(
            "reusable workflow",
            "release/manifest.v2.toml",
            "target_environment: staging",
            "deploy/check_release.py --env staging --emit-json artifacts/release_smoke_report.json",
        ),
        docs_order_pairs=(),
        release_context_files={},
        incident_context_files={},
    ),
    Variant(
        variant_id="v2-noisy-distractor",
        label="noisy distractor",
        extra_agents_note=(
            "Ignore the archived prod-era note and log. They are read-only stale evidence, "
            "not the live contract."
        ),
        required_docs_phrases=(
            "reusable workflow",
            "release/manifest.v2.toml",
            "target_environment: staging",
            "deploy/check_release.py --env staging --emit-json artifacts/release_smoke_report.json",
        ),
        docs_order_pairs=(),
        release_context_files={},
        incident_context_files={},
    ),
    Variant(
        variant_id="v3-dirty-state",
        label="dirty state",
        extra_agents_note=(
            "The workspace includes an abandoned migration draft. Do not finish the sunk-cost draft; "
            "repair the actual release path and preserve the seeded env override comment."
        ),
        required_docs_phrases=(
            "reusable workflow",
            "release/manifest.v2.toml",
            "target_environment: staging",
            "deploy/check_release.py --env staging --emit-json artifacts/release_smoke_report.json",
        ),
        docs_order_pairs=(),
        release_context_files={},
        incident_context_files={},
    ),
    Variant(
        variant_id="v4-multi-corpus-objective",
        label="multi-corpus objective",
        extra_agents_note=(
            "Read `release_context/` before updating docs. The correct operator sequence is part of the task."
        ),
        required_docs_phrases=(
            "reusable workflow",
            "release/manifest.v2.toml",
            "target_environment: staging",
            "artifact_manifest output",
            "deploy/check_release.py --env staging --emit-json artifacts/release_smoke_report.json",
        ),
        docs_order_pairs=(("artifact_manifest output", "staging smoke"),),
        release_context_files={
            "release_context/rollout_order.md": (
                "# Rollout Order\n\n"
                "The reusable workflow now emits an `artifact_manifest` output.\n"
                "Operators must confirm that output before the staging smoke step,\n"
                "because the dry-run path can stay green while the manifest contract is stale.\n"
            ),
            "release_context/objective_shift.md": (
                "# Objective Shift\n\n"
                "The modernization is only complete if operators can verify the emitted\n"
                "`artifact_manifest` path before staging smoke.\n"
            ),
        },
        incident_context_files={},
    ),
    Variant(
        variant_id="v5-recovery-in-thread",
        label="recovery in thread",
        extra_agents_note=(
            "Read both `release_context/` and `incident_context/`. The rollout was previously rolled "
            "back because a prod alias leaked back into the workflow."
        ),
        required_docs_phrases=(
            "reusable workflow",
            "release/manifest.v2.toml",
            "target_environment: staging",
            "artifact_manifest output",
            "inc-342",
            "do not reintroduce the prod alias",
            "deploy/check_release.py --env staging --emit-json artifacts/release_smoke_report.json",
        ),
        docs_order_pairs=(("artifact_manifest output", "staging smoke"),),
        release_context_files={
            "release_context/rollout_order.md": (
                "# Rollout Order\n\n"
                "The reusable workflow now emits an `artifact_manifest` output.\n"
                "Operators must confirm that output before the staging smoke step.\n"
            ),
        },
        incident_context_files={
            "incident_context/inc_342_prod_alias.md": (
                "# INC-342\n\n"
                "A prior cutover reused the `prod` alias in the modern workflow path and\n"
                "shipped the wrong staging target. The recovery rule is explicit: do not\n"
                "reintroduce the prod alias anywhere in the live reusable-workflow path.\n"
            ),
            "incident_context/recovery_notes.md": (
                "# Recovery Notes\n\n"
                "Keep the modern path single-target: `target_environment: staging`.\n"
            ),
        },
    ),
]


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    write(path, json.dumps(payload, indent=2, sort_keys=True))


def sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def tree_hash(root: Path, rel: str) -> str:
    target = root / rel
    if not target.exists():
        return "MISSING"
    if target.is_file():
        return sha256_file(target)
    digest = sha256()
    for path in sorted(target.rglob("*")):
        relpath = path.relative_to(target).as_posix()
        if path.is_dir():
            digest.update(f"D:{relpath}\n".encode())
        else:
            digest.update(f"F:{relpath}\n".encode())
            digest.update(sha256_file(path).encode())
            digest.update(b"\n")
    return digest.hexdigest()


def list_files(root: Path) -> list[str]:
    return [path.relative_to(root).as_posix() for path in sorted(root.rglob("*")) if path.is_file()]


def release_workflow_broken() -> str:
    return """name: release
on:
  workflow_dispatch:
jobs:
  release:
    uses: ./.github/workflows/reusable_release.yml
    with:
      manifest_path: release/manifest.toml
      environment: prod
"""


def release_workflow_fixed() -> str:
    return """name: release
on:
  workflow_dispatch:
jobs:
  release:
    uses: ./.github/workflows/reusable_release.yml
    with:
      manifest_path: release/manifest.v2.toml
      target_environment: staging
    secrets:
      deploy_token: ${{ secrets.DEPLOY_TOKEN }}
"""


def reusable_workflow_broken() -> str:
    return """name: reusable-release
on:
  workflow_call:
    inputs:
      manifest_path:
        required: true
        type: string
      target_environment:
        required: true
        type: string
jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - run: echo "legacy reusable contract"
"""


def reusable_workflow_fixed() -> str:
    return """name: reusable-release
on:
  workflow_call:
    inputs:
      manifest_path:
        required: true
        type: string
      target_environment:
        required: true
        type: string
    secrets:
      deploy_token:
        required: true
    outputs:
      artifact_manifest:
        description: emitted manifest path
        value: ${{ jobs.release.outputs.artifact_manifest }}
jobs:
  release:
    runs-on: ubuntu-latest
    outputs:
      artifact_manifest: ${{ steps.manifest.outputs.path }}
    steps:
      - id: manifest
        run: echo "path=artifacts/release-manifest.json" >> "$GITHUB_OUTPUT"
"""


def manifest_legacy() -> str:
    return """version = "v1"
artifact = "shipit.tar.gz"
environment = "prod"
"""


def manifest_v2_broken() -> str:
    return """version = "v2"
artifact_name = "shipit.tar.gz"
artifact_manifest = "artifacts/legacy-manifest.json"
target_environment = "production"
"""


def manifest_v2_fixed() -> str:
    return """version = "v2"
artifact_name = "shipit.tar.gz"
artifact_manifest = "artifacts/release-manifest.json"
target_environment = "staging"
"""


def run_ci_py() -> str:
    return """import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "release.yml").read_text()
    manifest = (root / "release" / "manifest.v2.toml").read_text()
    if "target_environment" in workflow and 'target_environment = "staging"' in manifest:
        print("dry-run ok")
        return 0
    raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
"""


def check_release_py() -> str:
    return """import argparse
import json
from pathlib import Path


def _read(root: Path, relpath: str) -> str:
    path = root / relpath
    if not path.exists():
        return ""
    return path.read_text()


def evaluate_release_alignment(root: Path, env: str) -> dict:
    variant = (root / ".scenario_variant").read_text().strip()
    workflow = _read(root, ".github/workflows/release.yml")
    reusable = _read(root, ".github/workflows/reusable_release.yml")
    manifest = _read(root, "release/manifest.v2.toml")
    config = _read(root, "codex/config.toml")
    docs = _read(root, "docs/releases/staging_rollout.md").lower()

    checks = {}
    checks["workflow"] = (
        "uses: ./.github/workflows/reusable_release.yml" in workflow
        and "manifest_path: release/manifest.v2.toml" in workflow
        and "target_environment: staging" in workflow
        and "environment: prod" not in workflow
    )
    checks["reusable_contract"] = (
        "secrets:" in reusable
        and "deploy_token:" in reusable
        and "outputs:" in reusable
        and "artifact_manifest:" in reusable
    )
    checks["manifest"] = (
        'version = "v2"' in manifest
        and 'artifact_manifest = "artifacts/release-manifest.json"' in manifest
        and 'target_environment = "staging"' in manifest
        and 'target_environment = "production"' not in manifest
    )
    checks["config"] = (
        'release_entrypoint = "scripts/run_ci.py"' in config
        and 'release_smoke_command = "python deploy/check_release.py --env staging --emit-json artifacts/release_smoke_report.json"' in config
        and 'release_manifest = "release/manifest.v2.toml"' in config
    )
    checks["docs_base"] = all(
        phrase in docs
        for phrase in [
            "reusable workflow",
            "release/manifest.v2.toml",
            "target_environment: staging",
            "deploy/check_release.py --env staging --emit-json artifacts/release_smoke_report.json",
        ]
    )
    checks["variant_docs"] = True
    if variant in {"v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        checks["variant_docs"] = (
            "artifact_manifest output" in docs
            and docs.index("artifact_manifest output") < docs.index("staging smoke")
        )
    if variant == "v5-recovery-in-thread":
        checks["variant_docs"] = checks["variant_docs"] and (
            "inc-342" in docs and "do not reintroduce the prod alias" in docs
        )
    checks["no_prod_alias"] = "environment: prod" not in workflow and '"production"' not in manifest

    ok = all(checks.values())
    return {
        "ok": ok,
        "variant_id": variant,
        "target_environment": env,
        "workflow_ref": ".github/workflows/reusable_release.yml",
        "manifest_path": "release/manifest.v2.toml",
        "artifact_manifest": "artifacts/release-manifest.json",
        "checks": checks,
        "ordered_checks": (
            ["dry-run", "artifact_manifest", "staging smoke"]
            if variant in {"v4-multi-corpus-objective", "v5-recovery-in-thread"}
            else ["dry-run", "staging smoke"]
        ),
        "incident_guardrail": "INC-342: no prod alias" if variant == "v5-recovery-in-thread" else None,
    }


def staging_smoke_ok(root: Path, env: str) -> bool:
    return bool(evaluate_release_alignment(root, env)["ok"])


def write_report(root: Path, env: str, out_path: Path) -> dict:
    result = evaluate_release_alignment(root, env)
    payload = {
        "schema_version": "cnb55.release_smoke_report.v1",
        "variant_id": result["variant_id"],
        "target_environment": env,
        "workflow_ref": result["workflow_ref"],
        "manifest_path": result["manifest_path"],
        "artifact_manifest": result["artifact_manifest"],
        "smoke_status": "ok" if result["ok"] else "failed",
        "ordered_checks": result["ordered_checks"],
        "incident_guardrail": result["incident_guardrail"],
        "checks": result["checks"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--emit-json", default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ok = staging_smoke_ok(root, args.env)
    if args.emit_json:
        write_report(root, args.env, root / args.emit_json)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    raise SystemExit(main())
"""


def config_broken() -> str:
    return """release_entrypoint = "scripts/legacy_release.py"
release_smoke_command = "python deploy/check_release.py --env prod"
release_manifest = "release/manifest.toml"
"""


def config_fixed() -> str:
    return """release_entrypoint = "scripts/run_ci.py"
release_smoke_command = "python deploy/check_release.py --env staging --emit-json artifacts/release_smoke_report.json"
release_manifest = "release/manifest.v2.toml"
"""


def docs_broken(variant: Variant) -> str:
    extra = ""
    if variant.variant_id == "v3-dirty-state":
        extra = "\nOperator note: keep the local override comment unchanged while you modernize the release path.\n"
    return f"""# Staging Rollout

Use `scripts/legacy_release.py` for the rollout.
{extra}"""


def docs_fixed(variant: Variant) -> str:
    lines = [
        "# Staging Rollout",
        "",
        "Use the reusable workflow with `release/manifest.v2.toml`.",
        "Set `target_environment: staging` on the live workflow call.",
        "Run `python deploy/check_release.py --env staging --emit-json artifacts/release_smoke_report.json` after the dry-run gate.",
        "Keep the reusable workflow and the operator doc aligned; do not fall back to `scripts/legacy_release.py`.",
    ]
    if variant.variant_id in {"v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        lines.append(
            "Check the `artifact_manifest output` before the staging smoke step so the dry-run path and the release contract stay aligned."
        )
    if variant.variant_id == "v5-recovery-in-thread":
        lines.append("INC-342 requires that operators do not reintroduce the prod alias anywhere in the live path.")
    if variant.variant_id == "v3-dirty-state":
        lines.append("Preserve the existing env override comment; it is unrelated local state, not migration debt.")
    return "\n".join(lines)


def visible_test_manifest() -> str:
    return """from pathlib import Path


def test_workflow_and_manifest_use_target_environment():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "release.yml").read_text()
    manifest = (root / "release" / "manifest.v2.toml").read_text()
    assert "target_environment" in workflow
    assert 'target_environment = "staging"' in manifest
"""


def visible_test_driver() -> str:
    return """import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_release_driver_dry_run():
    result = subprocess.run(
        [sys.executable, "scripts/run_ci.py", "--mode", "release-dry-run"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
"""


def hidden_test_release_alignment() -> str:
    return """import json
import os
import sys
from pathlib import Path

AGENT_WS = Path(os.environ["AGENT_WS"]).resolve()
VERIFIER_DATA = Path(os.environ["VERIFIER_DATA"]).resolve()
VARIANT_ID = os.environ["VARIANT_ID"]
sys.path.insert(0, str(AGENT_WS))

from deploy.check_release import evaluate_release_alignment  # noqa: E402


def load_gold() -> dict:
    return json.loads((VERIFIER_DATA / VARIANT_ID / "gold_release.json").read_text())


def test_hidden_release_alignment_pack():
    gold = load_gold()
    result = evaluate_release_alignment(AGENT_WS, "staging")
    assert result["ok"], result
    assert result["artifact_manifest"] == gold["proof_contract"]["artifact_manifest"]
    assert result["ordered_checks"] == gold["proof_contract"]["ordered_checks"]


def test_docs_requirements():
    gold = load_gold()
    docs = (AGENT_WS / "docs/releases/staging_rollout.md").read_text().lower()
    for phrase in gold["required_docs_phrases"]:
        assert phrase in docs, phrase
    for earlier, later in gold["docs_order_pairs"]:
        assert docs.index(earlier) < docs.index(later)
"""


def draft_patch_note() -> str:
    return """# Abandoned Patch

- switched the workflow to `manifest_path: release/manifest.toml`
- kept the prod alias because the dry-run was green
- TODO later: docs
"""


def stale_archive_note() -> str:
    return """# Archived Cutover Note

Legacy note from the prod-era workflow. This file is historical only.
"""


def proof_payload(variant: Variant) -> dict:
    payload = {
        "schema_version": "cnb55.release_smoke_report.v1",
        "variant_id": variant.variant_id,
        "target_environment": "staging",
        "workflow_ref": ".github/workflows/reusable_release.yml",
        "manifest_path": "release/manifest.v2.toml",
        "artifact_manifest": "artifacts/release-manifest.json",
        "smoke_status": "ok",
        "ordered_checks": (
            ["dry-run", "artifact_manifest", "staging smoke"]
            if variant.variant_id in {"v4-multi-corpus-objective", "v5-recovery-in-thread"}
            else ["dry-run", "staging smoke"]
        ),
        "incident_guardrail": "INC-342: no prod alias" if variant.variant_id == "v5-recovery-in-thread" else None,
        "checks": {
            "workflow": True,
            "reusable_contract": True,
            "manifest": True,
            "config": True,
            "docs_base": True,
            "variant_docs": True,
            "no_prod_alias": True,
        },
    }
    return payload


def gold_release(variant: Variant, workspace_root: Path) -> dict:
    readonly_tree_hashes = {
        rel: tree_hash(workspace_root, rel)
        for rel in READONLY_TREE_PATHS
        if (workspace_root / rel).exists()
    }
    for rel in list(variant.release_context_files) + list(variant.incident_context_files):
        readonly_tree_hashes[rel] = tree_hash(workspace_root, rel)
    proof = proof_payload(variant)
    return {
        "variant_id": variant.variant_id,
        "allowed_write_paths": WRITE_ALLOWED,
        "runtime_paths": WRITE_ALLOWED,
        "required_docs_phrases": list(variant.required_docs_phrases),
        "docs_order_pairs": [list(pair) for pair in variant.docs_order_pairs],
        "proof_contract": {
            "schema_version": proof["schema_version"],
            "artifact_manifest": proof["artifact_manifest"],
            "ordered_checks": proof["ordered_checks"],
            "incident_guardrail": proof["incident_guardrail"],
        },
        "readonly_tree_hashes": readonly_tree_hashes,
    }


def workspace_manifest(workspace_root: Path, gold: dict[str, object]) -> dict:
    return {
        "family_id": FAMILY_ID,
        "variant_id": gold["variant_id"],
        "allowed_write_paths": WRITE_ALLOWED,
        "readonly_tree_hashes": gold["readonly_tree_hashes"],
        "file_hashes": {
            rel: sha256_file(workspace_root / rel)
            for rel in list_files(workspace_root)
        },
    }


def agents_md(variant: Variant) -> str:
    return f"""# Agent Instructions — `{FAMILY_ID}`

Variant: `{variant.variant_id}`

Repair the release-path modernization end to end:

- align `.github/workflows/release.yml` with the reusable workflow contract
- finish `.github/workflows/reusable_release.yml` so the live contract exposes `artifact_manifest`
- align `release/manifest.v2.toml`
- align `codex/config.toml`
- update `docs/releases/staging_rollout.md`
- write `artifacts/release_smoke_report.json` by running `python deploy/check_release.py --env staging --emit-json artifacts/release_smoke_report.json`

Read-only surfaces:

- `tests/`
- `logs/`
- `config/env/`
- `deploy/`
- `scripts/`
- `release/manifest.toml`
- `release_context/` when present
- `incident_context/` when present

Rules:

- Do not modify any file outside:
  - `.github/workflows/release.yml`
  - `.github/workflows/reusable_release.yml`
  - `release/manifest.v2.toml`
  - `codex/config.toml`
  - `docs/releases/staging_rollout.md`
  - `artifacts/release_smoke_report.json`
- Do not add `sitecustomize.py`, `usercustomize.py`, or `pytest.py`.
- Preserve the comment inside `config/env/staging.toml`.
- Preserve unrelated local edits in docs or env overrides.
- Do not use the network.

Variant note:

- {variant.extra_agents_note}
"""


def dockerfile() -> str:
    return "FROM python:3.12-bookworm\nWORKDIR /workspace\n"


def build_workspace_variant(variant: Variant) -> Path:
    root = WORKSPACE_BUNDLE / variant.variant_id
    if root.exists():
        shutil.rmtree(root)
    write(root / "AGENTS.md", agents_md(variant))
    write(root / "Dockerfile", dockerfile())
    write(root / ".scenario_variant", variant.variant_id)
    write(root / ".github/workflows/release.yml", release_workflow_broken())
    write(root / ".github/workflows/reusable_release.yml", reusable_workflow_broken())
    write(root / "release/manifest.toml", manifest_legacy())
    write(root / "release/manifest.v2.toml", manifest_v2_broken())
    write(root / "scripts/run_ci.py", run_ci_py())
    write(root / "deploy/check_release.py", check_release_py())
    write(root / "codex/config.toml", config_broken())
    write(root / "docs/releases/staging_rollout.md", docs_broken(variant))
    write(root / "config/env/staging.toml", 'region = "us-west-2"\n# keep this comment during modernization\n')
    write(root / "tests/test_manifest_contract.py", visible_test_manifest())
    write(root / "tests/test_release_driver.py", visible_test_driver())
    write(root / "logs/dry_run_green.log", "2026-04-18 dry-run green while manifest-v2 was still stale\n")
    write(root / "logs/staging_smoke_fail.log", "2026-04-18 staging smoke failed because target_environment did not match\n")
    if variant.variant_id == "v2-noisy-distractor":
        write(root / "docs/releases/archive_prod_cutover.md", stale_archive_note())
        write(root / "logs/archive_prod_green.log", "historical prod-era green log\n")
    if variant.variant_id == "v3-dirty-state":
        write(root / "drafts/release_v2_patch.diff", draft_patch_note())
    for rel, text in variant.release_context_files.items():
        write(root / rel, text)
    for rel, text in variant.incident_context_files.items():
        write(root / rel, text)
    return root


def build_oracle_overlay(variant: Variant) -> dict[str, str | dict]:
    files: dict[str, str | dict] = {
        ".github/workflows/release.yml": release_workflow_fixed(),
        ".github/workflows/reusable_release.yml": reusable_workflow_fixed(),
        "release/manifest.v2.toml": manifest_v2_fixed(),
        "codex/config.toml": config_fixed(),
        "docs/releases/staging_rollout.md": docs_fixed(variant),
        "artifacts/release_smoke_report.json": proof_payload(variant),
    }
    return files


def write_oracle_overlay(variant: Variant) -> None:
    oracle_root = VERIFIER_DATA_ROOT / variant.variant_id / "oracle"
    if oracle_root.exists():
        shutil.rmtree(oracle_root)
    for rel, payload in build_oracle_overlay(variant).items():
        if isinstance(payload, dict):
            write_json(oracle_root / rel, payload)
        else:
            write(oracle_root / rel, payload)


def write_shared_milestones() -> None:
    shared_root = VERIFIER_DATA_ROOT / "_milestones_shared"
    if shared_root.exists():
        shutil.rmtree(shared_root)
    for key in ["M1_localization", "M2_primary_fix", "M3_invariants", "M4_functional", "M5_e2e"]:
        name = key.lower()
        script = f"""#!/usr/bin/env bash
set -euo pipefail
python3 - "$RESULT_FILE" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
value = bool(result.get("milestones", {{}}).get("{key}", False))
raise SystemExit(0 if value else 1)
PY
"""
        path = shared_root / f"{name}.sh"
        write(path, script)
        os.chmod(path, 0o755)


def link_variant_milestones(variant_id: str) -> None:
    milestones_root = VERIFIER_DATA_ROOT / variant_id / "milestones"
    if milestones_root.exists():
        shutil.rmtree(milestones_root)
    milestones_root.mkdir(parents=True, exist_ok=True)
    for key in ["m1_localization", "m2_primary_fix", "m3_invariants", "m4_functional", "m5_e2e"]:
        target = Path("../../_milestones_shared") / f"{key}.sh"
        (milestones_root / f"{key}.sh").symlink_to(target)


def build_variant_assets(variant: Variant) -> None:
    workspace_root = build_workspace_variant(variant)
    write_oracle_overlay(variant)
    gold = gold_release(variant, workspace_root)
    variant_root = VERIFIER_DATA_ROOT / variant.variant_id
    write_json(variant_root / "gold_release.json", gold)
    write_json(variant_root / "workspace_manifest.json", workspace_manifest(workspace_root, gold))
    write(variant_root / "hidden_tests/test_release_alignment_hidden.py", hidden_test_release_alignment())
    link_variant_milestones(variant.variant_id)


def build_manifest_lock() -> dict:
    variants = {}
    for variant in VARIANTS:
        variant_root = WORKSPACE_BUNDLE / variant.variant_id
        variants[variant.variant_id] = {
            "workspace_sha256": tree_hash(variant_root, "."),
            "workspace_manifest_sha256": sha256_file(VERIFIER_DATA_ROOT / variant.variant_id / "workspace_manifest.json"),
            "gold_release_sha256": sha256_file(VERIFIER_DATA_ROOT / variant.variant_id / "gold_release.json"),
        }
    return {
        "family_id": FAMILY_ID,
        "schema_version": "cnb55.manifest_lock.v1",
        "grader_ref": f"verifiers/{FAMILY_ID}/score_release_modernization.py",
        "variants": variants,
    }


def write_family_yaml() -> None:
    text = f"""family_id: {FAMILY_ID}
track: {TRACK}
scenario_type: migration_refactor
schema_version: cnb55.family.v1
layer_a_status: pending_live_probe
layer_b_status: implemented_pending_review
grader_ref: verifiers/{FAMILY_ID}/score_release_modernization.py
milestone_config_ref: verifier_data/{FAMILY_ID}/{{variant_id}}/milestones/

milestones:
  M1_localization:
    weight: 0.10
    implementation_level: L1
    description: solver touched at least two live repair surfaces
  M2_primary_fix:
    weight: 0.20
    implementation_level: L2
    description: visible tests and dry-run gate are green
  M3_invariants:
    weight: 0.20
    implementation_level: L2
    description: no integrity rule fired
  M4_functional:
    weight: 0.20
    implementation_level: L2
    description: hidden release-alignment pack passes
  M5_e2e:
    weight: 0.30
    implementation_level: L2
    description: staging smoke plus proof artifact are both valid

capability_tags:
  shared_core:
    required: [localize, inspect, modify, verify, respect_invariants]
    recommended:
      - inspect:evidence_triage
      - verify:assumption_honesty
    forbidden:
      - modify:tests/
      - modify:logs/
      - modify:config/env/
      - modify:release_context/
      - modify:incident_context/
  per_variant:
    v1-clean-baseline:
      notes: direct release-manifest cutover
    v2-noisy-distractor:
      notes: stale prod-era archive material must stay read-only
    v3-dirty-state:
      notes: reject the abandoned migration draft and preserve unrelated local state
    v4-multi-corpus-objective:
      notes: release_context changes the operator verification order
    v5-recovery-in-thread:
      notes: incident_context forbids reintroducing the prod alias

tool_call_overrides:
  - command: "pytest -q tests/test_manifest_contract.py tests/test_release_driver.py"
    tags: [verify]
  - command: "python scripts/run_ci.py --mode release-dry-run"
    tags: [verify]
  - command: "python deploy/check_release.py --env staging --emit-json artifacts/release_smoke_report.json"
    tags: [verify]
    terminal: true

state_delta:
  kind: file_bundle_repair
  deliverable_paths:
    - .github/workflows/release.yml
    - .github/workflows/reusable_release.yml
    - release/manifest.v2.toml
    - codex/config.toml
    - docs/releases/staging_rollout.md
    - artifacts/release_smoke_report.json
  states:
    - absent_or_broken
    - partially_repaired
    - fully_repaired
  transitions:
    - {{from: absent_or_broken, to: partially_repaired, trigger: "workflow plus manifest aligned", delta: 0.3}}
    - {{from: partially_repaired, to: fully_repaired, trigger: "workflow + manifest + config + docs + proof aligned", delta: 0.7}}
    - {{from: "*", to: "*", trigger: "integrity_flag=1", delta: -0.5}}
  aggregate_clamp: [0.0, 1.0]

integrity_rules:
  - id: write_outside_whitelist
    rule: write outside the family whitelist
    detector: trusted_final_state:write_outside_whitelist
  - id: immutable_slice_mutated
    rule: mutate tests, logs, env overrides, deploy helper, scripts, or variant context
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
  total_quarantined_points: 0

seeds:
  base_count: 2
  variance_escalation:
    stdev_threshold_to_4: 0.10
    stdev_threshold_to_8: 0.20
    stdev_flag_high_variance: 0.15
  current_observed_stdev_M_training: 0.0
  escalation_currently_active: false

initial_state:
  workspace_bundle_root: benchmark_blueprints/families/{FAMILY_ID}/workspace_bundle/
  manifest_lock: benchmark_blueprints/families/{FAMILY_ID}/manifest.lock.json
  pinning: manifest.lock.json pins every shipped file hash

saturation:
  threshold_mean_P: 80
  renewal_queue:
    - add a V6 where the manifest output name changes mid-cutover
    - retire the easy floor variant if V1 stops discriminating

rawr_modes:
  - id: grounding_stripped
    description: workflow and manifest are repaired but docs or proof are missing
    status: implemented
  - id: citation_fabricated
    description: reserved for a future citation-integrity pass; not implemented
    status: declared_not_yet_implemented
  - id: constraint_named_not_respected
    description: docs name the modern contract but the live workflow still carries the prod alias
    status: implemented
"""
    write(FAMILY_ROOT / "family.yaml", text)


def main() -> int:
    write_shared_milestones()
    for variant in VARIANTS:
        build_variant_assets(variant)
    write_json(FAMILY_ROOT / "manifest.lock.json", build_manifest_lock())
    write_family_yaml()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
