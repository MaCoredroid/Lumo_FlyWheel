import argparse
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
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
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
