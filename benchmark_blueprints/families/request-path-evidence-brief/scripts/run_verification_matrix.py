#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
FAMILY = REPO / "benchmark_blueprints/families/request-path-evidence-brief"
WS_BUNDLE = FAMILY / "workspace_bundle"
VERIFIER_DATA = REPO / "verifier_data/request-path-evidence-brief"
SCORER = REPO / "verifiers/request-path-evidence-brief/score_trace.py"


def load_oracle(variant: str) -> dict[str, str]:
    oracle_dir = VERIFIER_DATA / variant / "oracle"
    return {
        "path_map.json": (oracle_dir / "path_map.json").read_text(),
        "request_path_brief.md": (oracle_dir / "request_path_brief.md").read_text(),
        "docs_correction.md": (oracle_dir / "docs_correction.md").read_text(),
    }


def write_artifact_set(ws: Path, payloads: dict[str, str]) -> None:
    artifacts = ws / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        (artifacts / name).write_text(content, encoding="utf-8")


def make_oracle(ws: Path, variant: str) -> None:
    write_artifact_set(ws, load_oracle(variant))


def make_empty(ws: Path, variant: str) -> None:
    del ws, variant


def make_grounding_stripped(ws: Path, variant: str) -> None:
    payloads = load_oracle(variant)
    payloads["request_path_brief.md"] = "The path is in the repo.\n"
    payloads["docs_correction.md"] = "Docs need an update.\n"
    write_artifact_set(ws, payloads)


def make_store_decoy(ws: Path, variant: str) -> None:
    payload = {
        "schema_version": "cnb55.request_path_map.v1",
        "variant_id": variant,
        "summary": "The store decides both fields.",
        "live_path": [
            {
                "step": 1,
                "file": "sync_app/store.py",
                "symbol": "legacy_make_record_with_owner_source",
                "role": "storage_decision",
                "caller_symbol": "ops.support_note",
                "callee_symbol": None,
            }
        ],
        "field_derivations": {
            "owner_source": {"file": "sync_app/store.py", "symbol": "legacy_make_record_with_owner_source"},
            "routing_key": {"file": "sync_app/store.py", "symbol": "legacy_build_routing_key"},
            "emission": {"file": "sync_app/store.py", "symbol": "legacy_make_record_with_owner_source"},
        },
        "test_observations": [],
        "rejected_decoys": [],
    }
    write_artifact_set(
        ws,
        {
            "path_map.json": json.dumps(payload, indent=2, sort_keys=True) + "\n",
            "request_path_brief.md": "Storage owns the final payload, so the support note is correct.\n",
            "docs_correction.md": "No correction needed.\n",
        },
    )


def make_wrong_order(ws: Path, variant: str) -> None:
    payloads = load_oracle(variant)
    data = json.loads(payloads["path_map.json"])
    live_path = list(reversed(data["live_path"]))
    for idx, step in enumerate(live_path, start=1):
        step["step"] = idx
    data["live_path"] = live_path
    payloads["path_map.json"] = json.dumps(data, indent=2, sort_keys=True) + "\n"
    write_artifact_set(ws, payloads)


def make_delete_tests(ws: Path, variant: str) -> None:
    make_oracle(ws, variant)
    target = ws / "tests" / "test_trace_outputs.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")


TRAJECTORIES = [
    ("01_oracle", make_oracle, "Oracle"),
    ("02_empty", make_empty, "Empty"),
    ("03_rawr_grounding_stripped", make_grounding_stripped, "RAWR grounding_stripped"),
    ("04_pick_store_decoy", make_store_decoy, "Pick store decoy"),
    ("05_wrong_order", make_wrong_order, "Wrong live order"),
    ("06_delete_tests", make_delete_tests, "Delete-tests adversarial"),
]


def score_trajectory(variant: str, builder, tag: str) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"vmx_{tag}_") as tmp:
        ws = Path(tmp) / "workspace"
        shutil.copytree(WS_BUNDLE / variant, ws)
        result_file = Path(tmp) / "verify_result.json"
        builder(ws, variant)
        env = os.environ.copy()
        env.update(
            {
                "AGENT_WS": str(ws),
                "VERIFIER_DATA": str(VERIFIER_DATA),
                "VARIANT_ID": variant,
                "RESULT_FILE": str(result_file),
            }
        )
        subprocess.run([sys.executable, str(SCORER)], env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return json.loads(result_file.read_text())


def derive_g(result: dict) -> float:
    h = 1.0 if result.get("integrity_flag", 0) == 0 else 0.0
    i = 1.0 if result.get("milestones", {}).get("M1_localization", False) else 0.0
    s = float(result.get("M_training", 0.0))
    return round(0.15 * h + 0.25 * i + 0.60 * s, 4)


def derive_r(result: dict) -> float:
    h = int(result.get("integrity_flag", 0))
    m_agg = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
    if result.get("pass", False):
        return 1.0
    return round(0.6 * m_agg - 0.6 * h, 4)


def derive_sttc(result: dict) -> int:
    p = 1 if result.get("pass", False) else 0
    m_agg = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
    h = int(result.get("integrity_flag", 0))
    g = derive_g(result)
    return int(round(1000 * p + 100 * m_agg - 100 * h + 10 * g))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="v1-clean-baseline")
    parser.add_argument("--out", default=str(FAMILY / "verification_matrix.md"))
    args = parser.parse_args()

    rows = []
    for tag, builder, label in TRAJECTORIES:
        result = score_trajectory(args.variant, builder, tag)
        rows.append(
            {
                "label": label,
                "P_benchmark": result["P_benchmark"],
                "M_training": result["M_training"],
                "G": derive_g(result),
                "R": derive_r(result),
                "S_TTC": derive_sttc(result),
                "integrity_flag": result["integrity_flag"],
                "pass": result["pass"],
                "ceilings_applied": result.get("ceilings_applied", []),
                "integrity_rules_fired": result.get("integrity_rules_fired", []),
            }
        )

    out_path = Path(args.out)
    out_path.write_text(
        "# §5 verification matrix — {variant}\n\n".format(variant=args.variant)
        + "Generated by `benchmark_blueprints/families/request-path-evidence-brief/scripts/run_verification_matrix.py`.\n\n"
        + "| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |\n"
        + "|---|---:|---:|---:|---:|---:|---:|---|---|\n"
        + "\n".join(
            "| {label} | {P_benchmark} | {M_training:.4f} | {G:.3f} | {R:.3f} | {S_TTC} | {integrity_flag} | {passed} | {ceilings} |".format(
                label=row["label"],
                P_benchmark=row["P_benchmark"],
                M_training=row["M_training"],
                G=row["G"],
                R=row["R"],
                S_TTC=row["S_TTC"],
                integrity_flag=row["integrity_flag"],
                passed=row["pass"],
                ceilings=("H=" + ",".join(row["integrity_rules_fired"])) if row["integrity_rules_fired"] else (",".join(row["ceilings_applied"]) or "—"),
            )
            for row in rows
        )
        + "\n\n## Expected bands\n\n"
        + "- Oracle: P >= 90, M >= 0.90\n"
        + "- Empty: P = 0, M = 0\n"
        + "- RAWR grounding_stripped: weak markdown grounding ceiling or reduced grounding points\n"
        + "- Pick store decoy: storage-layer ceilings fire and P <= 30\n"
        + "- Delete-tests adversarial: integrity_flag = 1\n",
        encoding="utf-8",
    )
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
