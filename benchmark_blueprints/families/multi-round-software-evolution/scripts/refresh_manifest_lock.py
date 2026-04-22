#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
FAMILY = REPO / "benchmark_blueprints/families/multi-round-software-evolution"
WS_BUNDLE = FAMILY / "workspace_bundle"
VER_DATA = REPO / "verifier_data/multi-round-software-evolution"
LOCK = FAMILY / "manifest.lock.json"
SCORER = REPO / "verifiers/multi-round-software-evolution/score_round_plan.py"
VERIFY_SH = REPO / "verifiers/multi-round-software-evolution/verify.sh"
CLI = FAMILY / "bin" / "cnb55-evolution"

VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]

WS_TREES = (
    ".scenario_variant",
    "AGENTS.md",
    "Dockerfile",
    "artifacts",
    "bin",
    "brief",
    "focus_candidates",
    "round_history",
    "objective_metrics",
    "repo_state",
    "release_context",
    "incident_context",
    "tests",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_tree(root: Path, rel: str) -> str | None:
    target = root / rel
    if not target.exists():
        return None
    h = hashlib.sha256()
    if target.is_file():
        h.update(b"F")
        h.update(sha256_file(target).encode())
        return h.hexdigest()
    for p in sorted(target.rglob("*")):
        rp = p.relative_to(target).as_posix()
        if p.is_file():
            h.update(b"F:" + rp.encode() + b"\x00")
            h.update(sha256_file(p).encode() + b"\x00")
        elif p.is_dir():
            h.update(b"D:" + rp.encode() + b"\x00")
    return h.hexdigest()


def hidden_tests_sha(vd: Path) -> str | None:
    ht = vd / "hidden_tests"
    if not ht.exists():
        return None
    return sha256_tree(ht, ".")


def score_variant(variant: str, mode: str) -> int:
    ws = WS_BUNDLE / variant
    with tempfile.TemporaryDirectory(prefix=f"mrse_lock_{variant}_{mode}_") as tmp:
        tmp_ws = Path(tmp) / "workspace"
        shutil.copytree(ws, tmp_ws)
        if mode == "oracle":
            shutil.copy(VER_DATA / variant / "oracle" / "round_plan.json", tmp_ws / "brief" / "round_plan.json")
            shutil.copy(VER_DATA / variant / "oracle" / "round_plan.md", tmp_ws / "brief" / "round_plan.md")
        elif mode == "shortcut":
            shortcut = {
                "v1-clean-baseline": "contained-pager-cleanup",
                "v2-noisy-distractor": "contained-pager-cleanup",
                "v3-dirty-state": "cold-start-cache-tuning",
                "v4-multi-corpus-objective": "schema-snapshot-alignment",
                "v5-recovery-in-thread": "streaming-watermark-alignment",
            }[variant]
            payload = json.loads((VER_DATA / variant / "oracle" / "round_plan.json").read_text())
            payload["selected_focus"]["focus_id"] = shortcut
            payload["selected_focus"]["title"] = shortcut.replace("-", " ").title()
            payload["selected_focus"]["why_now"] = "Shortcut choice."
            payload["selected_focus"]["citations"] = [
                "objective_metrics/scoreboard.md",
                "round_history/round_log.md",
            ]
            payload["do_not_touch"] = [
                {
                    "area": "dashboard-polish",
                    "reason": "Skip unrelated polish work.",
                    "citations": ["focus_candidates/priority_map.md"],
                }
            ]
            payload["partial_progress_metric"]["baseline"] = "Improve it."
            payload["partial_progress_metric"]["target"] = "Make it better."
            payload["partial_progress_metric"]["guardrail"] = "Avoid obvious breakage while we move fast."
            payload["partial_progress_metric"]["measurement_plan"] = ["Check it later", "Ship if it looks okay"]
            inp = tmp_ws / "brief_input.json"
            inp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            subprocess.run([sys.executable, str(CLI), "submit", str(inp.name)], cwd=tmp_ws, check=True)
        result_file = tmp_ws / "verify_result.json"
        env = os.environ.copy()
        env.update(
            {
                "AGENT_WS": str(tmp_ws),
                "VERIFIER_DATA": str(VER_DATA),
                "VARIANT_ID": variant,
                "RESULT_FILE": str(result_file),
                "CNB55_SEED": "42",
            }
        )
        subprocess.run([sys.executable, str(SCORER)], env=env, check=True)
        return json.loads(result_file.read_text())["P_benchmark"]


def main() -> int:
    lock = {
        "family_id": "multi-round-software-evolution",
        "track": 10,
        "track_name": "Strategic Management & Long-Horizon Evolution",
        "schema_version": "cnb55.manifest.v2",
        "calibration_targets": {
            "family_mean_center": 20,
            "family_mean_window": [15, 25],
            "max_variant_score": 40,
            "min_hard_variant_score": 10,
            "monotonicity_tolerance": 3,
        },
        "determinism": {"cnb55_seed": 42, "result_key_sort": "sorted"},
        "cli": {"cnb55_evolution_sha256": sha256_file(CLI)},
        "grader": {
            "score_round_plan_py_sha256": sha256_file(SCORER),
            "verify_sh_sha256": sha256_file(VERIFY_SH),
        },
        "variants": {},
    }

    for variant in VARIANTS:
        vd = VER_DATA / variant
        ws = WS_BUNDLE / variant
        gold = json.loads((vd / "gold_plan.json").read_text())
        entry = {
            "observed_empty_brief_score": score_variant(variant, "empty"),
            "observed_oracle_score": score_variant(variant, "oracle"),
            "observed_shortcut_score": score_variant(variant, "shortcut"),
            "verifier_data": {
                "gold_plan_sha256": sha256_file(vd / "gold_plan.json"),
                "workspace_manifest_sha256": sha256_file(vd / "workspace_manifest.json"),
                "oracle_round_plan_json_sha256": sha256_file(vd / "oracle" / "round_plan.json"),
                "oracle_round_plan_md_sha256": sha256_file(vd / "oracle" / "round_plan.md"),
            },
            "workspace_trees": {},
        }
        ht = hidden_tests_sha(vd)
        if ht:
            entry["verifier_data"]["hidden_tests_tree_sha256"] = ht
        for rel in WS_TREES:
            digest = sha256_tree(ws, rel)
            if digest:
                entry["workspace_trees"][rel] = digest
        entry["gold_pass_bar"] = gold["pass_bar"]
        lock["variants"][variant] = entry

    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(f"wrote {LOCK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
