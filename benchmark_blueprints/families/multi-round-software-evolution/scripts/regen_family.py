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
CLI = FAMILY / "bin" / "cnb55-evolution"
SCORER = REPO / "verifiers/multi-round-software-evolution/score_round_plan.py"

VARIANT_ORDER = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]

READONLY_TREES = [
    ".scenario_variant",
    "AGENTS.md",
    "Dockerfile",
    "bin",
    "focus_candidates",
    "round_history",
    "objective_metrics",
    "repo_state",
    "release_context",
    "incident_context",
    "tests",
]

ORACLE_INPUTS = {
    "v1-clean-baseline": {
        "schema_version": "cnb55.evolution_plan.v1",
        "variant_id": "v1-clean-baseline",
        "selected_focus": {
            "focus_id": "schema-snapshot-alignment",
            "title": "Schema Snapshot Alignment",
            "why_now": "Round 4 replay still diverges because the snapshot writer and restore path disagree on field ordering. Fixing that invariant unblocks adaptive batching and keeps future streaming work from being built on sand.",
            "citations": [
                "focus_candidates/priority_map.md",
                "objective_metrics/scoreboard.md",
                "repo_state/open_threads.md",
            ],
        },
        "do_not_touch": [
            {
                "area": "contained-pager-cleanup",
                "reason": "The pager storm is already rate-limited and no longer blocks the roadmap; reopening it this round would burn time without unlocking the blocked follow-on work.",
                "citations": [
                    "round_history/round_log.md",
                    "objective_metrics/scoreboard.md",
                ],
            }
        ],
        "partial_progress_metric": {
            "metric_id": "snapshot-drift-repros",
            "baseline": "Replay diverges in 7 of 10 seeded restore runs after round 3.",
            "target": "Replay diverges in 0 of 10 seeded restore runs and adaptive-batching shadow runs stay byte-stable.",
            "guardrail": "Do not regress watermark mismatch rate above 1 in 20 replay runs while fixing snapshot ordering.",
            "measurement_plan": [
                "Run the seeded restore replay suite on every snapshot writer change.",
                "Compare the byte diff summary and watermark mismatch counter before declaring the round complete.",
            ],
        },
        "assumption_ledger": [
            {
                "topic": "Adaptive batching launch stays blocked until snapshot restore is stable",
                "status": "observed",
                "note": "Repo-state note ties the launch checklist directly to stable snapshot replay.",
            },
            {
                "topic": "Cold-start cache tuning remains secondary once snapshot drift is fixed",
                "status": "to_verify",
                "note": "The perf win matters, but the current scoreboard says it is not the gating dependency.",
            },
            {
                "topic": "Whether watermark parity still needs extra fixtures after snapshot alignment",
                "status": "missing",
                "note": "The current bundle does not say if the same snapshot fix is sufficient for all watermark edge cases.",
            },
        ],
    },
    "v2-noisy-distractor": {
        "schema_version": "cnb55.evolution_plan.v1",
        "variant_id": "v2-noisy-distractor",
        "selected_focus": {
            "focus_id": "schema-snapshot-alignment",
            "title": "Schema Snapshot Alignment",
            "why_now": "The stale pager snapshot makes the contained issue look loud again, but the current scoreboard and open-thread note still show snapshot drift as the blocker for any durable next step.",
            "citations": [
                "focus_candidates/priority_map.md",
                "objective_metrics/scoreboard.md",
                "repo_state/open_threads.md",
            ],
        },
        "do_not_touch": [
            {
                "area": "contained-pager-cleanup",
                "reason": "The noisy alert count is from the pre-rate-limit snapshot. Current pages are contained, so this should stay deferred unless fresh paging escapes the guardrail.",
                "citations": [
                    "round_history/round_log.md",
                    "objective_metrics/scoreboard.md",
                ],
            }
        ],
        "partial_progress_metric": {
            "metric_id": "snapshot-drift-repros",
            "baseline": "Replay diverges in 7 of 10 seeded restore runs while fresh pager pages remain at 0 in the last 7 days.",
            "target": "Replay diverges in 0 of 10 runs with fresh pager pages still at 0 and no new restore-order diffs.",
            "guardrail": "Do not reopen the contained pager path or reclassify stale snapshots as current signal.",
            "measurement_plan": [
                "Track seeded replay divergence and the fresh pager counter from the current scoreboard together.",
                "Require both replay stability and zero fresh-page regressions before closing the round.",
            ],
        },
        "assumption_ledger": [
            {
                "topic": "The stale pager snapshot is not a current blocker",
                "status": "observed",
                "note": "The scoreboard marks it historical and the round log says the rate limiter contained it.",
            },
            {
                "topic": "The schema ordering fix may reduce some watermark noise as a side-effect",
                "status": "to_verify",
                "note": "There is a plausible overlap, but the current evidence does not quantify it.",
            },
            {
                "topic": "Whether any fresh customer report bypasses the pager guardrail",
                "status": "missing",
                "note": "No post-containment customer report is included in the bundle.",
            },
        ],
    },
    "v3-dirty-state": {
        "schema_version": "cnb55.evolution_plan.v1",
        "variant_id": "v3-dirty-state",
        "selected_focus": {
            "focus_id": "schema-snapshot-alignment",
            "title": "Schema Snapshot Alignment",
            "why_now": "The abandoned cache-tuning patch is tempting, but it does not remove the replay invariant that keeps future work from sticking. The next round should fix the prerequisite before touching the partial perf path.",
            "citations": [
                "focus_candidates/priority_map.md",
                "objective_metrics/scoreboard.md",
                "repo_state/open_threads.md",
            ],
        },
        "do_not_touch": [
            {
                "area": "contained-pager-cleanup",
                "reason": "Still contained and not the gating dependency.",
                "citations": [
                    "round_history/round_log.md",
                    "objective_metrics/scoreboard.md",
                ],
            },
            {
                "area": "cold-start-cache-tuning",
                "reason": "The in-progress patch is abandoned partial work, not a head start; finishing it before the invariant is fixed would optimize a path that still replays incorrectly.",
                "citations": [
                    "repo_state/open_threads.md",
                    "focus_candidates/priority_map.md",
                ],
            },
        ],
        "partial_progress_metric": {
            "metric_id": "snapshot-drift-repros",
            "baseline": "Replay diverges in 7 of 10 seeded restore runs and the cache patch still sits half-wired behind a local flag.",
            "target": "Replay diverges in 0 of 10 runs while the abandoned cache patch remains deferred and untouched this round.",
            "guardrail": "Do not merge or extend the cold-start cache patch until replay determinism is proven stable.",
            "measurement_plan": [
                "Run the seeded restore replay suite after every ordering change.",
                "Verify the cache patch diff hash stays unchanged while replay divergence drops to zero.",
            ],
        },
        "assumption_ledger": [
            {
                "topic": "The partial cache patch does not unlock the blocked launch items",
                "status": "observed",
                "note": "Repo-state explicitly says the downstream work is still blocked by replay determinism.",
            },
            {
                "topic": "Cache tuning may be the follow-on round once snapshot replay is stable",
                "status": "to_verify",
                "note": "The focus map still lists it as worthwhile, just not first.",
            },
            {
                "topic": "Whether the abandoned patch contains any safe-to-reuse benchmark harness pieces",
                "status": "missing",
                "note": "The bundle shows the patch but not which pieces could be cleanly salvaged later.",
            },
        ],
    },
    "v4-multi-corpus-objective": {
        "schema_version": "cnb55.evolution_plan.v1",
        "variant_id": "v4-multi-corpus-objective",
        "selected_focus": {
            "focus_id": "streaming-watermark-alignment",
            "title": "Streaming Watermark Alignment",
            "why_now": "The release gate shifts the primary objective: the next launch is blocked more by streaming watermark mismatches than by raw replay drift. This round should move the gating customer objective first, while still respecting the invariant risk noted in the bundle.",
            "citations": [
                "focus_candidates/priority_map.md",
                "objective_metrics/scoreboard.md",
                "release_context/release_gate.md",
            ],
        },
        "do_not_touch": [
            {
                "area": "contained-pager-cleanup",
                "reason": "Still contained and not release-blocking.",
                "citations": [
                    "round_history/round_log.md",
                    "objective_metrics/scoreboard.md",
                ],
            },
            {
                "area": "cold-start-cache-tuning",
                "reason": "The partial cache patch remains a side quest relative to the release gate.",
                "citations": [
                    "repo_state/open_threads.md",
                    "release_context/release_gate.md",
                ],
            },
        ],
        "partial_progress_metric": {
            "metric_id": "watermark-mismatch-repros",
            "baseline": "Streaming watermark mismatch reproduces in 6 of 20 release-gate replay runs and blocks the customer launch checklist.",
            "target": "Streaming watermark mismatch reproduces in 0 of 20 release-gate replay runs while snapshot drift does not regress above its current 7 of 10 baseline.",
            "guardrail": "Do not treat the abandoned cache patch as progress and do not let snapshot replay drift get worse while chasing watermark parity.",
            "measurement_plan": [
                "Run the release-gate replay pack and count watermark mismatches after every candidate fix.",
                "Track the base snapshot-drift replay counter in parallel so the objective shift does not hide invariant regressions.",
            ],
        },
        "assumption_ledger": [
            {
                "topic": "The release gate makes watermark parity the immediate objective",
                "status": "observed",
                "note": "Release context names it as the blocker for the next customer rollout.",
            },
            {
                "topic": "Snapshot drift may still be a prerequisite risk for the watermark fix",
                "status": "to_verify",
                "note": "The release context elevates watermark parity, but repo-state still warns about the invariant under it.",
            },
            {
                "topic": "Whether the launch can tolerate partial improvement instead of zero mismatches",
                "status": "missing",
                "note": "The bundle states the blocker, but not the exact acceptability threshold short of zero.",
            },
        ],
    },
    "v5-recovery-in-thread": {
        "schema_version": "cnb55.evolution_plan.v1",
        "variant_id": "v5-recovery-in-thread",
        "selected_focus": {
            "focus_id": "schema-snapshot-alignment",
            "title": "Schema Snapshot Alignment",
            "why_now": "The watermark plan was already tried and rolled back because the unstable snapshot invariant underneath it corrupted recovery state. The next round should step back to the prerequisite invariant fix before retrying watermark work.",
            "citations": [
                "focus_candidates/priority_map.md",
                "repo_state/open_threads.md",
                "incident_context/rollback.md",
            ],
        },
        "do_not_touch": [
            {
                "area": "contained-pager-cleanup",
                "reason": "Still contained and still not the thing that failed the last attempt.",
                "citations": [
                    "round_history/round_log.md",
                    "objective_metrics/scoreboard.md",
                ],
            },
            {
                "area": "streaming-watermark-alignment",
                "reason": "Do not retry the rolled-back watermark plan until snapshot restore ordering is stable; otherwise the team repeats the same incident path.",
                "citations": [
                    "incident_context/rollback.md",
                    "repo_state/open_threads.md",
                ],
            },
        ],
        "partial_progress_metric": {
            "metric_id": "snapshot-drift-repros",
            "baseline": "The watermark round rolled back and replay still diverges in 7 of 10 seeded restore runs.",
            "target": "Replay diverges in 0 of 10 runs and the rollback reproduction no longer corrupts recovery state before any watermark retry is scheduled.",
            "guardrail": "Do not reopen the rolled-back watermark plan until the rollback reproduction passes cleanly and fresh customer streaming regressions stay flat.",
            "measurement_plan": [
                "Re-run the rollback reproduction after every snapshot-ordering change.",
                "Require both zero restore divergence and a clean rollback replay before re-queuing watermark alignment.",
            ],
        },
        "assumption_ledger": [
            {
                "topic": "The rollback was caused by retrying watermark work on top of unstable snapshot state",
                "status": "observed",
                "note": "Incident context names snapshot ordering as the prerequisite that was skipped.",
            },
            {
                "topic": "Watermark alignment may become the immediate next round once snapshot restore is stable",
                "status": "to_verify",
                "note": "That depends on whether the rollback reproduction stays clean after the invariant fix.",
            },
            {
                "topic": "Exact exit criteria for safely retrying the rolled-back watermark plan",
                "status": "missing",
                "note": "The incident writeup explains why the retry failed, but not the formal retry gate.",
            },
        ],
    },
}


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


def list_files(root: Path) -> list[str]:
    out: list[str] = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            if rel.startswith("brief/"):
                continue
            out.append(rel)
    return out


def readonly_hashes(ws: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in READONLY_TREES:
        digest = sha256_tree(ws, rel)
        if digest:
            out[rel] = digest
    return out


def run_submit(variant: str, data: dict) -> tuple[Path, Path]:
    with tempfile.TemporaryDirectory(prefix=f"mrse_oracle_{variant}_") as tmp:
        tmp_path = Path(tmp)
        ws = tmp_path / "workspace"
        shutil.copytree(WS_BUNDLE / variant, ws)
        inp = ws / "brief_input.json"
        inp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        subprocess.run([sys.executable, str(CLI), "submit", str(inp.name)], cwd=ws, check=True)
        oracle_dir = VER_DATA / variant / "oracle"
        oracle_dir.mkdir(parents=True, exist_ok=True)
        (oracle_dir / "brief_input.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        shutil.copy(ws / "brief" / "round_plan.json", oracle_dir / "round_plan.json")
        shutil.copy(ws / "brief" / "round_plan.md", oracle_dir / "round_plan.md")
        return oracle_dir / "round_plan.json", oracle_dir / "round_plan.md"


def score_workspace(ws: Path, variant: str) -> dict:
    result_file = ws / "verify_result.json"
    env = os.environ.copy()
    env.update(
        {
            "AGENT_WS": str(ws),
            "VERIFIER_DATA": str(VER_DATA),
            "VARIANT_ID": variant,
            "RESULT_FILE": str(result_file),
            "CNB55_SEED": "42",
        }
    )
    subprocess.run([sys.executable, str(SCORER)], env=env, check=True)
    return json.loads(result_file.read_text())


def shortcut_input(variant: str) -> dict:
    data = json.loads(json.dumps(ORACLE_INPUTS[variant]))
    if variant == "v4-multi-corpus-objective":
        data["selected_focus"]["focus_id"] = "schema-snapshot-alignment"
        data["selected_focus"]["title"] = "Schema Snapshot Alignment"
        data["selected_focus"]["why_now"] = "Keep pushing the old invariant work."
    elif variant == "v5-recovery-in-thread":
        data["selected_focus"]["focus_id"] = "streaming-watermark-alignment"
        data["selected_focus"]["title"] = "Streaming Watermark Alignment"
        data["selected_focus"]["why_now"] = "Retry the same watermark plan immediately."
    elif variant == "v3-dirty-state":
        data["selected_focus"]["focus_id"] = "cold-start-cache-tuning"
        data["selected_focus"]["title"] = "Cold-Start Cache Tuning"
        data["selected_focus"]["why_now"] = "There is already a partial patch, so finish it."
    else:
        data["selected_focus"]["focus_id"] = "contained-pager-cleanup"
        data["selected_focus"]["title"] = "Contained Pager Cleanup"
        data["selected_focus"]["why_now"] = "The loudest issue should get the whole round."
    data["selected_focus"]["citations"] = [
        "objective_metrics/scoreboard.md",
        "round_history/round_log.md",
    ]
    data["do_not_touch"] = [
        {
            "area": "dashboard-polish",
            "reason": "Skip unrelated polish work.",
            "citations": [
                "focus_candidates/priority_map.md",
            ],
        }
    ]
    data["partial_progress_metric"]["baseline"] = "Improve it."
    data["partial_progress_metric"]["target"] = "Make it better."
    data["partial_progress_metric"]["guardrail"] = "Avoid obvious breakage while we move fast."
    data["partial_progress_metric"]["measurement_plan"] = ["Check it later", "Ship if it looks okay"]
    return data


def main() -> int:
    VER_DATA.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []

    for variant in VARIANT_ORDER:
        ws = WS_BUNDLE / variant
        readonly = readonly_hashes(ws)
        files = list_files(ws)

        ver_variant = VER_DATA / variant
        (ver_variant / "hidden_tests").mkdir(parents=True, exist_ok=True)
        (ver_variant / "hidden_tests" / "README.md").write_text(
            "Hidden checks are encoded in score_round_plan.py and gold_plan.json for this family.\n"
        )

        gold = {
            "variant_id": variant,
            "accepted_focus_id": ORACLE_INPUTS[variant]["selected_focus"]["focus_id"],
            "expected_boundary_ids": [row["area"] for row in ORACLE_INPUTS[variant]["do_not_touch"]],
            "expected_assumption_keywords": {
                "v1-clean-baseline": ["watermark"],
                "v2-noisy-distractor": ["fresh customer report"],
                "v3-dirty-state": ["safe-to-reuse"],
                "v4-multi-corpus-objective": ["acceptability threshold"],
                "v5-recovery-in-thread": ["retry gate", "rolled-back"],
            }[variant],
            "unlock_keywords": {
                "v1-clean-baseline": ["adaptive batching", "future"],
                "v2-noisy-distractor": ["blocker", "current"],
                "v3-dirty-state": ["prerequisite", "future"],
                "v4-multi-corpus-objective": ["release", "customer"],
                "v5-recovery-in-thread": ["prerequisite", "incident"],
            }[variant],
            "pass_bar": 70,
            "shortcut_focus_id": shortcut_input(variant)["selected_focus"]["focus_id"],
            "readonly_tree_hashes": readonly,
            "test_evolution_plan_sha256": sha256_file(ws / "tests" / "test_evolution_plan.py"),
        }
        (ver_variant / "gold_plan.json").write_text(json.dumps(gold, indent=2, sort_keys=True) + "\n")

        manifest = {
            "variant_id": variant,
            "files": files,
            "readonly_tree_hashes": readonly,
            "test_evolution_plan_sha256": sha256_file(ws / "tests" / "test_evolution_plan.py"),
        }
        (ver_variant / "workspace_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        oracle_json, oracle_md = run_submit(variant, ORACLE_INPUTS[variant])

        with tempfile.TemporaryDirectory(prefix=f"mrse_score_{variant}_") as tmp:
            tmp_path = Path(tmp)

            oracle_ws = tmp_path / "oracle"
            shutil.copytree(ws, oracle_ws)
            (oracle_ws / "brief").mkdir(exist_ok=True)
            shutil.copy(oracle_json, oracle_ws / "brief" / "round_plan.json")
            shutil.copy(oracle_md, oracle_ws / "brief" / "round_plan.md")
            oracle_score = score_workspace(oracle_ws, variant)["P_benchmark"]

            empty_ws = tmp_path / "empty"
            shutil.copytree(ws, empty_ws)
            empty_score = score_workspace(empty_ws, variant)["P_benchmark"]

            shortcut_ws = tmp_path / "shortcut"
            shutil.copytree(ws, shortcut_ws)
            shortcut_inp = shortcut_ws / "brief_input.json"
            shortcut_inp.write_text(json.dumps(shortcut_input(variant), indent=2, sort_keys=True) + "\n")
            subprocess.run([sys.executable, str(CLI), "submit", str(shortcut_inp.name)], cwd=shortcut_ws, check=True)
            shortcut_score = score_workspace(shortcut_ws, variant)["P_benchmark"]

        summary.append(
            {
                "variant": variant,
                "oracle": oracle_score,
                "empty": empty_score,
                "shortcut": shortcut_score,
                "oracle_json_sha256": sha256_file(oracle_json),
                "oracle_md_sha256": sha256_file(oracle_md),
            }
        )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
