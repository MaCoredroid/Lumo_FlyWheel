#!/usr/bin/env python3
"""
Regenerate CNB-55 family artifacts to match the v2 (CLI-based) pipeline.

For each variant under
  benchmark_blueprints/families/proposal-ranking-manager-judgment/workspace_bundle/
and matching verifier_data/proposal-ranking-manager-judgment/{variant}/, we:

1. Recompute the readonly tree hashes to include bin/ and the rewritten
   AGENTS.md.
2. Rewrite verifier_data/.../workspace_manifest.json to list every workspace
   file (including bin/cnb55-brief) and the refreshed readonly_tree_hashes.
3. Refresh verifier_data/.../gold_ranking.json with the same readonly hashes.
4. Author an oracle brief_input.json at
   verifier_data/.../oracle/brief_input.json and run the v2 CLI inside a
   temp workspace copy to produce canonical brief/manager_brief.{json,md}.
   Copy the resulting canonical files back into verifier_data/.../oracle/.
5. Run score_ranking.py against a temp workspace populated with the oracle
   brief to confirm score >= 90. Also run empty-brief (score 0) and
   shortcut-brief (accept staffing-blocked P3) and record their scores.

Prints a single summary table at the end.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FAMILY = REPO / "benchmark_blueprints/families/proposal-ranking-manager-judgment"
WS_BUNDLE = FAMILY / "workspace_bundle"
VER_DATA = REPO / "verifier_data/proposal-ranking-manager-judgment"
CLI = FAMILY / "bin" / "cnb55-brief"
SCORER = REPO / "verifiers/proposal-ranking-manager-judgment/score_ranking.py"

VARIANTS = [
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
    "proposals",
    "repo_evidence",
    "release_context",
    "incident_context",
    "tests",
]


# --------------------------------------------------------------- hashing


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


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
        rel_p = p.relative_to(target).as_posix()
        if p.is_file():
            h.update(b"F:" + rel_p.encode() + b"\x00")
            h.update(sha256_file(p).encode() + b"\x00")
        elif p.is_dir():
            h.update(b"D:" + rel_p.encode() + b"\x00")
    return h.hexdigest()


def list_files(root: Path) -> list[str]:
    out: list[str] = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            # skip brief/ (agent output dir, not readonly)
            if rel.startswith("brief/"):
                continue
            out.append(rel)
    return out


def compute_readonly_hashes(ws: Path) -> dict[str, str]:
    h: dict[str, str] = {}
    for rel in READONLY_TREES:
        if (ws / rel).exists():
            digest = sha256_tree(ws, rel)
            if digest:
                h[rel] = digest
    return h


# ---------------------------------------------- manifest + gold updates


def update_manifest(variant: str, readonly_hashes: dict[str, str],
                    files: list[str]) -> None:
    p = VER_DATA / variant / "workspace_manifest.json"
    data = json.loads(p.read_text())
    data["files"] = files
    data["readonly_tree_hashes"] = readonly_hashes
    # keep existing test_ranking_brief_sha256 / variant_id
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def update_gold(variant: str, readonly_hashes: dict[str, str]) -> None:
    p = VER_DATA / variant / "gold_ranking.json"
    data = json.loads(p.read_text())
    data["readonly_tree_hashes"] = readonly_hashes
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------- oracle brief generators


def oracle_input_v1() -> dict:
    return {
        "schema_version": "cnb55.manager_brief.v2",
        "variant_id": "v1-clean-baseline",
        "accepted": "P4",
        "primary_risk": {
            "statement": (
                "Schema-cache warm-up cost on cold process start can regress "
                "p99 start-up latency for freshly scheduled replicas until the "
                "compiled cache fills, with Priya at only 40% Q3 (parental "
                "leave handoff) Ravi must cover the ramp-down reviewer slot."
            ),
            "mitigations": [
                "staged rollout behind a feature flag at 1%/10%/50%/100%",
                "pre-warm the cache via shadow traffic replay before real traffic",
                "kill switch reverts to the non-compiled path on SLO breach",
                "canary observability on start-up p99 for 24h before ramping",
                "Ravi cover during Priya ramp-down (mid-quarter handoff) so "
                "rollout gates are not blocked on a single 40% reviewer",
            ],
        },
        "ranking": [
            {
                "proposal_id": "P4",
                "rank": 1,
                "summary": (
                    "Surgical schema-compile warm-start pays down the dominant "
                    "hot-path segment and is staffable in Q3 with Ravi covering "
                    "during Priya's 40% parental leave ramp-down."
                ),
                "citations": [
                    "proposals/P4.md",
                    "repo_evidence/perf/schema_compile_bench.md",
                    "repo_evidence/meta/staffing.md",
                    "repo_evidence/meta/roster_memo.md",
                    "repo_evidence/code/normalizer_excerpt.py",
                    "repo_evidence/perf/hot_path_flame_2026_04_02.txt",
                ],
                "constraint_tags": ["staffing", "rollout", "perf"],
            },
            {
                "proposal_id": "P2",
                "rank": 2,
                "summary": (
                    "L2 response cache is a close-call trade-off vs P4; real "
                    "win but coupled to watermark without staged rollout."
                ),
                "citations": [
                    "proposals/P2.md",
                    "repo_evidence/perf/l2_cache_shadow_replay.md",
                    "repo_evidence/meta/rollout_history.md",
                ],
                "constraint_tags": ["rollout", "regression"],
            },
            {
                "proposal_id": "P1",
                "rank": 3,
                "summary": (
                    "Dropping structural validation is perf-positive but "
                    "regressed SDK contracts twice in Q2 rollout history."
                ),
                "citations": [
                    "proposals/P1.md",
                    "repo_evidence/perf/hot_path_flame_2026_04_02.txt",
                    "repo_evidence/meta/rollout_history.md",
                ],
                "constraint_tags": ["regression", "rollout"],
            },
            {
                "proposal_id": "P3",
                "rank": 4,
                "summary": (
                    "Rust rewrite is staffing-blocked this quarter (Diego at "
                    "0%); headline delta is unstaffable primary work."
                ),
                "citations": [
                    "proposals/P3.md",
                    "repo_evidence/perf/rust_prototype_notes.md",
                    "repo_evidence/meta/staffing.md",
                ],
                "constraint_tags": ["staffing", "regression"],
            },
        ],
        "assumption_ledger": [
            {
                "topic": "Schema-compile p99 start-up regression",
                "status": "to_verify",
                "note": "Must measure cold-start p99 on canary before ramp.",
            },
            {
                "topic": "L2 cache / watermark coupling failure enumeration",
                "status": "missing",
                "note": (
                    "Shadow replay flags the risk but does not enumerate the "
                    "failure paths; would want a deeper review before re-entering."
                ),
            },
            {
                "topic": "Diego Q4 return date",
                "status": "to_verify",
                "note": "Staffing doc lists 0% Q3 but no explicit Q4 allocation.",
            },
            {
                "topic": "Ravi cover bandwidth during Priya 40% ramp-down",
                "status": "to_verify",
                "note": (
                    "Mid-quarter staffing update pre-approves Ravi cover but "
                    "does not name a backup if Ravi is also over-subscribed."
                ),
            },
        ],
    }


def oracle_input_v2() -> dict:
    d = oracle_input_v1()
    d["variant_id"] = "v2-noisy-distractor"
    # Insert P5 as rank 3, push others down
    d["ranking"] = [
        d["ranking"][0],  # P4 rank 1 (inherits staffing_update cite)
        d["ranking"][1],  # P2 rank 2
        {
            "proposal_id": "P5",
            "rank": 3,
            "summary": (
                "Validator microservice perf numbers are stale (January 2026 "
                "bench / jan_bench, outdated and superseded pre-streaming) "
                "and ownership is ambiguous after Kenji handoff."
            ),
            "citations": [
                "proposals/P5.md",
                "repo_evidence/perf/validator_service_jan_bench.md",
                "repo_evidence/meta/staffing.md",
            ],
            "constraint_tags": ["staffing", "regression"],
        },
        {**d["ranking"][2], "rank": 4},  # P1 -> rank 4
        {**d["ranking"][3], "rank": 5},  # P3 -> rank 5
    ]
    d["assumption_ledger"].insert(
        0,
        {
            "topic": "Whether the stale Jan bench has been re-run post handoff",
            "status": "missing",
            "note": (
                "validator_service_jan_bench.md is outdated; no fresh bench is "
                "on file to supersede it."
            ),
        },
    )
    return d


def oracle_input_v3() -> dict:
    d = oracle_input_v2()
    d["variant_id"] = "v3-dirty-state"
    # V3 adds a SECOND abandoned prototype (in_progress_patch_v2 - validator
    # scaffolding). Reflect both in rationale and ledger.
    d["ranking"][4]["summary"] = (
        "Rust rewrite; the in-progress abandoned skeleton is NOT a head start "
        "— sunk cost, abandoned, staffing unchanged (Diego 0%)."
    )
    d["ranking"][4]["citations"] = [
        "proposals/P3.md",
        "repo_evidence/perf/rust_prototype_notes.md",
        "repo_evidence/meta/staffing.md",
        "repo_evidence/code/in_progress_patch/README.md",
    ]
    # P5 rejection rationale also notes the validator scaffolding is
    # abandoned, not a head start.
    d["ranking"][2]["summary"] = (
        "Validator microservice perf numbers are stale (January 2026 "
        "bench / jan_bench, outdated and superseded pre-streaming). "
        "The in_progress_patch_v2 validator scaffolding is abandoned "
        "(Kenji transferred) — sunk cost, not a head start."
    )
    d["ranking"][2]["citations"] = [
        "proposals/P5.md",
        "repo_evidence/perf/validator_service_jan_bench.md",
        "repo_evidence/meta/staffing.md",
        "repo_evidence/code/in_progress_patch_v2/README.md",
    ]
    d["assumption_ledger"].insert(
        0,
        {
            "topic": "Whether the in-progress Rust patch is resumed in Q3",
            "status": "missing",
            "note": (
                "Code excerpt looks abandoned; maintainability cost is unbounded "
                "and resumption is not staffed."
            ),
        },
    )
    d["assumption_ledger"].insert(
        1,
        {
            "topic": "Ownership handoff for abandoned validator scaffolding",
            "status": "missing",
            "note": (
                "in_progress_patch_v2 was authored by Kenji (transferred); "
                "no new owner, no target date. Treat as sunk cost."
            ),
        },
    )
    return d


def oracle_input_v4() -> dict:
    return {
        "schema_version": "cnb55.manager_brief.v2",
        "variant_id": "v4-multi-corpus-objective",
        "accepted": "P2",
        "primary_risk": {
            "statement": (
                "L2 cache invalidation coupling with the streaming watermark "
                "module can regress truncation guarantees under load, and "
                "watermark state changes landed AFTER the shadow replay numbers "
                "were collected (INC-2427) — so the replay baseline is a "
                "conservative lower-bound, not a ship-ready SLO."
            ),
            "mitigations": [
                "feature flag gated 1%/10% canary",
                "kill switch back to no-cache path on watermark lag",
                "shadow replay comparing watermark boundaries with/without cache",
                "SLO observability on truncation rate during rollout",
                "staged rollout halts at 10% if the INC-2427 watermark regression "
                "reappears; Mei pre-warm reviewer for the watermark coupling gate",
            ],
        },
        "ranking": [
            {
                "proposal_id": "P2",
                "rank": 1,
                "summary": (
                    "L2 cache now pairs latency with streaming reliability "
                    "isolation; matches the new objective (INC-2411/2419/2427)."
                ),
                "citations": [
                    "proposals/P2.md",
                    "repo_evidence/perf/l2_cache_shadow_replay.md",
                    "release_context/release_notes_2026_03.md",
                ],
                "constraint_tags": ["rollout", "release", "perf"],
            },
            {
                "proposal_id": "P5",
                "rank": 2,
                "summary": (
                    "Validator microservice delivers reliability isolation "
                    "under the new streaming objective; January 2026 jan_bench "
                    "perf is stale and superseded pre-streaming — flagged here."
                ),
                "citations": [
                    "proposals/P5.md",
                    "repo_evidence/perf/validator_service_jan_bench.md",
                    "release_context/release_notes_2026_03.md",
                ],
                "constraint_tags": ["release", "regression"],
            },
            {
                "proposal_id": "P4",
                "rank": 3,
                "summary": (
                    "Surgical warm-start is strong latency win but does NOT "
                    "advance streaming-reliability — mismatch vs current objective."
                ),
                "citations": [
                    "proposals/P4.md",
                    "repo_evidence/perf/schema_compile_bench.md",
                    "release_context/release_notes_2026_03.md",
                ],
                "constraint_tags": ["release", "perf"],
            },
            {
                "proposal_id": "P1",
                "rank": 4,
                "summary": (
                    "Fast-ship validation drop; does not advance streaming "
                    "reliability and keeps SDK regression risk."
                ),
                "citations": [
                    "proposals/P1.md",
                    "repo_evidence/meta/rollout_history.md",
                    "release_context/release_notes_2026_03.md",
                ],
                "constraint_tags": ["release", "regression"],
            },
            {
                "proposal_id": "P3",
                "rank": 5,
                "summary": (
                    "Rust rewrite; sunk-cost abandoned patch; staffing-blocked "
                    "(Diego 0%); does not advance streaming reliability."
                ),
                "citations": [
                    "proposals/P3.md",
                    "repo_evidence/perf/rust_prototype_notes.md",
                    "repo_evidence/meta/staffing.md",
                ],
                "constraint_tags": ["staffing", "regression"],
            },
        ],
        "assumption_ledger": [
            {
                "topic": "Streaming-reliability SLO numerical target for INC-2411/2419/2427",
                "status": "missing",
                "note": (
                    "Release notes describe truncation incidents but do not "
                    "set an explicit SLO; would need PM confirmation."
                ),
            },
            {
                "topic": "Whether stale Jan validator bench has been refreshed",
                "status": "missing",
                "note": "validator_service_jan_bench.md is outdated; no fresh bench found.",
            },
            {
                "topic": "Whether P2 shadow-replay results still hold after watermark changes",
                "status": "to_verify",
                "note": "Shadow replay predates INC-2427 watermark changes.",
            },
        ],
    }


def oracle_input_v5() -> dict:
    return {
        "schema_version": "cnb55.manager_brief.v2",
        "variant_id": "v5-recovery-in-thread",
        "accepted": "P5",
        "primary_risk": {
            "statement": (
                "Validator cold-start cost combined with the new RPC boundary "
                "becoming a new control-flow surface that must be observed "
                "under load post-INC-2481, while the watermark redesign "
                "timeline remains unknown."
            ),
            "mitigations": [
                "feature flag staged rollout 1%/10%/50%",
                "shadow replay validator responses against in-proc baseline",
                "kill switch to revert to in-proc validator on SLO breach",
                "canary SLO observability on RPC latency + error rate",
            ],
        },
        "ranking": [
            {
                "proposal_id": "P5",
                "rank": 1,
                "summary": (
                    "Validator microservice delivers reliability isolation and "
                    "does NOT touch the watermark module that caused INC-2481; "
                    "January 2026 jan_bench perf is stale / superseded "
                    "pre-streaming and flagged accordingly."
                ),
                "citations": [
                    "proposals/P5.md",
                    "incident_context/incident_2026_04_P2_rollback.md",
                    "release_context/release_notes_2026_03.md",
                    "repo_evidence/perf/validator_service_jan_bench.md",
                ],
                "constraint_tags": ["incident", "release", "rollout"],
            },
            {
                "proposal_id": "P4",
                "rank": 2,
                "summary": (
                    "Surgical warm-start; does not touch the watermark that "
                    "caused INC-2481; safe latency work as a fast-follow."
                ),
                "citations": [
                    "proposals/P4.md",
                    "repo_evidence/perf/schema_compile_bench.md",
                    "incident_context/incident_2026_04_P2_rollback.md",
                ],
                "constraint_tags": ["incident", "perf"],
            },
            {
                "proposal_id": "P2",
                "rank": 3,
                "summary": (
                    "L2 cache — was the ACCEPTED pick in prior cycle and was "
                    "ROLLED BACK via INC-2481; cannot lead until watermark redesign."
                ),
                "citations": [
                    "proposals/P2.md",
                    "incident_context/incident_2026_04_P2_rollback.md",
                    "incident_context/watermark_bug_notes.md",
                    "incident_context/prior_ranking.md",
                ],
                "constraint_tags": ["incident", "regression"],
            },
            {
                "proposal_id": "P1",
                "rank": 4,
                "summary": (
                    "Structural-validation drop; SDK contract risk unchanged; "
                    "does not advance streaming-reliability post INC-2481."
                ),
                "citations": [
                    "proposals/P1.md",
                    "repo_evidence/meta/rollout_history.md",
                    "release_context/release_notes_2026_03.md",
                ],
                "constraint_tags": ["release", "regression"],
            },
            {
                "proposal_id": "P3",
                "rank": 5,
                "summary": (
                    "Rust rewrite; staffing-blocked (Diego 0%) AND depends on "
                    "watermark redesign from INC-2481 — sunk cost, abandoned skeleton."
                ),
                "citations": [
                    "proposals/P3.md",
                    "repo_evidence/perf/rust_prototype_notes.md",
                    "repo_evidence/meta/staffing.md",
                    "incident_context/watermark_bug_notes.md",
                ],
                "constraint_tags": ["staffing", "incident"],
            },
        ],
        "assumption_ledger": [
            {
                "topic": "Timeline for watermark redesign referenced in INC-2481",
                "status": "missing",
                "note": (
                    "Incident doc notes a redesign is needed but no target "
                    "quarter is attached; P2 re-entry depends on it."
                ),
            },
            {
                "topic": "Validator RPC observability SLO",
                "status": "to_verify",
                "note": "Need PM sign-off on the RPC SLO before ramping P5.",
            },
            {
                "topic": "Whether stale Jan validator bench has been refreshed",
                "status": "missing",
                "note": "validator_service_jan_bench.md is outdated.",
            },
        ],
    }


ORACLES = {
    "v1-clean-baseline": oracle_input_v1,
    "v2-noisy-distractor": oracle_input_v2,
    "v3-dirty-state": oracle_input_v3,
    "v4-multi-corpus-objective": oracle_input_v4,
    "v5-recovery-in-thread": oracle_input_v5,
}


# ---------------------------------------------- oracle runner


def run_cli_in_workspace(ws: Path, brief_input: dict) -> subprocess.CompletedProcess:
    (ws / "brief").mkdir(exist_ok=True)
    inp = ws / "brief_input.json"
    inp.write_text(json.dumps(brief_input, indent=2))
    return subprocess.run(
        [sys.executable, str(ws / "bin" / "cnb55-brief"), "submit", str(inp)],
        cwd=str(ws),
        capture_output=True, text=True,
    )


def generate_oracle_brief(variant: str) -> None:
    ws_src = WS_BUNDLE / variant
    oracle_dir = VER_DATA / variant / "oracle"
    oracle_dir.mkdir(exist_ok=True, parents=True)

    brief_input = ORACLES[variant]()
    # Persist the brief_input.json alongside the canonical outputs.
    (oracle_dir / "brief_input.json").write_text(
        json.dumps(brief_input, indent=2, sort_keys=True) + "\n"
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_ws = Path(tmp) / "ws"
        shutil.copytree(ws_src, tmp_ws)
        cp = run_cli_in_workspace(tmp_ws, brief_input)
        if cp.returncode != 0:
            print(f"[{variant}] CLI submit FAILED rc={cp.returncode}")
            print("STDOUT:", cp.stdout)
            print("STDERR:", cp.stderr)
            raise SystemExit(1)
        # Copy canonical outputs into oracle_dir
        shutil.copy2(tmp_ws / "brief" / "manager_brief.json",
                     oracle_dir / "manager_brief.json")
        shutil.copy2(tmp_ws / "brief" / "manager_brief.md",
                     oracle_dir / "manager_brief.md")


# ---------------------------------------------- scorer runner


def score_in_ws(variant: str, brief_files: dict[str, str] | None) -> dict:
    """Copy variant workspace to tmp, populate brief/ with brief_files,
    run the scorer, return result JSON."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_ws = Path(tmp) / "ws"
        shutil.copytree(WS_BUNDLE / variant, tmp_ws)
        brief_dir = tmp_ws / "brief"
        brief_dir.mkdir(exist_ok=True)
        if brief_files is not None:
            for name, content in brief_files.items():
                (brief_dir / name).write_text(content)
        result_path = Path(tmp) / "result.json"
        env = os.environ.copy()
        env.update({
            "AGENT_WS": str(tmp_ws),
            "VERIFIER_DATA": str(VER_DATA),
            "RESULT_FILE": str(result_path),
            "VARIANT_ID": variant,
            "CNB55_SEED": "42",
        })
        cp = subprocess.run(
            [sys.executable, str(SCORER)],
            capture_output=True, text=True, env=env,
        )
        if not result_path.exists():
            print(f"[{variant}] scorer did not write result")
            print("STDOUT:", cp.stdout)
            print("STDERR:", cp.stderr)
            raise SystemExit(1)
        return json.loads(result_path.read_text())


def oracle_brief_files(variant: str) -> dict[str, str]:
    oracle = VER_DATA / variant / "oracle"
    return {
        "manager_brief.json": (oracle / "manager_brief.json").read_text(),
        "manager_brief.md": (oracle / "manager_brief.md").read_text(),
    }


def shortcut_brief_input(variant: str) -> dict:
    """Accept staffing-blocked P3 (shortcut). Still valid CLI input."""
    base = ORACLES[variant]()
    # Find P3 entry, promote it to rank 1; move current rank 1 to rank 2
    ranking = base["ranking"]
    # set all ranks
    p3 = next(e for e in ranking if e["proposal_id"] == "P3")
    others = [e for e in ranking if e["proposal_id"] != "P3"]
    new = [
        {**p3, "rank": 1,
         "summary": "Rust rewrite chosen for highest perf delta."},
    ]
    for i, e in enumerate(others, start=2):
        new.append({**e, "rank": i})
    base["ranking"] = new
    base["accepted"] = "P3"
    return base


def shortcut_brief_files(variant: str) -> dict[str, str]:
    inp = shortcut_brief_input(variant)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_ws = Path(tmp) / "ws"
        shutil.copytree(WS_BUNDLE / variant, tmp_ws)
        cp = run_cli_in_workspace(tmp_ws, inp)
        if cp.returncode != 0:
            raise SystemExit(
                f"shortcut CLI failed for {variant}: {cp.stderr}"
            )
        return {
            "manager_brief.json": (tmp_ws / "brief" / "manager_brief.json").read_text(),
            "manager_brief.md": (tmp_ws / "brief" / "manager_brief.md").read_text(),
        }


# ----------------------------------------------- main


def main() -> int:
    print("=== regenerating manifests, gold hashes, and oracles ===")

    # 1. refresh manifests + gold hashes per variant
    for v in VARIANTS:
        ws = WS_BUNDLE / v
        h = compute_readonly_hashes(ws)
        files = list_files(ws)
        update_manifest(v, h, files)
        update_gold(v, h)
        print(f"[{v}] manifest+gold updated: {len(files)} files, "
              f"{len(h)} readonly trees")

    # 2. regenerate oracle briefs via CLI
    for v in VARIANTS:
        generate_oracle_brief(v)
        print(f"[{v}] oracle brief regenerated via CLI")

    # 3. score oracle / empty / shortcut for each variant
    print()
    print(
        f"{'variant':<30} {'oracle':>8} {'empty':>7} {'shortcut':>10} "
        f"{'pass_bar':>9} {'status':<20}"
    )
    print("-" * 95)
    ok = True
    results = []
    for v in VARIANTS:
        gold = json.loads((VER_DATA / v / "gold_ranking.json").read_text())
        pass_bar = gold.get("pass_bar", 60)

        oracle_result = score_in_ws(v, oracle_brief_files(v))
        empty_result = score_in_ws(v, {})
        shortcut_result = score_in_ws(v, shortcut_brief_files(v))

        oracle_s = oracle_result["score"]
        empty_s = empty_result["score"]
        shortcut_s = shortcut_result["score"]

        status = []
        if oracle_s < 90:
            status.append("ORACLE_LOW")
            ok = False
        if empty_s != 0:
            status.append("EMPTY_NONZERO")
            ok = False
        if shortcut_s > 30:
            status.append("SHORTCUT_HIGH")
            ok = False
        if not status:
            status = ["OK"]

        print(
            f"{v:<30} {oracle_s:>8} {empty_s:>7} {shortcut_s:>10} "
            f"{pass_bar:>9} {', '.join(status):<20}"
        )
        results.append({
            "variant": v,
            "oracle_score": oracle_s,
            "oracle_raw": oracle_result.get("raw_score_pre_ceiling"),
            "oracle_ceilings": oracle_result.get("ceilings_applied"),
            "empty_score": empty_s,
            "empty_ceilings": empty_result.get("ceilings_applied"),
            "shortcut_score": shortcut_s,
            "shortcut_ceilings": shortcut_result.get("ceilings_applied"),
            "pass_bar": pass_bar,
        })

    print()
    print("VERDICT:", "OK" if ok else "FAIL")

    if not ok:
        print()
        print("Full result dumps for failing variants:")
        for r in results:
            v = r["variant"]
            print()
            print(f"--- {v} ---")
            print(json.dumps(r, indent=2))
            if r["oracle_score"] < 90:
                full = score_in_ws(v, oracle_brief_files(v))
                print("oracle full result:")
                print(json.dumps(full, indent=2))

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
