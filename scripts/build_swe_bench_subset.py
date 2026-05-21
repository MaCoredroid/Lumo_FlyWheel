#!/usr/bin/env python3
"""Build deterministic SWE-Bench subsets for the bounded-time spec.

Spec ref: docs/reports/auto_research/swe-bench-bounded-time-spec-20260520.md §5.

Outputs (per --tier and --dataset):
  - <out-dir>/swe-bench-tier{N}-{verified|pro}-instances-20260520.md
    Pre-registration document with the deterministic instance list,
    per-repo strata, the seed, and the generated_at timestamp.
  - <out-dir>/swe-bench-tier{N}-{verified|pro}-instances.json
    Machine-readable subset for the orchestrator.

Stratification rule (proportional sampling):
  - For each repo, allocate floor(target_n * repo_share) instances.
  - Distribute the residual to the largest-share repos by largest-remainder.
  - Within each repo, sort by instance_id and take a deterministic
    seeded.random.sample to fill the per-repo quota.

The spec §5 example table was based on a stale repo-share assumption
(it under-weighted django and over-weighted pylint/flask/requests).
We follow the actual published distribution at dataset commit pin.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

DATASETS = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "pro": "ScaleAI/SWE-bench_Pro",
}

TIER_SIZE = {
    0: 20,
    1: 100,
    2: None,
}


def _load_records(dataset_key: str) -> list[dict]:
    os.environ.setdefault(
        "HF_HOME",
        str(Path("/home/mark/shared/lumoFlyWheel/.cache/huggingface")),
    )
    from datasets import load_dataset

    name = DATASETS[dataset_key]
    ds = load_dataset(name, split="test")
    rows = []
    for ex in ds:
        rows.append({
            "instance_id": ex["instance_id"],
            "repo": ex.get("repo", "unknown"),
        })
    return rows


def _stratify(records: list[dict], target_n: int, seed: int) -> list[dict]:
    if target_n >= len(records):
        return sorted(records, key=lambda r: r["instance_id"])

    by_repo: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_repo[r["repo"]].append(r)
    for repo in by_repo:
        by_repo[repo].sort(key=lambda r: r["instance_id"])

    total = len(records)
    shares = {repo: len(rs) / total for repo, rs in by_repo.items()}
    raw_alloc = {repo: target_n * share for repo, share in shares.items()}
    floor_alloc = {repo: int(v) for repo, v in raw_alloc.items()}
    remainder = target_n - sum(floor_alloc.values())

    # Distribute remainder by largest fractional remainder, then repo size,
    # then repo name (alphabetical) so the result is deterministic.
    ranked = sorted(
        by_repo.keys(),
        key=lambda repo: (
            -(raw_alloc[repo] - floor_alloc[repo]),
            -len(by_repo[repo]),
            repo,
        ),
    )
    for repo in ranked:
        if remainder <= 0:
            break
        # Don't allocate more than the repo has.
        if floor_alloc[repo] < len(by_repo[repo]):
            floor_alloc[repo] += 1
            remainder -= 1

    # Ensure every non-empty repo has at least 1 if target_n permits.
    if target_n >= len(by_repo):
        for repo in by_repo:
            if floor_alloc[repo] == 0:
                # Steal from the largest over-allocated repo.
                donor = max(
                    floor_alloc.keys(),
                    key=lambda r: floor_alloc[r] - target_n * shares[r],
                )
                if donor != repo and floor_alloc[donor] > 1:
                    floor_alloc[donor] -= 1
                    floor_alloc[repo] += 1

    rng = random.Random(seed)
    chosen: list[dict] = []
    for repo in sorted(by_repo.keys()):
        n = floor_alloc.get(repo, 0)
        if n <= 0:
            continue
        chosen.extend(rng.sample(by_repo[repo], n))
    chosen.sort(key=lambda r: r["instance_id"])
    return chosen


def _write_markdown(
    out_path: Path,
    *,
    dataset_key: str,
    tier: int,
    seed: int,
    target_n: int,
    chosen: list[dict],
    full_total: int,
    full_repo_counts: dict[str, int],
    generated_at: str,
) -> None:
    chosen_repo_counts = Counter(r["repo"] for r in chosen)
    lines: list[str] = []
    lines.append(
        f"# SWE-Bench Tier {tier} Pre-Registered Instance List — {dataset_key.capitalize()}"
    )
    lines.append("")
    lines.append(f"**Generated:** {generated_at}  ")
    lines.append(f"**Dataset:** `{DATASETS[dataset_key]}` (split=test)  ")
    lines.append(f"**Seed:** `{seed}`  ")
    lines.append(f"**Target N:** {target_n}  ")
    lines.append(f"**Full benchmark size:** {full_total}  ")
    lines.append(
        "**Stratification:** proportional by repo, largest-remainder rounding, "
        "deterministic per-repo seeded random.sample."
    )
    lines.append("")
    lines.append(
        "> Pre-registered before evaluation results are read. Per spec §3, "
        "this file commits the campaign to this exact instance set before any "
        "score is computed. Do not modify after Tier 0 launches."
    )
    lines.append("")
    lines.append("## Per-repo allocation")
    lines.append("")
    lines.append("| Repo | Full count | Share | Subset count |")
    lines.append("|---|---:|---:|---:|")
    for repo in sorted(full_repo_counts):
        full = full_repo_counts[repo]
        sub = chosen_repo_counts.get(repo, 0)
        share = full / full_total
        lines.append(f"| `{repo}` | {full} | {share:.1%} | {sub} |")
    lines.append(f"| **Total** | **{full_total}** | **100.0%** | **{sum(chosen_repo_counts.values())}** |")
    lines.append("")
    lines.append("## Instance IDs")
    lines.append("")
    by_repo: dict[str, list[str]] = defaultdict(list)
    for r in chosen:
        by_repo[r["repo"]].append(r["instance_id"])
    for repo in sorted(by_repo):
        lines.append(f"### `{repo}` ({len(by_repo[repo])})")
        lines.append("")
        for iid in sorted(by_repo[repo]):
            lines.append(f"- `{iid}`")
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n")


def _write_json(
    out_path: Path,
    *,
    dataset_key: str,
    tier: int,
    seed: int,
    target_n: int,
    chosen: list[dict],
    generated_at: str,
) -> None:
    payload = {
        "dataset_name": DATASETS[dataset_key],
        "split": "test",
        "tier": tier,
        "seed": seed,
        "target_n": target_n,
        "generated_at": generated_at,
        "instance_ids": [r["instance_id"] for r in chosen],
        "by_repo": dict(Counter(r["repo"] for r in chosen)),
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/reports/auto_research"),
    )
    parser.add_argument(
        "--target-n",
        type=int,
        default=None,
        help="Override the per-tier default subset size.",
    )
    parser.add_argument("--date-stamp", default=None,
                        help="Override generated_at date stamp (default: today).")
    args = parser.parse_args(argv)

    target_n = args.target_n or TIER_SIZE[args.tier]
    records = _load_records(args.dataset)
    if target_n is None:
        chosen = sorted(records, key=lambda r: r["instance_id"])
        target_n = len(chosen)
    else:
        chosen = _stratify(records, target_n, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    date_stamp = args.date_stamp or _dt.date.today().strftime("%Y%m%d")
    generated_at = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    md_path = args.out_dir / f"swe-bench-tier{args.tier}-{args.dataset}-instances-{date_stamp}.md"
    json_path = args.out_dir / f"swe-bench-tier{args.tier}-{args.dataset}-instances-{date_stamp}.json"

    full_repo_counts = dict(Counter(r["repo"] for r in records))
    _write_markdown(
        md_path,
        dataset_key=args.dataset,
        tier=args.tier,
        seed=args.seed,
        target_n=target_n,
        chosen=chosen,
        full_total=len(records),
        full_repo_counts=full_repo_counts,
        generated_at=generated_at,
    )
    _write_json(
        json_path,
        dataset_key=args.dataset,
        tier=args.tier,
        seed=args.seed,
        target_n=target_n,
        chosen=chosen,
        generated_at=generated_at,
    )

    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    print(f"chose {len(chosen)} of {len(records)} across {len(set(r['repo'] for r in chosen))} repos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
