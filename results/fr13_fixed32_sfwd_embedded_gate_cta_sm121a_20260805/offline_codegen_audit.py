#!/usr/bin/env python3
"""Reproduce the fixed32 SFWD embedded-gate SM121a comparison offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise SystemExit("CUDA_VISIBLE_DEVICES must be explicitly empty")


CANDIDATE_REVISION = "086da781207322601fc4876f9f6d69292a4a71a1"
BASELINE_REVISION = "4d876d6fd1a9a9bfc4ca4f90651bfc2421439e31"
COMPILER_REVISION = "ee72339c39a83282bbd86298ea4796f71020d334"
COMPILER_PATH = (
    "results/fr13_fixed32_sfwd_conv_postprep_gatepack2_sm121a_20260805/"
    "offline_codegen_audit.py"
)
KERNEL_PATH = (
    "src/lumo_flywheel_serving/fr13_sfwd_conv_postprep_fusion_kernel.py"
)
LAUNCHER_PATH = "src/lumo_flywheel_serving/fr13_sfwd_conv_postprep_fusion.py"
GENERATOR_PATH = "scripts/fr13_generate_sfwd_conv_postprep_fusion_kernel.py"
PROFILES = {
    "b1": {"batch": 1, "block_c": 256, "num_warps": 4},
    "b4": {"batch": 4, "block_c": 256, "num_warps": 4},
}


def _git_text(repo: Path, revision: str, path: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{path}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _resolve(repo: Path, revision: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", f"{revision}^{{commit}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _work(profile: str, *, embedded: bool) -> dict[str, int]:
    batch = PROFILES[profile]["batch"]
    channel_ctas = 10240 // 256
    gate_groups = 32 // 8
    standalone_gate_ctas = 0 if embedded else gate_groups
    ctas_per_request_layer = channel_ctas + standalone_gate_ctas
    gate_bytes_per_request_layer = (
        2 * 32 * 48 * 2
        + 2 * 32 * 48 * 4
        + gate_groups * 48 * (4 + 2)
    )
    return {
        "batch": batch,
        "channel_ctas_per_request_layer": channel_ctas,
        "gate_computation_groups_per_request_layer": gate_groups,
        "standalone_gate_ctas_per_request_layer": standalone_gate_ctas,
        "embedded_gate_channel_ctas_per_request_layer": (
            gate_groups if embedded else 0
        ),
        "ctas_per_request_layer": ctas_per_request_layer,
        "ctas_whole_batch_all_48_layers": batch * ctas_per_request_layer * 48,
        "launched_warps_per_request_layer": ctas_per_request_layer * 4,
        "launched_warps_whole_batch_all_48_layers": (
            batch * ctas_per_request_layer * 4 * 48
        ),
        "requested_gate_bytes_per_request_layer": gate_bytes_per_request_layer,
        "requested_gate_bytes_whole_batch_all_48_layers": (
            batch * gate_bytes_per_request_layer * 48
        ),
        "kernel_launches_all_48_layers": 48,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--candidate", default=CANDIDATE_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    candidate = _resolve(repo, args.candidate)
    if candidate != CANDIDATE_REVISION:
        raise SystemExit(
            f"candidate revision drift: {candidate} != {CANDIDATE_REVISION}"
        )
    baseline = _resolve(repo, BASELINE_REVISION)

    compiler_source = _git_text(repo, COMPILER_REVISION, COMPILER_PATH)
    compiler = {
        "__name__": "fr13_sfwd_embedded_gate_offline_compiler",
        "__file__": f"{COMPILER_PATH}@{COMPILER_REVISION}",
    }
    exec(compile(compiler_source, compiler["__file__"], "exec"), compiler)
    compiler["DEPLOYMENT_CONFIGS"] = PROFILES

    sources = {
        "baseline": _git_text(repo, baseline, KERNEL_PATH),
        "candidate": _git_text(repo, candidate, KERNEL_PATH),
    }
    builds: dict[str, dict[str, dict[str, object]]] = {}
    for label, revision in (("baseline", baseline), ("candidate", candidate)):
        builds[label] = {
            profile: compiler["compile_one"](
                source=sources[label],
                revision=revision,
                output=output / label / profile,
                profile=profile,
            )
            for profile in ("b1", "b4")
        }

    work = {
        profile: {
            "baseline": _work(profile, embedded=False),
            "candidate": _work(profile, embedded=True),
        }
        for profile in ("b1", "b4")
    }
    work_deltas = {
        profile: {
            key: work[profile]["candidate"][key]
            - work[profile]["baseline"][key]
            for key in (
                "standalone_gate_ctas_per_request_layer",
                "ctas_per_request_layer",
                "ctas_whole_batch_all_48_layers",
                "launched_warps_per_request_layer",
                "launched_warps_whole_batch_all_48_layers",
                "requested_gate_bytes_per_request_layer",
                "requested_gate_bytes_whole_batch_all_48_layers",
                "kernel_launches_all_48_layers",
            )
        }
        for profile in ("b1", "b4")
    }
    codegen_fields = (
        "registers",
        "stack_bytes",
        "local_bytes",
        "elf_shared_bytes",
        "launch_shared_bytes",
        "encoded_sass_instructions",
        "static_sass_instructions",
        "ldg",
        "stg",
        "ldl",
        "stl",
        "calls",
        "cubin_bytes",
    )
    codegen_deltas = {
        profile: {
            key: int(builds["candidate"][profile][key])
            - int(builds["baseline"][profile][key])
            for key in codegen_fields
        }
        for profile in ("b1", "b4")
    }
    static_gate_pass = all(
        builds["candidate"][profile][key] == 0
        for profile in ("b1", "b4")
        for key in (
            "stack_bytes",
            "local_bytes",
            "elf_shared_bytes",
            "launch_shared_bytes",
            "ldl",
            "stl",
            "calls",
        )
    ) and all(
        builds["candidate"][profile]["registers"]
        <= builds["baseline"][profile]["registers"]
        and codegen_deltas[profile]["encoded_sass_instructions"] <= 0
        and codegen_deltas[profile]["static_sass_instructions"] <= 0
        and work_deltas[profile]["ctas_whole_batch_all_48_layers"] < 0
        and work_deltas[profile]["launched_warps_whole_batch_all_48_layers"] < 0
        and work_deltas[profile][
            "requested_gate_bytes_whole_batch_all_48_layers"
        ]
        == 0
        for profile in ("b1", "b4")
    )
    source_hashes = {
        label: {
            path: _sha256(_git_text(repo, revision, path))
            for path in (KERNEL_PATH, LAUNCHER_PATH, GENERATOR_PATH)
        }
        for label, revision in (
            ("baseline", baseline),
            ("candidate", candidate),
        )
    }
    summary = {
        "schema": "fr13.fixed32.sfwd.embedded_gate_cta.sm121a.offline_codegen.v1",
        "status": "PASS" if static_gate_pass else "FAIL",
        "offline_only": True,
        "gpu_api_used": False,
        "runtime_byte_correctness": False,
        "timing_claim": False,
        "performance_claim": False,
        "floor_acceptance_eligible": False,
        "revisions": {
            "baseline": baseline,
            "candidate": candidate,
            "compiler": COMPILER_REVISION,
        },
        "source_hashes": source_hashes,
        "compile_contract": {
            "target": "sm_121a",
            "physical_rows_per_request": 32,
            "logical_tree_nodes_lte": 32,
            "draft_vocab_k": 65536,
            "draft_vocab_root": 1,
            "block_c": 256,
            "gate_rows_per_group": 8,
            "gate_schedule": "append_to_first_four_channel_ctas",
            "profiles": PROFILES,
        },
        "builds": builds,
        "codegen_deltas": codegen_deltas,
        "work_model": work,
        "work_deltas": work_deltas,
        "traffic_classification": (
            "exact_source_address_requested_bytes_not_measured_dram_or_hbm"
        ),
        "static_gate_pass": static_gate_pass,
        "required_next_gate": (
            "real SWE-Verified B1 and B4 byte equality on exact merged source, "
            "then exact4 and exact16 full-step timing"
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "codegen_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if static_gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
