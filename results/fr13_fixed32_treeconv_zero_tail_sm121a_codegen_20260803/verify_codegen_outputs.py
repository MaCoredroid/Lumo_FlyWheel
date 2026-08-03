#!/usr/bin/env python3
"""Verify two isolated fixed32 tree-conv SM121a codegen builds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


RESOURCE_RE = re.compile(
    r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"verification failed: {message}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_bytes(argv: list[str]) -> bytes:
    return subprocess.run(
        argv,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout


def verify_build(path: Path, expected: dict[str, object]) -> dict[str, bytes]:
    observed = json.loads((path / "summary.json").read_text())
    require(observed == expected, f"{path} embedded summary")
    blobs = {
        name: (path / f"kernel.{name}").read_bytes()
        for name in ("cubin", "ptx", "sass")
    }
    require(len(blobs["cubin"]) == expected["cubin_bytes"], f"{path} cubin size")
    for name in ("cubin", "ptx", "sass"):
        require(
            sha256(blobs[name]) == expected[f"{name}_sha256"],
            f"{path} {name} hash",
        )
    resource = (path / "resource.txt").read_bytes()
    require(
        run_bytes(["/usr/local/cuda/bin/nvdisasm", "-c", str(path / "kernel.cubin")])
        == blobs["sass"],
        f"{path} fresh disassembly",
    )
    require(
        run_bytes(
            [
                "/usr/local/cuda/bin/cuobjdump",
                "--dump-resource-usage",
                str(path / "kernel.cubin"),
            ]
        )
        == resource,
        f"{path} fresh resource report",
    )
    match = RESOURCE_RE.search(resource.decode())
    require(match is not None, f"{path} resource parse")
    registers, stack_bytes, shared_bytes, local_bytes = map(int, match.groups())
    require(registers == expected["registers"], f"{path} registers")
    require(stack_bytes == expected["stack_bytes"], f"{path} stack")
    require(shared_bytes == expected["elf_shared_bytes"], f"{path} shared")
    require(local_bytes == expected["local_bytes"], f"{path} local")
    require(re.search(rb"(?m)^\.target sm_121a$", blobs["ptx"]) is not None, f"{path} target")
    require(registers <= 255, f"{path} register limit")
    require(
        stack_bytes
        == local_bytes
        == expected["ldl"]
        == expected["stl"]
        == expected["calls"]
        == 0,
        f"{path} spill/local/call gate",
    )
    return {**blobs, "resource": resource}


def verify_tree(root: Path) -> dict[str, object]:
    summary = json.loads((root / "summary.json").read_text())
    require(
        summary["schema"] == "fr13.fixed32.treeconv.zero_tail.sm121a.codegen.v1",
        "schema",
    )
    require(summary["offline_only"] is True, "offline only")
    require(summary["timing_claim"] is False, "no timing claim")
    require(
        summary["deployment_context"]
        == {
            "fixed_physical_rows": 32,
            "drafter_vocab_k": 65536,
            "root_reduction": 1,
        },
        "fixed32 K64 root1 deployment context",
    )
    contract = summary["compile_contract"]
    require(contract["target"] == "sm_121a", "target contract")
    require(contract["physical_rows_per_request"] == 32, "physical rows")
    require(contract["channels"] == 10240, "channels")
    require(contract["conv_width"] == 4, "width")
    require(contract["conv_state_len"] == 34, "state length")
    require(contract["source_rows_per_request"] == 36, "source rows")
    route = summary["fixed32_route"]
    require(route["generic_batched_writeback_guarded_out"] is True, "fixed route guard")
    require(route["full_node_writebacks_per_event"] == 0, "full-node bypass")
    require(route["candidate_default_off"] is True, "default-off selector")

    verified = {}
    comparisons = {}
    for kind in ("direct", "metadata"):
        verified[kind] = {}
        comparisons[kind] = {}
        variants = summary["direct_kernels"][kind]
        for batch_key, batch in (("b1", 1), ("b4", 4)):
            verified[kind][batch_key] = {}
            for label in ("incumbent", "retained_off", "candidate"):
                expected = variants[label][batch_key]
                verified[kind][batch_key][label] = verify_build(
                    root / kind / label / batch_key,
                    expected,
                )
                require(
                    expected["ctas_per_event"] == 480 * batch,
                    f"{kind}/{batch_key} CTA scaling",
                )
                require(
                    expected["destination_columns_stored_per_row"] == 34,
                    f"{kind}/{batch_key} full destination stores",
                )
            incumbent = variants["incumbent"][batch_key]
            retained = variants["retained_off"][batch_key]
            candidate = variants["candidate"][batch_key]
            for name in ("sass", "resource"):
                require(
                    verified[kind][batch_key]["incumbent"][name]
                    == verified[kind][batch_key]["retained_off"][name],
                    f"{kind}/{batch_key} selector-off machine-code identity {name}",
                )
            require(candidate["source_columns_loaded_per_row"] == 3, "candidate live cols")
            require(incumbent["source_columns_loaded_per_row"] == 34, "incumbent cols")
            require(candidate["source_read_bytes_per_event"] * 34 == incumbent["source_read_bytes_per_event"] * 3, "read-byte ratio")
            require(candidate["destination_write_bytes_per_event"] == incumbent["destination_write_bytes_per_event"], "store bytes")
            require(candidate["ctas_per_event"] == incumbent["ctas_per_event"], "CTA identity")
            require(candidate["ldg"] < incumbent["ldg"], "static LDG reduction")
            require(candidate["stg"] == incumbent["stg"], "static STG identity")
            comparisons[kind][batch_key] = {
                "selector_off_machine_code_identity": True,
                "selector_off_cubin_container_identity": (
                    verified[kind][batch_key]["incumbent"]["cubin"]
                    == verified[kind][batch_key]["retained_off"]["cubin"]
                ),
                "selector_off_ptx_container_identity": (
                    verified[kind][batch_key]["incumbent"]["ptx"]
                    == verified[kind][batch_key]["retained_off"]["ptx"]
                ),
                "source_read_bytes_saved": incumbent["source_read_bytes_per_event"] - candidate["source_read_bytes_per_event"],
                "modeled_global_bytes_saved": incumbent["modeled_global_bytes_per_event"] - candidate["modeled_global_bytes_per_event"],
                "roofline_ms_saved_at_273GBps": incumbent["roofline_ms_at_273GBps"] - candidate["roofline_ms_at_273GBps"],
                "ldg_delta": candidate["ldg"] - incumbent["ldg"],
                "stg_delta": candidate["stg"] - incumbent["stg"],
                "register_delta": candidate["registers"] - incumbent["registers"],
                "cubin_bytes_delta": candidate["cubin_bytes"] - incumbent["cubin_bytes"],
            }

    generic = summary["retained_generic"]
    verify_build(root / "retained_generic", generic)
    require(generic["b1_ctas_per_event"] == 15360, "generic B1 CTAs")
    require(generic["b4_ctas_per_event"] == 61440, "generic B4 CTAs")
    require(generic["b1_modeled_global_bytes_per_event"] == 2139095040, "generic B1 bytes")
    require(generic["b4_modeled_global_bytes_per_event"] == 8556380160, "generic B4 bytes")
    return {
        "schema": "fr13.fixed32.treeconv.zero_tail.sm121a.verification.v1",
        "verified": True,
        "offline_only": True,
        "timing_claim": False,
        "revisions": summary["revisions"],
        "deployment_context": summary["deployment_context"],
        "compile_contract": contract,
        "fixed32_route": route,
        "comparisons": comparisons,
        "absolute_codegen": summary["direct_kernels"],
        "retained_generic": generic,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--rebuild", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    primary = verify_tree(args.primary)
    rebuild = verify_tree(args.rebuild)
    require(primary == rebuild, "primary/rebuild report identity")
    rendered = json.dumps(primary, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
