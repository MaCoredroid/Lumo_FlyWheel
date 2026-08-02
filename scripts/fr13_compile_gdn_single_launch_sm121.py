#!/usr/bin/env python3
"""Offline SM121a compile/resource audit for fixed32 GDN single launch.

The coordinator runs two isolated workers. Each worker imports the kernel
source directly, compiles exact B1/B4 live specializations plus the current
two-launch comparators, inspects temporary cubins, and returns reduced JSON.
Raw compiler IR, PTX, cubins, and SASS stay in temporary directories.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
KERNEL_PATH = REPO / "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
PATCHER_PATH = REPO / "scripts/fr10_phase4_patch_vllm_tree_gdn.py"
EXPECTED_KERNEL_SHA256 = (
    "ca5ff6496c7cf3221996e6aa5971d36207e305e51f5c4a308f71d15165ab659a"
)
TARGET_ARCH = 121
TARGET_WARP_SIZE = 32
NUM_WARPS = 8
NUM_STAGES_SOURCE = "triton_default_unset"

POINTER_SIGNATURES = {
    "q": "*bf16",
    "k": "*bf16",
    "v": "*bf16",
    "g": "*fp32",
    "beta": "*fp32",
    "raw_a": "*bf16",
    "raw_b": "*bf16",
    "A_log": "*fp32",
    "dt_bias": "*fp32",
    "h0": "*fp32",
    "h0_indices": "*i64",
    "h0_num_accepted_tokens": "*i32",
    "invocation_counter": "*i32",
    "root_nodes": "*i32",
    "branch_nodes": "*i32",
    "branch_lengths": "*i32",
    "group_path_indices": "*i32",
    "group_path_counts": "*i32",
    "path_nodes": "*i32",
    "path_parent": "*i32",
    "path_parent_slots": "*i32",
    "path_lengths": "*i32",
    "state_export": "*fp32",
    "export_mask": "*i32",
    "out": "*bf16",
    "ring_k": "*bf16",
    "ring_v": "*bf16",
    "ring_a": "*bf16",
    "ring_b": "*bf16",
    "flags_ptr": "*i32",
}

COMMON_CONSTANTS = {
    "N_ACTUAL": 32,
    "NUM_KH": 16,
    "NUM_VH": 48,
    "DIM_K": 128,
    "DIM_V": 128,
    "BLOCK_V": 8,
    "OUTPUT_SCALE": 128**-0.5,
    "USE_QK_L2NORM_IN_KERNEL": True,
    "H0_INDEX_ROW": 0,
    # Fixed32 uses 31 speculative rows plus the root column.
    "H0_INDEX_BATCH_STRIDE": 32,
    "H0_BATCH_INDEX": 0,
    "H0_ACCEPTED_BATCH_STRIDE": 1,
    "H0_BANK_STRIDE": 48 * 128 * 128,
    "H0_USE_ACCEPTED_COLUMN": False,
    "RAW_GATING": True,
    "SCAN_ALIGN": False,
    "RING_EXPORT": True,
}


@dataclass(frozen=True)
class Variant:
    name: str
    function_name: str
    route: str
    batch: int
    level: int | None
    grid: tuple[int, int, int]
    constants: dict[str, object]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def _load_kernel():
    spec = importlib.util.spec_from_file_location(
        "_fr13_gdn_single_launch_compile_target", KERNEL_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {KERNEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _signature(jit_function) -> dict[str, str]:
    names = [
        parameter.name
        for parameter in jit_function.params
        if not parameter.is_constexpr
    ]
    unknown = sorted(set(names) - POINTER_SIGNATURES.keys())
    if unknown:
        raise RuntimeError(f"pointer signature missing for {unknown!r}")
    return {name: POINTER_SIGNATURES[name] for name in names}


def _variants(kernel) -> list[Variant]:
    levels = kernel._subtree_decompose(kernel._FR13_FIXED32_PARENT)
    contract = kernel._fr13_fixed32_gdn_single_launch_contract(levels)
    path_counts = tuple(len(level) for level in levels)
    max_lengths = tuple(max(len(path) for path, _parent in level) for level in levels)
    if path_counts != (1, 11) or max_lengths != (5, 7):
        raise RuntimeError(
            "fixed32 path descriptor drift: "
            f"counts={path_counts!r} max_lengths={max_lengths!r}"
        )
    if (
        int(contract["groups"]) != 5
        or int(contract["max_group_paths"]) != 3
        or int(contract["critical_node_steps"]) != 32
    ):
        raise RuntimeError(f"single-launch contract drift: {contract!r}")

    variants: list[Variant] = []
    for batch in (1, 4):
        single_constants = {
            **COMMON_CONSTANTS,
            "H0_IS_BANK": True,
            "COUNT_INVOCATION": True,
            "ROOT_STEPS": int(contract["groups"]),
            "MAX_PATH_LEN": max_lengths[1],
            "MAX_GROUP_PATHS": int(contract["max_group_paths"]),
            "NUM_GROUPS": int(contract["groups"]),
            "FLAGS_EXPORT": True,
            "FLAGS_ROWS": batch,
        }
        variants.append(
            Variant(
                name=f"b{batch}_single_launch",
                function_name="_tree_gdn_kernel_fixed32_single_launch",
                route="candidate_single_launch",
                batch=batch,
                level=None,
                grid=(48, 16, batch),
                constants=single_constants,
            )
        )

        for level, (num_paths, max_path_len) in enumerate(
            zip(path_counts, max_lengths, strict=True)
        ):
            path_constants = {
                **COMMON_CONSTANTS,
                "COUNT_INVOCATION": level == 0,
                "MAX_PATH_LEN": max_path_len,
                "STATE_SOURCE": 1 if level == 0 else 2,
                "EXPORT_MODE": 1 if level == 0 else 2,
                "FLAGS_EXPORT": level == 0,
                "FLAGS_ROWS": batch,
            }
            if batch == 1:
                path_constants.pop("H0_INDEX_BATCH_STRIDE")
                path_constants.pop("H0_ACCEPTED_BATCH_STRIDE")
                path_constants["H0_IS_BANK"] = True
                function_name = "_tree_gdn_path_kernel"
            else:
                path_constants.update(
                    {
                        "NUM_PATHS": num_paths,
                        "BATCH_SIZE": batch,
                        "EXPORT_SLOTS": 5,
                    }
                )
                function_name = "_tree_gdn_path_kernel_fixed32_batch"
            variants.append(
                Variant(
                    name=f"b{batch}_current_level{level}",
                    function_name=function_name,
                    route="current_two_launch",
                    batch=batch,
                    level=level,
                    grid=(48, 16, batch * num_paths),
                    constants=path_constants,
                )
            )
    return variants


def _run_tool(*arguments: str) -> str:
    completed = subprocess.run(
        list(arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout + completed.stderr


def _tool_version(path: str) -> list[str]:
    return [line for line in _run_tool(path, "--version").splitlines() if line]


def _sass_opcodes(sass: str) -> list[str]:
    opcodes: list[str] = []
    address = re.compile(r"^\s*/\*[^*]*\*/\s*")
    for raw_line in sass.splitlines():
        if address.match(raw_line) is None:
            continue
        line = address.sub("", raw_line, count=1).strip()
        if not line or line.startswith((".", "//")) or line.endswith(":"):
            continue
        fields = line.split()
        if fields and fields[0].startswith("@"):
            fields = fields[1:]
        if fields:
            opcodes.append(fields[0].split(".", 1)[0])
    return opcodes


def _resource_fields(resources: str, kernel_name: str) -> dict[str, int]:
    function_pattern = re.compile(
        rf"Function\s+{re.escape(kernel_name)}:\s*\n\s*([^\n]+)"
    )
    match = function_pattern.search(resources)
    if match is None:
        raise RuntimeError(f"resource entry missing for {kernel_name}")
    line = match.group(1)
    fields = {}
    for name in ("REG", "STACK", "SHARED", "LOCAL"):
        value = re.search(rf"\b{name}:(\d+)", line)
        if value is None:
            raise RuntimeError(f"resource {name} missing from {line!r}")
        fields[name.lower()] = int(value.group(1))
    constant = re.search(r"\bCONSTANT\[0\]:(\d+)", line)
    fields["constant0"] = int(constant.group(1)) if constant else 0
    return fields


def _elf_section_size(elf: str, section_name: str) -> int:
    for line in elf.splitlines():
        fields = line.split()
        if fields and fields[-1] == section_name:
            return int(fields[2], 16)
    raise RuntimeError(f"ELF section missing: {section_name}")


def _compile_variant(kernel, variant: Variant, scratch: Path) -> dict[str, Any]:
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    jit_function = getattr(kernel, variant.function_name)
    source = ASTSource(
        jit_function,
        signature=_signature(jit_function),
        constexprs=variant.constants,
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("cuda", TARGET_ARCH, TARGET_WARP_SIZE),
        options={"num_warps": NUM_WARPS},
    )
    cubin_bytes = compiled.asm["cubin"]
    if not isinstance(cubin_bytes, bytes):
        raise RuntimeError("Triton returned a non-binary cubin payload")
    cubin = scratch / f"{variant.name}.cubin"
    cubin.write_bytes(cubin_bytes)

    cuobjdump = shutil.which("cuobjdump")
    nvdisasm = shutil.which("nvdisasm")
    if cuobjdump is None or nvdisasm is None:
        raise RuntimeError("cuobjdump and nvdisasm are required")
    resources = _run_tool(cuobjdump, "--dump-resource-usage", str(cubin))
    cuobjdump_sass = _run_tool(cuobjdump, "--dump-sass", str(cubin))
    elf = _run_tool(cuobjdump, "--dump-elf", str(cubin))
    nvdisasm_sass = _run_tool(nvdisasm, "--print-code", str(cubin))

    resource = _resource_fields(resources, compiled.name)
    nv_opcodes = _sass_opcodes(nvdisasm_sass)
    cuobjdump_opcodes = _sass_opcodes(cuobjdump_sass)
    if len(nv_opcodes) != len(cuobjdump_opcodes):
        raise RuntimeError(
            f"independent SASS instruction counts differ for {variant.name}: "
            f"{len(nv_opcodes)} != {len(cuobjdump_opcodes)}"
        )
    header = next(
        (line.strip() for line in elf.splitlines() if "64-bit ELF:" in line),
        None,
    )
    if header is None or "sm=121a" not in header:
        raise RuntimeError(f"unexpected cubin target for {variant.name}: {header!r}")

    metadata = compiled.metadata
    result = {
        "name": variant.name,
        "route": variant.route,
        "batch": variant.batch,
        "level": variant.level,
        "kernel": compiled.name,
        "grid": list(variant.grid),
        "ctas_per_launch": variant.grid[0] * variant.grid[1] * variant.grid[2],
        "ctas_per_request": (
            variant.grid[0] * variant.grid[1] * variant.grid[2] // variant.batch
        ),
        "signature": _signature(jit_function),
        "constexprs": variant.constants,
        "compile_hash": metadata.hash,
        "cubin_sha256": _sha256(cubin_bytes),
        "cubin_bytes": len(cubin_bytes),
        "cubin_header": header,
        "registers_per_thread": resource["reg"],
        "registers_per_cta": resource["reg"] * NUM_WARPS * TARGET_WARP_SIZE,
        "stack_bytes": resource["stack"],
        "local_bytes": resource["local"],
        "shared_bytes_cuobjdump": resource["shared"],
        "constant0_bytes": resource["constant0"],
        "shared_bytes_triton_metadata": metadata.shared,
        "global_scratch_bytes": metadata.global_scratch_size,
        "tmem_bytes": metadata.tmem_size,
        "num_warps": metadata.num_warps,
        "num_stages": metadata.num_stages,
        "sass_instructions_nvdisasm": len(nv_opcodes),
        "sass_instructions_cuobjdump": len(cuobjdump_opcodes),
        "sass_primary_text_bytes": _elf_section_size(
            elf, f".text.{compiled.name}"
        ),
        "sass_capmerc_text_bytes": _elf_section_size(
            elf, f".nv.capmerc.text.{compiled.name}"
        ),
        "ldl_instructions": nv_opcodes.count("LDL"),
        "stl_instructions": nv_opcodes.count("STL"),
        "call_instructions": sum(
            nv_opcodes.count(opcode) for opcode in ("CALL", "CAL", "JCAL")
        ),
        "indirect_branch_instructions": sum(
            nv_opcodes.count(opcode) for opcode in ("BRX", "JMX")
        ),
        "resource_report_sha256": _sha256(resources.encode("utf-8")),
        "nvdisasm_sass_sha256": _sha256(nvdisasm_sass.encode("utf-8")),
        "cuobjdump_sass_sha256": _sha256(cuobjdump_sass.encode("utf-8")),
    }
    if result["sass_primary_text_bytes"] != len(nv_opcodes) * 16:
        raise RuntimeError(
            f"SASS byte/instruction mismatch for {variant.name}: "
            f"{result['sass_primary_text_bytes']} != {len(nv_opcodes)} * 16"
        )
    return result


def _worker(output: Path) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be explicitly empty")
    source_sha256 = _file_sha256(KERNEL_PATH)
    if source_sha256 != EXPECTED_KERNEL_SHA256:
        raise RuntimeError(
            f"kernel source drift: {source_sha256} != {EXPECTED_KERNEL_SHA256}"
        )

    import torch
    import triton
    from triton.backends.nvidia.compiler import get_ptxas

    kernel = _load_kernel()
    scratch = output.parent
    variants = [
        _compile_variant(kernel, variant, scratch) for variant in _variants(kernel)
    ]
    ptxas = Path(get_ptxas(TARGET_ARCH).path).resolve()
    triton_root = Path(triton.__file__).resolve().parent
    try:
        ptxas_identity = str(ptxas.relative_to(triton_root))
    except ValueError:
        ptxas_identity = ptxas.name
    cuobjdump = shutil.which("cuobjdump")
    nvdisasm = shutil.which("nvdisasm")
    assert cuobjdump is not None and nvdisasm is not None
    report = {
        "triton_version": triton.__version__,
        "torch_version": torch.__version__,
        "target": {
            "backend": "cuda",
            "triton_arch": TARGET_ARCH,
            "cubin_arch": "sm_121a",
            "warp_size": TARGET_WARP_SIZE,
            "num_warps": NUM_WARPS,
            "num_stages_source": NUM_STAGES_SOURCE,
        },
        "producer": {
            "role": "triton_selected_blackwell_ptxas",
            "identity": ptxas_identity,
            "sha256": _file_sha256(ptxas),
            "version": _tool_version(str(ptxas)),
        },
        "inspectors": {
            "cuobjdump": {
                "sha256": _file_sha256(Path(cuobjdump)),
                "version": _tool_version(cuobjdump),
            },
            "nvdisasm": {
                "sha256": _file_sha256(Path(nvdisasm)),
                "version": _tool_version(nvdisasm),
            },
        },
        "system_nvcc": {
            "invoked_by_harness": False,
            "compiler_role": "none",
        },
        "variants": variants,
    }
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return 0


def _stable_variant_fields(variant: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "resource_report_sha256",
        "nvdisasm_sass_sha256",
        "cuobjdump_sass_sha256",
    }
    return {key: value for key, value in variant.items() if key not in excluded}


def _coordinate(output: Path) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be explicitly empty")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    pass_reports = []
    for pass_index in (1, 2):
        with tempfile.TemporaryDirectory(
            prefix=f"fr13-gdn-single-sm121-pass{pass_index}-", dir="/tmp"
        ) as temporary:
            root = Path(temporary)
            worker_output = root / "worker.json"
            cache = root / "triton-cache"
            dump = root / "triton-dump"
            inductor = root / "torchinductor-cache"
            tmp = root / "tmp"
            for directory in (cache, dump, inductor, tmp):
                directory.mkdir()
            worker_env = dict(os.environ)
            worker_env.update(
                {
                    "CUDA_VISIBLE_DEVICES": "",
                    "TRITON_CACHE_DIR": str(cache),
                    "TRITON_DUMP_DIR": str(dump),
                    "TORCHINDUCTOR_CACHE_DIR": str(inductor),
                    "TMPDIR": str(tmp),
                }
            )
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker-output",
                    str(worker_output),
                ],
                cwd=REPO,
                env=worker_env,
                check=True,
            )
            pass_reports.append(json.loads(worker_output.read_text(encoding="ascii")))

    first, second = pass_reports
    first_variants = {variant["name"]: variant for variant in first["variants"]}
    second_variants = {variant["name"]: variant for variant in second["variants"]}
    if first_variants.keys() != second_variants.keys():
        raise RuntimeError("fresh-cache variant inventory differs")
    deterministic = {}
    for name in sorted(first_variants):
        stable_equal = _stable_variant_fields(first_variants[name]) == (
            _stable_variant_fields(second_variants[name])
        )
        deterministic[name] = {
            "stable_fields_equal": stable_equal,
            "cubin_sha256_equal": (
                first_variants[name]["cubin_sha256"]
                == second_variants[name]["cubin_sha256"]
            ),
            "compile_hash_equal": (
                first_variants[name]["compile_hash"]
                == second_variants[name]["compile_hash"]
            ),
        }
        if not all(deterministic[name].values()):
            raise RuntimeError(f"fresh-cache compile mismatch for {name}")

    candidate_variants = [
        variant
        for variant in first["variants"]
        if variant["route"] == "candidate_single_launch"
    ]
    unsafe = [
        variant["name"]
        for variant in candidate_variants
        if any(
            int(variant[field]) != 0
            for field in (
                "stack_bytes",
                "local_bytes",
                "ldl_instructions",
                "stl_instructions",
                "call_instructions",
                "indirect_branch_instructions",
                "global_scratch_bytes",
                "tmem_bytes",
            )
        )
    ]
    report = {
        "schema": "fr13.fixed32.gdn_single_launch.sm121_resource_audit.v1",
        "status": (
            "CODEGEN_VIABILITY_PASS_ZERO_SPILL"
            if not unsafe
            else "CODEGEN_REJECT_UNSAFE_RESOURCE_USE"
        ),
        "source": {
            "commit": _run_tool("git", "rev-parse", "HEAD").strip(),
            "kernel_path": str(KERNEL_PATH.relative_to(REPO)),
            "kernel_sha256": _file_sha256(KERNEL_PATH),
            "patcher_path": str(PATCHER_PATH.relative_to(REPO)),
            "patcher_sha256": _file_sha256(PATCHER_PATH),
            "harness_path": str(Path(__file__).resolve().relative_to(REPO)),
            "harness_sha256": _file_sha256(Path(__file__).resolve()),
        },
        "offline": {
            "compile_passes": 2,
            "fresh_process_per_pass": True,
            "fresh_temporary_cache_per_pass": True,
            "raw_compiler_artifacts_retained": False,
            "gpu_kernel_executed": False,
            "system_nvcc_invoked_by_harness": False,
        },
        "toolchain": {
            "triton_version": first["triton_version"],
            "torch_version": first["torch_version"],
            "target": first["target"],
            "producer": first["producer"],
            "inspectors": first["inspectors"],
            "system_nvcc": first["system_nvcc"],
        },
        "fresh_cache_determinism": deterministic,
        "unsafe_candidate_variants": unsafe,
        "variants": first["variants"],
    }
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps({"status": report["status"], "output": str(output)}))
    return 0 if not unsafe else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.output is None) == (args.worker_output is None):
        parser.error("provide exactly one of --output or --worker-output")
    if args.worker_output is not None:
        return _worker(args.worker_output.resolve())
    assert args.output is not None
    return _coordinate(args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
