from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
CHECKER = REPO / "scripts" / "fr13_check_bf16_gemvx_k64_m1_shuffle_r64_codegen.py"
R32_CUDA = REPO / "csrc" / "fr13_bf16_gemvx_k64_m1_shuffle.cu"
R64_CUDA = REPO / "csrc" / "fr13_bf16_gemvx_k64_m1_shuffle_r64.cu"

SPEC = importlib.util.spec_from_file_location("fr13_r64_codegen", CHECKER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _valid_sass() -> str:
    mnemonics = (
        ["NOP"] * 14
        + ["SHFL.DOWN"] * 4
        + ["FADD"] * 4
        + ["FFMA"] * 2
        + ["LDG.E.U16.CONSTANT"] * 2
        + ["STG.E.U16"]
        + ["F2FP.BF16.F32.PACK_AB"]
        + ["MOV"] * 36
    )
    lines = ["\t.target\tsm_121a", "//--------------------- .text.kernel"]
    lines.extend(
        f"        /*{index:04x}*/                   {mnemonic} R0, R0 ;"
        for index, mnemonic in enumerate(mnemonics)
    )
    return "\n".join(lines) + "\n"


def test_checker_accepts_only_exact_r64_source_arithmetic() -> None:
    contract = MODULE.audit_source(
        R64_CUDA.read_text(encoding="ascii"),
        R32_CUDA.read_text(encoding="ascii"),
    )

    assert contract["grid"] == [1024, 1, 1]
    assert contract["block"] == [16, 64, 1]
    assert contract["per_row_arithmetic_matches_r32_source"] is True


def test_checker_accepts_expected_spill_free_resource_tuple() -> None:
    resource_usage = """
arch = sm_121a
 Function mangled_shuffle_r64_kernel_args:
  REG:18 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:928 TEXTURE:0
"""

    assert MODULE.audit_resource_usage(resource_usage) == {
        "registers_per_thread": 18,
        "stack_bytes_per_thread": 0,
        "local_bytes_per_thread": 0,
        "static_shared_bytes_per_cta": 0,
        "constant0_bytes": 928,
    }
    with pytest.raises(MODULE.AuditError, match="stack usage"):
        MODULE.audit_resource_usage(resource_usage.replace("STACK:0", "STACK:8"))


def test_checker_requires_one_sm121a_cubin_and_1024_thread_bound() -> None:
    elf_list = "ELF file 1: candidate.1.sm_121a.cubin\n"
    elf_dump = """
CUDA Virtual SM: sm_121
CUDA Tool Kit Version: 13.0
register count: 18
frame size: 0x0
min stack size: 0x0
Attribute: EIATTR_MAX_THREADS
Format: EIFMT_SVAL
Value: 0x400 0x1 0x1
Attribute: EIATTR_CRS_STACK_SIZE
Format: EIFMT_SVAL
Value: 0x0
"""

    assert MODULE.audit_elf(elf_list, elf_dump)["max_threads"] == [1024, 1, 1]
    with pytest.raises(MODULE.AuditError, match="1024 threads"):
        MODULE.audit_elf(elf_list, elf_dump.replace("0x400", "0x200"))


def test_checker_rejects_barrier_local_call_and_atomic_sass() -> None:
    sass = _valid_sass()
    assert MODULE.audit_sass(sass)["operational_instructions"] == 50

    for forbidden in ("BAR.SYNC", "LDL", "STL", "CALL.ABS.NOINC", "ATOM.E.ADD"):
        invalid = sass.replace("MOV R0, R0 ;", f"{forbidden} R0, R0 ;", 1)
        with pytest.raises(MODULE.AuditError, match="forbidden SASS"):
            MODULE.audit_sass(invalid)


def test_checker_is_host_only_and_emits_no_qualification_claim() -> None:
    source = CHECKER.read_text(encoding="ascii")

    assert "import torch" not in source
    assert "nvidia-smi" not in source
    assert '"gpu_used": False' in source
    assert '"runtime_wired": False' in source
    assert '"performance_claim": False' in source
    assert '"acceptance_valid": False' in source
