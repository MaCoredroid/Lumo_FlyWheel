from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path("scripts/fr13_fa2_qrow32_static_gate.py")
    spec = importlib.util.spec_from_file_location("fr13_fa2_qrow32_static_gate", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, data: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="ascii")
    return path


def _fixture(tmp_path: Path):
    module = _module()
    source = _write(
        tmp_path / "flash_fwd_fr13_qrow32_hdim256_bf16_sm80.cu",
        module.FIXED32_QUERY_TILE32_TRANSLATION_UNIT,
    )
    qrow_object = _write(tmp_path / "qrow32.o", b"qrow32 object")
    stock_so = _write(tmp_path / "stock.so", b"stock shared object")
    candidate_so = _write(tmp_path / "candidate.so", b"candidate shared object")
    elf_list = _write(
        tmp_path / "elf-list.txt",
        "ELF file    1: qrow32.cu.1.sm_121a.cubin\n",
    )
    resource_usage = _write(
        tmp_path / "resource.txt",
        "\n".join(
            (
                "Fatbin elf code:",
                "arch = sm_121a",
                "Resource usage:",
                f" Function {module.TARGET_KERNEL}:",
                "  REG:244 STACK:0 SHARED:1024 LOCAL:0 CONSTANT[0]:1416",
                "",
            )
        ),
    )
    ptxas_log = _write(
        tmp_path / "ptxas.txt",
        "\n".join(
            (
                "ptxas info : Compiling entry function "
                f"'{module.TARGET_KERNEL}' for 'sm_121a'",
                f"ptxas info : Function properties for {module.TARGET_KERNEL}",
                "    0 bytes stack frame, 0 bytes spill stores, "
                "0 bytes spill loads",
                "ptxas info : Used 244 registers, used 1 barriers",
                "",
            )
        ),
    )
    sass = _write(
        tmp_path / "sass.txt",
        "\n".join(
            (
                f"Function : {module.TARGET_KERNEL}",
                "        /*0000*/ MOV R1, c[0x0][0x28] ;",
                "        /*0010*/ CALL.REL.NOINC 0x100 ;",
                "        /*0020*/ EXIT ;",
                "",
            )
        ),
    )
    defined = "FUNC GLOBAL DEFAULT exported_symbol\n"
    undefined = "FUNC GLOBAL DEFAULT imported_symbol\n"
    needed = "libc.so.6\nlibtorch.so\n"
    args = argparse.Namespace(
        qrow_source=source,
        qrow_object=qrow_object,
        stock_so=stock_so,
        candidate_so=candidate_so,
        elf_list=elf_list,
        resource_usage=resource_usage,
        ptxas_log=ptxas_log,
        sass=sass,
        stock_defined=_write(tmp_path / "stock-defined.txt", defined),
        candidate_defined=_write(tmp_path / "candidate-defined.txt", defined),
        stock_undefined=_write(tmp_path / "stock-undefined.txt", undefined),
        candidate_undefined=_write(
            tmp_path / "candidate-undefined.txt", undefined
        ),
        stock_needed=_write(tmp_path / "stock-needed.txt", needed),
        candidate_needed=_write(tmp_path / "candidate-needed.txt", needed),
        output=None,
    )
    return module, args


def test_static_gate_accepts_zero_spill_sm121a_and_exact_abi(tmp_path: Path) -> None:
    module, args = _fixture(tmp_path)
    result = module.verify_static(args)

    assert result["status"] == "PASS"
    assert result["target"] == {
        "arch": "sm_121a",
        "head_dim": 256,
        "block_m": 32,
        "block_n": 64,
        "warps": 2,
        "threads": 64,
        "split_k": False,
    }
    assert result["resources"] == {
        "registers": 244,
        "stack_bytes": 0,
        "static_shared_bytes": 1024,
        "static_local_bytes": 0,
    }
    assert result["ptxas"]["spill_load_bytes"] == 0
    assert result["sass"] == {"ldl": 0, "stl": 0, "call": 1}
    assert result["production_eligible"] is False
    assert result["timing_eligible"] is False
    assert result["default_off"] is True


@pytest.mark.parametrize(
    ("field", "old", "new", "message"),
    (
        ("resource_usage", "STACK:0", "STACK:8", "nonzero stack"),
        ("ptxas_log", "0 bytes spill loads", "8 bytes spill loads", "spills"),
        ("sass", "/*0020*/ EXIT", "/*0020*/ LDL R2, [R1]", "local-memory"),
    ),
)
def test_static_gate_rejects_resource_drift(
    tmp_path: Path,
    field: str,
    old: str,
    new: str,
    message: str,
) -> None:
    module, args = _fixture(tmp_path)
    path = getattr(args, field)
    path.write_text(path.read_text(encoding="ascii").replace(old, new), encoding="ascii")

    with pytest.raises(module.GateError, match=message):
        module.verify_static(args)


def test_static_gate_rejects_defined_or_undefined_abi_drift(tmp_path: Path) -> None:
    module, args = _fixture(tmp_path)
    args.candidate_defined.write_text(
        "FUNC GLOBAL DEFAULT added_symbol\nFUNC GLOBAL DEFAULT exported_symbol\n",
        encoding="ascii",
    )
    with pytest.raises(module.GateError, match="defined dynamic symbols ABI drifted"):
        module.verify_static(args)

    args.candidate_defined.write_text(
        "FUNC GLOBAL DEFAULT exported_symbol\n",
        encoding="ascii",
    )
    args.candidate_undefined.write_text(
        "FUNC GLOBAL DEFAULT added_import\nFUNC GLOBAL DEFAULT imported_symbol\n",
        encoding="ascii",
    )
    with pytest.raises(module.GateError, match="undefined dynamic symbols ABI drifted"):
        module.verify_static(args)


def test_static_gate_rejects_visible_launcher_and_source_drift(tmp_path: Path) -> None:
    module, args = _fixture(tmp_path)
    args.candidate_defined.write_text(
        "FUNC GLOBAL DEFAULT exported_symbol\n"
        "FUNC GLOBAL DEFAULT fr13_run_mha_fwd_fixed32_qrow32\n",
        encoding="ascii",
    )
    args.stock_defined.write_text(
        args.candidate_defined.read_text(encoding="ascii"),
        encoding="ascii",
    )
    with pytest.raises(module.GateError, match="launcher leaked"):
        module.verify_static(args)

    module, args = _fixture(tmp_path / "source-drift")
    args.qrow_source.write_text("drift\n", encoding="ascii")
    with pytest.raises(module.GateError, match="gated generator"):
        module.verify_static(args)
