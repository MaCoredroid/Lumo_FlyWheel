#!/usr/bin/env python3
"""Graft the division-free qrow16 fatbin onto the production host object."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import tempfile
from pathlib import Path


CANDIDATE_SYMBOL = (
    "_ZN5flash36fr13_flash_fwd_fixed32_qrow16_kernelENS_16Flash_fwd_paramsE"
)
HOST_SYMBOL = (
    "_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traits"
    "ILi256ELi16ELi64ELi1ELb0ELb0EN7cutlass10bfloat16_tE19Flash_kernel_traits"
    "ILi256ELi16ELi64ELi1ES3_EELb0ELb0ELb0ELb0ELb1ELb0ELb0ELb0EEEv"
    "NS_16Flash_fwd_paramsE"
)
ELF_HEADER_SHOFF = 0x28
ELF_HEADER_SHENTSIZE = 0x3A
ELF_HEADER_SHNUM = 0x3C
ELF_HEADER_SHSTRNDX = 0x3E
SECTION_HEADER = struct.Struct("<IIQQQQIIQQ")
SYMBOL = struct.Struct("<IBBHQQ")
HEX = frozenset("0123456789abcdef")


class GraftError(RuntimeError):
    pass


def _regular(path: Path, label: str) -> Path:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise GraftError(f"{label} must be an absolute regular non-symlink file")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in HEX for character in value):
        raise GraftError(f"{label} must be a lowercase SHA-256")
    return value


def _cstring(table: bytes, offset: int) -> bytes:
    if not 0 <= offset < len(table):
        raise GraftError("ELF string offset is out of range")
    end = table.find(b"\0", offset)
    if end < 0:
        raise GraftError("ELF string table is unterminated")
    return table[offset:end]


def _rewrite_string_table(
    table: bytes, old: bytes, new: bytes
) -> tuple[bytes, dict[int, int], int]:
    output = bytearray()
    offsets: dict[int, int] = {}
    replacements = 0
    position = 0
    while position < len(table):
        end = table.find(b"\0", position)
        if end < 0:
            raise GraftError("ELF string table is unterminated")
        value = table[position:end]
        offsets[position] = len(output)
        replacements += value.count(old)
        output.extend(value.replace(old, new))
        output.append(0)
        position = end + 1
    return bytes(output), offsets, replacements


def _align(output: bytearray, alignment: int = 8) -> None:
    output.extend(b"\0" * (-len(output) % alignment))


def _rename_cuda_elf_symbol(data: bytes, old: bytes, new: bytes) -> bytes:
    if data[:6] != b"\x7fELF\x02\x01":
        raise GraftError("candidate cubin is not ELF64 little-endian")
    section_offset = struct.unpack_from("<Q", data, ELF_HEADER_SHOFF)[0]
    section_size = struct.unpack_from("<H", data, ELF_HEADER_SHENTSIZE)[0]
    section_count = struct.unpack_from("<H", data, ELF_HEADER_SHNUM)[0]
    shstr_index = struct.unpack_from("<H", data, ELF_HEADER_SHSTRNDX)[0]
    if section_size != SECTION_HEADER.size or not 0 < shstr_index < section_count:
        raise GraftError("candidate cubin has an unsupported section table")
    if section_offset + section_count * section_size > len(data):
        raise GraftError("candidate cubin section table is truncated")

    sections = [
        list(SECTION_HEADER.unpack_from(data, section_offset + index * section_size))
        for index in range(section_count)
    ]
    shstr = sections[shstr_index]
    shstr_bytes = data[shstr[4] : shstr[4] + shstr[5]]
    section_names = [_cstring(shstr_bytes, section[0]) for section in sections]
    try:
        strtab_index = section_names.index(b".strtab")
        symtab_index = section_names.index(b".symtab")
    except ValueError as error:
        raise GraftError("candidate cubin lacks .strtab or .symtab") from error

    strtab = sections[strtab_index]
    strtab_bytes = data[strtab[4] : strtab[4] + strtab[5]]
    new_shstr, shstr_offsets, shstr_replacements = _rewrite_string_table(
        shstr_bytes, old, new
    )
    new_strtab, strtab_offsets, strtab_replacements = _rewrite_string_table(
        strtab_bytes, old, new
    )
    if shstr_replacements == 0 or strtab_replacements == 0:
        raise GraftError("candidate device symbol was not present in both string tables")

    output = bytearray(data[:section_offset])
    output[shstr[4] : shstr[4] + shstr[5]] = b"\0" * shstr[5]
    output[strtab[4] : strtab[4] + strtab[5]] = b"\0" * strtab[5]

    symtab = sections[symtab_index]
    if symtab[9] != SYMBOL.size or symtab[5] % SYMBOL.size:
        raise GraftError("candidate cubin has an unsupported symbol table")
    for offset in range(symtab[4], symtab[4] + symtab[5], SYMBOL.size):
        name_offset = struct.unpack_from("<I", output, offset)[0]
        try:
            renamed_offset = strtab_offsets[name_offset]
        except KeyError as error:
            raise GraftError("symbol name does not start at a string boundary") from error
        struct.pack_into("<I", output, offset, renamed_offset)

    _align(output)
    new_shstr_offset = len(output)
    output.extend(new_shstr)
    _align(output)
    new_strtab_offset = len(output)
    output.extend(new_strtab)
    _align(output)
    new_section_offset = len(output)

    for index, section in enumerate(sections):
        try:
            section[0] = shstr_offsets[section[0]]
        except KeyError as error:
            raise GraftError("section name does not start at a string boundary") from error
        if index == shstr_index:
            section[4] = new_shstr_offset
            section[5] = len(new_shstr)
        elif index == strtab_index:
            section[4] = new_strtab_offset
            section[5] = len(new_strtab)
        output.extend(SECTION_HEADER.pack(*section))
    struct.pack_into("<Q", output, ELF_HEADER_SHOFF, new_section_offset)
    if old in output or new not in output:
        raise GraftError("candidate cubin symbol rewrite was incomplete")
    return bytes(output)


def _run(command: list[str]) -> None:
    subprocess.run(
        command,
        check=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )


def graft(args: argparse.Namespace) -> dict[str, object]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise GraftError("CUDA_VISIBLE_DEVICES must be explicitly empty")
    cubin = _regular(args.candidate_cubin, "candidate cubin")
    ptx = _regular(args.candidate_ptx, "candidate PTX")
    host_object = _regular(args.host_object, "production host object")
    expected_host_object = _require_sha256(
        args.expected_host_object_sha256, "host object SHA-256"
    )
    if _sha256(host_object) != expected_host_object:
        raise GraftError("production host object SHA-256 mismatch")
    for output in (args.output_fatbin, args.output_object):
        if not output.is_absolute() or output.exists() or output.is_symlink():
            raise GraftError("outputs must be new absolute paths")
        output.parent.mkdir(parents=True, exist_ok=True)

    old = CANDIDATE_SYMBOL.encode("ascii")
    new = HOST_SYMBOL.encode("ascii")
    cubin_data = _rename_cuda_elf_symbol(cubin.read_bytes(), old, new)
    ptx_text = ptx.read_text(encoding="ascii")
    if re.search(r"\b(?:div|rem)\.s64\b", ptx_text):
        raise GraftError("candidate PTX still contains signed 64-bit division")
    ptx_replacements = ptx_text.count(CANDIDATE_SYMBOL)
    if ptx_replacements == 0:
        raise GraftError("candidate PTX does not contain the qrow16 device symbol")
    renamed_ptx = ptx_text.replace(CANDIDATE_SYMBOL, HOST_SYMBOL)
    if CANDIDATE_SYMBOL in renamed_ptx:
        raise GraftError("candidate PTX symbol rewrite was incomplete")

    with tempfile.TemporaryDirectory(
        prefix="fr13-qrow16-graft-", dir=args.output_object.parent
    ) as temporary:
        temporary_path = Path(temporary)
        renamed_cubin_path = temporary_path / "qrow16.sm_80.cubin"
        renamed_ptx_path = temporary_path / "qrow16.compute_80.ptx"
        renamed_cubin_path.write_bytes(cubin_data)
        renamed_ptx_path.write_text(renamed_ptx, encoding="ascii")
        _run(
            [
                str(args.fatbinary),
                "--create",
                str(args.output_fatbin),
                "--64",
                "--compress-all",
                "--image3",
                f"kind=elf,sm=80,file={renamed_cubin_path}",
                "--image3",
                f"kind=ptx,sm=80,file={renamed_ptx_path}",
            ]
        )
        _run(
            [
                str(args.objcopy),
                "--update-section",
                f".nv_fatbin={args.output_fatbin}",
                str(host_object),
                str(args.output_object),
            ]
        )

    return {
        "schema": "fr13.fixed32.fa2_qrow16_fatbin_graft.v1",
        "status": "PASS_HOST_ONLY_BUILD",
        "cuda_visible_devices": "",
        "candidate_symbol": CANDIDATE_SYMBOL,
        "host_symbol": HOST_SYMBOL,
        "ptx_symbol_replacements": ptx_replacements,
        "candidate_cubin_sha256": _sha256(cubin),
        "candidate_ptx_sha256": _sha256(ptx),
        "host_object_sha256": _sha256(host_object),
        "ptx_signed_64bit_division_count": 0,
        "output_fatbin_sha256": _sha256(args.output_fatbin),
        "output_object_sha256": _sha256(args.output_object),
        "output_object_size": args.output_object.stat().st_size,
        "gpu_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-cubin", type=Path, required=True)
    parser.add_argument("--candidate-ptx", type=Path, required=True)
    parser.add_argument("--host-object", type=Path, required=True)
    parser.add_argument("--expected-host-object-sha256", required=True)
    parser.add_argument("--output-fatbin", type=Path, required=True)
    parser.add_argument("--output-object", type=Path, required=True)
    parser.add_argument(
        "--fatbinary",
        type=Path,
        default=Path("/usr/local/cuda/bin/fatbinary"),
    )
    parser.add_argument("--objcopy", type=Path, default=Path("/usr/bin/objcopy"))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    result = graft(args)
    payload = json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.manifest is not None:
        if (
            not args.manifest.is_absolute()
            or args.manifest.exists()
            or args.manifest.is_symlink()
        ):
            raise GraftError("manifest must be a new absolute path")
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(payload, encoding="ascii")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
