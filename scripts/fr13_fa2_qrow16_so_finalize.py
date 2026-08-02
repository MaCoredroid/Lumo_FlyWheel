#!/usr/bin/env python3
"""Finalize a host-linked qrow16 SO without permitting dynamic-ABI drift."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ELF_HEADER_SHOFF = 0x28
ELF_HEADER_SHENTSIZE = 0x3A
ELF_HEADER_SHNUM = 0x3C
ELF_HEADER_SHSTRNDX = 0x3E
SECTION_HEADER = struct.Struct("<IIQQQQIIQQ")
SYMBOL = struct.Struct("<IBBHQQ")
DYNAMIC = struct.Struct("<qQ")
ET_DYN = 3
EM_AARCH64 = 183
SHN_UNDEF = 0
STB_GLOBAL = 1
STT_NOTYPE = 0
STT_FUNC = 2
STV_DEFAULT = 0
DT_NULL = 0
DT_NEEDED = 1
REPAIRED_SYMBOLS = (
    "_ZNK3c104cuda10CUDAStream5queryEv",
    "_ZNK3c104cuda10CUDAStream11synchronizeEv",
)
HEX = frozenset("0123456789abcdef")


class FinalizeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Section:
    name: bytes
    offset: int
    size: int
    link: int
    entry_size: int


@dataclass(frozen=True)
class SymbolRecord:
    name: str
    binding: int
    symbol_type: int
    visibility: int
    other: int
    undefined: bool
    size: int


def _regular(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise FinalizeError(f"{label} must be an absolute path")
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise FinalizeError(f"{label} does not exist") from error
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise FinalizeError(f"{label} must be a regular non-symlink file")
    return path


def _require_sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in HEX for character in value):
        raise FinalizeError(f"{label} must be a lowercase SHA-256")
    return value


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cstring(table: bytes, offset: int) -> bytes:
    if not 0 <= offset < len(table):
        raise FinalizeError("ELF string offset is out of range")
    end = table.find(b"\0", offset)
    if end < 0:
        raise FinalizeError("ELF string table is unterminated")
    return table[offset:end]


class ElfImage:
    def __init__(self, data: bytes, label: str) -> None:
        self.data = data
        self.label = label
        if data[:6] != b"\x7fELF\x02\x01":
            raise FinalizeError(f"{label} is not ELF64 little-endian")
        if struct.unpack_from("<H", data, 0x10)[0] != ET_DYN:
            raise FinalizeError(f"{label} is not a shared object")
        if struct.unpack_from("<H", data, 0x12)[0] != EM_AARCH64:
            raise FinalizeError(f"{label} is not AArch64")
        section_offset = struct.unpack_from("<Q", data, ELF_HEADER_SHOFF)[0]
        section_size = struct.unpack_from("<H", data, ELF_HEADER_SHENTSIZE)[0]
        section_count = struct.unpack_from("<H", data, ELF_HEADER_SHNUM)[0]
        shstr_index = struct.unpack_from("<H", data, ELF_HEADER_SHSTRNDX)[0]
        if (
            section_size != SECTION_HEADER.size
            or not 0 < shstr_index < section_count
        ):
            raise FinalizeError(f"{label} has an unsupported section table")
        if section_offset + section_count * section_size > len(data):
            raise FinalizeError(f"{label} has a truncated section table")
        raw_sections = [
            SECTION_HEADER.unpack_from(data, section_offset + index * section_size)
            for index in range(section_count)
        ]
        shstr = raw_sections[shstr_index]
        shstr_data = self._slice(shstr[4], shstr[5], "section-name table")
        self.sections = tuple(
            Section(
                name=_cstring(shstr_data, raw[0]),
                offset=raw[4],
                size=raw[5],
                link=raw[6],
                entry_size=raw[9],
            )
            for raw in raw_sections
        )

    def _slice(self, offset: int, size: int, label: str) -> bytes:
        if offset < 0 or size < 0 or offset + size > len(self.data):
            raise FinalizeError(f"{self.label} has an invalid {label}")
        return self.data[offset : offset + size]

    def section(self, name: bytes) -> Section:
        matches = [section for section in self.sections if section.name == name]
        if len(matches) != 1:
            raise FinalizeError(
                f"{self.label} must contain exactly one {name.decode('ascii')} section"
            )
        return matches[0]

    def section_data(self, name: bytes) -> bytes:
        section = self.section(name)
        return self._slice(section.offset, section.size, name.decode("ascii"))

    def dynamic_symbols(self) -> tuple[SymbolRecord, ...]:
        symbols = self.section(b".dynsym")
        if symbols.entry_size != SYMBOL.size or symbols.size % SYMBOL.size:
            raise FinalizeError(f"{self.label} has an unsupported dynamic symbol table")
        if not 0 <= symbols.link < len(self.sections):
            raise FinalizeError(f"{self.label} has an invalid dynamic string table link")
        strings = self.sections[symbols.link]
        string_data = self._slice(strings.offset, strings.size, "dynamic string table")
        records = []
        for offset in range(
            symbols.offset,
            symbols.offset + symbols.size,
            SYMBOL.size,
        ):
            name_offset, info, other, shndx, _value, size = SYMBOL.unpack_from(
                self.data, offset
            )
            try:
                name = _cstring(string_data, name_offset).decode("ascii")
            except UnicodeDecodeError as error:
                raise FinalizeError(
                    f"{self.label} has a non-ASCII dynamic symbol"
                ) from error
            records.append(
                SymbolRecord(
                    name=name,
                    binding=info >> 4,
                    symbol_type=info & 0xF,
                    visibility=other & 0x3,
                    other=other,
                    undefined=shndx == SHN_UNDEF,
                    size=size,
                )
            )
        return tuple(records)

    def needed(self) -> tuple[str, ...]:
        dynamic = self.section(b".dynamic")
        if dynamic.entry_size != DYNAMIC.size or dynamic.size % DYNAMIC.size:
            raise FinalizeError(f"{self.label} has an unsupported dynamic section")
        if not 0 <= dynamic.link < len(self.sections):
            raise FinalizeError(f"{self.label} has an invalid dynamic string table link")
        strings = self.sections[dynamic.link]
        string_data = self._slice(strings.offset, strings.size, "dynamic string table")
        needed = []
        for offset in range(
            dynamic.offset,
            dynamic.offset + dynamic.size,
            DYNAMIC.size,
        ):
            tag, value = DYNAMIC.unpack_from(self.data, offset)
            if tag == DT_NULL:
                break
            if tag == DT_NEEDED:
                try:
                    needed.append(_cstring(string_data, value).decode("ascii"))
                except UnicodeDecodeError as error:
                    raise FinalizeError(
                        f"{self.label} has a non-ASCII dependency"
                    ) from error
        return tuple(needed)


def _expected_record(name: str, symbol_type: int) -> SymbolRecord:
    return SymbolRecord(
        name=name,
        binding=STB_GLOBAL,
        symbol_type=symbol_type,
        visibility=STV_DEFAULT,
        other=STV_DEFAULT,
        undefined=True,
        size=0,
    )


def _repair_dynamic_symbol_types(data: bytes) -> bytes:
    image = ElfImage(data, "candidate")
    symbols = image.section(b".dynsym")
    strings = image.sections[symbols.link]
    string_data = image._slice(strings.offset, strings.size, "dynamic string table")
    output = bytearray(data)
    repaired: Counter[str] = Counter()
    for offset in range(
        symbols.offset,
        symbols.offset + symbols.size,
        SYMBOL.size,
    ):
        name_offset, info, other, shndx, value, size = SYMBOL.unpack_from(data, offset)
        try:
            name = _cstring(string_data, name_offset).decode("ascii")
        except UnicodeDecodeError as error:
            raise FinalizeError("candidate has a non-ASCII dynamic symbol") from error
        if name not in REPAIRED_SYMBOLS:
            continue
        if (
            info >> 4 != STB_GLOBAL
            or info & 0xF != STT_NOTYPE
            or other & 0x3 != STV_DEFAULT
            or shndx != SHN_UNDEF
            or value != 0
            or size != 0
        ):
            raise FinalizeError(f"refusing unexpected dynamic symbol record: {name}")
        output[offset + 4] = (STB_GLOBAL << 4) | STT_FUNC
        repaired[name] += 1
    if repaired != Counter({name: 1 for name in REPAIRED_SYMBOLS}):
        raise FinalizeError("candidate does not contain the exact repair symbol set")
    return bytes(output)


def finalize(args: argparse.Namespace) -> dict[str, object]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise FinalizeError("CUDA_VISIBLE_DEVICES must be explicitly empty")
    candidate = _regular(args.candidate, "candidate")
    reference = _regular(args.reference, "reference")
    expected_candidate = _require_sha256(
        args.expected_candidate_sha256, "candidate SHA-256"
    )
    expected_reference = _require_sha256(
        args.expected_reference_sha256, "reference SHA-256"
    )
    if _sha256(candidate) != expected_candidate:
        raise FinalizeError("candidate SHA-256 mismatch")
    if _sha256(reference) != expected_reference:
        raise FinalizeError("reference SHA-256 mismatch")
    if (
        not args.output.is_absolute()
        or args.output.exists()
        or args.output.is_symlink()
    ):
        raise FinalizeError("output must be a new absolute path")

    candidate_data = candidate.read_bytes()
    reference_image = ElfImage(reference.read_bytes(), "reference")
    candidate_image = ElfImage(candidate_data, "candidate")
    reference_symbols = reference_image.dynamic_symbols()
    candidate_symbols = candidate_image.dynamic_symbols()
    if candidate_image.needed() != reference_image.needed():
        raise FinalizeError("candidate DT_NEEDED entries differ from reference")
    for section_name in (b".gnu.hash", b".gnu.version", b".gnu.version_r"):
        if candidate_image.section_data(section_name) != reference_image.section_data(
            section_name
        ):
            raise FinalizeError(
                f"candidate {section_name.decode('ascii')} differs from reference"
            )
    candidate_only = Counter(candidate_symbols) - Counter(reference_symbols)
    reference_only = Counter(reference_symbols) - Counter(candidate_symbols)
    expected_candidate_only = Counter(
        _expected_record(name, STT_NOTYPE) for name in REPAIRED_SYMBOLS
    )
    expected_reference_only = Counter(
        _expected_record(name, STT_FUNC) for name in REPAIRED_SYMBOLS
    )
    if (
        candidate_only != expected_candidate_only
        or reference_only != expected_reference_only
    ):
        raise FinalizeError("candidate has dynamic-ABI drift beyond the repair allowlist")

    output_data = _repair_dynamic_symbol_types(candidate_data)
    finalized = ElfImage(output_data, "finalized candidate")
    if finalized.dynamic_symbols() != reference_symbols:
        raise FinalizeError(
            "finalized dynamic symbol table does not exactly match reference"
        )
    if finalized.needed() != reference_image.needed():
        raise FinalizeError(
            "finalized DT_NEEDED entries do not exactly match reference"
        )
    for section_name in (b".gnu.hash", b".gnu.version", b".gnu.version_r"):
        if finalized.section_data(section_name) != reference_image.section_data(
            section_name
        ):
            raise FinalizeError(
                f"finalized {section_name.decode('ascii')} differs from reference"
            )
    if len(output_data) != len(candidate_data):
        raise FinalizeError("finalization changed shared-object size")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(output_data)
    os.chmod(temporary, stat.S_IMODE(candidate.stat().st_mode))
    os.replace(temporary, args.output)
    return {
        "schema": "fr13.fixed32.fa2_qrow16_so_finalize.v1",
        "status": "PASS_HOST_ONLY_ABI_FINALIZE",
        "cuda_visible_devices": "",
        "gpu_used": False,
        "candidate_input_sha256": expected_candidate,
        "reference_sha256": expected_reference,
        "output_sha256": _sha256_bytes(output_data),
        "output_size": len(output_data),
        "dynamic_symbol_count": len(reference_symbols),
        "needed_count": len(reference_image.needed()),
        "repaired_symbol_names_sha256": _sha256_bytes(
            "\n".join(REPAIRED_SYMBOLS).encode("ascii")
        ),
        "repaired_symbol_count": len(REPAIRED_SYMBOLS),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--expected-reference-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    result = finalize(args)
    payload = json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.manifest is not None:
        if (
            not args.manifest.is_absolute()
            or args.manifest.exists()
            or args.manifest.is_symlink()
        ):
            raise FinalizeError("manifest must be a new absolute path")
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(payload, encoding="ascii")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
