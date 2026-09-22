#!/usr/bin/env python3
"""Export completed E1 campaign evidence without discarding failed attempts.

This is packaging, not scientific qualification. Independent reports determine
which sealed results support claims. Run only after the experiment worker exits.
"""
import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", type=Path, required=True)
    ap.add_argument("--campaign", action="append", required=True,
                    help="V2-relative campaign root; repeat to preserve superseded campaigns")
    ap.add_argument("--sources", type=Path, required=True,
                    help="JSON mapping V2-relative source/report paths to verified SHA-256 values")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    sidecar = Path(str(a.output) + ".manifest.json")
    assert not a.output.exists() and not sidecar.exists(), "Preserve previous exports"
    files = {a.sources}
    campaigns = []
    excluded = {"triton_cache", "__pycache__", ".cache"}
    for relative in a.campaign:
        root = a.paper / relative
        config = root / "campaign_snapshot/e1_cells.json"
        assert config.is_file(), f"Missing frozen cells: {root}"
        attempts = []
        for cell in sorted(root.glob("cell_*_a*")):
            if not cell.is_dir():
                continue
            result = cell / "cell_result.json"
            assert result.is_file(), f"Unclosed attempt: {cell}"
            r = json.loads(result.read_text())
            assert (r.get("terminal_seal") or {}).get("sealed"), f"Unsealed attempt: {cell}"
            attempts.append({"path": str(cell.relative_to(a.paper)),
                             "status": r.get("status"), "result_sha256": digest(result),
                             "terminal_seal": r["terminal_seal"]})
        campaigns.append({"root": relative, "frozen_cells_sha256": digest(config),
                          "planned_cells": len(json.loads(config.read_text())["cells"]),
                          "attempts": attempts})
        files.update(p for p in root.rglob("*")
                     if p.is_file() and not excluded.intersection(p.relative_to(root).parts))
    for relative, expected in json.loads(a.sources.read_text()).items():
        source = a.paper / relative
        assert digest(source) == expected, f"Source changed: {relative}"
        files.add(source)
    ordered = sorted(files)
    rows = [{"path": str(p.relative_to(a.paper)), "size": p.stat().st_size,
             "sha256": digest(p)} for p in ordered]
    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Private controlled-prefix replay evidence on the pinned stack; includes unsuccessful attempts. Packaging does not confer qualification.",
        "source_root": str(a.paper), "campaigns": campaigns, "files": rows,
        "excluded": ["model weights", *sorted(excluded)],
        "path_mapping": "Entries are V2-relative; original absolute provenance paths remain unchanged.",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(a.output, "w:gz", compresslevel=1, dereference=True) as archive:
        for p, row in zip(ordered, rows):
            assert digest(p) == row["sha256"], f"File changed during export: {p}"
            archive.add(p, arcname=row["path"], recursive=False)
        data = (json.dumps(manifest, indent=2) + "\n").encode()
        member = tarfile.TarInfo("E1-EVIDENCE-MANIFEST.json")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    manifest.update(archive_sha256=digest(a.output), archive_bytes=a.output.stat().st_size)
    sidecar.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("archive_sha256", "archive_bytes")}
                     | {"files": len(rows), "campaigns": len(campaigns)}))


if __name__ == "__main__":
    main()
