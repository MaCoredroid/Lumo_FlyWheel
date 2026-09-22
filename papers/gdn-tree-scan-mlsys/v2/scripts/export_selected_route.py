#!/usr/bin/env python3
"""Private evidence export for the three completed attention-policy pilots.

Preserves unsuccessful policies and original gate verdicts. A closing event seal
establishes capture completion, not qualification. An explicit reviewed source
manifest binds the later offline analyzers separately from boot snapshots.
"""
import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOTS = {
    "experiments/out-20260922T081202Z-e2-j1prime-p072-kvpolicyA-b1":
        "Incompatible slot policy and invalid async recorder; preserved failure",
    "experiments/out-20260922T082815Z-e2-b1-policyB-sync-p072":
        "Selected synchronous flat-slot B1 instrumentation pilot",
    "experiments/out-20260922T084319Z-e2-b4-policyB-sync-cohort4":
        "Selected synchronous flat-slot B4 pilot; original failed and repaired offline gates retained",
}
EXCLUDED = {"triton_cache", "__pycache__", ".cache"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", type=Path, required=True)
    ap.add_argument("--sources", type=Path, required=True,
                    help="JSON mapping v2-relative reviewed source/report paths to SHA-256")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    sidecar = Path(str(a.output) + ".manifest.json")
    assert not a.output.exists() and not sidecar.exists(), "Preserve earlier exports"
    files = {a.sources}
    captures = []
    for relative, scope in ROOTS.items():
        root = a.paper / relative
        runs = sorted(root.glob("e7b_*"))
        runs = [r for r in runs if r.is_dir()]
        assert len(runs) == 1, (root, runs)
        events = runs[0] / "logs/e1_events.jsonl"
        lines = events.read_text().splitlines()
        assert lines, events
        seal = json.loads(lines[-1])
        # The seal schema is checked against captured data before this exporter
        # is used; preserve its complete contents, including failed counters.
        assert "run_close" in (seal.get("event"), seal.get("kind"), seal.get("type")), seal
        captures.append({"root": relative, "scope": scope, "closing_seal": seal})
        files.update(p for p in root.rglob("*")
                     if p.is_file() and not EXCLUDED.intersection(p.parts))
    for relative, expected in json.loads(a.sources.read_text()).items():
        source = a.paper / relative
        assert digest(source) == expected, f"Reviewed file changed: {relative}"
        files.add(source)
    ordered = sorted(files)
    rows = [{"path": str(p.relative_to(a.paper)), "size": p.stat().st_size,
             "sha256": digest(p)} for p in ordered]
    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Private finite serving-contract evidence; no live conv/KV byte snapshots, full-model sequential-equivalence proof, warmed timing, or stochastic-law guarantee",
        "source_root": str(a.paper), "captures": captures, "files": rows,
        "excluded": ["model weights", *sorted(EXCLUDED)],
        "path_mapping": "Entries are relative to source_root; original absolute provenance paths are unchanged",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(a.output, "w:gz", compresslevel=1, dereference=True) as archive:
        for p, row in zip(ordered, rows):
            assert digest(p) == row["sha256"], f"File changed during export: {p}"
            archive.add(p, arcname=row["path"], recursive=False)
        data = (json.dumps(manifest, indent=2) + "\n").encode()
        member = tarfile.TarInfo("SELECTED-ROUTE-MANIFEST.json")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    manifest.update(archive_sha256=digest(a.output), archive_bytes=a.output.stat().st_size)
    sidecar.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("archive_sha256", "archive_bytes")}
                     | {"files": len(rows)}))


if __name__ == "__main__":
    main()
