#!/usr/bin/env python3
"""Export the two completed E7b diagnostic batches privately; no inference/upload.

Entries retain paths relative to v2. Original absolute provenance paths remain
unchanged. The E7a companion is separate. Model weights and JIT caches are omitted.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import tarfile

ROOTS = (
    "experiments/out-20260922T063019Z-e7b-r13-p072-batch",
    "experiments/out-20260922T071945Z-e7b-r16-p072-commit-batch",
)
SOURCES = {
    "experiments/e2/e7b_state_continuation.py": "9aaa30503c0d7c114c1e9492aab7ac253a8b5a840b21cda9426f81b62226301d",
    "experiments/e7a/e7a_core.py": "89feb49d2bf17b670fb8b5daf1032fd55467f51de3bf33a84f08a1ed555cabd7",
    "experiments/e2/e7b_reduce.py": "ebfaa9ef86fdd6a8f8dd4cb551e4c24bfdd177e757ecc9f3b3e36054f2fbd1c1",
}

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()
    assert not a.output.exists(), "Use a new snapshot name; preserve earlier exports"
    files = set()
    for root in ROOTS:
        d = a.paper / root
        assert "E7B LOOP DONE: 3 boots" in (d / "loop.log").read_text(), root
        files.update(f for f in d.rglob("*") if f.is_file()
                     and not {"triton_cache", "__pycache__", ".cache"}.intersection(f.parts))
    for name, expected in SOURCES.items():
        f = a.paper / name
        assert digest(f) == expected, f"Reviewed source changed: {name}"
        files.add(f)
    entries = [{"path": str(f.relative_to(a.paper)), "size": f.stat().st_size,
                "sha256": digest(f)} for f in sorted(files)]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(a.output, "w:gz", compresslevel=1, dereference=True) as archive:
        for f, entry in zip(sorted(files), entries):
            assert digest(f) == entry["sha256"], f"Source changed while exporting: {f}"
            archive.add(f, arcname=entry["path"], recursive=False)
    report = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Private one-prefix verifier/commit diagnostics; KV remap disabled; not full-route qualification",
        "archive_sha256": digest(a.output), "archive_bytes": a.output.stat().st_size,
        "source_root": str(a.paper), "files": entries,
        "excluded": ["model weights", "triton_cache", "__pycache__", ".cache"],
    }
    manifest = a.output.with_name(a.output.name + ".manifest.json")
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("archive_sha256", "archive_bytes")} | {"files": len(entries)}))

if __name__ == "__main__":
    main()
