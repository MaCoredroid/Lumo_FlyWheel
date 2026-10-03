#!/usr/bin/env python3
"""Package and clean-build the recorder's local manuscript dependencies.

A working source checkpoint is neither campaign completion nor publication.
Writes only into a new caller-selected directory; preserves all old artifacts.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--paper', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--delivery-source', action='store_true',
                    help='Label a manuscript source delivery; campaign disposition stays in its separate receipt.')
    args = ap.parse_args()
    paper = args.paper.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    stage = output / 'source'
    stage.mkdir()
    members = {'ref.bib', 'main.bbl', 'README.md'}
    external = set()
    for line in (paper / 'main.fls').read_text().splitlines():
        if not line.startswith('INPUT '):
            continue
        item = Path(line[6:])
        item = (item if item.is_absolute() else paper / item).resolve()
        try:
            relative = item.relative_to(paper)
        except ValueError:
            external.add(str(item))
            continue
        if item.suffix in ('.tex', '.tikz', '.bib', '.bbl', '.cls', '.sty', '.bst', '.pdf', '.png', '.jpg', '.jpeg'):
            if item == paper / 'main.pdf':
                raise ValueError('recorder unexpectedly reads its output PDF')
            members.add(str(relative))
    if not {'main.tex', 'abstract.tex', 'IEEEtran.cls'} <= members:
        raise ValueError('recorder is not a complete manuscript build')
    records = []
    for name in sorted(members):
        source = paper / name
        if source.is_symlink() or not source.is_file():
            raise ValueError('source is not a regular file: ' + name)
        destination = stage / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        records.append({'path': name, 'bytes': destination.stat().st_size, 'sha256': sha(destination)})
    heading = ('LumoTree manuscript source delivery. Scientific scope and campaign disposition are recorded in the adjacent delivery receipt.\n'
               if args.delivery_source else
               'LumoTree working manuscript source checkpoint. Not a final campaign delivery.\n')
    instructions = (
        heading +
        'Compile with a TeX installation providing IEEEtran bibliography style and latexmk:\n'
        'latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape main.tex\n'
        'This archive contains the manuscript dependency closure, not the private raw experiment archive.\n'
        'See SOURCE-MANIFEST.json and the adjacent checkpoint receipt for exact identities and build checks.\n'
    )
    (stage / 'BUILD.txt').write_text(instructions)
    manifest = {'schema': 'lumotree-working-source-checkpoint-v1', 'campaign_complete': False,
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'source_recorder_sha256': sha(paper / 'main.fls'),
                'builder_sha256': sha(Path(__file__).resolve()),
                'scope': 'MANUSCRIPT_SOURCE_CLOSURE_ONLY_NOT_RAW_EVIDENCE_BUNDLE',
                'members': records, 'external_tex_inputs': sorted(external)}
    if args.delivery_source:
        manifest['schema'] = 'lumotree-manuscript-source-delivery-v1'
        manifest.pop('campaign_complete')
        manifest['campaign_disposition'] = 'SEPARATE_DELIVERY_RECEIPT_NOT_INFERRED_FROM_COMPILATION'
    (stage / 'SOURCE-MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    archive = output / ('lumotree-source.tar.gz' if args.delivery_source else 'lumotree-working-source.tar.gz')
    with tarfile.open(archive, 'w:gz') as stream:
        for path in sorted(stage.rglob('*')):
            if path.is_file():
                stream.add(path, arcname=str(path.relative_to(stage)), recursive=False)
    extracted = output / 'clean-build'
    extracted.mkdir()
    with tarfile.open(archive) as stream:
        stream.extractall(extracted, filter='data')
    for row in records:
        if sha(extracted / row['path']) != row['sha256']:
            raise ValueError('extracted member hash differs: ' + row['path'])
    command = ['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error', '-file-line-error',
               '-no-shell-escape', 'main.tex']
    with (output / 'clean-build.stdout.txt').open('wb') as log:
        result = subprocess.run(command, cwd=extracted, stdout=log, stderr=subprocess.STDOUT)
    receipt = {'schema': 'lumotree-working-source-checkpoint-verification-v1',
               'campaign_complete': False, 'published': False,
               'archive_sha256': sha(archive), 'source_manifest_sha256': sha(stage / 'SOURCE-MANIFEST.json'),
               'command': command, 'returncode': result.returncode,
               'source_members_verified': len(records), 'build_log_sha256': sha(output / 'clean-build.stdout.txt')}
    if args.delivery_source:
        receipt['schema'] = 'lumotree-manuscript-source-delivery-verification-v1'
        receipt.pop('campaign_complete')
        receipt['campaign_disposition'] = 'SEPARATE_DELIVERY_RECEIPT_NOT_INFERRED_FROM_COMPILATION'
    if result.returncode == 0:
        text = (extracted / 'main.log').read_text(errors='replace')
        forbidden = ('There were undefined references', 'There were undefined citations',
                     'multiply defined', 'Overfull \\hbox')
        receipt['unresolved_or_horizontal_overflow'] = [key for key in forbidden if key in text]
        receipt['pdf_sha256'] = sha(extracted / 'main.pdf')
        receipt['pdf_bytes'] = (extracted / 'main.pdf').stat().st_size
        # Record actual local files read in the clean build; no workspace fallback.
        read_paths = set()
        for line in (extracted / 'main.fls').read_text().splitlines():
            if line.startswith('INPUT '):
                path = Path(line[6:])
                read_paths.add(str((path if path.is_absolute() else extracted / path).resolve()))
        receipt['original_workspace_inputs'] = sorted(x for x in read_paths
            if Path(x).is_relative_to(paper) and not Path(x).is_relative_to(extracted))
        receipt['status'] = 'PASS' if not receipt['unresolved_or_horizontal_overflow'] and not receipt['original_workspace_inputs'] else 'FAIL'
    else:
        receipt['status'] = 'FAIL'
    (output / 'VERIFICATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    if receipt['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
