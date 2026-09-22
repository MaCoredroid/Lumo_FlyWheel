# E8 companion packaging plan — 2026-09-22

The local-only helper `scripts/export_e8.py` is staged while timing runs. **Do not create the final archive until the worker has exited and all closed evidence has been copied locally.** It never contacts DGX, starts Docker, imports a model, rewrites a run or changes any attempted source. A missing campaign completion marker, any missing/unsealed/additional timing cell, failed campaign, changed source or changed raw evidence refuses the complete-campaign export. An insufficient-support cell is preserved and may make the frozen aggregate unavailable; it is not omitted or replaced.

## Byte set and identities

The exporter includes the complete contents, logs and statuses of these v2-relative roots, excluding only named caches (`__pycache__`, `triton_cache`, `.cache`):

| Root | Meaning / required identity |
|---|---|
| `experiments/e8-single-logits` | Original source, manifest `45da91477bf238e99f70c2201bc722059427ff9cced399754171a4371c41efcf` |
| `experiments/out-20260922T2119Z-e8-single-logits` | Preserved pre-container infrastructure failure, including its exact attempted stage and `FAILED.json`; no scientific request occurred |
| `experiments/e8-single-logits-qualification-v2` | Corrected source, manifest `bc2feab326cb67c916ea418b0399499f4986e18d6bba3e4efc7a20a50f0ce030` |
| `experiments/out-20260922T213142Z-e8-single-logits-v2` | Both actual qualification arms, raw API/events/loaded modules/heads, source stage and receipts; final gate `815b348bdb95c455122e7f1232449d4e12dd339edbd6c3c8e2b9a980c6342513` |
| `experiments/e8-single-logits-timing-v1` | Timing source, manifest `8bbb5b486fa9e11ff0bc0ef6d0373de65bbd8217392e973a02de015df39c2ab1`; all clean and qualification variant generators plus exact copied dependencies |
| `experiments/out-20260922T215254Z-e8-timing` | Exact executed stage, complete copied qualification evidence, all six cells, full API/event/config/loaded-module/ownership logs, terminal seals, campaign start/end and unchanged frozen aggregate |

Required pinned reviews are the original harness, infrastructure-v2, timing harness, actual qualification and route audit reports. The weight preflight and original P0 weight-hash manifest, parent qualification checker and both parent result receipts are included. A final **actual timing review** must be named by `--timing-review` and hash-bound in the final `--sources` JSON map; use that map to add its independent raw reducer/JSON, final worker reports and any final interpretation notes. The helper and this plan are also included. Mutable worker state is included only if explicitly selected and hashed after completion.

The copied stages already contain the E1 joiner, direct-ID mapper, recorder, workload, eight frozen prefixes/prompt pool, E8 head/shim code and six exact serving repository dependencies. To make the shipped source-generation and CPU controls readable/reproducible, the exporter additionally preserves their original path inputs: the six files under `p0/monitor/e1-source-20260922T1150Z`, the seven E1 helper/data files and base v7 launcher under `experiments/out-20260922T100028Z-e1-18cells/campaign_snapshot`, and the specific prior E1 tree-B1 fixture's cohort/API/event/join files and all five unmodified `loaded_backend/*.py` modules (the drafter plus the four unchanged modules whose pins the generator discovers). These old fixture bytes are labeled CPU regression/source inputs; their values are not new E8 observations. No prior E1 performance result is substituted for an E8 cell.

Model weight bytes and the Docker image are not redistributed. Their exact identity is preserved. Python3.10+ standard library is sufficient for CPU raw joins, qualification receipt verification and aggregate recomputation; fresh serving additionally needs the pinned image, named model weights and appropriate Linux/GPU runtime. In-memory full-logit comparisons are evidenced by the loaded gate source and closed receipts; no raw full-vocabulary tensors were archived, so the companion does not claim an independent offline replay of those comparisons.

## Produce only after completion

Create a JSON file under v2, for example `p0/monitor/e8-export-sources-final.json`, mapping each additional final report/source path to its SHA-256. Preserve the reviewed sources; do not edit their manifests to accommodate export. From the v2 directory:

```sh
python3 -B scripts/export_e8.py --selfcheck
python3 -B scripts/export_e8.py --paper "$PWD" \
  --sources "$PWD/p0/monitor/e8-export-sources-final.json" \
  --timing-review p0/monitor/ACTUAL-FINAL-TIMING-REVIEW.md --check-only
python3 -B scripts/export_e8.py --paper "$PWD" \
  --sources "$PWD/p0/monitor/e8-export-sources-final.json" \
  --timing-review p0/monitor/ACTUAL-FINAL-TIMING-REVIEW.md \
  --output "$PWD/artifacts/e8-single-logits-YYYYMMDDTHHMMZ.tar.gz"
```

Replace the review placeholder with the real completed review path. No final archive has been produced by staging this helper. The complete-campaign exporter does not accept an aborted or partially copied timing campaign; if timing aborts, preserve its root and make an explicitly incomplete checkpoint through a separately reviewed packaging decision rather than fabricating completion.

Both check-only and export verify all attempted-source manifests, qualification raw receipt hashes, all six terminal/raw seals and final aggregate binding. They rerun eight frozen CPU joins (two qualification, six timing) into temporary files and compare the full JSON and exit status, then rerun the frozen aggregate and compare every field. The only relocation normalization is the absolute `qualification.root` metadata string in the newly computed aggregate; the exporter checks its expected suffix and preserves the archived original bytes. No numeric result, source or input is normalized.

The archive has v2-relative members and an embedded `E8-EVIDENCE-MANIFEST.json` with per-file sizes/hashes, pins, path map, statuses and reproduction checks. The adjacent `.tar.gz.manifest.json` adds archive SHA-256/size, matching earlier companion conventions. Existing output or partial files are refused. Before publishing the final filename, the helper reopens the written archive and checks every member hash. Parent owns integration into the paper bundle and `build_review_bundle.py`.

## Extract, verify and recompute

Verify the archive hash against the sidecar, extract into a new empty directory and treat that directory as v2. All archive paths are relative to this root; absolute provenance paths inside JSON/logs stay unchanged. Then run:

```sh
python3 -B scripts/export_e8.py --paper "$PWD" --verify-extracted
python3 -B experiments/out-20260922T215254Z-e8-timing/stage/timing_verify.py \
  --stage experiments/out-20260922T215254Z-e8-timing/stage \
  --qualification experiments/out-20260922T215254Z-e8-timing/qualification_evidence
python3 -B experiments/out-20260922T215254Z-e8-timing/stage/aggregate.py \
  --stage experiments/out-20260922T215254Z-e8-timing/stage \
  --run experiments/out-20260922T215254Z-e8-timing
```

The first command verifies every embedded member and reproduces the eight raw joins and aggregate. The latter two expose the frozen admission and analysis directly without GPU work. For one raw cell, use its `logs/e1_events.jsonl`, `api_tokens.json`, `join_manifest.json` and the copied `stage/qualification-source/frozen/e1/e1_join.py --expect-reqs 1`; write any `--json` result to a fresh temporary path. Do not overwrite archived `join.json` or `aggregate.json`. The timing freeze's source-generation and test programs should be run from the included original `experiments/e8-single-logits-timing-v1` layout (prefer a disposable extracted copy); their prior E1 regression inputs are supplied at the original relative paths.
