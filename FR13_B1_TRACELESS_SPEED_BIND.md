# FR13 B=1 traceless chain5 speed bind

Date: 2026-06-11 UTC.

## Verdict

The FR13 forked-FA2 tree launcher no longer silently enables the three high-volume tree diagnostics on clean speed boots:

- `LUMO_MTP_DRAFT_TRACE_FILE`
- `LUMO_TREE_SAMPLER_DEBUG_LOG`
- `LUMO_TREE_PATH_LCP_LOG`

Those diagnostics remain explicitly available: callers can pass paths for S1/S2/engagement/lossless evidence, and the patcher still emits when `FR10_METRICS=1` or an explicit log path is present. The speed path now defaults to the FR10 clean-speed style: empty diagnostic paths, with no trace files created.

No B=4 work was run.

## Code change

Updated:

- `scripts/fr13_launch_forked_fa2_tree_server.sh`
- `tests/test_fr10_phase4_sampled_committer_wiring.py`

The launcher now initializes:

```bash
LUMO_MTP_DRAFT_TRACE_FILE=${LUMO_MTP_DRAFT_TRACE_FILE:-}
LUMO_TREE_SAMPLER_DEBUG_LOG=${LUMO_TREE_SAMPLER_DEBUG_LOG:-}
LUMO_TREE_PATH_LCP_LOG=${LUMO_TREE_PATH_LCP_LOG:-}
```

and forwards those values into the container instead of hard-coding `/logs/fr10_mtp_draft_trace.jsonl`, `/logs/tree_sampler_debug.jsonl`, and `/logs/tree_path_lcp.jsonl`.

## Static checks

```bash
bash -n scripts/fr10_launch_speed_server.sh scripts/fr13_launch_forked_fa2_tree_server.sh
pytest -q \
  tests/test_fr10_phase4_sampled_committer_wiring.py \
  tests/test_fr13_nsys_launcher_wiring.py \
  tests/test_fr13_replay_route_wiring.py::test_launcher_passthrough_defaults_replay_route_on \
  tests/test_fr13_s1_bonus_row.py::test_launcher_forwards_bonus_self_flag \
  tests/test_fr13_nondet_chase_fixes.py::test_launcher_forwards_fr13_chase_flags
```

Result: `20 passed in 0.74s`.

## Live no-log speed run

Run dir: `output/fr13_b1_traceless_speed_bind/chain_replay_on/`.

Launcher:

```bash
CONTAINER=fr13-b1-traceless-chain
PORT=9950
GPU_UTIL=0.82
MAX_NUM_SEQS=1
BATCH_INVARIANT=0
FR10_METRICS=0
FR13_REPLAY_ROUTE=1
LOG_DIR=$PWD/output/fr13_b1_traceless_speed_bind/chain_replay_on/logs
TREE='[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0)]'
scripts/fr13_launch_forked_fa2_tree_server.sh
```

Probe:

```bash
python3 scripts/fr12_deliverable_swe4_probe.py \
  --endpoint http://127.0.0.1:9950 \
  --model qwen3.6-27b \
  --mode tree_mtp \
  --label chain_replay_on_traceless \
  --out output/fr13_b1_traceless_speed_bind/chain_replay_on/chain_probe.json \
  --samples-per-prompt 1 \
  --batch-size 1 \
  --max-tokens 128 \
  --temperature 0.0 \
  --top-p 1.0 \
  --seed 1313 \
  --wait-health 1200 \
  --request-timeout 600
```

Speed basis is only `/metrics`:
`request_decode_time_seconds_sum / spec_decode_num_drafts_total`.

| arm | decode seconds | spec drafts | draft tokens | s/forward | accept/event | warm decode TPS |
|---|---:|---:|---:|---:|---:|---:|
| chain5 replay-on, no diagnostic logs | 39.737744 | 131 | 655 | 0.303342 | 2.931298 | 12.884476 |

Comparison:

| prior chain5 row | prior s/forward | current / prior |
|---|---:|---:|
| `FR13_B1_SPEED_TAX_ATTRIBUTION_BIND.md` replay-on | 0.303595 | 0.999166x |
| `FR13_B1_PROFILE_BIND.md` profiled replay-on | 0.304396 | 0.996538x |

The no-log speed result reproduces the previous chain5 speed band. Removing default trace files does not reveal a hidden near-native result; it removes diagnostic tax from clean speed verdicts.

## No-log artifact proof

Container env:

- `FR10_METRICS=0`
- `FR13_REPLAY_ROUTE=1`
- `VLLM_BATCH_INVARIANT=0`
- `LUMO_MTP_DRAFT_TRACE_FILE=`
- `LUMO_TREE_SAMPLER_DEBUG_LOG=`
- `LUMO_TREE_PATH_LCP_LOG=`

Log dir listing contained only `fr13_forked_fa2.sha256`. Explicit absence proof:

```text
ABSENT fr10_mtp_draft_trace.jsonl
ABSENT tree_sampler_debug.jsonl
ABSENT tree_path_lcp.jsonl
ABSENT tree_path_lcp_max.jsonl
```

CUDA graph proof from `docker_full.log`:

- `PIECEWISE=1 (largest=6), FULL=1 (largest=6)`
- graph capture finished in `6 secs`

## Short no-log Nsight profile

Run dir: `output/fr13_b1_traceless_speed_bind/chain_nsys/`.

Launcher differences from the speed run:

```bash
CONTAINER=fr13-b1-traceless-nsys
LUMO_NSYS_WRAP_VLLM=1
LUMO_NSYS_DELAY_S=360
LUMO_NSYS_DURATION_S=240
LUMO_NSYS_OUTPUT=/logs/chain_nsys
```

Probe used `--max-tokens 64` to compare with `FR13_B1_PROFILE_BIND.md`.

Profile artifacts:

- `output/fr13_b1_traceless_speed_bind/chain_nsys/logs/chain_nsys.nsys-rep`
- `output/fr13_b1_traceless_speed_bind/chain_nsys/logs/chain_nsys.sqlite`
- `output/fr13_b1_traceless_speed_bind/chain_nsys/nsys_reduce.json`

Profile speed:

| arm | decode seconds | spec drafts | draft tokens | s/forward | accept/event | warm decode TPS |
|---|---:|---:|---:|---:|---:|---:|
| chain5 replay-on, no-log nsys | 20.844408 | 68 | 340 | 0.306535 | 2.808824 | 12.281471 |

DtoH comparison against `FR13_B1_PROFILE_BIND.md` chain5 replay-on (`7,752` DtoH ops, `4,017,072` bytes, `36.621952 ms`):

| signal | prior profiled chain5 | no-log profiled chain5 | current / prior |
|---|---:|---:|---:|
| CUDA graph node events | 43,270 | 43,270 | 1.000000x |
| GPU DtoH memcpy count | 7,752 | 7,598 | 0.980134x |
| GPU DtoH memcpy bytes | 4,017,072 | 39,176 | 0.009752x |
| GPU DtoH memcpy time | 36.621952 ms | 7.293856 ms | 0.199166x |
| GPU HtoD memcpy count | 4,302 | 4,242 | 0.986053x |
| GPU HtoD memcpy bytes | 357,898 | 354,522 | 0.990567x |
| GPU HtoD memcpy time | 1.229216 ms | 2.127424 ms | 1.730716x |
| `cudaGraphLaunch` calls | 824 | 824 | 1.000000x |
| `cudaGraphLaunch` runtime API time | 58.725568 ms | 63.633408 ms | 1.083568x |
| `cudaMemcpyAsync` runtime API calls | 18,512 | 18,723 | 1.011398x |
| `cudaMemcpyAsync` runtime API time | 22,809.608800 ms | 10,281.481008 ms | 0.450750x |

The no-log profile separates two effects:

- Graph-node count and launch shape remain the same residual `tree_mtp` path.
- The large DtoH byte/time surface from the prior profile was mostly diagnostic trace payload, not required for a clean speed verdict.

## Next speed target

The next target is still the pure-spine `tree_mtp` graph/row-shape/scheduler path. The traceless profile removes the diagnostic DtoH byte tax from speed verdicts, but it does not change the core chain5 speed band: chain5 remains about `0.303 s/fwd`, not near native. Future speed claims must keep using `/metrics` decode seconds divided by spec drafts, and any lossless/superset gate that needs `tree_path_lcp` evidence must opt into that log explicitly.
