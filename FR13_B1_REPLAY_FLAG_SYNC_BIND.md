# FR13 B=1 replay flag-sync closeout

Date: 2026-06-11 UTC.
Closeout updated: 2026-06-12 UTC.

## Closeout verdict

This trail is an artifact bank, not a landed code bind. The code patch in the
worktree is rejected and must not be committed as-is.

Two representative-layer variants were tried:

- First sorted replay layer: reduced DtoH sync pressure, but failed live
  default-vs-strict safety and default-repeat determinism.
- Last/topological replay layer: default-repeat determinism recovered, but
  default-vs-strict still failed.

The representative-layer family is therefore dead for lossless work. It removes
host syncs by weakening all-layer freshness/row validation, and the served token
comparisons show that is not safe enough to bank.

The next concrete speed path is all-layer batched validation: preserve every
layer's fresh flag and staged-row check, but aggregate the small flag tensors on
device and perform one DtoH read instead of two `.item()` reads per GDN layer.

No B=4 work was run in this trail.

Current rejected dirty files:

- `scripts/fr10_phase4_patch_vllm_tree_gdn.py`
- `scripts/fr13_launch_forked_fa2_tree_server.sh`
- `tests/test_fr13_replay_route_wiring.py`

The artifact root is:

```text
output/fr13_b1_replay_flag_sync_bind/
```

## Last-layer follow-up

The last/topological-layer fence changed both committer twins to sort replay
layers by numeric `layers.<int>` order, then default-check only
`len(_fr13_replay_layer_state) - 1` while strict mode checked every layer.

Static checks passed in the worker:

```bash
bash -n scripts/fr13_launch_forked_fa2_tree_server.sh scripts/fr10_launch_speed_server.sh
python3 -m py_compile scripts/fr10_phase4_patch_vllm_tree_gdn.py tests/test_fr13_replay_route_wiring.py
pytest -q \
  tests/test_fr13_replay_route_wiring.py \
  tests/test_fr13_nsys_launcher_wiring.py \
  tests/test_fr10_phase4_sampled_committer_wiring.py \
  tests/test_fr13_s1_bonus_row.py::test_launcher_forwards_bonus_self_flag \
  tests/test_fr13_nondet_chase_fixes.py::test_launcher_forwards_fr13_chase_flags
```

Result: `34 passed`.

Live B=1 chain5 speed:

| arm | decode seconds | spec drafts | s/forward | accept/event | warm decode TPS |
|---|---:|---:|---:|---:|---:|
| last-layer default | 42.500922 | 141 | 0.301425 | 2.659574 | 12.046797 |
| last-layer default repeat | 42.474751 | 141 | 0.301239 | 2.659574 | 12.054220 |
| strict all-layer | 40.317079 | 133 | 0.303136 | 2.849624 | 12.699333 |

Safety comparisons:

| pair | exact token sequence match | result |
|---|---:|---|
| default vs default repeat | 4/4 | pass |
| default vs strict | 2/4 | fail |
| default repeat vs strict | 2/4 | fail |

First default-vs-strict mismatch:

```text
prompt_id=1 sample_index=0 first_diff=25 default_token=471 strict_token=14
```

Last-layer artifacts:

- `output/fr13_b1_replay_flag_sync_bind/last_layer_default/chain_probe.json`
- `output/fr13_b1_replay_flag_sync_bind/last_layer_default/chain_probe_repeat.json`
- `output/fr13_b1_replay_flag_sync_bind/last_layer_default/default_repeat_compare.json`
- `output/fr13_b1_replay_flag_sync_bind/last_layer_strict/chain_probe.json`
- `output/fr13_b1_replay_flag_sync_bind/last_layer_strict/default_vs_strict_compare.json`
- `output/fr13_b1_replay_flag_sync_bind/last_layer_strict/default_repeat_vs_strict_compare.json`

## Verdict

Speed/profile target: the default replay-route flag validation no longer does
two unconditional device-to-host flag reads for every registered GDN layer.
Default mode validates the first sorted replay layer as a representative
freshness/row-count guard, still requires `_fr13_replay_flags` and
`_fr13_replay_ssm_state` on every layer, launches replay for every layer, and
clears `_fr13_flags[0]` for every layer after launch.

Strict diagnostic mode remains available:

```text
FR13_REPLAY_STRICT_FLAG_CHECK=1
```

With that env set, every replay layer runs the old freshness and row-count
`.item()` validation before replay launch.

Safety acceptance status: not verified. A full B=1 chain5 default-vs-strict
served-token comparison was attempted, but token IDs were not byte-identical.
A repeat default run was also not byte-identical to the first default run, so
the live output comparison is not a clean attribution to the new flag-sync
change. Strict mode did engage tree mode without crash and was byte-identical
against a second strict run in the same container. Because the requested safety
layer did not pass, this work was not committed or pushed.

No B=4 work was run.

## Code change

Updated:

- `scripts/fr10_phase4_patch_vllm_tree_gdn.py`
- `scripts/fr13_launch_forked_fa2_tree_server.sh`
- `tests/test_fr13_replay_route_wiring.py`

The greedy and sampled replay committer twins now build:

```python
_fr13_replay_layer_items = sorted(_fr13_replay_layers.items())
_fr13_strict_flag_check = (
    __import__('os').environ.get(
        'FR13_REPLAY_STRICT_FLAG_CHECK', '0'
    ) == '1'
)
```

and gate flag reads with:

```python
_fr13_check_flags = (
    _fr13_strict_flag_check or _fr13_replay_layer_i == 0
)
```

Missing `_fr13_replay_flags` or `_fr13_replay_ssm_state` still fails loudly for
every layer without calling `.item()`. Strict mode is forwarded through the
canonical FR13 launcher with default `0`.

## Static checks

```bash
bash -n scripts/fr13_launch_forked_fa2_tree_server.sh scripts/fr10_launch_speed_server.sh
python3 -m py_compile scripts/fr10_phase4_patch_vllm_tree_gdn.py tests/test_fr13_replay_route_wiring.py
pytest -q \
  tests/test_fr13_replay_route_wiring.py \
  tests/test_fr13_nsys_launcher_wiring.py \
  tests/test_fr10_phase4_sampled_committer_wiring.py \
  tests/test_fr13_s1_bonus_row.py::test_launcher_forwards_bonus_self_flag \
  tests/test_fr13_nondet_chase_fixes.py::test_launcher_forwards_fr13_chase_flags
```

Result: `34 passed in 0.59s`.

The new static test asserts:

- no `for _fr13_prefix in sorted(_fr13_replay_layers):` loop remains;
- `FR13_REPLAY_STRICT_FLAG_CHECK` exists in both committer twins;
- the only two flag `.item()` reads per committer sit behind
  `if _fr13_check_flags:`;
- strict mode means all-layer validation because
  `_fr13_check_flags = _fr13_strict_flag_check or _fr13_replay_layer_i == 0`;
- default mode means representative-layer validation because only layer index
  zero passes that guard when strict is off.

## Default speed run

Run dir:

```text
output/fr13_b1_replay_flag_sync_bind/chain_replay_on/
```

Launcher shape:

```bash
CONTAINER=fr13-b1-flag-sync-chain
PORT=9950
GPU_UTIL=0.82
MAX_NUM_SEQS=1
BATCH_INVARIANT=0
FR10_METRICS=0
FR13_REPLAY_ROUTE=1
FR13_REPLAY_STRICT_FLAG_CHECK=0
TREE='[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0)]'
scripts/fr13_launch_forked_fa2_tree_server.sh
```

Probe used `max_tokens=128`, `temperature=0.0`, `top_p=1.0`, `seed=1313`,
`samples_per_prompt=1`, `batch_size=1`.

Speed basis is only `/metrics`:
`request_decode_time_seconds_sum / spec_decode_num_drafts_total`.

| arm | decode seconds | spec drafts | draft tokens | s/forward | accept/event | warm decode TPS |
|---|---:|---:|---:|---:|---:|---:|
| chain5 replay-on, default flag check | 43.308411 | 143 | 715 | 0.302856 | 2.566434 | 11.822184 |

Comparison:

| prior chain5 row | prior s/forward | current / prior |
|---|---:|---:|
| `FR13_B1_TRACELESS_SPEED_BIND.md` no-log | 0.303342 | 0.998398x |
| `FR13_B1_SPEED_TAX_ATTRIBUTION_BIND.md` replay-on | 0.303595 | 0.997566x |
| `FR13_B1_PROFILE_BIND.md` profiled replay-on | 0.304396 | 0.994941x |

Default repeat artifact:
`output/fr13_b1_replay_flag_sync_bind/chain_default_repeat/chain_probe.json`
measured `39.326723 / 130 = 0.302513 s/fwd`.

## Nsight profile

Run dir:

```text
output/fr13_b1_replay_flag_sync_bind/chain_nsys/
```

Launcher differences:

```bash
CONTAINER=fr13-b1-flag-sync-nsys
LUMO_NSYS_WRAP_VLLM=1
LUMO_NSYS_DELAY_S=360
LUMO_NSYS_DURATION_S=240
LUMO_NSYS_OUTPUT=/logs/chain_nsys
FR13_REPLAY_STRICT_FLAG_CHECK=0
```

Probe used `max_tokens=64`, matching the previous no-log profile window.

Artifacts:

- `output/fr13_b1_replay_flag_sync_bind/chain_nsys/logs/chain_nsys.nsys-rep`
- `output/fr13_b1_replay_flag_sync_bind/chain_nsys/logs/chain_nsys.sqlite`
- `output/fr13_b1_replay_flag_sync_bind/chain_nsys/nsys_reduce.json`

Profile speed:

| arm | decode seconds | spec drafts | draft tokens | s/forward | accept/event | warm decode TPS |
|---|---:|---:|---:|---:|---:|---:|
| chain5 replay-on, default flag check, nsys | 20.630580 | 68 | 340 | 0.303391 | 2.779412 | 12.408764 |

Profile comparison against `FR13_B1_TRACELESS_SPEED_BIND.md` no-log Nsight
profile (`7,598` DtoH ops, `39,176` bytes, `7.293856 ms`;
`cudaStreamSynchronize` `8,538` calls):

| signal | prior no-log profile | flag-sync profile | current / prior |
|---|---:|---:|---:|
| CUDA graph node events | 43,270 | 43,270 | 1.000000x |
| GPU DtoH memcpy count | 7,598 | 814 | 0.107133x |
| GPU DtoH memcpy bytes | 39,176 | 12,040 | 0.307331x |
| GPU DtoH memcpy time | 7.293856 ms | 4.055552 ms | 0.556023x |
| GPU HtoD memcpy count | 4,242 | 4,191 | 0.987977x |
| GPU HtoD memcpy bytes | 354,522 | 354,190 | 0.999064x |
| GPU HtoD memcpy time | 2.127424 ms | 2.062848 ms | 0.969645x |
| `cudaStreamSynchronize` calls | 8,538 | 1,703 | 0.199461x |
| `cudaMemcpyAsync` runtime API calls | 18,723 | 9,885 | 0.527426x |
| `cudaGraphLaunch` calls | 824 | 824 | 1.000000x |

This binds the intended DtoH sync-class reduction while leaving the broader
tree-mode graph shape unchanged.

## Safety check

Strict-mode full run:

```text
output/fr13_b1_replay_flag_sync_bind/chain_strict_equiv/chain_probe.json
```

Strict-mode repeat in the same container:

```text
output/fr13_b1_replay_flag_sync_bind/chain_strict_equiv/chain_probe_repeat.json
```

Comparison artifact:

```text
output/fr13_b1_replay_flag_sync_bind/default_strict_safety_compare.json
```

| pair | byte-identical served token IDs | result |
|---|---:|---|
| strict1 vs strict2 | yes | strict mode stable in-container |
| default1 vs strict1 | no | 3/4 prompts diverged |
| default2 vs strict1 | no | 3/4 prompts diverged |
| default1 vs default2 | no | 4/4 prompts diverged |

Strict speed:

| arm | decode seconds | spec drafts | draft tokens | s/forward | accept/event | warm decode TPS |
|---|---:|---:|---:|---:|---:|---:|
| strict1 | 40.466136 | 133 | 665 | 0.304257 | 2.909774 | 12.652555 |
| strict2 | 40.387729 | 133 | 665 | 0.303667 | 2.909774 | 12.677118 |

Safety interpretation:

- Static safety is present: strict mode restores all-layer `.item()` validation,
  default mode keeps a representative-layer freshness/row-count guard, and all
  layers still require buffers and clear freshness after replay.
- Live strict smoke is present: strict mode engaged tree mode twice without
  crash and produced byte-identical served tokens across the two strict runs.
- Live default-vs-strict equivalence is not verified. Default mode also failed
  byte identity against a default repeat, so this is not a clean proof that the
  flag-sync change alone changes outputs, but it is enough to block a verified
  safety claim.

## Artifact notes

All live arms used B=1 only. Diagnostic trace files remained absent in default,
profile, strict, and default-repeat log directories:

```text
ABSENT fr10_mtp_draft_trace.jsonl
ABSENT tree_sampler_debug.jsonl
ABSENT tree_path_lcp.jsonl
ABSENT tree_path_lcp_max.jsonl
```

CUDA graph proof from logs includes:

```text
PIECEWISE=1 (largest=6), FULL=1 (largest=6)
Graph capturing finished
```

## Decision

Do not commit or push this change as verified yet. The speed/profile target is
bound, but the added safety acceptance layer is not.
