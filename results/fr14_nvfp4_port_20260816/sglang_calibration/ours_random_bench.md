# FR14 apples-to-apples measurement 1 — OUR promoted stack on THEIR workload

**STATUS: THE MEASUREMENT WAS NOT TAKEN. GPU CONTENDED.**
**Lane: DIAGNOSTIC. This file banks a BLOCKER, two launcher findings, one
CPU-side verification, and a ready-to-run vehicle. It banks NO throughput
number and must not be cited as one.**

2026-08-19 · GB10 · HEAD at close `5efa41820dde43adac39c1adb6c4f5211bf06d1a`

## What this run was for

Run the **promoted production stack** — hydra27_fixed32 + K0 full-vocab drafting
+ split-K tier-B armed **by default** + fused draft top-k default, on the
RadixArk NVFP4 checkpoint — against **sglang's own bench shape**: random
1024/1024, 8 prompts, concurrency 1. The banked comparison is sglang EAGLE
3/1/4 on the same box and the same checkpoint:

| sglang EAGLE bs=1 random-1024 | |
|---|---:|
| output TPS | **26.151** |
| accept length | **2.901** |
| median TPOT | **36.089 ms** |
| mean TPOT | 37.784 ms |
| median E2E | 37.379 s |

(`sglang_calibration/bench_bs1.jsonl`, greedy, `--dataset-name random
--random-input-len 1024 --random-output-len 1024 --random-range-ratio 1
--num-prompts 8 --max-concurrency 1`, default `--seed 42`.)

## Why it was not taken

Pre-boot GPU discipline check at the top of the window found **docker empty and
zero compute processes**. Roughly forty minutes later, immediately before boot,
the re-check found:

```
fr13-bigdenom-hydra31_fixed32_promoab_Ch31q   Up 2 minutes
nvidia-smi compute-apps: pid 352261, 20625 MiB
```

Round 20's promotion A/B arm **Ch31q** had launched at `2026-08-19T18:19:22Z`.
The comparable hydra27 arm `Ch27n` ran **4 h 43 m** wall
(`output/fr14_promoab_Ch27n_20260819T112147Z/arm_meta.txt`). Our stack needs the
GB10's unified memory uncontended — the whole point of the leg-3 vehicle is
"engine only, so nothing competes" — and this measurement's bound was 75 min.

**The foreign serve was left untouched.** No container was stopped, no launcher
was executed, nothing was staged onto the GPU. The two launcher findings below
are therefore **static reads of HEAD, not observed refusals** — labelled as such
throughout.

## Finding 1 (BLOCKING, static) — the promoted split-K default cannot boot a plain launch

`scripts/fr13_launch_forked_fa2_tree_server.sh`, on a plain hydra27_fixed32 B1
launch with no B1 arm named:

1. **:1373-1398** the promoted default arms the tier-B serve —
   `FR13_FA2_QROW32_B1_TIER_B_ARM=gqa_pair_splitk` — and stages the binary,
   the credential path, the closure and both SASS digests. It sets **no**
   `FR13_FA2_QROW32_B1_SOURCE_COMMIT` and **no**
   `FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256`; both default empty at **:961-962**.
2. **:1611** a non-empty tier-B arm increments
   `_FR13_FA2_QROW32_B1_SELECTOR_COUNT` to 1.
3. **:2298-2310** `SELECTOR_COUNT > 0` opens the B1 selector provenance gate,
   which requires

   ```
   "$FR13_FA2_QROW32_B1_SOURCE_COMMIT" == "$(git rev-parse HEAD)"
   ```

   Empty ≠ HEAD ⇒ `exit 2`, *"FR13 qrow32 B1 selector requires Hydra27 B1 and
   exact binary/source provenance"*.

**This is the exact failure class pass 101 was written to remove, relocated.**
The comment at :1305-1315 describes it precisely for the *gqa_pair* default —
"the arm the default armed was guaranteed to refuse, and the launcher exited 2 …
STRUCTURALLY UNBOOTABLE at every HEAD since the credential was minted" — and
fixes it there by degrading to the incumbent. The split-K default deliberately
inverts that to **hard refusal** (:1380-1390 rationale: a promoted default that
silently serves something else is an unlabelled A/B). That inversion is sound.
What is not sound is that the default supplies none of the provenance the
selector gate it opens will demand, so the refusal fires on **every** plain
launch, not only on a stale credential.

Consistent with this, no serve has yet taken the path: the most recent hydra27
arm's launch log carries no split-K arming line at all, because that runner
**explicitly names the B1 production arm empty**, which sets
`_FR13_FA2_QROW32_B1_PRODUCTION_ARM_NAMED=1` and skips the whole default block
(`output/fr14_promoab_Ch27n_20260819T112147Z/hydra27_fixed32_promoab_Ch27n/launch.log`).

**Minimal operator fix, and it is only two variables:** export
`FR13_FA2_QROW32_B1_SOURCE_COMMIT=$(git rev-parse HEAD)` and
`FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256=$(sha256sum
scripts/fr13_patch_fa2_tree_bias.py | cut -d' ' -f1)`. Leave the gqa_pair gate
JSONs empty — see finding 2. The banked vehicle carries this behind
`RERUN_WITH_PROVENANCE=1` so the first attempt still exercises, and records, the
untouched default.

## Finding 2 (LATENT, static) — the two promoted defaults are mutually exclusive

The obvious operator response to finding 1 — "present the gqa_pair credential
so the provenance is real" — trips a *different* refusal:

* **:1400-1407** with `GQA_PAIR_GATE_JSON` + `GQA_PAIR_LIVE_RESULT_JSON` +
  `SOURCE_COMMIT == HEAD` all present, the **gqa_pair production default** also
  arms, inside the same block that already armed split-K.
* **:1606 + :1611** now count **two** selectors.
* **:1620** `SELECTOR_COUNT > 1` ⇒ `exit 2`, *"FR13 qrow32 B1 live A/B and
  production arms are mutually exclusive"*.

So the env that satisfies the selector provenance gate by the credentialed route
is the env that trips the two-selector guard. The narrow path that works is the
one in finding 1: the two provenance variables **alone**, gate JSONs empty.
Worth ruling deliberately rather than discovering at boot.

## Verified CPU-side: the numerics-bound credential DOES still validate at HEAD

The one thing the campaign asked to confirm, confirmed — and it needs no GPU.

```
scripts/fr13_qrow32_b1_pass_sidecar.py verify-tier-b … → rc=0
```

* credential file `results/fr14_nvfp4_port_20260816/fr14_splitk_tierb_credential.json`
  sha256 **255267fc18fa4eb5209e381133d53a48410b3d362c6bc573065a9dd5be87e6d5** —
  the digest the promotion earned at, unchanged.
* body digest `908ea0780d…` covers its own payload.
* `identity.source_commit` = **8dd868805b726e75813a01b3b97edfb9575ec73f**
  (the pass-100 promotion commit). HEAD is now **5efa41820** — at least six
  commits on, and it moved twice *during this session* under concurrent agents.
* **All nine pre-registered bounds re-derived from the recorded measurements and
  passed** (B1 determinism bitwise in-process and cross-process; B2 0.9331 ≥ 0.9;
  B3 1.384 ≤ 4; B4 4 ≤ 8; B5 3.81e-06 ≤ 1e-04; B6 1 flip ≤ incumbent's 2;
  B7 0.9608 ≤ 1.1; B8 0.8386 ≤ 1.1; B9 0 non-finite disagreements).
  All ten probe-strength floors met.
* `bounds_sha256 ee49c3a7…` matches the pre-registered pin, and
  `patch_source_sha256 e80ed4ea…` matches HEAD's
  `scripts/fr13_patch_fa2_tree_bias.py` **exactly**.
* staged binary present, not a symlink, sha256 `28570f83…`, size 300123792 —
  matches the launcher literal and the credential identity.

**The pass-101 re-scope works as designed.** `source_commit` sits in
`TIERB_RECORDED_FIELDS` (recorded, required well-formed, *not* matched) while
the nine numerics-determining fields sit in `TIERB_BINDING_FIELDS`. A HEAD move
that touches none of them cannot change what the kernel computes, and it does
not refuse. Finding 1 is **not** a credential problem — the credential is
healthy. It is the launcher's own selector gate, which is still commit-bound.

## The vehicle, ready to run

`/home/mark/shared/tmp-scratch/nvfp4_port/ours_random_bench/`

* `launch_nomiddleware.sh` — copy of
  `scripts/fr13_launch_forked_fa2_tree_server.sh`; **`diff` is exactly one
  line**, `FR13_FIXED32_MIDDLEWARE_FLAGS=""`. Asserted, not assumed.
* `boot.sh` — the leg-3 vehicle (`ablation_a_leg3_boot.sh`) retargeted from
  tail6/K64 to **hydra27/K0** and to RadixArk: K0 floor contract asserted
  (25430574256 B / 93.15228665201465 ms / walk cap 12), hydra27 mask and active
  drafts taken from `fr13_fixed32_topology` rather than hand-copied, canonical
  exact4 identity exported for the tier-B block, **B1 FA2 arm deliberately
  unnamed** so the promoted default must arm itself.
* `hfcache/` — ShareGPT `V3_unfiltered_cleaned_split.json` pre-fetched (642 MB),
  so the bench does not download mid-window.
* `sglang_serving.py`, `verify_tierb_at_head.json` — the client source read for
  methodology, and the verification above.

### Methodology the run would use, and every divergence from theirs

Replicating `sglang_calibration.sh` as closely as our stack allows:

| | theirs | ours | divergence |
|---|---|---|---|
| client | `sglang.bench_serving` | same, from their image, `--network host`, no GPU | none |
| backend | `--backend sglang` → `/generate` | `--backend vllm` → `/v1/completions` | **route only**; same client, same timing code |
| dataset | `random`, 1024/1024, ratio 1, 8 prompts, conc 1, seed 42 | identical | none — same ShareGPT text tiled to exactly 1024 tokens |
| sampling | **greedy, T=0** | **T=0.6 / top_p 0.95 / top_k 20** | **UNAVOIDABLE.** The fixed32 committer hard-refuses greedy — *"FR13 fixed32 requires sampled temp>0, no draft_probs, and max_spec_len=31"* — and killed the engine on the first request, twice, in leg 3 (`ablation_a_leg3_temp0_refusal.txt`). Favours **them** by an unknown small amount: greedy accepts more. |
| ingress | none | **fixed32 ASGI middleware dropped** | it admits only `/v1/chat/completions` and `/v1/responses` and authenticates a per-task HMAC only the SWE harness mints. `/v1/completions` returns 403 `fixed32_route_not_allowed`, so their raw-completions shape is unreachable behind it. Established campaign practice (`fr14_leg3_launch_nomiddleware.sh`, `fr14_armb_leg3_launch_nomiddleware.sh`). It is an ingress audit device, not a decode-path component. |
| harness | none | **no SWE runner** | engine-only, as leg 3 and the arm-B ablation ran. Consequence: the fixed32 flush transaction never runs, so `fr13_decode_step_wall_*` and `fr13_decode_forward_gpu_*` stay 0. **step_wall must come from the vLLM-native basis** — `d(request_decode_time_seconds_sum)/d(spec_decode_num_drafts_total)` — with accept from `spec_decode_num_accepted_tokens / spec_decode_num_drafts`, scraped `/metrics` pre and post. bench_serving's own `Accept length` line is blank on the vllm backend: it reads sglang's `/get_server_info`, which we do not serve. |
| launcher | — | promoted launcher, **one line changed** | the split-K arming, the credential staging and the in-container `verify-tier-b` are byte-identical to the production path. Only the ASGI guard is absent. |

### Why the existing nomiddleware copies could not be used

Both `scripts/fr14_leg3_launch_nomiddleware.sh` and
`scripts/fr14_armb_leg3_launch_nomiddleware.sh` are **behind HEAD**: neither
carries `FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE`, and their B1 selector still
hard-codes the K64/root1 identity (*"requires Hydra27 K64/root1 B1"*). Under the
promoted K0 config they refuse by construction. A fresh copy off HEAD is
required, which is what the vehicle carries.

## What to expect when it runs, and the honest framing that must travel with it

**This is our WORST-CASE workload and the result must be reported as such.**
Random 1024-token ShareGPT prompts through raw `/v1/completions` are
off-distribution for a stack whose acceptance was earned on agentic code traffic.
The mechanism is specific, not hand-waving: the hydra27 tree spends 31 physical
drafts every step, and on unpredictable text the **suffix tail earns close to
nothing** — the deep spine positions that pay for themselves on predictable code
are rejected, while their verify cost is fixed and paid in full.

The banked history on this checkpoint and this exact shape:

| | engine | output TPS | accept |
|---|---|---:|---:|
| sglang EAGLE 3/1/4, greedy | theirs | **26.10 / 26.15** | **2.90** |
| arm A conservative bytes | ours, tail6 K64 | 19.27 | 2.358 |
| arm B aggressive bytes | ours, tail6 K64 | 23.23 | 2.739 |
| arm B, full-vocab | ours, tail6 **K0** | **25.68** | **3.214** |

Against SWE agent traffic the same stack accepts **~4.1-4.5**. The gap — roughly
**0.9-1.3 accepted tokens per event** — is the price of the workload, not of the
kernel, and it is the single largest term in any comparison against their 26.15.
A promoted-stack number on this shape should be read as *the floor*, and the
hydra27 + split-K deltas should be read against the **25.68 / 3.214** K0 row,
which is the closest banked antecedent (tail6, no split-K, no fused top-k).

## Next step

Re-run when the GB10 is free. The vehicle is staged; the window needed is
~6 min boot + ~6 min bench (8 × 1024 tokens at ~25 TPS ≈ 330 s) + teardown.
Run `boot.sh` **once with the default untouched** to record the refusal in
finding 1 as an observation rather than a static read, then
`RERUN_WITH_PROVENANCE=1 boot.sh` for the measurement.

## Postscript — Ch31q crashed at 4 m 50 s, and the slot is held by its corpse

Written after the bank above. The foreign arm did not run 4 h; it **died at
`2026-08-19T18:24:12Z`, exit 1**, 4 m 50 s after starting:

```
FAIL: fixed32 terminal flush rc=1
fixed32 container incarnation drifted after engine-ledger materialization failure: b13398ebe046…
```

That is round 20's own held blocker reproducing — not anything this measurement
did, and it was diagnosed from `docker inspect` and their runlog without
touching either.

The GPU is now idle, **but the window was still not taken**, for one reason
worth stating plainly: `docker ps -a` is **not empty** — Ch31q's exited
container is still there, and it is round 20's forensic evidence for a failure
they are actively chasing. Removing it to satisfy the launcher's own
`docker ps -aq` emptiness assertion would destroy that evidence, and it is not
this measurement's to remove. Ch31p → Ch31q also shows a fast relaunch cadence,
so the slot should be assumed claimed.

**Whoever picks this up next:** clear Ch31q only once round 20 has taken what it
needs from it, then run the staged vehicle. Nothing else is in the way.

## GPU discipline

Docker empty and zero compute processes at open. One foreign container
(`fr13-bigdenom-hydra31_fixed32_promoab_Ch31q`, round 20) appeared mid-window,
was **left running**, and exited on its own. This measurement started nothing on
the GPU and removed nothing: zero containers created, zero removed. Net
container delta **0**. Zero compute processes at close.
