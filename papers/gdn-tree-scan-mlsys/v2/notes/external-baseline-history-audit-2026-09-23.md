# External-baseline Git-history audit — 2026-09-23

Read-only independent audit prompted by the user's correction that SGLang and DFlash had already been tested. **That correction is supported.** It would be false to describe DFlash-family work as entirely untested. Parent separately owns the SGLang SWE task census and historical native-MTP comparisons. No experiment was rerun.

## Search coverage and applicability

Searched all local refs: 8,367 reachable commits and 443 refs at inspection. Used commit-message searches, reachable object-path searches, and full-history content pickaxe for `DFlash2|DSpark|DSPARK|DFLASH2` over Markdown/JSON/Python/shell files (excluding unrelated benchmark blueprint/verifier workspace copies). Also searched DFlash/d-flash/diffusion and other engine/baseline names. The content history completed; its latest actual DFlash execution is the September 21 upstream investigation below. `diffusion` in June FR13 numerical-error notes is not a diffusion-drafter benchmark.

This inventory distinguishes executed synthetic serving, captured-input replay, capture-only preparation, and an actual serving smoke. Older dates do not invalidate a test. Different workload, target precision, runtime, and measured quantity prevent substituting these results for a matched long-running SWE task comparison.

## Primary evidence inventory

All paths below are relative to the repository root `/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel`, unless a Git ref is specified.

| Executed work | Exact historical commit and evidence | Primary observation | What it establishes / remaining boundary |
|---|---|---|---|
| DSpark serving calibration, August 19 | `6530b1f17d84714e8aaf98b1ea5431a403f1e3ea`; `results/fr14_nvfp4_port_20260816/sglang_calibration/dspark_bench_bs{1,8}.jsonl`; `REDTEAM_20260816.md:2096–2121` | Random 1024-input/1024-output benchmark: client max concurrency 1 gives 19.9274115546 output tokens/s, acceptance length 2.2402173913; max concurrency 8 gives aggregate 117.9320657348 output tokens/s, acceptance length 2.8339417315. | **Actual served baseline.** Synthetic random-token requests, not SWE tasks. Max concurrency 8 does not prove every physical step was B8. Raw reported observed concurrency is 7.2975. |
| DSpark drafter-in-tree fusion replay, August 19 (then called FR14 E1; unrelated to September E1) | `c0d5550f431fc23e23db8e1165fc54669220c7a1`; `results/fr14_nvfp4_port_20260816/REDTEAM_20260816.md:2219–2250` | Contemporaneous report: 30,971 simulated steps; tree survival 2.3291 vs MTP-tree 3.2685; predefined kill threshold <2.43. Twelve captured documents from four tasks. | **Reported executed captured-input experiment with negative result**, not complete free-running task comparison. Task-level SE 0.19 is explicitly distinguished from token/step-level pseudo-replication. Raw replay program, preregistration and full output were not found in reachable Git payloads in this bounded audit; retrieve those before promoting quantitative replay claims. |
| DFlash2 architecture/preflight and target-tap capture, August 19 | `f5ae4df63dc53a6467e46312bc57857def4f663e` (`REDTEAM:2928–2959`); capture commit `b2ff0aaefdbd71742a3c4f8bb4e70c1fe294be09`, `promotion_ab_campaign.md:3697–3719` | Recorded capture 487.7 s; 12 document metadata records identify post-layer 5/19/33/47/61 plus final norm; captured argmax and engine greedy token both 248046; 11 GB `taps_e2`, original 11 GB retained. | **Executed target capture and preflight**, not DFlash2 tree-quality or throughput result. The 3.27 quality bar, 3.678 wall break-even and 13.17 ms/pass are preregistered thresholds/cost estimates, not measured DFlash2 task performance. |
| DFlash2 actual FP8 serving smoke, September 21 | `92fea914a49a41bc0b07fb8de0e7dac01081d32e`, corrected by `632817854588fd36fe0e00ed71d178cad323e52a`; `origin/vllm-upstream/plan-v3:results/upstream/54076/{RESULT.md,CARD.md,CORRECTIONS.md,PROVENANCE.txt,runs/arm1_dflash2/argv.json}` | Actual DFlash2 arm booted and one chat-completions smoke returned HTTP 200 and `OK.`. Its cache/Mamba block geometry was 832/832. Explicit block-size 816 variant gave 1632/1632. | **DFlash2 really served on the later stack.** It supersedes an unqualified statement that DFlash2 cannot boot/serve. It is a geometry investigation with one smoke response, not SWE completion, rate benchmark or broad correctness proof. |

## DSpark raw denominator, model and control details

The two raw `output_throughput` values reproduce exactly as `total_output_tokens / duration`:

- B1 label: 8 completed requests, 8,192 output tokens / 411.09202655707486 s = 19.927411554557715 output tokens/s; mean TPOT 49.74383229469673 ms.
- B8 label: 32 completed requests, 32,768 output tokens / 277.8548802298028 s = 117.93206573481409 aggregate output tokens/s; mean TPOT 60.711368336015106 ms.

This is benchmark-wall output throughput. It is neither the Cqc10 inverse mean of request mean-TPOTs nor task completion wall time.

Raw `server_info` records target `/models/qwen3.8-27b-nvfp4-radixark-asshipped`, draft `/models/qwen3.8-27b-dspark-draft`, `DSPARK`, one speculative step, eight draft tokens, DSpark block size seven, explicitly unquantized draft; target dtype and KV dtype are `auto`. Version is `0.0.0.dev0+qwen38.27b.g561c8f3`, attention `flashinfer`, linear attention `triton`, static-memory fraction 0.7. The contemporaneous report identifies the GB10 rig and approximately 1.36B BF16 draft. No exact target/draft HF revision is present in these JSON records; do not invent one or infer an exact KV dtype solely from `auto`.

Actual EAGLE control JSONs in the same directory: `bench_bs1.jsonl` = 26.100488306129982 output tokens/s and `n2_bench_bs1.jsonl` = 26.15115122039521, both eight completed requests and acceptance 2.901056338028169. They use the same target path/runtime, EAGLE three steps/top-k one/four draft tokens and no separate draft path. Different recorded random seeds and absence of a complete raw request identity comparison mean this audit does not upgrade these into a rigorously paired same-token trial. The narrative's roughly 24% DSpark deficit is a valid description of these recorded calibration values, not an agent-task or same-method causal result. One DSpark B1/B8 sample per configuration was found; no replicated DSpark SWE result was found.

## DFlash2 chronology and later serving identity

The August preflight describes 1.92B BF16, five layers, block eight, hidden-state-modulated selector and tap layers 5/19/33/47/61. Its then-installed SGLang image predated support, and its selector expected a dense target head incompatible with the particular NVFP4-packed head. Renaming architecture was explicitly rejected because it would drop selector/convolution weights. These are specific then-current route blockers; they must not be presented as a present universal DFlash2 limitation.

The August replay remained queued/paused in `REDTEAM_20260816.md:7050–7060,7255–7272,7304–7312`, including commits `3d5e5cd76` and `90ecf6ced`. The all-ref content search found no later completed result for that FR14 DFlash2 fusion replay.

The September serving arm instead used **Qwen/Qwen3.8-27B-FP8**, revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, with **incoai/Qwen3.8-27B-DFlash2**, revision `dedf8df68adfb1afeaf7b7480c0a0243108177b4`, seven speculative tokens. vLLM 0.28.0, Torch 2.13.0+cu130, CUDA 13.0, GB10 driver 590.48.01; TP1, prefix caching on, Mamba cache mode align, max sequences four, context 4,096, GPU memory utilization 0.50. Wheel SHA-256 `817b8181f7f61b4a62dc1d5d9ab39f2bfb60a6cb86c29879a78a147b85756787`. The archive `results/upstream/54076/evidence_54076_20260921T220949Z.tar.gz` and `SHA256SUMS` preserve the deeper evidence. This audit inspected committed reports/config/provenance, not a fresh runtime.

The corrected report explicitly limits the negative geometry finding to this model pair/version, identifies logging wrappers in the instrumentation, and does not claim a main-branch runtime test. The divergent 200/800 arm was `extract_hidden_states`, a training-data workflow, not the DFlash2 arm. Startup time (~365 s) is not decode throughput.

## Other non-SGLang leads (do not promote without their own raw audit)

- Native vLLM ngram/PLD comparison: `390739964`, `docs/reports/auto_research/track-b-real-task-warmonly-pr39562-matrix-20260507.md`. Qwen3.5-era serving, inline `release-note-to-plan-translation/v1-clean-baseline` task content, cold request discarded, then one warm completion at c1 or four at c4. This is content-bearing request benchmarking, not a complete tool-using agent trajectory. That report specifically supersedes an earlier favorable synthetic result.
- Arctic `SuffixDecodingCache` integration and ablations: `1d3b2ea7e` (`track-b-round2-shipped-20260509.md`) is implementation/measurement-plan evidence and explicitly says real-data measurement still queued. `f4f33ad41` and `cd6cc54b2`, `docs/reports/auto_research/track-b-round3-{closeout,e2e-v3-closeout}-20260510.md`, later report an executed 13-task in-house harness sweep with four repeats and 13/13 local correctness, not SWE-bench Verified. The reported median-wall reduction is not independently raw-reconstructed by this audit; it is an additional prior workload-comparison pointer, not a qualified current-paper headline. It compares evolving harness-coupled suffix routes, not an isolated DFlash baseline.
- Parent separately found historical native 48-task campaign `51b7dafdf`; its rate interpretation was already corrected by P0. This note neither revives those rates nor duplicates the parent's raw SGLang SWE census.

## Recommended manuscript disposition

Acknowledge that external baselines were exercised. A compact coverage statement can say: SGLang native/EAGLE and DSpark were served in calibration; DSpark was also evaluated on captured inputs with a negative fusion result; DFlash2 reached verified target capture on the NVFP4 study and later served an FP8 smoke on vLLM 0.28.0. Use the separate recovered SGLang SWE evidence for any workload-specific comparison. Do not relabel these DFlash/DSpark tests as missing experiments, and do not put their synthetic rates or cost projections into a table restricted to long-running agent workloads. No new experiment is recommended merely to correct the historical record.

## Checked source identities (SHA-256)

- `REDTEAM_20260816.md`: `902814907c5642ce917fbda8f7ca955b3d859d4caae2d22570b448c4121d2736`
- `promotion_ab_campaign.md`: `d3462d1741e230e596aa6799da90be3318872fe16b3b580f0faf0d2f902d357c`
- `dspark_bench_bs1.jsonl`: `e9467cf4da82be76c70d64d334e4443d2b88f9e59b4e2c696ed99366a9ee1aab`
- `dspark_bench_bs8.jsonl`: `bb4ca75a2a29f4232e3690129619442899528ddad97c3d0df0f19d3f8c262af6`
- `bench_bs1.jsonl`: `798229e37f82de94b11c2334f24d228f8bfec519aa6f655ffeba00dbbfd5d031`
- `n2_bench_bs1.jsonl`: `0b0fd61d203e9c26b463a419fb9aef1eb11ff9658c6a169a93ea49b1bd95bef9`
- Branch `54076/RESULT.md`: `0896eb4a6c29c5eb1d3b9c2daa6b3911fedd556676f34694dafb925d286e01e3`
- Branch `54076/CORRECTIONS.md`: `ffd4bf715aa064994b56ffe734edcae3ae5ee1c91f54306857437de30633811f`
- Branch `54076/PROVENANCE.txt`: `09d608fe759ce1f3499c8303f5469d2dd1e238b101729681beb7c5eb7684f641`
- Branch `54076/runs/arm1_dflash2/argv.json`: `30ee79916cba51505e7e662a0a9b1f98bdae2aad231a991cb3a1b2adb92c5190`
