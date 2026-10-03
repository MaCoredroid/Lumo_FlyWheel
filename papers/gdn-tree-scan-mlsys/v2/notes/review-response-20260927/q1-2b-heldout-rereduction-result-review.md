# Independent held-out CPU re-reduction result review

2026-09-28 UTC. **Bounded result-audit PASS.** The separately authorized CPU re-reduction completed successfully and agrees with the independently reviewed raw case census. No material result mismatch found. Parent retains qualification/gate authority; this review does not qualify a full model, convolution/KV continuation, sampling, serving or timing.

New result directory: `experiments/review-response-20260927/runs/q1.2b-heldout-reduction/q12b-heldout-reduction-20260928T020837Z`. Original GPU-evidence directory remains `runs/q1.2b-heldout/q12b-heldout-20260928T013539Z`. Remote repository is `mark@100.103.10.122:/home/mark/lumotree-review-20260927`; paths here are relative to paper v2 unless stated otherwise.

## Verified result

The authorized CPU process ran `2026-09-28T02:10:03Z–02:11:02Z` and exited 0. The new receipt records `COMPLETED_reduce_rc=0`. The summary has zero malformed, structural, numerical or uncovered findings and exactly:

- **5,760 PASS ordinary cases per process**, comprising the same 3,072 BF16 output and 2,688 FP32 publication cases already audited, or 276,480 head observations per process.
- **11,520 numerical records recomputed** = 2 processes × 5,760 cases.
- **5,814 referenced tensor objects verified**, matching the earlier independent SHA census exactly.
- **Zero held-out negative mutations**, with power explicitly inherited from accepted calibration run `q12b-calibration-retry-20260928T010045Z` under the original gate.

The new summary retains the original A/B attestations byte-for-value, including the defective old self-helper map. Its compatibility record explicitly names the original reducer SHA and **original exit 2**, the two original refusal messages, and the exact self-helper reconciliation. It does not rewrite the original result into a historical PASS.

These completed tensor/C2 recomputations now corroborate the prior independent saved-array observations: all 2,688 published recurrent states per process match native bitwise; **54 BF16 output cases differ** (31 ordinary, 23 stress) while satisfying the unchanged paired criterion. Full A/B and within-process repeat identities were established in the preceding raw audit. Repeats and processes do not multiply the independent fixture count: this remains the two fixed evaluation fixtures, not additional unseen inputs.

The store still contains 6,198 objects; the **384 unscored native repeat-census states** at nodes 18, 23, 25 and 27 remain outside the reducer's 5,814 verified-object count. No stored/verified/scored denominator is inflated.

## Authorization, runtime and preservation

I verified the exact CPU-only authority snapshot (`2a732de1…`), new reduction ID, original run ID and original evidence pins. Its source acceptance, freeze, independent source-review hash, reducer and launcher all match the current bytes read remotely. The fixed command uses the pinned image, **no `--gpus`**, `--recompute all`, `--block evaluation`, read-only original run/repository mounts and a separate writable reduction directory.

Actual image inspection records image ID `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc` and repository digest `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`. CPU container `ee2dcdd32b19…` matches the runtime hostname. Runtime is Python 3.12.13 / Torch 2.11.0+cu130, OMP/MKL/torch threads 1, `CUDA_VISIBLE_DEVICES=''`, `cuda_available=false`.

I independently rehashed the **eight original terminal/summary/launch/gate/A-result/B-result/A-inventory/B-inventory files**, matching their pre-repair hashes. The original terminal remains `COMPLETED_reduce_rc=2`, SHA `7746dd7b…`; the original failed summary remains SHA `646a07b7…`. I also verified all **12 members** named by the new reduction receipt, both remotely and against the local mirror. There are 13 files including that receipt itself. The actual numerical computation was performed by the authorized frozen reducer, not repeated by this reviewer.

## Source-bound evidence

| Artifact | SHA-256 |
|---|---|
| New `summary.v2_2_1.json` | `a1a52aaf0ab290647865873e805646dcf3d87f5e3480132da8c6489c31951a71` |
| New `REDUCTION-RECEIPT.json` | `680d468a446574d908cca3c6921dccdf4bd9f9fd9748b87784a93625d55af283` |
| New `REDUCTION-BINDING.json` | `ff5a052453ab19875ffa5384e743fa4b6c7e36960fb23f47c8b5d38a960e93fc` |
| `AUTHORIZATION.snapshot.json` | `2a732de1ad7b18485d73638446f5d42e1f1b3ae0cdaf70d2028acec8071e2f42` |
| Repaired reducer | `a7decd00863f69b6970f7e1015a1d725681f046ebc65c63a535d5bc42da00cc8` |
| CPU launcher | `6aa6a9c3f374840b728d64f565c73bbd445e8843105d0bc06259fbe121c383c5` |
| Final source freeze | `cd1fddb44aaae81f1e20b9bf9eb58fa0b3c671a5fec239e52a23bc5f6353565e` |
| Parent source acceptance | `c5dcc1850e5ac812fddb357dcc1cb6a1ef00b45e2c86d7cca8f6836e5a93d640` |
| Independent source closure | `62596ec99a6cc282e50b89a7152fdd69efc0198337235bdfcd0e01dd6173e1de` |

Independent read-only reproducer and compact audit JSON are in `notes/review-response-20260927/raw-audit-q12b-heldout-reduction/`:

```sh
ssh mark@100.103.10.122 'python3 -B -' < notes/review-response-20260927/raw-audit-q12b-heldout-reduction/inspect_reduction.py > notes/review-response-20260927/raw-audit-q12b-heldout-reduction/independent-reduction-audit.json
```

This completed exit 0. Script SHA `4c1c7c4995c3e9a0ff1d2d25840f0a828e9c06d0cbe78a958b2d5f77e97d1bd3`; JSON SHA `b693b2615f7e750547511c0dd1a5f69d0c9797802399ab0e0887bd9a2d35d3ea`. It verifies receipts/source identities/counts and rehashes only small original metadata; no raw tensor rehash, numerical reducer, Docker or GPU execution occurs. The earlier full raw audit and its failure report remain unchanged. No further rerun is requested by this bounded review.
