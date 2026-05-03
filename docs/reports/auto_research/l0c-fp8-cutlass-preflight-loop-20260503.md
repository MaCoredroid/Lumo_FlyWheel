# L0c FP8 CUTLASS Preflight Loop Addendum

Generated: 2026-05-03

Round:
`output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260503T052540Z`

## Summary

This round reran CUTLASS-only `fp8_gemm` L0c after adding the cheap
`auto-research preflight-patch` command to the agent brief.

Terminal result:

- `outcome`: `ROUND_BLOCKED`
- `terminal_condition`: `compile_failures_3x`
- accepted candidates: `0`
- rejected candidates: `3`
- produced bundle: none

The terminal condition is mechanically from three candidates without
`mutation.patch`, but the candidate artifacts show the real reason: each agent
used the cheap preflight feedback and concluded that the only currently approved
CUTLASS bootstrap file cannot produce a valid preflight-clean mutation.

## Baseline

| Measurement | Eval throughput |
|---|---:|
| `cold_discard_00.json` | `0.034404` |
| `measurement_01.json` | `0.056745` |
| `measurement_02.json` | `0.056856` |
| `measurement_03.json` | `0.010705` |
| `measurement_04.json` | `0.057398` |
| `measurement_05.json` | `0.056918` |

Reported paired mean: `0.0477244`.

`measurement_03.json` is a severe low outlier relative to the other four
baseline measurements. The useful baseline signal is therefore that the normal
CUTLASS baseline remains around `0.0567`-`0.0574`, while this round's arithmetic
mean is contaminated by one runtime anomaly.

## Candidate Outcomes

| Iteration | Artifact | Outcome |
|---|---|---|
| `001` | `BLOCKED.md` | No `mutation.patch`; agent concluded every honest patch to the only allowed bootstrap file is preflight-demoted as `fp8_gemm_cutlass_python_wrapper_rewrite`. |
| `002` | `BLOCKED.md` | Same conclusion; refused to submit spoofed or alternate-format diff to bypass the path-based check. |
| `003` | `BLOCKED.md` | Same conclusion; also noted metadata-only edits would later fail the runtime-effect guard. |

No candidate called `auto-research apply-and-test`. The cheap preflight feedback
was available in the brief and was sufficient for agents to avoid wasting a vLLM
restart.

## Learning

The corrected agent contract is now better:

- agents run `patch --dry-run`;
- agents run `auto-research preflight-patch`;
- preflight output includes `matching_rule`, `code_snippet`, and
  `evidence_snippet`;
- agents revise locally when preflight fails;
- if no valid patch can pass the cheap checks, agents write `BLOCKED.md` instead
  of fabricating a patch.

The remaining issue is surface-level, not loop-level: the only current mutable
CUTLASS file is also the file the FP8 preflight rule forbids. That is internally
consistent for safety, but it means more budget on this surface cannot discover
a valid CUTLASS optimization.

## Recommendation

Do not run another loop on this exact repo-owned CUTLASS overlay bootstrap
surface. The next useful engineering step is one of:

1. expose a real vLLM/CUTLASS source-build mutation target;
2. build the isolated FP8 GEMM replay gate so selected wrapper/source
   replacements can be tested cheaply before Tier 4;
3. reframe the next round as backend/version selection against newer
   vLLM/CUTLASS builds.

Until one of those exists, extra attempts will produce more `BLOCKED.md` or
preflight-demoted candidates, not accepted throughput improvements.
