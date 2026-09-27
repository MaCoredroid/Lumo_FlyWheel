# Cqc10 agent-behavior claim — independent check, 2026-09-23

**Supported with a named diagnostic limit:** all ten Cqc10 attempts terminated, produced nonempty patches, and triggered no recorded trace degeneration flag. This supports ordinary observed agent activity in this selected run, not a guarantee of degeneration-free decoding, unchanged task quality, or speed relative to another mechanism.

Recommended abstract sentence:

> In a selected ten-task Astropy run, all agent attempts terminated and produced nonempty patches, with no degeneration detected by the recorded trace checks.

Recommended body clarification:

> The trace heuristic jointly flags at most 200 visible characters, zero tool calls, and at least 2,000 thinking characters; none of these ten traces triggered it. It can miss loops that continue narrating or using tools, so this observation does not establish degeneration-free generation or preserved task quality.

## Direct evidence

Read-only SSH inspected the original `qwen_trace.jsonl` for every task at `/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816/output/fr14_promoab_Cqc10_20260824T074813Z/hydra27_fixed32_promoab_Cqc10/swe_out/verified/per_task/`. The pinned repository eyeball functions were replayed in memory, without modifying remote files or launching model work. All ten trace hashes match their saved runner-metadata trace receipts; there are no malformed JSON lines. Head/tail visible-text excerpts describe repository inspection, edits and test execution; all ten end with task-specific summaries. Those self-reported success statements do not override the separate six-resolved/four-failed evaluator results.

| Task suffix | Assistant records | Tool calls | Visible characters | Degeneration flag | c5 |
|---|---:|---:|---:|---|---:|
| 13977 | 92 | 33 | 3,217 | None | 0.5744 |
| 14096 | 74 | 29 | 3,228 | None | 0.5831 |
| 14182 | 104 | 35 | 2,244 | None | 0.5812 |
| 14309 | 20 | 6 | 1,364 | None | 0.6169 |
| 14365 | 53 | 17 | 2,250 | None | 0.5182 |
| 14369 | 104 | 36 | 3,898 | None | 0.5863 |
| 14508 | 77 | 32 | 4,386 | None | 0.5793 |
| 14539 | 98 | 37 | 2,292 | None | 0.5533 |
| 14598 | 104 | 35 | 3,082 | None | 0.5288 |
| 14995 | 32 | 11 | 2,274 | None | 0.5928 |

All ten have zero malformed-tool-argument flags. “Assistant records” is the trace parser's count, not the 256 normal service requests. The parser's malformed flag covers string arguments that fail JSON parsing; it does not certify every tool call's semantics. `c5` is the per-task metric-bracket delta at accepted position 5 divided by position 4. Its ten values independently reproduce the campaign note and fall inside the recorded [0.40, 0.70] band. Neither this acceptance statistic nor absence of malformed arguments proves task correctness.

## Limits and counterexample

- `results/fr14_nvfp4_port_20260816/promotion_ab_eyeball.py:81–108` implements the conjunction. Lines 93–99 explicitly disclose only one known positive example and missed modes when looping retains narration or tools. Lines 1–7 place judgment in the campaign's reading of excerpts rather than treating statistics as a verdict. The retrospective threshold and this selected ten-task negative sample do not establish calibrated sensitivity.
- `promotion_ab_campaign.md:5413–5435` records the Cqc10 trace review; lines 5508–5534 describe the repaired visible-text/conjunction diagnostic and its limits. The older `promotion_ab_exact16_eyeball.json` actually holds only three Cqc16 task records, so its filename is not evidence of ten Cqc10 checks. This audit therefore reads Cqc10 original traces directly.
- The actual prior Cqc16/13236 trace is a counterexample to any campaign-wide “no degeneration” claim: one assistant record, zero visible characters, zero tools, 70,755 thinking characters, and a positive degeneration flag. Its `c5=1092/3121=0.349887856456264` is outside the band. Trace SHA `676bf2a7b79328052a9d4f7d1c40e60dbdd91173d5b0892634fca0bf53bb057a`. This earlier failed segment remains separate; Cqc10 does not erase it or identify a causal repair by itself.
- All normal-stop completions and all nonempty patches are supporting execution facts, not substitutes for the trace evidence. Four saved evaluator failures remain failures despite the agents' concluding success summaries.
- The 35.46 ms request-mean TPOT / 28.20 tokens/s inverse mean remains a descriptive Cqc10 statistic. These behavior checks add no matched native/tree or other-mechanism speed comparison. No new model experiment is required for the bounded sentence above; a comparative speed assertion still needs matched workload data.

## Bound artifacts

The source script is identical locally/remotely: `promotion_ab_eyeball.py`, SHA `f4cc87ade9f4e6866dafcdf8f2429085ede0dce5aceb96db380e0cc808aa2ea0`. The campaign note SHA is `d3462d1741e230e596aa6799da90be3318872fe16b3b580f0faf0d2f902d357c`; the older three-record eyeball JSON SHA is `f6985aec7c2226d39bc490d4027046a840189a9e1f28c618c8d2b58b8f04b8d1`.

New evidence is separate from the already sealed Cqc10 bundle:

- `results/agent-workload/cqc10-behavior-audit.json`, SHA `5c3c6501093cbebe6da8e6114cc548a7b2b748136d168478cad86653da0224a7`: exact trace/source/metric hashes, parser results, excerpts and ten-task plus prior-positive ledgers.
- `results/agent-workload/cqc10_behavior_readonly.py`, SHA `7a84011f97598b5eeb1576f88f5ac082cf8f1c28f049db0865215d3da1815087`: reproducible read-only remote reduction; no GPU use and no remote writes.
- `results/agent-workload/BEHAVIOR_MANIFEST.json`, SHA `60606fd37fa463844715cf586d824611c1b30587dd68c8811e87ccec7b6ccb59`: binds these additions and the unchanged prior evidence-manifest hash. Complete raw traces remain at their recorded source paths; selected excerpts and hashes are retained here.

## Final wording recheck

PASS on current `abstract.tex` SHA `4bedfb502db02d1f23ffe61461036d125748bf0fd6a7e959f6f9c5d21669ed1b` and `results/agent-workload/case-study.tex` SHA `997bceba4787944c56327822d4655e73c92a8e8c56805ffec9bf42098c538312`. Abstract line 2 uses the supported selected-run sentence and explicitly leaves relative speed and unchanged quality unestablished. Case-study line 33 includes the exact diagnostic conjunction and its narration/tool-loop blind spot; line 44 retains the earlier adverse lineage. No further experiment or wording correction is necessary for this bounded behavior claim.
