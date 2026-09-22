# E8 qualification-v2 infrastructure review — 2026-09-22

**PASS for one replacement infrastructure qualification attempt in a fresh output root.** Exact manifest: `bc2feab326cb67c916ea418b0399499f4986e18d6bba3e4efc7a20a50f0ce030`. This permits the already specified two untimed ON/OFF qualification boots; it is not a successful scientific qualification or permission to start timing. No GPU, real Docker container, model import, inference or memory-recovery operation was run by this reviewer.

The stage is `experiments/e8-single-logits-qualification-v2` under local `/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/papers/gdn-tree-scan-mlsys/v2` and remote `/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2`.

## Failure and exact repair

The first attempt `experiments/out-20260922T2119Z-e8-single-logits` failed before Docker creation. The generator replaced only `# REPO IS DERIVED` and inserted a newline, turning the remaining `, NOT HARDCODED. This read` into a validly parsed shell command. The original launcher exited127 at line11. `bash -n` could not detect this execution defect; my first review should not have treated syntax checking as sufficient execution validation. The failure is appended to the original `e8-harness-redteam.md` chronology.

Preserved failure evidence:

- Original stage manifest: `45da91477bf238e99f70c2201bc722059427ff9cced399754171a4371c41efcf`.
- `FAILED.json`: `60a01a22ed29afa94e0100f43bdd417746df1fa80905145eafeac10bab235096`.
- `qualification_on/launcher_stdout.txt`: `9cdc27b64fd8053d3cfa4d14e9c73ab96da6ac6e414cdd9e973a51d6f35a12fa`.

The sibling generator now replaces the full original comment line and retains it as one comment. Independent manifest/file comparison found only two changed existing payloads:

| File | Corrected SHA-256 |
|---|---|
| `e8_prepare.py` | `79183f68d02d2b4cfe3453ccf17e772399cc0aee7e5b4ca368b8755b32b2faae` |
| `e8_launch.v1.sh` | `37392279706b080d3d7b694cd4cf31cbf7b08bf947ce269672557b9054269d96` |

The filename is retained for the unchanged driver's reference; the sibling directory and manifest identify revision2. All non-file scientific manifest fields, four emitted eagle variants, head gate, shim, run verifier, qualification driver, original CPU tests and frozen E1/repository dependencies are byte-identical to the reviewed original. Added files are the infrastructure record and execution test:

- `INFRASTRUCTURE_FIX.md`: `fb5d77ab78f8fe9411d26742a2f466e8f212f8655a88f70d6ea422b9cbf0c3de`.
- `test_launcher_execution.py`: `7658338f30aceb7ef0572650372e389fbf2429637e22b4b58f20a28e8ebcc82b`.

No numerical criterion, workload, source-identity rule, receipt requirement or API coverage gate was relaxed. No scientific observation existed in the failed attempt; this is one transparently recorded infrastructure replacement, not repetition of a failed numerical result.

## Independent execution evidence

I read the stub runner and personally ran this command on the Linux DGX host:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python3 -B \
  /home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/e8-single-logits-qualification-v2/test_launcher_execution.py \
  --original /home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/e8-single-logits \
  --output /tmp/e8-qualification-v2-independent-20260922.json
```

**PASS: ON, OFF, original regression.** Independent output SHA: `46bd6f0611de59fe0885b3dfb0ab0d28ee98872b671668c8af36eee72eb5924a`.

The test executes the actual generated outer Bash script, real stage verifier and model-directory plumbing against a temporary mock model. Only the allowlisted memory recovery/check heredocs are stubbed. A strict Docker executable stub captures the generated arguments; its captured inner `bash -lc` command is then actually executed with strict Python/vLLM stubs. It validates frozen patcher → correct-arm qualification shim → frozen recorder shim → final vLLM command, explicit engine seed20260921, B1, Cat10, eager/synchronous/cache-off execution and all six read-only source overlays. No model or Docker executable is invoked.

Both corrected arms reached the vLLM stub. The unmodified original stage passed its real source-manifest validation and then reproduced exactly return127 / `,: command not found`, with only the stage-verification event and **no Docker call**. This is an execution regression control, not a corrupted-manifest refusal substituted for the actual defect.

I also independently reran the original **27/27 CPU controls** in a temporary copy of the corrected stage. The final manifest remained unchanged. The prior independent head/API/source-gate review continues to apply because those files are unchanged; no redundant GPU test was run.

## Execution disposition and limits

The parent may hand the corrected sibling to the sole GPU worker once existing weight/idle-host preflight is satisfied. Use a fresh uniquely named root, the sibling's unchanged `e8_qualify.py`, and explicit `--execute`. The original failed root and original source must remain preserved and must not be relaunched or regenerated. Exact command form:

```bash
E8_V2_RUN="/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/out-$(date -u +%Y%m%dT%H%M%SZ)-e8-single-logits-qualification-v2"
python3 -B /home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/e8-single-logits-qualification-v2/e8_qualify.py \
  --repo /home/mark/lumo-paper-v2-20260921 --run "$E8_V2_RUN" --port 9968 --execute
```

Resources, expected15–25-minute two-boot duration, fixed deadlines and first-failure stopping rule are unchanged. Any same-input head/candidate/hidden/RNG, source, API, coverage or seal failure must stop the experiment before timing. The driver still cannot launch clean timing. A later timing stage must bind this corrected qualification manifest and actual successful receipts, preserve the original failed infrastructure attempt, and receive its own execution/census/analysis review.
