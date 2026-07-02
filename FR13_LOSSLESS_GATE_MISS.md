# FR13 — Why our "tree/spine/FA2-forked = lossless vs native MTP" conclusion missed the structured-output carrier

**Date:** 2026-07-02. Workflow wyqldaxuz (5 agents, verified vs the on-disk gates/docs) + adversarial verdict.

## The one-sentence answer
**Every historical "lossless" gate validated a MODE-preserving or SELF-CONSISTENT invariant (argmax/greedy
match, LCP-superset over the tree's OWN drafts, byte-identical committer rewrite, teacher-forced re-convergence)
— exactly the invariant a kernel can PASS while corrupting the temp-0.6 SAMPLED tail at the high-entropy
tool-call / XML / codefence boundaries where the corruption actually lives. The one gate that measures the
deployment-binding invariant (temp-0.6 q-vs-p distributional TV/KL) was fully DESIGNED, found NOT COMPUTABLE
("q is not banked"), queued behind the speed campaign, and NEVER RUN.**

## Gate by gate (verified against source)
1. **Superset gates** (fr7/fr10/fr13-perevent, B1/B4 CPU): prove only "tree acceptance ≥ the tree's OWN path0
   LCP." `fr10_superset_gate_report._path0_proxy_native` LITERALLY substitutes the tree's own path0 as the
   "native" comparator → only "tree ≥ tree-path0" can ever fail; native-equivalence is structurally untestable.
   perevent defines lossless = served == greedy-argmax of a **teacher-forced** oracle (mode, not distribution),
   scores ~27 leaf positions, calls ~17 spine flips out-of-scope, and `spine_regressions==0` is true BY
   CONSTRUCTION (committer LCP tie-break).
2. **argmax/greedy/byte gates** (fr12_compare_argmax, fr13_argmax_lcp_localize @temp0, gpu_committer_byte_ab
   @greedy-LCP, apc_greedy_divergence @TEMP=0/TOP_K=1): all at temp-0/TOP_K=1 or a byte-identical rewrite →
   blind to sub-argmax tail reshaping. And **spec-decode is distribution-preserving ONLY under exact rejection
   sampling**; this path uses a deterministic **LCP-max GREEDY committer** (`_lumo_tree_path_lcp_max_greedy_sample`)
   to which the theorem does NOT apply — yet "lossless" was claimed as if it did. Temp-0 argmax rate cat9
   13.5% ≈ native 14.0% was the passing proxy — "necessary, not sufficient (class #9)"; the flips concentrate
   at format/codefence/tool-call boundaries (CARRIER_REOPEN).
3. **The sufficient gate never ran:** FR13_TEMP06_DRIFT_GATE.md fully designs the temp-0.6 q=softmax(verify/0.6)
   vs p=softmax(decode/0.6) TV/KL gate, argues it is the ONLY deployment-binding test, then §3: "NOT COMPUTABLE
   (q is not banked)"; GPU capture "queues AFTER the speed campaign." So the deployment-binding lossless number
   was never produced.
4. **FA2-carrier-overturned** (FR13_FA2_CARRIER_OVERTURNED_BIND): proved only FA2 is not the ORIGINATOR of the
   argmax-flip (moved it to L0-GDN co-residency); QPAD fixed the named carrier (L31 3.9e-3→0) yet e2e flips did
   NOT drop (24 vs 22, admitted trajectory-confounded) — read as "FA2 exonerated" instead of "the corrupting
   mechanism is elsewhere in the same forked/tree path, still uncaught."
5. **chain5-vs-native "Fisher p=0.44 NS":** a FALSE-NEGATIVE — N=1 resolve power AND codex's `/v1/responses`
   AUTO_CONTINUE nudge MASKED the malformed tool-calls (chain5 recovered to 3/5). The recovery layer masked the
   carrier in the very A/B meant to detect it. The SAME corruption is fatal 0/8 on nudge-less qwen-code.
6. **Teacher-forced recurrent oracle:** MANUFACTURES the snap-back read as "benign self-healing" (forced onto
   the served prefix → every divergence re-converges in 1-2 positions BY CONSTRUCTION; class-#12 trap in
   CARRIER_REOPEN).
7. **char-8 gate:** declared "cache-independent" behavioral catch-all, but it counts only "Unterminated string"
   (codex JSON) and keys turns on `item.completed` (codex JSONL). On the qwen-code forked arm it sees 0 → turns=0
   → char8=0 → **false PASS**; 10 qwen3xml parse-fails uncounted. **The agent swap silently voided our one
   behavioral gate.**

## Corruption profile (deep trace dive, angle A)
Onset at the natural-language→tool-call-XML boundary (clean prose, then the first `<tool_call>`/`<function=`
delimiter breaks). Catastrophic-per-turn, 6 sub-modes (malformed/mis-nested XML, token loop, off-topic derail,
empty-args, malformed-JSON, silent stall). **NOT context-length driven** — forked deaths span turn 1→46 at flat
~23-32k tokens while native is clean at overlapping/larger 22.6-44.7k. Native is functionally lossless (2.5%
benign slips, each self-recovered in 1 turn; 4/5 resolved). Discriminator is **recoverability**, not raw warning
count (native logs MORE: 25 vs 10). XML dialect of the codex char-8 malformed-tool-call carrier.

## The fix — add a generation-integrity tier (numerical-lossless ≠ behavioral-lossless)
1. **Structured-output carrier gate** (`scripts/fr13_structured_output_carrier_gate.py`, to build): dialect-aware
   exposure=assistant-turns (the existing char-8/replica gates score qwen-code as turns=0 → false PASS — MUST
   fix), per-turn 5-label classifier (clean / malformed-markup / degenerate-loop / off-topic / silent-drop)
   unifying char-8-JSON + malformed-XML, one-sided Poisson rate + replica self-noise floor.
2. **De-confounding grid, not an A/B:** {FLASH_ATTN, TREE_ATTN} × {num_spec 5, 8}, EXACT_SEED held ON, K≥4 seeds
   /cell @temp0.6, ≥8 shared tasks incl 13453; regress corrupt-turn rate on attention-family with num_spec as
   covariate. Only (TREE,8) failing while (FLASH,8) & (TREE,5) pass isolates the kernel.
3. **Hold agent+nudge constant** across arms (add the /v1/chat/completions nudge OR run codex on both); report
   per-turn emission-integrity SEPARATELY from terminal empty-patch outcome.
4. **Actually RUN the temp-0.6 q-vs-p TV gate** — the only sufficient distributional check. Demote
   bit-exact-@1024 / state-diff / drift-curve to NECESSARY-NOT-SUFFICIENT L0 numerical rungs. A config is
   "lossless" only after ALSO passing (a) the carrier gate on the grid with agent/nudge held constant and (b)
   the temp-0.6 q-vs-p TV gate. Numerical-lossless + behavioral-lossless = two required axes.

## Adversarial calibration (do NOT overclaim yet)
- **miss-diagnosis: SOUND** (high conf, source-verified).
- **The "forked kernel IS the carrier" conclusion is NOT yet established** — FOUR confounds: (i) kernel vs
  num_spec=8 vs topology coupled; (ii) the **nudge-net confound** — a qwen-code turn-1 stall is terminal
  regardless of kernel (AUTO_CONTINUE inert on chat/completions), so "forked carrier" is entangled with
  "qwen-code lacks a nudge net"; (iii) task-overlap is N=2 and only 13453 is an informative discordant stratum
  (12907 fails both arms = non-informative); (iv) the mechanism (temp-0.6 tail corruption) is asserted but the
  sufficient measurement (q-vs-p TV) was never run — outcome and mechanism not independently established.
- **The carrier gate as specified is NOT valid** — false-positives on native (code-indent trips the char-repeat
  rule), miscalibrated wordfrac threshold, the docker PARSE_FAIL join is anti-correlated with outcome (drop it
  from the pooled stat), SILENT-DROP re-imports the nudge confound (report separately). Fix + re-validate on
  13398/14182 before use.
- **Decisive experiment:** the de-confounding 2×2 grid (agent/nudge held constant) + the temp-0.6 q-vs-p TV gate.
