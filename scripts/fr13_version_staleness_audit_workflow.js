export const meta = {
  name: 'fr13-version-staleness-audit',
  description: 'READ-ONLY audit (user): is any FR13 work STALE — (1) vs the REAL running image (0.19.2, the /tmp/vllm_live_019=0.19.0 confound: 123-line causal_conv1d diff, etc. → re-verify load-bearing conclusions + find dangling stale-cache/deleted-image references), (2) vs vLLM LATEST upstream (has upstream moved on / made any of our patches redundant — #42960 GDN batch-invariance, the GDN/scan kernels, the FA2-fork / tree-conv / committer)? Produce a precise CLEANUP LIST (source + image references to fix/remove) + a RE-VERIFY list (conclusions read against stale source). Adversarial verify. (Claude cleans up afterwards.)',
  phases: [
    { title: 'Audit' },
    { title: 'Verify' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121). Repo /home/mark/shared/lumoFlyWheel. READ-ONLY (a GPU fix-verify workflow',
'edits the scan kernel/gate/patcher concurrently — do NOT edit any code; write ONLY',
'FR13_VERSION_STALENESS_AUDIT.md). Pathspec commits only. Use ONLINE research (WebSearch/WebFetch the vLLM',
'GitHub repo + releases) for the latest-upstream comparison.',
'',
'GROUNDING RULE (user, MANDATORY): the running image = vllm/vllm-openai@sha256:3dbe092e = cu130-nightly =',
'0.19.2rc1.dev134. Read its source ONLY via `scripts/vllm_src.sh <relpath>` (cats fresh from the pinned image).',
'NEVER read a /tmp cache (they are DELETED + they drifted). The trigger for this audit: /tmp/vllm_live_019 was',
'a STALE vLLM 0.19.0 extraction that every agent read, diverging 15/40/123 lines from the running 0.19.2',
'(fused_recurrent / fused_sigmoid_gating / causal_conv1d) — silently off-citing kernel analysis. See',
'FR13_VLLM_SOURCE_OF_TRUTH.md.',
'',
'YOUR JOB — two staleness axes + a cleanup list:',
'AXIS 1 (vs the REAL running 0.19.2 — the version-drift hygiene):',
'  (a) DANGLING REFERENCES: grep the binds (*.md) + scripts for references to DELETED paths/images — the',
'      stale caches `/tmp/vllm_live_019`, `/tmp/vllm_img_0192`, `/tmp/vllm_pristine_019`, `/tmp/vllm-0.22-*`,',
'      `/tmp/fr10_vllm_src`, and the DELETED images `vllm/vllm-openai:latest`, `lumo-vllm-audit:v0.22.0-cu129-min`.',
'      List every file:line so Claude can fix/annotate them (esp. any LAUNCHER or live script that would break,',
'      vs binds that are just historical narrative).',
'  (b) STALE LINE-CITATIONS in LOAD-BEARING binds: the analyses were written against the 0.19.0 cache. For the',
'      load-bearing native-kernel citations, re-confirm against the 0.19.2 image via vllm_src.sh. PRIME SUSPECT:',
'      the CONV conclusion — `causal_conv1d.py` diverged 123 LINES between 0.19.0 and 0.19.2. RE-VERIFY that',
'      "conv FIXED+CLOSED, 3-tap FIFO, native conv_state_token_offset=num_accepted-1" STILL holds on the REAL',
'      0.19.2 causal_conv1d.py (does the spec-path state_len / offset logic match what FR13_CONV_CROSSEVENT_',
'      INVESTIGATE.md claimed?). Also re-confirm fused_recurrent (packed-decode recurrent rank-1, num_warps=1 —',
'      already re-verified, just cite the 0.19.2 lines) + fused_sigmoid_gating. Flag any conclusion whose basis',
'      SHIFTED on the real source. Note: commit 7441fc43 (the scan A/B gate + fr13_native_packed_decode_ref.py)',
'      was BUILT against the stale 0.19.0 source — check its native-kernel reference call is correct for 0.19.2.',
'AXIS 2 (vs vLLM LATEST upstream — is our work REDUNDANT/obsolete now?): the FR9 study (2026-06-03) found',
'  upgrading buys nothing (GDN batch-invariance #42960 OPEN, no isolated recurrent-forward primitive). RE-CHECK',
'  CURRENT (online): (a) is #42960 (batch-invariant GDN_ATTN) STILL open / any merged PR? (b) did upstream',
'  latest (0.22.x / main) change the GDN kernels we align to (fused_recurrent packed-decode, fused_sigmoid_',
'  gating, the chunked path) or the spec-decode tree machinery, such that our 0.19.2 alignment target or',
'  conclusions would differ? (c) does upstream latest now provide NATIVELY anything we BUILT (the FA2 tree-bias',
'  fork, tree-conv-fused, the GDN tree-scan, the committer, no-copy tree verify) — making it redundant? Be',
'  precise: cite the upstream PR/issue/release. The ANSWER may well be "still nothing for us" (confirm the',
'  why-not-upgrade decision holds) — but check, do not assume.',
'',
'DELIVERABLE: FR13_VERSION_STALENESS_AUDIT.md = (1) CLEANUP LIST (exact file:line dangling refs to fix/remove,',
'split live-script-breaks vs historical-narrative), (2) RE-VERIFY LIST (load-bearing conclusions whose basis',
'must be re-confirmed on 0.19.2 — with your re-confirmation result for conv + the kernels), (3) UPSTREAM',
'verdict (is any work redundant vs latest; does why-not-upgrade still hold). Be SKEPTICAL + precise; this audit',
'exists because we already got burned by a stale source. Quote FR13_BUG_CLASS_PLAYBOOK rows (#10 codegen-',
'identity, #11 naming-slip/version-skew). Commit pathspec.'
].join('\n');

phase('Audit');
const A_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['danglingRefs','convConclusionRecheck','kernelCitationRecheck','commit7441Recheck','upstreamRedundancy','batchInvariantStatus','cleanupList','reverifyList','committed','notes'],
  properties: {
    danglingRefs: { type: 'string', description: 'file:line refs to DELETED /tmp caches + deleted images, split live-script-breaks vs historical-narrative' },
    convConclusionRecheck: { type: 'string', description: 'does "conv FIXED+CLOSED / 3-tap FIFO / native offset=num_accepted-1" STILL hold on the REAL 0.19.2 causal_conv1d.py (the 123-line-diff suspect)? re-confirmed via vllm_src.sh' },
    kernelCitationRecheck: { type: 'string', description: 'fused_recurrent (packed-decode recurrent/num_warps=1) + fused_sigmoid_gating re-cited on 0.19.2; any basis shift' },
    commit7441Recheck: { type: 'string', description: 'is 7441fc43 (scan A/B gate + native_packed_decode_ref, built vs stale source) correct for 0.19.2?' },
    upstreamRedundancy: { type: 'string', description: 'does vLLM latest provide natively anything we built (FA2-fork/tree-conv/scan/committer)? cite PR/issue' },
    batchInvariantStatus: { type: 'string', description: 'current status of #42960 (GDN batch-invariance) in latest/main — still open / merged?' },
    cleanupList: { type: 'string', description: 'the precise actionable cleanup list for Claude (refs to fix/remove)' },
    reverifyList: { type: 'string', description: 'load-bearing conclusions whose basis must be re-confirmed (+ which already re-confirmed here)' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const a = await agent(
  CTX + '\n\nTASK (Audit, no GPU, read-only). Do axes 1+2. Write FR13_VERSION_STALENESS_AUDIT.md, commit '
  + 'pathspec. Return the schema.',
  { label: 'version-staleness-audit', phase: 'Audit', schema: A_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','convRecheckGrounded','danglingComplete','upstreamGrounded','anyConclusionInvalidated','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    convRecheckGrounded: { type: 'string', description: 'is the conv-conclusion re-check grounded in the ACTUAL 0.19.2 causal_conv1d.py (via vllm_src.sh), not asserted?' },
    danglingComplete: { type: 'string', description: 'is the dangling-ref sweep complete (binds + scripts) + correctly split live-breaks vs narrative?' },
    upstreamGrounded: { type: 'string', description: 'is the upstream-redundancy / #42960 check grounded in current online sources (cited), not stale memory?' },
    anyConclusionInvalidated: { type: 'string', description: 'did the real-source re-check INVALIDATE any FR13 conclusion (esp conv)? = a real finding to escalate' },
    recommendation: { type: 'string', description: 'the prioritized cleanup + any conclusion to re-open. No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(a) + '. Default holds=false if the conv re-check is not '
  + 'grounded in the ACTUAL 0.19.2 source (vllm_src.sh), the upstream check rests on stale memory not current '
  + 'online sources, or the dangling-ref sweep is incomplete. The valuable output is whether any conclusion was '
  + 'INVALIDATED by reading the real source. No close/pass-fail; no reward-hack.',
  { label: 'verify-staleness-audit', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { a, v };
