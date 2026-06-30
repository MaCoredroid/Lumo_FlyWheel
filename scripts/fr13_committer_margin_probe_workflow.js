export const meta = {
  name: 'fr13-committer-margin-probe',
  description: 'COMMITTER-REPLAY MARGIN PROBE (user chose "probe then margin-damp"). The +16 leaf carrier is an LCP-committer trajectory fork (FR13_LEAF_CORESIDENCY_PATH, verify HOLDS): leaves commit drafts[leaf] when a leaf path LCP >= spine LCP. K1 is a verified PARTIAL kernel lever (de-cascaded 18->12, accept held) but +9 above native; the residual is the committer fork. DECISIVE QUESTION: are the residual cat9 forks (A) GENUINE leaf-LCP wins (the verify argmax/parent_target at the LCP-divergence node is CONFIDENT, top1-top2 > 1 nat = the leaf genuinely matched a confident verify decision = FUNDAMENTAL, removing it loses the accept edge) or (B) SUB-1-NAT NEAR-TIE nudges (the leaf won the LCP boundary only because a co-residency/FA2 perturbation tipped a near-tie verify argmax = FIXABLE by a deterministic rank-2 LCP near-tie margin-damp that does NOT force-pick the spine)? Instrument the committer (read-only diagnostics dump, default-OFF flag) to capture per-node verify top-2 logprobs at the fork positions, boot cat9 WITH K1 ON (FR13_SCAN_ALIGN=1 MODE=body = the candidate config), classify A vs B. Mostly B => margin-damp viable => K1+margin-damp = lossless+fast candidate. Mostly A => fundamental => relax. Single GPU boot + CPU classify. Adversarial verify.',
  phases: [
    { title: 'ProbeClassify' },
    { title: 'Verify' },
  ],
}

const CAT9_TREE = '[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 1), (0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 0, 1)]';

const CTX = [
'FR13 on DGX Spark GB10 (sm_121, bf16/fp8 GDN-hybrid Qwen3-Next). Repo /home/mark/shared/lumoFlyWheel.',
'GPU SERIALIZED. Pre-boot hygiene: source .venv; recover_host_memory(); assert MemAvailable>=100GiB + docker ps',
'empty. Teardown trap: docker rm -f + recover_host_memory after the boot. boot ENFORCE_EAGER=1.',
'',
'GROUNDING RULE (user): read vLLM source via scripts/vllm_src.sh (pinned image 3dbe092e = 0.19.2rc1.dev134),',
'NEVER a /tmp cache. CHECK ARTIFACT TIMESTAMPS before reading any output.',
'',
'COMPARE TARGET (user): lossless = cat9 vs native-E5 each-vs-its-own-no-spec-oracle (RECURRENT single-step',
'decode, scripts/fr13_recurrent_decode_oracle.py). native-E5=3 = BAR. cat9+K1 OFF-residual = 12 de-cascaded',
'(FR13_K1_STORE_BOUNDARY_BIND). DEPTH-MATCH: cat9 is depth-5 so cat9-vs-E5 is correct (do NOT compare a depth-3',
'arm to E5). int-view NEVER atol. clear-margin = deviation_nat>1.0 gold-margin (full oracle_topk).',
'',
'THE CARRIER (FR13_LEAF_CORESIDENCY_PATH.md cc602efc, verify HOLDS): the leaf forks are an LCP-COMMITTER',
'trajectory fork. The committer (_lumo_tree_path_lcp_max_greedy_sample, scripts/fr10_phase4_patch_vllm_tree_gdn.',
'py L6818-6896): for each root-to-leaf path, lcp = longest prefix where drafts[node]==parent_targets[node]',
'(L6827-6831; parent_targets[node] = argmax of the VERIFY forward logits at node). best_path = max-lcp path,',
'tie-break earliest-leaf (=spine for sorted trees) (L6839-6843). Commits drafts[node] for the lcp prefix +',
'bonus (L6874-6896). A FORK = a leaf path lcp >= spine lcp so best_path becomes the LEAF and drafts[leaf] is',
'served (a token the spine never serves), OR a co-resident leaf shifts parent_targets at a near-tie so the lcp',
'boundary moves one. EXISTING infra to build on: FR13_COMMIT_ARGMAX_GATE (L6900+) already accesses the verify',
'logits per committed row + names the exact verify row.',
'',
'THE CLASSIFICATION (the decisive fundamental-vs-fixable question):',
'At each FORK position (a clear-margin flip where cat9 served != recurrent-oracle greedy), find the LCP-',
'DIVERGENCE node = the node where the WINNING leaf path continued matching (draft==parent_target) but the spine',
'path stopped (or vice-versa = the node that decided the leaf-vs-spine lcp). At that node read the VERIFY',
'forward top-2 logprobs (the margin of parent_target = top1 - top2):',
'  (A) GENUINE leaf-LCP win = the deciding parent_target was CONFIDENT (top1-top2 > 1.0 nat). The verify forward',
'      strongly preferred the token the leaf matched -> the leaf is a real spec candidate; the flip is the',
'      verify-vs-decode realization gap at a confident node = FUNDAMENTAL (margin-damp would reject a genuine',
'      accept = loses the accept edge).',
'  (B) SUB-1-NAT NEAR-TIE = the deciding parent_target won by < 1.0 nat (verify nearly indifferent between the',
'      leaf token and the spine token). A co-residency/FA2 perturbation could tip this argmax -> FIXABLE: a',
'      deterministic rank-2 rule "do NOT let a leaf win the lcp boundary on a sub-1-nat parent_target" would',
'      stop the fork while genuine (A) leaf wins still serve. NOT force-spine (genuine leaf wins keep serving).',
'',
'YOUR JOB:',
'PHASE 1 (ProbeClassify, GPU):',
'  (1) Read the committer L6818-6896 + the FR13_COMMIT_ARGMAX_GATE verify-logit access (L6900+). Add a READ-ONLY',
'      diagnostics dump behind a NEW default-OFF flag (e.g. FR13_FORK_MARGIN_DUMP) that records, per spec-step:',
'      for each path its lcp + path nodes; for the lcp-divergence node of the winning path the VERIFY top-2',
'      logprobs (top1-top2 margin) + the parent_target id + the spine-token id at that node; the chosen best_',
'      path/best_leaf + committed token. Gate on the flag (default-OFF = NO dump = byte-identical served path,',
'      bug-class #10). Add `-e FR13_FORK_MARGIN_DUMP` to the FORKED launcher (like the 365da33b FR13_SCAN_ALIGN',
'      passthrough = the PROVEN worker channel; the locked launcher does NOT forward it). Commit pathspec',
'      (committer file + launcher), default-OFF.',
'  (2) Hygiene + boot cat9 via the FORKED launcher (TREE=' + CAT9_TREE + ', locked pipeline flags) WITH',
'      FR13_SCAN_ALIGN=1 FR13_SCAN_ALIGN_MODE=body (K1 ON = candidate config) + FR13_FORK_MARGIN_DUMP=1,',
'      ENFORCE_EAGER=1, temp 0.0 seed 1313 prompts_swe4. NON-VACUITY (playbook #9, 4 burned this session):',
'      (i) DUMP FLAG LIVE: bridge-needle worker /proc/<pid>/environ for FR13_FORK_MARGIN_DUMP=1 (fail loud if',
'          absent) AND the dump file is NON-EMPTY with per-node margins; (ii) K1 live: FR13_SCAN_ALIGN=1 in',
'          worker environ; (iii) RECURRENT_PATH_ENGAGED=True on the rescore; (iv) within-boot det.',
'  (3) Rescore the served stream vs scripts/fr13_recurrent_decode_oracle.py -> the clear-margin FORK positions.',
'      JOIN each fork to its committer-dump record (assert the fork positions MATCH dump records = non-vacuous,',
'      not a disjoint capture). Classify each fork A (margin>1nat) vs B (margin<1nat). Teardown + recover.',
'PHASE 2 (Verify). Count A vs B over the residual forks. VERDICT: (mostly B = FIXABLE) => the rank-2 LCP near-',
'tie margin-damp is viable -> K1+margin-damp is the lossless+fast candidate (recommend implementing + e2e',
'testing it next; user pre-authorized "probe then margin-damp"). (mostly A = FUNDAMENTAL) => the leaf forks are',
'genuine accept-edge wins, margin-damp would cost accept -> relax to accept/event-parity. Report the A/B count +',
'the margin distribution. Reward-hacks BANNED: the dump is READ-ONLY (changes NOTHING served); margin-damp (if',
'recommended) must be deterministic + NOT force-spine (force-spine-commit = BANNED, FR13_FORCE_SPINE_COMMIT',
'exists as a diagnostic ONLY); native = A/B oracle only; no copy/dense/multi-spine. Quote FR13_BUG_CLASS_',
'PLAYBOOK (#9 vacuous/flag-not-live, #10 codegen-dead, #12 trajectory). NO bake/close decision (user call).',
].join('\n');

phase('ProbeClassify');
const P_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['dumpInstrumentation','dumpFlagLive','k1Live','oracleEngaged','forkPositionsMatchDump','n_forks','n_fundamental_A','n_fixable_B','margin_distribution','committed','ok','notes'],
  properties: {
    dumpInstrumentation: { type: 'string', description: 'where the read-only per-node verify top-2 margin dump was added (committer line) + the default-OFF flag + launcher passthrough; default-OFF byte-identity proof' },
    dumpFlagLive: { type: ['boolean','null'], description: 'FR13_FORK_MARGIN_DUMP=1 in worker /proc/environ AND dump file non-empty with per-node margins?' },
    k1Live: { type: ['boolean','null'], description: 'FR13_SCAN_ALIGN=1 in worker environ (candidate config)?' },
    oracleEngaged: { type: ['boolean','null'] },
    forkPositionsMatchDump: { type: ['boolean','null'], description: 'the rescore fork positions JOIN to committer-dump records (non-vacuous, not disjoint)?' },
    n_forks: { type: ['integer','null'], description: 'number of clear-margin fork positions classified (the residual, ~12-18)' },
    n_fundamental_A: { type: ['integer','null'], description: 'forks where the deciding parent_target margin > 1 nat (genuine leaf win)' },
    n_fixable_B: { type: ['integer','null'], description: 'forks where the deciding parent_target margin < 1 nat (near-tie nudge)' },
    margin_distribution: { type: 'string', description: 'the per-fork deciding-node verify top1-top2 margins (the A/B split evidence)' },
    committed: { type: 'string' },
    ok: { type: 'boolean' },
    notes: { type: 'string' },
  },
};
const p = await agent(
  CTX + '\n\nTASK (ProbeClassify, GPU). Instrument the read-only dump (default-OFF), boot cat9+K1, PROVE dump-live'
  + ' + k1-live + oracle-engaged + fork-positions-match-dump BEFORE classifying. Classify A vs B. Teardown + '
  + 'recover. Return the schema.',
  { label: 'committer-margin-probe', phase: 'ProbeClassify', schema: P_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','nonVacuous','classificationSound','fundamentalVsFixable','marginDampViable','nextAction','rewardHackCheck','issues'],
  properties: {
    holds: { type: 'boolean' },
    nonVacuous: { type: 'string', description: 'dump-live + non-empty per-node margins + fork-positions-match-dump + oracle-engaged all proven? (not another vacuous instrument)' },
    classificationSound: { type: 'string', description: 'is the A/B split from the ACTUAL captured per-node verify margins (spot-check 2-3 forks), and is the >1nat/<1nat rule applied at the correct lcp-divergence node?' },
    fundamentalVsFixable: { type: 'string', description: 'verdict: mostly fundamental (A, relax) or mostly fixable (B, margin-damp viable)? with the count + margin evidence' },
    marginDampViable: { type: ['boolean','null'], description: 'is the rank-2 LCP near-tie margin-damp viable (enough B forks) without force-spine / accept loss?' },
    nextAction: { type: 'string', description: 'if fixable: implement+e2e-test K1+margin-damp (user pre-authorized). if fundamental: relax to accept/event-parity. For the user.' },
    rewardHackCheck: { type: 'string', description: 'dump READ-ONLY (nothing served changed); margin-damp deterministic + NOT force-spine; native=oracle only; no copy/dense/multi-spine' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(p) + '. Default holds=false if the dump was not proven live'
  + ' / non-empty, the fork positions do NOT join to dump records (disjoint capture = vacuous), the A/B split is'
  + ' not from actual captured per-node margins (spot-check), the divergence node is mis-identified, or any'
  + ' recommended margin-damp is force-spine / a reward-hack. Conclude honestly fundamental-vs-fixable. No'
  + ' bake/close decision.',
  { label: 'verify-committer-margin-probe', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { p, v };
