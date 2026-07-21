# FR13 Committer CUDA-Graph Port — Design (direction-2)

## Proven (micro-bench, scripts/fr13_committer_graph_microbench.py)
Capturing the 48-layer `fused_sigmoid` committer loop at FIXED shapes into ONE cuda graph is
**byte-identical** to the current varlen committer (max_diff=0.0) and **5.43× faster** (58.4→10.8 ms).
Enabler = **state-neutral padding**: pad tokens with `a=-1e4` (⇒ softplus→0 ⇒ decay=1) and `k=v=b=0`
(⇒ zero write) leave the GDN state EXACTLY unchanged. So variable accepted-path length no longer blocks
graph capture. `num_accepted_tokens` NOT needed. Distinct state-row indices per request are required
(collision ⇒ intra-call CTA race) — true in reality.

## The two shape variables that must be pinned for a single captured graph
1. **path length T_b** (1 + accepted): pad each request to `MAX_PATH` (const, = 1 + max tree depth; tail6 ~12).
2. **batch B = num_spec_decodes** (varies per step with eff_conc, 1..4): pad to `MAX_B` (= max_num_seqs for
   spec, i.e. BSIZE=4) with **dummy state-neutral requests** writing throwaway bank rows.
⇒ the captured loop always processes `MAX_B × MAX_PATH` tokens across `MAX_B` segments ⇒ ONE graph covers
every (B≤MAX_B, accepts≤MAX_PATH). Real requests occupy segments 0..B-1; segments B..MAX_B-1 are all-neutral
into a reserved scratch bank row (never read back).

## Persistent buffers (allocated once, module-level, graph-stable addresses)
Per layer `_L`: `kbuf[_L]`,`vbuf[_L]`,`abuf[_L]`,`bbuf[_L]` of shape `[MAX_B*MAX_PATH, heads, dim]`
(a/b are `[MAX_B*MAX_PATH, NVH]`). Shared: `qbuf` (zeros), `cu_fixed=[0,MAX_PATH,2*MAX_PATH,...]` (const),
`ssi_fixed[MAX_B, MAX_PATH]`. `banks_g` = the LIVE per-layer state banks (the graph reads+writes them
in-place via ssi_fixed — same tensors the model already owns; NOT copies).

## Per-commit replay (host, OUTSIDE the graph)
1. Reset pad region: `abuf.fill_(-1e4); kbuf.zero_(); vbuf.zero_(); bbuf.zero_()` once is wrong per-step —
   instead keep pad region permanently neutral and only overwrite the REAL slots each step (batched gather).
2. **Batched gather** (index_select from the k/v/a/b rings, like `_fr13_native_committer_all_layers_batched`)
   into the real slots `[b*MAX_PATH : b*MAX_PATH + T_b]` for b<B. Set `ssi_fixed[b,:]=col0[b]` for b<B and
   `ssi_fixed[B:,:] = <reserved scratch row>`.
3. `g.replay()` — the 48 fused_sigmoid as one graph.

## Capture-time (once, lazily on first commit after warmup)
Warmup the fixed loop 3× on a side stream, then `with torch.cuda.graph(g, pool=<persistent>): loop()`.
Guard with a module flag so it captures exactly once. Capture happens in the committer's EAGER region
(after the verify forward / during sampling-commit) — NOT inside vLLM's forward cuda graph.

## RISKS to clear before/at live bring-up (in order)
- **R-A nested/stream capture inside vLLM**: the committer runs eager (post-forward), but confirm no active
  capture and use a dedicated stream + persistent mem pool. If vLLM is mid graph-capture of the forward when
  the committer first runs, defer committer-capture to a later step (flag `..._GRAPH_WARMUP_STEPS`).
- **R-B pad region must be re-neutralized IFF the real region can shrink**: if step t used T_b=8 and step t+1
  uses T_b=3, slots [3:8] hold stale real data. FIX: after gather, explicitly neutralize
  `[b*MAX_PATH+T_b : (b+1)*MAX_PATH]` each step (cheap, batched). Verify in micro-bench with varying T.
- **R-C dummy-segment scratch row**: reserve one bank row never used by a real request; dummy writes land
  there and are never read. Confirm the reaper/allocator won't reclaim it.
- **R-D MAX_PATH overflow**: if accepts+1 > MAX_PATH (deeper than expected), FALL BACK to the eager varlen
  committer for that step (assert + counter). Never silently truncate (would be lossy).

## Flag + gates
Flag: extend `FR13_COMMITTER_NATIVE_BATCHED` with a `..._GRAPH=1` sub-mode (or new `FR13_COMMITTER_GRAPH`).
Default OFF. Gates in order:
1. micro-bench byte-identity + speed — **DONE** (0.0 / 5.43×).
2. micro-bench with VARYING T across replays (R-B) — pad re-neutralization correctness.
3. in-process live byte-identity: graph-committed banks == eager varlen committer banks, same boot, per step.
4. B=4 16-task SPEED gate (same pf/eff-conc as tail6 baseline, WALL=0): committer time collapse + accept
   unchanged + resolve band ~8/16. Label graph vs eager.
