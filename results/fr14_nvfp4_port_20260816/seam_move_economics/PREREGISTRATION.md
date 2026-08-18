# PRE-REGISTRATION — seam-move economics (written BEFORE any coverability measurement)

Frozen at HEAD 05987f682. Read-only. Written before running the coverability simulator.

## Architecture (established from code + census, not from the measurement)

- `scripts/fr13_mtp_suffix_assembly.py`: `assemble_tail_tree(..., head_depth=5, mtp_k=5)`.
  Head = depths 0..4, spine tokens are pure MTP (`mtp_k == head_depth` => "byte-identical to
  the baseline cat33333"), 2 MTP runner-up branches per depth. Tail = pure Arctic suffix chain,
  node j at path length `head_depth+1+j` => **0-indexed draft position 5+j**.
- Census (this serve): `drafter.main_tail_length=6`, `arctic_requested_tokens=12`
  (`main`=6, `rank1`=4, `rank2`=2), `mtp_forward_calls=4`, `active_nodes=23`, 31 physical drafts.
- => positions 0-4 are MTP-drafted; positions 5-10 are Arctic-suffix-drafted. The per-position
  counters end exactly at position 10 (tail6). **The seam is between position 4 and position 5.**
- The code names the seam explicitly: "the tail's arctic top-1 conditional is weakest at the
  handoff j=0/d6" (Direction-2 d6-handoff repair comment).

## Observed ladder (measured, this serve, `metrics_after_swe.txt` minus pre-snapshots)

steps = 21611; accepted = 92439; accept/step = 4.2774; committed/step = 5.2774.
N_k = 20469,17281,13322,10768,8796,5253,4219,3619,3212,2884,2616 (k=0..10).
Survivals s_k = N_k/N_{k-1} (N_-1 = steps):
  s0=.9472 s1=.8443 s2=.7709 s3=.8083 s4=.8169 | s5=.5972 | s6=.8032 s7=.8578 s8=.8875 s9=.8979 s10=.9071
  ^-------------- MTP --------------^   ^ suffix slot 1 ^  ^------- suffix slots 2..6 -------^

Step cost (GPU-compute basis, `deploy_speed_b1radix.json`):
  sfwd/verify 134.55 ms + drafter 52.674 ms + committer 20.642 ms = 207.87 ms
  => derived_tps_fullstep_gpu = 5.2774/0.20787 = 25.39 tok/s  (wall basis: 215.31 ms, 24.51 tok/s)

## Coverability rule (PRE-REGISTERED)

Per task t, build the ordered token stream Sigma_t = [initial prompt] + [interleaved tool
results / user turns] + [model emissions], tokenized with the SERVED tokenizer
(`/models/qwen3.8-27b-nvfp4-radixark`). E_t = indices of MODEL-EMITTED tokens.

Suffix proposer P (stand-in for `arctic_inference.suffix_decoding.SuffixDecodingCache`):
at index j with true prefix Sigma[:j], take the longest L in [Lmax..Lmin] such that the L-gram
Sigma[j-L:j] occurs earlier in Sigma[:j]; propose the most frequent next token over all earlier
occurrences (ties -> most recent). Lmax=32, Lmin=2. Chain: append the PROPOSED token to the
pattern and repeat (mirrors the suffix-tree walk and the tail assembly, where tail node j is
`suffix_rel[j][0]`).

S(j) = max m such that the chain's first m proposals equal Sigma[j..j+m-1].

Measured quantities (all over j in E_t, pooled across the 4 tasks):
- q1      = P(S >= 1)                      — suffix cold-start hit rate, UNCONDITIONAL
- r_m     = P(S >= m | S >= m-1), m=2..8    — the suffix chain ladder
- cov_d   = P(S >= d), d = 1,2,3            — shallow coverability (the "completeness" ask)

### Validation gate (decides whether the simulator is trusted)
The simulated ladder r_2..r_6 must land within +/-0.10 absolute of the OBSERVED tail survivals
(.8032,.8578,.8875,.8979,.9071) for >= 4 of the 5 slots.
- PASS  -> the simulator is a faithful stand-in; its q1 is used as the seam-3 cold-start estimate.
- FAIL  -> report the simulated ladder but run the seam-3 arithmetic on the OBSERVED ladder only,
           with q1 bracketed [q1_sim, 0.5972], and disclose the failure.

### Reconstruction-fidelity gate
Reconstructed emitted token count per task must be >= 85% of the trace-reported `output_tokens`
(4356 / 49211 / 25690 / 34467). Below that, the emission stream is treated as partial and every
coverability number is reported as a lower bound only.

## Seam-3 arithmetic (PRE-REGISTERED)

Seam at depth 3 = MTP covers positions 0,1,2 (2 MTP forwards instead of 4; 3 spine depths instead
of 5); Arctic suffix chain covers positions 3+.

E[accept] = sum_k prod_{j<=k} s_j with
  s_0,s_1,s_2 = .9472,.8443,.7709 (unchanged — same depths, same MTP heads, measured today)
  s_3         = q1_seam3   (first suffix slot)
  s_4..       = observed suffix continuation ladder .8032,.8578,.8875,.8979,.9071 (slots 2..6)
q1_seam3 bracket:
  OPTIMISTIC  = 0.5972 (today's handoff conditional). Optimistic because the depth-5 population is
                pre-selected by 5 correct MTP tokens => easier / more copyable regions than depth-3.
  PESSIMISTIC = q1 (simulated, unconditional over ALL emitted positions => unselected, <= true).
Tail variants: tail6 (max position 8) and tail8 (max position 10; the head frees 6 nodes, so 2 more
tail nodes fit inside the same fixed-32 budget).

Cost side: step_time_seam3 = 207.87 - Delta, Delta = 2 MTP passes, bracketed [18.4, 26.3] ms,
central 21.0 ms (= 2 x 52.674/5 as briefed). Verify (134.55 ms) and committer (20.64 ms) held
fixed — rows_per_step stays 32 in fixed32 mode.

### Verdict rule
- FAVORABLE   : PESSIMISTIC q1, tail8, central Delta beats 25.39 tok/s by >= 5% (i.e. >= 26.66).
- MARGINAL    : OPTIMISTIC beats 25.39 by >= 5% but PESSIMISTIC does not.
- UNFAVORABLE : even OPTIMISTIC fails to beat 25.39.
