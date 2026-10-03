# Candidate joint S0 mapper: initial bounded review

2026-09-29. **Blocked on one connected state-selection/metadata group.** The union import, bank-view binding, lazy alias handling and post-write identity checks are reusable. This report reviews the preserved initial mapper, not the parent's subsequent repair. No GPU, remote operation, engine execution or source edit was performed.

Reviewed source: `experiments/review-response-20260927/tools/q1_candidate_joint_common_o0_v1.py`, SHA `2c925f950de2eaff471fbd7708fa91c65e6b964130c8273bed38b61b58847648`; tests SHA `03d78aff8b91ef27204328f67a2d7c97b4500259c021b9e2f25bdf601f70977e`. Immutable copies and `SOURCE.json`, `supplied-tests.txt`, `ADVERSARIAL.json` are under `p0/monitor/review-response-20260927/candidate-joint-mapper-independent/`.

## F1: an allocated GDN page is not necessarily the running-state page

Mapper lines 44–50 accept SSI column zero whenever the page occurs anywhere in the CPU/GPU table and request allocation. They do not bind it to the actual logical running-state column. Nor do lines 19–25/45–46 check live sequence length, `mamba_state_idx`, non-spec counts or the complete scratch window. Consequently stale/native metadata and a different allocated page can pass the entire import.

Independent real CPU control (Torch 2.8.0, `/usr/bin/python3`, `CUDA_VISIBLE_DEVICES=''`, OMP/MKL/torch threads 1): starting with the supplied fixture, give group0 table/request blocks `[1,2]` and group1 `[2,1]`, then set all group0 SSI roots to 2 and group1 to 1. Both roots remain allocated and writes remain disjoint. **The frozen mapper completes the union import with one mutation latch**, although the actual S0 column is zero and requires roots 1 and 2. Additional mapper calls accepted `seq_lens[0]=66` for P=65, `mamba_state_idx['r']=999`, and metadata claiming both 32 spec tokens and nonzero prefill/decode populations. These are actual false acceptances, not hypothetical source-record tampering.

The production rule is source-derived:

- Pinned candidate `workload-plan/inspections/cpu-lumotree-source-20260929T012000Z/result/sources/v1/worker/gpu_model_runner.py:5712–5741` calls aligned `preprocess_mamba` before preparing the forward.
- The same inspection's `v1/worker/mamba_utils.py:575–592` sets `curr_state_idx = ceil((num_computed + num_scheduled)/block_size)-1`; speculative-block count cancels from the selected running column.
- Its `v1/attention/backends/gdn_attn.py:1066–1073,1153–1158` derives SSI from the aligned block-table gather. The immutable image utility at `workload-plan/inspections/cpu-native-state-map-20260929T121600Z/utils.py:879–893` gathers from `floor((seq_lens-1)/block_size)`.
- The candidate's `_fr13_mamba_spec_scratch_window` at `gdn_attn.py:97–125` keeps that running page in column zero and repeats the one spare page across logical columns 1–31 under the narrowed two-page route.

Thus at the declared S0, require the actual scheduled/query span 32, live `seq_lens[0] == P+32`, `column == (P+31)//1024 == runner.mamba_state_idx[rid]`, and active SSI root equal to the CPU/GPU/**request** GDN page at that exact column. Validate the actual MambaSpec and available running-plus-spare extent, the expanded 32-column active SSI window, and pure spec counts (`num_spec_decodes=1`, `num_spec_decode_tokens=32`, no non-spec decode/prefill tokens). Bind these values into the before/after identity snapshot so a change during import is caught.

Do not use the native one-token formula `P//1024`: the two formulas differ near a block boundary. Do not require exactly one physical SSI row: `gdn_attn.py:1294–1304` can expose graph-padded rows with inactive rows filled with NULL. The one active row/window and truthful metadata counts are the invariant. Update the supplied positive fixture: it presently retains the native `seq_lens=66` and width-one SSI at P=65, so its success does not prove deployed 32-row metadata compatibility.

## Checks that remain sound within this scope

All eight supplied tests passed when executed against the snapshotted mapper and tests, using real CPU tensors and the unchanged four v9 bank/registry methods extracted by the supplied AST seam. The independent MTP/target alias control refused before any write (`joint MTP destination overlaps target`). The single accepted `Transaction` still authenticates all source bytes before mutation, preserves the explicit union/complement rules, refuses reuse, and signals post-latch failures. The mapper does not itself authorize continuing a poisoned process; the future hook must retain v9's fail-stop handling of the mutation latch/import exception.

Lazy aliases are supported by pinned `gpu_model_runner.py:6989–7032`: both caches are populated at first publication after S0. Requiring both absent or both registry-bound is reasonable at the S0 mapper. The existing target publication witness separately checks ordered name-to-cache binding at v9 hooks lines 439–442; there is no need to reopen that accepted algorithm here.

Source admission, joint-prefix provenance, raw post-import records, candidate graph/categorical observations, final caller/freeze integration and runtime qualification remain outside this mapper's source-only closure. Fix F1 and run bounded state-column/boundary/metadata regressions before claiming the mapper ready; no broader campaign is requested.
