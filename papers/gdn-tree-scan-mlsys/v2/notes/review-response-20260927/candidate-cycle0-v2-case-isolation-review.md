# Candidate cycle-0 v2: per-case isolation review

**Bounded disposition: no concrete mapping or reference-selection blocker found in the reviewed delta.** Accepted as source/CPU preparation only. No accepted full native-corpus result exists yet, so this review does not admit a candidate run or establish qualification.

| Reviewed source | SHA-256 |
|---|---|
| `tools/q1_candidate_case_binding_v2.py` | `5876f349cbe8126c40d27e70a61d60d208978e5464fe732c76769fe835364882` |
| `tools/q1_candidate_job_v2.py` | `a077baabdf77b4c3a61e0d8458f378b2ca0dc531c6d074b3f40d75ce9e854b50` |
| `tools/q1_candidate_hooks_v2.py` | `b54c58fb8df781f5715e649244226d62254fccce0d97d70cf22645b781e9eaa0` |
| `tools/q1_candidate_driver_v2.py` | `a1a73199402a3776eae02f7af3d7e6f5973bd7a36530b75a3919f6d8a00cd6bb` |
| `tools/q1_patch_candidate_v2.py` | `408424da30044d097880aa4307e72764f6f0306017039742e0821bc3ebc0a70a` |
| `tools/tests/test_q1_candidate_case_binding_v2.py` | `61809529d965b7f90603c4f56c225a1cd2ab6fc7f340d29363e3a0971a15153b` |

The job builder requires the current source-bound native manifest, a hash-bound approved `Q1-NATIVE-CORPUS-RESULT`, a complete 672-observation reduction, and all 84 primary cases qualified. It selects the declared `aligned_nonpacked` process A run and repeat 0 **for every same case**, then matches the selected receipt/job hashes and each selected record's seal, O0/O1 digests, raw O2 hash and decision to the approved reduction. Native result acceptance is a prerequisite; it is not a search over alternative references. Neither candidate outcomes nor native agreement/margin are used to choose among A/B, r0/r1, packed/nonpacked or cases.

`validate_layout`, `case_view`, `control_view`, `prefix_ids` and `native_record` consistently bind the case ID, prefix ID/bytes, continuation tokens/positions, selected native run/arm/process/repeat, control identity and state/decision record. Scalar v1-style prefix/reference fields are forbidden in the 84-case job. The builder uses the actual frozen fixture cases, and native manifest equality fixes their complete 3-prefix × 28-path calibration population. The runtime rejects cross-case or incomplete bindings rather than substituting a default reference.

In the hooks, `_bind` validates the next control and native record before installing the new `CandCase`, its prefix, native binding and native O0. An unsealed preceding case or invalid next reference latches the process unusable. The new draft and acceptance paths use `c.prefix`; they no longer read a single job-wide prefix. The inherited shadow buffers are fully copied for each state snapshot rather than serving as cached prior-case data. S0 hydration, publication checks and raw tensor machinery are otherwise the preserved v1 implementation and were not broadly re-reviewed or re-executed here.

The driver obtains prompt bytes and the native comparator anew for each case, executes R=2 with distinct observation IDs, and checks the seal's native binding against that case's selected binding. Its complete-collection flag remains distinct from numerical `all_agree`; a successful collector return is not itself a numerical-qualification verdict. The patcher changes module/schema/marker references to v2 while retaining the same six S0–S5 insertion seams.

## CPU evidence and boundaries

`python3 -B -m unittest -v tools/tests/test_q1_candidate_case_binding_v2.py` passed all **5 test methods**. These use actual fixture/prefix bytes and explicitly synthetic native records/seals, including 168 mock transport calls. They do not execute model forwards or tensor hydration.

The reviewer additionally extracted the actual `CandCase`, `ProcessUnusable` and `_bind` AST into a stdlib-only harness. Across **336 metadata-only transitions** (84 fixed cases × R2 × candidate process A/B), distinct per-case synthetic O0/O1/O2 identities verified that both repeats in either candidate process select the same case's fixed native A/r0 and correct prefix. Independently refused controls covered cross-case native substitution, r1 reference substitution, process B reference substitution, cross-prefix substitution, an authenticated record file belonging to another case, and control change while the previous case remained unsealed. Temporary CPU fixture files were removed afterward.

The final driver includes the parent's scoped file-handle writes. All reviewed Python files parse. No scientific package, gate, launch source, result or authority was modified by this reviewer; no GPU, Docker, cache, SSH, network, model or workload operation was performed. A concrete result-backed job, final launcher/source bindings and parent admission are later stages. This review makes no continuous-cycle, held-out, MTP-state, lifecycle, timing or workload-quality claim.
