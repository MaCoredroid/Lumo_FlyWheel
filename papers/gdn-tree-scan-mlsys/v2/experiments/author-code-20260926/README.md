# Author-code component checks — 26 September 2026

Completed on GB10, using the existing pinned local container image in RUN-RECEIPT.json. The source manifest preserves upstream URLs, full commit IDs, licenses and SHA-256 hashes. Import adapters create package namespaces to avoid loading an unrelated installed serving stack. They change no author kernel arithmetic or test assertions.

- TreeWY: **39 selected tests passed** (16 GPU kernel/graph cases and 23 reference cases). Original JUnit and terminal log retained. The entire upstream test suite, native-kernel parity dependencies and full model stack were not run.
- Weaver: unchanged author BF16 script completed **9 numerical cases and 2 timing configurations**, with its original 65-node shape. The script has no numerical pass/fail threshold; completion is not qualification against a common reference. Timings omit state commitment and are not compared to LumoTree.

Reproduce inside the recorded CUDA environment:

```sh
python run_treewy_tests.py
python run_weaver_bench.py --tokens 65 --repetitions 1000
```

The Weaver serving backend has both chunk and fused verifier routes; both stash operands and invoke accepted-path recurrent replay. The optional factor-fold helper is not evidence that the serving backend commits that way. A next comparison must select and record the actual route.

Next: match current LumoTree and author-kernel inputs/topologies/dtypes, qualify outputs and consumed accepted state, then measure complete verification/commit cycles and separate allocations/traffic. Do not interpret these checks as a result on SWE-bench Verified or as evidence of an intrinsic numerical defect in any method.
