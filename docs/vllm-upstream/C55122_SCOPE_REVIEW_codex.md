GO on one brief scope reply; NO-GO on the draft as written. The explicit mention warrants clarification, although jschmied already attributes the speedup to k3dani—do not imply he misattributed that number.
Verified live: #55122 is OPEN at b2312b2de793ba12ac5822286e832e0ea332e9e0; both cited anchors resolve to our original result/rerun comments, with the stated dates and correctness-only scope.
Both kernel files are byte-identical between 19588c89 and b2312b2d. From tested 85f61e24b to 19588c89, the header changed in the untested FilteredTopK wrapper; the tested persistent path stayed unchanged. This establishes source continuity, not execution/integration validation of the current head.
Delete “so that re-run stands for the current head.” Delete “No objection to the reframing”: it adds no result and invites interpretation as endorsement. No new ping, benchmark offer, or merge position is needed.
Exact replacement (≤110 words):
To clarify the scope of our contribution: our [GB10 fallback results](https://github.com/vllm-project/vllm/pull/55122#issuecomment-5648007112) and [rerun at `85f61e24b`](https://github.com/vllm-project/vllm/pull/55122#discussion_r4043382139) checked exact-reference agreement and repeatability on the reported float32 grid. We measured no performance, so our results do not validate the 21–28% speedup. The tested persistent path remains unchanged in source, but we have not executed the current head. Operator integration and CUDA graphs were outside our harness coverage.

AI assistance was used.
