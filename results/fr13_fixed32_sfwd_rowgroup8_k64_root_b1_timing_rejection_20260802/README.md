# SFWD rowgroup8 K64/root1 B1 timing rejection

This reduced package records a matched stock/candidate timing diagnostic on one
authenticated real SWE-Verified task. Both arms resolved the task, passed its
tests, used the pinned K64/root1 route without fallback, reconciled every timing
counter, and tore down cleanly.

`fixed32_sfwd_state_fusion_rowgroup8_v3` is rejected for performance promotion.
It measured 244.569046597 ms/step versus 242.286839679 ms/step for stock: a
2.282206918 ms, or 0.941944235%, regression. Full-step wall TPS fell from
21.129126406 to 20.635455840. The candidate was authenticated as served, so this
is a measurement of the candidate rather than a silent reference fallback.

The stock phase breakdown was 170.855128182 ms SFWD, 35.233577381 ms DFWD,
25.274926515 ms CFWD, and 10.923207601 ms other. SFWD therefore owns 70.52% of
the measured wall and remains the primary optimization target. The candidate
regressed SFWD by 1.702695103 ms and its three GPU components by 2.774341027 ms.

The cited 119.658015414 ms value is only an optimistic mandatory-weight-read
lower bound. It is not a complete physical hardware-floor step. Against its
1.15x threshold of 137.606717726 ms, stock remains 104.680121953 ms high and
would require a 43.205% wall-time reduction. Even deleting all measured
non-GPU-component wall time would leave stock 93.756914353 ms above that cap.

This is a B1 diagnostic, not an acceptance result: one task cannot provide the
required one-sided U95. Exact4 B4 or exact16 remains mandatory for acceptance.
No raw task, model, request, response, patch, environment, process, container,
credential, or run-log data is included.
