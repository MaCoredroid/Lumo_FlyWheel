# Joint-native outer integration: narrow closure review

Disposition: all three findings in `q1-native-joint-outer-integration-review.md` are closed for the reviewed source/CPU scope. No remaining blocker was found in these corrective deltas. The original source, two powered failures and initial review remain unchanged. This acceptance is not a source-inventory result, frozen plan, launch gate or GPU qualification.

**R1, source path visibility: closed.** The config now authenticates the exact job bytes in `log_dir/q1_ref_hooks_job.json`, validates the canonical source manifest/run paths and joint scope, and adds exactly two read-only mounts with identical host/container destinations. The hook and host auditor can consequently use the same bound paths. The positive CPU renderer control uses a real 84-case fixture job with explicitly synthetic source locations. Removing the two added mount pairs restores the base Docker argv exactly; target serve arguments, environment, image, model and patched FA2 fields remain equal. Bad job SHA, escaped manifest path, foreign source-run location and target-only scope each refuse. No container mount was exercised live.

**R2, dependency execution before authentication: closed.** The plan records `source_paths`; the launcher's first heredoc uses stdlib to validate all 45 declared paths and their hashes against both bound manifest and gate, rejects escapes, checks completeness through AST local-import traversal, and authenticates the two external patch scripts before inserting the campaign import path or importing the plan. The original mismatched-dependency sentinel was replayed through the actual complete heredoc and refused without executing the sentinel. Separate missing-member, `..` escape, symlink escape, foreign gate hash and foreign external-script hash controls also refuse before campaign import. A valid declared closure reaches the end of the authentication block. These controls use temporary synthetic gate/manifest declarations, not an actual launch authority.

**R3, receipt description: closed.** The obsolete three-source-preparation wording is removed; the note now describes 84 calibration paths, R2 and one selected A/B process. Scientific/timing disclaimers remain bounded. Target driver, patch transforms, apply adapter and hook bytes are unchanged from the initial review, so their existing positive/fail-stop controls were not repeated.

Twelve focused CPU controls passed, plus `bash -n`. Evidence: `p0/monitor/review-response-20260927/native-joint-outer-repair-review-20260929/`, including captured sources, `check_closure.py`, exact results and source closure identities. No GPU, Docker, SSH, remote mutation, source edit or gate mutation occurred. Actual three-source inventory authentication and subsequent parent freeze/admission remain future operations.

| Reviewed file | SHA256 |
|---|---|
| `q1_native_joint_common_o0_plan_v1.py` | `efdef44ae470e4c61c91b0d2a6da979a4028a769cd2865410f91e1f40cdadd64` |
| `run_q1_native_joint_common_o0_v1.sh` | `dd83f2863e18d82350d34adfb5e43d55e07cc3b8e234955139d46f76974186a1` |
| `q1_spec_off_joint_common_o0_config_v1.py` | `233e4a9d8dbf537f5baa80b3f5084b2333ab6564a891b713ba1e307193c3c604` |
| `q1_reference_driver_joint_common_o0_v1.py` | `bcb975c8750ed0b59dc8de7b1aca224b605476dae5be174ec840743020ade19f` |
| `q1_patch_native_mtp_joint_common_o0_v1.py` | `d4b05bc7e400c856788a27513737e8a8939fa8cf1539ea8ca4b24d900cfc82c7` |
| `q1_apply_native_mtp_joint_common_o0_v1.py` | `a87bda8293721a395cfda8656c036cea8e0a581217593e65d07419421fe743f2` |
| `q1_reference_hooks_joint_common_o0_v1.py` | `6d4a95e06a95f8f00f708800a146ca92a6264a0e5d80e3210a97061866c18809` |
