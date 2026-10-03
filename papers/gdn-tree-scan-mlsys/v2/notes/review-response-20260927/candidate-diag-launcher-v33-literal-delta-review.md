# Diagnostic launcher v3.3: bounded literal-delta review

**Disposition: no blocker in the requested source-only delta.** This review grants no scientific gate or launch authority.

| Artifact | SHA-256 |
|---|---|
| `tools/q1_make_diag_launcher_v3_3.py` | `31dd7d86b3058d2f07983832aedc8dd729409195195075ff77267013f7abcc0c` |
| Pinned v3.2 base | `db89386fb76db8594c00b575bddf369f6f8b05a6183f425734d1679f81de5352` |
| `tools/generated/fr14_leg3_launch_nomiddleware.q1diag.v3_3.sh` | `d0a78fc5031319451f7d28664f59c0ede179e8e9ca2a7579d490e6706d2236a6` |
| `identity/launcher-diagnostic.v3_3-from-v3_2.diff` | `beba2c6f3255f926d5d1e193b028bb84185c7648e05f41b120f58cfb22199904` |

The actual generator output exactly equals the reviewed base with five `q1_patch_candidate_v1.py` → `q1_patch_candidate_v2.py` literal replacements. Inverse replacement restores the base byte-for-byte. The retained unified diff also exactly matches a freshly computed diff. The five sites are the host patcher path (line 7116), its failure description (7119), in-container presence check (8215), in-container invocation (8217), and provenance path/hash check (8302). Host and container therefore select the same v2 patcher, whose previously reviewed delta imports `q1_candidate_hooks_v2`.

No model, engine option, topology, numerical setting, environment forwarding, mount, timer, diagnostic exception or gate behavior changes in this literal delta. The generator checks the pinned base SHA before replacement and requires exactly five occurrences. It never executes the launcher.

Safe CPU controls confirmed the clean render, exact inverse delta, base-byte-drift refusal, four-occurrence and six-occurrence refusal, idempotent write of identical output, and refusal to replace different existing output. Occurrence controls adjusted only an in-memory fixture hash to reach the independent count guard; production pins and files were untouched.

Local macOS `/bin/bash` is Bash 3.2 and cannot parse the inherited `[[ -v ... ]]` guard at line 283. Its syntax failure is not a v3.3 regression; no claim of target Bash 5 validation is made by this review. Target-shell syntax checking remains parent-owned. No remote, Docker, GPU, cache, model or launcher operation was performed, and no implementation or gate was modified.
