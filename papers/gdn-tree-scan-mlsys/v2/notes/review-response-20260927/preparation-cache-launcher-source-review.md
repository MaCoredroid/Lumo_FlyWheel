# Preparation-cache launcher source review — 2026-09-29 UTC

**Initial disposition: the cache handoff fixes the explicit-compile/read-only-cache conflict, but hold the Lumo successor until its setup-failure branch explicitly exits. AR/chain do not have that control-flow defect.** This is prospective source/CPU review only, not launch or gate approval.

No full launcher, generator, Docker/container, SSH, GPU, or model operation was run. The tests below execute only extracted shell fragments and standard-library Python against reviewer-owned temporary files. Implementation, predecessor artifacts, and gates were not modified by this reviewer.

## Reviewed initial identities

Paths are under campaign `workload-plan/launchers/`.

| File | SHA-256 |
| --- | --- |
| `make_preparation_cache_launchers_v1.py` | `777d0dbe7bd5fae45528ad010871b8119dbbff280f87844155909c6e0236b5f9` |
| `PREPARATION-CACHE-LAUNCHERS-v1.json` | `e150af5e677a7b966756a39c80824fbd55627c01394e982b7eea1e731863f5e3` |
| `ar_agent_serving_v4.sh` | `9fa2d0ef65e7e84691797ba50ad65dff78d83e382628b4a2f3cf48bd410dd240` |
| `chain_mtp_agent_serving_v4.sh` | `2c20a63ee823d5700fd8911129714ec09335fc701edfa63e78cd0e3b3241f999` |
| `lumotree_workload_owned_v2_3.sh` | `f00aba7ac740d72b83c19a5d6e2ddc043839446337e94a97be572f4735ab498f` |

All three predecessor/successor/diff hashes and successor sizes match the manifest. Recomputing each unified diff matches its artifact exactly; removing only the inserted preparation initializer and final prefix restore recovers each predecessor byte for byte. The Lumo predecessor is v2.2, SHA `eb8143836beddc5446b132381ad39c5c4625c89b873aa4b4d7b5566f0b5a6d5d`; this review does not reopen or newly approve its unrelated deltas.

## Confirmed behavior

Generator lines 35–49 insert the initializer before any container preparation Python and restore the engine prefix immediately before the existing observer `PYTHONPATH` export/final exec. Docker startup environment and all observation mounts remain unchanged, so post-boot environment/mount checks continue to expect the read-only `/opt/lumotree-observer/empty-pycache`. The preparation directory is container-local `/tmp/lumotree-prepare-pycache`, created with mode 0700. `PYTHONDONTWRITEBYTECODE` remains enabled. Engine arguments, model/numerical settings, patcher order, and observer activation point are unchanged. SGLang v3 remains unchanged because its recipe has no preceding file patcher phase.

A positive control used the actual extracted initializer/restore fragments with only the two constant directory paths relocated into a temporary test directory. Starting from the engine-prefix environment, a real `python3 -B` explicitly compiled a harmless one-line file during preparation. Final `exec python3 -B` checked the restored prefix and no-write flag. Result: rc 0; preparation directory mode 0700; exactly one `.pyc` under preparation; engine prefix still empty; final interpreter prefix restored. This is a shell/interpreter handoff check, not a Docker read-only-mount or production Python/image qualification.

## R1 — Lumo setup failure is not fail-closed

`make_preparation_cache_launchers_v1.py:43` and generated Lumo v2.3 line 7653 place this standalone AND-list before the remaining preparation commands:

```bash
[[ ! -e /tmp/lumotree-prepare-pycache && ! -L /tmp/lumotree-prepare-pycache ]] && mkdir -m 700 /tmp/lumotree-prepare-pycache && export PYTHONPYCACHEPREFIX=/tmp/lumotree-prepare-pycache
```

`set -e` does not exit for a failed non-final command in that AND-list. A pre-existing directory makes the first test false; a failed `mkdir` can similarly stop only the list. Both permit the next source line to run with the old prefix. Reproduction using the actual extracted line, an existing temporary directory, and `printf PATCH_STAGE_REACHED` on the following line returned **rc 0 and `PATCH_STAGE_REACHED`**. This defeats the intended freshness/setup admission even though a later explicit compile may ultimately fail.

AR/chain insert the same list into the complete existing `&& ... && exec` chain, so its failure short-circuits their remaining work. For Lumo, append an explicit checked failure branch, e.g. `|| { echo 'private preparation cache setup refused' >&2; exit 3; }`, or use individually checked statements. Preserve the engine prefix restoration and read-only mount. Required bounded closure controls: fresh setup/compile/final-exec succeeds; existing directory, dangling symlink, and injected `mkdir` failure each refuse before a preparation-stage marker. No full launcher or model execution is needed for this correction.

The preparation exports from `cpu-reference-workload-source-review.md` remain valid independently of this launcher control-flow issue. A final parent freeze and actual runtime admission remain separate.

## R1 successor closure — final disposition

**R1 is closed in Lumo v2.4. Accept the bounded cache-lifecycle source repair: AR v4, chain v4, and Lumo v2.4. No remaining blocker in this corrective delta. This is not full-launcher, boot, or gate approval.** The original finding above remains applicable to preserved Lumo v2.3.

Successor `lumotree_workload_owned_v2_4.sh` SHA `02ef55ea8a83f49a8ace3605a9d482f22c65728c88f7d7dceeed4edb5a76f46d`; `LUMOTREE-PREPARATION-CACHE-GUARD-v1.json` SHA `003bd21879e2172b79feaa1e770acbd21154ee0aafef93bb40f2463761b33e29`; `.cache-guard.diff` SHA `0ac0a28dd11a771e3f923c414d018106fbde0e141e0a4d055495ba914063d57c`. All hashes match their manifest. Recomputed diff matches exactly. Replacing the single new anchor with the old anchor restores the pinned v2.3 predecessor byte for byte. The only addition is the explicit refusal/exit branch at line 7653; engine prefix restoration at line 8386, mounts, environment, model settings, numerical criteria, and serving command are unchanged.

Four controls used the actual v2.4 initializer and restore lines, with only their constant directory paths relocated into temporary reviewer-owned directories:

| Control | Result |
| --- | --- |
| Fresh directory; explicit preparation `py_compile`; final `exec python3 -B` | rc 0; mode 0700; one `.pyc` only under preparation; restored engine prefix stays empty; no-write flag retained |
| Existing preparation directory | rc 3; preparation-stage marker absent |
| Dangling preparation symlink | rc 3; preparation-stage marker absent |
| Injected `mkdir() { return 42; }` | rc 3; preparation-stage marker absent |

Every negative case emits only `private preparation cache setup refused` on stderr. These controls execute neither the full launcher nor any image/container/GPU path. Target-host Bash syntax verification remains parent-owned; the manifest remains a draft requiring parent freeze, and no launch or workload counter changes follow from this closure.
