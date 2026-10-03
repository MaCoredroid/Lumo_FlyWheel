# M1-Q C0 collector repair closure

2026-09-28. The two findings in `m1-q-c0-collector-bounded-review.md` are closed for collector SHA-256 `ce57f174960f6de218a7f0710d2505405d1ad62902a07679faf55ef685ea4fd2`. The derivative launcher was source-diff reviewed at SHA-256 `34de4761a587ac7bd1cd3ca0e58a26458e5267269c5239d28cb0bdf3059e68b7`.

The collector now registers and persists each method/repeat/path before dispatch. An exception escaping `S.run_method` marks that cycle `stage_error`, preserves its traceback and partial-evidence path, and records the exit time before the outer handler persists the stage failure. A tensor-index hash error is appended without bypassing the final receipt. Failure of the normal final receipt demotes the stage and writes `RECEIPT.failure.json`; failure to write either receipt still cannot yield success. These changes add retention around the existing scientific path, with no extra call or retry.

Ten independent AST/stdlib CPU controls pass. They cover the two exact earlier counterexamples, the connected outer exception handler, the alternate failure receipt, refusal before dispatch if pre-dispatch persistence fails, exact four-method/two-repeat order and eight separate paths, persisted nonfinite stopping, pristine-input mutation refusal, ordinary finalization, and equality of all 14 Lumo environment settings to the unchanged adapter contract. Only fixed opaque strings/dictionaries are used. No Torch, NumPy, candidate, runtime module, fixture generator, launcher or GPU code is imported or called; the only file output in a control is a temporary JSON failure receipt.

Snapshot: `p0/monitor/review-response-20260927/m1q-c0-collector-repair-reviewed-20260928T220045Z/`. It contains the reviewed collector and launcher, the comparison native launcher and adapter, independent control source, raw test output, this note, and a SHA-256 manifest. The initial source and both reproducible failures remain separately preserved in the initial review snapshot.

Reproduce from the repair snapshot:

```sh
/Users/zhiyuanma/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -m unittest -v reviewer_controls
```

No additional blocker was found in this bounded collector review or the launcher derivative diff. Accepted v3.5 scientific adapters, preflight, capture/observer lifetime and original native-launch lifecycle were not reopened. The final actual gate, reference-eligibility receipt, deployed hash bindings, live admission, C0 execution and numerical reduction remain parent-owned steps and were not executed or independently observed here. This note closes source defects; it does not open a gate or establish a candidate result.
