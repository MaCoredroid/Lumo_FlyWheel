The payments on-call handoff started over-collapsing alert bursts after the
dedupe path was simplified.

Start with `logs/failure.log`, `docs/handoff_contract.md`, and the failing
tests. Repair the parser plus dedupe flow so the grouped incidents follow the
documented payments contract:

- normalize environment aliases like `Production` and `prod` to the canonical
  `prod` token
- bucket `observed_at` into the 5-minute `window_start`
- keep `payment_lane` stable on the grouped handoff entry
- preserve `occurrence_count`, `first_seen_at`, and `last_seen_at`
- prefer the stable `dedupe_hint` when it is present, because processor-specific
  title suffixes are noisy display text and should not split a payment incident
  family

Keep the handoff human-readable: preserve the earliest display title inside a
merged record instead of replacing it with the machine-oriented `dedupe_hint`.

Do not delete the failing tests.
