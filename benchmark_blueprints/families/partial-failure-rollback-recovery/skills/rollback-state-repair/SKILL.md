# Rollback State Repair

Use this skill when a local maintenance failure leaves state split across rollback, queue, and ledger surfaces.

## Workflow
1. Identify which state must be restored and which evidence must survive.
2. Distinguish retryable failures from terminal cleanup.
3. Repair rollback logic before touching the ops note.
4. Validate both consistency and evidence preservation.

## Anti-Patterns
- Deleting evidence to simplify cleanup.
- Treating every failure as terminal.
- Fixing ledger state while leaving queue state inconsistent.

## Done Signal
- Consistency is restored and retry or audit evidence is preserved correctly.
