# Generation-5 lifecycle diagnosis

## Observed sequence

1. Physical B4 was reached on the real exact4 task set.
2. The batched-BV8 post-replay shadow gate passed all 48 layers and restored the
   reference state before serving it.
3. At 23:49:38Z, generation 5 reconciled while a drafter proposal was incomplete;
   the engine had three running requests.
4. The runtime emitted an error ack rather than a boundary snapshot.
5. Authenticated traffic continued for about 55 minutes. Engine and proxy
   ledgers finalized with 151 accepted/completed requests and no active work.
6. The terminal protocol retry inherited the generation-5 error state, so no
   terminal snapshot or work census was materialized.

## Classification

High-confidence sample/proposal-to-flush synchronization race. The flush worker
serializes with the execute lock, but host-side sample/proposal work can be live
outside that lock. Proposal reconciliation correctly fails closed when it sees
the incomplete state, but the snapshot protocol did not wait for the
sample-pending condition first.

The incomplete proposal was transient, not permanently stranded. Later proposal
traffic could not have begun if the singleton had remained populated. This
explains why inference continued while the generation-5 ack remained an error.

## Measurement boundary

The 640/640 counters in the error ack are observations at the failed snapshot,
not final values. No later successful snapshot exists. The empty work-census
file, orchestrator return code 1, terminal flush return code 2, and absent formal
verdict make all timing, TPS, acceptance, and hardware-floor claims invalid.

## Required repair

Wait on the sample condition until no sample/proposal entry is pending, then
reconcile while holding the condition's underlying execution lock. Preserve
fail-closed behavior if the sample failure state is set. Qualify the changed
source with a new real SWE-Verified exact4 B4 byte gate before any B4 timing.
