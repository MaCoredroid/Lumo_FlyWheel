# Focus Candidates

## schema-snapshot-alignment
- Step back and fix the prerequisite invariant that caused the rollback.
- Restores a safe base for any future watermark retry.

## contained-pager-cleanup
- Still contained and still not what caused the failed round.

## cold-start-cache-tuning
- Still abandoned partial work, still not the gating path.

## streaming-watermark-alignment
- Was the previous round's chosen focus.
- Must not be retried blindly after the rollback.
