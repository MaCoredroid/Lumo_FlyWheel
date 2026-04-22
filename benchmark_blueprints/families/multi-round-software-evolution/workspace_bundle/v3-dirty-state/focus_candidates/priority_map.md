# Focus Candidates

## schema-snapshot-alignment
- Fix the mismatch between snapshot writer ordering and restore ordering.
- Still the prerequisite for stable follow-on work.

## contained-pager-cleanup
- Already contained; still wrong to reopen first.

## cold-start-cache-tuning
- There is a half-finished patch, but it only tunes cold-start hydration.
- The patch does not remove replay divergence.

## streaming-watermark-alignment
- Still downstream of stable snapshot restore.
