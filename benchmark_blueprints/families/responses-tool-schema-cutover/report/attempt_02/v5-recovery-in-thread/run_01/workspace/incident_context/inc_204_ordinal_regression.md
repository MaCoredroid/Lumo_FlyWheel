# INC-204

An earlier repair grouped tool results by visible call ordinal instead of `call_id`.
The patch passed a narrow fixture and then rendered the wrong transcript when results
arrived out of order for repeated tool invocations.
