"""Abandoned attempt. Do not trust this ordinal-based join shim."""

def attach_results_by_position(calls, results):
    pairs = []
    for index, call in enumerate(calls):
        result = results[index] if index < len(results) else None
        pairs.append((call.get("tool_name"), result))
    return pairs
