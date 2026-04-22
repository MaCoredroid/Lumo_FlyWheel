import json


def serialize_events(events):
    return "\n".join(json.dumps(event, separators=(",", ":")) for event in events)


def replay_from_serialized(serialized):
    events = []
    for line in serialized.splitlines():
        if not line.strip():
            continue
        events.append(json.loads(line))
    return events
