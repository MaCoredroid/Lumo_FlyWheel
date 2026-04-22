import json


def serialize_events(events):
    lines = []
    for event in events:
        lines.append(json.dumps(event, sort_keys=True))
    return "\n".join(lines)


def replay_from_serialized(serialized):
    events = []
    for line in serialized.splitlines():
        if not line.strip():
            continue
        events.append(json.loads(line))
    return events
