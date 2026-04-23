import json


def serialize_events(events):
    return "\n".join(json.dumps(event, sort_keys=True) for event in events)


def replay_from_serialized(serialized):
    events = []
    for line in serialized.splitlines():
        if line:
            events.append(json.loads(line))
    return events
