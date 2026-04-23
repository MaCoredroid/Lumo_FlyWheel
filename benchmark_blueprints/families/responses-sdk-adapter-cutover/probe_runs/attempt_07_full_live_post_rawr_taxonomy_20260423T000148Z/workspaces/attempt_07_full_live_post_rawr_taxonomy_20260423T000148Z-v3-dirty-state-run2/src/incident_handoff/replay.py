import json


def serialize_events(events):
    return "\n".join(json.dumps(event, sort_keys=True) for event in events)


def replay_from_serialized(serialized):
    if not serialized:
        return []
    return [json.loads(line) for line in serialized.splitlines() if line.strip()]
