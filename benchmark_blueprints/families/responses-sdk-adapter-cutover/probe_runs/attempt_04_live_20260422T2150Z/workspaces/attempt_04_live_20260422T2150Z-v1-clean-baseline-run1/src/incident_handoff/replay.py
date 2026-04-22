import json


def serialize_events(events):
    return "\n".join(json.dumps(event) for event in events)


def replay_from_serialized(serialized):
    return [json.loads(line) for line in serialized.splitlines() if line]
