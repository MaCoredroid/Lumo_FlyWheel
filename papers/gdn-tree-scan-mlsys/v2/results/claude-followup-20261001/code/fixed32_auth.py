"""Per-request headers for replaying against a fixed32 (LumoTree) engine ingress.

Reads the READY.json written by the serve-only hook (secret file, canonical task ids)
and signs each request exactly as the agent proxy does: engine bearer, task key id,
and a fresh fr13-chat-<32 hex> wire id. Uses the repository's own helpers.
"""
import json, os, secrets, sys
sys.path.insert(0, "/home/mark/shared/lumotree-v2exp-20260930/src")
from lumo_flywheel_serving.inference_proxy import (  # noqa: E402
    load_fixed32_ingress_secrets, fixed32_task_key_id, parse_fixed32_task_ids,
    FIXED32_TASK_KEY_HEADER, FIXED32_REQUEST_ID_HEADER)

_state = {}


def headers(index, name, body):
    if not _state:
        ready = json.load(open(os.environ["V2EXP_READY_FILE"]))
        sec = load_fixed32_ingress_secrets(ready["secret_file"])
        tasks = parse_fixed32_task_ids(ready["task_ids"])
        _state.update(bearer=sec.engine_bearer, key=fixed32_task_key_id(tasks[0]))
    return {
        "Authorization": f"Bearer {_state['bearer']}",
        FIXED32_TASK_KEY_HEADER: _state["key"],
        FIXED32_REQUEST_ID_HEADER: f"fr13-chat-{secrets.token_hex(16)}",
    }
