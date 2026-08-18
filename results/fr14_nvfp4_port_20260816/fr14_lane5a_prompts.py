#!/usr/bin/env python3
"""FR14 lane 5A generation-probe prompt set: SWE-flavoured, degeneration-baited.

Emits the prompt list as JSON on stdout so the traces are reproducible from the
tree rather than from a shell heredoc.

DESIGN.  Mark's condition on this lane is that generation traces be READ for
degeneration at every stage that produces text.  A probe can only find the
signatures it gives the model room to produce, so each prompt is chosen for a
specific failure mode a damaged verifier head would surface:

* ``greedy_*``  -- temperature 0.  A quantised head shifts logits; greedy is the
  regime where a shift becomes a DIFFERENT TOKEN with no sampling noise to hide
  behind, and it is the regime the offline argmax-flip analysis is a model of.
* ``sampled_*`` -- the campaign's own serving parameters
  (temperature 0.6 / top_p 0.95 / top_k 20).
* ``repetition_trap`` -- an enumeration task with a high max_tokens.  Repetition
  loops are the classic low-precision-head signature and they need LENGTH to
  appear; a 256-token probe cannot see one.
* ``tool_call``      -- malformed tool calls are a named signature, and the tool
  schema forces exact-token JSON where a single flipped token is visible.
* ``long_code``      -- mid-word breaks and brace/indentation corruption show up
  in code far more legibly than in prose.
* ``exactness``      -- an instruction requiring a verbatim string plus an
  arithmetic fact: two independently checkable assertions in one short trace.
"""

import json

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the project's test suite, optionally filtered.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "test file or dir"},
                    "keyword": {"type": "string", "description": "-k expression"},
                    "verbose": {"type": "boolean"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
]

BUGGY = '''\
def merge_intervals(intervals):
    """Merge overlapping [start, end] intervals. Returns a new sorted list."""
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    out = [intervals[0]]
    for start, end in intervals[1:]:
        last = out[-1]
        if start < last[1]:
            last[1] = max(last[1], end)
        else:
            out.append([start, end])
    return out
'''

TRACEBACK = '''\
Traceback (most recent call last):
  File "/srv/app/pipeline/loader.py", line 214, in load_shard
    tensors[name] = f.get_tensor(name).to(self.device, non_blocking=True)
  File "/srv/app/pipeline/loader.py", line 188, in device
    return self._devices[self._rank % len(self._devices)]
ZeroDivisionError: integer modulo by zero
'''

PROMPTS = [
    {
        "id": "greedy_bugfix",
        "regime": "greedy",
        "max_tokens": 1536,
        "sampling": {"temperature": 0.0},
        "messages": [
            {
                "role": "user",
                "content": (
                    "This function has a boundary bug and a mutation bug. Identify both, "
                    "then give the corrected function.\n\n```python\n" + BUGGY + "```"
                ),
            }
        ],
    },
    {
        "id": "greedy_traceback",
        "regime": "greedy",
        "max_tokens": 1536,
        "sampling": {"temperature": 0.0},
        "messages": [
            {
                "role": "user",
                "content": (
                    "Diagnose this failure and propose a minimal patch as a unified diff.\n\n"
                    + TRACEBACK
                ),
            }
        ],
    },
    {
        "id": "greedy_exactness",
        "regime": "greedy",
        "max_tokens": 512,
        "sampling": {"temperature": 0.0},
        "messages": [
            {
                "role": "user",
                "content": (
                    "Reply with exactly the line: FR14 lane5A NVFP4 head alive. "
                    "Then on a new line state the value of 17*23, and on a third "
                    "line the value of 2**16."
                ),
            }
        ],
    },
    {
        "id": "greedy_tool_call",
        "regime": "greedy",
        "max_tokens": 768,
        "sampling": {"temperature": 0.0},
        "tools": TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "The test file tests/test_loader.py is failing on the shard-rank "
                    "tests. Run only those tests, verbosely, then read the loader "
                    "around the failing line."
                ),
            }
        ],
    },
    {
        "id": "sampled_long_code",
        "regime": "sampled",
        "max_tokens": 2048,
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a Python LRU cache with a fixed capacity, O(1) get and put, "
                    "and a thread-safe variant. Include pytest tests covering eviction "
                    "order, capacity 1, and concurrent access. Then explain the "
                    "invariant that makes get() O(1)."
                ),
            }
        ],
    },
    {
        "id": "sampled_repetition_trap",
        "regime": "sampled",
        "max_tokens": 2560,
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        "messages": [
            {
                "role": "user",
                "content": (
                    "List 40 distinct causes of flaky tests in a large Python "
                    "monorepo. Number them 1 to 40. For each, give the cause on one "
                    "line and a one-line detection hint. Do not repeat a cause."
                ),
            }
        ],
    },
]

# --------------------------------------------------------------------------
# PHASE 2.  Phase 1's tool prompt 400'd: the probe had booted without the
# production serve line's --enable-auto-tool-choice / --tool-call-parser
# qwen3_xml / --reasoning-parser qwen3, so vLLM refused the request before the
# model ever saw it.  That is a probe defect, not a model result, and it left
# Mark's named "malformed tool call" signature untested.  Phase 2 re-runs the
# tool prompts under the PRODUCTION parser flags.
#
# It also raises max_tokens on one code task to 6144.  Every phase-1 long trace
# finished with reason `length` INSIDE the reasoning block, so the probe read
# the model's deliberation but never its ANSWER -- and the answer is where
# truncated code, unbalanced delimiters and mid-word breaks would show.
PROMPTS_PHASE2 = [
    {
        "id": "p2_greedy_tool_call",
        "regime": "greedy",
        "max_tokens": 1536,
        "sampling": {"temperature": 0.0},
        "tools": TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "The test file tests/test_loader.py is failing on the shard-rank "
                    "tests. Run only those tests, verbosely, then read the loader "
                    "around the failing line."
                ),
            }
        ],
    },
    {
        "id": "p2_sampled_tool_call",
        "regime": "sampled",
        "max_tokens": 1536,
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        "tools": TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Before changing anything, read pipeline/loader.py lines 180-220 "
                    "and then run the loader tests with the keyword 'rank'."
                ),
            }
        ],
    },
    {
        "id": "p2_greedy_bugfix_to_completion",
        "regime": "greedy",
        "max_tokens": 6144,
        "sampling": {"temperature": 0.0},
        "messages": [
            {
                "role": "user",
                "content": (
                    "This function has a boundary bug and a mutation bug. Identify both, "
                    "then give the corrected function.\n\n```python\n" + BUGGY + "```"
                ),
            }
        ],
    },
]

if __name__ == "__main__":
    import sys

    which = PROMPTS_PHASE2 if "--phase2" in sys.argv else PROMPTS
    print(json.dumps(which, indent=1))
