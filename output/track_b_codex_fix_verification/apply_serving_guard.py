from pathlib import Path

path = Path('/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/responses/serving.py')
text = path.read_text(encoding='utf-8')

if '# Track B PR39055 workaround' in text:
    print('[PATCH] streaming-event guard already present')
else:
    sentinel_init = '        first_delta_sent = False\n        previous_delta_messages: list[DeltaMessage] = []'
    if sentinel_init not in text:
        raise RuntimeError('sentinel for init not found')
    text = text.replace(
        sentinel_init,
        sentinel_init + '\n        # Track B PR39055 workaround: init tool-call vars to prevent UnboundLocalError when arguments arrive before name\n        current_tool_call_name = None\n        current_tool_call_id = None',
        1,
    )

    old_use = '            tool_call_arguments = "".join(parts)\n            if tool_call_arguments:'
    new_use = '            tool_call_arguments = "".join(parts)\n            if tool_call_arguments and current_tool_call_name:'
    if old_use not in text:
        raise RuntimeError('use-site sentinel not found')
    text = text.replace(old_use, new_use, 1)

    path.write_text(text, encoding='utf-8')
    print('[PATCH] applied streaming-event guard for UnboundLocalError')

import py_compile
py_compile.compile(str(path), doraise=True)
print('[PATCH] file compiles cleanly')
