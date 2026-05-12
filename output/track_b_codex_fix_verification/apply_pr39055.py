#!/usr/bin/env python3
"""Apply PR #39055 (qwen3 reasoning parser tool-call recovery) to the vllm install in the container."""
from pathlib import Path

path = Path('/usr/local/lib/python3.12/dist-packages/vllm/reasoning/qwen3_reasoning_parser.py')
if not path.is_file():
    raise RuntimeError(f'vLLM qwen3 reasoning parser source missing: {path}')

text = path.read_text(encoding='utf-8')
if '_split_embedded_tool_calls' in text:
    print('[PATCH] PR39055 already present')
else:
    helper_block = (
        '_EMBEDDED_TOOL_CALL_RE = __import__("re").compile(\n'
        '    r"<tool_call>(.*?)</tool_call>|<tool_call>.*$",\n'
        '    __import__("re").DOTALL,\n'
        ')\n\n\n'
    )
    if 'class Qwen3ReasoningParser' not in text:
        raise RuntimeError('Qwen3ReasoningParser not found')
    text = text.replace('class Qwen3ReasoningParser', helper_block + 'class Qwen3ReasoningParser', 1)

    method_block = (
        '    @staticmethod\n'
        '    def _split_embedded_tool_calls(\n'
        '        reasoning,\n'
        '        content,\n'
        '    ):\n'
        '        """Promote tool-call XML blocks out of reasoning into content."""\n'
        '        if (\n'
        '            not reasoning\n'
        '            or "<tool_call>" not in reasoning\n'
        '            or "<function=" not in reasoning\n'
        '        ):\n'
        '            return reasoning, content\n'
        '        extracted_blocks = []\n'
        '        def _collect_or_keep(match):\n'
        '            block = match.group(0)\n'
        '            if "<function=" not in block:\n'
        '                return block\n'
        '            extracted_blocks.append(block.strip())\n'
        '            return ""\n'
        '        remaining_reasoning = _EMBEDDED_TOOL_CALL_RE.sub(_collect_or_keep, reasoning)\n'
        '        remaining_reasoning = remaining_reasoning.strip() or None\n'
        '        if not extracted_blocks:\n'
        '            return reasoning, content\n'
        '        content_parts = ["\\n\\n".join(extracted_blocks)]\n'
        '        if content:\n'
        '            content_parts.append(content)\n'
        '        merged_content = "\\n\\n".join(part for part in content_parts if part) or None\n'
        '        return remaining_reasoning, merged_content\n\n'
    )
    sentinel_def = '    def extract_reasoning(\n'
    if sentinel_def not in text:
        raise RuntimeError('def extract_reasoning insertion point not found')
    text = text.replace(sentinel_def, method_block + sentinel_def, 1)

    old_truncated = '            return model_output, None'
    new_truncated = '            return self._split_embedded_tool_calls(model_output, None)'
    if old_truncated not in text:
        raise RuntimeError('truncated-reasoning return not found')
    text = text.replace(old_truncated, new_truncated, 1)

    old_normal = '        return reasoning, final_content'
    new_normal = '        return self._split_embedded_tool_calls(reasoning, final_content)'
    if old_normal not in text:
        raise RuntimeError('normal return not found')
    text = text.replace(old_normal, new_normal, 1)

    path.write_text(text, encoding='utf-8')
    print('[PATCH] applied PR39055')

import py_compile
py_compile.compile(str(path), doraise=True)
print('[PATCH] file compiles cleanly')
print('[PATCH] new wc -l:', len(text.splitlines()))
