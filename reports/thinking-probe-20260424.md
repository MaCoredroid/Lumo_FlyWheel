# Serving Thinking Probe

- capture_date: 2026-04-24T23:25:47.178065Z
- base_url: http://127.0.0.1:8100/v1
- model: qwen3.5-27b
- outcome: row-3
- diagnosis: Thinking fires by default and with explicit override.

## Inputs

### Case A

```json
{
  "input": "Summarize: AI is useful.",
  "max_output_tokens": 2048,
  "model": "qwen3.5-27b"
}
```

### Case B

```json
{
  "extra_body": {
    "chat_template_kwargs": {
      "enable_thinking": true
    }
  },
  "input": "Prove that sqrt(2) is irrational. Show every step.",
  "max_output_tokens": 8192,
  "model": "qwen3.5-27b"
}
```

## Usage Summary

| case | input_tokens | output_tokens | reasoning_tokens | total_tokens |
|---|---:|---:|---:|---:|
| A | 18 | 2048 | 2048 | 2066 |
| B | 24 | 2154 | 2154 | 2178 |
