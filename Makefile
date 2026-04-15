PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
LUMOSERVE := $(VENV)/bin/lumoserve

.PHONY: venv bootstrap-runtime download-qwen35-27b serve stop smoke gate1 test

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e .[dev]

bootstrap-runtime:
	$(LUMOSERVE) bootstrap-runtime
	docker pull vllm/vllm-openai@sha256:d9a5c1c1614c959fde8d2a4d68449db184572528a6055afdd0caf1e66fb51504

download-qwen35-27b:
	$(LUMOSERVE) --registry model_registry.yaml download-model qwen3.5-27b --env-file /home/mark/shared/expansion_copy/continue.huggingface.env

serve:
	$(LUMOSERVE) --registry model_registry.yaml serve qwen3.5-27b

stop:
	$(LUMOSERVE) --registry model_registry.yaml stop

smoke:
	$(LUMOSERVE) --registry model_registry.yaml smoke-test qwen3.5-27b --enable-request-logging

gate1:
	VLLM_API_KEY=EMPTY scripts/codex_gate1_qwen35_27b.sh

test:
	$(PYTEST)
