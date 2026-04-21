# syntax=docker/dockerfile:1.6
FROM python:3.12-bookworm@sha256:1e034bf5ce1ca754be43d1491516da937c0fa03fc29b97a8b6a5e1ce7cb8bbf3

RUN pip install --no-cache-dir "pytest==8.3.3" "pyyaml==6.0.2"

WORKDIR /workspace
COPY . /workspace
RUN test -f /workspace/.scenario_variant && test -d /workspace/proposals

CMD ["bash"]
