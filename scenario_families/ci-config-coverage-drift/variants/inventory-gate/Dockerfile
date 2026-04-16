FROM python:3.12-bookworm@sha256:1e034bf5ce1ca754be43d1491516da937c0fa03fc29b97a8b6a5e1ce7cb8bbf3

COPY repo/ /workspace/
WORKDIR /workspace

RUN python -m pip install --no-cache-dir pytest
RUN set -eux; \
    set +e; \
    make ci >/tmp/build-smoke.log 2>&1; \
    status=$?; \
    cat /tmp/build-smoke.log; \
    test "$status" -ne 0
