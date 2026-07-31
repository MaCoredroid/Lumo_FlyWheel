#!/usr/bin/env python3
"""Strict host client for the FR13 fixed-32 runtime flush protocol.

The runtime publishes a generation-0 ``ready`` record to the ack path. The
host then serializes every ``snapshot`` or ``final`` request through one
process-global mutex, atomically replaces the request file, signals EngineCore
with SIGUSR2, and accepts only the matching atomic ack.

The ready and response ack records use the same exact shape::

    {
      "schema": "fr13-fixed32-flush-ack-v1",
      "mode": "tail6_fixed32",
      "producer_pid": 123,
      "generation": 0,
      "nonce": "0000...0000",
      "action": "ready",
      "status": "ok",
      "counters": {}
    }

Generation zero must use ``action=ready`` and ``READY_NONCE``. Later
generations must echo the corresponding request's mode, PID, generation,
nonce, and action. Any uncertain transaction poisons that endpoint for the
remainder of the host process; silently retrying an ambiguous generation
would make a late ack indistinguishable from the retry.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol


REQUEST_SCHEMA: Final = "fr13-fixed32-flush-request-v1"
ACK_SCHEMA: Final = "fr13-fixed32-flush-ack-v1"
RESULT_SCHEMA: Final = "fr13-fixed32-flush-client-result-v1"
SELF_TEST_SCHEMA: Final = "fr13-fixed32-flush-self-test-v1"

READY_ACTION: Final = "ready"
READY_NONCE: Final = "0" * 64
FLUSH_ACTIONS: Final = frozenset({"snapshot", "final"})
FIXED32_MODES: Final = frozenset(
    {"tail6_fixed32", "hydra27_fixed32", "hydra31_fixed32"}
)

REQUEST_KEYS: Final = frozenset(
    {
        "schema",
        "mode",
        "producer_pid",
        "prev_generation",
        "generation",
        "nonce",
        "action",
    }
)
ACK_KEYS: Final = frozenset(
    {
        "schema",
        "mode",
        "producer_pid",
        "generation",
        "nonce",
        "action",
        "status",
        "counters",
    }
)
NONCE_RE: Final = re.compile(r"^[0-9a-f]{64}$")
CONTAINER_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
MAX_ACK_BYTES: Final = 1024 * 1024


class FlushProtocolError(RuntimeError):
    """Base class for a fail-closed flush failure."""


class FlushConfigurationError(FlushProtocolError):
    """The client configuration is invalid."""


class FlushAckError(FlushProtocolError):
    """An ack is malformed or does not bind to the request."""


class FlushLivenessError(FlushProtocolError):
    """The container or EngineCore producer is dead."""


class FlushSignalError(FlushProtocolError):
    """SIGUSR2 delivery failed."""


class FlushTimeoutError(FlushProtocolError):
    """The matching atomic ack did not arrive before the deadline."""


class FlushRuntimeError(FlushProtocolError):
    """The runtime acknowledged the generation with a non-ok status."""


class FlushStateError(FlushProtocolError):
    """The process-global endpoint state is terminal or ambiguous."""


@dataclass(frozen=True)
class CommandResult:
    """Minimal result returned by an injectable command runner."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandRunner(Protocol):
    """Run one argv vector with a hard command timeout."""

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float,
    ) -> CommandResult:
        """Return command status and captured text."""


@dataclass(frozen=True)
class FlushAck:
    """Validated runtime acknowledgement."""

    mode: str
    producer_pid: int
    generation: int
    nonce: str
    action: str
    counters: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ACK_SCHEMA,
            "mode": self.mode,
            "producer_pid": self.producer_pid,
            "generation": self.generation,
            "nonce": self.nonce,
            "action": self.action,
            "status": "ok",
            "counters": self.counters,
        }


@dataclass
class _EndpointState:
    generation: int
    nonce: str
    action: str
    terminal: bool = False
    poisoned_reason: str | None = None


EndpointKey = tuple[str, str, str, int, str]
Clock = Callable[[], float]
Sleeper = Callable[[float], None]
NonceFactory = Callable[[], str]

_PROCESS_FLUSH_LOCK = threading.Lock()
_ENDPOINT_STATES: dict[EndpointKey, _EndpointState] = {}


def _default_command_runner(
    argv: Sequence[str],
    *,
    timeout_s: float,
) -> CommandResult:
    try:
        completed = subprocess.run(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as error:
        raise FlushProtocolError(
            f"command timed out after {timeout_s:.3f}s: {list(argv)!r}"
        ) from error
    except OSError as error:
        raise FlushProtocolError(
            f"cannot execute command {list(argv)!r}: {error}"
        ) from error
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _is_exact_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _duplicate_checked_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FlushAckError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise FlushAckError(f"non-finite JSON constant {value!r} is forbidden")


def _read_json_object(path: Path) -> dict[str, object]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise FlushAckError(f"cannot open ack {path}: {error}") from error
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise FlushAckError(f"ack is not a regular file: {path}")
        if file_stat.st_size > MAX_ACK_BYTES:
            raise FlushAckError(f"ack exceeds {MAX_ACK_BYTES} bytes: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(MAX_ACK_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(payload) > MAX_ACK_BYTES:
        raise FlushAckError(f"ack exceeds {MAX_ACK_BYTES} bytes: {path}")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise FlushAckError(f"ack is not UTF-8: {path}: {error}") from error
    try:
        raw = json.loads(
            text,
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=_reject_json_constant,
        )
    except FlushAckError:
        raise
    except json.JSONDecodeError as error:
        raise FlushAckError(f"ack is malformed JSON: {path}: {error}") from error
    if not isinstance(raw, dict):
        raise FlushAckError(f"ack root must be an object: {path}")
    return raw


def _validate_exact_keys(
    record: Mapping[str, object],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    actual = frozenset(record)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise FlushAckError(f"{label} keys mismatch: missing={missing} extra={extra}")


def _validate_nonce(nonce: object, *, label: str) -> str:
    if not isinstance(nonce, str) or NONCE_RE.fullmatch(nonce) is None:
        raise FlushAckError(f"{label} must be 64 lowercase hex characters")
    return nonce


def _validated_ack_shape(record: Mapping[str, object]) -> None:
    _validate_exact_keys(record, ACK_KEYS, label="ack")
    if record["schema"] != ACK_SCHEMA:
        raise FlushAckError(
            f"ack schema mismatch: expected {ACK_SCHEMA!r}, got {record['schema']!r}"
        )
    if not isinstance(record["mode"], str):
        raise FlushAckError("ack mode must be a string")
    if not _is_exact_int(record["producer_pid"]) or record["producer_pid"] <= 0:
        raise FlushAckError("ack producer_pid must be a positive integer")
    if not _is_exact_int(record["generation"]) or record["generation"] < 0:
        raise FlushAckError("ack generation must be a nonnegative integer")
    _validate_nonce(record["nonce"], label="ack nonce")
    if not isinstance(record["action"], str):
        raise FlushAckError("ack action must be a string")
    if not isinstance(record["status"], str):
        raise FlushAckError("ack status must be a string")
    if not isinstance(record["counters"], dict):
        raise FlushAckError("ack counters must be an object")


def _validate_common_identity(
    record: Mapping[str, object],
    *,
    mode: str,
    producer_pid: int,
) -> None:
    _validated_ack_shape(record)
    if record["mode"] != mode:
        raise FlushAckError(
            f"ack mode mismatch: expected {mode!r}, got {record['mode']!r}"
        )
    if record["producer_pid"] != producer_pid:
        raise FlushAckError(
            "ack producer_pid mismatch: "
            f"expected {producer_pid}, got {record['producer_pid']!r}"
        )


def _validated_ready_ack(
    record: Mapping[str, object],
    *,
    mode: str,
    producer_pid: int,
) -> FlushAck:
    _validate_common_identity(record, mode=mode, producer_pid=producer_pid)
    if record["generation"] != 0:
        raise FlushAckError(
            f"initial ack is not generation zero: {record['generation']!r}"
        )
    if record["nonce"] != READY_NONCE:
        raise FlushAckError("generation-zero ready ack has the wrong nonce")
    if record["action"] != READY_ACTION:
        raise FlushAckError("generation-zero ack action must be 'ready'")
    if record["status"] != "ok":
        raise FlushRuntimeError(
            f"runtime ready ack status is {record['status']!r}, not 'ok'"
        )
    return FlushAck(
        mode=mode,
        producer_pid=producer_pid,
        generation=0,
        nonce=READY_NONCE,
        action=READY_ACTION,
        counters=dict(record["counters"]),
    )


def _validated_current_ack(
    record: Mapping[str, object],
    *,
    mode: str,
    producer_pid: int,
) -> FlushAck:
    """Validate a ready or successful flush ack for strict process resume."""

    _validate_common_identity(record, mode=mode, producer_pid=producer_pid)
    if record["status"] != "ok":
        raise FlushRuntimeError(
            f"runtime current ack status is {record['status']!r}, not 'ok'"
        )
    generation = record["generation"]
    nonce = record["nonce"]
    action = record["action"]
    if generation == 0:
        return _validated_ready_ack(
            record,
            mode=mode,
            producer_pid=producer_pid,
        )
    if action not in FLUSH_ACTIONS:
        raise FlushAckError(
            "positive-generation ack action must be 'snapshot' or 'final'"
        )
    if nonce == READY_NONCE:
        raise FlushAckError("positive-generation ack uses the reserved ready nonce")
    return FlushAck(
        mode=mode,
        producer_pid=producer_pid,
        generation=generation,
        nonce=nonce,
        action=action,
        counters=dict(record["counters"]),
    )


def _atomic_write_json(path: Path, record: Mapping[str, object]) -> None:
    if path.is_symlink():
        raise FlushConfigurationError(f"refusing symlink request path: {path}")
    parent = path.parent
    if not parent.is_dir():
        raise FlushConfigurationError(
            f"request parent directory does not exist: {parent}"
        )
    content = (
        json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
        temporary_path = None
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_flags |= getattr(os, "O_CLOEXEC", 0)
        directory_fd = os.open(parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as error:
        raise FlushProtocolError(
            f"cannot atomically write flush request {path}: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class Fixed32FlushClient:
    """Mutex-protected client for one EngineCore flush endpoint."""

    def __init__(
        self,
        *,
        container: str,
        producer_pid: int,
        mode: str,
        request_path: str | Path,
        ack_path: str | Path,
        timeout_s: float = 30.0,
        poll_interval_s: float = 0.02,
        liveness_interval_s: float = 0.25,
        command_timeout_s: float = 5.0,
        command_runner: CommandRunner | None = None,
        clock: Clock = time.monotonic,
        sleeper: Sleeper = time.sleep,
        nonce_factory: NonceFactory = lambda: secrets.token_hex(32),
    ) -> None:
        if not isinstance(container, str) or CONTAINER_RE.fullmatch(container) is None:
            raise FlushConfigurationError(
                "container must match [A-Za-z0-9][A-Za-z0-9_.-]*"
            )
        if not _is_exact_int(producer_pid) or producer_pid <= 0:
            raise FlushConfigurationError("producer_pid must be a positive integer")
        if mode not in FIXED32_MODES:
            raise FlushConfigurationError(
                f"mode must be one of {sorted(FIXED32_MODES)}, got {mode!r}"
            )
        for label, value in (
            ("timeout_s", timeout_s),
            ("poll_interval_s", poll_interval_s),
            ("liveness_interval_s", liveness_interval_s),
            ("command_timeout_s", command_timeout_s),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise FlushConfigurationError(
                    f"{label} must be a positive finite number"
                )

        request = Path(request_path).expanduser()
        ack = Path(ack_path).expanduser()
        if not request.is_absolute():
            request = Path.cwd() / request
        if not ack.is_absolute():
            ack = Path.cwd() / ack
        try:
            request_parent = request.parent.resolve(strict=True)
            ack_parent = ack.parent.resolve(strict=True)
        except OSError as error:
            raise FlushConfigurationError(
                f"flush sidecar parent does not exist: {error}"
            ) from error
        if not request_parent.is_dir() or not ack_parent.is_dir():
            raise FlushConfigurationError("flush sidecar parents must be directories")
        self.request_path = request_parent / request.name
        self.ack_path = ack_parent / ack.name
        if self.request_path == self.ack_path:
            raise FlushConfigurationError("request_path and ack_path must be different")

        self.container = container
        self.producer_pid = producer_pid
        self.mode = mode
        self.timeout_s = float(timeout_s)
        self.poll_interval_s = float(poll_interval_s)
        self.liveness_interval_s = float(liveness_interval_s)
        self.command_timeout_s = float(command_timeout_s)
        self._command_runner = command_runner or _default_command_runner
        self._clock = clock
        self._sleeper = sleeper
        self._nonce_factory = nonce_factory
        self._key: EndpointKey = (
            str(self.request_path),
            str(self.ack_path),
            container,
            producer_pid,
            mode,
        )

    def _remaining(self, deadline: float) -> float:
        return deadline - self._clock()

    def _run_command(
        self,
        argv: Sequence[str],
        *,
        deadline: float,
        label: str,
    ) -> CommandResult:
        remaining = self._remaining(deadline)
        if remaining <= 0.0:
            raise FlushTimeoutError(f"timeout before {label}")
        timeout = min(self.command_timeout_s, remaining)
        try:
            result = self._command_runner(tuple(argv), timeout_s=timeout)
        except FlushProtocolError:
            raise
        except Exception as error:
            raise FlushProtocolError(
                f"{label} command runner failed: {type(error).__name__}: {error}"
            ) from error
        if (
            not hasattr(result, "returncode")
            or not _is_exact_int(result.returncode)
            or not isinstance(getattr(result, "stdout", None), str)
            or not isinstance(getattr(result, "stderr", None), str)
        ):
            raise FlushProtocolError(
                f"{label} command runner returned an invalid result"
            )
        return CommandResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def _check_liveness(self, *, deadline: float) -> None:
        container_result = self._run_command(
            (
                "docker",
                "inspect",
                "--format={{.State.Running}}",
                self.container,
            ),
            deadline=deadline,
            label="container liveness",
        )
        if (
            container_result.returncode != 0
            or container_result.stdout.strip().lower() != "true"
        ):
            detail = (
                container_result.stderr.strip()
                or container_result.stdout.strip()
                or f"rc={container_result.returncode}"
            )
            raise FlushLivenessError(
                f"container {self.container!r} is not running: {detail}"
            )

        process_result = self._run_command(
            (
                "docker",
                "exec",
                self.container,
                "kill",
                "-0",
                str(self.producer_pid),
            ),
            deadline=deadline,
            label="producer liveness",
        )
        if process_result.returncode != 0:
            detail = (
                process_result.stderr.strip()
                or process_result.stdout.strip()
                or f"rc={process_result.returncode}"
            )
            raise FlushLivenessError(
                f"producer PID {self.producer_pid} is not alive "
                f"in {self.container!r}: {detail}"
            )

    def _initialize_state(
        self,
        *,
        deadline: float,
        resume_current: bool = False,
    ) -> _EndpointState:
        existing = _ENDPOINT_STATES.get(self._key)
        if existing is not None:
            return existing

        next_liveness = self._clock()
        expected_ack = "current ack" if resume_current else "generation-zero ready ack"
        while True:
            now = self._clock()
            if now >= next_liveness:
                self._check_liveness(deadline=deadline)
                next_liveness = now + self.liveness_interval_s
            try:
                record = _read_json_object(self.ack_path)
            except FileNotFoundError:
                record = None
            if record is not None:
                if resume_current:
                    current = _validated_current_ack(
                        record,
                        mode=self.mode,
                        producer_pid=self.producer_pid,
                    )
                else:
                    current = _validated_ready_ack(
                        record,
                        mode=self.mode,
                        producer_pid=self.producer_pid,
                    )
                state = _EndpointState(
                    generation=current.generation,
                    nonce=current.nonce,
                    action=current.action,
                    terminal=current.action == "final",
                )
                _ENDPOINT_STATES[self._key] = state
                return state
            remaining = self._remaining(deadline)
            if remaining <= 0.0:
                raise FlushTimeoutError(
                    f"timeout waiting for {expected_ack}: {self.ack_path}"
                )
            self._sleeper(min(self.poll_interval_s, remaining))

    def connect(self) -> FlushAck:
        """Validate the generation-0 ready ack and initialize global state."""

        with _PROCESS_FLUSH_LOCK:
            deadline = self._clock() + self.timeout_s
            state = self._initialize_state(deadline=deadline)
            if state.poisoned_reason is not None:
                raise FlushStateError(
                    f"flush endpoint is poisoned: {state.poisoned_reason}"
                )
            if state.generation != 0 or state.action != READY_ACTION:
                raise FlushStateError(
                    "flush endpoint was already used; connect() only returns "
                    "the generation-zero ready record"
                )
            record = _read_json_object(self.ack_path)
            return _validated_ready_ack(
                record,
                mode=self.mode,
                producer_pid=self.producer_pid,
            )

    def connect_current(self) -> FlushAck:
        """Resume from the exact current ok ack in a fresh host process.

        This is intended for a teardown CLI process that follows task-runner
        snapshots. It accepts generation zero or a prior snapshot, while a
        prior final ack is terminal and fails closed.
        """

        with _PROCESS_FLUSH_LOCK:
            deadline = self._clock() + self.timeout_s
            state = self._initialize_state(
                deadline=deadline,
                resume_current=True,
            )
            if state.poisoned_reason is not None:
                raise FlushStateError(
                    f"flush endpoint is poisoned: {state.poisoned_reason}"
                )
            if state.terminal:
                raise FlushStateError(
                    "flush endpoint is terminal after a successful final ack"
                )
            record = _read_json_object(self.ack_path)
            current = _validated_current_ack(
                record,
                mode=self.mode,
                producer_pid=self.producer_pid,
            )
            if (
                current.generation != state.generation
                or current.nonce != state.nonce
                or current.action != state.action
            ):
                raise FlushAckError(
                    "current ack changed while seeding process-global state"
                )
            return current

    def _wait_for_ack(
        self,
        *,
        state: _EndpointState,
        generation: int,
        nonce: str,
        action: str,
        deadline: float,
    ) -> FlushAck:
        next_liveness = self._clock()
        last_stale = "ack is absent"
        while True:
            now = self._clock()
            if now >= next_liveness:
                self._check_liveness(deadline=deadline)
                next_liveness = now + self.liveness_interval_s
            try:
                record = _read_json_object(self.ack_path)
            except FileNotFoundError:
                record = None
            if record is not None:
                _validate_common_identity(
                    record,
                    mode=self.mode,
                    producer_pid=self.producer_pid,
                )
                observed_generation = record["generation"]
                if observed_generation < state.generation:
                    raise FlushAckError(
                        "ack generation regressed below committed state: "
                        f"{observed_generation} < {state.generation}"
                    )
                if observed_generation == state.generation:
                    if (
                        record["nonce"] != state.nonce
                        or record["action"] != state.action
                        or record["status"] != "ok"
                    ):
                        raise FlushAckError(
                            "stale ack does not match the last committed generation"
                        )
                    last_stale = (
                        f"stale generation {observed_generation} ({record['action']})"
                    )
                elif observed_generation == generation:
                    if record["nonce"] != nonce:
                        raise FlushAckError(
                            "ack nonce does not match the pending request"
                        )
                    if record["action"] != action:
                        raise FlushAckError(
                            "ack action does not match the pending request"
                        )
                    if record["status"] != "ok":
                        raise FlushRuntimeError(
                            "runtime returned non-ok flush status "
                            f"{record['status']!r} for generation {generation}"
                        )
                    return FlushAck(
                        mode=self.mode,
                        producer_pid=self.producer_pid,
                        generation=generation,
                        nonce=nonce,
                        action=action,
                        counters=dict(record["counters"]),
                    )
                else:
                    raise FlushAckError(
                        "ack generation is ahead of the pending request: "
                        f"{observed_generation} > {generation}"
                    )
            remaining = self._remaining(deadline)
            if remaining <= 0.0:
                raise FlushTimeoutError(
                    "timeout waiting for matching atomic ack "
                    f"generation={generation}: {last_stale}"
                )
            self._sleeper(min(self.poll_interval_s, remaining))

    def flush(self, action: str, *, resume_current: bool = False) -> FlushAck:
        """Issue one serialized snapshot or final flush."""

        if action not in FLUSH_ACTIONS:
            raise FlushConfigurationError(
                f"action must be one of {sorted(FLUSH_ACTIONS)}, got {action!r}"
            )
        with _PROCESS_FLUSH_LOCK:
            deadline = self._clock() + self.timeout_s
            state = self._initialize_state(
                deadline=deadline,
                resume_current=resume_current,
            )
            if state.poisoned_reason is not None:
                raise FlushStateError(
                    f"flush endpoint is poisoned: {state.poisoned_reason}"
                )
            if state.terminal:
                raise FlushStateError(
                    "flush endpoint is terminal after a successful final ack"
                )

            generation = state.generation + 1
            nonce = self._nonce_factory()
            try:
                nonce = _validate_nonce(nonce, label="generated nonce")
            except FlushAckError as error:
                raise FlushConfigurationError(str(error)) from error
            if nonce == READY_NONCE:
                raise FlushConfigurationError(
                    "nonce factory returned the reserved ready nonce"
                )
            request = {
                "schema": REQUEST_SCHEMA,
                "mode": self.mode,
                "producer_pid": self.producer_pid,
                "prev_generation": state.generation,
                "generation": generation,
                "nonce": nonce,
                "action": action,
            }
            if frozenset(request) != REQUEST_KEYS:
                raise AssertionError("internal request shape drift")

            request_written = False
            try:
                self._check_liveness(deadline=deadline)
                _atomic_write_json(self.request_path, request)
                request_written = True
                signal_result = self._run_command(
                    (
                        "docker",
                        "exec",
                        self.container,
                        "kill",
                        "-USR2",
                        str(self.producer_pid),
                    ),
                    deadline=deadline,
                    label="SIGUSR2 delivery",
                )
                if signal_result.returncode != 0:
                    detail = (
                        signal_result.stderr.strip()
                        or signal_result.stdout.strip()
                        or f"rc={signal_result.returncode}"
                    )
                    raise FlushSignalError(
                        "SIGUSR2 delivery failed for "
                        f"{self.container!r} PID {self.producer_pid}: {detail}"
                    )
                ack = self._wait_for_ack(
                    state=state,
                    generation=generation,
                    nonce=nonce,
                    action=action,
                    deadline=deadline,
                )
            except Exception as error:
                if request_written:
                    state.poisoned_reason = (
                        f"generation {generation} outcome is ambiguous after "
                        f"{type(error).__name__}: {error}"
                    )
                raise

            state.generation = generation
            state.nonce = nonce
            state.action = action
            state.terminal = action == "final"
            return ack

    def snapshot(self) -> FlushAck:
        """Issue a nonterminal snapshot flush."""

        return self.flush("snapshot")

    def finalize(self) -> FlushAck:
        """Issue the terminal flush."""

        return self.flush("final")


def request_flush(
    *,
    container: str,
    producer_pid: int,
    mode: str,
    request_path: str | Path,
    ack_path: str | Path,
    action: str,
    timeout_s: float = 30.0,
    poll_interval_s: float = 0.02,
    liveness_interval_s: float = 0.25,
    command_timeout_s: float = 5.0,
    command_runner: CommandRunner | None = None,
    resume_current: bool = False,
) -> FlushAck:
    """Convenience API backed by the process-global endpoint state."""

    client = Fixed32FlushClient(
        container=container,
        producer_pid=producer_pid,
        mode=mode,
        request_path=request_path,
        ack_path=ack_path,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
        liveness_interval_s=liveness_interval_s,
        command_timeout_s=command_timeout_s,
        command_runner=command_runner,
    )
    return client.flush(action, resume_current=resume_current)


def ready_ack(
    *,
    mode: str,
    producer_pid: int,
    counters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the exact generation-0 record expected from the runtime."""

    if mode not in FIXED32_MODES:
        raise FlushConfigurationError(f"unsupported fixed32 mode: {mode!r}")
    if not _is_exact_int(producer_pid) or producer_pid <= 0:
        raise FlushConfigurationError("producer_pid must be a positive integer")
    return {
        "schema": ACK_SCHEMA,
        "mode": mode,
        "producer_pid": producer_pid,
        "generation": 0,
        "nonce": READY_NONCE,
        "action": READY_ACTION,
        "status": "ok",
        "counters": dict(counters or {}),
    }


def response_ack(
    request: Mapping[str, object],
    *,
    counters: Mapping[str, Any] | None = None,
    status: str = "ok",
) -> dict[str, Any]:
    """Build an exact response fixture from a validated request object."""

    _validate_exact_keys(request, REQUEST_KEYS, label="request")
    if request["schema"] != REQUEST_SCHEMA:
        raise FlushConfigurationError("request schema mismatch")
    return {
        "schema": ACK_SCHEMA,
        "mode": request["mode"],
        "producer_pid": request["producer_pid"],
        "generation": request["generation"],
        "nonce": request["nonce"],
        "action": request["action"],
        "status": status,
        "counters": dict(counters or {}),
    }


def _write_test_json(path: Path, record: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class _FakeRuntime:
    def __init__(
        self,
        request_path: Path,
        ack_path: Path,
        *,
        mutate_ack: Callable[[dict[str, Any]], None] | None = None,
        no_ack: bool = False,
        container_alive: bool = True,
        process_alive: bool = True,
        signal_rc: int = 0,
        signal_delay_s: float = 0.0,
    ) -> None:
        self.request_path = request_path
        self.ack_path = ack_path
        self.mutate_ack = mutate_ack
        self.no_ack = no_ack
        self.container_alive = container_alive
        self.process_alive = process_alive
        self.signal_rc = signal_rc
        self.signal_delay_s = signal_delay_s
        self.requests: list[dict[str, Any]] = []
        self.active_signals = 0
        self.max_active_signals = 0
        self._lock = threading.Lock()

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float,
    ) -> CommandResult:
        del timeout_s
        command = tuple(argv)
        if command[:3] == (
            "docker",
            "inspect",
            "--format={{.State.Running}}",
        ):
            if self.container_alive:
                return CommandResult(0, "true\n", "")
            return CommandResult(1, "", "No such container")
        if len(command) >= 5 and command[0:2] == ("docker", "exec"):
            signal = command[-2]
            if signal == "-0":
                if self.process_alive:
                    return CommandResult(0, "", "")
                return CommandResult(1, "", "No such process")
            if signal == "-USR2":
                if self.signal_rc != 0:
                    return CommandResult(self.signal_rc, "", "signal rejected")
                with self._lock:
                    self.active_signals += 1
                    self.max_active_signals = max(
                        self.max_active_signals,
                        self.active_signals,
                    )
                try:
                    if self.signal_delay_s:
                        time.sleep(self.signal_delay_s)
                    request = json.loads(self.request_path.read_text(encoding="utf-8"))
                    self.requests.append(request)
                    if not self.no_ack:
                        ack = response_ack(
                            request,
                            counters={"events": request["generation"]},
                        )
                        if self.mutate_ack is not None:
                            self.mutate_ack(ack)
                        _atomic_write_json(self.ack_path, ack)
                finally:
                    with self._lock:
                        self.active_signals -= 1
                return CommandResult(0, "", "")
        return CommandResult(127, "", f"unexpected command: {command!r}")


def _expect_error(
    error_type: type[BaseException],
    function: Callable[[], object],
    *,
    contains: str | None = None,
) -> None:
    try:
        function()
    except error_type as error:
        if contains is not None and contains not in str(error):
            raise AssertionError(
                f"expected {contains!r} in {type(error).__name__}: {error}"
            ) from error
    else:
        raise AssertionError(f"expected {error_type.__name__}")


def _new_test_client(
    directory: Path,
    runtime: _FakeRuntime,
    *,
    mode: str = "tail6_fixed32",
    producer_pid: int = 4312,
    timeout_s: float = 0.08,
) -> Fixed32FlushClient:
    return Fixed32FlushClient(
        container="fr13-test",
        producer_pid=producer_pid,
        mode=mode,
        request_path=directory / "request.json",
        ack_path=directory / "ack.json",
        timeout_s=timeout_s,
        poll_interval_s=0.002,
        liveness_interval_s=0.01,
        command_timeout_s=0.02,
        command_runner=runtime,
    )


def _self_test_success_and_terminal() -> None:
    with tempfile.TemporaryDirectory(prefix="fr13-flush-success-") as raw:
        directory = Path(raw)
        request_path = directory / "request.json"
        ack_path = directory / "ack.json"
        _write_test_json(
            ack_path,
            ready_ack(mode="tail6_fixed32", producer_pid=4312),
        )
        runtime = _FakeRuntime(request_path, ack_path)
        client = _new_test_client(directory, runtime)
        first = client.snapshot()
        second = client.finalize()
        if first.generation != 1 or second.generation != 2:
            raise AssertionError("flush generations did not advance exactly")
        if [row["prev_generation"] for row in runtime.requests] != [0, 1]:
            raise AssertionError("request prev_generation chain is wrong")
        if [row["generation"] for row in runtime.requests] != [1, 2]:
            raise AssertionError("request generation chain is wrong")
        if runtime.requests[0]["nonce"] == runtime.requests[1]["nonce"]:
            raise AssertionError("flush nonces must be unique")
        _expect_error(
            FlushStateError,
            client.snapshot,
            contains="terminal",
        )


def _self_test_ready_tamper() -> None:
    mutations: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
        ("schema", lambda row: row.__setitem__("schema", "wrong")),
        ("mode", lambda row: row.__setitem__("mode", "hydra27_fixed32")),
        ("pid", lambda row: row.__setitem__("producer_pid", 4313)),
        ("generation", lambda row: row.__setitem__("generation", 1)),
        ("nonce", lambda row: row.__setitem__("nonce", "1" * 64)),
        ("action", lambda row: row.__setitem__("action", "snapshot")),
        ("status", lambda row: row.__setitem__("status", "runtime_error")),
        ("counters", lambda row: row.__setitem__("counters", [])),
        ("extra", lambda row: row.__setitem__("extra", 1)),
    )
    for label, mutate in mutations:
        with tempfile.TemporaryDirectory(prefix=f"fr13-flush-ready-{label}-") as raw:
            directory = Path(raw)
            request_path = directory / "request.json"
            ack_path = directory / "ack.json"
            row = ready_ack(mode="tail6_fixed32", producer_pid=4312)
            mutate(row)
            _write_test_json(ack_path, row)
            runtime = _FakeRuntime(request_path, ack_path)
            client = _new_test_client(directory, runtime)
            expected = FlushRuntimeError if label == "status" else FlushAckError
            _expect_error(expected, client.snapshot)


def _self_test_response_tamper() -> None:
    mutations: tuple[
        tuple[str, Callable[[dict[str, Any]], None], type[BaseException]], ...
    ] = (
        ("schema", lambda row: row.__setitem__("schema", "wrong"), FlushAckError),
        (
            "mode",
            lambda row: row.__setitem__("mode", "hydra27_fixed32"),
            FlushAckError,
        ),
        ("pid", lambda row: row.__setitem__("producer_pid", 4313), FlushAckError),
        ("generation", lambda row: row.__setitem__("generation", 7), FlushAckError),
        ("nonce", lambda row: row.__setitem__("nonce", "f" * 64), FlushAckError),
        ("action", lambda row: row.__setitem__("action", "final"), FlushAckError),
        (
            "status",
            lambda row: row.__setitem__("status", "runtime_error"),
            FlushRuntimeError,
        ),
        ("counters", lambda row: row.__setitem__("counters", []), FlushAckError),
        ("extra", lambda row: row.__setitem__("extra", 1), FlushAckError),
    )
    for label, mutate, expected in mutations:
        with tempfile.TemporaryDirectory(prefix=f"fr13-flush-response-{label}-") as raw:
            directory = Path(raw)
            request_path = directory / "request.json"
            ack_path = directory / "ack.json"
            _write_test_json(
                ack_path,
                ready_ack(mode="tail6_fixed32", producer_pid=4312),
            )
            runtime = _FakeRuntime(
                request_path,
                ack_path,
                mutate_ack=mutate,
            )
            client = _new_test_client(directory, runtime)
            _expect_error(expected, client.snapshot)
            _expect_error(
                FlushStateError,
                client.snapshot,
                contains="poisoned",
            )


def _self_test_liveness_signal_timeout() -> None:
    cases: tuple[tuple[str, dict[str, Any], type[BaseException], str], ...] = (
        (
            "dead-container",
            {"container_alive": False},
            FlushLivenessError,
            "not running",
        ),
        (
            "dead-process",
            {"process_alive": False},
            FlushLivenessError,
            "not alive",
        ),
        (
            "signal",
            {"signal_rc": 9},
            FlushSignalError,
            "delivery failed",
        ),
        (
            "timeout",
            {"no_ack": True},
            FlushTimeoutError,
            "stale generation",
        ),
    )
    for label, runtime_args, expected, contains in cases:
        with tempfile.TemporaryDirectory(prefix=f"fr13-flush-{label}-") as raw:
            directory = Path(raw)
            request_path = directory / "request.json"
            ack_path = directory / "ack.json"
            _write_test_json(
                ack_path,
                ready_ack(mode="tail6_fixed32", producer_pid=4312),
            )
            runtime = _FakeRuntime(request_path, ack_path, **runtime_args)
            client = _new_test_client(directory, runtime)
            _expect_error(expected, client.snapshot, contains=contains)


def _self_test_duplicate_and_truncated_json() -> None:
    payloads = (
        ('{"schema":"fr13-fixed32-flush-ack-v1","schema":"fr13-fixed32-flush-ack-v1"}'),
        '{"schema":',
        '{"schema":NaN}',
    )
    for index, payload in enumerate(payloads):
        with tempfile.TemporaryDirectory(prefix=f"fr13-flush-json-{index}-") as raw:
            directory = Path(raw)
            request_path = directory / "request.json"
            ack_path = directory / "ack.json"
            ack_path.write_text(payload, encoding="utf-8")
            runtime = _FakeRuntime(request_path, ack_path)
            client = _new_test_client(directory, runtime)
            _expect_error(FlushAckError, client.snapshot)


def _self_test_mutex_race() -> None:
    with tempfile.TemporaryDirectory(prefix="fr13-flush-race-") as raw:
        directory = Path(raw)
        request_path = directory / "request.json"
        ack_path = directory / "ack.json"
        _write_test_json(
            ack_path,
            ready_ack(mode="hydra27_fixed32", producer_pid=4312),
        )
        runtime = _FakeRuntime(
            request_path,
            ack_path,
            signal_delay_s=0.01,
        )
        first_client = _new_test_client(
            directory,
            runtime,
            mode="hydra27_fixed32",
            timeout_s=0.5,
        )
        second_client = _new_test_client(
            directory,
            runtime,
            mode="hydra27_fixed32",
            timeout_s=0.5,
        )
        barrier = threading.Barrier(3)
        results: list[FlushAck] = []
        failures: list[BaseException] = []
        result_lock = threading.Lock()

        def run(client: Fixed32FlushClient) -> None:
            barrier.wait()
            try:
                result = client.snapshot()
                with result_lock:
                    results.append(result)
            except BaseException as error:  # noqa: BLE001
                with result_lock:
                    failures.append(error)

        threads = (
            threading.Thread(target=run, args=(first_client,)),
            threading.Thread(target=run, args=(second_client,)),
        )
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=2.0)
            if thread.is_alive():
                raise AssertionError("mutex race test thread did not terminate")
        if failures:
            raise AssertionError(f"mutex race failed: {failures!r}")
        if sorted(ack.generation for ack in results) != [1, 2]:
            raise AssertionError("concurrent flushes did not serialize generations")
        if runtime.max_active_signals != 1:
            raise AssertionError("process-global flush mutex did not serialize")
        if [row["generation"] for row in runtime.requests] != [1, 2]:
            raise AssertionError("runtime observed a request overwrite race")


def _self_test_cross_process_resume() -> None:
    with tempfile.TemporaryDirectory(prefix="fr13-flush-resume-") as raw:
        directory = Path(raw)
        request_path = directory / "request.json"
        ack_path = directory / "ack.json"
        _write_test_json(
            ack_path,
            ready_ack(mode="tail6_fixed32", producer_pid=4312),
        )

        runtime = _FakeRuntime(request_path, ack_path)
        first_process_client = _new_test_client(directory, runtime)
        snapshot = first_process_client.snapshot()
        if snapshot.generation != 1:
            raise AssertionError("first process did not publish generation one")

        bin_dir = directory / "bin"
        bin_dir.mkdir()
        fake_docker = bin_dir / "docker"
        fake_docker.write_text(
            """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
import tempfile

args = sys.argv[1:]
if args and args[0] == "inspect":
    print("true")
    raise SystemExit(0)
if len(args) >= 5 and args[0] == "exec" and args[-2] == "-0":
    raise SystemExit(0)
if len(args) >= 5 and args[0] == "exec" and args[-2] == "-USR2":
    request_path = Path(os.environ["FR13_TEST_REQUEST"])
    ack_path = Path(os.environ["FR13_TEST_ACK"])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    ack = {
        "schema": "fr13-fixed32-flush-ack-v1",
        "mode": request["mode"],
        "producer_pid": request["producer_pid"],
        "generation": request["generation"],
        "nonce": request["nonce"],
        "action": request["action"],
        "status": "ok",
        "counters": {"events": request["generation"]},
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=ack_path.parent,
        prefix=".ack.",
        suffix=".tmp",
        encoding="utf-8",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(ack, handle, sort_keys=True)
        handle.write("\\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, ack_path)
    raise SystemExit(0)
print("unexpected fake docker argv: " + repr(args), file=sys.stderr)
raise SystemExit(127)
""",
            encoding="utf-8",
        )
        fake_docker.chmod(0o755)
        environment = os.environ.copy()
        environment["PATH"] = f"{bin_dir}{os.pathsep}{environment.get('PATH', '')}"
        environment["FR13_TEST_REQUEST"] = str(request_path)
        environment["FR13_TEST_ACK"] = str(ack_path)
        command = (
            sys.executable,
            str(Path(__file__).resolve()),
            "--container",
            "fr13-test",
            "--producer-pid",
            "4312",
            "--mode",
            "tail6_fixed32",
            "--request",
            str(request_path),
            "--ack",
            str(ack_path),
            "--action",
            "final",
            "--timeout-s",
            "1",
        )
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
            env=environment,
        )
        if completed.returncode != 0:
            raise AssertionError(
                "fresh CLI process could not resume generation one: "
                f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
            )
        result = json.loads(completed.stdout)
        child_ack = result["ack"]
        if child_ack["generation"] != 2 or child_ack["action"] != "final":
            raise AssertionError(
                f"fresh CLI process emitted the wrong final ack: {child_ack!r}"
            )

        terminal = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
            env=environment,
        )
        if terminal.returncode != 2 or "terminal" not in terminal.stderr:
            raise AssertionError(
                "fresh CLI process did not reject an existing final ack: "
                f"rc={terminal.returncode} stderr={terminal.stderr!r}"
            )


def self_test() -> None:
    """Run CPU-only protocol, tamper, liveness, timeout, and race tests."""

    tests = (
        _self_test_success_and_terminal,
        _self_test_ready_tamper,
        _self_test_response_tamper,
        _self_test_liveness_signal_timeout,
        _self_test_duplicate_and_truncated_json,
        _self_test_mutex_race,
        _self_test_cross_process_resume,
    )
    for test in tests:
        test()
    print(
        "PASS fr13_fixed32_flush_protocol "
        f"schema={SELF_TEST_SCHEMA} groups={len(tests)}",
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Issue one fail-closed fixed32 runtime snapshot or final flush")
    )
    parser.add_argument("--container")
    parser.add_argument("--producer-pid", type=int)
    parser.add_argument("--mode", choices=sorted(FIXED32_MODES))
    parser.add_argument("--request", type=Path)
    parser.add_argument("--ack", type=Path)
    parser.add_argument("--action", choices=sorted(FLUSH_ACTIONS))
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--poll-interval-s", type=float, default=0.02)
    parser.add_argument("--liveness-interval-s", type=float, default=0.25)
    parser.add_argument("--command-timeout-s", type=float, default=5.0)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run CPU-only fail-closed protocol tests",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    transaction_values = (
        args.container,
        args.producer_pid,
        args.mode,
        args.request,
        args.ack,
        args.action,
    )
    if args.self_test:
        if any(value is not None for value in transaction_values):
            parser.error("--self-test cannot be combined with transaction options")
        self_test()
        return 0
    if any(value is None for value in transaction_values):
        parser.error(
            "--container, --producer-pid, --mode, --request, --ack, and "
            "--action are required"
        )

    try:
        ack = request_flush(
            container=args.container,
            producer_pid=args.producer_pid,
            mode=args.mode,
            request_path=args.request,
            ack_path=args.ack,
            action=args.action,
            timeout_s=args.timeout_s,
            poll_interval_s=args.poll_interval_s,
            liveness_interval_s=args.liveness_interval_s,
            command_timeout_s=args.command_timeout_s,
            resume_current=True,
        )
    except FlushProtocolError as error:
        print(
            f"FR13 fixed32 flush failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2

    print(
        json.dumps(
            {
                "schema": RESULT_SCHEMA,
                "ack": ack.as_dict(),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
