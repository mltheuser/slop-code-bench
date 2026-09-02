"""Utilities for processing runtime output streams with threading.

This module provides utilities for handling streaming output from runtime processes
with proper threading and timeout management:

- **ensure_string**: Convert bytes to string with error handling
- **start_stream_pump**: Start threaded stream processing
- **make_timeout_fn**: Create timeout calculation functions
- **process_stream**: Main stream processing with timeout and filtering

The utilities support both Docker and local runtime streams, providing
consistent behavior across different execution environments with proper cleanup
and timeout handling.

Output completeness
-------------------
A process exiting is NOT the same as its output having been consumed: the OS
pipe buffer can still hold megabytes after the writer is gone, and the pump
thread needs time to drain it into the event queue. Callers rely on
``RuntimeResult.stdout`` being the *whole* output (agents parse it as a single
JSON document), so on the normal path this module waits for the pump thread to
signal end-of-stream before returning, rather than stopping as soon as the
queue momentarily runs dry. Only a timeout or an explicit kill may truncate,
and that is reported via ``RuntimeResult.timed_out``.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Iterator
from typing import Literal

import structlog

from slop_code.execution.runtime import RuntimeEvent
from slop_code.execution.runtime import RuntimeResult

logger = structlog.get_logger(__name__)

# Fallback deadline used only when a caller asks for one without naming it.
# ``timeout=None`` means *no* deadline (see make_timeout_fn): an agent run has
# no meaningful upper bound the harness can pick for it, and silently killing
# one at an arbitrary hour produces a truncated benchmark result rather than a
# slow one.
DEFAULT_WAIT_TIMEOUT = 7200.0  # 2 hours

# Idle limit for draining output after the process has exited. This bounds the
# gap *between* chunks, not the total drain, so arbitrarily large outputs still
# come through in full as long as they keep arriving. It exists for the case
# where a background grandchild inherited the pipe and keeps it open forever:
# then no chunk ever arrives, and we stop instead of hanging.
DRAIN_IDLE_TIMEOUT = 5.0

# How long to block on the queue before re-checking whether the process exited.
EXIT_POLL_INTERVAL = 0.1

# Courtesy join for the pump thread. On the normal path it has already
# finished, so this returns immediately. After a timeout it is typically
# blocked on a pipe read that only unblocks once the caller kills the process,
# so we abandon it rather than wait: it is a daemon thread writing to a queue
# we own, and dropping it is harmless. Joining unbounded here would hang the
# run, because the kill happens after this function returns.
THREAD_JOIN_TIMEOUT = 1.0


def ensure_string(data: bytes | str) -> str:
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data


def start_stream_pump(
    stream: Iterator[tuple[bytes | str, bytes | str]],
    event_queue: queue.Queue[
        tuple[Literal["stdout", "stderr", "finished"], str | None]
    ],
    stop_event: threading.Event,
) -> threading.Thread:
    """Start a thread to pump a demuxed stream into an event queue.

    Args:
        stream: Iterator yielding (stdout, stderr) tuples
        event_queue: Queue to receive events
        stop_event: Event to check for early termination
        ensure_string: Function to convert bytes to string
    """

    def pump() -> None:
        """Pump demuxed stream to event queue."""
        for stdout, stderr in stream:
            if stdout:
                contents = ensure_string(stdout)
                event_queue.put(("stdout", contents))
            if stderr:
                contents = ensure_string(stderr)
                event_queue.put(("stderr", contents))
            if stop_event.is_set():
                break
        event_queue.put(("finished", None))

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    return thread


def make_timeout_fn(
    timeout: float | None, start_time: float
) -> Callable[[], float]:
    """Build a function returning the seconds left before the deadline.

    Args:
        timeout: Seconds allowed, or None for no deadline at all.
        start_time: ``time.monotonic()`` reading when the work started.

    Returns:
        A callable returning remaining seconds; always positive (never
        expiring) when ``timeout`` is None.
    """
    if timeout is None:
        # No deadline. Returning a large constant keeps every caller's
        # "how long may I block?" arithmetic working without special cases,
        # while never reaching zero.
        return lambda: DEFAULT_WAIT_TIMEOUT

    deadline = start_time + timeout

    def timeout_fn() -> float:
        return deadline - time.monotonic()

    return timeout_fn


def process_stream(
    stream: Iterator[tuple[str | bytes, str | bytes]],
    timeout: float | None,
    poll_fn: Callable[[], int | None],
    yield_only_after: str | None = None,
) -> Generator[RuntimeEvent, None, RuntimeResult]:
    logger.debug("Starting to consume events with timeout", timeout=timeout)
    start_time = time.monotonic()
    timeout_fn = make_timeout_fn(timeout, start_time)
    stop_event = threading.Event()
    event_queue: queue.Queue[
        tuple[Literal["stdout", "stderr", "finished"], str | None]
    ] = queue.Queue()
    thread = start_stream_pump(stream, event_queue, stop_event)
    stdout = ""
    stderr = ""
    setup_stdout = ""
    setup_stderr = ""
    yielding_stdout = yield_only_after is None
    yielding_stderr = yield_only_after is None
    timed_out = False

    def handle_event(
        kind: Literal["stdout", "stderr"],
        payload: str,
    ) -> Iterator[RuntimeEvent]:
        nonlocal stdout, stderr, setup_stdout, setup_stderr
        nonlocal yielding_stdout, yielding_stderr

        if kind == "stdout":
            stdout += payload
            if (
                not yielding_stdout
                and yield_only_after
                and yield_only_after in stdout
            ):
                yielding_stdout = True
                setup_stdout, stdout = stdout.split(yield_only_after, 1)
                payload = stdout

            if yielding_stdout and payload.strip():
                yield RuntimeEvent(kind="stdout", text=payload)
            return

        if kind == "stderr":
            stderr += payload
            if (
                not yielding_stderr
                and yield_only_after
                and yield_only_after in stderr
            ):
                yielding_stderr = True
                setup_stderr, stderr = stderr.split(yield_only_after, 1)
                payload = stderr

            if yielding_stderr and payload.strip():
                yield RuntimeEvent(kind="stderr", text=payload)
            return

        logger.error("Received unknown event", kind=kind, payload=payload)

    # End-of-stream is signalled by the pump thread, not by process exit:
    # the pipe can still hold buffered output after the writer is gone.
    stream_exhausted = False
    exit_code: int | None = None

    def consume(payload: str | None, kind: str) -> Iterator[RuntimeEvent]:
        """Dispatch one queue item; sets stream_exhausted on the sentinel."""
        nonlocal stream_exhausted
        if kind == "finished":
            logger.debug("Received finished event")
            stream_exhausted = True
            return
        if payload is None:
            logger.error("Received empty stream event", kind=kind)
            stream_exhausted = True
            return
        yield from handle_event(kind, payload)  # type: ignore[arg-type]

    # Phase 1: the process is running. Wake up regularly so that its exit is
    # noticed promptly even while no output arrives.
    while not stream_exhausted:
        if (remaining := timeout_fn()) <= 0:
            timed_out = True
            break
        if (exit_code := poll_fn()) is not None:
            break
        try:
            kind, payload = event_queue.get(
                timeout=min(remaining, EXIT_POLL_INTERVAL)
            )
        except queue.Empty:
            continue
        yield from consume(payload, kind)

    # Phase 2: the process exited but its output may still be in flight. Wait
    # for the pump thread to reach end-of-stream instead of stopping at the
    # first momentarily empty queue, which would silently truncate the tail.
    if not stream_exhausted and not timed_out:
        while not stream_exhausted:
            try:
                kind, payload = event_queue.get(timeout=DRAIN_IDLE_TIMEOUT)
            except queue.Empty:
                # Nothing for a while and the writer is gone: either the pipe
                # is held open by a background grandchild, or the pump thread
                # is wedged. Either way no more output is coming.
                logger.warning(
                    "No output for %.0fs after process exit; stopping drain. "
                    "Captured output may be truncated if a background "
                    "process inherited the pipe.",
                    DRAIN_IDLE_TIMEOUT,
                    stdout_chars=len(stdout),
                    stderr_chars=len(stderr),
                )
                break
            yield from consume(payload, kind)

    # Phase 3: timed out or killed. Take whatever is already queued without
    # waiting; the caller kills the process and truncation is expected here.
    if timed_out:
        while not stream_exhausted:
            try:
                kind, payload = event_queue.get_nowait()
            except queue.Empty:
                break
            yield from consume(payload, kind)

    elapsed = time.monotonic() - start_time
    stop_event.set()
    thread.join(timeout=THREAD_JOIN_TIMEOUT)
    if thread.is_alive():
        # Expected after a timeout: the thread is parked in a blocking read
        # that ends when the caller kills the process. Nothing references it
        # afterwards, so abandoning it leaks nothing.
        logger.debug(
            "Stream pump thread still running; abandoning it",
            join_timeout=THREAD_JOIN_TIMEOUT,
            timed_out=timed_out,
        )

    if exit_code is None:
        exit_code = poll_fn()
    if exit_code is None:
        exit_code = -1
    logger.debug(
        "Setup stdout", setup_stdout=setup_stdout, setup_stderr=setup_stderr
    )
    return RuntimeResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        setup_stdout=setup_stdout,
        setup_stderr=setup_stderr,
        elapsed=elapsed,
        timed_out=timed_out,
    )
