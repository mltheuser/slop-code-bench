"""Tests for runtime stream processing helpers."""

from __future__ import annotations

import selectors
import subprocess
import sys
import threading
import time
from collections.abc import Iterator

from slop_code.execution import stream_processor
from slop_code.execution.local_streaming import _ChunkReader
from slop_code.execution.runtime import RuntimeResult
from slop_code.execution.stream_processor import ensure_string
from slop_code.execution.stream_processor import make_timeout_fn
from slop_code.execution.stream_processor import process_stream


def test_ensure_string_preserves_text_around_invalid_utf8_bytes() -> None:
    decoded = ensure_string(b'{"type":"message_update","data":"ok"}\xff\n')

    assert '{"type":"message_update","data":"ok"}' in decoded
    assert decoded.endswith("\n")


def drain(
    stream: Iterator[tuple[str | bytes, str | bytes]],
    timeout: float | None,
    poll_fn,
    **kwargs,
) -> RuntimeResult:
    """Run process_stream to completion and return its RuntimeResult."""
    generator = process_stream(stream, timeout, poll_fn, **kwargs)
    try:
        while True:
            next(generator)
    except StopIteration as stop:
        return stop.value


class TestProcessStreamCompleteness:
    """Output must be captured in full, even when it arrives after exit.

    A process exiting does not mean its output has been read: the pipe buffer
    can still hold megabytes. Agents parse stdout as a single JSON document,
    so losing the tail corrupts the run rather than merely trimming a log.
    """

    def test_captures_output_arriving_after_process_exit(self) -> None:
        """Output still buffered when the process exits is not dropped."""
        # The pump thread is deliberately slow, so the process has long exited
        # by the time the bulk of the payload reaches the queue.
        payload = "x" * 100_000
        exited = threading.Event()

        def slow_stream() -> Iterator[tuple[str, str]]:
            for index in range(0, len(payload), 1_000):
                if index >= 2_000:
                    exited.set()
                    time.sleep(0.005)
                yield payload[index : index + 1_000], ""

        def poll_fn() -> int | None:
            return 0 if exited.is_set() else None

        result = drain(slow_stream(), None, poll_fn)

        assert result.stdout == payload
        assert len(result.stdout) == len(payload)

    def test_captures_large_payload_from_real_subprocess(self) -> None:
        """End-to-end: a big write followed by immediate exit loses nothing."""
        size = 4 * 1024 * 1024
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                f"import sys; sys.stdout.write('x' * {size})",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
        )
        try:
            result = drain(_demux(proc), None, proc.poll)
        finally:
            proc.kill()
            proc.wait()

        assert len(result.stdout) == size
        assert result.exit_code == 0

    def test_captures_stderr_arriving_after_process_exit(self) -> None:
        """The same guarantee applies to stderr."""
        exited = threading.Event()

        def slow_stream() -> Iterator[tuple[str, str]]:
            yield "", "first"
            exited.set()
            time.sleep(0.05)
            yield "", "second"

        def poll_fn() -> int | None:
            return 0 if exited.is_set() else None

        result = drain(slow_stream(), None, poll_fn)

        assert result.stderr == "firstsecond"


class TestProcessStreamTermination:
    """Waiting for output must not turn into waiting forever."""

    def test_timeout_reports_timed_out_and_returns_promptly(self) -> None:
        """A process that never exits stops at the deadline."""

        def endless_stream() -> Iterator[tuple[str, str]]:
            while True:
                time.sleep(0.01)
                yield "tick", ""

        started = time.monotonic()
        result = drain(endless_stream(), 0.5, lambda: None)
        elapsed = time.monotonic() - started

        assert result.timed_out is True
        assert elapsed < 5.0

    def test_returns_when_stream_ends_without_exit_code(self) -> None:
        """End-of-stream ends the run even if poll never reports exit."""

        def short_stream() -> Iterator[tuple[str, str]]:
            yield "done", ""

        result = drain(short_stream(), 5.0, lambda: None)

        assert result.stdout == "done"
        assert result.timed_out is False
        assert result.exit_code == -1

    def test_reports_exit_code_from_poll(self) -> None:
        """The exit code observed while streaming is preserved."""

        def short_stream() -> Iterator[tuple[str, str]]:
            yield "out", ""

        result = drain(short_stream(), 5.0, lambda: 3)

        assert result.exit_code == 3

    def test_zero_exit_code_is_not_confused_with_missing(self) -> None:
        """Exit code 0 must survive: `exit_code or poll()` would lose it."""

        def short_stream() -> Iterator[tuple[str, str]]:
            yield "out", ""

        result = drain(short_stream(), 5.0, lambda: 0)

        assert result.exit_code == 0


class TestProcessStreamBackgroundChild:
    """A backgrounded child must not stall or truncate the foreground run.

    Agents start servers with `nohup ... &`. The shell exits immediately, but
    the child can hold the pipe's write end open for the rest of the run.
    """

    def test_completes_promptly_when_child_holds_pipe_open(self) -> None:
        """The command's own output is captured without awaiting the child."""
        proc = subprocess.Popen(
            ["sh", "-c", "sleep 60 & echo started"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
        )
        started = time.monotonic()
        try:
            result = drain(_demux(proc), None, proc.poll)
        finally:
            proc.kill()
            proc.wait()
        elapsed = time.monotonic() - started

        assert result.stdout.strip() == "started"
        assert elapsed < 30.0


class TestNoTimeoutMeansNoDeadline:
    """``timeout=None`` must impose no deadline whatsoever.

    Agent runs are passed ``timeout=None`` precisely because the harness has
    no sensible upper bound for them. Silently substituting a default would
    kill a long run mid-flight and record a truncated result as if it were
    the agent's own output.
    """

    def test_make_timeout_fn_never_expires(self) -> None:
        never = make_timeout_fn(None, time.monotonic() - 10_000_000)

        assert never() > 0

    def test_explicit_timeout_still_counts_down(self) -> None:
        expired = make_timeout_fn(5.0, time.monotonic() - 10.0)

        assert expired() < 0

    def test_run_outliving_the_default_is_not_killed(
        self, monkeypatch
    ) -> None:
        """A run longer than the fallback constant completes untouched."""
        monkeypatch.setattr(stream_processor, "DEFAULT_WAIT_TIMEOUT", 0.5)

        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import sys, time\n"
                "for i in range(4):\n"
                "    sys.stdout.write(f'{i}\\n'); sys.stdout.flush()\n"
                "    time.sleep(0.4)",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
        )
        try:
            result = drain(_demux(proc), None, proc.poll)
        finally:
            proc.kill()
            proc.wait()

        assert result.timed_out is False
        assert result.stdout.split() == ["0", "1", "2", "3"]


class TestProcessStreamSetupSplit:
    """The setup/command split must keep working."""

    def test_splits_setup_output_on_marker(self) -> None:
        """Text before the marker is setup output, the rest is the command."""

        def stream() -> Iterator[tuple[str, str]]:
            yield "setup logs<<<SPLIT>>>command out", ""

        result = drain(stream(), 5.0, lambda: 0, yield_only_after="<<<SPLIT>>>")

        assert result.setup_stdout == "setup logs"
        assert result.stdout == "command out"


def _demux(proc: subprocess.Popen) -> Iterator[tuple[str, str]]:
    """Yield (stdout, stderr) chunks from a live subprocess.

    Mirrors LocalStreamingRuntime, including its non-blocking chunk reads.
    """
    selector = selectors.DefaultSelector()
    readers = {
        "out": _ChunkReader(proc.stdout),
        "err": _ChunkReader(proc.stderr),
    }
    selector.register(proc.stdout, selectors.EVENT_READ, data="out")
    selector.register(proc.stderr, selectors.EVENT_READ, data="err")
    try:
        while selector.get_map():
            for key, _ in selector.select():
                reader = readers[key.data]
                chunk = reader.read(4096)
                if not chunk:
                    if reader.at_eof:
                        selector.unregister(key.fileobj)
                    continue
                if key.data == "out":
                    yield chunk, ""
                else:
                    yield "", chunk
    finally:
        selector.close()
