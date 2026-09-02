"""Unit tests for the momo agent."""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import pytest

from slop_code.agent_runner.agents.momo.agent import MomoAgent
from slop_code.agent_runner.agents.momo.agent import MomoConfig
from slop_code.agent_runner.models import AgentCostLimits
from slop_code.agent_runner.models import AgentError
from slop_code.common.llms import APIPricing
from slop_code.execution.runtime import RuntimeEvent
from slop_code.execution.runtime import RuntimeResult


@dataclass
class FakeRuntime:
    """Replays scripted `curl` results, then repeats a default forever."""

    scripted: list[tuple[str, int]] = field(default_factory=list)
    default: tuple[str, int] | None = None
    commands: list[str] = field(default_factory=list)

    def stream(
        self, command: str, env: dict, timeout: float | None
    ) -> Iterable[RuntimeEvent]:
        self.commands.append(command)
        if self.scripted:
            stdout, exit_code = self.scripted.pop(0)
        elif self.default is not None:
            stdout, exit_code = self.default
        else:
            stdout, exit_code = "", 7
        yield RuntimeEvent(
            kind="finished",
            result=RuntimeResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr="curl: (7) failed to connect",
                setup_stdout="",
                setup_stderr="",
                elapsed=0.0,
                timed_out=False,
            ),
        )


def curl_response(body: object, http_status: int = 200) -> tuple[str, int]:
    """Build what the agent's `curl -w '\\n%{http_code}'` produces."""
    return (json.dumps(body) + "\n" + str(http_status), 0)


def make_agent(
    runtime: FakeRuntime,
    *,
    poll_interval: float = 0.0,
    run_idle_timeout: float = 3600.0,
) -> MomoAgent:
    """Build an agent wired to a fake runtime, bypassing container setup."""
    agent = MomoAgent(
        config=MomoConfig(
            type="momo",
            server_dist=Path("/nonexistent"),
            harness_path=Path("/nonexistent"),
            poll_interval=poll_interval,
            run_idle_timeout=run_idle_timeout,
            cost_limits=AgentCostLimits(cost_limit=0, net_cost_limit=0),
        ),
        model_id="test-model",
        reasoning_effort=None,
        problem_name="test_problem",
        pricing=APIPricing(input=3.0, output=15.0),
        verbose=False,
    )
    agent._runtime = runtime
    agent._momo_session_id = "session-1"
    return agent


class TestPollResponseHandling:
    """A malformed status response must surface as an AgentError.

    Anything else escapes as a TypeError/KeyError, which the harness records
    as a non-agent crash rather than an ordinary checkpoint failure.
    """

    def test_non_dict_payload_raises_agent_error(self) -> None:
        runtime = FakeRuntime(scripted=[curl_response("a bare string")])
        agent = make_agent(runtime)

        with pytest.raises(AgentError, match="Unexpected momo session"):
            agent._poll_until_run_ends()

    def test_payload_without_status_raises_agent_error(self) -> None:
        runtime = FakeRuntime(scripted=[curl_response({"id": "session-1"})])
        agent = make_agent(runtime)

        with pytest.raises(AgentError, match="Unexpected momo session"):
            agent._poll_until_run_ends()

    def test_idle_status_ends_the_poll(self) -> None:
        runtime = FakeRuntime(scripted=[curl_response({"status": "idle"})])
        agent = make_agent(runtime)

        agent._poll_until_run_ends()

    def test_repeated_transport_failures_propagate(self) -> None:
        runtime = FakeRuntime(default=("", 7))
        agent = make_agent(runtime)

        with pytest.raises(AgentError, match="curl to momo server failed"):
            agent._poll_until_run_ends()


class TestWedgedRunDetection:
    """A run reporting 'running' while emitting nothing must not hang.

    Without a bound the poll loop spins forever and pins a worker for the
    rest of the benchmark.
    """

    def test_idle_run_is_abandoned(self) -> None:
        runtime = FakeRuntime(default=curl_response({"status": "running"}))
        agent = make_agent(
            runtime, poll_interval=0.01, run_idle_timeout=0.3
        )

        started = time.monotonic()
        with pytest.raises(AgentError, match="produced no events"):
            agent._poll_until_run_ends()

        assert time.monotonic() - started < 10.0

    def test_zero_timeout_disables_the_bound(self) -> None:
        runtime = FakeRuntime(default=curl_response({"status": "running"}))
        agent = make_agent(runtime, poll_interval=0.01, run_idle_timeout=0)

        finished = threading.Event()

        def poll() -> None:
            with contextlib.suppress(AgentError):
                agent._poll_until_run_ends()
            finished.set()

        threading.Thread(target=poll, daemon=True).start()
        finished.wait(timeout=1.0)

        assert not finished.is_set()

    def test_progress_resets_the_idle_window(self, tmp_path: Path) -> None:
        """Events arriving keep a long but healthy run alive."""
        runtime = FakeRuntime(default=curl_response({"status": "running"}))
        agent = make_agent(
            runtime, poll_interval=0.01, run_idle_timeout=0.3
        )
        # A real data dir whose event log keeps growing between polls.
        events = tmp_path / "sessions" / "session-1" / "events.jsonl"
        events.parent.mkdir(parents=True)
        events.write_text("")

        class DataTmp:
            name = str(tmp_path)

        agent._data_tmp = DataTmp()  # type: ignore[assignment]

        stop = threading.Event()

        def append_events() -> None:
            line = json.dumps({"type": "tool_call_started"}) + "\n"
            while not stop.is_set():
                with events.open("a") as handle:
                    handle.write(line)
                time.sleep(0.02)

        writer = threading.Thread(target=append_events, daemon=True)
        writer.start()
        finished = threading.Event()

        def poll() -> None:
            with contextlib.suppress(AgentError):
                agent._poll_until_run_ends()
            finished.set()

        threading.Thread(target=poll, daemon=True).start()
        # Well past the idle window: continuous progress must keep it alive.
        finished.wait(timeout=1.0)
        stop.set()
        writer.join(timeout=1.0)

        assert not finished.is_set()


class TestNoHarnessImposedLimits:
    """The harness must not end a run that momo would let continue.

    Momo self-bounds every run (256 turns, 3-day wall clock, 24h per tool
    call). Any additional harness-side deadline can only cut a legitimate
    run short and turn a real result into a truncated one.
    """

    def test_idle_bound_is_disabled_by_default(self) -> None:
        config = MomoConfig(
            type="momo",
            server_dist=Path("/nonexistent"),
            harness_path=Path("/nonexistent"),
            cost_limits=AgentCostLimits(cost_limit=0, net_cost_limit=0),
        )

        assert config.run_idle_timeout == 0

    def test_shipped_config_imposes_no_limits(self) -> None:
        """configs/agents/momo.yaml must not reintroduce a bound."""
        import yaml

        raw = yaml.safe_load(
            Path("configs/agents/momo.yaml").read_text()
        )
        config = MomoConfig(**raw)

        assert config.run_idle_timeout == 0
        assert config.cost_limits.step_limit == 0
        assert config.cost_limits.cost_limit == 0
        assert config.cost_limits.net_cost_limit == 0

    def test_default_agent_polls_indefinitely(self) -> None:
        """With the shipped defaults a long silent run is not aborted."""
        runtime = FakeRuntime(default=curl_response({"status": "running"}))
        agent = make_agent(
            runtime, poll_interval=0.01, run_idle_timeout=0
        )
        finished = threading.Event()

        def poll() -> None:
            with contextlib.suppress(AgentError):
                agent._poll_until_run_ends()
            finished.set()

        threading.Thread(target=poll, daemon=True).start()
        finished.wait(timeout=1.0)

        assert not finished.is_set()


class TestRunOutcomes:
    """Each terminal momo status must map to the right harness outcome."""

    def _run_with_status(self, status: str) -> MomoAgent:
        runtime = FakeRuntime(
            scripted=[
                curl_response({"accepted": True}, 202),
                curl_response({"status": "idle"}),
            ]
        )
        agent = make_agent(runtime)
        agent._ingest_new_events = lambda: setattr(  # type: ignore
            agent,
            "_last_run_finished",
            {"status": status, "turnsUsed": 7},
        )
        agent.run("task")
        return agent

    def test_completed_run_succeeds(self) -> None:
        agent = self._run_with_status("completed")

        assert agent._incomplete_runs == []

    @pytest.mark.parametrize(
        "status", ["stopped", "turns_exhausted", "timeout"]
    )
    def test_truncated_run_is_recorded_but_not_retried(
        self, status: str
    ) -> None:
        """Cut-off runs are scored as-is and flagged, never retried.

        Retrying would resume the session with a fresh per-run turn budget,
        silently granting the agent several times its intended allowance.
        """
        agent = self._run_with_status(status)

        assert agent._incomplete_runs == [status]

    def test_errored_run_raises(self) -> None:
        runtime = FakeRuntime(
            scripted=[
                curl_response({"accepted": True}, 202),
                curl_response({"status": "idle"}),
            ]
        )
        agent = make_agent(runtime)
        agent._ingest_new_events = lambda: setattr(  # type: ignore
            agent,
            "_last_run_finished",
            {"status": "error", "error": {"message": "provider exploded"}},
        )

        with pytest.raises(AgentError, match="provider exploded"):
            agent.run("task")

    def test_missing_run_finished_raises(self) -> None:
        """An empty event log means the run was never observed."""
        runtime = FakeRuntime(
            scripted=[
                curl_response({"accepted": True}, 202),
                curl_response({"status": "idle"}),
            ]
        )
        agent = make_agent(runtime)
        agent._ingest_new_events = lambda: None  # type: ignore[method-assign]

        with pytest.raises(AgentError, match="without a run_finished event"):
            agent.run("task")


class TestArtifacts:
    """Truncated runs must stay visible after the run."""

    def test_incomplete_runs_are_written(self, tmp_path: Path) -> None:
        runtime = FakeRuntime()
        agent = make_agent(runtime)
        agent._incomplete_runs = ["turns_exhausted"]

        agent.save_artifacts(tmp_path)

        payload = json.loads((tmp_path / "incomplete_runs.json").read_text())
        assert payload["statuses"] == ["turns_exhausted"]

    def test_no_marker_for_clean_runs(self, tmp_path: Path) -> None:
        agent = make_agent(FakeRuntime())

        agent.save_artifacts(tmp_path)

        assert not (tmp_path / "incomplete_runs.json").exists()


class TestUsageIngestion:
    """Usage is parsed incrementally from the mounted event logs."""

    def test_events_are_counted_once(self, tmp_path: Path) -> None:
        agent = make_agent(FakeRuntime())

        class DataTmp:
            name = str(tmp_path)

        agent._data_tmp = DataTmp()  # type: ignore[assignment]
        events = tmp_path / "sessions" / "session-1" / "events.jsonl"
        events.parent.mkdir(parents=True)
        events.write_text(
            json.dumps(
                {
                    "type": "llm_call_finished",
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                    },
                }
            )
            + "\n"
        )

        for _ in range(3):
            agent._ingest_new_events()

        assert agent.usage.steps == 1
        assert agent.usage.net_tokens.input == 100
        assert agent.usage.net_tokens.output == 50

    def test_partial_trailing_line_is_not_consumed(
        self, tmp_path: Path
    ) -> None:
        """A half-written line is re-read once its newline arrives."""
        agent = make_agent(FakeRuntime())

        class DataTmp:
            name = str(tmp_path)

        agent._data_tmp = DataTmp()  # type: ignore[assignment]
        events = tmp_path / "sessions" / "session-1" / "events.jsonl"
        events.parent.mkdir(parents=True)
        event = json.dumps(
            {
                "type": "llm_call_finished",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )
        events.write_text(event[:20])
        agent._ingest_new_events()
        assert agent.usage.steps == 0

        events.write_text(event + "\n")
        agent._ingest_new_events()

        assert agent.usage.steps == 1
