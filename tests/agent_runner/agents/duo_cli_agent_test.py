"""Unit tests for the GitLab Duo CLI agent."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import pytest

from slop_code.agent_runner.agents.duo_cli import DuoCliAgent
from slop_code.agent_runner.agents.duo_cli import DuoCliConfig
from slop_code.agent_runner.models import AgentCostLimits
from slop_code.agent_runner.models import AgentError
from slop_code.common.llms import APIPricing
from slop_code.execution.runtime import RuntimeEvent
from slop_code.execution.runtime import RuntimeResult

SESSION_LOG_LINE = (
    "2026-09-01T16:01:24:240 [info]: "
    "[DuoWorkflowNodeExecutor][6832711] Executor started\n"
)


@dataclass
class FakeRuntime:
    """Runtime stub that replays a scripted sequence of events per run."""

    scripted_runs: list[list[RuntimeEvent]] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)

    def stream(
        self,
        command: str,
        env: dict,
        timeout: float | None,
    ) -> Iterable[RuntimeEvent]:
        self.commands.append(command)
        yield from self.scripted_runs.pop(0)


def make_agent(runtime: FakeRuntime) -> DuoCliAgent:
    """Build an agent wired to a fake runtime, bypassing container setup."""
    agent = DuoCliAgent(
        config=DuoCliConfig(
            type="duo_cli",
            version="9.17.0",
            cost_limits=AgentCostLimits(cost_limit=0, net_cost_limit=0),
        ),
        model_id="claude_opus_5",
        problem_name="test_problem",
        pricing=APIPricing(input=3.0, output=15.0),
        verbose=False,
    )
    agent._runtime = runtime
    return agent


def finished(
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
) -> RuntimeEvent:
    """Build the terminal event of a run."""
    return RuntimeEvent(
        kind="finished",
        result=RuntimeResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            setup_stdout="",
            setup_stderr="",
            elapsed=1.0,
            timed_out=False,
        ),
    )


def result_document(session_id: str = "6832711") -> str:
    """Build a successful `duo run` result document."""
    return json.dumps(
        {
            "schemaVersion": "1.0",
            "sessionId": session_id,
            "status": "success",
            "response": "done",
            "elements": [],
        }
    )


class TestSessionIdCapture:
    """The session id must survive a run that produces no usable document.

    It is what lets a retry resume the same conversation. Reading it only
    from the result document loses it exactly when it matters most: when the
    run was killed or its stdout was truncated.
    """

    def test_captures_session_id_from_stderr_stream(self) -> None:
        runtime = FakeRuntime(
            scripted_runs=[
                [
                    RuntimeEvent(kind="stderr", text=SESSION_LOG_LINE),
                    finished(stdout=result_document()),
                ]
            ]
        )
        agent = make_agent(runtime)

        agent.run("task")

        assert agent._resume_session_id == "6832711"

    def test_captures_session_id_when_stdout_is_empty(self) -> None:
        """A run killed mid-flight writes no document but still logs its id."""
        runtime = FakeRuntime(
            scripted_runs=[
                [
                    RuntimeEvent(kind="stderr", text=SESSION_LOG_LINE),
                    finished(stdout="", exit_code=143),
                ]
            ]
        )
        agent = make_agent(runtime)

        with pytest.raises(AgentError, match="wrote no result document"):
            agent.run("task")

        assert agent._resume_session_id == "6832711"

    def test_captures_session_id_when_stdout_is_truncated(self) -> None:
        """Truncated stdout is unparseable, but the id is already known."""
        runtime = FakeRuntime(
            scripted_runs=[
                [
                    RuntimeEvent(kind="stderr", text=SESSION_LOG_LINE),
                    finished(stdout='{"sessionId": "68327'),
                ]
            ]
        )
        agent = make_agent(runtime)

        with pytest.raises(AgentError, match="not valid JSON"):
            agent.run("task")

        assert agent._resume_session_id == "6832711"

    def test_first_session_id_wins(self) -> None:
        """The id is stable per run; later log lines must not overwrite it."""
        runtime = FakeRuntime(
            scripted_runs=[
                [
                    RuntimeEvent(kind="stderr", text=SESSION_LOG_LINE),
                    RuntimeEvent(
                        kind="stderr",
                        text="[DuoWorkflowNodeExecutor][9999999] later\n",
                    ),
                    finished(stdout=result_document()),
                ]
            ]
        )
        agent = make_agent(runtime)

        agent.run("task")

        assert agent._resume_session_id == "6832711"

    def test_reset_clears_session_id(self) -> None:
        """Each checkpoint starts a fresh conversation."""
        runtime = FakeRuntime(
            scripted_runs=[
                [
                    RuntimeEvent(kind="stderr", text=SESSION_LOG_LINE),
                    finished(stdout=result_document()),
                ]
            ]
        )
        agent = make_agent(runtime)
        agent.run("task")

        agent.reset()

        assert agent._resume_session_id is None


class TestRetry:
    """A retry must resume the failed run, or not happen at all."""

    def test_retry_resumes_observed_session(self) -> None:
        runtime = FakeRuntime(
            scripted_runs=[
                [
                    RuntimeEvent(kind="stderr", text=SESSION_LOG_LINE),
                    finished(stdout="", exit_code=143),
                ],
                [finished(stdout=result_document())],
            ]
        )
        agent = make_agent(runtime)
        with pytest.raises(AgentError):
            agent.run("task")

        agent.retry()

        assert "--existing-session-id 6832711" in runtime.commands[1]
        assert "Continue from where you left off." in runtime.commands[1]

    def test_retry_without_session_id_raises(self) -> None:
        """Without a session the retry would be a contextless blank run."""
        runtime = FakeRuntime(
            scripted_runs=[[finished(stdout="", exit_code=143)]]
        )
        agent = make_agent(runtime)
        with pytest.raises(AgentError):
            agent.run("task")

        with pytest.raises(AgentError, match="no session id"):
            agent.retry()

        # No second invocation was attempted.
        assert len(runtime.commands) == 1

    def test_run_checkpoint_reports_error_when_retry_impossible(self) -> None:
        """The checkpoint fails cleanly instead of running a blind retry."""
        runtime = FakeRuntime(
            scripted_runs=[[finished(stdout="", exit_code=143)]]
        )
        agent = make_agent(runtime)

        result = agent.run_checkpoint("task")

        assert result.had_error is True
        assert len(runtime.commands) == 1


class TestArtifacts:
    """Failed runs must remain inspectable after the fact."""

    def test_saves_artifacts_for_failed_run(self, tmp_path: Path) -> None:
        runtime = FakeRuntime(
            scripted_runs=[
                [
                    RuntimeEvent(kind="stderr", text=SESSION_LOG_LINE),
                    finished(stdout="", exit_code=143),
                ]
            ]
        )
        agent = make_agent(runtime)
        with pytest.raises(AgentError):
            agent.run("task")

        agent.save_artifacts(tmp_path)

        assert (tmp_path / "run_1.result.json").exists()
        assert SESSION_LOG_LINE in (tmp_path / "run_1.stderr.log").read_text()
