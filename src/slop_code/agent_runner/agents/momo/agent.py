"""Momo agent: drives a momo-agent server running inside the workspace container.

The harness owns the container (built from the environment's base image, the
workspace bind-mounted at the environment workdir). This agent mounts a
host-built momo-agent server distribution and a harness folder read-only into
that container, starts the server inside it, and speaks its localhost HTTP API
via ``curl`` executed through the streaming runtime. The server's data
directory is a host temp dir mounted read-write, so event logs are parsed on
the host for usage tracking and saved as artifacts.

Requires:
- A Docker environment spec (the server runs inside the container).
- Java 21 in the environment base image (Debian trixie's default JRE).
- An ai-router instance reachable from inside the container
  (default: ``http://host.docker.internal:8787``).
"""

from __future__ import annotations

import json
import shlex
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from slop_code.agent_runner.agent import Agent
from slop_code.agent_runner.agent import AgentConfigBase
from slop_code.agent_runner.credentials import ProviderCredential
from slop_code.agent_runner.models import AgentError
from slop_code.agent_runner.models import AgentSetupError
from slop_code.agent_runner.models import UnsupportedEnvironmentError
from slop_code.agent_runner.registry import register_agent
from slop_code.common.llms import ModelDefinition
from slop_code.common.llms import ThinkingPreset
from slop_code.common.llms import TokenUsage
from slop_code.execution import Session
from slop_code.execution.docker_runtime.models import DockerEnvironmentSpec
from slop_code.execution.runtime import RuntimeResult

SERVER_MOUNT = "/opt/momo-server"
HARNESS_MOUNT = "/opt/momo-harness"
DATA_MOUNT = "/momo-data"

# Momo accepts none/low/medium/high; fold the benchmark's extremes onto them.
_THINKING_TO_EFFORT: dict[str, str] = {
    "none": "none",
    "disabled": "none",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
}


class MomoConfig(AgentConfigBase, agent_type="momo"):
    """Configuration for the momo agent.

    Attributes:
        server_dist: Host path to the momo-agent server installDist directory
            (``momo-agent/server/build/install/server``).
        harness_path: Host path to the momo harness folder
            (``harness.yaml`` + ``instructions.md``).
        port: Port the in-container server listens on.
        ai_router_base_url: ai-router URL as seen from inside the container.
        poll_interval: Seconds between session-status polls during a run.
        startup_timeout: Seconds to wait for the server to become ready.
        run_idle_timeout: Abort a run that reports ``running`` while writing
            no new events for this many seconds. This is an inactivity bound,
            not a total-runtime budget: momo's own wall clock (3 days by
            default) governs how long legitimate work may take, so bounding
            total time here would cut healthy long runs short. It exists to
            catch a wedged run that would otherwise pin a worker forever.
            0 disables the bound.
    """

    type: Literal["momo"] = "momo"
    server_dist: Path
    harness_path: Path
    port: int = 8420
    ai_router_base_url: str = "http://host.docker.internal:8787"
    poll_interval: float = 3.0
    startup_timeout: float = 60.0
    # Generous: a single tool call may legitimately run for a long time
    # (momo's own tool timeout is 24h) without emitting any event.
    run_idle_timeout: float = 3600.0


class MomoAgent(Agent):
    """Agent wrapper around a momo-agent server run inside the container."""

    def __init__(
        self,
        config: MomoConfig,
        model_id: str,
        reasoning_effort: str | None,
        problem_name: str,
        pricing,
        verbose: bool,
    ) -> None:
        super().__init__(
            "momo", problem_name, config.cost_limits, pricing, verbose
        )
        self.config = config
        self.model_id = model_id
        self.reasoning_effort = reasoning_effort
        self._session: Session | None = None
        self._runtime = None
        self._data_tmp: tempfile.TemporaryDirectory | None = None
        self._workspace_in_container: str | None = None
        self._momo_session_id: str | None = None
        # Byte offsets per host events.jsonl path, for incremental parsing.
        self._event_offsets: dict[Path, int] = {}
        self._last_run_finished: dict[str, Any] | None = None
        # Terminal statuses of runs that ended without answering, in order.
        # Surfaced via save_artifacts() so a truncated run is visible after
        # the fact rather than only in the live log.
        self._incomplete_runs: list[str] = []

    @classmethod
    def _from_config(
        cls,
        config: AgentConfigBase,
        model: ModelDefinition,
        credential: ProviderCredential,
        problem_name: str,
        verbose: bool,
        image: str | None,
        thinking_preset: ThinkingPreset | None = None,
        thinking_max_tokens: int | None = None,
    ) -> "MomoAgent":
        assert isinstance(config, MomoConfig)
        momo_settings = model.agent_specific.get("momo", {})
        model_id = momo_settings.get(
            "model", f"{model.internal_name}:cloud@{model.provider}"
        )
        reasoning_effort = None
        if thinking_preset is not None:
            reasoning_effort = _THINKING_TO_EFFORT.get(thinking_preset)
        return cls(
            config=config,
            model_id=model_id,
            reasoning_effort=reasoning_effort,
            problem_name=problem_name,
            pricing=model.pricing,
            verbose=verbose,
        )

    # --- transport -------------------------------------------------------

    def _exec(self, command: str, timeout: float | None) -> RuntimeResult:
        """Run a command in the container and return its buffered result."""
        if self._runtime is None:
            raise AgentError("Momo agent has not been set up with a runtime")
        result: RuntimeResult | None = None
        for event in self._runtime.stream(command, env={}, timeout=timeout):
            if event.kind == "finished":
                result = event.result
        if result is None:
            raise AgentError(f"Container exec yielded no result: {command}")
        return result

    def _api(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> tuple[int, Any]:
        """Call the in-container momo server; return (http_status, json|None)."""
        cmd = (
            f"curl -s -m {int(timeout)} -w '\\n%{{http_code}}' -X {method} "
            f"http://127.0.0.1:{self.config.port}{path}"
        )
        if body is not None:
            cmd += " -H 'Content-Type: application/json'"
            cmd += f" -d {shlex.quote(json.dumps(body))}"
        result = self._exec(cmd, timeout=timeout + 15)
        if result.exit_code != 0:
            raise AgentError(
                f"curl to momo server failed (exit {result.exit_code}): "
                f"{method} {path}: {result.stderr[-500:]}"
            )
        raw = result.stdout.strip()
        raw_body, _, status_line = raw.rpartition("\n")
        try:
            status = int(status_line.strip())
        except ValueError as exc:
            raise AgentError(
                f"Unparseable momo server response for {method} {path}: "
                f"{raw[-500:]}"
            ) from exc
        parsed: Any = None
        raw_body = raw_body.strip()
        if raw_body:
            try:
                parsed = json.loads(raw_body)
            except json.JSONDecodeError:
                parsed = raw_body
        return status, parsed

    # --- lifecycle -------------------------------------------------------

    def setup(self, session: Session) -> None:
        self._session = session
        if not isinstance(session.spec, DockerEnvironmentSpec):
            raise UnsupportedEnvironmentError(
                "The momo agent requires a Docker environment: the momo-agent "
                "server runs inside the workspace container."
            )
        server_dist = self.config.server_dist.expanduser().resolve()
        harness = self.config.harness_path.expanduser().resolve()
        if not (server_dist / "bin" / "server").is_file():
            raise AgentSetupError(
                f"momo server distribution not found at {server_dist} "
                "(expected bin/server; build it with "
                "`./gradlew :server:installDist`)."
            )
        if not (harness / "harness.yaml").is_file():
            raise AgentSetupError(
                f"momo harness not found at {harness} (expected harness.yaml)."
            )

        self._workspace_in_container = session.spec.docker.workdir
        self._data_tmp = tempfile.TemporaryDirectory(prefix="momo-data-")
        mounts: dict[str, dict[str, str]] = {
            str(server_dist): {"bind": SERVER_MOUNT, "mode": "ro"},
            str(harness): {"bind": HARNESS_MOUNT, "mode": "ro"},
            self._data_tmp.name: {"bind": DATA_MOUNT, "mode": "rw"},
        }
        self._runtime = session.spawn(
            mounts=mounts,
            env_vars={"AI_ROUTER_BASE_URL": self.config.ai_router_base_url},
            disable_setup=True,
        )

        start_cmd = (
            f"nohup {SERVER_MOUNT}/bin/server --port={self.config.port} "
            f"--data-dir={DATA_MOUNT} > {DATA_MOUNT}/server.log 2>&1 & "
            "echo started"
        )
        result = self._exec(start_cmd, timeout=60)
        if result.exit_code != 0:
            raise AgentSetupError(
                f"Starting the momo server failed: {result.stderr[-500:]}"
            )
        self._await_server_ready()
        self._create_momo_session()
        self.log.debug(
            "agent.momo.setup",
            workspace=str(session.working_dir),
            server_dist=str(server_dist),
            harness=str(harness),
            model=self.model_id,
        )

    def _await_server_ready(self) -> None:
        deadline = time.monotonic() + self.config.startup_timeout
        last_error = "no probe ran"
        while time.monotonic() < deadline:
            try:
                status, _ = self._api(
                    "GET",
                    f"/v1/sessions?workspace={self._workspace_in_container}",
                    timeout=5.0,
                )
            except AgentError as exc:
                last_error = str(exc)
            else:
                if status == 200:
                    return
                last_error = f"HTTP {status}"
            time.sleep(1.0)
        log_tail = ""
        if self._data_tmp is not None:
            log_path = Path(self._data_tmp.name) / "server.log"
            if log_path.exists():
                log_tail = log_path.read_text(errors="replace")[-1000:]
        raise AgentSetupError(
            f"momo server did not become ready within "
            f"{self.config.startup_timeout}s ({last_error}). "
            f"server.log tail: {log_tail}"
        )

    def _create_momo_session(self) -> None:
        status, info = self._api(
            "POST",
            "/v1/sessions",
            body={
                "harnessPath": HARNESS_MOUNT,
                "environment": {
                    "type": "local",
                    "workspace": self._workspace_in_container,
                },
                "title": self.problem_name,
            },
            timeout=60.0,
        )
        if status != 201:
            raise AgentSetupError(
                f"Creating a momo session failed with HTTP {status}: {info}"
            )
        self._momo_session_id = info["id"]

    # --- running ---------------------------------------------------------

    def run(self, task: str) -> None:
        if self._momo_session_id is None:
            raise AgentError("Momo agent has no session; setup() not run?")
        payload: dict[str, Any] = {"prompt": task, "model": self.model_id}
        if self.reasoning_effort is not None:
            payload["reasoningEffort"] = self.reasoning_effort
        status, info = self._api(
            "POST",
            f"/v1/sessions/{self._momo_session_id}/prompt",
            body=payload,
            timeout=60.0,
        )
        if status != 202:
            raise AgentError(
                f"Prompting the momo session failed with HTTP {status}: {info}"
            )

        self._last_run_finished = None
        self._poll_until_run_ends()

        self._ingest_new_events()
        finished = self._last_run_finished
        if finished is None:
            raise AgentError("momo run ended without a run_finished event")
        run_status = finished.get("status")
        if run_status == "error":
            error = finished.get("error") or {}
            raise AgentError(
                f"momo run failed: {error.get('message', 'unknown error')}"
            )
        if run_status != "completed":
            # The agent was cut off before it answered. This is NOT raised as
            # an error on purpose: 'turns_exhausted' and 'timeout' are momo's
            # own budgets doing their job, and the harness retry would resume
            # the session with a *fresh* per-run budget (turnsUsed counts per
            # run), quietly granting up to 3x the intended allowance and
            # making runs incomparable. The partial work stands and is
            # scored as-is; the marker below is what tells a reader that the
            # score reflects a truncated run rather than a finished one.
            self._incomplete_runs.append(str(run_status))
            self.log.warning(
                "momo run ended without completing",
                run_status=run_status,
                turns_used=finished.get("turnsUsed"),
                note=(
                    "scored as a truncated attempt; not retried because a "
                    "retry would reset momo's per-run budget"
                ),
            )
        self.log.debug(
            "momo run finished",
            run_status=run_status,
            turns_used=finished.get("turnsUsed"),
            steps=self.usage.steps,
            cost=self.usage.cost,
        )

    def _poll_until_run_ends(self) -> None:
        """Block until the momo session leaves the ``running`` state.

        Raises:
            AgentError: On repeated transport failures, an unusable
                status response, or if the run wedges (see
                ``run_idle_timeout``).
        """
        consecutive_failures = 0
        idle_limit = self.config.run_idle_timeout
        last_progress = time.monotonic()
        progress_marker = self._progress_marker()
        while True:
            time.sleep(self.config.poll_interval)
            self._ingest_new_events()

            # Any new event byte counts as progress. A run that keeps
            # reporting "running" while writing nothing at all is wedged;
            # without this the loop would poll forever and pin the worker.
            marker = self._progress_marker()
            if marker != progress_marker:
                progress_marker = marker
                last_progress = time.monotonic()
            elif (
                idle_limit > 0
                and time.monotonic() - last_progress > idle_limit
            ):
                raise AgentError(
                    f"momo run produced no events for {idle_limit:.0f}s while "
                    f"still reporting 'running'; treating it as wedged"
                )
            try:
                status, info = self._api(
                    "GET",
                    f"/v1/sessions/{self._momo_session_id}",
                    timeout=10.0,
                )
            except AgentError:
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    raise
                continue
            consecutive_failures = 0
            if status != 200:
                raise AgentError(
                    f"Polling the momo session failed with HTTP {status}: "
                    f"{info}"
                )
            # A healthy server always answers with a status-bearing object.
            # Anything else (proxy error page, truncated body) must surface
            # as an AgentError, not as a TypeError/KeyError escaping as a
            # non-agent crash.
            if not isinstance(info, dict) or "status" not in info:
                raise AgentError(
                    f"Unexpected momo session payload while polling: "
                    f"{str(info)[:200]}"
                )
            if info["status"] != "running":
                return

    def _progress_marker(self) -> tuple[int, int]:
        """Cheap liveness signal: how much event data has been consumed.

        Returns:
            (number of event files seen, total bytes consumed across them).
        """
        return (
            len(self._event_offsets),
            sum(self._event_offsets.values()),
        )

    def _ingest_new_events(self) -> None:
        """Parse new event-log lines from every momo session (subagents too)."""
        if self._data_tmp is None:
            return
        sessions_dir = Path(self._data_tmp.name) / "sessions"
        if not sessions_dir.is_dir():
            return
        for events_file in sorted(sessions_dir.glob("*/events.jsonl")):
            offset = self._event_offsets.get(events_file, 0)
            try:
                with events_file.open("rb") as handle:
                    handle.seek(offset)
                    chunk = handle.read()
            except OSError:
                continue
            if not chunk:
                continue
            # Only consume complete lines; a partial tail is re-read later.
            complete, _, partial = chunk.rpartition(b"\n")
            if not complete:
                continue
            self._event_offsets[events_file] = offset + len(complete) + 1
            is_root = (
                self._momo_session_id is not None
                and events_file.parent.name == self._momo_session_id
            )
            for line in complete.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._consume_event(event, is_root=is_root)

    def _consume_event(self, event: dict[str, Any], is_root: bool) -> None:
        kind = event.get("type")
        if kind == "llm_call_finished":
            usage = event.get("usage", {})
            tokens = TokenUsage(
                input=usage.get("prompt_tokens", 0),
                output=usage.get("completion_tokens", 0),
                cache_read=usage.get("cache_read_tokens", 0),
                reasoning=usage.get("reasoning_tokens", 0),
            )
            cost = self.pricing.get_cost(tokens) if self.pricing else 0.0
            self.usage.step(cost, tokens)
        elif kind == "run_finished" and is_root:
            self._last_run_finished = event

    # --- checkpoint boundaries and teardown -------------------------------

    def reset(self) -> None:
        """Start a fresh conversation: new momo session, same workspace."""
        if self._momo_session_id is not None:
            try:
                self._api(
                    "POST",
                    f"/v1/sessions/{self._momo_session_id}/close",
                    timeout=60.0,
                )
            except AgentError:
                self.log.warning(
                    "Closing the momo session failed; continuing",
                    session_id=self._momo_session_id,
                )
            self._momo_session_id = None
        self._last_run_finished = None
        self._incomplete_runs = []
        self._create_momo_session()

    def save_artifacts(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if self._incomplete_runs:
            # A run that never answered still produces a snapshot that gets
            # scored. Record it next to the logs so the resulting score can
            # be read as "truncated attempt", not "the model did poorly".
            (path / "incomplete_runs.json").write_text(
                json.dumps({"statuses": self._incomplete_runs}, indent=2)
            )
        if self._data_tmp is None:
            return
        data_dir = Path(self._data_tmp.name)
        for log_file in data_dir.glob("*.log"):
            shutil.copy2(log_file, path / log_file.name)
        sessions_dir = data_dir / "sessions"
        if sessions_dir.is_dir():
            shutil.copytree(
                sessions_dir, path / "sessions", dirs_exist_ok=True
            )

    def cleanup(self) -> None:
        if self._momo_session_id is not None and self._runtime is not None:
            try:
                self._api(
                    "POST",
                    f"/v1/sessions/{self._momo_session_id}/close",
                    timeout=30.0,
                )
            except AgentError:
                pass
            self._momo_session_id = None
        if self._data_tmp is not None:
            self._data_tmp.cleanup()
            self._data_tmp = None


register_agent("momo", MomoAgent)
