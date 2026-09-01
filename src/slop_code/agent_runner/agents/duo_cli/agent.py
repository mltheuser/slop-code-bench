"""GitLab Duo CLI agent: runs `duo run` headless inside the workspace container.

The Duo CLI (`glab duo cli` in production; here the standalone `duo` binary
baked into the agent image by docker.j2) executes one goal per invocation and
exits. `--output-format json` emits a single JSON document on stdout with the
run's transcript (`elements`), final `response`, and `sessionId`; logs go to
stderr. Headless runs auto-approve all tool calls and require no git repo in
the workspace.

Requires:
- A Docker environment spec (the CLI runs inside the container via exec).
- A GitLab token in the host environment (default: ``GITLAB_TOKEN``) whose
  user resolves a Duo-entitled namespace.
- Network access from the container to the GitLab host and cloud.gitlab.com.

Limitations, accepted deliberately:
- The CLI reports no token usage or cost anywhere, so only steps are tracked:
  one per started tool call, counted live from the stderr log markers. The
  JSON transcript's `elements` array is NOT used for counting — it compacts
  long histories and omits subagent activity, under-reporting actual tool
  calls by more than an order of magnitude on busy runs.
- The CLI has no run-level time or turn limits of its own; limits are
  enforced server-side by the flow.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path
from typing import Any, Literal

from jinja2 import Template

from slop_code.agent_runner.agent import Agent
from slop_code.agent_runner.agent import AgentConfigBase
from slop_code.agent_runner.credentials import ProviderCredential
from slop_code.agent_runner.models import AgentError
from slop_code.agent_runner.models import AgentSetupError
from slop_code.agent_runner.models import UnsupportedEnvironmentError
from slop_code.common.llms import ModelDefinition
from slop_code.common.llms import ThinkingPreset
from slop_code.execution import Session
from slop_code.execution.docker_runtime.models import DockerEnvironmentSpec
from slop_code.execution.runtime import RuntimeResult

# One step per started tool call, counted from the stderr log stream.
_TOOL_STARTED_MARKER = "[RunController] Tool started:"

# Keep accumulated stderr bounded; the interesting part is the tail.
_MAX_STDERR_BYTES = 4 * 1024 * 1024

# The session id also appears in the stderr log stream, seconds into the run
# and long before the result document is written, e.g.
#   [DuoWorkflowNodeExecutor][6832711] Executor started
# Reading it from here means a retry can resume the session even when the run
# ends without a parseable result document (killed mid-run, truncated stdout).
_SESSION_ID_PATTERN = re.compile(r"\[DuoWorkflowNodeExecutor\]\[(\d+)\]")


class DuoCliConfig(AgentConfigBase, agent_type="duo_cli"):
    """Configuration for the GitLab Duo CLI agent.

    Attributes:
        version: duo-cli release to bake into the agent image (generic
            package registry version, e.g. "9.17.0").
        gitlab_url: GitLab instance the CLI talks to.
        gitlab_token_env: Host environment variable holding the GitLab
            token passed into the container as GITLAB_TOKEN.
    """

    type: Literal["duo_cli"] = "duo_cli"
    docker_template: Path = Path(__file__).parent / "docker.j2"
    gitlab_url: str = "https://gitlab.com"
    gitlab_token_env: str = "GITLAB_TOKEN"

    def get_docker_file(self, base_image: str) -> str | None:
        if self.docker_template is None:
            return None
        template = self.docker_template.read_text()
        return Template(template).render(
            base_image=base_image, version=self.version
        )


class DuoCliAgent(Agent):
    """Agent wrapper around headless `duo run` in the container."""

    def __init__(
        self,
        config: DuoCliConfig,
        model_id: str | None,
        problem_name: str,
        pricing,
        verbose: bool,
    ) -> None:
        super().__init__(
            "duo_cli", problem_name, config.cost_limits, pricing, verbose
        )
        self.config = config
        self.model_id = model_id
        self._session: Session | None = None
        self._runtime = None
        self._image: str | None = None
        # Session id of the last run since reset(); a follow-up run() in the
        # same checkpoint (the harness's retry path) resumes it.
        self._resume_session_id: str | None = None
        # (result_json_text, stderr_text) per run since reset(), for artifacts.
        self._runs: list[tuple[str, str]] = []

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
    ) -> "DuoCliAgent":
        assert isinstance(config, DuoCliConfig)
        duo_settings = model.agent_specific.get("duo_cli", {})
        # GitLab model identifiers are the benchmark's internal names with
        # separators folded to underscores (claude-sonnet-5 -> claude_sonnet_5).
        model_id = duo_settings.get(
            "model", re.sub(r"[-.]", "_", model.internal_name)
        )
        agent = cls(
            config=config,
            model_id=model_id,
            problem_name=problem_name,
            pricing=model.pricing,
            verbose=verbose,
        )
        agent._image = image
        if thinking_preset is not None:
            agent.log.warning(
                "Duo CLI has no reasoning-effort control; ignoring thinking "
                "preset",
                thinking_preset=thinking_preset,
            )
        return agent

    def setup(self, session: Session) -> None:
        self._session = session
        if not isinstance(session.spec, DockerEnvironmentSpec):
            raise UnsupportedEnvironmentError(
                "The duo_cli agent requires a Docker environment."
            )
        token = os.environ.get(self.config.gitlab_token_env)
        if not token:
            raise AgentSetupError(
                f"GitLab token not found in ${self.config.gitlab_token_env}; "
                "export it on the host (e.g. from `glab auth status`)."
            )
        self._runtime = session.spawn(
            env_vars={
                "GITLAB_TOKEN": token,
                "GITLAB_URL": self.config.gitlab_url,
                "DUO_WORKFLOW_TELEMETRY_ENABLED": "false",
                "NO_COLOR": "1",
            },
            image=self._image,
            disable_setup=True,
        )
        result = self._collect("duo --version", timeout=120)
        if result.exit_code != 0:
            raise AgentSetupError(
                f"duo binary unusable in the agent image: "
                f"{result.stderr[-500:]}"
            )
        self.log.debug(
            "agent.duo_cli.setup",
            workspace=str(session.working_dir),
            duo_version=result.stdout.strip(),
            model=self.model_id,
        )

    def _collect(self, command: str, timeout: float | None) -> RuntimeResult:
        """Run a command in the container and return its buffered result."""
        if self._runtime is None:
            raise AgentError("Duo agent has not been set up with a runtime")
        result: RuntimeResult | None = None
        for event in self._runtime.stream(command, env={}, timeout=timeout):
            if event.kind == "finished":
                result = event.result
        if result is None:
            raise AgentError(f"Container exec yielded no result: {command}")
        return result

    def _observe_session_id(self, text: str) -> None:
        """Record the session id seen in a chunk of the stderr log stream.

        The id is stable for the lifetime of one `duo run`, so the first match
        wins and later chunks are ignored.

        Args:
            text: A chunk of stderr output from the running CLI.
        """
        if self._resume_session_id is not None:
            return
        match = _SESSION_ID_PATTERN.search(text)
        if match:
            self._resume_session_id = match.group(1)
            self.log.debug(
                "agent.duo_cli.session_observed",
                session_id=self._resume_session_id,
            )

    def retry(self) -> None:
        """Continue the current checkpoint after a failed run.

        Resuming requires a session id. Without one the CLI would start a
        blank session, and the retry prompt alone ("Continue from where you
        left off.") carries no task, leaving the agent to guess at a workspace
        it has no memory of. Failing loudly is better than burning a
        checkpoint on that.

        Raises:
            AgentError: If no session id was observed for the failed run.
        """
        if not self._resume_session_id:
            raise AgentError(
                "Cannot retry duo run: no session id was observed for the "
                "failed run, so the retry would start a fresh session with "
                "no task context."
            )
        self.log.info(
            "agent.duo_cli.retry",
            session_id=self._resume_session_id,
        )
        super().retry()

    def run(self, task: str) -> None:
        if self._runtime is None:
            raise AgentError("Duo agent has not been set up with a runtime")
        cmd = (
            f"duo run --goal {shlex.quote(task)} "
            "--output-format json --log-level info"
        )
        if self.model_id:
            cmd += f" --model {shlex.quote(self.model_id)}"
        if self._resume_session_id:
            cmd += f" --existing-session-id {shlex.quote(self._resume_session_id)}"

        stderr_parts: list[str] = []
        stderr_bytes = 0
        result: RuntimeResult | None = None
        for event in self._runtime.stream(cmd, env={}, timeout=None):
            if event.kind == "stderr" and event.text:
                if stderr_bytes < _MAX_STDERR_BYTES:
                    stderr_parts.append(event.text)
                    stderr_bytes += len(event.text)
                self.usage.steps += event.text.count(_TOOL_STARTED_MARKER)
                self._observe_session_id(event.text)
            elif event.kind == "finished":
                result = event.result
        if result is None:
            raise AgentError("duo run yielded no result")

        stderr_text = "".join(stderr_parts) or result.stderr
        # Late safety net: with a truncated stderr stream the id may only be
        # in the buffered copy.
        self._observe_session_id(stderr_text)
        stdout_text = result.stdout.strip()
        self._runs.append((stdout_text, stderr_text))

        if not stdout_text:
            raise AgentError(
                f"duo run wrote no result document (exit "
                f"{result.exit_code}). stderr tail: {stderr_text[-800:]}"
            )
        try:
            document = json.loads(stdout_text)
        except json.JSONDecodeError as exc:
            raise AgentError(
                f"duo run stdout is not valid JSON "
                f"({len(stdout_text)} chars, exit {result.exit_code}): "
                f"{stdout_text[-500:]}"
            ) from exc

        session_id = document.get("sessionId")
        if session_id:
            self._resume_session_id = str(session_id)
        elements = document.get("elements", [])

        if document.get("status") != "success":
            raise AgentError(
                f"duo run failed: {document.get('error', 'unknown error')} "
                f"(exit {result.exit_code})"
            )
        failed_tools = [
            element.get("name")
            for element in elements
            if element.get("type") == "tool"
            and element.get("state", {}).get("type") == "error"
        ]
        self.log.debug(
            "duo run finished",
            session_id=session_id,
            steps=self.usage.steps,
            elements=len(elements),
            failed_tools=failed_tools,
            response=str(document.get("response", ""))[:200],
        )

    def reset(self) -> None:
        """Forget the conversation: the next run starts a fresh session."""
        self._resume_session_id = None
        self._runs = []

    def save_artifacts(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for index, (stdout_text, stderr_text) in enumerate(self._runs, 1):
            (path / f"run_{index}.result.json").write_text(stdout_text)
            (path / f"run_{index}.stderr.log").write_text(stderr_text)

    def cleanup(self) -> None:
        # The runtime (container) is owned and cleaned up by the session;
        # a finished `duo run` leaves nothing behind.
        pass


from slop_code.agent_runner.registry import register_agent  # noqa: E402

register_agent("duo_cli", DuoCliAgent)
