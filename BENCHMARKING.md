# Benchmarking momo-agent and GitLab Duo CLI on this fork

This fork adds two agents to SlopCodeBench: **`momo`** (the
[momo-agent](https://github.com/mltheuser/momo-agent) server, run *inside* the
workspace container) and **`duo_cli`** (GitLab's headless Duo CLI). This is the
runbook: setup, per-agent prerequisites, running, reading results, stopping.

Verified on macOS + colima, 2026-09-02.

## One-time machine setup

```bash
colima start
export DOCKER_HOST=unix://$HOME/.colima/default/docker.sock
export TMPDIR=$HOME/.cache/scbench-tmp
mkdir -p "$TMPDIR"

uv sync -p 3.12          # Homebrew's 3.14 breaks the fastuuid build
uv run slop-code sync    # installs the 36-problem catalog into ~/.cache/scbench
```

**Export `DOCKER_HOST` and `TMPDIR` on every invocation, not just setup.**
`docker-py` ignores the Docker CLI context, and `/tmp` + `/var/folders` are not
shared into the colima VM — a bind mount rooted there is created *inside* the
VM, so container writes succeed while the host sees an empty directory. `momo`
reads usage and artifacts back through such a mount, so it fails with
`momo run ended without a run_finished event` and the checkpoint is recorded as
`state: error`.

## Running momo

`momo` mounts a host-built server distribution and harness folder read-only
into the container, starts the server there, and drives its HTTP API over
`docker exec`. Usage is parsed live from the server's `events.jsonl` via a
shared data mount, subagent sessions included.

```bash
# 1. ai-router on the host (momo's only LLM backend; reached at
#    host.docker.internal:8787 from inside the container)
cd ~/Develop/Private/ai-router && set -a && source .env && set +a && ./bin/ai-router serve &

# 2. The server distribution the adapter mounts
cd ~/Develop/Private/momo-codes/momo-agent && ./gradlew :server:installDist

# 3. Anthropic key: required by the harness's credential machinery for
#    provider "anthropic". momo itself never reads it.
export ANTHROPIC_API_KEY="$(grep -m1 '^AI_ROUTER_ANTHROPIC_API_KEY=' ~/Develop/Private/ai-router/.env | cut -d= -f2-)"
```

```bash
uv run slop-code run --agent momo --model anthropic/sonnet-5 \
  --environment configs/environments/docker-python3.12-uv.yaml \
  --prompt configs/prompts/just-solve.jinja \
  --problem file_backup
```

Host paths live in [`configs/agents/momo.yaml`](configs/agents/momo.yaml).
Model names map automatically (`anthropic/sonnet-5` →
`claude-sonnet-5:cloud@anthropic`); override via `agent_specific.momo.model`.

**Truncated runs are scored, not retried.** A run can end `turns_exhausted`,
`timeout` or `stopped` — the agent never answered. The harness scores whatever
was built rather than retrying, because a retry resumes the session with a
*fresh* per-run turn budget and would quietly grant several times the intended
allowance. Such runs are flagged in
`<checkpoint>/agent/incomplete_runs.json`: if that file exists, read the score
as "the agent was cut off", not "the model wrote weak code".

**A momo run outlives the harness.** The run lives in the container and is only
*observed*; if the harness dies, it keeps working (and spending) until momo's
own budgets end it. See [Stopping runs](#stopping-runs).

## Running GitLab Duo CLI

`duo_cli` bakes the self-contained `duo` binary into the agent image (version
pinned in [`configs/agents/duo_cli.yaml`](configs/agents/duo_cli.yaml)). Each
checkpoint is one headless `duo run --output-format json`; tool calls are
auto-approved and no git repo is needed.

```bash
# GitLab token whose user resolves a Duo-entitled namespace.
# glab's own token works; personal namespaces are NOT supported.
export GITLAB_TOKEN="$(grep -m1 '^\s*token:' ~/.config/glab-cli/config.yml | awk '{print $2}')"
# Anthropic key still required by the harness; duo never reads it.
```

```bash
uv run slop-code run --agent duo_cli --model anthropic/sonnet-5 \
  --environment configs/environments/docker-python3.12-uv.yaml \
  --prompt configs/prompts/just-solve.jinja \
  --problem file_backup
```

Model names fold separators to underscores (`claude-sonnet-5` →
`claude_sonnet_5`); override via `agent_specific.duo_cli.model`. Thinking
presets are ignored — the CLI has no such control.

**Reading duo results:**

- Duo emits **no token usage or cost** — cost columns are always 0.
- `steps` counts tool calls parsed from the stderr log, not from the JSON
  transcript's `elements` array (which compacts history and omits subagents:
  38 elements observed for 1069 actual tool calls).
- A failed run is retried by resuming the same Duo session. If the session id
  could not be observed, the retry is **refused** and the checkpoint fails
  loudly — a blank session prompted with "Continue from where you left off."
  has no task context and would silently produce misdirected work.

## Run limits

**The harness imposes none.** `step_limit`, `cost_limit` and `net_cost_limit`
are `0` for both agents, and agent runs are launched with no wall-clock
deadline. A run ends when the agent's own budgets end it — for momo: 256 turns,
3-day wall clock, 24h per tool call.

The one opt-in bound is momo's `run_idle_timeout`, which ships **disabled
(`0`)**. It aborts a run reporting `running` while emitting no events at all.
Enable it only to fail faster than momo's own budgets; any value below momo's
24h tool timeout risks killing legitimate work.

`max_retries` (default 2) only adds attempts *after* a failure; it never cuts a
healthy run short.

## Subset runs

The full 36-problem catalog is an overnight, three-digit-dollar affair. Three
combinable ways to narrow it:

1. **`--problem`** (repeatable) — pick problems by hand.
2. **Run configs** — [`lite.yaml`](configs/runs/lite.yaml) (5 problems) and
   [`lite_under20.yaml`](configs/runs/lite_under20.yaml) (2, historically
   under $20):
   `uv run slop-code run --config configs/runs/lite.yaml --agent momo --model anthropic/sonnet-5`
3. **`--num-workers N`** — problems in parallel, one container each. For momo,
   2–3 is a sensible laptop ceiling (each worker is a JVM plus ai-router load).

Reproducing HumanLayer's
[Opus 5 subset](https://github.com/humanlayer/advanced-context-engineering-for-coding-agents/blob/main/benchmarking-opus-5-on-slop-code-bench.md)
(3 problems, 17 checkpoints):

```bash
uv run slop-code run --agent momo --model anthropic/sonnet-5 \
  --environment configs/environments/docker-python3.12-uv.yaml \
  --prompt configs/prompts/just-solve.jinja \
  --problem circuit_eval --problem database_migration --problem dynamic_config_service_api
```

An interrupted run is not wasted: `slop-code run --resume outputs/<run-dir>`
continues from the last completed checkpoint (the installed catalog commit must
match the run's recorded one).

## Reading results

Outputs land in `outputs/<model>/<agent>_<prompt>_<thinking>_<timestamp>/`, one
folder per problem and checkpoint, each with the workspace `snapshot/`,
`evaluation.json`, and agent artifacts under `agent/`.

Checkpoints are evaluated inline by default; `uv run slop-code eval
outputs/<run-dir>/` re-runs or fills in the suites. Quality metrics come from
`slop-code metrics`. Keep `pass_policy` fixed when comparing agents — strict
("all-cases") and lenient ("core-cases") rates are not comparable.

**Distinguish a broken run from a bad solution** — both leave
`passed_policy: false`, so check `summary.state` in `run_info.yaml`:

| `state` | Meaning |
|---|---|
| `completed` | All checkpoints ran and passed. |
| `failed` | Agent worked fine; its code failed tests. **A valid result.** |
| `error` | Agent itself broke. **Discard**; see `error_message`. |

Per-checkpoint states appear under `summary.checkpoints` as `ran` / `error` /
`skipped`. Also sanity-check that `steps` (and momo's `cost`) are non-zero —
zero for both usually means a bind mount was invisible (see `TMPDIR` above).

## Stopping runs

`pkill -f 'slop-code run'` kills only the CLI: the multiprocessing **worker**
driving the agent survives orphaned and advances to the next checkpoint. Kill
by venv path, then remove containers:

```bash
pkill -9 -f 'slop-code-bench/.venv'
docker ps -q --filter ancestor=slop-code:python3.12 | xargs -r docker rm -f
docker ps -q --filter ancestor=slop-code:duo_cli-9.17.0-python3.12 | xargs -r docker rm -f
```

For momo, stop the in-flight runs *first* so the agent isn't left working
unobserved in a container you're about to remove:

```bash
c=$(docker ps -q --filter ancestor=slop-code:python3.12 | head -1)
for sid in $(docker exec $c ls /momo-data/sessions); do
  docker exec $c curl -s -X POST "http://127.0.0.1:8420/v1/sessions/$sid/stop"
done
docker rm -f $c
```
