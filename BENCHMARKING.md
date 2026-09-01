# Benchmarking momo-agent and GitLab Duo CLI on this fork

This fork adds two agents to SlopCodeBench: **`momo`** (the
[momo-agent](https://github.com/mltheuser/momo-agent) server, run *inside*
the workspace container) and **`duo_cli`** (GitLab's headless Duo CLI).
This document is the runbook: machine setup, per-agent prerequisites, full
and subset runs, evaluation, and the operational gotchas found while
smoke-testing both agents (2026-09-01, macOS + colima).

## One-time machine setup

```bash
# Docker daemon (colima) and the two env vars every run needs.
colima start
export DOCKER_HOST=unix://$HOME/.colima/default/docker.sock   # docker-py ignores the CLI context
export TMPDIR=$HOME/.cache/scbench-tmp                        # /var/folders + /tmp are NOT shared
mkdir -p "$TMPDIR"                                            #   with the colima VM; mounts from
                                                              #   there become empty root stubs
# Python 3.12 (Homebrew's 3.14 breaks the fastuuid build; .python-version pins it)
uv sync -p 3.12

# Install the managed problem catalog (36 problems) into ~/.cache/scbench
uv run slop-code sync
```

Both `DOCKER_HOST` and `TMPDIR` are needed on **every** invocation, not just
setup. `/tmp` and `/var/folders` are not shared into the colima VM, so a bind
mount rooted there is created inside the VM instead: the container writes
succeed, and the host sees an empty directory. For `momo`, whose usage
tracking and artifacts are read back from a mounted data dir, this means no
events are ever ingested — the run fails with
`momo run ended without a run_finished event` and the checkpoint is recorded
as an agent error (`state: error`).

Sanity check after any run: `steps` and (for momo) `cost` should be non-zero
in `run_info.yaml`. Zero for both usually means the mount was invisible.

## Running momo

The `momo` agent mounts a host-built momo-agent server distribution and a
harness folder read-only into the harness-owned container, starts the server
inside it, and drives its HTTP API via `curl` over `docker exec`. Usage
(steps, tokens, cost) is parsed live from the server's `events.jsonl`
through a shared data mount, subagent sessions included.

Prerequisites per run:

```bash
# 1. ai-router on the host with a provider key (momo's only LLM backend;
#    the container reaches it via host.docker.internal:8787)
cd ~/Develop/Private/ai-router && set -a && source .env && set +a && ./bin/ai-router serve &

# 2. The server distribution the adapter mounts
cd ~/Develop/Private/momo-codes/momo-agent && ./gradlew :server:installDist

# 3. The Anthropic key env var (required by the harness's credential
#    machinery for provider "anthropic"; momo itself never reads it)
export ANTHROPIC_API_KEY="$(grep -m1 '^AI_ROUTER_ANTHROPIC_API_KEY=' ~/Develop/Private/ai-router/.env | cut -d= -f2-)"
```

Host paths for the distribution and harness folder live in
[`configs/agents/momo.yaml`](configs/agents/momo.yaml) — adjust them if the
checkouts move. Model names map automatically
(`anthropic/sonnet-5` → `claude-sonnet-5:cloud@anthropic`); override per
model via `agent_specific.momo.model` in the model YAML.

```bash
uv run slop-code run --agent momo --model anthropic/sonnet-5 \
  --environment configs/environments/docker-python3.12-uv.yaml \
  --prompt configs/prompts/just-solve.jinja \
  --problem file_backup
```

**Truncated runs are scored, not retried.** A momo run can end without
answering: `turns_exhausted` (256 turns), `timeout` (3-day wall clock) or
`stopped`. The harness keeps whatever the agent built and scores it, rather
than retrying — a retry resumes the session with a *fresh* per-run turn
budget, which would quietly grant several times the intended allowance and
make runs incomparable. Such runs are flagged in
`<checkpoint>/agent/incomplete_runs.json`; if that file exists, read the
score as "the agent was cut off", not "the model wrote weak code".

**Wedged runs are abandoned.** `run_idle_timeout` (default 1h, in
`configs/agents/momo.yaml`) aborts a run that keeps reporting `running`
while writing no events at all. It is not a runtime budget — any new event
resets the window, so long tool calls are safe — it only stops a dead run
from pinning a worker for the rest of the benchmark. Set it to `0` to
disable.

**Known gotcha (accepted deliberately, 2026-09-01):** a momo run lives in
the container and merely gets *observed* by the harness — if the harness
process dies, the run keeps working (and spending) until momo's own library
budgets end it. After any harness crash: stop runs by hand, then remove the
container:

```bash
c=$(docker ps -q --filter ancestor=slop-code:python3.12 | head -1)
for sid in $(docker exec $c ls /momo-data/sessions); do
  docker exec $c curl -s -X POST "http://127.0.0.1:8420/v1/sessions/$sid/stop"
done
docker rm -f $c
```

## Running GitLab Duo CLI

The `duo_cli` agent bakes the self-contained `duo` binary (no Node.js) into
the agent image from the gitlab-lsp generic package registry — the version
is pinned in [`configs/agents/duo_cli.yaml`](configs/agents/duo_cli.yaml).
Each checkpoint is one headless `duo run --output-format json`; headless
runs auto-approve all tool calls and need no git repo. A failed run raises,
and the harness retry resumes the same Duo session via
`--existing-session-id`.

Prerequisites per run:

```bash
# GitLab token whose user resolves a Duo-entitled namespace
# (glab's own token works; personal namespaces are NOT supported)
export GITLAB_TOKEN="$(grep -m1 '^\s*token:' ~/.config/glab-cli/config.yml | awk '{print $2}')"
# The harness still demands an Anthropic key for the model's provider —
# export ANTHROPIC_API_KEY as above; duo never reads it.
```

```bash
uv run slop-code run --agent duo_cli --model anthropic/sonnet-5 \
  --environment configs/environments/docker-python3.12-uv.yaml \
  --prompt configs/prompts/just-solve.jinja \
  --problem file_backup
```

Model names fold separators to underscores (`claude-sonnet-5` →
`claude_sonnet_5`, a valid GitLab identifier); override via
`agent_specific.duo_cli.model`. Reasoning-effort/thinking presets are
ignored — the CLI has no such control.

**Limitations to keep in mind when reading results:**

- Duo emits **no token usage or cost anywhere** — cost columns are 0.
- `steps` counts started tool calls parsed from the stderr log. The JSON
  transcript's `elements` array is *not* used: it compacts history and
  omits subagent activity (observed under-reporting: 38 elements for 1069
  actual tool calls).
- Turn/time limits are enforced server-side by the flow; the CLI has none,
  so long checkpoints are normal (~30+ min observed on `file_backup`).

## Subset runs (quick evaluation rounds)

Running the full 36-problem catalog is an overnight, three-digit-dollar
affair. Three subsetting mechanisms, combinable:

1. **`--problem` flags** (repeatable) — pick problems by hand.
2. **Run configs with a `problems:` list** —
   [`configs/runs/lite.yaml`](configs/runs/lite.yaml) (5-problem
   debug/experimentation subset) and
   [`configs/runs/lite_under20.yaml`](configs/runs/lite_under20.yaml)
   (2 problems, historically under $20):
   `uv run slop-code run --config configs/runs/lite.yaml --agent momo --model anthropic/sonnet-5`
3. **`--num-workers N`** — problems in parallel, one container each. For
   momo, 2–3 is a sensible ceiling on a laptop (each worker is a JVM server
   plus ai-router traffic).

A good reference for this style of quick round is HumanLayer's
[Benchmarking Opus 5 on SlopCodeBench](https://github.com/humanlayer/advanced-context-engineering-for-coding-agents/blob/main/benchmarking-opus-5-on-slop-code-bench.md):
they picked 3 problems (17 checkpoints) mixing difficulty labels and scored
strict pass rate. Reproducing their subset here:

```bash
uv run slop-code run --agent momo --model anthropic/sonnet-5 \
  --environment configs/environments/docker-python3.12-uv.yaml \
  --prompt configs/prompts/just-solve.jinja \
  --problem circuit_eval --problem database_migration --problem dynamic_config_service_api
```

An interrupted run is not wasted: `slop-code run --resume outputs/<run-dir>`
continues from the last completed checkpoint (the installed problem-catalog
commit must match the run's recorded one).

## Evaluation and metrics

Checkpoints are evaluated inline during the run by default;
`uv run slop-code eval outputs/<run-dir>/` re-runs or fills in the pytest
suites (Core must pass; earlier checkpoints' tests become regression tests
for later ones). Quality/"slop" metrics come from `slop-code metrics`.
When comparing agents, keep the `pass_policy` fixed across runs — strict
("all-cases", the article's metric) and lenient ("core-cases") pass rates
are not comparable.

Outputs land under `outputs/<model>/<agent>_<prompt>_<thinking>_<timestamp>/`,
one folder per problem, one per checkpoint, each with the workspace
`snapshot/`, `evaluation.json`, and the agent's own artifacts under `agent/`
(momo: momo event logs and server log; duo: per-run result JSON and stderr).

## Stopping runs safely

`pkill -f 'slop-code run'` kills only the CLI process — the multiprocessing
**worker** driving the agent survives orphaned and keeps going (it will even
advance to the next checkpoint). Kill by venv path instead, then clean up
containers:

```bash
pkill -9 -f 'slop-code-bench/.venv'
docker ps -q --filter ancestor=slop-code:python3.12 | xargs -r docker rm -f      # momo
docker ps -q --filter "ancestor=slop-code:duo_cli-9.17.0-python3.12" | xargs -r docker rm -f  # duo
```

For momo, stop the in-flight runs first (see the momo gotcha above) so the
agent isn't left working unobserved inside a container you're about to keep.
