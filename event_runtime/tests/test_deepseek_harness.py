from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from harbor.agents.factory import AgentFactory
from harbor.agents.installed.deepseek_harness import DeepSeekHarness
from harbor.models.agent.name import AgentName


ROOT = Path(__file__).resolve().parents[2]
CONTAINER = ROOT / "event_runtime" / "container"


def test_factory_registers_deepseek_harness() -> None:
    assert AgentFactory.get_agent_class(AgentName.DEEPSEEK_HARNESS) is DeepSeekHarness
    assert DeepSeekHarness.name() == "deepseek-harness"


def test_harbor_adapter_checks_the_pinned_local_runtime_graph() -> None:
    version_command = DeepSeekHarness.get_version_command(None)  # type: ignore[arg-type]
    assert (
        "/opt/deepseek-harness/node_modules/"
        "@deepseek-ai/dsh-sdk-jsonrpc-demo/package.json"
    ) in version_command
    assert "/usr/local/lib/node_modules/@deepseek-ai" not in version_command


def test_deepseek_harness_rejects_nonbenchmark_reasoning_effort(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must be max"):
        DeepSeekHarness(
            logs_dir=tmp_path,
            model_name="deepseek/deepseek-v4-flash-vision-exp",
            reasoning_effort="high",
        )


def test_minimal_cordis_contract_is_sealed() -> None:
    config = (CONTAINER / "deepseek-harness-minimal.cordis.yml").read_text()
    assert "deepseek/deepseek-v4-flash-vision-exp" in config
    assert "contextWindow: 1048576" in config
    assert "maxTokens: 384000" in config
    assert "inputModalities: [text, image]" in config
    assert "thinking: enabled" in config
    assert "reasoningEffort: max" in config
    assert "@deepseek-ai/dsh-llm-retry" in config
    assert "mode: normal" in config
    assert "maxRetries: 5" in config
    for code in {
        "EMPTY_RESPONSE",
        "RATE_LIMIT",
        "SERVER",
        "TIMEOUT",
        "TRANSPORT",
        "STREAM_CLOSED",
    }:
        assert f"- {code}" in config
    assert "initialDelayMs: 500" in config
    assert "maxDelayMs: 10000" in config
    assert "jitterRatio: 0.1" in config
    assert "persona: You are a helpful software engineer assistant." in config
    assert "includeHarnessIdentity: false" in config
    assert "includeRuntimeContext: false" in config
    assert "workspaceContext: false" in config
    assert "enabled: false" in config
    assert "@deepseek-ai/dsh-tool-bash-persistent" in config
    assert "maxOutputChars: 16000" in config
    assert "@deepseek-ai/dsh-tool-str-replace-editor" in config
    assert "@deepseek-ai/dsh-session-persistence-jsonl" in config
    assert "@deepseek-ai/dsh-goal" in config
    assert "@deepseek-ai/dsh-tool-goal" in config
    assert "@deepseek-ai/dsh-goal-round-driver" in config
    assert "defaultMaxGoalRounds: !!js Number(process.env.DSH_GOAL_MAX_ROUNDS ?? 256)" in config
    assert "blockedAfterConsecutiveRounds: 3" in config
    assert "sprint-deepseek-goal-bootstrap.mjs" in config
    assert "compression: none" in config
    assert "dsh-tool-jobs" not in config
    assert "dsh-compaction" not in config


def test_wrapper_seals_route_and_wire_parameters() -> None:
    wrapper = (CONTAINER / "sprint-deepseek-harness-exec-wrapper.sh").read_text()
    marker = "REQUEST_CONTRACT='"
    start = wrapper.index(marker) + len(marker)
    contract = json.loads(wrapper[start : wrapper.index("'", start)])
    assert contract == {
        "model": "deepseek/deepseek-v4-flash-vision-exp",
        "stream": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 384_000,
        "reasoning_effort": "max",
    }
    assert "PROVIDER_ENDPOINT:-deepseek" in wrapper
    assert '--provider-endpoint "$PROVIDER_ENDPOINT"' in wrapper
    assert '--request-contract-json "$REQUEST_CONTRACT"' in wrapper
    assert '--upstream-api-key-stdin' in wrapper
    assert 'unset OPENROUTER_API_KEY' in wrapper
    assert 'export OPENROUTER_API_KEY=sprint-local-proxy-token' in wrapper
    assert 'PROXY_PROCESS_FILE="$AGENT_STATE_DIR/openrouter-proxy.pid"' in wrapper
    assert 'pending_request_count' in wrapper
    assert 'fail_closed_proxy_recovery' in wrapper
    assert "DeepSeek Harness native goal bootstrap is missing" in wrapper
    assert 'session_root="$DURABLE_DIR/runs/$RUN_ID/deepseek-harness/sessions"' in wrapper
    assert 'session_id="$RUN_ID"' in wrapper
    assert "deepseek-harness/cpu-attempt-$ATTEMPT/sessions" not in wrapper


def test_native_goal_bootstrap_precedes_first_model_step() -> None:
    bootstrap = (CONTAINER / "sprint-deepseek-goal-bootstrap.mjs").read_text()
    assert "ctx.on('agent/pre-step'" in bootstrap
    assert "ctx.goals.create(agent, { objective })" in bootstrap
    assert "ctx.goals.resume(agent, { id: current.id, revision: current.revision })" in bootstrap
    assert "textual /goal command" in bootstrap
    runner = (CONTAINER / "sprint-deepseek-harness-runner.py").read_text()
    assert 'os.environ["DSH_GOAL_OBJECTIVE"] = objective' in runner
    assert "uses native goal mode, not /goal prompt text" in runner
    assert "continuation round" not in runner
    template = (
        ROOT / "event_runtime/control/templates/deepseek-harness.j2"
    ).read_text()
    assert not template.lstrip().startswith("/goal")


def test_native_benchmark_goal_is_host_owned() -> None:
    bootstrap = CONTAINER / "sprint-deepseek-goal-bootstrap.mjs"
    script = f"""
process.env.DSH_GOAL_OBJECTIVE = 'fixed benchmark objective'
const plugin = await import({json.dumps(bootstrap.as_uri())})
const handlers = new Map()
const originalResume = () => 'resume-ok'
const goals = {{
  create: () => {{ throw new Error('not used') }},
  get: () => ({{ objective: 'fixed benchmark objective', phase: 'active', activation: 'armed' }}),
  edit: () => 'edit-must-not-run',
  pause: () => 'pause-must-not-run',
  resume: originalResume,
  complete: () => 'complete-must-not-run',
  block: () => 'block-must-not-run',
  clear: () => 'clear-must-not-run',
}}
const ctx = {{ goals, on: (name, handler) => handlers.set(name, handler) }}
plugin.apply(ctx)
for (const mutation of ['edit', 'pause', 'complete', 'block', 'clear']) {{
  let message = ''
  try {{ goals[mutation]() }} catch (error) {{ message = String(error.message) }}
  if (!message.includes('benchmark goal is host-owned')) throw new Error(`${{mutation}} was not fenced: ${{message}}`)
}}
if (goals.resume !== originalResume || goals.resume() !== 'resume-ok') {{
  throw new Error('resume must remain available for supervised relaunch')
}}
let continued = false
handlers.get('agent/pre-step')({{ agent: {{}} }}, () => {{ continued = true }})
if (!continued) throw new Error('pre-step did not continue')
"""
    subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        check=True,
        text=True,
        capture_output=True,
    )


def test_offline_probe_exercises_non_surface_stream_closed_retry() -> None:
    probe = (CONTAINER / "sprint-deepseek-harness-probe.py").read_text()
    assert 'FAILED_PARTIAL_TEXT = "partial-stream-content-must-not-surface"' in probe
    assert "len(Handler.request_payloads) == 4" in probe
    assert "first_request == retry_request" in probe
    assert 'retry["failure"]["code"] == "STREAM_CLOSED"' in probe
    assert "FAILED_PARTIAL_TEXT not in json.dumps(surface_messages)" in probe
    assert '"benchmark goal is host-owned" in mutation_surface' in probe
    assert 'get("operation") == "edit"' in probe


def test_runner_records_only_versioned_structured_notifications() -> None:
    runner = (CONTAINER / "sprint-deepseek-harness-runner.py").read_text()
    assert '"schema_version": 1' in runner
    assert '"method": raw["method"]' in runner
    assert '"payload": raw["payload"]' in runner
    assert "repr(notification)" not in runner


def test_image_pins_runtime_and_sdk_versions() -> None:
    image_source = (ROOT / "event_runtime/image.py").read_text()
    assert 'DEEPSEEK_HARNESS_VERSION = "0.1.1-rc.2"' in image_source
    assert '"deepseek-harness-sdk==0.1.1rc1"' in image_source
    manifest = json.loads(
        (
            ROOT
            / "event_runtime/container/deepseek-harness-node/package.json"
        ).read_text()
    )
    lock = json.loads(
        (
            ROOT
            / "event_runtime/container/deepseek-harness-node/package-lock.json"
        ).read_text()
    )
    required = {
        "dsh-sdk-jsonrpc-demo",
        "dsh-sdk-jsonrpc-server",
        "dsh-agent-spine-demo",
        "dsh-goal",
        "dsh-goal-round-driver",
        "dsh-tool-goal",
        "dsh-llm-deepseek",
        "dsh-llm-retry",
        "dsh-tool-bash-persistent",
        "dsh-tool-str-replace-editor",
        "dsh-session-persistence-jsonl",
    }
    for package in required:
        name = f"@deepseek-ai/{package}"
        assert manifest["dependencies"][name] == "0.1.1-rc.2"
        assert lock["packages"][f"node_modules/{name}"]["version"] == "0.1.1-rc.2"
    assert "npm ci --prefix /opt/deepseek-harness" in image_source
    assert "node_modules/.bin/dsh-jsonrpc-agent" in image_source
    assert "/usr/local/bin/dsh-jsonrpc-agent" in image_source
    assert "dsh-agent/package.json' | wc -l)" in image_source
    assert "dsh-scope/package.json' | wc -l)" in image_source
