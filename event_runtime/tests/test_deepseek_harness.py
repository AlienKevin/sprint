from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import subprocess
import sys
import types
from dataclasses import dataclass
import queue

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


@pytest.mark.parametrize("reasoning_effort", ["medium", "high", "max"])
def test_deepseek_harness_accepts_supported_reasoning_efforts(
    tmp_path: Path, reasoning_effort: str
) -> None:
    DeepSeekHarness(
        logs_dir=tmp_path,
        model_name="deepseek/deepseek-v4-flash-vision-exp",
        reasoning_effort=reasoning_effort,
    )


def test_deepseek_harness_rejects_unknown_reasoning_effort(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be one of"):
        DeepSeekHarness(
            logs_dir=tmp_path,
            model_name="deepseek/deepseek-v4-flash-vision-exp",
            reasoning_effort="ultra",
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
    assert "@deepseek-ai/dsh-tool-goal" not in config
    assert "@deepseek-ai/dsh-goal-round-driver" in config
    assert (
        "defaultMaxGoalRounds: !!js Number(process.env.DSH_GOAL_MAX_ROUNDS ?? 256)"
        in config
    )
    assert "Goal mutation tools are deliberately not mounted" in config
    assert "sprint-deepseek-goal-bootstrap.mjs" in config
    assert "compression: none" in config
    assert "dsh-tool-jobs" not in config
    assert "dsh-compaction" not in config


def test_wrapper_seals_route_and_wire_parameters() -> None:
    wrapper = (CONTAINER / "sprint-deepseek-harness-exec-wrapper.sh").read_text()
    assert "REASONING_EFFORT=${SPRINT_REASONING_EFFORT:-max}" in wrapper
    assert '"reasoning_effort":"' in wrapper
    assert "$REASONING_EFFORT" in wrapper
    assert "SPRINT_REASONING_EFFORT must be medium, high, or max" in wrapper
    assert '"max_tokens":384000' in wrapper
    assert '"temperature":1.0' in wrapper
    assert '"top_p":0.95' in wrapper
    assert "PROVIDER_ENDPOINT:-deepseek" in wrapper
    assert '--provider-endpoint "$PROVIDER_ENDPOINT"' in wrapper
    assert '--request-contract-json "$REQUEST_CONTRACT"' in wrapper
    assert "--upstream-api-key-stdin" in wrapper
    assert "unset OPENROUTER_API_KEY" in wrapper
    assert "export OPENROUTER_API_KEY=sprint-local-proxy-token" in wrapper
    assert 'PROXY_PROCESS_FILE="$AGENT_STATE_DIR/openrouter-proxy.pid"' in wrapper
    assert "pending_request_count" in wrapper
    assert "fail_closed_proxy_recovery" in wrapper
    assert "DeepSeek Harness native goal bootstrap is missing" in wrapper
    assert (
        'session_root="$DURABLE_DIR/runs/$RUN_ID/deepseek-harness/sessions"' in wrapper
    )
    assert 'session_id="$RUN_ID"' in wrapper
    assert 'lifecycle="$AGENT_LOG_DIR/goal-lifecycle.json"' in wrapper
    assert '--lifecycle "$lifecycle"' in wrapper
    assert "deepseek-harness/cpu-attempt-$ATTEMPT/sessions" not in wrapper


def test_native_goal_bootstrap_precedes_first_model_step() -> None:
    bootstrap = (CONTAINER / "sprint-deepseek-goal-bootstrap.mjs").read_text()
    assert "ctx.on('agent/pre-step'" in bootstrap
    assert "ctx.goals.create(agent, { objective })" in bootstrap
    assert "const trustedResume = ctx.goals.resume.bind(ctx.goals)" in bootstrap
    assert "textual /goal command" in bootstrap
    runner = (CONTAINER / "sprint-deepseek-harness-runner.py").read_text()
    assert 'os.environ["DSH_GOAL_OBJECTIVE"] = objective' in runner
    assert "uses native goal mode, not /goal prompt text" in runner
    assert "subscribe_session_notifications" in runner
    assert "client.session_prompt" in runner
    assert "session.run(" not in runner
    template = (
        ROOT / "event_runtime/control/templates/deepseek-harness.j2"
    ).read_text()
    assert not template.lstrip().startswith("/goal")
    assert "event history" not in template
    assert "event wait" not in template


def test_native_benchmark_goal_is_host_owned() -> None:
    bootstrap = CONTAINER / "sprint-deepseek-goal-bootstrap.mjs"
    script = f"""
process.env.DSH_GOAL_OBJECTIVE = 'fixed benchmark objective'
const plugin = await import({json.dumps(bootstrap.as_uri())})
const handlers = new Map()
const agent = {{ id: 'agent-1', status: 'running' }}
const session = {{ id: agent.id }}
agent.session = session
let current = {{
  id: 'goal-1',
  revision: 1,
  objective: 'fixed benchmark objective',
  phase: 'active',
  activation: 'armed',
  roundsStarted: 0,
}}
let resumed = 0
let disarmed = 0
const goals = {{
  create: () => {{ throw new Error('not used') }},
  get: () => ({{ ...current }}),
  disarm: () => {{ disarmed += 1; current.activation = 'disarmed'; return {{ ...current }} }},
  edit: () => 'edit-must-not-run',
  pause: () => 'pause-must-not-run',
  resume: () => {{ resumed += 1; current.activation = 'armed'; return {{ ...current }} }},
  complete: () => 'complete-must-not-run',
  block: () => 'trusted-driver-block',
  clear: () => 'clear-must-not-run',
}}
let scheduled
let scheduledDelay
globalThis.setTimeout = (callback, delay) => {{ scheduled = callback; scheduledDelay = delay; return 1 }}
globalThis.clearTimeout = () => {{}}
const ctx = {{
  agents: {{ get: (id) => id === agent.id ? agent : undefined }},
  goals,
  on: (name, handler) => handlers.set(name, handler),
}}
plugin.apply(ctx)
for (const mutation of ['edit', 'pause', 'resume', 'complete', 'clear']) {{
  let message = ''
  try {{ goals[mutation]() }} catch (error) {{ message = String(error.message) }}
  if (!message.includes('benchmark goal is host-owned')) throw new Error(`${{mutation}} was not fenced: ${{message}}`)
}}
if (goals.block() !== 'trusted-driver-block') throw new Error('trusted driver block was fenced')
let continued = false
handlers.get('agent/pre-step')({{ agent }}, () => {{ continued = true }})
if (!continued) throw new Error('pre-step did not continue')
handlers.get('session/event')(session, {{ type: 'turn/end', data: {{ reason: {{ kind: 'completed' }} }} }})
agent.status = 'idle'
handlers.get('agent/status')({{ agent, status: 'idle' }})
if (scheduledDelay !== 30000) throw new Error(`unexpected continuation watchdog: ${{scheduledDelay}}`)
scheduled()
if (disarmed !== 1 || resumed !== 1) throw new Error('stalled armed goal was not refreshed')
current.activation = 'disarmed'
handlers.get('session/event')(session, {{ type: 'turn/end', data: {{ reason: {{ kind: 'completed' }} }} }})
handlers.get('agent/status')({{ agent, status: 'idle' }})
if (resumed !== 2) throw new Error('normally completed disarmed goal was not rearmed')
"""
    subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        check=True,
        text=True,
        capture_output=True,
    )


@dataclass
class FakeNotification:
    method: str
    payload: dict[str, object]


class FakeSubscription:
    def __init__(self) -> None:
        self.items: queue.Queue[object] = queue.Queue()
        self.closed = False

    def next(self) -> FakeNotification:
        item = self.items.get(timeout=2)
        if isinstance(item, BaseException):
            raise item
        assert isinstance(item, FakeNotification)
        return item

    def close(self) -> None:
        self.closed = True


class FakeHarnessClient:
    def __init__(self, notifications: list[FakeNotification]) -> None:
        self.notifications = notifications
        self.subscription = FakeSubscription()
        self.prompt_calls: list[tuple[str, list[dict[str, str]]]] = []

    def subscribe_session_notifications(self, session_id: str) -> FakeSubscription:
        assert session_id == "same-session"
        return self.subscription

    def session_prompt(
        self,
        session_id: str,
        content: list[dict[str, str]],
        *,
        notification_subscription: FakeSubscription,
    ) -> str:
        assert notification_subscription is self.subscription
        self.prompt_calls.append((session_id, content))
        for notification in self.notifications:
            self.subscription.items.put(notification)
        return "initial-message"

    def _runtime_diagnostics(self) -> str:
        return "goal-round-driver warning with secret-for-test"


class FakeHarness:
    def __init__(self, notifications: list[FakeNotification]) -> None:
        self.client = FakeHarnessClient(notifications)
        self.closed = False

    def start_session(self, session_id: str) -> object:
        assert session_id == "same-session"
        return types.SimpleNamespace(id=session_id)

    def close(self) -> None:
        self.closed = True
        self.client.subscription.items.put(RuntimeError("runtime closed"))


def test_runner_keeps_same_session_through_second_native_goal_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk_stub = types.ModuleType("deepseek_harness")
    sdk_stub.DeepSeekHarness = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deepseek_harness", sdk_stub)
    runner_path = CONTAINER / "sprint-deepseek-harness-runner.py"
    spec = importlib.util.spec_from_file_location("sprint_dsh_runner_test", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)

    sequence = 0

    def status(value: str) -> FakeNotification:
        return FakeNotification(
            "session.status", {"sessionId": "same-session", "status": value}
        )

    def event(kind: str, data: dict[str, object]) -> FakeNotification:
        nonlocal sequence
        result = FakeNotification(
            "session.event",
            {
                "sessionId": "same-session",
                "event": {"seq": sequence, "type": kind, "data": data},
            },
        )
        sequence += 1
        return result

    notifications = [
        event(
            "agent/inbox/spliced",
            {"inserted": [{"id": "initial-message", "source": {"kind": "user"}}]},
        ),
        status("running"),
        event("turn/start", {"turn": 1}),
        event(
            "goal/change",
            {
                "operation": "create",
                "goal": {"phase": "active"},
                "roundsStarted": 0,
            },
        ),
        event(
            "assistant/message",
            {"message": {"content": [{"type": "text", "text": "round one final"}]}},
        ),
        event("turn/end", {"turn": 1, "reason": {"kind": "completed"}}),
        status("idle"),
        event(
            "agent/inbox/spliced",
            {
                "inserted": [
                    {
                        "id": "goal-round-1",
                        "source": {
                            "kind": "goal",
                            "goalId": "goal-1",
                            "revision": 1,
                            "round": 1,
                        },
                    }
                ]
            },
        ),
        status("running"),
        event("turn/start", {"turn": 2}),
        event(
            "user/message",
            {
                "id": "goal-round-1",
                "source": {"kind": "goal", "round": 1},
            },
        ),
        event(
            "assistant/message",
            {"message": {"content": [{"type": "text", "text": "round two final"}]}},
        ),
        event("turn/end", {"turn": 2, "reason": {"kind": "completed"}}),
        event(
            "goal/change",
            {
                "operation": "block",
                "goal": {"phase": "blocked"},
                "roundsStarted": 1,
            },
        ),
        status("idle"),
    ]
    harness = FakeHarness(notifications)
    recorded: list[FakeNotification] = []
    lifecycle = tmp_path / "goal-lifecycle.json"

    result = runner.run_goal_session(
        harness,
        session_id="same-session",
        objective="keep improving",
        record=recorded.append,
        stop_file=tmp_path / "stop",
        lifecycle_path=lifecycle,
        continuation_timeout_seconds=1,
    )

    assert result.exit_code == 0
    assert result.runner_state == "terminal"
    assert result.goal_status == "blocked"
    assert result.completed_turns == 2
    assert result.rounds_started == 1
    assert result.final_response == "round two final"
    assert harness.client.prompt_calls == [
        ("same-session", [{"type": "text", "text": "keep improving"}])
    ]
    assert harness.closed is True
    assert len(recorded) == len(notifications)
    lifecycle_payload = json.loads(lifecycle.read_text())
    assert lifecycle_payload["runner_state"] == "terminal"
    assert lifecycle_payload["completed_turns"] == 2
    assert lifecycle_payload["goal_rounds_started"] == 1


def test_runner_continuation_timeout_records_sanitized_runtime_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk_stub = types.ModuleType("deepseek_harness")
    sdk_stub.DeepSeekHarness = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deepseek_harness", sdk_stub)
    runner_path = CONTAINER / "sprint-deepseek-harness-runner.py"
    spec = importlib.util.spec_from_file_location(
        "sprint_dsh_runner_timeout_test", runner_path
    )
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)

    def event(sequence: int, kind: str, data: dict[str, object]) -> FakeNotification:
        return FakeNotification(
            "session.event",
            {
                "sessionId": "same-session",
                "event": {"seq": sequence, "type": kind, "data": data},
            },
        )

    notifications = [
        event(
            0,
            "agent/inbox/spliced",
            {"inserted": [{"id": "initial-message", "source": {"kind": "user"}}]},
        ),
        FakeNotification(
            "session.status", {"sessionId": "same-session", "status": "running"}
        ),
        event(1, "turn/start", {"turn": 1}),
        event(
            2,
            "goal/change",
            {
                "operation": "create",
                "goal": {"phase": "active"},
                "roundsStarted": 0,
            },
        ),
        event(3, "turn/end", {"turn": 1, "reason": {"kind": "completed"}}),
        FakeNotification(
            "session.status", {"sessionId": "same-session", "status": "idle"}
        ),
    ]
    harness = FakeHarness(notifications)
    lifecycle = tmp_path / "goal-lifecycle.json"
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-for-test")

    result = runner.run_goal_session(
        harness,
        session_id="same-session",
        objective="keep improving",
        record=lambda _notification: None,
        stop_file=tmp_path / "stop",
        lifecycle_path=lifecycle,
        continuation_timeout_seconds=0.02,
    )

    assert result.exit_code == runner.INFRA_FAILURE_EXIT
    assert result.runner_state == "invalid_infrastructure"
    payload = json.loads(lifecycle.read_text())
    assert payload["failure_code"] == "goal_continuation_timeout"
    assert payload["continuation_timeout_seconds"] == 0.02
    assert "goal-round-driver warning" in payload["runtime_diagnostics"]
    assert "secret-for-test" not in payload["runtime_diagnostics"]
    assert "[REDACTED]" in payload["runtime_diagnostics"]


def test_offline_probe_exercises_non_surface_stream_closed_retry() -> None:
    probe = (CONTAINER / "sprint-deepseek-harness-probe.py").read_text()
    assert 'FAILED_PARTIAL_TEXT = "partial-stream-content-must-not-surface"' in probe
    assert "len(Handler.request_payloads) == 3" in probe
    assert "first_request == retry_request" in probe
    assert "result.completed_turns == 2" in probe
    assert "result.rounds_started == 1" in probe
    assert '"<goal_round>" in json.dumps(goal_round_request)' in probe
    assert 'retry["failure"]["code"] == "STREAM_CLOSED"' in probe
    assert "FAILED_PARTIAL_TEXT not in json.dumps(surface_messages)" in probe
    assert '"Use goal tools" not in messages[0]["content"]' in probe
    assert 'for name in ("create_goal", "get_goal", "update_goal")' in probe
    assert 'DSH_PROBE_STRESS_CHUNKS", "20000"' in probe
    assert "continuation_seconds < 120" in probe


def test_runner_default_goal_continuation_timeout_is_ten_minutes() -> None:
    runner = (CONTAINER / "sprint-deepseek-harness-runner.py").read_text()
    assert "DEFAULT_CONTINUATION_TIMEOUT_SECONDS = 600" in runner


def test_runner_records_only_versioned_structured_notifications() -> None:
    runner = (CONTAINER / "sprint-deepseek-harness-runner.py").read_text()
    assert '"schema_version": 1' in runner
    assert '"method": raw["method"]' in runner
    assert '"payload": raw["payload"]' in runner
    assert "repr(notification)" not in runner


def test_native_goal_bootstrap_has_host_owned_continuation_watchdog() -> None:
    bootstrap = (CONTAINER / "sprint-deepseek-goal-bootstrap.mjs").read_text()
    assert "CONTINUATION_WATCHDOG_MS = 30_000" in bootstrap
    assert "event.data?.reason?.kind === 'completed'" in bootstrap
    assert "trustedDisarm(agent)" in bootstrap
    assert "trustedResume(agent, goalRef(current))" in bootstrap


def test_image_pins_runtime_and_sdk_versions() -> None:
    image_source = (ROOT / "event_runtime/image.py").read_text()
    assert 'DEEPSEEK_HARNESS_VERSION = "0.1.1-rc.2"' in image_source
    assert '"deepseek-harness-sdk==0.1.1rc1"' in image_source
    manifest = json.loads(
        (
            ROOT / "event_runtime/container/deepseek-harness-node/package.json"
        ).read_text()
    )
    lock = json.loads(
        (
            ROOT / "event_runtime/container/deepseek-harness-node/package-lock.json"
        ).read_text()
    )
    required = {
        "dsh-sdk-jsonrpc-demo",
        "dsh-sdk-jsonrpc-server",
        "dsh-agent-spine-demo",
        "dsh-goal",
        "dsh-goal-round-driver",
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
