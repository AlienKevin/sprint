from __future__ import annotations

import json
from pathlib import Path

import pytest

from harbor.agents.factory import AgentFactory
from harbor.agents.installed.deepseek_harness import DeepSeekHarness
from harbor.models.agent.name import AgentName


ROOT = Path(__file__).resolve().parents[2]
CONTAINER = ROOT / "event_runtime" / "container"


def test_factory_registers_deepseek_harness() -> None:
    assert AgentFactory.get_agent_class(AgentName.DEEPSEEK_HARNESS) is DeepSeekHarness
    assert DeepSeekHarness.name() == "deepseek-harness"


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
    assert "persona: You are a helpful software engineer assistant." in config
    assert "includeHarnessIdentity: false" in config
    assert "includeRuntimeContext: false" in config
    assert "workspaceContext: false" in config
    assert "enabled: false" in config
    assert "@deepseek-ai/dsh-tool-bash-persistent" in config
    assert "@deepseek-ai/dsh-tool-str-replace-editor" in config
    assert "@deepseek-ai/dsh-session-persistence-jsonl" in config
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


def test_image_pins_runtime_and_sdk_versions() -> None:
    dockerfile = (
        ROOT / "events/g1-100-metres/environment/Dockerfile"
    ).read_text()
    assert "ARG DEEPSEEK_HARNESS_VERSION=0.1.1-rc.2" in dockerfile
    assert '"deepseek-harness-sdk==0.1.1rc1"' in dockerfile
    required = {
        "dsh-sdk-jsonrpc-demo",
        "dsh-sdk-jsonrpc-server",
        "dsh-agent-spine-demo",
        "dsh-llm-deepseek",
        "dsh-tool-bash-persistent",
        "dsh-tool-str-replace-editor",
        "dsh-session-persistence-jsonl",
    }
    for package in required:
        assert f"@deepseek-ai/{package}@${{DEEPSEEK_HARNESS_VERSION}}" in dockerfile
