from __future__ import annotations

import pytest

from harbor.agents.installed.deepseek_harness import DeepSeekHarness


@pytest.mark.parametrize("reasoning_effort", ["medium", "high", "max"])
def test_supported_reasoning_efforts(temp_dir, reasoning_effort: str) -> None:
    DeepSeekHarness(
        logs_dir=temp_dir,
        model_name="deepseek/deepseek-v4-flash-vision-exp",
        reasoning_effort=reasoning_effort,
    )


def test_unsupported_reasoning_effort_is_rejected(temp_dir) -> None:
    with pytest.raises(ValueError, match="must be one of"):
        DeepSeekHarness(
            logs_dir=temp_dir,
            model_name="deepseek/deepseek-v4-flash-vision-exp",
            reasoning_effort="ultra",
        )
