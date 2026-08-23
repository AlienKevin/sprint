from __future__ import annotations

import shlex
from typing import override

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.agent.name import AgentName


class DeepSeekHarness(BaseInstalledAgent):
    """DeepSeek's official unattended minimal benchmark harness."""

    _OUTPUT_FILENAME = "deepseek-harness.txt"
    _RUNTIME_VERSION = "0.1.1-rc.2"

    def __init__(self, *args, reasoning_effort: str = "max", **kwargs):
        if reasoning_effort != "max":
            raise ValueError("DeepSeek Harness benchmark reasoning effort must be max")
        super().__init__(*args, **kwargs)

    @staticmethod
    @override
    def name() -> str:
        return AgentName.DEEPSEEK_HARNESS.value

    @override
    def get_version_command(self) -> str | None:
        return (
            "node -p \"require('/usr/local/lib/node_modules/"
            "@deepseek-ai/dsh-sdk-jsonrpc-demo/package.json').version\""
        )

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        # Experiments are network-sealed during agent setup.  The exact runtime
        # must therefore come from the preflighted immutable Modal image.
        check = await environment.exec(
            command=(
                "test -x /usr/local/bin/dsh-jsonrpc-agent && "
                "test -x /opt/sprint-deepseek-harness-exec-wrapper.sh && "
                "test -x /opt/sprint-deepseek-harness-runner.py && "
                f"test \"$({self.get_version_command()})\" = "
                f"{shlex.quote(self._RUNTIME_VERSION)} && "
                "python3 -c 'import deepseek_harness'"
            )
        )
        if check.return_code != 0:
            raise RuntimeError(
                "the immutable image does not contain the pinned DeepSeek Harness runtime"
            )

    @override
    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        api_key = self._get_env("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required")
        output = f"/logs/agent/{self._OUTPUT_FILENAME}"
        await self.exec_as_agent(
            environment,
            command=(
                "/opt/sprint-deepseek-harness-exec-wrapper.sh "
                f"{shlex.quote(instruction)} 2>&1 </dev/null | "
                f"stdbuf -oL tee {shlex.quote(output)}"
            ),
            env={"OPENROUTER_API_KEY": api_key},
        )
