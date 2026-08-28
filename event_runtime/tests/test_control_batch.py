from __future__ import annotations

import contextlib
import copy
import datetime as dt
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
from pathlib import Path
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "runs/ops"
sys.path.insert(0, str(OPS))
sys.path.insert(0, str(ROOT))

from event_runtime.control import batch as batch_eval  # noqa: E402
from event_runtime.container.sprint_openrouter_usage import (  # noqa: E402
    empty_token_usage,
)
from event_runtime.export import frontier as frontier_update  # noqa: E402
from event_runtime.preflight import canary as training_gpu_canary  # noqa: E402
from event_runtime.export import timeline as unified_timeline  # noqa: E402


def test_event_runtime_state_does_not_dirty_the_source_tree() -> None:
    result = subprocess.run(
        [
            "git",
            "check-ignore",
            "--quiet",
            "runs/ops/arbitrary-batch-name/model-1/trial-launch.json",
        ],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0


def test_launcher_provenance_guard_checks_source_not_generated_runs() -> None:
    launcher = (ROOT / "event_runtime/control/launch.sh").read_text()
    assert 'python3 "$ROOT/event_runtime/preflight/check_source.py"' in launcher
    source_check = (ROOT / "event_runtime/preflight/check_source.py").read_text()
    assert '"event_runtime/control"' in source_check
    assert '"event_runtime/event.py"' in source_check
    assert '"event_runtime/sync_verifier.py"' in source_check
    assert '"events/g1-100-metres"' in source_check
    assert '"harbor"' in source_check
    assert '"event_runtime/export"' not in source_check
    assert '"runs"' not in source_check


def test_launcher_forbids_same_trial_cpu_resume() -> None:
    launcher = (ROOT / "event_runtime/control/launch.sh").read_text()
    assert "CPU-agent resume is forbidden" in launcher
    assert "--supervised-launch" not in launcher
    assert "RESUMING" not in launcher


def test_launcher_requires_exact_gpu_job_index_and_paces_dispatch() -> None:
    launcher = (ROOT / "event_runtime/control/launch.sh").read_text()
    assert 'gpu-dispatch-loop --run-id "$RUN_ID" --poll-seconds 10' in launcher
    assert 'gpu-budget-pulse --run-id "$RUN_ID" --poll-seconds 15' in launcher


def test_batch_matrix_is_exact_six_arm_max_effort_contract() -> None:
    rows = batch_eval.matrix("eval-20260808")
    assert len(rows) == 6
    assert len({row["run_id"] for row in rows}) == 6
    assert {row["family"] for row in rows} == {"deepseek", "luna"}
    assert {row["reasoning_effort"] for row in rows} == {"max"}
    assert {row["codex_version"] for row in rows} == {"0.149.1"}
    assert {row["agent_kind"] for row in rows} == {"codex", "deepseek-harness"}
    assert {row["agent_kind"] for row in rows if row["family"] == "deepseek"} == {
        "deepseek-harness"
    }
    assert {row["agent_kind"] for row in rows if row["family"] == "luna"} == {"codex"}
    assert {row["goal_mode"] for row in rows} == {
        "codex_session_goal",
        "deepseek_native_goal",
    }
    assert (
        sum(row["model"] == "deepseek/deepseek-v4-flash-vision-exp" for row in rows)
        == 3
    )
    assert sum(row["model"] == "openai/gpt-5.6-luna" for row in rows) == 3
    assert {
        row["resolved_model_version"] for row in rows if row["family"] == "deepseek"
    } == {"deepseek/deepseek-v4-flash-vision-exp-20260821"}
    assert {
        row["resolved_model_version"] for row in rows if row["family"] == "luna"
    } == {"openai/gpt-5.6-luna-20260709"}
    assert {
        (row["provider_endpoint"], row["quantization"])
        for row in rows
        if row["family"] == "deepseek"
    } == {("deepseek", "unknown")}
    assert {
        Path(row["wrapper"]).name for row in rows if row["family"] == "deepseek"
    } == {"deepseek_harness.sh"}
    assert batch_eval.LIVE_SITE_DEPLOY_SECONDS == 20 * 60


def test_batch_matrix_accepts_explicit_reasoning_effort() -> None:
    rows = batch_eval.matrix(
        "eval-sol-medium",
        families=("sol",),
        trials_per_model=3,
        reasoning_effort="medium",
    )
    assert len(rows) == 3
    assert {row["family"] for row in rows} == {"sol"}
    assert {row["reasoning_effort"] for row in rows} == {"medium"}


def test_opus_matrix_uses_claude_code_goal_mode_and_official_route() -> None:
    rows = batch_eval.matrix(
        "eval-opus-medium",
        families=("opus",),
        trials_per_model=1,
        reasoning_effort="medium",
    )

    assert rows == [
        {
            "run_id": "eval-opus-medium-opus-1",
            "family": "opus",
            "model": "anthropic/claude-opus-5",
            "resolved_model_version": "anthropic/claude-opus-5-20260723",
            "reasoning_effort": "medium",
            "agent_kind": "claude-code",
            "goal_mode": "claude_code_native_goal",
            "codex_version": "0.149.1",
            "claude_code_version": "2.1.248",
            "wrapper": str(
                ROOT / "event_runtime/control/providers/anthropic_claude_code.sh"
            ),
            "trial": 1,
            "status": "planned",
            "provider": "Anthropic",
            "provider_endpoint": "anthropic",
            "quantization": "unknown",
            "context_window": "1000000",
        }
    ]


def test_glm_matrix_defaults_to_max_and_official_zai_route() -> None:
    rows = batch_eval.matrix("eval-glm-max", families=("glm",), trials_per_model=1)

    assert rows[0]["model"] == "z-ai/glm-5.3-flash"
    assert rows[0]["resolved_model_version"] == "z-ai/glm-5.3-flash-20260826"
    assert rows[0]["reasoning_effort"] == "max"
    assert rows[0]["agent_kind"] == "claude-code"
    assert rows[0]["goal_mode"] == "claude_code_native_goal"
    assert rows[0]["provider"] == "Z.AI"
    assert rows[0]["provider_endpoint"] == "z-ai/fp8"
    assert rows[0]["quantization"] == "fp8"
    assert Path(rows[0]["wrapper"]).name == "z_ai_claude_code.sh"
    assert batch_eval.agent_adapter_contract_ready(rows)


def test_opus_matrix_forces_medium_when_batch_default_is_max() -> None:
    rows = batch_eval.matrix(
        "eval-opus-default", families=("opus",), trials_per_model=1
    )

    assert rows[0]["reasoning_effort"] == "medium"


def test_glm_matrix_rejects_noncanonical_effort() -> None:
    with pytest.raises(ValueError, match="must be max for glm"):
        batch_eval.matrix(
            "eval-glm-medium",
            families=("glm",),
            trials_per_model=1,
            reasoning_effort="medium",
        )


@pytest.mark.parametrize("family", ["deepseek", "luna"])
@pytest.mark.parametrize("reasoning_effort", ["medium", "high"])
def test_batch_matrix_requires_max_for_fixed_effort_families(
    family: str, reasoning_effort: str
) -> None:
    with pytest.raises(ValueError, match=f"reasoning effort must be max for {family}"):
        batch_eval.matrix(
            f"eval-{family}-{reasoning_effort}",
            families=(family,),
            reasoning_effort=reasoning_effort,
        )


@pytest.mark.parametrize("reasoning_effort", ["medium", "high", "max"])
def test_batch_matrix_allows_sol_effort_sweep(reasoning_effort: str) -> None:
    rows = batch_eval.matrix(
        f"eval-sol-{reasoning_effort}",
        families=("sol",),
        reasoning_effort=reasoning_effort,
    )
    assert {row["reasoning_effort"] for row in rows} == {reasoning_effort}


def test_batch_matrix_accepts_exact_replacement_trial_slots() -> None:
    rows = batch_eval.matrix(
        "eval-sol-medium-replacement",
        families=("sol",),
        reasoning_effort="medium",
        trial_numbers=(3,),
    )

    assert [row["run_id"] for row in rows] == ["eval-sol-medium-replacement-sol-3"]
    assert [row["trial"] for row in rows] == [3]


@pytest.mark.parametrize("trial_numbers", [(), (0,), (51,), (3, 3), (True,)])
def test_batch_matrix_rejects_invalid_replacement_trial_slots(
    trial_numbers: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="trial numbers"):
        batch_eval.matrix("eval-invalid", trial_numbers=trial_numbers)


def test_batch_matrix_rejects_unknown_reasoning_effort() -> None:
    with pytest.raises(ValueError, match="reasoning effort must be one of"):
        batch_eval.matrix("eval-sol-invalid", reasoning_effort="ultra")


def test_one_shot_batch_monitor_does_not_claim_lifetime_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, Path]] = []
    env_file = tmp_path / ".env"
    monkeypatch.setattr(
        batch_eval,
        "monitor_cycle",
        lambda batch_id, *, env_file: (
            calls.append((batch_id, env_file))
            or {"batch_id": batch_id, "status": "running"}
        ),
    )
    monkeypatch.setattr(
        batch_eval.frontier_update,
        "file_lock",
        lambda *_args, **_kwargs: pytest.fail(
            "one-shot diagnostics must not claim the daemon owner lease"
        ),
    )

    result = batch_eval.run_monitor_command(
        "eval", env_file=env_file, loop=False, poll_seconds=10
    )

    assert result == {"batch_id": "eval", "status": "running"}
    assert calls == [("eval", env_file)]


def test_duplicate_batch_monitor_exits_without_running_a_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(batch_eval, "BATCH_ROOT", tmp_path / "batches")

    @contextlib.contextmanager
    def denied_owner(path: Path, *, blocking: bool = True):
        assert path == tmp_path / "batches/eval/monitor-owner.lock"
        assert blocking is False
        yield False

    monkeypatch.setattr(batch_eval.frontier_update, "file_lock", denied_owner)
    monkeypatch.setattr(
        batch_eval,
        "monitor_cycle",
        lambda *_args, **_kwargs: pytest.fail(
            "a duplicate monitor must not run even one controller cycle"
        ),
    )

    result = batch_eval.run_monitor_command(
        "eval", env_file=tmp_path / ".env", loop=True, poll_seconds=10
    )

    assert result["batch_id"] == "eval"
    assert result["status"] == "monitor_already_running"


def test_duplicate_monitor_control_record_is_safe_for_cli_output() -> None:
    result = {
        "schema_version": 1,
        "batch_id": "eval",
        "status": "monitor_already_running",
        "updated_at": "2026-08-24T00:00:00Z",
    }
    assert batch_eval.public_command_output("monitor", result) == result


def test_fixed_and_swept_efforts_are_propagated_to_agent_contracts() -> None:
    deepseek_rows = batch_eval.matrix(
        "max-deepseek-canary",
        families=("deepseek",),
        reasoning_effort="max",
    )
    sol_rows = batch_eval.matrix(
        "high-sol-canary",
        families=("sol",),
        reasoning_effort="high",
    )
    assert len(deepseek_rows) == 3
    assert len(sol_rows) == 3
    assert {row["reasoning_effort"] for row in deepseek_rows} == {"max"}
    assert {row["reasoning_effort"] for row in sol_rows} == {"high"}
    launcher = (ROOT / "event_runtime/control/launch.sh").read_text()
    deepseek_wrapper = (
        ROOT / "event_runtime/control/providers/deepseek_harness.sh"
    ).read_text()
    assert '--ae "SPRINT_REASONING_EFFORT=$REASONING_EFFORT"' in launcher
    assert (
        "DeepSeek Harness benchmark reasoning effort is sealed to max" not in launcher
    )
    assert '"reasoning_effort":"' in launcher
    assert 'REASONING_EFFORT="${REASONING_EFFORT:-max}"' in deepseek_wrapper
    assert "REASONING_EFFORT=max" not in deepseek_wrapper
    assert batch_eval.agent_adapter_contract_ready(deepseek_rows + sol_rows)


def test_agent_adapter_contract_rejects_unsupported_deepseek_effort() -> None:
    assert not batch_eval.agent_adapter_contract_ready(
        [
            {
                "agent_kind": "deepseek-harness",
                "model": "deepseek/deepseek-v4-flash-vision-exp",
                "reasoning_effort": "ultra",
            }
        ]
    )


def test_batch_monitor_holds_owner_until_terminal_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(batch_eval, "BATCH_ROOT", tmp_path / "batches")
    owner_open = False
    observed_owner: list[bool] = []

    @contextlib.contextmanager
    def acquired_owner(path: Path, *, blocking: bool = True):
        nonlocal owner_open
        assert path == tmp_path / "batches/eval/monitor-owner.lock"
        assert blocking is False
        owner_open = True
        try:
            yield True
        finally:
            owner_open = False

    def terminal_cycle(batch_id: str, *, env_file: Path) -> dict[str, object]:
        observed_owner.append(owner_open)
        assert batch_id == "eval"
        assert env_file == tmp_path / ".env"
        return {"batch_id": batch_id, "status": "complete"}

    monkeypatch.setattr(batch_eval.frontier_update, "file_lock", acquired_owner)
    monkeypatch.setattr(batch_eval, "monitor_cycle", terminal_cycle)
    monkeypatch.setattr(batch_eval, "public_batch", lambda payload: payload)
    disabled: list[list[str]] = []
    monkeypatch.setattr(
        batch_eval.subprocess,
        "run",
        lambda command, **_kwargs: disabled.append(command),
    )

    result = batch_eval.run_monitor_command(
        "eval", env_file=tmp_path / ".env", loop=True, poll_seconds=10
    )

    assert result == {"batch_id": "eval", "status": "complete"}
    assert observed_owner == [True]
    assert owner_open is False
    assert disabled == [
        ["systemctl", "--user", "disable", "sprint-batch-eval-monitor.service"]
    ]


def test_terminal_batch_bypasses_live_deploy_debounce() -> None:
    assert (
        batch_eval.deployment_debounce_seconds(
            {
                "arms": [
                    {"harbor_alive": False},
                    {"stop_ack": {"acknowledged_at": "now"}},
                ]
            }
        )
        == 0
    )


def test_agent_exit_makes_single_cpu_trial_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "eval-deepseek-1"
    state_dir = tmp_path / run_id
    state_dir.mkdir()
    (state_dir / "STOP_ACK.json").write_text('{"reason":"agent_exit"}\n')
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", tmp_path)
    assert batch_eval.arm_terminal(
        {
            "run_id": run_id,
            "harbor_alive": False,
            "stop_ack": {"reason": "agent_exit"},
        }
    )
    assert (
        batch_eval.deployment_debounce_seconds(
            {
                "arms": [
                    {
                        "run_id": run_id,
                        "harbor_alive": False,
                        "stop_ack": {"reason": "agent_exit"},
                    }
                ]
            }
        )
        == 0
    )
    assert (
        batch_eval.deployment_debounce_seconds(
            {"arms": [{"harbor_alive": False}, {"harbor_alive": True}]}
        )
        == batch_eval.LIVE_SITE_DEPLOY_SECONDS
    )


def test_performance_snapshot_uses_the_active_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    web = tmp_path / "web"
    calls: list[tuple[str, Path, dict[str, object]]] = []
    monkeypatch.setattr(batch_eval, "WEB", web)
    monkeypatch.setattr(
        batch_eval.performance_export,
        "build",
        lambda prefix, output, **kwargs: calls.append((prefix, output, kwargs)),
    )

    batch_eval.refresh_performance_snapshot(
        {
            "batch_id": "event-20260812-r4",
            "preflight": {"openrouter_credit_snapshot": {"per_trial_budget_usd": 12.5}},
        }
    )

    assert calls == [
        (
            "event-20260812-r4-",
            web / "data/performance/current.json",
            {"cost_cap": 12.5, "run_ids": []},
        )
    ]


def test_single_family_replacement_uses_coexisting_comparison_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    web = tmp_path / "web"
    batches = tmp_path / "batches"
    batches.mkdir()
    calls: list[tuple[str, Path, dict[str, object]]] = []
    monkeypatch.setattr(batch_eval, "WEB", web)
    monkeypatch.setattr(batch_eval, "BATCH_ROOT", batches)
    monkeypatch.setattr(
        batch_eval.performance_export,
        "build",
        lambda prefix, output, **kwargs: calls.append((prefix, output, kwargs)),
    )
    batch_eval.atomic_json(
        batch_eval.batch_path("comparison"),
        {
            "batch_id": "comparison",
            "updated_at": "now",
            "status": "running",
            "reasoning_effort": "max",
            "codex_version": "0.147.0",
            "run_hours": None,
            "arms": [
                {"run_id": "comparison-luna-1", "family": "luna"},
                {"run_id": "comparison-sol-1", "family": "sol"},
            ],
        },
    )

    batch_eval.refresh_performance_snapshot(
        {
            "batch_id": "replacement",
            "coexist_batch_ids": ["comparison"],
            "updated_at": "now",
            "status": "running",
            "reasoning_effort": "max",
            "codex_version": "0.147.0",
            "run_hours": None,
            "arms": [{"run_id": "replacement-luna-1", "family": "luna"}],
        }
    )

    assert calls == [
        (
            "replacement-",
            web / "data/performance/current.json",
            {
                "cost_cap": 10.0,
                "run_ids": [
                    "comparison-luna-1",
                    "comparison-sol-1",
                    "replacement-luna-1",
                ],
            },
        )
    ]


def test_deployed_projection_check_ignores_unrelated_run_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    web = tmp_path / "web"
    batch_id = "eval"
    run_id = "eval-1"
    batch_file = web / f"data/batches/{batch_id}.json"
    current_batch = web / "data/batches/current.json"
    performance = web / "data/performance/current.json"
    timeline = web / f"data/timelines/{run_id}.json"
    unrelated = web / "data/timelines/other.json"
    for path in (batch_file, current_batch, performance, timeline, unrelated):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{path.name}\n")
    monkeypatch.setattr(batch_eval, "WEB", web)
    payload = {
        "batch_id": batch_id,
        "arms": [{"run_id": run_id}],
    }
    publication = {
        "site_status": "deployed",
        "last_deployed_public_artifacts": frontier_update.public_artifact_hashes(web),
    }
    assert batch_eval.deployed_batch_current(payload, publication)
    unrelated.write_text("changed elsewhere\n")
    assert batch_eval.deployed_batch_current(payload, publication)
    performance.write_text("changed performance\n")
    assert not batch_eval.deployed_batch_current(payload, publication)
    performance.write_text("current.json\n")
    assert batch_eval.deployed_batch_current(payload, publication)
    timeline.write_text("changed in this run\n")
    assert not batch_eval.deployed_batch_current(payload, publication)


def test_batch_matrix_can_launch_three_deepseek_trials_only() -> None:
    rows = batch_eval.matrix("eval-deepseek", families=("deepseek",))
    assert len(rows) == 3
    assert {row["family"] for row in rows} == {"deepseek"}
    assert [row["trial"] for row in rows] == [1, 2, 3]
    assert {row["model"] for row in rows} == {"deepseek/deepseek-v4-flash-vision-exp"}
    assert {row["provider"] for row in rows} == {"DeepSeek"}
    assert {row["provider_endpoint"] for row in rows} == {"deepseek"}
    assert {row["quantization"] for row in rows} == {"unknown"}
    assert {Path(row["wrapper"]).name for row in rows} == {"deepseek_harness.sh"}
    assert {row["agent_kind"] for row in rows} == {"deepseek-harness"}
    assert {row["goal_mode"] for row in rows} == {"deepseek_native_goal"}


def test_batch_matrix_can_launch_three_luna_and_three_sol_trials() -> None:
    rows = batch_eval.matrix(
        "eval-openai",
        families=("luna", "sol"),
    )
    assert len(rows) == 6
    assert {row["family"] for row in rows} == {"luna", "sol"}
    assert {row["wrapper"] for row in rows} == {
        str(ROOT / "event_runtime/control/providers/openai.sh")
    }
    assert {row["model"] for row in rows if row["family"] == "luna"} == {
        "openai/gpt-5.6-luna"
    }
    assert {row["model"] for row in rows if row["family"] == "sol"} == {
        "openai/gpt-5.6-sol"
    }
    assert {row["openrouter_preset"] for row in rows} == {
        "@preset/sprint-gpt-5-6-luna-openai-standard",
        "@preset/sprint-gpt-5-6-sol-openai-standard",
    }
    assert [row["trial"] for row in rows if row["family"] == "sol"] == [1, 2, 3]


def test_batch_matrix_rejects_historical_nonofficial_provider_routes() -> None:
    with pytest.raises(ValueError, match="invalid model families"):
        batch_eval.matrix(
            "eval-ds-routes",
            families=("flash-baidu", "pro-alibaba"),
        )


def test_every_launchable_family_uses_its_official_openrouter_provider() -> None:
    rows = batch_eval.matrix(
        "eval-official-routes",
        families=batch_eval.SUPPORTED_FAMILIES,
    )
    assert rows
    for row in rows:
        assert row["provider_endpoint"].split("/", 1)[0] == row["model"].split(
            "/", 1
        )[0]
        assert row["provider"] in {"Anthropic", "DeepSeek", "OpenAI", "Z.AI"}
    assert {row["reasoning_effort"] for row in rows if row["family"] == "opus"} == {
        "medium"
    }
    assert {row["reasoning_effort"] for row in rows if row["family"] == "glm"} == {
        "max"
    }


def test_sol_model_lock_preserves_exact_codex_contract() -> None:
    lock = json.loads((ROOT / "event_runtime/models/sol.json").read_text())
    model = lock["model"]
    assert lock["source"].startswith("openai/codex rust-v0.149.1")
    assert lock["codex_version"] == "0.149.1"
    assert lock["model_messages_sha256"] == (
        "e1ab3222ab4ceb4196f381138bf63232456419dba5a03bc276137a323e4134aa"
    )
    assert model["slug"] == "gpt-5.6-sol"
    assert model["tool_mode"] == "code_mode_only"
    assert model["multi_agent_version"] == "v2"
    assert model["context_window"] == 272000
    assert model["max_context_window"] == 872000
    assert [row["effort"] for row in model["supported_reasoning_levels"]][-1] == "ultra"


def test_generic_openai_catalog_installer_supports_sol(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    env = {
        **os.environ,
        "CODEX_HOME": str(codex_home),
        "SPRINT_CODEX_OPENAI_MODEL_LOCK": str(ROOT / "event_runtime/models/sol.json"),
        "SPRINT_CODEX_OPENAI_MODEL_ID": "gpt-5.6-sol",
        "SPRINT_CODEX_OPENAI_MODEL": "@preset/test-sol",
        "SPRINT_CODEX_OPENAI_BASE_URL": "http://127.0.0.1:18080/api/v1",
        "SPRINT_CODEX_MODEL_MESSAGES_TEMPLATE_JSON": str(
            ROOT / "event_runtime/models/deepseek.json"
        ),
    }
    subprocess.run(
        [
            "bash",
            str(ROOT / "event_runtime/container/sprint-apply-openai-codex-config.sh"),
        ],
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    catalog = json.loads((codex_home / "models.json").read_text())
    model = catalog["models"][0]
    assert model["slug"] == "@preset/test-sol"
    assert model["multi_agent_version"] == "v2"
    assert model["support_verbosity"] is True
    assert model["supports_parallel_tool_calls"] is False
    config = (codex_home / "config.toml").read_text()
    assert 'model = "@preset/test-sol"' in config
    assert 'wire_api = "responses"' in config


def test_openai_openrouter_catalogs_match_codex_0_149_capabilities() -> None:
    for name in ("luna", "sol"):
        lock = json.loads((ROOT / f"event_runtime/models/{name}.json").read_text())
        assert lock["codex_version"] == "0.149.1"
        assert lock["model"]["support_verbosity"] is True
        assert lock["model"]["max_context_window"] == 872000
        assert "supports_parallel_tool_calls" not in lock["model"]


@pytest.mark.parametrize(
    "model_args",
    [
        ["--model", "gpt-5.6-luna"],
        ["-m", "gpt-5.6-luna"],
        ["--model=gpt-5.6-luna"],
    ],
)
def test_codex_wrapper_selects_pinned_openai_catalog_slug(
    tmp_path: Path, model_args: list[str]
) -> None:
    runtime = tmp_path / "run"
    logs = tmp_path / "logs"
    codex_home = tmp_path / "codex-home"
    durable = tmp_path / "durable"
    for path in (runtime, logs, codex_home, durable):
        path.mkdir(parents=True)

    recorded_args = tmp_path / "args.txt"
    native = tmp_path / "codex-native"
    native.write_text(
        f"#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > {recorded_args!s}\nsleep 0.5\n"
    )
    native.chmod(0o755)
    apply = tmp_path / "apply-openai.sh"
    apply.write_text("#!/usr/bin/env bash\nexit 0\n")
    apply.chmod(0o755)

    env = {
        **os.environ,
        "CODEX_HOME": str(codex_home),
        "SPRINT_RUNTIME_DIR": str(runtime),
        "SPRINT_AGENT_LOG_DIR": str(logs),
        "SPRINT_DURABLE_DIR": str(durable),
        "SPRINT_MODEL": "openai/gpt-5.6-luna",
        "SPRINT_CODEX_EXECUTABLE": str(native),
        "SPRINT_CODEX_OPENAI_MODEL": "@preset/sprint-gpt-5-6-luna-openai-standard",
        "SPRINT_APPLY_OPENAI_CODEX_CONFIG": str(apply),
    }
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "event_runtime/container/sprint-codex-exec-wrapper.sh"),
            "/unused/codex",
            "exec",
            "--json",
            *model_args,
            "--",
            "test prompt",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    args = recorded_args.read_text().splitlines()
    assert "gpt-5.6-luna" not in args
    assert "--model=gpt-5.6-luna" not in args
    if model_args[0].startswith("--model="):
        assert "--model=@preset/sprint-gpt-5-6-luna-openai-standard" in args
    else:
        selector = args.index(model_args[0])
        assert args[selector + 1] == "@preset/sprint-gpt-5-6-luna-openai-standard"


def test_deepseek_catalog_never_emits_unsupported_verbosity() -> None:
    catalog = json.loads((ROOT / "event_runtime/models/deepseek.json").read_text())
    assert catalog["models"]
    assert all(model["support_verbosity"] is False for model in catalog["models"])
    assert all(
        model["supports_parallel_tool_calls"] is False for model in catalog["models"]
    )


@pytest.mark.parametrize("trials_per_model", [2, 5])
def test_batch_matrix_supports_staged_trial_counts(trials_per_model: int) -> None:
    rows = batch_eval.matrix(
        "eval-staged",
        trials_per_model=trials_per_model,
    )
    assert len(rows) == 2 * trials_per_model
    assert sum(row["family"] == "deepseek" for row in rows) == trials_per_model
    assert sum(row["family"] == "luna" for row in rows) == trials_per_model
    assert [row["trial"] for row in rows if row["family"] == "deepseek"] == list(
        range(1, trials_per_model + 1)
    )


@pytest.mark.parametrize("trials_per_model", [0, 51, True])
def test_batch_matrix_rejects_invalid_trial_counts(trials_per_model: int) -> None:
    with pytest.raises(ValueError, match="trials per model"):
        batch_eval.matrix("eval-invalid", trials_per_model=trials_per_model)


def test_env_loader_reads_only_required_launch_settings(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        'OPENROUTER_API_KEY="openrouter-secret"\n'
        "OPENROUTER_MANAGEMENT_KEY=management-secret\n"
        "SPRINT_OPENROUTER_AUTO_RECHARGE_CONFIRMED=1\n"
        "MODAL_TOKEN_SECRET=must-not-load\n"
    )
    assert batch_eval.load_env(path) == {
        "OPENROUTER_MANAGEMENT_KEY": "management-secret",
        "SPRINT_OPENROUTER_AUTO_RECHARGE_CONFIRMED": "1",
    }


def test_local_provider_billed_cost_uses_durable_summary_when_live_field_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "eval-luna-1"
    state_dir = tmp_path / run_id
    (state_dir / "telemetry").mkdir(parents=True)
    (state_dir / "telemetry" / "agent-cost.json").write_text(
        json.dumps(
            {
                "as_of": "2026-08-23T00:01:00Z",
                "components": {"model_api": {"cost_usd": 1.25}},
            }
        )
    )
    summary_dir = state_dir / "provider-api-usage" / "api-usage"
    summary_dir.mkdir(parents=True)
    (summary_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "run_id": run_id,
                "provider_billed_model_api_usd": 1.25,
                "pending_request_count": 1,
                "in_flight_request_count": 1,
                "updated_at": "2026-08-23T00:01:01Z",
                "token_usage": empty_token_usage(),
            }
        )
    )
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", tmp_path)

    assert batch_eval._local_provider_billed_cost(run_id) == (
        1.25,
        1,
        "2026-08-23T00:01:01Z",
    )


def test_local_provider_billed_cost_combines_newest_trusted_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "eval-luna-1"
    state_dir = tmp_path / run_id
    (state_dir / "telemetry").mkdir(parents=True)
    (state_dir / "telemetry" / "agent-cost.json").write_text(
        json.dumps(
            {
                "as_of": "2026-08-23T00:01:02Z",
                "components": {
                    "model_api": {
                        "provider_billed_cost_usd": 1.30,
                        "pending_request_count": 0,
                    }
                },
            }
        )
    )
    summary_dir = state_dir / "provider-api-usage" / "api-usage"
    summary_dir.mkdir(parents=True)
    (summary_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "run_id": run_id,
                "provider_billed_model_api_usd": 1.25,
                "pending_request_count": 1,
                "updated_at": "2026-08-23T00:01:01Z",
                "token_usage": empty_token_usage(),
            }
        )
    )
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", tmp_path)

    assert batch_eval._local_provider_billed_cost(run_id) == (
        1.30,
        1,
        "2026-08-23T00:01:02Z",
    )


def test_local_provider_billed_cost_accepts_live_epoch_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "eval-luna-1"
    state_dir = tmp_path / run_id
    (state_dir / "telemetry").mkdir(parents=True)
    (state_dir / "telemetry" / "agent-cost.json").write_text(
        json.dumps(
            {
                "as_of": None,
                "as_of_epoch_ms": 1787443323456,
                "components": {
                    "model_api": {
                        "provider_billed_cost_usd": 1.30,
                        "pending_request_count": 1,
                    }
                },
            }
        )
    )
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", tmp_path)

    assert batch_eval._local_provider_billed_cost(run_id) == (
        1.30,
        1,
        "2026-08-23T00:02:03.456Z",
    )


def test_child_key_usage_audit_does_not_call_missing_ledger_a_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "arms": [
            {
                "run_id": "eval-luna-1",
                "status": "running",
                "openrouter_credential": {"key_hash": "hash-1"},
            }
        ],
        "alerts": [],
    }

    class Client:
        def keys_usage(self, key_hashes: set[str]) -> dict[str, dict[str, float]]:
            assert key_hashes == {"hash-1"}
            return {"hash-1": {"usage": 0.25}}

    monkeypatch.setattr(
        batch_eval,
        "_local_provider_billed_cost",
        lambda _run_id: (0.0, 0, None),
    )
    stopped: list[tuple[str, str]] = []
    monkeypatch.setattr(
        batch_eval.sprintctl,
        "persist_stop_request",
        lambda run_id, *, reason: stopped.append((run_id, reason)),
    )

    alerts = batch_eval.audit_openrouter_child_usage(
        payload,
        Client(),
        now=dt.datetime(2026, 8, 23, tzinfo=dt.timezone.utc),
    )

    assert alerts == [
        {
            "run_id": "eval-luna-1",
            "kind": "openrouter_key_usage_audit",
            "source": "OpenRouterManagementError",
            "count_in_tail": "1",
        }
    ]
    assert stopped == []
    assert "openrouter_usage_audit" not in payload["arms"][0]


@pytest.mark.parametrize("status", ["stopping", "stopped", "finalized"])
def test_child_key_usage_audit_skips_terminal_credentials(status: str) -> None:
    payload = {
        "arms": [
            {
                "run_id": "eval-luna-1",
                "status": status,
                "openrouter_credential": {"key_hash": "revoked-hash"},
            }
        ],
        "alerts": [],
    }

    class Client:
        def keys_usage(self, _key_hashes: set[str]):
            raise AssertionError("terminal credentials must not be audited")

    assert (
        batch_eval.audit_openrouter_child_usage(
            payload,
            Client(),
            now=dt.datetime(2026, 8, 23, tzinfo=dt.timezone.utc),
        )
        == []
    )


def test_child_key_usage_audit_skips_acknowledged_stop() -> None:
    payload = {
        "arms": [
            {
                "run_id": "eval-luna-1",
                "status": "running",
                "stop_ack": {"reason": "budget_telemetry_unavailable"},
                "openrouter_credential": {"key_hash": "revoked-hash"},
            }
        ],
        "alerts": [],
    }

    class Client:
        def keys_usage(self, _key_hashes: set[str]):
            raise AssertionError("acknowledged stop credentials must not be audited")

    assert (
        batch_eval.audit_openrouter_child_usage(
            payload,
            Client(),
            now=dt.datetime(2026, 8, 23, tzinfo=dt.timezone.utc),
        )
        == []
    )


def test_child_key_usage_audit_defers_missing_ledger_during_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "arms": [
            {
                "run_id": "eval-luna-1",
                "status": "launched",
                "launched_at": "2026-08-23T00:00:00Z",
                "openrouter_credential": {"key_hash": "hash-1"},
            }
        ],
        "alerts": [],
    }

    class Client:
        def keys_usage(self, key_hashes: set[str]) -> dict[str, dict[str, float]]:
            assert key_hashes == {"hash-1"}
            return {"hash-1": {"usage": 0.01}}

    monkeypatch.setattr(
        batch_eval,
        "_local_provider_billed_cost",
        lambda _run_id: (0.0, 0, None),
    )

    alerts = batch_eval.audit_openrouter_child_usage(
        payload,
        Client(),
        now=dt.datetime(2026, 8, 23, 0, 1, tzinfo=dt.timezone.utc),
    )

    assert alerts == []
    audit = payload["arms"][0]["openrouter_usage_audit"]
    assert audit["reconciliation_deferred"] == "trusted_proxy_ledger_starting"
    assert audit["startup_age_seconds"] == 60.0
    assert audit["key_usage_usd"] == 0.01


def test_child_key_usage_audit_defers_while_proxy_request_is_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "arms": [
            {
                "run_id": "eval-luna-1",
                "status": "running",
                "openrouter_credential": {"key_hash": "hash-1"},
                "openrouter_usage_audit": {
                    "mismatch_first_seen_at": "2026-08-22T23:00:00Z"
                },
            }
        ],
        "alerts": [],
    }

    class Client:
        def keys_usage(self, key_hashes: set[str]) -> dict[str, dict[str, float]]:
            assert key_hashes == {"hash-1"}
            return {"hash-1": {"usage": 0.25}}

    monkeypatch.setattr(
        batch_eval,
        "_local_provider_billed_cost",
        lambda _run_id: (0.10, 1, "2026-08-23T00:00:00Z"),
    )
    stopped: list[tuple[str, str]] = []
    monkeypatch.setattr(
        batch_eval.sprintctl,
        "persist_stop_request",
        lambda run_id, *, reason: stopped.append((run_id, reason)),
    )

    alerts = batch_eval.audit_openrouter_child_usage(
        payload,
        Client(),
        now=dt.datetime(2026, 8, 23, 1, tzinfo=dt.timezone.utc),
    )

    assert alerts == []
    assert stopped == []
    audit = payload["arms"][0]["openrouter_usage_audit"]
    assert "mismatch_first_seen_at" not in audit
    assert audit["reconciliation_deferred"] == "trusted_proxy_request_in_flight"


def test_child_key_usage_audit_fetches_all_arms_in_one_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "arms": [
            {
                "run_id": f"eval-luna-{index}",
                "status": "running",
                "openrouter_credential": {"key_hash": f"hash-{index}"},
            }
            for index in (1, 2)
        ],
        "alerts": [],
    }
    calls: list[set[str]] = []

    class Client:
        def keys_usage(self, key_hashes: set[str]) -> dict[str, dict[str, float]]:
            calls.append(key_hashes)
            return {
                key_hash: {"usage": float(index)}
                for index, key_hash in enumerate(sorted(key_hashes), start=1)
            }

    monkeypatch.setattr(
        batch_eval,
        "_local_provider_billed_cost",
        lambda run_id: (
            1.0 if run_id.endswith("-1") else 2.0,
            0,
            "2026-08-23T00:00:00Z",
        ),
    )

    assert (
        batch_eval.audit_openrouter_child_usage(
            payload,
            Client(),
            now=dt.datetime(2026, 8, 23, tzinfo=dt.timezone.utc),
        )
        == []
    )
    assert calls == [{"hash-1", "hash-2"}]


def test_controller_recovers_generation_and_uploads_summary_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "eval-luna-recovery"
    ledger = tmp_path / run_id / "provider-api-usage" / "api-usage"
    requests = ledger / "requests"
    requests.mkdir(parents=True)
    complete = {
        "run_id": run_id,
        "ledger_request_id": "a" * 32,
        "state": "complete",
        "provider_reported_cost_usd": 0.1,
        "benchmark_cost_usd": 0.1,
        "cost_basis": "openrouter_list_price_before_endpoint_discount",
    }
    pending = {
        "schema_version": 3,
        "run_id": run_id,
        "cpu_attempt": 1,
        "ledger_request_id": "b" * 32,
        "generation_id": "gen-recover",
        "state": "in_flight",
        "promotion_snapshot": {
            "discount_fraction": 0.0,
            "deepseek_peak_pricing_usd_per_token": None,
            "cost_basis": "openrouter_list_price_before_endpoint_discount",
        },
    }
    (requests / f"{'a' * 32}.json").write_text(json.dumps(complete))
    (requests / f"{'b' * 32}.json").write_text(json.dumps(pending))
    (ledger / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "run_id": run_id,
                "provider_billed_model_api_usd": 0.1,
                "pending_request_count": 1,
                "token_usage": empty_token_usage(),
            }
        )
    )
    arm = {
        "run_id": run_id,
        "model": "openai/gpt-5.6-luna",
        "resolved_model_version": "openai/gpt-5.6-luna-20260709",
        "provider": "OpenAI",
    }

    class Client:
        def generation_usage(self, generation_id: str) -> dict[str, object]:
            assert generation_id == "gen-recover"
            return {
                "total_cost": 0.03,
                "native_tokens_prompt": 100,
                "native_tokens_cached": 0,
                "native_tokens_completion": 10,
                "model": "openai/gpt-5.6-luna-20260709",
                "provider": "OpenAI",
            }

    uploads: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(
        batch_eval.sprintctl,
        "load_run",
        lambda _run_id: (
            tmp_path / run_id,
            {
                "volume_name": "volume",
                "budget_enforcement": {
                    "api_budget_cost_basis": (
                        "openrouter_list_price_before_endpoint_discount"
                    )
                },
            },
        ),
    )
    monkeypatch.setattr(
        batch_eval.sprintctl,
        "volume_upload",
        lambda _run, source, remote: uploads.append(
            (remote, json.loads(source.read_text()))
        ),
    )

    assert batch_eval.reconcile_openrouter_child_ledger(
        arm, Client(), upstream_usage=0.13
    )

    recovered = json.loads((requests / f"{'b' * 32}.json").read_text())
    summary = json.loads((ledger / "summary.json").read_text())
    assert recovered["state"] == "recovered_complete"
    assert recovered["usage"]["input_tokens"] == 100
    assert summary["pending_request_count"] == 0
    assert summary["provider_billed_model_api_usd"] == pytest.approx(0.13)
    assert summary["token_usage"] == {
        "input_tokens": 100,
        "ordinary_uncached_input_tokens": 100,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "output_tokens": 10,
        "reasoning_output_tokens": 0,
        "total_tokens": 110,
    }
    assert uploads[-1][0].endswith("/api-usage/summary.json")


def test_controller_marks_generationless_zero_delta_unbilled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "eval-deepseek-unbilled"
    ledger = tmp_path / run_id / "provider-api-usage" / "api-usage"
    requests = ledger / "requests"
    requests.mkdir(parents=True)
    request_id = "c" * 32
    (requests / f"{request_id}.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "run_id": run_id,
                "ledger_request_id": request_id,
                "state": "cost_recovery_required",
                "generation_id": None,
            }
        )
    )
    (ledger / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "run_id": run_id,
                "provider_billed_model_api_usd": 0.0,
                "pending_request_count": 1,
                "token_usage": empty_token_usage(),
            }
        )
    )
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(
        batch_eval.sprintctl,
        "load_run",
        lambda _run_id: (
            tmp_path / run_id,
            {
                "volume_name": "volume",
                "budget_enforcement": {
                    "api_budget_cost_basis": (
                        "openrouter_list_price_with_deepseek_peak_floor"
                    )
                },
            },
        ),
    )
    monkeypatch.setattr(
        batch_eval.sprintctl, "volume_upload", lambda *_args, **_kwargs: None
    )

    assert batch_eval.reconcile_openrouter_child_ledger(
        {"run_id": run_id}, object(), upstream_usage=0.0
    )
    record = json.loads((requests / f"{request_id}.json").read_text())
    assert record["state"] == "rejected_not_billed"
    assert record["provider_reported_cost_usd"] == 0.0
    assert record["reconciled_from_child_key_total"] is True


def test_child_key_usage_audit_resolves_prior_health_alert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "arms": [
            {
                "run_id": "eval-luna-1",
                "status": "running",
                "openrouter_credential": {"key_hash": "hash-1"},
            }
        ],
        "alerts": [
            {
                "run_id": "eval-luna-1",
                "kind": "openrouter_key_usage_audit",
                "source": "OpenRouterManagementError",
            }
        ],
    }

    class Client:
        def keys_usage(self, key_hashes: set[str]) -> dict[str, dict[str, float]]:
            assert key_hashes == {"hash-1"}
            return {"hash-1": {"usage": 0.25}}

    monkeypatch.setattr(
        batch_eval,
        "_local_provider_billed_cost",
        lambda _run_id: (0.25, 0, "2026-08-23T00:00:00Z"),
    )

    assert (
        batch_eval.audit_openrouter_child_usage(
            payload,
            Client(),
            now=dt.datetime(2026, 8, 23, 1, tzinfo=dt.timezone.utc),
        )
        == []
    )
    assert payload["alerts"] == []
    assert payload["resolved_alerts"][0]["kind"] == "openrouter_key_usage_audit"


def test_child_key_usage_audit_stops_persistent_proxy_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "arms": [
            {
                "run_id": "eval-luna-1",
                "status": "running",
                "openrouter_credential": {"key_hash": "hash-1"},
            }
        ],
        "alerts": [],
    }

    class Client:
        def keys_usage(self, key_hashes: set[str]) -> dict[str, dict[str, float]]:
            assert key_hashes == {"hash-1"}
            return {"hash-1": {"usage": 0.25}}

    monkeypatch.setattr(
        batch_eval,
        "_local_provider_billed_cost",
        lambda _run_id: (0.10, 0, "2026-08-23T00:00:00Z"),
    )
    stopped: list[tuple[str, str]] = []
    monkeypatch.setattr(
        batch_eval.sprintctl,
        "persist_stop_request",
        lambda run_id, *, reason: stopped.append((run_id, reason)),
    )
    first_now = dt.datetime(2026, 8, 23, tzinfo=dt.timezone.utc)
    assert (
        batch_eval.audit_openrouter_child_usage(payload, Client(), now=first_now) == []
    )
    payload["arms"][0]["openrouter_usage_audit"]["mismatch_first_seen_at"] = (
        first_now.isoformat().replace("+00:00", "Z")
    )

    alerts = batch_eval.audit_openrouter_child_usage(
        payload,
        Client(),
        now=first_now
        + dt.timedelta(seconds=batch_eval.OPENROUTER_USAGE_AUDIT_GRACE_SECONDS + 1),
    )

    assert alerts[0]["kind"] == "openrouter_proxy_bypass"
    assert stopped == [("eval-luna-1", "openrouter_proxy_bypass_detected")]
    assert payload["arms"][0]["status"] == "stopping"


def test_openrouter_credit_requirement_covers_full_matrix_with_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGENT_COST_BUDGET_USD", raising=False)

    requirement = batch_eval.openrouter_credit_requirement(6)

    assert requirement == {
        "per_trial_budget_usd": 10.0,
        "trial_count": 6,
        "maximum_combined_budget_usd": 60.0,
        "safety_factor": 1.05,
        "required_credit_usd": 63.0,
    }


def test_openrouter_credit_query_reports_remaining_without_secrets() -> None:
    response = io.BytesIO(
        json.dumps({"data": {"total_credits": 100.0, "total_usage": 25.25}}).encode()
    )
    with mock.patch.object(
        batch_eval.urllib.request, "urlopen", return_value=response
    ) as urlopen:
        snapshot = batch_eval.fetch_openrouter_credit("secret-key")

    assert snapshot == {
        "total_credits_usd": 100.0,
        "total_usage_usd": 25.25,
        "remaining_credit_usd": 74.75,
    }
    request = urlopen.call_args.args[0]
    assert request.full_url == "https://openrouter.ai/api/v1/credits"
    assert "secret-key" not in json.dumps(snapshot)


def test_functional_gpu_canary_must_match_both_warmed_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warmup = tmp_path / "warmup.json"
    canary = tmp_path / "canary.json"
    fixture = tmp_path / "train_sprint.py"
    fixture.write_text("print('pinned canary')\n")
    fixture_sha256 = hashlib.sha256(fixture.read_bytes()).hexdigest()
    warmup.write_text(
        json.dumps(
            {
                "completed": True,
                "contexts": {
                    "agent_training": {"image_id": "im-agent"},
                    "verifier": {"image_id": "im-verifier"},
                },
            }
        )
    )
    canary.write_text(
        json.dumps(
            {
                "schema_version": 5,
                "completed": True,
                "full_path_verified": True,
                "verifier_equivalence_verified": True,
                "cost_equivalence_verified": True,
                "gpu_budget_mirror_verified": True,
                "gpu_budget_mirror": {
                    "completed": True,
                    "updates_verified": 2,
                    "observed_sequences": [1, 2],
                },
                "cost_equivalence": {
                    "completed": True,
                    "comparison": {
                        "verified": True,
                        "tolerance_usd": 1e-9,
                        "max_absolute_delta_usd": 0.0,
                        "comparisons": {
                            "model_api_usd": {},
                            "cpu_agent_usd": {},
                            "training_sandboxes_usd": {},
                            "total_usd": {},
                        },
                    },
                },
                "image_id": "im-agent",
                "verifier_image_id": "im-verifier",
                "training_fixture_sha256": fixture_sha256,
            }
        )
    )
    monkeypatch.setattr(batch_eval, "WARMUP_MANIFEST", warmup)
    monkeypatch.setattr(batch_eval, "FUNCTIONAL_CANARY_REPORT", canary)
    monkeypatch.setattr(batch_eval, "FUNCTIONAL_CANARY_FIXTURE", fixture)
    assert batch_eval.functional_gpu_canary_ready()
    payload = json.loads(canary.read_text())
    payload["verifier_image_id"] = "im-stale"
    canary.write_text(json.dumps(payload))
    assert not batch_eval.functional_gpu_canary_ready()

    payload["verifier_image_id"] = "im-verifier"
    canary.write_text(json.dumps(payload))
    fixture.write_text("print('changed canary')\n")
    assert not batch_eval.functional_gpu_canary_ready()


def test_functional_gpu_canary_fails_closed_on_cost_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warmup = tmp_path / "warmup.json"
    canary = tmp_path / "canary.json"
    warmup.write_text(
        json.dumps(
            {
                "completed": True,
                "contexts": {
                    "agent_training": {"image_id": "im-agent"},
                    "verifier": {"image_id": "im-verifier"},
                },
            }
        )
    )
    canary.write_text(
        json.dumps(
            {
                "schema_version": 5,
                "completed": True,
                "full_path_verified": True,
                "verifier_equivalence_verified": True,
                "cost_equivalence_verified": False,
                "gpu_budget_mirror_verified": True,
                "gpu_budget_mirror": {
                    "completed": True,
                    "updates_verified": 2,
                    "observed_sequences": [1, 2],
                },
                "cost_equivalence": {
                    "completed": True,
                    "comparison": {"verified": False},
                },
                "image_id": "im-agent",
                "verifier_image_id": "im-verifier",
            }
        )
    )
    monkeypatch.setattr(batch_eval, "WARMUP_MANIFEST", warmup)
    monkeypatch.setattr(batch_eval, "FUNCTIONAL_CANARY_REPORT", canary)
    assert not batch_eval.functional_gpu_canary_ready()


def test_cost_canary_compares_every_component_and_fails_closed() -> None:
    payload = {
        "total_usd": 1.0,
        "components": {
            "model_api": {"cost_usd": 0.1},
            "cpu_agent": {"cost_usd": 0.2},
            "training_sandboxes": {"cost_usd": 0.7},
        },
    }
    expected = {
        "model_api_usd": 0.1,
        "cpu_agent_usd": 0.2,
        "training_sandboxes_usd": 0.7,
        "total_usd": 1.0,
    }
    result = training_gpu_canary._assert_cost_match(payload, expected=expected)
    assert result["verified"]
    assert result["max_absolute_delta_usd"] == 0

    expected["training_sandboxes_usd"] = 0.70001
    with pytest.raises(RuntimeError, match="event cost differs"):
        training_gpu_canary._assert_cost_match(payload, expected=expected)


def test_training_gpu_fleet_probe_requires_every_exact_worker_and_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warmup = tmp_path / "warmup.json"
    warmup.write_text(
        json.dumps(
            {
                "completed": True,
                "contexts": {"agent_training": {"image_id": "im-exact"}},
            }
        )
    )
    monkeypatch.setattr(batch_eval, "WARMUP_MANIFEST", warmup)
    monkeypatch.setattr(
        batch_eval.tempfile,
        "TemporaryDirectory",
        lambda **_kwargs: contextlib.nullcontext(str(tmp_path)),
    )

    def successful_run(command, **_kwargs):
        report = Path(command[command.index("--report") + 1])
        worker_ids = [
            command[index + 1]
            for index, value in enumerate(command)
            if value == "--worker-id"
        ]
        report.write_text(
            json.dumps(
                {
                    "completed": True,
                    "image_id": "im-exact",
                    "worker_ids": worker_ids,
                    "cpu_workers": [
                        {"worker_id": worker_id, "ready": True}
                        for worker_id in worker_ids
                    ],
                    "training_gpu_workers": [
                        {"worker_id": worker_id, "ready": True}
                        for worker_id in worker_ids
                    ],
                }
            )
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok")

    monkeypatch.setattr(batch_eval.subprocess, "run", successful_run)
    planned_workers = [f"eval-{index}" for index in range(1, 16)]
    ready, report = batch_eval.training_gpu_fleet_probe(
        batch_id="eval", modal_profile="test", worker_ids=planned_workers
    )
    assert ready
    assert [row["worker_id"] for row in report["cpu_workers"]] == planned_workers
    assert [
        row["worker_id"] for row in report["training_gpu_workers"]
    ] == planned_workers

    def incomplete_run(command, **kwargs):
        result = successful_run(command, **kwargs)
        report = Path(command[command.index("--report") + 1])
        payload = json.loads(report.read_text())
        payload["cpu_workers"].pop()
        report.write_text(json.dumps(payload))
        return result

    monkeypatch.setattr(batch_eval.subprocess, "run", incomplete_run)
    ready, _ = batch_eval.training_gpu_fleet_probe(
        batch_id="eval", modal_profile="test", worker_ids=planned_workers
    )
    assert not ready


def test_sprint_modal_resource_audit_rejects_only_live_sprint_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def dirty_listing(command, **_kwargs):
        calls.append(command)
        if "container" in command:
            return json.dumps(
                [
                    {
                        "container_id": "ta-sprint",
                        "app_id": "ap-sprint",
                        "app_name": "sprint-old-run",
                        "start_time": "now",
                    },
                    {
                        "container_id": "ta-unrelated",
                        "app_id": "ap-unrelated",
                        "app_name": "kevin-unrelated",
                        "start_time": "now",
                    },
                ]
            )
        return json.dumps(
            [
                {
                    "app_id": "ap-sprint",
                    "description": "sprint-old-run",
                    "state": "deployed",
                    "tasks": "0",
                },
                {
                    "app_id": "ap-stopped",
                    "description": "sprint-finished-run",
                    "state": "stopped",
                    "tasks": "0",
                },
                {
                    "app_id": "ap-unrelated",
                    "description": "kevin-unrelated",
                    "state": "deployed",
                    "tasks": "1",
                },
            ]
        )

    monkeypatch.setattr(batch_eval, "run_checked", dirty_listing)
    ready, report = batch_eval.sprint_modal_resource_audit(modal_profile="test")
    assert not ready
    assert report["live_sprint_apps"] == [
        {
            "app_id": "ap-sprint",
            "description": "sprint-old-run",
            "state": "deployed",
            "tasks": "0",
        }
    ]
    assert report["live_sprint_containers"][0]["container_id"] == "ta-sprint"
    assert report["unrelated_resources_ignored"] is True
    assert len(calls) == 2


def test_sprint_modal_resource_audit_accepts_stopped_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def clean_listing(command, **_kwargs):
        if "container" in command:
            return "[]"
        return json.dumps(
            [
                {
                    "app_id": "ap-stopped",
                    "description": "sprint-finished-run",
                    "state": "stopped",
                    "tasks": "0",
                },
                {
                    "app_id": "ap-unrelated",
                    "description": "kevin-unrelated",
                    "state": "deployed",
                    "tasks": "1",
                },
            ]
        )

    monkeypatch.setattr(batch_eval, "run_checked", clean_listing)
    ready, report = batch_eval.sprint_modal_resource_audit(modal_profile="test")
    assert ready
    assert report["live_sprint_apps"] == []
    assert report["live_sprint_containers"] == []


def test_sprint_modal_resource_audit_allows_explicit_companion_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def listing(command, **_kwargs):
        if "container" in command:
            return json.dumps(
                [
                    {
                        "container_id": "ta-companion",
                        "app_id": "ap-companion",
                        "app_name": "sprint-companion-luna-1-training",
                        "start_time": "now",
                    }
                ]
            )
        return json.dumps(
            [
                {
                    "app_id": "ap-companion",
                    "description": "sprint-companion-luna-1",
                    "state": "deployed",
                    "tasks": "1",
                }
            ]
        )

    monkeypatch.setattr(batch_eval, "run_checked", listing)
    ready, report = batch_eval.sprint_modal_resource_audit(
        modal_profile="test", allowed_app_prefixes=("sprint-companion-",)
    )
    assert ready
    assert report["live_sprint_apps"] == []
    assert report["live_sprint_containers"] == []
    assert report["allowed_live_prefixes"] == ["sprint-companion-"]


def test_vercel_daily_quota_error_backs_off_for_24_hours() -> None:
    observed = dt.datetime(2026, 8, 9, 23, 45, tzinfo=dt.timezone.utc)
    state: dict[str, object] = {}
    batch_eval.record_deployment_error(
        state,
        RuntimeError(
            "Resource is limited - try again in 24 hours "
            '(more than 100, code: "api-deployments-free-per-day")'
        ),
        now=observed,
    )
    assert state["site_status"] == "quota_limited"
    assert state["quota_code"] == "api-deployments-free-per-day"
    assert state["retry_not_before"] == "2026-08-10T23:45:00Z"
    assert not batch_eval.deployment_retry_due(
        state, now=observed + dt.timedelta(hours=23, minutes=59)
    )
    assert batch_eval.deployment_retry_due(state, now=observed + dt.timedelta(hours=24))


def test_nonquota_deployment_error_remains_normally_retryable() -> None:
    state: dict[str, object] = {}
    batch_eval.record_deployment_error(state, RuntimeError("transient deploy failure"))
    assert state["site_status"] == "error"
    assert "retry_not_before" not in state
    assert batch_eval.deployment_retry_due(state)


def test_provider_inference_probe_records_usage_without_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return io.BytesIO(
            json.dumps(
                {
                    "id": "resp_test",
                    "model": "gpt-5.6-luna",
                    "status": "completed",
                    "service_tier": "default",
                    "usage": {"input_tokens": 8, "output_tokens": 1},
                }
            ).encode()
        )

    monkeypatch.setattr(batch_eval.urllib.request, "urlopen", fake_urlopen)
    result = batch_eval.provider_inference_probe(
        "https://api.openai.com/v1/responses",
        "top-secret-key",
        {"model": "gpt-5.6-luna", "input": "Return OK."},
    )

    request = captured["request"]
    assert captured["timeout"] == 120
    assert json.loads(request.data) == {
        "model": "gpt-5.6-luna",
        "input": "Return OK.",
    }
    assert request.get_header("Authorization") == "Bearer top-secret-key"
    assert result["request_id"] == "resp_test"
    assert result["usage"] == {"input_tokens": 8, "output_tokens": 1}
    assert "top-secret-key" not in json.dumps(result)


def test_provider_model_discovery_retries_transport_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    delays: list[float] = []

    def flaky_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        assert timeout == 30
        assert request.get_header("Authorization") == "Bearer top-secret-key"
        if calls < 3:
            raise TimeoutError("timed out")
        return io.BytesIO(json.dumps({"data": [{"id": "gpt-test"}]}).encode())

    monkeypatch.setattr(batch_eval.urllib.request, "urlopen", flaky_urlopen)
    monkeypatch.setattr(batch_eval.time, "sleep", delays.append)

    assert batch_eval.provider_models(
        "https://api.example/models", "top-secret-key"
    ) == {"gpt-test"}
    assert calls == 3
    assert delays == [1.0, 2.0]


def test_provider_model_discovery_does_not_retry_auth_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fail_urlopen(_request, timeout):
        nonlocal calls
        calls += 1
        assert timeout == 30
        raise urllib.error.HTTPError(
            "https://api.example/models", 401, "Unauthorized", {}, io.BytesIO()
        )

    monkeypatch.setattr(batch_eval.urllib.request, "urlopen", fail_urlopen)
    with pytest.raises(RuntimeError, match="HTTP 401"):
        batch_eval.provider_models("https://api.example/models", "top-secret-key")
    assert calls == 1


def test_provider_inference_probe_retries_transient_server_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    delays: list[float] = []

    def flaky_urlopen(_request, timeout):
        nonlocal calls
        calls += 1
        assert timeout == 120
        if calls < 3:
            raise urllib.error.HTTPError(
                "https://api.openai.com/v1/responses",
                500,
                "Server Error",
                {},
                io.BytesIO(b'{"error":{"type":"server_error"}}'),
            )
        return io.BytesIO(
            json.dumps(
                {
                    "id": "resp_retry",
                    "model": "gpt-5.6-luna",
                    "status": "completed",
                    "usage": {"total_tokens": 10},
                }
            ).encode()
        )

    monkeypatch.setattr(batch_eval.urllib.request, "urlopen", flaky_urlopen)
    monkeypatch.setattr(batch_eval.time, "sleep", delays.append)

    result = batch_eval.provider_inference_probe(
        "https://api.openai.com/v1/responses",
        "top-secret-key",
        {"model": "gpt-5.6-luna", "input": "Return OK."},
    )
    assert result["request_id"] == "resp_retry"
    assert calls == 3
    assert delays == [5.0, 10.0]


def test_provider_inference_probe_audits_eventual_openrouter_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    delays: list[float] = []

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            assert timeout == 120
            response = io.BytesIO(
                json.dumps(
                    {
                        "id": "resp_test",
                        "model": "deepseek/deepseek-v4-flash-0731",
                        "status": "completed",
                        "usage": {"cost": 0.001},
                    }
                ).encode()
            )
            response.headers = {"X-Generation-Id": "gen_test"}
            return response
        assert timeout == 30
        assert "id=gen_test" in request.full_url
        if calls == 2:
            raise urllib.error.HTTPError(
                request.full_url, 404, "Not Found", {}, io.BytesIO()
            )
        return io.BytesIO(
            json.dumps(
                {
                    "data": {
                        "provider_name": "DeepSeek",
                        "model": "deepseek/deepseek-v4-flash-20260731",
                        "preset_id": "preset-fixture",
                        "total_cost": 0.001,
                        "provider_responses": [{"status": 200}],
                    }
                }
            ).encode()
        )

    monkeypatch.setattr(batch_eval.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(batch_eval.time, "sleep", delays.append)

    result = batch_eval.provider_inference_probe(
        "https://openrouter.ai/api/v1/responses",
        "top-secret-key",
        {"model": "@preset/test", "input": "Return OK."},
        generation_audit_url="https://openrouter.ai/api/v1/generation",
    )

    assert result["provider"] == "DeepSeek"
    assert result["request_id"] == "resp_test"
    assert result["generation_id"] == "gen_test"
    assert result["resolved_model"] == "deepseek/deepseek-v4-flash-20260731"
    assert result["preset_id"] == "preset-fixture"
    assert result["provider_reported_total_cost_usd"] == 0.001
    assert delays == [1.0]


def test_provider_inference_probe_preserves_sanitized_spend_limit_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    body = io.BytesIO(
        json.dumps(
            {
                "error": {
                    "type": "usage_limit_reached",
                    "message": "Project reached its enforced spend limit.\nUpdate billing.",
                }
            }
        ).encode()
    )

    def fail_urlopen(_request, timeout):
        nonlocal calls
        calls += 1
        assert timeout == 120
        raise urllib.error.HTTPError(
            "https://api.openai.com/v1/responses",
            400,
            "Bad Request",
            {},
            body,
        )

    monkeypatch.setattr(batch_eval.urllib.request, "urlopen", fail_urlopen)
    with pytest.raises(RuntimeError) as caught:
        batch_eval.provider_inference_probe(
            "https://api.openai.com/v1/responses",
            "top-secret-key",
            {"model": "gpt-5.6-luna", "input": "Return OK."},
        )
    message = str(caught.value)
    assert "HTTP 400" in message
    assert "enforced spend limit" in message
    assert "\n" not in message
    assert "top-secret-key" not in message
    assert calls == 1


def test_vercel_project_link_accepts_cli_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    web = tmp_path / "web"
    (web / ".vercel").mkdir(parents=True)
    (web / ".vercel/project.json").write_text(
        json.dumps(
            {
                "projectId": frontier_update.PROJECT_ID,
                "orgId": frontier_update.ORG_ID,
                "projectName": "sprint",
                "settings": {"nodeVersion": "24.x"},
            }
        )
    )
    monkeypatch.setattr(batch_eval, "WEB", web)
    assert batch_eval.vercel_project_link_ready()


def test_public_batch_never_contains_secrets_or_host_paths() -> None:
    payload = {
        "batch_id": "eval",
        "updated_at": "now",
        "status": "running",
        "reasoning_effort": "max",
        "codex_version": "0.147.0",
        "run_hours": None,
        "alerts": [],
        "arms": [
            {
                "run_id": "eval-luna-1",
                "family": "luna",
                "model": "openai/gpt-5.6-luna",
                "resolved_model_version": "gpt-5.6-luna",
                "provider_endpoint": "openai",
                "trial": 1,
                "status": "running",
                "OPENAI_API_KEY": "secret",
                "env_file": "/private/.env",
            }
        ],
    }
    encoded = json.dumps(batch_eval.public_batch(payload))
    assert "secret" not in encoded
    assert "/private" not in encoded


def test_write_public_batch_updates_active_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(batch_eval, "WEB", tmp_path / "web")
    payload = {
        "batch_id": "eval",
        "updated_at": "2026-08-19T00:00:00Z",
        "status": "running",
        "reasoning_effort": "max",
        "codex_version": "0.147.0",
        "run_hours": None,
        "arms": [],
    }

    historical = batch_eval.write_public_batch(payload)

    assert historical == tmp_path / "web/data/batches/eval.json"
    current = tmp_path / "web/data/batches/current.json"
    assert json.loads(historical.read_text()) == json.loads(current.read_text())


def test_write_public_batch_ignores_observer_only_timestamp_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(batch_eval, "WEB", tmp_path / "web")
    payload = {
        "batch_id": "eval",
        "updated_at": "2026-08-19T00:00:00Z",
        "status": "running",
        "reasoning_effort": "max",
        "codex_version": "0.147.0",
        "run_hours": None,
        "arms": [{"run_id": "eval-1", "last_monitor_at": "first"}],
    }
    path = batch_eval.write_public_batch(payload)
    first = path.read_bytes()
    payload["updated_at"] = "2026-08-19T00:01:00Z"
    payload["arms"][0]["last_monitor_at"] = "second"

    batch_eval.write_public_batch(payload)

    assert path.read_bytes() == first


def test_write_public_batch_tracks_explicitly_coexisting_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(batch_eval, "WEB", tmp_path / "web")
    monkeypatch.setattr(batch_eval, "BATCH_ROOT", tmp_path / "batches")
    (tmp_path / "batches").mkdir()
    base = {
        "batch_id": "base",
        "updated_at": "2026-08-19T00:00:00Z",
        "status": "running",
        "reasoning_effort": "max",
        "codex_version": "0.147.0",
        "run_hours": None,
        "arms": [{"run_id": "base-sol-1", "family": "sol"}],
    }
    replacement = {
        **base,
        "batch_id": "replacement",
        "coexist_batch_ids": ["base"],
        "arms": [{"run_id": "replacement-luna-1", "family": "luna"}],
    }
    batch_eval.atomic_json(batch_eval.batch_path("base"), base)

    historical = batch_eval.write_public_batch(replacement)

    assert [arm["run_id"] for arm in json.loads(historical.read_text())["arms"]] == [
        "replacement-luna-1"
    ]
    current = json.loads((tmp_path / "web/data/batches/current.json").read_text())
    assert current["tracked_batch_ids"] == ["base", "replacement"]
    assert [arm["run_id"] for arm in current["arms"]] == [
        "base-sol-1",
        "replacement-luna-1",
    ]


def test_tracking_batch_prefers_newer_duplicate_model_effort_trial_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(batch_eval, "BATCH_ROOT", tmp_path / "batches")
    (tmp_path / "batches").mkdir()
    base = {
        "batch_id": "base",
        "updated_at": "2026-08-19T00:00:00Z",
        "status": "complete",
        "reasoning_effort": "max",
        "codex_version": "0.149.1",
        "run_hours": None,
        "arms": [
            {
                "run_id": "base-sol-1",
                "family": "sol",
                "reasoning_effort": "max",
                "trial": 1,
            }
        ],
    }
    tracking = {
        **base,
        "batch_id": "tracking",
        "status": "running",
        "coexist_batch_ids": ["base"],
        "arms": [
            {
                "run_id": "tracking-sol-1",
                "family": "sol",
                "reasoning_effort": "max",
                "trial": 1,
            }
        ],
    }
    batch_eval.atomic_json(batch_eval.batch_path("base"), base)

    current = batch_eval.public_tracking_batch(tracking)

    assert [arm["run_id"] for arm in current["arms"]] == ["tracking-sol-1"]


def test_write_public_batch_can_leave_active_pointer_to_tracking_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(batch_eval, "WEB", tmp_path / "web")
    current = tmp_path / "web/data/batches/current.json"
    current.parent.mkdir(parents=True)
    current.write_text('{"batch_id":"replacement"}\n')
    source = {
        "batch_id": "source",
        "updated_at": "2026-08-21T00:00:00Z",
        "status": "running",
        "reasoning_effort": "max",
        "codex_version": "0.147.0",
        "run_hours": None,
        "arms": [],
    }

    historical = batch_eval.write_public_batch(source, update_current=False)

    assert json.loads(historical.read_text())["batch_id"] == "source"
    assert json.loads(current.read_text())["batch_id"] == "replacement"


def test_tracking_batch_ignores_delegated_site_alerts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(batch_eval, "BATCH_ROOT", tmp_path / "batches")
    (tmp_path / "batches").mkdir()
    base = {
        "batch_id": "base",
        "updated_at": "2026-08-19T00:00:00Z",
        "status": "running",
        "reasoning_effort": "max",
        "codex_version": "0.147.0",
        "run_hours": None,
        "arms": [
            {
                "run_id": "base-luna-1",
                "family": "luna",
                "status": "finalized",
            }
        ],
        "alerts": [
            {"run_id": "batch", "kind": "performance_export"},
            {"run_id": "batch", "kind": "website_deploy"},
            {"run_id": "base-luna-1", "kind": "finalization"},
            {"run_id": "base-luna-1", "kind": "monitor_error"},
        ],
    }
    replacement = {
        "batch_id": "replacement",
        "updated_at": "2026-08-19T00:00:00Z",
        "status": "running",
        "reasoning_effort": "max",
        "codex_version": "0.147.0",
        "run_hours": None,
        "coexist_batch_ids": ["base"],
        "arms": [{"run_id": "replacement-luna-1", "family": "luna"}],
        "alerts": [{"run_id": "batch", "kind": "website_deploy"}],
    }
    batch_eval.atomic_json(batch_eval.batch_path("base"), base)

    current = batch_eval.public_tracking_batch(replacement)

    assert current["alerts"] == [
        {"run_id": "base-luna-1", "kind": "monitor_error"},
        {"run_id": "batch", "kind": "website_deploy"},
    ]


@pytest.mark.parametrize(
    ("invalidated_reason", "stop_ack", "expected_reason"),
    [
        (
            None,
            {"reason": "budget_telemetry_unavailable"},
            "budget_telemetry_unavailable",
        ),
        ("untrusted_agent_stop_marker", None, "untrusted_agent_stop_marker"),
    ],
)
def test_tracking_batch_keeps_invalid_lane_visible_until_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalidated_reason: str | None,
    stop_ack: dict | None,
    expected_reason: str,
) -> None:
    monkeypatch.setattr(batch_eval, "BATCH_ROOT", tmp_path / "batches")
    (tmp_path / "batches").mkdir()
    base = {
        "batch_id": "base",
        "updated_at": "now",
        "status": "running",
        "reasoning_effort": "max",
        "codex_version": "0.147.0",
        "run_hours": None,
        "arms": [
            {
                "run_id": "base-luna-1",
                "family": "luna",
                "invalidated_reason": invalidated_reason,
                "stop_ack": stop_ack,
            },
            {"run_id": "base-sol-1", "family": "sol"},
        ],
    }
    replacement = {
        **base,
        "batch_id": "replacement",
        "coexist_batch_ids": ["base"],
        "arms": [{"run_id": "replacement-luna-1", "family": "luna"}],
    }
    batch_eval.atomic_json(batch_eval.batch_path("base"), base)

    current = batch_eval.public_tracking_batch(replacement)

    assert [arm["run_id"] for arm in current["arms"]] == [
        "base-luna-1",
        "base-sol-1",
        "replacement-luna-1",
    ]
    assert current["excluded_arms"] == [
        {
            "run_id": "base-luna-1",
            "reason": expected_reason,
            "source_batch_id": "base",
        }
    ]


def test_batches_are_operator_stopped_without_a_fixed_deadline() -> None:
    assert batch_eval.RUN_HOURS is None


def test_run_control_units_cover_every_host_process_for_one_trial() -> None:
    assert batch_eval.run_control_units("eval-luna-1") == (
        "sprint-trial-eval-luna-1.service",
        "sprint-monitor-eval-luna-1.service",
        "sprint-pulse-eval-luna-1.service",
        "sprint-gpu-dispatch-eval-luna-1.service",
    )


def test_retire_run_control_services_accepts_units_already_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(batch_eval.sprintctl, "OPS_ROOT", tmp_path)
    run_dir = tmp_path / "eval-luna-1"
    run_dir.mkdir()
    (run_dir / "run.json").write_text('{"run_id":"eval-luna-1","agent_kind":"codex"}\n')
    completed = mock.Mock(returncode=5, stderr="Unit not loaded")
    absent = mock.Mock(returncode=3)
    monkeypatch.setattr(
        batch_eval.subprocess,
        "run",
        mock.Mock(side_effect=[completed, absent, absent, absent, absent]),
    )

    batch_eval.retire_run_control_services("eval-luna-1")


def test_batch_run_checked_returns_captured_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = mock.Mock(stdout="expected output\n")
    run = mock.Mock(return_value=completed)
    monkeypatch.setattr(batch_eval.subprocess, "run", run)

    assert batch_eval.run_checked(["example", "command"]) == "expected output\n"
    run.assert_called_once()


def test_retire_run_control_services_stops_zero_task_modal_apps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(batch_eval.sprintctl, "OPS_ROOT", tmp_path)
    run_dir = tmp_path / "eval-luna-1"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "eval-luna-1",
                "agent_kind": "codex",
                "app_name": "sprint-eval-luna-1",
                "training_app_name": "sprint-eval-luna-1-training",
                "verifier_app_name": "sprint-eval-luna-1-verifier",
            }
        )
        + "\n"
    )
    systemctl = mock.Mock(returncode=0, stderr="")
    absent = mock.Mock(returncode=3)
    monkeypatch.setattr(
        batch_eval.subprocess,
        "run",
        mock.Mock(side_effect=[systemctl, absent, absent, absent, absent]),
    )
    modal_calls: list[list[str]] = []

    def run_command(argv, **_kwargs):
        modal_calls.append(argv)
        if argv[-2:] == ["list", "--json"]:
            return mock.Mock(returncode=0, stdout="[]", stderr="")
        return mock.Mock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(batch_eval.sprintctl, "run_command", run_command)

    batch_eval.retire_run_control_services("eval-luna-1")

    assert [call[-1] for call in modal_calls[:-1]] == [
        "sprint-eval-luna-1",
        "sprint-eval-luna-1-training",
        "sprint-eval-luna-1-verifier",
    ]


def test_batch_stop_fails_closed_when_host_controllers_survive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_id = "host-survives"
    run_id = "host-survives-luna-1"
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", tmp_path / "ops")
    monkeypatch.setattr(batch_eval, "BATCH_ROOT", tmp_path / "batches")
    run_dir = batch_eval.SCRIPT_DIR / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text("{}\n")
    batch_eval.atomic_json(
        batch_eval.batch_path(batch_id),
        {
            "batch_id": batch_id,
            "arms": [{"run_id": run_id, "status": "running"}],
            "alerts": [],
            "credential_status": "revoked",
        },
    )

    with (
        mock.patch.object(
            batch_eval.sprintctl,
            "persist_stop_request",
            lambda *_args, **_kwargs: None,
        ),
        mock.patch.object(
            batch_eval.sprintctl,
            "request_stop",
            lambda *_args, **_kwargs: {"status": "acknowledged"},
        ),
        mock.patch.object(
            batch_eval,
            "retire_run_control_services",
            side_effect=RuntimeError("monitor still active"),
        ),
    ):
        result = batch_eval.stop_batch(batch_id)

    assert result["status"] == "stop_failed"
    assert result["arms"][0]["status"] == "stop_failed"
    assert "monitor still active" in result["arms"][0]["stop_dispatch_error"]


def test_batch_stop_persists_all_intents_before_slow_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_id = "stop-all"
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", tmp_path / "ops")
    monkeypatch.setattr(batch_eval, "BATCH_ROOT", tmp_path / "batches")
    arms = [{"run_id": f"run-{index}", "status": "running"} for index in range(1, 4)]
    for arm in arms:
        run_dir = batch_eval.SCRIPT_DIR / arm["run_id"]
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text("{}\n")
    batch_eval.atomic_json(
        batch_eval.batch_path(batch_id),
        {
            "batch_id": batch_id,
            "arms": arms,
            "alerts": [],
            "credential_status": "active",
        },
    )
    persisted: list[str] = []
    dispatched: list[str] = []
    events: list[str] = []

    def persist(run_id: str, *, reason: str):
        assert reason == "operator_batch_stop"
        persisted.append(run_id)
        events.append(f"persist:{run_id}")
        return tmp_path, {}, {}

    def dispatch(run_id: str, *, reason: str, wait_for_termination: bool):
        assert persisted == ["run-1", "run-2", "run-3"]
        assert "intent:published" in events
        assert "credentials:revoked" in events
        assert wait_for_termination is True
        dispatched.append(run_id)
        events.append(f"dispatch:{run_id}")
        if run_id == "run-1":
            raise RuntimeError("provider unavailable")
        return {"status": "acknowledged"}

    def revoke(payload: dict[str, object], env_file: Path):
        assert env_file == tmp_path / ".env"
        assert "intent:published" in events
        payload["credential_status"] = "revoked"
        payload["credentials_revoked_at"] = "now"
        payload["credential_cleanup_errors"] = []
        events.append("credentials:revoked")
        return []

    @contextlib.contextmanager
    def recording_lock(path: Path, *, blocking: bool = True):
        assert path == batch_eval.batch_path(batch_id).with_suffix(".lock")
        assert blocking is False
        assert persisted == ["run-1", "run-2", "run-3"]
        if "intent:published" not in events:
            assert dispatched == []
            events.append("intent:published")
        else:
            assert set(dispatched) == {"run-1", "run-2", "run-3"}
            events.append("final:published")
        yield True
        events.append("lock:exited")

    with (
        mock.patch.object(batch_eval.sprintctl, "persist_stop_request", persist),
        mock.patch.object(batch_eval.sprintctl, "request_stop", dispatch),
        mock.patch.object(
            batch_eval, "retire_run_control_services", lambda _run_id: None
        ),
        mock.patch.object(batch_eval, "revoke_batch_credentials", revoke),
        mock.patch.object(batch_eval.frontier_update, "file_lock", recording_lock),
    ):
        result = batch_eval.stop_batch(batch_id, env_file=tmp_path / ".env")

    assert set(dispatched) == {"run-1", "run-2", "run-3"}
    assert [arm["status"] for arm in result["arms"]] == [
        "stop_failed",
        "stopped",
        "stopped",
    ]
    assert result["arms"][0]["stop_dispatch_error"].startswith("RuntimeError:")
    assert result["arms"][1]["stop_dispatch_status"] == "acknowledged"
    assert result["status"] == "stop_failed"
    assert result["teardown_error"]["runs"]["run-1"]
    assert result["credential_status"] == "revoked"
    stored = json.loads(batch_eval.batch_path(batch_id).read_text())
    assert [arm["status"] for arm in stored["arms"]] == [
        "stop_failed",
        "stopped",
        "stopped",
    ]
    assert events[-1] == "lock:exited"


def test_batch_stop_dispatches_lanes_concurrently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_id = "stop-concurrent"
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", tmp_path / "ops")
    monkeypatch.setattr(batch_eval, "BATCH_ROOT", tmp_path / "batches")
    arms = [{"run_id": f"run-{index}", "status": "running"} for index in range(3)]
    for arm in arms:
        run_dir = batch_eval.SCRIPT_DIR / arm["run_id"]
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text("{}\n")
    batch_eval.atomic_json(
        batch_eval.batch_path(batch_id),
        {
            "batch_id": batch_id,
            "arms": arms,
            "alerts": [],
            "credential_status": "revoked",
        },
    )
    rendezvous = threading.Barrier(len(arms))

    def dispatch(_run_id: str, *, reason: str, wait_for_termination: bool):
        assert reason == "operator_batch_stop"
        assert wait_for_termination is True
        rendezvous.wait(timeout=2)
        return {"status": "acknowledged"}

    with (
        mock.patch.object(
            batch_eval.sprintctl, "persist_stop_request", lambda *_args, **_kwargs: None
        ),
        mock.patch.object(batch_eval.sprintctl, "request_stop", dispatch),
        mock.patch.object(
            batch_eval, "retire_run_control_services", lambda _run_id: None
        ),
    ):
        result = batch_eval.stop_batch(batch_id)

    assert all(arm["stop_dispatch_status"] == "acknowledged" for arm in result["arms"])
    assert all(arm["status"] == "stopped" for arm in result["arms"])
    assert result["status"] == "stopped"


def test_batch_stop_journals_transition_when_monitor_holds_state_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_id = "stop-journal"
    run_id = "stop-journal-luna-1"
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", tmp_path / "ops")
    monkeypatch.setattr(batch_eval, "BATCH_ROOT", tmp_path / "batches")
    run_dir = batch_eval.SCRIPT_DIR / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text("{}\n")
    batch_eval.atomic_json(
        batch_eval.batch_path(batch_id),
        {
            "batch_id": batch_id,
            "status": "running",
            "arms": [{"run_id": run_id, "status": "running"}],
            "alerts": [],
            "credential_status": "revoked",
        },
    )

    @contextlib.contextmanager
    def busy_lock(_path: Path, *, blocking: bool = True):
        assert blocking is False
        yield False

    with (
        mock.patch.object(
            batch_eval.sprintctl, "persist_stop_request", lambda *_args, **_kwargs: None
        ),
        mock.patch.object(
            batch_eval.sprintctl,
            "request_stop",
            lambda *_args, **_kwargs: {"status": "acknowledged"},
        ),
        mock.patch.object(
            batch_eval, "retire_run_control_services", lambda _run_id: None
        ),
        mock.patch.object(batch_eval.frontier_update, "file_lock", busy_lock),
    ):
        projected = batch_eval.stop_batch(batch_id)

    assert projected["arms"][0]["status"] == "stopped"
    assert batch_eval.batch_stop_marker_path(batch_id).is_file()
    # The monitor-owned batch document is untouched until it can consume the
    # journal, so the stop path cannot clobber a concurrent monitor commit.
    assert batch_eval.read_batch(batch_id)["arms"][0]["status"] == "running"

    stored = batch_eval.read_batch(batch_id)
    consumed = batch_eval.consume_batch_stop_transition(batch_id, stored)
    assert consumed["arms"][0]["status"] == "stopped"
    assert consumed["arms"][0]["stop_dispatch_status"] == "acknowledged"
    assert not batch_eval.batch_stop_marker_path(batch_id).exists()
    assert batch_eval.read_batch(batch_id)["arms"][0]["status"] == "stopped"


def test_website_projection_never_mutates_batch_control_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_id = "deploy-unlocked"
    monkeypatch.setattr(batch_eval, "BATCH_ROOT", tmp_path / "batches")
    monkeypatch.setattr(batch_eval, "WEB", tmp_path / "web")
    monkeypatch.setattr(
        batch_eval.frontier_update, "PIPELINE_LOCK", tmp_path / "pipeline.lock"
    )
    state_path = batch_eval.batch_path(batch_id)
    payload = {
        "batch_id": batch_id,
        "updated_at": "2026-08-24T00:20:00Z",
        "arms": [],
        "alerts": [],
        "credential_status": "active",
        "status": "running",
        "reasoning_effort": "max",
        "codex_version": "0.149.1",
        "run_hours": None,
    }
    batch_eval.atomic_json(state_path, payload)
    stale = copy.deepcopy(payload)
    stale["updated_at"] = "2026-08-24T00:00:00Z"
    batch_eval.write_public_batch(stale)
    # Observer-only monitor cycles intentionally do not rewrite these files.
    batch_eval.write_public_batch(payload)
    current_path = tmp_path / "web/data/batches/current.json"
    assert json.loads(current_path.read_text())["updated_at"] == stale["updated_at"]
    events: list[str] = []

    @contextlib.contextmanager
    def deployment_lock(path: Path, *, blocking: bool = True):
        assert path == tmp_path / "pipeline.lock"
        assert blocking is False
        events.append("deploy-lock:entered")
        yield True
        events.append("deploy-lock:exited")

    def refresh(current: dict[str, object]) -> None:
        assert current["batch_id"] == batch_id
        events.append("performance")

    def deploy(state: dict[str, object], **_kwargs) -> tuple[bool, str]:
        assert (
            json.loads(current_path.read_text())["updated_at"] == payload["updated_at"]
        )
        events.append("vercel")
        concurrent = json.loads(state_path.read_text())
        concurrent["credential_status"] = "revoked"
        concurrent["status"] = "stopping"
        batch_eval.atomic_json(state_path, concurrent)
        state["site_status"] = "deployed"
        state["last_deployed_site_hash"] = "new"
        return True, "https://deployment.example"

    with (
        mock.patch.object(batch_eval.frontier_update, "file_lock", deployment_lock),
        mock.patch.object(batch_eval, "refresh_performance_snapshot", refresh),
        mock.patch.object(batch_eval.frontier_update, "deploy_if_needed", deploy),
    ):
        publication = {"batch_id": batch_id, "last_deployed_site_hash": "old"}
        error, error_kind, attempted = batch_eval.deploy_website_projection(
            payload,
            publication,
            now=dt.datetime(2026, 8, 24, tzinfo=dt.timezone.utc),
            debounce_seconds=1200,
        )

    assert attempted is True
    assert error is None
    assert error_kind is None
    assert publication["site_status"] == "deployed"
    # A concurrent experiment transition survives untouched because the
    # publisher has no write path to batch.json.
    updated = batch_eval.read_batch(batch_id)
    assert updated["credential_status"] == "revoked"
    assert updated["status"] == "stopping"
    assert events == [
        "deploy-lock:entered",
        "performance",
        "vercel",
        "deploy-lock:exited",
    ]


def test_hung_website_deploy_does_not_block_operator_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_id = "deploy-stop"
    run_id = "deploy-stop-luna-1"
    monkeypatch.setattr(batch_eval, "BATCH_ROOT", tmp_path / "batches")
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", tmp_path / "ops")
    monkeypatch.setattr(batch_eval, "WEB", tmp_path / "web")
    monkeypatch.setattr(
        batch_eval.frontier_update, "PIPELINE_LOCK", tmp_path / "pipeline.lock"
    )
    run_dir = batch_eval.SCRIPT_DIR / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text("{}\n")
    state_path = batch_eval.batch_path(batch_id)
    batch_eval.atomic_json(
        state_path,
        {
            "batch_id": batch_id,
            "status": "running",
            "reasoning_effort": "max",
            "codex_version": "0.149.1",
            "run_hours": None,
            "arms": [{"run_id": run_id, "status": "running"}],
            "alerts": [],
            "credential_status": "revoked",
        },
    )
    deploy_started = threading.Event()
    allow_deploy_to_finish = threading.Event()
    monitor_errors: list[BaseException] = []

    def blocking_deploy(*_args, **_kwargs) -> tuple[bool, str]:
        deploy_started.set()
        assert allow_deploy_to_finish.wait(timeout=5)
        return True, "https://deployment.example"

    def monitor() -> None:
        try:
            payload = batch_eval.read_batch(batch_id)
            batch_eval.deploy_website_projection(
                payload,
                {"batch_id": batch_id},
                now=dt.datetime(2026, 8, 24, tzinfo=dt.timezone.utc),
                debounce_seconds=0,
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            monitor_errors.append(exc)

    with (
        mock.patch.object(batch_eval, "refresh_performance_snapshot", lambda _: None),
        mock.patch.object(
            batch_eval.frontier_update, "deploy_if_needed", blocking_deploy
        ),
        mock.patch.object(
            batch_eval.sprintctl, "persist_stop_request", lambda *_args, **_kwargs: None
        ),
        mock.patch.object(
            batch_eval.sprintctl,
            "request_stop",
            lambda *_args, **_kwargs: {"status": "acknowledged"},
        ),
        mock.patch.object(
            batch_eval, "retire_run_control_services", lambda _run_id: None
        ),
    ):
        worker = threading.Thread(target=monitor, daemon=True)
        worker.start()
        assert deploy_started.wait(timeout=2)
        began = time.monotonic()
        stopped = batch_eval.stop_batch(batch_id)
        stop_elapsed = time.monotonic() - began
        allow_deploy_to_finish.set()
        worker.join(timeout=5)

    assert stop_elapsed < 1
    assert stopped["arms"][0]["status"] == "stopped"
    assert not worker.is_alive()
    assert monitor_errors == []
    merged = batch_eval.read_batch(batch_id)
    assert merged["arms"][0]["status"] == "stopped"


def test_hung_publisher_does_not_delay_health_monitor_cycles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_id = "publish-health"
    run_id = "publish-health-luna-1"
    monkeypatch.setattr(batch_eval, "BATCH_ROOT", tmp_path / "batches")
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", tmp_path / "ops")
    monkeypatch.setattr(batch_eval, "WEB", tmp_path / "web")
    monkeypatch.setattr(
        batch_eval.frontier_update, "PIPELINE_LOCK", tmp_path / "pipeline.lock"
    )
    run_dir = batch_eval.SCRIPT_DIR / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text("{}\n")
    batch_eval.atomic_json(
        batch_eval.batch_path(batch_id),
        {
            "batch_id": batch_id,
            "status": "running",
            "reasoning_effort": "max",
            "codex_version": "0.149.1",
            "run_hours": None,
            "arms": [{"run_id": run_id, "family": "luna", "status": "running"}],
            "alerts": [],
            "credential_status": "revoked",
        },
    )
    deploy_started = threading.Event()
    allow_deploy_to_finish = threading.Event()
    publisher_errors: list[BaseException] = []

    def blocking_deploy(state: dict[str, object], **_kwargs: object):
        deploy_started.set()
        assert allow_deploy_to_finish.wait(timeout=5)
        state["site_status"] = "deployed"
        return True, "https://deployment.example"

    def publisher() -> None:
        try:
            batch_eval.publish_cycle(batch_id)
        except BaseException as exc:  # pragma: no cover - surfaced below
            publisher_errors.append(exc)

    with (
        mock.patch.object(batch_eval, "refresh_performance_snapshot", lambda _: None),
        mock.patch.object(
            batch_eval.frontier_update, "deploy_if_needed", blocking_deploy
        ),
        mock.patch.object(
            batch_eval, "mark_deployed_runs", lambda _payload, _publication: []
        ),
        mock.patch.object(
            batch_eval,
            "deployed_batch_current",
            lambda _payload, _publication: False,
        ),
        mock.patch.object(
            batch_eval,
            "live_run_monitor_status",
            lambda _run_id: {
                "harbor_alive": True,
                "ledger": {},
                "snapshot_heartbeat_ok": True,
            },
        ),
        mock.patch.object(batch_eval, "log_alerts", lambda _run_id: []),
        mock.patch.object(
            batch_eval, "verifier_lane_stall_alerts", lambda *_args, **_kwargs: []
        ),
        mock.patch.object(
            batch_eval, "continuous_ledger_error_alerts", lambda _payload: []
        ),
    ):
        worker = threading.Thread(target=publisher, daemon=True)
        worker.start()
        assert deploy_started.wait(timeout=2)
        began = time.monotonic()
        health = batch_eval.monitor_cycle(batch_id)
        health_elapsed = time.monotonic() - began
        allow_deploy_to_finish.set()
        worker.join(timeout=5)

    assert health_elapsed < 1
    assert health["arms"][0]["last_monitor_at"]
    assert not worker.is_alive()
    assert publisher_errors == []


def test_verifier_lane_stall_alert_is_scoped_to_one_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", tmp_path)
    ledger = (
        tmp_path
        / "eval-luna-1/harbor-jobs/job/task__trial/artifacts/continuous/ledger.jsonl"
    )
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps(
            {
                "submitted_at": "2026-08-08T11:59:59Z",
                "accepted_at": "2026-08-08T11:59:59Z",
                "accepted": True,
                "finished_at": None,
                "error": None,
            }
        )
        + "\n"
    )
    payload = {
        "arms": [
            {
                "run_id": "eval-luna-1",
                "ledger": {"queued": 2, "running": 0},
            }
        ]
    }
    now = batch_eval.parse_time("2026-08-08T12:20:00Z")
    alerts = batch_eval.verifier_lane_stall_alerts(payload, now=now)
    assert len(alerts) == 1
    assert alerts[0]["run_id"] == "eval-luna-1"
    assert alerts[0]["kind"] == "verifier_lane_stalled"
    assert alerts[0]["pending_submissions"] == "2"

    ledger.write_text(
        json.dumps(
            {
                "submitted_at": "2026-08-08T11:59:59Z",
                "accepted_at": "2026-08-08T11:59:59Z",
                "verification_started_at": "2026-08-08T12:00:01Z",
                "accepted": True,
                "finished_at": None,
                "error": None,
            }
        )
        + "\n"
    )
    assert batch_eval.verifier_lane_stall_alerts(payload, now=now) == []
    payload["arms"][0]["ledger"] = {"queued": 0, "running": 0}
    assert batch_eval.verifier_lane_stall_alerts(payload, now=now) == []


def test_continuous_ledger_errors_are_explicit_batch_alerts() -> None:
    payload = {
        "arms": [
            {"run_id": "clean", "ledger": {"error": 0}},
            {"run_id": "lost", "ledger": {"error": 2}},
        ]
    }
    assert batch_eval.continuous_ledger_error_alerts(payload) == [
        {
            "run_id": "lost",
            "kind": "continuous_ledger_error",
            "source": "continuous/ledger.jsonl",
            "count_in_tail": "2",
        }
    ]


def test_rejected_submission_cannot_mask_an_old_stuck_verifier_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ops = tmp_path / "ops"
    ledger = (
        ops
        / "eval-deepseek-1/harbor-jobs/job/task__trial/artifacts/continuous/ledger.jsonl"
    )
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps(
            {
                "submitted_at": "2026-08-08T11:50:00Z",
                "accepted_at": "2026-08-08T11:50:00Z",
                "accepted": True,
                "finished_at": None,
                "error": None,
            }
        )
        + "\n"
        + json.dumps(
            {
                "submitted_at": "2026-08-08T12:19:00Z",
                "accepted": False,
                "finished_at": "2026-08-08T12:19:00Z",
                "error": "submission cooldown active",
            }
        )
        + "\n"
    )
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", ops)
    payload = {
        "arms": [
            {
                "run_id": "eval-deepseek-1",
                "ledger": {"queued": 1, "running": 0},
            }
        ]
    }
    now = batch_eval.parse_time("2026-08-08T12:20:00Z")
    alerts = batch_eval.verifier_lane_stall_alerts(payload, now=now)
    assert len(alerts) == 1
    assert alerts[0]["run_id"] == "eval-deepseek-1"

    ledger.write_text(
        json.dumps(
            {
                "submitted_at": "2026-08-08T11:59:59Z",
                "finished_at": None,
                "error": None,
            }
        )
        + "\n"
        + ledger.read_text()
    )
    alerts = batch_eval.verifier_lane_stall_alerts(payload, now=now)
    assert len(alerts) == 1
    assert alerts[0]["last_progress_at"] == "2026-08-08T11:59:59+00:00"


def test_resumed_cpu_attempt_progress_prevents_false_verifier_stall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ops = tmp_path / "ops"
    old_ledger = (
        ops
        / "eval-deepseek-1/harbor-jobs/job/task__old/artifacts/continuous/ledger.jsonl"
    )
    resumed_ledger = (
        ops
        / "eval-deepseek-1/cpu-attempts/04/harbor-jobs/job/task__current"
        / "artifacts/continuous/ledger.jsonl"
    )
    old_ledger.parent.mkdir(parents=True)
    resumed_ledger.parent.mkdir(parents=True)
    old_ledger.write_text(
        json.dumps(
            {
                "submitted_at": "2026-08-08T11:00:00Z",
                "accepted": True,
                "finished_at": None,
            }
        )
        + "\n"
    )
    resumed_ledger.write_text(
        json.dumps(
            {
                "submitted_at": "2026-08-08T12:19:30Z",
                "accepted": True,
                "finished_at": None,
            }
        )
        + "\n"
    )
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", ops)
    payload = {
        "arms": [
            {
                "run_id": "eval-deepseek-1",
                "ledger": {"queued": 0, "running": 1},
            }
        ]
    }

    now = batch_eval.parse_time("2026-08-08T12:20:00Z")
    assert batch_eval.verifier_lane_stall_alerts(payload, now=now) == []


def test_recovered_verifier_stall_moves_to_resolved_history() -> None:
    payload = {
        "arms": [
            {"run_id": "recovered", "ledger": {"queued": 0, "running": 0}},
            {"run_id": "still-stuck", "ledger": {"queued": 1, "running": 0}},
        ],
        "alerts": [
            {
                "run_id": "recovered",
                "kind": "verifier_lane_stalled",
                "source": "continuous/ledger.jsonl",
            },
            {
                "run_id": "still-stuck",
                "kind": "verifier_lane_stalled",
                "source": "continuous/ledger.jsonl",
            },
        ],
    }
    active = [
        {
            "run_id": "still-stuck",
            "kind": "verifier_lane_stalled",
            "source": "continuous/ledger.jsonl",
        }
    ]

    batch_eval.resolve_recovered_verifier_stalls(payload, active)

    assert [alert["run_id"] for alert in payload["alerts"]] == ["still-stuck"]
    assert payload["resolved_alerts"][0]["run_id"] == "recovered"
    assert "progressed" in payload["resolved_alerts"][0]["resolution"]


def test_dq_replay_is_queued_and_public_index_is_path_safe(tmp_path: Path) -> None:
    job = tmp_path / "job"
    trial = job / "task__trial"
    web = tmp_path / "web"
    state_path = tmp_path / "eval-deepseek-1" / "frontier-state.json"
    attempt = trial / "artifacts/continuous/attempts/0001-fall.pt"
    policy = attempt / "artifacts/app/submission/policy.pt"
    replay = attempt / "verifier/replay.json"
    details = attempt / "verifier/sprint_results.json"
    policy.parent.mkdir(parents=True)
    replay.parent.mkdir(parents=True)
    web.mkdir()
    state_path.parent.mkdir()
    policy.write_bytes(b"failed-policy")
    replay.write_text('{"schema_version":1,"body_names":[],"frames":[]}\n')
    details.write_text(
        json.dumps(
            {
                "valid_run": False,
                "max_distance_m": 14.5,
                "max_distance_semantics": (
                    "legal_prefix_until_first_terminal_condition"
                ),
                "failed_gates": ["in_lane"],
            }
        )
    )
    ledger = trial / "artifacts/continuous/ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        json.dumps(
            {
                "index": 1,
                "name": "fall.pt",
                "submitted_at": "2026-08-08T00:00:00Z",
                "finished_at": "2026-08-08T00:01:00Z",
                "rewards": {"valid_run": 0.0, "gate_in_lane": 0.0},
            }
        )
        + "\n"
    )
    state = frontier_update.scan_frontier(
        job=job, trial=trial, state_path=state_path, web=web
    )
    digest = frontier_update.sha256_file(policy)
    assert state["policies"][digest]["failed_gates"] == ["in_lane"]
    assert state["policies"][digest]["replay_path"] == str(replay)
    queued = next(
        item for item in state["capture_queue"] if item["policy_hash"] == digest
    )
    assert queued["status"] == "queued"
    assert queued["story"] == "failure"

    state["captures"][digest] = {
        "valid": True,
        "web_html": f"replay/frontier-{digest[:12]}.html",
    }
    registry = state_path.parent / "gpu-job-registry"
    registry.mkdir()
    (registry / "job-1.json").write_text(
        json.dumps(
            {
                "created_at": "2026-08-07T23:58:00Z",
                "progress": {"submission_results": [{"policy_sha256": digest}]},
            }
        )
    )
    frontier_update.write_web_policy_indexes(state_path, state, web)
    public = json.loads((web / "data/policies/eval-deepseek-1.json").read_text())
    encoded = json.dumps(public)
    assert str(tmp_path) not in encoded
    assert public["policies"][0]["replay_ready"] is True
    assert public["policies"][0]["replay_url"].startswith("/replay/frontier-")
    assert public["policies"][0]["enqueued_at"] == "2026-08-07T23:58:00Z"
    assert public["policies"][0]["enqueued_at_basis"] == "gpu_job_enqueued"
    assert public["policies"][0]["max_distance_semantics"] == (
        "legal_prefix_until_first_terminal_condition"
    )


def test_public_policy_index_uses_bridge_observation_before_verifier_admission(
    tmp_path: Path,
) -> None:
    web = tmp_path / "web"
    state_path = tmp_path / "eval-sol-1" / "frontier-state.json"
    state_path.parent.mkdir()
    (state_path.parent / "run.json").write_text(
        json.dumps({"run_id": "eval-sol-1", "created_at": "2026-08-07T12:00:00Z"})
    )
    bridge = state_path.parent / "submission-bridge"
    bridge.mkdir()
    digest = "a" * 64
    (bridge / "120500-policy.json").write_text(
        json.dumps(
            {
                "policy_sha256": digest,
                "observed_at": "2026-08-07T12:05:00Z",
                "forwarded_at": "2026-08-07T12:05:06Z",
            }
        )
    )
    state = {
        "captures": {},
        "policies": {
            digest: {
                "index": 1,
                "submitted_at": "2026-08-07T12:40:00Z",
                "finished_at": "2026-08-07T12:41:00Z",
                "effective_speed_mps": 2.5,
            }
        },
    }

    frontier_update.write_web_policy_indexes(state_path, state, web)

    public = json.loads((web / "data/policies/eval-sol-1.json").read_text())
    policy = public["policies"][0]
    assert policy["enqueued_at"] == "2026-08-07T12:05:00Z"
    assert policy["enqueued_at_basis"] == "gpu_output_observed"
    assert policy["submitted_at"] == "2026-08-07T12:40:00Z"


def test_public_policy_index_keeps_only_configured_newest_runs(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(frontier_update, "PUBLIC_RUN_LIMIT", 6)
    web = tmp_path / "web"
    index_path = web / "data" / "policies" / "index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runs": [
                    {
                        "run_id": f"old-{index}",
                        "created_at": f"2026-08-0{index}T00:00:00Z",
                    }
                    for index in range(1, 8)
                ],
            }
        )
    )
    state_path = tmp_path / "current" / "frontier-state.json"
    state_path.parent.mkdir()
    (state_path.parent / "run.json").write_text(
        json.dumps(
            {
                "run_id": "current",
                "created_at": "2026-08-09T00:00:00Z",
                "model": "openai/gpt-5.6-luna",
            }
        )
    )
    frontier_update.write_web_policy_indexes(state_path, {"policies": {}}, web)
    index = json.loads(index_path.read_text())
    assert len(index["runs"]) == frontier_update.PUBLIC_RUN_LIMIT
    assert [row["run_id"] for row in index["runs"]] == [
        "current",
        "old-7",
        "old-6",
        "old-5",
        "old-4",
        "old-3",
    ]


def test_public_policy_index_preserves_observer_clock_without_changes(
    tmp_path: Path,
) -> None:
    web = tmp_path / "web"
    state_path = tmp_path / "stable" / "frontier-state.json"
    state_path.parent.mkdir()
    (state_path.parent / "run.json").write_text(
        json.dumps({"run_id": "stable", "created_at": "2026-08-19T00:00:00Z"})
    )
    state = {"policies": {}, "captures": {}}
    frontier_update.write_web_policy_indexes(state_path, state, web)
    run_path = web / "data" / "policies" / "stable.json"
    previous = json.loads(run_path.read_text())
    previous["updated_at"] = "2000-01-01T00:00:00Z"
    frontier_update.atomic_write_json(run_path, previous)

    frontier_update.write_web_policy_indexes(state_path, state, web)

    public = json.loads(run_path.read_text())
    index = json.loads((web / "data" / "policies" / "index.json").read_text())
    assert public["updated_at"] == "2000-01-01T00:00:00Z"
    assert index["runs"][0]["updated_at"] == "2000-01-01T00:00:00Z"


def test_homepage_uses_compact_timeline_index_summaries() -> None:
    source = (ROOT / "web" / "app.js").read_text()
    assert "dashboard_artifacts" in source
    assert "usage_summary" in source
    assert "/data/batches/current.json" in source
    assert "renderExperimentTracker" in source
    assert "updateExperimentClocks" in source
    assert 'class="experiment-table"' in source
    assert 'scope="rowgroup"' in source
    assert "${row.submitted}</td>" in source
    assert "Effective Speed" in source
    assert "Total cost" in source
    assert "await json(tMeta.path)" not in source
    assert "showReadout" in source
    assert "readout-timeline" in source
    assert ".append(document.createElementNS" not in source
    assert "final_api_cost_usd" in source
    assert "modal_provider_billing" in source
    assert "modalRoleCost(run,'verifier_gpu')" not in source
    assert "Full-screen replay" not in source
    assert "replay_url" in source


def test_timeline_json_export_replaces_nonfinite_numbers(tmp_path: Path) -> None:
    path = tmp_path / "timeline.json"
    unified_timeline.atomic_json(path, {"value": float("inf")})
    assert json.loads(path.read_text()) == {"value": None}


def test_replay_renderer_exposes_complete_cli() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "web/renderers/g1-100-metres/render.py"),
            "--help",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout
    assert "--capture" in completed.stdout
    assert "--meta-policy" in completed.stdout


def test_capture_retries_are_bounded_and_renderer_versioned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = tmp_path / "job"
    trial = job / "task__trial"
    web = tmp_path / "web"
    state_path = tmp_path / "eval-luna-1" / "frontier-state.json"
    attempt = trial / "artifacts/continuous/attempts/0001-fall.pt"
    policy = attempt / "artifacts/app/submission/policy.pt"
    replay = attempt / "verifier/replay.json"
    policy.parent.mkdir(parents=True)
    replay.parent.mkdir(parents=True)
    state_path.parent.mkdir()
    web.mkdir()
    policy.write_bytes(b"failed-policy")
    replay.write_text('{"schema_version":1,"body_names":[],"frames":[]}\n')
    (attempt / "verifier/sprint_results.json").write_text(
        json.dumps({"valid_run": False, "failed_gates": ["finished"]})
    )
    ledger = trial / "artifacts/continuous/ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        json.dumps(
            {
                "index": 1,
                "name": "fall.pt",
                "submitted_at": "2026-08-08T00:00:00Z",
                "finished_at": "2026-08-08T00:01:00Z",
                "rewards": {"valid_run": 0.0, "gate_finished": 0.0},
            }
        )
        + "\n"
    )
    monkeypatch.setattr(frontier_update, "PIPELINE_LOCK", tmp_path / "pipeline.lock")
    monkeypatch.setattr(frontier_update, "renderer_source_hash", lambda: "renderer-a")
    calls = 0

    def broken_capture(**_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("renderer diagnostic")

    monkeypatch.setattr(frontier_update, "capture_and_render", broken_capture)
    state = frontier_update.run_worker(
        job=job,
        trial=trial,
        state_path=state_path,
        web=web,
        deploy=False,
        debounce_seconds=0,
    )
    current = [
        item
        for item in state["capture_queue"]
        if item.get("renderer_hash") == "renderer-a"
    ]
    assert calls == frontier_update.CAPTURE_MAX_ATTEMPTS
    assert len(current) == 1
    assert current[0]["status"] == "error"
    assert current[0]["attempts"] == frontier_update.CAPTURE_MAX_ATTEMPTS
    assert "renderer diagnostic" in current[0]["error"]

    frontier_update.run_worker(
        job=job,
        trial=trial,
        state_path=state_path,
        web=web,
        deploy=False,
        debounce_seconds=0,
    )
    assert calls == frontier_update.CAPTURE_MAX_ATTEMPTS

    monkeypatch.setattr(frontier_update, "renderer_source_hash", lambda: "renderer-b")
    updated = frontier_update.scan_frontier(
        job=job,
        trial=trial,
        state_path=state_path,
        web=web,
    )
    replacement = [
        item
        for item in updated["capture_queue"]
        if item.get("renderer_hash") == "renderer-b"
    ]
    assert len(replacement) == 1
    assert replacement[0]["status"] == "queued"
    assert replacement[0]["attempts"] == 0


def test_replay_worker_waits_for_shared_renderer_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[bool] = []

    @contextlib.contextmanager
    def recording_lock(_path: Path, *, blocking: bool = True):
        observed.append(blocking)
        yield True

    state = {"capture_queue": [], "policies": {}, "captures": {}}
    monkeypatch.setattr(frontier_update, "file_lock", recording_lock)
    monkeypatch.setattr(frontier_update, "scan_frontier", lambda **_kwargs: state)
    monkeypatch.setattr(
        frontier_update, "write_web_policy_indexes", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        frontier_update, "atomic_write_json", lambda *_args, **_kwargs: None
    )

    result = frontier_update.run_worker(
        job=tmp_path / "job",
        trial=tmp_path / "trial",
        state_path=tmp_path / "state.json",
        web=tmp_path / "web",
        deploy=False,
        debounce_seconds=0,
    )
    assert result is state
    assert observed == [True]


def test_run_checked_preserves_failure_output() -> None:
    with pytest.raises(RuntimeError, match="renderer diagnostic"):
        frontier_update.run_checked(
            [sys.executable, "-c", "print('renderer diagnostic'); raise SystemExit(9)"]
        )


def test_run_checked_bounds_stalled_commands() -> None:
    with pytest.raises(RuntimeError, match="command timed out after 0.01s"):
        frontier_update.run_checked(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            timeout_seconds=0.01,
        )


def test_batch_control_services_separate_health_monitor_from_publisher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    config_home = tmp_path / "config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-appear-in-unit")
    monkeypatch.setattr(batch_eval.shutil, "which", lambda name: "/tools/vercel")
    monkeypatch.setattr(
        batch_eval,
        "run_checked",
        lambda command, **_kwargs: commands.append(command) or "ok",
    )
    monkeypatch.setattr(batch_eval.subprocess, "run", lambda *args, **kwargs: None)
    batch_eval.start_batch_control_services("eval", tmp_path / ".env", "profile-a")
    unit_dir = config_home / "systemd/user"
    monitor = (unit_dir / "sprint-batch-eval-monitor.service").read_text()
    publisher = (unit_dir / "sprint-batch-eval-publisher.service").read_text()
    timer = (unit_dir / "sprint-batch-eval-publisher.timer").read_text()
    for unit in (monitor, publisher):
        assert 'Environment="UV=/home/ubuntu/.local/bin/uv"' in unit
        assert 'Environment="MODAL_PROFILE=profile-a"' in unit
        assert "/tools" in unit
        assert str(batch_eval.HARBOR_PYTHON.parent) in unit
        assert str(batch_eval.HARBOR_PYTHON) in unit
        assert f"WorkingDirectory={batch_eval.ROOT}" in unit
        assert "must-not-appear-in-unit" not in unit
    assert '"monitor"' in monitor
    assert '"--loop"' in monitor
    assert '"publish"' not in monitor
    assert "Restart=on-failure" in monitor
    assert "WantedBy=default.target" in monitor
    assert '"publish"' in publisher
    assert '"--loop"' not in publisher
    assert "Type=oneshot" in publisher
    assert "TimeoutStartSec=35min" in publisher
    assert "KillMode=control-group" in publisher
    assert "Unit=sprint-batch-eval-publisher.service" in timer
    assert f"OnUnitInactiveSec={batch_eval.LIVE_SITE_DEPLOY_SECONDS}s" in timer
    assert "Persistent=true" in timer
    assert commands == [
        ["systemctl", "--user", "daemon-reload"],
        [
            "systemctl",
            "--user",
            "enable",
            "sprint-batch-eval-monitor.service",
        ],
        [
            "systemctl",
            "--user",
            "restart",
            "sprint-batch-eval-monitor.service",
        ],
        ["systemctl", "--user", "daemon-reload"],
        [
            "systemctl",
            "--user",
            "enable",
            "sprint-batch-eval-publisher.timer",
        ],
        [
            "systemctl",
            "--user",
            "restart",
            "sprint-batch-eval-publisher.timer",
        ],
        [
            "systemctl",
            "--user",
            "start",
            "--no-block",
            "sprint-batch-eval-publisher.service",
        ],
    ]


def test_missing_vercel_never_rolls_back_health_supervision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(batch_eval, "BATCH_ROOT", tmp_path / "batches")
    monkeypatch.setattr(batch_eval.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        batch_eval,
        "run_checked",
        lambda command, **_kwargs: commands.append(command) or "ok",
    )
    monkeypatch.setattr(batch_eval.subprocess, "run", lambda *args, **kwargs: None)

    batch_eval.start_batch_control_services("eval", tmp_path / ".env", "profile-a")

    assert commands == [
        ["systemctl", "--user", "daemon-reload"],
        [
            "systemctl",
            "--user",
            "enable",
            "sprint-batch-eval-monitor.service",
        ],
        [
            "systemctl",
            "--user",
            "restart",
            "sprint-batch-eval-monitor.service",
        ],
    ]
    publication = batch_eval.read_publication("eval")
    assert publication["site_status"] == "setup_error"
    assert publication["setup_error"]["type"] == "RuntimeError"


def test_tracking_batch_quiesces_only_coexisting_publishers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    quiesced: list[str] = []
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(batch_eval.shutil, "which", lambda _name: "/tools/vercel")
    monkeypatch.setattr(batch_eval, "run_checked", lambda *_args, **_kwargs: "ok")
    monkeypatch.setattr(
        batch_eval,
        "quiesce_batch_publisher",
        lambda batch_id: quiesced.append(batch_id),
    )

    batch_eval.start_batch_control_services(
        "tracking",
        tmp_path / ".env",
        "profile-a",
        coexist_batch_ids=("source-a", "source-b"),
    )

    assert quiesced == ["source-a", "source-b"]


def test_no_publish_keeps_monitor_without_quiescing_coexisting_publishers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    quiesced: list[str] = []
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(batch_eval, "BATCH_ROOT", tmp_path / "batches")
    monkeypatch.setattr(batch_eval.shutil, "which", lambda _name: "/tools/vercel")
    monkeypatch.setattr(
        batch_eval,
        "run_checked",
        lambda command, **_kwargs: commands.append(command) or "ok",
    )
    monkeypatch.setattr(batch_eval.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        batch_eval,
        "quiesce_batch_publisher",
        lambda batch_id: quiesced.append(batch_id),
    )

    batch_eval.start_batch_control_services(
        "isolated",
        tmp_path / ".env",
        "profile-a",
        coexist_batch_ids=("protected",),
        publish_site=False,
    )

    assert commands == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "sprint-batch-isolated-monitor.service"],
        ["systemctl", "--user", "restart", "sprint-batch-isolated-monitor.service"],
    ]
    assert quiesced == ["isolated"]
    publication = batch_eval.read_publication("isolated")
    assert publication["site_status"] == "disabled"
    assert publication["reason"] == "launch_no_publish"


def test_batch_monitor_reads_live_lane_status_without_duplicate_poll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "eval-luna-1"
    state_dir = tmp_path / run_id
    state_dir.mkdir()
    (state_dir / "monitor.pid").write_text("1234\n")
    expected = {
        "schema_version": 2,
        "run_id": run_id,
        "updated_at": "2026-08-08T23:00:00Z",
        "harbor_alive": True,
    }
    (state_dir / "status.json").write_text(json.dumps(expected))
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(batch_eval.sprintctl, "process_alive", lambda *_args: True)
    monkeypatch.setattr(
        batch_eval.sprintctl,
        "monitor_once",
        lambda *_args, **_kwargs: pytest.fail("duplicate monitor poll"),
    )
    assert batch_eval.live_run_monitor_status(run_id) == expected


def test_batch_monitor_reads_stopped_lane_locally_without_modal_poll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "eval-deepseek-1"
    state_dir = tmp_path / run_id
    state_dir.mkdir()
    (state_dir / "STOP_ACK.json").write_text("{}\n")
    expected = {
        "schema_version": 2,
        "run_id": run_id,
        "harbor_alive": False,
        "stop_ack": {"reason": "operator_stop"},
    }
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(
        batch_eval.sprintctl, "load_run", lambda _run_id: (state_dir, {})
    )
    monkeypatch.setattr(
        batch_eval.sprintctl,
        "status_snapshot",
        lambda actual_dir, _run, *, include_remote: (
            expected
            if actual_dir == state_dir and include_remote is False
            else pytest.fail("stopped lane must use local status")
        ),
    )
    monkeypatch.setattr(
        batch_eval.sprintctl,
        "monitor_once",
        lambda *_args, **_kwargs: pytest.fail("stopped lane polled Modal"),
    )

    assert batch_eval.live_run_monitor_status(run_id) == expected


def test_batch_monitor_treats_agent_exit_ack_as_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "eval-deepseek-1"
    state_dir = tmp_path / run_id
    state_dir.mkdir()
    (state_dir / "STOP_ACK.json").write_text('{"reason":"agent_exit"}\n')
    (state_dir / "monitor.pid").write_text("1234\n")
    (state_dir / "status.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": run_id,
                "harbor_alive": True,
                "stop_ack": {"reason": "agent_exit"},
            }
        )
    )
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(batch_eval.sprintctl, "process_alive", lambda *_args: True)
    monkeypatch.setattr(
        batch_eval.sprintctl,
        "status_snapshot",
        lambda *_args, **_kwargs: pytest.fail("terminal lane must not be polled"),
    )

    assert batch_eval.live_run_monitor_status(run_id) is None


def test_batch_monitor_recovers_status_when_run_state_appears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_id = "eval"
    run_id = "eval-luna-1"
    ops = tmp_path / "ops"
    run_dir = ops / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text("{}\n")
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", ops)
    monkeypatch.setattr(batch_eval, "BATCH_ROOT", ops / "batches")
    monkeypatch.setattr(batch_eval, "WEB", tmp_path / "web")
    batch_eval.atomic_json(
        batch_eval.batch_path(batch_id),
        {
            "batch_id": batch_id,
            "status": "running",
            "reasoning_effort": "max",
            "codex_version": "0.147.0",
            "run_hours": None,
            "arms": [{"run_id": run_id, "status": "missing_run_state"}],
            "alerts": [],
        },
    )
    monkeypatch.setattr(
        batch_eval,
        "live_run_monitor_status",
        lambda _run_id: {
            "harbor_alive": True,
            "ledger": {},
            "snapshot_heartbeat_ok": True,
        },
    )
    monkeypatch.setattr(batch_eval, "log_alerts", lambda _run_id: [])
    monkeypatch.setattr(
        batch_eval, "verifier_lane_stall_alerts", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        batch_eval, "continuous_ledger_error_alerts", lambda _payload: []
    )
    monkeypatch.setattr(batch_eval, "mark_deployed_runs", lambda _payload: [])

    result = batch_eval.monitor_cycle(batch_id)

    assert result["arms"][0]["status"] == "running"


def test_stopped_batch_is_terminal_for_health_monitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_id = "stopped-batch"
    monkeypatch.setattr(batch_eval, "BATCH_ROOT", tmp_path / "batches")
    batch_eval.atomic_json(
        batch_eval.batch_path(batch_id),
        {
            "batch_id": batch_id,
            "status": "stopped",
            "arms": [{"run_id": "stopped-luna-1", "status": "stopped"}],
            "alerts": [],
        },
    )
    monkeypatch.setattr(
        batch_eval,
        "audit_openrouter_child_usage",
        lambda *_args, **_kwargs: pytest.fail("stopped batch audited provider usage"),
    )
    monkeypatch.setattr(
        batch_eval,
        "live_run_monitor_status",
        lambda *_args, **_kwargs: pytest.fail("stopped batch polled a run"),
    )

    result = batch_eval.monitor_cycle(batch_id)

    assert result["status"] == "stopped"
    assert result["arms"][0]["status"] == "stopped"


def test_health_monitor_completes_without_any_site_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_id = "finalizing"
    run_id = "finalizing-luna-1"
    ops = tmp_path / "ops"
    run_dir = ops / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text("{}\n")
    (run_dir / "FINALIZED.json").write_text(
        json.dumps(
            {
                "complete": True,
                "integrity": {
                    "schema_version": batch_eval.sprintctl.run_integrity.SCHEMA_VERSION,
                    "benchmark_valid": True,
                },
                "timeline_schema_version": batch_eval.sprintctl.UNIFIED_TIMELINE_SCHEMA_VERSION,
                "conditions": {"provider_usage_ledger_settled": True},
            }
        )
        + "\n"
    )
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", ops)
    monkeypatch.setattr(batch_eval, "BATCH_ROOT", ops / "batches")
    monkeypatch.setattr(batch_eval, "WEB", tmp_path / "web")
    batch_eval.atomic_json(
        batch_eval.batch_path(batch_id),
        {
            "batch_id": batch_id,
            "status": "running",
            "reasoning_effort": "max",
            "codex_version": "0.149.1",
            "run_hours": None,
            "arms": [{"run_id": run_id, "family": "luna", "status": "running"}],
            "alerts": [],
            "credential_status": "cleanup_error",
            "credential_cleanup_errors": ["transient management API failure"],
        },
    )
    monkeypatch.setattr(
        batch_eval,
        "live_run_monitor_status",
        lambda _run_id: {"harbor_alive": False, "ledger": {}, "stop_ack": {}},
    )
    monkeypatch.setattr(batch_eval, "log_alerts", lambda _run_id: [])
    monkeypatch.setattr(
        batch_eval, "verifier_lane_stall_alerts", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        batch_eval, "continuous_ledger_error_alerts", lambda _payload: []
    )
    monkeypatch.setattr(
        batch_eval.sprintctl,
        "load_run",
        lambda _run_id: (run_dir, {"provider_usage_ledger_required": False}),
    )
    cleanup_attempts: list[Path] = []

    def revoke(payload: dict[str, object], env_file: Path) -> list[str]:
        cleanup_attempts.append(env_file)
        payload["credential_status"] = "revoked"
        payload["credential_cleanup_errors"] = []
        return []

    monkeypatch.setattr(batch_eval, "revoke_batch_credentials", revoke)
    result = batch_eval.monitor_cycle(batch_id)

    assert result["status"] == "complete"
    assert result["arms"][0]["status"] == "finalized"
    assert "deploy" not in result
    assert not batch_eval.publication_path(batch_id).exists()
    assert result["credential_status"] == "revoked"
    assert cleanup_attempts == [batch_eval.ROOT / ".env"]


def test_batch_observer_never_finalizes_terminal_lane_inline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_id = "terminal-finalizing"
    run_id = "terminal-finalizing-sol-1"
    ops = tmp_path / "ops"
    run_dir = ops / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text("{}\n")
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", ops)
    monkeypatch.setattr(batch_eval, "BATCH_ROOT", ops / "batches")
    batch_eval.atomic_json(
        batch_eval.batch_path(batch_id),
        {
            "batch_id": batch_id,
            "status": "running",
            "reasoning_effort": "max",
            "codex_version": "0.149.1",
            "run_hours": None,
            "arms": [{"run_id": run_id, "family": "sol", "status": "running"}],
            "alerts": [],
            "credential_status": "revoked",
        },
    )
    monkeypatch.setattr(
        batch_eval,
        "live_run_monitor_status",
        lambda _run_id: {"harbor_alive": False, "ledger": {}, "stop_ack": {}},
    )
    monkeypatch.setattr(batch_eval, "log_alerts", lambda _run_id: [])
    monkeypatch.setattr(
        batch_eval, "verifier_lane_stall_alerts", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        batch_eval, "continuous_ledger_error_alerts", lambda _payload: []
    )
    monkeypatch.setattr(
        batch_eval.sprintctl,
        "finalize",
        lambda *_args, **_kwargs: pytest.fail("batch observer ran finalization"),
    )

    result = batch_eval.monitor_cycle(batch_id)

    arm = result["arms"][0]
    assert arm["status"] == "finalizing"
    assert arm["finalization_conditions"] == {
        "independent_run_finalizer_complete": False
    }


def test_batch_observer_recovers_transient_finalizing_lane_to_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_id = "recovered-finalizing"
    run_id = "recovered-finalizing-sol-1"
    ops = tmp_path / "ops"
    run_dir = ops / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text("{}\n")
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", ops)
    monkeypatch.setattr(batch_eval, "BATCH_ROOT", ops / "batches")
    batch_eval.atomic_json(
        batch_eval.batch_path(batch_id),
        {
            "batch_id": batch_id,
            "status": "running",
            "reasoning_effort": "max",
            "codex_version": "0.149.1",
            "run_hours": None,
            "arms": [
                {
                    "run_id": run_id,
                    "family": "sol",
                    "status": "finalizing",
                    "finalization_conditions": {
                        "independent_run_finalizer_complete": False
                    },
                }
            ],
            "alerts": [],
            "credential_status": "revoked",
        },
    )
    monkeypatch.setattr(
        batch_eval,
        "live_run_monitor_status",
        lambda _run_id: {
            "harbor_alive": True,
            "snapshot_heartbeat_ok": True,
            "ledger": {},
            "stop_ack": None,
        },
    )
    monkeypatch.setattr(batch_eval, "log_alerts", lambda _run_id: [])
    monkeypatch.setattr(
        batch_eval, "verifier_lane_stall_alerts", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        batch_eval, "continuous_ledger_error_alerts", lambda _payload: []
    )

    result = batch_eval.monitor_cycle(batch_id)

    arm = result["arms"][0]
    assert arm["status"] == "running"
    assert "finalization_conditions" not in arm


def test_run_monitor_startup_barrier_requires_live_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = iter(
        [
            {"run_id": "sol-1", "harbor_alive": False},
            {"run_id": "sol-1", "harbor_alive": True},
        ]
    )
    monkeypatch.setattr(
        batch_eval, "live_run_monitor_status", lambda _run_id: next(snapshots)
    )
    monkeypatch.setattr(batch_eval.time, "sleep", lambda _seconds: None)

    batch_eval.wait_for_run_monitors_ready(
        ["sol-1"], timeout_seconds=10, poll_seconds=0
    )


def test_run_monitor_startup_barrier_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(batch_eval, "live_run_monitor_status", lambda _run_id: None)

    with pytest.raises(RuntimeError, match="startup barrier timed out"):
        batch_eval.wait_for_run_monitors_ready(
            ["deepseek-1"], timeout_seconds=0, poll_seconds=0
        )


def test_batch_observer_recovers_missing_lane_monitor_without_remote_poll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_id = "recover-monitor"
    run_id = "recover-monitor-luna-1"
    ops = tmp_path / "ops"
    run_dir = ops / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text("{}\n")
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", ops)
    monkeypatch.setattr(batch_eval, "BATCH_ROOT", ops / "batches")
    batch_eval.atomic_json(
        batch_eval.batch_path(batch_id),
        {
            "batch_id": batch_id,
            "status": "running",
            "reasoning_effort": "max",
            "codex_version": "0.149.1",
            "run_hours": None,
            "arms": [{"run_id": run_id, "family": "luna", "status": "running"}],
            "alerts": [],
            "credential_status": "revoked",
        },
    )
    monkeypatch.setattr(batch_eval, "live_run_monitor_status", lambda _run_id: None)
    monkeypatch.setattr(
        batch_eval,
        "local_run_status",
        lambda _run_id: {
            "run_id": run_id,
            "harbor_alive": True,
            "ledger": {},
            "snapshot_heartbeat_ok": True,
        },
    )
    recovered: list[str] = []
    monkeypatch.setattr(
        batch_eval,
        "request_run_monitor_recovery",
        lambda candidate: recovered.append(candidate) is None or True,
    )
    monkeypatch.setattr(
        batch_eval.sprintctl,
        "monitor_once",
        lambda *_args, **_kwargs: pytest.fail("batch observer performed remote poll"),
    )
    monkeypatch.setattr(batch_eval, "log_alerts", lambda _run_id: [])
    monkeypatch.setattr(
        batch_eval, "verifier_lane_stall_alerts", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        batch_eval, "continuous_ledger_error_alerts", lambda _payload: []
    )

    result = batch_eval.monitor_cycle(batch_id)

    assert recovered == [run_id]
    assert result["arms"][0]["last_monitor_at"]
    assert result["arms"][0]["monitor_recovery_requested_at"]
    assert result["alerts"][0]["kind"] == "run_monitor_unavailable"


def test_resolved_website_alert_leaves_active_list_but_preserves_history() -> None:
    payload = {
        "alerts": [
            {
                "run_id": "batch",
                "kind": "website_deploy",
                "source": "FileNotFoundError",
                "first_seen_at": "2026-08-09T08:40:21Z",
            },
            {"run_id": "run-1", "kind": "provider_auth", "source": "agent.log"},
        ]
    }

    batch_eval.resolve_alerts(
        payload,
        run_id="batch",
        kind="website_deploy",
        resolution="subsequent_site_snapshot_succeeded",
    )

    assert payload["alerts"] == [
        {"run_id": "run-1", "kind": "provider_auth", "source": "agent.log"}
    ]
    assert payload["resolved_alerts"][0]["source"] == "FileNotFoundError"
    assert payload["resolved_alerts"][0]["resolved_at"]
    assert (
        payload["resolved_alerts"][0]["resolution"]
        == "subsequent_site_snapshot_succeeded"
    )


def test_provider_auth_alert_does_not_match_decimal_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "eval-luna-1"
    state_dir = tmp_path / run_id
    state_dir.mkdir()
    (state_dir / "frontier-worker.log").write_text(
        '{"max_distance_m": 4.401}\n'
        '"b9611a72b8f8f8146821ca6d4f83fa59e9812b7235e762825cf499d9b403e164"\n'
        "HTTP status 401 unauthorized\n"
    )
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", tmp_path)

    alerts = batch_eval.log_alerts(run_id)

    assert alerts == [
        {
            "run_id": run_id,
            "kind": "provider_auth",
            "source": "frontier-worker.log",
            "count_in_tail": "1",
        }
    ]


def test_provider_auth_alert_does_not_match_heartbeat_age(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "eval-sol-1"
    state_dir = tmp_path / run_id
    state_dir.mkdir()
    (state_dir / "monitor.log").write_text(
        '{"snapshot_heartbeat_age_seconds": 401, "snapshot_heartbeat_ok": true}\n'
    )
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", tmp_path)

    assert batch_eval.log_alerts(run_id) == []


def test_provider_rate_limit_alert_does_not_match_watchdog_age(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "eval-sol-1"
    state_dir = tmp_path / run_id
    state_dir.mkdir()
    (state_dir / "controller-errors.jsonl").write_text(
        '{"message": "budget pulse watchdog snapshot is stale (429.2s)"}\n'
    )
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", tmp_path)

    assert batch_eval.log_alerts(run_id) == []


def test_provider_rate_limit_alert_matches_http_429(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "eval-sol-1"
    state_dir = tmp_path / run_id
    state_dir.mkdir()
    (state_dir / "controller-errors.jsonl").write_text(
        '{"message": "OpenRouter response status 429"}\n'
    )
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", tmp_path)

    assert batch_eval.log_alerts(run_id) == [
        {
            "run_id": run_id,
            "kind": "provider_rate_limit",
            "source": "controller-errors.jsonl",
            "count_in_tail": "1",
        }
    ]


def test_recovered_log_alert_is_archived() -> None:
    payload = {
        "alerts": [
            {
                "run_id": "run-1",
                "kind": "provider_auth",
                "source": "monitor.log",
            },
            {
                "run_id": "run-1",
                "kind": "monitor_error",
                "source": "RuntimeError",
            },
        ]
    }

    batch_eval.resolve_recovered_log_alerts(payload, [])

    assert payload["alerts"] == [
        {
            "run_id": "run-1",
            "kind": "monitor_error",
            "source": "RuntimeError",
        }
    ]
    assert payload["resolved_alerts"][0]["kind"] == "provider_auth"
    assert (
        payload["resolved_alerts"][0]["resolution"]
        == "matching log condition recovered"
    )


def test_website_javascript_recognizes_every_launchable_model_family() -> None:
    source = (ROOT / "web/app.js").read_text()
    assert "DeepSeek V4 Flash Vision Exp" in source
    assert "DeepSeek V4 Flash 0731 · Baidu" in source
    assert "GPT‑5.6 Luna" in source
    assert "GPT‑5.6 Sol" in source
    assert "value.includes('gpt-5.6-sol')?'sol'" in source
    assert "Officially disqualified or unfinished" in source
    assert "lane exit" in source
    assert "Claude Opus 5" in source
    assert "GLM‑5.3‑Flash" in source
    assert "value.includes('claude-opus-5')?'opus'" in source
    assert "value.includes('glm-5.3-flash')?'glm'" in source
    spec = importlib.util.find_spec("json")
    assert spec is not None


def test_partial_batch_launch_is_safely_rolled_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arms = [
        {
            "run_id": f"eval-luna-{index}",
            "family": "luna",
            "model": "openai/gpt-5.6-luna",
            "resolved_model_version": "gpt-5.6-luna",
            "provider_endpoint": "openai",
            "reasoning_effort": "max",
            "codex_version": "0.147.0",
            "wrapper": f"wrapper-{index}",
            "trial": index,
            "status": "planned",
        }
        for index in (1, 2)
    ]
    monkeypatch.setattr(batch_eval, "BATCH_ROOT", tmp_path / "batches")
    monkeypatch.setattr(batch_eval, "matrix", lambda _batch_id, **_kwargs: arms)
    preflight_calls: list[dict[str, object]] = []

    def fake_preflight(**kwargs):
        preflight_calls.append(kwargs)
        return {
            "ready": True,
            "checks": {},
            "deepseek_pricing_snapshot": {"fixture": True},
        }

    monkeypatch.setattr(batch_eval, "preflight", fake_preflight)
    monkeypatch.setattr(
        batch_eval,
        "load_env",
        lambda _path: {"OPENROUTER_MANAGEMENT_KEY": "m" * 32},
    )
    monkeypatch.setattr(batch_eval, "OpenRouterManagementClient", lambda _key: object())

    class FakeCredential:
        def __init__(self, arm: dict[str, object]) -> None:
            self.run_id = str(arm["run_id"])
            self.api_key = f"child-{self.run_id}"

        def public_metadata(self) -> dict[str, object]:
            return {
                "run_id": self.run_id,
                "key_hash": f"hash-{self.run_id}",
                "guardrail_id": f"guard-{self.run_id}",
            }

    monkeypatch.setattr(
        batch_eval,
        "provision_trial_credentials",
        lambda _client, _specs, *, journal_path: [FakeCredential(arm) for arm in arms],
    )
    cleanup: list[dict[str, object]] = []
    monkeypatch.setattr(
        batch_eval,
        "revoke_trial_credentials",
        lambda _client, credentials, *, journal_path: cleanup.extend(credentials) or [],
    )
    monkeypatch.setattr(batch_eval.time, "sleep", lambda _seconds: None)
    calls = 0

    def fake_run(_command: list[str], *, env: dict[str, str] | None = None) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise subprocess.CalledProcessError(1, _command)
        return "launched"

    stopped: list[tuple[str, str]] = []
    monkeypatch.setattr(batch_eval, "run_checked", fake_run)
    monkeypatch.setattr(
        batch_eval.sprintctl,
        "request_stop",
        lambda run_id, *, reason: stopped.append((run_id, reason)),
    )
    with pytest.raises(subprocess.CalledProcessError):
        batch_eval.launch("eval", tmp_path / ".env", "test-profile")
    state = json.loads((tmp_path / "batches/eval/batch.json").read_text())
    assert "claude_code_version" not in state
    assert state["status"] == "launch_error"
    assert state["arms"][0]["status"] == "stopping_after_launch_rollback"
    assert stopped == [("eval-luna-1", "partial_batch_launch_rollback")]
    assert len(cleanup) == 2
    assert preflight_calls[0]["probe_training_fleet"] is True


def test_deployment_marker_is_durable_idempotent_publication_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    web = tmp_path / "web"
    ops = tmp_path / "ops"
    batches = tmp_path / "batches"
    run_id = "eval-luna-1"
    state_dir = ops / run_id
    policy_index = web / f"data/policies/{run_id}.json"
    policy_index.parent.mkdir(parents=True)
    state_dir.mkdir(parents=True)
    (batches / "eval").mkdir(parents=True)
    policy_index.write_text('{"policies":[]}\n')
    (state_dir / "frontier-state.json").write_text(
        json.dumps(
            {
                "capture_queue": [{"status": "captured"}],
                "policies": {
                    "abc": {"replay_path": "/sealed/replay.json"},
                },
                "captures": {"abc": {"valid": True}},
            }
        )
    )
    monkeypatch.setattr(batch_eval, "WEB", web)
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", ops)
    monkeypatch.setattr(batch_eval, "BATCH_ROOT", batches)
    monkeypatch.setattr(
        batch_eval.sprintctl,
        "load_run",
        lambda _run_id: (state_dir, {"run_id": run_id}),
    )
    uploads: list[str] = []
    monkeypatch.setattr(
        batch_eval.sprintctl,
        "volume_upload",
        lambda _run, _source, destination: uploads.append(destination),
    )
    site_hash = frontier_update.site_tree_hash(web)
    artifact_hashes = frontier_update.public_artifact_hashes(web)
    payload = {
        "batch_id": "eval",
        "arms": [{"run_id": run_id}],
    }
    publication = {
        "site_status": "deployed",
        "last_deployed_site_hash": site_hash,
        "last_deployed_public_artifacts": artifact_hashes,
        "last_deployed_at": "2026-08-08T00:00:00Z",
    }
    assert batch_eval.mark_deployed_runs(payload, publication) == []
    assert uploads == [
        f"runs/{run_id}/state/deployment-provenance/"
        f"{frontier_update.sha256_file(policy_index)}.json",
        f"runs/{run_id}/state/BATCH_SITE_DEPLOYED.json",
    ]
    assert batch_eval.mark_deployed_runs(payload, publication) == []
    assert len(uploads) == 2

    marker = json.loads((state_dir / "BATCH_SITE_DEPLOYED.json").read_text())
    snapshot = state_dir / marker["public_artifact_snapshot_path"]
    assert frontier_update.sha256_file(snapshot) == marker["public_artifact_sha256"]
    policy_index.write_text('{"policies":[{"new":true}]}\n')
    # The immutable snapshot records what reached production; it is observer
    # provenance and no longer gates benchmark finalization.
    assert frontier_update.sha256_file(snapshot) == marker["public_artifact_sha256"]
    snapshot.chmod(0o600)
    snapshot.write_text("tampered\n")
    assert frontier_update.sha256_file(snapshot) != marker["public_artifact_sha256"]


def test_deployment_marker_uses_timeline_when_run_has_no_submissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    web = tmp_path / "web"
    ops = tmp_path / "ops"
    batches = tmp_path / "batches"
    run_id = "eval-empty"
    state_dir = ops / run_id
    timeline = web / f"data/timelines/{run_id}.json"
    timeline.parent.mkdir(parents=True)
    timeline.write_text('{"coverage":{"ready":true}}\n')
    state_dir.mkdir(parents=True)
    (batches / "eval").mkdir(parents=True)
    (state_dir / "frontier-state.json").write_text(
        json.dumps({"capture_queue": [], "policies": {}, "captures": {}})
    )
    monkeypatch.setattr(batch_eval, "WEB", web)
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", ops)
    monkeypatch.setattr(batch_eval, "BATCH_ROOT", batches)
    monkeypatch.setattr(
        batch_eval.sprintctl,
        "load_run",
        lambda _run_id: (state_dir, {"run_id": run_id}),
    )
    monkeypatch.setattr(
        batch_eval.sprintctl, "volume_upload", lambda *_args, **_kwargs: None
    )
    payload = {
        "batch_id": "eval",
        "arms": [{"run_id": run_id}],
    }
    publication = {
        "site_status": "deployed",
        "last_deployed_site_hash": frontier_update.site_tree_hash(web),
        "last_deployed_public_artifacts": frontier_update.public_artifact_hashes(web),
        "last_deployed_at": "2026-08-09T00:00:00Z",
    }

    assert batch_eval.mark_deployed_runs(payload, publication) == []
    marker = json.loads((state_dir / "BATCH_SITE_DEPLOYED.json").read_text())
    assert marker["schema_version"] == 2
    assert marker["public_artifact_path"] == f"data/timelines/{run_id}.json"
    timeline.write_text('{"coverage":{"ready":false}}\n')
    snapshot = state_dir / marker["public_artifact_snapshot_path"]
    assert frontier_update.sha256_file(snapshot) == marker["public_artifact_sha256"]
    snapshot.chmod(0o600)
    snapshot.write_text("tampered\n")
    assert frontier_update.sha256_file(snapshot) != marker["public_artifact_sha256"]
