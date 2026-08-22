from __future__ import annotations

import contextlib
import datetime as dt
import importlib.util
import io
import json
import os
import subprocess
import sys
import urllib.error
from pathlib import Path
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "runs/ops"
sys.path.insert(0, str(OPS))
sys.path.insert(0, str(ROOT))

from event_runtime.control import batch as batch_eval  # noqa: E402
from event_runtime.export import frontier as frontier_update  # noqa: E402
from event_runtime.preflight import canary as training_gpu_canary  # noqa: E402
from event_runtime.export import timeline as unified_timeline  # noqa: E402


def test_event_runtime_state_does_not_dirty_the_source_tree() -> None:
    result = subprocess.run(
        [
            "git",
            "check-ignore",
            "--quiet",
            "runs/ops/arbitrary-batch-name/model-1/supervisor.json",
        ],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0


def test_launcher_provenance_guard_checks_source_not_generated_runs() -> None:
    launcher = (ROOT / "event_runtime/control/launch.sh").read_text()
    assert (
        "status --porcelain --untracked-files=all -- event_runtime events harbor"
        in launcher
    )
    assert (
        "status --porcelain --untracked-files=all -- events harbor runs" not in launcher
    )


def test_resume_volume_guard_uses_untruncated_modal_json() -> None:
    launcher = (ROOT / "event_runtime/control/launch.sh").read_text()
    assert "modal volume list --json" in launcher
    assert 'row.get("name") == target' in launcher
    assert 'modal volume list 2>/dev/null | grep -qF "$VOLUME_NAME"' not in launcher


def test_batch_matrix_is_exact_six_arm_max_effort_contract() -> None:
    rows = batch_eval.matrix("eval-20260808")
    assert len(rows) == 6
    assert len({row["run_id"] for row in rows}) == 6
    assert {row["family"] for row in rows} == {"deepseek", "luna"}
    assert {row["reasoning_effort"] for row in rows} == {"max"}
    assert {row["codex_version"] for row in rows} == {"0.147.0"}
    assert sum(
        row["model"] == "deepseek/deepseek-v4-flash-0731" for row in rows
    ) == 3
    assert sum(row["model"] == "openai/gpt-5.6-luna" for row in rows) == 3
    assert {
        row["resolved_model_version"] for row in rows if row["family"] == "deepseek"
    } == {"Baidu | deepseek/deepseek-v4-flash-20260731"}
    assert {
        (row["provider_endpoint"], row["quantization"])
        for row in rows
        if row["family"] == "deepseek"
    } == {("baidu/fp8", "fp8")}
    assert batch_eval.LIVE_SITE_DEPLOY_SECONDS == 20 * 60


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


def test_provider_retry_does_not_make_batch_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "eval-deepseek-1"
    state_dir = tmp_path / run_id
    state_dir.mkdir()
    (state_dir / "STOP_ACK.json").write_text('{"reason":"agent_exit"}\n')
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(batch_eval, "supervisor_active", lambda _run_id: True)

    assert not batch_eval.arm_terminal(
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
        == batch_eval.LIVE_SITE_DEPLOY_SECONDS
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
            "preflight": {
                "openrouter_credit_snapshot": {"per_trial_budget_usd": 12.5}
            },
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
                ]
            },
        )
    ]


def test_final_site_gate_ignores_unrelated_run_changes(
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
        "deploy": {
            "site_status": "deployed",
            "last_deployed_public_artifacts": frontier_update.public_artifact_hashes(
                web
            ),
        },
    }
    assert batch_eval.deployed_batch_current(payload)
    unrelated.write_text("changed elsewhere\n")
    assert batch_eval.deployed_batch_current(payload)
    performance.write_text("changed performance\n")
    assert not batch_eval.deployed_batch_current(payload)
    performance.write_text("current.json\n")
    assert batch_eval.deployed_batch_current(payload)
    timeline.write_text("changed in this run\n")
    assert not batch_eval.deployed_batch_current(payload)


def test_batch_matrix_can_launch_three_deepseek_trials_only() -> None:
    rows = batch_eval.matrix("eval-deepseek", families=("deepseek",))
    assert len(rows) == 3
    assert {row["family"] for row in rows} == {"deepseek"}
    assert [row["trial"] for row in rows] == [1, 2, 3]
    assert {row["model"] for row in rows} == {
        "deepseek/deepseek-v4-flash-0731"
    }
    assert {row["provider"] for row in rows} == {"Baidu"}
    assert {row["provider_endpoint"] for row in rows} == {"baidu/fp8"}
    assert {row["quantization"] for row in rows} == {"fp8"}


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


def test_batch_matrix_can_seal_baidu_and_alibaba_deepseek_routes() -> None:
    rows = batch_eval.matrix(
        "eval-ds-routes", families=("flash-baidu", "pro-alibaba")
    )
    assert len(rows) == 6
    assert {row["provider_endpoint"] for row in rows} == {
        "baidu/fp8",
        "alibaba",
    }
    assert {
        (row["family"], row["model"], row["quantization"])
        for row in rows
    } == {
        (
            "flash-baidu",
            "deepseek/deepseek-v4-flash-0731",
            "fp8",
        ),
        (
            "pro-alibaba",
            "deepseek/deepseek-v4-pro-0813",
            "unknown",
        ),
    }
    assert {row["wrapper"] for row in rows} == {
        str(ROOT / "event_runtime/control/providers/deepseek.sh")
    }


def test_sol_model_lock_preserves_exact_codex_contract() -> None:
    lock = json.loads((ROOT / "event_runtime/models/sol.json").read_text())
    model = lock["model"]
    assert lock["source"].startswith("openai/codex rust-v0.147.0")
    assert lock["codex_version"] == "0.147.0"
    assert lock["model_messages_sha256"] == (
        "e1ab3222ab4ceb4196f381138bf63232456419dba5a03bc276137a323e4134aa"
    )
    assert model["slug"] == "gpt-5.6-sol"
    assert model["tool_mode"] == "code_mode_only"
    assert model["multi_agent_version"] == "v2"
    assert model["context_window"] == 272000
    assert model["max_context_window"] == 272000
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
        "SPRINT_CODEX_DEEPSEEK_MODELS_JSON": str(
            ROOT / "event_runtime/models/deepseek.json"
        ),
    }
    subprocess.run(
        [
            "bash",
            str(
                ROOT
                / "event_runtime/container/sprint-apply-openai-codex-config.sh"
            ),
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
    config = (codex_home / "config.toml").read_text()
    assert 'model = "@preset/test-sol"' in config
    assert 'wire_api = "responses"' in config


def test_generic_deepseek_catalog_installer_supports_pro(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    env = {
        **os.environ,
        "CODEX_HOME": str(codex_home),
        "SPRINT_CODEX_DEEPSEEK_MODELS_JSON": str(
            ROOT / "event_runtime/models/deepseek.json"
        ),
        "SPRINT_CODEX_DEEPSEEK_MODEL": "deepseek/deepseek-v4-pro-0813",
        "SPRINT_CODEX_DEEPSEEK_CONTEXT_WINDOW": "1000000",
        "SPRINT_CODEX_DEEPSEEK_BASE_URL": "http://127.0.0.1:18080/api/v1",
    }
    subprocess.run(
        [
            "bash",
            str(
                ROOT
                / "event_runtime/container/sprint-apply-deepseek-codex-config.sh"
            ),
        ],
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    catalog = json.loads((codex_home / "models.json").read_text())
    assert len(catalog["models"]) == 1
    model = catalog["models"][0]
    assert model["slug"] == "deepseek/deepseek-v4-pro-0813"
    assert model["context_window"] == 1_000_000
    assert model["max_context_window"] == 1_000_000
    assert model["display_name"] == "DeepSeek-V4-Pro"
    assert model["support_verbosity"] is False


def test_deepseek_catalog_never_emits_unsupported_verbosity() -> None:
    catalog = json.loads(
        (ROOT / "event_runtime/models/deepseek.json").read_text()
    )
    assert catalog["models"]
    assert all(model["support_verbosity"] is False for model in catalog["models"])
    assert all(
        model["supports_parallel_tool_calls"] is False
        for model in catalog["models"]
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


def test_env_loader_reads_only_required_model_keys(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "OPENAI_API_KEY='openai-secret'\n"
        'OPENROUTER_API_KEY="openrouter-secret"\n'
        "MODAL_TOKEN_SECRET=must-not-load\n"
    )
    assert batch_eval.load_env(path) == {
        "OPENAI_API_KEY": "openai-secret",
        "OPENROUTER_API_KEY": "openrouter-secret",
    }


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
        json.dumps(
            {"data": {"total_credits": 100.0, "total_usage": 25.25}}
        ).encode()
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
            }
        )
    )
    monkeypatch.setattr(batch_eval, "WARMUP_MANIFEST", warmup)
    monkeypatch.setattr(batch_eval, "FUNCTIONAL_CANARY_REPORT", canary)
    assert batch_eval.functional_gpu_canary_ready()
    payload = json.loads(canary.read_text())
    payload["verifier_image_id"] = "im-stale"
    canary.write_text(json.dumps(payload))
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
                    "workers": [
                        {"worker_id": worker_id, "ready": True}
                        for worker_id in worker_ids
                    ],
                }
            )
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok")

    monkeypatch.setattr(batch_eval.subprocess, "run", successful_run)
    ready, report = batch_eval.training_gpu_fleet_probe(
        batch_id="eval", modal_profile="test", worker_ids=["eval-1", "eval-2"]
    )
    assert ready
    assert [row["worker_id"] for row in report["workers"]] == ["eval-1", "eval-2"]

    def incomplete_run(command, **kwargs):
        result = successful_run(command, **kwargs)
        report = Path(command[command.index("--report") + 1])
        payload = json.loads(report.read_text())
        payload["workers"].pop()
        report.write_text(json.dumps(payload))
        return result

    monkeypatch.setattr(batch_eval.subprocess, "run", incomplete_run)
    ready, _ = batch_eval.training_gpu_fleet_probe(
        batch_id="eval", modal_profile="test", worker_ids=["eval-1", "eval-2"]
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
    current = json.loads(
        (tmp_path / "web/data/batches/current.json").read_text()
    )
    assert current["tracked_batch_ids"] == ["base", "replacement"]
    assert [arm["run_id"] for arm in current["arms"]] == [
        "base-sol-1",
        "replacement-luna-1",
    ]


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
        (None, {"reason": "budget_telemetry_unavailable"}, "budget_telemetry_unavailable"),
        ("untrusted_agent_stop_marker", None, "untrusted_agent_stop_marker"),
    ],
)
def test_tracking_batch_excludes_invalid_replaced_lane(
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
        {"batch_id": batch_id, "arms": arms, "alerts": []},
    )
    persisted: list[str] = []
    dispatched: list[str] = []

    def persist(run_id: str, *, reason: str):
        assert reason == "operator_batch_stop"
        persisted.append(run_id)
        return tmp_path, {}, {}

    def dispatch(run_id: str, *, reason: str):
        assert persisted == ["run-1", "run-2", "run-3"]
        dispatched.append(run_id)
        if run_id == "run-1":
            raise RuntimeError("provider unavailable")
        return {"status": "requested"}

    with (
        mock.patch.object(batch_eval.sprintctl, "persist_stop_request", persist),
        mock.patch.object(batch_eval.sprintctl, "request_stop", dispatch),
    ):
        result = batch_eval.stop_batch(batch_id)

    assert dispatched == ["run-1", "run-2", "run-3"]
    assert all(arm["status"] == "stopping" for arm in result["arms"])
    assert result["arms"][0]["stop_dispatch_error"].startswith("RuntimeError:")
    assert result["arms"][1]["stop_dispatch_status"] == "requested"
    stored = json.loads(batch_eval.batch_path(batch_id).read_text())
    assert [arm["status"] for arm in stored["arms"]] == [
        "stopping",
        "stopping",
        "stopping",
    ]


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
                "max_distance_semantics": ("legal_prefix_until_first_disqualification"),
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
    frontier_update.write_web_policy_indexes(state_path, state, web)
    public = json.loads((web / "data/policies/eval-deepseek-1.json").read_text())
    encoded = json.dumps(public)
    assert str(tmp_path) not in encoded
    assert public["policies"][0]["replay_ready"] is True
    assert public["policies"][0]["replay_url"].startswith("/replay/frontier-")
    assert public["policies"][0]["max_distance_semantics"] == (
        "legal_prefix_until_first_disqualification"
    )


def test_public_policy_index_keeps_only_six_newest_runs(tmp_path: Path) -> None:
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
    assert "cache read" in source
    assert "${row.submitted} submitted</b>" in source
    assert "Total cost" in source
    assert "await json(tMeta.path)" not in source
    assert "showReadout" in source
    assert "readout-timeline" in source
    assert ".append(document.createElementNS" not in source
    assert "final_api_cost_usd" in source
    assert "modal_provider_billing" in source
    assert "verifier sandbox excluded" in source
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


def test_batch_monitor_service_carries_absolute_uv_vercel_and_modal_profile(
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
    batch_eval.start_monitor_service("eval", tmp_path / ".env", "profile-a")
    unit_path = config_home / "systemd/user/sprint-batch-eval-monitor.service"
    unit = unit_path.read_text()
    assert 'Environment="UV=/home/ubuntu/.local/bin/uv"' in unit
    assert 'Environment="MODAL_PROFILE=profile-a"' in unit
    assert "/tools" in unit
    assert str(batch_eval.HARBOR_PYTHON.parent) in unit
    assert str(batch_eval.HARBOR_PYTHON) in unit
    assert f"WorkingDirectory={batch_eval.ROOT}" in unit
    assert "Restart=on-failure" in unit
    assert "WantedBy=default.target" in unit
    assert "must-not-appear-in-unit" not in unit
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


def test_batch_monitor_does_not_treat_agent_exit_ack_as_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "eval-deepseek-1"
    state_dir = tmp_path / run_id
    state_dir.mkdir()
    (state_dir / "STOP_ACK.json").write_text('{"reason":"agent_exit"}\n')
    (state_dir / "monitor.pid").write_text("1234\n")
    expected = {
        "schema_version": 2,
        "run_id": run_id,
        "harbor_alive": True,
        "stop_ack": {"reason": "agent_exit"},
    }
    (state_dir / "status.json").write_text(json.dumps(expected))
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(batch_eval.sprintctl, "process_alive", lambda *_args: True)
    monkeypatch.setattr(
        batch_eval.sprintctl,
        "status_snapshot",
        lambda *_args, **_kwargs: pytest.fail("retry ack is not terminal"),
    )

    assert batch_eval.live_run_monitor_status(run_id) == expected


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

    result = batch_eval.monitor_cycle(batch_id, deploy=False)

    assert result["arms"][0]["status"] == "running"


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


def test_website_javascript_parses_and_has_no_legacy_opus_copy() -> None:
    source = (ROOT / "web/app.js").read_text()
    assert "DeepSeek V4 Flash 0731" in source
    assert "GPT‑5.6 Luna" in source
    assert "GPT‑5.6 Sol" in source
    assert "value.includes('gpt-5.6-sol')?'sol'" in source
    assert "did not finish" in source
    assert "left the lane" in source
    assert "Opus" not in source
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
    monkeypatch.setattr(batch_eval, "load_env", lambda _path: {})
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
    assert state["status"] == "launch_error"
    assert state["arms"][0]["status"] == "stopping_after_launch_rollback"
    assert stopped == [("eval-luna-1", "partial_batch_launch_rollback")]
    assert preflight_calls[0]["probe_training_fleet"] is True


def test_deployment_marker_is_durable_idempotent_and_gates_finalization(
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
        "deploy": {
            "site_status": "deployed",
            "last_deployed_site_hash": site_hash,
            "last_deployed_public_artifacts": artifact_hashes,
            "last_deployed_at": "2026-08-08T00:00:00Z",
        },
        "arms": [{"run_id": run_id}],
    }
    assert batch_eval.mark_deployed_runs(payload) == []
    assert uploads == [
        f"runs/{run_id}/state/deployment-provenance/"
        f"{frontier_update.sha256_file(policy_index)}.json",
        f"runs/{run_id}/state/BATCH_SITE_DEPLOYED.json",
    ]
    assert batch_eval.mark_deployed_runs(payload) == []
    assert len(uploads) == 2

    run = {
        "run_id": run_id,
        "batch_id": "eval",
        "site_dir": str(web),
    }
    assert batch_eval.sprintctl.batch_site_deployed_ready(state_dir, run)
    policy_index.write_text('{"policies":[{"new":true}]}\n')
    assert batch_eval.sprintctl.batch_site_deployed_ready(state_dir, run)
    snapshot = (
        state_dir
        / json.loads((state_dir / "BATCH_SITE_DEPLOYED.json").read_text())[
            "public_artifact_snapshot_path"
        ]
    )
    snapshot.chmod(0o600)
    snapshot.write_text("tampered\n")
    assert not batch_eval.sprintctl.batch_site_deployed_ready(state_dir, run)


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
        "deploy": {
            "site_status": "deployed",
            "last_deployed_site_hash": frontier_update.site_tree_hash(web),
            "last_deployed_public_artifacts": frontier_update.public_artifact_hashes(
                web
            ),
            "last_deployed_at": "2026-08-09T00:00:00Z",
        },
        "arms": [{"run_id": run_id}],
    }

    assert batch_eval.mark_deployed_runs(payload) == []
    marker = json.loads((state_dir / "BATCH_SITE_DEPLOYED.json").read_text())
    assert marker["schema_version"] == 2
    assert marker["public_artifact_path"] == f"data/timelines/{run_id}.json"
    run = {"run_id": run_id, "batch_id": "eval", "site_dir": str(web)}
    assert batch_eval.sprintctl.batch_site_deployed_ready(state_dir, run)
    timeline.write_text('{"coverage":{"ready":false}}\n')
    assert batch_eval.sprintctl.batch_site_deployed_ready(state_dir, run)
    snapshot = state_dir / marker["public_artifact_snapshot_path"]
    snapshot.chmod(0o600)
    snapshot.write_text("tampered\n")
    assert not batch_eval.sprintctl.batch_site_deployed_ready(state_dir, run)
