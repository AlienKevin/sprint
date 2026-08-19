from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "runs/ops"
sys.path.insert(0, str(OPS))
sys.path.insert(0, str(ROOT))

from event_runtime.compute import worker as gpu_worker  # noqa: E402
from event_runtime.cost import agent as agent_cost  # noqa: E402


def timeline_fixture() -> dict:
    pricing = {
        "id": "fixture",
        "rates_usd_per_second": {"CPU": "1", "Memory": "2", "A10G": "10"},
    }
    return {
        "generated_at": "2026-08-11T00:00:04Z",
        "run": {"run_id": "run-1", "model": "openai/gpt-5.6-luna"},
        "clock": {"origin_epoch_ms": 0, "end_epoch_ms": 4000},
        "events": [
            {"kind": "cpu_allocated", "epoch_ms": 0},
            {"kind": "gpu_allocated", "epoch_ms": 1000, "lease_id": "lease"},
            {
                "kind": "model_request_usage",
                "epoch_ms": 2000,
                "calculated_cost_usd": 0.5,
                "cost_components_usd": {"output": 0.5},
            },
            {"kind": "gpu_released", "epoch_ms": 3000, "lease_id": "lease"},
        ],
        "usage_summary": {
            "request_count": 1,
            "ordinary_uncached_input_tokens": 2,
            "cached_input_tokens": 3,
            "cache_write_input_tokens": 4,
            "output_tokens": 5,
            "reasoning_output_tokens": 1,
        },
        "resource_usage_summary": {
            "resource_contract": {
                "cpu_agent": {
                    "physical_cpu_cores": 1,
                    "memory_mb": 1024,
                    "gpu_count": 0,
                },
                "training_worker": {
                    "physical_cpu_cores": 1,
                    "memory_mb": 1024,
                    "gpu_count": 1,
                    "gpu_type": "A10G",
                },
            },
            "modal_estimate": {
                "pricing_snapshot": pricing,
                "by_role": {
                    "cpu_agent": {
                        "allocated_ms": 4000,
                        "quantities": {"CPU": 1, "Memory": 1, "A10G": 0},
                        "cost_components_usd": {"CPU": 4, "Memory": 8},
                        "estimated_cost_usd": 12,
                    },
                    "training_gpu": {
                        "allocated_ms": 2000,
                        "quantities": {"CPU": 1, "Memory": 1, "A10G": 1},
                        "cost_components_usd": {"CPU": 2, "Memory": 4, "A10G": 20},
                        "estimated_cost_usd": 26,
                    },
                },
            },
        },
    }


def test_snapshot_has_one_matching_component_ledger_and_all_constants(
    tmp_path: Path,
) -> None:
    usage = tmp_path / "usage" / "run-usage-audit.json"
    usage.parent.mkdir()
    usage.write_text(
        json.dumps(
            {
                "pricing_snapshots": [
                    {
                        "id": "api-fixture",
                        "unit_tokens": 1_000_000,
                        "rates_usd_per_million_tokens": {"output": "1.20"},
                    }
                ]
            }
        )
    )
    payload = agent_cost.build_snapshot(timeline_fixture(), state_dir=tmp_path)

    assert payload["total_usd"] == 38.5
    assert payload["components"]["model_api"]["cost_usd"] == 0.5
    assert payload["components"]["cpu_agent"]["cost_usd"] == 12
    assert payload["components"]["training_sandboxes"]["cost_usd"] == 26
    assert payload["components"]["model_api"]["cost_components_usd"] == {"output": 0.5}
    assert payload["constants"]["api_pricing_snapshots"][0]["id"] == "api-fixture"
    assert payload["constants"]["a10g_memory_billing"].startswith("included")
    assert "official_verifier" in payload["excluded"]


def test_host_conservatively_merges_watchdog_and_host_allocation_ledgers(
    tmp_path: Path,
) -> None:
    canonical = {
        "schema_version": 2,
        "run_id": "run-1",
        "status": "within_budget",
        "total_usd": 7.25,
        "components": {
            "model_api": {
                "cost_usd": 6.0,
                "cost_source": "openrouter_reported_per_request",
                "cost_components_usd": {"output": 0.0},
                "tokens": {
                    "ordinary_uncached_input_tokens": 0,
                    "cached_input_tokens": 0,
                    "cache_write_input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_output_tokens": 0,
                },
            },
            "cpu_agent": {
                "cost_usd": 0.25,
                "cost_components_usd": {"CPU": 0.0, "Memory": 0.0},
            },
            "training_sandboxes": {
                "cost_usd": 1.0,
                "cost_components_usd": {
                    "CPU": 0.0,
                    "Memory": 0.0,
                    "A10G": 0.0,
                },
            },
        },
    }
    path = tmp_path / "telemetry/budget-watchdog.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(canonical))

    payload = agent_cost.build_snapshot(timeline_fixture(), state_dir=tmp_path)

    # The provider-reported API charge wins over the older host request view,
    # while the host's fresher CPU/GPU lifecycle values win over a stale Modal
    # Volume mount.  The merged total is the one mirrored back to the agent.
    assert payload["components"]["model_api"]["cost_usd"] == 6.0
    assert payload["components"]["cpu_agent"]["cost_usd"] == 12
    assert payload["components"]["training_sandboxes"]["cost_usd"] == 26
    assert payload["components"]["model_api"]["tokens"] == {
        "ordinary_uncached_input_tokens": 2,
        "cached_input_tokens": 3,
        "cache_write_input_tokens": 4,
        "output_tokens": 5,
        "reasoning_output_tokens": 1,
    }
    assert payload["components"]["model_api"]["cost_components_usd"] == {
        "output": 0.5
    }
    assert payload["components"]["cpu_agent"]["cost_components_usd"] == {
        "CPU": 4,
        "Memory": 8,
    }
    assert payload["components"]["training_sandboxes"][
        "cost_components_usd"
    ] == {"CPU": 2, "Memory": 4, "A10G": 20}
    assert payload["component_totals_usd"] == {
        "model_api_usd": 6.0,
        "cpu_agent_usd": 12,
        "training_sandboxes_usd": 26,
    }
    assert payload["total_usd"] == 44.0
    assert payload["training_allocated_seconds"] == 2.0
    assert payload["schema_version"] == 2


def test_host_merged_cost_uses_previous_snapshot_as_monotonic_floor(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry"
    telemetry.mkdir(parents=True)
    (telemetry / "budget-watchdog.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "run-1",
                "status": "within_budget",
                "budget_usd": 100.0,
                "stop_threshold_usd": 99.9,
                "total_usd": 1.75,
                "components": {
                    "model_api": {"cost_usd": 0.5, "request_count": 1},
                    "cpu_agent": {"cost_usd": 0.25, "allocated_seconds": 1.0},
                    "training_sandboxes": {
                        "cost_usd": 1.0,
                        "allocated_seconds": 1.0,
                    },
                },
            }
        )
    )
    (telemetry / "agent-cost.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "run-1",
                "status": "within_budget",
                "total_usd": 53.5,
                "components": {
                    "model_api": {"cost_usd": 0.5, "request_count": 1},
                    "cpu_agent": {"cost_usd": 13.0, "allocated_seconds": 5.0},
                    "training_sandboxes": {
                        "cost_usd": 40.0,
                        "allocated_seconds": 4.0,
                    },
                },
            }
        )
    )

    payload = agent_cost.build_snapshot(timeline_fixture(), state_dir=tmp_path)

    assert payload["component_totals_usd"] == {
        "model_api_usd": 0.5,
        "cpu_agent_usd": 13.0,
        "training_sandboxes_usd": 40.0,
    }
    assert payload["total_usd"] == 53.5
    assert payload["cpu_allocated_seconds"] == 5.0
    assert payload["training_allocated_seconds"] == 4.0
    assert payload["budget_remaining_usd"] == 46.5


def test_live_ledger_caps_open_training_interval_at_snapshot_time() -> None:
    timeline = timeline_fixture()
    timeline["events"] = timeline["events"][:-1]
    ledger = agent_cost.build_cost_ledger(timeline)
    # CPU: 4 * (1 CPU + 1 GiB memory) = 12; training: 3 * 13 = 39; API: .5.
    assert agent_cost.cumulative_cost_at_epoch(ledger, 4000) == 51.5


def test_event_cost_is_a_single_json_interface(tmp_path: Path, monkeypatch) -> None:
    path = ROOT / "event_runtime/agent/cost.py"
    spec = importlib.util.spec_from_loader(
        "event_cost_cli", SourceFileLoader("event_cost_cli", str(path))
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    snapshot = tmp_path / "cost.json"
    snapshot.write_text(json.dumps({"schema_version": 1, "total_usd": 1.25}))
    monkeypatch.setattr(module, "SNAPSHOT", snapshot)
    monkeypatch.setattr(module, "MIRRORED_SNAPSHOT", snapshot)
    monkeypatch.setattr(module.sys, "argv", ["event cost"])
    assert module.main() == 0


def test_event_cost_prefers_fresh_host_mirror_over_durable_mount(
    tmp_path: Path, monkeypatch
) -> None:
    path = ROOT / "event_runtime/agent/cost.py"
    spec = importlib.util.spec_from_loader(
        "event_cost_mirror_priority",
        SourceFileLoader("event_cost_mirror_priority", str(path)),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mirror = tmp_path / "run/cost.json"
    mirror.parent.mkdir(parents=True)
    mirror.write_text(json.dumps({"schema_version": 2, "total_usd": 2.0}))
    monkeypatch.setattr(module, "MIRRORED_SNAPSHOT", mirror)
    monkeypatch.setenv("SPRINT_RUN_ID", "run-1")

    assert module.snapshot_path() == mirror


def test_host_atomically_mirrors_cost_and_cli(tmp_path: Path, monkeypatch) -> None:
    mirror = tmp_path / "mirror"
    cli = tmp_path / "event_runtime" / "agent" / "cost.py"
    cli.parent.mkdir(parents=True)

    def local_exec(
        _run: dict, container_id: str, command: str, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert container_id == "ta-agent"
        return subprocess.run(
            command,
            shell=True,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    monkeypatch.setattr(gpu_worker, "AGENT_GPU_MIRROR_ROOT", str(mirror))
    monkeypatch.setattr(gpu_worker, "AGENT_COST_CLI_PATH", str(cli))
    monkeypatch.setattr(gpu_worker.sprintctl, "exec_container", local_exec)
    detail = gpu_worker.mirror_agent_cost(
        {"agent_container_id": "ta-agent"},
        {"schema_version": 1, "as_of": "now", "total_usd": 2.5},
    )

    assert detail["agent_cost_mirror"] == "updated"
    assert json.loads((mirror / "cost.json").read_text())["total_usd"] == 2.5
    assert cli.read_bytes() == (ROOT / "event_runtime/agent/cost.py").read_bytes()


def test_agent_cost_mirror_refuses_to_replace_newer_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    cli = tmp_path / "event_runtime" / "agent" / "cost.py"
    cli.parent.mkdir(parents=True)
    current = {
        "schema_version": 2,
        "checked_at_epoch_s": 200.0,
        "total_usd": 3.0,
    }
    (mirror / "cost.json").write_text(json.dumps(current))

    def local_exec(
        _run: dict, _container_id: str, command: str, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            shell=True,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    monkeypatch.setattr(gpu_worker, "AGENT_GPU_MIRROR_ROOT", str(mirror))
    monkeypatch.setattr(gpu_worker, "AGENT_COST_CLI_PATH", str(cli))
    monkeypatch.setattr(gpu_worker.sprintctl, "exec_container", local_exec)
    detail = gpu_worker.mirror_agent_cost(
        {"agent_container_id": "ta-agent"},
        {
            "schema_version": 2,
            "checked_at_epoch_s": 100.0,
            "total_usd": 9.0,
        },
    )

    assert detail["agent_cost_mirror"] == "stale_ignored"
    assert json.loads((mirror / "cost.json").read_text()) == current


@pytest.mark.parametrize("marker", ["STOP_REQUESTED.json", "STOP_ACK.json"])
def test_host_skips_cost_mirror_after_agent_stop(
    tmp_path: Path, monkeypatch, marker: str
) -> None:
    (tmp_path / marker).write_text("{}")
    execute = mock.Mock(side_effect=AssertionError("stopped agent must not be called"))
    monkeypatch.setattr(gpu_worker.sprintctl, "exec_container", execute)

    detail = gpu_worker.mirror_agent_cost(
        {"agent_container_id": "ta-agent", "state_dir": str(tmp_path)},
        {"schema_version": 1, "total_usd": 10.0},
    )

    assert detail == {"agent_cost_mirror": "agent_stopped"}
    execute.assert_not_called()


def test_host_treats_finished_agent_task_as_stopped(monkeypatch) -> None:
    monkeypatch.setattr(
        gpu_worker.sprintctl,
        "exec_container",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [],
            1,
            "",
            "Task has already finished with status success",
        ),
    )

    detail = gpu_worker.mirror_agent_cost(
        {"agent_container_id": "ta-agent"},
        {"schema_version": 1, "total_usd": 10.0},
    )

    assert detail == {"agent_cost_mirror": "agent_stopped"}
