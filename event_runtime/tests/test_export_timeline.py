from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "runs/ops"
ENV = ROOT / "event_runtime/container"
sys.path.insert(0, str(OPS))
sys.path.insert(0, str(ROOT))

from event_runtime.export import timeline as unified_timeline  # noqa: E402
from event_runtime.container.sprint_openrouter_usage import (  # noqa: E402
    empty_token_usage,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_deepseek_harness_trace_uses_nested_event_clock_and_shape() -> None:
    row = {
        "schema_version": 1,
        "method": "session.event",
        "payload": {
            "event": {
                "seq": 17,
                "type": "tool/call",
                "time": 1_787_571_739_098,
                "data": {"name": "exec", "callId": "call-17"},
            }
        },
    }

    assert unified_timeline.Builder._record_timestamp(row) == 1_787_571_739_098
    kind, data = unified_timeline.Builder._trace_shape(row)
    assert kind == "tool_call"
    assert data == {
        "trace_type": "deepseek_harness",
        "trace_subtype": "tool/call",
        "harness_sequence": 17,
        "tool": "exec",
        "call_id": "call-17",
    }


def fixture_run(
    tmp_path: Path, *, missing_artifact: bool = False, training_lifecycle: bool = True
) -> Path:
    state = tmp_path / "run-state"
    run_id = "timeline-fixture"
    trial = state / "harbor-jobs" / run_id / "task__abc"
    run = {
        "schema_version": 2,
        "run_id": run_id,
        "created_at": "2026-08-07T12:00:00Z",
        "model": "openai/test-model",
        "agent_kind": "codex",
        "cpu_agent_gpu_worker": True,
        "scoring_queue_scope": "per_run_model",
        "scoring_queue_key": run_id,
        "scoring_max_concurrent": 1,
        "resource_contract": {
            "cpu_agent": {
                "physical_cpu_cores": 2,
                "vcpus_equivalent": 4,
                "memory_mb": 8192,
                "gpus": 0,
            },
            "training_worker": {
                "physical_cpu_cores": 8,
                "vcpus_equivalent": 16,
                "memory_mb": 32768,
                "gpu_count": 1,
                "gpu_type": "A10G",
            },
            "verifier": {
                "physical_cpu_cores": 4,
                "vcpus_equivalent": 8,
                "memory_mb": 10240,
                "gpu_count": 1,
                "gpu_type": "A10G",
            },
        },
    }
    state.mkdir(parents=True)
    (state / "run.json").write_text(json.dumps(run))
    write_jsonl(
        state / "telemetry" / "cpu_lifecycle.jsonl",
        [
            {
                "event": "cpu_launch_started",
                "attempt": 1,
                "at": "2026-08-07T12:00:01Z",
            },
        ],
    )
    telemetry = [
        {
            "epoch_s": 1786104010,
            "ts_utc": "2026-08-07T12:00:10Z",
            "role": "cpu-agent",
            "container_id": "cpu-1",
            "sample_index": 1,
            "cpu_util_pct": 42.5,
            "mem_used_kib": 100,
            "mem_total_kib": 1000,
            "gpus": [],
        },
        {
            "epoch_s": 1786104011,
            "ts_utc": "2026-08-07T12:00:11Z",
            "role": "training-gpu",
            "container_id": "gpu-1",
            "job_id": "job-1",
            "attempt": 1,
            "lease_id": "lease-1",
            "sample_index": 1,
            "cpu_util_pct": 12.0,
            "gpus": [
                {
                    "gpu_index": 0,
                    "util_gpu_pct": 88,
                    "util_mem_pct": 51,
                    "mem_used_mib": 8192,
                    "mem_total_mib": 24576,
                    "pipeline_metrics_source": "cupti-pm-sampling",
                    "pipeline_metrics_status": "ok",
                    "pipeline_metrics_window_ms": 1000,
                    "pipeline_metrics_sample_count": 10,
                    "sm_active_pct": 60.0,
                    "sm_occupancy_pct": 40.0,
                    "tensor_pipe_active_pct": 20.0,
                    "fp32_fma_pipe_active_pct": 30.0,
                    "fp16_instruction_pct_of_peak_active": 10.0,
                    "dram_throughput_pct": 50.0,
                }
            ],
        },
        {
            "epoch_s": 1786104015,
            "role": "training-gpu",
            "container_id": "gpu-1",
            "job_id": "job-1",
            "attempt": 1,
            "lease_id": "lease-1",
            "sample_index": 2,
            "gpus": [
                {
                    "gpu_index": 0,
                    "util_gpu_pct": 80,
                    "mem_used_mib": 8000,
                    "mem_total_mib": 24576,
                    "pipeline_metrics_source": "cupti-pm-sampling",
                    "pipeline_metrics_status": "ok",
                    "pipeline_metrics_window_ms": 1000,
                    "pipeline_metrics_sample_count": 10,
                    "sm_active_pct": 80.0,
                    "sm_occupancy_pct": 60.0,
                    "tensor_pipe_active_pct": 40.0,
                    "fp32_fma_pipe_active_pct": 50.0,
                    "fp16_instruction_pct_of_peak_active": 30.0,
                    "dram_throughput_pct": 70.0,
                }
            ],
        },
        {
            "epoch_s": 1786104025,
            "role": "training-gpu",
            "container_id": "gpu-2",
            "job_id": "job-1",
            "attempt": 2,
            "lease_id": "lease-2",
            "sample_index": 1,
            "gpus": [
                {
                    "gpu_index": 0,
                    "util_gpu_pct": 84,
                    "mem_used_mib": 8100,
                    "mem_total_mib": 24576,
                }
            ],
        },
        {
            "epoch_s": 1786104035,
            "role": "training-gpu",
            "container_id": "gpu-2",
            "job_id": "job-1",
            "attempt": 2,
            "lease_id": "lease-2",
            "sample_index": 2,
            "gpus": [
                {
                    "gpu_index": 0,
                    "util_gpu_pct": 86,
                    "mem_used_mib": 8200,
                    "mem_total_mib": 24576,
                }
            ],
        },
    ]
    write_jsonl(state / "telemetry" / "host-samples.jsonl", telemetry + telemetry)
    lifecycle = [
        {
            "event_id": "a",
            "epoch_s": 1786104005,
            "phase": "gpu_lifecycle",
            "action": "instant",
            "job_id": "job-1",
            "attempt": 1,
            "lease_id": "lease-1",
            "detail": {"event": "gpu_allocated"},
        },
        {
            "event_id": "b",
            "epoch_s": 1786104020,
            "phase": "gpu_lifecycle",
            "action": "instant",
            "job_id": "job-1",
            "attempt": 1,
            "lease_id": "lease-1",
            "detail": {"event": "gpu_preempted", "reason": "worker_lost"},
        },
        {
            "event_id": "c",
            "epoch_s": 1786104022,
            "phase": "gpu_lifecycle",
            "action": "instant",
            "job_id": "job-1",
            "attempt": 2,
            "lease_id": "lease-2",
            "detail": {"event": "gpu_reallocated"},
        },
        {
            "event_id": "d",
            "epoch_s": 1786104040,
            "phase": "gpu_lifecycle",
            "action": "instant",
            "job_id": "job-1",
            "attempt": 2,
            "lease_id": "lease-2",
            "detail": {"event": "gpu_released", "reason": "succeeded"},
        },
    ]
    if training_lifecycle:
        write_jsonl(state / "telemetry" / "gpu_timeline.jsonl", lifecycle)

    secret = "SUPER_SECRET_TOOL_ARGUMENT"
    trace = [
        {
            "timestamp": "2026-08-07T12:00:12.000Z",
            "type": "session_meta",
            "payload": {"id": "s"},
        },
        {
            "timestamp": "2026-08-07T12:00:13.000Z",
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call",
                "name": "exec",
                "call_id": "call-1",
                "input": secret,
            },
        },
        {
            "timestamp": "2026-08-07T12:00:14.000Z",
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call_output",
                "call_id": "call-1",
                "output": secret,
            },
        },
        {
            "timestamp": "2026-08-07T12:00:15.000Z",
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call",
                "name": "view_image",
                "call_id": "call-2",
                "input": secret,
            },
        },
        {
            "timestamp": "2026-08-07T12:00:16.000Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "call-shell-start",
                "arguments": json.dumps({"cmd": secret}),
            },
        },
        {
            "timestamp": "2026-08-07T12:00:17.000Z",
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call-shell-start",
                "output": (
                    "Wall time: 1.0000 seconds\nProcess running with session ID 23796\n"
                ),
            },
        },
        {
            "timestamp": "2026-08-07T12:00:18.000Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "write_stdin",
                "call_id": "call-shell-finish",
                "arguments": json.dumps({"session_id": 23796, "yield_time_ms": 30_000}),
            },
        },
        {
            "timestamp": "2026-08-07T12:00:20.000Z",
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call-shell-finish",
                "output": "Wall time: 2.0000 seconds\nProcess exited with code 0\n",
            },
        },
    ]
    trace_path = trial / "agent" / "sessions" / "rollout.jsonl"
    write_jsonl(trace_path, trace)
    duplicate = trial / "agent" / "codex-state" / "sessions" / "rollout.jsonl"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(trace_path.read_bytes())

    ledger_rows = []
    for index in (1, 2):
        name = f"policy-{index}.pt"
        relative = (
            f"continuous/attempts/{index:04d}-{name}/artifacts/app/submission/policy.pt"
        )
        artifact = trial / "artifacts" / relative
        if not (missing_artifact and index == 2):
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(f"policy-{index}".encode())
        ledger_rows.append(
            {
                "index": index,
                "name": name,
                "submitted_at": f"2026-08-07T12:00:{20 + index:02d}Z",
                "started_at": "2026-08-07T12:00:22Z"
                if index == 1
                else "2026-08-07T12:00:25Z",
                "finished_at": "2026-08-07T12:00:24Z"
                if index == 1
                else "2026-08-07T12:00:27Z",
                "artifact_path": relative,
                "rewards": {"best_100m_s": 50 - index, "valid_run": 1},
                "error": None,
                "verification_attempts": 2 if index == 1 else 1,
                "verification_retry_events": [
                    {
                        "attempt": 1,
                        "failed_at": "2026-08-07T12:00:22.500Z",
                        "error_type": "NotFoundError",
                        "error": "Modal Sandbox ta-old not found",
                    }
                ]
                if index == 1
                else [],
            }
        )
    write_jsonl(trial / "artifacts" / "continuous" / "ledger.jsonl", ledger_rows)
    verifier_windows = [
        (
            trial
            / "artifacts"
            / "continuous"
            / "attempts"
            / "0001-policy-1.pt"
            / "verifier"
            / "telemetry",
            "2026-08-07T12:00:22.100Z",
            "2026-08-07T12:00:23.900Z",
            [1786104022.2, 1786104023.8],
            "verify-1",
        ),
        (
            trial
            / "artifacts"
            / "continuous"
            / "attempts"
            / "0002-policy-2.pt"
            / "verifier"
            / "telemetry",
            "2026-08-07T12:00:25.100Z",
            "2026-08-07T12:00:26.900Z",
            [1786104025.2, 1786104026.8],
            "verify-2",
        ),
        (
            trial / "verifier" / "telemetry",
            "2026-08-07T12:00:30.100Z",
            "2026-08-07T12:00:39.900Z",
            [1786104032, 1786104038],
            "verify-final",
        ),
    ]
    for telemetry_dir, started, finished, epochs, container in verifier_windows:
        telemetry_dir.mkdir(parents=True, exist_ok=True)
        (telemetry_dir / "lifecycle.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "role": "verifier-gpu",
                    "started_at": started,
                    "finished_at": finished,
                    "sample_count": len(epochs),
                    "complete": True,
                }
            )
        )
        write_jsonl(
            telemetry_dir / "samples.jsonl",
            [
                {
                    "epoch_s": epoch,
                    "role": "verifier-gpu",
                    "container_id": container,
                    "sample_index": sample_index,
                    "gpus": [
                        {
                            "gpu_index": 0,
                            "util_gpu_pct": 40 + sample_index,
                            "mem_used_mib": 2700 + sample_index * 100,
                            "mem_total_mib": 23028,
                        }
                    ],
                }
                for sample_index, epoch in enumerate(epochs, 1)
            ],
        )
    (trial / "result.json").write_text(
        json.dumps(
            {
                "verifier": {
                    "started_at": "2026-08-07T12:00:30Z",
                    "finished_at": "2026-08-07T12:00:40Z",
                }
            }
        )
    )
    return state


def test_unified_timeline_is_joined_deduplicated_and_public_safe(
    tmp_path: Path,
) -> None:
    state = fixture_run(tmp_path)
    web = tmp_path / "web"
    payload = unified_timeline.build_timeline(state, web_dir=web, bucket_seconds=60)

    assert payload["coverage"]["ready"] is True
    assert payload["coverage"]["requirements"] == {
        "all_submitted_artifacts_captured": True,
        "cpu_agent_lifecycle": True,
        "cpu_agent_metrics": True,
        "model_usage_and_cost": True,
        "submission_ledger": True,
        "timestamped_agent_trace": True,
        "training_gpu_lifecycle": True,
        "training_gpu_metrics": True,
        "verifier_gpu_lifecycle": True,
        "verifier_gpu_metrics": True,
    }
    assert [event["epoch_ms"] for event in payload["events"]] == sorted(
        event["epoch_ms"] for event in payload["events"]
    )
    assert sum(event["kind"] == "resource_sample" for event in payload["events"]) == 11
    assert {
        event.get("role")
        for event in payload["events"]
        if event["kind"] == "resource_sample"
    } == {"cpu-agent", "training-gpu", "verifier-gpu"}
    assert {event["kind"] for event in payload["events"]} >= {
        "gpu_preempted",
        "gpu_reallocated",
        "artifact_submitted",
        "evaluation_retry",
        "tool_call",
    }
    starts = [
        event
        for event in payload["events"]
        if event["kind"] == "evaluation_started"
        and event.get("evaluation_scope") == "continuous"
    ]
    assert all(event["scoring_queue_key"] == "timeline-fixture" for event in starts)
    assert {event["queue_wait_ms"] for event in starts} == {1000, 3000}
    assert payload["run"]["scoring_queue_scope"] == "per_run_model"
    assert len(payload["artifacts"]) == 2
    assert all(item["captured"] and item["sha256"] for item in payload["artifacts"])
    assert payload["artifacts"][0]["verification_attempts"] == 2
    assert (
        payload["artifacts"][0]["verification_retry_events"][0]["error_type"]
        == "NotFoundError"
    )
    retry = next(
        event for event in payload["events"] if event["kind"] == "evaluation_retry"
    )
    assert retry["failed_attempt"] == 1
    assert retry["error_type"] == "NotFoundError"
    assert payload["tool_call_buckets"]["buckets"][0]["by_tool"] == {
        "exec": 1,
        "exec_command": 1,
        "view_image": 1,
        "write_stdin": 1,
    }
    exec_call = next(
        event
        for event in payload["events"]
        if event["kind"] == "tool_call" and event.get("tool") == "exec"
    )
    assert exec_call["duration_ms"] == 1000
    assert exec_call["finished_ts"] == "2026-08-07T12:00:14.000Z"
    assert payload["tool_timing_summary"]["exec"] == {
        "call_count": 1,
        "timed_call_count": 1,
        "total_duration_ms": 1000,
        "mean_duration_ms": 1000,
        "p50_duration_ms": 1000,
        "p95_duration_ms": 1000,
        "max_duration_ms": 1000,
    }
    assert payload["shell_session_summary"] == [
        {
            "cpu_attempt": 1,
            "shell_session_id": "23796",
            "started_epoch_ms": 1786104016000,
            "started_ts": "2026-08-07T12:00:16.000Z",
            "finished_epoch_ms": 1786104020000,
            "finished_ts": "2026-08-07T12:00:20.000Z",
            "duration_ms": 4000,
            "tool_call_count": 2,
            "timed_tool_call_ms": 3000,
            "exit_code": 0,
            "aborted": False,
            "complete": True,
        }
    ]
    assert payload["comparison_summary"]["best_100m_s"] == 48.0
    assert payload["comparison_summary"]["best_result_epoch_ms"] == 1786104027000
    assert payload["comparison_summary"]["time_to_best_ms"] == 27_000
    assert (
        payload["comparison_summary"]["modal_estimated_cost_at_best_usd"] == 0.022042568
    )
    assert payload["comparison_summary"]["valid_submission_count"] == 2
    assert payload["comparison_summary"]["tool_call_count"] == 4
    assert payload["resource_usage_summary"]["training_gpu"]["allocation_count"] == 2
    training_pipeline = payload["resource_usage_summary"]["gpu_pipeline"][
        "training_gpu"
    ]
    assert training_pipeline["collector"] == "cupti-pm-sampling"
    assert training_pipeline["metrics"]["sm_active_pct"] == {
        "sample_count": 2,
        "window_weighted_mean_pct": 70.0,
        "p50_pct": 60.0,
        "p95_pct": 80.0,
        "max_pct": 80.0,
    }
    resource_event = next(
        event
        for event in payload["events"]
        if event["kind"] == "resource_sample"
        and event.get("role") == "training-gpu"
        and (event.get("metrics") or {}).get("gpus", [{}])[0].get("sm_active_pct")
        == 60.0
    )
    assert resource_event["metrics"]["gpus"][0]["tensor_pipe_active_pct"] == 20.0
    assert (
        payload["resource_usage_summary"]["modal_estimate"]["estimated_cost_usd"]
        == 0.038489292
    )
    assert payload["artifacts"][1]["cost_at_submission"]["epoch_ms"] == 1786104022000
    assert payload["artifacts"][1]["cost_at_result"] == {
        "epoch_ms": 1786104027000,
        "api_calculated_usd": None,
        "modal_tariff_estimated_usd": 0.022042568,
        "agent_modal_tariff_estimated_usd": 0.0201332,
        "verifier_measurement_overhead_estimated_usd": 0.001909368,
        "total_estimated_usd": None,
        "modal_estimate_kind": "requested_resource_floor",
    }
    encoded = json.dumps(payload)
    assert "SUPER_SECRET_TOOL_ARGUMENT" not in encoded
    assert (state / "telemetry" / "unified-timeline.json").is_file()
    assert (web / "data" / "timelines" / "timeline-fixture.json").is_file()
    overview = json.loads(
        (web / "data" / "timeline-overviews" / "timeline-fixture.json").read_text()
    )
    assert overview["clock"] == payload["clock"]
    assert overview["tool_call_buckets"] == payload["tool_call_buckets"]
    assert overview["events"]
    assert {event["category"] for event in overview["events"]} <= {
        "metrics",
        "infrastructure",
    }
    assert all("source" not in event for event in overview["events"])
    index = json.loads((web / "data" / "timelines" / "index.json").read_text())
    assert index["runs"][0]["path"] == "/data/timelines/timeline-fixture.json"
    assert (
        index["runs"][0]["overview_path"]
        == "/data/timeline-overviews/timeline-fixture.json"
    )
    assert index["runs"][0]["comparison_summary"]["best_100m_s"] == 48.0
    assert index["runs"][0]["usage_summary"] == payload["usage_summary"]
    assert index["runs"][0]["dashboard_artifacts"] == [
        {
            "submission_index": artifact.get("submission_index"),
            "finished_epoch_ms": unified_timeline.parse_epoch_ms(
                artifact.get("finished_at")
            ),
            "rewards": {
                "valid_run": (artifact.get("rewards") or {}).get("valid_run"),
                "best_100m_s": (artifact.get("rewards") or {}).get("best_100m_s"),
                "gate_finished": (artifact.get("rewards") or {}).get("gate_finished"),
                "gate_in_lane": (artifact.get("rewards") or {}).get("gate_in_lane"),
                "gate_self_collision": (artifact.get("rewards") or {}).get(
                    "gate_self_collision"
                ),
                "peak_speed_mps": (artifact.get("rewards") or {}).get("peak_speed_mps"),
            },
            "cost_at_result": artifact.get("cost_at_result"),
        }
        for artifact in payload["artifacts"]
    ]


def test_public_timeline_index_keeps_only_configured_newest_runs(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(unified_timeline, "PUBLIC_RUN_LIMIT", 6)
    web = tmp_path / "web"
    index_path = web / "data" / "timelines" / "index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text(
        json.dumps(
            {
                "schema_version": unified_timeline.SCHEMA_VERSION,
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
    state = fixture_run(tmp_path)
    unified_timeline.build_timeline(state, web_dir=web)
    index = json.loads(index_path.read_text())
    assert len(index["runs"]) == unified_timeline.PUBLIC_RUN_LIMIT
    assert [row["run_id"] for row in index["runs"]] == [
        "timeline-fixture",
        "old-7",
        "old-6",
        "old-5",
        "old-4",
        "old-3",
    ]


def test_timeline_preserves_generated_clock_without_material_changes(
    tmp_path: Path,
) -> None:
    web = tmp_path / "web"
    state = fixture_run(tmp_path)
    unified_timeline.build_timeline(state, web_dir=web)
    internal = state / "telemetry" / "unified-timeline.json"
    previous = json.loads(internal.read_text())
    previous["generated_at"] = "2000-01-01T00:00:00Z"
    unified_timeline.atomic_json(internal, previous)

    payload = unified_timeline.build_timeline(state, web_dir=web)

    public = json.loads(
        (web / "data" / "timelines" / "timeline-fixture.json").read_text()
    )
    index = json.loads((web / "data" / "timelines" / "index.json").read_text())
    assert payload["generated_at"] == "2000-01-01T00:00:00Z"
    assert public["generated_at"] == "2000-01-01T00:00:00Z"
    assert index["runs"][0]["generated_at"] == "2000-01-01T00:00:00Z"


def test_durable_gpu_lifecycle_copy_is_deduplicated_by_event_id(
    tmp_path: Path,
) -> None:
    state = fixture_run(tmp_path)
    source = state / "telemetry" / "gpu_timeline.jsonl"
    (state / "telemetry" / "durable-gpu-timeline.jsonl").write_text(source.read_text())
    payload = unified_timeline.build_timeline(state)
    lifecycle = {
        "gpu_allocated",
        "gpu_preempted",
        "gpu_reallocated",
        "gpu_released",
    }
    assert sum(event["kind"] in lifecycle for event in payload["events"]) == 4
    assert payload["resource_usage_summary"]["training_gpu"]["allocation_count"] == 2


def test_training_cost_excludes_archive_pin_before_sandbox_create(
    tmp_path: Path,
) -> None:
    state = fixture_run(tmp_path)
    lifecycle_path = state / "telemetry" / "gpu_timeline.jsonl"
    lifecycle = [json.loads(line) for line in lifecycle_path.read_text().splitlines()]
    lifecycle.extend(
        [
            {
                "event_id": "billing-start-1",
                "epoch_s": 1786104002,
                "phase": "gpu_worker_starting",
                "action": "enter",
                "job_id": "job-1",
                "attempt": 1,
                "lease_id": "lease-1",
                "detail": {"source": "dispatch"},
            },
            {
                "event_id": "billing-start-2",
                "epoch_s": 1786104021,
                "phase": "gpu_worker_starting",
                "action": "enter",
                "job_id": "job-1",
                "attempt": 2,
                "lease_id": "lease-2",
                "detail": {"source": "dispatch"},
            },
            {
                "event_id": "sandbox-create-1",
                "epoch_s": 1786104004,
                "phase": "gpu_sandbox_create",
                "action": "enter",
                "job_id": "job-1",
                "attempt": 1,
                "lease_id": "lease-1",
                "detail": {"source": "dispatch"},
            },
            {
                "event_id": "sandbox-create-2",
                "epoch_s": 1786104021,
                "phase": "gpu_sandbox_create",
                "action": "enter",
                "job_id": "job-1",
                "attempt": 2,
                "lease_id": "lease-2",
                "detail": {"source": "dispatch"},
            },
        ]
    )
    write_jsonl(lifecycle_path, lifecycle)

    payload = unified_timeline.build_timeline(state)
    training = payload["resource_usage_summary"]["training_gpu"]

    assert training["allocated_ms"] == 33_000
    # 35 seconds starts immediately before Sandbox.create. The earlier
    # gpu_worker_starting events include two seconds of host archive pinning
    # and must not inflate the live Modal estimate.
    assert training["billing_upper_bound_allocated_ms"] == 35_000
    assert (
        payload["resource_usage_summary"]["modal_estimate"]["by_role"]["training_gpu"][
            "allocated_ms"
        ]
        == 35_000
    )


def test_training_billing_uses_provider_exit_observation_during_volume_lag(
    tmp_path: Path,
) -> None:
    state = fixture_run(tmp_path)
    lifecycle_path = state / "telemetry" / "gpu_timeline.jsonl"
    lifecycle = [json.loads(line) for line in lifecycle_path.read_text().splitlines()]
    lifecycle.extend(
        [
            {
                "event_id": "sandbox-create-1",
                "epoch_s": 1786104004,
                "phase": "gpu_sandbox_create",
                "action": "enter",
                "job_id": "job-1",
                "attempt": 1,
                "lease_id": "lease-1",
            },
            {
                "event_id": "provider-exit-1",
                "epoch_s": 1786104023,
                "phase": "gpu_lifecycle",
                "action": "instant",
                "job_id": "job-1",
                "attempt": 1,
                "lease_id": "lease-1",
                "detail": {"event": "gpu_provider_exit_observed"},
            },
        ]
    )
    write_jsonl(lifecycle_path, lifecycle)

    payload = unified_timeline.build_timeline(state)
    training = payload["resource_usage_summary"]["training_gpu"]
    attempt_one = next(
        interval
        for interval in training["billing_upper_bound_intervals"]
        if interval.get("gpu_attempt") == 1
    )

    # The worker's exact terminal record is 12:00:20, while Modal was observed
    # exited at 12:00:23. Billing uses the later provider observation so an
    # in-flight estimate can reconcile by at most the poll interval, not the
    # full terminal Volume visibility grace.
    assert attempt_one["start_epoch_ms"] == 1786104004000
    assert attempt_one["end_epoch_ms"] == 1786104023000
    assert training["intervals"][0]["end_epoch_ms"] == 1786104020000


def test_stop_ack_closes_single_cpu_allocation(
    tmp_path: Path,
) -> None:
    state = fixture_run(tmp_path)
    run_path = state / "run.json"
    run = json.loads(run_path.read_text())
    run.update(
        {
            "cpu_execution_policy": "single_process_no_resume",
        }
    )
    run_path.write_text(json.dumps(run))
    telemetry_path = state / "telemetry" / "host-samples.jsonl"
    rows = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    for row in rows:
        if row.get("role") == "cpu-agent":
            row["cpu_attempt"] = 1
    write_jsonl(telemetry_path, rows)
    (state / "STOP_ACK.json").write_text(
        json.dumps({"acknowledged_at": "2026-08-07T12:00:20Z"})
    )
    payload = unified_timeline.build_timeline(state)
    ack = next(
        event for event in payload["events"] if event["kind"] == "stop_acknowledged"
    )
    assert ack["reason"] == ""
    cpu = payload["coverage"]["cpu_metric_coverage"]["attempts"]
    assert cpu == [
        {
            "start_epoch_ms": 1786104001000,
            "end_epoch_ms": 1786104020000,
            "cpu_attempt": 1,
            "sample_count": 1,
            "max_gap_ms": 10000,
            "covered": True,
        }
    ]
    assert payload["coverage"]["requirements"]["cpu_agent_lifecycle"] is True
    assert payload["coverage"]["requirements"]["cpu_agent_metrics"] is True


def test_post_run_accounting_does_not_stretch_activity_clock(tmp_path: Path) -> None:
    state = fixture_run(tmp_path)
    (state / "STOP_ACK.json").write_text(
        json.dumps({"acknowledged_at": "2026-08-07T12:00:20Z"})
    )
    write_jsonl(
        state / "telemetry" / "cpu_lifecycle.jsonl",
        [
            {
                "event": "cpu_launch_started",
                "attempt": 1,
                "at": "2026-08-07T12:00:01Z",
            },
            {
                "event": "cpu_launch_exited",
                "attempt": 1,
                "at": "2026-08-07T14:00:00Z",
                "exit_code": 0,
            },
        ],
    )
    audit = {
        "session_id": "late-accounting",
        "request_count": 1,
        "cost_reconstruction_complete": True,
        "calculated_api_usage_usd": 0.25,
        "requests": [
            {
                "api_call_id": "late-api-call",
                "usage_reported_at": "2026-08-07T14:00:00Z",
                "model": "test-model",
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "calculated_cost_usd": 0.25,
                "cost_reconstruction_status": "complete",
            }
        ],
    }
    path = next(state.glob("harbor-jobs/*/*/agent")) / "usage-audit.json"
    path.write_text(json.dumps(audit))

    payload = unified_timeline.build_timeline(state)

    assert payload["clock"]["end_epoch_ms"] == 1786104020000
    assert payload["clock"]["activity_end_basis"] == "stop_acknowledged"
    assert payload["clock"]["observer_end_epoch_ms"] == 1786111200000
    assert payload["clock"]["post_run_event_count"] >= 1
    late = next(
        event for event in payload["events"] if event["kind"] == "model_request_usage"
    )
    assert late["post_run"] is True


def test_missing_submitted_artifact_blocks_readiness(tmp_path: Path) -> None:
    payload = unified_timeline.build_timeline(
        fixture_run(tmp_path, missing_artifact=True)
    )
    assert payload["coverage"]["ready"] is False
    assert (
        payload["coverage"]["requirements"]["all_submitted_artifacts_captured"] is False
    )
    assert payload["coverage"]["counts"]["missing_submission_artifacts"] == 1


def test_failed_submission_ingestion_is_preserved_without_claiming_policy_bytes(
    tmp_path: Path,
) -> None:
    state = fixture_run(tmp_path, missing_artifact=True)
    ledger = next(state.rglob("continuous/ledger.jsonl"))
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    rows[1].update(
        {
            "artifact_sha256": None,
            "error": "NotFoundError: sandbox unavailable",
            "error_type": "NotFoundError",
            "retryable_infrastructure_error": True,
        }
    )
    write_jsonl(ledger, rows)

    payload = unified_timeline.build_timeline(state)

    assert (
        payload["coverage"]["requirements"]["all_submitted_artifacts_captured"] is True
    )
    assert payload["coverage"]["counts"]["failed_submission_ingestions"] == 1
    failed = [
        event
        for event in payload["events"]
        if event["kind"] == "artifact_ingestion_failed"
    ]
    assert len(failed) == 1
    artifact = next(item for item in payload["artifacts"] if item["ingestion_failed"])
    assert artifact["captured"] is False


def test_request_costs_are_joined_to_performance_on_the_same_clock(
    tmp_path: Path,
) -> None:
    state = fixture_run(tmp_path)
    run_path = state / "run.json"
    run = json.loads(run_path.read_text())
    run["usage_audit_required"] = True
    run_path.write_text(json.dumps(run))
    audit = {
        "session_id": "session-cost",
        "request_count": 1,
        "cost_reconstruction_complete": True,
        "calculated_api_usage_usd": 0.25,
        "calculated_api_usage_cost_basis": "published_standard_list_price",
        "pricing_snapshots": [{"id": "price-v1"}],
        "requests": [
            {
                "api_call_id": "api_call_1",
                "usage_reported_at": "2026-08-07T12:00:18Z",
                "model": "test-model",
                "reasoning_effort": "high",
                "input_tokens": 100,
                "cached_input_tokens": 50,
                "cache_write_input_tokens": 0,
                "output_tokens": 20,
                "reasoning_output_tokens": 10,
                "total_tokens": 120,
                "pricing_snapshot_id": "price-v1",
                "calculated_cost_usd": 0.25,
                "cost_reconstruction_status": "complete",
            }
        ],
    }
    path = next(state.glob("harbor-jobs/*/*/agent")) / "usage-audit.json"
    path.write_text(json.dumps(audit))

    payload = unified_timeline.build_timeline(state)

    assert payload["coverage"]["requirements"]["model_usage_and_cost"] is True
    assert payload["usage_summary"]["request_count"] == 1
    assert payload["usage_summary"]["calculated_api_usage_usd"] == 0.25
    assert payload["usage_summary"]["calculated_api_usage_cost_basis"] == [
        "published_standard_list_price"
    ]
    assert payload["comparison_summary"]["final_api_cost_basis"] == [
        "published_standard_list_price"
    ]
    assert payload["comparison_summary"]["api_cost_at_best_usd"] == 0.25
    assert payload["artifacts"][1]["cost_at_result"]["api_calculated_usd"] == 0.25
    assert payload["artifacts"][1]["cost_at_result"]["total_estimated_usd"] == 0.2701332
    assert (
        payload["artifacts"][1]["cost_at_result"][
            "verifier_measurement_overhead_estimated_usd"
        ]
        == 0.001909368
    )
    request = next(
        event for event in payload["events"] if event["kind"] == "model_request_usage"
    )
    assert request["elapsed_ms"] == 18_000


def test_attested_zero_request_failure_has_complete_zero_cost_timeline(
    tmp_path: Path,
) -> None:
    state = fixture_run(tmp_path)
    run_path = state / "run.json"
    run = json.loads(run_path.read_text())
    run["usage_audit_required"] = True
    run_path.write_text(json.dumps(run))
    audit_path = state / "usage" / "run-usage-audit.json"
    audit_path.parent.mkdir()
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "timeline-fixture",
                "request_count": 0,
                "requests": [],
                "pricing_snapshots": [],
                "cost_reconstruction_complete": True,
                "calculated_api_usage_usd": 0.0,
                "zero_request_reason": (
                    "no completed model request was present in any captured CPU attempt"
                ),
            }
        )
    )

    payload = unified_timeline.build_timeline(state)

    assert payload["coverage"]["requirements"]["model_usage_and_cost"] is True
    assert payload["coverage"]["counts"]["complete_usage_audits"] == 1
    assert payload["coverage"]["counts"]["attested_zero_request_usage_audits"] == 1
    assert payload["usage_summary"]["request_count"] == 0
    assert payload["usage_summary"]["calculated_api_usage_usd"] == 0.0


def test_openrouter_chat_ledger_is_generic_usage_source(tmp_path: Path) -> None:
    state = fixture_run(tmp_path)
    run_path = state / "run.json"
    run = json.loads(run_path.read_text())
    run.update(
        {
            "agent_kind": "deepseek-harness",
            "model": "deepseek/deepseek-v4-flash-vision-exp",
            "reasoning_effort": "max",
            "provider_usage_ledger_required": True,
            "usage_audit_required": False,
        }
    )
    run_path.write_text(json.dumps(run))
    ledger = state / "provider-api-usage" / "api-usage"
    (ledger / "requests").mkdir(parents=True)
    (ledger / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "run_id": "timeline-fixture",
                "model_api_usd": 0.025,
                "provider_billed_model_api_usd": 0.02,
                "completed_request_count": 1,
                "pending_request_count": 0,
                "in_flight_request_count": 0,
                "cost_recovery_required_count": 0,
                "token_usage": {
                    "input_tokens": 1_000,
                    "ordinary_uncached_input_tokens": 200,
                    "cached_input_tokens": 800,
                    "cache_write_input_tokens": 0,
                    "output_tokens": 100,
                    "reasoning_output_tokens": 60,
                    "total_tokens": 1_100,
                },
            }
        )
    )
    (ledger / "requests" / "abc.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "ledger_request_id": "a" * 32,
                "run_id": "timeline-fixture",
                "cpu_attempt": 1,
                "requested_at": "2026-08-07T12:00:16Z",
                "completed_at": "2026-08-07T12:00:18Z",
                "requested_model": "deepseek/deepseek-v4-flash-vision-exp",
                "response_model": "deepseek/deepseek-v4-flash-vision-exp",
                "state": "complete",
                "provider_reported_cost_usd": 0.02,
                "benchmark_cost_usd": 0.025,
                "cost_basis": "openrouter_list_price_with_deepseek_peak_floor",
                "provider_cost_basis": "openrouter_reported_per_request",
                "promotion_discount_fraction": 0.2,
                "usage": {
                    "prompt_tokens": 1_000,
                    "prompt_tokens_details": {"cached_tokens": 800},
                    "completion_tokens": 100,
                    "completion_tokens_details": {"reasoning_tokens": 60},
                    "total_tokens": 1_100,
                },
            }
        )
    )

    payload = unified_timeline.build_timeline(state)

    assert payload["coverage"]["requirements"]["model_usage_and_cost"] is True
    assert payload["usage_summary"]["request_count"] == 1
    assert payload["usage_summary"]["input_tokens"] == 1_000
    assert payload["usage_summary"]["ordinary_uncached_input_tokens"] == 200
    assert payload["usage_summary"]["cached_input_tokens"] == 800
    assert payload["usage_summary"]["output_tokens"] == 100
    assert payload["usage_summary"]["reasoning_output_tokens"] == 60
    assert payload["usage_summary"]["total_tokens"] == 1_100
    assert payload["usage_summary"]["calculated_api_usage_usd"] == 0.025
    assert payload["usage_summary"]["provider_billed_api_usage_usd"] == 0.02
    request = next(
        event for event in payload["events"] if event["kind"] == "model_request_usage"
    )
    assert request["elapsed_ms"] == 18_000


def test_live_openrouter_summary_exposes_cumulative_tokens_without_request_scan(
    tmp_path: Path,
) -> None:
    state = fixture_run(tmp_path)
    run_path = state / "run.json"
    run = json.loads(run_path.read_text())
    run.update(
        {
            "agent_kind": "deepseek-harness",
            "model": "deepseek/deepseek-v4-flash-vision-exp",
            "reasoning_effort": "max",
            "provider_usage_ledger_required": True,
            "usage_audit_required": False,
        }
    )
    run_path.write_text(json.dumps(run))
    ledger = state / "provider-api-usage" / "api-usage"
    ledger.mkdir(parents=True)
    (ledger / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "run_id": "timeline-fixture",
                "updated_at": "2026-08-07T12:00:18Z",
                "model_api_usd": 0.25,
                "provider_billed_model_api_usd": 0.20,
                "model_api_cost_basis": "benchmark-list-price",
                "provider_billed_cost_basis": "openrouter-reported",
                "completed_request_count": 7,
                "pending_request_count": 0,
                "in_flight_request_count": 0,
                "cost_recovery_required_count": 0,
                "in_flight_request_ids": [],
                "cost_recovery_required_request_ids": [],
                "token_usage": {
                    "input_tokens": 10_000,
                    "ordinary_uncached_input_tokens": 1_500,
                    "cached_input_tokens": 8_000,
                    "cache_write_input_tokens": 500,
                    "output_tokens": 900,
                    "reasoning_output_tokens": 600,
                    "total_tokens": 10_900,
                },
            }
        )
    )

    payload = unified_timeline.build_timeline(state)

    assert payload["coverage"]["requirements"]["model_usage_and_cost"] is True
    assert payload["usage_summary"]["request_count"] == 7
    assert payload["usage_summary"]["input_tokens"] == 10_000
    assert payload["usage_summary"]["cached_input_tokens"] == 8_000
    assert payload["usage_summary"]["cache_write_input_tokens"] == 500
    assert payload["usage_summary"]["output_tokens"] == 900
    assert payload["usage_summary"]["reasoning_output_tokens"] == 600
    assert payload["usage_summary"]["calculated_api_usage_usd"] == 0.25
    assert payload["usage_summary"]["provider_billed_api_usage_usd"] == 0.20
    events = [
        event for event in payload["events"] if event["kind"] == "model_request_usage"
    ]
    assert len(events) == 1
    assert events[0]["cumulative_summary"] is True


def test_codex_usage_still_requires_settled_provider_summary(tmp_path: Path) -> None:
    state = fixture_run(tmp_path)
    run_path = state / "run.json"
    run = json.loads(run_path.read_text())
    run["provider_usage_ledger_required"] = True
    run["usage_audit_required"] = True
    run_path.write_text(json.dumps(run))
    audit_path = next(state.glob("harbor-jobs/*/*/agent")) / "usage-audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "session_id": "session-cost",
                "request_count": 1,
                "cost_reconstruction_complete": True,
                "calculated_api_usage_usd": 0.25,
                "calculated_api_usage_cost_basis": "test",
                "pricing_snapshots": [{"id": "price-v1"}],
                "requests": [
                    {
                        "api_call_id": "api-call",
                        "usage_reported_at": "2026-08-07T12:00:18Z",
                        "model": "test-model",
                        "reasoning_effort": "high",
                        "input_tokens": 100,
                        "cached_input_tokens": 50,
                        "output_tokens": 20,
                        "total_tokens": 120,
                        "pricing_snapshot_id": "price-v1",
                        "calculated_cost_usd": 0.25,
                        "cost_reconstruction_status": "complete",
                    }
                ],
            }
        )
    )
    ledger = state / "provider-api-usage" / "api-usage"
    ledger.mkdir(parents=True)
    (ledger / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "run_id": "timeline-fixture",
                "pending_request_count": 1,
                "in_flight_request_count": 1,
                "cost_recovery_required_count": 0,
                "token_usage": empty_token_usage(),
            }
        )
    )

    payload = unified_timeline.build_timeline(state)

    assert payload["usage_summary"]["request_count"] > 0
    assert payload["coverage"]["counts"]["incomplete_provider_usage_summaries"] == 1
    assert payload["coverage"]["requirements"]["model_usage_and_cost"] is False
    assert payload["coverage"]["ready"] is False


def test_openrouter_chat_ledger_rejects_non_numeric_tokens(tmp_path: Path) -> None:
    state = fixture_run(tmp_path)
    run_path = state / "run.json"
    run = json.loads(run_path.read_text())
    run.update(
        {
            "agent_kind": "deepseek-harness",
            "provider_usage_ledger_required": True,
            "usage_audit_required": False,
        }
    )
    run_path.write_text(json.dumps(run))
    ledger = state / "provider-api-usage" / "api-usage"
    (ledger / "requests").mkdir(parents=True)
    (ledger / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "timeline-fixture",
                "pending_request_count": 0,
                "cost_recovery_required_count": 0,
            }
        )
    )
    (ledger / "requests" / "bad.json").write_text(
        json.dumps(
            {
                "ledger_request_id": "b" * 32,
                "run_id": "timeline-fixture",
                "state": "complete",
                "provider_reported_cost_usd": 0.02,
                "benchmark_cost_usd": 0.025,
                "usage": {
                    "prompt_tokens": "not-a-number",
                    "completion_tokens": 10,
                },
            }
        )
    )

    payload = unified_timeline.build_timeline(state)

    assert payload["coverage"]["requirements"]["model_usage_and_cost"] is False
    assert payload["coverage"]["counts"]["malformed_provider_usage_records"] == 1
    assert payload["usage_summary"]["request_count"] == 0


def test_all_submission_result_set_has_no_privileged_primary(
    tmp_path: Path,
) -> None:
    state = fixture_run(tmp_path)

    payload = unified_timeline.build_timeline(state)

    assert payload["comparison_summary"]["best_100m_s"] == 48.0
    assert payload["comparison_summary"]["evaluation_result_policy"] == (
        "all_blind_archival_submissions"
    )
    assert "primary_score_policy" not in payload["comparison_summary"]
    assert "primary_final_100m_s" not in payload["comparison_summary"]
    assert all("primary_final" not in item for item in payload["artifacts"])


def test_provider_billing_is_selected_and_error_text_is_not_published(
    tmp_path: Path,
) -> None:
    state = fixture_run(tmp_path)
    modal_payload = {
        "schema_version": 1,
        "run_id": "timeline-fixture",
        "provider_complete": True,
        "status": "complete",
        "billing_basis": "provider_report_precredits",
        "provider_cost_precredits_usd": 1.25,
        "by_role_usd": {"cpu_agent": 0.5, "training_gpu": 0.75},
        "by_category_usd": {"CPU": 0.4, "Memory": 0.1, "A10G": 0.75},
        "by_role_category_usd": {
            "cpu_agent": {"CPU": 0.4, "Memory": 0.1},
            "training_gpu": {"A10G": 0.75},
        },
        "error": "PRIVATE_PROVIDER_DIAGNOSTIC",
    }
    path = state / "telemetry" / "modal-cost.json"
    path.write_text(json.dumps(modal_payload))

    payload = unified_timeline.build_timeline(state)

    summary = payload["resource_usage_summary"]
    assert summary["usd_cost"] == 1.25
    assert summary["usd_cost_kind"] == "provider_report_precredits"
    assert "PRIVATE_PROVIDER_DIAGNOSTIC" not in json.dumps(payload)


def test_required_cgroup_scope_fails_closed_on_host_wide_metrics(
    tmp_path: Path,
) -> None:
    state = fixture_run(tmp_path)
    run_path = state / "run.json"
    run = json.loads(run_path.read_text())
    run["cgroup_telemetry_required"] = True
    run_path.write_text(json.dumps(run))

    payload = unified_timeline.build_timeline(state)

    assert payload["coverage"]["requirements"]["cgroup_scoped_cpu_memory"] is False
    assert payload["coverage"]["ready"] is False
    assert any("host-wide fallback" in item for item in payload["coverage"]["warnings"])

    for path in state.rglob("*samples.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        for row in rows:
            row["resource_accounting_scope"] = "cgroup-v1"
        write_jsonl(path, rows)
    fixed = unified_timeline.build_timeline(state)
    assert fixed["coverage"]["requirements"]["cgroup_scoped_cpu_memory"] is True
    assert fixed["coverage"]["ready"] is True


def test_scoring_ledger_is_a_gpu_lifecycle_source_without_training(
    tmp_path: Path,
) -> None:
    payload = unified_timeline.build_timeline(
        fixture_run(tmp_path, training_lifecycle=False)
    )
    coverage = payload["coverage"]
    assert coverage["sources"]["gpu_lifecycle_files"] == 0
    assert coverage["counts"]["kind:evaluation_started"] == 3
    assert coverage["counts"]["kind:evaluation_finished"] == 3
    assert coverage["requirements"]["training_gpu_lifecycle"] is True
    assert coverage["requirements"]["training_gpu_metrics"] is True
    assert coverage["requirements"]["verifier_gpu_lifecycle"] is True
    assert coverage["requirements"]["verifier_gpu_metrics"] is True
    assert coverage["ready"] is True


def test_host_registry_closes_missing_training_terminal_event(tmp_path: Path) -> None:
    state = fixture_run(tmp_path)
    lifecycle_path = state / "telemetry" / "gpu_timeline.jsonl"
    rows = [
        row
        for row in (
            json.loads(line) for line in lifecycle_path.read_text().splitlines()
        )
        if row.get("event_id") != "d"
    ]
    write_jsonl(lifecycle_path, rows)
    registry = state / "gpu-job-registry" / "job-1.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            {
                "job_id": "job-1",
                "attempt": 2,
                "status": "terminated",
                "terminated_at": "2026-08-07T12:00:40Z",
                "termination_reason": "operator_stop",
            }
        )
    )

    payload = unified_timeline.build_timeline(state)

    recovered = [
        event
        for event in payload["events"]
        if event.get("gpu_job_id") == "job-1"
        and event.get("gpu_attempt") == 2
        and event["kind"] == "gpu_released"
    ]
    assert len(recovered) == 1
    assert recovered[0]["lifecycle_recovered"] is True
    assert recovered[0]["lifecycle_recovery_source"] == "host_job_registry"
    assert payload["coverage"]["requirements"]["training_gpu_lifecycle"] is True
    assert payload["coverage"]["requirements"]["training_gpu_metrics"] is True


def test_host_registry_recovers_provider_exit_billing_boundary(tmp_path: Path) -> None:
    state = fixture_run(tmp_path)
    lifecycle_path = state / "telemetry" / "gpu_timeline.jsonl"
    lifecycle = [json.loads(line) for line in lifecycle_path.read_text().splitlines()]
    lifecycle.append(
        {
            "event_id": "sandbox-create-1",
            "epoch_s": 1786104004,
            "phase": "gpu_sandbox_create",
            "action": "enter",
            "job_id": "job-1",
            "attempt": 1,
            "lease_id": "lease-1",
        }
    )
    write_jsonl(lifecycle_path, lifecycle)
    registry = state / "gpu-job-registry" / "job-1.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            {
                "job_id": "job-1",
                "attempt": 1,
                "status": "running",
                "lease_id": "lease-1",
                "provider_exit_observed_epoch_s": 1786104023,
                "provider_exit_code": 0,
            }
        )
    )

    payload = unified_timeline.build_timeline(state)
    attempt_one = next(
        interval
        for interval in payload["resource_usage_summary"]["training_gpu"][
            "billing_upper_bound_intervals"
        ]
        if interval.get("gpu_attempt") == 1
    )

    assert attempt_one["end_epoch_ms"] == 1786104023000
    assert payload["coverage"]["counts"]["gpu_registry_provider_exit_events"] == 1


def test_host_registry_clamps_pre_spawn_training_interval(
    tmp_path: Path,
) -> None:
    state = fixture_run(tmp_path)
    lifecycle_path = state / "telemetry" / "gpu_timeline.jsonl"
    rows = [
        {
            "event_id": "lifecycle-start",
            "epoch_s": 1786104000,
            "phase": "gpu_lifecycle",
            "action": "instant",
            "job_id": "short-lived",
            "attempt": 1,
            "detail": {"event": "gpu_allocated"},
        },
        {
            "event_id": "lifecycle-end",
            "epoch_s": 1786104101,
            "phase": "gpu_lifecycle",
            "action": "instant",
            "job_id": "short-lived",
            "attempt": 1,
            "detail": {"event": "gpu_released"},
        },
    ]
    write_jsonl(lifecycle_path, rows)
    registry = state / "gpu-job-registry" / "short-lived.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            {
                "job_id": "short-lived",
                "attempt": 1,
                "status": "terminated",
                "dispatched_at_epoch_s": 1786104045,
                "terminated_at_epoch_s": 1786104057,
                "termination_reason": "operator_stop",
            }
        )
    )

    payload = unified_timeline.build_timeline(state)

    coverage = payload["coverage"]["gpu_metric_coverage"]["training"]
    assert coverage == [
        {
            "start_epoch_ms": 1786104045000,
            "end_epoch_ms": 1786104057000,
            "gpu_job_id": "short-lived",
            "gpu_attempt": 1,
            "raw_start_epoch_ms": 1786104000000,
            "raw_end_epoch_ms": 1786104101000,
            "lifecycle_bounds_source": "host_job_registry",
            "telemetry_expected": False,
            "telemetry_not_expected_reason": "sandbox_terminated_before_worker_start",
            "sample_count": 0,
            "max_gap_ms": 12000,
            "covered": True,
            "leading_gap_ms": None,
            "internal_max_gap_ms": None,
            "trailing_gap_ms": None,
            "terminal_tail_grace_ms": 180000,
            "terminal_tail_grace_used": False,
            "coverage_status": "not_applicable",
            "coverage_reason": "sandbox_terminated_before_worker_start",
        }
    ]
    counts = payload["coverage"]["counts"]
    assert counts["gpu_registry_start_bounds_applied"] == 1
    assert counts["gpu_registry_end_bounds_applied"] == 1
    assert payload["coverage"]["requirements"]["training_gpu_metrics"] is True


def test_host_registry_clamps_telemetry_to_worker_start(tmp_path: Path) -> None:
    state = fixture_run(tmp_path)
    lifecycle = state / "telemetry" / "gpu_timeline.jsonl"
    rows = [json.loads(line) for line in lifecycle.read_text().splitlines()]
    rows.append(
        {
            "event_id": "sandbox-create-job-1",
            "epoch_s": 1786104004,
            "phase": "gpu_sandbox_create",
            "action": "enter",
            "job_id": "job-1",
            "attempt": 1,
        }
    )
    write_jsonl(lifecycle, rows)
    registry = state / "gpu-job-registry" / "job-1.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            {
                "job_id": "job-1",
                "attempt": 1,
                "status": "succeeded",
                "dispatched_at_epoch_s": 1786104006,
                "started_at_epoch_s": 1786104010,
                "finished_at_epoch_s": 1786104020,
            }
        )
    )

    payload = unified_timeline.build_timeline(state)

    attempt_one = next(
        item
        for item in payload["coverage"]["gpu_metric_coverage"]["training"]
        if item.get("gpu_job_id") == "job-1" and item.get("gpu_attempt") == 1
    )
    assert attempt_one["raw_start_epoch_ms"] == 1786104005000
    assert attempt_one["start_epoch_ms"] == 1786104010000
    assert attempt_one["leading_gap_ms"] == 1000
    assert attempt_one["covered"] is True
    assert (
        payload["resource_usage_summary"]["training_gpu"][
            "billing_upper_bound_intervals"
        ][0]["start_epoch_ms"]
        == 1786104004000
    )


def test_preworker_terminated_allocation_does_not_require_impossible_gpu_samples(
    tmp_path: Path,
) -> None:
    state = fixture_run(tmp_path)
    run_path = state / "run.json"
    run = json.loads(run_path.read_text())
    run["gpu_pipeline_telemetry_required"] = True
    run_path.write_text(json.dumps(run))
    lifecycle_path = state / "telemetry" / "gpu_timeline.jsonl"
    write_jsonl(
        lifecycle_path,
        [
            {
                "event_id": "preworker-start",
                "epoch_s": 1786104000,
                "phase": "gpu_lifecycle",
                "action": "instant",
                "job_id": "preworker-stop",
                "attempt": 1,
                "detail": {"event": "gpu_allocated"},
            },
            {
                "event_id": "preworker-end",
                "epoch_s": 1786104200,
                "phase": "gpu_lifecycle",
                "action": "instant",
                "job_id": "preworker-stop",
                "attempt": 1,
                "detail": {"event": "gpu_released"},
            },
        ],
    )
    registry = state / "gpu-job-registry" / "preworker-stop.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            {
                "job_id": "preworker-stop",
                "attempt": 1,
                "status": "terminated",
                "dispatched_at_epoch_s": 1786104005,
                "terminated_at_epoch_s": 1786104195,
                "termination_reason": "operator_stop",
            }
        )
    )

    payload = unified_timeline.build_timeline(state)

    coverage = payload["coverage"]["gpu_metric_coverage"]["training"]
    assert coverage == [
        {
            "start_epoch_ms": 1786104005000,
            "end_epoch_ms": 1786104195000,
            "gpu_job_id": "preworker-stop",
            "gpu_attempt": 1,
            "raw_start_epoch_ms": 1786104000000,
            "raw_end_epoch_ms": 1786104200000,
            "lifecycle_bounds_source": "host_job_registry",
            "telemetry_expected": False,
            "telemetry_not_expected_reason": "sandbox_terminated_before_worker_start",
            "sample_count": 0,
            "max_gap_ms": 190000,
            "covered": True,
            "leading_gap_ms": None,
            "internal_max_gap_ms": None,
            "trailing_gap_ms": None,
            "terminal_tail_grace_ms": 180000,
            "terminal_tail_grace_used": False,
            "coverage_status": "not_applicable",
            "coverage_reason": "sandbox_terminated_before_worker_start",
        }
    ]
    assert payload["coverage"]["requirements"]["training_gpu_metrics"] is True
    assert payload["coverage"]["requirements"]["training_gpu_pipeline_metrics"] is True
    assert (
        payload["coverage"]["counts"]["training_gpu_preworker_terminated_intervals"]
        == 1
    )


def test_host_registry_drops_allocation_published_after_terminal_attempt(
    tmp_path: Path,
) -> None:
    state = fixture_run(tmp_path)
    lifecycle_path = state / "telemetry" / "gpu_timeline.jsonl"
    write_jsonl(
        lifecycle_path,
        [
            {
                "event_id": "terminal-before-controller-return",
                "epoch_s": 1786104057,
                "phase": "gpu_lifecycle",
                "action": "instant",
                "job_id": "short-race",
                "attempt": 2,
                "lease_id": "lease-race",
                "detail": {"event": "gpu_released"},
            },
            {
                "event_id": "late-controller-allocation",
                "epoch_s": 1786104060,
                "phase": "gpu_lifecycle",
                "action": "instant",
                "job_id": "short-race",
                "attempt": 2,
                "lease_id": "lease-race",
                "detail": {"event": "gpu_reallocated"},
            },
        ],
    )
    registry = state / "gpu-job-registry" / "short-race.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            {
                "job_id": "short-race",
                "attempt": 2,
                "status": "failed",
                "started_at_epoch_s": 1786104056,
                "finished_at_epoch_s": 1786104057,
            }
        )
    )

    payload = unified_timeline.build_timeline(state)

    intervals = payload["resource_usage_summary"]["training_gpu"]["intervals"]
    assert not any(item.get("gpu_job_id") == "short-race" for item in intervals)
    assert (
        payload["coverage"]["counts"]["gpu_registry_post_terminal_starts_dropped"] == 1
    )


def test_training_gpu_samples_do_not_satisfy_verifier_coverage(tmp_path: Path) -> None:
    state = fixture_run(tmp_path)
    for path in state.rglob("verifier/telemetry/samples.jsonl"):
        path.unlink()
    coverage = unified_timeline.build_timeline(state)["coverage"]
    assert coverage["requirements"]["training_gpu_metrics"] is True
    assert coverage["requirements"]["verifier_gpu_metrics"] is False
    assert coverage["counts"]["gpu_metric_role:training-gpu"] == 4
    assert coverage["counts"].get("gpu_metric_role:verifier-gpu", 0) == 0
    assert coverage["ready"] is False


def test_short_training_allocation_without_sample_stays_within_gap_budget() -> None:
    coverage = unified_timeline.Builder._metric_coverage(
        [
            {
                "start_epoch_ms": 1_000,
                "end_epoch_ms": 30_000,
                "gpu_job_id": "startup-failure",
                "gpu_attempt": 1,
            }
        ],
        [],
        max_gap_ms=45_000,
        match_fields=("gpu_job_id", "gpu_attempt"),
        allow_empty_within_gap=True,
    )
    assert coverage == [
        {
            "start_epoch_ms": 1_000,
            "end_epoch_ms": 30_000,
            "gpu_job_id": "startup-failure",
            "gpu_attempt": 1,
            "sample_count": 0,
            "max_gap_ms": 29_000,
            "covered": True,
        }
    ]


def test_terminal_tail_grace_never_masks_an_internal_metric_gap() -> None:
    interval = {
        "start_epoch_ms": 0,
        "end_epoch_ms": 200_000,
        "gpu_job_id": "cleanup-tail",
        "gpu_attempt": 1,
    }
    continuous = [
        {
            "epoch_ms": epoch_ms,
            "gpu_job_id": "cleanup-tail",
            "gpu_attempt": 1,
        }
        for epoch_ms in (10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000)
    ]
    covered = unified_timeline.Builder._metric_coverage(
        [interval],
        continuous,
        max_gap_ms=45_000,
        match_fields=("gpu_job_id", "gpu_attempt"),
        terminal_tail_grace_ms=180_000,
    )[0]
    assert covered["covered"] is True
    assert covered["trailing_gap_ms"] == 130_000
    assert covered["terminal_tail_grace_used"] is True

    with_internal_gap = continuous[:2] + continuous[-1:]
    rejected = unified_timeline.Builder._metric_coverage(
        [interval],
        with_internal_gap,
        max_gap_ms=45_000,
        match_fields=("gpu_job_id", "gpu_attempt"),
        terminal_tail_grace_ms=180_000,
    )[0]
    assert rejected["internal_max_gap_ms"] == 50_000
    assert rejected["covered"] is False


def test_pipeline_coverage_counts_explicit_failed_attempts_without_values() -> None:
    interval = {
        "start_epoch_ms": 0,
        "end_epoch_ms": 100_000,
        "gpu_job_id": "job-one",
        "gpu_attempt": 1,
    }
    samples = []
    for epoch_ms, group, status in (
        (10_000, 0, "ok"),
        (20_000, 1, "ok"),
        (30_000, 2, "ok"),
        (40_000, 0, "unavailable"),
        (50_000, 1, "unavailable"),
        (60_000, 2, "unavailable"),
        (70_000, 0, "ok"),
        (80_000, 1, "ok"),
        (90_000, 2, "ok"),
    ):
        gpu = {
            "pipeline_metrics_source": "cupti-pm-sampling",
            "pipeline_metrics_group": group,
            "pipeline_metrics_status": status,
        }
        if status == "ok":
            for field, groups in unified_timeline.GPU_PIPELINE_FIELD_GROUPS.items():
                if group in groups:
                    gpu[field] = 1.0
        samples.append(
            {
                "epoch_ms": epoch_ms,
                "gpu_job_id": "job-one",
                "gpu_attempt": 1,
                "metrics": {"gpus": [gpu]},
            }
        )

    coverage = unified_timeline.Builder._pipeline_metric_coverage(
        [interval],
        samples,
        max_gap_ms=45_000,
        match_fields=("gpu_job_id", "gpu_attempt"),
        allow_empty_within_gap=True,
    )

    assert all(rows[0]["covered"] for rows in coverage.values())
    tensor = coverage["tensor_pipe_active_pct"][0]
    assert tensor["measurement_count"] == 2
    assert tensor["unavailable_attempt_count"] == 1
    assert tensor["coverage_start_epoch_ms"] == 10_000
    assert tensor["coverage_end_epoch_ms"] == 90_000
    assert tensor["start_epoch_ms"] == 0
    assert tensor["end_epoch_ms"] == 100_000


def test_single_out_of_window_verifier_sample_fails_coverage(tmp_path: Path) -> None:
    state = fixture_run(tmp_path, training_lifecycle=False)
    for path in state.rglob("verifier/telemetry/samples.jsonl"):
        write_jsonl(
            path,
            [
                {
                    "epoch_s": 1786104050,
                    "role": "verifier-gpu",
                    "container_id": "late-verifier",
                    "sample_index": 1,
                    "gpus": [{"gpu_index": 0, "util_gpu_pct": 50}],
                }
            ],
        )
    coverage = unified_timeline.build_timeline(state)["coverage"]
    assert coverage["counts"]["gpu_metric_role:verifier-gpu"] == 1
    assert coverage["counts"]["verifier_gpu_intervals"] == 3
    assert coverage["counts"]["verifier_gpu_intervals_covered"] == 0
    assert coverage["requirements"]["verifier_gpu_metrics"] is False
    assert coverage["ready"] is False


def test_missing_verifier_sampler_lifecycle_blocks_readiness(tmp_path: Path) -> None:
    state = fixture_run(tmp_path)
    for path in state.rglob("verifier/telemetry/lifecycle.json"):
        path.unlink()
    coverage = unified_timeline.build_timeline(state)["coverage"]
    assert coverage["counts"]["verifier_evaluation_intervals"] == 3
    assert coverage["counts"]["verifier_gpu_intervals"] == 0
    assert coverage["requirements"]["verifier_gpu_lifecycle"] is False
    assert coverage["requirements"]["verifier_gpu_metrics"] is False


def test_cpu_only_agent_gpu_sample_is_not_inferred_as_verifier(tmp_path: Path) -> None:
    state = fixture_run(tmp_path)
    telemetry = state / "telemetry" / "host-samples.jsonl"
    with telemetry.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "epoch_s": 1786104012,
                    "ts_utc": "2026-08-07T12:00:12Z",
                    "role": "cpu-agent",
                    "container_id": "final-verifier",
                    "cpu_util_pct": 55.0,
                    "gpus": [
                        {
                            "gpu_index": 0,
                            "util_gpu_pct": 48.0,
                            "mem_used_mib": 2750.0,
                            "mem_total_mib": 23028.0,
                        }
                    ],
                }
            )
            + "\n"
        )
    payload = unified_timeline.build_timeline(state)
    sample = next(
        event
        for event in payload["events"]
        if event.get("container_id") == "final-verifier"
    )
    assert sample["role"] == "unknown"
    assert sample["reported_role"] == "cpu-agent"
    assert payload["coverage"]["counts"]["impossible_cpu_agent_gpu_samples"] == 1


def load_trace_mirror():
    spec = importlib.util.spec_from_file_location(
        "sprint_trace_mirror", ROOT / "event_runtime/container/sprint-trace-mirror.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_trace_mirror_only_publishes_complete_lines_and_resumes(tmp_path: Path) -> None:
    mirror = load_trace_mirror()
    home = tmp_path / "codex"
    source = home / "sessions" / "rollout.jsonl"
    source.parent.mkdir(parents=True)
    first = b'{"timestamp":"2026-08-07T12:00:00Z","type":"a"}\n'
    second = b'{"timestamp":"2026-08-07T12:00:01Z","type":"b"}\n'
    partial = b'{"timestamp":"2026-08-07T12:00:02Z"'
    source.write_bytes(first + second + partial)
    durable = tmp_path / "durable"
    root = durable / "runs" / "r" / "trace" / "raw" / "cpu-attempt-001"

    row = mirror.mirror_source(
        source, home, root, run_id="r", agent_kind="codex", cpu_attempt=1
    )
    assert row["bytes"] == len(first + second)
    chunks = list(root.rglob("*.jsonl"))
    assert len(chunks) == 1
    assert chunks[0].read_bytes() == first + second

    source.write_bytes(first + second + partial + b"}\n")
    row = mirror.mirror_source(
        source, home, root, run_id="r", agent_kind="codex", cpu_attempt=1
    )
    assert row["bytes"] == len(partial) + 2
    assert (
        b"".join(path.read_bytes() for path in sorted(root.rglob("*.jsonl")))
        == first + second + partial + b"}\n"
    )


def test_trace_mirror_replays_safely_if_cursor_is_lost(tmp_path: Path) -> None:
    mirror = load_trace_mirror()
    home = tmp_path / "codex"
    source = home / "sessions" / "rollout.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text('{"timestamp":"2026-08-07T12:00:00Z","type":"a"}\n')
    root = tmp_path / "durable"
    mirror.mirror_source(
        source, home, root, run_id="r", agent_kind="codex", cpu_attempt=1
    )
    cursor = next(root.rglob("cursor.json"))
    cursor.unlink()
    mirror.mirror_source(
        source, home, root, run_id="r", agent_kind="codex", cpu_attempt=1
    )
    assert len(list(root.rglob("*.jsonl"))) == 1
    assert next(root.rglob("cursor.json")).is_file()
