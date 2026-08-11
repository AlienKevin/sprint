# Event runtime

This package contains mechanisms shared by every event: trusted cost
accounting, compute-job lifecycle, recovery, telemetry, archival scheduling,
and public-data export. Event rules, embodiments, policy interfaces, metrics,
and verifiers stay under `events/<event>/`.

The runtime must not reinterpret an event result. It transports trusted event
outputs and applies the comparison-cost inclusion policy used by both the agent
and the website.

Run the shared and operator tests together:

```bash
uv run --project harbor pytest -q event_runtime/tests runs/ops/tests
```
