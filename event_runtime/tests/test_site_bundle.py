from __future__ import annotations

import json
from pathlib import Path

import pytest

from event_runtime.export.site_bundle import build_site_bundle


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def build_fixture(tmp_path: Path) -> tuple[Path, Path]:
    web = tmp_path / "web"
    bundle = tmp_path / "bundle"
    web.mkdir()
    (web / "index.html").write_text("current site\n")
    (web / ".vercel").mkdir()
    (web / ".vercel/project.json").write_text('{"projectId":"test"}\n')
    (web / ".vercel/private.env").write_text("SECRET=no\n")
    (web / ".publisher.tmp").write_text("unpublished\n")

    current_run = "batch-current-luna-1"
    stale_run = "batch-old-deepseek-1"
    batch = {
        "schema_version": 1,
        "batch_id": "batch-current",
        "arms": [{"run_id": current_run}],
    }
    write_json(web / "data/batches/current.json", batch)
    write_json(web / "data/batches/batch-current.json", batch)
    write_json(
        web / "data/batches/batch-old.json",
        {"batch_id": "batch-old", "arms": [{"run_id": stale_run}]},
    )
    write_json(
        web / "data/performance/current.json",
        {"schema_version": 1, "runs": [{"run_id": current_run}], "models": []},
    )

    for family in ("policies", "timelines", "trajectories"):
        write_json(
            web / "data" / family / "index.json",
            {
                "schema_version": 1,
                "runs": [
                    {
                        "run_id": current_run,
                        "path": f"/data/{family}/{current_run}.json",
                    },
                    {
                        "run_id": stale_run,
                        "path": f"/data/{family}/{stale_run}.json",
                    },
                ],
            },
        )
        current_payload: dict = {"schema_version": 1, "run_id": current_run}
        if family == "policies":
            current_payload["policies"] = [
                {
                    "replay_url": "/replay/frontier-current",
                    "web_html": "/replay/frontier-current.html",
                }
            ]
        write_json(web / "data" / family / f"{current_run}.json", current_payload)
        write_json(
            web / "data" / family / f"{stale_run}.json",
            {"schema_version": 1, "run_id": stale_run},
        )
    write_json(
        web / "data/timeline-overviews" / f"{current_run}.json",
        {"schema_version": 1, "run_id": current_run},
    )
    write_json(
        web / "data/timeline-overviews" / f"{stale_run}.json",
        {"schema_version": 1, "run_id": stale_run},
    )

    (web / "replay").mkdir()
    (web / "replay/frontier-current.html").write_text("current replay\n")
    (web / "replay/frontier-old.html").write_text("old replay\n")
    (web / "captures").mkdir()
    (web / "captures/frontier-old.json").write_text("{}\n")
    return web, bundle


def test_bundle_contains_only_current_batch_results(tmp_path: Path) -> None:
    web, bundle = build_fixture(tmp_path)

    report = build_site_bundle(web, bundle, require_current=True)

    assert report["mode"] == "current_batch"
    assert report["run_ids"] == ["batch-current-luna-1"]
    assert (bundle / "index.html").read_text() == "current site\n"
    assert (bundle / ".vercel/project.json").is_file()
    assert not (bundle / ".vercel/private.env").exists()
    assert not (bundle / ".publisher.tmp").exists()
    assert (bundle / "data/batches/current.json").is_file()
    assert (bundle / "data/batches/batch-current.json").is_file()
    assert not (bundle / "data/batches/batch-old.json").exists()
    assert (bundle / "replay/frontier-current.html").is_file()
    assert not (bundle / "replay/frontier-old.html").exists()
    assert not (bundle / "captures/frontier-old.json").exists()

    for family in ("policies", "timelines", "trajectories"):
        index = json.loads((bundle / "data" / family / "index.json").read_text())
        assert [row["run_id"] for row in index["runs"]] == ["batch-current-luna-1"]
        assert (bundle / "data" / family / "batch-current-luna-1.json").is_file()
        assert not (bundle / "data" / family / "batch-old-deepseek-1.json").exists()
    assert (
        bundle / "data/timeline-overviews/batch-current-luna-1.json"
    ).is_file()
    assert not (
        bundle / "data/timeline-overviews/batch-old-deepseek-1.json"
    ).exists()


def test_bundle_fails_closed_on_missing_current_replay(tmp_path: Path) -> None:
    web, bundle = build_fixture(tmp_path)
    (web / "replay/frontier-current.html").unlink()

    with pytest.raises(RuntimeError, match="references a missing asset"):
        build_site_bundle(web, bundle, require_current=True)


def test_bundle_falls_back_for_static_site_without_batch(tmp_path: Path) -> None:
    web = tmp_path / "web"
    bundle = tmp_path / "bundle"
    (web / "data/timelines").mkdir(parents=True)
    (web / "index.html").write_text("static\n")
    (web / "data/timelines/run.json").write_text("{}\n")

    report = build_site_bundle(web, bundle)

    assert report["mode"] == "full_tree_without_current_batch"
    assert (bundle / "data/timelines/run.json").is_file()
