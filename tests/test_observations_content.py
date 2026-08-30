"""Keep the model observations short, concrete, and illustrated by real runs."""

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def observations() -> str:
    page = (ROOT / "web/index.html").read_text()
    return re.search(r'<section[^>]+id="observations">([\s\S]*?)</section>', page).group(1)


def test_observations_are_three_short_model_summaries_without_example_links() -> None:
    section = observations()
    assert re.findall(r'<h3>(.*?)</h3>', section) == ["DeepSeek", "Luna", "GLM"]
    assert "Common approach" not in section
    assert "Recurring stumbles" not in section
    assert "<a " not in section
    paragraphs = re.findall(r'<p>(.*?)</p>', section)
    assert len(paragraphs) == 3
    for paragraph in paragraphs:
        assert len(re.split(r'(?<=[.!?])\s+(?=[A-Z])', paragraph)) == 4
        assert 55 <= len(paragraph.split()) <= 75
    assert "No trial finished 100m" in section
    assert "GLM’s fastest simulated 100m took 9.90 s" in section
    assert "0.32 s shy of Usain Bolt’s 9.58 s world record" in section
    assert "from a standing start without starting blocks" in section
    assert "one of five trials" in section
    assert "short forward launches" not in section
    assert paragraphs[1].endswith("For example, this policy takes one big step, then falls flat on its face.")
    for phrase in ["In its best trial", "reinforcement learning", "scripted crawling",
                   "using PPO", "lane and collision rules", "working gait worse",
                   "complete collision shapes", "steering corrections"]:
        assert phrase in section


def test_observations_use_three_requested_lazy_replays() -> None:
    section = observations()
    frames = re.findall(r'<iframe([^>]+)>', section)
    assert len(frames) == 3
    assert all('loading="lazy"' in frame and '?example=1' in frame and 'title="' in frame for frame in frames)
    assert all('&amp;autoplay=0' in frame for frame in frames)
    page = (ROOT / "web/index.html").read_text()
    assert page.count('autoplay=0') == 3, "Only the three Observations demos opt out"
    assert '/replay/frontier-76d7d31f8c51?example=1' in frames[0]
    assert '/replay/frontier-dd28af63bf84?example=1' in frames[1]
    assert '/replay/frontier-5eb6fee88ef0?example=1' in frames[2]
    luna = re.findall(r'<article>([\s\S]*?)</article>', section)[1]
    assert 'class="dq-replay observation-replay"' in luna
    assert 'title="Luna trial 2 policy 9 taking a big step before falling"' in luna
    assert "Replay coming soon" not in luna


def test_observation_replay_choices_match_published_results() -> None:
    data = json.loads((ROOT / "web/data/performance/current.json").read_text())
    submissions = [{**point, "run_id": run["run_id"]} for run in data["runs"] for point in run["points"]]
    deepseek = [row for row in submissions if "deepseek" in row["run_id"]]
    best = max(deepseek, key=lambda row: row.get("continuous_score_mps") or 0)
    assert best["replay_url"] == "/replay/frontier-76d7d31f8c51.html"
    assert 77 < best["max_legal_distance_m"] < 79
    glm = next(row for row in submissions if row.get("replay_url") == "/replay/frontier-5eb6fee88ef0.html")
    assert glm["submission_index"] == 1
    assert glm["termination_reason"] == "in_lane"
    glm_finishes = [row for row in submissions if "glm" in row["run_id"] and row.get("termination_reason") == "finished"]
    fastest = min(row["time_to_max_legal_distance_s"] for row in glm_finishes)
    assert fastest == 9.90
    assert round(fastest - 9.58, 2) == 0.32
    luna = next(row for row in submissions if row.get("replay_url") == "/replay/frontier-dd28af63bf84.html")
    assert luna["run_id"] == "s10-vexp-r123-20260828-luna-2"
    assert luna["submission_index"] == 9
    assert luna["queue_source_step_id"] == "a1-s1020"
    assert luna["termination_reason"] == "timeout"
    assert 2 < luna["max_legal_distance_m"] < 2.01


def test_scoring_captions_describe_the_visible_body_parts() -> None:
    page = (ROOT / "web/index.html").read_text()
    scoring = re.search(r'<div class="scoring-copy">([\s\S]*?)</div>', page).group(1)
    assert scoring.count('<p>') == 2
    assert '<p>We call this <strong>Effective Speed</strong>' in scoring
    assert "The robot’s right hand crossed the right lane boundary." in page
    assert "The robot’s feet bumped into each other as it fell." in page
    assert '/replay/frontier-8e2feee19eac?example=1' in page
    assert 'src="/replay/frontier-bca4f7ab8e3c?example=1"' in page
    assert '/trajectory?run=s10-vexp-r123-20260828-luna-5&amp;policies=1&amp;focus=1&amp;step=a1-s183' in page


def test_effective_speed_term_uses_white_text_without_a_title_chip() -> None:
    page = (ROOT / "web/index.html").read_text()
    css = (ROOT / "web/styles.css").read_text()
    assert '<strong>Effective Speed</strong>' in page
    assert '.scoring-copy p > strong:first-of-type { color: var(--text); }' in css


def test_observation_replays_have_room_to_show_the_gait() -> None:
    css = (ROOT / "web/styles.css").read_text()
    assert ".observation-cards { display: grid; grid-template-columns: 1fr;" in css
    assert "@media (max-width: 1000px)" in css
    assert ".observation-cards article { grid-template-columns: 1fr; gap: 16px; }" in css
    assert ".observation-replay { min-height: 240px; }" in css
