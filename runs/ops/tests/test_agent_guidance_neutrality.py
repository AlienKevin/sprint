from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "challenge/g1-sprint-100m-lane"

# These are the benchmark-authored guidance surfaces available to the model.
# Runtime implementation code may necessarily use words such as "rewards" for
# trusted result-schema fields; it is not task guidance and is excluded here.
GUIDANCE = (
    TASK / "instruction.md",
    TASK / "README.md",
    TASK / "environment/train/README.md",
    TASK / "environment/train/robot.py",
    TASK / "environment/train/spec.py",
    TASK / "environment/bin/sprint-submit",
    TASK / "environment/bin/sprint-board",
    TASK / "environment/bin/sprint-check",
    TASK / "environment/bin/sprint-gpu-train",
    TASK / "environment/bin/sprint-verify",
    ROOT / "runs/codex-goal.j2",
    ROOT / "runs/codex-goal-slash.j2",
)

NAMED_METHOD_CUES = (
    "ppo",
    "rsl-rl",
    "reinforcement learning",
    "reinforcement-learning",
    "policy gradient",
    "actor-critic",
    "imitation learning",
    "evolution strategy",
    "curriculum",
    "velocity-derived",
    "g1flatenvcfg",
    "training tip",
)

PERFORMANCE_PRIORS = (
    "published result",
    "published g1",
    "world record",
    "guinness",
    "reference time",
    "reference speed",
)


def test_agent_guidance_is_method_neutral() -> None:
    combined = "\n".join(path.read_text().lower() for path in GUIDANCE)
    for cue in (*NAMED_METHOD_CUES, *PERFORMANCE_PRIORS):
        assert cue not in combined, f"agent guidance contains method cue {cue!r}"


def test_agent_and_verifier_submission_contracts_match() -> None:
    agent = (TASK / "environment/bin/sprint-check").read_text()
    verifier = (TASK / "tests/check_submission.py").read_text()
    assert (
        agent.replace(
            "    sprint-check policy.pt", "    python check_submission.py policy.pt"
        )
        == verifier
    )


def test_published_verifier_is_an_exact_reviewed_source_mirror() -> None:
    published = TASK / "environment/verifier"
    manifest = json.loads((published / "SOURCE_MANIFEST.json").read_text())
    assert manifest["source"] == "challenge/g1-sprint-100m-lane/tests"
    for relative, expected in manifest["files"].items():
        trusted = TASK / "tests" / relative
        exposed = published / relative
        assert exposed.read_bytes() == trusted.read_bytes()
        assert hashlib.sha256(exposed.read_bytes()).hexdigest() == expected


def test_local_verifier_uses_only_the_trial_training_queue() -> None:
    helper = (TASK / "environment/bin/sprint-verify").read_text()
    dockerfile = (TASK / "environment/Dockerfile").read_text()
    assert '"sprint-gpu-train"' in helper
    assert '"--max-attempts",\n        "1"' in helper
    assert "/opt/sprint-verifier/test.sh" in helper
    assert "COPY verifier /opt/sprint-verifier" in dockerfile
    assert "chmod -R a-w /opt/sprint-verifier" in dockerfile


def test_equivalence_gate_accepts_only_matching_canonical_outputs(
    tmp_path: Path,
) -> None:
    compare = TASK / "tests/compare_results.py"
    agent = tmp_path / "agent.json"
    official = tmp_path / "official.json"
    output = tmp_path / "proof.json"
    result = {
        "valid_run": False,
        "best_valid_100m_s": None,
        "max_distance_m": 1.2344,
        "max_distance_semantics": "legal_prefix_until_first_disqualification",
        "lanes_finished": 0,
        "lanes_valid": 0,
        "lanes_total": 1,
        "runs": 1,
        "all_valid_times_s": [],
        "failed_gates": ["finished"],
    }
    agent.write_text(json.dumps(result))
    official.write_text(json.dumps({**result, "max_distance_m": 1.23449}))
    command = [
        sys.executable,
        str(compare),
        "--agent",
        str(agent),
        "--official",
        str(official),
        "--out",
        str(output),
    ]
    assert subprocess.run(command, check=False).returncode == 0
    assert json.loads(output.read_text())["equivalent"] is True

    official.write_text(json.dumps({**result, "failed_gates": ["in_lane"]}))
    assert subprocess.run(command, check=False).returncode != 0
