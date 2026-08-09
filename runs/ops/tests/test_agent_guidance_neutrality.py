from __future__ import annotations

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
    assert agent.replace(
        "    sprint-check policy.pt", "    python check_submission.py policy.pt"
    ) == verifier
