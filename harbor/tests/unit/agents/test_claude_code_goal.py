"""The claude-code agent's `goal` kwarg."""

from harbor.agents.installed.claude_code import ClaudeCode

INSTRUCTION = "Train a control policy that runs the G1 100 m as fast as possible."


def _agent(goal) -> ClaudeCode:
    agent = ClaudeCode.__new__(ClaudeCode)
    agent.goal = goal
    return agent


def test_off_by_default_leaves_the_instruction_alone():
    assert _agent(None)._apply_goal(INSTRUCTION) == INSTRUCTION
    assert _agent(False)._apply_goal(INSTRUCTION) == INSTRUCTION


def test_true_makes_the_instruction_the_condition():
    assert _agent(True)._apply_goal(INSTRUCTION) == f"/goal {INSTRUCTION}"


def test_a_string_states_the_condition_and_keeps_the_brief():
    condition = "Keep going until the time stops improving."
    assert _agent(condition)._apply_goal(INSTRUCTION) == (
        f"/goal {condition}\n\n{INSTRUCTION}"
    )


def test_a_blank_string_falls_back_to_the_instruction():
    """Otherwise `--ak goal=` would send a bare `/goal` with no condition."""
    assert _agent("   ")._apply_goal(INSTRUCTION) == f"/goal {INSTRUCTION}"


def test_a_quoted_on_switch_is_not_taken_as_the_condition():
    """`--ak goal=true` parses to a bool, but a quoted one would not.

    `/goal true` is a nonsense condition that still looks like it worked.
    """
    for spelling in ("true", "True", "yes", "on", "1"):
        assert _agent(spelling)._apply_goal(INSTRUCTION) == f"/goal {INSTRUCTION}"


def test_the_instruction_survives_an_unrecognised_slash_command():
    """The wrapper has to degrade, not destroy.

    A CLI that does not know `/goal` still receives the whole brief after it,
    so the worst case is a stray token rather than an agent with no task.
    """
    prompt = _agent(True)._apply_goal(INSTRUCTION)
    assert prompt.endswith(INSTRUCTION)
