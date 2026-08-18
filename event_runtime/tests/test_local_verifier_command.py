from __future__ import annotations

from pathlib import Path

from event_runtime.agent.test_policy import build_command


def test_public_event_test_command_explicitly_bootstraps_isaac() -> None:
    command = build_command(Path("/app/policy.pt"), "local verifier")

    assert command[:2] == ["/usr/local/bin/event", "gpu"]
    assert command[command.index("--job-kind") + 1] == "verify"
    assert command[-3:-1] == ["bash", "-lc"]
    shell = command[-1]
    assert "SPRINT_ISAAC_BOOTSTRAP=/opt/sprint-isaac-bootstrap.py" in shell
    assert "exec bash /opt/event-verifier/test.sh" in shell


def test_published_verifier_honors_optional_bootstrap_without_requiring_it() -> None:
    script = (
        Path(__file__).resolve().parents[2] / "events/g1-100-metres/tests/test.sh"
    ).read_text()

    assert 'if [ -n "${SPRINT_ISAAC_BOOTSTRAP:-}" ]' in script
    assert 'ISAAC_PYTHON+=("$SPRINT_ISAAC_BOOTSTRAP")' in script
    assert '"${ISAAC_PYTHON[@]}" "$TESTS_DIR/verify.py"' in script
