from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "events/g1-100-metres"

# These are the benchmark-authored guidance surfaces available to the model.
# Runtime implementation code may necessarily use words such as "rewards" for
# trusted result-schema fields; it is not task guidance and is excluded here.
GUIDANCE = (
    TASK / "instruction.md",
    TASK / "README.md",
    TASK / "environment/README.md",
    TASK / "environment/robot.py",
    TASK / "environment/spec.py",
    ROOT / "event_runtime/container/bin/event",
    TASK / "tests/check_submission.py",
    ROOT / "event_runtime/agent/archive.py",
    ROOT / "event_runtime/agent/gpu.py",
    ROOT / "event_runtime/agent/test_policy.py",
    ROOT / "event_runtime/control/templates/codex.j2",
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

REMOVED_REPOSITORY_PATHS = (
    "challenge/g1-sprint-100m-lane",
    "challenge/g1-sprint-100m",
    "events/g1-100-metres/environment/train",
    "events/g1-100-metres/environment/verifier",
    "events/g1-100-metres/tools",
    "events/g1-100-metres/visualization",
    "sprint-web",
)

PATH_AUDIT_SUFFIXES = {
    ".dockerignore",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}


def test_agent_guidance_is_method_neutral() -> None:
    combined = "\n".join(path.read_text().lower() for path in GUIDANCE)
    for cue in (*NAMED_METHOD_CUES, *PERFORMANCE_PRIORS):
        assert cue not in combined, f"agent guidance contains method cue {cue!r}"


def test_documented_container_paths_match_the_built_agent_image() -> None:
    dockerfile = (TASK / "environment/Dockerfile").read_text()
    image = (ROOT / "event_runtime/image.py").read_text()
    gpu = (ROOT / "event_runtime/agent/gpu.py").read_text()
    instruction = (TASK / "instruction.md").read_text()
    guide = (TASK / "environment/README.md").read_text()

    assert "COPY README.md __init__.py asset_probe.py robot.py spec.py " in dockerfile
    assert "standing_start.py /app/train/" in dockerfile
    assert "`/app/train/README.md`" in instruction
    assert "The writable workspace is `/app`" in guide
    assert "event gpu -- python3 -u /app/YOUR_SCRIPT.py" in guide
    assert "/app/train/YOUR_SCRIPT.py" not in guide
    assert 'AGENT_WORKSPACE_ROOT = Path("/app")' in gpu
    assert 'submit.add_argument("--workdir", default="/app")' in gpu
    assert '"ln -sfn /opt/event-verifier /app/verifier"' in image


def test_removed_repository_paths_do_not_reappear() -> None:
    roots = (ROOT / "events", ROOT / "event_runtime", ROOT / "tests", ROOT / "web")
    paths = [ROOT / "README.md"]
    for root in roots:
        paths.extend(root.rglob("*"))
    for path in paths:
        if path.resolve() == Path(__file__).resolve():
            continue
        if not path.is_file() or path.suffix not in PATH_AUDIT_SUFFIXES:
            continue
        if "__pycache__" in path.parts or "public" in path.parts:
            continue
        text = path.read_text(errors="replace")
        for removed in REMOVED_REPOSITORY_PATHS:
            assert removed not in text, f"{path.relative_to(ROOT)} uses {removed}"


def test_event_command_runs_from_its_installed_path(tmp_path: Path) -> None:
    installed = tmp_path / "usr/local/bin/event"
    installed.parent.mkdir(parents=True)
    installed.write_bytes((ROOT / "event_runtime/container/bin/event").read_bytes())
    environment = {"PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, str(installed), "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    assert "event cost" in result.stdout


def test_codex_comparison_models_pin_provider_compatible_tool_contracts(
    tmp_path: Path,
) -> None:
    environment = ROOT / "event_runtime/container"
    models = ROOT / "event_runtime/models"
    deepseek = json.loads((models / "deepseek.json").read_text())
    for model in deepseek["models"]:
        assert model["tool_mode"] is None
        assert model["multi_agent_version"] == "v2"

    luna_lock = json.loads((models / "luna.json").read_text())
    assert luna_lock["codex_version"] == "0.147.0"
    assert luna_lock["model"]["slug"] == "gpt-5.6-luna"
    assert luna_lock["model"]["tool_mode"] == "code_mode_only"
    assert luna_lock["model"]["multi_agent_version"] == "v1"

    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        '[projects."/app"]\ntrust_level = "trusted"\n'
    )
    result = subprocess.run(
        ["bash", str(environment / "sprint-apply-luna-codex-config.sh")],
        check=False,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "CODEX_HOME": str(codex_home),
            "SPRINT_CODEX_LUNA_MODEL_LOCK": str(models / "luna.json"),
            "SPRINT_CODEX_DEEPSEEK_MODELS_JSON": str(models / "deepseek.json"),
        },
    )
    assert result.returncode == 0, result.stderr
    locked_catalog = json.loads((codex_home / "models.json").read_text())
    assert [model["slug"] for model in locked_catalog["models"]] == [
        "@preset/sprint-gpt-5-6-luna-openai-standard"
    ]
    assert locked_catalog["models"][0]["tool_mode"] == "code_mode_only"
    assert locked_catalog["models"][0]["multi_agent_version"] == "v1"
    config = (codex_home / "config.toml").read_text()
    assert 'model = "@preset/sprint-gpt-5-6-luna-openai-standard"' in config
    assert 'model_provider = "sprint_openrouter"' in config
    assert 'base_url = "http://127.0.0.1:18080/api/v1"' in config
    assert f'model_catalog_json = "{codex_home / "models.json"}"' in config

    official_deepseek = json.loads((models / "deepseek.json").read_text())
    result = subprocess.run(
        ["bash", str(environment / "sprint-apply-deepseek-codex-config.sh")],
        check=False,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "CODEX_HOME": str(codex_home),
            "SPRINT_CODEX_DEEPSEEK_MODELS_JSON": str(models / "deepseek.json"),
            "SPRINT_CODEX_DEEPSEEK_BASE_URL": "https://openrouter.ai/api/v1",
            "SPRINT_CODEX_DEEPSEEK_MODEL": (
                "@preset/sprint-deepseek-v4-flash-0731-official"
            ),
        },
    )
    assert result.returncode == 0, result.stderr
    locked_catalog = json.loads((codex_home / "models.json").read_text())
    assert locked_catalog["models"][0]["slug"] == (
        "@preset/sprint-deepseek-v4-flash-0731-official"
    )
    expected_model = official_deepseek["models"][0] | {
        "slug": "@preset/sprint-deepseek-v4-flash-0731-official"
    }
    expected_catalog = official_deepseek | {
        "models": [expected_model, *official_deepseek["models"][1:]]
    }
    assert locked_catalog == expected_catalog
    config = (codex_home / "config.toml").read_text()
    assert 'model = "@preset/sprint-deepseek-v4-flash-0731-official"' in config
    assert 'base_url = "https://openrouter.ai/api/v1"' in config
    assert 'wire_api = "responses"' in config


def test_published_verifier_is_an_exact_reviewed_source_mirror() -> None:
    from event_runtime.event import load_event
    from event_runtime.sync_verifier import materialize_public_verifier

    with tempfile.TemporaryDirectory() as raw:
        published = Path(raw) / "verifier"
        materialize_public_verifier(load_event(repository_root=ROOT), published)
        manifest = json.loads((published / "SOURCE_MANIFEST.json").read_text())
        assert manifest["source"] == "events/g1-100-metres/tests"
        for relative, expected in manifest["files"].items():
            trusted = TASK / "tests" / relative
            exposed = published / relative
            assert exposed.read_bytes() == trusted.read_bytes()
            assert hashlib.sha256(exposed.read_bytes()).hexdigest() == expected


def test_training_and_both_verifiers_share_one_exact_standing_start() -> None:
    trusted = TASK / "tests/verifier/standing_start.py"
    training = TASK / "environment/standing_start.py"
    assert trusted.read_bytes() == training.read_bytes()

    source = trusted.read_text()
    assert 'CANONICAL_START_NAME = "standard-standing-start-v1"' in source
    assert "ROOT_LINEAR_VELOCITY_M_S = (0.0, 0.0, 0.0)" in source
    assert "ROOT_ANGULAR_VELOCITY_RAD_S = (0.0, 0.0, 0.0)" in source
    assert 'JOINT_VELOCITIES_RAD_S = {".*": 0.0}' in source

    official_cfg = (TASK / "tests/verifier/environment.py").read_text()
    training_robot = (TASK / "environment/robot.py").read_text()
    needle = "apply_canonical_standing_start(robot)"
    assert needle in official_cfg
    assert needle in training_robot


def test_standing_start_adds_no_blocks_or_track_physics() -> None:
    source = (TASK / "tests/verifier/standing_start.py").read_text().lower()
    assert "startingblock" not in source
    assert "starting_block" not in source
    assert "foot plate" not in source


def test_course_has_no_per_rollout_start_perturbation() -> None:
    source = (TASK / "tests/verifier/forward_command.py").read_text()
    assert "ATTEMPT_YAW_OFFSETS" not in source
    assert "reset_base_by_lane" not in source
    config = (TASK / "tests/verifier/environment.py").read_text()
    assert '"pose_range": {}' in config
    assert '"velocity_range": {}' in config


def test_policy_cannot_choose_a_private_pre_start_pose() -> None:
    source = (TASK / "tests/verifier/rollout.py").read_text()
    settle = source.index("# 1. Initialize PhysX/contact state")
    gun = source.index("# 2. The starting gun")
    release = source.index("# 3. release")
    pre_start = source[settle:gun]
    at_gun = source[gun:release]

    assert "env.step(zero_actions)" in pre_start
    assert "env.step(policy(obs_t))" not in pre_start
    assert (
        "with torch.inference_mode():\n            env.step(zero_actions)"
        not in pre_start
    )
    assert "obs, _ = env.reset()" in at_gun
    assert 'getattr(policy, "reset", None)' in at_gun
    assert "reset_policy(torch.ones" in at_gun


def test_official_loader_preserves_optional_policy_reset() -> None:
    source = (TASK / "tests/verify.py").read_text()
    assert 'module_reset = getattr(module, "reset", None)' in source
    assert "infer.reset = reset" in source


def test_canonical_standing_start_sets_every_initial_state_field() -> None:
    source = TASK / "tests/verifier/standing_start.py"
    module_spec = importlib.util.spec_from_file_location("standing_start", source)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    state = SimpleNamespace(
        pos=None,
        rot=None,
        lin_vel=None,
        ang_vel=None,
        joint_pos=None,
        joint_vel=None,
    )
    robot = SimpleNamespace(init_state=state)
    assert module.apply_canonical_standing_start(robot) is robot
    assert state.pos == module.ROOT_POSITION_M
    assert state.rot == module.ROOT_ORIENTATION_WXYZ
    assert state.lin_vel == module.ROOT_LINEAR_VELOCITY_M_S
    assert state.ang_vel == module.ROOT_ANGULAR_VELOCITY_RAD_S
    assert state.joint_pos == module.JOINT_POSITIONS_RAD
    assert state.joint_vel == module.JOINT_VELOCITIES_RAD_S


def test_local_verifier_uses_only_the_trial_training_queue() -> None:
    helper = (ROOT / "event_runtime/agent/test_policy.py").read_text()
    image = (ROOT / "event_runtime/image.py").read_text()
    assert '"gpu"' in helper
    assert '"--max-attempts",\n        "1"' in helper
    assert "/opt/event-verifier/test.sh" in helper
    assert 'public_verifier, "/opt/event-verifier"' in image
    assert '"chmod -R a-w /opt/event-verifier; "' in image


def test_equivalence_gate_accepts_only_matching_canonical_outputs(
    tmp_path: Path,
) -> None:
    compare = ROOT / "event_runtime/preflight/compare_results.py"
    agent = tmp_path / "agent.json"
    official = tmp_path / "official.json"
    output = tmp_path / "proof.json"
    result = {
        "valid_run": False,
        "best_valid_100m_s": None,
        "max_distance_m": 1.2344,
        "max_distance_semantics": "legal_prefix_until_first_disqualification",
        "lane_containment_semantics": "whole_body_collision_envelope_between_vertical_boundaries",
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
