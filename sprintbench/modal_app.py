"""Modal app that runs the sprint benchmark on a GPU.

Isaac Sim needs an RTX-capable GPU and a large container, so nothing here runs
locally.  The image follows NVIDIA's documented "Isaac Sim pip" install: a CUDA
12.8 PyTorch build, the Isaac Sim wheels (with the extension cache baked in so
the container does not go shopping for extensions at run time), then the pinned
Isaac Lab source tree installed editable, exactly as ``isaaclab.sh --install``
does it.

    modal run modal_app.py::smoke          # prove the stack imports and steps
    modal run modal_app.py::evaluate       # run the benchmark
"""

import os

import modal

ISAACSIM_VERSION = "5.1.0"
ISAACLAB_TAG = "v2.3.2"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
NVIDIA_INDEX = "https://pypi.nvidia.com"

HERE = os.path.dirname(os.path.abspath(__file__))

# Isaac Sim is a Kit application: it wants the usual desktop GL/X/Vulkan
# libraries present even when it never opens a window.
SYSTEM_DEPS = [
    "build-essential", "ca-certificates", "curl", "git", "unzip",
    "libatomic1", "libegl1", "libgl1", "libglu1-mesa", "libgomp1",
    "libsm6", "libice6", "libxi6", "libxrandr2", "libxt6", "libxext6",
    "libx11-6", "libxrender1", "libxcursor1", "libxinerama1",
    "libfreetype6", "libfontconfig1", "libglib2.0-0",
    "libxkbcommon0", "libxkbcommon-x11-0", "libvulkan1", "mesa-vulkan-drivers",
]

# The GLVND/EGL/Vulkan loaders Kit needs to find the NVIDIA driver when it
# renders.  These go in a late layer on purpose: adding them to SYSTEM_DEPS
# would invalidate the 15 GB Isaac Sim layer built on top of it.
GRAPHICS_DEPS = "libglvnd0 libglx0 libegl1 libgles2 libxcb1 libxau6 vulkan-tools"

# Modal's GPU containers expose the driver with compute+utility capabilities
# only, and mount no ICD manifests, so the Vulkan and EGL loaders have nothing
# to point at.  These two files are what the NVIDIA container runtime would
# normally install; the driver libraries themselves are already in the image.
# Kept on one line each: Modal turns run_commands into Dockerfile RUN lines, and
# a newline inside one ends the instruction.
VULKAN_ICD = '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3"}}'
EGL_VENDOR = '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}'

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install(*SYSTEM_DEPS)
    .env({
        # Isaac Sim refuses to start without an explicit EULA acknowledgement.
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "ACCEPT_EULA": "Y",
        "PRIVACY_CONSENT": "Y",
        "ISAACLAB_PATH": "/opt/IsaacLab",
        "PYTHONUNBUFFERED": "1",
    })
    .pip_install("torch==2.7.0", "torchvision==0.22.0", index_url=TORCH_INDEX)
    .pip_install(
        f"isaacsim[all,extscache]=={ISAACSIM_VERSION}",
        extra_index_url=NVIDIA_INDEX,
    )
    .run_commands(
        f"git clone --depth 1 --branch {ISAACLAB_TAG}"
        " https://github.com/isaac-sim/IsaacLab.git /opt/IsaacLab",
        # Isaac Lab's per-extension pyproject asks for an unpinned setuptools,
        # and setuptools >= 81 no longer ships pkg_resources, which its build
        # backend still reaches for.  Pin the build tools and skip isolation
        # rather than let pip resolve a setuptools that cannot build this.
        "pip install 'setuptools<80' wheel toml",
        # what isaaclab.sh --install does, minus the conda/symlink discovery
        "pip install --no-build-isolation -e /opt/IsaacLab/source/isaaclab",
        "pip install --no-build-isolation -e /opt/IsaacLab/source/isaaclab_assets",
        "pip install --no-build-isolation -e /opt/IsaacLab/source/isaaclab_mimic",
        "pip install --no-build-isolation -e '/opt/IsaacLab/source/isaaclab_rl[rsl-rl]'",
        "pip install --no-build-isolation -e /opt/IsaacLab/source/isaaclab_tasks",
    )
    .pip_install("rsl-rl-lib==3.1.2")
    .run_commands(
        f"apt-get update && apt-get install -y --no-install-recommends {GRAPHICS_DEPS}"
        " && rm -rf /var/lib/apt/lists/*",
        "mkdir -p /usr/share/vulkan/icd.d /usr/share/glvnd/egl_vendor.d",
        f"printf '%s' '{VULKAN_ICD}' > /usr/share/vulkan/icd.d/nvidia_icd.json",
        f"printf '%s' '{EGL_VENDOR}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json",
    )
    .env({
        # also late, for the same reason
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        "NVIDIA_VISIBLE_DEVICES": "all",
        "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
        "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
    })
    .add_local_dir(HERE, remote_path="/root/sprintbench", ignore=["results/*", "logs/*", "**/__pycache__"])
)

app = modal.App("g1-sprint-bench", image=image)

# Isaac Sim compiles shaders and caches downloaded USD assets on first run.  Both
# are slow and both are worth keeping between runs.
cache = modal.Volume.from_name("isaacsim-cache", create_if_missing=True)
results = modal.Volume.from_name("sprintbench-results", create_if_missing=True)

CACHE_MOUNTS = {
    "/root/.cache/ov": cache,
    "/results": results,
}

GPU = os.environ.get("SPRINT_GPU", "A10G")


@app.function(gpu=GPU, timeout=600)
def gpu_graphics_report() -> str:
    """Is there a rendering-capable driver in here, or only a compute one?

    Isaac Sim's RTX renderer needs the driver's GLX/EGL/Vulkan libraries, which
    a container runtime configured for compute alone does not mount.  Cheaper
    to ask directly than to infer it from a failed render.
    """
    import glob
    import os
    import subprocess

    out = []
    for pattern in ("libGLX_nvidia.so*", "libEGL_nvidia.so*", "libnvidia-glcore.so*",
                    "libnvidia-rtcore.so*", "libnvoptix.so*", "libcuda.so*",
                    "libnvidia-glvkspirv.so*", "libnvidia-gpucomp.so*",
                    "libnvidia-eglcore.so*", "libnvidia-tls.so*"):
        hits = glob.glob(f"/usr/lib/x86_64-linux-gnu/{pattern}") + glob.glob(f"/usr/lib64/{pattern}")
        out.append(f"{pattern}: {hits or 'MISSING'}")
    out.append("\ndevice nodes: " + str(sorted(glob.glob("/dev/nvidia*"))))
    for cmd in (["vulkaninfo", "--summary"], ["nvidia-smi", "-L"]):
        try:
            env = dict(os.environ, VK_LOADER_DEBUG="error,warn")
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
            out.append(f"\n$ {' '.join(cmd)}\n{(r.stdout or '')[:900]}\n--stderr--\n{(r.stderr or '')[:1200]}")
        except Exception as e:  # noqa: BLE001
            out.append(f"\n$ {' '.join(cmd)} -> {e}")
    return "\n".join(out)


@app.function(gpu=GPU, timeout=3600, volumes=CACHE_MOUNTS)
def smoke():
    """Launch Isaac Sim headless, build the stock G1 env, step it once."""
    import subprocess
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, enable_cameras=False, device="cuda:0")
    sim_app = launcher.app

    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401  (registers the task ids)
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg("Isaac-Velocity-Flat-G1-v0", device="cuda:0", num_envs=4)
    env = gym.make("Isaac-Velocity-Flat-G1-v0", cfg=cfg)
    obs, _ = env.reset()
    tensor = obs["policy"] if isinstance(obs, dict) else obs
    print("observation:", tuple(tensor.shape), "action:", env.action_space.shape)
    for _ in range(10):
        env.step(torch.zeros((4, env.action_space.shape[-1]), device="cuda:0"))
    print("stepped 10 control steps OK")
    env.close()
    sim_app.close()
    return {"obs_dim": int(tensor.shape[-1]), "ok": True}


@app.function(gpu=GPU, timeout=7200, volumes=CACHE_MOUNTS)
def evaluate(checkpoint: str, speeds: str, label: str, distance: float = 100.0,
             max_seconds: float = 200.0, video: bool = False,
             video_lane: int = 0, video_length: int = 1200,
             show_fall: bool = False) -> dict:
    """Run the sprint benchmark and return the result document."""
    import json
    import os
    import subprocess

    out = f"/results/{label}.json"
    ckpt = checkpoint if checkpoint == "zero" else f"/root/sprintbench/checkpoints/{checkpoint}"
    cmd = [
        "python", "/root/sprintbench/scripts/evaluate.py",
        "--checkpoint", ckpt,
        "--speeds", speeds, "--distance", str(distance),
        "--max-seconds", str(max_seconds), "--label", label,
        "--out", out, "--headless", "--geometry", "/results/collision_geometry.json",
    ]
    trace_out = f"/results/{label}.trace.json"
    cmd += ["--trace-out", trace_out, "--trace-lane", str(video_lane)]
    if show_fall:
        cmd.append("--show-fall")
    if video:
        cmd += ["--video", "--video-lane", str(video_lane),
                "--video-length", str(video_length)]
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd="/root/sprintbench")
    if proc.returncode != 0:
        raise RuntimeError(f"evaluate.py exited {proc.returncode}")

    results.commit()
    with open(out) as f:
        doc = json.load(f)
    if os.path.exists(trace_out):
        with open(trace_out) as f:
            doc["_trace"] = json.load(f)
    return doc


@app.function(gpu=GPU, timeout=1800, volumes=CACHE_MOUNTS)
def collision_audit(asset: str = "minimal") -> dict:
    """Which bodies can touch the ground at all?"""
    import json
    import subprocess

    out = f"/results/collision_audit_{asset}.json"
    proc = subprocess.run(
        ["python", "/root/sprintbench/scripts/collision_audit.py", "--headless", "--out", out, "--asset", asset],
        cwd="/root/sprintbench")
    if proc.returncode != 0:
        raise RuntimeError(f"collision_audit.py exited {proc.returncode}")
    results.commit()
    return json.load(open(out))


@app.function(gpu=GPU, timeout=1800, volumes=CACHE_MOUNTS)
def extract_geometry() -> dict:
    """Per-body collision hulls, so penetration can be measured exactly."""
    import json
    import subprocess

    out = "/results/collision_geometry.json"
    proc = subprocess.run(
        ["python", "/root/sprintbench/scripts/extract_collision_geometry.py",
         "--headless", "--out", out], cwd="/root/sprintbench")
    if proc.returncode != 0:
        raise RuntimeError(f"extract_collision_geometry.py exited {proc.returncode}")
    results.commit()
    return json.load(open(out))


@app.function(gpu=GPU, timeout=7200, volumes=CACHE_MOUNTS)
def verify(runs: int = 5, skip_robustness: bool = False,
           max_seconds: float = 60.0) -> dict:
    """Run the challenge verifier against the reference policy.

    Exercises the path a real submission takes: /tests mounted separately from
    the workspace, the policy loaded as a TorchScript archive, scoring done
    against the verifier's own copy of the environment.
    """
    import json
    import shutil
    import subprocess

    shutil.rmtree("/tests", ignore_errors=True)
    shutil.copytree("/root/sprintbench/challenge/tests", "/tests")
    # The reference wrapped to the task's 120-D interface: verifier-side only,
    # here to exercise the harness against a policy whose behaviour is known.
    subprocess.run(["python", "/tests/reference_policy.py",
                    "--checkpoint", "/root/sprintbench/checkpoints/Isaac-Velocity-Flat-G1-v0.pt",
                    "--out", "/tmp/policy.pt"], check=True)

    cmd = ["python", "/tests/verify.py", "--policy", "/tmp/policy.pt",
           "--logs", "/results/verifier", "--tests", "/tests", "--headless",
           "--runs", str(runs), "--max-seconds", str(max_seconds)]
    if skip_robustness:
        cmd.append("--skip-robustness")
    proc = subprocess.run(cmd)
    results.commit()
    out = {"exit_code": proc.returncode}
    for name in ("sprint_results.json",):
        path = f"/results/verifier/{name}"
        if os.path.exists(path):
            out[name] = json.load(open(path))
    log = "/results/verifier/results.log"
    if os.path.exists(log):
        out["log"] = open(log).read()
    return out


@app.function(gpu=GPU, timeout=3600, volumes=CACHE_MOUNTS)
def adversarial() -> dict:
    """Attack the floor, and report how deep anything got.

    The benchmark no longer disqualifies a run for penetration, so the physics
    has to be the thing that prevents it.  This is what makes that a measurement
    rather than a hope.
    """
    import json
    import subprocess

    out = {}
    for mode in ("slam", "drill", "pile"):
      for soft in (False, True):
        subprocess.run(["python", "/root/sprintbench/challenge/tests/adversarial_policy.py",
                        "--mode", mode, "--out", f"/tmp/adv_{mode}.pt"], check=True)
        key = f"{mode}-{'stock' if soft else 'hardened'}"
        res = f"/results/adv_{key}.json"
        cmd = ["python", "/root/sprintbench/scripts/evaluate.py",
               "--checkpoint", f"/tmp/adv_{mode}.pt", "--speeds", "4.0,4.0,4.0",
               "--max-seconds", "20", "--label", key, "--out", res,
               "--geometry", "/root/sprintbench/results/collision_geometry.json",
               "--headless"]
        if soft:
            cmd.append("--soft-physics")
        subprocess.run(cmd, cwd="/root/sprintbench")
        if os.path.exists(res):
            doc = json.load(open(res))
            deep = max(next(c["value"] for c in r["checks"]
                            if c["name"] == "no_ground_penetration") for r in doc["runs"])
            out[key] = {"deepest_penetration_cm": round(deep * 100, 4),
                        "max_distance_m": max(r["distance_m"] for r in doc["runs"])}
    results.commit()
    return out


@app.local_entrypoint()
def main(
    action: str = "smoke",
    checkpoint: str = "Isaac-Velocity-Flat-G1-v0.pt",
    speeds: str = "0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0",
    label: str = "isaaclab-g1-flat",
    distance: float = 100.0,
    max_seconds: float = 200.0,
    video: bool = False,
    video_lane: int = 0,
    video_length: int = 1200,
    show_fall: bool = False,
):
    import json
    import pathlib

    if action == "smoke":
        print(smoke.remote())
        return
    if action == "graphics":
        print(gpu_graphics_report.remote())
        return
    if action == "verify":
        doc = verify.remote(5, video, max_seconds)
        print(doc.get("log", "(no log)"))
        out = pathlib.Path(__file__).parent / "results" / "verifier_reference.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(doc.get("sprint_results.json", {}), indent=2))
        print(f"\nwrote {out}")
        return
    if action == "adversarial":
        doc = adversarial.remote()
        for mode, v in sorted(doc.items()):
            print(f"{mode:>16}: deepest {v['deepest_penetration_cm']:+.4f} cm, "
                  f"travelled {v['max_distance_m']:.1f} m")
        return
    if action == "geometry":
        doc = extract_geometry.remote()
        out = pathlib.Path(__file__).parent / "results" / "collision_geometry.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(doc))
        print(f"{len(doc['bodies'])} bodies with hulls; "
              f"{len(doc['no_collision'])} without: {doc['no_collision']}")
        return
    if action == "collision":
        doc = collision_audit.remote(checkpoint if checkpoint in ("minimal","full") else "minimal")
        out = pathlib.Path(__file__).parent / "results" / "collision_audit.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(doc, indent=2))
        print(f"colliding bodies ({len(doc['colliding'])}): {doc['colliding']}")
        print(f"ghost bodies    ({len(doc['ghosts'])}): {doc['ghosts']}")
        return

    doc = evaluate.remote(checkpoint, speeds, label, distance, max_seconds,
                          video, video_lane, video_length, show_fall)
    root = pathlib.Path(__file__).parent / "results"
    root.mkdir(exist_ok=True)
    trace = doc.pop("_trace", None)
    (root / f"{label}.json").write_text(json.dumps(doc, indent=2))
    print(f"wrote {root / f'{label}.json'}")
    if trace:
        (root / f"{label}.trace.json").write_text(json.dumps(trace, separators=(",", ":")))
        print(f"wrote {root / f'{label}.trace.json'} ({len(trace['frames'])} frames)")
    for run in doc["runs"]:
        failed = [c["name"] for c in run["checks"] if not c["passed"]]
        finish = f"{run['finish_time_s']:.2f}s" if run["finish_time_s"] else "DNF"
        print(f"  cmd {run['commanded_speed']:.2f} -> {run['mean_speed_mps']:.2f} m/s, "
              f"100 m {finish}, valid={run['valid']}, failed={failed or '-'}")
