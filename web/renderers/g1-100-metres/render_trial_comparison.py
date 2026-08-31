#!/usr/bin/env python3
"""Build the static, query-driven multi-policy trajectory replay shell."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from render import G1_PARENT, PREFERRED, model_identity, replay_i18n_html, shared_asset_html
from render_comparison import add_mobile_closeup


ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / "web/replay-template.html"
SCENE = Path(__file__).with_name("scene.js")
HQ_DEFAULT = Path(__file__).with_name("g1_hq.json")
POLICY_INDEX_DEFAULT = ROOT / "web/data/policies/index.json"


def load_registry(index_path: Path) -> dict[str, dict]:
    """Return every public policy whose authoritative capture is available."""

    registry: dict[str, dict] = {}
    index = json.loads(index_path.read_text())
    for run in index.get("runs", []):
        policy_path = ROOT / "web" / str(run["path"]).lstrip("/")
        payload = json.loads(policy_path.read_text())
        identity = model_identity(payload.get("model")) or {}
        for policy in payload.get("policies", []):
            replay_url = str(policy.get("replay_url") or "")
            capture_id = Path(replay_url).stem
            # Some captures were published after the policy export's readiness
            # snapshot. Recover only the exact submitted artifact, never an
            # arbitrary capture that happens to share its short URL prefix.
            needs_validation = not policy.get("replay_ready") or not replay_url
            policy_sha = str(policy.get("policy_sha256") or "").lower()
            if needs_validation:
                if not re.fullmatch(r"[a-f0-9]{64}", policy_sha):
                    continue
                capture_id = f"frontier-{policy_sha[:12]}"
            capture_path = ROOT / "web/captures" / f"{capture_id}.json"
            if (
                not capture_id.startswith("frontier-")
                or not capture_path.is_file()
            ):
                continue
            if needs_validation:
                try:
                    capture = json.loads(capture_path.read_text())
                except (OSError, ValueError):
                    continue
                if (
                    capture.get("policy_sha256") != policy_sha
                    or capture.get("schema_version") != 2
                    or not capture.get("body_names")
                    or not any(capture.get("frames") or [])
                ):
                    continue
            registry[capture_id] = {
                "captureId": capture_id,
                "url": f"/captures/{capture_id}.json",
                "policyNumber": int(policy["submission_index"]),
                "label": f"Policy #{int(policy['submission_index'])}",
                "color": identity.get("color") or "#6E97C4",
                "identity": identity,
                "runId": payload.get("run_id"),
                "model": payload.get("model"),
            }
    return registry


def _safe_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":")).replace("</", "<\\/")


def build_shell(*, registry: dict, hq: dict, shared_assets_dir: Path | None = None) -> str:
    marker = "<script>const DATA="
    head = TEMPLATE.read_text().split(marker, 1)[0]
    head = head.replace(
        '<div class="lanes">',
        '<div class="lanes" aria-label="Selected policies">',
        1,
    )
    bootstrap = f"""
<script>
const TRIAL_BOOT={{
  registry:{_safe_json(registry)},
  hq:{_safe_json(hq)},
  preferred:{_safe_json(PREFERRED)},
  parents:{_safe_json(G1_PARENT)},
  scene:{_safe_json(add_mobile_closeup(SCENE.read_text()))}
}};
{Path(__file__).with_name('trial-comparison.js').read_text()}
</script>
"""
    shell_css = """
<style>
html,body{margin:0;background:#070908}.wrap{max-width:none;padding:0}.wrap>.eyebrow,.wrap>h1,.wrap>.lede,.polsel,.cap,.story{display:none}.stagewrap{margin:0;border:0;border-radius:0;box-shadow:none}
/* The result under the timer must not move the independent policy list. */
.hud:has(.clock-status:not(:empty)) .lanes{top:12px}
.lc.emphasized{box-shadow:inset 3px 0 0 var(--emphasis,#fff);border-color:var(--emphasis,#fff);background:color-mix(in srgb,var(--emphasis,#fff) 20%,#0a1118)}
.lc .policy-follow{display:flex;align-items:center;gap:8px;flex:1;min-width:0;border:0;background:none;padding:0;color:inherit;font:inherit;text-align:left;cursor:pointer}
.lc .policy-remove{flex:none;width:24px;height:24px;border:0;border-radius:4px;background:rgba(255,255,255,.08);color:inherit;font:700 18px/1 sans-serif;cursor:pointer}
.lc .policy-remove:hover{background:rgba(255,255,255,.2)}.lc button:focus-visible{outline:2px solid #fff;outline-offset:3px}
</style>
"""
    html = replay_i18n_html(head.replace('<div class="wrap">', shell_css + '<div class="wrap">', 1) + bootstrap)
    return shared_asset_html(html, shared_assets_dir) if shared_assets_dir is not None else html


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-index", type=Path, default=POLICY_INDEX_DEFAULT)
    parser.add_argument("--hq", type=Path, default=HQ_DEFAULT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shared-assets", type=Path, help="Website shared asset directory; omit for inline assets")
    args = parser.parse_args()
    registry = load_registry(args.policy_index)
    hq = json.loads(args.hq.read_text())["meshes"]
    html = build_shell(registry=registry, hq=hq, shared_assets_dir=args.shared_assets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(json.dumps({"output": str(args.out), "usableCaptures": len(registry)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
