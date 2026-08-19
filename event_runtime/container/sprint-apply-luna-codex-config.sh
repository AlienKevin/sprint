#!/usr/bin/env bash
# Install a static Luna catalog copied from the exact pinned Codex release.
# This prevents a model refresh from changing tool/collaboration semantics and
# routes the pinned OpenRouter/OpenAI preset through the durable cost proxy.
set -euo pipefail

CODEX_HOME_DIR=${CODEX_HOME:-/tmp/codex-home}
LOCK_SRC=${SPRINT_CODEX_LUNA_MODEL_LOCK:-/opt/sprint-codex-luna-model-lock.json}
TEMPLATE_SRC=${SPRINT_CODEX_DEEPSEEK_MODELS_JSON:-/opt/sprint-codex-deepseek-models.json}
MODEL_SLUG=${SPRINT_CODEX_LUNA_MODEL:-@preset/sprint-gpt-5-6-luna-openai-standard}
BASE_URL=${SPRINT_CODEX_LUNA_BASE_URL:-http://127.0.0.1:18080/api/v1}
CATALOG_PATH="$CODEX_HOME_DIR/models.json"
CONFIG_PATH="$CODEX_HOME_DIR/config.toml"

for required in "$LOCK_SRC" "$TEMPLATE_SRC"; do
  if [[ ! -f "$required" ]]; then
    echo "pinned Codex model input missing: $required" >&2
    exit 1
  fi
done

umask 077
mkdir -p "$CODEX_HOME_DIR"
python3 - "$CONFIG_PATH" "$CATALOG_PATH" "$LOCK_SRC" "$TEMPLATE_SRC" \
  "$MODEL_SLUG" "$BASE_URL" <<'PY'
import hashlib
import json
import pathlib
import re
import sys

config_path = pathlib.Path(sys.argv[1])
catalog_path = pathlib.Path(sys.argv[2])
lock_path = pathlib.Path(sys.argv[3])
template_path = pathlib.Path(sys.argv[4])
model_slug = sys.argv[5]
base_url = sys.argv[6]

lock = json.loads(lock_path.read_text(encoding="utf-8"))
if lock.get("codex_version") != "0.147.0":
    raise SystemExit("Luna catalog lock must target Codex 0.147.0")
model = lock.get("model")
if not isinstance(model, dict) or model.get("slug") != "gpt-5.6-luna":
    raise SystemExit("Luna catalog lock is missing gpt-5.6-luna")
if model.get("tool_mode") != "code_mode_only" or model.get("multi_agent_version") != "v1":
    raise SystemExit("Luna catalog lock does not preserve the comparison contract")

template_catalog = json.loads(template_path.read_text(encoding="utf-8"))
template = next(
    (item for item in template_catalog.get("models", []) if item.get("slug") == "deepseek-v4-flash"),
    None,
)
messages = template.get("model_messages") if isinstance(template, dict) else None
if not isinstance(messages, dict):
    raise SystemExit("Codex instruction template is missing from the pinned catalog")
canonical = json.dumps(messages, sort_keys=True, separators=(",", ":")).encode()
if hashlib.sha256(canonical).hexdigest() != lock.get("model_messages_sha256"):
    raise SystemExit("Codex instruction template does not match the Luna lock")

locked_model = dict(model)
locked_model["slug"] = model_slug
locked_model["model_messages"] = messages
catalog_path.write_text(
    json.dumps({"models": [locked_model]}, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

raw = config_path.read_text(encoding="utf-8") if config_path.exists() else ""

def key_of(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped or stripped.startswith("["):
        return None
    return stripped.split("=", 1)[0].strip().strip("\"'")

out = []
skip_section = False
for line in raw.splitlines():
    stripped = line.strip()
    if stripped.startswith("["):
        header = stripped.strip("[]").strip().strip("\"'")
        skip_section = header == "model_providers.sprint_openrouter" or header.startswith(
            "model_providers.sprint_openrouter."
        )
        if skip_section:
            continue
        out.append(line)
        continue
    if skip_section:
        continue
    if key_of(line) in {
        "model",
        "model_catalog_json",
        "model_provider",
        "preferred_auth_method",
        "forced_login_method",
        "openai_base_url",
    }:
        continue
    out.append(line)

leading = [
    f'model = "{model_slug}"',
    'model_provider = "sprint_openrouter"',
    'preferred_auth_method = "apikey"',
    'forced_login_method = "api"',
    f'model_catalog_json = "{catalog_path}"',
]
provider = [
    "[model_providers.sprint_openrouter]",
    'name = "sprint_openrouter"',
    f'base_url = "{base_url}"',
    'wire_api = "responses"',
    'env_key = "OPENAI_API_KEY"',
]
body = "\n".join(out).strip("\n")
parts = ["\n".join(leading), "\n".join(provider)]
if body:
    parts.append(body)
text = "\n\n".join(parts).rstrip() + "\n"
config_path.write_text(re.sub(r"\n{3,}", "\n\n", text), encoding="utf-8")
PY

echo "Applied pinned Luna Codex 0.147.0 catalog in $CODEX_HOME_DIR" >&2
