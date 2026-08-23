#!/usr/bin/env bash
# Rewrite $CODEX_HOME config to the audited DeepSeek Codex harness shape.
# See: https://api-docs.deepseek.com/quick_start/agent_integrations/codex/
#
# Harbor only appends openai_base_url; that routes Codex through the built-in
# OpenAI provider (~258k window + remote compaction). This script installs
# DeepSeek's models.json catalog (1M context) and [model_providers.deepseek]
# with wire_api=responses, then removes openai_base_url.
#
# Auth uses env_key=OPENAI_API_KEY (Harbor already injects the key). Do not
# write experimental_bearer_token; config.toml is copied into agent logs.
set -euo pipefail

CODEX_HOME_DIR=${CODEX_HOME:-/tmp/codex-home}
MODELS_SRC=${SPRINT_CODEX_DEEPSEEK_MODELS_JSON:-/opt/sprint-codex-deepseek-models.json}
BASE_URL=${SPRINT_CODEX_DEEPSEEK_BASE_URL:-https://openrouter.ai/api/v1}
MODEL_SLUG=${SPRINT_CODEX_DEEPSEEK_MODEL:-deepseek/deepseek-v4-flash-0731}
CONTEXT_WINDOW=${SPRINT_CODEX_DEEPSEEK_CONTEXT_WINDOW:-1048576}

if [[ ! -f "$MODELS_SRC" ]]; then
  echo "DeepSeek Codex models.json missing: $MODELS_SRC" >&2
  exit 1
fi

umask 077
mkdir -p "$CODEX_HOME_DIR"
cp -f -- "$MODELS_SRC" "$CODEX_HOME_DIR/models.json"

CONFIG_PATH="$CODEX_HOME_DIR/config.toml"
python3 - "$CONFIG_PATH" "$CODEX_HOME_DIR/models.json" "$BASE_URL" "$MODEL_SLUG" "$CONTEXT_WINDOW" <<'PY'
import pathlib
import json
import os
import re
import sys

config_path = pathlib.Path(sys.argv[1])
models_path = pathlib.Path(sys.argv[2])
base_url = sys.argv[3]
model_slug = sys.argv[4]
context_window = int(sys.argv[5])
if not 1_000 <= context_window <= 10_000_000:
    raise SystemExit("DeepSeek context window is invalid")

# The provider preset is the wire-level model identifier. Preserve every field
# from DeepSeek's official Codex catalog while making its slug match the model
# passed to Codex, otherwise Codex silently falls back to generic metadata.
catalog = json.loads(models_path.read_text(encoding="utf-8"))
models = catalog.get("models")
if not isinstance(models, list):
    raise SystemExit("DeepSeek model catalog has no model list")
flash_models = [model for model in models if model.get("slug") == "deepseek-v4-flash"]
if len(flash_models) != 1:
    raise SystemExit("DeepSeek model catalog must contain exactly one V4 Flash model")
selected = flash_models[0]
selected["slug"] = model_slug
selected["context_window"] = context_window
selected["max_context_window"] = context_window
# Codex must compact before input + the explicit provider completion allowance
# can exceed the endpoint's total context. Keep the benchmark's official
# completion ceiling, but reserve it when calculating the local compaction
# trigger. The extra margin covers tokenizer/accounting skew and tool framing.
contract_raw = os.environ.get("SPRINT_OPENROUTER_REQUEST_CONTRACT_JSON", "").strip()
contract = json.loads(contract_raw) if contract_raw else {}
if not isinstance(contract, dict):
    raise SystemExit("OpenRouter request contract must be a JSON object")
completion_reserve = int(
    contract.get("max_output_tokens") or contract.get("max_tokens") or 384000
)
context_safety = int(os.environ.get("SPRINT_CODEX_CONTEXT_SAFETY_TOKENS", "32768"))
auto_compact_limit = context_window - completion_reserve - context_safety
if completion_reserve <= 0 or context_safety < 0 or auto_compact_limit < 1000:
    raise SystemExit("DeepSeek completion reserve leaves no usable context window")
selected["auto_compact_token_limit"] = auto_compact_limit
if "pro" in model_slug.lower():
    selected["display_name"] = "DeepSeek-V4-Pro"
catalog["models"] = [selected]
models_path.write_text(
    json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)

raw = config_path.read_text(encoding="utf-8") if config_path.exists() else ""

# Drop keys that mask or contradict the official DeepSeek provider + catalog.
drop_keys = {
    "openai_base_url",
    "profile",
    "oss_provider",
    "model_context_window",
    "model_auto_compact_token_limit",
    "model_auto_compact_token_limit_scope",
    "base_instructions",
    "model_instructions_file",
    "compact_prompt",
    "experimental_compact_prompt_file",
    "service_tier",
    "model_verbosity",
    "model_reasoning_summary",
    "plan_mode_reasoning_effort",
    "experimental_use_unified_exec_tool",
}
# Keys we rewrite to the official values (keep Harbor/CLI effort if already set).
force_keys = {
    "model",
    "model_provider",
    "preferred_auth_method",
    "forced_login_method",
    "model_catalog_json",
}

def is_header(line: str) -> bool:
    return line.lstrip().startswith("[")

def key_of(line: str) -> str | None:
    s = line.strip()
    if not s or s.startswith("#") or "=" not in s or is_header(s):
        return None
    return s.split("=", 1)[0].strip().strip("\"'")

out: list[str] = []
skip_section = False
seen_effort = False
i = 0
lines = raw.splitlines()
while i < len(lines):
    line = lines[i]
    trimmed = line.strip()
    if trimmed.startswith("["):
        header = trimmed.strip("[]").strip().strip("\"'")
        skip_section = header == "model_providers.deepseek" or header.startswith(
            "model_providers.deepseek."
        )
        if skip_section:
            i += 1
            continue
        out.append(line)
        i += 1
        continue
    if skip_section:
        i += 1
        continue
    key = key_of(line)
    if key in drop_keys or key in force_keys:
        i += 1
        continue
    if key == "model_reasoning_effort":
        seen_effort = True
    out.append(line)
    i += 1

leading = [
    f'model = "{model_slug}"',
    'model_provider = "deepseek"',
    'preferred_auth_method = "apikey"',
    'forced_login_method = "api"',
    f'model_catalog_json = "{models_path}"',
]
if not seen_effort:
    # Bakeoff launchers pass -c model_reasoning_effort=max; default high matches docs.
    leading.append('model_reasoning_effort = "high"')

provider = [
    "",
    "[model_providers.deepseek]",
    'name = "deepseek"',
    f'base_url = "{base_url}"',
    'wire_api = "responses"',
    # Harbor already exports OPENAI_API_KEY into the Codex process.
    'env_key = "OPENAI_API_KEY"',
    "",
]

body = "\n".join(out).strip("\n")
parts = ["\n".join(leading), "\n".join(provider).rstrip()]
if body:
    parts.append(body)
text = "\n\n".join(parts).rstrip() + "\n"
# Avoid triple blank lines when provider already ends with a blank.
text = re.sub(r"\n{3,}", "\n\n", text)
config_path.write_text(text, encoding="utf-8")
PY

echo "Applied audited DeepSeek Codex config in $CODEX_HOME_DIR" >&2
