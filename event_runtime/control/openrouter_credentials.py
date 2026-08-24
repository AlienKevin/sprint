#!/usr/bin/env python3
"""Provision and audit experiment-scoped OpenRouter credentials.

Only a management key may call this module's control-plane endpoints. Plaintext
child keys exist in memory just long enough to launch their owning trial and are
never written to the batch manifest or journal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


API_ROOT = "https://openrouter.ai/api/v1"
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,80}$")
READ_RETRY_ATTEMPTS = 4
READ_RETRY_INITIAL_SECONDS = 0.5
READ_RETRY_MAX_SECONDS = 10.0
KEY_LIST_PAGE_SIZE = 100
KEY_LIST_MAX_PAGES = 100


class OpenRouterManagementError(RuntimeError):
    """A sanitized OpenRouter management-plane failure."""


@dataclass(frozen=True)
class TrialCredentialSpec:
    run_id: str
    model: str
    resolved_model: str
    provider: str
    budget_usd: float = 10.0


@dataclass
class ProvisionedTrialCredential:
    run_id: str
    model: str
    resolved_model: str
    provider: str
    budget_usd: float
    key_hash: str
    key_name: str
    guardrail_id: str
    guardrail_name: str
    expires_at: str
    api_key: str = field(repr=False)

    def public_metadata(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model": self.model,
            "resolved_model": self.resolved_model,
            "provider": self.provider,
            "budget_usd": self.budget_usd,
            "key_hash": self.key_hash,
            "key_name": self.key_name,
            "guardrail_id": self.guardrail_id,
            "guardrail_name": self.guardrail_name,
            "expires_at": self.expires_at,
        }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class OpenRouterManagementClient:
    def __init__(self, management_key: str, *, api_root: str = API_ROOT) -> None:
        if len(management_key) < 16:
            raise ValueError("OPENROUTER_MANAGEMENT_KEY is missing or too short")
        parsed = urllib.parse.urlsplit(api_root)
        if parsed.scheme != "https" or parsed.hostname != "openrouter.ai":
            raise ValueError("OpenRouter management API root must be official HTTPS")
        self._key = management_key
        self._api_root = api_root.rstrip("/")

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        expected: tuple[int, ...] = (200,),
    ) -> dict[str, Any]:
        if not path.startswith("/") or ".." in path:
            raise ValueError("unsafe OpenRouter management API path")
        method = method.upper()
        body = None if payload is None else json.dumps(payload).encode()
        # Only idempotent reads are retried. Provisioning and revocation calls
        # must never be replayed after an ambiguous network response.
        attempts = READ_RETRY_ATTEMPTS if method == "GET" and body is None else 1
        result: Any = None
        status = 0
        for attempt in range(attempts):
            request = urllib.request.Request(
                self._api_root + path,
                data=body,
                method=method,
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Accept": "application/json",
                    **(
                        {"Content-Type": "application/json"}
                        if body is not None
                        else {}
                    ),
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    status = int(response.status)
                    raw = response.read()
                    result = json.loads(raw) if raw else {}
                break
            except urllib.error.HTTPError as exc:
                if exc.code in expected:
                    exc.read()
                    return {}
                message = ""
                try:
                    error = json.load(exc).get("error", {})
                    message = str(error.get("message") or error.get("code") or "")
                except Exception:  # noqa: BLE001 - do not expose raw body
                    pass
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if retryable and attempt + 1 < attempts:
                    retry_after = None
                    try:
                        retry_after = float(exc.headers.get("Retry-After", ""))
                    except (AttributeError, TypeError, ValueError):
                        pass
                    delay = min(
                        retry_after
                        if retry_after is not None and retry_after >= 0
                        else READ_RETRY_INITIAL_SECONDS * (2**attempt),
                        READ_RETRY_MAX_SECONDS,
                    )
                    time.sleep(delay)
                    continue
                suffix = f": {' '.join(message.split())[:300]}" if message else ""
                raise OpenRouterManagementError(
                    f"OpenRouter management {method} {path} failed: "
                    f"HTTP {exc.code}{suffix}"
                ) from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt + 1 < attempts:
                    time.sleep(
                        min(
                            READ_RETRY_INITIAL_SECONDS * (2**attempt),
                            READ_RETRY_MAX_SECONDS,
                        )
                    )
                    continue
                raise OpenRouterManagementError(
                    f"OpenRouter management {method} {path} failed after "
                    f"{attempts} attempts: {type(exc).__name__}"
                ) from exc
        if status not in expected or not isinstance(result, dict):
            raise OpenRouterManagementError(
                f"OpenRouter management {method} {path} returned HTTP {status}"
            )
        return result

    def create_guardrail(self, spec: TrialCredentialSpec, name: str) -> dict[str, Any]:
        result = self.request(
            "POST",
            "/guardrails",
            {
                "name": name,
                "description": f"Ephemeral sealed eval guardrail for {spec.run_id}",
                "allowed_models": [spec.model],
                "allowed_providers": [spec.provider],
                "ignored_models": None,
                "ignored_providers": None,
                "limit_usd": spec.budget_usd,
                "reset_interval": "monthly",
                # Do not silently change the benchmark's account-level privacy
                # policy. Any stricter workspace rule still wins by intersection.
                "enforce_zdr_anthropic": False,
                "enforce_zdr_google": False,
                "enforce_zdr_openai": False,
                "enforce_zdr_other": False,
            },
            expected=(201,),
        )
        data = result.get("data")
        if not isinstance(data, dict) or not data.get("id"):
            raise OpenRouterManagementError("created guardrail response lacked an ID")
        return data

    def create_key(
        self, spec: TrialCredentialSpec, name: str, expires_at: str
    ) -> tuple[dict[str, Any], str]:
        result = self.request(
            "POST",
            "/keys",
            {
                "name": name,
                "expires_at": expires_at,
                "include_byok_in_limit": True,
                "limit": spec.budget_usd,
                "limit_reset": None,
            },
            expected=(201,),
        )
        data = result.get("data")
        key = result.get("key")
        if not isinstance(data, dict) or not data.get("hash"):
            raise OpenRouterManagementError("created API key response lacked a hash")
        if not isinstance(key, str) or len(key) < 16:
            raise OpenRouterManagementError(
                "created API key response lacked plaintext key"
            )
        return data, key

    def assign_key(self, guardrail_id: str, key_hash: str) -> None:
        result = self.request(
            "POST",
            f"/guardrails/{urllib.parse.quote(guardrail_id, safe='')}/assignments/keys",
            {"key_hashes": [key_hash]},
        )
        if result.get("assigned_count") != 1:
            raise OpenRouterManagementError("guardrail did not assign exactly one key")

    def verify(self, credential: ProvisionedTrialCredential) -> None:
        guardrail_id = urllib.parse.quote(credential.guardrail_id, safe="")
        key_hash = urllib.parse.quote(credential.key_hash, safe="")
        guardrail = self.request("GET", f"/guardrails/{guardrail_id}").get("data")
        key = self.request("GET", f"/keys/{key_hash}").get("data")
        assignments = self.request(
            "GET", f"/guardrails/{guardrail_id}/assignments/keys?limit=100"
        ).get("data")
        if not isinstance(guardrail, dict) or not isinstance(key, dict):
            raise OpenRouterManagementError("guardrail/key verification was incomplete")
        if guardrail.get("allowed_models") != [credential.resolved_model]:
            raise OpenRouterManagementError("guardrail model allowlist mismatch")
        if guardrail.get("allowed_providers") != [credential.provider]:
            raise OpenRouterManagementError("guardrail provider allowlist mismatch")
        try:
            guardrail_limit = float(guardrail.get("limit_usd"))
            key_limit = float(key.get("limit"))
        except (TypeError, ValueError) as exc:
            raise OpenRouterManagementError("guardrail/key budget was invalid") from exc
        if not math.isclose(guardrail_limit, credential.budget_usd, abs_tol=1e-12):
            raise OpenRouterManagementError("guardrail budget mismatch")
        if not math.isclose(key_limit, credential.budget_usd, abs_tol=1e-12):
            raise OpenRouterManagementError("API key budget mismatch")
        if key.get("limit_reset") is not None or key.get("disabled") is True:
            raise OpenRouterManagementError("API key reset/disabled state mismatch")
        hashes = {
            row.get("key_hash") for row in assignments or [] if isinstance(row, dict)
        }
        if credential.key_hash not in hashes:
            raise OpenRouterManagementError("guardrail key assignment is missing")

    def key_usage(self, key_hash: str) -> dict[str, Any]:
        data = self.request(
            "GET", f"/keys/{urllib.parse.quote(key_hash, safe='')}"
        ).get("data")
        if not isinstance(data, dict):
            raise OpenRouterManagementError("API key usage response was incomplete")
        return data

    def keys_usage(self, key_hashes: set[str]) -> dict[str, dict[str, Any]]:
        """Read a coherent usage snapshot for a batch of ephemeral keys."""
        wanted = {str(key_hash) for key_hash in key_hashes if key_hash}
        if not wanted:
            return {}
        found: dict[str, dict[str, Any]] = {}
        offset = 0
        for _page in range(KEY_LIST_MAX_PAGES):
            query = urllib.parse.urlencode(
                {"include_disabled": "true", "offset": offset}
            )
            data = self.request("GET", f"/keys?{query}").get("data")
            if not isinstance(data, list):
                raise OpenRouterManagementError(
                    "API key list usage response was incomplete"
                )
            for row in data:
                if not isinstance(row, dict):
                    continue
                key_hash = row.get("hash")
                if isinstance(key_hash, str) and key_hash in wanted:
                    found[key_hash] = row
            if found.keys() >= wanted:
                return found
            if len(data) < KEY_LIST_PAGE_SIZE:
                break
            offset += len(data)
        missing = len(wanted - found.keys())
        raise OpenRouterManagementError(
            f"API key list usage response omitted {missing} requested key(s)"
        )

    def generation_usage(self, generation_id: str) -> dict[str, Any]:
        """Return OpenRouter's authoritative audit for one billed generation."""
        if not generation_id or len(generation_id) > 200:
            raise ValueError("invalid OpenRouter generation ID")
        query = urllib.parse.urlencode({"id": generation_id})
        data = self.request("GET", f"/generation?{query}").get("data")
        if not isinstance(data, dict):
            raise OpenRouterManagementError(
                "generation usage response was incomplete"
            )
        return data

    def verify_denied_inference(
        self,
        credential: ProvisionedTrialCredential,
        *,
        model: str,
        provider: str,
    ) -> None:
        """Prove a sealed child key cannot use a forbidden route."""
        request = urllib.request.Request(
            self._api_root + "/responses",
            data=json.dumps(
                {
                    "model": model,
                    "input": "Return OK.",
                    "max_output_tokens": 1,
                    "provider": {
                        "only": [provider],
                        "allow_fallbacks": False,
                    },
                }
            ).encode(),
            method="POST",
            headers={
                "Authorization": f"Bearer {credential.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                status = int(response.status)
                response.read()
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            exc.read(4096)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise OpenRouterManagementError(
                "child-key negative route check was unavailable"
            ) from exc
        if status not in {400, 403, 404}:
            raise OpenRouterManagementError(
                f"child key accepted or ambiguously handled forbidden route: HTTP {status}"
            )

    def delete_key(self, key_hash: str) -> None:
        self.request(
            "DELETE",
            f"/keys/{urllib.parse.quote(key_hash, safe='')}",
            expected=(200, 204, 404),
        )

    def delete_guardrail(self, guardrail_id: str) -> None:
        self.request(
            "DELETE",
            f"/guardrails/{urllib.parse.quote(guardrail_id, safe='')}",
            expected=(200, 204, 404),
        )


def validate_specs(specs: list[TrialCredentialSpec]) -> None:
    if not specs:
        raise ValueError("at least one trial credential spec is required")
    seen: set[str] = set()
    for spec in specs:
        if not SAFE_NAME.fullmatch(spec.run_id) or spec.run_id in seen:
            raise ValueError(f"unsafe or duplicate run ID: {spec.run_id!r}")
        seen.add(spec.run_id)
        if (
            "/" not in spec.model
            or "/" not in spec.resolved_model
            or not SAFE_NAME.fullmatch(spec.provider)
        ):
            raise ValueError(f"invalid OpenRouter route for {spec.run_id}")
        if not math.isfinite(spec.budget_usd) or spec.budget_usd <= 0:
            raise ValueError(f"invalid trial budget for {spec.run_id}")


def provision_trial_credentials(
    client: OpenRouterManagementClient,
    specs: list[TrialCredentialSpec],
    *,
    journal_path: Path,
    lifetime_hours: int = 48,
) -> list[ProvisionedTrialCredential]:
    validate_specs(specs)
    if not 1 <= lifetime_hours <= 168:
        raise ValueError("credential lifetime must be between 1 and 168 hours")
    expires_at = (
        (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=lifetime_hours))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    provisioned: list[ProvisionedTrialCredential] = []
    journal: dict[str, Any] = {
        "schema_version": 1,
        "status": "provisioning",
        "expires_at": expires_at,
        "credentials": [],
    }
    _atomic_json(journal_path, journal)
    try:
        for spec in specs:
            guardrail_name = f"sprint-{spec.run_id}-guardrail"
            key_name = f"sprint-{spec.run_id}-key"
            guardrail = client.create_guardrail(spec, guardrail_name)
            journal["credentials"].append(
                {
                    "run_id": spec.run_id,
                    "model": spec.model,
                    "resolved_model": spec.resolved_model,
                    "provider": spec.provider,
                    "budget_usd": spec.budget_usd,
                    "guardrail_id": str(guardrail["id"]),
                    "guardrail_name": guardrail_name,
                    "key_hash": None,
                    "key_name": key_name,
                    "expires_at": expires_at,
                }
            )
            _atomic_json(journal_path, journal)
            key_data, api_key = client.create_key(spec, key_name, expires_at)
            credential = ProvisionedTrialCredential(
                run_id=spec.run_id,
                model=spec.model,
                resolved_model=spec.resolved_model,
                provider=spec.provider,
                budget_usd=spec.budget_usd,
                key_hash=str(key_data["hash"]),
                key_name=key_name,
                guardrail_id=str(guardrail["id"]),
                guardrail_name=guardrail_name,
                expires_at=expires_at,
                api_key=api_key,
            )
            journal["credentials"][-1] = credential.public_metadata()
            _atomic_json(journal_path, journal)
            client.assign_key(credential.guardrail_id, credential.key_hash)
            client.verify(credential)
            forbidden_model = (
                "openai/gpt-5.6-luna"
                if spec.model != "openai/gpt-5.6-luna"
                else "deepseek/deepseek-v4-flash-vision-exp"
            )
            forbidden_provider = "openai" if spec.provider != "openai" else "deepseek"
            client.verify_denied_inference(
                credential,
                model=forbidden_model,
                provider=spec.provider,
            )
            client.verify_denied_inference(
                credential,
                model=spec.model,
                provider=forbidden_provider,
            )
            provisioned.append(credential)
        journal["status"] = "active"
        _atomic_json(journal_path, journal)
        return provisioned
    except Exception:
        revoke_trial_credentials(
            client, journal["credentials"], journal_path=journal_path
        )
        raise


def revoke_trial_credentials(
    client: OpenRouterManagementClient,
    credentials: list[dict[str, Any]],
    *,
    journal_path: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    for row in credentials:
        key_hash = row.get("key_hash")
        guardrail_id = row.get("guardrail_id")
        if key_hash:
            try:
                client.delete_key(str(key_hash))
            except OpenRouterManagementError as exc:
                errors.append(f"{row.get('run_id')}: key cleanup failed: {exc}")
        if guardrail_id:
            try:
                client.delete_guardrail(str(guardrail_id))
            except OpenRouterManagementError as exc:
                errors.append(f"{row.get('run_id')}: guardrail cleanup failed: {exc}")
    if journal_path is not None:
        _atomic_json(
            journal_path,
            {
                "schema_version": 1,
                "status": "cleanup_error" if errors else "revoked",
                "credentials": credentials,
                "errors": errors,
            },
        )
    return errors
