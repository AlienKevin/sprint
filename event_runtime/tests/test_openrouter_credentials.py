from __future__ import annotations

import json
from pathlib import Path
import urllib.error
from unittest import mock

import pytest

from event_runtime.control.openrouter_credentials import (
    OpenRouterManagementClient,
    OpenRouterManagementError,
    TrialCredentialSpec,
    provision_trial_credentials,
    revoke_trial_credentials,
    validate_specs,
)


class FakeManagementClient:
    def __init__(self, *, fail_run: str | None = None) -> None:
        self.fail_run = fail_run
        self.guardrails: dict[str, dict[str, object]] = {}
        self.keys: dict[str, dict[str, object]] = {}
        self.assignments: dict[str, set[str]] = {}
        self.deleted_keys: list[str] = []
        self.deleted_guardrails: list[str] = []
        self.denied_checks: list[tuple[str, str, str]] = []

    def create_guardrail(self, spec: TrialCredentialSpec, name: str):
        guardrail_id = f"gr-{spec.run_id}"
        self.guardrails[guardrail_id] = {
            "id": guardrail_id,
            "name": name,
            "model": spec.model,
            "provider": spec.provider,
            "limit": spec.budget_usd,
        }
        return {"id": guardrail_id}

    def create_key(self, spec: TrialCredentialSpec, name: str, expires_at: str):
        if spec.run_id == self.fail_run:
            raise OpenRouterManagementError("injected key failure")
        key_hash = f"hash-{spec.run_id}"
        plaintext = f"sk-or-v1-plaintext-{spec.run_id}"
        self.keys[key_hash] = {
            "hash": key_hash,
            "name": name,
            "expires_at": expires_at,
            "limit": spec.budget_usd,
        }
        return {"hash": key_hash}, plaintext

    def assign_key(self, guardrail_id: str, key_hash: str) -> None:
        self.assignments.setdefault(guardrail_id, set()).add(key_hash)

    def verify(self, credential) -> None:
        assert credential.key_hash in self.assignments[credential.guardrail_id]

    def verify_denied_inference(self, credential, *, model: str, provider: str) -> None:
        assert model != credential.model or provider != credential.provider
        self.denied_checks.append((credential.run_id, model, provider))

    def delete_key(self, key_hash: str) -> None:
        self.deleted_keys.append(key_hash)

    def delete_guardrail(self, guardrail_id: str) -> None:
        self.deleted_guardrails.append(guardrail_id)


def specs() -> list[TrialCredentialSpec]:
    return [
        TrialCredentialSpec(
            run_id="eval-deepseek-1",
            model="deepseek/deepseek-v4-flash-vision-exp",
            resolved_model="deepseek/deepseek-v4-flash-vision-exp-20260821",
            provider="deepseek",
        ),
        TrialCredentialSpec(
            run_id="eval-luna-1",
            model="openai/gpt-5.6-luna",
            resolved_model="openai/gpt-5.6-luna-20260709",
            provider="openai",
        ),
    ]


def test_validate_specs_accepts_official_provider_variant_tag() -> None:
    validate_specs(
        [
            TrialCredentialSpec(
                run_id="eval-glm-1",
                model="z-ai/glm-5.3-flash",
                resolved_model="z-ai/glm-5.3-flash-20260826",
                provider="z-ai/fp8",
            )
        ]
    )


@pytest.mark.parametrize(
    "provider",
    ("/fp8", "z-ai/", "z-ai/fp8/extra", "z-ai//fp8", "z-ai/fp 8"),
)
def test_validate_specs_rejects_malformed_provider_variant_tag(provider: str) -> None:
    with pytest.raises(ValueError, match="invalid OpenRouter route"):
        validate_specs(
            [
                TrialCredentialSpec(
                    run_id="eval-glm-1",
                    model="z-ai/glm-5.3-flash",
                    resolved_model="z-ai/glm-5.3-flash-20260826",
                    provider=provider,
                )
            ]
        )


def test_provision_seals_each_trial_and_never_journals_plaintext(
    tmp_path: Path,
) -> None:
    client = FakeManagementClient()
    journal = tmp_path / "credentials.json"

    credentials = provision_trial_credentials(
        client, specs(), journal_path=journal, lifetime_hours=24
    )

    assert len(credentials) == 2
    assert len({row.key_hash for row in credentials}) == 2
    assert {row.budget_usd for row in credentials} == {10.0}
    assert len(client.denied_checks) == 4
    payload = json.loads(journal.read_text())
    assert payload["status"] == "active"
    assert journal.stat().st_mode & 0o777 == 0o600
    assert "sk-or-v1-plaintext" not in journal.read_text()
    assert all("api_key" not in row for row in payload["credentials"])


def test_partial_provision_rolls_back_every_created_resource(tmp_path: Path) -> None:
    client = FakeManagementClient(fail_run="eval-luna-1")
    journal = tmp_path / "credentials.json"

    with pytest.raises(OpenRouterManagementError, match="injected"):
        provision_trial_credentials(client, specs(), journal_path=journal)

    payload = json.loads(journal.read_text())
    assert payload["status"] == "revoked"
    assert client.deleted_keys == ["hash-eval-deepseek-1"]
    assert set(client.deleted_guardrails) == {
        "gr-eval-deepseek-1",
        "gr-eval-luna-1",
    }


def test_revoke_is_complete_and_journaled(tmp_path: Path) -> None:
    client = FakeManagementClient()
    journal = tmp_path / "credentials.json"
    credentials = provision_trial_credentials(client, specs(), journal_path=journal)

    errors = revoke_trial_credentials(
        client,
        [credential.public_metadata() for credential in credentials],
        journal_path=journal,
    )

    assert errors == []
    assert len(client.deleted_keys) == 2
    assert len(client.deleted_guardrails) == 2
    assert json.loads(journal.read_text())["status"] == "revoked"


def test_bulk_key_usage_reads_one_coherent_snapshot() -> None:
    client = OpenRouterManagementClient("m" * 32)
    calls: list[tuple[str, str]] = []

    def fake_request(method: str, path: str):
        calls.append((method, path))
        return {
            "data": [
                {"hash": "hash-1", "usage": 1.25},
                {"hash": "hash-2", "usage": 2.5},
            ]
        }

    client.request = fake_request  # type: ignore[method-assign]

    assert client.keys_usage({"hash-1", "hash-2"}) == {
        "hash-1": {"hash": "hash-1", "usage": 1.25},
        "hash-2": {"hash": "hash-2", "usage": 2.5},
    }
    assert calls == [("GET", "/keys?include_disabled=true&offset=0")]


def test_management_get_retries_transient_429_and_honors_retry_after() -> None:
    client = OpenRouterManagementClient("m" * 32)
    transient = urllib.error.HTTPError(
        "https://openrouter.ai/api/v1/keys",
        429,
        "rate limited",
        {"Retry-After": "0"},
        None,
    )
    response = mock.MagicMock()
    response.__enter__.return_value = response
    response.status = 200
    response.read.return_value = b'{"data": []}'

    with (
        mock.patch(
            "event_runtime.control.openrouter_credentials.urllib.request.urlopen",
            side_effect=[transient, response],
        ) as urlopen,
        mock.patch(
            "event_runtime.control.openrouter_credentials.time.sleep"
        ) as sleep,
    ):
        assert client.request("GET", "/keys") == {"data": []}

    assert urlopen.call_count == 2
    sleep.assert_called_once_with(0.0)


def test_management_mutation_is_never_retried() -> None:
    client = OpenRouterManagementClient("m" * 32)
    transient = urllib.error.HTTPError(
        "https://openrouter.ai/api/v1/keys",
        500,
        "server error",
        {},
        None,
    )
    with mock.patch(
        "event_runtime.control.openrouter_credentials.urllib.request.urlopen",
        side_effect=transient,
    ) as urlopen:
        with pytest.raises(OpenRouterManagementError, match="HTTP 500"):
            client.request("POST", "/keys", {"name": "probe"}, expected=(201,))
    assert urlopen.call_count == 1
