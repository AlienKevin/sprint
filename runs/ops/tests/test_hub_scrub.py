#!/usr/bin/env python3
"""Unit tests for Hub scrub + leak-gate (no network, no real secrets)."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HUB = ROOT / "runs/hub_track_upload.py"


def load_hub():
    spec = importlib.util.spec_from_file_location("hub_track_upload", HUB)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["hub_track_upload"] = mod
    spec.loader.exec_module(mod)
    return mod


class HubScrubTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hub = load_hub()

    def test_scrub_redacts_key_prefixes(self):
        sample = "\n".join(
            [
                "OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz0123456789",
                "Authorization: Bearer sk-ant-oat01-abcdefghijklmnopqrstuvwxyz",
                "KIMI_MODEL_API_KEY=wk-abc123.ws-secretvaluehere",
                "url=https://user:hunter2@api.example.com/v1",
                "plain sk-abcdefghijklmnopqrstuvwxyz012345",
            ]
        )
        out, counts = self.hub.scrub_text(sample, set())
        self.assertGreater(sum(counts.values()), 0)
        self.assertNotIn("sk-proj-", out)
        self.assertNotIn("sk-ant-oat01-", out)
        self.assertNotIn("wk-abc123.ws-", out)
        self.assertNotIn("user:hunter2@", out)
        self.assertIn("[REDACTED", out)

    def test_leak_gate_fails_on_residual(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "leak.txt").write_text(
                "still has sk-abcdefghijklmnopqrstuvwxyz012345\n", encoding="utf-8"
            )
            clean, msg = self.hub.assert_scrubbed_clean(root)
            self.assertFalse(clean)
            self.assertIn("FAIL", msg)
            # Message must not echo the secret body.
            self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz012345", msg)

    def test_leak_gate_passes_clean_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ok.txt").write_text(
                "OPENAI_API_KEY=[REDACTED]\nAuthorization: Bearer [REDACTED_BEARER]\n",
                encoding="utf-8",
            )
            clean, msg = self.hub.assert_scrubbed_clean(root)
            self.assertTrue(clean, msg)


if __name__ == "__main__":
    unittest.main()
