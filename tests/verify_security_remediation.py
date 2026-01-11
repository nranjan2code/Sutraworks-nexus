import unittest
import os
import shutil
from fastapi.testclient import TestClient

# Ensure env is set before import to ensure AuthManager picks it up
os.environ["NEXUS_API_KEY"] = "test-secret-key"

from nexus.service.server import app, update_env_file, auth_manager


class TestSecurityRemediation(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Ensure key is set in the instance used by app
        auth_manager.config.api_key = "test-secret-key"
        # Backup .env if exists
        if os.path.exists(".env"):
            shutil.copy(".env", ".env.bak")

    def tearDown(self):
        # Restore .env
        if os.path.exists(".env.bak"):
            shutil.move(".env.bak", ".env")
        else:
            if os.path.exists(".env"):
                os.remove(".env")

    def test_auth_enforcement(self):
        """Verify endpoints return 403/401 without key."""

        # Use a VALID body for config to avoid 400 if auth is bypassed
        valid_body = {"ollama_host": "http://localhost:11434", "ollama_model": "llama2"}

        endpoints = [
            ("/api/config", "POST", valid_body),
            ("/api/config", "GET", None),
            ("/api/control", "POST", {"action": "pause"}),
            ("/api/interact", "POST", {"prompt": "hi"}),
            ("/api/ollama/tags", "GET", None),
            ("/api/status", "GET", None),
        ]

        for ep, method, body in endpoints:
            if method == "POST":
                resp = self.client.post(ep, json=body)
            else:
                resp = self.client.get(ep)

            self.assertIn(
                resp.status_code,
                [401, 403],
                f"Endpoint {ep} should be protected (Got {resp.status_code})",
            )

    def test_ssrf_protection(self):
        """Verify unsafe URLs are rejected."""
        headers = {"X-API-Key": "test-secret-key"}

        # Test internal IP
        unsafe_hosts = ["http://192.168.1.1", "ftp://localhost", "file:///etc/passwd"]

        for host in unsafe_hosts:
            resp = self.client.post(
                "/api/config", json={"ollama_host": host, "ollama_model": "llama2"}, headers=headers
            )
            # Should fail validation or SSRF check
            self.assertEqual(resp.status_code, 400, f"Host {host} should be rejected")

    def test_env_safety(self):
        """Verify update_env_file preserves other keys."""
        with open(".env", "w") as f:
            f.write("SECRET_KEY=keep_me\nNEXUS_API_KEY=keep_me_too\n")

        update_env_file({"OLLAMA_HOST": "updated"})

        with open(".env", "r") as f:
            content = f.read()

        self.assertIn("SECRET_KEY=keep_me", content)
        self.assertIn("NEXUS_API_KEY=keep_me_too", content)
        self.assertIn("OLLAMA_HOST=updated", content)


if __name__ == "__main__":
    unittest.main()
