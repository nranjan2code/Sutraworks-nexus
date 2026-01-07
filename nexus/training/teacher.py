"""
Nexus Teacher Module
====================

Interfaces with a stronger "Teacher" model (e.g., Ollama) to guide Nexus's learning.
Uses teacher-student distillation: Nexus learns to mimic the teacher's reasoning.
"""

import requests
import logging
from typing import Optional, Dict

logger = logging.getLogger("nexus.teacher")


class OllamaTeacher:
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize Ollama Teacher.

        Args:
            base_url: URL of Ollama server. Defaults to env OLLAMA_HOST or localhost.
            model: Model name. Defaults to env OLLAMA_MODEL or llama2.
        """
        import os

        # Priority: explicit arg > env var > default
        self.base_url = (base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

        self.model = model or os.getenv("OLLAMA_MODEL", "llama2")

        self.available = False
        self._check_availability()

    def _check_availability(self):
        """Check if Ollama is reachable."""
        try:
            res = requests.get(f"{self.base_url}/api/tags", timeout=1.0)
            if res.status_code == 200:
                self.available = True
                logger.info(f"Connected to Ollama Teacher at {self.base_url}")
            else:
                logger.warning(f"Ollama reachable but returned {res.status_code}")
        except requests.exceptions.RequestException:
            logger.warning(
                f"Ollama Teacher not found at {self.base_url}. Learning will be self-supervised only."
            )
            self.available = False

    def ask(self, prompt: str) -> Optional[str]:
        """Ask the teacher for an answer."""
        if not self.available:
            return None

        try:
            payload = {"model": self.model, "prompt": prompt, "stream": False}
            res = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=30)
            if res.status_code == 200:
                return res.json().get("response")
            return None
        except Exception as e:
            logger.error(f"Error asking teacher: {e}")
            return None

    def generate_synthetic_example(self, topic: Optional[str] = None) -> Optional[Dict[str, str]]:
        """
        Ask teacher to generate a training example (Prompt + Answer).
        Returns: {"prompt": str, "response": str}
        """
        if not self.available:
            return None

        if topic:
            context = f"about '{topic}'"
        else:
            context = "about science, history, or philosophy"

        meta_prompt = (
            f"Generate a random, interesting question {context}. "
            "Then provide a concise, accurate answer. "
            "Format: Question: <question>\\nAnswer: <answer>"
        )

        raw = self.ask(meta_prompt)
        if not raw:
            return None

        # Basic parsing
        try:
            parts = raw.split("Answer:")
            if len(parts) == 2:
                q = parts[0].replace("Question:", "").strip()
                a = parts[1].strip()
                return {"prompt": q, "response": a}
        except Exception:
            pass

        return None
