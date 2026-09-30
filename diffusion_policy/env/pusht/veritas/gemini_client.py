"""Gemini API client for structured outputs.

Ported from veritas/src/agent/utils/gemini_client.py, keeping only what the waypoint-plan
call reaches. Changed: ``_save_vlm_log`` takes an explicit filename, since the original's
per-second timestamp let parallel episodes overwrite each other.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Callable, Optional, Type, TypeVar

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

T = TypeVar("T")

try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    log.warning("google-genai not available. Install with: pip install google-genai")


class GeminiClient:
    """Client for Gemini API with structured outputs."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Args:
            api_key: Gemini API key (default: GEMINI_API_KEY env var)
            model_name: Model name (default: GEMINI_MODEL env var, then "gemini-3.8-flash";
                Veritas's gemini-2.5-flash is closed to new API users)
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai not installed. Install with: pip install google-genai")

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set. Provide api_key or set GEMINI_API_KEY environment variable.")

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        self.last_usage = None
        log.info(f"Initialized Gemini client with model: {self.model_name}")

    def _generate(self, parts, generation_config: dict):
        cfg = genai_types.GenerateContentConfig(
            response_mime_type=generation_config.get("response_mime_type"),
            response_schema=generation_config.get("response_schema"),
        )
        return self.client.models.generate_content(
            model=self.model_name,
            contents=parts,
            config=cfg,
        )

    def _image_to_part(self, image: np.ndarray):
        """[H, W, 3] uint8 RGB -> PIL image (Gemini accepts PIL directly). No resize."""
        return Image.fromarray(image)

    def _call_with_retry(
        self,
        parts,
        generation_config: dict,
        schema: Type[T],
        label: str = "GeminiCall",
        max_retries: int = 3,
        initial_retry_delay: float = 5.0,
        fallback_factory: Optional[Callable[[], T]] = None,
        on_success: Optional[Callable[[T, str], None]] = None,
    ) -> Optional[T]:
        """Shared retry loop for Gemini API calls with rate-limit handling.

        Args:
            parts: Content parts to send to Gemini.
            generation_config: Generation config dict.
            schema: Pydantic model class used to validate the response.
            label: Human-readable label for log messages.
            max_retries: Maximum number of attempts.
            initial_retry_delay: Starting delay in seconds for exponential backoff.
            fallback_factory: Callable returning a fallback instance on total failure.
                If *None*, returns *None* on failure.
            on_success: Optional callback ``(parsed, response_text) -> None``
                invoked after successful parse (e.g. for logging or saving).

        Returns:
            Parsed schema instance, fallback, or *None*.
        """
        retry_delay = initial_retry_delay

        for attempt in range(max_retries):
            try:
                response = self._generate(parts, generation_config)
                response_text = response.text
                # token counts of the successful attempt (thoughts_token_count included)
                usage = getattr(response, "usage_metadata", None)
                self.last_usage = (usage.model_dump(mode="json", exclude_none=True)
                                   if usage is not None else None)
                log.debug(f"[{label}] response: {response_text}")

                parsed = schema.model_validate_json(response_text)

                if on_success is not None:
                    on_success(parsed, response_text)

                return parsed

            except Exception as e:
                error_str = str(e)
                # 503 "high demand" is added to Veritas's list: retried with no delay, all
                # attempts land in the same overload spike.
                is_rate_limit = (
                    "429" in error_str
                    or "quota" in error_str.lower()
                    or "rate limit" in error_str.lower()
                    or "Please retry" in error_str
                    or "503" in error_str
                    or "UNAVAILABLE" in error_str
                )
                # Truncated / malformed JSON (e.g. max_output_tokens hit mid-output) is
                # transient: a fresh call usually parses fine.
                is_parse_error = (
                    "json_invalid" in error_str
                    or "Invalid JSON" in error_str
                    or "EOF while parsing" in error_str
                    or "JSONDecodeError" in type(e).__name__
                )

                if is_rate_limit and attempt < max_retries - 1:
                    retry_match = re.search(
                        r"retry in ([\d.]+)s", error_str, re.IGNORECASE
                    )
                    if retry_match:
                        retry_delay = float(retry_match.group(1)) + 1.0
                    else:
                        retry_delay *= 2
                    log.warning(
                        f"[{label}] rate limited (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {retry_delay:.1f}s..."
                    )
                    time.sleep(retry_delay)
                    continue
                elif is_parse_error and attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
                    continue
                else:
                    log.error(f"[{label}] call failed: {e}", exc_info=True)
                    if attempt + 1 == max_retries:
                        break

        if fallback_factory is not None:
            log.warning(f"[{label}] Using fallback response")
            return fallback_factory()
        return None

    @staticmethod
    def _save_vlm_log(text: str, log_dir: Optional[str], name: str) -> None:
        """Persist raw VLM response text to ``<log_dir>/<name>.json``."""
        if not log_dir:
            return
        try:
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, f"{name}.json"), "w") as f:
                f.write(text)
        except Exception as e:
            log.warning(f"Failed to save VLM log: {e}")
