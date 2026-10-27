"""Gemini API client for structured outputs.

Ported from veritas/src/agent/utils/gemini_client.py, keeping only what the waypoint-plan
call reaches. Changed: ``_save_vlm_log`` takes an explicit filename, since the original's
per-second timestamp let parallel episodes overwrite each other.
"""

from __future__ import annotations

import glob
import logging
import os
import random
import re
import time
from typing import Callable, List, Optional, Tuple, Type, TypeVar

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


DEFAULT_KEY_GLOB = "~/.config/gemini/api_key*"
# backoff jitter, so cells rate-limited together do not retry together; a private RNG
# leaves the global one (and any eval seeding) untouched
_JITTER = random.Random()


def load_api_keys() -> List[Tuple[str, str]]:
    """``(label, key)`` for GEMINI_API_KEY, then each file matching GEMINI_KEY_GLOB
    (default ``~/.config/gemini/api_key*``, one key per file), deduplicated in order."""
    found = []
    if os.environ.get("GEMINI_API_KEY"):
        found.append(("GEMINI_API_KEY", os.environ["GEMINI_API_KEY"].strip()))
    pattern = os.path.expanduser(os.environ.get("GEMINI_KEY_GLOB", DEFAULT_KEY_GLOB))
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path) as f:
                found.append((os.path.basename(path), f.read().strip()))
        except OSError as e:
            log.warning(f"cannot read Gemini key file {path}: {e}")
    keys, seen = [], set()
    for label, key in found:
        if key and key not in seen:
            seen.add(key)
            keys.append((label, key))
    return keys


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

        # (label, key) pairs; the label (env var or file name) is what logs show, never the key
        self._keys = [("api_key", api_key)] if api_key else load_api_keys()
        if not self._keys:
            raise ValueError("GEMINI_API_KEY not set. Provide api_key, set GEMINI_API_KEY, or "
                             f"put keys in {DEFAULT_KEY_GLOB}.")
        self._clients = {}
        # spread parallel jobs over the keys: each starts on its own key
        seed = os.environ.get("SLURM_JOB_ID") or os.getpid()
        self._key_idx = int(seed) % len(self._keys)
        self.model_name = model_name or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        self.last_usage = None
        log.info(f"Initialized Gemini client with model: {self.model_name}, "
                 f"keys: {[label for label, _ in self._keys]}")

    @property
    def api_key(self) -> str:
        return self._keys[self._key_idx][1]

    @property
    def key_label(self) -> str:
        return self._keys[self._key_idx][0]

    @property
    def client(self):
        """The ``genai.Client`` of the current key (one per key, kept open)."""
        key = self.api_key
        if key not in self._clients:
            self._clients[key] = genai.Client(api_key=key)
        return self._clients[key]

    def _next_key(self) -> None:
        self._key_idx = (self._key_idx + 1) % len(self._keys)

    def _drop_current_key(self) -> None:
        self._clients.pop(self.api_key, None)
        del self._keys[self._key_idx]
        self._key_idx %= len(self._keys)

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
        # a rate-limited call first tries every other key at once; only when all keys are
        # limited does it count an attempt and back off
        attempt, keys_tried = 0, 1

        while attempt < max_retries:
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

                is_bad_key = ("401" in error_str or "UNAUTHENTICATED" in error_str
                              or "API_KEY_INVALID" in error_str or "PERMISSION_DENIED" in error_str)

                if is_bad_key and len(self._keys) > 1:
                    log.warning(f"[{label}] key {self.key_label} rejected ({error_str[:60]}); "
                                f"dropping it for this process")
                    self._drop_current_key()
                    continue
                if is_rate_limit and keys_tried < len(self._keys):
                    self._next_key()
                    keys_tried += 1
                    log.warning(f"[{label}] rate limited; switching to key {self.key_label}")
                    continue
                keys_tried = 1
                if len(self._keys) > 1:
                    self._next_key()
                if is_rate_limit and attempt < max_retries - 1:
                    retry_match = re.search(
                        r"retry in ([\d.]+)s", error_str, re.IGNORECASE
                    )
                    if retry_match:
                        retry_delay = float(retry_match.group(1)) + 1.0
                    else:
                        retry_delay *= 2
                    sleep_s = retry_delay * _JITTER.uniform(0.5, 1.5)
                    log.warning(
                        f"[{label}] rate limited (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {sleep_s:.1f}s..."
                    )
                    time.sleep(sleep_s)
                    attempt += 1
                    continue
                elif is_parse_error and attempt < max_retries - 1:
                    log.warning(f"[{label}] unparseable response (attempt {attempt + 1}/"
                                f"{max_retries}): {error_str[:200]}")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
                    attempt += 1
                    continue
                else:
                    log.error(f"[{label}] call failed: {e}", exc_info=True)
                    attempt += 1

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
