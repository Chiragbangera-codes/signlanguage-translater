"""Turns recognized sign glosses into natural sentences in any spoken language.

The gesture model emits a bare stream of glosses ("hello how you"). Sign
languages drop articles, copulas and inflections, so a rule table can only
cover the handful of phrasings someone thought to write down. This service
hands the gloss stream to Gemini instead, which handles arbitrary length and
produces the translation in the same call.

The frontend keeps its local rule-based constructor as an offline fallback, so
this service is allowed to be unavailable: it reports that clearly instead of
returning a degraded sentence.
"""

import json
import logging
import os
import threading
from collections import OrderedDict
from typing import List, Optional, Tuple

from pydantic import BaseModel

logger = logging.getLogger("backend_api")

# Gemini model IDs move fairly often; override with SENTENCE_MODEL rather than
# editing this file.
DEFAULT_MODEL = os.getenv("SENTENCE_MODEL", "gemini-2.5-flash")
DEFAULT_LANGUAGE_NAME = "English"
CACHE_SIZE = 256

SYSTEM_PROMPT = """You convert sign language gloss into natural spoken language.

The input is an ordered list of signs recognized from a webcam, oldest first. \
Sign languages drop articles, copulas, plural markers and tense, and their word \
order differs from spoken language, so the gloss is never a finished sentence.

Rules:
- Produce ONE natural, grammatical utterance that a hearing person would say.
- Add the function words, tense, agreement and politeness the gloss implies \
(is/are, the/a, to, do, question inversion). Never add facts the gloss does not \
support: no invented names, numbers, places, or reasons.
- Adjacent repeated glosses mean one sign was held too long. Collapse them.
- A gloss you cannot place sensibly may be dropped rather than forced in.
- Digits stand for the number they spell out; keep them as spoken numbers.
- Punctuate as a question when the gloss carries a question sign (what, how, \
where, when, who, why, which) or clearly asks something.
- No quotes, no gloss echo, no commentary — just the utterance."""

STYLE_INSTRUCTIONS = {
    "natural": (
        "Length: match the gloss. A two-sign gloss becomes a short sentence; a longer "
        "gloss becomes a correspondingly longer one. Do not pad."
    ),
    "expanded": (
        "Length: the signer is holding a longer conversation and wants to say more than "
        "the gloss literally spells out. Expand it into a fuller, natural message of two "
        "or three connected sentences, joining the glosses into clauses with the "
        "conjunctions and phrasing a fluent speaker would use. Stay strictly within what "
        "the gloss supports — elaborate the phrasing, never the facts."
    ),
}


class GeneratedSentence(BaseModel):
    """Schema the model is constrained to when generating a sentence."""

    english: str
    translation: str


class SentenceUnavailableError(RuntimeError):
    """Raised when no API key is configured, the SDK is missing, or the model
    returned nothing usable."""


class SentenceService:
    """Thread-safe singleton wrapping the Gemini client used for sentence generation."""

    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SentenceService, cls).__new__(cls)
                cls._instance._client = None
                cls._instance._cache = OrderedDict()
            return cls._instance

    @staticmethod
    def api_key() -> Optional[str]:
        """The configured key, if any.

        ``GEMINI_API_KEY`` is the name the Gemini SDK documents;
        ``GOOGLE_API_KEY`` is accepted too since the SDK also reads it.
        """
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    @staticmethod
    def credentials_configured() -> bool:
        return bool(SentenceService.api_key())

    @property
    def is_available(self) -> bool:
        if not self.credentials_configured():
            return False
        try:
            self._get_client()
        except SentenceUnavailableError:
            return False
        return True

    def _get_client(self):
        with self._lock:
            if self._client is not None:
                return self._client

            try:
                from google import genai
            except ImportError as e:
                raise SentenceUnavailableError(
                    "The 'google-genai' package is not installed. "
                    "Run: pip install -r backend/requirements.txt"
                ) from e

            key = self.api_key()
            if not key:
                raise SentenceUnavailableError("GEMINI_API_KEY is not set.")

            self._client = genai.Client(api_key=key)
            return self._client

    def _cache_get(self, key: Tuple) -> Optional[GeneratedSentence]:
        with self._lock:
            result = self._cache.get(key)
            if result is not None:
                self._cache.move_to_end(key)
            return result

    def _cache_put(self, key: Tuple, value: GeneratedSentence) -> None:
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)

    def generate(
        self,
        words: List[str],
        language: str = "en",
        language_name: Optional[str] = None,
        style: str = "natural",
        mode: str = "words",
    ) -> Tuple[GeneratedSentence, bool]:
        """Builds a sentence from ``words``. Returns ``(sentence, from_cache)``."""
        cleaned = [w.strip() for w in words if w and w.strip()]
        if not cleaned:
            raise ValueError("No glosses were provided.")

        target_name = (language_name or "").strip() or (
            DEFAULT_LANGUAGE_NAME if language.lower().startswith("en") else language
        )
        style_key = style if style in STYLE_INSTRUCTIONS else "natural"

        cache_key = (tuple(w.lower() for w in cleaned), language, target_name, style_key, mode)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached, True

        client = self._get_client()

        gloss = " ".join(cleaned)
        mode_note = (
            "These glosses are digits signed one at a time; read them as the number they spell."
            if mode == "numbers"
            else "These glosses are word signs."
        )

        user_prompt = (
            f"Sign glosses (oldest first): {gloss}\n"
            f"{mode_note}\n"
            f"{STYLE_INSTRUCTIONS[style_key]}\n\n"
            f"Return the utterance in English, and its translation into {target_name}. "
            f"If {target_name} is English, the translation is the same text."
        )

        try:
            response = client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=user_prompt,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    # Constrain the reply to the GeneratedSentence shape so no
                    # prose parsing is needed.
                    "response_mime_type": "application/json",
                    "response_schema": GeneratedSentence,
                    "temperature": 0.3,
                },
            )
        except Exception as e:
            # A wrong model ID and an auth failure both land here; keep the
            # original message so the cause is visible in the API response.
            raise SentenceUnavailableError(
                f"Gemini request failed ({DEFAULT_MODEL}): {e}"
            ) from e

        result = self._parse(response)

        # Guard against an empty translation for a non-English target.
        if not result.translation.strip():
            result.translation = result.english

        self._cache_put(cache_key, result)
        return result, False

    @staticmethod
    def _parse(response) -> GeneratedSentence:
        """Extracts the structured sentence from a Gemini response.

        Prefers the SDK's own parsed object, falling back to decoding the JSON
        text. Either can be empty when a safety filter blocks the reply, which
        surfaces as an unavailability rather than a crash.
        """
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, GeneratedSentence) and parsed.english.strip():
            return parsed

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            reason = ""
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                reason = f" (finish_reason={getattr(candidates[0], 'finish_reason', None)})"
            raise SentenceUnavailableError(
                f"The model returned no sentence{reason}."
            )

        try:
            return GeneratedSentence.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValueError) as e:
            raise SentenceUnavailableError(
                f"Could not read the generated sentence: {e}"
            ) from e
