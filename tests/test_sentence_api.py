import os
import sys

import pytest
from fastapi.testclient import TestClient

# Add workspace root and backend directory to sys.path
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
backend_root = os.path.join(workspace_root, "backend")
if workspace_root not in sys.path:
    sys.path.append(workspace_root)
if backend_root not in sys.path:
    sys.path.append(backend_root)

from backend.app.main import app  # noqa: E402
from backend.app.services.sentence_service import (  # noqa: E402
    GeneratedSentence,
    SentenceService,
    SentenceUnavailableError,
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def credentials(monkeypatch):
    """Makes the service look configured without hitting the real API."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-used")


@pytest.fixture
def no_credentials(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def _stub_generate(english: str, translation: str, from_cache: bool = False):
    def _generate(self, words, language="en", language_name=None, style="natural", mode="words"):
        return GeneratedSentence(english=english, translation=translation), from_cache

    return _generate


# ---------------------------------------------------------------------------
# Endpoint behaviour
# ---------------------------------------------------------------------------

def test_health_reports_sentence_generation(client):
    """GET /api/v1/health exposes whether sentence generation is configured."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert "sentence_generation" in response.json()


def test_sentence_requires_words(client, credentials):
    """An empty gloss list is rejected before any API call."""
    response = client.post("/api/v1/sentence", json={"words": []})
    assert response.status_code == 400
    assert response.json()["error"] == "INVALID_INPUT"


def test_sentence_unconfigured_returns_503(client, no_credentials):
    """Without credentials the endpoint reports unavailability so the UI can fall back."""
    response = client.post("/api/v1/sentence", json={"words": ["hello", "how", "you"]})
    assert response.status_code == 503
    json_data = response.json()
    assert json_data["error"] == "SERVICE_UNAVAILABLE"
    assert "GEMINI_API_KEY" in json_data["message"]


def test_sentence_success(client, credentials, monkeypatch):
    """A generated sentence is returned with its English reference and metadata."""
    monkeypatch.setattr(
        SentenceService, "generate", _stub_generate("How are you?", "आप कैसे हैं?")
    )

    response = client.post(
        "/api/v1/sentence",
        json={
            "words": ["hello", "how", "you"],
            "language": "hi",
            "language_name": "Hindi",
            "style": "natural",
            "mode": "words",
        },
    )

    assert response.status_code == 200
    json_data = response.json()
    assert json_data["sentence"] == "आप कैसे हैं?"
    assert json_data["english"] == "How are you?"
    assert json_data["language"] == "hi"
    assert json_data["language_name"] == "Hindi"
    assert json_data["source"] == "llm"
    assert json_data["processing_time_ms"] >= 0.0


def test_sentence_reports_cache_hits(client, credentials, monkeypatch):
    """A repeated gloss sequence is served from the cache and labelled as such."""
    monkeypatch.setattr(
        SentenceService, "generate", _stub_generate("Thank you.", "Thank you.", from_cache=True)
    )

    response = client.post("/api/v1/sentence", json={"words": ["thank_you"]})
    assert response.status_code == 200
    assert response.json()["source"] == "cache"


def test_sentence_defaults_to_english(client, credentials, monkeypatch):
    """Omitting the language yields an English sentence."""
    monkeypatch.setattr(SentenceService, "generate", _stub_generate("I am fine.", "I am fine."))

    response = client.post("/api/v1/sentence", json={"words": ["i", "fine"]})
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["language"] == "en"
    assert json_data["language_name"] == "English"


def test_sentence_rejects_unknown_style(client, credentials):
    """`style` is constrained to the two supported values."""
    response = client.post(
        "/api/v1/sentence", json={"words": ["hello"], "style": "shakespearean"}
    )
    assert response.status_code == 400
    assert response.json()["error"] == "INVALID_INPUT"


def test_sentence_upstream_failure_returns_503(client, credentials, monkeypatch):
    """A model-side failure surfaces as 503 rather than a 500."""

    def _raise(self, *args, **kwargs):
        raise SentenceUnavailableError("The model returned an empty sentence.")

    monkeypatch.setattr(SentenceService, "generate", _raise)

    response = client.post("/api/v1/sentence", json={"words": ["hello"]})
    assert response.status_code == 503
    assert response.json()["error"] == "SERVICE_UNAVAILABLE"

def test_credentials_configured_follows_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert SentenceService.credentials_configured() is False

    monkeypatch.setenv("GEMINI_API_KEY", "key")
    assert SentenceService.credentials_configured() is True


def test_google_api_key_also_accepted(monkeypatch):
    """The Gemini SDK reads GOOGLE_API_KEY too, so honour both names."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    assert SentenceService.credentials_configured() is True


def test_generate_rejects_blank_glosses():
    """Whitespace-only input never reaches the API."""
    with pytest.raises(ValueError):
        SentenceService().generate(words=["", "   "])


def test_generate_caches_repeated_glosses(monkeypatch):
    """The second identical request is served from the cache, not the API."""
    service = SentenceService()
    service._cache.clear()

    calls = {"count": 0}

    class _FakeResponse:
        parsed = None
        text = '{"english": "How are you?", "translation": "How are you?"}'
        candidates = []

    class _FakeModels:
        def generate_content(self, **kwargs):
            calls["count"] += 1
            return _FakeResponse()

    class _FakeClient:
        models = _FakeModels()

    monkeypatch.setattr(SentenceService, "_get_client", lambda self: _FakeClient())

    first, first_cached = service.generate(words=["how", "you"])
    second, second_cached = service.generate(words=["How", "You"])  # case-insensitive key

    assert calls["count"] == 1
    assert first_cached is False
    assert second_cached is True
    assert second.english == first.english
