import logging
import time

from fastapi import APIRouter, HTTPException

from ...schemas.sentence import SentenceRequest, SentenceResponse
from ...services.sentence_service import (
    DEFAULT_LANGUAGE_NAME,
    SentenceService,
    SentenceUnavailableError,
)

logger = logging.getLogger("backend_api")

router = APIRouter()
sentence_service = SentenceService()


@router.post(
    "/sentence",
    response_model=SentenceResponse,
    summary="Build a natural sentence from recognized sign glosses",
)
def build_sentence(payload: SentenceRequest):
    """Converts a stream of sign glosses into a natural sentence in the target language.

    Declared sync on purpose: the Gemini call blocks, so FastAPI runs this in a
    threadpool rather than stalling the event loop (and concurrent /predict
    requests) for the length of the generation.

    Returns ``503`` when no API key is configured — the frontend
    falls back to its local rule-based constructor in that case.
    """
    start_time = time.perf_counter()

    if not sentence_service.credentials_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "Sentence generation is not configured. Set GEMINI_API_KEY on the "
                "backend to enable natural sentences and translation."
            ),
        )

    try:
        result, from_cache = sentence_service.generate(
            words=payload.words,
            language=payload.language,
            language_name=payload.language_name,
            style=payload.style,
            mode=payload.mode,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except SentenceUnavailableError as e:
        logger.warning(f"Sentence generation unavailable: {e}")
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.exception(f"Sentence generation failed: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Sentence generation failed: {e}",
        ) from e

    processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
    language_name = (payload.language_name or "").strip() or DEFAULT_LANGUAGE_NAME

    logger.info(
        f"Sentence generated ({language_name}, {'cache' if from_cache else 'llm'}) "
        f"in {processing_time_ms}ms"
    )

    return SentenceResponse(
        sentence=result.translation,
        english=result.english,
        language=payload.language,
        language_name=language_name,
        source="cache" if from_cache else "llm",
        processing_time_ms=processing_time_ms,
    )
