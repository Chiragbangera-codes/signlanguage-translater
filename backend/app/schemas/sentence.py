from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class SentenceRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "words": ["hello", "how", "you"],
                "language": "hi",
                "language_name": "Hindi",
                "style": "natural",
                "mode": "words",
            }
        }
    )

    words: List[str] = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Ordered sign glosses recognized by the translator, oldest first."
    )

    language: str = Field(
        "en",
        max_length=16,
        description="BCP-47 style code of the language the sentence should be spoken in (e.g. 'en', 'hi', 'ta')."
    )

    language_name: Optional[str] = Field(
        None,
        max_length=64,
        description="Human readable name of the target language (e.g. 'Hindi'). Defaults to English."
    )

    style: Literal["natural", "expanded"] = Field(
        "natural",
        description=(
            "'natural' keeps the utterance as short as the glosses justify. "
            "'expanded' produces a fuller, more conversational message for longer conversations."
        )
    )

    mode: str = Field(
        "words",
        description="Translator mode the glosses came from: 'words' or 'numbers'."
    )


class SentenceResponse(BaseModel):
    sentence: str = Field(..., description="The final sentence in the requested target language.")
    english: str = Field(..., description="The same sentence in English, for on-screen reference.")
    language: str = Field(..., description="Code of the language 'sentence' is written in.")
    language_name: str = Field(..., description="Human readable name of the target language.")
    source: str = Field(..., description="'llm' for a generated sentence, 'cache' for a repeated one.")
    processing_time_ms: float = Field(..., description="Server-side generation time in milliseconds.")
