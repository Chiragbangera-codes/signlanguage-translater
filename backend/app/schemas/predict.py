from typing import Annotated, List

from pydantic import BaseModel, ConfigDict, Field

SequenceFrame = Annotated[List[float], Field(min_length=127, max_length=127)]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sequence": [[0.0] + [0.0] * 63 + [-1.0] * 63] * 30,
                "mode": "numbers"
            }
        }
    )

    sequence: List[SequenceFrame] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="Thirty landmark frames, each containing uses_two_hands followed by 63 left- and 63 right-hand coordinates."
    )

    mode: str = Field(
        "numbers",
        description="Inference mode: 'numbers' routes to the digit (0-9) model, 'words' routes to the alphabet model."
    )


class PredictionItem(BaseModel):
    label: str = Field(..., description="Predicted label (digit or alphabet letter)")
    confidence: float = Field(..., description="Confidence score in percentage (0-100)")

class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Top predicted label")
    confidence: float = Field(..., description="Main prediction confidence score (0-100)")
    processing_time_ms: float = Field(..., description="Inference and preprocessing execution time in milliseconds")
    top_predictions: List[PredictionItem] = Field(
        ...,
        description="List of top predictions sorted by confidence score"
    )

