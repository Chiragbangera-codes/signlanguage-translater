from typing import List

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "left_hand": [0.0] * 63,
                "right_hand": [-1.0] * 63,
                "uses_two_hands": 0.0
            }
        }
    )

    left_hand: List[float] = Field(
        ...,
        min_length=63,
        max_length=63,
        description="63 floating-point coordinates for left hand landmarks (21 landmarks * 3 coords x,y,z)"
    )
    right_hand: List[float] = Field(
        ...,
        min_length=63,
        max_length=63,
        description="63 floating-point coordinates for right hand landmarks (21 landmarks * 3 coords x,y,z)"
    )
    uses_two_hands: float = Field(
        ...,
        description="Binary flag (1.0 if both hands are detected, 0.0 otherwise)"
    )


class PredictionItem(BaseModel):
    label: str = Field(..., description="Predicted character (A-Z)")
    confidence: float = Field(..., description="Confidence score in percentage (0-100)")

class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Top predicted sign alphabet letter")
    confidence: float = Field(..., description="Main prediction confidence score (0-100)")
    processing_time_ms: float = Field(..., description="Inference and preprocessing execution time in milliseconds")
    top_predictions: List[PredictionItem] = Field(
        ...,
        description="List of top predictions sorted by confidence score"
    )

