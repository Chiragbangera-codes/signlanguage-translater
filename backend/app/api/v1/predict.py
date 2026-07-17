import logging
import os
import sys
import time

import numpy as np
from fastapi import APIRouter, HTTPException

# Add workspace root to sys.path to access ml module
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

from ml.preprocess import normalize_hand

from ...schemas.predict import PredictionItem, PredictionRequest, PredictionResponse
from ...services.model_loader import ModelLoaderService

# Set up logging
logger = logging.getLogger("backend_api")

router = APIRouter()
model_loader = ModelLoaderService()

@router.post("/predict", response_model=PredictionResponse, summary="Translate Hand Landmarks to Alphabet Character")
async def predict_gesture(payload: PredictionRequest):
    """Predicts the Indian Sign Language alphabet from hand landmarks.

    Accepts coordinates of both hands and uses_two_hands status,
    applies the exact preprocessing pipeline (wrist origin offset and hand scaling),
    and feeds the 127 features to the neural network model.
    """
    start_time = time.perf_counter()

    logger.info("Received translation request.")

    if not model_loader.is_loaded:
        logger.error("Model loader service is not initialized.")
        raise HTTPException(
            status_code=503,
            detail="Prediction model is not initialized or loaded."
        )

    try:
        # 1. Load coordinates from request
        left_hand_raw = np.array(payload.left_hand, dtype=np.float64)
        right_hand_raw = np.array(payload.right_hand, dtype=np.float64)
        uses_two_hands = float(payload.uses_two_hands)

        # DEBUG: log whether hands have real data or are padded
        left_is_empty = np.allclose(left_hand_raw, -1.0)
        right_is_empty = np.allclose(right_hand_raw, -1.0)
        logger.info(f"[DEBUG] uses_two_hands={uses_two_hands}, left_empty={left_is_empty}, right_empty={right_is_empty}")
        if not left_is_empty:
            logger.info(f"[DEBUG] left_hand wrist (first 3): {left_hand_raw[:3]}")
        if not right_is_empty:
            logger.info(f"[DEBUG] right_hand wrist (first 3): {right_hand_raw[:3]}")

        # 2. Apply instance-wise normalization matching the training pipeline exactly
        left_hand_norm = normalize_hand(left_hand_raw)
        right_hand_norm = normalize_hand(right_hand_raw)

        # 3. Concatenate: [uses_two_hands] + [left_hand_normalized] + [right_hand_normalized]
        features = np.zeros(127, dtype=np.float32)
        features[0] = uses_two_hands
        features[1:64] = left_hand_norm
        features[64:127] = right_hand_norm

        # 4. Run model prediction
        probs = model_loader.predict(features)
        logger.info(f"[DEBUG] top-3 raw probs: {sorted(enumerate(probs), key=lambda x: -x[1])[:3]}")

        # 5. Extract top predictions (class probabilities)
        # Sort classes in descending order of probabilities
        sorted_indices = np.argsort(probs)[::-1]

        top_predictions = []
        for rank in range(3): # Return top 3 predictions
            idx = sorted_indices[rank]
            prob = probs[idx]
            label = model_loader.decode_label(idx)
            top_predictions.append(
                PredictionItem(label=label, confidence=round(float(prob) * 100, 2))
            )

        prediction_letter = top_predictions[0].label
        prediction_conf = top_predictions[0].confidence

        # Calculate execution time
        end_time = time.perf_counter()
        processing_time_ms = round((end_time - start_time) * 1000, 2)

        logger.info(f"Prediction successful. Result: {prediction_letter} ({prediction_conf}%) in {processing_time_ms}ms")

        return PredictionResponse(
            prediction=prediction_letter,
            confidence=prediction_conf,
            processing_time_ms=processing_time_ms,
            top_predictions=top_predictions
        )


    except Exception as e:
        logger.exception(f"Unexpected prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference pipeline execution error: {str(e)}"
        ) from e

