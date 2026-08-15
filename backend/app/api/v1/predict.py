import logging
import os
import sys
import time

import numpy as np
from fastapi import APIRouter, HTTPException

# Add workspace root to sys.path to access ml module
workspace_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

from ml.preprocess import normalize_hand

from ...schemas.predict import PredictionItem, PredictionRequest, PredictionResponse
from ...services.model_loader import ModelLoaderService

# Set up logging
logger = logging.getLogger("backend_api")

router = APIRouter()
model_loader = ModelLoaderService()

@router.post("/predict", response_model=PredictionResponse, summary="Classify a landmark sequence (digits or words)")
async def predict_gesture(payload: PredictionRequest):
    """Predicts a gesture from 30 webcam landmark frames for the requested mode."""
    start_time = time.perf_counter()

    mode = (payload.mode or "numbers").lower()
    logger.info(f"Received translation request for mode: {mode}")

    if not model_loader.is_loaded:
        logger.error("Model loader service is not initialized.")
        raise HTTPException(
            status_code=503,
            detail="Prediction model is not initialized or loaded."
        )

    if mode not in model_loader.available_modes():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown mode '{mode}'. Supported modes: {model_loader.available_modes()}"
        )

    if not model_loader.is_mode_available(mode):
        logger.warning(f"Model for mode '{mode}' is not trained yet.")
        raise HTTPException(
            status_code=503,
            detail=(
                f"The '{mode}' model is not trained yet. "
                f"Collect the corresponding dataset and run preprocessing/training "
                f"to enable {mode} predictions."
            )
        )

    try:
        # The LSTM model expects the full 30-frame sequence (30, 127),
        # not a single (127,) frame.  Normalize every frame's hand
        # landmarks per-frame, matching the preprocessing pipeline
        # used during training (preprocess_features in preprocess.py).
        sequence_input = np.asarray(payload.sequence, dtype=np.float32)

        if sequence_input.shape != (30, 127):
            raise HTTPException(
                status_code=400,
                detail=f"Expected a (30, 127) landmark sequence, got shape {sequence_input.shape}."
            )

        # Validate that at least one frame contains a detected hand
        # (left-hand coords not all -1.0 padding — the sentinel for
        # "no hand detected").
        has_valid_frame = False
        for frame in sequence_input:
            if not np.allclose(frame[1:64], -1.0):
                has_valid_frame = True
                break

        if not has_valid_frame:
            logger.warning("No valid hand detected in any of the 30 sequence frames.")
            raise HTTPException(
                status_code=422,
                detail=(
                    "No hand was detected in the provided sequence. "
                    "Ensure a hand is clearly visible in the camera frame."
                ),
            )

        # Normalize all frames in the sequence (same per-frame logic
        # as preprocess_features in preprocess.py).
        for frame in sequence_input:
            frame[1:64] = normalize_hand(frame[1:64])
            frame[64:127] = normalize_hand(frame[64:127])

        probs = model_loader.predict(sequence_input, mode=mode)

        # Extract the three most likely labels for the requested mode.
        # Sort classes in descending order of probabilities
        sorted_indices = np.argsort(probs)[::-1]

        top_predictions = []
        for rank in range(3): # Return top 3 predictions
            idx = sorted_indices[rank]
            prob = probs[idx]
            label = model_loader.decode_label(idx, mode=mode)
            top_predictions.append(
                PredictionItem(label=label, confidence=round(float(prob) * 100, 2))
            )

        prediction_label = top_predictions[0].label
        prediction_conf = top_predictions[0].confidence

        # Calculate execution time
        end_time = time.perf_counter()
        processing_time_ms = round((end_time - start_time) * 1000, 2)

        logger.info(f"Prediction successful ({mode}). Result: {prediction_label} ({prediction_conf}%) in {processing_time_ms}ms")

        return PredictionResponse(
            prediction=prediction_label,
            confidence=prediction_conf,
            processing_time_ms=processing_time_ms,
            top_predictions=top_predictions
        )


    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference pipeline execution error: {str(e)}"
        ) from e

