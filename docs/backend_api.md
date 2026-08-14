# Backend API Documentation

This document describes the FastAPI inference service structure, request/response formats, error codes, and deployment details for the SignSpeak AI translation API.

---

## 1. API Architecture

The backend is built with FastAPI following a modular layout:

```text
backend/
  ├── app/
  │     ├── api/
  │     │     └── v1/
  │     │           └── predict.py   <-- POST /predict router
  │     ├── core/                  <-- Configurations and security
  │     ├── schemas/
  │     │     └── predict.py       <-- Pydantic schemas (Request / Response)
  │     ├── services/
  │     │     └── model_loader.py  <-- Thread-safe Singleton model loader
  │     └── main.py                <-- FastAPI app setup and Lifespan manager
  ├── requirements.txt
  └── .env
```

### Components
*   **Lifespan manager**: On startup, loads and initializes the TensorFlow model and Label Encoder. On shutdown, releases resources.
*   **ModelLoaderService**: Singleton service designed with a thread lock to ensure thread safety when executing `model.predict()` concurrently across incoming async FastAPI requests.
*   **API v1 predict router**: Validates the payload using Pydantic, normalizes landmarks utilizing the exact sample-wise preprocessing functions from the training suite, merges coordinates, and calls the loader.

---

## 2. API Endpoints

### 2.1 API Root Information
`GET /`
*   **Description**: Index endpoint to verify server status and link to auto-generated OpenAPI documentation.
*   **Response (200 OK)**:
    ```json
    {
      "message": "Welcome to SignSpeak AI Translation API.",
      "health_check": "/api/v1/health",
      "docs": "/docs"
    }
    ```

### 2.2 Health Check Status
`GET /api/v1/health`
*   **Description**: Exposes service health and checks if neural network checkpoints were loaded.
*   **Response (200 OK)**:
    ```json
    {
      "status": "ok",
      "version": "1.0.0",
      "model_loaded": true,
      "tensorflow": "2.x"
    }
    ```


### 2.3 Translate Gesture
`POST /api/v1/predict`
*   **Description**: Accepts a sequence of raw MediaPipe hand landmark arrays, normalizes the coordinates, uses a sequence-to-single-frame bridge to isolate the latest valid gesture snapshot, and returns the top 3 prediction labels and confidence scores.
*   **Request Headers**: `Content-Type: application/json`
*   **Request Body**:
    *   `sequence`: An array of exactly 30 frames. Each frame is an array of exactly 127 floats:
        *   `uses_two_hands` (index 0): `1.0` if both hands are detected, `0.0` otherwise.
        *   `left_hand` landmarks (indices 1-63): 21 landmarks $\times$ 3 coordinates $x, y, z$. Padded with `-1.0` if no hand is detected.
        *   `right_hand` landmarks (indices 64-126): 21 landmarks $\times$ 3 coordinates $x, y, z$. Padded with `-1.0` if no hand is detected.
    *   `mode`: Inference mode string. Supports `"numbers"` (routes to the digits 0-9 and signs 10-25 model) and `"words"` (routes to the words model, if trained).

    **Example request payload**:
    ```json
    {
      "sequence": [
        [0.0, 0.25, 0.43, -0.05, 0.28, 0.40, -0.08, ..., -1.0, -1.0, -1.0, ...],
        ...
      ],
      "mode": "numbers"
    }
    ```

*   **Response Body**:
    *   `prediction`: String representing the top predicted gesture label (e.g. `"21"`).
    *   `confidence`: Float score (0.0 to 100.0) of the top prediction.
    *   `processing_time_ms`: Float representing the preprocessing and model execution time in milliseconds.
    *   `top_predictions`: List of the top 3 prediction candidates sorted by confidence in descending order.

    **Example response payload (200 OK)**:
    ```json
    {
      "prediction": "21",
      "confidence": 100.0,
      "processing_time_ms": 6.84,
      "top_predictions": [
        {"label": "21", "confidence": 100.0},
        {"label": "5", "confidence": 0.0},
        {"label": "3", "confidence": 0.0}
      ]
    }
    ```

---

## 3. Error and Status Codes

The API implements structured JSON error responses with standard HTTP status codes:

| HTTP Status | Error Detail | Scenario |
| :---: | --- | --- |
| **200 OK** | Successful execution | Prediction returned successfully. |
| **400 Bad Request** | Request parsing error | Body contains malformed JSON or validation fails (e.g. sequence is not 30 frames, or frame features are not 127). |
| **422 Unprocessable** | Validation error | Valid input schema but no hand was detected in any of the 30 frames. |
| **503 Service Unavailable**| Model checkpoints not loaded | The TensorFlow model file is missing or failed to initialize on startup. |
| **500 Server Error** | Internal pipeline failure | Inference calculations failed due to runtime exceptions. |
