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
*   **Description**: Accepts raw MediaPipe hand landmark arrays, processes coordinates, and returns the top sign translation characters.
*   **Request Headers**: `Content-Type: application/json`
*   **Request Body**:
    *   `left_hand`: Array of exactly 63 floats (21 landmarks $\times$ 3 coordinates $x, y, z$).
    *   `right_hand`: Array of exactly 63 floats.
    *   `uses_two_hands`: Float binary flag (`1.0` if both hands are detected, `0.0` if only one hand is detected).
    *   *Note*: Padded hands (e.g. right hand missing) must be filled with `-1.0` coordinates.

    **Example request payload**:
    ```json
    {
      "left_hand": [0.25, 0.43, -0.05, 0.28, 0.40, -0.08, ...],
      "right_hand": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, ...],
      "uses_two_hands": 0.0
    }
    ```

*   **Response Body**:
    *   `prediction`: String representing the top predicted alphabet character (A-Z).
    *   `confidence`: Float score (0.0 to 100.0) of the top prediction.
    *   `top_predictions`: List of the top 3 prediction candidates sorted by confidence in descending order.

    **Example response payload (200 OK)**:
    ```json
    {
      "prediction": "A",
      "confidence": 99.82,
      "top_predictions": [
        {"label": "A", "confidence": 99.82},
        {"label": "S", "confidence": 0.12},
        {"label": "E", "confidence": 0.06}
      ]
    }
    ```

---

## 3. Error and Status Codes

The API implements structured JSON error responses with standard HTTP status codes:

| HTTP Status | Error Detail | Scenario |
| :---: | --- | --- |
| **200 OK** | Successful execution | Prediction returned successfully. |
| **400 Bad Request** | Request parsing error | Body contains malformed JSON. |
| **422 Unprocessable** | Validation error | Input arrays do not have exactly 63 items, or flags are missing. |
| **503 Service Unavailable**| Model checkpoints not loaded | The TensorFlow model file is missing or failed to initialize on startup. |
| **500 Server Error** | Internal pipeline failure | Inference calculations failed due to runtime exceptions. |
