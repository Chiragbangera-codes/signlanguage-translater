# SignSpeak AI — Project Report

## 1. Abstract

SignSpeak AI is a real-time sign language gesture translation system that converts hand gestures into text and speech. The system captures 30-frame MediaPipe hand landmark sequences from a webcam, processes them through a TensorFlow/Keras LSTM neural network, and returns the top-3 predicted gesture labels with confidence scores. The backend is built on FastAPI, served via Uvicorn, and the frontend is a Next.js dashboard with live camera tracking, prediction stabilization, and word/sentence construction UI.

---

## 2. Technical Architecture

### 2.1 Stack Overview

| Layer | Technology |
|-------|-----------|
| **Frontend** | Next.js 15 (App Router), React 19, TypeScript, Tailwind CSS, Framer Motion, Zustand |
| **Backend** | FastAPI, Uvicorn (async), Pydantic v2, NumPy 1.26.4 |
| **ML Framework** | TensorFlow 2.15.0, Keras 2.15.0 |
| **Computer Vision** | MediaPipe Hands (client-side landmark extraction), OpenCV |
| **Data Science** | Scikit-Learn (LabelEncoder), SciPy, h5py 3.16.0 |
| **Python** | 3.10 |

**Full pinned dependencies** (`backend/requirements.txt`):

| Package | Version |
|---------|---------|
| fastapi | 0.141.1 |
| uvicorn | 0.52.3 |
| numpy | 1.26.4 |
| tensorflow | 2.15.0 |
| keras | 2.15.0 |
| protobuf | 4.25.9 |
| scikit-learn | 1.7.2 |
| pydantic | 2.13.4 |
| python-dotenv | 1.2.2 |
| h5py | 3.16.0 |

### 2.2 System Flow

```
User Camera → MediaPipe Hands → 30-frame landmark sequences (30×127)
    → FastAPI /predict endpoint → LSTM model inference → Top-3 labels + confidence
    → Frontend displays prediction → Text-to-speech output
```

### 2.3 File Structure

```text
SignSpeakAI/
  ├── backend/
  │     ├── app/
  │     │     ├── api/v1/predict.py      # POST /api/v1/predict
  │     │     ├── schemas/predict.py      # Pydantic models
  │     │     ├── services/model_loader.py# Thread-safe Singleton model loader
  │     │     └── main.py                 # FastAPI app + lifespan, CORS
  │     └── requirements.txt
  ├── frontend/
  │     ├── src/
  │     │     ├── app/                    # Next.js pages (landing + dashboard)
  │     │     ├── components/translator/  # CameraCard, Controls, PredictionCard,
  │     │     │                            # WordBuilder, SentenceBuilder, HistoryPanel, etc.
  │     │     ├── store/useTranslatorStore.ts  # Zustand global state
  │     │     ├── lib/speech.ts           # Text-to-speech utilities
  │     │     └── styles/
  │     └── package.json
  ├── ml/
  │     ├── preprocess.py                 # Normalization, label encoding, train/val/test split
  │     ├── train.py                      # LSTM training (numbers mode)
  │     ├── train_words.py                # LSTM training (words mode)
  │     ├── evaluate.py                   # Test-set evaluation + artifacts
  │     ├── evaluate_words.py             # Words model evaluation
  │     ├── model/                        # sign_speak_lstm.h5, sign_speak_words_lstm.h5
  │     └── artifacts/                    # Confusion matrices, training curves
  ├── dataset/
  │     ├── digits/[0-9]/                 # 500 .npy sequences (50/digit × 10)
  │     └── words/[label]/                # 700 .npy sequences (50/word × 14)
  ├── scripts/
  │     └── collect_sequences.py          # Webcam-based landmark collection
  ├── tests/
  │     └── test_api.py                   # 9 API integration tests
  └── README.md
```

---

## 3. Dataset

### 3.1 Collection Method

All data was collected using `scripts/collect_sequences.py`, which captures 30-frame MediaPipe landmark sequences from a webcam for each gesture label. Each sequence contains 30 frames, and each frame has 127 features:

- **Feature 0**: `uses_two_hands` flag (1.0 if two hands, 0.0 if one)
- **Features 1–63**: Left hand landmarks (21 points × 3D coordinates)
- **Features 64–126**: Right hand landmarks (21 points × 3D coordinates)

Missing hands are padded with `-1.0` sentinel values.

### 3.2 Dataset Summary

| Mode | Classes | Samples per Class | Total Samples | Frames per Sequence | Features per Frame |
|------|---------|-------------------|---------------|---------------------|-------------------|
| Numbers | 10 (digits 0–9) | 50 | 500 | 30 | 127 |
| Words | 14 (hello, you, how, …, welcome) | 50 | 700 | 30 | 127 |

### 3.3 Preprocessing Pipeline

1. **Translation normalization**: Wrist landmark (index 0) is shifted to the origin `(0, 0, 0)`, making features location-invariant.
2. **Scale normalization**: All coordinates are divided by the maximum Euclidean distance from the wrist to any landmark, normalizing to a unit sphere.
3. **Missing-hand preservation**: Coordinates of absent hands remain `-1.0` to preserve the dual-hand flag signal.
4. **Label encoding**: `LabelEncoder` from scikit-learn maps class names to integer indices (e.g., `0→"0"`, `3→"3"` for digits; `0→"fine"`, `5→"hello"` for words).

### 3.4 Train/Validation/Test Split

- **Random seed**: `42` (fixed for reproducibility)
- **Split ratios**: 80% training / 10% validation / 10% testing
- **Stratification**: Applied on target labels to maintain class balance across all splits
- **No preprocessing leakage**: Normalization is instance-based (per-hand), not group-based

---

## 4. Model Architecture

### 4.1 Numbers LSTM Model (`sign_speak_lstm.h5`)

| Layer | Type | Configuration |
|-------|------|---------------|
| Input | InputLayer | `(None, 30, 127)` |
| LSTM | LSTM | 64 units, tanh activation, return_sequences=False |
| Dropout | Dropout | Rate = 0.2 |
| Dense | Dense | 64 units, ReLU activation |
| Dropout | Dropout | Rate = 0.2 |
| Dense | Dense | 10 units, Softmax activation |

- **Total parameters**: 53,962
- **Optimizer**: Adam (lr = 0.001)
- **Loss**: Sparse Categorical Crossentropy
- **Batch size**: 64
- **Epochs**: 35 (with EarlyStopping patience=8, ReduceLROnPlateau)

### 4.2 Words LSTM Model (`sign_speak_words_lstm.h5`)

| Layer | Type | Configuration |
|-------|------|---------------|
| Input | InputLayer | `(None, 30, 127)` |
| LSTM | LSTM | 64 units, tanh activation, return_sequences=False |
| Dropout | Dropout | Rate = 0.2 |
| Dense | Dense | 64 units, ReLU activation |
| Dropout | Dropout | Rate = 0.2 |
| Dense | Dense | 14 units, Softmax activation |

- **Total parameters**: 54,222
- Same hyperparameters as Numbers model

---

## 5. Training Results

### 5.1 Numbers Mode (Digits 0–9)

| Metric | Training (final epoch) | Validation (final epoch) | Test Set |
|--------|----------------------|------------------------|-----------|
| **Accuracy** | 99.50% | 100.00% | 100.00% |
| **Loss** | 0.110 | 0.041 | — |

**Training progression (35 epochs)**:
- Epoch 1: training acc 10.5%, val acc 14.0%
- Epoch 20: training acc 92.0%, val acc 100.0%
- Epoch 35 (final): training acc 99.5%, val acc 100.0%

### 5.2 Words Mode (14 word signs)

| Metric | Test Set |
|--------|----------|
| **Accuracy** | 100.00% |
| **Precision (weighted)** | 1.0000 |
| **Recall (weighted)** | 1.0000 |
| **F1 Score (weighted)** | 1.0000 |
| **Test samples** | 70 (5 per class × 14 classes) |

### 5.3 Model Selection

Three MLP architecture variants (A: Simple, B: Medium, C: Complex) were compared during initial exploration:

| Architecture | Parameters | Val Accuracy | Val Loss |
|-------------|------------|-------------|----------|
| Model A (Simple) | 11,226 | 99.00% | 0.890 |
| Model B (Medium) | 27,898 | 95.00% | 0.355 |
| Model C (Complex) | 78,074 | 99.00% | 0.082 |

The LSTM architecture was selected as the final production model for both modes.

---

## 6. Evaluation & Validation

### 6.1 Data Integrity Verification

- **Index disjointness**: `train_test_split(random_state=42, stratify=y)` produces disjoint train/val/test index sets — verified with zero overlap between any pair.
- **Sample duplication**: No duplicate sequences in any split.
- **Preprocessing isolation**: Normalization is computed per-sample (wrist translation + hand-size scaling), with no statistics shared across splits.

### 6.2 Test Set Composition

| Mode | Total Samples | Test Set | Test per Class | Classes |
|------|--------------|----------|----------------|---------|
| Numbers | 500 | 50 (10%) | 5 | 10 (digits 0–9) |
| Words | 700 | 70 (10%) | 5 | 14 (hello, you, how, …) |

### 6.3 Inference API Tests

All 9 API integration tests pass:

| Test | Description | Status |
|------|-------------|--------|
| `test_health_check` | GET /api/v1/health returns model_loaded=True | ✅ |
| `test_root_portal` | GET / returns welcome message | ✅ |
| `test_predict_empty_payload` | Empty body → 400 INVALID_INPUT | ✅ |
| `test_predict_wrong_sequence_length_short` | 10 frames → 400 INVALID_INPUT | ✅ |
| `test_predict_wrong_frame_feature_size` | 10 features → 400 INVALID_INPUT | ✅ |
| `test_predict_no_hand_detected` | All -1.0 frames → 422 | ✅ |
| `test_predict_success_numbers` | Valid 30×127 → 200 with prediction | ✅ |
| `test_predict_success_only_last_valid_frame` | 29 empty + 1 valid → 200 | ✅ |
| `test_predict_unknown_mode` | "invalid_mode" → 400 | ✅ |

### 6.4 Data Leakage Note

All sequences in both datasets were collected from a **single signer**. The random train/test split (seed=42) does not perform signer-aware grouping, meaning signer-specific characteristics appear in both training and test sets. The 100% test accuracy should be interpreted as **within-signer generalization** rather than cross-signer robustness.

---

## 7. Backend API

### 7.1 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API root portal with docs link |
| GET | `/api/v1/health` | Service health + model status |
| POST | `/api/v1/predict` | Gesture classification (top-3 + confidence) |

### 7.2 Prediction Request/Response

**Request** (`POST /api/v1/predict`):
```json
{
  "sequence": [[0.0, 0.25, 0.43, -0.05, ..., -1.0, ...], ...],  // 30 frames × 127 floats
  "mode": "numbers"  // or "words"
}
```

**Response (200 OK)**:
```json
{
  "prediction": "5",
  "confidence": 98.76,
  "processing_time_ms": 6.84,
  "top_predictions": [
    {"label": "5", "confidence": 98.76},
    {"label": "3", "confidence": 1.24},
    {"label": "0", "confidence": 0.00}
  ]
}
```

### 7.3 Model Loader Service

- Thread-safe Singleton pattern with `threading.RLock`
- Eagerly loads the default (numbers) model at startup via FastAPI lifespan
- Lazily loads the words model on first request
- Checks file existence before loading; returns 503 if model unavailable

### 7.4 Error Response Format

All errors return structured JSON:

| HTTP Status | Error Code | Detail | Scenario |
|------------|------------|--------|----------|
| 200 | — | Prediction returned | Success |
| 400 | INVALID_INPUT | Validation failed at '[field]': [msg] | Malformed JSON, wrong sequence length, wrong feature count |
| 422 | SERVER_ERROR | [detail] | No hand detected in any frame |
| 503 | SERVICE_UNAVAILABLE | Prediction model is not initialized or loaded. | Model failed to load at startup |
| 500 | SERVER_ERROR | Inference pipeline execution error: [str(e)] | Unhandled exception |

### 7.5 CORS Configuration

The API allows origins from environment variable `ALLOWED_ORIGINS` plus defaults:
- `http://localhost:3000`
- `http://127.0.0.1:3000`

All methods and headers are permitted with credentials enabled.

---

## 8. Frontend Features

### 8.1 Dashboard

- **Live Camera Feed**: MediaPipe Hands tracking with canvas overlay drawing
- **Stabilization Pipeline**: Rolling 10-frame buffer → 60% majority vote → 700ms hold → 500ms cooldown
- **Dual-Hand Support**: Both one-handed and two-handed gestures
- **Numbers/Words Mode Toggle**: Switches between digit (0–9) and word sign models
- **Word & Sentence Builder**: Compose words, construct sentences, archive results
- **History Panel**: Previously predicted gestures and spoken sentences
- **Settings Panel**: Confidence threshold, speech rate, voice selection, camera mirroring
- **Text-to-Speech**: Native Web Speech API synthesis with digit-to-word conversion (100 → "one hundred")
- **Keyboard Shortcuts**: Space=commit, Backspace=delete, Enter=speak, Escape=clear

### 8.2 Prediction Response Handling

The frontend processes the API response:
- Top prediction displayed in the main prediction card
- Confidence bar visualization
- Top-3 predictions shown in a dropdown
- Stabilization logic buffers predictions to prevent flickering

---

## 9. Development Setup

```bash
# 1. Create venv (Python 3.10)
python -m venv venv

# 2. Install backend requirements
venv\Scripts\pip install -r backend/requirements.txt

# 3. Start API server
venv\Scripts\uvicorn backend.app.main:app --reload

# 4. Install frontend dependencies
cd frontend
npm install

# 5. Start frontend
npm run dev
```

---

## 10. Conclusion

SignSpeak AI delivers a complete real-time sign language translation pipeline with:

- **100% test accuracy** on both numbers (10 digits) and words (14 signs) modes
- **Real-time inference** with ~3ms mean model latency (down from ~24ms, optimized via `tf.function` graph compilation)
- **Privacy-first design**: webcam processing is local; only 127 float coordinates are sent to the API
- **Production-ready deployment**: pinned dependencies (Python 3.10, TensorFlow 2.15.0, Keras 2.15.0), all 9 API tests passing

The primary limitation is single-signer data — future work should collect multi-signer datasets to validate cross-signer generalization.