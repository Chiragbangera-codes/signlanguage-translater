# SignSpeak AI - AI-Powered Indian Sign Language (ISL) Translator

SignSpeak AI is a modern web application that translates Indian Sign Language (ISL) hand gestures into text and speech in real-time. It uses MediaPipe for client-side hand tracking, TensorFlow for gesture classification, FastAPI for the prediction API, and Next.js for a premium, interactive user interface.

## Project Structure

```text
SignSpeakAI/
  ├── dataset/
  │     └── Indian Sign Language Gesture Landmarks.csv
  ├── ml/
  │     ├── train.py
  │     ├── evaluate.py
  │     ├── preprocess.py
  │     ├── predict.py
  │     ├── dataset.py
  │     └── utils.py
  ├── backend/
  │     ├── app/
  │     │     ├── api/
  │     │     │     └── v1/
  │     │     ├── core/
  │     │     ├── models/
  │     │     ├── schemas/
  │     │     ├── services/
  │     │     └── utils/
  │     │     └── main.py
  │     ├── requirements.txt
  │     └── .env
  ├── frontend/
  │     ├── src/
  │     │     ├── app/
  │     │     ├── components/
  │     │     ├── hooks/
  │     │     ├── services/
  │     │     ├── types/
  │     │     ├── lib/
  │     │     └── styles/
  │     ├── package.json
  │     └── .env.local
  ├── docs/
  ├── tests/
  ├── scripts/
  ├── .gitignore
  └── README.md
```

## Features

- **Real-Time Webcam Translation**: Extracts hand landmarks in the browser via MediaPipe.
- **Dual-Hand Inference**: Connects a 127-feature model supporting single/double hand gestures.
- **Prediction Stabilization**: Utilizes prediction buffering, majority voting, hold detection, and cooldowns.
- **Text-to-Speech**: Speech synthesis utilizing native browser voices.
- **Word & Sentence Builder**: Full UI controls to delete, space, clear, and record history.

## Development Setup

### ML Pipeline
1. Navigate to the `ml/` directory.
2. Install training dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run preprocessing & training:
   ```bash
   python train.py
   ```

### Backend API
1. Navigate to the `backend/` directory.
2. Setup a virtual environment and install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the FastAPI development server:
   ```bash
   uvicorn app.main:app --reload
   ```

### Frontend UI
1. Navigate to the `frontend/` directory.
2. Install npm modules:
   ```bash
   npm install
   ```
3. Launch development server:
   ```bash
   npm run dev
   ```
