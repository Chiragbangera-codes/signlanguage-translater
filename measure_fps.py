import sys
import time
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else str(
    Path(__file__).resolve().parent / "ml" / "model" / "sign_speak_words_lstm_s2.h5"
)
WARMUP = 20        # frames/runs to ignore at the start
N_FRAMES = 300     # frames to measure in the end-to-end test
N_RUNS = 100       # runs for the model-only test

model = load_model(MODEL_PATH)
seq_len, feat = model.input_shape[1], model.input_shape[2]
print(f"Model input: sequence length={seq_len}, features={feat}")

# ---------- 1. Model-only ----------
x = np.random.rand(1, seq_len, feat).astype("float32")
for _ in range(WARMUP):
    model.predict(x, verbose=0)

t = time.perf_counter()
for _ in range(N_RUNS):
    model.predict(x, verbose=0)
total = time.perf_counter() - t
avg_ms = total / N_RUNS * 1000
print(f"\n[Model only] {N_RUNS} runs, avg {avg_ms:.2f} ms -> {1000 / avg_ms:.2f} FPS")

# ---------- 2. End-to-end (camera + MediaPipe + model) ----------
hands = mp.solutions.hands.Hands(
    static_image_mode=False, max_num_hands=2,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract(results):
    vec = []
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks[:2]:
            for lm in hand.landmark:
                vec.extend([lm.x, lm.y, lm.z])
    vec = vec[:feat] + [0.0] * max(0, feat - len(vec))
    return np.array(vec, dtype="float32")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

buffer = deque(maxlen=seq_len)
t_mp, t_model, t_total = [], [], []
count = 0
print("\nMeasuring end-to-end... keep your hand in front of the camera.")

while count < N_FRAMES + WARMUP:
    start = time.perf_counter()
    ok, frame = cap.read()
    if not ok:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    a = time.perf_counter()
    results = hands.process(rgb)
    buffer.append(extract(results))
    b = time.perf_counter()

    if len(buffer) == seq_len:
        model.predict(np.expand_dims(np.array(buffer), 0), verbose=0)
    c = time.perf_counter()

    count += 1
    if count > WARMUP and len(buffer) == seq_len:
        t_mp.append((b - a) * 1000)
        t_model.append((c - b) * 1000)
        t_total.append((c - start) * 1000)

cap.release()
hands.close()

if t_total:
    avg_total = np.mean(t_total)
    print(f"[MediaPipe]   avg {np.mean(t_mp):.2f} ms")
    print(f"[Model]       avg {np.mean(t_model):.2f} ms")
    print(f"[End-to-end]  avg {avg_total:.2f} ms/frame -> {1000 / avg_total:.2f} FPS")
else:
    print("Not enough frames were measured. Check your camera.")