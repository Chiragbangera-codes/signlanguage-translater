import sys, time
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import load_model

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else str(
    Path(__file__).resolve().parent / "ml" / "model" / "sign_speak_words_lstm_s2.h5"
)
WARMUP = 20
N_FRAMES = 300

model = load_model(MODEL_PATH)
seq_len, feat = model.input_shape[1], model.input_shape[2]
print(f"Model input: {seq_len} x {feat}")

@tf.function(reduce_retracing=True)
def infer(x):
    return model(x, training=False)

# warm-up so tracing time is not measured
dummy = np.zeros((1, seq_len, feat), dtype="float32")
infer(tf.constant(dummy))
model.predict(dummy, verbose=0)

def predict_slow(x):
    return model.predict(x, verbose=0)

def predict_fast(x):
    return infer(tf.constant(x)).numpy()

def extract(results):
    vec = []
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks[:2]:
            for lm in hand.landmark:
                vec.extend([lm.x, lm.y, lm.z])
    vec = vec[:feat] + [0.0] * max(0, feat - len(vec))
    return np.array(vec, dtype="float32")

def run(name, complexity, every, fast):
    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2,
        model_complexity=complexity,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    predict = predict_fast if fast else predict_slow
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    buffer = deque(maxlen=seq_len)
    t_total, t_mp, t_model = [], [], []
    frame_idx = 0
    print(f"\n>>> {name}: keep your hand in view...")

    while frame_idx < N_FRAMES + WARMUP:
        start = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        a = time.perf_counter()
        results = hands.process(rgb)
        buffer.append(extract(results))        # buffer updates EVERY frame
        b = time.perf_counter()

        predicted = False
        if len(buffer) == seq_len and frame_idx % every == 0:
            predict(np.expand_dims(np.array(buffer), 0))
            predicted = True
        c = time.perf_counter()

        frame_idx += 1
        if frame_idx > WARMUP and len(buffer) == seq_len:
            t_total.append((c - start) * 1000)
            t_mp.append((b - a) * 1000)
            if predicted:
                t_model.append((c - b) * 1000)

    cap.release()
    hands.close()
    if not t_total:
        return (name, None)
    avg = np.mean(t_total)
    return (name, dict(mp=np.mean(t_mp),
                       model=np.mean(t_model) if t_model else 0,
                       total=avg, fps=1000 / avg, pred=1000 / avg / every))

configs = [
    ("Baseline (predict, cx=1, every 1)", 1, 1, False),
    ("+ tf.function",                      1, 1, True),
    ("+ complexity 0",                     0, 1, True),
    ("+ predict every 2",                  0, 2, True),
    ("+ predict every 3",                  0, 3, True),
]

rows = [run(*c) for c in configs]

print("\n" + "=" * 86)
print(f"{'Config':36}{'MP ms':>8}{'Model ms':>10}{'Total ms':>10}{'Loop FPS':>10}{'Pred/s':>8}")
for name, r in rows:
    if r is None:
        print(f"{name:36}  (no frames measured)")
    else:
        print(f"{name:36}{r['mp']:8.1f}{r['model']:10.1f}{r['total']:10.1f}{r['fps']:10.2f}{r['pred']:8.2f}")