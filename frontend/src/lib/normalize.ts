export const LANDMARKS_PER_HAND = 21;
export const COORDS_PER_HAND = LANDMARKS_PER_HAND * 3;
export const FEATURES_PER_FRAME = 1 + COORDS_PER_HAND * 2;
export const SEQUENCE_LENGTH = 30;

// Words model (v2): 127 landmark features + raw wrist x, y of hand slot A.
// Must match ADD_WRIST in ml/preprocess_words.py.
export const WRIST_FEATURES = 2;
export const MODEL_FEATURES = FEATURES_PER_FRAME + WRIST_FEATURES; // 129

export const MISSING = -1.0;

function isMissingValue(value: number): boolean {
  return Math.abs(value + 1.0) <= 1e-8 + 1e-5;
}

export function isHandMissing(hand: Float32Array | number[]): boolean {
  if (hand[0] === MISSING) return true;
  for (let i = 0; i < hand.length; i++) {
    if (!isMissingValue(hand[i])) return false;
  }
  return true;
}

export function normalizeHand(hand: Float32Array): Float32Array {
  const out = new Float32Array(COORDS_PER_HAND);

  if (isHandMissing(hand)) {
    out.fill(MISSING);
    return out;
  }

  const wristX = hand[0];
  const wristY = hand[1];
  const wristZ = hand[2];

  let maxDist = 0;
  for (let i = 0; i < LANDMARKS_PER_HAND; i++) {
    const x = hand[i * 3] - wristX;
    const y = hand[i * 3 + 1] - wristY;
    const z = hand[i * 3 + 2] - wristZ;
    out[i * 3] = x;
    out[i * 3 + 1] = y;
    out[i * 3 + 2] = z;

    const dist = Math.sqrt(x * x + y * y + z * z);
    if (dist > maxDist) maxDist = dist;
  }

  const scale = Math.max(maxDist, 1e-8);
  for (let i = 0; i < COORDS_PER_HAND; i++) {
    out[i] /= scale;
  }

  return out;
}

/**
 * Frames where MediaPipe found no hand (feature 1 === -1) copy the
 * nearest earlier frame that had a hand; frames before the first hand
 * copy that first hand. Mirrors fill_missing_frames() in
 * ml/preprocess_words.py (currently OFF there: USE_FILL = False).
 */
export function fillMissingFrames(frames: number[][]): number[][] {
  const missing = (f: number[]) => Math.abs(f[1] + 1) < 1e-5;
  const first = frames.find((f) => !missing(f));
  if (!first) return frames;
  let last = first;
  return frames.map((f) => (missing(f) ? last : (last = f)));
}

/**
 * Normalises a sequence of raw 127-value frames for the model.
 * - fill:     gap-filling (must match USE_FILL in preprocess_words.py)
 * - addWrist: append the RAW wrist x, y of hand slot A (129 features,
 *             must match ADD_WRIST in preprocess_words.py). A missing hand
 *             gives -1, -1, exactly as in training.
 */
export function normalizeSequence(
  rawFrames: number[][],
  fill = false,
  addWrist = false
): Float32Array {
  const frames = fill ? fillMissingFrames(rawFrames) : rawFrames;
  const width = addWrist ? MODEL_FEATURES : FEATURES_PER_FRAME;
  const out = new Float32Array(frames.length * width);

  for (let f = 0; f < frames.length; f++) {
    const frame = frames[f];
    const base = f * width;

    out[base] = frame[0];

    for (const offset of [1, 1 + COORDS_PER_HAND]) {
      const hand = new Float32Array(COORDS_PER_HAND);
      for (let i = 0; i < COORDS_PER_HAND; i++) {
        hand[i] = frame[offset + i];
      }
      const normalized = normalizeHand(hand);
      for (let i = 0; i < COORDS_PER_HAND; i++) {
        out[base + offset + i] = normalized[i];
      }
    }

    if (addWrist) {
      out[base + FEATURES_PER_FRAME] = frame[1];     // raw wrist x (slot A)
      out[base + FEATURES_PER_FRAME + 1] = frame[2]; // raw wrist y (slot A)
    }
  }

  return out;
}