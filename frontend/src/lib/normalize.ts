export const LANDMARKS_PER_HAND = 21;
export const COORDS_PER_HAND = LANDMARKS_PER_HAND * 3;
export const FEATURES_PER_FRAME = 1 + COORDS_PER_HAND * 2;
export const SEQUENCE_LENGTH = 30;

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

export function normalizeSequence(frames: number[][]): Float32Array {
  const out = new Float32Array(frames.length * FEATURES_PER_FRAME);

  for (let f = 0; f < frames.length; f++) {
    const frame = frames[f];
    const base = f * FEATURES_PER_FRAME;

    out[base] = frame[0];

    for (const [slot, offset] of [
      [0, 1],
      [1, 1 + COORDS_PER_HAND],
    ] as const) {
      void slot;
      const hand = new Float32Array(COORDS_PER_HAND);
      for (let i = 0; i < COORDS_PER_HAND; i++) {
        hand[i] = frame[offset + i];
      }
      const normalized = normalizeHand(hand);
      for (let i = 0; i < COORDS_PER_HAND; i++) {
        out[base + offset + i] = normalized[i];
      }
    }
  }

  return out;
}
