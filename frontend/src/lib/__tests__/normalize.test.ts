import { describe, expect, it } from "vitest";
import fixtures from "./normalize.fixtures.json";
import {
  COORDS_PER_HAND,
  FEATURES_PER_FRAME,
  isHandMissing,
  MISSING,
  normalizeHand,
  normalizeSequence,
} from "../normalize";

/**
 * Parity tests against `normalize_hands_batch` in ml/preprocess.py.
 *
 * The fixtures were produced by running the Python function itself (see
 * normalize.fixtures.json). If these fail, the browser is feeding the model
 * different numbers than training did — which produces confident wrong
 * predictions rather than an error, so this is the guard that catches it.
 */

type Fixture = { input: number[][]; expected: number[][] };
const cases = fixtures as unknown as Record<string, Fixture>;

describe("normalizeHand — parity with ml/preprocess.py", () => {
  for (const [name, { input, expected }] of Object.entries(cases)) {
    it(`matches Python for "${name}"`, () => {
      input.forEach((hand, row) => {
        const actual = normalizeHand(Float32Array.from(hand));
        expect(actual.length).toBe(COORDS_PER_HAND);
        actual.forEach((value, i) => {
          // float32 round-tripping through JSON; 1e-6 is far tighter than any
          // difference that could change a prediction.
          expect(value).toBeCloseTo(expected[row][i], 6);
        });
      });
    });
  }
});

describe("isHandMissing", () => {
  it("treats an all -1.0 hand as padding", () => {
    expect(isHandMissing(new Array(COORDS_PER_HAND).fill(MISSING))).toBe(true);
  });

  it("treats a -1.0 wrist as padding even when later values are real", () => {
    const hand = new Array(COORDS_PER_HAND).fill(0.5);
    hand[0] = MISSING;
    expect(isHandMissing(hand)).toBe(true);
  });

  it("treats a detected hand as present", () => {
    expect(isHandMissing(new Array(COORDS_PER_HAND).fill(0.3))).toBe(false);
  });
});

describe("normalizeSequence", () => {
  it("produces a flat array of frames x 127", () => {
    const frames = Array.from({ length: 30 }, () =>
      Array.from({ length: FEATURES_PER_FRAME }, (_, i) => (i === 0 ? 0 : Math.random()))
    );
    expect(normalizeSequence(frames).length).toBe(30 * FEATURES_PER_FRAME);
  });

  it("carries the uses_two_hands flag through untouched", () => {
    const frame = new Array(FEATURES_PER_FRAME).fill(0.5);
    frame[0] = 1.0;
    const out = normalizeSequence([frame]);
    expect(out[0]).toBe(1.0);
  });

  it("keeps a padded second hand as -1.0", () => {
    const frame = new Array(FEATURES_PER_FRAME).fill(0.5);
    frame[0] = 0.0;
    for (let i = 0; i < COORDS_PER_HAND; i++) {
      frame[1 + COORDS_PER_HAND + i] = MISSING;
    }
    const out = normalizeSequence([frame]);
    for (let i = 0; i < COORDS_PER_HAND; i++) {
      expect(out[1 + COORDS_PER_HAND + i]).toBe(MISSING);
    }
  });
});
