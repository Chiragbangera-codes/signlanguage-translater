import type { PredictionItem } from "../store/useTranslatorStore";
import { normalizeSequence, FEATURES_PER_FRAME, SEQUENCE_LENGTH } from "./normalize";

type TF = typeof import("@tensorflow/tfjs");

export interface LocalPrediction {
  prediction: string;
  confidence: number;
  topPredictions: PredictionItem[];
  processingTimeMs: number;
}

interface LoadedModel {
  model: import("@tensorflow/tfjs").LayersModel;
  labels: string[];
  tf: TF;
}

const cache = new Map<string, Promise<LoadedModel>>();

let lastError: string | null = null;

export function getLocalModelError(): string | null {
  return lastError;
}

async function fetchLabels(mode: string): Promise<string[]> {
  const response = await fetch(`/model/${mode}/labels.json`);
  if (!response.ok) {
    throw new Error(`labels.json missing for "${mode}" (${response.status})`);
  }
  const labels = await response.json();
  if (!Array.isArray(labels) || labels.length === 0) {
    throw new Error(`labels.json for "${mode}" is empty or malformed`);
  }
  return labels as string[];
}

export function loadLocalModel(mode: string): Promise<LoadedModel> {
  const existing = cache.get(mode);
  if (existing) return existing;

  const pending = (async (): Promise<LoadedModel> => {
    const tf = await import("@tensorflow/tfjs");
    const [model, labels] = await Promise.all([
      tf.loadLayersModel(`/model/${mode}/model.json`),
      fetchLabels(mode),
    ]);

    const outputShape = model.outputShape as number[];
    const numClasses = outputShape[outputShape.length - 1];
    if (numClasses !== labels.length) {
      model.dispose();
      throw new Error(
        `Model outputs ${numClasses} classes but labels.json lists ${labels.length}. ` +
          `Re-run scripts/export_tfjs.py so both come from the same training run.`
      );
    }

    const warmup = tf.zeros([1, SEQUENCE_LENGTH, FEATURES_PER_FRAME]);
    const result = model.predict(warmup) as import("@tensorflow/tfjs").Tensor;
    await result.data();
    warmup.dispose();
    result.dispose();

    lastError = null;
    return { model, labels, tf };
  })();

  pending.catch((e) => {
    lastError = e instanceof Error ? e.message : String(e);
    cache.delete(mode);
  });

  cache.set(mode, pending);
  return pending;
}

export async function predictLocally(
  frames: number[][],
  mode: string
): Promise<LocalPrediction> {
  const started = performance.now();
  const { model, labels, tf } = await loadLocalModel(mode);

  const normalized = normalizeSequence(frames);

  const probabilities = tf.tidy(() => {
    const input = tf.tensor3d(normalized, [1, SEQUENCE_LENGTH, FEATURES_PER_FRAME]);
    return model.predict(input) as import("@tensorflow/tfjs").Tensor;
  });

  const scores = await probabilities.data();
  probabilities.dispose();

  const ranked = Array.from(scores)
    .map((confidence, index) => ({
      label: labels[index],
      confidence: Math.round(confidence * 100 * 100) / 100,
    }))
    .sort((a, b) => b.confidence - a.confidence);

  const topPredictions = ranked.slice(0, 3);

  return {
    prediction: topPredictions[0].label,
    confidence: topPredictions[0].confidence,
    topPredictions,
    processingTimeMs: Math.round((performance.now() - started) * 100) / 100,
  };
}

export async function isLocalModelAvailable(mode: string): Promise<boolean> {
  try {
    const response = await fetch(`/model/${mode}/model.json`, { method: "HEAD" });
    return response.ok;
  } catch {
    return false;
  }
}
