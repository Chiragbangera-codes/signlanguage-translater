import type { PredictionItem } from "../store/useTranslatorStore";
import { normalizeSequence, MODEL_FEATURES, SEQUENCE_LENGTH } from "./normalize";

type TF = typeof import("@tensorflow/tfjs");

export interface LocalPrediction {
  prediction: string;
  confidence: number;
  topPredictions: PredictionItem[];
  processingTimeMs: number;
}

interface LoadedModel {
  // One model, or several whose predictions are averaged (an ensemble).
  models: import("@tensorflow/tfjs").LayersModel[];
  labels: string[];
  tf: TF;
  features: number; // 127 (numbers model) or 129 (words model with wrist position)
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

/**
 * Optional /model/<mode>/members.json lists the model files of an ensemble,
 * e.g. ["model.json", "member2/model.json"]. Without it, the single
 * /model/<mode>/model.json is used, exactly as before.
 */
async function fetchMembers(mode: string): Promise<string[]> {
  try {
    const response = await fetch(`/model/${mode}/members.json`);
    if (!response.ok) return ["model.json"];
    const members = await response.json();
    if (Array.isArray(members) && members.length > 0) return members as string[];
  } catch {
    // no members.json -> single model
  }
  return ["model.json"];
}

export function loadLocalModel(mode: string): Promise<LoadedModel> {
  const existing = cache.get(mode);
  if (existing) return existing;

  const pending = (async (): Promise<LoadedModel> => {
    const tf = await import("@tensorflow/tfjs");
    const [members, labels] = await Promise.all([fetchMembers(mode), fetchLabels(mode)]);
    const models = await Promise.all(
      members.map((file) => tf.loadLayersModel(`/model/${mode}/${file}`))
    );

    // Read the feature count from the model itself, so a 127- and a
    // 129-feature model both work without code changes.
    const features = (models[0].inputs[0].shape as number[])[2];

    for (const model of models) {
      const outputShape = model.outputShape as number[];
      const numClasses = outputShape[outputShape.length - 1];
      const modelFeatures = (model.inputs[0].shape as number[])[2];
      if (numClasses !== labels.length || modelFeatures !== features) {
        models.forEach((m) => m.dispose());
        throw new Error(
          `Model outputs ${numClasses} classes / ${modelFeatures} features but labels.json lists ` +
            `${labels.length} classes (expected ${features} features). ` +
            `Re-export so all models and labels come from the same training setup.`
        );
      }
    }

    for (const model of models) {
      const warmup = tf.zeros([1, SEQUENCE_LENGTH, features]);
      const result = model.predict(warmup) as import("@tensorflow/tfjs").Tensor;
      await result.data();
      warmup.dispose();
      result.dispose();
    }

    lastError = null;
    return { models, labels, tf, features };
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
  const { models, labels, tf, features } = await loadLocalModel(mode);

  // Gap-filling is OFF (USE_FILL = False in ml/preprocess_words.py).
  // Wrist features are added only when the model was trained with them.
  const normalized = normalizeSequence(frames, false, features === MODEL_FEATURES);

  // Average the probabilities of every model in the ensemble.
  const probabilities = tf.tidy(() => {
    const input = tf.tensor3d(normalized, [1, SEQUENCE_LENGTH, features]);
    const outputs = models.map((m) => m.predict(input) as import("@tensorflow/tfjs").Tensor);
    return outputs.length === 1 ? outputs[0] : tf.stack(outputs).mean(0);
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