import { getLanguage } from "./languages";

export type SentenceStyle = "natural" | "expanded";

export interface SentenceResult {
  sentence: string;
  english: string;
  language: string;
  languageName: string;
  source: string;
  processingTimeMs: number;
}

export class SentenceApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "SentenceApiError";
    this.status = status;
  }
}

function apiBase(): string {
  return process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";
}

export async function requestSentence(
  rawWords: string,
  languageCode: string,
  style: SentenceStyle = "natural",
  mode: string = "words"
): Promise<SentenceResult> {
  const words = rawWords
    .split(/\s+/)
    .map((w) => w.trim())
    .filter((w) => w.length > 0);

  if (words.length === 0) {
    throw new SentenceApiError("No words to build a sentence from.", 400);
  }

  const language = getLanguage(languageCode);

  let response: Response;
  try {
    response = await fetch(`${apiBase()}/sentence`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        words,
        language: language.code,
        language_name: language.name,
        style,
        mode,
      }),
    });
  } catch {
    throw new SentenceApiError("Sentence service unreachable.", 0);
  }

  if (!response.ok) {
    let detail = `Sentence service returned ${response.status}.`;
    try {
      const errorData = await response.json();
      detail = errorData.message || errorData.detail || detail;
    } catch {
    }
    throw new SentenceApiError(detail, response.status);
  }

  const data = await response.json();

  return {
    sentence: data.sentence,
    english: data.english,
    language: data.language,
    languageName: data.language_name,
    source: data.source,
    processingTimeMs: data.processing_time_ms,
  };
}
