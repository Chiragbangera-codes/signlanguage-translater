/**
 * POST /api/sentence  —  turns recognised sign glosses into a natural sentence
 * and translates it, using Google Gemini.
 *
 * This runs as a Vercel serverless function next to the app, so the deployed
 * app needs no separate Python backend. It mirrors
 * backend/app/services/sentence_service.py (same prompt, same JSON shape).
 *
 * Needs the environment variable GEMINI_API_KEY (set it in Vercel ->
 * Project -> Settings -> Environment Variables). Optional: SENTENCE_MODEL.
 * Without a key it returns 503 and the app falls back to its offline rules.
 */

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const SYSTEM_PROMPT = `You convert sign language gloss into natural spoken language.

The input is an ordered list of signs recognized from a webcam, oldest first. Sign languages drop articles, copulas, plural markers and tense, and their word order differs from spoken language, so the gloss is never a finished sentence.

Rules:
- Produce ONE natural, grammatical utterance that a hearing person would say.
- Add the function words, tense, agreement and politeness the gloss implies (is/are, the/a, to, do, question inversion). Never add facts the gloss does not support: no invented names, numbers, places, or reasons.
- Adjacent repeated glosses mean one sign was held too long. Collapse them.
- A gloss you cannot place sensibly may be dropped rather than forced in.
- Digits stand for the number they spell out; keep them as spoken numbers.
- Punctuate as a question when the gloss carries a question sign (what, how, where, when, who, why, which) or clearly asks something.
- No quotes, no gloss echo, no commentary — just the utterance.`;

const STYLE_INSTRUCTIONS: Record<string, string> = {
  natural:
    "Length: match the gloss. A two-sign gloss becomes a short sentence; a longer gloss becomes a correspondingly longer one. Do not pad.",
  expanded:
    "Length: the signer is holding a longer conversation and wants to say more than the gloss literally spells out. Expand it into a fuller, natural message of two or three connected sentences, joining the glosses into clauses with the conjunctions and phrasing a fluent speaker would use. Stay strictly within what the gloss supports — elaborate the phrasing, never the facts.",
};

// Tried in order; a model ID that no longer exists is skipped.
const MODELS = [process.env.SENTENCE_MODEL, "gemini-flash-latest", "gemini-2.5-flash"].filter(
  (m): m is string => Boolean(m && m.trim())
);

type Generated = { english: string; translation: string };

// Small per-instance cache so repeated sentences are instant and free.
const cache = new Map<string, Generated>();
const CACHE_SIZE = 256;

function error(status: number, message: string) {
  return Response.json({ detail: message, message }, { status });
}

async function callGemini(key: string, model: string, prompt: string): Promise<Generated> {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`;
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-goog-api-key": key },
    body: JSON.stringify({
      systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] },
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      generationConfig: {
        temperature: 0.3,
        responseMimeType: "application/json",
        responseSchema: {
          type: "OBJECT",
          properties: { english: { type: "STRING" }, translation: { type: "STRING" } },
          required: ["english", "translation"],
        },
      },
    }),
  });

  if (!response.ok) {
    const body = await response.text();
    const err = new Error(`Gemini ${model} returned ${response.status}: ${body.slice(0, 300)}`);
    (err as Error & { status?: number }).status = response.status;
    throw err;
  }

  const data = await response.json();
  const text: string = data?.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
  if (!text.trim()) {
    throw new Error(`Gemini ${model} returned no sentence (${data?.candidates?.[0]?.finishReason ?? "no candidates"}).`);
  }
  const parsed = JSON.parse(text) as Generated;
  if (!parsed.english || !parsed.english.trim()) throw new Error("Empty sentence from Gemini.");
  if (!parsed.translation || !parsed.translation.trim()) parsed.translation = parsed.english;
  return parsed;
}

export async function POST(request: Request) {
  const started = performance.now();

  const key = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
  if (!key) {
    return error(503, "Sentence generation is not configured. Set GEMINI_API_KEY in the Vercel project settings.");
  }

  let body: { words?: unknown; language?: unknown; language_name?: unknown; style?: unknown; mode?: unknown };
  try {
    body = await request.json();
  } catch {
    return error(400, "Request body must be JSON.");
  }

  const words = Array.isArray(body.words)
    ? body.words.filter((w): w is string => typeof w === "string").map((w) => w.trim()).filter(Boolean)
    : [];
  if (words.length === 0 || words.length > 64) return error(400, "Send 1 to 64 words.");

  const language = typeof body.language === "string" && body.language.length <= 16 ? body.language : "en";
  const rawName = typeof body.language_name === "string" ? body.language_name.trim().slice(0, 64) : "";
  const languageName = rawName || (language.toLowerCase().startsWith("en") ? "English" : language);
  const style = body.style === "expanded" ? "expanded" : "natural";
  const mode = body.mode === "numbers" ? "numbers" : "words";

  const cacheKey = JSON.stringify([words.map((w) => w.toLowerCase()), language, languageName, style, mode]);
  let result = cache.get(cacheKey);
  const fromCache = Boolean(result);

  if (!result) {
    const modeNote =
      mode === "numbers"
        ? "These glosses are digits signed one at a time; read them as the number they spell."
        : "These glosses are word signs.";
    const prompt =
      `Sign glosses (oldest first): ${words.join(" ")}\n` +
      `${modeNote}\n` +
      `${STYLE_INSTRUCTIONS[style]}\n\n` +
      `Return the utterance in English, and its translation into ${languageName}. ` +
      `If ${languageName} is English, the translation is the same text.`;

    let lastError: unknown = null;
    for (const model of MODELS) {
      try {
        result = await callGemini(key, model, prompt);
        break;
      } catch (e) {
        lastError = e;
        const status = (e as { status?: number }).status;
        // Only move on to the next model when this one doesn't exist.
        if (status !== 404 && status !== 400) break;
      }
    }

    if (!result) {
      const message = lastError instanceof Error ? lastError.message : String(lastError);
      console.error("[sentence]", message);
      return error(503, `Sentence generation failed: ${message}`);
    }

    cache.set(cacheKey, result);
    if (cache.size > CACHE_SIZE) cache.delete(cache.keys().next().value as string);
  }

  return Response.json({
    sentence: result.translation,
    english: result.english,
    language,
    language_name: languageName,
    source: fromCache ? "cache" : "llm",
    processing_time_ms: Math.round((performance.now() - started) * 100) / 100,
  });
}
