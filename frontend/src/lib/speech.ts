const BELOW_TWENTY = [
  "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
  "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
  "seventeen", "eighteen", "nineteen",
];

const TENS_WORDS = [
  "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

const SCALES = ["", "thousand", "million", "billion"];

function convertChunk(chunk: number): string[] {
  const parts: string[] = [];
  const hundreds = Math.floor(chunk / 100);
  const remainder = chunk % 100;

  if (hundreds > 0) {
    parts.push(`${BELOW_TWENTY[hundreds]} hundred`);
  }

  if (remainder > 0) {
    if (remainder < 20) {
      parts.push(BELOW_TWENTY[remainder]);
    } else {
      const tens = Math.floor(remainder / 10);
      const ones = remainder % 10;
      parts.push(ones ? `${TENS_WORDS[tens]}-${BELOW_TWENTY[ones]}` : TENS_WORDS[tens]);
    }
  }

  return parts.length ? parts : [];
}

export function numberToWords(n: number): string {
  if (!Number.isFinite(n)) return String(n);
  if (n === 0) return "zero";
  if (n < 0) return `minus ${numberToWords(-n)}`;

  const words: string[] = [];
  let remaining = n;
  let scaleIndex = 0;

  while (remaining > 0) {
    const chunk = remaining % 1000;
    remaining = Math.floor(remaining / 1000);
    if (chunk > 0) {
      const chunkWords = convertChunk(chunk);
      if (scaleIndex > 0 && SCALES[scaleIndex]) {
        chunkWords.push(SCALES[scaleIndex]);
      }
      words.unshift(...chunkWords);
    }
    scaleIndex += 1;
  }

  return words.join(" ");
}

export function expandDigits(text: string): string {
  return text.replace(/\d+/g, (match) => {
    const num = parseInt(match, 10);
    return Number.isNaN(num) ? match : numberToWords(num);
  });
}

/**
 * Picks a voice for `bcp47`, preferring an exact locale match, then any voice
 * for the same base language. Returns null when the browser ships no voice for
 * that language — the utterance still carries `lang`, so some platforms cope.
 */
export function findVoiceForLanguage(
  voices: SpeechSynthesisVoice[],
  bcp47: string
): SpeechSynthesisVoice | null {
  const target = bcp47.toLowerCase();
  const base = target.split("-")[0];

  return (
    voices.find((voice) => voice.lang.toLowerCase().replace("_", "-") === target) ??
    voices.find((voice) => voice.lang.toLowerCase().split(/[-_]/)[0] === base) ??
    null
  );
}

/**
 * Speaks `sentence`. When `bcp47` is given, a voice for that language wins over
 * `voiceName` — the configured voice is almost always English and would read a
 * Hindi or Tamil sentence as gibberish.
 */
export function speakSentence(
  sentence: string,
  voiceName: string | null | undefined,
  rate: number,
  bcp47: string = "en-US"
): boolean {
  if (!sentence.trim()) return false;
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return false;

  window.speechSynthesis.cancel();

  const isEnglish = bcp47.toLowerCase().startsWith("en");

  // Digit expansion and lowercasing are English-specific: "100" is already
  // read correctly by a Hindi voice, and lowercasing breaks nothing in
  // non-cased scripts but does strip meaning from cased non-English text.
  const spoken = isEnglish ? expandDigits(sentence).toLowerCase() : sentence;
  const utterance = new SpeechSynthesisUtterance(spoken);

  const voices = window.speechSynthesis.getVoices();
  const languageVoice = findVoiceForLanguage(voices, bcp47);
  const namedVoice = voices.find((voice) => voice.name === voiceName) ?? null;

  // For a non-English target the language match wins; for English the user's
  // configured voice wins.
  const targetVoice = isEnglish ? namedVoice ?? languageVoice : languageVoice ?? namedVoice;
  if (targetVoice) {
    utterance.voice = targetVoice;
  }

  utterance.lang = targetVoice?.lang || bcp47;
  utterance.rate = rate;
  utterance.pitch = 1.0;
  window.speechSynthesis.speak(utterance);
  return true;
}
