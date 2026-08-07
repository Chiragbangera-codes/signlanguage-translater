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

export function speakSentence(
  sentence: string,
  voiceName: string | null | undefined,
  rate: number
): boolean {
  if (!sentence.trim()) return false;
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return false;

  window.speechSynthesis.cancel();

  const spoken = expandDigits(sentence).toLowerCase();
  const utterance = new SpeechSynthesisUtterance(spoken);

  const voices = window.speechSynthesis.getVoices();
  const targetVoice = voices.find((voice) => voice.name === voiceName);
  if (targetVoice) {
    utterance.voice = targetVoice;
  }

  utterance.rate = rate;
  utterance.pitch = 1.0;
  window.speechSynthesis.speak(utterance);
  return true;
}
