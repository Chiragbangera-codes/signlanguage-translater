const QUESTION_WORDS = new Set([
  "how", "what", "where", "when", "who", "why", "which",
]);

function normalizeWords(raw: string): string[] {
  return raw
    .split(/\s+/)
    .map((w) => w.toLowerCase().trim())
    .filter((w) => w.length > 0);
}

function hasWord(words: string[], target: string): boolean {
  return words.includes(target.toLowerCase());
}

interface GrammarPattern {
  match: (words: string[]) => boolean;
  apply: (words: string[]) => string;
}

const PATTERNS: GrammarPattern[] = [
  {
    match: (w) => hasWord(w, "how") && hasWord(w, "you"),
    apply: () => "How are you?",
  },
  {
    match: (w) => hasWord(w, "what") && hasWord(w, "name"),
    apply: () => "What is your name?",
  },
  {
    match: (w) => hasWord(w, "where") && hasWord(w, "go"),
    apply: () => "Where are you going?",
  },
  {
    match: (w) =>
      hasWord(w, "what") &&
      hasWord(w, "you") &&
      hasWord(w, "do"),
    apply: () => "What are you doing?",
  },
  {
    match: (w) => hasWord(w, "what") && hasWord(w, "want") && hasWord(w, "you"),
    apply: () => "What do you want?",
  },
  {
    match: (w) => hasWord(w, "what") && hasWord(w, "like") && hasWord(w, "you"),
    apply: () => "What do you like?",
  },
  {
    match: (w) =>
      hasWord(w, "what") &&
      hasWord(w, "like") &&
      (hasWord(w, "i") || hasWord(w, "me")),
    apply: () => "What do I like?",
  },
  {
    match: (w) => hasWord(w, "nice") && hasWord(w, "meet") && hasWord(w, "you"),
    apply: () => "Nice to meet you.",
  },
  {
    match: (w) =>
      (hasWord(w, "i") || hasWord(w, "me")) &&
      hasWord(w, "love") &&
      hasWord(w, "you"),
    apply: () => "I love you.",
  },
  {
    match: (w) => hasWord(w, "good") && hasWord(w, "morning"),
    apply: () => "Good morning.",
  },
  {
    match: (w) => hasWord(w, "good") && hasWord(w, "afternoon"),
    apply: () => "Good afternoon.",
  },
  {
    match: (w) => hasWord(w, "good") && hasWord(w, "evening"),
    apply: () => "Good evening.",
  },
  {
    match: (w) => hasWord(w, "good") && hasWord(w, "night"),
    apply: () => "Good night.",
  },
  {
    match: (w) => hasWord(w, "thank") && hasWord(w, "you"),
    apply: () => "Thank you.",
  },
  {
    match: (w) => hasWord(w, "thank"),
    apply: () => "Thank you.",
  },
  {
    match: (w) => hasWord(w, "sorry"),
    apply: () => "I'm sorry.",
  },
  {
    match: (w) => {
      if (w.length === 1 && hasWord(w, "hello")) return true;
      return false;
    },
    apply: () => "Hello.",
  },
  {
    match: (w) => w.length === 1 && hasWord(w, "yes"),
    apply: () => "Yes.",
  },
  {
    match: (w) => w.length === 1 && hasWord(w, "no"),
    apply: () => "No.",
  },
  {
    match: (w) => w.length <= 2 && hasWord(w, "please"),
    apply: () => "Please.",
  },
  {
    match: (w) => hasWord(w, "i") && hasWord(w, "fine"),
    apply: () => "I am fine.",
  },
  {
    match: (w) => hasWord(w, "i") && hasWord(w, "hungry"),
    apply: () => "I am hungry.",
  },
  {
    match: (w) => hasWord(w, "i") && hasWord(w, "thirsty"),
    apply: () => "I am thirsty.",
  },
  {
    match: (w) => hasWord(w, "i") && hasWord(w, "tired"),
    apply: () => "I am tired.",
  },
  {
    match: (w) => hasWord(w, "i") && hasWord(w, "happy"),
    apply: () => "I am happy.",
  },
  {
    match: (w) => hasWord(w, "i") && hasWord(w, "sad"),
    apply: () => "I am sad.",
  },
  {
    match: (w) => hasWord(w, "i") && hasWord(w, "study"),
    apply: () => "I am studying.",
  },
  {
    match: (w) => hasWord(w, "you") && hasWord(w, "study"),
    apply: () => "Are you studying?",
  },
  {
    match: (w) =>
      (hasWord(w, "i") || hasWord(w, "me")) &&
      hasWord(w, "want") &&
      hasWord(w, "water"),
    apply: () => "I want water.",
  },
  {
    match: (w) =>
      (hasWord(w, "i") || hasWord(w, "me")) &&
      hasWord(w, "want") &&
      hasWord(w, "food"),
    apply: () => "I want food.",
  },
  {
    match: (w) =>
      (hasWord(w, "i") || hasWord(w, "me")) && hasWord(w, "understand"),
    apply: () => "I understand.",
  },
  {
    match: (w) =>
      (hasWord(w, "you") && hasWord(w, "understand")) ||
      (hasWord(w, "you") && hasWord(w, "know")),
    apply: () => "Do you understand?",
  },
  {
    match: (w) => hasWord(w, "i") && hasWord(w, "help"),
    apply: () => "I need help.",
  },
  {
    match: (w) => hasWord(w, "come") && hasWord(w, "in"),
    apply: () => "Come in.",
  },
  {
    match: (w) => hasWord(w, "go") && hasWord(w, "away"),
    apply: () => "Go away.",
  },
  {
    match: (w) => hasWord(w, "stop"),
    apply: () => "Stop.",
  },
  {
    match: (w) => hasWord(w, "more"),
    apply: () => "More, please.",
  },
  {
    match: (w) => hasWord(w, "finish"),
    apply: () => "I am finished.",
  },
  {
    match: (w) =>
      hasWord(w, "i") &&
      hasWord(w, "want") &&
      (hasWord(w, "home") || hasWord(w, "go") || hasWord(w, "house")),
    apply: () => "I want to go home.",
  },
  {
    match: (w) => hasWord(w, "home") && hasWord(w, "good"),
    apply: () => "Welcome home.",
  },
  {
    match: (w) => hasWord(w, "welcome"),
    apply: () => "Welcome.",
  },
  {
    match: (w) => hasWord(w, "happy") && hasWord(w, "birthday"),
    apply: () => "Happy birthday.",
  },
  {
    match: (w) => hasWord(w, "congratulations"),
    apply: () => "Congratulations!",
  },
];

function applyFallback(words: string[]): string {
  if (words.length === 0) return "";

  // Heuristic: if the sentence starts with a question word, treat as a question.
  const startsWithQuestion = QUESTION_WORDS.has(words[0]);

  let sentence: string;

  if (startsWithQuestion) {
    // Detect question patterns and restructure to English SVO order.
    sentence = buildQuestion(words);
  } else if (
    hasWord(words, "i") ||
    hasWord(words, "me") ||
    hasWord(words, "my") ||
    hasWord(words, "we")
  ) {
    // Subject-prominent sentence: ensure "I" is capitalized and in subject position.
    sentence = buildStatement(words);
  } else if (hasWord(words, "you")) {
    sentence = buildStatement(words);
  } else {
    // Generic fallback: join words, capitalize, punctuate.
    sentence = words.join(" ");
  }

  // Capitalize "i" pronoun
  sentence = sentence.replace(/\bi\b/g, "I");

  // Capitalize first letter
  sentence = sentence.charAt(0).toUpperCase() + sentence.slice(1);

  // Ensure proper ending punctuation
  if (!sentence.endsWith(".") && !sentence.endsWith("?") && !sentence.endsWith("!")) {
    sentence += startsWithQuestion || sentence.includes("?") ? "." : ".";
  }

  return sentence;
}

function buildQuestion(words: string[]): string {
  // Common ISL question patterns:
  // "what name you" -> "What is your name?"
  // "how you" -> "How are you?"
  // "where go you" -> "Where are you going?"
  // "you what doing" -> "What are you doing?"

  if (words.includes("what") && words.includes("name")) {
    return "What is your name?";
  }

  if (words.includes("how") && words.includes("you")) {
    return "How are you?";
  }

  if (words.includes("where") && words.includes("go")) {
    return "Where are you going?";
  }

  if (
    words.includes("what") &&
    words.some((w) => w === "do" || w === "doing") &&
    words.includes("you")
  ) {
    return "What are you doing?";
  }

  if (words.includes("what") && words.includes("want") && words.includes("you")) {
    return "What do you want?";
  }

  // Generic question: capitalize and add question mark
  return words.join(" ") + "?";
}

function buildStatement(words: string[]): string {
  // Generic statement: just join and capitalize.
  return words.join(" ");
}

export function constructMeaningfulSentence(rawWords: string): string {
  const words = normalizeWords(rawWords);

  if (words.length === 0) return "";

  // Try each pattern in order; first match wins.
  for (const pattern of PATTERNS) {
    if (pattern.match(words)) {
      return pattern.apply(words);
    }
  }

  // No explicit pattern matched — apply heuristic fallback.
  return applyFallback(words);
}
