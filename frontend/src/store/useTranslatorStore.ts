import { create } from "zustand";
import { constructMeaningfulSentence } from "../lib/grammar";
import { requestSentence } from "../lib/sentence";
import type { SentenceStyle } from "../lib/sentence";
import { DEFAULT_LANGUAGE_CODE, getLanguage } from "../lib/languages";

export interface PredictionItem {
  label: string;
  confidence: number;
}

export type TranslatorMode = "numbers" | "words";

/** An archived sentence plus the language it was generated in, so replaying it
 *  picks the right synthesizer voice. */
export interface HistoryEntry {
  text: string;
  language: string;
}

interface TranslatorState {
  // Webcam & Pipeline status
  webcamActive: boolean;
  isTranslating: boolean;
  modelLoaded: boolean;
  statusBarMessage: string;

  // Active prediction states
  currentPrediction: string | null;
  currentConfidence: number;
  topPredictions: PredictionItem[];
  processingTimeMs: number;

  // Language construction states
  currentWord: string;
  constructedSentence: string;
  meaningfulSentence: string | null;
  /** English reference text when the sentence was translated. */
  englishSentence: string | null;
  /** Language `meaningfulSentence` is actually written in — not the requested
   *  target, which differs whenever the offline fallback ran. */
  sentenceLanguage: string;
  isConstructing: boolean;
  targetLanguage: string;
  sentenceStyle: SentenceStyle;
  history: HistoryEntry[];

  // Settings State
  confidenceThreshold: number; // 0-100%
  speechRate: number; // 0.5-2.0
  selectedVoiceName: string | null;
  cameraMirrored: boolean;
  activeMode: TranslatorMode;

  // Performance stats state
  cameraFps: number;
  apiHealthy: boolean;
  /** Which path served the last prediction: on-device, remote API, or neither yet. */
  inferenceSource: "local" | "api" | null;

  // Actions
  setWebcamActive: (active: boolean) => void;
  setIsTranslating: (translating: boolean) => void;
  setModelLoaded: (loaded: boolean) => void;
  setStatusBarMessage: (msg: string) => void;
  setPrediction: (
    prediction: string | null,
    confidence: number,
    topPredictions: PredictionItem[],
    processingTimeMs: number
  ) => void;

  // Word & Sentence manipulation
  appendLetterToWord: (letter: string) => void;
  appendWordToCurrentWord: (word: string) => void;
  backspaceWord: () => void;
  clearWord: () => void;
  commitWordToSentence: () => void;
  addWordToSentence: (word: string) => void;
  clearSentence: () => void;
  addSentenceToHistory: (sentence: string, language?: string) => void;
  clearHistory: () => void;

  // Grammar correction
  setMeaningfulSentence: (sentence: string | null) => void;
  constructMeaningfulSentence: () => Promise<void>;
  setTargetLanguage: (code: string) => void;
  setSentenceStyle: (style: SentenceStyle) => void;

  // Settings & Performance Actions
  setConfidenceThreshold: (val: number) => void;
  setSpeechRate: (val: number) => void;
  setSelectedVoiceName: (name: string | null) => void;
  setCameraMirrored: (val: boolean) => void;
  setActiveMode: (mode: TranslatorMode) => void;
  setCameraFps: (fps: number) => void;
  setApiHealthy: (healthy: boolean) => void;
  setInferenceSource: (source: "local" | "api" | null) => void;
}

export const useTranslatorStore = create<TranslatorState>((set, get) => ({
  // Initial states
  webcamActive: false,
  isTranslating: false,
  modelLoaded: true,
  statusBarMessage: "System Ready. Connect webcam to start translation.",
  
  currentPrediction: null,
  currentConfidence: 0,
  topPredictions: [],
  processingTimeMs: 0,
  
  currentWord: "",
  constructedSentence: "",
  meaningfulSentence: null,
  englishSentence: null,
  sentenceLanguage: DEFAULT_LANGUAGE_CODE,
  isConstructing: false,
  targetLanguage: DEFAULT_LANGUAGE_CODE,
  sentenceStyle: "natural",
  history: [
    { text: "HELLO WORLD", language: "en" },
    { text: "WELCOME TO SIGNSPEAK AI", language: "en" },
    { text: "INDIAN SIGN LANGUAGE IS AMAZING", language: "en" }
  ],

  // Default settings
  // A 33-class softmax rarely exceeds 80% even when correct, so the old gate
  // discarded most valid predictions and the app felt dead. The stabilizer
  // (6-of-10 majority + 700ms hold) is what filters noise, not this threshold.
  confidenceThreshold: 55,
  speechRate: 1.0,
  selectedVoiceName: null,
  cameraMirrored: true,
  // Words is the mode with a trained model and an exported on-device bundle;
  // Numbers has neither, so defaulting to it silently fell back to the API.
  activeMode: "words",

  // Performance stats
  cameraFps: 0,
  apiHealthy: true,
  inferenceSource: null,

  // Actions
  setWebcamActive: (active) => set(() => ({
    webcamActive: active,
    statusBarMessage: active 
      ? "Webcam stream active. Ready for tracking." 
      : "Webcam disabled. Ready to connect."
  })),
  
  setIsTranslating: (translating) => set(() => ({
    isTranslating: translating,
    statusBarMessage: translating 
      ? "Translating gestures. Hold gesture for 700ms to log." 
      : "Translation paused."
  })),

  
  setModelLoaded: (loaded) => set({ modelLoaded: loaded }),
  
  setStatusBarMessage: (msg) => set({ statusBarMessage: msg }),
  
  setPrediction: (prediction, confidence, topPredictions, processingTimeMs) => set({
    currentPrediction: prediction,
    currentConfidence: confidence,
    topPredictions,
    processingTimeMs
  }),
  
  appendLetterToWord: (letter) => set((state) => ({
    currentWord: state.currentWord + letter,
    statusBarMessage: `Registered letter: ${letter}`
  })),

  appendWordToCurrentWord: (word) => set((state) => ({
    currentWord: state.currentWord
      ? `${state.currentWord} ${word}`
      : word,
    statusBarMessage: `Registered word: ${word}`
  })),

  backspaceWord: () => set((state) => {
    if (!state.currentWord.trim()) {
      return { currentWord: "", statusBarMessage: "Word is empty." };
    }
    if (state.activeMode === "words") {
      const parts = state.currentWord.trim().split(" ");
      parts.pop();
      const newWord = parts.join(" ");
      return {
        currentWord: newWord,
        statusBarMessage: newWord ? "Removed last word." : "Word cleared."
      };
    }
    return {
      currentWord: state.currentWord.slice(0, -1),
      statusBarMessage: "Deleted last character."
    };
  }),
  
  clearWord: () => set({ 
    currentWord: "",
    statusBarMessage: "Word cleared."
  }),
  
  commitWordToSentence: () => set((state) => {
    if (!state.currentWord.trim()) return {};
    const newSentence = state.constructedSentence 
      ? `${state.constructedSentence} ${state.currentWord}` 
      : state.currentWord;
    return {
      constructedSentence: newSentence,
      currentWord: "",
      statusBarMessage: `Committed word: ${state.currentWord}`
    };
  }),
  
  clearSentence: () => set({ 
    constructedSentence: "",
    meaningfulSentence: null,
    englishSentence: null,
    statusBarMessage: "Sentence cleared."
  }),

  addWordToSentence: (word) => set((state) => {
    const trimmed = word.trim();
    if (!trimmed) return {};
    const newSentence = state.constructedSentence
      ? `${state.constructedSentence} ${trimmed}`
      : trimmed;
    return {
      constructedSentence: newSentence,
      currentWord: "",
      statusBarMessage: `Added word: ${trimmed}`,
    };
  }),
  
  addSentenceToHistory: (sentence, language = "en") => set((state) => ({
    history: [{ text: sentence, language }, ...state.history],
    statusBarMessage: "Sentence committed to history."
  })),
  
  clearHistory: () => set({ 
    history: [],
    statusBarMessage: "History cleared."
  }),

  setMeaningfulSentence: (sentence) => set({
    meaningfulSentence: sentence,
    englishSentence: null,
    sentenceLanguage: "en",
    statusBarMessage: sentence
      ? "Meaningful sentence constructed."
      : "Meaningful sentence cleared."
  }),

  setTargetLanguage: (code) => set({
    targetLanguage: code,
    statusBarMessage: `Output language set to ${getLanguage(code).name}.`
  }),

  setSentenceStyle: (style) => set({
    sentenceStyle: style,
    statusBarMessage:
      style === "expanded"
        ? "Sentence style: expanded (fuller, multi-clause output)."
        : "Sentence style: natural (matches the signed words)."
  }),

  // Sends the glosses to the backend, which builds a natural sentence of any
  // length and translates it. Falls back to the local rule table when the
  // service is unreachable or unconfigured, so the app still works offline.
  constructMeaningfulSentence: async () => {
    const state = get();
    const raw = state.constructedSentence.trim();

    if (!raw) {
      set({
        meaningfulSentence: null,
        englishSentence: null,
        statusBarMessage: "No words to construct a sentence from."
      });
      return;
    }

    if (state.isConstructing) return;

    set({ isConstructing: true, statusBarMessage: "Constructing sentence..." });

    try {
      const result = await requestSentence(
        raw,
        state.targetLanguage,
        state.sentenceStyle,
        state.activeMode
      );

      set({
        meaningfulSentence: result.sentence,
        sentenceLanguage: result.language,
        englishSentence:
          result.language === "en" || result.english === result.sentence
            ? null
            : result.english,
        isConstructing: false,
        statusBarMessage: `Sentence constructed in ${result.languageName} (${Math.round(result.processingTimeMs)}ms).`
      });
    } catch {
      // Offline fallback: local rules, English only.
      const fallback = constructMeaningfulSentence(raw);
      const isEnglish = state.targetLanguage === "en";

      set({
        meaningfulSentence: fallback,
        // The offline rules only produce English, whatever was requested.
        sentenceLanguage: "en",
        englishSentence: null,
        isConstructing: false,
        statusBarMessage: isEnglish
          ? "Sentence service unavailable — used offline grammar rules."
          : `Sentence service unavailable — showing English instead of ${getLanguage(state.targetLanguage).name}.`
      });
    }
  },

  // Settings & Performance Actions
  setConfidenceThreshold: (val) => set({ confidenceThreshold: val }),
  setSpeechRate: (val) => set({ speechRate: val }),
  setSelectedVoiceName: (name) => set({ selectedVoiceName: name }),
  setCameraMirrored: (val) => set({ cameraMirrored: val }),
  setActiveMode: (mode) => set({
    activeMode: mode,
    statusBarMessage:
      mode === "numbers"
        ? "Mode switched to Numbers (digits 0-9)."
        : "Mode switched to Words (word signs)."
  }),
  setCameraFps: (fps) => set({ cameraFps: fps }),
  setApiHealthy: (healthy) => set({ apiHealthy: healthy }),
  setInferenceSource: (source) => set({ inferenceSource: source })
}));
