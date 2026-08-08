import { create } from "zustand";
import { constructMeaningfulSentence } from "../lib/grammar";

export interface PredictionItem {
  label: string;
  confidence: number;
}

export type TranslatorMode = "numbers" | "words";

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
  history: string[];

  // Settings State
  confidenceThreshold: number; // 0-100%
  speechRate: number; // 0.5-2.0
  selectedVoiceName: string | null;
  cameraMirrored: boolean;
  activeMode: TranslatorMode;

  // Performance stats state
  cameraFps: number;
  apiHealthy: boolean;

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
  addSentenceToHistory: (sentence: string) => void;
  clearHistory: () => void;

  // Grammar correction
  setMeaningfulSentence: (sentence: string | null) => void;
  constructMeaningfulSentence: () => void;

  // Settings & Performance Actions
  setConfidenceThreshold: (val: number) => void;
  setSpeechRate: (val: number) => void;
  setSelectedVoiceName: (name: string | null) => void;
  setCameraMirrored: (val: boolean) => void;
  setActiveMode: (mode: TranslatorMode) => void;
  setCameraFps: (fps: number) => void;
  setApiHealthy: (healthy: boolean) => void;
}

export const useTranslatorStore = create<TranslatorState>((set) => ({
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
  history: [
    "HELLO WORLD",
    "WELCOME TO SIGNSPEAK AI",
    "INDIAN SIGN LANGUAGE IS AMAZING"
  ],

  // Default settings
  confidenceThreshold: 80,
  speechRate: 1.0,
  selectedVoiceName: null,
  cameraMirrored: true,
  activeMode: "numbers",

  // Performance stats
  cameraFps: 0,
  apiHealthy: true,

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
  
  addSentenceToHistory: (sentence) => set((state) => ({
    history: [sentence, ...state.history],
    statusBarMessage: "Sentence committed to history."
  })),
  
  clearHistory: () => set({ 
    history: [],
    statusBarMessage: "History cleared."
  }),

  setMeaningfulSentence: (sentence) => set({
    meaningfulSentence: sentence,
    statusBarMessage: sentence
      ? "Meaningful sentence constructed."
      : "Meaningful sentence cleared."
  }),

  constructMeaningfulSentence: () => set((state) => {
    if (!state.constructedSentence.trim()) {
      return {
        meaningfulSentence: null,
        statusBarMessage: "No words to construct a sentence from."
      };
    }
    const result = constructMeaningfulSentence(state.constructedSentence);
    return {
      meaningfulSentence: result,
      statusBarMessage: "Sign words converted to meaningful sentence."
    };
  }),

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
        : "Mode switched to Words (alphabet). The words model is not trained yet."
  }),
  setCameraFps: (fps) => set({ cameraFps: fps }),
  setApiHealthy: (healthy) => set({ apiHealthy: healthy })
}));
