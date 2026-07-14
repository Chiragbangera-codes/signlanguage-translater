import { create } from "zustand";

export interface PredictionItem {
  label: string;
  confidence: number;
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
  history: string[];

  // Settings State
  confidenceThreshold: number; // 0-100%
  speechRate: number; // 0.5-2.0
  selectedVoiceName: string | null;
  cameraMirrored: boolean;

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
  backspaceWord: () => void;
  clearWord: () => void;
  commitWordToSentence: () => void;
  clearSentence: () => void;
  addSentenceToHistory: (sentence: string) => void;
  clearHistory: () => void;

  // Settings & Performance Actions
  setConfidenceThreshold: (val: number) => void;
  setSpeechRate: (val: number) => void;
  setSelectedVoiceName: (name: string | null) => void;
  setCameraMirrored: (val: boolean) => void;
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
  
  backspaceWord: () => set((state) => ({
    currentWord: state.currentWord.slice(0, -1),
    statusBarMessage: "Deleted last character."
  })),
  
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
  
  addSentenceToHistory: (sentence) => set((state) => ({
    history: [sentence, ...state.history],
    statusBarMessage: "Sentence committed to history."
  })),
  
  clearHistory: () => set({ 
    history: [],
    statusBarMessage: "History cleared."
  }),

  // Settings & Performance Actions
  setConfidenceThreshold: (val) => set({ confidenceThreshold: val }),
  setSpeechRate: (val) => set({ speechRate: val }),
  setSelectedVoiceName: (name) => set({ selectedVoiceName: name }),
  setCameraMirrored: (val) => set({ cameraMirrored: val }),
  setCameraFps: (fps) => set({ cameraFps: fps }),
  setApiHealthy: (healthy) => set({ apiHealthy: healthy })
}));
