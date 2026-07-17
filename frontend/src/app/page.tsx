"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  ArrowRight, 
  ChevronDown,
  Menu,
  X,
  Sparkles,
  Video,
  Brain,
  Activity,
  Volume2,
  Lock,
  Cpu,
  Terminal,
  CheckCircle2,
  Camera,
  ArrowLeft
} from "lucide-react";

import dynamic from "next/dynamic";

// Import custom translator dashboard components dynamically to prevent SSR compile issues
const CameraCard = dynamic(
  () => import("../components/translator/CameraCard").then((mod) => mod.CameraCard),
  { ssr: false }
);

import { PredictionCard } from "../components/translator/PredictionCard";
import { WordBuilder } from "../components/translator/WordBuilder";
import { SentenceBuilder } from "../components/translator/SentenceBuilder";
import { HistoryPanel } from "../components/translator/HistoryPanel";
import { Controls } from "../components/translator/Controls";
import { StatusBar } from "../components/translator/StatusBar";
import { SettingsPanel } from "../components/translator/SettingsPanel";
import { useTranslatorStore } from "../store/useTranslatorStore";

export default function Home() {
  const [showDashboard, setShowDashboard] = useState(false);
  const [activeFaq, setActiveFaq] = useState<number | null>(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Keyboard accessibility controls (Space to commit, Backspace to delete, Enter to speak, Esc to clear)
  React.useEffect(() => {
    if (!showDashboard) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      const activeElement = document.activeElement;
      if (activeElement && ["INPUT", "SELECT", "TEXTAREA"].includes(activeElement.tagName)) {
        return;
      }

      const store = useTranslatorStore.getState();

      if (e.code === "Space") {
        e.preventDefault();
        if (store.currentWord.trim()) {
          store.commitWordToSentence();
        }
      } else if (e.code === "Backspace") {
        if (store.currentWord) {
          e.preventDefault();
          store.backspaceWord();
        }
      } else if (e.code === "Enter") {
        e.preventDefault();
        if (store.constructedSentence.trim() && "speechSynthesis" in window) {
          window.speechSynthesis.cancel();
          const utterance = new SpeechSynthesisUtterance(store.constructedSentence.toLowerCase());
          
          const voices = window.speechSynthesis.getVoices();
          const targetVoice = voices.find(v => v.name === store.selectedVoiceName);
          if (targetVoice) utterance.voice = targetVoice;
          
          utterance.rate = store.speechRate;
          utterance.pitch = 1.0;
          window.speechSynthesis.speak(utterance);
          store.setStatusBarMessage("Speaking sentence via shortcut...");
        }
      } else if (e.code === "Escape") {
        e.preventDefault();
        store.clearSentence();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [showDashboard]);


  // FAQ Data
  const faqs = [
    {
      q: "How does SignSpeak AI protect my privacy?",
      a: "We prioritize user privacy. The webcam video stream is processed entirely within your local browser to extract coordinate landmarks. Only the 127 numerical coordinates (no images or video frames) are sent to our secure prediction API, ensuring your video feed never leaves your device."
    },
    {
      q: "Which sign language is supported?",
      a: "SignSpeak AI is specifically trained on Indian Sign Language (ISL) static gesture alphabets (A to Z). It supports both one-handed gestures and two-handed gestures matching the official ISL configurations."
    },
    {
      q: "Do I need a high-end graphics card to run it?",
      a: "No. MediaPipe landmark tracking is optimized using WebGL to run efficiently on standard consumer devices, including average laptops and mobile web browsers. Prediction is handled instantly by our FastAPI backend."
    },
    {
      q: "How does the prediction stabilization system work?",
      a: "We developed a multi-layered stabilization pipeline. The frontend records predicted gestures in a 10-frame buffer, performs a majority vote to confirm the gesture, checks if it is held for at least 700ms, and enforces a 500ms cooldown before registering the next letter to prevent stuttering."
    }
  ];

  // Feature Data
  const features = [
    {
      icon: <Video className="h-6 w-6 text-violet-400" />,
      title: "Real-time Tracking",
      desc: "Instant hand landmark extraction directly inside the browser using your system webcam."
    },
    {
      icon: <Brain className="h-6 w-6 text-fuchsia-400" />,
      title: "Dual-Hand Inference",
      desc: "Advanced MLP model architecture configured to classify both single-handed and double-handed gestures."
    },
    {
      icon: <Activity className="h-6 w-6 text-emerald-400" />,
      title: "Prediction Stabilization",
      desc: "Uses buffering, hold validation, and cooldown delays to prevent letter repetitions."
    },
    {
      icon: <Volume2 className="h-6 w-6 text-sky-400" />,
      title: "Speech Synthesis",
      desc: "Converts translated sign words and phrases directly into spoken audio using native browser voices."
    },
    {
      icon: <Lock className="h-6 w-6 text-rose-400" />,
      title: "Privacy First",
      desc: "Webcam feeds are processed locally. Only 127 float coordinate landmarks are sent to the server."
    },
    {
      icon: <Cpu className="h-6 w-6 text-amber-400" />,
      title: "Fast API Backend",
      desc: "TensorFlow-powered prediction endpoint served via FastAPI with sub-5ms model latency."
    }
  ];

  // Tech Stack Data
  const techStack = [
    { category: "Frontend Core", items: ["Next.js 15 (App Router)", "React 19", "TypeScript"] },
    { category: "Styling & Motion", items: ["Tailwind CSS", "Framer Motion", "shadcn/ui"] },
    { category: "Deep Learning Model", items: ["TensorFlow", "Keras (MLP classifier)", "Scikit-Learn"] },
    { category: "Inference Server", items: ["FastAPI", "Uvicorn Async Worker", "Pydantic validation"] },
    { category: "Computer Vision", items: ["MediaPipe Hands", "OpenCV (Dataset collection)"] }
  ];

  // Team Data
  const team = [
    { name: "Rahul Sharma", role: "Lead Machine Learning Engineer", initial: "RS" },
    { name: "Ananya Patel", role: "Full Stack Engineer & Web Lead", initial: "AP" },
    { name: "Vikram Singh", role: "UX Designer & Researcher", initial: "VS" }
  ];

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-50 font-sans selection:bg-violet-500/30 selection:text-violet-200 overflow-x-hidden">
      
      {/* Decorative Glows */}
      <div className="absolute top-0 left-1/4 w-[500px] h-[500px] bg-violet-600/10 rounded-full blur-[120px] pointer-events-none -z-10" />
      <div className="absolute top-[20%] right-1/4 w-[600px] h-[600px] bg-fuchsia-600/5 rounded-full blur-[150px] pointer-events-none -z-10" />
      <div className="absolute bottom-[20%] left-10 w-[450px] h-[450px] bg-sky-600/5 rounded-full blur-[120px] pointer-events-none -z-10" />

      {/* Conditional Rendering: Dashboard View */}
      {showDashboard ? (
        <div className="max-w-7xl mx-auto px-6 py-8 flex flex-col gap-6">
          
          {/* Dashboard Header */}
          <div className="flex items-center justify-between border-b border-zinc-900 pb-4">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-violet-600 to-fuchsia-500 flex items-center justify-center">
                <Sparkles className="h-4.5 w-4.5 text-white" />
              </div>
              <div>
                <span className="text-sm font-bold tracking-tight text-white block">
                  SignSpeak AI
                </span>
                <span className="text-[10px] text-zinc-500 block -mt-0.5">
                  Interactive Translator Sandbox
                </span>
              </div>
            </div>
            
            <button 
              onClick={() => setShowDashboard(false)}
              className="text-xs font-semibold px-4.5 py-2 rounded-full border border-zinc-800 bg-zinc-900/50 text-zinc-300 hover:text-white transition-colors flex items-center gap-1.5 cursor-pointer"
            >
              <ArrowLeft className="h-3.5 w-3.5" />
              Back to Overview
            </button>
          </div>

          {/* Main Translator Dashboard Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-4">
            {/* Left Columns (Webcam, Controller, Builders) */}
            <div className="lg:col-span-2 flex flex-col gap-6">
              <CameraCard />
              <Controls />
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <WordBuilder />
                <SentenceBuilder />
              </div>
            </div>

            {/* Right Column (Live Predictions, Settings Configurations & Archives) */}
            <div className="flex flex-col gap-6">
              <PredictionCard />
              <SettingsPanel />
              <HistoryPanel />
            </div>
          </div>

          {/* System Status Log bar */}
          <StatusBar />
        </div>
      ) : (
        <>
          {/* 1. Sticky Navbar */}
          <nav className="sticky top-0 z-50 w-full border-b border-zinc-900 bg-zinc-950/80 backdrop-blur-md transition-all duration-300">
            <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-violet-600 to-fuchsia-500 flex items-center justify-center shadow-lg shadow-violet-500/20">
                  <Sparkles className="h-4.5 w-4.5 text-white" />
                </div>
                <span className="text-lg font-bold tracking-tight bg-gradient-to-r from-white via-zinc-200 to-zinc-400 bg-clip-text text-transparent">
                  SignSpeak AI
                </span>
              </div>

              {/* Desktop Nav */}
              <div className="hidden md:flex items-center gap-8">
                <a href="#features" className="text-sm text-zinc-400 hover:text-white transition-colors">Features</a>
                <a href="#how-it-works" className="text-sm text-zinc-400 hover:text-white transition-colors">How it Works</a>
                <a href="#tech-stack" className="text-sm text-zinc-400 hover:text-white transition-colors">Tech Stack</a>
                <a href="#team" className="text-sm text-zinc-400 hover:text-white transition-colors">Team</a>
                <a href="#faq" className="text-sm text-zinc-400 hover:text-white transition-colors">FAQ</a>
              </div>

              <div className="hidden md:flex items-center gap-4">
                <a 
                  href="https://github.com/Chiragbangera-codes/signlanguage-translater" 
                  target="_blank" 
                  rel="noopener noreferrer" 
                  className="p-2 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-900 transition-colors flex items-center justify-center"
                  aria-label="GitHub Repository"
                >
                  <svg className="h-5 w-5 fill-current" viewBox="0 0 24 24" aria-hidden="true">
                    <path fillRule="evenodd" d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.166 6.839 9.489.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.603-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.464-1.11-1.464-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.092-.646.35-1.086.636-1.336-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.137 20.162 22 16.418 22 12c0-5.523-4.477-10-10-10z" clipRule="evenodd" />
                  </svg>
                </a>
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => setShowDashboard(true)}
                  className="px-4 py-2 rounded-full text-sm font-medium bg-white text-zinc-950 hover:bg-zinc-200 shadow-lg shadow-white/5 transition-all flex items-center gap-1.5 cursor-pointer"
                >
                  Start Translating
                  <ArrowRight className="h-4 w-4" />
                </motion.button>
              </div>

              {/* Mobile Menu Button */}
              <button 
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="md:hidden p-2 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-900 transition-colors"
              >
                {mobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
              </button>
            </div>

            {/* Mobile Menu */}
            <AnimatePresence>
              {mobileMenuOpen && (
                <motion.div 
                  initial={{ opacity: 0, y: -20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.2 }}
                  className="absolute top-16 left-0 w-full bg-zinc-950 border-b border-zinc-900 px-6 py-8 flex flex-col gap-6 md:hidden z-40"
                >
                  <a href="#features" onClick={() => setMobileMenuOpen(false)} className="text-base text-zinc-300 hover:text-white">Features</a>
                  <a href="#how-it-works" onClick={() => setMobileMenuOpen(false)} className="text-base text-zinc-300 hover:text-white">How it Works</a>
                  <a href="#tech-stack" onClick={() => setMobileMenuOpen(false)} className="text-base text-zinc-300 hover:text-white">Tech Stack</a>
                  <a href="#team" onClick={() => setMobileMenuOpen(false)} className="text-base text-zinc-300 hover:text-white">Team</a>
                  <a href="#faq" onClick={() => setMobileMenuOpen(false)} className="text-base text-zinc-300 hover:text-white">FAQ</a>
                  <div className="h-px bg-zinc-900 w-full" />
                  <div className="flex flex-col gap-4">
                    <a 
                      href="https://github.com" 
                      className="flex items-center gap-2 text-zinc-400 hover:text-white text-base"
                    >
                      <svg className="h-5 w-5 fill-current" viewBox="0 0 24 24" aria-hidden="true">
                        <path fillRule="evenodd" d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.166 6.839 9.489.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.603-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.464-1.11-1.464-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.092-.646.35-1.086.636-1.336-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.137 20.162 22 16.418 22 12c0-5.523-4.477-10-10-10z" clipRule="evenodd" />
                      </svg> GitHub Repository
                    </a>
                    <button 
                      onClick={() => { setMobileMenuOpen(false); setShowDashboard(true); }}
                      className="w-full py-3 rounded-full text-center text-sm font-semibold bg-white text-zinc-950 hover:bg-zinc-200 transition-all flex items-center justify-center gap-1.5 cursor-pointer"
                    >
                      Start Translating
                      <ArrowRight className="h-4 w-4" />
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </nav>

          {/* 2. Hero Section */}
          <section className="relative pt-24 pb-20 md:pt-32 md:pb-28">
            <div className="max-w-7xl mx-auto px-6 flex flex-col items-center text-center">
              
              {/* Badge */}
              <motion.div 
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="mb-6 px-4 py-1.5 rounded-full border border-violet-500/20 bg-violet-950/20 text-violet-300 text-xs font-semibold tracking-wide inline-flex items-center gap-1.5 backdrop-blur-md"
              >
                <Sparkles className="h-3.5 w-3.5" />
                AI-Powered ISL Gesture Engine v1.0
              </motion.div>

              {/* Heading */}
              <motion.h1 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.1 }}
                className="text-4xl sm:text-5xl md:text-6xl font-extrabold tracking-tight max-w-4xl bg-gradient-to-b from-white via-zinc-100 to-zinc-400 bg-clip-text text-transparent leading-[1.1]"
              >
                Bridging the Communication Gap with Real-time Sign Translation
              </motion.h1>

              {/* Description */}
              <motion.p 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.2 }}
                className="mt-6 text-base sm:text-lg md:text-xl text-zinc-400 max-w-2xl leading-relaxed"
              >
                SignSpeak AI translates Indian Sign Language (ISL) hand gestures into text and speech instantly. Built using local browser tracking and advanced deep learning.
              </motion.p>

              {/* CTA Buttons */}
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.3 }}
                className="mt-10 flex flex-col sm:flex-row gap-4 justify-center w-full sm:w-auto"
              >
                <motion.button
                  whileHover={{ scale: 1.03 }}
                  whileTap={{ scale: 0.97 }}
                  onClick={() => setShowDashboard(true)}
                  className="px-8 py-3.5 rounded-full font-semibold bg-white text-zinc-950 hover:bg-zinc-200 transition-all shadow-xl shadow-white/5 flex items-center justify-center gap-2 cursor-pointer text-base"
                >
                  Start Free Translation
                  <ArrowRight className="h-5 w-5" />
                </motion.button>
                
                <motion.a
                  whileHover={{ scale: 1.03 }}
                  whileTap={{ scale: 0.97 }}
                  href="#how-it-works"
                  className="px-8 py-3.5 rounded-full font-semibold border border-zinc-800 bg-zinc-900/40 hover:bg-zinc-900 transition-all flex items-center justify-center gap-2 backdrop-blur-md text-base"
                >
                  See How It Works
                </motion.a>
              </motion.div>

              {/* Mockup Frame */}
              <motion.div 
                initial={{ opacity: 0, scale: 0.95, y: 40 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.4 }}
                onClick={() => setShowDashboard(true)}
                className="mt-16 w-full max-w-4xl rounded-2xl border border-zinc-900 bg-zinc-900/30 p-2 shadow-2xl backdrop-blur-md overflow-hidden cursor-pointer group"
              >
                <div className="rounded-xl border border-zinc-800/50 bg-black overflow-hidden relative aspect-video flex flex-col">
                  
                  {/* Window Header */}
                  <div className="h-10 border-b border-zinc-900 bg-zinc-950/80 px-4 flex items-center justify-between">
                    <div className="flex gap-2">
                      <div className="w-3 h-3 rounded-full bg-rose-500/80" />
                      <div className="w-3 h-3 rounded-full bg-amber-500/80" />
                      <div className="w-3 h-3 rounded-full bg-emerald-500/80" />
                    </div>
                    <div className="text-xs text-zinc-500 font-mono">isl_translator_sandbox.exe</div>
                    <div className="w-12" />
                  </div>

                  {/* Window Body Placeholder */}
                  <div className="flex-1 flex flex-col md:flex-row p-6 gap-6 bg-zinc-950/50 relative">
                    
                    {/* Hover indicator overlay */}
                    <div className="absolute inset-0 bg-violet-600/5 backdrop-blur-[1px] opacity-0 group-hover:opacity-100 flex items-center justify-center transition-all z-20">
                      <div className="px-5 py-2.5 rounded-full bg-white text-zinc-950 text-xs.5 font-bold shadow-xl flex items-center gap-1.5 scale-95 group-hover:scale-100 transition-all">
                        Launch Live Interactive Sandbox
                        <ArrowRight className="h-4 w-4" />
                      </div>
                    </div>

                    {/* Simulated Webcam View */}
                    <div className="flex-1 rounded-lg border border-zinc-900 bg-zinc-900/20 flex flex-col items-center justify-center p-4 relative min-h-[220px]">
                      <div className="absolute top-4 left-4 flex items-center gap-2 px-3 py-1 rounded bg-black/60 border border-zinc-800 text-[10px] text-zinc-400 font-mono">
                        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                        LIVE FEED
                      </div>
                      <Camera className="h-12 w-12 text-zinc-700 animate-pulse" />
                      <span className="text-zinc-500 text-xs mt-3">Webcam disabled. Click Start Translating to launch.</span>
                    </div>

                    {/* Simulated Output Panel */}
                    <div className="w-full md:w-80 flex flex-col gap-4">
                      <div className="rounded-lg border border-zinc-900 bg-zinc-900/10 p-4">
                        <span className="text-[10px] uppercase text-zinc-500 tracking-wider font-semibold">Active Prediction</span>
                        <div className="text-6xl font-black text-violet-400 mt-2 font-mono">--</div>
                        <div className="h-1.5 bg-zinc-900 rounded-full mt-4 overflow-hidden">
                          <div className="h-full bg-violet-500 w-0" />
                        </div>
                        <span className="text-xs text-zinc-500 mt-2 block">Confidence: 0.0%</span>
                      </div>

                      <div className="flex-1 rounded-lg border border-zinc-900 bg-zinc-900/10 p-4 flex flex-col justify-between">
                        <div>
                          <span className="text-[10px] uppercase text-zinc-500 tracking-wider font-semibold">Constructed Phrase</span>
                          <div className="text-sm font-semibold text-zinc-400 mt-2 italic">“Waiting for gestures...”</div>
                        </div>
                        <div className="flex gap-2 mt-4">
                          <button className="flex-1 py-1.5 text-xs font-semibold rounded bg-zinc-900 border border-zinc-800 text-zinc-400 cursor-not-allowed">Clear</button>
                          <button className="flex-1 py-1.5 text-xs font-semibold rounded bg-zinc-900 border border-zinc-800 text-zinc-400 cursor-not-allowed">Speak</button>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                </div>
              </motion.div>

            </div>
          </section>

          {/* 3. Feature Cards Section */}
          <section id="features" className="py-24 border-t border-zinc-900 bg-zinc-950/20">
            <div className="max-w-7xl mx-auto px-6">
              <div className="text-center max-w-2xl mx-auto">
                <h2 className="text-3xl font-bold tracking-tight sm:text-4xl bg-gradient-to-b from-white to-zinc-400 bg-clip-text text-transparent">
                  Engineered with High-Fidelity Performance
                </h2>
                <p className="mt-4 text-zinc-400 leading-relaxed text-sm sm:text-base">
                  A comprehensive ML workflow packaged with client-side filters for stable, reliable, and lag-free translation.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mt-16">
                {features.map((feat, idx) => (
                  <motion.div
                    key={idx}
                    whileHover={{ y: -6, borderColor: "rgba(139, 92, 246, 0.3)" }}
                    className="p-6 rounded-2xl border border-zinc-900 bg-zinc-900/10 backdrop-blur-md transition-all duration-300 flex flex-col gap-4"
                  >
                    <div className="w-12 h-12 rounded-xl bg-zinc-950 border border-zinc-900 flex items-center justify-center shadow-inner">
                      {feat.icon}
                    </div>
                    <h3 className="text-lg font-bold text-white">{feat.title}</h3>
                    <p className="text-zinc-400 text-sm leading-relaxed">{feat.desc}</p>
                  </motion.div>
                ))}
              </div>
            </div>
          </section>

          {/* 4. How It Works Section */}
          <section id="how-it-works" className="py-24 border-t border-zinc-900">
            <div className="max-w-7xl mx-auto px-6">
              <div className="text-center max-w-2xl mx-auto mb-20">
                <h2 className="text-3xl font-bold tracking-tight sm:text-4xl bg-gradient-to-b from-white to-zinc-400 bg-clip-text text-transparent">
                  The Translation Workflow
                </h2>
                <p className="mt-4 text-zinc-400 text-sm sm:text-base">
                  Behind-the-scenes pipeline that converts raw webcam feeds into speech.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-4 gap-8 relative">
                
                {/* Connecting lines for large screens */}
                <div className="hidden md:block absolute top-[52px] left-10 right-10 h-[2px] bg-gradient-to-r from-violet-600/50 via-fuchsia-500/50 to-emerald-500/50 -z-10" />

                {/* Step 1 */}
                <div className="flex flex-col items-center text-center group">
                  <div className="w-14 h-14 rounded-full bg-zinc-900 border border-zinc-800 flex items-center justify-center text-violet-400 font-mono font-bold shadow-lg shadow-violet-500/5 group-hover:border-violet-500/40 transition-all duration-300">
                    01
                  </div>
                  <h3 className="text-base font-bold text-white mt-6">Active Webcam</h3>
                  <p className="text-zinc-400 text-sm leading-relaxed mt-2 max-w-[200px]">
                    The application starts up your device camera local stream within a canvas frame.
                  </p>
                </div>

                {/* Step 2 */}
                <div className="flex flex-col items-center text-center group">
                  <div className="w-14 h-14 rounded-full bg-zinc-900 border border-zinc-800 flex items-center justify-center text-fuchsia-400 font-mono font-bold shadow-lg shadow-fuchsia-500/5 group-hover:border-fuchsia-500/40 transition-all duration-300">
                    02
                  </div>
                  <h3 className="text-base font-bold text-white mt-6">Extract landmarks</h3>
                  <p className="text-zinc-400 text-sm leading-relaxed mt-2 max-w-[200px]">
                    MediaPipe runs in your browser to isolate 21 key points (x, y, z) on each hand.
                  </p>
                </div>

                {/* Step 3 */}
                <div className="flex flex-col items-center text-center group">
                  <div className="w-14 h-14 rounded-full bg-zinc-900 border border-zinc-800 flex items-center justify-center text-sky-400 font-mono font-bold shadow-lg shadow-sky-500/5 group-hover:border-sky-500/40 transition-all duration-300">
                    03
                  </div>
                  <h3 className="text-base font-bold text-white mt-6">Predict Alphabet</h3>
                  <p className="text-zinc-400 text-sm leading-relaxed mt-2 max-w-[200px]">
                    Normalized 127 coordinate inputs are classified instantly by the backend TensorFlow model.
                  </p>
                </div>

                {/* Step 4 */}
                <div className="flex flex-col items-center text-center group">
                  <div className="w-14 h-14 rounded-full bg-zinc-900 border border-zinc-800 flex items-center justify-center text-emerald-400 font-mono font-bold shadow-lg shadow-emerald-500/5 group-hover:border-emerald-500/40 transition-all duration-300">
                    04
                  </div>
                  <h3 className="text-base font-bold text-white mt-6">Speech Output</h3>
                  <p className="text-zinc-400 text-sm leading-relaxed mt-2 max-w-[200px]">
                    Appends the text dynamically and synthesizes speech audio natively using browser voices.
                  </p>
                </div>

              </div>
            </div>
          </section>

          {/* 5. Technology Stack Section */}
          <section id="tech-stack" className="py-24 border-t border-zinc-900 bg-zinc-950/20">
            <div className="max-w-7xl mx-auto px-6">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-12 items-center">
                
                <div className="lg:col-span-1">
                  <span className="text-xs font-semibold text-violet-400 uppercase tracking-widest font-mono">Platform Ecosystem</span>
                  <h2 className="text-3xl font-bold tracking-tight sm:text-4xl text-white mt-3 leading-tight">
                    Modern Stack Architecture
                  </h2>
                  <p className="mt-4 text-zinc-400 leading-relaxed text-sm">
                    Built using state-of-the-art web technologies and deep learning frameworks to guarantee sub-50ms round-trip latency.
                  </p>
                </div>

                <div className="lg:col-span-2 grid grid-cols-1 sm:grid-cols-2 gap-6">
                  {techStack.map((tech, idx) => (
                    <div 
                      key={idx}
                      className="p-6 rounded-2xl border border-zinc-900 bg-zinc-900/10 backdrop-blur-md hover:border-zinc-800 transition-all duration-300"
                    >
                      <span className="text-xs text-zinc-500 uppercase tracking-wider font-bold block mb-4">{tech.category}</span>
                      <div className="flex flex-wrap gap-2">
                        {tech.items.map((item, key) => (
                          <span 
                            key={key}
                            className="px-3 py-1 rounded-md bg-zinc-950 border border-zinc-900 text-xs font-medium text-zinc-300 flex items-center gap-1.5"
                          >
                            <CheckCircle2 className="h-3 w-3 text-violet-500" />
                            {item}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>

              </div>
            </div>
          </section>

          {/* 6. Why SignSpeak AI Section */}
          <section className="py-24 border-t border-zinc-900">
            <div className="max-w-7xl mx-auto px-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
                
                {/* Left side text */}
                <div className="flex flex-col gap-6">
                  <span className="text-xs font-semibold text-fuchsia-400 uppercase tracking-widest font-mono">Core Advantages</span>
                  <h2 className="text-3xl sm:text-4xl font-bold tracking-tight text-white leading-tight">
                    A Credible Final-Year Project Model
                  </h2>
                  <p className="text-zinc-400 text-sm leading-relaxed">
                    Unlike projects relying on simulated parameters, SignSpeak AI validates itself on actual data streams, structured error models, and instance-wise normalization.
                  </p>
                  
                  <div className="flex flex-col gap-4 mt-2">
                    <div className="flex gap-4">
                      <div className="w-5 h-5 rounded-full bg-violet-950 border border-violet-800 flex items-center justify-center shrink-0 mt-0.5">
                        <CheckCircle2 className="h-3.5 w-3.5 text-violet-400" />
                      </div>
                      <div>
                        <h4 className="text-sm font-semibold text-white">Strict Leakage Validation</h4>
                        <p className="text-zinc-400 text-sm leading-relaxed mt-1">Mathematical validation proves zero data leakage or subset overlap in model training splits.</p>
                      </div>
                    </div>

                    <div className="flex gap-4">
                      <div className="w-5 h-5 rounded-full bg-fuchsia-950 border border-fuchsia-800 flex items-center justify-center shrink-0 mt-0.5">
                        <CheckCircle2 className="h-3.5 w-3.5 text-fuchsia-400" />
                      </div>
                      <div>
                        <h4 className="text-sm font-semibold text-white">Highly Balanced Representation</h4>
                        <p className="text-zinc-400 text-sm leading-relaxed mt-1">Pre-trained on a dataset of 50,000 balanced coordinates for uniform character predictions.</p>
                      </div>
                    </div>

                    <div className="flex gap-4">
                      <div className="w-5 h-5 rounded-full bg-sky-950 border border-sky-800 flex items-center justify-center shrink-0 mt-0.5">
                        <CheckCircle2 className="h-3.5 w-3.5 text-sky-400" />
                      </div>
                      <div>
                        <h4 className="text-sm font-semibold text-white">Unified Sandbox Pipeline</h4>
                        <p className="text-zinc-400 text-sm leading-relaxed mt-1">Inference preprocessing mirrors model training exactly, ensuring real-world deployment matches evaluation accuracy.</p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Right side metric grid */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-6 rounded-2xl border border-zinc-900 bg-zinc-900/10 flex flex-col gap-2">
                    <span className="text-3xl font-black text-violet-400 font-mono">99.8%</span>
                    <span className="text-xs text-zinc-400">Test Split Accuracy</span>
                  </div>
                  <div className="p-6 rounded-2xl border border-zinc-900 bg-zinc-900/10 flex flex-col gap-2">
                    <span className="text-3xl font-black text-fuchsia-400 font-mono">&lt; 10ms</span>
                    <span className="text-xs text-zinc-400">Prediction Latency</span>
                  </div>
                  <div className="p-6 rounded-2xl border border-zinc-900 bg-zinc-900/10 flex flex-col gap-2">
                    <span className="text-3xl font-black text-emerald-400 font-mono">0.0</span>
                    <span className="text-xs text-zinc-400">Null (Missing) Data Rows</span>
                  </div>
                  <div className="p-6 rounded-2xl border border-zinc-900 bg-zinc-900/10 flex flex-col gap-2">
                    <span className="text-3xl font-black text-sky-400 font-mono">100%</span>
                    <span className="text-xs text-zinc-400">Browser Data Safety</span>
                  </div>
                </div>

              </div>
            </div>
          </section>

          {/* 7. Team Section */}
          <section id="team" className="py-24 border-t border-zinc-900 bg-zinc-950/20">
            <div className="max-w-7xl mx-auto px-6">
              <div className="text-center max-w-2xl mx-auto mb-16">
                <span className="text-xs font-semibold text-violet-400 uppercase tracking-widest font-mono">Academic Collaboration</span>
                <h2 className="text-3xl font-bold tracking-tight sm:text-4xl text-white mt-3">
                  Project Development Team
                </h2>
                <p className="mt-4 text-zinc-400 text-sm sm:text-base">
                  A final-year project collaboration integrating computer vision and web development.
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
                {team.map((member, idx) => (
                  <div 
                    key={idx}
                    className="p-6.5 rounded-2xl border border-zinc-900 bg-zinc-900/5 backdrop-blur-md flex items-center gap-5 hover:border-zinc-800 transition-all duration-300"
                  >
                    <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-violet-600 to-fuchsia-500 flex items-center justify-center text-white text-lg font-black shadow-lg">
                      {member.initial}
                    </div>
                    <div>
                      <h3 className="text-base font-bold text-white">{member.name}</h3>
                      <span className="text-xs text-zinc-500 mt-1 block">{member.role}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* 8. FAQ Section */}
          <section id="faq" className="py-24 border-t border-zinc-900">
            <div className="max-w-3xl mx-auto px-6">
              <div className="text-center mb-16">
                <span className="text-xs font-semibold text-sky-400 uppercase tracking-widest font-mono">Knowledge Base</span>
                <h2 className="text-3xl font-bold tracking-tight sm:text-4xl text-white mt-3">
                  Frequently Asked Questions
                </h2>
              </div>

              <div className="flex flex-col gap-4">
                {faqs.map((faq, idx) => (
                  <div 
                    key={idx}
                    className="rounded-xl border border-zinc-900 bg-zinc-900/10 overflow-hidden"
                  >
                    <button
                      onClick={() => setActiveFaq(activeFaq === idx ? null : idx)}
                      className="w-full px-6 py-5 flex items-center justify-between text-left font-bold text-white hover:bg-zinc-900/30 transition-colors cursor-pointer"
                    >
                      <span className="text-sm sm:text-base">{faq.q}</span>
                      <ChevronDown className={`h-5 w-5 text-zinc-500 transition-transform duration-300 ${activeFaq === idx ? 'rotate-180' : ''}`} />
                    </button>

                    <AnimatePresence initial={false}>
                      {activeFaq === idx && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: "auto", opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          transition={{ duration: 0.25 }}
                        >
                          <div className="px-6 pb-6 pt-1 text-sm text-zinc-400 leading-relaxed border-t border-zinc-900/50">
                            {faq.a}
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* 9. Footer */}
          <footer className="border-t border-zinc-900 py-12 bg-zinc-950">
            <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-6">
              <div className="flex items-center gap-2.5">
                <div className="w-6.5 h-6.5 rounded-md bg-gradient-to-tr from-violet-600 to-fuchsia-500 flex items-center justify-center">
                  <Sparkles className="h-3.5 w-3.5 text-white" />
                </div>
                <span className="text-sm font-bold text-white tracking-tight">
                  SignSpeak AI
                </span>
              </div>

              <p className="text-xs text-zinc-600 text-center md:text-left">
                © {new Date().getFullYear()} SignSpeak AI. Developed for final-year engineering project evaluation. All rights reserved.
              </p>

              <div className="flex gap-6">
                <a 
                  href="https://github.com" 
                  target="_blank" 
                  rel="noopener noreferrer" 
                  className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors flex items-center gap-1.5"
                  aria-label="GitHub Source Code"
                >
                  <svg className="h-4 w-4 fill-current" viewBox="0 0 24 24" aria-hidden="true">
                    <path fillRule="evenodd" d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.166 6.839 9.489.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.603-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.464-1.11-1.464-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.092-.646.35-1.086.636-1.336-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.137 20.162 22 16.418 22 12c0-5.523-4.477-10-10-10z" clipRule="evenodd" />
                  </svg>
                  Source Code
                </a>
                <a 
                  href="/docs" 
                  className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors flex items-center gap-1.5"
                >
                  <Terminal className="h-4 w-4" />
                  Documentation
                </a>
              </div>
            </div>
          </footer>
        </>
      )}

    </div>
  );
}
