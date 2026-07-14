import React from "react";
import { History, Play, Trash2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useTranslatorStore } from "../../store/useTranslatorStore";

export const HistoryPanel: React.FC = () => {
  const { 
    history, 
    clearHistory, 
    setStatusBarMessage,
    speechRate,
    selectedVoiceName
  } = useTranslatorStore();

  const handleSpeakHistoryItem = (itemText: string) => {
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(itemText.toLowerCase());
      
      const voices = window.speechSynthesis.getVoices();
      const targetVoice = voices.find(v => v.name === selectedVoiceName);
      if (targetVoice) {
        utterance.voice = targetVoice;
      }
      utterance.rate = speechRate;
      utterance.pitch = 1.0;
      
      window.speechSynthesis.speak(utterance);
      setStatusBarMessage(`Replaying archived speech: "${itemText}"`);
    }
  };


  return (
    <div className="rounded-2xl border border-zinc-900 bg-zinc-950/40 p-6 backdrop-blur-md flex flex-col justify-between h-full min-h-[300px]">
      
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-zinc-900/60 mb-4">
        <div className="flex items-center gap-2">
          <History className="h-4.5 w-4.5 text-emerald-400" />
          <span className="text-sm font-semibold text-zinc-300">Translation Archives</span>
        </div>
        {history.length > 0 && (
          <button 
            onClick={clearHistory}
            className="p-1.5 rounded-lg text-zinc-500 hover:text-rose-400 hover:bg-rose-500/10 transition-colors cursor-pointer"
            title="Clear Archives"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        )}
      </div>

      {/* History List */}
      <div className="flex-1 overflow-y-auto max-h-[320px] flex flex-col gap-2.5 pr-1">
        <AnimatePresence initial={false}>
          {history.length > 0 ? (
            history.map((sentence, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
                transition={{ duration: 0.2 }}
                className="p-3.5 rounded-xl border border-zinc-900 bg-zinc-900/20 hover:border-zinc-800/80 flex items-center justify-between gap-3 group transition-all"
              >
                <span className="text-xs.5 font-bold font-mono text-zinc-300 leading-normal truncate flex-1 select-text">
                  {sentence}
                </span>
                
                <button
                  onClick={() => handleSpeakHistoryItem(sentence)}
                  className="w-7 h-7 rounded-lg border border-zinc-800 bg-zinc-950 flex items-center justify-center text-zinc-500 hover:text-emerald-400 hover:bg-emerald-950/20 transition-all opacity-100 md:opacity-40 group-hover:opacity-100 cursor-pointer shadow-md shrink-0"
                  title="Replay Audio"
                >
                  <Play className="h-3 w-3 fill-current" />
                </button>
              </motion.div>
            ))
          ) : (
            <div className="flex-1 flex items-center justify-center min-h-[160px]">
              <span className="text-xs.5 text-zinc-600 italic select-none">
                No archived translations.
              </span>
            </div>
          )}
        </AnimatePresence>
      </div>

    </div>
  );
};
