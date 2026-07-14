import React from "react";
import { motion } from "framer-motion";
import { RotateCcw, Delete, CornerDownLeft, Type } from "lucide-react";
import { useTranslatorStore } from "../../store/useTranslatorStore";

export const WordBuilder: React.FC = () => {
  const { currentWord, clearWord, backspaceWord, commitWordToSentence, isTranslating } = useTranslatorStore();

  return (
    <div className="rounded-2xl border border-zinc-900 bg-zinc-950/40 p-6 backdrop-blur-md flex flex-col justify-between min-h-[160px] h-full">
      
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-zinc-900/60">
        <div className="flex items-center gap-2">
          <Type className="h-4.5 w-4.5 text-fuchsia-400" />
          <span className="text-sm font-semibold text-zinc-300">Word Constructor</span>
        </div>
      </div>

      {/* Word Box Display */}
      <div className="flex-1 flex items-center justify-start py-4 overflow-x-auto min-h-[60px]">
        {currentWord ? (
          <div className="flex gap-1.5 items-center">
            {currentWord.split("").map((letter, idx) => (
              <motion.div
                key={idx}
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ type: "spring", stiffness: 400, damping: 25 }}
                className="w-10 h-10 rounded-lg bg-zinc-900 border border-zinc-800 flex items-center justify-center text-lg font-bold text-white font-mono shadow-md"
              >
                {letter}
              </motion.div>
            ))}
            <motion.div 
              animate={{ opacity: [0, 1, 0] }}
              transition={{ repeat: Infinity, duration: 0.8 }}
              className="w-1.5 h-6.5 bg-violet-400 rounded-full ml-1"
            />
          </div>
        ) : (
          <span className="text-sm text-zinc-600 italic select-none">
            {isTranslating ? "Hold ISL gestures to construct letters..." : "Connect camera to start constructing words..."}
          </span>
        )}
      </div>

      {/* Action Controls */}
      <div className="flex gap-3.5 mt-auto pt-3 border-t border-zinc-900/60">
        <button
          onClick={clearWord}
          disabled={!currentWord}
          className="flex-1 py-2 px-3 rounded-lg border border-zinc-900 bg-zinc-900/10 text-zinc-400 hover:text-white hover:bg-zinc-900 disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-xs font-semibold flex items-center justify-center gap-1.5 cursor-pointer"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          Clear
        </button>

        <button
          onClick={backspaceWord}
          disabled={!currentWord}
          className="flex-1 py-2 px-3 rounded-lg border border-zinc-900 bg-zinc-900/10 text-zinc-400 hover:text-white hover:bg-zinc-900 disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-xs font-semibold flex items-center justify-center gap-1.5 cursor-pointer"
        >
          <Delete className="h-3.5 w-3.5" />
          Delete
        </button>

        <button
          onClick={commitWordToSentence}
          disabled={!currentWord}
          className="flex-2 py-2 px-3 rounded-lg bg-violet-600 hover:bg-violet-500 text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-xs font-semibold flex items-center justify-center gap-1.5 cursor-pointer shadow-lg shadow-violet-500/10"
        >
          <CornerDownLeft className="h-3.5 w-3.5" />
          Append Word
        </button>
      </div>

    </div>
  );
};
