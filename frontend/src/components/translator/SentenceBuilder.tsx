import React from "react";
import { Volume2, Archive, Trash2, AlignLeft } from "lucide-react";
import { useTranslatorStore } from "../../store/useTranslatorStore";

export const SentenceBuilder: React.FC = () => {
  const { 
    constructedSentence, 
    clearSentence, 
    addSentenceToHistory,
    setStatusBarMessage,
    speechRate,
    selectedVoiceName
  } = useTranslatorStore();

  const handleSpeak = () => {
    if (!constructedSentence.trim()) return;
    
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel(); // Cancel any ongoing speech
      const utterance = new SpeechSynthesisUtterance(constructedSentence.toLowerCase());
      
      // Find and assign the selected speech voice
      const voices = window.speechSynthesis.getVoices();
      const targetVoice = voices.find(v => v.name === selectedVoiceName);
      if (targetVoice) {
        utterance.voice = targetVoice;
      }
      
      utterance.rate = speechRate;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
      setStatusBarMessage("Synthesizing speech output...");
    } else {
      setStatusBarMessage("Speech synthesis is not supported on this browser.");
      alert("Text-to-Speech is not supported by your browser.");
    }
  };




  const handleArchive = () => {
    if (!constructedSentence.trim()) return;
    addSentenceToHistory(constructedSentence);
    clearSentence();
  };


  return (
    <div className="rounded-2xl border border-zinc-900 bg-zinc-950/40 p-6 backdrop-blur-md flex flex-col justify-between min-h-[160px] h-full">
      
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-zinc-900/60">
        <div className="flex items-center gap-2">
          <AlignLeft className="h-4.5 w-4.5 text-sky-400" />
          <span className="text-sm font-semibold text-zinc-300">Sentence Constructor</span>
        </div>
      </div>

      {/* Sentence Display Area */}
      <div className="flex-1 flex items-center justify-start py-4 overflow-y-auto min-h-[60px]">
        {constructedSentence ? (
          <p className="text-lg font-bold text-white font-mono leading-relaxed select-text">
            {constructedSentence}
          </p>
        ) : (
          <span className="text-sm text-zinc-600 italic select-none">
            Awaiting word commitments to form a full sentence...
          </span>
        )}
      </div>

      {/* Actions */}
      <div className="flex gap-3.5 mt-auto pt-3 border-t border-zinc-900/60">
        <button
          onClick={clearSentence}
          disabled={!constructedSentence}
          className="flex-1 py-2 px-3 rounded-lg border border-zinc-900 bg-zinc-900/10 text-zinc-400 hover:text-white hover:bg-zinc-900 disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-xs font-semibold flex items-center justify-center gap-1.5 cursor-pointer"
        >
          <Trash2 className="h-3.5 w-3.5" />
          Clear
        </button>

        <button
          onClick={handleArchive}
          disabled={!constructedSentence}
          className="flex-1 py-2 px-3 rounded-lg border border-zinc-900 bg-zinc-900/10 text-zinc-400 hover:text-white hover:bg-zinc-900 disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-xs font-semibold flex items-center justify-center gap-1.5 cursor-pointer"
        >
          <Archive className="h-3.5 w-3.5" />
          Archive
        </button>

        <button
          onClick={handleSpeak}
          disabled={!constructedSentence}
          className="flex-2 py-2 px-3 rounded-lg bg-sky-600 hover:bg-sky-500 text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-xs font-semibold flex items-center justify-center gap-1.5 cursor-pointer shadow-lg shadow-sky-500/10"
        >
          <Volume2 className="h-3.5 w-3.5" />
          Speak Out Loud
        </button>
      </div>

    </div>
  );
};
