import React from "react";
import { Volume2, Archive, Trash2, AlignLeft, Sparkle } from "lucide-react";
import { useTranslatorStore } from "../../store/useTranslatorStore";
import { speakSentence } from "../../lib/speech";

export const SentenceBuilder: React.FC = () => {
  const { 
    constructedSentence, 
    meaningfulSentence,
    clearSentence, 
    addSentenceToHistory,
    setStatusBarMessage,
    speechRate,
    selectedVoiceName,
    constructMeaningfulSentence,
    setMeaningfulSentence
  } = useTranslatorStore();

  const handleSpeak = () => {
    const target = meaningfulSentence || constructedSentence;
    if (!target.trim()) return;
    
    if ("speechSynthesis" in window) {
      speakSentence(target, selectedVoiceName, speechRate);
      setStatusBarMessage(`Synthesizing speech output...`);
    } else {
      setStatusBarMessage("Speech synthesis is not supported on this browser.");
      alert("Text-to-Speech is not supported by your browser.");
    }
  };

  const handleConstruct = () => {
    if (!constructedSentence.trim()) return;
    constructMeaningfulSentence();
  };

  const handleClearConstructed = () => {
    setMeaningfulSentence(null);
    setStatusBarMessage("Meaningful sentence cleared.");
  };

  const handleArchive = () => {
    const target = meaningfulSentence || constructedSentence;
    if (!target.trim()) return;
    addSentenceToHistory(target);
    clearSentence();
    setMeaningfulSentence(null);
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

       {/* Raw Sign-Word Display */}
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

       {/* Meaningful Sentence Output */}
       {meaningfulSentence && (
         <div className="mt-3 mb-2 px-4 py-3 rounded-xl border border-zinc-900 bg-zinc-900/20 overflow-y-auto">
           <div className="flex items-center justify-between mb-1.5">
             <span className="text-[10px] font-semibold text-emerald-400 uppercase tracking-wider">
               Meaningful Sentence
             </span>
             <button
               onClick={handleClearConstructed}
               className="text-[9px] text-zinc-600 hover:text-rose-400 transition-colors cursor-pointer"
               title="Clear meaningful sentence"
             >
               Clear
             </button>
           </div>
           <p className="text-xl font-bold text-emerald-300 font-sans leading-relaxed select-text">
             {meaningfulSentence}
           </p>
         </div>
       )}

       {/* Actions */}
       <div className="flex gap-3.5 mt-auto pt-3 border-t border-zinc-900/60">
         <button
           onClick={clearSentence}
           disabled={!constructedSentence}
            className="flex-1 w-full min-w-0 py-2 px-3 rounded-lg border border-zinc-900 bg-zinc-900/10 text-zinc-400 hover:text-white hover:bg-zinc-900 disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-xs font-semibold flex items-center justify-center gap-1.5 cursor-pointer"
          >
            <Trash2 className="h-3.5 w-3.5" />
            Clear
          </button>

          <button
            onClick={handleConstruct}
            disabled={!constructedSentence}
            className="flex-1 w-full min-w-0 py-2 px-3 rounded-lg border border-zinc-900 bg-zinc-900/10 text-zinc-400 hover:text-white hover:bg-zinc-900 disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-xs font-semibold flex items-center justify-center gap-1.5 cursor-pointer"
          >
            <Sparkle className="h-3.5 w-3.5" />
            Construct Sentence
          </button>

          <button
            onClick={handleArchive}
            disabled={!constructedSentence}
            className="flex-1 w-full min-w-0 py-2 px-3 rounded-lg border border-zinc-900 bg-zinc-900/10 text-zinc-400 hover:text-white hover:bg-zinc-900 disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-xs font-semibold flex items-center justify-center gap-1.5 cursor-pointer"
          >
            <Archive className="h-3.5 w-3.5" />
            Archive
          </button>

          <button
            onClick={handleSpeak}
            disabled={!(meaningfulSentence || constructedSentence)}
            className="flex-1 w-full min-w-0 py-2 px-3 rounded-lg bg-sky-600 hover:bg-sky-500 text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-xs font-semibold flex items-center justify-center gap-1.5 cursor-pointer shadow-md shadow-sky-500/10"
         >
           <Volume2 className="h-3.5 w-3.5" />
           Speak Out Loud
         </button>
       </div>

    </div>
  );
};
