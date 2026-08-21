import React from "react";
import { Volume2, Archive, Trash2, AlignLeft, Sparkle, Languages, Loader2 } from "lucide-react";
import { useTranslatorStore } from "../../store/useTranslatorStore";
import { speakSentence } from "../../lib/speech";
import { SUPPORTED_LANGUAGES, getLanguage } from "../../lib/languages";

export const SentenceBuilder: React.FC = () => {
  const {
    constructedSentence,
    meaningfulSentence,
    englishSentence,
    sentenceLanguage,
    isConstructing,
    targetLanguage,
    sentenceStyle,
    clearSentence,
    addSentenceToHistory,
    setStatusBarMessage,
    speechRate,
    selectedVoiceName,
    constructMeaningfulSentence,
    setMeaningfulSentence,
    setTargetLanguage,
    setSentenceStyle
  } = useTranslatorStore();

  // The language the sentence is actually in. This trails the picker: changing
  // the target does not retranslate what is already on screen, so labelling and
  // speaking must follow the sentence, not the dropdown.
  const actual = getLanguage(sentenceLanguage);

  const handleSpeak = () => {
    const target = meaningfulSentence || constructedSentence;
    if (!target.trim()) return;

    if ("speechSynthesis" in window) {
      // Only a constructed sentence is in the target language; raw glosses stay English.
      const bcp47 = meaningfulSentence ? actual.bcp47 : "en-US";
      speakSentence(target, selectedVoiceName, speechRate, bcp47);
      setStatusBarMessage(`Synthesizing speech output...`);
    } else {
      setStatusBarMessage("Speech synthesis is not supported on this browser.");
      alert("Text-to-Speech is not supported by your browser.");
    }
  };

  const handleConstruct = () => {
    if (!constructedSentence.trim()) return;
    void constructMeaningfulSentence();
  };

  const handleClearConstructed = () => {
    setMeaningfulSentence(null);
    setStatusBarMessage("Meaningful sentence cleared.");
  };

  const handleArchive = () => {
    const target = meaningfulSentence || constructedSentence;
    if (!target.trim()) return;
    addSentenceToHistory(target, meaningfulSentence ? sentenceLanguage : "en");
    clearSentence();
  };


  return (
    <div className="rounded-2xl border border-zinc-900 bg-zinc-950/40 p-6 backdrop-blur-md flex flex-col justify-between min-h-[160px] h-full min-w-0 overflow-hidden">

       {/* Header */}
       <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2 pb-3 border-b border-zinc-900/60">
         <div className="flex items-center gap-2 shrink-0">
           <AlignLeft className="h-4.5 w-4.5 text-sky-400" />
           <span className="text-sm font-semibold text-zinc-300">Sentence Constructor</span>
         </div>

         <div className="flex items-center gap-2 min-w-0 flex-1 justify-end">
           {/* Sentence length style */}
           <button
             onClick={() => setSentenceStyle(sentenceStyle === "natural" ? "expanded" : "natural")}
             title={
               sentenceStyle === "expanded"
                 ? "Expanded: builds a fuller, multi-sentence message"
                 : "Natural: keeps the sentence as short as the signs justify"
             }
             className="shrink-0 px-2 py-1 rounded-md border border-zinc-800 bg-zinc-900/40 text-[10px] font-semibold text-zinc-400 hover:text-white hover:border-zinc-700 transition-colors cursor-pointer"
           >
             {sentenceStyle === "expanded" ? "Expanded" : "Natural"}
           </button>

           {/* Output language */}
           <div className="flex items-center gap-1.5 min-w-0 flex-1">
             <Languages className="h-3.5 w-3.5 text-zinc-500 shrink-0" />
             <select
               value={targetLanguage}
               onChange={(e) => setTargetLanguage(e.target.value)}
               title="Language the sentence is generated and spoken in"
               className="w-full min-w-0 max-w-[170px] truncate bg-zinc-900 border border-zinc-800 text-[11px] rounded-md px-2 py-1 text-zinc-300 font-medium focus:outline-none focus:ring-1 focus:ring-sky-500/50 cursor-pointer"
             >
               {SUPPORTED_LANGUAGES.map((lang) => (
                 <option key={lang.code} value={lang.code}>
                   {lang.label}
                 </option>
               ))}
             </select>
           </div>
         </div>
       </div>

       {/* Raw Sign-Word Display */}
       <div className="flex-1 min-h-[60px] max-h-[140px] overflow-y-auto py-4 flex items-start justify-start">
         {constructedSentence ? (
           <p className="w-full text-lg font-bold text-white font-mono leading-relaxed select-text break-words">
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
         <div className="mt-3 mb-2 px-4 py-3 rounded-xl border border-zinc-900 bg-zinc-900/20 max-h-[180px] overflow-y-auto shrink-0">
           <div className="flex items-center justify-between mb-1.5">
             <span className="text-[10px] font-semibold text-emerald-400 uppercase tracking-wider">
               Meaningful Sentence · {actual.name}
             </span>
             <button
               onClick={handleClearConstructed}
               className="text-[9px] text-zinc-600 hover:text-rose-400 transition-colors cursor-pointer"
               title="Clear meaningful sentence"
             >
               Clear
             </button>
           </div>
           <p
             lang={actual.bcp47}
             className="text-xl font-bold text-emerald-300 font-sans leading-relaxed select-text break-words"
           >
             {meaningfulSentence}
           </p>
           {englishSentence && (
             <p className="mt-1.5 text-xs text-zinc-500 italic leading-relaxed select-text break-words">
               {englishSentence}
             </p>
           )}
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
            disabled={!constructedSentence || isConstructing}
            className="flex-1 w-full min-w-0 py-2 px-3 rounded-lg border border-zinc-900 bg-zinc-900/10 text-zinc-400 hover:text-white hover:bg-zinc-900 disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-xs font-semibold flex items-center justify-center gap-1.5 cursor-pointer"
          >
            {isConstructing ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Sparkle className="h-3.5 w-3.5" />
            )}
            {isConstructing ? "Building..." : "Construct Sentence"}
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
