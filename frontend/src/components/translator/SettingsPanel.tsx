import React, { useEffect, useState } from "react";
import { Settings, Sliders, Volume2, ToggleLeft, ToggleRight } from "lucide-react";
import { useTranslatorStore } from "../../store/useTranslatorStore";

export const SettingsPanel: React.FC = () => {
  const {
    confidenceThreshold,
    speechRate,
    selectedVoiceName,
    cameraMirrored,
    setConfidenceThreshold,
    setSpeechRate,
    setSelectedVoiceName,
    setCameraMirrored
  } = useTranslatorStore();

  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);

  useEffect(() => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;

    const loadVoices = () => {
      const allVoices = window.speechSynthesis.getVoices();
      // Filter out duplicates and only show common languages (e.g. English, regional)
      const uniqueVoices = allVoices.filter(
        (voice, index, self) => self.findIndex(v => v.name === voice.name) === index
      );
      setVoices(uniqueVoices);

      // Select default voice if none active
      if (!selectedVoiceName && uniqueVoices.length > 0) {
        // Prefer English
        const englishVoice = uniqueVoices.find(v => v.lang.startsWith("en-"));
        if (englishVoice) {
          setSelectedVoiceName(englishVoice.name);
        } else {
          setSelectedVoiceName(uniqueVoices[0].name);
        }
      }
    };

    loadVoices();
    window.speechSynthesis.onvoiceschanged = loadVoices;
  }, [selectedVoiceName, setSelectedVoiceName]);

  return (
    <div className="rounded-2xl border border-zinc-900 bg-zinc-950/40 p-6 backdrop-blur-md flex flex-col justify-between h-full min-h-[300px]">
      
      {/* Header */}
      <div className="flex items-center gap-2 pb-4 border-b border-zinc-900/60 mb-4">
        <Settings className="h-4.5 w-4.5 text-violet-400" />
        <span className="text-sm font-semibold text-zinc-300">System Configurations</span>
      </div>

      {/* Settings Grid */}
      <div className="flex-1 flex flex-col gap-5 justify-start">
        
        {/* Mirror Camera Switch */}
        <div className="flex items-center justify-between">
          <div className="flex flex-col gap-0.5">
            <span className="text-xs font-semibold text-zinc-300">Mirror Webcam Feed</span>
            <span className="text-[10px] text-zinc-500">Horizontal flip for standard mirror visual.</span>
          </div>
          <button
            onClick={() => setCameraMirrored(!cameraMirrored)}
            className="text-zinc-400 hover:text-white transition-colors cursor-pointer"
          >
            {cameraMirrored ? (
              <ToggleRight className="h-9 w-9 text-violet-500" />
            ) : (
              <ToggleLeft className="h-9 w-9 text-zinc-600" />
            )}
          </button>
        </div>

        {/* Confidence Threshold Slider */}
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-zinc-300 flex items-center gap-1.5">
              <Sliders className="h-3.5 w-3.5 text-zinc-500" />
              Inference Gate Threshold
            </span>
            <span className="font-mono text-violet-400 font-bold">{confidenceThreshold}%</span>
          </div>
          <input
            type="range"
            min="50"
            max="95"
            step="5"
            value={confidenceThreshold}
            onChange={(e) => setConfidenceThreshold(parseInt(e.target.value))}
            className="w-full h-1 bg-zinc-900 rounded-lg appearance-none cursor-pointer accent-violet-500 border border-zinc-800"
          />
          <span className="text-[9px] text-zinc-500">Only accepts predictions higher than this confidence level.</span>
        </div>

        {/* Speech Rate Slider */}
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-zinc-300 flex items-center gap-1.5">
              <Volume2 className="h-3.5 w-3.5 text-zinc-500" />
              TTS Speech Playback Rate
            </span>
            <span className="font-mono text-sky-400 font-bold">{speechRate.toFixed(1)}x</span>
          </div>
          <input
            type="range"
            min="0.5"
            max="1.5"
            step="0.1"
            value={speechRate}
            onChange={(e) => setSpeechRate(parseFloat(e.target.value))}
            className="w-full h-1 bg-zinc-900 rounded-lg appearance-none cursor-pointer accent-sky-500 border border-zinc-800"
          />
        </div>

        {/* Voice Selection Dropdown */}
        {voices.length > 0 && (
          <div className="flex flex-col gap-2.5">
            <span className="text-xs font-semibold text-zinc-300">Synthesizer Voice Voice</span>
            <select
              value={selectedVoiceName || ""}
              onChange={(e) => setSelectedVoiceName(e.target.value || null)}
              className="w-full bg-zinc-900 border border-zinc-800 text-xs rounded-lg px-3 py-2 text-zinc-300 font-medium focus:outline-none focus:ring-1 focus:ring-violet-500/50 cursor-pointer"
            >
              {voices.map((voice, idx) => (
                <option key={idx} value={voice.name}>
                  {voice.name} ({voice.lang})
                </option>
              ))}
            </select>
          </div>
        )}

      </div>

    </div>
  );
};
