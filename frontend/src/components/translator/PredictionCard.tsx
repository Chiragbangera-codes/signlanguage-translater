import React from "react";
import { Brain, Cpu } from "lucide-react";
import { useTranslatorStore } from "../../store/useTranslatorStore";
import { ConfidenceBar } from "./ConfidenceBar";

export const PredictionCard: React.FC = () => {
  const { 
    currentPrediction, 
    currentConfidence, 
    topPredictions, 
    processingTimeMs,
    isTranslating,
    cameraFps,
    apiHealthy
  } = useTranslatorStore();

  return (
    <div className="rounded-2xl border border-zinc-900 bg-zinc-950/40 p-6 backdrop-blur-md flex flex-col justify-between h-full min-h-[350px]">
      
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-zinc-900/60">
        <div className="flex items-center gap-2">
          <Brain className="h-4.5 w-4.5 text-violet-400" />
          <span className="text-sm font-semibold text-zinc-300">Live Inference HUD</span>
        </div>
        {isTranslating && (
          <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-violet-500/10 border border-violet-500/20 text-violet-300 animate-pulse">
            MODEL ACTIVE
          </span>
        )}
      </div>

      {/* Main Prediction Letter display */}
      <div className="flex-1 flex items-center justify-center py-6">
        <div className="text-center">
          <div className="relative inline-flex items-center justify-center">
            {isTranslating && currentPrediction ? (
              <span className="text-8xl font-black tracking-tighter text-white font-mono bg-gradient-to-b from-white to-zinc-400 bg-clip-text text-transparent">
                {currentPrediction}
              </span>
            ) : (
              <span className="text-7xl font-light text-zinc-700 font-mono italic select-none">
                --
              </span>
            )}
          </div>
          <div className="text-xs text-zinc-500 font-medium mt-2">
            {isTranslating && currentConfidence > 0 
              ? `Main confidence: ${currentConfidence.toFixed(1)}%` 
              : "Awaiting input..."}
          </div>
        </div>
      </div>

      {/* Performance Stats Panel */}
      <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-[10px] font-mono text-zinc-500 border-t border-b border-zinc-900/60 py-3.5 my-3 select-none">
        <div className="flex justify-between border-r border-zinc-900/40 pr-3">
          <span className="text-zinc-600">Camera FPS:</span>
          <span className="text-zinc-300 font-bold">{isTranslating && cameraFps > 0 ? `${cameraFps} fps` : "0 fps"}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-zinc-600">API Status:</span>
          <span className={`font-bold ${apiHealthy ? 'text-emerald-500' : 'text-rose-500'}`}>
            {apiHealthy ? "Healthy" : "Offline"}
          </span>
        </div>
        <div className="flex justify-between border-r border-zinc-900/40 pr-3 pt-1">
          <span className="text-zinc-600">Detection:</span>
          <span className="text-zinc-300 font-bold">
            {!currentPrediction ? "No hands" : topPredictions.length >= 2 ? "Two hands" : "One hand"}
          </span>
        </div>
        <div className="flex justify-between pt-1">
          <span className="text-zinc-600">Model:</span>
          <span className="text-zinc-300 font-bold">{isTranslating ? "Active" : "Idle"}</span>
        </div>
      </div>

      {/* Top 3 Predictions & Stats */}
      <div className="flex flex-col gap-4 mt-auto">
        <div className="flex flex-col gap-3">
          {isTranslating && topPredictions && topPredictions.length > 0 ? (
            topPredictions.map((pred, idx) => (
              <ConfidenceBar 
                key={idx} 
                label={pred.label} 
                confidence={pred.confidence} 
              />
            ))
          ) : (
            <div className="flex flex-col gap-3 opacity-20">
              <ConfidenceBar label="-" confidence={0} />
              <ConfidenceBar label="-" confidence={0} />
            </div>
          )}
        </div>

        {/* Inference Latency Metric */}
        <div className="pt-4 border-t border-zinc-900/60 flex items-center justify-between text-[11px] font-semibold font-mono text-zinc-500">
          <div className="flex items-center gap-1.5">
            <Cpu className="h-3.5 w-3.5 text-zinc-600" />
            <span>MODEL LATENCY</span>
          </div>
          <span>
            {isTranslating && processingTimeMs > 0 
              ? `${processingTimeMs.toFixed(2)} ms` 
              : "0.00 ms"}
          </span>
        </div>
      </div>

    </div>
  );
};
