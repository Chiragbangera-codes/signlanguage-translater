import React from "react";

import { Camera, VideoOff, Play, Pause, Sparkles } from "lucide-react";
import { useTranslatorStore } from "../../store/useTranslatorStore";

export const Controls: React.FC = () => {
  const { 
    webcamActive, 
    isTranslating, 
    setWebcamActive, 
    setIsTranslating,
    setStatusBarMessage
  } = useTranslatorStore();

  const handleToggleCamera = () => {
    const nextState = !webcamActive;
    setWebcamActive(nextState);
    if (!nextState && isTranslating) {
      setIsTranslating(false);
    }
  };

  const handleToggleTranslation = () => {
    if (!webcamActive) {
      setStatusBarMessage("Please enable the webcam before starting translation.");
      alert("Please connect the camera first.");
      return;
    }
    setIsTranslating(!isTranslating);
  };


  return (
    <div className="rounded-2xl border border-zinc-900 bg-zinc-950/40 p-6 backdrop-blur-md flex flex-col sm:flex-row items-center gap-6 justify-between w-full shadow-lg">
      
      <div className="flex flex-col gap-1 text-center sm:text-left">
        <h3 className="text-sm.5 font-bold text-white flex items-center justify-center sm:justify-start gap-1.5">
          <Sparkles className="h-4 w-4 text-violet-400" />
          Translation Pipeline Controller
        </h3>
        <p className="text-xs text-zinc-500 max-w-sm">
          Initialize camera capture feeds and start/pause real-time classification engines.
        </p>
      </div>

      <div className="flex flex-col sm:flex-row gap-3 w-full sm:w-auto shrink-0">
        
        {/* Toggle Camera Button */}
        <button
          onClick={handleToggleCamera}
          className={`py-2.5 px-5.5 rounded-full text-xs.5 font-bold flex items-center justify-center gap-2 cursor-pointer transition-all border shadow-sm ${
            webcamActive 
              ? "bg-zinc-900 border-zinc-800 text-zinc-300 hover:text-white"
              : "bg-white text-zinc-950 border-white hover:bg-zinc-200"
          }`}
        >
          {webcamActive ? (
            <>
              <VideoOff className="h-4 w-4" />
              Disconnect Camera
            </>
          ) : (
            <>
              <Camera className="h-4 w-4" />
              Enable Camera
            </>
          )}
        </button>

        {/* Toggle Translation Button */}
        <button
          onClick={handleToggleTranslation}
          className={`py-2.5 px-5.5 rounded-full text-xs.5 font-bold flex items-center justify-center gap-2 cursor-pointer transition-all border shadow-md ${
            isTranslating
              ? "bg-rose-600/10 border-rose-500/20 text-rose-300 hover:bg-rose-600/20"
              : "bg-gradient-to-r from-violet-600 to-fuchsia-500 text-white border-transparent hover:from-violet-500 hover:to-fuchsia-400"
          }`}
        >
          {isTranslating ? (
            <>
              <Pause className="h-4 w-4" />
              Pause Translation
            </>
          ) : (
            <>
              <Play className="h-4 w-4 fill-current" />
              Start Translation
            </>
          )}
        </button>

      </div>

    </div>
  );
};
