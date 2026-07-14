import React from "react";
import { Terminal, ShieldCheck, Cpu } from "lucide-react";
import { useTranslatorStore } from "../../store/useTranslatorStore";

export const StatusBar: React.FC = () => {
  const { statusBarMessage, modelLoaded } = useTranslatorStore();

  return (
    <div className="rounded-xl border border-zinc-905 bg-zinc-950/80 px-4 py-2.5 backdrop-blur-md flex flex-col sm:flex-row items-center justify-between gap-3 text-[11px] font-medium font-mono text-zinc-500 shadow-inner">
      
      {/* Active system log message */}
      <div className="flex items-center gap-2">
        <Terminal className="h-4 w-4 text-violet-400 shrink-0 animate-pulse" />
        <span className="text-zinc-400 text-center sm:text-left truncate max-w-lg">
          LOG: {statusBarMessage}
        </span>
      </div>

      {/* Network and Framework state parameters */}
      <div className="flex items-center gap-4 shrink-0">
        <div className="flex items-center gap-1.5">
          <Cpu className="h-3.5 w-3.5 text-zinc-600" />
          <span className="text-zinc-400">TF-API: v1.0.0</span>
        </div>
        
        <div className="flex items-center gap-1.5">
          <div className={`w-1.5 h-1.5 rounded-full ${modelLoaded ? 'bg-emerald-500' : 'bg-rose-500'}`} />
          <span className="text-zinc-400">MODEL: LOADED</span>
        </div>

        <div className="flex items-center gap-1.5">
          <ShieldCheck className="h-3.5 w-3.5 text-emerald-500" />
          <span className="text-emerald-500">SECURE LOCAL PIPELINE</span>
        </div>
      </div>

    </div>
  );
};
