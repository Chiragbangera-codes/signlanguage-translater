import React from "react";
import { motion } from "framer-motion";

interface ConfidenceBarProps {
  label: string;
  confidence: number;
}

export const ConfidenceBar: React.FC<ConfidenceBarProps> = ({ label, confidence }) => {
  // Select color depending on confidence value
  const getColor = (conf: number) => {
    if (conf >= 90) return "bg-gradient-to-r from-violet-500 to-fuchsia-500";
    if (conf >= 50) return "bg-gradient-to-r from-sky-500 to-indigo-500";
    return "bg-gradient-to-r from-zinc-700 to-zinc-500";
  };

  return (
    <div className="flex flex-col gap-1.5 w-full">
      <div className="flex items-center justify-between text-xs font-semibold font-mono">
        <span className="text-zinc-300">Letter {label}</span>
        <span className="text-zinc-400">{confidence.toFixed(1)}%</span>
      </div>
      <div className="h-2 w-full rounded-full bg-zinc-900 border border-zinc-800/40 overflow-hidden relative">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${confidence}%` }}
          transition={{ duration: 0.4, ease: "easeOut" }}
          className={`h-full rounded-full ${getColor(confidence)}`}
        />
      </div>
    </div>
  );
};
