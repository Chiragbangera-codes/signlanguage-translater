/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useEffect, useRef, useState, useCallback } from "react";
import { VideoOff, WifiOff } from "lucide-react";
import { useTranslatorStore } from "../../store/useTranslatorStore";

// Draw hand joints & connection lines in canvas context (declared outside to remain pure)
const drawHandSkeleton = (ctx: CanvasRenderingContext2D, landmarks: any[]) => {
  const width = ctx.canvas.width;
  const height = ctx.canvas.height;

  const connections = [
    [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8], // Index
    [5, 9], [9, 10], [10, 11], [11, 12], // Middle
    [9, 13], [13, 14], [14, 15], [15, 16], // Ring
    [13, 17], [0, 17], [17, 18], [18, 19], [19, 20] // Pinky
  ];

  // Connections drawing configuration
  ctx.strokeStyle = "rgba(168, 85, 247, 0.7)"; // violet-500
  ctx.lineWidth = 4;
  ctx.lineCap = "round";

  for (const [start, end] of connections) {
    const pt1 = landmarks[start];
    const pt2 = landmarks[end];
    if (pt1 && pt2) {
      ctx.beginPath();
      ctx.moveTo(pt1.x * width, pt1.y * height);
      ctx.lineTo(pt2.x * width, pt2.y * height);
      ctx.stroke();
    }
  }

  // Joint keys drawing configuration
  for (let i = 0; i < landmarks.length; i++) {
    const pt = landmarks[i];
    ctx.beginPath();
    ctx.arc(pt.x * width, pt.y * height, 5, 0, 2 * Math.PI);
    
    // Highlight fingertip nodes
    if ([4, 8, 12, 16, 20].includes(i)) {
      ctx.fillStyle = "#f43f5e"; // rose-500
    } else {
      ctx.fillStyle = "#ffffff";
    }
    ctx.fill();
  }
};

export const CameraCard: React.FC = () => {
  const { 
    webcamActive, 
    isTranslating, 
    setWebcamActive, 
    setPrediction, 
    appendLetterToWord, 
    setStatusBarMessage,
    setCameraFps,
    setApiHealthy
  } = useTranslatorStore();

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  
  const handsRef = useRef<any>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  
  // Pipeline Performance and Stabilization Refs
  const lastPredictionTime = useRef<number>(0);
  const predictionBuffer = useRef<string[]>([]);
  
  // Hold detection and cooldown filters
  const lastAppendedLetter = useRef<string | null>(null);
  const lastAppendTime = useRef<number>(0);
  const currentMajorityLetter = useRef<string | null>(null);
  const majorityLetterStartTime = useRef<number>(0);

  // FPS Counter tracking refs
  const fpsLastTime = useRef<number>(0);
  const fpsFrames = useRef<number>(0);

  const [cameraError, setCameraError] = useState<string | null>(null);

  // Main frame processing loop ref to resolve accessed-before-declaration warnings
  const processVideoFrameRef = useRef<() => void>(() => {});

  // Update loop callback reference whenever FPS settings updates
  useEffect(() => {
    processVideoFrameRef.current = async () => {
      if (!streamRef.current || !videoRef.current || !handsRef.current) return;

      if (videoRef.current.readyState === videoRef.current.HAVE_ENOUGH_DATA) {
        try {
          await handsRef.current.send({ image: videoRef.current });
          
          // Track frame rendering cycles for FPS calculation
          fpsFrames.current++;
          const now = performance.now();
          if (now - fpsLastTime.current >= 1000) {
            const calculatedFps = Math.round((fpsFrames.current * 1000) / (now - fpsLastTime.current));
            setCameraFps(calculatedFps);
            fpsFrames.current = 0;
            fpsLastTime.current = now;
          }
        } catch (err) {
          console.error("Frame processing exception:", err);
        }
      }
      animationFrameRef.current = requestAnimationFrame(processVideoFrameRef.current);
    };
  }, [setCameraFps]);

  // Stabilization: Buffer, Majority Vote, Hold Trigger (700ms), and Cooldown Lock (500ms)
  const runStabilizationPipeline = useCallback((letter: string, confidence: number) => {
    // Read dynamic settings from store
    const { confidenceThreshold } = useTranslatorStore.getState();

    // 1. Enforce confidence threshold (ignore predictions below configured setting)
    if (confidence < confidenceThreshold) return;

    // 2. Add to rolling window buffer (size = 10)
    predictionBuffer.current.push(letter);
    if (predictionBuffer.current.length > 10) {
      predictionBuffer.current.shift();
    }

    // 3. Count frequencies in the buffer (Majority Vote)
    const counts: Record<string, number> = {};
    let majorityLetter = "";
    let majorityCount = 0;
    
    for (const l of predictionBuffer.current) {
      counts[l] = (counts[l] || 0) + 1;
      if (counts[l] > majorityCount) {
        majorityCount = counts[l];
        majorityLetter = l;
      }
    }

    // 4. Threshold: 60% majority (6 out of 10)
    if (majorityCount >= 6) {
      const now = Date.now();
      
      if (majorityLetter !== currentMajorityLetter.current) {
        // New majority letter detected: start hold timer
        currentMajorityLetter.current = majorityLetter;
        majorityLetterStartTime.current = now;
      } else {
        // Sustained majority: check hold duration (700ms)
        const holdDuration = now - majorityLetterStartTime.current;
        if (holdDuration >= 700) {
          // Check cooldown lock (500ms) to prevent stuttering
          const cooldownElapsed = now - lastAppendTime.current;
          if (cooldownElapsed >= 500) {
            lastAppendTime.current = now;
            lastAppendedLetter.current = majorityLetter;
            
            // Append prediction to the constructed word
            appendLetterToWord(majorityLetter);
            setStatusBarMessage(`Added letter "${majorityLetter}" (Held for ${holdDuration}ms)`);
          }
        }
      }
    } else {
      // Clear majority tracking
      currentMajorityLetter.current = null;
      majorityLetterStartTime.current = 0;
    }
  }, [appendLetterToWord, setStatusBarMessage]);

  // Maps MediaPipe landmarks to 127 float format and makes API call
  const executeInference = useCallback(async (results: any) => {
    const landmarksList = results.multiHandLandmarks || [];
    const handednessList = results.multiHandedness || [];

    // Initialize padded landmarks (-1.0)
    let leftHandCoords = Array(63).fill(-1.0);
    let rightHandCoords = Array(63).fill(-1.0);

    for (let i = 0; i < landmarksList.length; i++) {
      const landmarks = landmarksList[i];
      const handedness = handednessList[i];

      const flatCoords: number[] = [];
      for (const lm of landmarks) {
        flatCoords.push(lm.x, lm.y, lm.z);
      }

      // Fill coordinate bins matching anatomical side labels
      if (handedness.label === "Left") {
        leftHandCoords = flatCoords;
      } else if (handedness.label === "Right") {
        rightHandCoords = flatCoords;
      }
    }

    const usesTwoHands = landmarksList.length >= 2 ? 1.0 : 0.0;

    try {
      const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const response = await fetch(`${apiBase}/api/v1/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          left_hand: leftHandCoords,
          right_hand: rightHandCoords,
          uses_two_hands: usesTwoHands
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        setStatusBarMessage(`API Error: ${errorData.message || "Failed to process coordinates."}`);
        setApiHealthy(false);
        return;
      }

      const data = await response.json();
      setApiHealthy(true);
      
      // Update store state with prediction
      setPrediction(
        data.prediction, 
        data.confidence, 
        data.top_predictions, 
        data.processing_time_ms
      );

      // Run stabilization pipeline
      runStabilizationPipeline(data.prediction, data.confidence);

    } catch (err) {
      console.error("FastAPI backend connection error:", err);
      setStatusBarMessage("Backend unavailable. Verify FastAPI is running on port 8000.");
      setPrediction(null, 0, [], 0);
      setApiHealthy(false);
    }
  }, [setPrediction, setStatusBarMessage, setApiHealthy, runStabilizationPipeline]);

  // MediaPipe Results Processing
  const onHandResults = useCallback(async (results: any) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Adjust canvas draw dimensions to match video stream resolution dynamically
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const width = canvas.width;
    const height = canvas.height;

    // Read mirroring preference and translating status from Zustand state dynamically
    const { cameraMirrored, isTranslating } = useTranslatorStore.getState();

    // 1. Draw camera feed with horizontal mirroring option
    ctx.save();
    if (cameraMirrored) {
      ctx.translate(width, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(video, 0, 0, width, height);

    // 2. Draw landmarks if detected
    if (results.multiHandLandmarks) {
      for (const landmarks of results.multiHandLandmarks) {
        drawHandSkeleton(ctx, landmarks);
      }
    }
    ctx.restore();

    // 3. Inference execution logic
    if (isTranslating) {
      const landmarksList = results.multiHandLandmarks || [];
      
      // Handle "No hand detected" scenario
      if (landmarksList.length === 0) {
        setPrediction(null, 0, [], 0);
        setStatusBarMessage("No hands in frame. Position your hands to translate.");
        // Clear majority state
        currentMajorityLetter.current = null;
        majorityLetterStartTime.current = 0;
        return;
      }

      // Throttle predictions (Max 5 requests per second / 200ms) to ensure smooth 30 FPS rendering
      const now = Date.now();
      if (now - lastPredictionTime.current >= 200) {
        lastPredictionTime.current = now;
        await executeInference(results);
      }
    }
  }, [setPrediction, setStatusBarMessage, executeInference]);

  const stopCameraAndLoop = useCallback(() => {
    // 1. Stop animation loop
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    
    // 2. Stop camera tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    // 3. Clear Canvas drawing
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    // 4. Reset prediction states
    setPrediction(null, 0, [], 0);
    predictionBuffer.current = [];
    currentMajorityLetter.current = null;
    majorityLetterStartTime.current = 0;
    setCameraFps(0);
  }, [setPrediction, setCameraFps]);

  const startCamera = useCallback(async () => {
    setCameraError(null);
    try {
      console.log("[CameraCard] Requesting webcam access...");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: { ideal: 640 }, 
          height: { ideal: 480 },
          facingMode: "user"
        },
        audio: false
      });
      
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current) {
            videoRef.current.play();
            // Reset FPS counters
            fpsLastTime.current = performance.now();
            fpsFrames.current = 0;
            // Start the frame processor loop
            animationFrameRef.current = requestAnimationFrame(processVideoFrameRef.current);
          }
        };
      }
      setStatusBarMessage("Webcam connected. Feed active.");
    } catch (err: any) {
      console.error("[CameraCard] Webcam access denied or unavailable:", err);
      setWebcamActive(false);
      setCameraError("Permission Denied");
      setStatusBarMessage("Webcam permission denied or device busy.");
      alert("Camera access denied. Please grant permission in browser settings.");
    }
  }, [setWebcamActive, setStatusBarMessage]);

  // Initialize MediaPipe Hands on client mount
  useEffect(() => {
    let active = true;

    const initHands = async () => {
      try {
        setStatusBarMessage("Loading MediaPipe tracking models...");
        
        // Dynamically load the @mediapipe/hands package on the client side only via require
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        const mpHands = require("@mediapipe/hands");
        
        if (!active) return;

        const HandsConstructor = mpHands.Hands;
        if (!HandsConstructor) {
          throw new Error("Hands export not found in @mediapipe/hands package.");
        }

        const hands = new HandsConstructor({
          locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });

        hands.setOptions({
          maxNumHands: 2,
          modelComplexity: 1,
          minDetectionConfidence: 0.6,
          minTrackingConfidence: 0.6
        });

        hands.onResults(onHandResults);
        handsRef.current = hands;
        console.log("[CameraCard] MediaPipe Hands package initialized dynamically.");
        setStatusBarMessage("System Ready. Connect webcam to start translation.");
      } catch (e) {
        console.error("Failed to initialize MediaPipe Hands package:", e);
        setStatusBarMessage("Error: Failed to load MediaPipe tracking models.");
      }
    };

    initHands();

    return () => {
      active = false;
      stopCameraAndLoop();
    };
  }, [isTranslating, stopCameraAndLoop, onHandResults, setStatusBarMessage]);

  // Handle webcam activation / deactivation
  useEffect(() => {
    if (webcamActive) {
      // Defer execution using microtasks to avoid synchronous cascading setState errors inside trigger effects
      Promise.resolve().then(() => {
        startCamera();
      });
    } else {
      stopCameraAndLoop();
    }
  }, [webcamActive, startCamera, stopCameraAndLoop]);

  return (
    <div className="rounded-2xl border border-zinc-900 bg-zinc-950/40 p-1 backdrop-blur-md overflow-hidden aspect-video relative flex flex-col justify-between shadow-2xl min-h-[300px]">
      
      <div className="flex-1 rounded-xl bg-black/60 relative flex flex-col items-center justify-center p-4">
        
        {/* Hidden video element for MediaPipe stream feed */}
        <video 
          ref={videoRef} 
          className="hidden" 
          playsInline 
          muted 
        />
        
        {/* Mirror view overlay drawing canvas */}
        <canvas 
          ref={canvasRef} 
          className="absolute inset-0 w-full h-full rounded-xl object-cover" 
        />

        {/* Glow overlay */}
        {webcamActive && (
          <div className="absolute inset-0 bg-violet-500/5 rounded-xl border border-violet-500/10 pointer-events-none" />
        )}

        {/* Active Webcam and Signal details */}
        <div className="absolute top-4 left-4 flex items-center gap-2 px-3 py-1 rounded bg-black/80 border border-zinc-800/80 text-[10px] font-mono tracking-wider font-semibold z-10">
          {webcamActive ? (
            <>
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-zinc-300">WEBCAM LIVE</span>
            </>
          ) : (
            <>
              <div className="w-2 h-2 rounded-full bg-zinc-700" />
              <span className="text-zinc-500">FEED OFFLINE</span>
            </>
          )}
        </div>

        {/* Informative placeholder text */}
        {!webcamActive && (
          <div className="flex flex-col items-center gap-4 text-center z-10 select-none">
            {cameraError ? (
              <>
                <div className="w-16 h-16 rounded-full bg-rose-950/30 border border-rose-500/30 flex items-center justify-center text-rose-400">
                  <WifiOff className="h-7 w-7" />
                </div>
                <span className="text-sm font-semibold text-rose-400">Webcam Denied or Busy</span>
                <span className="text-[11px] text-zinc-500 max-w-[240px] leading-relaxed">
                  Allow camera permissions in browser settings and reload this page to start.
                </span>
              </>
            ) : (
              <>
                <div className="w-16 h-16 rounded-full bg-zinc-900 border border-zinc-855 flex items-center justify-center text-zinc-600">
                  <VideoOff className="h-7 w-7" />
                </div>
                <span className="text-sm font-semibold text-zinc-500">Camera Feed Idle</span>
                <span className="text-[11px] text-zinc-700 max-w-[240px] leading-relaxed">
                  Click &apos;Enable Camera&apos; in the dashboard controller to start live translation.
                </span>
              </>
            )}
          </div>
        )}

        {/* Custom video corners scan indicators */}
        {webcamActive && (
          <>
            <div className="absolute top-6 right-6 w-4 h-4 border-t-2 border-r-2 border-zinc-700 pointer-events-none" />
            <div className="absolute bottom-6 left-6 w-4 h-4 border-b-2 border-l-2 border-zinc-700 pointer-events-none" />
          </>
        )}
      </div>

    </div>
  );
};
