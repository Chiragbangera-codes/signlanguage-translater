# Frontend Architecture Documentation

This document describes the Next.js 15 client architecture, Zustand state structure, MediaPipe tracking pipeline, prediction stabilization logic, and accessibility details for the SignSpeak AI translation dashboard.

---

## 1. Application Architecture Overview

The frontend is built using Next.js 15 (App Router), React 19, TypeScript, Tailwind CSS, Framer Motion, and Zustand:

```text
frontend/
  ├── src/
  │     ├── app/
  │     │     ├── layout.tsx         <-- Root layout & metadata
  │     │     ├── globals.css        <-- Tailwind CSS configurations
  │     │     └── page.tsx           <-- Main entry (Portal / Dashboard switcher)
  │     ├── components/
  │     │     └── translator/        <-- Modular Dashboard Components
  │     │           ├── CameraCard.tsx
  │     │           ├── Controls.tsx
  │     │           ├── PredictionCard.tsx
  │     │           ├── ConfidenceBar.tsx
  │     │           ├── WordBuilder.tsx
  │     │           ├── SentenceBuilder.tsx
  │     │           ├── SettingsPanel.tsx
  │     │           ├── HistoryPanel.tsx
  │     │           └── StatusBar.tsx
  │     └── store/
  │           └── useTranslatorStore.ts <-- Zustand Store (Global State Manager)
  ├── package.json
  └── postcss.config.js
```

### Rendering Architecture
*   **SSR & Client Isolation**: Next.js App Router prerenders pages on the server. However, third-party libraries like MediaPipe Hands and window-bound features (webcam streams, speech synthesis) are client-only. To prevent SSR bundle failures during compile time, `CameraCard` is imported dynamically using `next/dynamic` with `ssr: false`.

---

## 2. Global State Management (Zustand)

Global UI and translation states are managed inside [useTranslatorStore.ts](file:///C:/Users/User/Desktop/SignSpeakAI/frontend/src/store/useTranslatorStore.ts):

*   **Active Pipeline State**: `webcamActive` (camera connected), `isTranslating` (inference active), `modelLoaded` (ML status), `statusBarMessage` (system logger log).
*   **Inference Values**: `currentPrediction` (top character), `currentConfidence`, `topPredictions` (top 3 classes), `processingTimeMs` (latency).
*   **Language Buffers**: `currentWord` (active word builder), `constructedSentence` (active sentence builder), `history` (previously spoken logs).
*   **Settings Parameters**: `confidenceThreshold` (filtering cut-off), `speechRate` (voice playback speed), `selectedVoiceName` (browser voice configuration), `cameraMirrored` (visual flip option).
*   **Performance Metrics**: `cameraFps` (capture frequency), `apiHealthy` (backend status check).

---

## 3. MediaPipe Hands & Video Processing Pipeline

The hand coordinate tracking flow resides inside [CameraCard.tsx](file:///C:/Users/User/Desktop/SignSpeakAI/frontend/src/components/translator/CameraCard.tsx):

1.  **Dynamic Script Loading**: To guarantee build stability, the MediaPipe script (`hands.js`) is loaded dynamically via CDN from `cdn.jsdelivr.net` at client mount time.
2.  **Webcam Capture**: Prompts the user for video permissions and plays the live stream inside a hidden `<video>` element.
3.  **Horizontal Mirroring**: A drawing loop runs at $\ge$ 30 FPS using browser `requestAnimationFrame`. Prior to drawing onto the canvas context, a coordinate translation scale is applied:
    $$\text{ctx.translate}(w, 0); \quad \text{ctx.scale}(-1, 1)$$
    This mirrors both the raw webcam image and drawing landmarks horizontally.
4.  **Skeleton Overlay Drawing**: Hand joints and coordinates are drawn with customizable glowing styles (rose fingertip nodes and glowing purple connection lines).
5.  **Pydantic Feature Mapping**: MediaPipe outputs are parsed. If a hand is missing, its 63-coordinate array remains padded with `-1.0` values. `uses_two_hands` is mapped based on detection length. The resulting JSON payload matches the backend Pydantic model:
    ```json
    {
      "left_hand": [...63 floats...],
      "right_hand": [...63 floats...],
      "uses_two_hands": 1.0 or 0.0
    }
    ```
6.  **Throttling Filter**: Inference requests are throttled at 5 requests per second (200ms interval) to conserve client bandwidth, while landmark drawing continues at full 30 FPS.

---

## 4. Hold & Cooldown Prediction Stabilization

Raw predictions fluctuate under bad framing. We implemented a multi-stage stabilization pipeline:

```text
                 [ Raw Inference Output ]
                            │
            Does Confidence >= Threshold (e.g. 80%)?
                            ├── [No] ──> Discard prediction
                            └── [Yes]
                                │
                    [ Rolling Buffer (Size=10) ]
                                │
                    Enforce 60% Majority Vote?
                            ├── [No] ──> Reset Majority Hold
                            └── [Yes]
                                │
                     Sustained Hold >= 700ms?
                            ├── [No] ──> Maintain Hold Timer
                            └── [Yes]
                                │
                      Is Cooldown >= 500ms?
                            ├── [No] ──> Await Cooldown expiry
                            └── [Yes]
                                │
                     [ Commit Letter to Word ]
```

1.  **Confidence Gate**: Ignores predictions falling below the configured threshold slider (defaults to 80.0%).
2.  **Rolling Buffer**: Pushes predictions into a sliding array window of size 10.
3.  **Majority Vote**: Enforces a 60% agreement threshold (6/10 identical values in the buffer) to recognize a gesture letter.
4.  **Hold Trigger**: The majority letter must stay dominant for at least 700ms.
5.  **Cooldown Lock**: Blocks new appends for 500ms after a character is added, preventing double-letter stuttering.

---

## 5. Speech Synthesis

Speech playout is handled locally in the browser utilizing the native HTML5 Web Speech API:
*   **Voice Customization**: Listens to `window.speechSynthesis.onvoiceschanged` to load all system-installed TTS voices in a selector dropdown.
*   **Speed Adaptation**: Binds the `speechRate` variable to `utterance.rate` (supporting 0.5x to 1.5x speed).

---

## 6. Accessibility & Keyboard Shortcuts

The translator sandbox incorporates keyboard accessibility features. Users can operate translation functions directly from the keyboard without relying on mouse pointers:

| Hotkey | Action | Target Component |
| :---: | --- | --- |
| `Space` | Commits current word | `WordBuilder` $\rightarrow$ `SentenceBuilder` |
| `Backspace` | Deletes last character | `WordBuilder` |
| `Enter` | Speaks the sentence out loud | `SentenceBuilder` $\rightarrow$ Web Speech API |
| `Escape` | Clears the active sentence | `SentenceBuilder` |

*Note: Shortcuts are bypassed if user focus is active inside input or voice selection select elements.*
