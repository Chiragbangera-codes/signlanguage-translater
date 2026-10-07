import type { MetadataRoute } from "next";

// Web app manifest: makes SignSpeak installable as an app on phones
// ("Add to Home screen" / "Install app"). Served at /manifest.webmanifest.
export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "SignSpeak AI - Sign Language Translator",
    short_name: "SignSpeak",
    description:
      "Real-time sign language recognition: signs to words, sentences and speech, running on your device.",
    start_url: "/",
    scope: "/",
    display: "standalone",
    orientation: "portrait",
    background_color: "#09090b",
    theme_color: "#09090b",
    categories: ["education", "accessibility", "utilities"],
    icons: [
      { src: "/icon-192.png", sizes: "192x192", type: "image/png", purpose: "any" },
      { src: "/icon-512.png", sizes: "512x512", type: "image/png", purpose: "any" },
      { src: "/icon-maskable-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
    ],
  };
}
