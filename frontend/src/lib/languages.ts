
export interface TargetLanguage {
  code: string;
  label: string;
  name: string;
  bcp47: string;
}

export const SUPPORTED_LANGUAGES: TargetLanguage[] = [
  { code: "en", label: "English", name: "English", bcp47: "en-US" },
  { code: "hi", label: "हिन्दी — Hindi", name: "Hindi", bcp47: "hi-IN" },
  { code: "bn", label: "বাংলা — Bengali", name: "Bengali", bcp47: "bn-IN" },
  { code: "ta", label: "தமிழ் — Tamil", name: "Tamil", bcp47: "ta-IN" },
  { code: "te", label: "తెలుగు — Telugu", name: "Telugu", bcp47: "te-IN" },
  { code: "mr", label: "मराठी — Marathi", name: "Marathi", bcp47: "mr-IN" },
  { code: "kn", label: "ಕನ್ನಡ — Kannada", name: "Kannada", bcp47: "kn-IN" },
  { code: "ml", label: "മലയാളം — Malayalam", name: "Malayalam", bcp47: "ml-IN" },
  { code: "gu", label: "ગુજરાતી — Gujarati", name: "Gujarati", bcp47: "gu-IN" },
  { code: "pa", label: "ਪੰਜਾਬੀ — Punjabi", name: "Punjabi", bcp47: "pa-IN" },
  { code: "ur", label: "اردو — Urdu", name: "Urdu", bcp47: "ur-IN" },
  { code: "es", label: "Español — Spanish", name: "Spanish", bcp47: "es-ES" },
  { code: "fr", label: "Français — French", name: "French", bcp47: "fr-FR" },
  { code: "de", label: "Deutsch — German", name: "German", bcp47: "de-DE" },
  { code: "pt", label: "Português — Portuguese", name: "Portuguese", bcp47: "pt-BR" },
  { code: "ar", label: "العربية — Arabic", name: "Arabic", bcp47: "ar-SA" },
  { code: "zh", label: "中文 — Chinese", name: "Chinese (Simplified)", bcp47: "zh-CN" },
  { code: "ja", label: "日本語 — Japanese", name: "Japanese", bcp47: "ja-JP" },
  { code: "ru", label: "Русский — Russian", name: "Russian", bcp47: "ru-RU" },
];

export const DEFAULT_LANGUAGE_CODE = "en";

export function getLanguage(code: string): TargetLanguage {
  return (
    SUPPORTED_LANGUAGES.find((lang) => lang.code === code) ?? SUPPORTED_LANGUAGES[0]
  );
}
