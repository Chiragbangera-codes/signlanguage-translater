import { describe, expect, it } from "vitest";
import { findVoiceForLanguage } from "../speech";
import { SUPPORTED_LANGUAGES, getLanguage, DEFAULT_LANGUAGE_CODE } from "../languages";

function voice(name: string, lang: string): SpeechSynthesisVoice {
  return { name, lang } as SpeechSynthesisVoice;
}

const VOICES = [
  voice("Microsoft David", "en-US"),
  voice("Microsoft Hazel", "en-GB"),
  voice("Google हिन्दी", "hi-IN"),
  voice("Google Français", "fr-FR"),
];

describe("findVoiceForLanguage", () => {
  it("prefers an exact locale match", () => {
    expect(findVoiceForLanguage(VOICES, "en-GB")?.name).toBe("Microsoft Hazel");
  });

  it("falls back to any voice for the same base language", () => {
    expect(findVoiceForLanguage(VOICES, "en-AU")?.name).toBe("Microsoft David");
  });

  it("matches regardless of case and underscore separators", () => {
    expect(findVoiceForLanguage([voice("Hindi", "hi_IN")], "hi-IN")?.name).toBe("Hindi");
  });

  it("returns null when the browser ships no voice for the language", () => {
    expect(findVoiceForLanguage(VOICES, "ta-IN")).toBeNull();
  });

  it("returns null for an empty voice list", () => {
    expect(findVoiceForLanguage([], "en-US")).toBeNull();
  });
});

describe("language catalogue", () => {
  it("has unique codes", () => {
    const codes = SUPPORTED_LANGUAGES.map((l) => l.code);
    expect(new Set(codes).size).toBe(codes.length);
  });

  it("gives every language a base-language-matching bcp47 tag", () => {
    for (const lang of SUPPORTED_LANGUAGES) {
      expect(lang.bcp47.toLowerCase().split("-")[0]).toBe(lang.code.toLowerCase());
    }
  });

  it("falls back to the default language for an unknown code", () => {
    expect(getLanguage("klingon").code).toBe(DEFAULT_LANGUAGE_CODE);
  });
});
