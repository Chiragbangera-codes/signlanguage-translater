import { describe, it, expect } from "vitest";
import { constructMeaningfulSentence } from "../grammar";

describe("constructMeaningfulSentence", () => {
  describe("question patterns", () => {
    it("converts 'how you' to 'How are you?'", () => {
      expect(constructMeaningfulSentence("how you")).toBe("How are you?");
    });

    it("converts 'you how' to 'How are you?'", () => {
      expect(constructMeaningfulSentence("you how")).toBe("How are you?");
    });

    it("converts 'what name you' to 'What is your name?'", () => {
      expect(constructMeaningfulSentence("what name you")).toBe(
        "What is your name?"
      );
    });

    it("converts 'your name what' to 'What is your name?'", () => {
      expect(constructMeaningfulSentence("your name what")).toBe(
        "What is your name?"
      );
    });

    it("converts 'name what' to 'What is your name?'", () => {
      expect(constructMeaningfulSentence("name what")).toBe(
        "What is your name?"
      );
    });

    it("converts 'where go you' to 'Where are you going?'", () => {
      expect(constructMeaningfulSentence("where go you")).toBe(
        "Where are you going?"
      );
    });

    it("converts 'what you doing' to 'What are you doing?'", () => {
      expect(constructMeaningfulSentence("what you doing")).toBe(
        "What are you doing?"
      );
    });

    it("converts 'you what want' to 'What do you want?'", () => {
      expect(constructMeaningfulSentence("you what want")).toBe(
        "What do you want?"
      );
    });
  });

  describe("statement patterns", () => {
    it("converts 'i love you' to 'I love you.'", () => {
      expect(constructMeaningfulSentence("i love you")).toBe("I love you.");
    });

    it("converts 'good morning' to 'Good morning.'", () => {
      expect(constructMeaningfulSentence("good morning")).toBe(
        "Good morning."
      );
    });

    it("converts 'good night' to 'Good night.'", () => {
      expect(constructMeaningfulSentence("good night")).toBe("Good night.");
    });

    it("converts 'thank you' to 'Thank you.'", () => {
      expect(constructMeaningfulSentence("thank you")).toBe("Thank you.");
    });

    it("converts 'sorry' to 'I'm sorry.'", () => {
      expect(constructMeaningfulSentence("sorry")).toBe("I'm sorry.");
    });

    it("converts 'i fine' to 'I am fine.'", () => {
      expect(constructMeaningfulSentence("i fine")).toBe("I am fine.");
    });

    it("converts 'i hungry' to 'I am hungry.'", () => {
      expect(constructMeaningfulSentence("i hungry")).toBe("I am hungry.");
    });

    it("converts 'i want water' to 'I want water.'", () => {
      expect(constructMeaningfulSentence("i want water")).toBe("I want water.");
    });

    it("converts 'i understand' to 'I understand.'", () => {
      expect(constructMeaningfulSentence("i understand")).toBe(
        "I understand."
      );
    });
  });

  describe("single-word patterns", () => {
    it("converts 'hello' to 'Hello.'", () => {
      expect(constructMeaningfulSentence("hello")).toBe("Hello.");
    });

    it("converts 'yes' to 'Yes.'", () => {
      expect(constructMeaningfulSentence("yes")).toBe("Yes.");
    });

    it("converts 'no' to 'No.'", () => {
      expect(constructMeaningfulSentence("no")).toBe("No.");
    });

    it("converts 'congratulations' to 'Congratulations!'", () => {
      expect(constructMeaningfulSentence("congratulations")).toBe(
        "Congratulations!"
      );
    });

    it("converts 'happy birthday' to 'Happy birthday.'", () => {
      expect(constructMeaningfulSentence("happy birthday")).toBe(
        "Happy birthday."
      );
    });
  });

  describe("fallback heuristic", () => {
    it("handles single word with no pattern match (capitalizes + period)", () => {
      expect(constructMeaningfulSentence("world")).toBe("World.");
    });

    it("handles question word with unknown pattern (question mark)", () => {
      const result = constructMeaningfulSentence("how are you");
      expect(result).toBe("How are you?");
    });

    it("handles unknown multi-word sentence with statement", () => {
      expect(constructMeaningfulSentence("good day today")).toBe(
        "Good day today."
      );
    });

    it("capitalizes lowercase 'i' in fallback", () => {
      const result = constructMeaningfulSentence("i am here");
      expect(result).toBe("I am here.");
    });
  });

  describe("edge cases", () => {
    it("returns empty string for empty input", () => {
      expect(constructMeaningfulSentence("")).toBe("");
    });

    it("returns empty string for whitespace-only input", () => {
      expect(constructMeaningfulSentence("   ")).toBe("");
    });

    it("handles mixed case input", () => {
      expect(constructMeaningfulSentence("HOW YOU")).toBe("How are you?");
    });

    it("handles extra whitespace", () => {
      expect(constructMeaningfulSentence("  how   you  ")).toBe(
        "How are you?"
      );
    });
  });
});
