import re
from difflib import SequenceMatcher
from typing import List
from functools import lru_cache
import logging


@lru_cache(maxsize=1)
def generate_injection_keywords() -> List[str]:
    """
    Generate prompt injection phrases based on common verbs, adjectives, objects, and suffixes.
    """
    verbs = [
        "ignore", "disregard", "skip", "forget", "neglect", "overlook", "omit",
        "bypass", "pay no attention to", "do not follow", "do not obey"
    ]

    adjectives = [
        "", "prior", "previous", "preceding", "above", "foregoing", "earlier", "initial"
    ]

    objects = [
        "content", "text", "instructions", "instruction", "directives", "directive",
        "commands", "command", "context", "conversation", "input", "inputs", "data",
        "message", "messages", "communication", "response", "responses", "request", "requests"
    ]

    suffixes = [
        "", "and start over", "and start anew", "and begin afresh", "and start from scratch"
    ]

    phrases = []
    for verb in verbs:
        for adjective in adjectives:
            for obj in objects:
                for suffix in suffixes:
                    phrase = f"{verb} {adjective} {obj} {suffix}".strip()
                    phrases.append(" ".join(phrase.split()))

    return phrases


def normalize_string(s: str) -> str:
    """Lowercase and remove punctuation and extra spaces from a string."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def get_substrings(text: str, n: int) -> List[str]:
    """Extract n-gram substrings from text."""
    words = text.split()
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def get_heuristic_score(sub: str, keyword_parts: List[str], max_match: int) -> float:
    matches = sum(1 for a, b in zip(sub.split(), keyword_parts) if a == b)
    if matches > 0:
        base_score = 0.5 + 0.5 * min(matches / max_match, 1.0)
        fuzz_penalty = SequenceMatcher(None, sub, " ".join(keyword_parts)).ratio()
        return max(0.0, base_score - fuzz_penalty * 0.1)
    return 0.0


def detect_prompt_injection_using_heuristic_on_input(user_input: str) -> float:
    """
    Detect prompt injection using heuristic keyword matching.

    Args:
        user_input (str): User input string.

    Returns:
        float: Score between 0.0–1.0 indicating likelihood of injection.
    """
    try:
        max_score = 0.0
        max_match_words = 5
        user_input = normalize_string(user_input)

        for phrase in generate_injection_keywords():
            keyword = normalize_string(phrase)
            keyword_parts = keyword.split()
            substrings = get_substrings(user_input, len(keyword_parts))

            for sub in substrings:
                score = get_heuristic_score(sub, keyword_parts, max_match_words)
                max_score = max(max_score, score)

        return round(max_score, 4)
    except Exception as e:
        logging.error(f"Heuristic detection failed: {e}")
        return 0.0