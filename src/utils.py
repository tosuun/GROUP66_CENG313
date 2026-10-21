"""Text measurements used in exploratory analysis."""

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def count_sentences(text):
    sentences = re.split(r"[.!?]+(?:\s+|$)", text.strip())
    return sum(bool(sentence.strip()) for sentence in sentences)


def non_ascii_ratio(text):
    if len(text) == 0:
        return 0

    non_ascii_count = 0

    for character in text:
        if not character.isascii():
            non_ascii_count += 1

    return non_ascii_count / len(text)


def unique_word_ratio(text):
    words = text.lower().split()

    if len(words) == 0:
        return 0

    unique_words = set(words)

    return len(unique_words) / len(words)


def word_similarity(text_1, text_2):
    words_1 = set(text_1.lower().split())
    words_2 = set(text_2.lower().split())

    common_words = words_1 & words_2
    all_words = words_1 | words_2

    if len(all_words) == 0:
        return 0

    return len(common_words) / len(all_words)


def prepare_words(text):
    words = re.findall(r"\b[a-z]+\b", text.lower())

    filtered_words = []

    for word in words:
        if word not in ENGLISH_STOP_WORDS and len(word) > 1:
            filtered_words.append(word)

    return filtered_words
