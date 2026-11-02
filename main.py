"""Simple semantic-similarity baseline for a 16-word Connections-style puzzle."""
from itertools import combinations # generate all possible combinations
import numpy as np
from sentence_transformers import SentenceTransformer # Model for generating sentence embeddings

# Example words datasets for the puzzle
WORDS_SAMPLE1 = [
    "apple", "dog", "piano", "red",
    "banana", "cat", "guitar", "blue",
    "orange", "horse", "drum", "green",
    "grape", "cow", "violin", "yellow",
]

WORDS_SAMPLE2 = [
    "rose", "tulip", "daisy", "lily",
    "penny", "nickel", "dime", "quarter",
    "winter", "spring", "summer", "autumn",
    "mercury", "venus", "earth", "mars",
]

WORDS_SAMPLE3 = [
    "hammer", "pliers", "wrench", "screwdriver",
    "bread", "rice", "pasta", "potato",
    "north", "south", "east", "west",
    "novel", "poem", "essay", "play",
]

WORDS_SAMPLE4 = [
    "eagle", "falcon", "hawk", "owl",
    "ruby", "emerald", "sapphire", "diamond",
    "tennis", "soccer", "hockey", "baseball",
    "chair", "table", "desk", "shelf",
]

WORDS_SAMPLE5 = [
    "coffee", "tea", "juice", "water",
    "jupiter", "saturn", "uranus", "neptune",
    "shark", "whale", "dolphin", "tuna",
    "cello", "flute", "trumpet", "clarinet",
]


def similarity_matrix(words: list[str], model: SentenceTransformer) -> np.ndarray:
    # Normalized embeddings to 16x16 matrix
    embeddings = model.encode(words, normalize_embeddings=True)
    # Return the similarity matrix as a NumPy array
    return np.asarray(embeddings) @ np.asarray(embeddings).T


def group_score(indices: tuple[int, ...], similarity: np.ndarray) -> float:
    # Return the mean pairwise cosine similarity of the six pairs in a group of four.
    return float(np.mean([similarity[i, j] for i, j in combinations(indices, 2)]))


def greedy_groups(words: list[str], similarity: np.ndarray) -> list[tuple[tuple[int, ...], float]]:
    # Baseline: choose highest-scoring 4-word group, then repeat.
    # Remaining is the set of words indicies that haven't been assigned to a group
    remaining = set(range(len(words)))
    result = []
    # While remaining is not empty, keep selecting the best group of 4 words
    while remaining:
        choices = combinations(sorted(remaining), 4)
        # Keep track of the best group found so far
        best = max(choices, key=lambda g: group_score(g, similarity))
        # Calculate the score of the best group found so far
        score = group_score(best, similarity)
        result.append((best, score))
        # Update remaining by removing the words that have been assigned to the best group
        remaining.difference_update(best)
    return result


def main() -> None:
    samples = [
        ("Sample 1", WORDS_SAMPLE1),
        ("Sample 2", WORDS_SAMPLE2),
        ("Sample 3", WORDS_SAMPLE3),
        ("Sample 4", WORDS_SAMPLE4),
        ("Sample 5", WORDS_SAMPLE5),
    ]
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    for sample_name, words in samples:
        print(f"\n{'=' * 20} {sample_name} {'=' * 20}")
        if len(words) != 16 or len(set(words)) != 16:
            raise ValueError(f"{sample_name} puzzle must contain 16 distinct words")

        similarity = similarity_matrix(words, model)
        print("Top similar pairs:")
        pairs = sorted(
            ((similarity[i, j], words[i], words[j]) for i, j in combinations(range(16), 2)),
            reverse=True,
        )
        for score, a, b in pairs[:10]:
            print(f"  {a:>8} - {b:<8}: {score:.3f}")
        print("\nPredicted groups (greedy baseline):")
        for number, (group, score) in enumerate(greedy_groups(words, similarity), 1):
            print(f"  Group {number}: {', '.join(words[i] for i in group)}  (score={score:.3f})")

if __name__ == "__main__":
    main()
