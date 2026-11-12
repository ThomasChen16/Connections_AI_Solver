"""Simple semantic-similarity baseline for a 16-word Connections-style puzzle."""
from itertools import combinations # generate all possible combinations
import numpy as np
from sentence_transformers import SentenceTransformer # Model for generating sentence embeddings
from constraint_solver import solve

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

# Deliberately ambiguous words, with several different kinds of groups.
# Intended groups: fish, birds of prey/other fliers, banking terms, tree parts.
WORDS_SAMPLE6 = [
    "bass", "trout", "salmon", "perch",
    "bat", "owl", "eagle", "hawk",
    "bank", "teller", "loan", "vault",
    "bark", "leaf", "root", "trunk",
]

# Categories rely on familiar compounds and phrases, not just close meanings.
# Intended groups: ___fly, ___light, ___ball, and ___brush.
WORDS_SAMPLE7 = [
    "fire", "house", "dragon", "butter",
    "traffic", "stop", "green", "candle",
    "snow", "foot", "basket", "wall",
    "tooth", "hair", "nail", "paint",
]

# Less-obvious synonym sets, including several relatively uncommon words.
# Intended groups: concise, joyful, cautious, and sluggish.
WORDS_SAMPLE8 = [
    "terse", "laconic", "pithy", "succinct",
    "elated", "ebullient", "buoyant", "jubilant",
    "wary", "circumspect", "guarded", "leery",
    "torpid", "lethargic", "languid", "sluggish",
]


def similarity_matrix(words: list[str], model: SentenceTransformer) -> np.ndarray:
    # Normalized embeddings to 16x16 matrix
    embeddings = model.encode(words, normalize_embeddings=True)
    # Return the similarity matrix as a NumPy array
    return np.asarray(embeddings) @ np.asarray(embeddings).T


def group_score(indices: tuple[int, ...], similarity: np.ndarray) -> float:
    # Return the mean pairwise cosine similarity of the six pairs in a group of four.
    # Calculate pairwise similarities because cosine similarity compares two vectors at a time and then we average them for the group
    return float(np.mean([similarity[i, j] for i, j in combinations(indices, 2)]))


def greedy_groups(words: list[str], similarity: np.ndarray) -> list[tuple[tuple[int, ...], float]]:
    # Baseline: choose highest-scoring 4-word group, then repeat.
    # Remaining is the set of words indices that haven't been assigned to a group
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
        ("Sample 5", WORDS_SAMPLE5),
        ("Sample 6 (ambiguous words)", WORDS_SAMPLE6),
        ("Sample 7 (common phrases)", WORDS_SAMPLE7),
        ("Sample 8 (less-obvious synonyms)", WORDS_SAMPLE8),
    ]
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    for sample_name, words in samples:
        print(f"\n{'=' * 20} {sample_name} {'=' * 20}")
        if len(words) != 16 or len(set(words)) != 16:
            raise ValueError(f"{sample_name} puzzle must contain 16 distinct words")

        similarity = similarity_matrix(words, model)
        
        # Greedy baseline approach
        print("\nPredicted groups (Greedy Baseline): ")
        
        greedy_result = greedy_groups(words, similarity)
        greedy_total = 0
        
        for number, (group, score) in enumerate(greedy_result, 1):
            print(
                f"  Group {number}: "
                f"{', '.join(words[i] for i in group)} "
                f"(score={score:.3f})"
            )
            greedy_total += score
        
        print(f"Greedy Total Score: {greedy_total:.3f}")
        
        # Recursive backtracking approach
        print("\nPredicted groups (Contraint-Based Search): ")
        
        remaining_words = tuple(range(len(words)))
        
        best_score, best_groups = solve(remaining_words, similarity)
        
        for number, group in enumerate(best_groups, 1):
            score = group_score(group, similarity)
            
            print(
                f"  Group {number}: "
                f"{', '.join(words[i] for i in group)} "
                f"(score={score:.3f})"
            )
        
        print(f"Constraint-Based Total Score: {best_score:.3f}")
        
        # Compare the total scores
        print("\nComparison:")
        print(f"  Greedy:     {greedy_total:.3f}")
        print(f"  Backtracking: {best_score:.3f}")
        print(f"  Improvement:  {best_score - greedy_total:+.3f}")
        

if __name__ == "__main__":
    main()
