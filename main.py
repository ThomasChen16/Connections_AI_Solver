"""Simple semantic-similarity baseline for a 16-word Connections-style puzzle."""
from itertools import combinations # generate all possible combinations
import numpy as np
from solvers.greedy_solver import greedy_groups
from solvers.constraint_solver import solve
from solvers.scoring import group_score
from models.minilm import load_model_minilm
from models.qwen3 import load_model_qwen3


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


def similarity_matrix(words: list[str], model) -> np.ndarray:
    # Normalized embeddings to 16x16 matrix
    embeddings = model.encode(words, normalize_embeddings=True)
    
    # Convert embeddings into an NumPy array
    embeddings = np.asarray(embeddings)
    
    # Calculate the cosine similarity matrix by multiplying the embeddings with their transpose
    return embeddings @ embeddings.T

def main() -> None:
    samples = [
        ("Sample 5", WORDS_SAMPLE5),
        ("Sample 6 (ambiguous words)", WORDS_SAMPLE6),
        ("Sample 7 (common phrases)", WORDS_SAMPLE7),
        ("Sample 8 (less-obvious synonyms)", WORDS_SAMPLE8),
    ]
    
    test_words = [
    "apple, a type of fruit",
    "banana, a type of fruit",
    "car, a type of vehicle",
    "truck, a type of vehicle"
    ]
    
    # Load the MiniLM model for generating sentence embeddings
    models = {
        "MiniLM": load_model_minilm(),
        "Qwen3": load_model_qwen3()
    }
    for name, model in models.items():
        similarity = similarity_matrix(test_words, model)
    
        print(f"\n{name}:")
        print(f"Apple-Banana: {similarity[0][1]:.4f}")
        print(f"Car-Truck: {similarity[2][3]:.4f}")
        print(f"Apple-Car: {similarity[0][2]:.4f}")
        print(f"Banana-Truck: {similarity[1][3]:.4f}")
    """
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
    """

if __name__ == "__main__":
    main()
