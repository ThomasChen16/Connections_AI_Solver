from solvers.scoring import group_score
from itertools import combinations # generate all possible combinations
import numpy as np

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