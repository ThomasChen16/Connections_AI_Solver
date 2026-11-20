from itertools import combinations
from solvers.scoring import group_score

# Memoization dictionary to store previously computed results for specific sets of remaining words
memo = {}

# Generate candidate groups of 4 words based on their pairwise similarity
def generate_candidate_groups(words, similarity):
    candidates = []
    
    # Generate all possible combinations of 4 words from the list
    for indices in combinations(range(len(words)), 4):
        # Generate all pairs of the 4 selected words
        pairs = list(combinations(indices, 2))
        
        # Calculate the average pair wise similarity
        score = sum(
            similarity[i][j] for i, j in pairs
        ) / len(pairs)
        
        candidates.append({
            "indices": indices,
            "score": score
        })
    
    return candidates

# Recursive backtracking explores different possible groupings before decidng which combination is best
def solve(remaining_words, similarity, memo=None):
    if memo is None:
        memo = {}
    
    # Check if the current set of remaining words has already been computed and stored in memo
    if remaining_words in memo:
        return memo[remaining_words]
    
    # Base case: Ends the recursion when no remaining words are left
    if not remaining_words:
        memo[remaining_words] = (0, [])
        return 0, []
    
    # Take the first word from the remaining words list to avoid redundant computation
    first_word = remaining_words[0]
    
    # Holds the best possible total score and corresponding group for each call
    best_score = float("-inf")
    best_groups = []
    
    # Generate candidate groups that include the first word
    for others in combinations(remaining_words[1:], 3):
        # Create a group consisting of the first word and the other three selected words
        group = (first_word,) + others
        
        # Update the remaining words by removing the current group
        remaining = tuple(
            word for word in remaining_words
            if word not in group
        )

        # Pause the current fucntion call and start executing the new function call with the updated remaining words
        remaining_score, remaining_groups = solve(remaining, similarity, memo)
        
        # Calculate the total score for the current group and the remaining groups
        total_score = (
            group_score(group, similarity) + remaining_score
        )
        
        # Update the best score and corresponding group as we backtrack
        # Compares the current total score with the best score found so far
        if total_score > best_score:
            best_score = total_score
            best_groups = [group] + remaining_groups
    
    # Store the best score and corresponding groups for the current set of remaining words in memo
    memo[remaining_words] = (best_score, best_groups)

    return best_score, best_groups
    