from itertools import combinations

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
def solve(remaining_words, similarity):
    # Base case: Ends the recursion when no remaining words are left
    if not remaining_words:
        return 0, []
    
    # Take the first word from the remaining words list to avoid redundant computation
    first_word = remaining_words[0]
    
    # Holds the best possible total score and corresponding group for each call
    best_score = float("-inf")
    best_groups = []
    
    # Generate candidate groups that include the first word
    for others in combinations(remaining_words[1:], 3):
        group = (first_word,) + others
        
        # Update the remaining words by removing the current group
        remaining = tuple(
            word for word in remaining_words
            if word not in group
        )

        # Recursively solve the remaining words to get best possible scores and groups
        remaining_score, remaining_groups = solve(remaining, similarity)
        
        # Calculate the total score for the current group and the remaining groups
        total_score = (
            group_score(group, similarity) + remaining_score
        )
        
        # Update the best score and corresponding group as we backtrack
        # Compares the current total score with the best score found so far
        if total_score > best_score:
            best_score = total_score
            best_groups = [group] + remaining_groups

    return best_score, best_groups
        
        

# Calculate the average pairwise similarity for a given group of words
def group_score(group, similarity):
    pairs = list(combinations(group, 2))
    
    # Calculate the total similarity for all pairs in the group
    # Similarity matrix where similarity[i][j] represents the similarity between the i-th and j-th word in the group
    total = sum(
        similarity[i][j] for i, j in pairs
    )
    
    # Return the average similarity for the group
    return total / len(pairs)
    