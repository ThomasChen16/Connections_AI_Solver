from itertools import combinations

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