import numpy as np
import matplotlib.pyplot as plt
from subject_projection import compute_subject_projection, merge_subject_embeddings

# Set random seed for reproducibility
np.random.seed(42)

# Test parameters
N = 5  # Number of subjects
Z = 4  # Number of embeddings per subject
K = 8  # Dimension of each embedding

# Create synthetic embeddings with specific patterns
# For each subject, we'll create embeddings where some are more similar than others
embeddings = np.zeros((N, Z, K))

# Fill embeddings with patterns that will have clear optimal weights
for n in range(N):
    # First embedding: high quality (distinctive pattern)
    embeddings[n, 0] = np.random.normal(0, 0.5, K) + np.array([3, 2, 1, 0, 0, 0, 0, 0])
    
    # Second embedding: medium quality
    embeddings[n, 1] = np.random.normal(0, 1, K) + np.array([1, 1, 1, 1, 0, 0, 0, 0])
    
    # Third embedding: low quality
    embeddings[n, 2] = np.random.normal(0, 2, K) + np.array([0, 0, 0, 0, 0, 0, 0, 0])
    
    # Fourth embedding: noise
    embeddings[n, 3] = np.random.normal(0, 3, K)

# Add subject-specific patterns to make subjects distinguishable
for n in range(N):
    # Add subject-specific pattern to all embeddings of this subject
    subject_pattern = np.zeros(K)
    subject_pattern[n % K] = 3  # Each subject has a strong signal in different dimension
    
    for z in range(Z):
        embeddings[n, z] += subject_pattern

# Parameters for the algorithm
T = 50  # Number of iterations
t_G = 2  # Number of neighbors for G computation
sigma = 0.1  # Gaussian kernel bandwidth
lr_b = 0.1  # Learning rate for embedding weights

# Compute the subject projection weights
b = compute_subject_projection(embeddings, T, t_G, sigma, lr_b)

# Merge embeddings using learned weights
merged_embeddings = merge_subject_embeddings(embeddings, b)

# Print to console
print("Original Embeddings for Each Subject:")
for n in range(N):
    print(f"\nSubject {n}:")
    for z in range(Z):
        print(f"  Embedding {z}: {embeddings[n, z].round(3)}")

print("\nLearned Weights for Merging:")
for n in range(N):
    print(f"Subject {n}: {b[n].round(3)}")

print("\nMerged Embeddings:")
for n in range(N):
    print(f"Subject {n}: {merged_embeddings[n].round(3)}")

# Save results to a txt file
with open('embedding_results.txt', 'w') as f:
    f.write("Original Embeddings for Each Subject:\n")
    for n in range(N):
        f.write(f"\nSubject {n}:\n")
        for z in range(Z):
            f.write(f"  Embedding {z}: {embeddings[n, z].round(3)}\n")
    
    f.write("\nLearned Weights for Merging:\n")
    for n in range(N):
        f.write(f"Subject {n}: {b[n].round(3)}\n")
    
    f.write("\nMerged Embeddings:\n")
    for n in range(N):
        f.write(f"Subject {n}: {merged_embeddings[n].round(3)}\n")

print("\nResults saved to embedding_results.txt") 