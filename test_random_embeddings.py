import numpy as np
import matplotlib.pyplot as plt
from subject_projection import compute_subject_projection, merge_subject_embeddings

# Set random seed for reproducibility
np.random.seed(42)

# Test parameters
N = 5  # Number of subjects
Z = 4  # Number of embeddings per subject
K = 8  # Dimension of each embedding

# Create purely random embeddings with different means and standard deviations
embeddings = np.zeros((N, Z, K))

# Different means and standard deviations for each embedding type
means = [2.0, 1.0, 0.5, 0.0]  # Decreasing means
stds = [0.5, 1.0, 2.0, 3.0]   # Increasing standard deviations

# Generate embeddings: purely random with different distributions
for n in range(N):
    for z in range(Z):
        embeddings[n, z] = np.random.normal(means[z], stds[z], K)

# Parameters for the algorithm
T = 50  # Number of iterations
t_G = 2  # Number of neighbors for G computation
sigma = 0.1  # Gaussian kernel bandwidth
lr_b = 0.1  # Learning rate for embedding weights

print(f"Shape of embeddings: {embeddings.shape}")

# Compute the subject projection weights
print("\nRunning compute_subject_projection...")
b = compute_subject_projection(
    embeddings, 
    T=T, 
    t_G=t_G, 
    sigma=sigma, 
    lr_b=lr_b
)

print(f"\nShape of b: {b.shape}")

# Display weights for each subject
print("\nLearned Weights for Merging:")
for n in range(N):
    print(f"Subject {n}: {b[n].round(3)}")

# Merge embeddings using learned weights
merged_embeddings = merge_subject_embeddings(embeddings, b)
print(f"\nShape of merged embeddings: {merged_embeddings.shape}")

# Save results to a txt file
with open('random_embedding_results.txt', 'w') as f:
    f.write("Random Embeddings with Different Distributions:\n")
    f.write(f"Means used: {means}\n")
    f.write(f"Standard deviations used: {stds}\n\n")
    
    f.write("Original Embeddings for Each Subject:\n")
    for n in range(N):
        f.write(f"\nSubject {n}:\n")
        for z in range(Z):
            f.write(f"  Embedding {z} (mean={means[z]}, std={stds[z]}): {embeddings[n, z].round(3)}\n")
    
    f.write("\nLearned Weights for Merging:\n")
    for n in range(N):
        f.write(f"Subject {n}: {b[n].round(3)}\n")
    
    f.write("\nMerged Embeddings:\n")
    for n in range(N):
        f.write(f"Subject {n}: {merged_embeddings[n].round(3)}\n")

print("\nTest complete. Results saved as 'random_embedding_results.txt'")

# Average weights across subjects for each embedding
avg_weights = np.mean(b, axis=0)
print("\nAverage weights across subjects:")
for z in range(Z):
    print(f"Embedding {z} (mean={means[z]}, std={stds[z]}): {avg_weights[z]:.4f}")

# Check if first embedding (highest mean, lowest std) got highest weight
if np.argmax(avg_weights) == 0:
    print("\nExpected result: The embedding with highest mean and lowest std (index 0) received the highest weight!")
else:
    print(f"\nNote: The highest weight went to embedding {np.argmax(avg_weights)} instead of the expected highest quality embedding (index 0)") 