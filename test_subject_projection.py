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
T = 50  # Number of iterations (increased from 15 to 50)
t_G = 2  # Number of neighbors for G computation
sigma = 0.1  # Gaussian kernel bandwidth
lr_b = 0.1  # Learning rate for embedding weights (increased from 0.05 to 0.1)

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
print("\nLearned embedding weights for each subject:")
for n in range(N):
    print(f"Subject {n}: {b[n]}")

# Merge embeddings using learned weights
merged_embeddings = merge_subject_embeddings(embeddings, b)
print(f"\nShape of merged embeddings: {merged_embeddings.shape}")

# Also merge using uniform weights for comparison
uniform_merged = merge_subject_embeddings(embeddings)

# Visualize the results
plt.figure(figsize=(15, 10))

# Plot 1: Learned weights for each subject
plt.subplot(221)
for n in range(N):
    plt.plot(range(Z), b[n], marker='o', label=f"Subject {n}")
plt.title('Learned Weights for Each Subject')
plt.xlabel('Embedding Index')
plt.ylabel('Weight')
plt.grid(True)
plt.legend()

# Plot 2: Compare weights across subjects
plt.subplot(222)
width = 0.15
positions = np.arange(Z)
for n in range(N):
    plt.bar(positions + n*width, b[n], width=width, label=f"Subject {n}")
plt.title('Weight Distribution Across Subjects')
plt.xlabel('Embedding Index')
plt.ylabel('Weight')
plt.xticks(positions + width*2, [f"Emb {z}" for z in range(Z)])
plt.legend()

# Plot 3: Subject distances before and after merging
plt.subplot(223)
# Flatten embeddings for each subject
flat_original = embeddings.reshape(N, Z*K)
# Compute pairwise distances between subjects
original_distances = np.zeros((N, N))
weighted_distances = np.zeros((N, N))
uniform_distances = np.zeros((N, N))

for i in range(N):
    for j in range(i+1, N):
        # Original distance (average of distances between all embeddings)
        emb_distances = []
        for z in range(Z):
            emb_distances.append(np.linalg.norm(embeddings[i, z] - embeddings[j, z]))
        original_distances[i, j] = original_distances[j, i] = np.mean(emb_distances)
        
        # Weighted merged distance
        weighted_distances[i, j] = weighted_distances[j, i] = np.linalg.norm(
            merged_embeddings[i] - merged_embeddings[j]
        )
        
        # Uniform merged distance
        uniform_distances[i, j] = uniform_distances[j, i] = np.linalg.norm(
            uniform_merged[i] - uniform_merged[j]
        )

# Plot distance matrices
plt.imshow(original_distances, cmap='viridis')
plt.title('Average Distances Between Original Embeddings')
plt.colorbar()

# Plot 4: Compare distances with learned weights vs uniform weights
plt.subplot(224)
n_pairs = N * (N-1) // 2  # Number of unique pairs
pair_idx = 0
pair_labels = []
orig_dist_values = []
weighted_dist_values = []
uniform_dist_values = []

for i in range(N):
    for j in range(i+1, N):
        pair_labels.append(f"{i}-{j}")
        orig_dist_values.append(original_distances[i, j])
        weighted_dist_values.append(weighted_distances[i, j])
        uniform_dist_values.append(uniform_distances[i, j])
        pair_idx += 1

width = 0.25
positions = np.arange(n_pairs)
plt.bar(positions - width, orig_dist_values, width=width, label="Original Avg")
plt.bar(positions, weighted_dist_values, width=width, label="Weighted")
plt.bar(positions + width, uniform_dist_values, width=width, label="Uniform")
plt.title('Distance Between Subject Pairs')
plt.xlabel('Subject Pair')
plt.ylabel('Distance')
plt.xticks(positions, pair_labels)
plt.legend()

plt.tight_layout()
plt.savefig('subject_projection_results.png')
plt.close()

print("\nTest complete. Results plot saved as 'subject_projection_results.png'")

# Additional analysis - compute the correlation between weights and embedding quality
# First embedding should get highest weight since it has the most distinctive pattern
avg_weights = np.mean(b, axis=0)
print("\nAverage weights across subjects:")
for z in range(Z):
    print(f"Embedding {z}: {avg_weights[z]:.4f}")
    
# Check if first embedding (best quality) got highest weight
if np.argmax(avg_weights) == 0:
    print("\nSuccess: The highest quality embedding (index 0) received the highest weight!")
else:
    print(f"\nNote: The highest weight went to embedding {np.argmax(avg_weights)} instead of the expected highest quality embedding (index 0)") 