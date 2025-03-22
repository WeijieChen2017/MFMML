import numpy as np
import matplotlib.pyplot as plt
from subject_projection import compute_subject_G, update_b, merge_subject_embeddings

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
T = 200  # Increased number of iterations to see more gradual changes
t_G = 2  # Number of neighbors for G computation
sigma = 0.1  # Gaussian kernel bandwidth
lr_b = 0.01  # Very slow learning rate (reduced from 0.1 to 0.01)

print(f"Shape of embeddings: {embeddings.shape}")
print(f"Using learning rate: {lr_b}")

# Compute subject similarity matrix G
G = compute_subject_G(embeddings, t_G, sigma)

# Initialize weights uniformly
b = np.ones((N, Z)) / Z

# Track weight evolution for each subject
weight_history = np.zeros((T+1, N, Z))
weight_history[0] = b.copy()  # Initial weights

# Run the algorithm step by step and track weights
print("\nTracking weight evolution over iterations...\n")
for t in range(T):
    # Update embedding weights
    b = update_b(embeddings, b, G, t_G, lr_b)
    
    # Store weights at this iteration
    weight_history[t+1] = b.copy()
    
    # Print progress at specific intervals
    if (t+1) % 20 == 0 or t == 0:
        avg_weights = np.mean(b, axis=0)
        print(f"Iteration {t+1:3d}: Average weights = [{avg_weights[0]:.3f}, {avg_weights[1]:.3f}, {avg_weights[2]:.3f}, {avg_weights[3]:.3f}]")
        
        # Show weights for subject 0 as an example
        print(f"               Subject 0 weights = [{b[0,0]:.3f}, {b[0,1]:.3f}, {b[0,2]:.3f}, {b[0,3]:.3f}]")

# Final weights
print("\nFinal weights for each subject:")
for n in range(N):
    print(f"Subject {n}: [{b[n,0]:.3f}, {b[n,1]:.3f}, {b[n,2]:.3f}, {b[n,3]:.3f}]")

# Average weights across subjects
avg_weights = np.mean(b, axis=0)
print("\nFinal average weights:")
for z in range(Z):
    print(f"Embedding {z} (mean={means[z]}, std={stds[z]}): {avg_weights[z]:.4f}")

# Plot weight evolution
plt.figure(figsize=(15, 10))

# Plot average weights across subjects over iterations
plt.subplot(2, 1, 1)
for z in range(Z):
    avg_weight_over_time = np.mean(weight_history[:, :, z], axis=1)
    plt.plot(range(T+1), avg_weight_over_time, label=f"Emb {z} (μ={means[z]}, σ={stds[z]})")

plt.title('Average Weight Evolution Across All Subjects')
plt.xlabel('Iteration')
plt.ylabel('Average Weight')
plt.legend()
plt.grid(True)

# Plot weights for subject 0 as an example
plt.subplot(2, 1, 2)
for z in range(Z):
    subj_weight_over_time = weight_history[:, 0, z]
    plt.plot(range(T+1), subj_weight_over_time, label=f"Emb {z} (μ={means[z]}, σ={stds[z]})")

plt.title('Weight Evolution for Subject 0')
plt.xlabel('Iteration')
plt.ylabel('Weight')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('weight_evolution.png')

print("\nWeight evolution plot saved as 'weight_evolution.png'")

# Save results to a txt file
with open('weight_evolution_results.txt', 'w') as f:
    f.write("Weight Evolution with Slow Learning Rate\n")
    f.write(f"Learning rate: {lr_b}\n")
    f.write(f"Iterations: {T}\n")
    f.write(f"Embedding means: {means}\n")
    f.write(f"Embedding standard deviations: {stds}\n\n")
    
    f.write("Weight progression (every 20 iterations):\n")
    for t in range(0, T+1, 20):
        f.write(f"\nIteration {t}:\n")
        avg_weights = np.mean(weight_history[t], axis=0)
        f.write(f"  Average: [{avg_weights[0]:.4f}, {avg_weights[1]:.4f}, {avg_weights[2]:.4f}, {avg_weights[3]:.4f}]\n")
        
        for n in range(N):
            f.write(f"  Subject {n}: [{weight_history[t,n,0]:.4f}, {weight_history[t,n,1]:.4f}, {weight_history[t,n,2]:.4f}, {weight_history[t,n,3]:.4f}]\n")
    
    f.write("\nFinal weights:\n")
    for n in range(N):
        f.write(f"Subject {n}: [{b[n,0]:.4f}, {b[n,1]:.4f}, {b[n,2]:.4f}, {b[n,3]:.4f}]\n")

print("Detailed results saved to 'weight_evolution_results.txt'") 