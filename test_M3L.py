import numpy as np
import matplotlib.pyplot as plt
from M3L import compute_projection, compute_F, compute_H

# Set random seed for reproducibility
np.random.seed(42)

# Test parameters
N = 5  # Number of subjects
Z = 4  # Number of embeddings per subject
K = 8  # Dimension of each embedding - changed from 16 to 8

# Generate random embeddings with specific patterns
embeddings = np.zeros((N, Z, K))

# Fill embeddings with specific patterns:
# First 2 dimensions: mean=0, std=1
# Next 2 dimensions: mean=1, std=1
# Last 4 dimensions: mean=5, std=1
for n in range(N):
    for z in range(Z):
        # First 2 dimensions (0 mean, 1 std)
        embeddings[n, z, 0:2] = np.random.normal(0, 1, 2)
        
        # Next 2 dimensions (1 mean, 1 std)
        embeddings[n, z, 2:4] = np.random.normal(1, 1, 2)
        
        # Last 4 dimensions (5 mean, 1 std)
        embeddings[n, z, 4:8] = np.random.normal(5, 1, 4)

# Normalize embeddings
embeddings = embeddings / np.linalg.norm(embeddings, axis=2, keepdims=True)

# Parameters for the algorithm
T = 10  # Number of iterations
q = 0.5  # Tuning parameter
t_F = 2  # Number of neighbors for F computation
t_H = 2  # Number of neighbors for H computation
sigma = 0.1  # Gaussian kernel bandwidth
lr_w = 0.02  # Learning rate for W
lr_a = 0.05  # Learning rate for a

# Initial computation of F and H
W_init = np.ones((Z, K))
W_init = W_init / np.linalg.norm(W_init, axis=1, keepdims=True)
F_init = compute_F(embeddings, W_init, t_F, sigma)
H_init = compute_H(embeddings, W_init, t_H, sigma)

print(f"Shape of embeddings: {embeddings.shape}")
print(f"Shape of initial F: {F_init.shape}")
print(f"Shape of initial H: {H_init.shape}")

# Test the compute_projection function
print("\nRunning compute_projection...")
W, a = compute_projection(
    embeddings, 
    T=T, 
    q=q, 
    t_F=t_F, 
    t_H=t_H, 
    sigma=sigma, 
    lr_w=lr_w, 
    lr_a=lr_a
)

print(f"\nShape of W: {W.shape}")
print(f"Shape of a: {a.shape}")

# Display some results
print("\nProjection matrix W (first row):")
print(W[0])
print("\nCoefficients vector a:")
print(a)

# Compute final F and H
F_final = compute_F(embeddings, W, t_F, sigma)
H_final = compute_H(embeddings, W, t_H, sigma)

# Check if F and H have changed
print(f"\nInitial F mean: {np.mean(F_init)}")
print(f"Final F mean: {np.mean(F_final)}")
print(f"Initial H mean: {np.mean(H_init)}")
print(f"Final H mean: {np.mean(H_final)}")

# Visualize the results
plt.figure(figsize=(15, 5))

# Plot W matrix as heatmap
plt.subplot(131)
plt.imshow(W, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Projection Matrix W')
plt.xlabel('K')
plt.ylabel('Z')

# Plot a vector
plt.subplot(132)
plt.bar(range(K), a)
plt.title('Coefficients Vector a')
plt.xlabel('Dimension K')
plt.ylabel('Weight')

# Plot the original embeddings and their group means
plt.subplot(133)

# Before normalization - show the pattern of different means across dimensions
mean_dims = []
for k in range(K):
    mean_dims.append(np.mean([np.random.normal(0 if k < 2 else (1 if k < 4 else 5), 1) for _ in range(100)]))

plt.bar(range(K), mean_dims, color='green', alpha=0.6, label='Original Pattern')

# Show the learned weights (a) which should ideally identify the most distinctive dimensions
norm_a = a / np.max(a)  # Normalize for comparison
plt.bar(range(K), norm_a, color='red', alpha=0.5, label='Learned Weights')

plt.title('Original Pattern vs Learned Weights')
plt.xlabel('Dimension K')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('M3L_results.png')
plt.close()

print("\nTest complete. Results plot saved as 'M3L_results.png'")

# Additional analysis - check if weights reflect the pattern
print("\nAverage weights by dimension group:")
print(f"First 2 dimensions (0 mean): {np.mean(a[0:2])}")
print(f"Middle 2 dimensions (1 mean): {np.mean(a[2:4])}")
print(f"Last 4 dimensions (5 mean): {np.mean(a[4:8])}") 