# M3L: Multi-Modal Matrix Learning

M3L is a Python implementation of the Multi-feature Multi-Manifold Learning (M3L) algorithm for multi-modal data analysis. This implementation is based on the paper by Yan et al. for single-sample face recognition.

## Overview

This project provides an implementation of a matrix learning algorithm that:

1. Takes embedding vectors across multiple subjects
2. Computes intra-embedding neighbor matrices (F) and inter-subject neighbor matrices (H)
3. Iteratively optimizes a projection matrix (W) and coefficients vector (a)
4. Produces a final projection that can be used to highlight the most important dimensions

The algorithm is particularly useful for multi-modal data analysis where each subject has multiple related embeddings, and you want to find a common projection that preserves both within-subject and between-subject relationships.

## Installation

Clone this repository:

```bash
git clone https://github.com/yourusername/M3L.git
cd M3L
```

### Requirements

- NumPy
- SciPy
- Matplotlib (for visualization)

## Usage

### Basic Example

```python
from M3L import compute_projection
import numpy as np

# Generate or load your embeddings (shape N, Z, K)
# N: number of subjects
# Z: number of embeddings per subject
# K: dimension of each embedding
embeddings = np.random.randn(5, 4, 8)  # Example with 5 subjects, 4 embeddings each, 8 dimensions

# Set parameters
T = 20  # Number of iterations
q = 0.5  # Tuning parameter
t_F = 2  # Number of neighbors for F computation
t_H = 2  # Number of neighbors for H computation
sigma = 0.1  # Gaussian kernel bandwidth
lr_w = 0.02  # Learning rate for W
lr_a = 0.05  # Learning rate for a

# Compute projection matrix and coefficients
W, a = compute_projection(embeddings, T, q, t_F, t_H, sigma, lr_w, lr_a)

# Use the projection on new data
projected_embeddings = np.zeros_like(embeddings)
for n in range(embeddings.shape[0]):
    for z in range(embeddings.shape[1]):
        projected_embeddings[n, z] = W[z] * a * embeddings[n, z]
```

### Subject Embedding Merging

The repository also includes functionality to merge multiple embeddings per subject into a single high-quality representation:

```python
from subject_projection import compute_subject_projection, merge_subject_embeddings

# For embeddings of shape (N, Z, K)
# N: number of subjects
# Z: number of embeddings per subject
# K: dimension of each embedding
embeddings = np.random.randn(5, 4, 8)

# Set parameters
T = 50  # Number of iterations
t_G = 2  # Number of neighbors for similarity computation
sigma = 0.1  # Gaussian kernel bandwidth
lr_b = 0.05  # Learning rate for embedding weights

# Compute optimal weights for merging embeddings
b = compute_subject_projection(embeddings, T, t_G, sigma, lr_b)

# Merge embeddings using computed weights
merged_embeddings = merge_subject_embeddings(embeddings, b)
# Result shape: (N, K) - one embedding per subject
```

#### How Weights are Determined

The algorithm selects optimal weights for merging multiple embeddings per subject through:

1. **Quality Assessment**: Each embedding is evaluated based on signal strength and clarity metrics, favoring embeddings with:
   - Higher signal-to-noise ratio
   - More consistent patterns
   - Stronger distinctive features
   
2. **Inter-Subject Relationships**: The algorithm builds a similarity graph between subjects (G matrix) and uses it to:
   - Identify which embeddings best preserve subject relationships
   - Encourage similar subjects to use similar embeddings
   
3. **Iterative Refinement**: Weights are updated through gradient descent, balancing:
   - Intrinsic embedding quality
   - How well an embedding distinguishes a subject from others
   - Preservation of neighborhood relationships between subjects
   
4. **Learning Rate Control**: The pace of weight updates is controlled by the learning rate parameter:
   - Lower rates (e.g., 0.01) produce more balanced weights across embeddings
   - Higher rates (e.g., 0.1) more aggressively favor high-quality embeddings

The resulting weights (b matrix) represent the optimal contribution of each embedding to a subject's final representation, emphasizing the most informative embeddings while reducing the influence of noisy or redundant ones.

### Running Tests

A test script is provided to validate the implementation:

```bash
python3 test_M3L.py
python3 test_subject_projection.py
```

These will generate random data, run the algorithms, and save visualizations to `M3L_results.png` and `subject_projection_results.png`.

## Algorithm Details

The algorithm consists of the following key components:

1. **Neighbor Matrices Computation**:
   - F matrix: Captures relationships between embeddings within each subject
   - H matrix: Captures relationships between subjects for each embedding position
   - G matrix: Captures overall similarity between subjects for embedding merging

2. **Iterative Optimization**:
   - W update: Updates the projection matrix using gradient descent
   - a update: Updates the coefficient vector to weigh different dimensions
   - b update: Updates embedding weights for merging multiple embeddings per subject

3. **Projection and Merging**:
   - Final W matrix: Projection weights for each embedding position
   - Final a vector: Global importance weights for each dimension
   - Final b matrix: Optimal weights for merging multiple embeddings per subject

## Parameters

- `N`: Number of subjects
- `Z`: Number of embeddings per subject
- `K`: Dimension of each embedding
- `T`: Number of iterations
- `q`: Tuning parameter
- `t_F`: Number of neighbors to consider for F matrix
- `t_H`: Number of neighbors to consider for H matrix
- `t_G`: Number of neighbors to consider for G matrix (subject similarity)
- `sigma`: Gaussian kernel bandwidth
- `lr_w`: Learning rate for W updates
- `lr_a`: Learning rate for a updates
- `lr_b`: Learning rate for b updates (embedding merging weights)

## Citation

This implementation is based on the following paper:

Yan, H., Lu, J., Zhou, X., & Shang, Y. (2014). Multi-feature multi-manifold learning for single-sample face recognition. Neurocomputing, 143, 134-143.
https://doi.org/10.1016/j.neucom.2014.06.012

## License

This project is licensed under the MIT License - see the LICENSE file for details.
