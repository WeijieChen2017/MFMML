# M3L: Multi-Modal Matrix Learning

M3L is a Python implementation of a novel approach for learning projection matrices from multi-modal embeddings. The algorithm learns optimal projection weights to capture the most important dimensions across similar embeddings while respecting both intra-embedding and inter-subject relationships.

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

### Running Tests

A test script is provided to validate the implementation:

```bash
python3 test_M3L.py
```

This will generate random data, run the algorithm, and save a visualization to `M3L_results.png`.

## Algorithm Details

The algorithm consists of the following key components:

1. **Neighbor Matrices Computation**:
   - F matrix: Captures relationships between embeddings within each subject
   - H matrix: Captures relationships between subjects for each embedding position

2. **Iterative Optimization**:
   - W update: Updates the projection matrix using gradient descent
   - a update: Updates the coefficient vector to weigh different dimensions

3. **Projection**:
   - Final W matrix: Projection weights for each embedding position
   - Final a vector: Global importance weights for each dimension

## Parameters

- `N`: Number of subjects
- `Z`: Number of embeddings per subject
- `K`: Dimension of each embedding
- `T`: Number of iterations
- `q`: Tuning parameter
- `t_F`: Number of neighbors to consider for F matrix
- `t_H`: Number of neighbors to consider for H matrix
- `sigma`: Gaussian kernel bandwidth
- `lr_w`: Learning rate for W updates
- `lr_a`: Learning rate for a updates

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@misc{M3L2023,
  author = {Your Name},
  title = {M3L: Multi-Modal Matrix Learning},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/yourusername/M3L}}
}
```
