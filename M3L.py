import numpy as np
from typing import Tuple
from scipy.spatial.distance import cdist
from scipy.linalg import solve

def compute_F(embeddings: np.ndarray, W: np.ndarray, t_F: int, sigma: float = 0.01) -> np.ndarray:
    """
    Compute the intra embedding neighbor matrix F based on t_F nearest neighbors using Gaussian kernel.
    Uses projected distances based on current W.
    
    Args:
        embeddings (np.ndarray): Input embeddings array of shape (N, Z, K)
        W (np.ndarray): Projection matrix of shape (Z, K)
        t_F (int): Number of nearest neighbors to consider
        sigma (float): Gaussian kernel bandwidth parameter. Defaults to 0.01
    
    Returns:
        np.ndarray: Intra embedding neighbor matrix F with shape (N, Z, Z)
    """
    N, Z, K = embeddings.shape
    F = np.zeros((N, Z, Z))
    
    # For each subject
    for n in range(N):
        # Project embeddings using W
        projected_embeddings = np.zeros((Z, K))
        for z in range(Z):
            projected_embeddings[z] = W[z] * embeddings[n, z]
        
        # Compute pairwise distances between all Z projected embeddings for this subject
        distances = cdist(projected_embeddings, projected_embeddings)
        
        # Compute Gaussian kernel values
        gaussian_values = np.exp(-distances**2 / (sigma**2))
        
        # For each embedding, find its t_F nearest neighbors
        for i in range(Z):
            # Get indices of t_F nearest neighbors (excluding self)
            nearest_indices = np.argsort(distances[i])[1:t_F+1]
            # Set F value to gaussian kernel value for nearest neighbors
            F[n, i, nearest_indices] = gaussian_values[i, nearest_indices]
            
    return F

def compute_H(embeddings: np.ndarray, W: np.ndarray, t_H: int, sigma: float = 0.01) -> np.ndarray:
    """
    Compute the inter-subject neighbor matrix H based on t_H nearest neighbors using Gaussian kernel.
    Uses projected distances based on current W.
    
    Args:
        embeddings (np.ndarray): Input embeddings array of shape (N, Z, K)
        W (np.ndarray): Projection matrix of shape (Z, K)
        t_H (int): Number of nearest neighbors to consider
        sigma (float): Gaussian kernel bandwidth parameter. Defaults to 0.01
    
    Returns:
        np.ndarray: Inter-subject neighbor matrix H with shape (Z, N, N)
    """
    N, Z, K = embeddings.shape
    H = np.zeros((Z, N, N))
    
    # For each embedding position
    for z in range(Z):
        # Get embeddings at position z for all subjects and project them
        projected_embeddings = np.zeros((N, K))
        for n in range(N):
            projected_embeddings[n] = W[z] * embeddings[n, z]
        
        # Compute pairwise distances between all N projected subjects at position z
        distances = cdist(projected_embeddings, projected_embeddings)
        
        # Compute Gaussian kernel values
        gaussian_values = np.exp(-distances**2 / (sigma**2))
        
        # For each subject, find its t_H nearest neighbors
        for i in range(N):
            # Get indices of t_H nearest neighbors (excluding self)
            nearest_indices = np.argsort(distances[i])[1:t_H+1]
            # Set H value to gaussian kernel value for nearest neighbors
            H[z, i, nearest_indices] = gaussian_values[i, nearest_indices]
    
    return H

def update_projection_at_position(embeddings: np.ndarray, position: int, W: np.ndarray, 
                               a: np.ndarray, F: np.ndarray, H: np.ndarray, t_F: int, t_H: int,
                               lr_w: float = 0.01) -> np.ndarray:
    """
    Update projection function W at a specific position using gradient descent.
    
    Args:
        embeddings (np.ndarray): Embeddings of shape (N, Z, K)
        position (int): Position in Z to update
        W (np.ndarray): Current projection matrix of shape (Z, K)
        a (np.ndarray): Current coefficients vector of shape (K,)
        F (np.ndarray): Intra embedding neighbor matrix of shape (N, Z, Z)
        H (np.ndarray): Inter-subject neighbor matrix of shape (Z, N, N)
        t_F (int): Number of F neighbors to consider
        t_H (int): Number of H neighbors to consider
        lr_w (float): Learning rate for W gradient descent
    
    Returns:
        np.ndarray: Updated projection vector at position of shape (K,)
    """
    N, Z, K = embeddings.shape
    z = position
    w = W[z].copy()  # Current projection weights for position z
    
    # Initialize gradient
    gradient = np.zeros(K)
    
    # Iterate through each subject
    for n in range(N):
        # Get current embedding for subject n at position z
        x = embeddings[n, z, :]  # Original embedding that remains fixed
        
        # Part 1: Gradient from F (intra-embedding relationships)
        # Find t_F closest neighbors according to F(n, z, :)
        f_neighbors = np.argsort(-F[n, z, :])[:t_F]  # Get indices of highest F values
        
        for neighbor_idx in f_neighbors:
            if F[n, z, neighbor_idx] > 0:  # Only consider non-zero connections
                f_weight = F[n, z, neighbor_idx]
                neighbor_x = embeddings[n, neighbor_idx, :]  # Neighbor embedding
                
                # Current projected values
                proj_x = w * x  # Element-wise multiplication
                proj_neighbor = W[neighbor_idx] * neighbor_x  # Element-wise multiplication
                
                # Difference between projections
                diff = proj_x - proj_neighbor
                
                # Gradient contribution from this neighbor
                gradient += f_weight * (diff * x)  # Element-wise multiplication with x
        
        # Part 2: Gradient from H (inter-subject relationships)
        # Find t_H closest neighbors according to H(z, n, :)
        h_neighbors = np.argsort(-H[z, n, :])[:t_H]  # Get indices of highest H values
        
        for neighbor_idx in h_neighbors:
            if H[z, n, neighbor_idx] > 0:  # Only consider non-zero connections
                h_weight = H[z, n, neighbor_idx]
                neighbor_x = embeddings[neighbor_idx, z, :]  # Neighbor embedding
                
                # Current projected values
                proj_x = w * x  # Element-wise multiplication
                proj_neighbor = w * neighbor_x  # Same projection weights for same position
                
                # Difference between projections
                diff = proj_x - proj_neighbor
                
                # Gradient contribution from this neighbor
                gradient += h_weight * (diff * x)  # Element-wise multiplication with x
    
    # Update weights using gradient descent
    w_new = w - lr_w * gradient
    
    # Normalize the weights to prevent explosion/vanishing
    w_norm = np.linalg.norm(w_new)
    if w_norm > 0:
        w_new = w_new / w_norm
    
    return w_new

def update_W(embeddings: np.ndarray, W: np.ndarray, a: np.ndarray, F: np.ndarray, H: np.ndarray, 
           t_F: int, t_H: int, lr_w: float = 0.01) -> np.ndarray:
    """
    Update the projection matrix W based on current state.
    
    Args:
        embeddings (np.ndarray): Input embeddings array of shape (N, Z, K)
        W (np.ndarray): Current projection matrix of shape (Z, K)
        a (np.ndarray): Current coefficients vector of shape (K,)
        F (np.ndarray): Intra embedding neighbor matrix of shape (N, Z, Z)
        H (np.ndarray): Inter-subject neighbor matrix of shape (Z, N, N)
        t_F (int): Number of F neighbors to consider
        t_H (int): Number of H neighbors to consider
        lr_w (float): Learning rate for W gradient descent
    
    Returns:
        np.ndarray: Updated projection matrix W of shape (Z, K)
    """
    N, Z, K = embeddings.shape
    new_W = np.zeros_like(W)
    
    # Update each row of W separately
    for z in range(Z):
        new_W[z] = update_projection_at_position(embeddings, z, W, a, F, H, t_F, t_H, lr_w)
    
    return new_W

def update_weights_at_embedding(embeddings: np.ndarray, dimension: int, W: np.ndarray, 
                             a: np.ndarray, F: np.ndarray, H: np.ndarray, t_F: int, t_H: int) -> float:
    """
    Update coefficient a at a specific embedding dimension using weighted sum across subjects and positions.
    
    Args:
        embeddings (np.ndarray): Embeddings of shape (N, Z, K)
        dimension (int): Dimension in K to update
        W (np.ndarray): Current projection matrix of shape (Z, K)
        a (np.ndarray): Current coefficients vector of shape (K,)
        F (np.ndarray): Intra embedding neighbor matrix of shape (N, Z, Z)
        H (np.ndarray): Inter-subject neighbor matrix of shape (Z, N, N)
        t_F (int): Number of F neighbors to consider
        t_H (int): Number of H neighbors to consider
    
    Returns:
        float: Updated coefficient for dimension k
    """
    N, Z, K = embeddings.shape
    k = dimension
    
    # Initialize numerator and denominator for weighted contribution
    numerator = 0.0
    denominator = 0.0
    
    # Iterate through all subjects and embedding positions
    for n in range(N):
        for z in range(Z):
            # Get current embedding value at dimension k
            x_k = embeddings[n, z, k]
            
            # Part 1: Contribution from F (intra-embedding relationships)
            # Find t_F closest neighbors according to F(n, z, :)
            f_neighbors = np.argsort(-F[n, z, :])[:t_F]  # Get indices of highest F values
            
            for neighbor_idx in f_neighbors:
                if F[n, z, neighbor_idx] > 0:  # Only consider non-zero connections
                    f_weight = F[n, z, neighbor_idx]
                    neighbor_x_k = embeddings[n, neighbor_idx, k]  # Neighbor embedding at dimension k
                    
                    # Weighted contribution to the coefficient
                    diff_k = x_k - neighbor_x_k
                    numerator += f_weight * (W[z, k] * diff_k)**2
                    denominator += f_weight * (diff_k)**2
            
            # Part 2: Contribution from H (inter-subject relationships)
            # Find t_H closest neighbors according to H(z, n, :)
            h_neighbors = np.argsort(-H[z, n, :])[:t_H]  # Get indices of highest H values
            
            for neighbor_idx in h_neighbors:
                if H[z, n, neighbor_idx] > 0:  # Only consider non-zero connections
                    h_weight = H[z, n, neighbor_idx]
                    neighbor_x_k = embeddings[neighbor_idx, z, k]  # Neighbor embedding at dimension k
                    
                    # Weighted contribution to the coefficient
                    diff_k = x_k - neighbor_x_k
                    numerator += h_weight * (W[z, k] * diff_k)**2
                    denominator += h_weight * (diff_k)**2
    
    # Compute new coefficient
    if denominator > 0:
        a_new = numerator / denominator
    else:
        a_new = a[k]  # Keep the same if denominator is zero
    
    return a_new

def update_a(embeddings: np.ndarray, W: np.ndarray, a: np.ndarray, F: np.ndarray, H: np.ndarray, 
           t_F: int, t_H: int, lr_a: float = 0.01) -> np.ndarray:
    """
    Update the coefficients vector a based on current state.
    
    Args:
        embeddings (np.ndarray): Input embeddings array of shape (N, Z, K)
        W (np.ndarray): Current projection matrix of shape (Z, K)
        a (np.ndarray): Current coefficients vector of shape (K,)
        F (np.ndarray): Intra embedding neighbor matrix of shape (N, Z, Z)
        H (np.ndarray): Inter-subject neighbor matrix of shape (Z, N, N)
        t_F (int): Number of F neighbors to consider
        t_H (int): Number of H neighbors to consider
        lr_a (float): Learning rate for a gradient descent
    
    Returns:
        np.ndarray: Updated coefficients vector a of shape (K,)
    """
    N, Z, K = embeddings.shape
    new_a = np.zeros_like(a)
    
    # Update each element of a separately
    for k in range(K):
        # Get the optimal value for a[k]
        a_optimal = update_weights_at_embedding(embeddings, k, W, a, F, H, t_F, t_H)
        
        # Apply gradient descent with learning rate
        new_a[k] = a[k] + lr_a * (a_optimal - a[k])
    
    # Normalize a to sum to 1
    new_a = new_a / np.sum(new_a)
    
    return new_a

def compute_projection(embeddings: np.ndarray, T: int, q: float, 
                      t_F: int, t_H: int, sigma: float = 0.01,
                      lr_w: float = 0.01, lr_a: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the projection function W and coefficients a from subject embeddings through iteration.
    
    Args:
        embeddings (np.ndarray): Input embeddings array of shape (N, Z, K)
            where N is number of subjects
            Z is the number of embeddings per subject
            K is the dimension of each embedding
        T (int): Number of iterations
        q (float): Tuning parameter for future use
        t_F (int): Number of neighbors for F computation
        t_H (int): Number of neighbors for H computation
        sigma (float): Gaussian kernel bandwidth parameter. Defaults to 0.01
        lr_w (float): Learning rate for W gradient descent. Defaults to 0.01
        lr_a (float): Learning rate for a gradient descent. Defaults to 0.01
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - W: Final projection matrix of shape (Z, K)
            - a: Final coefficients vector of shape (K,)
    """
    # Get dimensions from input
    N, Z, K = embeddings.shape
    
    # Initialize matrices
    W = np.ones((Z, K))
    W = W / np.linalg.norm(W, axis=1, keepdims=True)  # Normalize each row
    a = np.full(K, fill_value=1/K)
    F = compute_F(embeddings, W, t_F, sigma)
    H = compute_H(embeddings, W, t_H, sigma)
    
    # Iterative optimization
    for t in range(T):
        # Update W and a alternately
        W = update_W(embeddings, W, a, F, H, t_F, t_H, lr_w)
        a = update_a(embeddings, W, a, F, H, t_F, t_H, lr_a)

        # Update F and H based on new W
        F = compute_F(embeddings, W, t_F, sigma)
        H = compute_H(embeddings, W, t_H, sigma)
    
    return W, a
