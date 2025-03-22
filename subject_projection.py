import numpy as np
from typing import Tuple
from scipy.spatial.distance import cdist

def compute_subject_G(embeddings: np.ndarray, t_G: int, sigma: float = 0.01) -> np.ndarray:
    """
    Compute the subject similarity matrix G based on t_G nearest neighbors using Gaussian kernel.
    This measures similarity between subjects based on their full embedding sets.
    
    Args:
        embeddings (np.ndarray): Input embeddings array of shape (N, Z, K)
        t_G (int): Number of nearest neighbors to consider
        sigma (float): Gaussian kernel bandwidth parameter. Defaults to 0.01
    
    Returns:
        np.ndarray: Subject similarity matrix G with shape (N, N)
    """
    N, Z, K = embeddings.shape
    G = np.zeros((N, N))
    
    # Flatten embeddings for each subject to get overall representation
    flat_embeddings = embeddings.reshape(N, Z*K)
    
    # Compute pairwise distances between all N subjects
    distances = cdist(flat_embeddings, flat_embeddings)
    
    # Compute Gaussian kernel values
    gaussian_values = np.exp(-distances**2 / (sigma**2))
    
    # For each subject, find its t_G nearest neighbors
    for i in range(N):
        # Get indices of t_G nearest neighbors (excluding self)
        nearest_indices = np.argsort(distances[i])[1:t_G+1]
        # Set G value to gaussian kernel value for nearest neighbors
        G[i, nearest_indices] = gaussian_values[i, nearest_indices]
    
    return G

def update_embedding_weight_for_subject(embeddings: np.ndarray, subject_idx: int, 
                                      b: np.ndarray, G: np.ndarray, t_G: int,
                                      lr_b: float = 0.01) -> np.ndarray:
    """
    Update the embedding weights for a specific subject using gradient descent.
    
    Args:
        embeddings (np.ndarray): Input embeddings array of shape (N, Z, K)
        subject_idx (int): Index of the subject to update
        b (np.ndarray): Current embedding weights matrix of shape (N, Z)
        G (np.ndarray): Subject similarity matrix of shape (N, N)
        t_G (int): Number of G neighbors to consider
        lr_b (float): Learning rate for b gradient descent
    
    Returns:
        np.ndarray: Updated embedding weights for subject of shape (Z,)
    """
    N, Z, K = embeddings.shape
    n = subject_idx
    b_n = b[n].copy()  # Current weights for subject n
    
    # Initialize gradient and quality score for each embedding
    gradient = np.zeros(Z)
    quality_scores = np.zeros(Z)
    
    # Get current subject embeddings
    subject_embeddings = embeddings[n]  # Shape (Z, K)
    
    # Find t_G closest neighbors according to G(n, :)
    g_neighbors = np.argsort(-G[n])[:t_G]  # Get indices of highest G values
    
    # For each embedding position, compute a quality score with more balanced metrics
    for z in range(Z):
        # Get the embedding vector
        emb = subject_embeddings[z]
        
        # 1. Signal strength (L2 norm) - but tempered to avoid extreme differences
        signal_strength = np.log1p(np.linalg.norm(emb))
        
        # 2. Signal-to-noise ratio - compare largest values to average, with dampening
        mean_abs = np.mean(np.abs(emb))
        if mean_abs > 0:
            snr = np.sqrt(np.max(np.abs(emb)) / mean_abs)  # Square root dampens extreme values
        else:
            snr = 1.0
        
        # Create a more balanced quality score with smaller coefficient differences
        quality_scores[z] = signal_strength * 0.7 + snr * 0.3
    
    # Normalize quality scores with softmax instead of direct normalization
    # This creates smoother differences between scores
    quality_scores = np.exp(quality_scores - np.max(quality_scores))
    quality_score_sum = np.sum(quality_scores)
    if quality_score_sum > 0:
        quality_scores = quality_scores / quality_score_sum
    else:
        quality_scores = np.ones(Z) / Z
    
    # Directly integrate quality scores into gradient calculation
    for neighbor_idx in g_neighbors:
        if G[n, neighbor_idx] > 0:  # Only consider non-zero connections
            g_weight = G[n, neighbor_idx]
            neighbor_embeddings = embeddings[neighbor_idx]  # Shape (Z, K)
            neighbor_weights = b[neighbor_idx]  # Shape (Z,)
            
            # For each embedding position
            for z in range(Z):
                # Current weighted embedding for this subject at position z
                weighted_embedding = b_n[z] * subject_embeddings[z]  # Shape (K,)
                
                # Current weighted embedding for neighbor at position z
                weighted_neighbor = neighbor_weights[z] * neighbor_embeddings[z]  # Shape (K,)
                
                # Difference between weighted embeddings
                diff = weighted_embedding - weighted_neighbor  # Shape (K,)
                
                # Gradient contribution from this neighbor
                gradient[z] += g_weight * np.sum(diff * subject_embeddings[z])
    
    # Use a much smaller coefficient (0.5 instead of 5.0) to allow more balanced weights
    combined_gradient = gradient - 0.5 * (b_n - quality_scores)
    
    # Update weights using gradient descent, controlled by learning rate
    b_new = b_n - lr_b * combined_gradient
    
    # Ensure non-negative weights
    b_new = np.maximum(b_new, 0)
    
    # Normalize the weights to sum to 1
    b_sum = np.sum(b_new)
    if b_sum > 0:
        b_new = b_new / b_sum
    else:
        b_new = np.ones(Z) / Z  # Default to uniform if sum is zero
    
    return b_new

def update_b(embeddings: np.ndarray, b: np.ndarray, G: np.ndarray, 
           t_G: int, lr_b: float = 0.01) -> np.ndarray:
    """
    Update the embedding weights matrix b based on current state.
    
    Args:
        embeddings (np.ndarray): Input embeddings array of shape (N, Z, K)
        b (np.ndarray): Current embedding weights matrix of shape (N, Z)
        G (np.ndarray): Subject similarity matrix of shape (N, N)
        t_G (int): Number of G neighbors to consider
        lr_b (float): Learning rate for b gradient descent
    
    Returns:
        np.ndarray: Updated embedding weights matrix b of shape (N, Z)
    """
    N, Z, K = embeddings.shape
    new_b = np.zeros_like(b)
    
    # Update each subject's weights separately
    for n in range(N):
        new_b[n] = update_embedding_weight_for_subject(embeddings, n, b, G, t_G, lr_b)
    
    return new_b

def compute_subject_projection(embeddings: np.ndarray, T: int, 
                           t_G: int, sigma: float = 0.01,
                           lr_b: float = 0.01) -> np.ndarray:
    """
    Compute the subject projection weights to merge Z embeddings into 1 embedding per subject.
    This function learns weights for each subject to combine their multiple embeddings.
    
    Args:
        embeddings (np.ndarray): Input embeddings array of shape (N, Z, K)
            where N is number of subjects
            Z is the number of embeddings per subject
            K is the dimension of each embedding
        T (int): Number of iterations
        t_G (int): Number of neighbors for G computation
        sigma (float): Gaussian kernel bandwidth parameter. Defaults to 0.01
        lr_b (float): Learning rate for embedding weights gradient descent. Defaults to 0.01
    
    Returns:
        np.ndarray: Final embedding weights matrix b of shape (N, Z)
            Each row contains weights to combine Z embeddings into 1 embedding for that subject
    """
    # Get dimensions from input
    N, Z, K = embeddings.shape
    
    # Initialize embedding weights matrix
    b = np.ones((N, Z)) / Z  # Start with uniform weights
    
    # Compute subject similarity matrix
    G = compute_subject_G(embeddings, t_G, sigma)
    
    # If t_G is too small, nudge it up to ensure meaningful learning
    if t_G < 1 and N > 1:
        t_G = min(1, N-1)
    
    # Iterative optimization
    for t in range(T):
        # Update embedding weights
        b = update_b(embeddings, b, G, t_G, lr_b)
        
        # Optional: Update G based on new weighted embeddings
        if t % 5 == 0 and t > 0:  # Update every 5 iterations to save computation
            # Create weighted embeddings
            weighted_embeddings = np.zeros((N, K))
            for n in range(N):
                for z in range(Z):
                    weighted_embeddings[n] += b[n, z] * embeddings[n, z]
            
            # Recompute G based on weighted embeddings
            distances = cdist(weighted_embeddings, weighted_embeddings)
            gaussian_values = np.exp(-distances**2 / (sigma**2))
            
            G = np.zeros((N, N))
            for i in range(N):
                nearest_indices = np.argsort(distances[i])[1:t_G+1]
                G[i, nearest_indices] = gaussian_values[i, nearest_indices]
    
    return b

def merge_subject_embeddings(embeddings: np.ndarray, b: np.ndarray = None) -> np.ndarray:
    """
    Merge multiple embeddings per subject into a single embedding using learned weights.
    
    Args:
        embeddings (np.ndarray): Input embeddings array of shape (N, Z, K)
        b (np.ndarray, optional): Embedding weights matrix of shape (N, Z).
            If None, uses uniform weights. Defaults to None.
    
    Returns:
        np.ndarray: Merged embeddings of shape (N, K)
    """
    N, Z, K = embeddings.shape
    
    # If no weights provided, use uniform weights
    if b is None:
        b = np.ones((N, Z)) / Z
    
    # Initialize merged embeddings
    merged_embeddings = np.zeros((N, K))
    
    # Combine embeddings for each subject
    for n in range(N):
        for z in range(Z):
            merged_embeddings[n] += b[n, z] * embeddings[n, z]
    
    return merged_embeddings 