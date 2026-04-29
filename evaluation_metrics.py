import numpy as np


def build_pseudo_causal_ground_truth(sensor_distances, distance_threshold=2000.0, k_nearest=5):
    """
    Constructs a pseudo-causal ground truth graph for traffic datasets (METR-LA, PeMSD4).
    Since strict SCM ground truth is unavailable in real-world traffic, we construct a
    plausible physical causal skeleton based on spatial proximity, which serves as the
    reference graph for SHD, FDR, and TPR evaluation.

    Args:
        sensor_distances: A [N, N] numpy array containing physical distances (e.g., in meters).
        distance_threshold: Maximum physical distance to be considered a direct causal link.
        k_nearest: Ensure only the k physically closest nodes can have directed edges
                   (representing actual traffic flow limits).

    Returns:
        A binary adjacency matrix [N, N] representing the physical pseudo-causal graph.
    """
    N = sensor_distances.shape[0]
    A_true = np.zeros((N, N))

    for i in range(N):
        # Find neighbors within the physical distance threshold
        valid_neighbors = np.where(sensor_distances[i] <= distance_threshold)[0]

        # Further constrain by k-nearest to prevent implausible dense hubs
        if len(valid_neighbors) > k_nearest + 1:  # +1 includes self
            # Sort valid neighbors by distance and pick top k
            sorted_idx = np.argsort(sensor_distances[i, valid_neighbors])
            valid_neighbors = valid_neighbors[sorted_idx[:k_nearest + 1]]

        for j in valid_neighbors:
            if i != j:
                # Directed edge assumption: traffic physically flows from upstream to downstream.
                # (In practice, usually symmetric unless direction flow data is parsed)
                A_true[i, j] = 1.0

    return A_true


def evaluate_causal_discovery(A_pred, A_true, threshold=0.1):
    """
    Evaluates the learned causal skeleton against the pseudo-causal ground truth.

    Returns:
        SHD: Structural Hamming Distance (lower is better)
        TPR: True Positive Rate (higher is better)
        FDR: False Discovery Rate (lower is better)
    """
    # Binarize prediction based on continuous edge weights
    A_pred_bin = (A_pred > threshold).astype(int)
    A_true_bin = A_true.astype(int)

    # SHD: Number of edge insertions, deletions, or flips needed to match ground truth
    shd = np.sum(np.abs(A_pred_bin - A_true_bin))

    # TPR: TP / (TP + FN)
    true_positives = np.sum((A_pred_bin == 1) & (A_true_bin == 1))
    actual_positives = np.sum(A_true_bin == 1)
    tpr = true_positives / actual_positives if actual_positives > 0 else 0.0

    # FDR: FP / (FP + TP)
    false_positives = np.sum((A_pred_bin == 1) & (A_true_bin == 0))
    predicted_positives = np.sum(A_pred_bin == 1)
    fdr = false_positives / predicted_positives if predicted_positives > 0 else 0.0

    return {"SHD": shd, "TPR": tpr, "FDR": fdr}