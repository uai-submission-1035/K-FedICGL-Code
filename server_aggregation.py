import torch
import torch.nn as nn


class FederatedInvariantAlignment(nn.Module):
    """
    Server-side Invariant Causal Subgraph Alignment (Section 3.3).
    This module maps independent client subgraphs into a Common Node Space
    to compute cross-client consensus and variance.
    """

    def __init__(self, num_global_nodes, tau_c=0.5, gamma=1.0, tau_s=0.5):
        super().__init__()
        self.N = num_global_nodes
        self.tau_c = tau_c  # Threshold for consensus (Eq. 5)
        self.gamma = gamma  # Penalty coefficient for variance (Eq. 6)
        self.tau_s = tau_s  # Threshold for final invariant mask (Eq. 7)

    def forward(self, client_adj_list, client_node_indices):
        """
        Args:
            client_adj_list: List of local adjacency matrices A^(k) from clients.
                             Each shape: [N_k, N_k]
            client_node_indices: List of lists containing global indices for each client's nodes.
                                 e.g., client 0 has global nodes [2, 5, 12, ...]
        Returns:
            A_inv: The globally aligned invariant causal skeleton. Shape: [N, N]
        """
        num_clients = len(client_adj_list)

        # Step 1: Map local subgraphs to the Global Common Node Space
        # We use a tensor to store all padded matrices and a mask to track valid observations
        global_A_stacked = torch.zeros((num_clients, self.N, self.N))
        observation_mask = torch.zeros((num_clients, self.N, self.N))

        for k in range(num_clients):
            local_A = client_adj_list[k]
            idx = client_node_indices[k]

            # Create a meshgrid to map [N_k, N_k] back to [N, N]
            idx_grid_x, idx_grid_y = torch.meshgrid(torch.tensor(idx), torch.tensor(idx), indexing='ij')
            global_A_stacked[k, idx_grid_x, idx_grid_y] = local_A

            # Mark these edges as 'observed' by client k
            observation_mask[k, idx_grid_x, idx_grid_y] = 1.0

        # Step 2: Compute valid average across clients (ignoring unobserved edges)
        # Avoid division by zero by clamping the denominator
        num_observations = observation_mask.sum(dim=0).clamp(min=1e-5)
        A_mean = (global_A_stacked * observation_mask).sum(dim=0) / num_observations

        # Step 3: Compute Structural Consensus Matrix C (Eq. 5)
        # Only clients that actually observed the edge contribute to the consensus
        is_above_threshold = (global_A_stacked > self.tau_c).float() * observation_mask
        C = is_above_threshold.sum(dim=0) / num_observations

        # Step 4: Compute Cross-environment Variance V (Eq. 6 text)
        # Variance is only calculated among clients that observed the specific edge
        squared_diff = ((global_A_stacked - A_mean.unsqueeze(0)) ** 2) * observation_mask
        V = squared_diff.sum(dim=0) / num_observations

        # Step 5: Active Alignment Score S (Eq. 6)
        S = C * torch.exp(-self.gamma * V)

        # Step 6: Extract Global Invariant Skeleton A_inv (Eq. 7)
        M_inv = (S > self.tau_s).float()
        A_inv = M_inv * A_mean

        # Zero out edges that had no observations at all
        A_inv = A_inv * (num_observations > 1e-5).float()

        return A_inv