import torch
import torch.nn as nn


class LocalCausalGraphGenerator(nn.Module):
    """
    Client-side Knowledge-Driven Causal Graph Construction (Section 3.2).
    Learns a local adjacency matrix constrained by physical road network priors and DAG properties.
    """

    def __init__(self, num_nodes, lambda_prior=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.lambda_prior = lambda_prior

        # Learnable continuous adjacency matrix A^(k)
        # Initialized near zero
        self.A = nn.Parameter(torch.zeros(num_nodes, num_nodes))

    def _compute_dag_constraint(self):
        """
        Computes the trace constraint h(A) = tr(e^(A o A)) - N = 0 to ensure a DAG structure.
        """
        # Hadamard product A o A ensures non-negativity
        A_sq = self.A * self.A
        # Matrix exponential
        E = torch.matrix_exp(A_sq)
        h = torch.trace(E) - self.num_nodes
        return h

    def compute_local_discovery_loss(self, L_rec, K_prior, rho, alpha):
        """
        Computes the Augmented Lagrangian loss for causal discovery (Eq. 4).

        Args:
            L_rec: Data likelihood reconstruction loss (e.g., MSE of feature reconstruction).
            K_prior: Domain knowledge prior matrix [N_k, N_k] (1 for physical connection, 0 otherwise).
            rho: Penalty parameter for Augmented Lagrangian.
            alpha: Lagrange multiplier.
        """
        # Step 1: Prior Penalty Mask M = 1 - K (Eq. 3)
        M = 1.0 - K_prior

        # Frobenius norm of the Hadamard product imposes penalty on spurious edges violating priors
        L_prior = torch.norm(self.A * M, p='fro') ** 2

        # Step 2: DAG Constraint h(A)
        h = self._compute_dag_constraint()

        # Step 3: Total Augmented Lagrangian Loss (Eq. 4)
        total_loss = L_rec + (self.lambda_prior * L_prior) + (0.5 * rho * h ** 2) + (alpha * h)

        return total_loss, h