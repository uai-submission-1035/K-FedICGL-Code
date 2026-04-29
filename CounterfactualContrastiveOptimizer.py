import torch.nn.functional as F


class CounterfactualContrastiveOptimizer(nn.Module):
    """
    Client-side Counterfactual Contrastive Learning (Section 3.4).
    Implements the intervention-inspired augmentation (Eq. 8) by replacing
    spurious transmission paths with Gaussian noise, forcing the predictor to
    rely on the global invariant skeleton.
    """

    def __init__(self, tau_nce=0.5):
        super().__init__()
        self.tau_nce = tau_nce

    def forward_intervention(self, gnn_encoder, X, A_local, A_inv, noise_std=0.1):
        """
        Executes the do-intervention on spurious edges (Eq. 8).
        """
        # Step 1: Decouple spurious local edges A_spur = max(0, A_local - A_inv)
        A_spur = F.relu(A_local - A_inv)

        # Step 2: Generate Gaussian random noise epsilon ~ N(0, sigma^2 I)
        epsilon = torch.randn_like(X) * noise_std

        # Step 3: Counterfactual Representation Generation (Eq. 8)
        # NOTE: Act as an intervention-inspired augmentation rather than strict SCM counterfactual
        # Anchor: Pure causal paths driven by invariant skeleton
        Z_inv = gnn_encoder(X, A_inv)

        # Spurious path driven by noise
        Z_noise = gnn_encoder(epsilon, A_spur)

        # Counterfactual positive sample
        Z_cf = Z_inv + Z_noise

        return Z_inv, Z_cf

    def compute_infonce_loss(self, Z_inv, Z_cf):
        """
        Computes the Causal Graph Contrastive Loss (Eq. 9).
        Treats the invariant representation as the anchor and the counterfactual
        representation as the positive sample.
        """
        # Normalize representations for cosine similarity
        Z_inv_norm = F.normalize(Z_inv, p=2, dim=-1)
        Z_cf_norm = F.normalize(Z_cf, p=2, dim=-1)

        # Compute cosine similarity matrix [N_k, N_k]
        # Diagonal elements are positive pairs, off-diagonal are negative pairs
        similarity_matrix = torch.matmul(Z_inv_norm, Z_cf_norm.transpose(-2, -1)) / self.tau_nce

        # Create labels (diagonal indices: 0, 1, 2, ..., N_k-1)
        N_k = Z_inv.shape[0]
        labels = torch.arange(N_k, device=Z_inv.device)

        # InfoNCE loss (CrossEntropy treats rows as predictions and diagonal as target classes)
        L_cgc = F.cross_entropy(similarity_matrix, labels)

        return L_cgc