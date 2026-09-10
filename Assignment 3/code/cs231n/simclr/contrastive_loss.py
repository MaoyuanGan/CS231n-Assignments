import torch
import numpy as np


def sim(z_i, z_j):
    """Normalized dot product between two vectors.

    Inputs:
    - z_i: 1xD tensor.
    - z_j: 1xD tensor.

    Returns:
    - A scalar value that is the normalized dot product between z_i and z_j.
    """
    norm_dot_product = torch.dot(z_i, z_j) / (
        torch.linalg.norm(z_i) * torch.linalg.norm(z_j)
    )
    return norm_dot_product


def simclr_loss_naive(out_left, out_right, tau):
    """Compute the contrastive loss L over a batch (naive loop version).

    Input:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch.
    The same row in out_left and out_right form a positive pair.
    In other words, (out_left[k], out_right[k]) form a positive pair for all k=0...N-1.
    - tau: scalar value, temperature parameter that determines how fast the exponential increases.

    Returns:
    - A scalar value; the total loss across all positive pairs in the batch.
    """
    N = out_left.shape[0]  # total number of training examples

    # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]

    total_loss = 0
    for k in range(N):  # loop through each positive pair (k, k+N)
        z_k, z_k_N = out[k], out[k + N]

        # ----- l(k, k+N) -----
        # 分子：正样本对的相似度
        numerator = torch.exp(sim(z_k, z_k_N) / tau)
        # 分母：除自身外的所有 2N-1 个样本（包含正样本）
        denominator = torch.zeros((), dtype=out.dtype, device=out.device)
        for i in range(2 * N):
            if i != k:
                denominator = denominator + torch.exp(sim(z_k, out[i]) / tau)
        total_loss = total_loss - torch.log(numerator / denominator)

        # ----- l(k+N, k) -----
        numerator = torch.exp(sim(z_k_N, z_k) / tau)
        denominator = torch.zeros((), dtype=out.dtype, device=out.device)
        for i in range(2 * N):
            if i != k + N:
                denominator = denominator + torch.exp(sim(z_k_N, out[i]) / tau)
        total_loss = total_loss - torch.log(numerator / denominator)

    # In the end, we need to divide the total loss by 2N, the number of samples in the batch.
    total_loss = total_loss / (2 * N)
    return total_loss


def sim_positive_pairs(out_left, out_right):
    """Normalized dot product between positive pairs.

    Inputs:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch.
    The same row in out_left and out_right form a positive pair.

    Returns:
    - A Nx1 tensor; each row k is the normalized dot product between out_left[k] and out_right[k].
    """
    pos_pairs = torch.sum(out_left * out_right, dim=1) / (
        torch.linalg.norm(out_left, dim=1) * torch.linalg.norm(out_right, dim=1)
    )                                   # [N]
    return pos_pairs.view(-1, 1)        # [N, 1]


def compute_sim_matrix(out):
    """Compute a 2N x 2N matrix of normalized dot products between all pairs
    of augmented examples in a batch.

    Inputs:
    - out: 2N x D tensor; each row is the z-vector (output of projection head) of a single
      augmented example. There are a total of 2N augmented examples in the batch.

    Returns:
    - sim_matrix: 2N x 2N tensor; each element i, j in the matrix is the normalized dot
      product between out[i] and out[j].
    """
    out_norm = out / torch.linalg.norm(out, dim=1, keepdim=True)  # [2N, D]
    sim_matrix = torch.mm(out_norm, out_norm.t())                 # [2N, 2N]
    return sim_matrix


def simclr_loss_vectorized(out_left, out_right, tau, device='cuda'):
    """Compute the contrastive loss L over a batch (vectorized version). No loops are allowed.
    Inputs and output are the same as in simclr_loss_naive.
    """
    N = out_left.shape[0]

    # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]

    # Compute similarity matrix between all pairs of augmented examples in the batch.
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]

    ###########################################################################
    # Step 1: Use sim_matrix to compute the denominator value for all augmented samples.
    ###########################################################################
    exponential = torch.exp(sim_matrix / tau)  # [2N, 2N]

    # This binary mask zeros out terms where k=i.
    mask = (torch.ones_like(exponential, device=device)
            - torch.eye(2 * N, device=device)).to(device).bool()

    # We apply the binary mask.
    exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]

    # Compute the denominator values for all augmented samples: [2N, 1]
    denom = torch.sum(exponential, dim=1, keepdim=True)

    ###########################################################################
    # Step 2: Compute similarity between positive pairs.
    ###########################################################################
    pos_sim = sim_positive_pairs(out_left, out_right)  # [N, 1]

    ###########################################################################
    # Step 3: Compute the numerator value for all augmented samples.
    # 每个正对贡献两项：l(k, k+N) 和 l(k+N, k)，因此复制拼接成 [2N, 1]
    ###########################################################################
    numerator = torch.exp(torch.cat([pos_sim, pos_sim], dim=0) / tau)  # [2N, 1]

    ###########################################################################
    # Step 4: Compute the total loss.
    ###########################################################################
    loss = torch.sum(-torch.log(numerator / denom)) / (2 * N)
    return loss


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
