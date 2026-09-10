from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Forward pass for an affine (fully-connected) layer."""
    out = x.reshape(x.shape[0], -1).dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Backward pass for an affine layer."""
    x, w, b = cache
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = dout.sum(axis=0)
    return dx, dw, db


def relu_forward(x):
    """Forward pass for a layer of rectified linear units (ReLUs)."""
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Backward pass for a layer of rectified linear units (ReLUs)."""
    dx, x = None, cache
    dx = dout * (x > 0)
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization."""
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        sample_mean = x.mean(axis=0)
        sample_var = x.var(axis=0)
        inv_std = 1.0 / np.sqrt(sample_var + eps)
        x_hat = (x - sample_mean) * inv_std
        out = gamma * x_hat + beta

        # 更新滑动平均（注意：归一化时用的是 minibatch 统计量，不是 running 统计量）
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        cache = (x, x_hat, gamma, sample_mean, sample_var, eps)
    elif mode == "test":
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta
        cache = None
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var
    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization (computational graph version)."""
    dx, dgamma, dbeta = None, None, None
    x, x_hat, gamma, mu, var, eps = cache
    N, D = dout.shape
    inv_std = 1.0 / np.sqrt(var + eps)

    dbeta = dout.sum(axis=0)
    dgamma = np.sum(x_hat * dout, axis=0)

    dxhat = dout * gamma
    divar = np.sum(dxhat * (x - mu), axis=0) * -0.5 * inv_std ** 3
    dmu = np.sum(dxhat * -inv_std, axis=0) + divar * -2.0 * np.mean(x - mu, axis=0)
    dx = dxhat * inv_std + divar * 2.0 * (x - mu) / N + dmu / N
    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Simplified backward pass for batch normalization."""
    dx, dgamma, dbeta = None, None, None
    x, x_hat, gamma, mu, var, eps = cache
    N, D = dout.shape
    inv_std = 1.0 / np.sqrt(var + eps)

    dbeta = dout.sum(axis=0)
    dgamma = np.sum(x_hat * dout, axis=0)

    dxhat = dout * gamma
    dx = (inv_std / N) * (N * dxhat
                          - np.sum(dxhat, axis=0)
                          - x_hat * np.sum(dxhat * x_hat, axis=0))
    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization (normalize each sample)."""
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)

    mu = x.mean(axis=1, keepdims=True)          # (N, 1)
    var = x.var(axis=1, keepdims=True)          # (N, 1)
    inv_std = 1.0 / np.sqrt(var + eps)          # (N, 1)
    x_hat = (x - mu) * inv_std
    out = gamma * x_hat + beta

    cache = (x_hat, gamma, inv_std)
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization."""
    dx, dgamma, dbeta = None, None, None
    x_hat, gamma, inv_std = cache
    D = dout.shape[1]

    dbeta = dout.sum(axis=0)
    dgamma = np.sum(x_hat * dout, axis=0)

    dxhat = dout * gamma
    dx = (inv_std / D) * (D * dxhat
                          - np.sum(dxhat, axis=1, keepdims=True)
                          - x_hat * np.sum(dxhat * x_hat, axis=1, keepdims=True))
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout. p is the KEEP probability."""
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask, out = None, None
    if mode == "train":
        mask = (np.random.rand(*x.shape) < p) / p   # 除以 p，测试期期望不变
        out = x * mask
    elif mode == "test":
        mask = None
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)
    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout."""
    dropout_param, mask = cache
    mode = dropout_param["mode"]
    dx = None
    if mode == "train":
        dx = dout * mask
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer."""
    out = None
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param["stride"]
    pad = conv_param["pad"]

    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    out = np.zeros((N, F, H_out, W_out), dtype=x.dtype)
    for i in range(H_out):
        for j in range(W_out):
            patch = x_pad[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
            out[:, :, i, j] = np.einsum("nchw,fchw->nf", patch, w)
    out += b.reshape(1, F, 1, 1)

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer."""
    dx, dw, db = None, None, None
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param["stride"]
    pad = conv_param["pad"]
    _, _, H_out, W_out = dout.shape

    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = dout.sum(axis=(0, 2, 3))

    for i in range(H_out):
        for j in range(W_out):
            patch = x_pad[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
            dout_ij = dout[:, :, i, j]                                   # (N, F)
            dw += np.einsum("nf,nchw->fchw", dout_ij, patch)
            dx_pad[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW] += \
                np.einsum("nf,fchw->nchw", dout_ij, w)

    dx = dx_pad[:, :, pad:pad + H, pad:pad + W]
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer."""
    out = None
    N, C, H, W = x.shape
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]

    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride

    out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
    for i in range(H_out):
        for j in range(W_out):
            patch = x[:, :, i * stride:i * stride + pool_height,
                      j * stride:j * stride + pool_width]
            out[:, :, i, j] = patch.max(axis=(2, 3))

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer."""
    dx = None
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]
    _, _, H_out, W_out = dout.shape

    dx = np.zeros_like(x)
    for i in range(H_out):
        for j in range(W_out):
            patch = x[:, :, i * stride:i * stride + pool_height,
                      j * stride:j * stride + pool_width]
            max_mask = (patch == patch.max(axis=(2, 3), keepdims=True))
            dx[:, :, i * stride:i * stride + pool_height,
               j * stride:j * stride + pool_width] += \
                max_mask * dout[:, :, i, j][:, :, None, None]
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for spatial batch normalization."""
    out, cache = None, None
    N, C, H, W = x.shape
    # (N, C, H, W) -> (N*H*W, C)：把每个通道当作一个特征
    x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)
    out_flat, cache = batchnorm_forward(x_flat, gamma, beta, bn_param)
    out = out_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Backward pass for spatial batch normalization."""
    dx, dgamma, dbeta = None, None, None
    N, C, H, W = dout.shape
    dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    dx_flat, dgamma, dbeta = batchnorm_backward(dout_flat, cache)
    dx = dx_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Forward pass for spatial group normalization."""
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    N, C, H, W = x.shape

    xg = x.reshape(N, G, -1)                     # (N, G, M), M = (C/G)*H*W
    mu = xg.mean(axis=2, keepdims=True)          # (N, G, 1)
    var = xg.var(axis=2, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    xg_hat = (xg - mu) * inv_std                 # (N, G, M)
    x_hat = xg_hat.reshape(N, C, H, W)
    out = gamma * x_hat + beta                   # gamma/beta: (1, C, 1, 1)

    cache = (xg_hat, inv_std, G)
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Backward pass for spatial group normalization."""
    dx, dgamma, dbeta = None, None, None
    xg_hat, inv_std, G = cache
    N, C, H, W = dout.shape
    M = xg_hat.shape[2]

    x_hat = xg_hat.reshape(N, C, H, W)
    dbeta = dout.sum(axis=(0, 2, 3), keepdims=True)
    dgamma = (dout * x_hat).sum(axis=(0, 2, 3), keepdims=True)

    dxg = (dout * gamma).reshape(N, G, M)
    dxg = (inv_std / M) * (M * dxg
                           - dxg.sum(axis=2, keepdims=True)
                           - xg_hat * (dxg * xg_hat).sum(axis=2, keepdims=True))
    dx = dxg.reshape(N, C, H, W)
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """Computes the loss and gradient for multiclass SVM classification."""
    loss, dx = None, None
    N = x.shape[0]
    correct_scores = x[np.arange(N), y][:, None]         # (N, 1)
    margins = np.maximum(0, x - correct_scores + 1.0)    # (N, C)
    margins[np.arange(N), y] = 0.0
    loss = np.sum(margins) / N

    dx = (margins > 0).astype(x.dtype)
    dx[np.arange(N), y] -= dx.sum(axis=1)
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification."""
    loss, dx = None, None
    N = x.shape[0]
    shifted = x - np.max(x, axis=1, keepdims=True)       # 数值稳定
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    loss = -np.sum(log_probs[np.arange(N), y]) / N

    dx = np.exp(log_probs)
    dx[np.arange(N), y] -= 1.0
    dx /= N
    return loss, dx
