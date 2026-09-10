from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. (已完成)
    ###########################################################################
    N = x.shape[0]
    out = x.reshape(N, -1).dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass. (已完成)
    ###########################################################################
    N = x.shape[0]
    x_flat = x.reshape(N, -1)
    dx = (dout.dot(w.T)).reshape(x.shape)
    dw = x_flat.T.dot(dout)
    db = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass. (已完成)
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass. (已完成)
    ###########################################################################
    dx = dout * (x > 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
    ###########################################################################
    # TODO: Implement the softmax loss. (已完成, 数值稳定版本)
    ###########################################################################
    N = x.shape[0]
    shifted = x - np.max(x, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    probs = np.exp(log_probs)
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var  = momentum * running_var  + (1 - momentum) * sample_var

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var: Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None

    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm. (已完成)
        #######################################################################
        sample_mean = np.mean(x, axis=0)                      # (D,)
        sample_var = np.var(x, axis=0)                        # (D,)
        inv_std = 1.0 / np.sqrt(sample_var + eps)             # (D,)
        x_centered = x - sample_mean                          # (N, D)
        x_hat = x_centered * inv_std                          # (N, D)
        out = gamma * x_hat + beta

        # 更新滑动平均
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        cache = (x_hat, gamma, inv_std, x_centered)
        #######################################################################
        #                       END OF YOUR CODE                              #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. (已完成)
        #######################################################################
        inv_std = 1.0 / np.sqrt(running_var + eps)
        x_hat = (x - running_mean) * inv_std
        out = gamma * x_hat + beta
        cache = None
        #######################################################################
        #                       END OF YOUR CODE                              #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. (已完成)
    ###########################################################################
    x_hat, gamma, inv_std, x_centered = cache
    N = dout.shape[0]

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)

    # 沿计算图逐节点反传
    dxhat = dout * gamma                                  # dL/dx_hat
    dx_centered_1 = dxhat * inv_std                       # 经 x_hat = x_centered * inv_std
    dinv_std = np.sum(dxhat * x_centered, axis=0)         # (D,)
    dvar = -0.5 * dinv_std * inv_std ** 3                 # inv_std = (var+eps)^(-1/2)
    # dmean = -sum(dx_centered) + dvar * mean(-2 * x_centered)；后者因 x_centered 零均值而为 0
    dmean = -np.sum(dx_centered_1, axis=0) \
            + dvar * np.mean(-2.0 * x_centered, axis=0)

    dx = dx_centered_1 + (2.0 / N) * dvar * x_centered + dmean / N
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. (已完成, 化简公式)
    ###########################################################################
    x_hat, gamma, inv_std, _ = cache
    N = dout.shape[0]

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)

    dx = (gamma * inv_std / N) * (
        N * dout
        - np.sum(dout, axis=0)
        - x_hat * np.sum(dout * x_hat, axis=0)
    )
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per
    data-point, before being scaled by gamma and beta parameters identical to
    that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train
    and test-time for layer normalization are identical, and we do not need to
    keep track of running averages of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm. (已完成)
    ###########################################################################
    mean = np.mean(x, axis=1, keepdims=True)              # (N, 1)
    var = np.var(x, axis=1, keepdims=True)                # (N, 1)
    inv_std = 1.0 / np.sqrt(var + eps)                    # (N, 1)
    x_hat = (x - mean) * inv_std                          # (N, D)
    out = gamma * x_hat + beta
    cache = (x_hat, gamma, inv_std)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm. (已完成)
    ###########################################################################
    x_hat, gamma, inv_std = cache
    N, D = dout.shape

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)

    # 与 BN 的化简公式同构：归一化轴换成 axis=1，N 换成 D
    dxhat = dout * gamma
    dx = (inv_std / D) * (
        D * dxhat
        - np.sum(dxhat, axis=1, keepdims=True)
        - x_hat * np.sum(dxhat * x_hat, axis=1, keepdims=True)
    )
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout. (已完成)
        #######################################################################
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout. (已完成)
        #######################################################################
        out = x
        mask = None
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout (已完成)
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where
        H' = 1 + (H + 2 * pad - HH) / stride
        W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass. (已完成)
    ###########################################################################
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param["stride"]
    pad = conv_param["pad"]

    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")

    out = np.zeros((N, F, H_out, W_out), dtype=x.dtype)
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    hs, ws = i * stride, j * stride
                    window = x_pad[n, :, hs:hs + HH, ws:ws + WW]
                    out[n, f, i, j] = np.sum(window * w[f]) + b[f]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass. (已完成)
    ###########################################################################
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param["stride"]
    pad = conv_param["pad"]
    _, _, H_out, W_out = dout.shape

    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.sum(dout, axis=(0, 2, 3))

    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    hs, ws = i * stride, j * stride
                    dx_pad[n, :, hs:hs + HH, ws:ws + WW] += w[f] * dout[n, f, i, j]
                    dw[f] += x_pad[n, :, hs:hs + HH, ws:ws + WW] * dout[n, f, i, j]

    dx = dx_pad[:, :, pad:pad + H, pad:pad + W]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where
        H' = 1 + (H - pool_height) / stride
        W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass (已完成)
    ###########################################################################
    N, C, H, W = x.shape
    ph = pool_param["pool_height"]
    pw = pool_param["pool_width"]
    stride = pool_param["stride"]

    H_out = 1 + (H - ph) // stride
    W_out = 1 + (W - pw) // stride
    out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    hs, ws = i * stride, j * stride
                    out[n, c, i, j] = np.max(x[n, c, hs:hs + ph, ws:ws + pw])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass (已完成)
    ###########################################################################
    x, pool_param = cache
    N, C, H, W = x.shape
    ph = pool_param["pool_height"]
    pw = pool_param["pool_width"]
    stride = pool_param["stride"]
    _, _, H_out, W_out = dout.shape

    dx = np.zeros_like(x)
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    hs, ws = i * stride, j * stride
                    window = x[n, c, hs:hs + ph, ws:ws + pw]
                    mask = (window == np.max(window))
                    dx[n, c, hs:hs + ph, ws:ws + pw] += mask * dout[n, c, i, j]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var: Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization. (已完成)
    ###########################################################################
    N, C, H, W = x.shape
    # (N, C, H, W) -> (N*H*W, C)，对每个通道做普通 BN
    x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)
    out_flat, cache = batchnorm_forward(x_flat, gamma, beta, bn_param)
    out = out_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization. (已完成)
    ###########################################################################
    N, C, H, W = dout.shape
    dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    dx_flat, dgamma, dbeta = batchnorm_backward(dout_flat, cache)
    dx = dx_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.

    In contrast to layer normalization, group normalization splits each entry
    in the data into G contiguous pieces, which it then normalizes independently.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer number of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization. (已完成)
    ###########################################################################
    N, C, H, W = x.shape
    D = (C // G) * H * W                     # 每组内的元素个数

    xg = x.reshape(N, G, D)
    mean = np.mean(xg, axis=2, keepdims=True)            # (N, G, 1)
    var = np.var(xg, axis=2, keepdims=True)              # (N, G, 1)
    inv_std = 1.0 / np.sqrt(var + eps)                   # (N, G, 1)
    xhat_g = (xg - mean) * inv_std                       # (N, G, D)
    x_hat = xhat_g.reshape(N, C, H, W)

    out = gamma * x_hat + beta
    cache = (x_hat, gamma, inv_std, G)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization. (已完成)
    ###########################################################################
    x_hat, gamma, inv_std, G = cache
    N, C, H, W = dout.shape
    D = (C // G) * H * W

    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)              # (1, C, 1, 1)
    dgamma = np.sum(dout * x_hat, axis=(0, 2, 3), keepdims=True)     # (1, C, 1, 1)

    dxhat_g = (dout * gamma).reshape(N, G, D)
    xhat_g = x_hat.reshape(N, G, D)
    dx_g = (inv_std / D) * (
        D * dxhat_g
        - np.sum(dxhat_g, axis=2, keepdims=True)
        - xhat_g * np.sum(dxhat_g * xhat_g, axis=2, keepdims=True)
    )
    dx = dx_g.reshape(N, C, H, W)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
