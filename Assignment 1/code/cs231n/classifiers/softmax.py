from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)
        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)
        loss -= logp[y[i]]  # negative log probability is the loss

        #############################################################################
        # TODO:                                                                     #
        # Compute the gradient of the loss function and store it dW.                #
        # Rather that first computing the loss and then computing the derivative,   #
        # it may be simpler to compute the derivative at the same time that the     #
        # loss is being computed. As a result you may need to modify some of the    #
        # code above to compute the gradient.                                       #
        #############################################################################
        # 对 scores 的梯度: dscores_k = p_k - 1(k == y_i)
        dscores = p.copy()                          # shape (C,)
        dscores[y[i]] -= 1.0

        # 链式法则: dW += x_i · dscores^T  (外积)
        dW += X[i].reshape(-1, 1).dot(dscores.reshape(1, -1))   # (D,1)·(1,C) -> (D,C)

    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + 2 * reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_train = X.shape[0]

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################
    # 1) 计算所有样本的 scores，并做数值稳定化（减去每行最大值）
    scores = X.dot(W)                                   # (N, C)
    scores -= np.max(scores, axis=1, keepdims=True)     # (N, C)

    # 2) 计算 softmax 概率矩阵 P
    p = np.exp(scores)
    p /= p.sum(axis=1, keepdims=True)                   # (N, C)

    # 3) 计算交叉熵损失: L = -mean( log p[i, y[i]] ) + reg*||W||^2
    correct_logprobs = -np.log(p[np.arange(num_train), y])   # (N,)
    loss = np.sum(correct_logprobs) / num_train
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # 对 scores 的梯度: dscores = P - onehot(y)
    dscores = p.copy()                                  # (N, C)
    dscores[np.arange(num_train), y] -= 1.0             # 正确类别位置减 1

    # 链式法则: dW = X^T · dscores / N + 2*reg*W
    dW = X.T.dot(dscores)                               # (D,N)·(N,C) -> (D,C)
    dW = dW / num_train + 2 * reg * W

    return loss, dW
