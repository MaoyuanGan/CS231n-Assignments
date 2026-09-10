from builtins import range
from builtins import object
import os
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """A two-layer fully-connected neural network: affine - relu - affine - softmax."""

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        self.params = {}
        self.reg = reg

        self.params["W1"] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params["b1"] = np.zeros(hidden_dim)
        self.params["W2"] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params["b2"] = np.zeros(num_classes)

    def loss(self, X, y=None):
        scores = None
        # ----- forward -----
        a1, fc_relu_cache = affine_relu_forward(X, self.params["W1"], self.params["b1"])
        scores, fc_cache = affine_forward(a1, self.params["W2"], self.params["b2"])

        # test mode
        if y is None:
            return scores

        # ----- backward -----
        loss, grads = 0, {}
        loss, dscores = softmax_loss(scores, y)
        # L2 正则必须带 0.5 因子
        loss += 0.5 * self.reg * (np.sum(self.params["W1"] ** 2)
                                  + np.sum(self.params["W2"] ** 2))

        da1, dW2, db2 = affine_backward(dscores, fc_cache)
        _, dW1, db1 = affine_relu_backward(da1, fc_relu_cache)

        grads["W1"] = dW1 + self.reg * self.params["W1"]
        grads["b1"] = db1
        grads["W2"] = dW2 + self.reg * self.params["W2"]
        grads["b2"] = db2

        return loss, grads

    def save(self, fname):
        """Save model parameters."""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        params = self.params
        np.save(fpath, params)
        print(fname, "saved.")

    def load(self, fname):
        """Load model parameters."""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        if not os.path.exists(fpath):
            print(fname, "not available.")
            return False
        else:
            params = np.load(fpath, allow_pickle=True).item()
            self.params = params
            print(fname, "loaded.")
            return True


class FullyConnectedNet(object):
    """{affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax"""

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout_keep_ratio=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        dims = [input_dim] + list(hidden_dims) + [num_classes]
        for i in range(self.num_layers):
            self.params["W%d" % (i + 1)] = weight_scale * np.random.randn(dims[i], dims[i + 1])
            self.params["b%d" % (i + 1)] = np.zeros(dims[i + 1])

        # gamma / beta 只存在于 L-1 个隐藏层的 norm 之后
        if self.normalization in ("batchnorm", "layernorm"):
            for i in range(self.num_layers - 1):
                self.params["gamma%d" % (i + 1)] = np.ones(hidden_dims[i])
                self.params["beta%d" % (i + 1)] = np.zeros(hidden_dims[i])

        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode

        scores = None
        L = self.num_layers
        caches = []

        # ----- forward：前 L-1 层为 affine -> [norm] -> relu -> [dropout] -----
        out = X
        for i in range(1, L):
            out, fc_cache = affine_forward(out, self.params["W%d" % i], self.params["b%d" % i])
            norm_cache = None
            if self.normalization == "batchnorm":
                out, norm_cache = batchnorm_forward(
                    out, self.params["gamma%d" % i], self.params["beta%d" % i],
                    self.bn_params[i - 1])
            elif self.normalization == "layernorm":
                out, norm_cache = layernorm_forward(
                    out, self.params["gamma%d" % i], self.params["beta%d" % i],
                    self.bn_params[i - 1])
            out, relu_cache = relu_forward(out)
            drop_cache = None
            if self.use_dropout:
                out, drop_cache = dropout_forward(out, self.dropout_param)
            caches.append((fc_cache, norm_cache, relu_cache, drop_cache))

        # 最后一层 affine -> softmax
        scores, final_cache = affine_forward(
            out, self.params["W%d" % L], self.params["b%d" % L])

        # test mode
        if mode == "test":
            return scores

        # ----- backward -----
        loss, grads = 0.0, {}
        loss, dscores = softmax_loss(scores, y)
        for i in range(1, L + 1):
            loss += 0.5 * self.reg * np.sum(self.params["W%d" % i] ** 2)

        # 最后一层
        dout, dW, db = affine_backward(dscores, final_cache)
        grads["W%d" % L] = dW + self.reg * self.params["W%d" % L]
        grads["b%d" % L] = db

        # 倒序回传 L-1 个 block（gamma/beta 不做正则）
        for i in range(L - 1, 0, -1):
            fc_cache, norm_cache, relu_cache, drop_cache = caches[i - 1]
            if self.use_dropout:
                dout = dropout_backward(dout, drop_cache)
            dout = relu_backward(dout, relu_cache)
            if norm_cache is not None:
                if self.normalization == "batchnorm":
                    dout, dgamma, dbeta = batchnorm_backward(dout, norm_cache)
                else:  # layernorm
                    dout, dgamma, dbeta = layernorm_backward(dout, norm_cache)
                grads["gamma%d" % i] = dgamma
                grads["beta%d" % i] = dbeta
            dout, dW, db = affine_backward(dout, fc_cache)
            grads["W%d" % i] = dW + self.reg * self.params["W%d" % i]
            grads["b%d" % i] = db

        return loss, grads

    def save(self, fname):
        """Save model parameters."""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        params = self.params
        np.save(fpath, params)
        print(fname, "saved.")

    def load(self, fname):
        """Load model parameters."""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        if not os.path.exists(fpath):
            print(fname, "not available.")
            return False
        else:
            params = np.load(fpath, allow_pickle=True).item()
            self.params = params
            print(fname, "loaded.")
            return True
