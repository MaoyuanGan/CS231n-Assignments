import numpy as np


def sgd(w, dw, config=None):
    """Performs vanilla stochastic gradient descent."""
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """Performs stochastic gradient descent with momentum."""
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    v = config["momentum"] * v - config["learning_rate"] * dw
    next_w = w + v

    config["velocity"] = v
    return next_w, config


def rmsprop(w, dw, config=None):
    """Uses the RMSProp update rule."""
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    next_w = None
    cache = config["decay_rate"] * config["cache"] \
        + (1 - config["decay_rate"]) * (dw * dw)
    next_w = w - config["learning_rate"] * dw / (np.sqrt(cache) + config["epsilon"])

    config["cache"] = cache
    return next_w, config


def adam(w, dw, config=None):
    """Uses the Adam update rule.

    NOTE: In order to match the reference output, t is modified _before_
    using it in any calculations.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None
    t = config["t"] + 1                                    # 先更新 t
    m = config["beta1"] * config["m"] + (1 - config["beta1"]) * dw
    v = config["beta2"] * config["v"] + (1 - config["beta2"]) * (dw * dw)
    m_hat = m / (1 - config["beta1"] ** t)                 # 偏置修正
    v_hat = v / (1 - config["beta2"] ** t)
    next_w = w - config["learning_rate"] * m_hat / (np.sqrt(v_hat) + config["epsilon"])

    config["m"] = m
    config["v"] = v
    config["t"] = t
    return next_w, config
