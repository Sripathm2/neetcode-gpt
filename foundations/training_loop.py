import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        w = np.zeros(len(X[0]))
        b = 0
        n = len(X)
        for i in range(epochs):
            y_hat = X @ w + b
            error = (y_hat - y)
            MSE = (1/n) * np.sum(error*error)
            dl_dw = (2/n) * X.T @ error
            dl_db = (2/n) * np.sum(error)
            w = w - lr*dl_dw
            b = b - lr*dl_db
        return (np.round(w, 5), round(b, 5))
