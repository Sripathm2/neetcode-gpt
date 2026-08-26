import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)
        
        z1 = W1 @ x + b1
        a1 = np.maximum(0, z1)
        y_hat = W2 @ a1 + b2
        error = y_hat-y_true
        loss = np.sum(error*error)

        dl_dy_hat =  2 * (error) / len(y_true)
        dl_b2 = dl_dy_hat
        dl_dw2 = np.outer(dl_dy_hat, a1)
        dl_da1 = dl_dy_hat @ W2
        dl_dz1 = dl_da1 * (z1 > 0)
        dl_dw1 = np.outer(dl_dz1, x)
        dl_b1 = dl_dz1

        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dl_dw1, 4),
            'db1': np.round(dl_b1, 4),
            'dW2': np.round(dl_dw2, 4),
            'db2': np.round(dl_b2, 4)
        }
