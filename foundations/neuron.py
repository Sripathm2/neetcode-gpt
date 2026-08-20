import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        z = np.dot(x,w) + b
        if "sigmoid" in activation:
            sigmoid_value  = 1/(1 + np.exp(-1*z))
            return np.round(sigmoid_value, 5)
        elif "relu" in activation:
            return np.round(np.max([0,z]),5)
        else:
            return -1.0
