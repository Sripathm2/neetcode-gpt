import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        temp_x = x
        for i in range(len(weights)-1):
            temp_x = temp_x @ weights[i]  + biases[i]
            temp_x = np.maximum(0, temp_x)
        ouput = temp_x @ weights[-1] + biases[-1]
        return np.round(ouput,5)


