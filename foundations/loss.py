import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)

        binary_cross_entropy =  0;
        for i in range(len(y_true)):
            binary_cross_entropy += (y_true[i]*np.log(y_pred[i]+ 1e-7)) + ((1-y_true[i])*np.log(1-y_pred[i]))
        binary_cross_entropy /= -1*len(y_pred)
        return round(binary_cross_entropy,4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)

        categoritcal_cross_entropy = 0
        for i in range(len(y_true)):
            for j in range(len(y_true[i])):
                categoritcal_cross_entropy += y_true[i][j]*np.log(y_pred[i][j]+ 1e-7)
        categoritcal_cross_entropy /= -1*len(y_pred)


        return round(categoritcal_cross_entropy,4)
