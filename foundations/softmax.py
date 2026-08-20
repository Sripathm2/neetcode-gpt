import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max_val = np.max(z)
        exp_value = [ np.exp(x-max_val) for x in z ]
        sum_value = np.sum(exp_value)
        softmax_val = [np.round(x/sum_value,4) for x in exp_value]
        return softmax_val
