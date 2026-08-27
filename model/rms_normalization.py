import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list

        mean = np.mean(np.array(x)*np.array(x))
        RMS = np.power(mean+eps, 0.5)
        x_hat = np.array(x)/RMS
        out = gamma*x_hat
        return np.round(out,4)
