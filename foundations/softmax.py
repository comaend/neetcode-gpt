import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        z = z - np.max(z)  # numerical stability
        exp_z = np.exp(z)
        softmax_vals = exp_z / np.sum(exp_z)
        
        return np.round(softmax_vals, 4)