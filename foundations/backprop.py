import numpy as np
from numpy.typing import NDArray
from typing import Tuple

class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        
        # 1. Forward Pass
        z = np.dot(x, w) + b
        y_hat = 1 / (1 + np.exp(-z))
        
        # 2. Compute Chain Rule Derivatives (Backpropagation)
        dL_dyhat = y_hat - y_true
        dyhat_dz = y_hat * (1 - y_hat)
        
        # Combined gradient with respect to pre-activation z
        dL_dz = dL_dyhat * dyhat_dz
        
        # Gradients with respect to weights and bias
        dL_dw = dL_dz * x
        dL_db = dL_dz
        
        # 3. Round results to 5 decimal places
        dL_dw_rounded = np.round(dL_dw, 5)
        dL_db_rounded = float(np.round(dL_db, 5))
        
        return dL_dw_rounded, dL_db_rounded