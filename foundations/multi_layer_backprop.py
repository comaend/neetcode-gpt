import numpy as np
from typing import List

class Solution:
    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]], b1: List[float],
        W2: List[List[float]], b2: List[float],
        y_true: List[float]
    ) -> dict:
        # Convert inputs to NumPy arrays
        x_arr = np.array(x, dtype=np.float64)
        W1_arr = np.array(W1, dtype=np.float64)
        b1_arr = np.array(b1, dtype=np.float64)
        W2_arr = np.array(W2, dtype=np.float64)
        b2_arr = np.array(b2, dtype=np.float64)
        y_true_arr = np.array(y_true, dtype=np.float64)
        
        # 1. Forward Pass
        z1 = np.dot(W1_arr, x_arr) + b1_arr
        a1 = np.maximum(0.0, z1)
        predictions = np.dot(W2_arr, a1) + b2_arr
        
        # Compute Mean Squared Error Loss
        loss = np.mean((predictions - y_true_arr) ** 2)
        
        # 2. Backward Pass
        d_pred = (2.0 / predictions.size) * (predictions - y_true_arr)
        
        # Layer 2 gradients
        dW2 = np.outer(d_pred, a1)
        db2 = d_pred
        
        # Backpropagate to hidden layer
        da1 = np.dot(W2_arr.T, d_pred)
        
        # Gradient through ReLU (strict > 0 removes derivative ambiguity at exactly zero)
        dz1 = da1 * (z1 > 0).astype(np.float64)
        
        # Layer 1 gradients
        dW1 = np.outer(dz1, x_arr)
        db1 = dz1
        
        # 3. Clean up negative zeros (-0.0) by adding 0.0
        loss = loss + 0.0
        dW1 = dW1 + 0.0
        db1 = db1 + 0.0
        dW2 = dW2 + 0.0
        db2 = db2 + 0.0
        
        # Return cleanly rounded python native structures
        return {
            'loss': float(np.round(loss, 4)),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist()
        }