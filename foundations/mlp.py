import numpy as np
from numpy.typing import NDArray
from typing import List

class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        
        current_layer = np.array(x, dtype=np.float64)
        num_layers = len(weights)
        
        for i in range(num_layers):
            W = np.array(weights[i], dtype=np.float64)
            b = np.array(biases[i], dtype=np.float64)
            
            # Compute the linear transformation: z = xW + b (or Wx + b depending on dimension ordering)
            # Given x is 1D and weights are structured to map inputs to outputs, np.dot handles it cleanly.
            z = np.dot(current_layer, W) + b
            
            # Apply ReLU activation after each hidden layer, but NOT on the final output layer
            if i < num_layers - 1:
                current_layer = np.maximum(0, z)
            else:
                current_layer = z
                
        # Return the final result rounded to 5 decimal places
        return np.round(current_layer, 5)