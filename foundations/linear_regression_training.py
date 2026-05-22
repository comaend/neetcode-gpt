import numpy as np
from numpy.typing import NDArray

class Solution:
    # Set the learning rate as provided
    learning_rate = 0.01

    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # Note: (model_prediction - ground_truth) provides the correct gradient direction 
        # when performing a standard subtraction update (weights -= lr * grad)
        return (2 / N) * np.dot(model_prediction - ground_truth, X[:, desired_weight])

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.dot(X, weights)

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        
        # Copy weights to avoid mutating the initial input array directly
        weights = np.array(initial_weights, dtype=np.float64)
        N = len(X)
        num_weights = len(weights)
        
        for _ in range(num_iterations):
            # 1. Compute predictions
            predictions = self.get_model_prediction(X, weights)
            
            # Create a temporary array to hold updates so we don't mix old and new weights mid-iteration
            gradients = np.zeros(num_weights)
            
            # 2. Compute gradients for each weight
            for j in range(num_weights):
                gradients[j] = self.get_derivative(predictions, Y, N, X, j)
                
            # 3. Update all weights simultaneously
            weights -= self.learning_rate * gradients
            
        # Return final weights rounded to 5 decimal places
        return np.round(weights, 5)