import numpy as np

class Perceptron:
    def __init__(self, num_epochs=10):
        self.num_epochs = num_epochs
        self.w = None
        self.b = 0

    def train(self, X, y):
        """
        TODO: Implement the Perceptron Update Rule.
        1. Init w and b to zeros (w is a vector and b is a scalar).
        2. Loop epochs.
        3. Loop examples:
           If prediction is wrong:
              w = w + y * x
              b = b + y
        """
        data_points, dimensions = X.shape
        self.w = np.zeros(dimensions)
        self.b = 0.0

        for epoch in range(self.num_epochs):
            for i in range(data_points):
                if y[i] * (np.dot(self.w, X[i]) + self.b) <= 0:
                    self.w += y[i] * X[i]
                    self.b += y[i]

    def predict(self, X):
        """
        TODO: Compute w*x + b. Return +1 or -1.
        """
        original_results = X @ self.w + self.b
        return np.where(original_results >= 0, 1.0, -1.0)