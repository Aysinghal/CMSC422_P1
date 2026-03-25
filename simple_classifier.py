import numpy as np

class MostFrequentClassClassifier:
    def __init__(self):
        self.prediction = 0

    def train(self, X, y):
        """
        TODO: Find the most frequent label in y and store it in self.prediction.
        """
        num_pos = np.sum(y == 1.0)
        num_neg = np.sum(y == -1.0)
        if num_pos >= num_neg:
            self.prediction = 1.0
        else:
            self.prediction = -1.0

    def predict(self, X):
        """
        TODO: Return a vector of predictions, all equal to self.prediction.
        """
        return np.full(X.shape[0], self.prediction)