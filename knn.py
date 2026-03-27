import numpy as np

def compute_euclidean_distance(x1, x2):
    """
    TODO: Compute Euclidean distance between x1 and x2.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        """
        TODO: Store the training data.
        """
        self.X_train = X
        self.y_train = y

    def get_k_neighbors_indices(self, x_single):
        """
        TODO: 
        Given a single test example x_single, calculate its distance to every point 
        in self.X_train and return the INDICES of the k nearest neighbors.
        """
        distances = []
        for i in range(len(self.X_train)):
            dist = compute_euclidean_distance(x_single, self.X_train[i])
            distances.append((dist, i))
        distances.sort(key=lambda pair: pair[0])
        k_indices = [pair[1] for pair in distances[:self.k]]
        
        return k_indices


    def predict(self, X):
        """
        TODO:
        1. Initialize empty predictions.
        2. Loop through every input example in X.
        3. For each example:
           a. Use get_k_neighbors_indices to find the k nearest neighbors.
           b. Get the labels of those neighbors.
           c. Vote (Majority wins). 
        """
        predictions = []

        for i in X:
            neighbor_indices = self.get_k_neighbors_indices(i)
            neighbor_labels = self.y_train[neighbor_indices]

            if np.sum(neighbor_labels) >= 0:
                predictions.append(1.0)
            else:
                predictions.append(-1.0)


        return np.array(predictions)