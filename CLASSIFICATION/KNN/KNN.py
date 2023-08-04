


import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3, d = 'l'):
        self.k = k
        self.d = d

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def lorentzian_distance(self, x1, x2):
        squared_distance = np.sum((x1 - x2) ** 2)
        return np.log(1 + squared_distance)
    
    def mahalanobis_distance(self, x1, x2):
        self.cov_matrix = np.cov(X_train.T)
        diff = x1 - x2
        inv_cov_matrix = np.linalg.inv(self.cov_matrix)
        distance = np.sqrt(np.dot(np.dot(diff, inv_cov_matrix), diff.T))
        return distance
    
    def predict(self, X_test):
        y_pred = [self._predict(x) for x in X_test]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between the input point and all training points
        if self.d == 'e':
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        elif self.d == 'l':
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        elif self.d == 'm':
            distances = [self.mahalanobis_distance(x, x_train) for x_train in self.X_train]

        # Get the k-nearest points and their corresponding classes
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_classes = [self.y_train[i] for i in k_indices]

        # Return the majority class among k-nearest neighbors
        most_common = Counter(k_nearest_classes).most_common(1)
        return most_common[0][0]

# Example usage of the KNN algorithm
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    # X_test = np.array([[4, 2], [2, 5], [6, 3]])
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples = 500, n_features = 2, centers = 4,cluster_std = 1.5, random_state = 4)
    
    # plt.style.use('seaborn')
    # plt.figure(figsize = (10,10))
    # plt.scatter(X[:,0], X[:,1], c=y, s = 10, edgecolors = 'black')
    # plt.show()
    
    #%---Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
    # Create and train the KNN classifier with k=3
    knn = KNN(k = 3, d = 'l')
    knn.fit(X_train, y_train)

    # Predict the classes of the test data
    y_pred = knn.predict(X_test)
    print("Predicted Classes:", y_pred)
    
    plt.style.use('seaborn')
    plt.figure(figsize = (10,10))
    plt.scatter(X_train[:,0], X_train[:,1], s = 10, edgecolors = 'black')
    plt.scatter(X_test[:,0], X_test[:,1], c = y_pred, s = 20, edgecolors = 'black')
    plt.show()

    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))
    
    
