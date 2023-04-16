import numpy as np
class Linear_regression():

    def __init__(self, learning_rate=0.01, n_iterations=1000):

        # Initializing the parameters and hyperparameters
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):

            # making the predictions
            y_predictions = np.dot(X, self.weights) + self.bias

            # computing partial derivative w.r.t to weights
            dw = (2/n_samples) * np.dot(X.T, (y_predictions-y))
            # computing partial derivative w.r.t to bias
            db = (2/n_samples) * np.sum(y_predictions-y)

            # Updating the weights and bias
            self.weights = self.weights - (self.learning_rate * dw)
            self.bias = self.bias - (self.learning_rate * db)

    def predict(self, X):

        # Predicting the outcome (dependent variable) after obtaining the weights and bias
        y_predictions = np.dot(X, self.weights) + self.bias
        return y_predictions