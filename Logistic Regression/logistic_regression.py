import numpy as np

def sigmoid(x):

    s = (1/(1+np.exp(-x)))
    return s
class Logistic_regression():

    def __init__(self, iterations=1000, learning_rate=0.001):

        self.iterations = iterations
        self.lr = learning_rate
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            linear_pred = np.dot(X, self.weights) + self.bias
            y_predictions = sigmoid(linear_pred)

            dw = (2/n_samples)*(np.dot(X.T, y_predictions-y))
            db = (2/n_samples)*(np.sum(y_predictions-y))

            self.weights -= (self.lr * dw)
            self.bias -= (self.lr * db)

    def predict(self, X):

        linear_pred = np.dot(X, self.weights) + self.bias
        y_predictions = sigmoid(linear_pred)
        class_pred = [0 if i<=0.5 else 1 for i in y_predictions]
        return class_pred


