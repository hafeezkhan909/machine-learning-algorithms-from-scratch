from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from LinearRegression import Linear_regression
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=0)

# train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

fig = plt.figure(figsize=(8, 6))
plt.scatter(X, y, color="g", marker="o", s=30)
plt.show()

reg_model = Linear_regression(learning_rate=0.02)
reg_model.fit(X_train, y_train)
predictions = reg_model.predict(X_test)

def mean_squared_error(y_test, y_pred):

    return np.mean((y_test-y_pred)**2)

mse = mean_squared_error(y_test, predictions)
print('This is the mean squared error between the actual and the predicted values of y: ',mse)

y_pred_line = reg_model.predict(X)

cmap = plt.get_cmap('viridis')
fig2 = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=30)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=30)
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()