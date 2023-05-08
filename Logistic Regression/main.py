import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from logistic_regression import Logistic_regression

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier = Logistic_regression(learning_rate=0.05)
classifier.fit(X_train, y_train)
y_predictions = classifier.predict(X_test)

def acc(y_pred, y_test):

    accuracy = np.sum(y_pred == y_test) / len(y_test)
    return accuracy

acc_percentage = acc(y_predictions, y_test) * 100

print("The accuracy of the logistic regression model is:", acc_percentage)


