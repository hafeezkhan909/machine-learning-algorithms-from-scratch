import matplotlib.pyplot as plt
from knn import KNN
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
iris_dataset = datasets.load_iris()

cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])
X, y = iris_dataset.data, iris_dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()

knn_classifier = KNN()
knn_classifier.fit(X_train, y_train)
predictions = knn_classifier.predict(X_test)

print(predictions)

accuracy = (np.sum(predictions == y_test)/len(y_test))*100
print('The accuracy attained by the classifier is', accuracy, '%')