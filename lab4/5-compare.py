from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from lab4.tools import report
import numpy as np

data = np.load("shared/s_datasets.npz")

X_train = data["X_train"]
Y_train = data["Y_train"]
x_test = data["x_test"]
y_test = data["y_test"]

svc = SVC()
knn = KNeighborsClassifier()
nb = GaussianNB()

svc.fit(X_train, Y_train)
knn.fit(X_train, Y_train)
nb.fit(X_train, Y_train)

report("SVC", svc.predict(x_test), y_test)
report("Neighbors", knn.predict(x_test), y_test)
report("Naive Bayes", nb.predict(x_test), y_test)
