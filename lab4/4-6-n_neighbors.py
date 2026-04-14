from sklearn.neighbors import KNeighborsClassifier
from lab4.tools import report
import numpy as np

data = np.load("shared/s_datasets.npz")

X_train = data["X_train"]
Y_train = data["Y_train"]
x_test = data["x_test"]
y_test = data["y_test"]

for k in range(1, 10):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, Y_train)

    #report(f"KNN (k={k})", model.predict(x_test), y_test)
    print(f"KNN (k={k}): {model.score(x_test, y_test):.2%}")

