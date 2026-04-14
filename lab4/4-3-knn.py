from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import joblib

data = np.load("shared/s_datasets.npz")

X_train = data["X_train"]
Y_train = data["Y_train"]

knn = KNeighborsClassifier()

knn.fit(X_train, Y_train)

joblib.dump(knn, "shared/knn_model.pkl")