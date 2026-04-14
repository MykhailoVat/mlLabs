import numpy as np
import joblib

data = np.load("shared/s_datasets.npz")
x_test = data["x_test"]
y_test = data["y_test"]

knn = joblib.load("shared/knn_model.pkl")

predicted = knn.predict(x_test)
expected = y_test

print(f"Predicted:{predicted[:36]}")
print(f"Expected: {expected[:36]}")
