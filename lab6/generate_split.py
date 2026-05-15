from generators import generate_dataset_split
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

X_train, X_test, y_train, y_test = generate_dataset_split()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

np.savez("data/dataset_1.npz",
         X_train=X_train,
         y_train=y_train,
         X_test=X_test,
         y_test=y_test)