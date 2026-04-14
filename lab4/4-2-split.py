from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

digits = load_digits()

X_train, x_test, Y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.2, random_state = 11)

np.savez("shared/s_datasets.npz",
         X_train=X_train,
         Y_train=Y_train,
         x_test=x_test,
         y_test=y_test)
